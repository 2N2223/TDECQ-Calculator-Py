# -*- coding: utf-8 -*-
"""FFE tap-bound constraints and bounded optimisation methods (802.3dj style).

Reference-receiver tap limits
-----------------------------
IEEE 802.3dj draft discussions (``welch_3dj_01_2405`` + D2.1 rulings #313 /
#183, see ``work/ieee_docs/``) constrain the reference FFE/DFE tap weights so
that a measurement cannot be won by unbounded noise amplification / over-
equalisation.  This module implements those constraints as *global algorithm
constants* (not per-case lookups, not reference-value back-filling): a
relative-offset bound table that is generic in the tap count ``N`` and the
main-tap index ``m``, unit-gain normalisation, and the pre/post differential
limit ``|w(+1) - w(-1)| / w(0) <= 0.25`` (no DFE, ``b(1) = 0``).

Bounded optimisation
--------------------
The unconstrained MMSE is only a baseline: when it violates the bounds the
final taps must come from a bounded, constrained optimisation:

* ``solve_bounded_qp``   - SLSQP quadratic programme (box + unit gain +
  differential limit);
* ``solve_active_set``   - iterative clamp + re-optimise of the free taps;
* ``solve_damped``       - ``identity + alpha*(MMSE - identity)`` projected
  into the feasible set;
* ``progressive_cd``     - constrained projection coordinate descent on the
  TDECQ metric with gradual tap expansion.

Every public function is deterministic and takes ``(N, main_index)`` so it is
generic in the FFE geometry.  Only numpy + scipy are used.
"""
from __future__ import annotations

import numpy as np

try:  # scipy is a declared dependency of the tool
    from scipy.optimize import lsq_linear, minimize  # noqa: F401
except Exception:  # pragma: no cover - imported lazily inside solvers too
    lsq_linear = minimize = None


# ---------------------------------------------------------------------------
# global constraint configuration (algorithm constants, not per-case values)
# ---------------------------------------------------------------------------

# relative-offset bound table, keyed by n = tap_index - main_index.  The
# bounds are normalised by the main tap (w(0) ~ 1).  |n| >= 3 uses the far-
# tap band uniformly (802.3dj treats n >= 3 the same way).
_BOUND_TABLE = {
    0: (0.8, 2.5),     # main tap
    -1: (-0.4, 0.05),  # near pre-cursor
    1: (-0.4, 0.05),   # near post-cursor
    -2: (-0.1, 0.2),   # far pre-cursor
    2: (-0.1, 0.2),    # far post-cursor
}
_FAR_BAND = (-0.1, 0.1)  # |n| >= 3


class FFEConstraints:
    """Global, immutable FFE tap-bound configuration.

    Attributes
    ----------
    N_candidates : tuple of int
        candidate tap counts (the tap-count dimension of the candidate pool).
    max_pre, max_post : int
        main-tap position limits (802.3dj structural caps: 3 pre-cursors,
        13 post-cursors).
    diff_limit : float
        pre/post differential limit ``|w(+1)-w(-1)|/w(0)`` (b(1) = 0).
    sum_tol, box_tol : float
        feasibility tolerances for the unit-gain and the box checks.
    step, max_iter : float / int
        progressive coordinate-descent step and sweep count.
    extend_delta : float
        minimum metric improvement (dB) to keep a tap-expansion step.
    lam_dffe, lam_ceq, lam_taps : float
        selection-score penalties: ``score = metric + lam_dffe*D_FFE +
        lam_ceq*C_eq + lam_taps*max(0, N-5)`` (dB-scale weights).
    """

    N_candidates = (5, 7, 9)
    max_pre = 3
    max_post = 13
    diff_limit = 0.25
    sum_tol = 1e-6
    box_tol = 1e-6
    step = 0.02
    max_iter = 10
    extend_delta = 0.05
    lam_dffe = 0.2
    lam_ceq = 0.1
    lam_taps = 0.05
    min_sym = 256

    def bound_for(self, n):
        """Box bound (lo, hi) for a relative offset ``n`` (main tap = 0)."""
        if n in _BOUND_TABLE:
            return _BOUND_TABLE[n]
        return _FAR_BAND


# one shared global instance - every case uses the same values
FFE_BOUNDS = FFEConstraints()


# ---------------------------------------------------------------------------
# bounds / feasibility
# ---------------------------------------------------------------------------


def bounds_for(N, main_index):
    """Per-tap box bounds (lb, ub) for tap count ``N`` and main index ``m``.

    The bounds are defined in the *relative* coordinate ``n = i - m`` so the
    table is independent of the absolute tap numbering: a 15-tap filter with
    the main tap at index 2 maps tap 0 to ``n = -2`` exactly like a 5-tap
    filter with the main tap at index 0.

    Parameters
    ----------
    N : int
        total tap count (>= 1).
    main_index : int
        index of the main tap (0 <= main_index < N).

    Returns
    -------
    (lb, ub) : np.ndarray of length N.
    """
    idx = np.arange(N, dtype=int)
    rel = idx - int(main_index)
    lb = np.array([FFE_BOUNDS.bound_for(int(n))[0] for n in rel], dtype=float)
    ub = np.array([FFE_BOUNDS.bound_for(int(n))[1] for n in rel], dtype=float)
    return lb, ub


def check_feasible(w, main_index, atol=None):
    """Check an FFE tap vector against every bound.

    Parameters
    ----------
    w : array-like of length N.
    main_index : int
        main-tap index (defines the relative-offset coordinate).
    atol : float or None
        tolerance used for the box / unit-gain / differential checks
        (defaults to ``FFE_BOUNDS.box_tol``).

    Returns
    -------
    (ok, violations) : ``ok`` is True only when every check passes;
    ``violations`` is a list of human-readable strings describing what failed
    (empty when ``ok``).
    """
    w = np.asarray(w, dtype=float)
    N = len(w)
    m = int(main_index)
    if m < 0 or m >= N:
        return False, ["main index %d out of range for N=%d" % (m, N)]
    atol = FFE_BOUNDS.box_tol if atol is None else float(atol)
    lb, ub = bounds_for(N, m)
    viol = []
    for i in range(N):
        if w[i] < lb[i] - atol:
            viol.append("tap %d (n=%+d) %.4f < lb %.4f"
                        % (i, i - m, w[i], lb[i]))
        if w[i] > ub[i] + atol:
            viol.append("tap %d (n=%+d) %.4f > ub %.4f"
                        % (i, i - m, w[i], ub[i]))
    if abs(w.sum() - 1.0) > max(FFE_BOUNDS.sum_tol, atol):
        viol.append("unit gain violated: sum=%.6f" % w.sum())
    if m - 1 >= 0 and m + 1 < N and w[m] > 0:
        d = abs(w[m + 1] - w[m - 1]) / w[m]
        if d > FFE_BOUNDS.diff_limit + atol:
            viol.append("differential limit: |w(+1)-w(-1)|/w(0)=%.4f > %.2f"
                        % (d, FFE_BOUNDS.diff_limit))
    return (len(viol) == 0), viol


def project_feasible(w, main_index, iters=12):
    """Project a tap vector into the feasible set (box + unit gain + diff).

    The projection alternates box clipping, unit-gain renormalisation (the
    main tap absorbs the residual so ``sum = 1`` exactly) and the pre/post
    differential-limit projection.  It is an approximation (the set is not
    convex) - the result is always passed through :func:`check_feasible` by
    the callers, and the QP solver is the *exact* bounded method.

    Parameters
    ----------
    w : array-like of length N.
    main_index : int
    iters : int
        projection iterations.

    Returns
    -------
    np.ndarray of length N (approximately feasible).
    """
    w = np.asarray(w, dtype=float).copy()
    N = len(w)
    m = int(main_index)
    lb, ub = bounds_for(N, m)
    for _ in range(iters):
        w = np.clip(w, lb, ub)
        s = w.sum()
        if s > 1e-12:
            w = w / s
        w[m] = float(np.clip(w[m], lb[m], ub[m]))
        oth = np.ones(N, dtype=bool)
        oth[m] = False
        tgt = 1.0 - w[m]
        so = w[oth].sum()
        if abs(so) > 1e-12:
            w[oth] *= tgt / so
        if m - 1 >= 0 and m + 1 < N and w[m] > 1e-12:
            d = w[m + 1] - w[m - 1]
            lim = FFE_BOUNDS.diff_limit * w[m]
            if abs(d) > lim:
                excess = abs(d) - lim
                sgn = 1.0 if d > 0 else -1.0
                w[m + 1] -= sgn * excess / 2.0
                w[m - 1] += sgn * excess / 2.0
    return np.clip(w, lb, ub)


# ---------------------------------------------------------------------------
# main-tap position from the pulse-response peak
# ---------------------------------------------------------------------------


def main_index_from_pulse(v, spu, tru, nom, cdr, L=15):
    """Estimate the channel pulse-response peak and return its tap delay.

    The received waveform ``v`` is modelled as ``y ~= conv(x, h)`` with the
    PAM4 level sequence ``x = nom[tru]``; the channel impulse response ``h``
    is estimated by least squares on the sampling phase ``cdr`` and the main
    tap is placed at the index of the largest |h|, clamped to the structural
    pre-cursor cap.

    Parameters
    ----------
    v : np.ndarray
        uniformly sampled waveform.
    spu : int
        samples per UI.
    tru : np.ndarray
        aligned reference symbols (first ``n_sym`` used).
    nom : np.ndarray of 4 nominal PAM4 levels.
    cdr : int
        sampling phase.
    L : int
        channel-pulse length in taps (>= max tap count).

    Returns
    -------
    int main-tap index candidate (0..FFE_BOUNDS.max_pre).
    """
    n = min(len(tru), (len(v) // spu) - 1)
    if n < 32:
        return 0
    x = nom[np.asarray(tru, dtype=int)[:n]]
    y = np.asarray(v, dtype=float)[cdr::spu][:n]
    X = np.zeros((n, L), dtype=float)
    for i in range(L):
        X[i:, i] = x[: n - i]
    h, _res, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    m0 = int(np.argmax(np.abs(h)))
    return int(np.clip(m0, 0, FFE_BOUNDS.max_pre))


# ---------------------------------------------------------------------------
# bounded solvers (all return a tap vector of length N, main at main_index)
# ---------------------------------------------------------------------------


def _design(X, y, main_index):
    """Return the least-squares objective data and a unit-gain initial guess.

    ``X`` is the (n_sym, N) tap matrix at one sampling phase and ``y`` the
    target PAM4 levels; the returned ``x0`` is a clipped, unit-gain start.
    """
    N = X.shape[1]
    m = int(main_index)
    lb, ub = bounds_for(N, m)
    x0 = np.full(N, 1.0 / N)
    x0[m] = 1.0
    x0 = np.clip(x0, lb, ub)
    x0 = x0 / x0.sum()
    return lb, ub, x0


def solve_bounded_qp(X, y, main_index, diff_lim=None):
    """Bounded quadratic programme (SLSQP): box + unit gain + differential.

    Minimises ``||X w - y||^2`` subject to ``lb <= w <= ub``, ``sum(w) = 1``
    and (when both neighbours of the main tap exist)
    ``-diff_lim*w(m) <= w(m+1) - w(m-1) <= diff_lim*w(m)``.

    Parameters
    ----------
    X : (n_sym, N) tap matrix at the sampling phase.
    y : (n_sym,) target PAM4 levels.
    main_index : int
    diff_lim : float or None
        differential limit (defaults to ``FFE_BOUNDS.diff_limit``).

    Returns
    -------
    np.ndarray of length N.  The result is re-checked by the caller with
    :func:`check_feasible`; when SLSQP fails to converge, a box-only
    ``lsq_linear`` solution on the free taps (main tap eliminated by the
    unit-gain sum) is returned as an approximately feasible fallback.
    """
    N = X.shape[1]
    m = int(main_index)
    diff_lim = FFE_BOUNDS.diff_limit if diff_lim is None else float(diff_lim)
    lb, ub, x0 = _design(X, y, m)
    if minimize is None:  # pragma: no cover
        return project_feasible(x0, m)

    def fun(w):
        r = X @ w - y
        return float(r @ r)

    cons = [{"type": "eq", "fun": lambda w: float(w.sum()) - 1.0}]
    if m - 1 >= 0 and m + 1 < N:
        cons.append({"type": "ineq",
                     "fun": lambda w: diff_lim * w[m] - (w[m + 1] - w[m - 1])})
        cons.append({"type": "ineq",
                     "fun": lambda w: diff_lim * w[m] + (w[m + 1] - w[m - 1])})
    import warnings as _warnings
    with _warnings.catch_warnings():
        # SLSQP temporarily evaluates infeasible iterates outside the box
        # ("Values in x were outside bounds during a minimize step"); the
        # final result is re-checked by check_feasible, so the message is
        # noise here.
        _warnings.simplefilter("ignore", RuntimeWarning)
        res = minimize(fun, x0, method="SLSQP", bounds=list(zip(lb, ub)),
                       constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-12})
    if res.success:
        return np.asarray(res.x, dtype=float)
    # fallback: eliminate the main tap (sum = 1) and solve a box-only LS
    others = [i for i in range(N) if i != m]
    A = X[:, others] - X[:, [m]]
    b = y - X[:, m]
    res2 = lsq_linear(A, b, bounds=(lb[others], ub[others]))
    w = np.zeros(N)
    w[others] = res2.x
    w[m] = 1.0 - float(res2.x.sum())
    return project_feasible(w, m)


def solve_active_set(X, y, main_index, max_iter=50):
    """Active-set estimate: clamp to the bounds, then re-optimise the rest.

    Iterates (clip -> solve least squares on the free taps -> unit-gain
    projection -> differential-limit projection) until the taps stop
    changing.  This is a degraded approximation of the QP and is only one
    candidate of the pool.

    Parameters
    ----------
    X : (n_sym, N) tap matrix.
    y : (n_sym,) targets.
    main_index : int
    max_iter : int

    Returns
    -------
    np.ndarray of length N.
    """
    N = X.shape[1]
    m = int(main_index)
    lb, ub = bounds_for(N, m)
    w = project_feasible(np.ones(N) / N, m)
    for _ in range(max_iter):
        prev = w.copy()
        w = np.clip(w, lb, ub)
        free = [i for i in range(N)
                if lb[i] - FFE_BOUNDS.box_tol < w[i] < ub[i] + FFE_BOUNDS.box_tol
                and i != m]  # main tap stays the unit-gain anchor
        fixed = [i for i in range(N) if i not in free]
        if free:
            wf = np.linalg.lstsq(X[:, free], y - X[:, fixed] @ w[fixed],
                                 rcond=None)[0]
            w[free] = wf
        w = project_feasible(w, m)
        if np.max(np.abs(w - prev)) < 1e-12:
            break
    return w


def solve_damped(X, y, main_index, alpha):
    """Damped MMSE: ``identity + alpha*(MMSE - identity)``, projected.

    Parameters
    ----------
    X : (n_sym, N) tap matrix.
    y : (n_sym,) targets.
    main_index : int
    alpha : float in (0, 1]

    Returns
    -------
    np.ndarray of length N (projected into the feasible set).
    """
    N = X.shape[1]
    m = int(main_index)
    wm = np.linalg.solve(X.T @ X + 1e-9 * np.eye(N), X.T @ y)
    s = wm.sum()
    if abs(s) > 1e-12:
        wm = wm / s
    ident = np.zeros(N)
    ident[m] = 1.0
    w = (1.0 - alpha) * ident + alpha * wm
    return project_feasible(w, m)


def progressive_cd(v, spu, n_ui, tru, cdr, main_index, max_N=None,
                   n_sym_limit=32768, decimate=1):
    """Constrained projection coordinate descent with gradual tap expansion.

    Stage 1 runs coordinate descent from the identity at the starting tap
    count (the first of ``FFE_BOUNDS.N_candidates``), keeping the taps inside
    the box / unit-gain / differential limits and accepting only moves that
    improve the sample-based TDECQ metric.  Stage 2 expands the filter by one
    pre- and one post-cursor tap at a time and keeps the expansion only while
    it improves the metric by more than ``FFE_BOUNDS.extend_delta`` dB.

    Parameters
    ----------
    v : np.ndarray
        uniformly sampled waveform (length >= ``n_ui * spu``).
    spu : int
        samples per UI.
    n_ui : int
        record length in UI (the starting window is ``n_ui - (N0-1)``
        symbols, one fewer per added tap pair as the filter expands).
    tru : np.ndarray
        reference symbols of the full window.
    cdr : int
        sampling phase.
    main_index : int
        main-tap index candidate of the starting geometry (clamped to the
        structural pre-cursor cap).
    max_N : int or None
        expansion ceiling (defaults to the last of ``N_candidates``).
    n_sym_limit : int
        records longer than this skip the metric-heavy CD (the QP /
        active-set / damped candidates still cover them).
    decimate : int
        step for the internal move-ranking metric (e.g. 4 ranks the moves on
        every 4th symbol).  Only the move *order* uses the decimated metric -
        the caller re-evaluates the returned taps at full precision.

    Returns
    -------
    (w, N_final) : the best feasible tap vector and its tap count, or None
    when the record is too long for the metric evaluations.
    """
    from . import eq  # local import to avoid a cycle
    max_N = FFE_BOUNDS.N_candidates[-1] if max_N is None else int(max_N)
    dec = max(1, int(decimate))
    N0 = int(FFE_BOUNDS.N_candidates[0])
    m = int(np.clip(main_index, 0, FFE_BOUNDS.max_pre))
    if N0 - 1 - m > FFE_BOUNDS.max_post:
        m = int(max(0, N0 - 1 - FFE_BOUNDS.max_post))
    n_sym0 = n_ui - (N0 - 1)
    if n_sym0 < FFE_BOUNDS.min_sym or n_sym0 > n_sym_limit:
        return None
    tru0 = np.asarray(tru[:n_sym0])
    Mj = eq.build_Mj(v, spu, n_sym0, m, N0 - 1 - m)
    N = N0
    w = np.zeros(N)
    w[m] = 1.0
    w = project_feasible(w, m)

    def metric_of(Mj, w, tru, cdr):
        M = np.tensordot(Mj, w, axes=(0, 0))
        if dec > 1:
            M = M[::dec]
            tru = tru[::dec]
        mdb, _ = eq.metric_detail(M, w, tru, cdr, spu, "samp")
        return mdb

    def descend(Mj, w, tru, m, best):
        changed = True
        for _ in range(FFE_BOUNDS.max_iter):
            moved = False
            for i in range(len(w)):
                if i == m:
                    continue
                for d in (FFE_BOUNDS.step, -FFE_BOUNDS.step):
                    wt = w.copy()
                    wt[i] += d
                    wt[m] = 1.0 - (wt.sum() - wt[m])
                    wt = project_feasible(wt, m)
                    ok, _ = check_feasible(wt, m)
                    if not ok:
                        continue
                    mt = metric_of(Mj, wt, tru, cdr)
                    if mt < best - 1e-12:
                        w, best = wt, mt
                        moved = True
            if not moved:
                break
        return w, best

    w, best = descend(Mj, w, tru0, m, metric_of(Mj, w, tru0, cdr))
    while N < max_N:
        N2 = N + 2
        m2 = m + 1
        if N2 - 1 - m2 > FFE_BOUNDS.max_post:
            break
        n_sym2 = n_ui - (N2 - 1)
        if n_sym2 < FFE_BOUNDS.min_sym:
            break
        w2 = np.zeros(N2)
        w2[1:N + 1] = w          # existing taps shift right by one pre-cursor
        w2 = project_feasible(w2, m2)
        Mj2 = eq.build_Mj(v, spu, n_sym2, m2, N2 - 1 - m2)
        tru2 = np.asarray(tru[:n_sym2])
        w2, best2 = descend(Mj2, w2, tru2, m2, best)
        if best2 < best - FFE_BOUNDS.extend_delta:
            w, best, N, m = w2, best2, N2, m2
        else:
            break
    return w, N


def at_bound_info(w, main_index, atol=None):
    """Which taps sit on a bound (for plot annotations).

    Parameters
    ----------
    w : array-like of length N.
    main_index : int
        main-tap index (defines the relative-offset coordinate).
    atol : float or None
        tolerance used to decide that a tap touches a bound (defaults to
        ``FFE_BOUNDS.box_tol``).

    Returns
    -------
    list of length N with ``"lb"`` (tap at the lower bound), ``"ub"`` (tap at
    the upper bound) or ``None`` for interior taps.
    """
    w = np.asarray(w, dtype=float)
    N = len(w)
    m = int(main_index)
    atol = FFE_BOUNDS.box_tol if atol is None else float(atol)
    lb, ub = bounds_for(N, m)
    out = []
    for i in range(N):
        if abs(w[i] - lb[i]) <= atol:
            out.append("lb")
        elif abs(w[i] - ub[i]) <= atol:
            out.append("ub")
        else:
            out.append(None)
    return out
