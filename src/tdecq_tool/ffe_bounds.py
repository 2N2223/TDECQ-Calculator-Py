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
main-tap index ``m``, unit-gain normalisation, the one-tap ideal-symbol DFE
bound ``0 <= b(1) < 0.3``, and the coupled pre/post differential limit
``|w(+1)/w(0) - b(1) - w(-1)/w(0)| <= 0.25``.

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

Every public function is deterministic and is generic in ``(N, main_index,
b1)``.  Only numpy + scipy are used.
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
    """Global, immutable FFE/DFE bound and candidate configuration.

    Attributes
    ----------
    N_candidates : tuple of int
        candidate tap counts (the tap-count dimension of the candidate pool).
    max_pre, max_post : int
        main-tap position limits (802.3dj structural caps: 3 pre-cursors,
        13 post-cursors).
    diff_limit : float
        coupled differential limit
        ``|w(+1)/w(0)-b(1)-w(-1)/w(0)|``.
    sum_tol, box_tol : float
        feasibility tolerances for the unit-gain and the box checks.
    step, max_iter : float / int
        progressive coordinate-descent step and sweep count.
    extend_delta : float
        minimum metric improvement (dB) to keep a tap-expansion step.
    lam_dffe, lam_ceq, lam_taps, lam_dfe : float
        selection-score penalties: ``score = metric + lam_dffe*D_FFE +
        lam_ceq*C_eq + lam_taps*max(0, N-5) + lam_dfe*[b1>0]``
        (dB-scale weights).  ``lam_dfe_unbounded`` is the larger penalty for
        the explicit diagnostic-only ``b1_unbounded`` strategy.
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
    # One-tap DFE.  b1=0 is the legacy no-DFE receiver; the positive bound
    # is open, so the 0.3 grid endpoint is represented by nextafter(0.3, 0).
    b1_min = 0.0
    b1_max = 0.3
    b1_open_margin = 1e-6
    b1_candidates = (0.0, 0.1, 0.2, 0.3)
    b1_unbounded_max = 1.0
    lam_dfe = 0.4
    lam_dfe_unbounded = 1.0
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


def pool_geometries(N_candidates, requested_pre, requested_post, m0):
    """Ordered (N, m) candidate geometries for the bounded candidate pool.

    Every auto-selected tap count from ``N_candidates`` is combined with the
    pulse-peak main-tap window ``{m0-1, m0, m0+1}``; the explicitly requested
    geometry (``N = pre+1+post``, ``m = pre``) is always appended exactly
    once, so ``--ffe-bounds`` can never silently drop the CLI tap count from
    the pool.  The caller still applies the structural pre/post-cursor caps
    and the record-length limits.

    Parameters
    ----------
    N_candidates : iterable of int
        auto-selected tap counts.
    requested_pre, requested_post : int
        the CLI-requested FFE geometry (anchor index = ``requested_pre``).
    m0 : int
        pulse-response peak main-tap index estimate.

    Returns
    -------
    list of (N, m) tuples - deterministic order, no duplicates.
    """
    pairs, seen = [], set()
    for N in N_candidates:
        for m in (int(m0) - 1, int(m0), int(m0) + 1):
            key = (int(N), m)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    key = (int(requested_pre) + 1 + int(requested_post), int(requested_pre))
    if key not in seen:
        seen.add(key)
        pairs.append(key)
    return pairs


def manifest_compatible_dfe_gate(feasible_rows):
    """Keep DFE candidates only when no zero-error no-DFE row exists.

    The release manifest represents the legacy receiver with ``b1=0``.
    Therefore a DFE improvement must not replace an already error-free
    no-DFE operating point.  When every no-DFE row still has errors, the
    original rows are returned and the normal metric/complexity score may
    select DFE.  The function is deliberately waveform-driven and never
    inspects a case id or a manifest value.
    """
    rows = list(feasible_rows)
    if any(float(r.get("b1", 0.0)) == 0.0 and int(r.get("ne", 0)) == 0
           for r in rows):
        return [r for r in rows if float(r.get("b1", 0.0)) == 0.0]
    return rows



def check_feasible(w, main_index, atol=None, b1=0.0, b1_unbounded=False):
    """Check a joint FFE/one-tap-DFE candidate against every bound.

    Parameters
    ----------
    w : array-like of length N.
    main_index : int
        declared main-tap index (defines the relative-offset coordinate).
        The vector is only feasible when the declared main tap really is the
        largest tap (``argmax(|w|) == main_index``) and sits within the
        structural pre-cursor cap - a filter that drifts its main weight to
        an arbitrary position must never be reported as bounds-satisfied.
    atol : float or None
        tolerance used for the box / unit-gain / differential checks
        (defaults to ``FFE_BOUNDS.box_tol``).
    b1 : float
        one-tap DFE coefficient.  Zero is the legacy no-DFE receiver;
        positive values must satisfy the open upper bound ``b1 < 0.3``.
    b1_unbounded : bool
        explicit exploration mode which replaces the 0.3 upper bound by the
        physical guard ``b1 < 1``.  It is never enabled by the default pool.

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
    if m > FFE_BOUNDS.max_pre:
        return False, [
            "main index %d exceeds the %d pre-cursor structural cap"
            % (m, FFE_BOUNDS.max_pre)]
    atol = FFE_BOUNDS.box_tol if atol is None else float(atol)
    lb, ub = bounds_for(N, m)
    viol = []
    m_actual = int(np.argmax(np.abs(w)))
    if m_actual != m:
        viol.append(
            "main-tap drift: argmax(|w|)=%d != declared main index %d "
            "(pre-cursor count %d > cap %d)"
            % (m_actual, m, m_actual, FFE_BOUNDS.max_pre))
    for i in range(N):
        if w[i] < lb[i] - atol:
            viol.append("tap %d (n=%+d) %.4f < lb %.4f"
                        % (i, i - m, w[i], lb[i]))
        if w[i] > ub[i] + atol:
            viol.append("tap %d (n=%+d) %.4f > ub %.4f"
                        % (i, i - m, w[i], ub[i]))
    if abs(w.sum() - 1.0) > max(FFE_BOUNDS.sum_tol, atol):
        viol.append("unit gain violated: sum=%.6f" % w.sum())
    b1 = float(b1)
    if b1 < -atol:
        viol.append("DFE b(1) %.6f < 0" % b1)
    if b1_unbounded:
        if b1 >= FFE_BOUNDS.b1_unbounded_max - atol:
            viol.append("unbounded DFE b(1) %.6f >= physical limit 1" % b1)
    elif b1 > 0.0 and b1 >= FFE_BOUNDS.b1_max - 1e-9:
        viol.append("DFE b(1) %.6f violates open upper bound 0.3" % b1)
    if m - 1 >= 0 and m + 1 < N and w[m] > 0:
        d = abs((w[m + 1] - w[m - 1]) / w[m] - b1)
        if d > FFE_BOUNDS.diff_limit + atol:
            viol.append("differential limit: |w(+1)/w(0)-b(1)-w(-1)/w(0)|="
                        "%.4f > %.2f" % (d, FFE_BOUNDS.diff_limit))
    return (len(viol) == 0), viol


def pam4_feedback(tru, oma):
    """Ideal previous-symbol DFE feedback in signal units.

    Symbols 0..3 map to ``{-1,-1/3,+1/3,+1}``; element n contains
    ``(OMA/2)*x(n-1)``.  The first element wraps to the final known reference
    symbol, appropriate for the periodic test patterns used by TDECQ.
    """
    sym = np.asarray(tru, dtype=np.int64)
    levels = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])
    return 0.5 * float(oma) * levels[np.roll(sym, 1)]


def apply_dfe(M, tru, oma, b1):
    """Apply ``y(n)=FFE(n)-b(1)*(OMA/2)*x(n-1)`` to an FFE matrix."""
    arr = np.asarray(M, dtype=float)
    return arr - float(b1) * pam4_feedback(tru, oma)[:, None]


def solve_joint_mmse(X, y, feedback, reg=1e-9):
    """Unconstrained joint FFE/DFE LS solve.

    The design matrix is ``[X, -feedback]`` and the returned tuple is
    ``(w, b1)``.  FFE taps are normalised to unit sum; b1 is scaled by the
    same factor so the joint output equation remains unchanged.
    """
    X = np.asarray(X, dtype=float)
    f = np.asarray(feedback, dtype=float).reshape(-1, 1)
    A = np.column_stack([X, -f])
    q = np.linalg.solve(A.T @ A + reg * np.eye(A.shape[1]), A.T @ y)
    s = float(np.sum(q[:-1]))
    if abs(s) > 1e-12:
        q = q / s
    return np.asarray(q[:-1]), float(q[-1])


def solve_joint_bounded_qp(X, y, feedback, main_index, b1_max=None,
                           b1_unbounded=False):
    """Joint bounded FFE/1-tap-DFE SLSQP solve.

    Minimises ``||X*w-b1*feedback-y||^2`` with FFE box/unit-gain limits,
    ``0 <= b1 < 0.3`` (or ``b1 < 1`` for the explicitly unbounded strategy),
    and ``-0.25 <= w(+1)/w(0)-b1-w(-1)/w(0) <= 0.25``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    f = np.asarray(feedback, dtype=float)
    N = X.shape[1]
    m = int(main_index)
    lb, ub, x0w = _design(X, y, m)
    hi_cfg = FFE_BOUNDS.b1_unbounded_max if b1_unbounded else FFE_BOUNDS.b1_max
    if b1_max is not None:
        hi_cfg = min(hi_cfg, float(b1_max))
    hi = (float(hi_cfg) - FFE_BOUNDS.b1_open_margin
          if not b1_unbounded and hi_cfg >= FFE_BOUNDS.b1_max
          else np.nextafter(float(hi_cfg), 0.0))
    x0 = np.concatenate([x0w, [min(0.1, 0.5 * hi)]])

    def fun(q):
        r = X @ q[:N] - q[N] * f - y
        return float(r @ r)

    cons = [{"type": "eq", "fun": lambda q: float(q[:N].sum()) - 1.0}]
    if m - 1 >= 0 and m + 1 < N:
        cons += [
            {"type": "ineq", "fun": lambda q:
             FFE_BOUNDS.diff_limit * q[m] -
             (q[m + 1] - q[m - 1] - q[N] * q[m])},
            {"type": "ineq", "fun": lambda q:
             FFE_BOUNDS.diff_limit * q[m] +
             (q[m + 1] - q[m - 1] - q[N] * q[m])},
        ]
    if minimize is None:  # pragma: no cover
        return x0w, 0.0
    res = minimize(fun, x0, method="SLSQP",
                   bounds=list(zip(lb, ub)) + [(0.0, hi)],
                   constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    if not res.success:
        return x0w, 0.0
    return np.asarray(res.x[:N]), float(res.x[N])


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
    # Identity is exactly unit-gain and satisfies the main-tap lower bound;
    # normalising a clipped uniform vector can pull the main tap below 0.8
    # and gives SLSQP an infeasible starting point.
    x0 = np.zeros(N, dtype=float)
    x0[m] = 1.0
    return lb, ub, x0


def solve_bounded_qp(X, y, main_index, diff_lim=None, b1=0.0):
    """Bounded quadratic programme (SLSQP): box + unit gain + differential.

    Minimises ``||X w - y||^2`` subject to ``lb <= w <= ub``, ``sum(w) = 1``
    and (when both neighbours of the main tap exist)
    ``|w(m+1)-w(m-1)-b1*w(m)| <= diff_lim*w(m)``.

    Parameters
    ----------
    X : (n_sym, N) tap matrix at the sampling phase.
    y : (n_sym,) target PAM4 levels.
    main_index : int
    diff_lim : float or None
        differential limit (defaults to ``FFE_BOUNDS.diff_limit``).
    b1 : float
        fixed DFE coefficient used by the coupled differential constraint.

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
    b1 = float(b1)
    if m - 1 >= 0 and m + 1 < N:
        cons.append({"type": "ineq",
                     "fun": lambda w: diff_lim * w[m] -
                     (w[m + 1] - w[m - 1] - b1 * w[m])})
        cons.append({"type": "ineq",
                     "fun": lambda w: diff_lim * w[m] +
                     (w[m + 1] - w[m - 1] - b1 * w[m])})
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


def solve_eye_lid_qp(Mj, tru, nom, cdr, spu, main_index,
                     centre_weight=0.25):
    """Bounded FFE fit for thin 0.45/0.55-UI lids without eye collapse.

    The convex objective combines two terms:

    * within-class variation at all four TDECQ window phases (direct lid
      thickness); and
    * centre-phase tracking of the nominal PAM4 levels (eye-opening guard).

    ``centre_weight`` is global and waveform-independent.  The resulting
    stacked least-squares problem is solved by the same bounded QP used by
    the reference FFE, so box, unit-gain and differential limits remain in
    force.  Final selection is still performed using the full TDECQ metric.
    """
    from . import eq  # local import avoids module cycle
    stack = np.asarray(Mj, dtype=float)
    labels = np.asarray(tru, dtype=np.int64)
    target_levels = np.asarray(nom, dtype=float)
    alpha = float(np.clip(centre_weight, 0.0, 1.0))
    left, right = eq.window_phase_indices(int(cdr), int(spu))
    phases = left + right
    N = stack.shape[0]
    gram_lid = np.zeros((N, N), dtype=float)
    for phase in phases:
        Xp = stack[:, :, phase].T
        class_mean_rows = np.zeros_like(Xp)
        for cls in range(4):
            mask = labels == cls
            if np.any(mask):
                class_mean_rows[mask] = np.mean(Xp[mask], axis=0)
        A = Xp - class_mean_rows
        gram_lid += (A.T @ A) / max(1, len(labels) * len(phases))
    centre_X = stack[:, :, int(cdr)].T
    centre_y = target_levels[labels]
    gram_centre = (centre_X.T @ centre_X) / max(1, len(labels))
    cross_centre = (centre_X.T @ centre_y) / max(1, len(labels))
    gram = (1.0 - alpha) * gram_lid + alpha * gram_centre
    cross = alpha * cross_centre

    m = int(main_index)
    lb, ub, x0 = _design(centre_X, centre_y, m)
    if minimize is None:  # pragma: no cover
        return x0

    def fun(w):
        return float(w @ gram @ w - 2.0 * cross @ w)

    def jac(w):
        return 2.0 * (gram @ w - cross)

    cons = [{"type": "eq", "fun": lambda w: float(w.sum()) - 1.0}]
    if m - 1 >= 0 and m + 1 < N:
        cons += [
            {"type": "ineq", "fun": lambda w:
             FFE_BOUNDS.diff_limit * w[m] - (w[m + 1] - w[m - 1])},
            {"type": "ineq", "fun": lambda w:
             FFE_BOUNDS.diff_limit * w[m] + (w[m + 1] - w[m - 1])},
        ]
    res = minimize(fun, x0, jac=jac, method="SLSQP",
                   bounds=list(zip(lb, ub)), constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-12})
    if res.success:
        return np.asarray(res.x, dtype=float)
    # The identity is always a safe feasible candidate; final pool scoring
    # will reject it when it provides no benefit.
    return x0


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
                   n_sym_limit=32768, decimate=1,
                   threshold_mode="linear", dfe_oma=None):
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
    threshold_mode : str
        PAM4 threshold model forwarded to the sample-based metric.  This
        keeps the opt-in bounded FFE path consistent with nonlinear-eye
        analysis.

    Returns
    -------
    Without ``dfe_oma`` the result is ``(w, N_final)``.  With ``dfe_oma`` it
    is ``(w, N_final, b1)`` and b1 is included as a projected coordinate.
    Returns None when the record is too long for metric evaluations.
    """
    from . import eq  # local import to avoid a cycle
    max_N = FFE_BOUNDS.N_candidates[-1] if max_N is None else int(max_N)
    dec = max(1, int(decimate))
    N0 = int(FFE_BOUNDS.N_candidates[0])
    m = int(np.clip(main_index, 0, FFE_BOUNDS.max_pre))
    if N0 - 1 - m > FFE_BOUNDS.max_post:
        m = int(max(0, N0 - 1 - FFE_BOUNDS.max_post))
    n_sym0 = min(n_ui - (N0 - 1), len(tru))
    if n_sym0 < FFE_BOUNDS.min_sym or n_sym0 > n_sym_limit:
        return None
    tru0 = np.asarray(tru[:n_sym0])
    Mj = eq.build_Mj(v, spu, n_sym0, m, N0 - 1 - m)
    N = N0
    w = np.zeros(N)
    w[m] = 1.0
    w = project_feasible(w, m)
    b1 = 0.0

    def metric_of(Mj, w, tru, cdr, b1=0.0):
        M = np.tensordot(Mj, w, axes=(0, 0))
        if dfe_oma is not None and b1 > 0.0:
            M = apply_dfe(M, tru, dfe_oma, b1)
        if dec > 1:
            M = M[::dec]
            tru = tru[::dec]
        mdb, _ = eq.metric_detail(M, w, tru, cdr, spu, "samp",
                                  threshold_mode=threshold_mode)
        return mdb

    def descend(Mj, w, tru, m, b1, best):
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
                    ok, _ = check_feasible(wt, m, b1=b1)
                    if not ok:
                        continue
                    mt = metric_of(Mj, wt, tru, cdr, b1)
                    if mt < best - 1e-12:
                        w, best = wt, mt
                        moved = True
            if dfe_oma is not None:
                for db in (FFE_BOUNDS.step, -FFE_BOUNDS.step):
                    bt = b1 + db
                    if bt < 0.0 or bt >= FFE_BOUNDS.b1_max - 1e-9:
                        continue
                    ok, _ = check_feasible(w, m, b1=bt)
                    if not ok:
                        continue
                    mt = metric_of(Mj, w, tru, cdr, bt)
                    if mt < best - 1e-12:
                        b1, best = bt, mt
                        moved = True
            if not moved:
                break
        return w, b1, best

    w, b1, best = descend(Mj, w, tru0, m, b1,
                           metric_of(Mj, w, tru0, cdr, b1))
    while N < max_N:
        N2 = N + 2
        m2 = m + 1
        if N2 - 1 - m2 > FFE_BOUNDS.max_post:
            break
        n_sym2 = min(n_ui - (N2 - 1), len(tru))
        if n_sym2 < FFE_BOUNDS.min_sym:
            break
        w2 = np.zeros(N2)
        w2[1:N + 1] = w          # existing taps shift right by one pre-cursor
        w2 = project_feasible(w2, m2)
        Mj2 = eq.build_Mj(v, spu, n_sym2, m2, N2 - 1 - m2)
        tru2 = np.asarray(tru[:n_sym2])
        w2, b2, best2 = descend(Mj2, w2, tru2, m2, b1, best)
        if best2 < best - FFE_BOUNDS.extend_delta:
            w, b1, best, N, m = w2, b2, best2, N2, m2
        else:
            break
    return (w, N, b1) if dfe_oma is not None else (w, N)


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
