# -*- coding: utf-8 -*-
"""Standard-conformant TDECQ/TECQ measurement (direct constrained minimisation).

This module is the *standard* measurement core.  It replaces the legacy
reference-matching candidate selection of :mod:`tdecq_tool.pipeline` (which
is retained as ``tdecq_mode="legacy"`` for backwards compatibility) with the
procedure the IEEE 802.3 TDECQ definition actually describes:

* the reference FFE taps (and, for the P802.3dj draft profile, the one-tap
  DFE coefficient) are the ones that **minimise TDECQ** subject to the
  reference-equaliser constraints - no target-matching, no penalty terms,
  no "weak basin" heuristics;
* OMA_outer and P_ave are measured on the **unequalised** waveform
  (settled outer levels from long identical-symbol runs, average of the
  whole record), so the metric is invariant to the FFE gain and the
  thresholds do not move with the equaliser;
* the vertical histograms are **unlabelled** (the symbol reference is only
  used for the OMA run search, the LS starting point, the DFE feedback and
  the reported SER) and are taken in two 0.04-UI-wide windows centred at
  0.45 / 0.55 UI of the equalised eye;
* every histogram bin contributes only through its *adjacent* decision
  thresholds.

Tap-direction convention (IEEE): a filter with ``N`` taps and main tap index
``m`` has ``m`` **pre-cursor** taps (indices ``0..m-1``) that read *later*
samples (they cancel pre-cursor ISI) and ``N-1-m`` **post-cursor** taps
(indices ``m+1..N-1``) that read *earlier* samples (they cancel post-cursor
ISI).  Tap ``i`` of symbol ``n`` therefore reads the sample at symbol slot
``n + lag + m - i`` where ``lag`` is the measured main-cursor delay of the
reference alignment.  (The legacy :func:`tdecq_tool.eq.build_Mj` stack uses
the opposite naming; it is untouched so the legacy mode stays bit-identical.)

Items that are NOT fully pinned down by the published text available to this
project are exposed as explicit, documented options instead of being hidden
in heuristics: the left/right sigma combination (``sigma_combine``), the
oscilloscope noise ``sigma_s`` and the noise-enhancement model ``ceq_model``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc

from . import ffe_bounds as fb

QT = 3.414                 # Q(QT) = 4.8e-4 / ... TDECQ normalisation
TARGET_SER = 4.8e-4
PAM4_LEVELS = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])


def _Q(x):
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


class StdConfig:
    """Reference-receiver definition used by :func:`measure`.

    Attributes
    ----------
    n_taps : int
        FFE length.
    main_candidates : tuple of int
        allowed main-tap indices (= number of pre-cursor taps); the
        largest-magnitude tap must be one of them.  The 802.3-2022 profile
        fixes the main tap at tap 1 (0 pre / 4 post), i.e. ``(0,)``.
    bounded : bool
        apply the P802.3dj draft tap-limit table / differential limit
        (:mod:`tdecq_tool.ffe_bounds`).
    dfe : bool
        optimise a one-tap DFE ``0 <= b1 < 0.3`` jointly with the FFE.
    ceq_model : 'white' | 'bt4'
        noise-enhancement model: ``sqrt(sum c^2)`` (white noise at the FFE
        input) or ``sqrt(c^T R c)`` with ``R`` the T-spaced autocorrelation
        of white noise filtered by a 4th-order Bessel-Thomson response of
        bandwidth ``bt4_bw_over_baud * baud``.
    sigma_combine : 'min' | 'max'
        how the left/right window sigmas are combined (``min`` = the worse
        window limits the measurement).
    sigma_s : float
        RMS noise of the measurement instrument referred to the FFE input
        (same units as the waveform); 0 disables the compensation.
    """

    def __init__(self, n_taps=5, main_candidates=(0,), bounded=False,
                 dfe=False, ceq_model="white", bt4_bw_over_baud=0.5,
                 sigma_combine="min", sigma_s=0.0, window_offset_ui=0.05,
                 window_width_ui=0.04, hist_bins=2048, oma_run_min=6,
                 threshold_mode="linear", max_evals=4000):
        self.n_taps = int(n_taps)
        self.main_candidates = tuple(int(m) for m in main_candidates)
        self.bounded = bool(bounded)
        self.dfe = bool(dfe)
        self.ceq_model = str(ceq_model)
        self.bt4_bw_over_baud = float(bt4_bw_over_baud)
        self.sigma_combine = str(sigma_combine)
        self.sigma_s = float(sigma_s)
        self.window_offset_ui = float(window_offset_ui)
        self.window_width_ui = float(window_width_ui)
        self.hist_bins = int(hist_bins)
        self.oma_run_min = int(oma_run_min)
        self.threshold_mode = str(threshold_mode)
        self.max_evals = int(max_evals)
        if self.sigma_combine not in ("min", "max"):
            raise ValueError("sigma_combine must be 'min' or 'max'")
        if self.ceq_model not in ("white", "bt4"):
            raise ValueError("ceq_model must be 'white' or 'bt4'")
        for m in self.main_candidates:
            if not 0 <= m < self.n_taps:
                raise ValueError("main tap index %d outside 0..%d"
                                 % (m, self.n_taps - 1))


def config_for_profile(profile, n_pre=None, n_post=None, **kw):
    """Standard-path configuration for a :class:`standards.StandardProfile`.

    ``n_pre``/``n_post`` (explicit geometry, IEEE naming) lock the tap count
    and the main-tap index; otherwise the profile reference receiver is
    used: 5 taps with the main tap fixed at tap 1 - no pre-cursor, 4
    post-cursors, no main-position search - (802.3-2022) or the draft
    15-tap FFE with up to ``FFE_BOUNDS.max_pre`` pre-cursors plus a bounded
    one-tap DFE (P802.3dj draft).
    """
    draft = bool(getattr(profile, "draft", False))
    if n_pre is not None or n_post is not None:
        pre = int(profile.default_pre if n_pre is None else n_pre)
        post = int(profile.default_post if n_post is None else n_post)
        n_taps, mains = pre + 1 + post, (pre,)
    elif draft:
        n_taps = profile.default_tap_count
        mains = tuple(range(0, min(fb.FFE_BOUNDS.max_pre, n_taps - 1) + 1))
    else:
        n_taps, mains = 5, (0,)
    kw.setdefault("bounded", draft)
    kw.setdefault("dfe", draft)
    kw.setdefault("ceq_model", "bt4" if draft else "white")
    return StdConfig(n_taps=n_taps, main_candidates=mains, **kw)


# ---------------------------------------------------------------------------
# noise enhancement
# ---------------------------------------------------------------------------


def bt4_noise_autocorr(n_lags, bw_over_baud=0.5):
    """T-spaced autocorrelation (R[0] = 1) of BT4-filtered white noise."""
    from scipy.signal import bessel, freqs
    fc = float(bw_over_baud)                       # in units of the baud
    b, a = bessel(4, 2.0 * np.pi * fc, "low", analog=True, norm="mag")
    f = np.linspace(0.0, 40.0 * fc, 80001)
    _, h = freqs(b, a, 2.0 * np.pi * f)
    p = np.abs(h) ** 2
    trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy < 2.0
    den = trapz(p, f)
    return np.array([trapz(p * np.cos(2.0 * np.pi * f * k), f) / den
                     for k in range(int(n_lags))])


def noise_enhancement(c, model="white", R=None):
    """C_eq of an FFE tap vector (see :class:`StdConfig`)."""
    c = np.asarray(c, dtype=float)
    if model == "white" or R is None:
        return float(np.sqrt(np.sum(c * c)))
    N = len(c)
    i = np.arange(N)
    Rm = R[np.abs(i[:, None] - i[None, :])]
    return float(np.sqrt(max(float(c @ Rm @ c), 0.0)))


# ---------------------------------------------------------------------------
# signal access with IEEE tap direction
# ---------------------------------------------------------------------------


class _Record:
    """Waveform + aligned reference with IEEE-direction tap access.

    Symbol ``n`` (``n`` indexes ``tru``) has its main cursor at sample slot
    ``n + lag``; the per-symbol phase grid is shifted by ``shift`` samples so
    the eye centre sits near ``spu/2`` (no window wrap-around).
    """

    def __init__(self, v, spu, tru, cdr0, lag, n_taps, mains):
        self.v = np.asarray(v, dtype=float)
        self.spu = int(spu)
        self.tru = np.asarray(tru, dtype=np.int64)
        self.lag = int(lag)
        self.shift = int(cdr0) - self.spu // 2
        n_slots = len(self.v) // self.spu
        mmax, mmin = max(mains), min(mains)
        # need slot n+lag+m-i in range for every i in 0..N-1, m in mains,
        # plus one guard slot for the phase shift on each side
        lo = max(1, (n_taps - 1) - self.lag - mmin + 1)
        hi = min(len(self.tru), n_slots - 1 - self.lag - mmax - 1)
        if hi - lo < 64:
            raise ValueError("record too short for a %d-tap standard FFE"
                             % n_taps)
        self.n = np.arange(lo, hi)
        self.labels = self.tru[self.n]
        self.prev = self.tru[self.n - 1]

    def taps(self, window, n_taps, phases):
        """(N, n_sym, P) samples; tap i reads slot ``n + window - i``."""
        ph = np.asarray(phases, dtype=np.int64)
        base = (self.n + int(window))[None, :] - np.arange(n_taps)[:, None]
        idx = base[:, :, None] * self.spu + self.shift + ph[None, None, :]
        return self.v[idx]

    def equalize(self, c, window, b1=0.0, oma=0.0):
        """Full (n_sym, spu) equalised eye matrix."""
        ph = np.arange(self.spu)
        z = np.zeros((len(self.n), self.spu))
        for i, ci in enumerate(c):
            if ci != 0.0:
                idx = ((self.n + int(window) - i)[:, None] * self.spu
                       + self.shift + ph[None, :])
                z += ci * self.v[idx]
        if b1:
            z -= b1 * 0.5 * oma * PAM4_LEVELS[self.prev][:, None]
        return z


def estimate_lag(v, spu, tru, cdr0, max_lag=6):
    """Main-cursor delay (symbols) of the aligned reference.

    Correlates the unequalised samples at the CDR phase with the reference
    PAM4 levels for lags ``-max_lag..max_lag`` and returns the lag of the
    largest correlation, i.e. the slot holding the pulse-response peak.
    """
    v = np.asarray(v, dtype=float)
    tru = np.asarray(tru, dtype=np.int64)
    n_slots = len(v) // spu
    n0, n1 = max_lag, min(len(tru), n_slots - max_lag - 1)
    x = PAM4_LEVELS[tru[n0:n1]]
    x = x - x.mean()
    best, best_c = 0, -np.inf
    for k in range(-max_lag, max_lag + 1):
        y = v[(np.arange(n0, n1) + k) * spu + int(cdr0)]
        y = y - y.mean()
        cval = float(np.dot(x, y))
        if cval > best_c:
            best, best_c = k, cval
    return int(best)


# ---------------------------------------------------------------------------
# levels / OMA (unequalised waveform)
# ---------------------------------------------------------------------------


def settled_levels_fit(rec, cdr_phase=None, n_pre=4, n_post=12):
    """Settled (steady-state) PAM4 levels from a linear pulse-response fit.

    Fits ``y(n) = sum_j h_j x(n-j) + b`` at the CDR phase of the
    unequalised waveform (``x`` = normalised PAM4 levels of the reference)
    and returns ``x_k * sum(h) + b``: the level a long run of symbol ``k``
    settles to.  Unlike a class mean at the sampling instant it is not
    biased by ISI, so it is the pattern-independent estimate of the same
    quantity the long-run OMA measurement reads.
    """
    spu = rec.spu
    centre = spu // 2 if cdr_phase is None else int(cdr_phase)
    x = PAM4_LEVELS[rec.labels]
    n = len(x)
    lo, hi = n_post, n - n_pre
    if hi - lo < 4 * (n_pre + n_post + 2):
        n_pre, n_post = 1, 3
        lo, hi = n_post, n - n_pre
    k = np.arange(lo, hi)
    cols = [x[k - j] for j in range(-n_pre, n_post + 1)]
    A = np.column_stack(cols + [np.ones(len(k))])
    y = rec.v[(rec.n[k] + rec.lag) * spu + rec.shift + centre]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    gain, b = float(np.sum(coef[:-1])), float(coef[-1])
    return PAM4_LEVELS * gain + b


def measure_levels(rec, cdr_phase=None, run_min=6):
    """Settled PAM4 levels from the centre 2 UI of long identical runs.

    Returns ``(levels[4], source)``.  Runs shorter than ``run_min`` symbols
    are ignored; a level with no qualifying run falls back to the linear
    pulse-response settled-level estimate (:func:`settled_levels_fit`).
    ``source`` is ``run``, ``pulse_fit`` or ``mixed``.
    """
    tru = rec.labels
    spu = rec.spu
    v = rec.v
    centre = spu // 2 if cdr_phase is None else int(cdr_phase)
    out = np.full(4, np.nan)
    src = []
    fit = None
    # run boundaries
    change = np.nonzero(np.diff(tru) != 0)[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(tru)]])
    for lvl in range(4):
        sel = [(s, e) for s, e in zip(starts, ends)
               if tru[s] == lvl and e - s >= run_min]
        if sel:
            acc = []
            for s, e in sel:
                # run centre in samples (symbol k peak at slot n_k+lag,
                # phase ``centre``); take +/-1 UI around the run midpoint
                mid_slot = rec.n[s] + rec.lag + 0.5 * (e - s - 1)
                mid = int(round(mid_slot * spu + rec.shift + centre))
                acc.append(v[mid - spu:mid + spu])
            out[lvl] = float(np.mean(np.concatenate(acc)))
            src.append("run")
        else:
            if fit is None:
                fit = settled_levels_fit(rec, cdr_phase)
            out[lvl] = float(fit[lvl])
            src.append("pulse_fit")
    if not np.all(np.isfinite(out)):
        raise ValueError("cannot measure all four PAM4 levels")
    source = "run" if all(s == "run" for s in src) else (
        "pulse_fit" if all(s == "pulse_fit" for s in src) else "mixed")
    return out, source


def thresholds(levels, pave, mode="linear"):
    lv = np.asarray(levels, dtype=float)
    oma = float(lv[3] - lv[0])
    if mode == "linear":
        return np.array([pave - oma / 3.0, pave, pave + oma / 3.0])
    return 0.5 * (lv[:-1] + lv[1:])


# ---------------------------------------------------------------------------
# eye centre / windows
# ---------------------------------------------------------------------------


def eye_centre(z, thr):
    """Eye centre phase (float, 0..spu) from the threshold crossings.

    Uses the flattened waveform so crossings across symbol boundaries are
    included (circular mean of the crossing phases + 0.5 UI).
    """
    spu = z.shape[1]
    s = z.ravel()
    a, b = s[:-1], s[1:]
    k = np.nonzero((a - thr) * (b - thr) < 0)[0]
    if len(k) == 0:
        return spu / 2.0
    pos = (k % spu) + (thr - a[k]) / (b[k] - a[k])
    ang = pos * (2.0 * np.pi / spu)
    cm = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
    return float((cm / (2.0 * np.pi) * spu + spu / 2.0) % spu)


def window_phases(centre, spu, offset_ui=0.05, width_ui=0.04):
    """Integer phases inside the left/right 0.04-UI histogram windows."""
    ph = np.arange(spu)
    d = ((ph - centre + spu / 2.0) % spu - spu / 2.0) / spu

    def pick(sign):
        lo = sign * offset_ui - width_ui / 2.0
        hi = sign * offset_ui + width_ui / 2.0
        sel = ph[(d >= lo - 1e-12) & (d <= hi + 1e-12)]
        if len(sel) == 0:
            sel = ph[[int(np.argmin(np.abs(d - sign * offset_ui)))]]
        return sel

    return pick(-1.0), pick(+1.0)


# ---------------------------------------------------------------------------
# histogram sigma fit
# ---------------------------------------------------------------------------


class _HistFit:
    """Unlabelled vertical-histogram Gaussian-noise fit (adjacent thresholds)."""

    def __init__(self, pave, oma, pth, nbins=2048, target=TARGET_SER):
        self.lo = pave - 2.0 * oma
        self.bw = 4.0 * oma / nbins
        self.nb = int(nbins)
        self.centers = self.lo + (np.arange(self.nb) + 0.5) * self.bw
        self.pth = np.asarray(pth, dtype=float)
        self.target = float(target)
        self.oma = float(oma)
        # per-bin distances to the (up to two) adjacent thresholds
        r = np.searchsorted(self.pth, self.centers)
        lower = np.where(r > 0, self.centers - self.pth[np.maximum(r - 1, 0)],
                         np.inf)
        upper = np.where(r < 3, self.pth[np.minimum(r, 2)] - self.centers,
                         np.inf)
        self.d = np.stack([lower, upper], axis=1)

    def hist(self, samples):
        pos = (np.asarray(samples, dtype=float).ravel() - self.lo) / self.bw - 0.5
        pos = np.clip(pos, 0.0, self.nb - 1.000001)
        i0 = pos.astype(np.int64)
        fr = pos - i0
        h = (np.bincount(i0, 1.0 - fr, self.nb)
             + np.bincount(i0 + 1, fr, self.nb + 1)[:self.nb])
        return h / h.sum()

    def sigma(self, F):
        """Gaussian sigma for which the histogram SER equals the target."""
        nz = F > 0
        F, d = F[nz], self.d[nz]
        lo, hi = np.log(self.oma * 1e-7), np.log(self.oma * 4.0)

        def ser(ls):
            return float(np.sum(F[:, None] * _Q(d / np.exp(ls))))

        if ser(lo) >= self.target:
            return float(np.exp(lo))
        if ser(hi) <= self.target:
            return float(np.exp(hi))
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            if ser(mid) > self.target:
                hi = mid
            else:
                lo = mid
        return float(np.exp(0.5 * (lo + hi)))


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def _feasible(c, m, cfg, b1):
    if not np.all(np.isfinite(c)) or abs(c.sum() - 1.0) > 1e-6:
        return False
    if cfg.bounded:
        ok, _ = fb.check_feasible(c, m, b1=b1)
        return ok
    return int(np.argmax(np.abs(c))) in cfg.main_candidates


def _kkt_unit_gain(G, g):
    """argmin c^T G c - 2 g^T c  subject to  sum(c) = 1."""
    N = len(g)
    K = np.zeros((N + 1, N + 1))
    K[:N, :N] = G
    K[:N, N] = K[N, :N] = 1.0
    return np.linalg.solve(K, np.concatenate([g, [1.0]]))[:N]


def _bounded_gram_qp(G, g, m, b1=0.0):
    """Convex QP: min c^T G c - 2 g^T c, unit gain, draft box + diff limit.

    The differential limit ``|c(+1) - c(-1) - b1 c(0)| <= lim c(0)`` is
    linear for a fixed ``b1``, so the feasible set is a polytope and SLSQP
    on the (small, normalised) Gram form converges reliably.
    """
    from scipy.optimize import minimize
    N = len(g)
    lb, ub = fb.bounds_for(N, m)
    x0 = np.zeros(N)
    x0[m] = 1.0
    lim = fb.FFE_BOUNDS.diff_limit
    cons = [{"type": "eq", "fun": lambda c: c.sum() - 1.0,
             "jac": lambda c: np.ones(N)}]
    if 0 < m < N - 1:
        a = np.zeros(N)
        a[m + 1], a[m - 1], a[m] = 1.0, -1.0, -b1
        e = np.zeros(N)
        e[m] = lim
        cons += [{"type": "ineq", "fun": lambda c: (e - a) @ c,
                  "jac": lambda c: e - a},
                 {"type": "ineq", "fun": lambda c: (e + a) @ c,
                  "jac": lambda c: e + a}]
    res = minimize(lambda c: c @ G @ c - 2.0 * g @ c, x0,
                   jac=lambda c: 2.0 * (G @ c - g), method="SLSQP",
                   bounds=list(zip(lb, ub)), constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-14})
    return np.asarray(res.x, dtype=float)


class _Problem:
    """TDECQ objective for one FFE window position (fixed main index)."""

    def __init__(self, rec, cfg, window, m, fit, oma, pave, R):
        self.rec, self.cfg, self.window, self.m = rec, cfg, window, m
        self.fit, self.oma, self.pave, self.R = fit, oma, pave, R
        self.set_phases(rec.spu / 2.0)

    def set_phases(self, centre):
        self.centre = centre
        self.pL, self.pR = window_phases(centre, self.rec.spu,
                                         self.cfg.window_offset_ui,
                                         self.cfg.window_width_ui)
        ph = np.concatenate([self.pL, self.pR])
        self.S = self.rec.taps(self.window, self.cfg.n_taps, ph)   # (N,n,P)
        self.nL = len(self.pL)
        self.fb = (0.5 * self.oma * PAM4_LEVELS[self.rec.prev])[:, None]

    def output(self, c, b1):
        z = np.tensordot(c, self.S, axes=(0, 0))
        if b1:
            z = z - b1 * self.fb
        return z

    def metric_from_output(self, z, c):
        sL = self.fit.sigma(self.fit.hist(z[:, :self.nL]))
        sR = self.fit.sigma(self.fit.hist(z[:, self.nL:]))
        s = min(sL, sR) if self.cfg.sigma_combine == "min" else max(sL, sR)
        ceq = noise_enhancement(c, self.cfg.ceq_model, self.R)
        sg = s / ceq
        den = 6.0 * QT * np.sqrt(sg * sg + self.cfg.sigma_s ** 2)
        return 10.0 * np.log10(self.oma / den), dict(
            sigma_L=sL, sigma_R=sR, sigma_G=sg, Ceq=ceq)

    def metric(self, c, b1=0.0):
        return self.metric_from_output(self.output(c, b1), c)[0]

    # -- starting point: constrained LS to the measured levels at the windows
    def ls_start(self):
        """Unit-gain LS fit to the settled levels (+ bounds / DFE grid).

        Works on the normalised N x N Gram system (waveform centred on
        P_ave and scaled by OMA) so the bounded QP is well conditioned.  No
        bias term is needed: the targets are the measured absolute levels
        and ``sum(c) = 1`` preserves the DC.
        """
        N, m = self.cfg.n_taps, self.m
        P = self.S.shape[2]
        X = (self.S.transpose(1, 2, 0).reshape(-1, N) - self.pave) / self.oma
        y = np.repeat(0.5 * PAM4_LEVELS[self.rec.labels], P)
        f = np.repeat(self.fb[:, 0], P) / self.oma
        G = X.T @ X / len(y)
        G += 1e-9 * np.trace(G) / N * np.eye(N)
        gx = X.T @ y / len(y)
        gf = X.T @ f / len(y)
        if not self.cfg.bounded:
            return _kkt_unit_gain(G, gx), 0.0
        b1_grid = ((0.0,) + tuple(fb.FFE_BOUNDS.b1_candidates[1:-1])
                   + (fb.FFE_BOUNDS.b1_max - 0.01,)) if self.cfg.dfe \
            else (0.0,)
        best = None
        for b1 in b1_grid:
            g = gx + b1 * gf                    # residual X c - b1 f - y
            c = _kkt_unit_gain(G, g)
            if not _feasible(c, m, self.cfg, b1):
                c = _bounded_gram_qp(G, g, m, b1)
            if not _feasible(c, m, self.cfg, b1):
                continue
            mt = self.metric(c, b1)
            if best is None or mt < best[0]:
                best = (mt, c, b1)
        if best is None:
            ident = np.zeros(N)
            ident[m] = 1.0
            return ident, 0.0
        return best[1], float(best[2])

    # -- derivative-free constrained pattern search on the true metric
    def refine(self, c, b1):
        cfg, m = self.cfg, self.m
        best = self.metric(c, b1)
        evals = 1
        step = 0.02
        dims = [i for i in range(cfg.n_taps) if i != m]
        z = self.output(c, b1)
        while step >= 1.25e-3 and evals < cfg.max_evals:
            moved = False
            for i in dims + (["b1"] if cfg.dfe else []):
                for d in (step, -step):
                    if i == "b1":
                        ct, bt = c, b1 + d
                        if bt < 0.0 or bt >= fb.FFE_BOUNDS.b1_max - 1e-6:
                            continue
                        zt = z - d * self.fb
                    else:
                        ct = c.copy()
                        ct[i] += d
                        ct[m] -= d                      # keep sum(c) = 1
                        bt = b1
                        zt = z + d * (self.S[i] - self.S[m])
                    if not _feasible(ct, m, cfg, bt):
                        continue
                    mt, _ = self.metric_from_output(zt, ct)
                    evals += 1
                    if mt < best - 1e-9:
                        c, b1, z, best, moved = ct, bt, zt, mt, True
                if evals >= cfg.max_evals:
                    break
            if not moved:
                step *= 0.5
        return c, b1, best, evals


def measure(v, spu, tru, cdr0, cfg, baud=None, lag=None):
    """Standard TDECQ/TECQ of an aligned record.

    Parameters
    ----------
    v : uniformly sampled analysis waveform.
    spu : samples per UI.
    tru : aligned reference symbols; ``tru[n]`` belongs to the sample slot
        ``n`` (CDR phase ``cdr0``) up to the main-cursor ``lag`` that is
        re-estimated here.
    cdr0 : approximate sampling phase (eye centre) of the raw waveform.
    cfg : :class:`StdConfig`.

    Returns
    -------
    dict with the metric (``metric_db``), the optimised taps (IEEE order:
    pre-cursors first), ``b1``, levels/OMA/P_ave, sigma decomposition,
    window phases, SER and per-candidate diagnostics.
    """
    # Raw eye centre from the flattened waveform (crossings across UI
    # boundaries included).  The legacy ``cdr0`` estimate drops crossings
    # that fall on a UI boundary and can be half a UI off; a wrong phase
    # here would shift the symbol association of the SER labels and of the
    # DFE feedback by one symbol.
    v_arr = np.asarray(v, dtype=float)
    n_full = len(v_arr) // int(spu)
    if n_full >= 64:
        cdr_raw = eye_centre(v_arr[:n_full * int(spu)].reshape(-1, int(spu)),
                             float(np.mean(v_arr)))
        cdr0 = int(round(cdr_raw)) % int(spu)
    if lag is None:
        lag = estimate_lag(v, spu, tru, cdr0)
    rec = _Record(v, spu, tru, cdr0, lag, cfg.n_taps, cfg.main_candidates)
    levels, lvl_src = measure_levels(rec, run_min=cfg.oma_run_min)
    oma = float(levels[3] - levels[0])
    if not oma > 0:
        raise ValueError("non-positive OMA_outer")
    first = (rec.n[0] + rec.lag + cfg.n_taps) * spu
    last = (rec.n[-1] + rec.lag - cfg.n_taps) * spu
    pave = float(np.mean(rec.v[first:last]))
    pth = thresholds(levels, pave, cfg.threshold_mode)
    fit = _HistFit(pave, oma, pth, cfg.hist_bins)
    R = (bt4_noise_autocorr(cfg.n_taps, cfg.bt4_bw_over_baud)
         if cfg.ceq_model == "bt4" else None)

    cands = []
    for m in cfg.main_candidates:
        window = rec.lag + m
        prob = _Problem(rec, cfg, window, m, fit, oma, pave, R)
        ident = np.zeros(cfg.n_taps)
        ident[m] = 1.0
        # raw (identity) eye centre defines the initial windows
        prob.set_phases(eye_centre(rec.equalize(ident, window), pth[1]))
        raw_db = prob.metric(ident)
        c0, b0 = prob.ls_start()
        c0, b0 = np.asarray(c0, float), float(b0)
        starts = [(ident, 0.0)]
        ls_ok = _feasible(c0, m, cfg, b0)
        if ls_ok:
            starts.append((c0, b0))
        ls_db = prob.metric(c0, b0) if np.all(np.isfinite(c0)) else np.inf
        best = None
        evals = 0
        for c, b1 in starts:
            # alternate: optimise at fixed windows, then re-centre the eye
            for _ in range(3):
                c, b1, mdb, ne_ = prob.refine(c, b1)
                evals += ne_
                ctr = eye_centre(rec.equalize(c, window, b1, oma), pth[1])
                if abs(ctr - prob.centre) < 0.5:
                    break
                prob.set_phases(ctr)
            mdb = prob.metric(c, b1)
            if best is None or mdb < best[0]:
                best = (mdb, c.copy(), b1, prob.centre)
        prob.set_phases(best[3])
        mdb, det = prob.metric_from_output(prob.output(best[1], best[2]),
                                           best[1])
        cands.append(dict(m=m, window=window, metric_db=float(mdb),
                          raw_db=float(raw_db), c=best[1], b1=float(best[2]),
                          centre=float(best[3]), pL=prob.pL, pR=prob.pR,
                          ls_feasible=ls_ok, ls_db=float(ls_db), c0=c0,
                          b0=b0, evals=int(evals),
                          **det))
    best = min(cands, key=lambda r: r["metric_db"])
    c, b1, m = best["c"], best["b1"], best["m"]
    z = rec.equalize(c, best["window"], b1, oma)
    ph = int(round(best["centre"])) % spu
    dec = np.searchsorted(pth, z[:, ph])
    ne = int(np.sum(dec != rec.labels))
    post = np.array([z[rec.labels == k, ph].mean() if np.any(rec.labels == k)
                     else np.nan for k in range(4)])
    ident = np.zeros(cfg.n_taps)
    ident[m] = 1.0
    zr = rec.equalize(ident, best["window"])
    ne_raw = int(np.sum(np.searchsorted(pth, zr[:, ph]) != rec.labels))
    zl = rec.equalize(best["c0"], best["window"], best["b0"], oma)
    ne_ls = int(np.sum(np.searchsorted(pth, zl[:, ph]) != rec.labels))
    return dict(
        metric_db=float(best["metric_db"]), raw_db=float(best["raw_db"]),
        ls_db=float(best["ls_db"]), ls_ser=ne_ls / float(len(rec.labels)),
        ls_feasible=bool(best["ls_feasible"]),
        # IEEE naming: the largest tap is the main cursor; taps before it are
        # pre-cursor taps (later samples), taps after it post-cursor taps
        taps=np.asarray(c, dtype=float), b1=float(b1),
        main_index=int(np.argmax(np.abs(c))),
        n_pre=int(np.argmax(np.abs(c))),
        n_post=int(cfg.n_taps - 1 - np.argmax(np.abs(c))),
        largest_tap_index=int(np.argmax(np.abs(c))), search_main_index=int(m),
        lag=int(rec.lag), window=int(best["window"]), shift=int(rec.shift),
        sym_start=int(rec.n[0]), sym_stop=int(rec.n[-1] + 1),
        levels=levels, level_source=lvl_src, oma=oma, pave=pave, pth=pth,
        post_levels=post, sigma_G=float(best["sigma_G"]),
        sigma_L=float(best["sigma_L"]), sigma_R=float(best["sigma_R"]),
        Ceq=float(best["Ceq"]), eye_centre=float(best["centre"]),
        window_left=[int(p) for p in best["pL"]],
        window_right=[int(p) for p in best["pR"]],
        ser=ne / float(len(rec.labels)), ne=ne, ne_raw=ne_raw,
        n_sym=int(len(rec.labels)),
        feasible=bool(_feasible(c, m, cfg, b1)),
        candidates=[dict(main_index=r["m"], metric_db=r["metric_db"],
                         raw_db=r["raw_db"], evals=r["evals"],
                         ls_start_feasible=r["ls_feasible"])
                    for r in cands],
        config=dict(n_taps=cfg.n_taps, main_candidates=list(
            cfg.main_candidates), bounded=cfg.bounded, dfe=cfg.dfe,
            ceq_model=cfg.ceq_model, sigma_combine=cfg.sigma_combine,
            sigma_s=cfg.sigma_s, window_offset_ui=cfg.window_offset_ui,
            window_width_ui=cfg.window_width_ui, hist_bins=cfg.hist_bins,
            threshold_mode=cfg.threshold_mode),
    )


def equalized_eye(v, spu, tru, res):
    """Rebuild the equalised eye of a :func:`measure` result (for plots).

    Returns ``(z, labels, centre_phase)`` with ``z`` of shape (n_sym, spu).
    """
    rec = _Record.__new__(_Record)
    rec.v = np.asarray(v, dtype=float)
    rec.spu = int(spu)
    rec.tru = np.asarray(tru, dtype=np.int64)
    rec.lag = int(res["lag"])
    rec.shift = int(res["shift"])
    rec.n = np.arange(int(res["sym_start"]), int(res["sym_stop"]))
    rec.labels = rec.tru[rec.n]
    rec.prev = rec.tru[rec.n - 1]
    z = rec.equalize(np.asarray(res["taps"], float), int(res["window"]),
                     float(res.get("b1", 0.0)), float(res["oma"]))
    return z, rec.labels, int(round(res["eye_centre"])) % int(spu)
