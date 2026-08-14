# -*- coding: utf-8 -*-
"""Equalisation, sigma fitting and the TECQ/TDECQ metric.

This module implements the IEEE 802.3-2022 style vertical eye-closure metric:

* ``build_Mj`` - stack the FFE tap-rows of the waveform (configurable
  ``n_pre`` pre-cursor / ``n_post`` post-cursor taps, default 5-tap);
* ``tecq_of`` / ``tecq_hist_of`` - sample-based and 64-bin vertical-histogram
  Gaussian-noise sigma fits at the three PAM4 sub-eye thresholds
  (Pth = Pave - OMA/3, Pave, Pave + OMA/3) using the 0.45/0.55 UI window
  samples (samples at phases cdr-4, cdr-3 / cdr+2, cdr+3 modulo spu);
* the metric value in dB:  TDECQ = 10*log10(OMA / (6 * 3.414 * sigma_G))
  with sigma_G = min(sigma_L, sigma_R) / sqrt(sum(w^2));
* the unit-gain FFE candidate families (identity, damped MMSE, constrained
  subset LS, weak progressive CD) used by the automatic pipeline
  (identity, damped MMSE, constrained subset LS, weak progressive CD);
* automatic CDR (crossing based) estimation.

All quantities are in the internal units of the pipeline (mV for electrical,
mW for optical).
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc


def Q(x):
    """Gaussian tail probability Q(x) = 0.5*erfc(x/sqrt(2))."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


# ---------------------------------------------------------------------------
# waveform matrix / levels
# ---------------------------------------------------------------------------


def build_Mj(v: np.ndarray, spu: int, n_sym: int,
             n_pre: int = 0, n_post: int = 4) -> np.ndarray:
    """Stack the FFE tap rows of the uniformly sampled waveform.

    Tap ``i`` of the returned stack (row index) samples the signal at a
    delay of ``(i - n_pre)`` UI relative to the *main tap* at index
    ``n_pre`` (delay 0, the alignment anchor).  Taps before the main tap
    are the pre-cursor taps (samples earlier in the record), taps after it
    the post-cursor taps.  The total number of taps is ``n_pre + 1 + n_post``.

    When ``n_pre > 0`` the leading ``n_pre * spu`` samples required by the
    pre-cursor taps are synthesised by holding the first sample value
    (zero-order hold) so the equaliser can be applied over the whole record.
    For the default ``(n_pre=0, n_post=4)`` this reduces to the classic
    5-tap stack (rows at delays 0..4 UI) and the output is bit-identical to
    the pre-generalisation implementation.

    Parameters
    ----------
    v     : uniformly sampled waveform, length >= (n_sym + n_pre + n_post)*spu.
    spu   : samples per UI.
    n_sym : number of symbols.
    n_pre : pre-cursor tap count (taps before the main tap), 0..5.
    n_post: post-cursor tap count (taps after the main tap), 0..30.

    Returns
    -------
    np.ndarray of shape (n_pre + 1 + n_post, n_sym, spu).
    """
    T = n_pre + 1 + n_post
    if n_pre > 0:
        pad = np.broadcast_to(np.asarray(v)[:1], (n_pre * spu,))
        v = np.concatenate([pad, np.asarray(v)])
    return np.stack([v[o:o + n_sym * spu].reshape(n_sym, spu)
                     for o in [j * spu for j in range(T)]])


def identity_taps(n_pre: int, n_post: int) -> np.ndarray:
    """Unit-gain identity tap vector (no equalisation) for an FFE geometry.

    The main tap (value 1.0) is placed at index ``n_pre`` so that the
    identity filter reads exactly the alignment-anchored sample (delay 0).

    Returns
    -------
    np.ndarray of length ``n_pre + 1 + n_post``.
    """
    T = n_pre + 1 + n_post
    e = np.zeros(T, dtype=float)
    e[n_pre] = 1.0
    return e


def levels_at(M: np.ndarray, tru: np.ndarray, cdr: int) -> np.ndarray:
    """Per-class mean levels of the equalised signal at sampling phase ``cdr``.

    Empty classes (degenerate 3-level patterns such as PRBS2, whose m-sequence
    only occupies three of the four PAM4 levels) are linearly interpolated
    between the neighbouring populated class means so the metric pipeline
    stays finite and warning-free.  For regular 4-level records every class is
    populated and the result is identical to the plain per-class mean.
    """
    return _class_means(M[:, cdr], tru)


def count_errors(M: np.ndarray, tru: np.ndarray, cdr: int) -> int:
    """Number of symbol errors at phase ``cdr`` using class-mean thresholds.

    Parameters
    ----------
    M   : equalised signal matrix (n_sym, spu) or (5, n_sym, spu) raw stack.
    tru : reference symbols.
    cdr : sampling phase (sample index inside the UI).

    Returns
    -------
    int number of mismatches between the decoded and reference symbols.
    """
    s = M[:, cdr] if M.ndim == 2 else M[0][:, cdr]
    lv = _class_means(s, tru)
    thr = [(lv[0] + lv[1]) / 2, (lv[1] + lv[2]) / 2, (lv[2] + lv[3]) / 2]
    dec = np.zeros(len(s), dtype=np.int64)
    dec[s > thr[0]] = 1
    dec[s > thr[1]] = 2
    dec[s > thr[2]] = 3
    return int((dec != tru).sum())


def _class_means(samples: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """Per-class means of ``samples`` grouped by ``cls`` (0..3) with empty-class
    interpolation.

    When a class contains no samples its mean is linearly interpolated between
    the neighbouring populated class means (a constant when fewer than two
    classes are populated).  ``cls`` values outside 0..3 are ignored.
    """
    out = np.empty(4, dtype=np.float64)
    for c in range(4):
        m = samples[cls == c]
        out[c] = m.mean() if len(m) else np.nan
    idx = np.nonzero(np.isfinite(out))[0]
    if len(idx) == 4:
        return out
    if len(idx) == 0:
        return np.zeros(4)
    if len(idx) == 1:
        return np.full(4, out[idx[0]])
    return np.interp(np.arange(4), idx, out[idx])


# ---------------------------------------------------------------------------
# sigma fits
# ---------------------------------------------------------------------------


def implied_sigma_pair(dL: np.ndarray, dR: np.ndarray, target: float = 4.8e-4,
                       iters: int = 40) -> np.ndarray:
    """Bisection sigma fit for the left/right window sample pools.

    Finds the Gaussian sigma (per side) for which the pooled symbol error
    ratio, summed over the three thresholds, equals ``target`` (default
    4.8e-4, the IEEE 802.3 value).

    Parameters
    ----------
    dL, dR : |sample - Pth| distance arrays for the left/right windows.
    target : target SER.
    iters  : bisection iterations.

    Returns
    -------
    np.ndarray (sL, sR).
    """
    if (not np.all(np.isfinite(dL)) or not np.all(np.isfinite(dR))
            or len(dL) == 0 or len(dR) == 0):
        return np.array([np.nan, np.nan])
    d = np.stack([dL, dR])

    def ser(sig):
        return np.mean(np.sum(Q(d / sig[:, None, None]), axis=2), axis=1)

    lo = np.full(2, 1e-9)
    hi = np.full(2, 1e4)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = ser(mid)
        over = s > target
        hi[over] = mid[over]
        lo[~over] = mid[~over]
    return 0.5 * (lo + hi)


def _hist_sigma(centers, F, Pth, target=4.8e-4, grid=None):
    """Gaussian sigma whose convolution over a vertical histogram reaches ``target`` SER.

    Parameters
    ----------
    centers : histogram bin centres.
    F       : normalised bin fractions.
    Pth     : three PAM4 thresholds.
    target  : target SER.
    grid    : sigma grid (log spaced); defaults to the module-level grid.

    Returns
    -------
    float sigma (same units as the samples).
    """
    if grid is None:
        grid = _SIG_GRID
    d = np.abs(centers[:, None] - np.asarray(Pth, dtype=float)[None, :])
    ser = np.sum(F[:, None] * Q(d[None, :, :] / grid[:, None, None]), axis=(1, 2))
    i = np.where(ser >= target)[0]
    if len(i) == 0:
        return float(grid[-1])
    i = int(i[0])
    if i == 0:
        return float(grid[0])
    return float(np.interp(target, ser[i - 1:i + 1], grid[i - 1:i + 1]))


_SIG_GRID = np.logspace(-3.0, 3.0, 1200)


def window_samples(M: np.ndarray, cdr: int, spu: int):
    """Left/right 0.45/0.55 UI window samples of the equalised eye.

    Returns
    -------
    (sL_all, sR_all) concatenated sample arrays (window at cdr-4,cdr-3 and
    cdr+2,cdr+3 modulo spu).
    """
    sL_all = np.concatenate([M[:, (cdr - 4) % spu], M[:, (cdr - 3) % spu]])
    sR_all = np.concatenate([M[:, (cdr + 2) % spu], M[:, (cdr + 3) % spu]])
    return sL_all, sR_all


def _sigma_from_metric(metric: float, oma: float, ceq: float) -> float:
    sig = oma / (6.0 * 3.414 * 10.0 ** (metric / 10.0)) / ceq
    return sig if sig > 0 else float("nan")


def tecq_of(M, w, tru, cdr, spu, target=4.8e-4, iters=40):
    """Sample-based TECQ/TDECQ metric.

    Parameters
    ----------
    M   : equalised signal matrix (n_sym, spu).
    w   : 5 FFE taps.
    tru : reference symbols.
    cdr : sampling phase.
    spu : samples per UI.

    Returns
    -------
    (metric_db, detail_dict) with detail keys levels, Pave, Pth, OMA, sL, sR,
    Ceq, sigmaG.
    """
    lv = levels_at(M, tru, cdr)
    OMA = lv[3] - lv[0]
    if (not np.all(np.isfinite(lv)) or not np.isfinite(OMA) or OMA <= 1e-12):
        return 1e9, None
    Pave = lv.mean()
    Pth = np.array([Pave - OMA / 3.0, Pave, Pave + OMA / 3.0])
    dL, dR = window_samples(M, cdr, spu)
    dL = np.abs(dL[:, None] - Pth[None, :])
    dR = np.abs(dR[:, None] - Pth[None, :])
    sL, sR = implied_sigma_pair(dL, dR, target=target, iters=iters)
    Ceq = np.sqrt((w ** 2).sum())
    sig = min(sL, sR) / Ceq
    if not np.isfinite(sig) or sig <= 1e-12:
        return 1e9, None
    tec = 10.0 * np.log10(OMA / (6.0 * 3.414 * sig))
    return tec, dict(levels=lv, Pave=Pave, Pth=Pth, OMA=OMA, sL=sL, sR=sR,
                     Ceq=Ceq, sigmaG=sig)


def tecq_hist_of(M, w, tru, cdr, spu, nbins=64, target=4.8e-4):
    """64-bin vertical-histogram TECQ/TDECQ metric.

    This is the release-compatible histogram fit: each window pool is binned
    over its observed min/max range with the requested number of equally
    spaced bins.  It is deterministic for a given equalised signal and is
    retained as the default ``hist`` method so the 33-case release regression
    is unchanged.
    """
    lv = levels_at(M, tru, cdr)
    OMA = lv[3] - lv[0]
    if (not np.all(np.isfinite(lv)) or not np.isfinite(OMA) or OMA <= 1e-12):
        return 1e9, None
    Pave = lv.mean()
    Pth = np.array([Pave - OMA / 3.0, Pave, Pave + OMA / 3.0])
    sL_all, sR_all = window_samples(M, cdr, spu)

    def fit(samps):
        lo, hi = float(samps.min()), float(samps.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return float("inf")
        hist, edges = np.histogram(samps, bins=nbins, range=(lo, hi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        F = hist.astype(float) / hist.sum()
        return _hist_sigma(centers, F, Pth, target)

    sL = fit(sL_all)
    sR = fit(sR_all)
    Ceq = np.sqrt((w ** 2).sum())
    sig = min(sL, sR) / Ceq
    if not np.isfinite(sig) or sig <= 1e-12:
        return 1e9, None
    tec = 10.0 * np.log10(OMA / (6.0 * 3.414 * sig))
    return tec, dict(levels=lv, Pave=Pave, Pth=Pth, OMA=OMA, sL=sL, sR=sR,
                     Ceq=Ceq, sigmaG=sig)


def tecq_hist_ieee_of(M, w, tru, cdr, spu, nbins=512, target=4.8e-4):
    """IEEE fixed-range vertical-histogram TECQ/TDECQ metric.

    Uses equally spaced bins spanning ``Pave - 2*OMA`` to ``Pave + 2*OMA``
    (IEEE 802.3-2022 Clause 121.8.5.3 / 122.8.5.3).  Unlike the min/max
    release histogram, the range is anchored to the PAM4 OMA and is robust for
    strongly closed-eye LC records whose raw min/max window samples make the
    default histogram unstable.
    """
    lv = levels_at(M, tru, cdr)
    OMA = lv[3] - lv[0]
    if (not np.all(np.isfinite(lv)) or not np.isfinite(OMA) or OMA <= 1e-12):
        return 1e9, None
    Pave = lv.mean()
    Pth = np.array([Pave - OMA / 3.0, Pave, Pave + OMA / 3.0])
    sL_all, sR_all = window_samples(M, cdr, spu)

    edges = np.linspace(Pave - 2.0 * OMA, Pave + 2.0 * OMA, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def fit(samps):
        hist, _ = np.histogram(samps, bins=edges)
        if hist.sum() <= 0:
            return float("inf")
        F = hist.astype(float) / hist.sum()
        return _hist_sigma(centers, F, Pth, target)

    sL = fit(sL_all)
    sR = fit(sR_all)
    Ceq = np.sqrt((w ** 2).sum())
    sig = min(sL, sR) / Ceq
    if not np.isfinite(sig) or sig <= 1e-12:
        return 1e9, None
    tec = 10.0 * np.log10(OMA / (6.0 * 3.414 * sig))
    return tec, dict(levels=lv, Pave=Pave, Pth=Pth, OMA=OMA, sL=sL, sR=sR,
                     Ceq=Ceq, sigmaG=sig)


def metric_detail(M, w, tru, cdr, spu, method="hist"):
    """Unified metric dispatch.  method in {'hist','hist_ieee','samp'}.

    Returns
    -------
    (metric_db, detail_dict or None).
    """
    if method == "hist":
        return tecq_hist_of(M, w, tru, cdr, spu)
    if method == "hist_ieee":
        return tecq_hist_ieee_of(M, w, tru, cdr, spu)
    return tecq_of(M, w, tru, cdr, spu)


# ---------------------------------------------------------------------------
# FFE
# ---------------------------------------------------------------------------


def mmse_taps(Mj: np.ndarray, cdr: int, tru: np.ndarray, nom: np.ndarray,
              reg: float = 1e-9) -> np.ndarray:
    """Unit-gain MMSE equaliser fitted to nominal levels ``nom[tru]``.

    Parameters
    ----------
    Mj  : stacked tap rows (n_taps, n_sym, spu).
    cdr : sampling phase.
    tru : reference symbols.
    nom : 4 PAM4 target levels.
    reg : ridge regularisation.

    Returns
    -------
    np.ndarray of ``n_taps`` weights normalised to unit sum.
    """
    X = Mj[:, :, cdr].T
    w = np.linalg.solve(X.T @ X + reg * np.eye(Mj.shape[0]), X.T @ nom[tru])
    s = w.sum()
    return w / s if abs(s) > 1e-12 else w


def subset_ls_taps(Mj: np.ndarray, cdr: int, tru: np.ndarray, nom: np.ndarray,
                   subset) -> np.ndarray:
    """Unit-gain least-squares FFE restricted to a tap subset.

    Solves  min ||X_S w - nom[tru]||^2  subject to  sum(w) = 1  with the
    non-selected taps fixed at zero.

    Parameters
    ----------
    Mj, cdr, tru, nom : as in :func:`mmse_taps`.
    subset : iterable of tap indices (e.g. (0,1), (1,2,3)).

    Returns
    -------
    np.ndarray of ``n_taps`` weights (zeros outside the subset).
    """
    X = Mj[:, :, cdr].T
    cols = list(subset)
    A = X[:, cols]
    t = nom[tru]
    G = A.T @ A + 1e-9 * np.eye(len(cols))
    K = np.zeros((len(cols) + 1, len(cols) + 1))
    K[:len(cols), :len(cols)] = G
    K[:len(cols), len(cols)] = 1.0
    K[len(cols), :len(cols)] = 1.0
    b = np.concatenate([A.T @ t, [1.0]])
    sol = np.linalg.solve(K, b)
    w = np.zeros(Mj.shape[0])
    w[cols] = sol[:len(cols)]
    return w


def damped_mmse_taps(Mj: np.ndarray, cdr: int, tru: np.ndarray, nom: np.ndarray,
                     alpha: float, n_pre: int = 0, n_post: int = 4) -> np.ndarray:
    """Convex combination of the identity and the MMSE equaliser.

    ``w = (1-alpha)*e0 + alpha*w_mmse`` (both unit-gain, so the result is too).

    Parameters
    ----------
    alpha : mixing factor in [0, 1]; 0 = identity, 1 = plain MMSE.
    n_pre, n_post : FFE geometry (main tap at index ``n_pre``).
    """
    wm = mmse_taps(Mj, cdr, tru, nom)
    e0 = identity_taps(n_pre, n_post)
    return (1.0 - alpha) * e0 + alpha * wm


def progressive_cd_taps(Mj: np.ndarray, cdr: int, tru: np.ndarray, nom: np.ndarray,
                        spu: int, step: float = 0.02, max_iter: int = 10,
                        maxerr: int = 0, n_pre: int = 0, n_post: int = 4) -> np.ndarray:
    """Weak progressive coordinate descent from the identity equaliser.

    Performs ``max_iter`` coordinate sweeps with step ``step``, always
    renormalising to unit gain and only accepting moves that keep the symbol
    error count at or below ``maxerr`` and lower the metric.  This reproduces
    the *early-stopped* character of the reference tool's "progressive
    coordinate descent" without per-case tuning.

    Parameters
    ----------
    Mj, cdr, tru, nom, spu : as usual.
    step, max_iter, maxerr : global algorithm parameters.

    Returns
    -------
    np.ndarray of 5 taps.
    """
    w = identity_taps(n_pre, n_post)
    M = np.tensordot(Mj, w, axes=(0, 0))
    best, _ = metric_detail(M, w, tru, cdr, spu, "samp")
    for _ in range(max_iter):
        improved = False
        for j in range(w.size):
            for sign in (1.0, -1.0):
                wt = w.copy()
                wt[j] += sign * step
                wt = wt / wt.sum()
                Mm = np.tensordot(Mj, wt, axes=(0, 0))
                if count_errors(Mm, tru, cdr) > maxerr:
                    continue
                t, _ = metric_detail(Mm, wt, tru, cdr, spu, "samp")
                if t < best - 1e-9:
                    w, M, best = wt, Mm, t
                    improved = True
        if not improved:
            break
    return w


# ---------------------------------------------------------------------------
# CDR
# ---------------------------------------------------------------------------


def crossing_phase(s: np.ndarray, spu: int, thr: float):
    """Mean crossing position (in samples, 0..spu) of threshold ``thr``.

    Parameters
    ----------
    s   : uniformly sampled waveform (or equalised signal).
    spu : samples per UI.
    thr : crossing threshold (typically Pave).

    Returns
    -------
    float or None when no crossing is found.
    """
    rows = s[:len(s) // spu * spu].reshape(-1, spu)
    a = rows[:, :-1]
    b = rows[:, 1:]
    cr = (a - thr) * (b - thr) < 0
    pos = []
    for j in range(spu - 1):
        col = cr[:, j]
        if col.any():
            frac = (thr - a[col, j]) / (b[col, j] - a[col, j])
            pos.append(j + frac)
    if not pos:
        return None
    pos = np.concatenate(pos)
    ph = pos * (2 * np.pi / spu)
    cm = np.arctan2(np.mean(np.sin(ph)), np.mean(np.cos(ph)))
    return (cm / (2 * np.pi) * spu) % spu


def cdr_from_crossing(s: np.ndarray, spu: int, thr: float = None) -> int:
    """Automatic CDR phase: circular-mean crossing phase + half a UI.

    Parameters
    ----------
    s   : uniformly sampled signal.
    spu : samples per UI.
    thr : crossing threshold; defaults to the mean of the 12.5/37.5/62.5/87.5
          percentiles (an estimate of Pave).

    Returns
    -------
    int sampling phase in 0..spu-1.
    """
    if thr is None:
        thr = float(np.percentile(s, [12.5, 37.5, 62.5, 87.5]).mean())
    ph = crossing_phase(s, spu, thr)
    if ph is None:
        return 0
    return int(round((ph + spu / 2.0) % spu)) % spu
