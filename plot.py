# -*- coding: utf-8 -*-
"""Diagnostic plotting for the TDECQ/TECQ tool (matplotlib, non-interactive).

``make_diagnostic_plot`` draws an IEEE 802.3 style density eye diagram of the
equalised PAM4 signal.  The eye panel follows the visual language of the
``bjarkeaf/pam4-link-sim`` reference visualisation: a power-normalised 2-D
density map on a dark background, the three PAM4 decision-threshold lines
annotated with their OMA definitions, and the two 0.45 UI / 0.55 UI histogram
windows marked as vertical bands.  The density map defaults to blue and can be
recoloured with ``eye_color``.

A compact waveform/sample overview panel is kept underneath.  The module uses
the matplotlib ``Agg`` backend so it works head-less; the CLI only calls it
when ``-p/--output-plot`` is given (``--no-plot`` disables it).

Reference
---------
The eye-diagram annotations and layout are adapted from
``bjarkeaf/pam4-link-sim`` (sim.py, ``plot_tdecq``).  This implementation is
independent and uses the data structures of the present clean-room pipeline.
"""
from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb  # noqa: E402

from . import eq  # noqa: E402
from .pipeline import prepare_plot_data  # noqa: E402
from . import ffe_bounds  # noqa: E402

# eye-diagram x-axis in UI (one symbol period)
_EYE_X = 1.0

# default eye-diagram density colour
DEFAULT_EYE_COLOR = "blue"

# IEEE 802.3 histogram window positions inside one UI (Clause 121.8.5.3)
_HIST_WINDOWS = ((0.45, "0.45 UI"), (0.55, "0.55 UI"))
_HIST_HALF_UI = 0.02


def _cmap_for(color):
    """Return an alpha-blended, black-to-``color`` density colormap.

    Parameters
    ----------
    color : str
        Any matplotlib colour specification (``"blue"``, ``"#1f77b4"``, ...).

    Returns
    -------
    LinearSegmentedColormap
        Transparent at zero density, saturated ``color`` at high density.
    """
    rgb = np.asarray(to_rgb(color), dtype=float)
    n = 256
    vals = np.zeros((n, 4), dtype=float)
    ramp = np.linspace(0.0, 1.0, n)
    vals[:, :3] = rgb[None, :] * ramp[:, None]
    vals[:, 3] = ramp
    return LinearSegmentedColormap.from_list("eye_%s" % color, vals)


def _eye_phase(M, cdr, spu):
    """Fold the equalised matrix into one UI, with ``cdr`` as time zero.

    ``M`` is an (n_sym, spu) matrix.  The returned matrix has column 0 at the
    sampling phase, so the 0.45/0.55 UI histogram windows correspond to the
    same phases as the pipeline's metric evaluation.
    """
    if cdr == 0:
        return M
    return np.concatenate([M[:, cdr:], M[:, :cdr]], axis=1)


def _density_eye(M, tru, cdr, spu, bins=(220, 220), xmax=1.0):
    """2-D histogram of the equalised signal over one UI.

    Samples of the equalised matrix ``M`` (n_sym, spu) are folded into one UI
    using the sampling phase ``cdr`` as the origin.  Returns (x, y, H) with
    x in UI (0..xmax) and y in the signal units.
    """
    n_sym = M.shape[0]
    eye = _eye_phase(M, cdr, spu)
    # Sampling phase is the centre of the displayed eye: column j of ``eye``
    # is phase (cdr+j)%spu, so its UI position is (0.5 + j/spu) modulo 1 UI.
    phase_frac = xmax * ((0.5 + np.arange(spu) / float(spu)) % 1.0)
    H, xe, ye = np.histogram2d(
        np.tile(phase_frac[None, :], (n_sym, 1)).ravel(),
        eye.ravel(), bins=bins,
        range=[[0.0, xmax], [float(eye.min()), float(eye.max())]])
    x = 0.5 * (xe[:-1] + xe[1:])
    y = 0.5 * (ye[:-1] + ye[1:])
    return x, y, H


def _collect_center_histograms(M, cdr, spu, p_ave, oma_outer, n_bins=512):
    """Collect normalised vertical histograms at 0.45/0.55 UI.

    The two windows are centred at 0.45 UI and 0.55 UI relative to the
    displayed eye, whose sampling phase is at 0.5 UI.  Bin centres span
    ``p_ave - 2*OMA`` to ``p_ave + 2*OMA``, matching the bjarkeaf
    ``collect_histograms`` visualisation.

    Returns
    -------
    (y_bins, F_left, F_right).
    """
    def window_samples(phase_frac):
        centre = (cdr + int(round((phase_frac - 0.5) * spu))) % spu
        half = int(round(0.02 * spu))
        cols = (centre + np.arange(-half, half + 1)) % spu
        return np.asarray(M)[:, cols].ravel()

    sL_all = window_samples(0.45)
    sR_all = window_samples(0.55)

    # Robust bin range.  Degenerate closed-eye fallback records can produce
    # non-finite / non-positive ``oma_outer`` (inverted class-mean levels);
    # in that case fall back to the observed window-sample min/max so the bin
    # edges are always strictly monotonic and np.histogram never raises.
    if not np.isfinite(p_ave):
        p_ave = 0.0
    if np.isfinite(oma_outer) and oma_outer > 0:
        lo, hi = p_ave - 2.0 * oma_outer, p_ave + 2.0 * oma_outer
    else:
        lo = float(np.min([sL_all.min(), sR_all.min()]))
        hi = float(np.max([sL_all.max(), sR_all.max()]))
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo, hi = p_ave - 1.0, p_ave + 1.0
        else:
            pad = 0.05 * (hi - lo)
            lo, hi = lo - pad, hi + pad
    y_edges = np.linspace(lo, hi, n_bins + 1)
    y_bins = 0.5 * (y_edges[:-1] + y_edges[1:])

    def hist_for(samples):
        counts, _ = np.histogram(samples, bins=y_edges)
        if counts.sum() <= 0:
            return np.zeros_like(y_bins)
        return counts.astype(float) / counts.sum()

    return y_bins, hist_for(sL_all), hist_for(sR_all)


def _format_tap_label(value, is_main=False, at_bound=None):
    """Format one FFE tap value for the suptitle with emphasis markers.

    The main tap is set in bold (``\\mathbf``); a tap touching a lower
    bound is underlined and one touching an upper bound overlined
    (``\\underline`` / ``\\overline``), so IEEE-802.3dj bounded runs show
    which cursors sit on a constraint.

    Parameters
    ----------
    value : float
        tap weight.
    is_main : bool
        main-tap emphasis (bold).
    at_bound : str or None
        ``"lb"`` / ``"ub"`` when the tap touches a bound (from
        :func:`ffe_bounds.at_bound_info`).

    Returns
    -------
    str a matplotlib-mathtext label.
    """
    s = "%.3f" % float(value)
    if at_bound == "lb":
        s = r"$\underline{%s}$" % s
    elif at_bound == "ub":
        s = r"$\overline{%s}$" % s
    if is_main:
        s = r"$\mathbf{%s}$" % s
    return s


def make_diagnostic_plot(input_csv, out, png_path, rate_gbps=112,
                         domain="electrical", samples_per_ui=64,
                         window="auto", case_id=None, pattern=None,
                         prbs_order=None, eye_color=DEFAULT_EYE_COLOR,
                         ffe_pre_cursor=0, ffe_post_cursor=4):
    """Render a diagnostic PNG for one measurement.

    Parameters
    ----------
    input_csv : str
        waveform CSV (auto-detected format).
    out : dict
        the SPEC section-5 output JSON produced by ``pipeline.analyze``.
    png_path : str
        destination PNG path.
    rate_gbps, domain, samples_per_ui, window, case_id, pattern, prbs_order :
        same semantics as ``run_tdecq.py``.
    eye_color : str
        colour of the equalised-eye density map (default ``"blue"``).
    ffe_pre_cursor, ffe_post_cursor : int
        FFE geometry (main tap at index ``ffe_pre_cursor``, total taps
        ``ffe_pre_cursor + 1 + ffe_post_cursor``).

    Returns
    -------
    str the PNG path.
    """
    if pattern is None:
        pattern = out.get("test_pattern", "auto")
        if pattern == "prbs" and out.get("prbs_register"):
            if out.get("seed_auto_detected"):
                # generic auto-seed run: keep the 'prbs' family and pass the
                # order explicitly (legacy names 11/31 are reserved)
                prbs_order = int(out["prbs_register"])
            else:
                pattern = "prbs%d" % int(out["prbs_register"])
    # The reported FFE geometry wins when it is consistent with the JSON tap
    # weights (the bounded multi-tap branch can auto-select a different
    # pre/post cursor count than the CLI defaults, and the eye / sample panel
    # must be drawn at that geometry).
    if out.get("ffe_tap_count") is not None:
        n_t = int(out["ffe_tap_count"])
        pre = int(out.get("ffe_pre_cursor", ffe_pre_cursor))
        post = int(out.get("ffe_post_cursor", ffe_post_cursor))
        taps = out.get("tap_weights")
        if pre + 1 + post == n_t and n_t == len(taps):
            ffe_pre_cursor, ffe_post_cursor = pre, post
    data = prepare_plot_data(input_csv, pattern=pattern,
                             rate_gbps=rate_gbps, domain=domain,
                             samples_per_ui=samples_per_ui, window=window,
                             case_id=case_id, prbs_order=prbs_order,
                             ffe_pre_cursor=ffe_pre_cursor,
                             ffe_post_cursor=ffe_post_cursor)
    M = data["M"]
    tru = data["tru"]
    cdr = data["cdr"]
    spu = data["spu"]
    w = data["w"]
    unit = "mV" if domain == "electrical" else "mW"
    unit_label = "Voltage (mV)" if domain == "electrical" else "Power (mW)"

    # bjarkeaf/pam4-link-sim visual constants: one UI in ps.
    ui_ps = (1.0 / rate_gbps) * 1e12

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1],
                          width_ratios=[1.0, 0.55],
                          hspace=0.34, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])       # left: eye
    ax_hist = fig.add_subplot(gs[0, 1])   # right: probability mass
    ax2 = fig.add_subplot(gs[1, :])       # bottom: waveform

    # ---- equalised density eye (bjarkeaf/pam4-link-sim style) ----
    eye = _eye_phase(M, cdr, spu)
    n_sym = eye.shape[0]
    # cdr-first folded columns: column j is phase (cdr+j)%spu, mapped to
    # 0.5 UI + j/spu (wrapped into [0, 1) UI).  This places the sampling
    # instant at the centre of the eye and keeps the 0.45/0.55 UI windows
    # correctly positioned around it.
    t_col = ui_ps * ((0.5 + np.arange(spu) / float(spu)) % 1.0)
    t_eye = np.tile(t_col, n_sym)
    p_eye = eye.flatten()
    counts, t_edges, p_edges = np.histogram2d(
        t_eye, p_eye, bins=[spu, 200],
        range=[[0.0, ui_ps], [float(p_eye.min()), float(p_eye.max())]])
    cmap = _cmap_for(eye_color)
    ax1.set_facecolor("#05050f")
    ax1.pcolormesh(t_edges, p_edges, counts.T, cmap=cmap,
                   norm=PowerNorm(gamma=0.2, vmin=0.0,
                                  vmax=max(float(counts.max()), 1e-12)),
                   shading="auto")

    # PAM4 thresholds from class-mean Pave/OMA.  On degenerate closed-eye
    # fallback records the class means may come out inverted (lv[3]-lv[0]<=0);
    # keep the displayed thresholds monotonic by using a symmetric span around
    # Pave derived from the observed level magnitude.
    lv = eq.levels_at(M, tru, cdr)
    p_ave = float(lv.mean())
    oma_outer = float(lv[3] - lv[0])
    if np.isfinite(oma_outer) and oma_outer > 0:
        oma_plot = oma_outer
    else:
        level_span = float(np.max(np.abs(lv))) if np.all(np.isfinite(lv)) else 0.0
        oma_plot = 2.0 * max(level_span, 1e-9)
    p_th1 = p_ave - oma_plot / 3.0
    p_th2 = p_ave
    p_th3 = p_ave + oma_plot / 3.0
    for p, label in [(p_th1, r'$P_{th1} = P_{ave} - \mathrm{OMA}/3$'),
                     (p_th2, r'$P_{th2} = P_{ave}$'),
                     (p_th3, r'$P_{th3} = P_{ave} + \mathrm{OMA}/3$')]:
        ax1.axhline(p, color="white", lw=0.8, ls="--", alpha=0.7)
        ax1.text(0.01, p, label, transform=ax1.get_yaxis_transform(),
                 va="bottom", ha="left", fontsize=9, color="white")
    # Histogram windows (0.04 UI wide per IEEE 802.3 Clause 121.8.5.3).
    half_win = _HIST_HALF_UI * ui_ps
    for phase_frac, label in _HIST_WINDOWS:
        cx = phase_frac * ui_ps
        ax1.axvspan(cx - half_win, cx + half_win, color="white", alpha=0.15)
        ax1.text(cx, 0.97, label, transform=ax1.get_xaxis_transform(),
                 va="top", ha="center", fontsize=9, color="white", rotation=90)
    y_lo = min(0.0, float(p_eye.min()))
    y_hi = float(p_eye.max()) + 0.1 * max(1.0, float(np.ptp(p_eye)))
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel(unit_label)
    ax1.set_title("Equalized eye diagram")

    # ---- probability mass at the 0.45/0.55 UI histogram windows ----
    y_bins, F_left, F_right = _collect_center_histograms(
        M, cdr, spu, p_ave, oma_outer)
    delta_y = y_bins[1] - y_bins[0]
    ax_hist.barh(y_bins, F_left, height=delta_y, alpha=0.5,
                 color="tab:blue", label="0.45 UI")
    ax_hist.barh(y_bins, F_right, height=delta_y, alpha=0.5,
                 color="tab:orange", label="0.55 UI")
    sigma_eff = out.get("sigma_G_mV", out.get("R_mV", None))
    if sigma_eff is None:
        sigma_eff = float("nan")
    scale = max(float(F_left.max()), float(F_right.max()), 1e-12)
    if np.isfinite(float(sigma_eff)) and float(sigma_eff) > 0:
        for pth in (p_th1, p_th2, p_th3):
            kernel = np.exp(-0.5 * ((y_bins - pth) / float(sigma_eff)) ** 2)
            ax_hist.plot(kernel * scale, y_bins, "k--", lw=0.8, alpha=0.55,
                         label="Gaussian kernel" if pth == p_th1 else None)
    for i, pth in enumerate((p_th1, p_th2, p_th3)):
        ax_hist.axhline(pth, color="red", lw=0.8, ls="--", alpha=0.45,
                        label="Decision thresholds" if i == 0 else None)
    ax_hist.set_ylim(ax1.get_ylim())
    ax_hist.set_xlabel("Probability mass")
    ax_hist.set_ylabel(unit_label)
    ax_hist.set_title("Histograms at 0.45 / 0.55 UI")
    ax_hist.legend(fontsize=7, framealpha=0.35)

    # ---- waveform / sample overview: raw and FFE waveforms, with explicit
    # sampling instants marked by vertical orange lines and dots ----
    v = data["v"]
    n_ui = data["n_ui"]
    n_sym_wave = M.shape[0]
    common = n_sym_wave * spu
    tt = np.arange(common) / spu
    raw_wave = np.asarray(v[:common], dtype=float)
    eq_wave = np.asarray(M.reshape(-1)[:common], dtype=float)

    ax2.plot(tt, raw_wave, lw=0.55, color="#7fb3d5", alpha=0.85,
             label="raw waveform")
    ax2.plot(tt, eq_wave, lw=0.55, color="#2ca02c", alpha=0.9,
             label="FFE-equalised waveform")

    xmax_wave = min(n_ui, 48)
    sample_x = (cdr + np.arange(n_sym_wave) * spu) / spu
    mask = sample_x < xmax_wave
    sample_x = sample_x[mask]
    raw_samples = np.asarray(v)[(cdr + np.arange(n_sym_wave) * spu)[mask]]
    eq_samples = np.asarray(M)[np.arange(n_sym_wave), cdr][mask]

    for x in sample_x:
        ax2.axvline(x, color="orange", lw=0.55, alpha=0.32)
    ax2.plot(sample_x, raw_samples, linestyle="none", marker="o", ms=3.2,
             mfc="none", mec="orange", mew=1.0, alpha=0.95,
             label="raw sample instants")
    ax2.plot(sample_x, eq_samples, linestyle="none", marker="s", ms=2.6,
             color="red", alpha=0.95, label="FFE sample instants")

    ax2.set_xlim(0, xmax_wave)
    ax2.set_xlabel("UI")
    ax2.set_ylabel("waveform [%s]" % unit)
    ax2.set_title("Raw vs FFE waveform (orange lines = sampling instants)")
    ax2.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.35)

    # the suptitle annotates the *reported* FFE (the measurement result), not
    # the lightweight eye re-analysis: main tap bold, bound-touching taps
    # underlined / overlined when the bounded mode is enabled.
    w_disp = np.asarray(out.get("tap_weights") or w, dtype=float)
    main_idx = int(out.get("ffe_pre_cursor", ffe_pre_cursor))
    bounds_on = bool(out.get("ffe_bounds_enabled", False))
    if bounds_on:
        atb = ffe_bounds.at_bound_info(w_disp, main_idx)
    else:
        atb = [None] * len(w_disp)
    taps_txt = ",".join(
        _format_tap_label(x, i == main_idx, atb[i])
        for i, x in enumerate(w_disp))
    fig.suptitle("TDECQ=%.3f dB  taps=[%s]  cdr=%d" % (
        out.get("tdecq_db", out.get("tecq_db", float("nan"))),
        taps_txt, cdr), fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.08,
                         hspace=0.42, wspace=0.32)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    return png_path
