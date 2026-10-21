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

# CJK titles on the reference-pattern panel: fall back gracefully when no
# CJK font is installed (matplotlib warns and substitutes).
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

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


def _phase_density(eye, n_y_bins=200):
    """Build an eye density grid without floating-point x-bin collisions.

    ``eye`` columns are discrete sample phases, so their horizontal bins are
    assigned with integer indices.  Constructing floating-point phase values
    and feeding them back to ``histogram2d`` is unsafe at e.g. 100 samples/UI:
    round-off puts two boundary-valued phases in one bin and leaves the next
    bin empty, which renders false vertical breaks in an otherwise continuous
    eye.  The returned horizontal edges are normalised UI coordinates.
    """
    eye = np.asarray(eye, dtype=float)
    if eye.ndim != 2 or eye.shape[1] < 1:
        raise ValueError("eye must have shape (n_symbols, samples_per_ui)")
    n_sym, spu = eye.shape
    p_lo = float(np.min(eye))
    p_hi = float(np.max(eye))
    if not np.isfinite(p_lo) or not np.isfinite(p_hi):
        raise ValueError("eye contains non-finite samples")
    if p_hi <= p_lo:
        p_lo -= 0.5
        p_hi += 0.5
    p_edges = np.linspace(p_lo, p_hi, int(n_y_bins) + 1)
    counts = np.zeros((spu, int(n_y_bins)), dtype=float)
    # Column zero is the CDR sampling phase and is displayed at the centre.
    # Integer permutation guarantees exactly one source phase per x bin.
    phase_bins = (np.arange(spu, dtype=int) + spu // 2) % spu
    for source_col, display_bin in enumerate(phase_bins):
        counts[display_bin], _ = np.histogram(
            eye[:, source_col], bins=p_edges)
    t_edges_ui = np.linspace(0.0, 1.0, spu + 1)
    assert np.all(counts.sum(axis=1) == n_sym)
    return t_edges_ui, p_edges, counts


def _density_eye(M, tru, cdr, spu, bins=(220, 220), xmax=1.0):
    """2-D histogram of the equalised signal over one UI.

    Samples of the equalised matrix ``M`` (n_sym, spu) are folded into one UI
    using the sampling phase ``cdr`` as the origin.  Returns (x, y, H) with
    x in UI (0..xmax) and y in the signal units.
    """
    eye = _eye_phase(M, cdr, spu)
    n_y = int(bins[1] if hasattr(bins, "__len__") else bins)
    xe_ui, ye, H = _phase_density(eye, n_y_bins=n_y)
    xe = xe_ui * float(xmax)
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


def eye_ylim(values, below=0.06, above=0.14):
    """Eye-diagram y-range relative to the signal span (any unit/scale).

    The margins are fractions of the peak-to-peak span (the top margin
    leaves room for the window labels), so W, mW, V and mV records all fill
    the panel; the previous fixed 0.1-unit margin squashed records whose
    span is far below one unit.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if span <= 0:
        span = abs(hi) if hi != 0 else 1.0
    return lo - below * span, hi + above * span


THRESHOLD_LABELS = (r'$P_{th1} = P_{ave} - \mathrm{OMA}/3$',
                    r'$P_{th2} = P_{ave}$',
                    r'$P_{th3} = P_{ave} + \mathrm{OMA}/3$')


def eye_overlay(M, tru, cdr, spu, out=None):
    """Decision thresholds and histogram windows to draw on an eye.

    A standard-mode result (``out['standard_measurement']``) carries the
    measured OMA_outer, Pave, thresholds and the exact window phases, so the
    overlay shows what the metric used.  Otherwise the thresholds come from
    the class means (with a monotonic fallback span on degenerate eyes) and
    the windows sit at the nominal 0.45/0.55 UI.

    Returns dict(p_ave, oma, pth, windows); ``windows`` holds
    ``(x0_ui, x1_ui, label)`` in the display coordinates of
    :func:`_phase_density` (sampling phase ``cdr`` at 0.5 UI).
    """
    sm = (out or {}).get("standard_measurement")
    if sm and sm.get("pave") is not None and sm.get("pth"):
        p_ave = float(sm["pave"])
        oma = float(sm["oma"])
        pth = [float(x) for x in sm["pth"]]
        windows = []
        for key, name in (("window_left", "left"), ("window_right", "right")):
            ph = np.asarray(sm.get(key) or [], dtype=int)
            if ph.size == 0:
                continue
            d = ((ph - int(cdr)) % spu + spu // 2) % spu
            x0, x1 = d.min() / spu, (d.max() + 1) / spu
            windows.append((x0, x1, "%.2f UI" % (0.5 * (x0 + x1))))
        return dict(p_ave=p_ave, oma=oma, pth=pth, windows=windows)
    lv = eq.levels_at(M, tru, cdr)
    p_ave = float(lv.mean())
    oma = float(lv[3] - lv[0])
    if not (np.isfinite(oma) and oma > 0):
        level_span = (float(np.max(np.abs(lv)))
                      if np.all(np.isfinite(lv)) else 0.0)
        oma = 2.0 * max(level_span, 1e-9)
    pth = [p_ave - oma / 3.0, p_ave, p_ave + oma / 3.0]
    windows = [(f - _HIST_HALF_UI, f + _HIST_HALF_UI, label)
               for f, label in _HIST_WINDOWS]
    return dict(p_ave=p_ave, oma=oma, pth=pth, windows=windows)


def _format_tap_label(value, is_main=False, at_bound=None):
    """Format one FFE tap value with portable ASCII-only markers.

    ``M`` marks the declared main tap and ``LB``/``UB`` mark lower/upper
    bound contact.  Mathtext commands such as ``\\mathbf`` are deliberately
    avoided: some remote Matplotlib builds reject them during ``savefig``,
    and nesting bold with bound under/overlines is not parser-portable.

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
    str such as ``"0.125"``, ``"M:0.900"`` or ``"M/LB:0.800"``.
    """
    marks = []
    if is_main:
        marks.append("M")
    if at_bound == "lb":
        marks.append("LB")
    elif at_bound == "ub":
        marks.append("UB")
    prefix = ("/".join(marks) + ":") if marks else ""
    return "%s%.3f" % (prefix, float(value))


def make_diagnostic_plot(input_csv, out, png_path, rate_gbps=112,
                         domain="electrical", samples_per_ui=64,
                         window="auto", case_id=None, pattern=None,
                         eye_color=DEFAULT_EYE_COLOR,
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
    rate_gbps, domain, samples_per_ui, window, case_id, pattern :
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
            # Compatibility with JSON emitted before explicit ``prbsN`` modes:
            # old generic reports used ``test_pattern='prbs'`` plus a separate
            # register field.  Convert them to the new unambiguous spelling.
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
                             case_id=case_id,
                             ffe_pre_cursor=ffe_pre_cursor,
                             ffe_post_cursor=ffe_post_cursor,
                             tap_weights=out.get("tap_weights"),
                             dfe_b1=out.get("dfe_b1", 0.0),
                             standard_measurement=out.get(
                                 "standard_measurement"))
    M = data["M"]
    tru = data["tru"]
    cdr = data["cdr"]
    spu = data["spu"]
    w = data["w"]
    unit = "mV" if domain == "electrical" else "mW"
    unit_label = "Voltage (mV)" if domain == "electrical" else "Power (mW)"

    # One UI in ps.  rate_gbps is the aggregate PAM4 data rate (2 bits per
    # symbol), so one UI is 2/(rate_gbps*1e9) s = 2e3/rate_gbps ps (the same
    # UI = 2/rate convention as the measurement front end).
    ui_ps = 2.0e3 / float(rate_gbps)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 0.85],
                          width_ratios=[1.0, 0.55],
                          hspace=0.36, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])       # left: eye
    ax_hist = fig.add_subplot(gs[0, 1])   # right: probability mass
    ax2 = fig.add_subplot(gs[1, :])       # middle: waveform
    ax3 = fig.add_subplot(gs[2, :])       # bottom: ideal reference pattern

    # ---- equalised density eye (bjarkeaf/pam4-link-sim style) ----
    eye = _eye_phase(M, cdr, spu)
    n_sym = eye.shape[0]
    p_eye = eye.flatten()
    t_edges_ui, p_edges, counts = _phase_density(eye, n_y_bins=200)
    t_edges = t_edges_ui * ui_ps
    cmap = _cmap_for(eye_color)
    ax1.set_facecolor("#05050f")
    ax1.pcolormesh(t_edges, p_edges, counts.T, cmap=cmap,
                   norm=PowerNorm(gamma=0.2, vmin=0.0,
                                  vmax=max(float(counts.max()), 1e-12)),
                   shading="auto")

    # PAM4 thresholds and histogram windows: the measured values of a
    # standard-mode result, else class means and the nominal 0.45/0.55 UI.
    ov = eye_overlay(M, tru, cdr, spu, out)
    p_ave, oma_outer = ov["p_ave"], ov["oma"]
    for p, label in zip(ov["pth"], THRESHOLD_LABELS):
        ax1.axhline(p, color="white", lw=0.8, ls="--", alpha=0.7)
        ax1.text(0.01, p, label, transform=ax1.get_yaxis_transform(),
                 va="bottom", ha="left", fontsize=9, color="white")
    # Histogram windows (0.04 UI wide per IEEE 802.3 Clause 121.8.5.3).
    for x0, x1, label in ov["windows"]:
        ax1.axvspan(x0 * ui_ps, x1 * ui_ps, color="white", alpha=0.15)
        ax1.text(0.5 * (x0 + x1) * ui_ps, 0.97, label,
                 transform=ax1.get_xaxis_transform(),
                 va="top", ha="center", fontsize=9, color="white", rotation=90)
    ax1.set_ylim(*eye_ylim(p_eye))
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
    sigma_eff = next((out[k] for k in ("sigma_G_mV", "sigma_G_mW", "R_mV", "R_mW")
                      if out.get(k) is not None), None)
    if sigma_eff is None:
        sigma_eff = float("nan")
    scale = max(float(F_left.max()), float(F_right.max()), 1e-12)
    if np.isfinite(float(sigma_eff)) and float(sigma_eff) > 0:
        for pth in ov["pth"]:
            kernel = np.exp(-0.5 * ((y_bins - pth) / float(sigma_eff)) ** 2)
            ax_hist.plot(kernel * scale, y_bins, "k--", lw=0.8, alpha=0.55,
                         label="Gaussian kernel" if pth == ov["pth"][0] else None)
    for i, pth in enumerate(ov["pth"]):
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

    # ---- ideal reference pattern reconstructed from the detected seed ----
    # ``data['tru']`` is the aligned reference symbol stream that ``_run_core``
    # built from the recovered seed (or the canonical fallback); mapping it
    # through the nominal PAM4 levels gives the ideal waveform the FFE is
    # equalising against.  On a failed seed recovery the panel reports the
    # warning / canonical fallback instead of a misleading waveform.
    # Prefer the measured post-FFE class levels in the report.  ``data['nom']``
    # can be a fixed release target (for example +/-500 mV) while an exported
    # Virtuoso waveform legitimately has a much smaller source swing; using
    # the fixed target here makes the ideal panel look incorrectly amplified.
    level_key = "RX_levels_mV" if domain == "electrical" else "RX_levels_mW"
    measured_levels = out.get(level_key)
    use_measured = measured_levels is not None
    nom_arr = np.asarray(measured_levels if use_measured else data["nom"], dtype=float)
    if nom_arr.shape != (4,) or not np.all(np.isfinite(nom_arr)) \
            or np.any(np.diff(nom_arr) <= 0):
        nom_arr = np.asarray(data["nom"], dtype=float)
        use_measured = False
    ideal_label = ("ideal pattern (measured RX levels)" if use_measured
                   else "ideal pattern (nominal levels)")
    tru_arr = np.asarray(tru, dtype=np.int64)
    n_show = min(48, int(len(tru_arr)))
    ideal = np.repeat(nom_arr[tru_arr[:n_show]], spu)
    x_ideal = np.arange(len(ideal)) / spu
    ax3.step(x_ideal, ideal, where="post", lw=1.0, color="#1f77b4",
             alpha=0.95, label=ideal_label)
    ax3.plot(x_ideal[::spu], nom_arr[tru_arr[:n_show]], linestyle="none",
             marker="o", ms=2.6, mfc="none", mec="#1f77b4", mew=0.8,
             label="symbol centres")
    seed = out.get("prbs_seed")
    warn = out.get("pattern_match_warning")
    seed_ok = bool(out.get("seed_auto_detected"))
    off_disp = int(out.get("offset_estimated") or 0)
    if warn:
        ax3.set_title("Seed recovery failed: %s" % str(warn)[:70], fontsize=8,
                      color="#c0392b")
    elif seed is None:
        ax3.set_title("Periodic pattern (%s, off=%d)" %
                      (out.get("test_pattern", "auto"), off_disp), fontsize=8)
    elif seed_ok:
        ax3.set_title("Recovered Seed pattern: seed=%s (kind=%s, source=%s)" %
                      (seed, out.get("record_kind", ""),
                       out.get("seed_source", "")), fontsize=8)
    else:
        ax3.set_title("Seed recovery failed (canonical fallback, seed=%s): %s" %
                      (seed, out.get("seed_source", "")), fontsize=8,
                      color="#c0392b")
    ax3.set_xlim(0, n_show)
    ax3.set_xlabel("UI")
    ax3.set_ylabel("ideal [%s]" % unit)
    ax3.legend(loc="upper right", fontsize=7, framealpha=0.35)

    # the suptitle annotates the *reported* FFE (the measurement result), not
    # the lightweight eye re-analysis.  Portable ASCII markers avoid remote
    # Matplotlib mathtext incompatibilities: M=main, LB/UB=bound contact.
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
    ser = out.get("actual_SER")
    if ser is not None and np.isfinite(float(ser)):
        # PAM4 Gray mapping: one symbol error flips ~1 of the 2 bits, so
        # BER(est) ~= SER / 2 (IEEE 802.3 assumption on a Gray-coded PAM4).
        ber_est = float(ser) / 2.0
        ser_txt = "SER=%.2e  BER(est)=%.2e (SER/2)" % (ser, ber_est)
    else:
        ser_txt = "SER=n/a"
    fig.suptitle("TDECQ=%.3f dB  taps=[%s]  cdr=%d   %s" % (
        out.get("tdecq_db", out.get("tecq_db", float("nan"))),
        taps_txt, cdr, ser_txt), fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.08,
                         hspace=0.42, wspace=0.32)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    return png_path
