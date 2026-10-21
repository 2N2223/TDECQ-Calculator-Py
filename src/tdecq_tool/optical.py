# -*- coding: utf-8 -*-
"""Optical-domain signal conditioning (SPEC.md 4.6).

Standard measurement mode uses :func:`reference_receiver_bt4` (the IEEE
802.3 TDECQ 4th-order Bessel-Thomson reference receiver, 0.5 x baud) and no
dispersion model.  :func:`optical_process` below is the LEGACY front end,
kept only so ``tdecq_mode='legacy'`` reproduces the historical results.

For TDECQ the received optical waveform (in mW) is processed with the IEEE
802.3 receiver model:

* chromatic dispersion: 0.8 ps/nm at 1310 nm (applied as a pure delay equal
  to the fibre dispersion, in samples);
* receiver bandwidth limit: first-order low-pass with a 3 dB bandwidth of
  42 GHz (the default ``0.75 * baud`` for 112G records).

The first 120 UI are discarded (filter transient) and the next 1927 UI are
kept for analysis, matching the reference bookkeeping (``total_UI = 1927``,
``total_symbols = 1923``).

Rate adaptation (P2-6): :func:`optical_process` is parameterised by the
native grid (``spu`` samples per UI), the symbol rate (``baud``) and the RX
bandwidth.  The legacy 112G behaviour (56 GBd / 64 spu / 42 GHz / crop at
120 UI from the zero-padded head) is preserved exactly when ``legacy_crop``
is set (the release-compatible ``o_*.csv`` records), while non-112G records
are resampled to the analysis grid, use ``rx_bw = 0.75 * baud`` and crop the
1927-UI window so it starts 120 UI after the *dispersion-delayed* signal
begins (the raw release crop only removes the positive transient when the
dispersion delay is shorter than 120 UI, which fails at 200+ GBd).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter, resample as fft_resample

DT_112 = 2.790178571429e-13          # sample step of the 112G records (s)
UI_112 = 1.0 / 56e9                  # UI duration at 56 GBd (s)
DISPERSION_PS_PER_NM = 0.8
WAVELENGTH_NM = 1310.0
RX_BW_HZ = 42e9
N_UI_KEEP = 1927
TRANSIENT_UI = 120
ANALYSIS_SPU = 64


def optical_process(v: np.ndarray, dt: float = None, spu: int = ANALYSIS_SPU,
                    baud: float = None, rx_bw: float = None,
                    n_ui: int = N_UI_KEEP, transient_ui: int = TRANSIENT_UI,
                    analysis_spu: int = ANALYSIS_SPU,
                    legacy_crop: bool = True) -> np.ndarray:
    """Apply dispersion + receiver low-pass and crop the steady-state window.

    Parameters
    ----------
    v  : received optical waveform (mW), uniformly sampled with step ``dt``
         on the native grid.
    dt : sample step in seconds.  Defaults to ``1 / (baud * spu)`` (the
         native-grid step at the symbol rate).
    spu : input samples per UI (native grid).
    baud : symbol rate (Hz).  Required when ``dt`` is omitted.
    rx_bw : RX 3 dB bandwidth (Hz).  Default ``0.75 * baud``; for the
         release-compatible 112G records pass ``RX_BW_HZ`` (42 GHz).
    n_ui : analysis window length (UI), default 1927.
    transient_ui : discarded filter transient (UI), default 120.
    analysis_spu : output samples per UI, default 64.
    legacy_crop : if True keep the release crop ``transient_ui * 64``
         samples counted from the zero-padded head (112G release records);
         otherwise crop ``D + transient_ui * 64`` so the window starts
         ``transient_ui`` UI after the dispersion-delayed signal begins.

    Returns
    -------
    np.ndarray of length ``n_ui * analysis_spu``.
    """
    v = np.asarray(v, dtype=float)
    if dt is None:
        if baud is None:
            raise ValueError("optical_process requires the symbol rate "
                             "(baud) or the sample step (dt)")
        dt = 1.0 / (baud * spu)
    if rx_bw is None:
        rx_bw = 0.75 * baud if baud is not None else RX_BW_HZ
    # resample the native grid onto the analysis grid (identity when equal)
    if spu != analysis_spu:
        n64 = int(round(len(v) * analysis_spu / spu))
        v = fft_resample(v, n64)
        dt = dt * spu / analysis_spu
    D = int(round(DISPERSION_PS_PER_NM * 1e-12 * WAVELENGTH_NM / dt))
    tau = 1.0 / (2 * np.pi * rx_bw)
    a = dt / (tau + dt)
    y = np.concatenate([np.zeros(D), v])
    y = lfilter([a], [1, -(1 - a)], y)
    if legacy_crop:
        start = transient_ui * analysis_spu
    else:
        start = D + transient_ui * analysis_spu
    end = start + n_ui * analysis_spu
    if end > len(y):
        raise ValueError(
            "optical_process: record too short for the %d-UI analysis window "
            "(needs %d samples after the %d-UI transient and the %d-sample "
            "dispersion delay, have %d)" % (n_ui, end, transient_ui, D,
                                            len(y)))
    return y[start:end]


# ---------------------------------------------------------------------------
# IEEE 802.3 TDECQ reference receiver (standard measurement mode)
# ---------------------------------------------------------------------------
REF_RX_BW_OVER_BAUD = 0.5


def reference_receiver_bt4(v: np.ndarray, spu: int,
                           bw_over_baud: float = REF_RX_BW_OVER_BAUD
                           ) -> np.ndarray:
    """4th-order Bessel-Thomson reference receiver.

    The TDECQ reference receiver is a fourth-order Bessel-Thomson response
    whose 3 dB bandwidth is ``bw_over_baud`` times the signalling rate
    (0.5 x baud, e.g. 26.5625 GHz at 53.125 GBd).  It is applied to the
    uniformly sampled optical waveform (``spu`` samples per UI) as a
    magnitude-normalised digital Bessel filter; at the analysis resolution
    (fc/fs = 0.5/spu) the bilinear frequency warping is negligible.  The
    filter state is initialised to the first sample so the record start
    does not produce a step transient.  Chromatic dispersion is NOT a
    receiver property and is not modelled here (it belongs to the test
    fibre / channel, i.e. to the waveform itself).

    Parameters
    ----------
    v : uniformly sampled waveform.
    spu : samples per UI.
    bw_over_baud : 3 dB bandwidth in units of the symbol rate.

    Returns
    -------
    np.ndarray, same length as ``v`` (delayed by the BT4 group delay,
    about 0.67 UI at 0.5 x baud; the pattern alignment absorbs it).
    """
    from scipy.signal import bessel, sosfilt, sosfilt_zi
    v = np.asarray(v, dtype=float)
    fc = float(bw_over_baud)
    if not 0.0 < fc < 0.5 * int(spu):
        raise ValueError("reference receiver bandwidth %.3g x baud is not "
                         "representable at %d samples/UI" % (fc, int(spu)))
    sos = bessel(4, fc, btype="low", norm="mag", output="sos", fs=float(spu))
    y, _ = sosfilt(sos, v, zi=sosfilt_zi(sos) * v[0])
    return y
