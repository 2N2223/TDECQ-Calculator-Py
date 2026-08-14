# -*- coding: utf-8 -*-
"""Optical-domain signal conditioning (SPEC.md 4.6).

For TDECQ the received optical waveform (in mW) is processed with the IEEE
802.3 receiver model:

* chromatic dispersion: 0.8 ps/nm at 1310 nm (applied as a pure delay equal
  to the fibre dispersion, in samples);
* receiver bandwidth limit: first-order low-pass with a 3 dB bandwidth of
  42 GHz.

The first 120 UI are discarded (filter transient) and the next 1927 UI are
kept for analysis, matching the reference bookkeeping (``total_UI = 1927``,
``total_symbols = 1923``).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

DT_112 = 2.790178571429e-13          # sample step of the 112G records (s)
UI_112 = 1.0 / 56e9                  # UI duration at 56 GBd (s)
DISPERSION_PS_PER_NM = 0.8
WAVELENGTH_NM = 1310.0
RX_BW_HZ = 42e9
N_UI_KEEP = 1927
TRANSIENT_UI = 120


def optical_process(v: np.ndarray, dt: float = DT_112) -> np.ndarray:
    """Apply dispersion + receiver low-pass and crop the steady-state window.

    Parameters
    ----------
    v  : received optical waveform (mW), uniformly sampled with step ``dt``.
    dt : sample step in seconds.

    Returns
    -------
    np.ndarray of length N_UI_KEEP * 64 (1927 UI @ 64 spu).
    """
    D = int(round(DISPERSION_PS_PER_NM * 1e-12 * WAVELENGTH_NM / dt))
    tau = 1.0 / (2 * np.pi * RX_BW_HZ)
    a = dt / (tau + dt)
    y = np.concatenate([np.zeros(D), v])
    y = lfilter([a], [1, -(1 - a)], y)
    start = TRANSIENT_UI * 64
    return y[start:start + N_UI_KEEP * 64]
