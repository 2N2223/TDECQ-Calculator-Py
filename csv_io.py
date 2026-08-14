# -*- coding: utf-8 -*-
"""Automatic waveform CSV parsing for the TDECQ/TECQ tool.

The tool must accept *any* CSV exported by a simulator without a manual
``csv_format`` hint.  This module implements heuristic detection of:

* the header / first data row;
* which column is the *time* column (monotonically increasing) and which is
  the *signal* column;
* the signal unit scale (Virtuoso exports voltage in volts for electrical
  waveforms and must be scaled to mV; psf2csv electrical waveforms are
  already in mV; optical waveforms are in mW).

Supported formats (SPEC.md section 3.2):

* Format A - Virtuoso export, header like ``VT("/p") X,VT("/p") Y`` (node name
  is arbitrary), two columns: time (s), voltage (V).
* Format B - psf2csv export, header ``time,out``, two columns: time (s),
  signal (mV for electrical / mW for optical).

All numerical work is vectorised with numpy.
"""
from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------------
# header / first-data-row detection
# ---------------------------------------------------------------------------


def sniff(path: str) -> dict:
    """Read the beginning of a CSV file and return a small descriptor.

    Returns
    -------
    dict with keys:
        header          : raw first non-empty line (str)
        header_tokens   : comma-separated tokens of the header (list[str])
        first_data_row  : 1-based line number of the first purely-numeric row
        n_header_lines  : number of skipped header lines for np.loadtxt
        col_count       : number of numeric columns in the first data row
        is_virtuoso     : True when the header contains ``VT(`` / ``X`` / ``Y``
        time_header     : name of the time column token (lower-cased)
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header = ""
        header_tokens: list[str] = []
        first_row = None
        line_no = 0
        while line_no < 40:
            line = fh.readline()
            if line == "":
                break
            line_no += 1
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    floats = [float(p) for p in parts]
                except ValueError:
                    if not header:
                        header = stripped
                        header_tokens = [t.strip().strip('"') for t in stripped.split(",")]
                    continue
                if first_row is None:
                    first_row = floats
                if header:
                    break
        if first_row is None:
            raise ValueError("could not find numeric data in %s" % path)
        n_header = line_no - 1
        if not header:
            # no textual header at all
            header = ""
            header_tokens = []
            n_header = line_no - 1
        txt = header.lower()
        is_virtuoso = "vt(" in txt
        if not is_virtuoso and len(header_tokens) == 2:
            # Virtuoso-style pair without the VT() label: exactly two
            # columns whose tokens are X (time scale) and Y (signal).
            t0 = header_tokens[0].strip().strip('"').lower()
            t1 = header_tokens[1].strip().strip('"').lower()
            is_virtuoso = (t0 == "x" and t1 == "y")
        time_header = None
        for tok in header_tokens:
            low = tok.strip().lower()
            if low in ("time", "t", "x", "time (s)", "time(s)"):
                time_header = low
        return dict(
            header=header,
            header_tokens=header_tokens,
            first_data_row=line_no,
            n_header_lines=n_header,
            col_count=len(first_row),
            is_virtuoso=is_virtuoso,
            time_header=time_header,
        )


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_columns(path: str):
    """Load all numeric columns of a CSV into a 2-D float array.

    Parameters
    ----------
    path : str
        CSV path.

    Returns
    -------
    (info, data) : (dict from :func:`sniff`, np.ndarray of shape (n_rows, n_cols))
    """
    info = sniff(path)
    if info["col_count"] < 2:
        raise ValueError("waveform CSV must have at least two numeric columns: %s" % path)
    data = np.loadtxt(path, delimiter=",", skiprows=info["n_header_lines"])
    data = np.atleast_2d(data)
    return info, data


def grid_is_uniform(t: np.ndarray, rel_tol: float = 1e-6) -> bool:
    """True when the time column is a uniformly spaced grid.

    The release Virtuoso exports use an adaptive (non-uniform) time grid;
    simulator exports with a fixed step (including Virtuoso-table-header
    ``VT(...) X,Y`` files whose columns are a regular grid) are uniform and
    can be sliced directly without resampling.

    Parameters
    ----------
    t : time column (s).
    rel_tol : maximum allowed relative deviation of a step from the median.

    Returns
    -------
    bool.
    """
    t = np.asarray(t, dtype=float)
    if len(t) < 3:
        return True
    dt = np.diff(t)
    if dt.min() <= 0.0:
        return False
    med = float(np.median(dt))
    if med <= 0.0:
        return False
    return bool(np.max(np.abs(dt - med)) <= rel_tol * med)


def _monotonic_fraction(col: np.ndarray) -> float:
    d = np.diff(col)
    if len(d) == 0:
        return 1.0
    return float(np.mean(d > 0))


def identify_columns(info: dict, data: np.ndarray):
    """Identify the time and signal columns of a numeric CSV.

    Rules (automatic, no per-file configuration):

    * a column whose values are monotonically non-decreasing is the time
      column (tie-broken by the header name);
    * every other numeric column is a candidate signal column; the one with
      the largest variance is used when there are several.

    Parameters
    ----------
    info : dict
        descriptor from :func:`sniff`.
    data : np.ndarray
        numeric table (n_rows, n_cols).

    Returns
    -------
    (t, s, t_col, s_col) : time array, signal array, column indices.
    """
    n_cols = data.shape[1]
    # header tokens take priority for the time column in *every* file shape
    # (two-column files included) - a token of "time" / "t" / "x" wins;
    # otherwise the most monotonic column is the time scale.
    t_col = None
    if info.get("time_header"):
        for c in range(n_cols):
            low = str(info["header_tokens"][c]).strip().lower() if c < len(info["header_tokens"]) else ""
            if low in ("time", "t", "x"):
                t_col = c
                break
    if t_col is None:
        mono = [(c, _monotonic_fraction(data[:, c])) for c in range(n_cols)]
        mono.sort(key=lambda kv: -kv[1])
        t_col = mono[0][0]
    s_cols = [c for c in range(n_cols) if c != t_col]
    if not s_cols:
        raise ValueError("could not separate time and signal columns in %s" % os.path.basename(path))
    s_col = max(s_cols, key=lambda c: float(np.var(data[:, c])))
    return data[:, t_col], data[:, s_col], t_col, s_col


def auto_scale(signal: np.ndarray, domain: str, is_virtuoso: bool) -> float:
    """Determine the multiplicative scale to reach the tool's internal units.

    Internal units are mV for electrical and mW for optical (SPEC.md 4.8).

    Rules (automatic, header-driven only - no amplitude guessing):

    * optical waveforms are already in mW -> scale 1.0;
    * Virtuoso exports (``VT(...) X,VT(...) Y`` or ``x,y`` pairs) are in
      volts -> x1000 for electrical;
    * psf2csv electrical exports (``time,out``) are already in mV -> 1.0.

    An amplitude heuristic (``swing < 20 => x1000``) is deliberately NOT
    used: it would wrongly scale genuine small-swing mV records, and the
    manifest electrical records all exceed 140 mV peak-to-peak, so no
    shipped case depends on it.

    Parameters
    ----------
    signal : np.ndarray
        raw signal column (kept for API symmetry).
    domain : str
        'electrical' or 'optical'.
    is_virtuoso : bool
        header heuristic result.

    Returns
    -------
    float scale factor.
    """
    if domain != "electrical":
        return 1.0
    return 1000.0 if is_virtuoso else 1.0


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def read_waveform(path: str, domain: str = "electrical"):
    """Read a waveform CSV and return time / scaled signal arrays.

    Parameters
    ----------
    path : str
        input CSV.
    domain : str
        'electrical' (units mV) or 'optical' (units mW).

    Returns
    -------
    (t, s, meta) :
        t        : time column (s), float array
        s        : signal column, scaled to mV (electrical) / mW (optical)
        meta     : dict with keys is_virtuoso, scale, n_header_lines,
                   first_data_row, header, t_col, s_col
    """
    info, data = load_columns(path)
    t, s, t_col, s_col = identify_columns(info, data)
    scale = auto_scale(s, domain, info["is_virtuoso"])
    return t, s * scale, dict(
        is_virtuoso=info["is_virtuoso"],
        uniform_grid=grid_is_uniform(t),
        scale=scale,
        n_header_lines=info["n_header_lines"],
        first_data_row=info["first_data_row"],
        header=info["header"],
        t_col=int(t_col),
        s_col=int(s_col),
    )


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------


def resample_uniform(t: np.ndarray, s: np.ndarray, n_ui: int, spu: int,
                     ui: float, phase: float = 0.0, t0: float = None):
    """Resample a waveform onto a uniform UI grid via linear interpolation.

    Parameters
    ----------
    t, s   : time (s) and signal arrays from :func:`read_waveform`.
    n_ui   : number of unit intervals to cover.
    spu    : samples per UI on the target grid.
    ui     : UI duration in seconds (2 / rate_gbps * 1e-9).
    phase  : fractional-sample offset of the sampling grid (0 .. 1).
    t0     : start time; defaults to t[0].

    Returns
    -------
    np.ndarray of length n_ui*spu, the uniformly sampled signal.
    """
    if t0 is None:
        t0 = float(t[0])
    n = n_ui * spu
    tgrid = t0 + (np.arange(n) + phase) * (ui / spu)
    return np.interp(tgrid, t, s)


def auto_window(pattern: str, n_ui_available: int, spu: int) -> int:
    """Choose the test-window length (number of UI) automatically.

    Semantic of ``--window auto`` (SPEC.md 3.3 / manifest ``window=auto``):

    * PRBS31 / SSPRQ records that contain a full pattern period use the full
      period (65535 UI);
    * PRBS13 uses the record length (the 128G records contain 6400 UI);
    * all other patterns use the record length.

    Parameters
    ----------
    pattern : str
        'prbs11', 'prbs13', 'prbs31', 'ssprq' or 'auto'.
    n_ui_available : int
        number of whole UI present in the record.
    spu : int
        samples per UI.

    Returns
    -------
    int number of UI to take from the record.
    """
    if pattern in ("prbs31", "ssprq"):
        period = 65535
        if n_ui_available >= period:
            return period
        return max(4, n_ui_available)
    if pattern == "prbs13":
        return min(n_ui_available, 6400)
    return n_ui_available


def auto_window_prbsn(n, n_ui_available, spu):
    """Automatic test-window length (UI) for a generic PRBSn record.

    * n <= 16: one full pattern period (``2**n - 1`` UI) when the record
      contains a full period, otherwise the record length;
    * n >  16: full periods are impractically long, so a full-period record is
      treated as a linear segment - use ``min(record length, 65535)`` UI,
      matching the PRBS31 long-record convention.

    Parameters
    ----------
    n : int in 2..31.
    n_ui_available : int, whole UI present in the record.
    spu : int, samples per UI (kept for API symmetry).

    Returns
    -------
    int number of UI to take from the record.
    """
    period = (1 << int(n)) - 1
    if n <= 16:
        if n_ui_available >= period:
            w = period
        else:
            w = n_ui_available
        # the 5-tap FFE needs a workable minimum number of symbols
        return min(n_ui_available, max(16, w))
    return min(n_ui_available, 65535)
