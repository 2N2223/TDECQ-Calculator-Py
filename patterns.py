# -*- coding: utf-8 -*-
"""PAM4 test-pattern generation and waveform alignment.

Implements the IEEE 802.3-2022 style PAM4 test patterns used by the tool:

* PRBS11 / PRBS31 - standard LFSR sequences packed MSB-first into PAM4
  symbols (2 bits per symbol);
* PRBS13 - the 802.3 "mzm dual-plane" construction used by the 128G records
  (two LFSR planes A and B, symbol = 2*A + (A xor B));
* SSPRQ - the official IEEE 802.3-2022 Clause 120 short stress pattern random
  quaternary sequence (65535 symbols), read from the shipped reference file
  and used both as the reference pattern and for self-checking.

Alignment routines correlate a decoded symbol sequence with the reference
pattern.  For periodic patterns a circular (FFT) correlation over one period
is used; the returned offset is such that ``tru[k] = pattern[(k + off) % p]``.
"""
from __future__ import annotations

import os

import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
OFFICIAL_SSPRQ = os.path.join(DATA_ROOT, "official", "Clause_120_SSPRQ_sequence.csv")


# ---------------------------------------------------------------------------
# LFSR machinery
# ---------------------------------------------------------------------------


def lfsr_bits(seed: int, nreg: int, length: int, taps):
    """Generate ``length`` bits of an LFSR sequence.

    Parameters
    ----------
    seed  : integer initial state (low ``nreg`` bits used).
    nreg  : register width in bits.
    length: number of output bits.
    taps  : iterable of bit indices (0-based, LSB=0) fed back with xor.

    Returns
    -------
    np.ndarray of int64 with values 0/1.  Bit ``nreg-1`` (MSB) is emitted
    first, matching the standard PRBS definition.
    """
    mask = (1 << nreg) - 1
    st = seed & mask
    out = np.empty(length, dtype=np.int64)
    for i in range(length):
        out[i] = (st >> (nreg - 1)) & 1
        fb = 0
        for t in taps:
            fb ^= (st >> t) & 1
        st = ((st << 1) | fb) & mask
    return out


def prbs_symbols(bits: np.ndarray, n: int) -> np.ndarray:
    """Pack a bit sequence MSB-first into PAM4 symbols (2 bits per symbol).

    Parameters
    ----------
    bits : 0/1 array; the pattern is used cyclically.
    n    : number of symbols to produce.

    Returns
    -------
    np.ndarray of int64 in 0..3.
    """
    B = len(bits)
    out = np.empty(n, dtype=np.int64)
    for k in range(n):
        i = 2 * k % B
        out[k] = (bits[i] << 1) | bits[(i + 1) % B]
    return out


PRBS11_BITS = lfsr_bits(1, 11, 2 * 2047, (10, 8))
PRBS31_BITS = lfsr_bits(0x402, 31, 2 * 2047, (30, 27))
PRBS11_SYMBOLS = prbs_symbols(PRBS11_BITS, 2047)
PRBS31_SYMBOLS = prbs_symbols(PRBS31_BITS, 2047)


# ---------------------------------------------------------------------------
# PRBS13 (mzm dual-plane)
# ---------------------------------------------------------------------------
_PRBS13_A = None
_PRBS13_B = None


def _prbs13_planes(n: int):
    """Two PRBS13 LFSR planes (A seed 1, B seed 137, poly x^13+x^12+x^10+x^9+1).

    The pre-roll offsets (3167 / 4507 symbols) reproduce the 802.3 PAM4
    dual-plane construction used by the 128G reference records.
    """
    global _PRBS13_A, _PRBS13_B
    if _PRBS13_A is None:
        _PRBS13_A = lfsr_bits(1, 13, 3167 + 65536, (12, 3, 2, 0))
        _PRBS13_B = lfsr_bits(137, 13, 4507 + 65536, (12, 3, 2, 0))
    return _PRBS13_A[3167:3167 + n], _PRBS13_B[4507:4507 + n]


def prbs13_symbols(n: int) -> np.ndarray:
    """PAM4 symbols of the 802.3 PRBS13 dual-plane construction."""
    A, B = _prbs13_planes(n)
    return 2 * A + (A ^ B)


# ---------------------------------------------------------------------------
# SSPRQ
# ---------------------------------------------------------------------------
_SSPRQ_CACHE = None


def ssprq_symbols(n=None, path=OFFICIAL_SSPRQ):
    """Official SSPRQ sequence (65535 symbols) from the shipped reference CSV.

    Parameters
    ----------
    n : int or None.  When given, the sequence is wrapped cyclically.
    path : str, path of the official single-column CSV.

    Returns
    -------
    np.ndarray of int64.
    """
    global _SSPRQ_CACHE
    if _SSPRQ_CACHE is None:
        _SSPRQ_CACHE = np.loadtxt(path, dtype=np.int64)
    if n is None:
        return _SSPRQ_CACHE
    return _SSPRQ_CACHE[np.arange(n) % len(_SSPRQ_CACHE)]


def _parse_prbsn_name(ptype):
    """Return (n, taps) when ``ptype`` names a generic single-LFSR PRBSn."""
    s = str(ptype or "").strip().lower()
    if s.startswith("prbs"):
        rest = s[4:]
        if rest.isdigit():
            n = int(rest)
            if 2 <= n <= 31 and n not in (11, 13, 31):
                return n, PRBS_STANDARD_TAPS[n]
    return None


def pattern_symbols(ptype: str, n: int) -> np.ndarray:
    """Generate ``n`` symbols of pattern ``ptype``.

    Supported families: 'prbs11'/'prbs13'/'prbs31'/'ssprq' and generic
    single-LFSR 'prbsN' (N in 2..31, canonical seed 1 - used by the
    canonical-seed fallback of the generic PRBSn path).
    """
    if ptype == "prbs11":
        return PRBS11_SYMBOLS[np.arange(n) % 2047]
    if ptype == "prbs31":
        return PRBS31_SYMBOLS[np.arange(n) % 2047]
    if ptype == "prbs13":
        return prbs13_symbols(n)
    if ptype == "ssprq":
        return ssprq_symbols(n)
    g = _parse_prbsn_name(ptype)
    if g is not None:
        nn, taps = g
        period = (1 << nn) - 1
        if period > 10 ** 6:
            # long orders: linear generation without the cyclic wrap
            return prbs_symbols(lfsr_bits(1, nn, 2 * n + 64, taps), n)
        return prbs_symbols(lfsr_bits(1, nn, 2 * period, taps), n)
    raise ValueError("unknown pattern %r" % (ptype,))


def pattern_period(ptype: str) -> int:
    """Length of the pattern period in symbols (PRBS13 dual-plane is aperiodic for the 6400-UI records)."""
    if ptype in ("prbs31", "prbs11"):
        return 2047
    if ptype == "ssprq":
        return 65535
    if ptype == "prbs13":
        return 8191
    g = _parse_prbsn_name(ptype)
    if g is not None:
        return (1 << g[0]) - 1
    raise ValueError(ptype)


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------


def decode_levels(samples: np.ndarray, thr) -> np.ndarray:
    """Slice a PAM4 sample array into symbol indices 0..3 with thresholds ``thr`` (length 3)."""
    dec = np.zeros(len(samples), dtype=np.int64)
    dec[samples > thr[0]] = 1
    dec[samples > thr[1]] = 2
    dec[samples > thr[2]] = 3
    return dec


def align_symbols_fft(dec: np.ndarray, psym: np.ndarray, period: int) -> int:
    """Circular correlation alignment of decoded symbols against a periodic pattern.

    Parameters
    ----------
    dec  : decoded symbol array (length <= period for exactness).
    psym : full periodic reference symbol array (length = period).
    period : pattern period.

    Returns
    -------
    int offset ``off`` with ``tru[k] = psym[(k + off) % period]``.
    """
    n = len(dec)
    best = (-1, 0)
    for lv in range(4):
        a = (dec == lv).astype(np.float64)
        b = (psym == lv).astype(np.float64)
        xp = np.zeros(period)
        xp[:n] = a
        c = np.fft.irfft(np.fft.rfft(xp, period) * np.conj(np.fft.rfft(b, period)), period)[:period]
        j = int(np.argmax(c))
        o = (period - j) % period
        sc = int((dec == psym[(np.arange(n) + o) % period]).sum())
        if sc > best[0]:
            best = (sc, o)
    return best[1]


def bit_offset_to_symbol_offset(bo: int, period: int) -> int:
    """Map a bit-domain correlation offset to a symbol-domain offset.

    ``align_bits`` correlates ``2*period`` interleaved PAM4 bits; the
    returned ``bo`` is a bit offset with two artefacts:

    * a full PRBS *bit* period is invisible to the circular correlation, so
      ``bo`` may carry an extra ``period`` (e.g. 10 + 2047 for PRBS11);
    * an odd ``bo`` is the half-symbol (bit-pair) phase.  The bit-pair
      reference it implies equals the symbol-domain reference shifted by
      ``(bo + period) // 2`` (an interleaved-PRBS property), while the naive
      ``bo // 2`` symbol offset matches only ~25% of the symbols.

    Parameters
    ----------
    bo : raw bit offset returned by :func:`align_bits`.
    period : symbol period of the pattern.

    Returns
    -------
    int symbol offset in ``0..period-1``.
    """
    bo %= period
    if bo % 2 == 0:
        return bo // 2
    return ((bo + period) // 2) % period



def align_bits(dec: np.ndarray, patbits: np.ndarray, period_bits: int) -> int:
    """Bit-domain alignment offset for PRBS patterns.

    Parameters
    ----------
    dec : decoded PAM4 symbols (0..3).
    patbits : reference bit array of length >= period_bits.
    period_bits : bit period.

    Returns
    -------
    int bit offset.
    """
    bits = np.empty(2 * len(dec), dtype=np.int64)
    bits[0::2] = (dec >> 1) & 1
    bits[1::2] = dec & 1
    n = len(bits)
    if n >= period_bits:
        # Fold the long bit vector modulo the pattern period and correlate
        # circularly.  Maximising sum(x[k] * b[(k+off) % P]) with +/-1 coding
        # is exactly the same argmax as the direct match count (the direct
        # O(P*n) loop is intractable for 65535-symbol long records), and the
        # circular FFT correlation below returns the identical offset.
        x = 2.0 * bits - 1.0
        xf = np.zeros(period_bits, dtype=np.float64)
        np.add.at(xf, np.arange(n) % period_bits, x)
        b = 2.0 * patbits[:period_bits] - 1.0
        c = np.fft.irfft(np.fft.rfft(xf) * np.conj(np.fft.rfft(b)), period_bits)
        j = int(np.argmax(c))
        return (period_bits - j) % period_bits
    a = 2.0 * bits - 1.0
    b = 2.0 * patbits[:period_bits] - 1.0
    xp = np.zeros(period_bits)
    xp[:n] = a
    c = np.fft.irfft(np.fft.rfft(xp) * np.conj(np.fft.rfft(b)), period_bits)
    j = int(np.argmax(c))
    return (period_bits - j) % period_bits


def align_pattern(dec: np.ndarray, ptype: str, n_sym: int, period: int) -> np.ndarray:
    """Return the aligned reference symbol array ``tru`` for decoded symbols ``dec``.

    Parameters
    ----------
    dec : decoded PAM4 symbols.
    ptype : 'prbs11'/'prbs31'/'prbs13'/'ssprq'.
    n_sym : number of reference symbols to return.
    period : pattern period (symbols).

    Returns
    -------
    np.ndarray ``tru`` with ``tru[k] = pattern[(k + off) % period]``.
    """
    if ptype in ("prbs11", "prbs31"):
        bits = PRBS11_BITS if ptype == "prbs11" else PRBS31_BITS
        bo = align_bits(dec, bits, 2 * period)
        # full bit-period shifts are alignment no-ops
        bo %= period
        # an odd ``bo`` is the half-symbol (bit-pair) phase; map it to the
        # symbol-domain offset that reproduces the same reference so the
        # returned ``tru`` stays consistent with ``tru[k] = ps[(k+off)%p]``
        off = (bo // 2) if (bo % 2 == 0) else ((bo + period) // 2) % period
        ps = pattern_symbols(ptype, period)
        return ps[(np.arange(n_sym) + off) % period]
    ps = ssprq_symbols()
    off = align_symbols_fft(dec, ps, period)
    return ps[(np.arange(n_sym) + off) % period]


def lfsr_linear_symbols(nreg: int, n_ui: int, spu: int, taps) -> np.ndarray:
    """Generate a *linear* (non-folded) PAM4 symbol sequence for long PRBS records.

    Used for the 65535-symbol PRBS31 long records: the bit stream is generated
    as one continuous LFSR run (period 2**31-1) and packed MSB-first without
    folding to the 2047-symbol period.

    Parameters
    ----------
    nreg : register width (31).
    n_ui : number of UI (symbols) needed plus alignment margin.
    spu  : samples per UI (kept for API symmetry).
    taps : feedback tap indices.

    Returns
    -------
    np.ndarray of int64 length n_ui + margin.
    """
    length = 2 * (n_ui + 64)
    bits = lfsr_bits(0x402, nreg, length, taps)
    return np.array([(bits[2 * k] << 1) | bits[2 * k + 1] for k in range(n_ui + 64)],
                    dtype=np.int64)


# ---------------------------------------------------------------------------
# generic PRBSn (n = 2..31) with automatic seed / offset recovery
# ---------------------------------------------------------------------------
# The taps below are the LFSR feedback bit indices (0 = LSB) of a Fibonacci
# LFSR whose output is the MSB, i.e. the same convention as :func:`lfsr_bits`.
# For n with an IEEE/ITU standard the standard polynomial is used; the other
# entries are deterministic primitive polynomials (maximal-length, period
# 2**n - 1).  This is a fixed mathematical table - it is NOT per-case
# calibration.
PRBS_STANDARD_TAPS = {
    2: (1, 0),          # x^2+x+1
    3: (2, 0),          # x^3+x+1
    4: (3, 0),          # x^4+x+1
    5: (4, 1),          # x^5+x^2+1
    6: (5, 0),          # x^6+x+1
    7: (6, 5),          # PRBS7  x^7+x^6+1
    8: (7, 6, 1, 0), # x^8+x^7+x^2+x+1
    9: (8, 4),          # PRBS9  x^9+x^5+1
    10: (9, 2),         # x^10+x^3+1
    11: (10, 8),        # PRBS11 x^11+x^9+1
    12: (11, 5, 3, 0),  # x^12+x^6+x^4+x+1
    13: (12, 11, 9, 8), # PRBS13 x^13+x^12+x^10+x^9+1 (single LFSR)
    14: (13, 9, 5, 0),  # x^14+x^10+x^6+x+1
    15: (14, 13),       # PRBS15 x^15+x^14+1
    16: (15, 11, 2, 0), # x^16+x^12+x^3+x+1
    17: (16, 2),        # x^17+x^3+1
    18: (17, 6),        # x^18+x^7+1
    19: (18, 4, 1, 0),  # x^19+x^5+x^2+x+1
    20: (19, 2),        # PRBS20 x^20+x^3+1
    21: (20, 1),        # x^21+x^2+1
    22: (21, 0),        # x^22+x+1
    23: (22, 17),       # PRBS23 x^23+x^18+1
    24: (23, 6, 1, 0),  # x^24+x^7+x^2+x+1
    25: (24, 2),        # x^25+x^3+1
    26: (25, 5, 1, 0),  # x^26+x^6+x^2+x+1
    27: (26, 4, 1, 0),  # x^27+x^5+x^2+x+1
    28: (27, 2),        # x^28+x^3+1
    29: (28, 1),        # x^29+x^2+1
    30: (29, 5, 3, 0),  # x^30+x^6+x^4+x+1
    31: (30, 27),       # PRBS31 x^31+x^28+1
}


def prbsn_taps(n):
    """Feedback tap indices (0-based) of the PRBSn polynomial for order ``n``.

    Parameters
    ----------
    n : int in 2..31.

    Returns
    -------
    tuple of tap bit indices (ascending).
    """
    n = int(n)
    if not (2 <= n <= 31):
        raise ValueError("PRBSn order must be in 2..31, got %r" % (n,))
    return tuple(sorted(PRBS_STANDARD_TAPS[n]))


def prbsn_period(n):
    """Symbol period of a maximal-length PRBSn: ``2**n - 1``."""
    return (1 << int(n)) - 1


def prbs_polynomial_str(n, taps):
    """Human-readable polynomial string, e.g. ``x^7+x^6+x+1``."""
    terms = []
    for t in sorted(taps, reverse=True):
        e = int(t) + 1
        terms.append("x" if e == 1 else "x^%d" % e)
    return "+".join(terms) + "+1"


def pack_bits(bits):
    """Pack an even-length bit array MSB-first into PAM4 symbols (2 bits each).

    Parameters
    ----------
    bits : np.ndarray of int64 0/1.

    Returns
    -------
    np.ndarray of int64 symbols in 0..3.
    """
    bits = np.asarray(bits, dtype=np.int64)
    m = len(bits) // 2
    return (bits[0:2 * m:2] << 1) | bits[1:2 * m + 1:2]


def unpack_symbols(sym):
    """Unpack PAM4 symbols into a bit array (MSB of each symbol first)."""
    sym = np.asarray(sym, dtype=np.int64)
    out = np.empty(2 * len(sym), dtype=np.int64)
    out[0::2] = (sym >> 1) & 1
    out[1::2] = sym & 1
    return out


def recurrence_residual(bits, n, taps):
    """Vectorised LFSR-recurrence residual over every length-(n+1) window.

    residual[i] == 1 iff bits[i..i+n] violate
    ``b_{i+n} = XOR_{t in taps} b_{i+t}``.

    Parameters
    ----------
    bits : np.ndarray of int64 0/1.
    n : register width.
    taps : feedback tap indices.

    Returns
    -------
    np.ndarray of int64 with length ``max(0, len(bits) - n)``.
    """
    bits = np.asarray(bits, dtype=np.int64)
    L = len(bits)
    m = L - n
    if m <= 0:
        return np.zeros(0, dtype=np.int64)
    # The output recurrence of the Fibonacci LFSR implemented by
    # :func:`lfsr_bits` is  b[i+n] = XOR_{t in taps} b[i + n - 1 - t]
    # (register bit t is emitted n-1-t output positions later).  The tap
    # indices are therefore reversed relative to the output positions.
    rtaps = sorted((n - 1 - int(t)) for t in taps)
    r = bits[n:n + m].copy()
    for t in rtaps:
        r ^= bits[t:t + m]
    return r


def verify_prbs_bits(bits, n, taps):
    """Fraction of length-(n+1) windows that satisfy the LFSR recurrence.

    A random bit stream scores about 0.5; a clean matching PRBSn stream
    scores 1.0.  Used both for seed acceptance and for the pattern-mismatch
    warning.

    Parameters
    ----------
    bits : np.ndarray of int64 0/1.
    n, taps : PRBSn parameters.

    Returns
    -------
    float in [0, 1].
    """
    bits = np.asarray(bits, dtype=np.int64)
    L = len(bits)
    if L <= n:
        return 0.0
    r = recurrence_residual(bits, n, taps)
    return float((r == 0).mean())


def find_clean_state(bits, n, taps, min_run=None):
    """Find a bit position whose next ``n`` bits form a consistent LFSR state.

    Parameters
    ----------
    bits : np.ndarray of int64 0/1.
    n, taps : PRBSn parameters.
    min_run : minimum number of consecutive recurrence-satisfying windows
        (defaults to ``n``).

    Returns
    -------
    (start, state) where ``state`` is the register state at bit position
    ``start``, or None when no long enough clean run exists.
    """
    bits = np.asarray(bits, dtype=np.int64)
    L = len(bits)
    if L < 2 * n + 2:
        return None
    if min_run is None:
        min_run = n
    ok = (recurrence_residual(bits, n, taps) == 0).astype(np.int8)
    # vectorised longest-run scan over the boolean residual array
    pad = np.concatenate([[0], ok, [0]])
    d = np.diff(pad)
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    if len(starts) == 0:
        return None
    lens = ends - starts
    j = int(np.argmax(lens))
    if lens[j] < min_run:
        return None
    start = int(starts[j])
    state = 0
    for j in range(n):
        state = (state << 1) | int(bits[start + j])
    return start, state


def _gf2_matmul(A, B):
    """GF(2) matrix product ``A @ B`` for small int arrays."""
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    m, k = A.shape
    k2, p = B.shape
    assert k == k2
    C = np.zeros((m, p), dtype=np.int64)
    for i in range(m):
        for j in range(p):
            C[i, j] = int((A[i] & B[:, j]).sum() & 1)
    return C


def _gf2_invert(T):
    """GF(2) matrix inverse by Gauss-Jordan elimination."""
    n = T.shape[0]
    A = T.copy()
    I = np.eye(n, dtype=np.int64)
    for col in range(n):
        piv = np.nonzero(A[col:, col])[0]
        if len(piv) == 0:
            raise ValueError("LFSR transition matrix is not invertible")
        piv = col + piv[0]
        if piv != col:
            A[[col, piv]] = A[[piv, col]]
            I[[col, piv]] = I[[piv, col]]
        for r in range(n):
            if r != col and A[r, col]:
                A[r] ^= A[col]
                I[r] ^= I[col]
    return I


def _lfsr_transition(n, taps):
    """GF(2) transition matrix T with ``state_{k+1} = T @ state_k``."""
    T = np.zeros((n, n), dtype=np.int64)
    for i in range(1, n):
        T[i, i - 1] = 1
    for t in taps:
        if 0 <= int(t) < n:
            T[0, int(t)] = 1
    return T


def state_shift_forward(state, n, taps, k):
    """LFSR register state ``k`` steps later in the sequence.

    Mirrors :func:`state_shift_back`: returns the register state whose next
    output bit is the ``k``-th future bit of ``state``'s stream.  Used by the
    dual-plane PRBS13 seed recovery to move a phase-shifted recovered seed pair
    back to the true record start.
    """
    mask = (1 << n) - 1
    state &= mask
    if k <= 0:
        return state
    bits = lfsr_bits(state, n, k + n, taps)
    st = 0
    for j in range(n):
        st = (st << 1) | int(bits[k + j])
    return st & mask


def _gf2_identity(n):
    return np.eye(n, dtype=np.int64)


def state_shift_back(state, n, taps, k):
    """LFSR register state ``k`` steps earlier in the sequence.

    Returns the state whose sequence, after ``k`` forward steps, equals
    ``state``.  Implemented with a GF(2) matrix power ``(T^-1)^k``.

    Parameters
    ----------
    state : int register state.
    n : register width.
    taps : feedback taps.
    k : number of backward steps (>= 0).

    Returns
    -------
    int state.
    """
    mask = (1 << n) - 1
    if k <= 0:
        return state & mask
    T = _lfsr_transition(n, taps)
    Tinv = _gf2_invert(T)
    M = _gf2_identity(n)
    base = Tinv
    e = int(k)
    while e:
        if e & 1:
            M = _gf2_matmul(M, base)
        base = _gf2_matmul(base, base)
        e >>= 1
    vec = np.array([(state >> i) & 1 for i in range(n)], dtype=np.int64)
    out = (M @ vec) % 2
    return int((out * (1 << np.arange(n))).sum()) & mask


def _output_rows(n, taps, nrows):
    """GF(2) coefficient rows c_k with ``b_k = c_k . state0`` (state at bit 0).

    c_0 = e_{n-1}; c_{k+1}[a] = c_k[a+1] xor (c_k[0] if a in taps else 0).
    """
    rows = []
    c = np.zeros(n, dtype=np.int64)
    c[n - 1] = 1
    for _ in range(nrows):
        rows.append(c.copy())
        cn = np.zeros(n, dtype=np.int64)
        if n > 1:
            cn[:n - 1] = c[1:]
        for a in taps:
            if 0 <= int(a) < n:
                cn[int(a)] ^= c[0]
        c = cn
    return np.array(rows)


def _gf2_eliminate(rows, b, n):
    """Solve an overdetermined GF(2) system, skipping contradictory rows.

    Parameters
    ----------
    rows : (L, n) coefficient matrix.
    b : (L,) right-hand side.

    Returns
    -------
    (x, ok_frac) : unique solution when the rank is n, else (None, 0.0);
    ``ok_frac`` is the fraction of rows satisfied by ``x``.
    """
    L = len(rows)
    m = rows.copy()
    bb = np.asarray(b, dtype=np.int64).copy()
    piv = {}
    for i in range(L):
        r = m[i].copy()
        v = int(bb[i])
        for col in sorted(piv.keys()):
            if r[col]:
                r ^= m[piv[col]]
                v ^= bb[piv[col]]
        nz = np.nonzero(r)[0]
        if len(nz) == 0:
            continue
        col = int(nz[0])
        m[i] = r
        bb[i] = v
        piv[col] = i
        if len(piv) == n:
            break
    if len(piv) < n:
        return None, 0.0
    x = np.zeros(n, dtype=np.int64)
    for col in sorted(piv.keys(), reverse=True):
        r = m[piv[col]]
        v = int(bb[piv[col]])
        x[col] = v ^ int((r * x).sum() & 1)
    ok = int(((rows @ x) % 2 == b).sum())
    return x, ok / L


def _gf2_solve(bits, n, taps):
    """Recover the LFSR state at bit 0 from an overdetermined GF(2) system."""
    L = len(bits)
    nrows = min(8 * n, L - n)
    if nrows < n:
        return None
    rows = _output_rows(n, taps, nrows)
    b = np.asarray(bits[:nrows], dtype=np.int64)
    x, _ = _gf2_eliminate(rows, b, n)
    if x is None:
        return None
    chk_len = min(8 * n, L)
    chk = lfsr_bits(int((x * (1 << np.arange(n))).sum()), n, chk_len, taps)
    if float((chk == bits[:chk_len]).mean()) < 0.95:
        return None
    return int((x * (1 << np.arange(n))).sum())


def recover_seed(bits, n, taps):
    """Automatically recover the LFSR seed of a PRBSn bit stream.

    The seed is the register state at **bit position 0** of ``bits``; the
    returned bit offset is the position of the clean run used for recovery.
    No a-priori seed or offset information is used.

    Algorithm
    ---------
    1. vectorised recurrence-residual scan finds the longest run of windows
       that satisfy ``b_{i+n} = XOR_t b_{i+t}`` (exact for clean records);
    2. the n bits at the run start form a valid LFSR state; a GF(2) matrix
       power shifts it back to bit position 0;
    3. if no clean run exists (noisy / mismatched record), an overdetermined
       GF(2) solve with contradiction skipping is tried;
    4. the candidate is accepted only when it reproduces >= 95% of the next
       8n bits (this rejects false positives on random / mismatched data).

    Parameters
    ----------
    bits : np.ndarray of int64 0/1 (decoded bits of the waveform).
    n, taps : PRBSn parameters.

    Returns
    -------
    (seed, bit_offset, consistency) or None when the stream does not look
    like a PRBSn stream with the given taps.  ``consistency`` is the fraction
    of recurrence windows satisfied over the whole stream.
    """
    bits = np.asarray(bits, dtype=np.int64)
    L = len(bits)
    if L < 2 * n + 8:
        return None
    if int(bits.min()) == int(bits.max()):
        # A maximal-length LFSR never emits n+1 consecutive identical bits
        # (the all-zero state is forbidden and the all-ones state feeds back
        # away); an all-constant segment would "satisfy" the recurrence
        # trivially with the degenerate zero state, so it is rejected here
        # instead of being reported as a recovered seed.
        return None
    found = find_clean_state(bits, n, taps)
    start, seed = 0, None
    if found is not None:
        start, state = found
        chk_len = min(8 * n, L - start)
        chk = lfsr_bits(state, n, chk_len, taps)
        if float((chk == bits[start:start + chk_len]).mean()) >= 0.95:
            seed = state_shift_back(state, n, taps, start)
    if seed is None:
        seed = _gf2_solve(bits, n, taps)
        if seed is not None:
            start = 0
    if seed is None:
        return None
    if int(seed) == 0:
        # The all-zero register state is never visited by a maximal-length
        # LFSR, and its all-zero output stream satisfies the recurrence
        # trivially on any run of zeros - a mostly-zero (or chance-window)
        # decoded plane could otherwise be "recovered" as the degenerate
        # zero state.  Reject it: a valid recovery always yields a nonzero
        # state whose regenerated stream is ~50% ones.
        return None
    consistency = verify_prbs_bits(bits, n, taps)
    return int(seed), int(start), float(consistency)
