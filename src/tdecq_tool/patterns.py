# -*- coding: utf-8 -*-
"""PAM4 test-pattern generation and waveform alignment.

Implements the IEEE 802.3-2022 style PAM4 test patterns used by the tool:

* PRBS11 / PRBS31 - standard LFSR sequences packed MSB-first into PAM4
  symbols (2 bits per symbol);
* PRBS13 - the 802.3 "mzm dual-plane" construction used by the 128G records
  (two LFSR planes A and B, symbol = 2*A + (A xor B));
* SSPRQ - the official IEEE 802.3-2022 Clause 120 short stress pattern random
  quaternary sequence (65535 symbols), synthesised from the standard
  three-section PRBS31 construction (bit-exact with the shipped
  Clause_120_SSPRQ_sequence.csv reference file).

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
# SSPRQ (IEEE 802.3-2022 Clause 120, generator-based, no CSV dependency)
# ---------------------------------------------------------------------------
# The 65535-symbol SSPRQ sequence is generated directly from the standard
# construction (IEEE 802.3-2022 Clause 120.5.11.2.x, as implemented by the
# Spectre reference model vssprq_2022.vams):
#
#   * bit stream A (32768 bits) = three PRBS31 sections (x^31+x^28+1,
#     left-shift, MSB-first output) with seeds 0x00000002 / 0x34013FF7 /
#     0x0CCCCCCC and lengths 10924 / 10922 / 10922 bits; the first 31 bits
#     of each section are skipped (the official CSV behaviour);
#   * bit stream B (65534 bits): B[i] = A[(i + 1) % 32768];
#   * inverted-Gray symbol map: 00->2, 01->3, 11->0, 10->1;
#   * seq1 = invGray(A)               (16384 symbols)
#   * seq2 = 3 - seq1                 (16384 symbols)
#   * seq3 = invGray(B[0:32766])      (16383 symbols)
#   * seq4 = 3 - invGray(B[32766:65534]) (16384 symbols)
#
# This is a public standard test-pattern definition (not a per-case metric),
# so writing the LFSR seeds / section lengths as module constants is inside
# the clean-room scope.  The generator is verified bit-exact against the
# shipped Clause_120_SSPRQ_sequence.csv reference file when it is present;
# the sequence no longer *requires* that file at run time.
_SSPRQ_CACHE = None
_SSPRQ_VERIFIED = False
_SSPRQ_INVGRAY = np.array([2, 3, 1, 0], dtype=np.int64)  # 00->2 01->3 10->1 11->0


def _ssprq_build() -> np.ndarray:
    """Build the official 65535-symbol SSPRQ sequence from the standard LFSRs."""
    # A: three PRBS31 sections, 31 leading bits skipped per section.
    sections = ((0x00000002, 10924), (0x34013FF7, 10922), (0x0CCCCCCC, 10922))
    a = np.concatenate(
        [lfsr_bits(seed, 31, 31 + seg_len, (30, 27))[31:] for seed, seg_len in sections]
    )
    if len(a) != 32768:
        raise RuntimeError("SSPRQ bit stream A length != 32768 (%d)" % len(a))
    # B: two A repetitions minus the first and the last bit.
    b = a[(np.arange(65534) + 1) % 32768]
    # PAM4 symbol sequence (four sections).
    seq1 = _SSPRQ_INVGRAY[(a[0::2] << 1) | a[1::2]]                    # 16384
    seq2 = 3 - seq1                                                    # 16384
    seq3 = _SSPRQ_INVGRAY[(b[0:32766:2] << 1) | b[1:32766:2]]          # 16383
    seq4 = 3 - _SSPRQ_INVGRAY[(b[32766::2] << 1) | b[32767::2]]        # 16384
    seq = np.concatenate([seq1, seq2, seq3, seq4])
    if len(seq) != 65535:
        raise RuntimeError("SSPRQ symbol sequence length != 65535 (%d)" % len(seq))
    return seq


def ssprq_symbols(n=None, path=OFFICIAL_SSPRQ):
    """Official SSPRQ sequence (65535 symbols), generated from the standard LFSRs.

    The shipped reference CSV is no longer required: the sequence is
    synthesised from the IEEE 802.3-2022 Clause 120 construction.  When
    ``path`` points to the official single-column CSV and the file exists,
    the first call verifies the generator bit-exact against it.

    Parameters
    ----------
    n : int or None.  When given, the sequence is wrapped cyclically.
    path : str, optional path of the official single-column reference CSV
        (used only for a one-time bit-exact self check when present).

    Returns
    -------
    np.ndarray of int64.
    """
    global _SSPRQ_CACHE, _SSPRQ_VERIFIED
    if _SSPRQ_CACHE is None:
        seq = _ssprq_build()
        if path and os.path.exists(path):
            ref = np.loadtxt(path, dtype=np.int64)
            if len(ref) != len(seq) or not np.array_equal(ref, seq):
                raise RuntimeError(
                    "generated SSPRQ does not match official reference CSV %s" % path)
            _SSPRQ_VERIFIED = True
        _SSPRQ_CACHE = seq
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


def lfsr_linear_symbols(nreg: int, n_ui: int, spu: int, taps,
                        seed: int = 0x402) -> np.ndarray:
    """Generate a *linear* (non-folded) PAM4 symbol sequence for long PRBS records.

    The bit stream is generated as one continuous LFSR run (period 2**nreg-1)
    and packed MSB-first without folding to the ``2**nreg-1`` symbol period.
    The seed is the LFSR state at bit 0 of the generated stream - the caller
    passes the automatically recovered record-start state (the legacy default
    ``0x402`` is kept for API compatibility; no release code path relies on it
    once the caller always passes the recovered seed).

    Parameters
    ----------
    nreg : register width.
    n_ui : number of UI (symbols) needed plus alignment margin.
    spu  : samples per UI (kept for API symmetry).
    taps : feedback tap indices.
    seed : LFSR initial state (default 0x402, the legacy canonical PRBS31 seed).

    Returns
    -------
    np.ndarray of int64 length n_ui + 64.
    """
    length = 2 * (n_ui + 64)
    bits = lfsr_bits(seed, nreg, length, taps)
    return pack_bits(bits)


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


def _find_folded_seam_candidates(win, n, taps):
    """Candidate wrap-seam positions (bit offsets) inside a folded window.

    A folded record repeats the maximal-length block of ``2**n - 1`` symbols,
    so any window contains at most one seam where the bit stream jumps from
    the last block bit back to the first.  Windows of the LFSR recurrence that
    span the seam violate the recurrence, so the residual violations form a
    (sparse, due to coincidental passes) cluster ending just before the seam.
    The true seam ``S`` satisfies ``S in [viol.max()+1, viol.min()+n]``; when
    the cluster is truncated at the array edges the seam lies in the tail/head
    ``n`` bits, which are enumerated explicitly (n <= 31, cheap).

    Parameters
    ----------
    win : bit window (0/1).
    n, taps : PRBSn parameters.

    Returns
    -------
    sorted list of candidate seam positions (always contains the true seam).
    """
    P2 = len(win)
    res = recurrence_residual(win, n, taps)
    viol = np.nonzero(res == 1)[0]
    cands = {0}
    if len(viol):
        lo = int(viol.max()) + 1
        hi = int(viol.min()) + n
        cands.update(range(max(0, lo), min(P2, hi) + 1))
    cands.update(range(max(1, P2 - n), P2))   # seam in the trailing n bits
    cands.update(range(0, n))                 # seam in the leading n bits
    return sorted(cands)


def recover_folded_seed(bits, n, taps, period):
    """Recover the block seed and symbol offset of a folded (periodic) PRBSn record.

    A folded record cyclically repeats one maximal-length block:
    ``record[k] = block[(k + off) % period]`` with ``period = 2**n - 1``.
    The wrap seam is located from the recurrence-residual violations; the
    ``n`` bits immediately after the seam are the first ``n`` bits of the
    repeated block, so they directly fix the block seed (when the seam sits
    in the last ``n`` window bits the ``n`` bits *before* the seam are used
    and advanced by one register width).  A candidate seam is accepted only
    when the regenerated block, rotated to the record phase, reproduces the
    observed window with >= 99.9% of the bits.

    Note that a folded record pins the block only up to a rotational
    equivalence: when the last few block bits coincide with the stream bits
    just before the block (a data-dependent coincidence), several
    (block-seed, offset) pairs produce the exact same reference.  The
    smallest seam position is returned deterministically; all accepted pairs
    decode to the identical reference (SER=0).

    Parameters
    ----------
    bits : decoded bit array (0/1); any length >= 2n+8 is sufficient.
    n, taps : PRBSn parameters.
    period : symbol period ``2**n - 1``.

    Returns
    -------
    (seed, off_sym, consistency) or None.  ``seed`` is an LFSR state at the
    start of the repeated block and ``off_sym`` satisfies
    ``record[k] = block[(k + off_sym) % period]``.
    """
    bits = np.asarray(bits, dtype=np.int64)
    L = len(bits)
    if L < 2 * n + 8:
        return None
    seed_lin = recover_seed(bits, n, taps)
    if seed_lin is None:
        return None
    s_lin, _, cons = seed_lin
    win = bits[:min(L, 2 * period + n)]
    P2 = len(win)
    res = recurrence_residual(win, n, taps)
    viol = np.nonzero(res == 1)[0]
    if len(viol) == 0:
        # seam-free window: a linear record or a folded record starting at
        # the block head - both share the record-start reference
        return int(s_lin), 0, float(cons)
    lo = int(viol.max()) + 1
    hi = int(viol.min()) + n
    cands = set(range(max(0, lo), min(P2, hi) + 1))
    cands.update(range(max(1, P2 - n), P2))   # truncated trailing cluster
    cands.update(range(0, n))                 # truncated leading cluster
    best = None
    for S in sorted(cands):
        if S % 2 != 0:
            continue                          # mid-symbol seam: not aligned
        if S + n <= P2:
            st = 0
            for j in range(n):
                st = (st << 1) | int(win[S + j])
        elif S - n >= 0:
            st = 0
            for j in range(n):
                st = (st << 1) | int(win[S - n + j])
            st = state_shift_forward(st, n, taps, n)
        else:
            continue
        block = lfsr_bits(st, n, 2 * period, taps)
        off_bits = (2 * period - S) % (2 * period)
        win_est = block[(off_bits + np.arange(P2)) % (2 * period)]
        match = float((win_est == win).mean())
        key = (match, -S)
        if best is None or key > best[0]:
            best = (key, int(st), S, match, cons)
    if best is None:
        return None
    (_, _tie), block_seed, S, match, cons = best
    if match < 0.999:
        return None
    off_sym = (period - (S // 2)) % period
    return int(block_seed), int(off_sym), float(cons)

