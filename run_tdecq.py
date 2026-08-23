#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TDECQ (optical) / TECQ (electrical) measurement tool - CLI entry, V1.3.

Fully automatic measurement of PAM4 received waveforms (IEEE 802.3-2022
style TDECQ / TECQ).  Only the waveform CSV, the pattern family, the data
rate, the domain and the samples-per-UI are required:

* ``--pattern prbsN`` (N in 2..31) automatically recovers a single-LFSR PRBS
  seed, symbol offset and test-window length from the waveform (no a-priori
  seed / offset); this includes ``--pattern prbs13``;
* ``--pattern prbs13dual`` automatically searches the two 802.3 MZM dual-plane
  LFSR seeds (``seed_a`` / ``seed_b``) and their phases - no hardcoded
  (1,137) seed;
* ``--pattern auto`` resolves the pattern from the waveform / CSV header
  (never from the optional ``--case`` diagnostic label);
* ``--pattern prbs11 / prbs31 / ssprq`` use the release reference streams.

The FFE tap geometry is configurable: ``--ffe-pre-cursor N`` (0..5) and
``--ffe-post-cursor M`` (0..30) place the main tap at index ``N`` with
``N + 1 + M`` total taps; the default (0, 4) reproduces the 5-tap release
geometry bit-for-bit.  ``--ffe-taps T`` is a convenience total-tap form
(equivalent to ``--ffe-post-cursor T-1-<pre>`` unless the post count is
explicit).

The IEEE-802.3dj-style bounded multi-tap FFE is opt-in: ``--ffe-bounds``
enables the constrained candidate pool (bounded QP / active-set / damped
MMSE / progressive coordinate descent over tap counts N in {5, 7, 9} plus
the explicitly requested geometry, main-tap position from the pulse-response
peak).  It is an *experimental* 802.3dj draft constraint (not part of IEEE
802.3-2022 TDECQ) and is off by default so the release behaviour stays
bit-for-bit.  When the pool auto-selects a geometry that differs from the
requested one, or the reported taps still violate the bound table, the JSON
carries explicit ``ffe_geometry_override`` / ``ffe_bounds_warning`` fields
instead of silently downgrading.
``--no-equalized-validation`` disables the validated-equalised branch on
closed-eye canonical-fallback records.

CSV parsing is fully automatic: Virtuoso exports (``VT(...) X,Y`` headers,
any node name, uniform or adaptive grid) and plain ``time,out`` records are
detected from the header, and the time/signal columns and units are
normalised without any per-file configuration.  A pattern-mismatch
``RuntimeWarning`` is emitted and stored in the JSON as
``pattern_match_warning``.

Usage:
    python run_tdecq.py --input <waveform.csv> --rate-gbps <112|128> \
                        --pattern <auto|prbs11|prbs13|prbs13dual|prbs31|ssprq|prbsN> \
                        --domain <electrical|optical> \
                        [--samples-per-ui auto] [--window auto] \
                        [--ffe-pre-cursor <0..5>] [--ffe-post-cursor <0..30>] \
                        [--ffe-taps <T>] [--ffe-bounds] \
                        [--threshold-mode <linear|class_midpoint>] \
                        [--seed-diagnostics [--seed-diagnostics-file <path>]] \
                        [--no-equalized-validation] --out <output.json> \
                        [-p <plot.png>] [--eye-color <color>] [--case <id>] [--no-plot]
"""
import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from tdecq_tool.pipeline import analyze
from tdecq_tool.plot import make_diagnostic_plot


def _ascii_text(value):
    """Escape non-ASCII text so every CLI byte is safe for remote logs."""
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def _choices():
    return ["auto", "prbs11", "prbs13", "prbs13dual", "prbs31", "ssprq"] + \
        ["prbs%d" % n for n in range(2, 32) if n not in (11, 13, 31)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="input waveform CSV")
    ap.add_argument("--rate-gbps", type=float, default=112)
    ap.add_argument("--pattern", default="auto", choices=_choices())
    ap.add_argument("--domain", default="electrical",
                    choices=["electrical", "optical"])
    ap.add_argument("--samples-per-ui", default="auto",
                    help="samples per UI: 32 for 128G, 64 for 112G, or 'auto'")
    ap.add_argument("--window", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", default=None,
                    help="diagnostic case label (never affects the algorithm)")
    ap.add_argument("-p", "--output-plot", default=None)
    ap.add_argument("--eye-color", default="blue",
                    help="eye-diagram density colour for --output-plot")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--ffe-pre-cursor", type=int, default=0, metavar="N",
                    help="FFE pre-cursor tap count before the main tap, 0..5 "
                         "(default 0 = V1.1.1 5-tap geometry)")
    ap.add_argument("--ffe-post-cursor", type=int, default=None, metavar="M",
                    help="FFE post-cursor tap count after the main tap, 0..30 "
                         "(default 4 = V1.1.1 5-tap geometry)")
    ap.add_argument("--ffe-taps", type=int, default=None, metavar="T",
                    help="convenience total tap count; equivalent to "
                         "--ffe-post-cursor T-1-<pre> unless --ffe-post-cursor "
                         "is given explicitly")
    ap.add_argument("--no-equalized-validation", action="store_true",
                    help="disable the validated-equalised branch on closed-eye "
                         "canonical-fallback records (always report the honest "
                         "identity eye)")
    ap.add_argument("--ffe-bounds", action="store_true",
                    help="enable the IEEE 802.3dj-style bounded multi-tap FFE "
                         "candidate pool (bounded QP/active-set/damped/"
                         "progressive-CD over N in {5,7,9} plus the requested "
                         "geometry; experimental 802.3dj draft constraint, "
                         "off by default to keep the release behaviour "
                         "bit-for-bit)")
    ap.add_argument("--threshold-mode", default="linear",
                    choices=["linear", "class_midpoint"],
                    help="PAM4 threshold model for sigma fitting and eye "
                    "diagnostics: linear keeps the IEEE-compatible "
                    "Pave+/-OMA/3 thresholds; class_midpoint uses adjacent "
                    "measured-level midpoints for nonlinear eyes")
    ap.add_argument("--seed-diagnostics", action="store_true",
                    help="emit ASCII-only PRBS seed/phase/threshold "
                    "diagnostics to stderr (default: off)")
    ap.add_argument("--seed-diagnostics-file", default=None, metavar="PATH",
                    help="append ASCII-only PRBS seed diagnostics to PATH; "
                    "also enables diagnostics")
    args = ap.parse_args()

    if not (0 <= args.ffe_pre_cursor <= 5):
        raise SystemExit("--ffe-pre-cursor must be in 0..5")
    if args.ffe_post_cursor is not None and not (0 <= args.ffe_post_cursor <= 30):
        raise SystemExit("--ffe-post-cursor must be in 0..30")
    if args.ffe_taps is not None:
        if args.ffe_taps < 1 or args.ffe_taps > 36:
            raise SystemExit("--ffe-taps must be in 1..36")
        if args.ffe_post_cursor is not None:
            # explicit post-cursor: only validate consistency
            if args.ffe_taps != args.ffe_pre_cursor + 1 + args.ffe_post_cursor:
                raise SystemExit(
                    "--ffe-taps must equal --ffe-pre-cursor+1+--ffe-post-cursor")
        else:
            args.ffe_post_cursor = args.ffe_taps - 1 - args.ffe_pre_cursor
            if args.ffe_post_cursor < 0 or args.ffe_post_cursor > 30:
                raise SystemExit("--ffe-taps gives out-of-range post-cursor %d"
                                 % args.ffe_post_cursor)
    if args.ffe_post_cursor is None:
        args.ffe_post_cursor = 4

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        out = analyze(args.input, pattern=args.pattern,
                      rate_gbps=args.rate_gbps, domain=args.domain,
                      samples_per_ui=args.samples_per_ui, window=args.window,
                      case_id=args.case,
                      ffe_pre_cursor=args.ffe_pre_cursor,
                      ffe_post_cursor=args.ffe_post_cursor,
                      equalized_validation=not args.no_equalized_validation,
                      ffe_bounds_enabled=args.ffe_bounds,
                      threshold_mode=args.threshold_mode,
                      seed_diagnostics=args.seed_diagnostics,
                      seed_diagnostics_file=args.seed_diagnostics_file,
                      verbose=True)
    for w in wlist:
        if issubclass(w.category, RuntimeWarning):
            print("WARNING: %s" % _ascii_text(w.message), file=sys.stderr)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    key = "tecq_db" if args.domain == "electrical" else "tdecq_db"
    print("wrote %s (%s = %.4f dB)" % (_ascii_text(args.out), key,
                                       out.get(key)))
    if out.get("pattern_match_warning"):
        print("NOTE: pattern_match_warning = %s" %
              _ascii_text(out["pattern_match_warning"]))
    if out.get("ffe_bounds_warning"):
        print("NOTE: ffe_bounds_warning = %s" %
              _ascii_text(out["ffe_bounds_warning"]))
    if out.get("ffe_geometry_override"):
        print("NOTE: ffe_geometry_override = %s"
              % _ascii_text(out.get("ffe_geometry_override_reason")))
    if args.output_plot and not args.no_plot:
        make_diagnostic_plot(args.input, out, args.output_plot,
                             rate_gbps=args.rate_gbps, domain=args.domain,
                             samples_per_ui=args.samples_per_ui,
                             case_id=args.case, pattern=args.pattern,
                             ffe_pre_cursor=out.get("ffe_pre_cursor",
                                                    args.ffe_pre_cursor),
                             ffe_post_cursor=out.get("ffe_post_cursor",
                                                     args.ffe_post_cursor),
                             eye_color=args.eye_color)
        print("wrote %s" % _ascii_text(args.output_plot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
