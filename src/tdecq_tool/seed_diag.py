# -*- coding: utf-8 -*-
"""Opt-in, ASCII-only diagnostics for PRBS seed recovery.

The measurement API deliberately keeps these diagnostics out of normal output:
they are intended for remote waveform triage, where the operator needs to see
which sampling phase / slicer thresholds failed without transferring samples.
Set ``TDECQ_SEED_DIAGNOSTICS=1`` (or pass the CLI flag) to enable them.  When
``TDECQ_SEED_DIAGNOSTICS_FILE`` is set, records are appended to that file;
otherwise they are written to stderr.
"""
from __future__ import annotations

import contextvars
import math
import os
import sys


_PREFIX = "TDECQ_SEED_DIAG"
_CONFIG = contextvars.ContextVar("tdecq_seed_diag_config", default=None)


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def configure(enabled=None, path=None) -> bool:
    """Configure diagnostics for the current analysis context.

    ``enabled=None`` leaves enablement to ``TDECQ_SEED_DIAGNOSTICS``.  A
    supplied file path enables diagnostics even without the boolean switch,
    which makes ``--seed-diagnostics-file`` convenient for batch jobs.
    """
    env_enabled = _truthy(os.environ.get("TDECQ_SEED_DIAGNOSTICS"))
    env_path = os.environ.get("TDECQ_SEED_DIAGNOSTICS_FILE")
    out_path = path or env_path
    active = env_enabled or bool(enabled) or bool(out_path)
    _CONFIG.set((active, out_path))
    if active:
        emit("start", sink=("file" if out_path else "stderr"))
    return active


def enabled() -> bool:
    """Whether seed diagnostics are enabled in the current context."""
    cfg = _CONFIG.get()
    return bool(cfg and cfg[0])


def _value(value) -> str:
    """Format one value as a compact ASCII token (never waveform samples)."""
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return format(value, ".9g")
    if isinstance(value, (tuple, list)):
        return ",".join(_value(x) for x in value)
    text = str(value)
    text = text.encode("ascii", "backslashreplace").decode("ascii")
    return "".join(ch if (ch.isalnum() or ch in "._,:+-=") else "_"
                   for ch in text)


def emit(event: str, **fields) -> None:
    """Append one stable key=value diagnostic line when enabled.

    The format is intentionally line-oriented and ASCII-only so it can be
    collected from a remote CLI's stderr and parsed with standard shell tools.
    """
    cfg = _CONFIG.get()
    if not cfg or not cfg[0]:
        return
    parts = [_PREFIX, "event=" + _value(event)]
    for key in sorted(fields):
        parts.append(_value(key) + "=" + _value(fields[key]))
    line = " ".join(parts) + "\n"
    path = cfg[1]
    if path:
        with open(path, "a", encoding="ascii", errors="backslashreplace") as fh:
            fh.write(line)
    else:
        sys.stderr.write(line)
        sys.stderr.flush()


def threshold_values(thresholds):
    """Return thresholds rounded only for diagnostic text."""
    return tuple(float(x) for x in thresholds)
