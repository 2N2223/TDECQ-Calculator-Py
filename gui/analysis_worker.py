# -*- coding: utf-8 -*-
"""Analysis worker process for the GUI.

The GUI runs every analysis (one record, or a whole batch) in a separate
process started with :func:`start`, so the Stop button can terminate it at
any point - a Python thread cannot be interrupted inside ``analyze``.

The child reports through a ``multiprocessing`` queue with the same
``(kind, payload)`` messages the GUI already handles: ``log``, ``clear``,
``single-done``, ``batch-progress``, ``batch-row``, ``batch-done`` and
``error``.  This module imports no tkinter, so the child stays light.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import queue
import sys
import threading
import traceback
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

LEGACY_OPTICAL_NOTE = ("legacy mode uses the historical optical front end "
                       "(no reference receiver, dispersion as a delay); its "
                       "optical results are not standard-compliant")


class QueueWriter:
    """File-like object that forwards writes into the GUI queue."""

    def __init__(self, q):
        self.q = q

    def write(self, text):
        if text:
            self.q.put(("log", str(text)))

    def flush(self):
        pass


def metric_key(result):
    return "tecq_db" if result.get("tecq_db") is not None else "tdecq_db"


def start(job, *args, in_process=False):
    """Start ``job`` ('single' or 'batch') in a new process.

    Returns ``(process, queue)``.  A fresh queue per run means a process
    terminated while writing can never corrupt the next run's messages.
    ``in_process=True`` runs the job in a thread of this process instead
    (fallback for hosts where the worker process cannot start); Stop then
    abandons the thread - its remaining output is discarded - but cannot
    end its computation early.
    """
    target = _run_single if job == "single" else _run_batch
    if in_process:
        q = queue.Queue()
        job_ = _ThreadJob(target, (q,) + tuple(args))
        return job_, q
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=target, args=(q,) + tuple(args), daemon=True)
    proc.start()
    return proc, q


class _StdoutRouter:
    """sys.stdout/stderr replacement that routes writes per thread."""

    def __init__(self, fallback):
        self.fallback = fallback
        self.routes = {}

    def write(self, text):
        q = self.routes.get(threading.get_ident())
        if q is None:
            return self.fallback.write(text)
        if text:
            q.put(("log", str(text)))
        return len(text)

    def flush(self):
        if threading.get_ident() not in self.routes:
            self.fallback.flush()


_ROUTERS = []


def _route_output(q):
    """Send this thread's stdout/stderr to ``q`` (in-process mode)."""
    if not _ROUTERS:
        _ROUTERS.append((_StdoutRouter(sys.stdout), _StdoutRouter(sys.stderr)))
        sys.stdout, sys.stderr = _ROUTERS[0]
    for r in _ROUTERS[0]:
        r.routes[threading.get_ident()] = q


def _unroute_output():
    if _ROUTERS:
        for r in _ROUTERS[0]:
            r.routes.pop(threading.get_ident(), None)


class _ThreadJob:
    """Process-like handle of an in-process (thread) job."""

    exitcode = 0

    def __init__(self, target, args):
        self.thread = threading.Thread(target=self._run, args=(target, args),
                                       daemon=True)
        self.thread.start()

    @staticmethod
    def _run(target, args):
        _route_output(args[0])
        try:
            target(*args)
        finally:
            _unroute_output()

    def is_alive(self):
        return self.thread.is_alive()

    def terminate(self):
        pass        # a thread cannot be killed; the GUI drops its queue

    def join(self, timeout=None):
        pass


def _capture(q):
    """Redirect stdout/stderr of a worker *process* into ``q``."""
    if threading.current_thread() is threading.main_thread():
        sys.stdout = sys.stderr = QueueWriter(q)


def _emit_notes(result, wlist, kwargs):
    for wmsg in wlist:
        if issubclass(wmsg.category, (RuntimeWarning, UserWarning)):
            print("WARNING: %s" % wmsg.message)
    if result.get("metric_credible") is False:
        print("NOTE: result not credible: %s"
              % result.get("metric_credible_reason"))
    if result.get("pattern_match_warning"):
        print("NOTE: pattern_match_warning = %s"
              % result["pattern_match_warning"])
    if result.get("ffe_bounds_warning"):
        print("NOTE: ffe_bounds_warning = %s" % result["ffe_bounds_warning"])
    if result.get("ffe_geometry_override"):
        print("NOTE: ffe_geometry_override = %s"
              % result.get("ffe_geometry_override_reason"))
    if result.get("ffe_geometry_locked"):
        print("NOTE: ffe_geometry_locked = requested %d-tap geometry "
              "(pre=%d/post=%d) was honored"
              % (result.get("ffe_tap_count", 0),
                 result.get("ffe_pre_cursor", 0),
                 result.get("ffe_post_cursor", 4)))
    if kwargs["tdecq_mode"] == "legacy" and kwargs["domain"] == "optical":
        print("NOTE: " + LEGACY_OPTICAL_NOTE)


def _analyze_one(path, case_id, kwargs, out_json):
    from tdecq_tool.pipeline import analyze
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        result = analyze(path, case_id=case_id, **kwargs)
    key = metric_key(result)
    if out_json:
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print("wrote %s (%s = %.4f dB)" % (out_json, key, result.get(key)))
    else:
        print("%s = %.4f dB  (JSON output disabled)" % (key, result.get(key)))
    _emit_notes(result, wlist, kwargs)
    return result


def _plot_data(path, result, kwargs):
    from tdecq_tool.pipeline import prepare_plot_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return prepare_plot_data(
            path, pattern=kwargs["pattern"], rate_gbps=kwargs["rate_gbps"],
            domain=kwargs["domain"], samples_per_ui=kwargs["samples_per_ui"],
            window=kwargs["window"],
            ffe_pre_cursor=int(result.get("ffe_pre_cursor", 0)),
            ffe_post_cursor=int(result.get("ffe_post_cursor", 4)),
            tap_weights=result.get("tap_weights"),
            dfe_b1=float(result.get("dfe_b1", 0.0) or 0.0),
            standard_measurement=result.get("standard_measurement"))


def _plain(obj):
    """Plot data without the raw waveform (not drawn; keeps the IPC small)."""
    return {k: v for k, v in obj.items() if k not in ("v", "Mj")}


def _run_single(q, path, out, kwargs, want_plot):
    _capture(q)
    try:
        result = _analyze_one(path, os.path.basename(path), kwargs, out)
        plot_data = None
        if want_plot:
            try:
                plot_data = _plain(_plot_data(path, result, kwargs))
            except Exception as exc:
                print("plot failed: %r" % (exc,))
        q.put(("single-done", (path, result, plot_data, kwargs)))
    except Exception as exc:
        traceback.print_exc()
        q.put(("error", "".join(
            traceback.format_exception_only(type(exc), exc))))


def _run_batch(q, files, outdir, kwargs):
    _capture(q)
    n = 0
    for i, path in enumerate(files):
        name = os.path.basename(path)
        q.put(("batch-progress", "running %d/%d: %s"
               % (i + 1, len(files), name)))
        # every new record clears the log area, like the single view
        q.put(("clear", None))
        try:
            out_json = (os.path.join(outdir, os.path.splitext(name)[0]
                                     + "_tdecq.json") if outdir else None)
            result = _analyze_one(path, name, kwargs, out_json)
            notes = []
            if result.get("metric_credible") is False:
                notes.append("not credible: %s"
                             % result.get("metric_credible_reason"))
            if result.get("pattern_match_warning"):
                notes.append(str(result["pattern_match_warning"]))
            if kwargs["tdecq_mode"] == "legacy" and \
                    kwargs["domain"] == "optical":
                notes.append("legacy optical front end")
            row = dict(file=name, path=path,
                       mode=result.get("tdecq_mode", "legacy"),
                       metric=result.get(metric_key(result)),
                       credible=result.get("metric_credible"),
                       ser=result.get("actual_SER"),
                       ceq=result.get("C_eq"),
                       seed_ok=result.get("seed_auto_detected"),
                       seed=result.get("prbs_seed"),
                       seed_a=result.get("prbs_seed_a"),
                       seed_b=result.get("prbs_seed_b"),
                       pattern=result.get("test_pattern"),
                       taps=result.get("n_taps"),
                       strategy=result.get("chosen_strategy"),
                       warn=bool(notes), warn_text="; ".join(notes),
                       json=out_json, error=None)
        except Exception as exc:
            traceback.print_exc()
            row = dict(file=name, path=path, mode=kwargs["tdecq_mode"],
                       metric=None, credible=None, ser=None, ceq=None,
                       seed_ok=None, seed=None, seed_a=None, seed_b=None,
                       pattern=None, taps=None, strategy=None, warn=True,
                       warn_text="error: %s" % exc, json=None,
                       error="%s: %s" % (type(exc).__name__, exc))
        n += 1
        q.put(("batch-row", row))
    q.put(("batch-done", (n, len(files), False)))
