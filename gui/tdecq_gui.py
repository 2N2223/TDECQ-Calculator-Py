#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TDECQ/TECQ measurement tool - cross-platform tkinter GUI.

Single-file GUI on top of ``tdecq_tool.pipeline.analyze``:

* single-record analysis with the CLI measurement options exposed,
* live analysis log + key-metric summary + embedded equalised-eye plot,
* batch mode: queue a folder / file selection, run all records in the
  background, tabulate the results and export them as CSV.

Worker process
--------------
All analysis work (``analyze`` and the plot re-analysis) runs in a separate
process (:mod:`gui.analysis_worker`), one run at a time, so the Stop buttons
can end it immediately.  The Tk main thread only reads the settings, renders
the messages it receives and draws the eye.

Cross-platform notes
--------------------
* tkinter/ttk ships with the Python installers for Windows and macOS; on
  Debian/Ubuntu Linux install ``python3-tk`` (``sudo apt install python3-tk``)
  and on Fedora ``python3-tkinter``.
* matplotlib (TkAgg backend) is required only for the embedded eye plot;
  without it the GUI still runs and just disables the plot panel.

Launch with ``python run_gui.py`` from the repository root.
"""
from __future__ import annotations

import csv
import os
import queue
import sys

# see run_gui.py: Intel OpenMP fork handlers abort the analysis worker on
# MKL builds of numpy; must be set before numpy is imported
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
for _p in (SRC, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from tdecq_tool import patterns, standards
from tdecq_tool.optical import REF_RX_BW_OVER_BAUD
from tdecq_tool.pipeline import PATTERN_PRBS13_DUAL, validate_pattern_request

from gui import analysis_worker
from gui.analysis_worker import LEGACY_OPTICAL_NOTE

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.colors as mcolors
    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                   NavigationToolbar2Tk)
    from matplotlib.figure import Figure
    from tdecq_tool import plot as tplot
    HAVE_MPL = True
except Exception:  # pragma: no cover - plot-less environments
    HAVE_MPL = False

APP_TITLE = "TDECQ / TECQ Measurement Tool"
EYE_BG = "#05050f"
EYE_FG = "#d8d8e0"
PATTERNS = (["auto", "custom", "ssprq", "prbs13q", "prbs31q", "prbs13dual"]
            + ["prbs%d" % n for n in range(2, 32)])
CUSTOM = "custom"
EYE_FORMATS = [("PNG image", "*.png"), ("SVG vector", "*.svg"),
               ("PDF", "*.pdf")]


def pattern_description(pattern, custom_poly="", custom_gray=False):
    """One-line description of the polynomial / mapping a pattern uses.

    Returns ``(text, ok)``; ``ok`` is False when a custom polynomial does
    not parse (``text`` is then the error).
    """
    p = str(pattern or "").strip().lower()
    if p == "auto":
        return "detected from the waveform (header / record shape)", True
    if p == CUSTOM:
        if not str(custom_poly).strip():
            return "enter a polynomial, e.g. x^13+x^4+x^3+x+1", False
        name = ("prbsq:" if custom_gray else "prbs:") + str(custom_poly)
        try:
            n, taps, gray = patterns.parse_custom_prbs_name(name)
        except ValueError as exc:
            return str(exc), False
        return ("PRBS%d %s, %s mapping, exactly these taps" % (
            n, patterns.prbs_polynomial_str(n, taps),
            "802.3 Gray" if gray else "binary"), True)
    if p == "ssprq":
        return "SSPRQ (802.3 Clause 120.5.11.2.3, 65535-symbol pattern)", True
    if p == PATTERN_PRBS13_DUAL:
        return ("two-plane PRBS13 (mzm, two x^13+x^12+x^10+x^9+1 LFSRs)",
                True)
    q = patterns.parse_prbsq_name(p)
    if q is not None:
        return ("PRBS%dQ %s, 802.3 Gray mapping" % (
            q[0], patterns.prbs_polynomial_str(*q)), True)
    if p.startswith("prbs") and p[4:].isdigit():
        n = int(p[4:])
        text = "PRBS%d %s" % (n, patterns.prbs_polynomial_str(
            n, patterns.prbsn_taps(n)))
        if n == 13:
            text += " (or x^13+x^4+x^3+x+1)"
        return text + ", level map detected automatically", True
    return "", True

_metric_key = analysis_worker.metric_key


def _fmt(v, spec="%.4g"):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return spec % v if np.isfinite(v) else str(v)
    return str(v)


def summary_lines(result):
    """Human-readable result summary (mode-aware, units from the domain)."""
    std = result.get("tdecq_mode") == "standard"
    unit = "mV" if result.get("Swing_mV") is not None else "mW"
    mkey = _metric_key(result)
    name = "TECQ" if mkey == "tecq_db" else "TDECQ"
    pre = "tecq" if mkey == "tecq_db" else "tdecq"
    out = []

    def add(label, value, spec="%.4g"):
        if value is not None:
            out.append("%-18s %s" % (label + ":", _fmt(value, spec)))

    add(name + " (dB)", result.get(mkey), "%.3f")
    add("mode", result.get("tdecq_mode", "legacy"))
    if std:
        add("credible", result.get("metric_credible"))
        if result.get("metric_credible") is False:
            add("  reason", result.get("metric_credible_reason"))
    raw = result.get(pre + "_raw_db")
    # the legacy path reports a 1e9 dB sentinel when the unequalised eye
    # is closed
    add("no-FFE (dB)", "eye closed" if raw is not None and raw >= 1e6
        else raw, "%.3f")
    add("actual SER", result.get("actual_SER"), "%.3g")
    add("C_eq", result.get("C_eq"), "%.4f")
    add("sigma_G (%s)" % unit, result.get("sigma_G_" + unit))
    add("OMA (mW)" if unit == "mW" else "Swing (mV)",
        result.get("OMA_mW" if unit == "mW" else "Swing_mV"))
    add("P_ave (%s)" % unit, result.get("P_ave_" + unit))
    taps = result.get("tap_weights")
    if taps:
        add("FFE taps", "%d (pre=%s, post=%s)" % (
            len(taps), result.get("ffe_pre_cursor"),
            result.get("ffe_post_cursor")))
        add("tap weights", ", ".join("%.4f" % x for x in taps))
        add("largest tap idx", result.get("ffe_largest_tap_index"))
    if result.get("dfe_b1"):
        add("DFE b1", result.get("dfe_b1"), "%.4f")
    sm = result.get("standard_measurement") or {}
    if std:
        add("eye centre", sm.get("eye_centre"), "%.2f samples")
        add("level source", result.get("level_source"))
        if sm.get("sigma_s"):
            add("sigma_S (%s)" % unit, sm.get("sigma_s"))
        rx = result.get("reference_receiver")
        if rx:
            add("reference RX", "%s, %.4g GHz (%.2f x baud)" % (
                rx.get("model"), rx.get("bw_ghz"), rx.get("bw_over_baud")))
    else:
        add("cdr phase", result.get("cdr_estimated"))
        add("sigma method", result.get("sigma_method"))
        add("family", result.get("selected_family"))
        add("strategy", result.get("chosen_strategy"))
    add("pattern", result.get("test_pattern"))
    add("PRBS order", result.get("prbs_register"))
    if result.get("prbs_polynomial") not in (None, "standard"):
        add("polynomial", result.get("prbs_polynomial"))
    if result.get("gray_mapping") not in (None, "standard"):
        add("level map", result.get("gray_mapping"))
    add("seed auto", result.get("seed_auto_detected"))
    if result.get("prbs_seed") is not None:
        add("seed", result.get("prbs_seed"))
    elif result.get("prbs_seed_a") is not None:
        add("seeds", "A=%s B=%s" % (result.get("prbs_seed_a"),
                                     result.get("prbs_seed_b")))
    add("offset", result.get("offset_estimated"))
    add("standard", result.get("standard"))
    add("RLM", result.get("RLM"))
    add("warning", result.get("pattern_match_warning") or None)
    add("elapsed (s)", result.get("elapsed_sec"), "%.2f")
    return out


class TdecqGui:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1180x860")
        self.root.minsize(900, 640)
        self.log_q = None          # queue of the running worker process
        self.proc = None           # running worker process
        self.job = None            # 'single' / 'batch' while busy
        self.busy = False
        # run jobs in a thread when the worker process cannot start (set
        # automatically after a start failure, or TDECQ_GUI_INPROCESS=1)
        self.in_process = os.environ.get("TDECQ_GUI_INPROCESS", "") == "1"
        self._job_args = None
        self._got_msg = False
        self.batch_rows = []
        self.batch_files = []
        self._build_ui()
        self._on_mode_change()
        self._on_standard_change()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(120, self._drain_log)

    def _on_close(self):
        # never leave an orphaned analysis process behind
        if self.proc is not None and self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=5)
        self.root.destroy()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        self.tab_single = ttk.Frame(self.notebook)
        self.tab_batch = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_single, text="Single analysis")
        self.notebook.add(self.tab_batch, text="Batch")

        self._build_single_tab()
        self._build_batch_tab()

    def _build_single_tab(self):
        pad = dict(padx=6, pady=4)
        top = ttk.LabelFrame(self.tab_single, text="Measurement settings")
        top.pack(fill="x", **pad)

        self.var_input = tk.StringVar()
        self.var_out = tk.StringVar()
        self.var_rate = tk.StringVar(value="112")
        self.var_pattern = tk.StringVar(value="auto")
        self.var_domain = tk.StringVar(value="electrical")
        self.var_spu = tk.StringVar(value="auto")
        self.var_window = tk.StringVar(value="auto")
        self.var_standard = tk.StringVar(value=standards.DEFAULT_STANDARD)
        self.var_pre = tk.StringVar(value="")
        self.var_post = tk.StringVar(value="")
        self.var_auto_geom = tk.BooleanVar(value=False)
        self.var_thr = tk.StringVar(value="linear")
        self.var_mode = tk.StringVar(value="standard")
        self.var_ceq = tk.StringVar(value="auto")
        self.var_combine = tk.StringVar(value="min")
        self.var_sigma_s = tk.StringVar(value="0")
        self.var_rxbw = tk.StringVar(value="%g" % REF_RX_BW_OVER_BAUD)
        self.var_custom_poly = tk.StringVar(value="")
        self.var_custom_gray = tk.BooleanVar(value=False)
        self.var_poly_info = tk.StringVar(value="")
        self.var_seed_diag = tk.BooleanVar(value=False)
        self.var_plot = tk.BooleanVar(value=HAVE_MPL)

        ttk.Label(top, text="Waveform CSV:").grid(row=0, column=0, sticky="e", **pad)
        ttk.Entry(top, textvariable=self.var_input, width=58).grid(row=0, column=1, **pad)
        ttk.Button(top, text="Browse...", command=self._pick_input).grid(row=0, column=2, **pad)

        ttk.Label(top, text="Output JSON (optional):").grid(row=1, column=0, sticky="e", **pad)
        ttk.Entry(top, textvariable=self.var_out, width=58).grid(row=1, column=1, **pad)
        ttk.Button(top, text="Browse...", command=self._pick_out).grid(row=1, column=2, **pad)

        grid = ttk.Frame(top)
        grid.grid(row=2, column=0, columnspan=3, sticky="we", **pad)
        # (label, variable, row, column, values or None for a free entry)
        spec = [
            ("Data rate (Gbps)", self.var_rate, 0, 0, None),
            ("Pattern", self.var_pattern, 0, 2, PATTERNS),
            ("Domain", self.var_domain, 0, 4, ("electrical", "optical")),
            ("Samples/UI", self.var_spu, 1, 0, None),
            ("Window (UI)", self.var_window, 1, 2, None),
            ("Standard", self.var_standard, 1, 4,
             tuple(standards.profile_names())),
            ("FFE pre", self.var_pre, 2, 0, None),
            ("FFE post", self.var_post, 2, 2, None),
            ("Threshold mode", self.var_thr, 2, 4,
             ("linear", "class_midpoint")),
            ("TDECQ mode", self.var_mode, 3, 0, ("standard", "legacy")),
            ("Ceq model", self.var_ceq, 3, 2, ("auto", "white", "bt4")),
            ("Sigma combine", self.var_combine, 3, 4, ("min", "max")),
            ("Sigma_S (mW/mV)", self.var_sigma_s, 4, 0, None),
            ("RX BW (x baud)", self.var_rxbw, 4, 2, None),
        ]
        self.std_widgets = []
        for label, var, r, c, values in spec:
            ttk.Label(grid, text=label + ":").grid(row=r, column=c, sticky="e", padx=4, pady=3)
            if values is None:
                widget = ttk.Entry(grid, textvariable=var, width=18)
            else:
                widget = ttk.Combobox(grid, textvariable=var, width=16,
                                      state="readonly", values=values)
            widget.grid(row=r, column=c + 1, sticky="w", padx=4, pady=3)
            if label in ("Ceq model", "Sigma combine", "Sigma_S (mW/mV)",
                         "RX BW (x baud)"):
                self.std_widgets.append((widget, values is not None))
            if label == "FFE pre":
                self.ent_pre = widget
            elif label == "FFE post":
                self.ent_post = widget
        self.lbl_ffe = ttk.Label(grid, text="", foreground="#505050")
        self.lbl_ffe.grid(row=4, column=4, columnspan=2, sticky="w", padx=4)
        self._ffe_saved = ("", "")
        self.var_standard.trace_add("write",
                                    lambda *a: self._on_standard_change())

        self.chk_auto_geom = ttk.Checkbutton(
            grid, text="auto geometry selection (legacy only)",
            variable=self.var_auto_geom)
        self.chk_auto_geom.grid(row=5, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Checkbutton(grid, text="seed diagnostics (to log)",
                        variable=self.var_seed_diag)\
            .grid(row=5, column=2, columnspan=2, sticky="w", padx=4)
        self.chk_plot = ttk.Checkbutton(grid, text="eye plot",
                                        variable=self.var_plot)
        self.chk_plot.grid(row=5, column=4, columnspan=2, sticky="w", padx=4)
        if not HAVE_MPL:
            self.chk_plot.state(["disabled"])
        ttk.Label(grid, text="Custom PRBS poly:").grid(row=6, column=0, sticky="e", padx=4, pady=3)
        self.ent_custom = ttk.Entry(grid, textvariable=self.var_custom_poly,
                                    width=34)
        self.ent_custom.grid(row=6, column=1, columnspan=2, sticky="w",
                             padx=4, pady=3)
        self.chk_custom_gray = ttk.Checkbutton(
            grid, text="802.3 Gray mapping", variable=self.var_custom_gray)
        self.chk_custom_gray.grid(row=6, column=3, columnspan=2, sticky="w",
                                  padx=4)
        ttk.Label(grid, text="Polynomial:").grid(row=7, column=0, sticky="e", padx=4)
        self.lbl_poly = ttk.Label(grid, textvariable=self.var_poly_info)
        self.lbl_poly.grid(row=7, column=1, columnspan=5, sticky="w", padx=4)
        self.lbl_note = ttk.Label(grid, text="", foreground="#b05000")
        self.lbl_note.grid(row=8, column=0, columnspan=6, sticky="w", padx=4)
        self.var_mode.trace_add("write", lambda *a: self._on_mode_change())
        self.var_domain.trace_add("write", lambda *a: self._on_mode_change())
        for var in (self.var_pattern, self.var_custom_poly,
                    self.var_custom_gray):
            var.trace_add("write", lambda *a: self._on_pattern_change())
        self._on_pattern_change()

        runbar = ttk.Frame(top)
        runbar.grid(row=3, column=1, columnspan=2, sticky="e", **pad)
        self.btn_run = ttk.Button(runbar, text="Run analysis",
                                  command=self._run_single)
        self.btn_run.pack(side="left", padx=2)
        self.btn_stop_single = ttk.Button(runbar, text="Stop",
                                          command=self._stop,
                                          state="disabled")
        self.btn_stop_single.pack(side="left", padx=2)

        body = ttk.Panedwindow(self.tab_single, orient="horizontal")
        body.pack(fill="both", expand=True, **pad)
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=2)

        logframe = ttk.LabelFrame(left, text="Log")
        logframe.pack(fill="both", expand=True)
        self.log = tk.Text(logframe, height=14, width=58, wrap="none",
                           state="disabled")
        self.log.pack(fill="both", expand=True)

        resframe = ttk.LabelFrame(left, text="Result summary")
        resframe.pack(fill="both", expand=True)
        self.summary = tk.Text(resframe, height=18, width=58, wrap="none",
                               state="disabled")
        self.summary.pack(fill="both", expand=True)

        plotframe = ttk.LabelFrame(right, text="Equalised eye (centred, 2 UI)")
        plotframe.pack(fill="both", expand=True)
        self._eye_name = None
        bar = ttk.Frame(plotframe)
        bar.pack(fill="x")
        self.btn_save_eye = ttk.Button(bar, text="Save eye plot...",
                                       command=self._save_eye,
                                       state="disabled")
        self.btn_save_eye.pack(side="right", padx=4, pady=2)
        if HAVE_MPL:
            self.fig = Figure(figsize=(6.4, 5.2), dpi=100)
            self.fig.patch.set_facecolor(EYE_BG)
            self.ax = self.fig.add_subplot(111)
            self._style_axes()
            self.canvas = FigureCanvasTkAgg(self.fig, master=plotframe)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(self.canvas, plotframe)
        else:
            ttk.Label(plotframe, text="matplotlib not available - plot disabled"
                      ).pack(fill="both", expand=True)

    def _build_batch_tab(self):
        pad = dict(padx=6, pady=4)
        top = ttk.LabelFrame(self.tab_batch, text="Batch queue")
        top.pack(fill="x", **pad)
        self.batch_list = tk.Listbox(top, height=8, selectmode="extended")
        self.batch_list.pack(fill="both", expand=True, side="left", padx=6, pady=6)
        btns = ttk.Frame(top)
        btns.pack(side="left", fill="y", padx=6)
        self.queue_buttons = [
            ttk.Button(btns, text="Add files...", command=self._batch_add_files),
            ttk.Button(btns, text="Add folder...", command=self._batch_add_folder),
            ttk.Button(btns, text="Remove selected", command=self._batch_remove),
            ttk.Button(btns, text="Clear", command=self._batch_clear),
        ]
        for b in self.queue_buttons:
            b.pack(fill="x", pady=2)

        opts = ttk.Frame(self.tab_batch)
        opts.pack(fill="x", **pad)
        self.var_batch_dir = tk.StringVar(value="")
        ttk.Label(opts, text="JSON output dir (optional):").pack(side="left", padx=4)
        ttk.Entry(opts, textvariable=self.var_batch_dir, width=36).pack(side="left", padx=4)
        ttk.Button(opts, text="Browse...", command=self._pick_batch_dir).pack(side="left", padx=2)
        self.btn_batch = ttk.Button(opts, text="Run batch", command=self._run_batch)
        self.btn_batch.pack(side="left", padx=(10, 2))
        self.btn_stop = ttk.Button(opts, text="Stop", command=self._stop,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=2)
        ttk.Button(opts, text="Export CSV...", command=self._export_csv).pack(side="left", padx=4)
        self.var_progress = tk.StringVar(value="")
        ttk.Label(opts, textvariable=self.var_progress).pack(side="left", padx=8)
        ttk.Label(self.tab_batch, text="Batch runs use the single-tab "
                  "settings (rate / pattern / domain / standard / mode / FFE "
                  "geometry / standard-mode options).").pack(anchor="w", padx=10)

        frame = ttk.LabelFrame(self.tab_batch, text="Batch results")
        frame.pack(fill="both", expand=True, **pad)
        cols = ("file", "metric", "credible", "ser", "ceq", "seed_ok",
                "pattern", "warn")
        self.tree = ttk.Treeview(frame, columns=cols, show="headings", height=18)
        headers = {"file": "record", "metric": "dB", "credible": "credible",
                   "ser": "SER", "ceq": "C_eq", "seed_ok": "seed ok",
                   "pattern": "pattern", "warn": "warning"}
        widths = {"file": 300, "metric": 80, "credible": 70, "ser": 80,
                  "ceq": 70, "seed_ok": 70, "pattern": 90, "warn": 260}
        for c in cols:
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=widths[c], anchor="w")
        self.tree.tag_configure("bad", foreground="#b00000")
        self.tree.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(frame, command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor(EYE_BG)
        ax.tick_params(colors=EYE_FG, which="both")
        for spine in ax.spines.values():
            spine.set_color(EYE_FG)
        ax.xaxis.label.set_color(EYE_FG)
        ax.yaxis.label.set_color(EYE_FG)
        ax.title.set_color(EYE_FG)

    def _on_mode_change(self):
        std = self.var_mode.get() == "standard"
        if std:
            self.var_auto_geom.set(False)
            self.chk_auto_geom.state(["disabled"])
        else:
            self.chk_auto_geom.state(["!disabled"])
        for widget, is_combo in self.std_widgets:
            widget.state(["!disabled", "readonly"] if std and is_combo
                         else ["!disabled"] if std else ["disabled"])
        if not std and self.var_domain.get() == "optical":
            self.lbl_note.configure(text="Note: " + LEGACY_OPTICAL_NOTE)
        else:
            self.lbl_note.configure(text="")

    def _on_standard_change(self):
        """802.3-2022 fixes the reference FFE: 5 taps, 0 pre / 4 post.

        The main tap is tap 1 (no pre-cursor, no main-position search), so
        the pre/post entries are locked and greyed out.  The draft profile
        keeps them editable.
        """
        fixed = self.var_standard.get() == standards.STANDARD_2022
        if fixed:
            if self.ent_pre.instate(["!disabled"]):
                self._ffe_saved = (self.var_pre.get(), self.var_post.get())
            self.var_pre.set("0")
            self.var_post.set("4")
            for w in (self.ent_pre, self.ent_post):
                w.state(["disabled"])
            self.lbl_ffe.configure(
                text="802.3-2022: fixed 5-tap FFE (main = tap 1)")
        else:
            if self.ent_pre.instate(["disabled"]):
                self.var_pre.set(self._ffe_saved[0])
                self.var_post.set(self._ffe_saved[1])
            for w in (self.ent_pre, self.ent_post):
                w.state(["!disabled"])
            self.lbl_ffe.configure(text="")

    def _on_pattern_change(self):
        custom = self.var_pattern.get() == CUSTOM
        for w in (self.ent_custom, self.chk_custom_gray):
            w.state(["!disabled"] if custom else ["disabled"])
        text, ok = pattern_description(self.var_pattern.get(),
                                       self.var_custom_poly.get(),
                                       self.var_custom_gray.get())
        self.var_poly_info.set(text)
        self.lbl_poly.configure(foreground="" if ok else "#b00000")

    def _pattern_arg(self):
        """Pattern name for analyze() (raises ValueError on bad input)."""
        pattern = self.var_pattern.get()
        if pattern != CUSTOM:
            return pattern
        poly = self.var_custom_poly.get().strip()
        if not poly:
            raise ValueError("Pattern 'custom': enter a PRBS polynomial, "
                             "e.g. x^13+x^4+x^3+x+1")
        return validate_pattern_request(
            ("prbsq:" if self.var_custom_gray.get() else "prbs:") + poly)

    def _pick_batch_dir(self):
        path = filedialog.askdirectory(title="JSON output folder")
        if path:
            self.var_batch_dir.set(path)

    def _save_eye(self):
        if not HAVE_MPL or self._eye_name is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save eye plot", defaultextension=".png",
            initialfile=self._eye_name + "_eye.png",
            filetypes=EYE_FORMATS)
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=200,
                             facecolor=self.fig.get_facecolor())
        except Exception as exc:
            messagebox.showerror(APP_TITLE, "Cannot save %s: %s" % (path, exc))
            return
        self._log("saved eye plot %s\n" % path)

    def _set_busy(self, busy, job=None):
        self.busy = busy
        self.job = job if busy else None
        state = "disabled" if busy else "normal"
        self.btn_run.configure(state=state)
        self.btn_batch.configure(state=state)
        for b in self.queue_buttons:
            b.configure(state=state)
        self.btn_stop.configure(
            state="normal" if busy and job == "batch" else "disabled")
        self.btn_stop_single.configure(
            state="normal" if busy and job == "single" else "disabled")

    def _start(self, job, *args):
        self._job_args = (job,) + tuple(args)
        self._got_msg = False
        self.proc, self.log_q = analysis_worker.start(
            job, *args, in_process=self.in_process)
        self._set_busy(True, job)

    def _stop(self):
        """Terminate the running analysis (single record or batch)."""
        proc, job = self.proc, self.job
        if proc is None or not self.busy:
            return
        self._drain_queue()            # keep what already arrived
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        self.proc, self.log_q = None, None
        self._set_busy(False)
        if job == "batch":
            done = len(self.batch_rows)
            self.var_progress.set("stopped after %d record(s)" % done)
            self._log("\n== batch stopped by user (%d record(s) finished)\n"
                      % done)
        else:
            self._clear_result("stopped")
            self._log("\n== analysis stopped by user\n")

    def _clear_result(self, message=""):
        """Blank the summary and the eye (no stale plot during a new run)."""
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.configure(state="disabled")
        self._eye_name = None
        self.btn_save_eye.configure(state="disabled")
        if HAVE_MPL:
            self.ax.clear()
            self._style_axes()
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            if message:
                self.ax.text(0.5, 0.5, message, transform=self.ax.transAxes,
                             ha="center", va="center", color=EYE_FG,
                             fontsize=11)
            self.canvas.draw()

    # ------------------------------------------------------------ helpers
    def _pick_input(self):
        path = filedialog.askopenfilename(
            title="Select waveform CSV",
            filetypes=[("Waveform CSV", "*.csv"), ("All files", "*.*")])
        if path:
            self.var_input.set(path)

    def _pick_out(self):
        path = filedialog.asksaveasfilename(
            title="Output JSON", defaultextension=".json",
            filetypes=[("JSON", "*.json")])
        if path:
            self.var_out.set(path)

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _log_clear(self):
        """Clear the log area (called at the start of every new run)."""
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _drain_queue(self):
        q = self.log_q
        while q is not None and q is self.log_q:
            try:
                kind, payload = q.get_nowait()
            except (queue.Empty, EOFError, OSError):
                break
            self._got_msg = True
            try:
                self._handle(kind, payload)
            except Exception as exc:   # never kill the polling loop
                self._log("\nGUI error (%s): %r\n" % (kind, exc))

    def _drain_log(self):
        try:
            self._drain_queue()
            proc = self.proc
            if self.busy and proc is not None and not proc.is_alive():
                # the worker exited: collect its last messages, then make
                # sure a crashed worker never leaves the GUI busy
                self._drain_queue()
                if self.busy and self.proc is proc:
                    self._set_busy(False)
                    if not self._got_msg and not self.in_process:
                        # the worker died before reporting anything (e.g.
                        # an OpenMP runtime aborting the new process):
                        # rerun this job in-process from now on
                        self.in_process = True
                        self._log("\nNOTE: the analysis process could not "
                                  "start (exit code %s); running analyses "
                                  "inside the GUI process from now on (Stop "
                                  "then only discards the running analysis)"
                                  "\n" % proc.exitcode)
                        self._start(*self._job_args)
                    else:
                        self._log("\nERROR: analysis process exited "
                                  "unexpectedly (code %s)\n" % proc.exitcode)
        finally:
            self.root.after(120, self._drain_log)

    def _handle(self, kind, payload):
        if kind == "log":
            self._log(payload)
        elif kind == "clear":
            self._log_clear()
        elif kind == "single-done":
            self._finish()
            self._show_result(*payload)
        elif kind == "batch-row":
            self._add_batch_row(payload)
        elif kind == "batch-progress":
            self.var_progress.set(payload)
        elif kind == "batch-done":
            self._finish()
            n, total, stopped = payload
            msg = "%d/%d records%s" % (n, total, " (stopped)" if stopped else "")
            self.var_progress.set("done: " + msg)
            self._log("\n== batch finished: %s\n" % msg)
        elif kind == "error":
            self._finish()
            self._clear_result("analysis failed - see the log")
            self._log("\nERROR: %s\n" % payload)

    def _finish(self):
        """The worker reported its final message."""
        self.proc = None
        self._set_busy(False)

    def _collect_args(self, verbose=False):
        """Validated analyze() keyword arguments (raises ValueError)."""
        def number(var, name, kind=float, minimum=None, allow_auto=False,
                   allow_empty=False):
            text = var.get().strip()
            if allow_auto and text.lower() in ("", "auto"):
                return "auto"
            if allow_empty and not text:
                return None
            try:
                val = kind(text)
            except ValueError:
                raise ValueError("%s: '%s' is not a valid %s" % (
                    name, text, "integer" if kind is int else "number"))
            if minimum is not None and val < minimum:
                raise ValueError("%s must be >= %g" % (name, minimum))
            return val

        rate = number(self.var_rate, "Data rate", minimum=1e-6)
        spu = number(self.var_spu, "Samples/UI", int, 2, allow_auto=True)
        window = number(self.var_window, "Window (UI)", int, 1,
                        allow_auto=True)
        if self.var_standard.get() == standards.STANDARD_2022:
            pre = post = None          # fixed reference FFE (see above)
        else:
            pre = number(self.var_pre, "FFE pre", int, 0, allow_empty=True)
            post = number(self.var_post, "FFE post", int, 0,
                          allow_empty=True)
        mode = self.var_mode.get()
        kwargs = dict(
            pattern=self._pattern_arg(),
            rate_gbps=rate,
            domain=self.var_domain.get(),
            samples_per_ui=spu if spu == "auto" else str(spu),
            window=window if window == "auto" else str(window),
            standard=self.var_standard.get(),
            threshold_mode=self.var_thr.get(),
            ffe_auto_geometry=(mode == "legacy"
                               and bool(self.var_auto_geom.get())),
            tdecq_mode=mode,
            seed_diagnostics=bool(self.var_seed_diag.get()) or None,
            verbose=verbose,
        )
        if mode == "standard":
            sigma_s = number(self.var_sigma_s, "Sigma_S", minimum=0.0,
                             allow_empty=True)
            rxbw = number(self.var_rxbw, "RX BW", minimum=1e-3,
                          allow_empty=True)
            kwargs.update(
                sigma_s=sigma_s or 0.0,
                sigma_combine=self.var_combine.get(),
                ceq_model=(None if self.var_ceq.get() == "auto"
                           else self.var_ceq.get()),
                rx_bw_over_baud=(REF_RX_BW_OVER_BAUD if rxbw is None
                                 else rxbw))
        if pre is not None:
            kwargs["ffe_pre_cursor"] = pre
        if post is not None:
            kwargs["ffe_post_cursor"] = post
        kwargs["ffe_geometry_explicit"] = pre is not None or post is not None
        return kwargs

    # ----------------------------------------------------------- actions
    def _run_single(self):
        if self.busy:
            return
        path = self.var_input.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror(APP_TITLE, "Select an existing waveform CSV first.")
            return
        # tkinter variables must be read on the main thread: collect and
        # validate every setting here and hand frozen kwargs to the worker.
        try:
            kwargs = self._collect_args(verbose=True)
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        out = self.var_out.get().strip()   # optional: empty -> no JSON written
        want_plot = bool(self.var_plot.get()) and HAVE_MPL
        self._log_clear()
        self._log("== analysing %s\n" % os.path.basename(path))
        self._clear_result("analysing %s ..." % os.path.basename(path))
        self._start("single", path, out, kwargs, want_plot)

    def _show_result(self, path, result, plot_data, kwargs):
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("end", "\n".join(summary_lines(result)))
        self.summary.configure(state="disabled")
        if plot_data is not None and HAVE_MPL:
            if self._draw_eye(result, plot_data, kwargs):
                self._eye_name = os.path.splitext(os.path.basename(path))[0]
                self.btn_save_eye.configure(state="normal")

    def _draw_eye(self, result, data, kwargs):
        """Equalised-eye panel in the diagnostic-plot style, over 2 UI.

        Dark power-normalised density map, the three PAM4 decision
        thresholds and the two histogram windows.  For a standard-mode
        result the thresholds and windows are the measured ones (see
        :func:`tdecq_tool.plot.eye_overlay`).
        """
        try:
            M, tru, cdr, spu = data["M"], data["tru"], data["cdr"], data["spu"]
            domain = kwargs["domain"]
            unit_label = ("Voltage (mV)" if domain == "electrical"
                          else "Power (mW)")
            # PAM4: UI = 2 / data rate
            ui_ps = 2.0e3 / float(kwargs["rate_gbps"])
            n_periods = 2

            self.ax.clear()
            self._style_axes()
            eye = tplot._eye_phase(M, cdr, spu)
            _, p_edges, counts = tplot._phase_density(eye, n_y_bins=200)
            # tile the folded density (sampling phase at 0.5 UI) over 2 UI
            # and shift it right by half a UI: half an eye, the full eye
            # centred at 1 UI with both sampling windows, half an eye
            shift = spu - spu // 2
            counts2 = np.roll(np.vstack([counts] * n_periods), shift, axis=0)
            shift_ui = shift / float(spu)
            t_edges = np.linspace(0.0, n_periods * ui_ps, counts2.shape[0] + 1)
            self.ax.pcolormesh(t_edges, p_edges, counts2.T,
                               cmap=tplot._cmap_for("blue"),
                               norm=mcolors.PowerNorm(
                                   gamma=0.2, vmin=0.0,
                                   vmax=max(float(counts.max()), 1e-12)),
                               shading="auto", rasterized=True)
            ov = tplot.eye_overlay(M, tru, cdr, spu, result)
            for p, label in zip(ov["pth"], tplot.THRESHOLD_LABELS):
                self.ax.axhline(p, color="white", lw=0.8, ls="--", alpha=0.7)
                self.ax.text(0.01, p, label,
                             transform=self.ax.get_yaxis_transform(),
                             va="bottom", ha="left", fontsize=8,
                             color="white")
            for x0, x1, label in ov["windows"]:
                # labels give the window position within the UI (sampling
                # phase = 0.5 UI)
                self.ax.axvspan((x0 + shift_ui) * ui_ps,
                                (x1 + shift_ui) * ui_ps,
                                color="white", alpha=0.15)
                self.ax.text((x0 + x1) / 2 * ui_ps + shift_ui * ui_ps, 0.97,
                             label, transform=self.ax.get_xaxis_transform(),
                             va="top", ha="center", fontsize=8,
                             color="white", rotation=90)
            for k in (0.5, 1.5):
                # UI boundaries (eye crossings) of the centred eye
                self.ax.axvline(k * ui_ps, color=EYE_FG, lw=0.6, ls=":",
                                alpha=0.5)
            self.ax.set_ylim(*tplot.eye_ylim(eye.ravel()))
            self.ax.set_xlim(0.0, n_periods * ui_ps)
            self.ax.set_xlabel("Time (ps)")
            self.ax.set_ylabel(unit_label)
            key = _metric_key(result)
            metric = result.get(key)
            ser = result.get("actual_SER")
            ser_txt = ("SER=%.2e" % ser
                       if ser is not None and np.isfinite(float(ser))
                       else "SER=n/a")
            flag = ("" if result.get("metric_credible", True)
                    else "  [not credible]")
            self.ax.set_title("%s = %.3f dB, %s  (%s)%s" % (
                "TECQ" if key == "tecq_db" else "TDECQ",
                metric if metric is not None else float("nan"), ser_txt,
                result.get("tdecq_mode", "legacy"), flag),
                fontsize=9, color=EYE_FG)
            try:
                self.fig.tight_layout()
            except Exception:
                pass
            self.canvas.draw()
            return True
        except Exception as exc:
            self._log("plot failed: %r\n" % (exc,))
            return False

    # ------------------------------------------------------------- batch
    def _batch_add_files(self):
        paths = filedialog.askopenfilenames(
            title="Add waveform CSVs",
            filetypes=[("Waveform CSV", "*.csv"), ("All files", "*.*")])
        for p in paths:
            if p not in self.batch_files:
                self.batch_files.append(p)
                self.batch_list.insert("end", p)

    def _batch_add_folder(self):
        folder = filedialog.askdirectory(title="Add all CSVs from folder")
        if not folder:
            return
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(".csv"):
                p = os.path.join(folder, name)
                if p not in self.batch_files:
                    self.batch_files.append(p)
                    self.batch_list.insert("end", p)

    def _batch_remove(self):
        for idx in reversed(self.batch_list.curselection()):
            self.batch_list.delete(idx)
            del self.batch_files[idx]

    def _batch_clear(self):
        self.batch_files.clear()
        self.batch_list.delete(0, "end")
        self.tree.delete(*self.tree.get_children())
        self.batch_rows.clear()
        self.var_progress.set("")

    def _run_batch(self):
        if self.busy:
            return
        if not self.batch_files:
            messagebox.showerror(APP_TITLE, "The batch queue is empty.")
            return
        try:
            kwargs = self._collect_args(verbose=True)
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        outdir = self.var_batch_dir.get().strip()   # optional: empty -> no JSON
        if outdir:
            try:
                os.makedirs(outdir, exist_ok=True)
            except OSError as exc:
                messagebox.showerror(APP_TITLE, "Cannot create %s: %s"
                                     % (outdir, exc))
                return
        self.tree.delete(*self.tree.get_children())
        self.batch_rows.clear()
        self._log_clear()
        self._start("batch", list(self.batch_files), outdir, kwargs)

    def _add_batch_row(self, row):
        self.batch_rows.append(row)
        vals = (row["file"],
                _fmt(row["metric"], "%.4f"),
                _fmt(row["credible"]),
                _fmt(row["ser"], "%.3g"),
                _fmt(row["ceq"], "%.3f"),
                _fmt(row["seed_ok"]),
                row["pattern"] or "-",
                row["warn_text"] or "")
        bad = row["error"] or row["credible"] is False
        self.tree.insert("", "end", values=vals, tags=("bad",) if bad else ())

    def _export_csv(self):
        if not self.batch_rows:
            messagebox.showerror(APP_TITLE, "Run a batch first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export batch results", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        cols = ["file", "mode", "metric", "credible", "ser", "ceq", "seed_ok",
                "seed", "seed_a", "seed_b", "pattern", "taps", "strategy",
                "warn", "warn_text", "json", "error", "path"]
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for row in self.batch_rows:
                w.writerow(["" if row.get(c) is None else row.get(c)
                            for c in cols])
        self._log("exported %s\n" % path)


def main():
    root = tk.Tk()
    TdecqGui(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
