#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch the TDECQ/TECQ tkinter GUI (see gui/tdecq_gui.py).

Run from anywhere with ``python run_gui.py``.  The interpreter that started
this script is checked first (Python and numpy / scipy / matplotlib
versions, tkinter); when it does not qualify - e.g. an EDA tool's bundled
Python with old MKL builds - and the project virtual environment ``.venv``
exists, the GUI is relaunched with the ``.venv`` interpreter automatically.

This file deliberately keeps to Python 3.6 syntax so it can report a clear
message on old system interpreters instead of failing on import.
"""
import os
import sys

# Intel OpenMP (MKL builds of numpy/scipy, e.g. an EDA tool's bundled
# Python) registers fork handlers that can abort a forked child with
# "OMP: Error #13: Assertion failure at z_Linux_util.cpp".  The GUI starts
# its analysis worker (multiprocessing 'spawn' = fork + exec) after numpy has
# initialised OpenMP, so disable those handlers before numpy is imported.
# The child execs a fresh interpreter immediately, so it needs none of them.
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

ROOT = os.path.dirname(os.path.abspath(__file__))
# hard minimum (the code uses Python 3.7 syntax); the versions below are the
# tested ones (requirements.txt) - older libraries only produce a warning
MIN_PYTHON = (3, 7)
# minimum versions (requirements.txt)
MIN_PACKAGES = (("numpy", (1, 24)), ("scipy", (1, 10)),
                ("matplotlib", (3, 5)))


def _version_tuple(text):
    out = []
    for part in str(text).split("."):
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def _package_version(name):
    """Installed version string of ``name`` without importing it, or None."""
    try:
        from importlib import metadata          # Python 3.8+
        return metadata.version(name)
    except Exception:
        pass
    try:
        import pkg_resources
        return pkg_resources.get_distribution(name).version
    except Exception:
        return None


def _check():
    """``(problems, old)``: blocking problems and below-tested versions."""
    problems, old = [], []
    if sys.version_info[:2] < MIN_PYTHON:
        problems.append("Python %d.%d (need >= %d.%d)"
                        % (tuple(sys.version_info[:2]) + MIN_PYTHON))
        return problems, old  # the packages would not run on it anyway
    for name, minimum in MIN_PACKAGES:
        ver = _package_version(name)
        if ver is None:
            problems.append("%s not installed" % name)
        elif _version_tuple(ver) < minimum:
            old.append("%s %s (tested with >= %s)"
                       % (name, ver, ".".join(str(x) for x in minimum)))
    try:
        import tkinter  # noqa: F401
    except Exception:
        problems.append("tkinter not available")
    return problems, old


def _venv_python():
    for rel in (("bin", "python3"), ("bin", "python"),
                ("Scripts", "python.exe")):
        cand = os.path.join(ROOT, ".venv", *rel)
        if os.path.isfile(cand):
            return cand
    return None


def main():
    problems, old = _check()
    venv = _venv_python()
    if (problems or old) and venv and \
            os.path.realpath(venv) != os.path.realpath(sys.executable) and \
            not os.environ.get("TDECQ_GUI_RELAUNCHED"):
        # prefer the project environment over a missing / old interpreter
        os.environ["TDECQ_GUI_RELAUNCHED"] = "1"
        sys.stderr.write("run_gui: %s: %s; relaunching with %s\n"
                         % (sys.executable, "; ".join(problems + old), venv))
        sys.stderr.flush()
        os.execv(venv, [venv, os.path.abspath(__file__)] + sys.argv[1:])
    if old and not problems:
        sys.stderr.write("run_gui: warning: older libraries than tested: "
                         "%s\n" % "; ".join(old))
    if problems:
        sys.stderr.write(
            "run_gui: cannot start the GUI with %s:\n  - %s\n"
            "The tool needs Python >= %d.%d with numpy, scipy, matplotlib "
            "and tkinter (tested: numpy >= 1.24, scipy >= 1.10, "
            "matplotlib >= 3.5).\n"
            "Create the project environment %s with such a Python, e.g.\n"
            "  <python3.10+> -m venv %s\n"
            "  %s -m pip install -r %s\n"
            "and run this script again (it switches to .venv by itself).\n"
            % (sys.executable, "\n  - ".join(problems), MIN_PYTHON[0],
               MIN_PYTHON[1], os.path.join(ROOT, ".venv"),
               os.path.join(ROOT, ".venv"),
               os.path.join(ROOT, ".venv", "bin", "python"),
               os.path.join(ROOT, "requirements.txt")))
        return 1
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from gui.tdecq_gui import main as gui_main
    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
