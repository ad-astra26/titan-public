"""Phase 10L (rFP §3G) — CI regression guard against pre-Phase-C remnants.

Scans the ``titan_hcl/`` source tree (via AST, so docstring/comment examples are
ignored — only real import statements count) for patterns that Phase 10
(spirit_loop.py retirement) eliminated, so they can't silently regress:

  1. No module imports ``titan_hcl.modules.spirit_loop`` — the orphan helper
     module was deleted (10I); its live logic moved to logic/ homes.
  2. No module imports a reflex-intuition helper directly from a worker body
     (``from titan_hcl.modules.*_worker import _compute_*_reflex_intuition``) —
     they live in logic/{body,mind,spirit}_helpers.py and are consumed via
     logic/reflex_intuition (SPEC §11.B.4).
  3. ``agno_hooks`` contains ZERO ``from titan_hcl.modules.*_worker`` imports
     (the Phase 11 §11.I orchestrator-isolation invariant; completed by 10J).
  4. No module imports the pre-rename ``titan_plugin`` package (renamed to
     ``titan_hcl`` — D-SPEC-110).

NOTE: this does NOT assert the absence of ``_HIGH_RATE_BROADCAST_TYPES`` (the
original 10L draft listed it, but it is LIVE load-bearing broadcast backpressure
in bus.py — kept; the RFP's "retired" note is stale).
"""
import ast
import os
import re

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.join(_HERE, "titan_hcl")

_WORKER_RE = re.compile(r"^titan_hcl\.modules\.\w*_worker$")
_REFLEX_RE = re.compile(r"^_compute_\w+_reflex_intuition$")


def _py_files():
    for root, _dirs, files in os.walk(_PKG):
        if "__pycache__" in root:
            continue
        for fn in files:
            if fn.endswith(".py"):
                yield os.path.join(root, fn)


def _imports(path):
    """Yield (lineno, module, name) for every real import node.

    For ``import X`` → (lineno, X, None). For ``from M import N`` →
    (lineno, M, N) per imported name. AST-based, so docstrings/comments that
    merely contain import-like text are ignored.
    """
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, None
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                yield node.lineno, mod, alias.name


def _rel(path):
    return os.path.relpath(path, _HERE)


def test_no_spirit_loop_imports():
    """spirit_loop.py is deleted (10I) — nothing may import it."""
    offenders = []
    for path in _py_files():
        for ln, mod, _name in _imports(path):
            if mod.startswith("titan_hcl.modules.spirit_loop"):
                offenders.append(f"{_rel(path)}:{ln}: from {mod} import ...")
    assert not offenders, (
        "spirit_loop.py was retired (Phase 10I) — remove these imports:\n"
        + "\n".join(offenders)
    )


def test_no_reflex_helper_imports_from_worker_bodies():
    """Reflex-intuition helpers live in logic/, not worker bodies (10C/10J)."""
    offenders = []
    for path in _py_files():
        for ln, mod, name in _imports(path):
            if name and _WORKER_RE.match(mod) and _REFLEX_RE.match(name):
                offenders.append(f"{_rel(path)}:{ln}: from {mod} import {name}")
    assert not offenders, (
        "reflex intuition must be imported from logic/ (logic.reflex_intuition / "
        "logic.{body,mind,spirit}_helpers), not worker bodies:\n" + "\n".join(offenders)
    )


def test_agno_hooks_has_no_worker_body_imports():
    """SPEC §11.B.4 — agno_hooks must not reach into any *_worker module body."""
    agno = os.path.join(_PKG, "modules", "agno_hooks.py")
    offenders = [
        f"{ln}: from {mod} import {name}"
        for ln, mod, name in _imports(agno)
        if _WORKER_RE.match(mod)
    ]
    assert not offenders, (
        "agno_hooks.py must contain ZERO 'from titan_hcl.modules.*_worker import' "
        "(Phase 11 orchestrator isolation):\n" + "\n".join(offenders)
    )


def test_no_titan_plugin_package_imports():
    """The package was renamed titan_plugin → titan_hcl (D-SPEC-110)."""
    offenders = []
    for path in _py_files():
        for ln, mod, _name in _imports(path):
            if mod == "titan_plugin" or mod.startswith("titan_plugin."):
                offenders.append(f"{_rel(path)}:{ln}: import {mod}")
    assert not offenders, (
        "the titan_plugin package was renamed to titan_hcl (D-SPEC-110):\n"
        + "\n".join(offenders)
    )
