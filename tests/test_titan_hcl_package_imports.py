"""Guard test for the titan_plugin → titan_hcl package rename (D-SPEC-110).

Two layers:
  1. compileall — every .py under titan_hcl/ parses (catches any sed-induced
     syntax breakage across the 857-file rename).
  2. import-smoke — the load-bearing entry modules + every worker main + the
     api surface actually import (catches broken import paths / stale
     titan_plugin references).

Run: python -m pytest tests/test_titan_hcl_package_imports.py -v -p no:anchorpy
"""
from __future__ import annotations

import compileall
import importlib
import os

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG_DIR = os.path.join(_REPO_ROOT, "titan_hcl")


def test_titan_hcl_package_exists_not_titan_plugin():
    """The package directory is titan_hcl (the legacy titan_plugin is gone)."""
    assert os.path.isdir(_PKG_DIR), "titan_hcl/ package dir missing"
    assert not os.path.isdir(os.path.join(_REPO_ROOT, "titan_plugin")), \
        "legacy titan_plugin/ dir still present — rename incomplete"
    assert os.path.isfile(os.path.join(_REPO_ROOT, "scripts", "titan_hcl.py")), \
        "scripts/titan_hcl.py entry missing"
    assert not os.path.isfile(os.path.join(_REPO_ROOT, "scripts", "titan_main.py")), \
        "legacy scripts/titan_main.py still present"


def test_titan_hcl_compiles():
    """Every .py under titan_hcl/ compiles (syntax-clean after the rename)."""
    ok = compileall.compile_dir(_PKG_DIR, quiet=1, force=True)
    assert ok, "compileall found syntax errors under titan_hcl/"


# Load-bearing modules that MUST import cleanly (entry + kernel + bus + every
# worker main + the api surface). Heavy ML deps are fine — these are imported
# one-by-one in this single process; the list excludes nothing critical.
_CRITICAL_MODULES = [
    "titan_hcl",
    "titan_hcl.bus",
    "titan_hcl.guardian",
    "titan_hcl.core.plugin",
    "titan_hcl.core.kernel",
    "titan_hcl.api.maker",
    "titan_hcl.api.shm_reader_bank",
    "titan_hcl.api.state_accessor",
    "titan_hcl.modules.observatory_worker",
    "titan_hcl.modules.agno_worker",
    "titan_hcl.modules.dream_state_worker",
    "titan_hcl.modules.cognitive_worker",
    "titan_hcl.modules.recorder_worker",
    "titan_hcl.modules.meditation_worker",
]


@pytest.mark.parametrize("modname", _CRITICAL_MODULES)
def test_critical_module_imports(modname):
    """Each load-bearing module imports without ImportError."""
    importlib.import_module(modname)


def test_orchestrator_class_renamed():
    """The orchestrator class is TitanHCL (not the legacy TitanPlugin)."""
    plugin_mod = importlib.import_module("titan_hcl.core.plugin")
    assert hasattr(plugin_mod, "TitanHCL"), "TitanHCL class missing"
    assert not hasattr(plugin_mod, "TitanPlugin"), \
        "legacy TitanPlugin name still exported"
