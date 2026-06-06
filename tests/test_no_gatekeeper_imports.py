"""Retirement guard — the gatekeeper + offline-RL/torch subsystem is GONE
(RFP_synthesis_decision_authority P1, INV-SDA-6).

Asserts (a) the 6 retired modules no longer import, and (b) no production code
re-imports the retired classes/symbols — so the IQL/torch subsystem can't creep
back. Execution-mode routing is the grounded router; sovereignty is the ONE
S = 0.7·E + 0.3·V.
"""
import importlib
import os
import subprocess

import pytest

# The 6 deleted modules (§1.2 RETIRED block).
RETIRED_MODULES = [
    "titan_hcl.logic.sage.gatekeeper",
    "titan_hcl.logic.sage.scholar",
    "titan_hcl.core.sage.recorder",
    "titan_hcl.modules.recorder_worker",
    "titan_hcl.proxies.rl_proxy",
    "titan_hcl.logic.rl_state_publisher",
]


@pytest.mark.parametrize("mod", RETIRED_MODULES)
def test_retired_module_is_gone(mod):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(mod)


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_production_import_of_retired_symbols():
    """grep the production tree (titan_hcl/, excluding comments) for live imports
    of the retired classes/entry-points. Comment mentions of the retirement are
    fine; an actual `import` is not."""
    root = os.path.join(_repo_root(), "titan_hcl")
    # Patterns that would only appear in a real import/use, not a prose comment.
    patterns = [
        "import SageGatekeeper",
        "import SageScholar",
        "import SageRecorder",
        "import RLProxy",
        "import recorder_worker_main",
        "from titan_hcl.proxies.rl_proxy",
        "from titan_hcl.core.sage",
        "from titan_hcl.logic.rl_state_publisher",
        "from titan_hcl.modules.recorder_worker",
    ]
    offenders = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    stripped = line.lstrip()
                    if stripped.startswith("#"):
                        continue
                    for pat in patterns:
                        if pat in line:
                            offenders.append(f"{path}:{i}: {line.strip()}")
    assert not offenders, (
        "Live imports of the retired offline-RL subsystem found:\n"
        + "\n".join(offenders))


def test_no_titan_recorder_module_spec():
    """The recorder ModuleSpec (the ~3000MB torch worker) is not registered."""
    src = open(
        os.path.join(_repo_root(), "titan_hcl", "module_catalog.py"),
        encoding="utf-8").read()
    assert 'name="recorder"' not in src, \
        "the recorder ModuleSpec must be gone (no torch worker, ~366 MB/Titan reclaimed)"
