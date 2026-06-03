"""Phase C guard — the canonical root bootstrap `setup_titan.sh` (W1.h, hardened).

Static checks only: the script apt-installs + clones + execs the wizard, so it
isn't safely runnable in CI. These guard the PUBLIC one-liner against syntax
errors + the base-deps / handoff contract, and assert the Phase-C duplicate is
gone (one canonical bootstrap, at repo root, where filter_lib.sh ships it).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "setup_titan.sh"          # canonical location: repo ROOT (curl|bash)


def test_exists_and_executable():
    assert SCRIPT.exists(), "root setup_titan.sh missing"
    assert os.access(SCRIPT, os.X_OK), "setup_titan.sh must be chmod +x"


def test_bash_syntax_clean():
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_no_duplicate_in_scripts_dir():
    # the Phase-C duplicate must be retired — exactly one canonical bootstrap
    assert not (REPO / "scripts" / "setup_titan.sh").exists()


def test_installs_provisioner_base_deps():
    body = SCRIPT.read_text()
    for dep in ("git", "python3-venv", "build-essential",
                "pkg-config", "libssl-dev", "xdelta3", "nftables"):
        assert dep in body, f"bootstrap apt line missing {dep}"


def test_robust_tty_open_and_wizard_handoff():
    body = SCRIPT.read_text()
    # real open-test (ENXIO-safe under nohup/CI), not just a `[ -e /dev/tty ]` probe
    assert "{ : < /dev/tty; }" in body
    assert "python3 -m scripts.setup_titan install" in body
