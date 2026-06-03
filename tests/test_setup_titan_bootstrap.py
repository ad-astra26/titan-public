"""Phase C tests — the curl|bash bootstrap scripts/setup_titan.sh (INV-PROV-4).

Syntax-checks the script and exercises its TITAN_BOOTSTRAP_DRYRUN path (echoes
every step, touches nothing, stops before the wizard) to assert the contract:
base deps, idempotent clone, MINIMAL venv (textual only — no full -e ., no CUDA),
and the exec handoff with forwarded args. No chromium / docker in the bootstrap.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "setup_titan.sh"


def test_script_exists_and_is_executable():
    assert SCRIPT.exists(), "scripts/setup_titan.sh missing"
    assert os.access(SCRIPT, os.X_OK), "scripts/setup_titan.sh must be chmod +x"


def test_bash_syntax_clean():
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.fixture
def dry_run(tmp_path):
    if not all(__import__("shutil").which(c) for c in ("bash", "curl", "sudo")):
        pytest.skip("bootstrap dry-run needs bash + curl + sudo present")
    env = dict(os.environ)
    env.update(TITAN_BOOTSTRAP_DRYRUN="1", TITAN_ID="T4", HOME=str(tmp_path),
               REPO_URL="https://example.com/titan-public.git")
    r = subprocess.run(["bash", str(SCRIPT), "--mode", "devnet"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    return r.stdout


def test_installs_base_deps(dry_run):
    for dep in ("git", "build-essential", "python3-venv", "xdelta3", "nftables"):
        assert dep in dry_run, f"base dep {dep} not provisioned"


def test_clones_id_scoped_repo(dry_run):
    assert "titan-T4" in dry_run                       # TITAN_ID → ~/titan-T4
    assert "example.com/titan-public.git" in dry_run   # REPO_URL honored


def test_minimal_venv_textual_only_no_heavy_install(dry_run):
    assert "test_env" in dry_run and "textual" in dry_run
    # the bootstrap must NOT do the heavy editable install (that's the CPU-index
    # venv phase) — and must never touch chromium/docker/CUDA.
    assert "pip install -e ." not in dry_run
    for forbidden in ("chromium", "playwright install", "docker", "cuda"):
        assert forbidden.lower() not in dry_run.lower(), f"bootstrap must not reference {forbidden}"


def test_exec_handoff_forwards_args(dry_run):
    assert "scripts.setup_titan install --mode devnet" in dry_run
