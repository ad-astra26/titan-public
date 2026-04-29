"""Tests for rFP_backup_worker Phase 8.2 — arch_map backup --verify-chain CLI.

Covers:
  1. run_backup_verify_chain: empty chain → 0
  2. run_backup_verify_chain: intact chain → 0
  3. run_backup_verify_chain: broken chain → 1
  4. Normal run_backup_diagnostics includes chain-integrity line
"""
import json
import os
import subprocess
import sys
from unittest import mock

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _write_chain_file(titan_id: str, anchors: list, cwd=None):
    """Write a chain file in the cwd's data/ directory."""
    cwd = cwd or os.getcwd()
    os.makedirs(os.path.join(cwd, "data"), exist_ok=True)
    p = os.path.join(cwd, "data", f"backup_anchor_chain_{titan_id}.json")
    with open(p, "w") as f:
        json.dump({"version": 1, "titan_id": titan_id, "anchors": anchors}, f)
    return p


# ────────────────────────────────────────────────────────────────────────────
# Direct function tests
# ────────────────────────────────────────────────────────────────────────────

def test_verify_chain_empty_returns_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # Import fresh from parent dir
    sys.path.insert(0, REPO_ROOT)
    from scripts.arch_map import run_backup_verify_chain
    rc = run_backup_verify_chain(all_titans=False)
    assert rc == 0
    out = capsys.readouterr().out
    assert "chain file absent" in out


def test_verify_chain_intact_returns_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    anchors = [
        {"backup_id": 0, "archive_hash": "a" * 64, "prev_anchor_hash": "",
         "tx": "sig1", "ts": 1, "backup_type": "personality", "size_mb": 25.0},
        {"backup_id": 1, "archive_hash": "b" * 64, "prev_anchor_hash": "a" * 64,
         "tx": "sig2", "ts": 2, "backup_type": "personality", "size_mb": 25.1},
    ]
    _write_chain_file("T1", anchors, cwd=str(tmp_path))
    sys.path.insert(0, REPO_ROOT)
    from scripts.arch_map import run_backup_verify_chain
    rc = run_backup_verify_chain(all_titans=False)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Chain INTACT" in out
    assert "2 entries" in out


def test_verify_chain_broken_returns_one(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    anchors = [
        {"backup_id": 0, "archive_hash": "a" * 64, "prev_anchor_hash": "",
         "tx": "sig1", "ts": 1, "backup_type": "personality", "size_mb": 25.0},
        {"backup_id": 1, "archive_hash": "b" * 64, "prev_anchor_hash": "X" * 64,
         "tx": "sig2", "ts": 2, "backup_type": "personality", "size_mb": 25.0},
    ]
    _write_chain_file("T1", anchors, cwd=str(tmp_path))
    sys.path.insert(0, REPO_ROOT)
    from scripts.arch_map import run_backup_verify_chain
    rc = run_backup_verify_chain(all_titans=False)
    assert rc == 1
    out = capsys.readouterr().out
    assert "Chain BROKEN" in out
    assert "index 1" in out


# ────────────────────────────────────────────────────────────────────────────
# Full CLI subprocess test
# ────────────────────────────────────────────────────────────────────────────

def test_cli_subprocess_intact_exit_0(tmp_path):
    anchors = [
        {"backup_id": 0, "archive_hash": "a" * 64, "prev_anchor_hash": "",
         "tx": "sig1", "ts": 1, "backup_type": "personality", "size_mb": 25.0},
    ]
    _write_chain_file("T1", anchors, cwd=str(tmp_path))
    # Use sys.executable (the python running pytest) rather than a hardcoded
    # test_env path. Worktrees don't have their own test_env, but they do
    # inherit the venv python via sys.executable when pytest is launched
    # from the activated venv.
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts", "arch_map.py"),
         "backup", "--verify-chain"],
        cwd=str(tmp_path),
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    assert "Chain INTACT" in proc.stdout


def test_cli_subprocess_broken_exit_1(tmp_path):
    anchors = [
        {"backup_id": 0, "archive_hash": "a" * 64, "prev_anchor_hash": "",
         "tx": "sig1", "ts": 1, "backup_type": "personality", "size_mb": 25.0},
        {"backup_id": 1, "archive_hash": "b" * 64, "prev_anchor_hash": "X" * 64,
         "tx": "sig2", "ts": 2, "backup_type": "personality", "size_mb": 25.0},
    ]
    _write_chain_file("T1", anchors, cwd=str(tmp_path))
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts", "arch_map.py"),
         "backup", "--verify-chain"],
        cwd=str(tmp_path),
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    assert "Chain BROKEN" in proc.stdout
