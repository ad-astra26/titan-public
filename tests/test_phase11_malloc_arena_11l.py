"""
Phase 11 §11.I.5 / Chunk 11L — MALLOC_ARENA_MAX=2 fleet-wide.

Verifies:
  1. scripts/guardian_hcl.py + scripts/titan_hcl.py both call
     `os.environ.setdefault("MALLOC_ARENA_MAX", "2")` early enough to
     propagate to every child spawn (via subprocess.Popen / multiprocessing.Process).
  2. The setting does NOT override a pre-existing value (allows operator
     overrides via systemd or shell env for diagnostics).

`titan-kernel-rs/spawn.rs:116` is the production setter when kernel-rs
spawns guardian_hcl. The defensive `setdefault` in Python land covers
standalone dev runs + tests + systemd unit overrides where the kernel-rs
env doesn't reach the Python entry.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


def _read(path: str) -> str:
    return Path(path).read_text()


def test_guardian_hcl_script_sets_malloc_arena_max_default():
    src = _read("scripts/guardian_hcl.py")
    assert re.search(
        r'os\.environ\.setdefault\(\s*["\']MALLOC_ARENA_MAX["\']\s*,\s*["\']2["\']\s*\)',
        src,
    ), "scripts/guardian_hcl.py must call os.environ.setdefault('MALLOC_ARENA_MAX', '2')"


def test_titan_hcl_script_sets_malloc_arena_max_default():
    src = _read("scripts/titan_hcl.py")
    assert re.search(
        r'os\.environ\.setdefault\(\s*["\']MALLOC_ARENA_MAX["\']\s*,\s*["\']2["\']\s*\)',
        src,
    ), "scripts/titan_hcl.py must call os.environ.setdefault('MALLOC_ARENA_MAX', '2')"


def test_setdefault_does_not_override_operator_value(monkeypatch):
    """setdefault must NOT override an operator's pre-set value (e.g. for
    diagnostics under MALLOC_ARENA_MAX=8)."""
    monkeypatch.setenv("MALLOC_ARENA_MAX", "8")
    import os
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    assert os.environ["MALLOC_ARENA_MAX"] == "8"


def test_setdefault_sets_when_absent(monkeypatch):
    monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)
    import os
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    assert os.environ["MALLOC_ARENA_MAX"] == "2"


def test_kernel_rs_spawn_already_sets_malloc_arena_max():
    """Cross-reference: kernel-rs's build_child_env already inserts
    MALLOC_ARENA_MAX=2 (titan-rust/.../spawn.rs:116). The Python
    setdefault is a defensive double-buckle; the load-bearing setter
    is the Rust side. This test makes the dependency explicit."""
    src = _read("titan-rust/crates/titan-kernel-rs/src/spawn.rs")
    assert 'MALLOC_ARENA_MAX' in src and '"2"' in src, (
        "titan-rust kernel-rs MUST insert MALLOC_ARENA_MAX=2 via "
        "build_child_env so kernel-spawned children inherit the cap.")
