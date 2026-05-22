"""SPEC §24 Phase 5.5 — RebirthBackup._tier_specs_from_paths directory
recursion tests.

Ensures directory entries in PERSONALITY_PATHS / WEEKLY_EXTRA_PATHS
(e.g. data/sage_memory/, data/mini_reasoning/) are recursed into
per-file specs so the unified pipeline preserves the same backup
coverage as the legacy full-tarball cascade + L5 local_diff.
"""

from __future__ import annotations

import os

import pytest

from titan_hcl.logic.backup import RebirthBackup


def _rb_for_paths(monkeypatch, paths):
    """Construct a RebirthBackup test instance with a stubbed network so
    the method-under-test (_tier_specs_from_paths) doesn't trigger any
    init side effects. We patch the class attribute and call the method
    directly on a bare instance."""
    rb = RebirthBackup.__new__(RebirthBackup)
    rb._titan_id = "T1"
    rb._arweave_store = None
    return rb


def test_tier_specs_file_entries(tmp_path):
    rb = RebirthBackup.__new__(RebirthBackup)
    f1 = tmp_path / "a.json"; f1.write_text("{}")
    f2 = tmp_path / "b.bin"; f2.write_bytes(b"x")
    paths = [
        (str(f1), "a.json"),
        (str(f2), "b.bin"),
    ]
    specs = rb._tier_specs_from_paths(paths)
    arcs = sorted(s.arc_name for s in specs)
    assert arcs == ["a.json", "b.bin"]


def test_tier_specs_recurses_into_directory(tmp_path):
    rb = RebirthBackup.__new__(RebirthBackup)
    d = tmp_path / "sage_memory"
    d.mkdir()
    (d / "buffer.bin").write_bytes(b"x")
    (d / "meta.json").write_text("{}")
    sub = d / "sub"
    sub.mkdir()
    (sub / "deep.dat").write_bytes(b"y")
    paths = [(str(d), "sage_memory")]
    specs = rb._tier_specs_from_paths(paths)
    arcs = sorted(s.arc_name for s in specs)
    assert arcs == ["sage_memory/buffer.bin", "sage_memory/meta.json",
                    "sage_memory/sub/deep.dat"]


def test_tier_specs_directory_trailing_slash(tmp_path):
    rb = RebirthBackup.__new__(RebirthBackup)
    d = tmp_path / "mini"
    d.mkdir()
    (d / "x.json").write_text("{}")
    # Source path WITH trailing slash should also be treated as a dir
    paths = [(str(d) + "/", "mini")]
    specs = rb._tier_specs_from_paths(paths)
    arcs = sorted(s.arc_name for s in specs)
    assert arcs == ["mini/x.json"]


def test_tier_specs_skips_backup_artifacts(tmp_path):
    rb = RebirthBackup.__new__(RebirthBackup)
    d = tmp_path / "vm"
    d.mkdir()
    (d / "live.bin").write_bytes(b"x")
    (d / "live.bin.bak").write_bytes(b"old")
    (d / "live.bin.bak.prev").write_bytes(b"older")
    (d / "scratch.tmp").write_bytes(b"tmp")
    (d / "fork.repair").write_bytes(b"r")
    paths = [(str(d), "vm")]
    specs = rb._tier_specs_from_paths(paths)
    arcs = sorted(s.arc_name for s in specs)
    assert arcs == ["vm/live.bin"]


def test_tier_specs_missing_source_silently_skipped(tmp_path):
    rb = RebirthBackup.__new__(RebirthBackup)
    f1 = tmp_path / "exists.json"; f1.write_text("{}")
    missing = tmp_path / "gone.json"  # not created
    paths = [
        (str(f1), "exists.json"),
        (str(missing), "gone.json"),
    ]
    specs = rb._tier_specs_from_paths(paths)
    arcs = [s.arc_name for s in specs]
    assert arcs == ["exists.json"]


def test_tier_specs_timechain_hint_set(tmp_path):
    rb = RebirthBackup.__new__(RebirthBackup)
    tc_dir = tmp_path / "timechain"
    tc_dir.mkdir()
    (tc_dir / "chain_main.bin").write_bytes(b"x")
    (tc_dir / "chain_episodic.bin").write_bytes(b"y")
    paths = [(str(tc_dir), "timechain")]
    specs = rb._tier_specs_from_paths(paths)
    # All timechain .bin files should pick up the timechain_bin hint
    assert all(s.format_hint == "timechain_bin" for s in specs)


def test_tier_specs_explicit_format_hint_passed_through(tmp_path):
    rb = RebirthBackup.__new__(RebirthBackup)
    f = tmp_path / "weird.dat"
    f.write_bytes(b"x")
    specs = rb._tier_specs_from_paths(
        [(str(f), "weird.dat")], format_hint="db",
    )
    assert specs[0].format_hint == "db"


def test_tier_specs_empty_dir_yields_no_specs(tmp_path):
    rb = RebirthBackup.__new__(RebirthBackup)
    d = tmp_path / "empty"; d.mkdir()
    specs = rb._tier_specs_from_paths([(str(d), "empty")])
    assert specs == []


def test_tier_specs_real_personality_paths_classmethod(tmp_path, monkeypatch):
    """Sanity-check that the production PERSONALITY_PATHS class attribute
    has at least some directory entries that need recursion (which is why
    the recursion exists)."""
    has_dirs = any(
        isinstance(e, tuple) and len(e) >= 1 and
        (e[0].endswith("/") or
         (os.path.isdir(e[0]) if os.path.exists(e[0]) else False))
        for e in RebirthBackup.PERSONALITY_PATHS
    )
    # We're not asserting this holds for all environments — but the
    # mini_reasoning + titan_vm_v2 entries are dirs by design (Phase 1)
    dir_entries = [
        e for e in RebirthBackup.PERSONALITY_PATHS
        if isinstance(e, tuple) and len(e) >= 1 and e[0].endswith("/")
    ]
    assert any("mini_reasoning" in e[0] or "titan_vm_v2" in e[0]
               for e in dir_entries), (
        "PERSONALITY_PATHS should have directory entries (mini_reasoning, "
        "titan_vm_v2) that the recursion handles; if these were removed, "
        "this test needs updating."
    )
