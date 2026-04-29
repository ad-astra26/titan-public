"""
Tests for TitanKernel.hibernate_runtime + restore_from_snapshot — B.1 §3.

Mocks Soul + Guardian + RegistryBank to avoid full kernel boot. Verifies
the snapshot serialization path + the restore-time compat check.

PLAN: titan-docs/PLAN_microkernel_phase_b1_shadow_swap.md §3
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from titan_plugin.core import shadow_protocol as sp


def _make_minimal_kernel(
    *,
    titan_id: str = "T1",
    soul_gen: int = 42,
    guardian_modules: list[str] | None = None,
    registry_writers: dict[str, int] | None = None,
    bus_subscribers: dict[str, list] | None = None,
):
    """Build a TitanKernel-like mock with just the attributes hibernate_runtime
    + restore_from_snapshot read.

    We don't call TitanKernel.__init__ — that boots Soul + Guardian + bus
    + registers shm writers, all of which require disk/secrets/network.
    Instead we manufacture the minimal duck-type the hibernate methods need.
    """
    from titan_plugin.core.kernel import TitanKernel

    kernel = TitanKernel.__new__(TitanKernel)
    kernel._titan_id = titan_id
    kernel._kernel_version_cache = "abc12345"

    # Soul mock
    soul = MagicMock()
    soul.current_gen = soul_gen
    kernel.soul = soul

    # Guardian mock
    guardian = MagicMock()
    guardian._modules = {name: MagicMock() for name in (guardian_modules or [])}
    kernel.guardian = guardian

    # RegistryBank mock — populate _writers dict with mock writers having _seq
    bank = MagicMock()
    writers = {}
    for name, seq in (registry_writers or {}).items():
        w = MagicMock()
        w._seq = seq
        writers[name] = w
    bank._writers = writers
    kernel.registry_bank = bank

    # Bus mock
    bus = MagicMock()
    bus._subscribers = bus_subscribers or {}
    kernel.bus = bus

    return kernel


# ── hibernate_runtime() ────────────────────────────────────────────

class TestHibernateRuntime:
    def test_writes_snapshot_with_all_fields(self, tmp_path):
        kernel = _make_minimal_kernel(
            titan_id="T1",
            soul_gen=42,
            guardian_modules=["spirit", "body", "mind"],
            registry_writers={"trinity_state": 100, "neuromod_state": 50},
            bus_subscribers={"spirit": [object()], "body": [object(), object()]},
        )
        snapshot_path = tmp_path / "test.msgpack"
        eid = sp.new_event_id()

        result_path = kernel.hibernate_runtime(eid, str(snapshot_path))

        assert result_path == str(snapshot_path)
        assert snapshot_path.exists()
        snap = sp.deserialize_snapshot(snapshot_path)
        assert snap.titan_id == "T1"
        assert snap.soul_current_gen == 42
        assert snap.event_id == eid
        assert snap.kernel_version == "abc12345"
        assert sorted(snap.guardian_modules) == ["body", "mind", "spirit"]
        assert snap.registry_seqs == {"trinity_state": 100, "neuromod_state": 50}
        assert snap.bus_subscriber_count == 3  # 1 + 2

    def test_handles_no_soul_limbo_mode(self, tmp_path):
        kernel = _make_minimal_kernel(soul_gen=0)
        kernel.soul = None  # limbo mode
        snapshot_path = tmp_path / "limbo.msgpack"

        kernel.hibernate_runtime("eid123", str(snapshot_path))

        snap = sp.deserialize_snapshot(snapshot_path)
        assert snap.soul_current_gen == 0  # default when no soul

    def test_handles_no_registry_bank(self, tmp_path):
        kernel = _make_minimal_kernel()
        kernel.registry_bank = None
        snapshot_path = tmp_path / "no_bank.msgpack"

        kernel.hibernate_runtime("eid123", str(snapshot_path))

        snap = sp.deserialize_snapshot(snapshot_path)
        assert snap.registry_seqs == {}

    def test_uses_default_path_when_omitted(self, tmp_path, monkeypatch):
        kernel = _make_minimal_kernel()
        # Redirect the default snapshot dir into tmp_path so we don't touch /tmp
        monkeypatch.setattr(sp, "DEFAULT_SNAPSHOT_DIR", tmp_path)

        result_path = kernel.hibernate_runtime("eid123")

        assert Path(result_path).parent == tmp_path
        assert Path(result_path).name == sp.DEFAULT_SNAPSHOT_NAME
        assert Path(result_path).exists()

    def test_event_id_propagates_to_snapshot(self, tmp_path):
        kernel = _make_minimal_kernel()
        eid1 = sp.new_event_id()
        eid2 = sp.new_event_id()
        path1 = tmp_path / "a.msgpack"
        path2 = tmp_path / "b.msgpack"

        kernel.hibernate_runtime(eid1, str(path1))
        kernel.hibernate_runtime(eid2, str(path2))

        assert sp.deserialize_snapshot(path1).event_id == eid1
        assert sp.deserialize_snapshot(path2).event_id == eid2
        assert eid1 != eid2


# ── restore_from_snapshot() ────────────────────────────────────────

class TestRestoreFromSnapshot:
    def _write_snap(self, path: Path, **overrides):
        defaults = dict(
            kernel_version="abc12345",
            soul_current_gen=42,
            titan_id="T1",
            registry_seqs={},
            guardian_modules=["spirit", "body"],
            bus_subscriber_count=10,
            written_at=time.time(),
            event_id=sp.new_event_id(),
        )
        defaults.update(overrides)
        snap = sp.RuntimeSnapshot(**defaults)
        sp.serialize_snapshot(snap, path)
        return snap

    def test_verifies_compatible_snapshot(self, tmp_path):
        snap_path = tmp_path / "ok.msgpack"
        original = self._write_snap(snap_path)

        kernel = _make_minimal_kernel(
            titan_id="T1",
            guardian_modules=["spirit", "body", "extra_module"],  # superset OK
        )

        result = kernel.restore_from_snapshot(str(snap_path))

        assert result["verified"] is True
        assert result["reason"] == "ok"
        assert result["event_id"] == original.event_id
        assert result["kernel_version_from"] == "abc12345"
        assert result["soul_gen_from"] == 42

    def test_handles_missing_snapshot_gracefully(self, tmp_path):
        kernel = _make_minimal_kernel(titan_id="T1")
        result = kernel.restore_from_snapshot(str(tmp_path / "nope.msgpack"))

        assert result["verified"] is False
        assert result["reason"] == "file_not_found"
        assert result["event_id"] == ""

    def test_handles_corrupt_snapshot_gracefully(self, tmp_path):
        snap_path = tmp_path / "corrupt.msgpack"
        snap_path.write_bytes(b"this is not msgpack")

        kernel = _make_minimal_kernel(titan_id="T1")
        result = kernel.restore_from_snapshot(str(snap_path))

        assert result["verified"] is False
        assert result["reason"].startswith("deserialize_error")
        assert result["event_id"] == ""

    def test_refuses_titan_id_mismatch(self, tmp_path):
        snap_path = tmp_path / "wrong_titan.msgpack"
        self._write_snap(snap_path, titan_id="T1")

        kernel = _make_minimal_kernel(
            titan_id="T2",  # mismatch
            guardian_modules=["spirit", "body"],
        )
        result = kernel.restore_from_snapshot(str(snap_path))

        assert result["verified"] is False
        assert "titan_id" in result["reason"]

    def test_refuses_stale_snapshot(self, tmp_path):
        snap_path = tmp_path / "stale.msgpack"
        self._write_snap(snap_path, written_at=time.time() - 1000.0)  # 16+ min old

        kernel = _make_minimal_kernel(
            titan_id="T1",
            guardian_modules=["spirit", "body"],
        )
        result = kernel.restore_from_snapshot(str(snap_path), max_age_seconds=300.0)

        assert result["verified"] is False
        assert "stale" in result["reason"]

    def test_refuses_missing_module_in_target(self, tmp_path):
        snap_path = tmp_path / "missing_mod.msgpack"
        self._write_snap(
            snap_path, guardian_modules=["spirit", "body", "old_module"],
        )

        kernel = _make_minimal_kernel(
            titan_id="T1",
            guardian_modules=["spirit", "body"],  # missing "old_module"
        )
        result = kernel.restore_from_snapshot(str(snap_path))

        assert result["verified"] is False
        assert "old_module" in result["reason"]


# ── kernel_version property ────────────────────────────────────────

class TestKernelVersion:
    def test_caches_after_first_call(self):
        kernel = _make_minimal_kernel()
        # Cache pre-set in our test factory; verify it's returned verbatim
        assert kernel.kernel_version == "abc12345"
        assert kernel._kernel_version_cache == "abc12345"

    def test_falls_back_to_unknown_if_not_cached_and_git_fails(self, monkeypatch):
        from titan_plugin.core.kernel import TitanKernel
        kernel = TitanKernel.__new__(TitanKernel)
        # Don't pre-set _kernel_version_cache — force the live lookup
        kernel._kernel_version_cache = None

        # Force subprocess.run to fail
        import subprocess
        def fake_run(*a, **kw):
            raise FileNotFoundError("git")
        monkeypatch.setattr(subprocess, "run", fake_run)

        version = kernel.kernel_version
        assert version == "unknown"
