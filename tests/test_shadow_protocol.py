"""
Tests for titan_plugin.core.shadow_protocol — B.1 §2.

Covers dataclass round-trips, msgpack serialization, compatibility checks,
event_id generation, and SHA-256 file checksums.

PLAN: titan-docs/PLAN_microkernel_phase_b1_shadow_swap.md §2
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from titan_plugin.core import shadow_protocol as sp


# ── Constants (smoke-test the canonical values) ────────────────────

class TestConstants:
    def test_grace_is_120s_no_force_design(self):
        assert sp.READINESS_GRACE_SECONDS == 120.0

    def test_poll_interval_under_grace(self):
        assert 0.5 <= sp.READINESS_POLL_INTERVAL <= 2.0

    def test_per_layer_timeouts_monotonic(self):
        # L0 < L1 < L2 < L3 (deeper layers may need longer to save state)
        t = sp.HIBERNATE_ACK_TIMEOUT_BY_LAYER
        assert t["L0"] < t["L1"] < t["L2"] < t["L3"]

    def test_ping_pong_ports_distinct(self):
        a, b = sp.PING_PONG_PORTS
        assert a != b
        assert {a, b} == {7777, 7779}

    def test_schema_version_is_one(self):
        assert sp.SNAPSHOT_SCHEMA_VERSION == 1


# ── HardBlocker / SoftBlocker ──────────────────────────────────────

class TestBlockers:
    def test_hard_blocker_to_dict_round_trip(self):
        b = sp.HardBlocker(name="x_post_in_flight", eta_seconds=8.4, since=1745695200.0)
        d = b.to_dict()
        assert d == {
            "name": "x_post_in_flight",
            "eta_seconds": 8.4,
            "since": 1745695200.0,
        }

    def test_soft_blocker_default_metadata_empty(self):
        b = sp.SoftBlocker(name="dream_cycle", eta_seconds=42.0)
        assert b.metadata == {}

    def test_soft_blocker_with_metadata(self):
        b = sp.SoftBlocker(
            name="reasoning_chain",
            eta_seconds=4.0,
            metadata={"chain_id": 4521, "step": 3, "of": 7},
        )
        d = b.to_dict()
        assert d["metadata"]["chain_id"] == 4521
        assert d["metadata"]["of"] == 7

    def test_hard_blocker_frozen(self):
        # frozen=True so blockers can't be mutated after construction
        b = sp.HardBlocker(name="x", eta_seconds=1.0, since=0.0)
        with pytest.raises(Exception):
            b.name = "y"


# ── ReadinessReport ────────────────────────────────────────────────

class TestReadinessReport:
    def test_empty_blockers_means_ready(self):
        r = sp.ReadinessReport(src="spirit", ready=False)  # claim not-ready
        # __post_init__ should fix the invariant — empty lists ⇒ ready
        assert r.ready is True

    def test_any_hard_blocker_means_not_ready(self):
        r = sp.ReadinessReport(
            src="spirit", ready=True,
            hard=[sp.HardBlocker(name="x", eta_seconds=1.0, since=0.0)],
        )
        assert r.ready is False

    def test_any_soft_blocker_means_not_ready(self):
        r = sp.ReadinessReport(
            src="spirit", ready=True,
            soft=[sp.SoftBlocker(name="dream", eta_seconds=10.0)],
        )
        assert r.ready is False

    def test_payload_round_trip(self):
        original = sp.ReadinessReport(
            src="api_subprocess",
            ready=False,
            hard=[
                sp.HardBlocker(name="chat_session", eta_seconds=12.0, since=1.0),
                sp.HardBlocker(name="x_post", eta_seconds=4.0, since=2.0),
            ],
            soft=[
                sp.SoftBlocker(name="research_query", eta_seconds=20.0,
                               metadata={"query_id": "q42"}),
            ],
            module_health="ok",
        )
        payload = original.to_payload()
        restored = sp.ReadinessReport.from_payload(payload)
        assert restored.src == original.src
        assert restored.ready == original.ready
        assert len(restored.hard) == 2
        assert restored.hard[0].name == "chat_session"
        assert restored.hard[1].name == "x_post"
        assert len(restored.soft) == 1
        assert restored.soft[0].metadata == {"query_id": "q42"}
        assert restored.module_health == "ok"


# ── HibernateAck ───────────────────────────────────────────────────

class TestHibernateAck:
    def test_payload_round_trip(self):
        a = sp.HibernateAck(
            src="spirit",
            layer="L1",
            state_paths=["data/spirit_state.json", "data/sphere_clock.json"],
            state_checksum="ab12cd34",
            elapsed_ms=42.5,
        )
        restored = sp.HibernateAck.from_payload(a.to_payload())
        assert restored.src == "spirit"
        assert restored.layer == "L1"
        assert restored.state_paths == ["data/spirit_state.json", "data/sphere_clock.json"]
        assert restored.state_checksum == "ab12cd34"
        assert restored.elapsed_ms == 42.5

    def test_ack_defaults_safe(self):
        # Minimal ack — only src + layer required
        a = sp.HibernateAck(src="body", layer="L1")
        assert a.state_paths == []
        assert a.state_checksum == ""
        assert a.elapsed_ms == 0.0


# ── RuntimeSnapshot serialization ──────────────────────────────────

class TestRuntimeSnapshot:
    def _sample_snap(self, **overrides) -> sp.RuntimeSnapshot:
        defaults = dict(
            kernel_version="abc12345",
            soul_current_gen=42,
            titan_id="T1",
            registry_seqs={"trinity_state": 100, "neuromod_state": 50},
            guardian_modules=["spirit", "body", "mind", "memory"],
            bus_subscriber_count=27,
            written_at=time.time(),
            event_id=sp.new_event_id(),
        )
        defaults.update(overrides)
        return sp.RuntimeSnapshot(**defaults)

    def test_serialize_then_deserialize_round_trip(self, tmp_path):
        snap = self._sample_snap()
        path = tmp_path / "runtime.msgpack"
        sp.serialize_snapshot(snap, path)
        assert path.exists()
        restored = sp.deserialize_snapshot(path)
        assert restored.kernel_version == snap.kernel_version
        assert restored.soul_current_gen == snap.soul_current_gen
        assert restored.titan_id == snap.titan_id
        assert restored.registry_seqs == snap.registry_seqs
        assert restored.guardian_modules == snap.guardian_modules
        assert restored.bus_subscriber_count == snap.bus_subscriber_count
        assert restored.event_id == snap.event_id
        assert restored.schema_version == snap.schema_version

    def test_serialize_atomic_no_partial_file(self, tmp_path):
        # After serialize_snapshot returns, no .tmp file remains
        snap = self._sample_snap()
        path = tmp_path / "runtime.msgpack"
        sp.serialize_snapshot(snap, path)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Stray tmp files: {tmp_files}"

    def test_deserialize_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            sp.deserialize_snapshot(tmp_path / "nope.msgpack")

    def test_deserialize_corrupt_raises(self, tmp_path):
        path = tmp_path / "corrupt.msgpack"
        path.write_bytes(b"this is not msgpack")
        with pytest.raises(Exception):
            sp.deserialize_snapshot(path)

    def test_default_snapshot_path_creates_parent(self, tmp_path):
        sub = tmp_path / "deep" / "nested" / "shadow"
        p = sp.default_snapshot_path(sub)
        assert p.parent.exists()
        assert p.name == sp.DEFAULT_SNAPSHOT_NAME


# ── verify_compatible() ────────────────────────────────────────────

class TestCompatibility:
    def _snap(self, **overrides):
        defaults = dict(
            kernel_version="abc12345",
            soul_current_gen=42,
            titan_id="T1",
            registry_seqs={},
            guardian_modules=["spirit", "body", "mind"],
            bus_subscriber_count=10,
            written_at=time.time(),
            event_id=sp.new_event_id(),
        )
        defaults.update(overrides)
        return sp.RuntimeSnapshot(**defaults)

    def test_compatible_pass(self):
        snap = self._snap()
        ok, reason = sp.verify_compatible(
            snap, target_titan_id="T1",
            target_modules=["spirit", "body", "mind"],
        )
        assert ok is True
        assert reason == "ok"

    def test_target_can_be_superset_of_snapshot(self):
        # New modules added in shadow — fine
        snap = self._snap(guardian_modules=["spirit", "body"])
        ok, reason = sp.verify_compatible(
            snap, target_titan_id="T1",
            target_modules=["spirit", "body", "mind", "new_module"],
        )
        assert ok is True

    def test_schema_version_mismatch_refused(self):
        snap = self._snap()
        snap.schema_version = 99
        ok, reason = sp.verify_compatible(
            snap, target_titan_id="T1", target_modules=snap.guardian_modules,
        )
        assert ok is False
        assert "schema_version" in reason

    def test_titan_id_mismatch_refused(self):
        snap = self._snap(titan_id="T1")
        ok, reason = sp.verify_compatible(
            snap, target_titan_id="T2", target_modules=snap.guardian_modules,
        )
        assert ok is False
        assert "titan_id" in reason

    def test_stale_snapshot_refused(self):
        snap = self._snap(written_at=time.time() - 1000.0)  # 16+ min old
        ok, reason = sp.verify_compatible(
            snap, target_titan_id="T1", target_modules=snap.guardian_modules,
            max_age_seconds=300.0,
        )
        assert ok is False
        assert "stale" in reason

    def test_target_missing_modules_refused(self):
        snap = self._snap(guardian_modules=["spirit", "body", "old_module"])
        ok, reason = sp.verify_compatible(
            snap, target_titan_id="T1",
            target_modules=["spirit", "body"],  # missing "old_module"
        )
        assert ok is False
        assert "old_module" in reason


# ── event_id + sha256 helpers ──────────────────────────────────────

class TestHelpers:
    def test_new_event_id_unique(self):
        ids = {sp.new_event_id() for _ in range(100)}
        assert len(ids) == 100, "UUID collision in 100 calls (vanishingly unlikely)"

    def test_new_event_id_hex_format(self):
        eid = sp.new_event_id()
        assert len(eid) == 32
        assert all(c in "0123456789abcdef" for c in eid)

    def test_sha256_deterministic(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"hello")
        f2.write_bytes(b"world")
        a = sp.sha256_of_files([f1, f2])
        b = sp.sha256_of_files([f1, f2])
        assert a == b
        assert len(a) == 64  # SHA-256 hex

    def test_sha256_order_sensitive(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"hello")
        f2.write_bytes(b"world")
        a = sp.sha256_of_files([f1, f2])
        b = sp.sha256_of_files([f2, f1])
        assert a != b, "Re-ordering files must change the checksum"

    def test_sha256_handles_missing_file(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f1.write_bytes(b"hello")
        missing = tmp_path / "does_not_exist.bin"
        # Must not raise — folds in <MISSING> sentinel
        result = sp.sha256_of_files([f1, missing])
        assert len(result) == 64
        # And changes if we now create the missing file
        missing.write_bytes(b"appeared")
        result2 = sp.sha256_of_files([f1, missing])
        assert result != result2
