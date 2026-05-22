"""
Tests for §4.D meditation_worker (D-SPEC-57, SPEC v1.8.3).

Coverage:
  • MeditationStatePublisher: cold-boot defaults, tracker mutators, completion
    recording, watchdog snapshot update, alert recording, msgpack round-trip,
    encode defense
  • MeditationStateReader: SHM round-trip, cold-boot returns None, is_in_meditation
    + get_count shortcuts
  • Bus event constants: MEDITATION_PHASE_CHANGED, MEDITATION_INTERRUPTED,
    MEDITATION_FORCE_END defined; producer-column updates verified
  • Constants TOML wiring: MEDITATION_STATE_SCHEMA_VERSION + MAX_BYTES
  • meditation_state_specs: RegistrySpec wiring (slot name, schema, max bytes)
  • Worker entry-point: imports, helpers present (SHM readers, persistence,
    backup_trigger file, InFlightRegistry pattern)
  • MeditationProxy: get_state / get_tracker / get_watchdog_health /
    is_in_meditation / get_count cold-defaults; force_end publish

Per CLAUDE.md: pytest -p no:anchorpy, separate process per file.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import msgpack
import pytest

from titan_hcl._phase_c_constants import (
    MEDITATION_STATE_MAX_BYTES,
    MEDITATION_STATE_SCHEMA_VERSION,
)
from titan_hcl.logic.meditation_state_publisher import MeditationStatePublisher
from titan_hcl.logic.meditation_state_reader import MeditationStateReader
from titan_hcl.logic.meditation_state_specs import (
    MEDITATION_STATE_SLOT,
    MEDITATION_STATE_SPEC,
)


# ── Per-test SHM root isolation ────────────────────────────────────


@pytest.fixture
def titan_id(tmp_path) -> str:
    """Per-test titan_id; SHM root cleaned up after test."""
    tid = f"T_TEST_{int(time.time() * 1e6) % 1_000_000}"
    yield tid
    shutil.rmtree(Path(f"/dev/shm/titan_{tid}"), ignore_errors=True)


# ── meditation_state_specs wiring ──────────────────────────────────


def test_meditation_state_slot_name():
    assert MEDITATION_STATE_SLOT == "meditation_state"


def test_meditation_state_spec_schema_version():
    assert MEDITATION_STATE_SPEC.schema_version == MEDITATION_STATE_SCHEMA_VERSION
    assert MEDITATION_STATE_SPEC.schema_version == 1


def test_meditation_state_spec_max_bytes():
    assert MEDITATION_STATE_SPEC.payload_bytes == MEDITATION_STATE_MAX_BYTES
    assert MEDITATION_STATE_SPEC.payload_bytes == 1024


def test_meditation_state_spec_is_variable_size():
    assert MEDITATION_STATE_SPEC.variable_size is True


# ── Constants TOML wiring ──────────────────────────────────────────


def test_constants_wired_per_dspec57():
    assert MEDITATION_STATE_SCHEMA_VERSION == 1
    assert MEDITATION_STATE_MAX_BYTES == 1024


# ── Bus event constants (NEW v1.8.3) + producer-column updates ─────


def test_bus_event_constants_defined():
    from titan_hcl import bus
    assert bus.MEDITATION_REQUEST == "MEDITATION_REQUEST"
    assert bus.MEDITATION_COMPLETE == "MEDITATION_COMPLETE"
    assert bus.MEDITATION_PHASE_CHANGED == "MEDITATION_PHASE_CHANGED"
    assert bus.MEDITATION_INTERRUPTED == "MEDITATION_INTERRUPTED"
    assert bus.MEDITATION_FORCE_END == "MEDITATION_FORCE_END"
    assert bus.MEDITATION_HEALTH_ALERT == "MEDITATION_HEALTH_ALERT"
    assert bus.MEDITATION_RECOVERY_TIER_1 == "MEDITATION_RECOVERY_TIER_1"
    assert bus.MEDITATION_RECOVERY_TIER_2 == "MEDITATION_RECOVERY_TIER_2"


# ── MeditationStatePublisher ───────────────────────────────────────


def test_publisher_cold_boot_defaults(titan_id):
    pub = MeditationStatePublisher(titan_id)
    pub.publish()
    snap = pub.snapshot()
    tracker = snap["tracker"]
    assert tracker["last_epoch"] == 0
    assert tracker["count"] == 0
    assert tracker["count_since_nft"] == 0
    assert tracker["last_ts"] == 0.0
    assert tracker["in_meditation"] is False
    assert tracker["current_phase"] == "idle"
    assert snap["last_alert"] is None
    assert snap["last_completion"] is None
    assert snap["schema_version"] == 1


def test_publisher_restore_tracker(titan_id):
    pub = MeditationStatePublisher(titan_id)
    pub.restore_tracker({
        "last_epoch": 1234,
        "count": 7,
        "count_since_nft": 2,
        "last_ts": 1700000000.0,
    })
    snap = pub.snapshot()
    tracker = snap["tracker"]
    assert tracker["last_epoch"] == 1234
    assert tracker["count"] == 7
    assert tracker["count_since_nft"] == 2
    assert tracker["last_ts"] == 1700000000.0
    # in_meditation always restored False on boot (safer than reviving stale True)
    assert tracker["in_meditation"] is False
    assert tracker["current_phase"] == "idle"


def test_publisher_set_in_meditation(titan_id):
    pub = MeditationStatePublisher(titan_id)
    assert pub.is_in_meditation() is False
    pub.set_in_meditation(True)
    assert pub.is_in_meditation() is True
    pub.set_in_meditation(False)
    assert pub.is_in_meditation() is False


def test_publisher_set_phase_valid(titan_id):
    pub = MeditationStatePublisher(titan_id)
    for phase in ("idle", "entering", "deep", "exiting"):
        pub.set_phase(phase)
        assert pub.snapshot()["tracker"]["current_phase"] == phase


def test_publisher_set_phase_invalid_coerced_to_idle(titan_id):
    pub = MeditationStatePublisher(titan_id)
    pub.set_phase("entering")
    pub.set_phase("garbage")
    assert pub.snapshot()["tracker"]["current_phase"] == "idle"


def test_publisher_record_completion(titan_id):
    pub = MeditationStatePublisher(titan_id)
    pub.set_in_meditation(True)
    pub.set_phase("exiting")
    completion = {
        "epoch": 100, "promoted": 5, "pruned": 2,
        "trigger": "emergent_driver", "success": True, "ts": time.time(),
    }
    pub.record_completion(epoch_id=100, completion=completion)
    snap = pub.snapshot()
    assert snap["tracker"]["in_meditation"] is False
    assert snap["tracker"]["last_epoch"] == 100
    assert snap["tracker"]["count"] == 1
    assert snap["tracker"]["count_since_nft"] == 1
    assert snap["tracker"]["current_phase"] == "idle"
    assert snap["last_completion"]["promoted"] == 5
    assert snap["last_completion"]["pruned"] == 2


def test_publisher_record_alert(titan_id):
    pub = MeditationStatePublisher(titan_id)
    pub.record_alert({
        "severity": "HIGH",
        "failure_mode": "F3_F6_STUCK",
        "detail": "stuck for 12 min",
        "ts": 1700000000.0,
    })
    snap = pub.snapshot()
    assert snap["last_alert"]["severity"] == "HIGH"
    assert snap["last_alert"]["failure_mode"] == "F3_F6_STUCK"
    assert snap["last_alert"]["detail"] == "stuck for 12 min"
    assert snap["last_alert"]["ts"] == 1700000000.0


def test_publisher_update_watchdog_snapshot(titan_id):
    pub = MeditationStatePublisher(titan_id)
    pub.update_watchdog_snapshot({
        "last_check_ts": 1700000000.0,
        "gap_samples": 12,
        "expected_interval_hours": 6.5,
        "in_meditation_since_ts": 0.0,
        "consecutive_zero_promoted": 0,
        "selftest_done": True,
        "selftest_pass": True,
    })
    wd = pub.snapshot()["watchdog"]
    assert wd["last_check_ts"] == 1700000000.0
    assert wd["gap_samples"] == 12
    assert wd["expected_interval_hours"] == 6.5
    assert wd["selftest_done"] is True
    assert wd["selftest_pass"] is True


def test_publisher_reset_count_since_nft(titan_id):
    pub = MeditationStatePublisher(titan_id)
    pub.restore_tracker({"count_since_nft": 3})
    assert pub.get_count_since_nft() == 3
    pub.reset_count_since_nft()
    assert pub.get_count_since_nft() == 0


def test_publisher_msgpack_round_trip(titan_id):
    pub = MeditationStatePublisher(titan_id)
    pub.set_phase("deep")
    pub.set_in_meditation(True)
    pub.record_alert({
        "severity": "MEDIUM", "failure_mode": "F7_NOT_DISTILLING",
        "detail": "3 zero-promoted streak", "ts": time.time(),
    })
    payload = pub.snapshot()
    encoded = msgpack.packb(payload, use_bin_type=True)
    assert len(encoded) < MEDITATION_STATE_MAX_BYTES, (
        f"encoded payload {len(encoded)}B must fit in {MEDITATION_STATE_MAX_BYTES}B slot")
    decoded = msgpack.unpackb(encoded, raw=False)
    assert decoded["tracker"]["current_phase"] == "deep"
    assert decoded["tracker"]["in_meditation"] is True
    assert decoded["last_alert"]["failure_mode"] == "F7_NOT_DISTILLING"


# ── MeditationStateReader ──────────────────────────────────────────


def test_reader_cold_boot_returns_none(titan_id):
    rdr = MeditationStateReader(titan_id)
    # No publisher has written yet — slot file absent.
    assert rdr.read() is None
    assert rdr.get_tracker() is None
    assert rdr.is_in_meditation() is False  # default
    assert rdr.get_count() == 0


def test_reader_round_trip_via_publisher(titan_id):
    pub = MeditationStatePublisher(titan_id)
    pub.restore_tracker({
        "last_epoch": 42, "count": 7, "count_since_nft": 2,
        "last_ts": 1700000000.0,
    })
    pub.set_phase("entering")
    pub.set_in_meditation(True)
    pub.publish()

    rdr = MeditationStateReader(titan_id)
    snap = rdr.read()
    assert snap is not None
    assert snap["tracker"]["count"] == 7
    assert snap["tracker"]["current_phase"] == "entering"
    assert snap["tracker"]["in_meditation"] is True
    assert snap["schema_version"] == 1

    # Convenience APIs.
    assert rdr.is_in_meditation() is True
    assert rdr.get_count() == 7


def test_reader_get_tracker_returns_section(titan_id):
    pub = MeditationStatePublisher(titan_id)
    pub.publish()
    rdr = MeditationStateReader(titan_id)
    tracker = rdr.get_tracker()
    assert isinstance(tracker, dict)
    assert "current_phase" in tracker
    assert "count" in tracker


def test_reader_get_watchdog_returns_section(titan_id):
    pub = MeditationStatePublisher(titan_id)
    pub.update_watchdog_snapshot({
        "last_check_ts": 1700000000.0,
        "gap_samples": 10,
        "expected_interval_hours": 5.0,
        "in_meditation_since_ts": 0.0,
        "consecutive_zero_promoted": 0,
        "selftest_done": True,
        "selftest_pass": True,
    })
    pub.publish()
    rdr = MeditationStateReader(titan_id)
    wd = rdr.get_watchdog()
    assert isinstance(wd, dict)
    assert wd["gap_samples"] == 10
    assert wd["selftest_pass"] is True


# ── Persistence + backup_trigger ───────────────────────────────────


def test_persistence_round_trip(tmp_path, monkeypatch):
    """Tracker persists to data/meditation_state.json + restores on second load."""
    monkeypatch.chdir(tmp_path)
    from titan_hcl.modules.meditation_worker import (
        TRACKER_PERSISTENCE_PATH,
        _load_tracker_from_disk,
        _persist_tracker,
    )
    _persist_tracker({
        "last_epoch": 99, "count": 4, "count_since_nft": 1,
        "last_ts": 1700000000.0,
    })
    assert os.path.exists(TRACKER_PERSISTENCE_PATH)
    loaded = _load_tracker_from_disk()
    assert loaded["last_epoch"] == 99
    assert loaded["count"] == 4
    assert loaded["count_since_nft"] == 1


def test_persistence_missing_file_returns_empty_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from titan_hcl.modules.meditation_worker import _load_tracker_from_disk
    loaded = _load_tracker_from_disk()
    assert loaded == {}


def test_backup_trigger_atomic_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from titan_hcl.modules.meditation_worker import (
        BACKUP_TRIGGER_PATH,
        _write_backup_trigger_file,
    )
    payload = {"epoch": 100, "promoted": 5, "pruned": 2, "success": True,
               "trigger": "emergent_driver", "ts": time.time()}
    _write_backup_trigger_file(payload, meditation_count=42)
    assert os.path.exists(BACKUP_TRIGGER_PATH)
    with open(BACKUP_TRIGGER_PATH) as f:
        data = json.load(f)
    assert data["meditation_count"] == 42
    assert data["payload"]["promoted"] == 5
    # No leftover tmp file.
    assert not os.path.exists(BACKUP_TRIGGER_PATH + ".tmp")


# ── Worker entry-point + InFlightRegistry pattern ──────────────────


def test_worker_module_imports():
    """Verify worker entry imports cleanly + key helpers exist."""
    from titan_hcl.modules import meditation_worker
    assert hasattr(meditation_worker, "meditation_worker_main")
    assert callable(meditation_worker.meditation_worker_main)
    assert hasattr(meditation_worker, "_InFlightRegistry")
    assert hasattr(meditation_worker, "_OrchestratorState")
    assert hasattr(meditation_worker, "_orchestrator_loop")
    assert hasattr(meditation_worker, "_build_shm_readers")
    assert hasattr(meditation_worker, "_persist_tracker")
    assert hasattr(meditation_worker, "_write_backup_trigger_file")
    # Chunk 1G — RFP §1.4 daemon-thread file-I/O executor helpers.
    assert hasattr(meditation_worker, "_get_io_executor")
    assert hasattr(meditation_worker, "_shutdown_io_executor")
    assert hasattr(meditation_worker, "_schedule_persist_tracker")
    assert hasattr(meditation_worker, "_schedule_backup_trigger")


def test_schedule_persist_tracker_writes_via_daemon_thread(tmp_path, monkeypatch):
    """Chunk 1G — _schedule_persist_tracker offloads to the io executor and
    eventually produces meditation_state.json on disk."""
    monkeypatch.chdir(tmp_path)
    from titan_hcl.modules.meditation_worker import (
        TRACKER_PERSISTENCE_PATH,
        _schedule_persist_tracker,
        _shutdown_io_executor,
    )
    try:
        _schedule_persist_tracker({
            "last_epoch": 7, "count": 11, "count_since_nft": 2,
            "last_ts": 1700000000.0,
        })
        # Drain — guarantees the submitted write completed.
        _shutdown_io_executor(wait=True)
        assert os.path.exists(TRACKER_PERSISTENCE_PATH)
        with open(TRACKER_PERSISTENCE_PATH) as f:
            data = json.load(f)
        assert data["last_epoch"] == 7
        assert data["count"] == 11
    finally:
        _shutdown_io_executor(wait=True)


def test_schedule_backup_trigger_writes_via_daemon_thread(tmp_path, monkeypatch):
    """Chunk 1G — _schedule_backup_trigger offloads to the io executor and
    eventually produces backup_trigger.json on disk. send_queue receives
    no alert when the write succeeds."""
    import queue as _q
    monkeypatch.chdir(tmp_path)
    from titan_hcl.modules.meditation_worker import (
        BACKUP_TRIGGER_PATH,
        _schedule_backup_trigger,
        _shutdown_io_executor,
    )
    sq: _q.Queue = _q.Queue()
    try:
        payload = {"epoch": 50, "promoted": 3, "pruned": 1, "success": True,
                   "trigger": "maker_manual", "ts": time.time()}
        _schedule_backup_trigger(sq, "meditation", payload, 17, "T1")
        _shutdown_io_executor(wait=True)
        assert os.path.exists(BACKUP_TRIGGER_PATH)
        with open(BACKUP_TRIGGER_PATH) as f:
            data = json.load(f)
        assert data["meditation_count"] == 17
        assert data["payload"]["epoch"] == 50
        # No alert emitted on success.
        assert sq.empty()
    finally:
        _shutdown_io_executor(wait=True)


def test_in_flight_registry_register_resolve():
    from titan_hcl.modules.meditation_worker import _InFlightRegistry
    registry = _InFlightRegistry()
    fut = registry.register("abc123")
    assert not fut.done()
    resolved = registry.resolve({"rid": "abc123", "type": "RESPONSE",
                                 "payload": {"success": True, "promoted": 3}})
    assert resolved is True
    assert fut.done()
    result = fut.result(timeout=0.1)
    assert result["payload"]["promoted"] == 3


def test_in_flight_registry_unknown_rid_passes():
    from titan_hcl.modules.meditation_worker import _InFlightRegistry
    registry = _InFlightRegistry()
    assert registry.resolve({"rid": "unknown", "type": "RESPONSE"}) is False
    assert registry.resolve({"type": "BROADCAST"}) is False


def test_in_flight_registry_cancel():
    from titan_hcl.modules.meditation_worker import _InFlightRegistry
    registry = _InFlightRegistry()
    fut = registry.register("xyz")
    registry.cancel("xyz")
    # Subsequent resolve should not crash but returns False (rid was popped).
    assert registry.resolve({"rid": "xyz", "type": "RESPONSE"}) is False


def test_orchestrator_state_enqueue_dedup():
    from titan_hcl.modules.meditation_worker import _OrchestratorState
    state = _OrchestratorState()
    assert state.enqueue({"source": "emergent_driver"}) is True
    # Second enqueue while busy is rejected.
    assert state.enqueue({"source": "watchdog_tier1"}) is False
    # After mark_idle, accepts again.
    state.mark_idle()
    assert state.enqueue({"source": "kin_sense"}) is True


# ── MeditationProxy ────────────────────────────────────────────────


def test_meditation_proxy_cold_defaults_no_worker(titan_id, monkeypatch):
    """Proxy returns default tracker/watchdog when SHM slot is empty."""
    # Stub out bus + guardian for unit isolation.
    class _StubBus:
        def publish(self, msg): return True

    class _StubGuardian:
        pass

    from titan_hcl.proxies.meditation_proxy import (
        _DEFAULT_TRACKER, _DEFAULT_WATCHDOG, MeditationProxy,
    )

    # Patch resolve_titan_id to return our test titan_id.
    monkeypatch.setattr(
        "titan_hcl.proxies.meditation_proxy.resolve_titan_id",
        lambda: titan_id,
    )

    proxy = MeditationProxy(bus=_StubBus(), guardian=_StubGuardian())
    tracker = proxy.get_tracker()
    assert tracker["count"] == 0
    assert tracker["in_meditation"] is False
    assert tracker["current_phase"] == "idle"
    wd = proxy.get_watchdog_health()
    assert wd["selftest_done"] is False
    assert proxy.is_in_meditation() is False
    assert proxy.get_count() == 0


def test_meditation_proxy_reads_published_state(titan_id, monkeypatch):
    """Proxy round-trips a fresh publisher write."""
    class _StubBus:
        def publish(self, msg): return True

    class _StubGuardian:
        pass

    from titan_hcl.proxies.meditation_proxy import MeditationProxy

    monkeypatch.setattr(
        "titan_hcl.proxies.meditation_proxy.resolve_titan_id",
        lambda: titan_id,
    )

    # Publisher writes a snapshot.
    pub = MeditationStatePublisher(titan_id)
    pub.restore_tracker({
        "last_epoch": 50, "count": 3, "count_since_nft": 1,
        "last_ts": 1700000000.0,
    })
    pub.set_phase("deep")
    pub.set_in_meditation(True)
    pub.publish()

    proxy = MeditationProxy(bus=_StubBus(), guardian=_StubGuardian())
    tracker = proxy.get_tracker()
    assert tracker["count"] == 3
    assert tracker["in_meditation"] is True
    assert tracker["current_phase"] == "deep"
    assert proxy.is_in_meditation() is True
    assert proxy.get_count() == 3


def test_meditation_proxy_force_end_publishes(titan_id, monkeypatch):
    """force_end() publishes a MEDITATION_FORCE_END bus event."""
    published: list[dict] = []

    class _CapturingBus:
        def publish(self, msg):
            published.append(msg)
            return True

    class _StubGuardian:
        pass

    from titan_hcl.proxies.meditation_proxy import MeditationProxy

    monkeypatch.setattr(
        "titan_hcl.proxies.meditation_proxy.resolve_titan_id",
        lambda: titan_id,
    )

    proxy = MeditationProxy(bus=_CapturingBus(), guardian=_StubGuardian())
    ok = proxy.force_end(reason="test_reason", source="unit_test")
    assert ok is True
    assert len(published) == 1
    msg = published[0]
    assert msg["type"] == "MEDITATION_FORCE_END"
    assert msg["dst"] == "meditation"
    assert msg["payload"]["reason"] == "test_reason"
    assert msg["payload"]["source"] == "unit_test"


# ── KERNEL_PROXY_ALIASES wiring ────────────────────────────────────


def test_meditation_proxy_in_kernel_aliases():
    """meditation_proxy must be in KERNEL_PROXY_ALIASES per SPEC v1.3.0
    multi-name BUS_SUBSCRIBE so proxy frames share the parent's bus
    connection."""
    from titan_hcl.core.kernel import KERNEL_PROXY_ALIASES
    assert "meditation_proxy" in KERNEL_PROXY_ALIASES
