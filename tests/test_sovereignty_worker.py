"""tests/test_sovereignty_worker.py — §4.L sovereignty_worker (D-SPEC-57, SPEC v1.8.3)

Coverage (worker-specific integration; class-level coverage lives in
tests/test_sovereignty_tracker.py):

  • Bus event constants defined: SOVEREIGNTY_EPOCH, SOVEREIGNTY_CONFIRM_MAKER,
    SOVEREIGNTY_INCREMENT_GREAT_CYCLE
  • Worker module imports cleanly
  • SOVEREIGNTY_EPOCH dispatch → tracker.record_epoch(...) called with
    correct payload-to-arg mapping (epoch_id, neuromods→neuromod_levels,
    dev_age→developmental_age, great_pulse_fired)
  • SOVEREIGNTY_CONFIRM_MAKER dispatch → tracker._maker_confirmed=True +
    JSON persisted
  • SOVEREIGNTY_CONFIRM_MAKER idempotent — second emit no-ops
  • SOVEREIGNTY_INCREMENT_GREAT_CYCLE dispatch → tracker._great_cycle += 1
    + JSON persisted
  • MODULE_READY emitted on boot with sovereignty_mode + great_cycle payload
  • MODULE_SHUTDOWN dispatch → tracker._save_state() called + clean exit
  • MODULE_HEARTBEAT thread runs (verifies daemon doesn't crash)
  • JSON state load on boot (resumes from disk)
  • 100-message criteria snapshot log (smoke test — log emission doesn't crash)

Per CLAUDE.md: pytest -p no:anchorpy, separate process per file.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from typing import Optional

import pytest

from titan_hcl import bus
from titan_hcl.bus import (
    MODULE_HEARTBEAT,
    MODULE_READY,
    MODULE_SHUTDOWN,
    SOVEREIGNTY_CONFIRM_MAKER,
    SOVEREIGNTY_EPOCH,
    SOVEREIGNTY_INCREMENT_GREAT_CYCLE,
    make_msg,
)
from titan_hcl.logic import sovereignty as sov_module
from titan_hcl.modules.sovereignty_worker import (
    HEARTBEAT_INTERVAL_S,
    sovereignty_worker_main,
)


# ── Bus event constants ────────────────────────────────────────────


def test_bus_event_constants_defined():
    """Bus constants for §4.L are defined as expected strings."""
    assert bus.SOVEREIGNTY_EPOCH == "SOVEREIGNTY_EPOCH"
    assert bus.SOVEREIGNTY_CONFIRM_MAKER == "SOVEREIGNTY_CONFIRM_MAKER"
    assert bus.SOVEREIGNTY_INCREMENT_GREAT_CYCLE == "SOVEREIGNTY_INCREMENT_GREAT_CYCLE"


def test_worker_module_exports():
    """Module exports `sovereignty_worker_main` entry-point + heartbeat const."""
    assert callable(sovereignty_worker_main)
    assert isinstance(HEARTBEAT_INTERVAL_S, (int, float))
    assert HEARTBEAT_INTERVAL_S > 0


# ── Test harness: drive worker in a thread, capture send_queue output ──


@pytest.fixture
def tmp_state_file(tmp_path, monkeypatch):
    """Redirect SovereigntyTracker PERSISTENCE_FILE to a per-test tmp path."""
    tmp_file = tmp_path / "sovereignty_state.json"
    monkeypatch.setattr(sov_module, "PERSISTENCE_FILE", str(tmp_file))
    return tmp_file


class _WorkerHarness:
    """Spawn sovereignty_worker_main in a background thread for tests."""

    def __init__(self, name: str = "sovereignty"):
        self.name = name
        self.recv: queue.Queue = queue.Queue()
        self.send: queue.Queue = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.exc: Optional[BaseException] = None

    def start(self) -> None:
        def _run():
            try:
                sovereignty_worker_main(
                    self.recv, self.send, self.name, {})
            except BaseException as e:
                self.exc = e

        self.thread = threading.Thread(
            target=_run, daemon=True,
            name=f"sov-worker-test-{self.name}")
        self.thread.start()

    def send_event(self, msg_type: str, payload: dict,
                   src: str = "test", rid: Optional[str] = None) -> None:
        self.recv.put(make_msg(msg_type, src, self.name, payload, rid=rid))

    def shutdown(self, timeout: float = 5.0) -> None:
        self.recv.put(make_msg(MODULE_SHUTDOWN, "guardian", self.name, {}))
        if self.thread is not None:
            self.thread.join(timeout=timeout)
            assert not self.thread.is_alive(), "worker did not exit on MODULE_SHUTDOWN"
        if self.exc is not None:
            raise self.exc

    def drain_send(self, max_wait: float = 2.0) -> list[dict]:
        """Collect all messages emitted to send_queue up to MODULE_READY arrival
        + a small drain window."""
        out: list[dict] = []
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                msg = self.send.get(timeout=0.1)
            except queue.Empty:
                continue
            out.append(msg)
        return out

    def wait_for_ready(self, timeout: float = 5.0) -> dict:
        """Block until MODULE_READY is observed in send_queue. Returns the
        READY message."""
        deadline = time.time() + timeout
        captured: list[dict] = []
        while time.time() < deadline:
            try:
                msg = self.send.get(timeout=0.2)
            except queue.Empty:
                continue
            captured.append(msg)
            if msg.get("type") == MODULE_READY:
                # Put captured non-READY messages back? No — tests can
                # assert on later state directly. Just return.
                return msg
        pytest.fail(f"MODULE_READY not observed in {timeout}s; got: "
                    f"{[m.get('type') for m in captured]}")


# ── Boot lifecycle ─────────────────────────────────────────────────


def test_worker_boots_and_emits_module_ready(tmp_state_file):
    h = _WorkerHarness()
    h.start()
    try:
        ready = h.wait_for_ready(timeout=5.0)
        payload = ready.get("payload", {})
        assert payload.get("module") == "sovereignty_worker"
        assert payload.get("version") == "1.8.3"
        assert payload.get("spec_ref") == "D-SPEC-57"
        # Cold boot — fresh tracker has ENFORCING mode + great_cycle=0.
        assert payload.get("sovereignty_mode") == "ENFORCING"
        assert payload.get("great_cycle") == 0
    finally:
        h.shutdown()


def test_worker_emits_heartbeat(tmp_state_file, monkeypatch):
    """Worker daemon heartbeat thread emits MODULE_HEARTBEAT at cadence."""
    # Shorten heartbeat cadence to make the test fast.
    monkeypatch.setattr(
        "titan_hcl.modules.sovereignty_worker.HEARTBEAT_INTERVAL_S", 0.1)
    h = _WorkerHarness()
    h.start()
    try:
        h.wait_for_ready(timeout=5.0)
        # Collect for ~0.5s — should see ≥2 heartbeats.
        deadline = time.time() + 0.5
        heartbeats = 0
        while time.time() < deadline:
            try:
                msg = h.send.get(timeout=0.1)
            except queue.Empty:
                continue
            if msg.get("type") == MODULE_HEARTBEAT:
                heartbeats += 1
        assert heartbeats >= 2, (
            f"expected ≥2 heartbeats in 0.5s at 0.1s cadence; got {heartbeats}")
    finally:
        h.shutdown()


# ── SOVEREIGNTY_EPOCH dispatch ─────────────────────────────────────


def test_sovereignty_epoch_dispatch_payload_mapping(tmp_state_file):
    """SOVEREIGNTY_EPOCH payload fields map correctly to tracker.record_epoch."""
    h = _WorkerHarness()
    h.start()
    try:
        h.wait_for_ready(timeout=5.0)
        # Send 1 epoch with realistic payload.
        h.send_event(SOVEREIGNTY_EPOCH, {
            "epoch_id": 100,
            "neuromods": {"DA": 0.6, "5HT": 0.7, "NE": 0.5,
                          "ACh": 0.65, "Endorphin": 0.55, "GABA": 0.2},
            "dev_age": 250,
            "great_pulse_fired": True,
            "total_great_pulses": 1,
        })
        # Allow worker to drain the queue.
        time.sleep(0.3)
    finally:
        h.shutdown()

    # State file persists after MODULE_SHUTDOWN (worker calls _save_state).
    assert tmp_state_file.exists(), \
        "MODULE_SHUTDOWN should persist sovereignty_state.json"
    data = json.loads(tmp_state_file.read_text())
    # great_pulse_fired=True → total_great_pulses += 1
    assert data.get("total_great_pulses") == 1
    # dev_age=250 → developmental_age tracked
    assert data.get("developmental_age") == 250


def test_sovereignty_epoch_great_pulse_count_increments(tmp_state_file):
    """Multiple epochs with great_pulse_fired increment total_great_pulses."""
    h = _WorkerHarness()
    h.start()
    try:
        h.wait_for_ready(timeout=5.0)
        for epoch_id in (10, 20, 30, 40, 50):
            h.send_event(SOVEREIGNTY_EPOCH, {
                "epoch_id": epoch_id,
                "neuromods": {"DA": 0.5, "5HT": 0.5},
                "dev_age": 100,
                "great_pulse_fired": True,
            })
        time.sleep(0.3)
    finally:
        h.shutdown()

    data = json.loads(tmp_state_file.read_text())
    assert data.get("total_great_pulses") == 5


# ── SOVEREIGNTY_CONFIRM_MAKER dispatch ─────────────────────────────


def test_confirm_maker_dispatch_flips_flag(tmp_state_file):
    """SOVEREIGNTY_CONFIRM_MAKER → tracker._maker_confirmed=True + persisted."""
    h = _WorkerHarness()
    h.start()
    try:
        h.wait_for_ready(timeout=5.0)
        h.send_event(SOVEREIGNTY_CONFIRM_MAKER, {
            "tx_signature": "abcdef0123456789abcdef0123456789",
            "ts": time.time(),
        })
        time.sleep(0.3)
    finally:
        h.shutdown()

    data = json.loads(tmp_state_file.read_text())
    assert data.get("maker_confirmed") is True


def test_confirm_maker_idempotent(tmp_state_file):
    """Second SOVEREIGNTY_CONFIRM_MAKER emit no-ops after first sets flag."""
    h = _WorkerHarness()
    h.start()
    try:
        h.wait_for_ready(timeout=5.0)
        h.send_event(SOVEREIGNTY_CONFIRM_MAKER, {
            "tx_signature": "first_tx",
            "ts": time.time(),
        })
        time.sleep(0.2)
        h.send_event(SOVEREIGNTY_CONFIRM_MAKER, {
            "tx_signature": "second_tx",
            "ts": time.time(),
        })
        time.sleep(0.2)
    finally:
        h.shutdown()

    data = json.loads(tmp_state_file.read_text())
    # Flag is True (set by first event); second emit didn't crash.
    assert data.get("maker_confirmed") is True


# ── SOVEREIGNTY_INCREMENT_GREAT_CYCLE dispatch ─────────────────────


def test_increment_great_cycle_dispatch(tmp_state_file):
    """SOVEREIGNTY_INCREMENT_GREAT_CYCLE → great_cycle += 1 + persisted."""
    h = _WorkerHarness()
    h.start()
    try:
        h.wait_for_ready(timeout=5.0)
        h.send_event(SOVEREIGNTY_INCREMENT_GREAT_CYCLE, {
            "ts": time.time(),
            "source": "resurrection",
        })
        time.sleep(0.3)
    finally:
        h.shutdown()

    data = json.loads(tmp_state_file.read_text())
    assert data.get("great_cycle") == 1


def test_increment_great_cycle_multiple(tmp_state_file):
    """Multiple SOVEREIGNTY_INCREMENT_GREAT_CYCLE emits increment cumulatively."""
    h = _WorkerHarness()
    h.start()
    try:
        h.wait_for_ready(timeout=5.0)
        for _ in range(3):
            h.send_event(SOVEREIGNTY_INCREMENT_GREAT_CYCLE, {
                "ts": time.time(),
                "source": "test_repeated",
            })
            time.sleep(0.05)
        time.sleep(0.3)
    finally:
        h.shutdown()

    data = json.loads(tmp_state_file.read_text())
    assert data.get("great_cycle") == 3


# ── State persistence: JSON load on boot ───────────────────────────


def test_worker_loads_existing_state_on_boot(tmp_state_file):
    """If sovereignty_state.json exists pre-boot, worker resumes from it."""
    # Pre-populate the state file.
    tmp_state_file.write_text(json.dumps({
        "great_cycle": 2,
        "total_great_pulses": 1234,
        "sovereignty_mode": "ADVISORY",
        "saturation_violations": 5,
        "collapse_violations": 3,
        "transition_epoch": 50000,
        "maker_confirmed": True,
        "developmental_age": 2500,
        "updated_at": time.time(),
    }))

    h = _WorkerHarness()
    h.start()
    try:
        ready = h.wait_for_ready(timeout=5.0)
        payload = ready.get("payload", {})
        # Worker boots with persisted state.
        assert payload.get("sovereignty_mode") == "ADVISORY"
        assert payload.get("great_cycle") == 2
    finally:
        h.shutdown()


def test_worker_shutdown_persists_state(tmp_state_file):
    """MODULE_SHUTDOWN handler calls _save_state before exit."""
    h = _WorkerHarness()
    h.start()
    try:
        h.wait_for_ready(timeout=5.0)
        # Mutate state via an event.
        h.send_event(SOVEREIGNTY_INCREMENT_GREAT_CYCLE, {
            "ts": time.time(),
            "source": "test_shutdown_persistence",
        })
        time.sleep(0.2)
        # Delete the JSON file to verify shutdown rewrites it.
        if tmp_state_file.exists():
            tmp_state_file.unlink()
    finally:
        h.shutdown()

    assert tmp_state_file.exists(), \
        "MODULE_SHUTDOWN should persist sovereignty_state.json"
    data = json.loads(tmp_state_file.read_text())
    # great_cycle was incremented to 1 above.
    assert data.get("great_cycle") == 1


# ── Mixed dispatch (integration) ───────────────────────────────────


def test_mixed_dispatch_integration(tmp_state_file):
    """End-to-end: all 3 event types interleaved produce correct final state."""
    h = _WorkerHarness()
    h.start()
    try:
        h.wait_for_ready(timeout=5.0)
        # SOVEREIGNTY_EPOCH with great pulse
        h.send_event(SOVEREIGNTY_EPOCH, {
            "epoch_id": 100,
            "neuromods": {"DA": 0.6, "GABA": 0.2},
            "dev_age": 500,
            "great_pulse_fired": True,
        })
        time.sleep(0.05)
        # SOVEREIGNTY_CONFIRM_MAKER
        h.send_event(SOVEREIGNTY_CONFIRM_MAKER, {
            "tx_signature": "tx_integration_test",
            "ts": time.time(),
        })
        time.sleep(0.05)
        # Another epoch
        h.send_event(SOVEREIGNTY_EPOCH, {
            "epoch_id": 110,
            "neuromods": {"DA": 0.55},
            "dev_age": 550,
            "great_pulse_fired": True,
        })
        time.sleep(0.05)
        # SOVEREIGNTY_INCREMENT_GREAT_CYCLE
        h.send_event(SOVEREIGNTY_INCREMENT_GREAT_CYCLE, {
            "ts": time.time(),
            "source": "test_integration",
        })
        time.sleep(0.3)
    finally:
        h.shutdown()

    data = json.loads(tmp_state_file.read_text())
    assert data.get("total_great_pulses") == 2
    assert data.get("maker_confirmed") is True
    assert data.get("great_cycle") == 1
    assert data.get("developmental_age") == 550


# ── 100-message criteria snapshot smoke ────────────────────────────


def test_criteria_snapshot_logged_at_100_events(tmp_state_file, caplog):
    """Worker logs criteria snapshot every 100 SOVEREIGNTY_EPOCH events
    (smoke test — verifies the log path doesn't crash with realistic
    payloads + large number of events)."""
    import logging
    caplog.set_level(logging.INFO,
                     logger="titan_hcl.modules.sovereignty_worker")
    h = _WorkerHarness()
    h.start()
    try:
        h.wait_for_ready(timeout=5.0)
        # Send 100 epochs to trigger the snapshot path.
        for epoch_id in range(1, 101):
            h.send_event(SOVEREIGNTY_EPOCH, {
                "epoch_id": epoch_id,
                "neuromods": {"DA": 0.5, "5HT": 0.5, "NE": 0.5,
                              "ACh": 0.5, "Endorphin": 0.5, "GABA": 0.3},
                "dev_age": 100 + epoch_id,
                "great_pulse_fired": False,
            })
        time.sleep(1.0)  # generous drain
    finally:
        h.shutdown()

    # Verify at least one Criteria snapshot log line was emitted.
    snapshot_logs = [r for r in caplog.records
                     if "Criteria snapshot" in r.getMessage()]
    assert len(snapshot_logs) >= 1, (
        f"expected ≥1 'Criteria snapshot' log; got {len(snapshot_logs)} "
        f"(captured {len(caplog.records)} records total)")


# ── Robust payload handling ────────────────────────────────────────


def test_malformed_payload_does_not_crash(tmp_state_file):
    """Malformed payloads (missing keys, wrong types) don't crash the worker."""
    h = _WorkerHarness()
    h.start()
    try:
        h.wait_for_ready(timeout=5.0)
        # Empty payload
        h.send_event(SOVEREIGNTY_EPOCH, {})
        # Missing neuromods
        h.send_event(SOVEREIGNTY_EPOCH, {"epoch_id": 1, "dev_age": 100})
        # Non-dict neuromods
        h.send_event(SOVEREIGNTY_EPOCH, {
            "epoch_id": 2, "neuromods": None, "dev_age": 100})
        # CONFIRM_MAKER with missing tx_signature
        h.send_event(SOVEREIGNTY_CONFIRM_MAKER, {"ts": time.time()})
        # INCREMENT_GREAT_CYCLE with missing source
        h.send_event(SOVEREIGNTY_INCREMENT_GREAT_CYCLE, {"ts": time.time()})
        time.sleep(0.3)
    finally:
        # Worker should still be alive and shut down cleanly.
        h.shutdown()

    # State file should exist (shutdown persisted it).
    assert tmp_state_file.exists()
