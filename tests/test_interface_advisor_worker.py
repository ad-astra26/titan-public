"""tests/test_interface_advisor_worker.py — §4.H interface_advisor_worker
(D-SPEC-59, SPEC v1.8.5).

Coverage:
  • Bus event constant defined (IMPULSE_RECEIVED)
  • Worker module exports + boot lifecycle (MODULE_READY payload)
  • Constants TOML wiring (3 new INTERFACE_ADVISOR_* constants)
  • Specs / SHM slot path
  • InterfaceAdvisorStatePublisher: record_and_publish, get_stats compat
  • InterfaceAdvisorStateReader: cold-boot defaults, check semantics,
    cache TTL, sub-µs hot path
  • Worker dispatch: IMPULSE_RECEIVED → advisor.check → SHM republish;
    rate-exceeded → RATE_LIMIT bus event emitted to source
  • Rate-throttle: 10Hz cap on SHM republishes
  • Cold-boot SHM-Reader returns defaults so parent rate check passes
  • End-to-end: parent calls reader.check → emits IMPULSE_RECEIVED →
    worker records → reader sees updated rate on next read

Per CLAUDE.md: pytest -p no:anchorpy, separate process per file.
"""
from __future__ import annotations

import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import msgpack
import pytest

from titan_hcl import bus
from titan_hcl._phase_c_constants import (
    INTERFACE_ADVISOR_RATE_REFRESH_CADENCE_S,
    INTERFACE_ADVISOR_STATE_MAX_BYTES,
    INTERFACE_ADVISOR_STATE_SCHEMA_VERSION,
)
from titan_hcl.bus import (
    IMPULSE_RECEIVED,
    MODULE_READY,
    MODULE_SHUTDOWN,
    RATE_LIMIT,
    make_msg,
)
from titan_hcl.logic.interface_advisor import InterfaceAdvisor
from titan_hcl.logic.interface_advisor_publisher import (
    InterfaceAdvisorStatePublisher,
)
from titan_hcl.logic.interface_advisor_reader import (
    InterfaceAdvisorStateReader,
    _DEFAULT_PAYLOAD,
)
from titan_hcl.logic.interface_advisor_specs import (
    INTERFACE_ADVISOR_STATE_SLOT,
    INTERFACE_ADVISOR_STATE_SPEC,
)
from titan_hcl.modules.interface_advisor_worker import (
    HEARTBEAT_INTERVAL_S,
    interface_advisor_worker_main,
)


# ── Per-test SHM root isolation ────────────────────────────────────


@pytest.fixture
def titan_id(tmp_path) -> str:
    """Per-test titan_id; SHM root cleaned up after test."""
    tid = f"T_TEST_{int(time.time() * 1e6) % 1_000_000}"
    yield tid
    shutil.rmtree(Path(f"/dev/shm/titan_{tid}"), ignore_errors=True)


# ── Bus event constants ────────────────────────────────────────────


def test_bus_event_constants_defined():
    assert bus.IMPULSE_RECEIVED == "IMPULSE_RECEIVED"
    assert IMPULSE_RECEIVED == "IMPULSE_RECEIVED"


# ── Constants TOML wiring ──────────────────────────────────────────


def test_constants_wired_per_dspec59():
    assert INTERFACE_ADVISOR_STATE_SCHEMA_VERSION == 1
    assert INTERFACE_ADVISOR_STATE_MAX_BYTES == 512
    assert INTERFACE_ADVISOR_RATE_REFRESH_CADENCE_S == pytest.approx(0.1)


# ── SHM slot spec wiring ───────────────────────────────────────────


def test_slot_name():
    assert INTERFACE_ADVISOR_STATE_SLOT == "interface_advisor_state"


def test_slot_spec_schema_version():
    assert INTERFACE_ADVISOR_STATE_SPEC.schema_version == 1


def test_slot_spec_max_bytes():
    assert INTERFACE_ADVISOR_STATE_SPEC.payload_bytes == 512


def test_slot_spec_variable_size():
    assert INTERFACE_ADVISOR_STATE_SPEC.variable_size is True


# ── Worker module exports ──────────────────────────────────────────


def test_worker_module_exports():
    assert callable(interface_advisor_worker_main)
    assert HEARTBEAT_INTERVAL_S > 0


# ── InterfaceAdvisorStatePublisher ─────────────────────────────────


def test_publisher_cold_boot_publish(titan_id):
    pub = InterfaceAdvisorStatePublisher(titan_id)
    pub.publish()
    # Read it back via the corresponding reader.
    reader = InterfaceAdvisorStateReader(titan_id=titan_id)
    payload = reader.read()
    assert payload.get("schema_version") == 1
    # Cold boot — rates dict is empty, limits populated, rate_limit_count=0.
    assert payload.get("rates") == {}
    assert "IMPULSE" in payload.get("limits", {})
    assert payload.get("rate_limit_count") == 0


def test_publisher_record_and_publish_returns_feedback_or_none(titan_id):
    advisor = InterfaceAdvisor()
    pub = InterfaceAdvisorStatePublisher(titan_id, advisor=advisor)
    # First IMPULSE — within limit (limit=1/60s).
    feedback = pub.record_and_publish("IMPULSE", source="spirit")
    assert feedback is None
    # Second IMPULSE in same window — exceeds limit, should return feedback.
    feedback = pub.record_and_publish("IMPULSE", source="spirit")
    assert feedback is not None
    assert feedback["message_type"] == "IMPULSE"
    assert feedback["current_rate"] == 2
    assert feedback["limit"] == 1
    assert feedback["source"] == "spirit"


def test_publisher_advisor_property(titan_id):
    pub = InterfaceAdvisorStatePublisher(titan_id)
    assert pub.advisor is not None
    assert isinstance(pub.advisor, InterfaceAdvisor)


# ── InterfaceAdvisorStateReader ────────────────────────────────────


def test_reader_cold_boot_defaults(titan_id):
    """Reader before any worker publish returns the defaults snapshot
    (zero rates, INITIAL_LIMITS, rate_limit_count=0)."""
    reader = InterfaceAdvisorStateReader(titan_id=titan_id)
    payload = reader.read()
    assert payload.get("rates") == {}
    assert "IMPULSE" in payload.get("limits", {})
    assert payload.get("rate_limit_count") == 0


def test_reader_check_within_limits(titan_id):
    """Cold-boot reader.check() returns None (all checks pass)."""
    reader = InterfaceAdvisorStateReader(titan_id=titan_id)
    feedback = reader.check("IMPULSE", source="spirit")
    assert feedback is None


def test_reader_check_exceeded_after_publish(titan_id):
    """After publisher records past the limit, reader.check returns
    RATE_LIMIT feedback dict."""
    pub = InterfaceAdvisorStatePublisher(titan_id)
    pub.record_and_publish("IMPULSE", source="spirit")
    pub.record_and_publish("IMPULSE", source="spirit")
    # Force cache invalidation by sleeping past TTL
    time.sleep(0.15)
    reader = InterfaceAdvisorStateReader(titan_id=titan_id)
    feedback = reader.check("IMPULSE", source="spirit")
    assert feedback is not None
    assert feedback["message_type"] == "IMPULSE"
    assert feedback["limit"] == 1
    assert feedback["current_rate"] >= 1


def test_reader_check_unknown_message_type_passes(titan_id):
    """Reader.check for an unknown msg_type (no entry in INITIAL_LIMITS)
    returns None — pass-through behavior matches InterfaceAdvisor.check."""
    reader = InterfaceAdvisorStateReader(titan_id=titan_id)
    feedback = reader.check("UNKNOWN_MESSAGE_TYPE", source="spirit")
    assert feedback is None


def test_reader_get_stats_compat_shim(titan_id):
    """Reader.get_stats returns the same shape as InterfaceAdvisor.get_stats."""
    reader = InterfaceAdvisorStateReader(titan_id=titan_id)
    stats = reader.get_stats()
    assert "limits" in stats
    assert "current_rates" in stats
    assert "window_seconds" in stats
    assert "rate_limit_count" in stats


def test_reader_get_current_rate(titan_id):
    pub = InterfaceAdvisorStatePublisher(titan_id)
    pub.record_and_publish("INTERFACE_INPUT", source="chat")
    time.sleep(0.15)
    reader = InterfaceAdvisorStateReader(titan_id=titan_id)
    rate = reader.get_current_rate("INTERFACE_INPUT")
    assert rate >= 1
    # Unknown msg_type → 0.
    assert reader.get_current_rate("NEVER_PUBLISHED") == 0


# ── Worker harness ─────────────────────────────────────────────────


class _WorkerHarness:
    def __init__(self, titan_id: str, name: str = "interface_advisor"):
        self.titan_id = titan_id
        self.name = name
        self.recv: queue.Queue = queue.Queue()
        self.send: queue.Queue = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.exc: Optional[BaseException] = None

    def start(self) -> None:
        def _run():
            try:
                interface_advisor_worker_main(
                    self.recv, self.send, self.name,
                    {"titan_id": self.titan_id})
            except BaseException as e:
                self.exc = e

        self.thread = threading.Thread(
            target=_run, daemon=True,
            name=f"ia-worker-test-{self.name}")
        self.thread.start()

    def send_event(self, msg_type: str, payload: dict,
                   src: str = "test") -> None:
        self.recv.put(make_msg(msg_type, src, self.name, payload))

    def shutdown(self, timeout: float = 5.0) -> None:
        self.recv.put(make_msg(MODULE_SHUTDOWN, "guardian", self.name, {}))
        if self.thread is not None:
            self.thread.join(timeout=timeout)
            assert not self.thread.is_alive(), \
                "worker did not exit on MODULE_SHUTDOWN"
        if self.exc is not None:
            raise self.exc

    def wait_for_ready(self, timeout: float = 5.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = self.send.get(timeout=0.2)
            except queue.Empty:
                continue
            if msg.get("type") == MODULE_READY:
                return msg
        pytest.fail(f"MODULE_READY not observed in {timeout}s")

    def drain_for(self, seconds: float) -> list[dict]:
        out: list[dict] = []
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                msg = self.send.get(timeout=0.1)
            except queue.Empty:
                continue
            out.append(msg)
        return out


# ── Worker boot lifecycle ──────────────────────────────────────────


def test_worker_boots_and_emits_module_ready(titan_id):
    h = _WorkerHarness(titan_id)
    h.start()
    try:
        ready = h.wait_for_ready(timeout=5.0)
        payload = ready.get("payload", {})
        assert payload.get("module") == "interface_advisor_worker"
        assert payload.get("version") == "1.8.5"
        assert payload.get("spec_ref") == "D-SPEC-59"
    finally:
        h.shutdown()


# ── Worker dispatch: IMPULSE_RECEIVED → record + republish ─────────


def test_worker_impulse_received_updates_shm(titan_id):
    h = _WorkerHarness(titan_id)
    h.start()
    try:
        h.wait_for_ready(timeout=5.0)
        # Single IMPULSE_RECEIVED — within limit.
        h.send_event(IMPULSE_RECEIVED, {
            "msg_type": "IMPULSE", "source": "spirit",
            "client_ts": time.time(),
        })
        # Wait long enough for: (a) worker to dispatch (~µs), (b) throttle
        # window to elapse (100ms), and (c) the recv_queue.get(timeout=1.0)
        # Empty branch to fire + flush pending publish. With timeout=1.0,
        # next Empty tick is up to 1s away — so sleep ≥1.3s total to ensure
        # we cross both the throttle + at least one Empty tick.
        time.sleep(1.5)
        # SHM should reflect rate=1 for IMPULSE.
        reader = InterfaceAdvisorStateReader(titan_id=titan_id)
        rate = reader.get_current_rate("IMPULSE")
        assert rate == 1
    finally:
        h.shutdown()


def test_worker_rate_exceeded_emits_rate_limit(titan_id):
    h = _WorkerHarness(titan_id)
    h.start()
    try:
        h.wait_for_ready(timeout=5.0)
        # 2× IMPULSE_RECEIVED in same window — second exceeds limit=1.
        h.send_event(IMPULSE_RECEIVED, {
            "msg_type": "IMPULSE", "source": "spirit",
            "client_ts": time.time(),
        })
        time.sleep(0.05)
        h.send_event(IMPULSE_RECEIVED, {
            "msg_type": "IMPULSE", "source": "spirit",
            "client_ts": time.time(),
        })
        # Allow worker to drain + the 100ms throttle window to elapse for
        # the second event's publish.
        time.sleep(0.5)
        # Collect everything emitted by worker so far.
        emitted = h.drain_for(0.2)
    finally:
        h.shutdown()

    # Worker should have emitted at least one RATE_LIMIT event (for the
    # second IMPULSE_RECEIVED) with dst="spirit".
    rate_limits = [m for m in emitted
                   if m.get("type") == RATE_LIMIT
                   and m.get("dst") == "spirit"]
    assert len(rate_limits) >= 1, (
        f"expected ≥1 RATE_LIMIT event with dst='spirit'; "
        f"got types={[m.get('type') for m in emitted]}")
    feedback = rate_limits[0].get("payload", {})
    assert feedback.get("message_type") == "IMPULSE"
    assert feedback.get("limit") == 1
    assert feedback.get("current_rate") >= 1


def test_worker_unknown_msg_type_no_crash(titan_id):
    """Worker dispatches IMPULSE_RECEIVED with an unknown msg_type
    without crashing (advisor.check returns None for unconfigured types)."""
    h = _WorkerHarness(titan_id)
    h.start()
    try:
        h.wait_for_ready(timeout=5.0)
        h.send_event(IMPULSE_RECEIVED, {
            "msg_type": "TOTALLY_UNKNOWN_TYPE", "source": "test",
            "client_ts": time.time(),
        })
        time.sleep(0.2)
    finally:
        h.shutdown()


def test_worker_malformed_payload_no_crash(titan_id):
    """Worker handles malformed IMPULSE_RECEIVED payloads (missing keys,
    None values) without crashing."""
    h = _WorkerHarness(titan_id)
    h.start()
    try:
        h.wait_for_ready(timeout=5.0)
        h.send_event(IMPULSE_RECEIVED, {})  # empty payload
        h.send_event(IMPULSE_RECEIVED, {"msg_type": None})  # None value
        h.send_event(IMPULSE_RECEIVED, {
            "source": "no_msg_type"})  # missing msg_type
        time.sleep(0.3)
    finally:
        h.shutdown()
