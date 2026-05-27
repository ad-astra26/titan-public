"""Tests for Phase 11 Chunk 11D: module-state SHM contract + probe RPC.

Per RFP_phase_c_enhancements.md §3H.2 chunk 11D verification:
  SHM-write → titan_hcl-poll-detection → probe-RPC → SHM-RUNNING happy path
  probe-fails → 3-state escalation per locked D4

Per SPEC §11.I.2 / §11.I.3 / §11.I.5 / §11.I.8 (D-SPEC-141 / v1.65.0).
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time

import pytest

from titan_hcl.bus import (
    MODULE_PROBE_REQUEST,
    MODULE_PROBE_RESPONSE,
)
from titan_hcl.core.module_state import (
    BootPriority,
    ModuleStateEntry,
    ModuleStateReader,
    ModuleStateReaderBank,
    ModuleStateWriter,
    ProbeResult,
)
from titan_hcl.core.probe_dispatcher import (
    PROBE_TIMEOUT_S,
    ProbeDispatcher,
    handle_module_probe_request,
)
from titan_hcl.errors import ModuleError, ModuleErrorCode, Severity
from titan_hcl.guardian_hcl.module_registry import ModuleSpec, ModuleState


@pytest.fixture
def shm_root(monkeypatch):
    """Use a tempdir as TITAN_SHM_ROOT so tests don't collide with running Titans."""
    with tempfile.TemporaryDirectory(prefix="titan_phase11_test_") as td:
        monkeypatch.setenv("TITAN_SHM_ROOT", td)
        yield td


# ─── ModuleState enum: BOOTED + PROBING added ─────────────────────────────────

class TestModuleStateEnumExtension:
    def test_booted_and_probing_present(self):
        assert ModuleState.BOOTED.value == "booted"
        assert ModuleState.PROBING.value == "probing"

    def test_existing_states_preserved(self):
        # Regression: Chunk 11D additions must not remove pre-existing states.
        for name in ("STOPPED", "STARTING", "RUNNING", "UNHEALTHY", "CRASHED", "DISABLED"):
            assert hasattr(ModuleState, name)


# ─── ModuleSpec: probe_fn + boot_priority fields added ────────────────────────

class TestModuleSpecExtension:
    def test_defaults(self):
        spec = ModuleSpec(name="x", entry_fn=lambda: None)
        assert spec.probe_fn is None
        assert spec.boot_priority == "mandatory"

    def test_custom_probe_fn(self):
        def my_probe(bus):
            return ProbeResult.ok_()
        spec = ModuleSpec(name="x", entry_fn=lambda: None, probe_fn=my_probe)
        assert spec.probe_fn is my_probe

    def test_custom_boot_priority(self):
        spec = ModuleSpec(name="x", entry_fn=lambda: None,
                          boot_priority=BootPriority.OPTIONAL_POST_BOOT.value)
        assert spec.boot_priority == "post_boot"


# ─── BootPriority enum ────────────────────────────────────────────────────────

class TestBootPriorityEnum:
    def test_values(self):
        assert BootPriority.MANDATORY.value == "mandatory"
        assert BootPriority.OPTIONAL_POST_BOOT.value == "post_boot"
        assert BootPriority.LAZY.value == "lazy"

    def test_is_str_subclass(self):
        assert isinstance(BootPriority.MANDATORY, str)


# ─── ProbeResult ──────────────────────────────────────────────────────────────

class TestProbeResult:
    def test_ok_constructor(self):
        r = ProbeResult.ok_(latency_ms=42.0)
        assert r.ok is True
        assert r.latency_ms == 42.0
        assert r.error_envelope is None

    def test_fail_constructor(self):
        err = ModuleError(
            module_name="m", subsystem="s", error_code="C",
            severity=Severity.ERROR, message="m",
        )
        r = ProbeResult.fail(error=err, latency_ms=1500.0)
        assert r.ok is False
        assert r.error_envelope is err

    def test_round_trip(self):
        err = ModuleError(
            module_name="m", subsystem="s", error_code="PROBE_FAILED",
            severity=Severity.ERROR, message="probe broken",
        )
        r1 = ProbeResult.fail(error=err, latency_ms=1234.5)
        r2 = ProbeResult.from_wire_dict(r1.as_wire_dict())
        assert r2.ok == r1.ok
        assert r2.latency_ms == r1.latency_ms
        assert r2.error_envelope is not None
        assert r2.error_envelope.error_code == "PROBE_FAILED"


# ─── ModuleStateEntry round-trip ──────────────────────────────────────────────

class TestModuleStateEntryWire:
    def test_round_trip_minimal(self):
        e = ModuleStateEntry(
            name="agno_worker", layer="L2",
            boot_priority=BootPriority.MANDATORY, state="running", pid=1234,
        )
        e2 = ModuleStateEntry.from_wire_dict(e.as_wire_dict())
        assert e2.name == "agno_worker"
        assert e2.layer == "L2"
        assert e2.boot_priority is BootPriority.MANDATORY
        assert e2.state == "running"
        assert e2.pid == 1234

    def test_round_trip_with_full_envelope(self):
        err = ModuleError(
            module_name="agno_worker", subsystem="ovg.warmup",
            error_code="OVG_TIMECHAIN_OPEN_FAILED",
            severity=Severity.FATAL, message="ovg open failed",
            detail="full traceback here",
            suggested_remediation="restart with --rebuild-cache",
        )
        probe = ProbeResult.fail(error=err, latency_ms=2400.0)
        e = ModuleStateEntry(
            name="agno_worker", layer="L2",
            boot_priority=BootPriority.MANDATORY, state="unhealthy",
            pid=999, started_at=1000.0, booted_at=1010.0, running_at=0.0,
            last_heartbeat=1015.0, last_probe_result=probe, last_error=err,
            restart_count=3, error_count_24h=12,
        )
        e2 = ModuleStateEntry.from_wire_dict(e.as_wire_dict())
        assert e2.last_probe_result is not None
        assert e2.last_probe_result.ok is False
        assert e2.last_error is not None
        assert e2.last_error.error_code == "OVG_TIMECHAIN_OPEN_FAILED"
        assert e2.last_error.severity is Severity.FATAL


# ─── ModuleStateWriter ↔ Reader SHM round-trip ───────────────────────────────

class TestModuleStateShmRoundTrip:
    def test_writer_publishes_reader_observes(self, shm_root):
        writer = ModuleStateWriter(
            module_name="health_monitor", layer="L3",
            boot_priority=BootPriority.MANDATORY,
        )
        try:
            writer.write_state("starting")
            time.sleep(0.005)
            reader = ModuleStateReader(module_name="health_monitor")
            try:
                entry = reader.read()
                assert entry is not None
                assert entry.name == "health_monitor"
                assert entry.state == "starting"
                assert entry.layer == "L3"
                assert entry.boot_priority is BootPriority.MANDATORY
                assert entry.pid == os.getpid()
                assert entry.started_at > 0
                # Now publish a state change and verify the reader sees it.
                writer.write_state("booted")
                time.sleep(0.005)
                entry2 = reader.read()
                assert entry2 is not None
                assert entry2.state == "booted"
                assert entry2.booted_at > 0
            finally:
                reader.close()
        finally:
            writer.close()

    def test_writer_running_carries_probe_result(self, shm_root):
        writer = ModuleStateWriter(
            module_name="output_verifier", layer="L2",
            boot_priority=BootPriority.MANDATORY,
        )
        try:
            writer.write_state("starting")
            writer.write_state("booted")
            writer.write_state("probing")
            probe = ProbeResult.ok_(latency_ms=120.5)
            writer.write_state("running", last_probe_result=probe)
            reader = ModuleStateReader(module_name="output_verifier")
            try:
                entry = reader.read()
                assert entry is not None
                assert entry.state == "running"
                assert entry.running_at > 0
                assert entry.last_probe_result is not None
                assert entry.last_probe_result.ok is True
                assert entry.last_probe_result.latency_ms == pytest.approx(120.5)
            finally:
                reader.close()
        finally:
            writer.close()

    def test_writer_unhealthy_carries_error(self, shm_root):
        writer = ModuleStateWriter(
            module_name="memory_worker", layer="L2",
            boot_priority=BootPriority.MANDATORY,
        )
        try:
            err = ModuleError(
                module_name="memory_worker", subsystem="faiss.warmup",
                error_code="MEMORY_INDEX_NOT_READY",
                severity=Severity.WARN, message="warmup taking too long",
            )
            probe = ProbeResult.fail(error=err, latency_ms=2400.0)
            writer.write_state("unhealthy", last_probe_result=probe, last_error=err)
            reader = ModuleStateReader(module_name="memory_worker")
            try:
                entry = reader.read()
                assert entry is not None
                assert entry.state == "unhealthy"
                assert entry.last_error is not None
                assert entry.last_error.error_code == "MEMORY_INDEX_NOT_READY"
            finally:
                reader.close()
        finally:
            writer.close()

    def test_heartbeat_updates_last_heartbeat(self, shm_root):
        writer = ModuleStateWriter(
            module_name="cognitive_worker", layer="L2",
            boot_priority=BootPriority.MANDATORY,
        )
        try:
            writer.write_state("starting")
            writer.write_state("booted")
            writer.write_state("probing")
            writer.write_state("running")
            time.sleep(0.01)
            writer.heartbeat()
            reader = ModuleStateReader(module_name="cognitive_worker")
            try:
                entry = reader.read()
                assert entry is not None
                assert entry.state == "running"
                assert entry.last_heartbeat > 0
            finally:
                reader.close()
        finally:
            writer.close()


# ─── ModuleStateReaderBank ────────────────────────────────────────────────────

class TestModuleStateReaderBank:
    def test_lazy_reader_creation_and_read_all(self, shm_root):
        # Create writers for two modules.
        writer_a = ModuleStateWriter(
            module_name="bank_test_a", layer="L2",
            boot_priority=BootPriority.MANDATORY,
        )
        writer_b = ModuleStateWriter(
            module_name="bank_test_b", layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        try:
            writer_a.write_state("running")
            writer_b.write_state("starting")
            bank = ModuleStateReaderBank()
            try:
                snap = bank.read_all(["bank_test_a", "bank_test_b", "bank_test_missing"])
                assert snap["bank_test_a"] is not None
                assert snap["bank_test_a"].state == "running"
                assert snap["bank_test_b"] is not None
                assert snap["bank_test_b"].state == "starting"
                assert snap["bank_test_b"].boot_priority is BootPriority.OPTIONAL_POST_BOOT
                # Missing module returns None (no SHM file).
                assert snap["bank_test_missing"] is None
            finally:
                bank.close()
        finally:
            writer_a.close()
            writer_b.close()


# ─── Worker-side: handle_module_probe_request ────────────────────────────────

class _FakeSendQueue:
    def __init__(self):
        self.sent: list[dict] = []
    def put_nowait(self, msg: dict) -> None:
        self.sent.append(msg)


class TestHandleModuleProbeRequest:
    def test_trivial_pass_when_probe_fn_none(self, shm_root):
        # Legacy worker: no probe_fn defined → trivial-pass per SPEC §11.I.2.
        writer = ModuleStateWriter(
            module_name="probe_legacy", layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        try:
            writer.write_state("starting")
            writer.write_state("booted")
            q = _FakeSendQueue()
            msg = {
                "type": MODULE_PROBE_REQUEST, "src": "titan_hcl",
                "dst": "probe_legacy", "rid": "probe-1",
                "payload": {"name": "probe_legacy", "probe_id": "probe-1"},
            }
            result = handle_module_probe_request(
                msg, probe_fn=None, send_queue=q,
                module_name="probe_legacy", state_writer=writer,
            )
            assert result.ok is True
            # Response sent
            assert len(q.sent) == 1
            response = q.sent[0]
            assert response["type"] == MODULE_PROBE_RESPONSE
            assert response["rid"] == "probe-1"
            assert response["payload"]["result"]["ok"] is True
            # SHM transitioned to running
            reader = ModuleStateReader(module_name="probe_legacy")
            try:
                entry = reader.read()
                assert entry is not None
                assert entry.state == "running"
            finally:
                reader.close()
        finally:
            writer.close()

    def test_probe_fn_returning_ok_result(self, shm_root):
        writer = ModuleStateWriter(
            module_name="probe_real", layer="L2",
            boot_priority=BootPriority.MANDATORY,
        )
        try:
            writer.write_state("booted")
            q = _FakeSendQueue()
            def my_probe(bus):
                return ProbeResult.ok_()
            msg = {
                "type": MODULE_PROBE_REQUEST, "src": "titan_hcl",
                "dst": "probe_real", "rid": "probe-2",
                "payload": {"probe_id": "probe-2"},
            }
            result = handle_module_probe_request(
                msg, probe_fn=my_probe, send_queue=q,
                module_name="probe_real", state_writer=writer,
            )
            assert result.ok is True
            assert q.sent[0]["payload"]["result"]["ok"] is True
        finally:
            writer.close()

    def test_probe_fn_raising_caught_and_envelope_emitted(self, shm_root):
        writer = ModuleStateWriter(
            module_name="probe_raise", layer="L2",
            boot_priority=BootPriority.MANDATORY,
        )
        try:
            writer.write_state("booted")
            q = _FakeSendQueue()
            def bad_probe(bus):
                raise RuntimeError("probe blew up")
            msg = {
                "type": MODULE_PROBE_REQUEST, "src": "titan_hcl",
                "dst": "probe_raise", "rid": "probe-3",
                "payload": {"probe_id": "probe-3"},
            }
            result = handle_module_probe_request(
                msg, probe_fn=bad_probe, send_queue=q,
                module_name="probe_raise", state_writer=writer,
            )
            assert result.ok is False
            assert result.error_envelope is not None
            assert result.error_envelope.error_code == "PROBE_FAILED"
            assert "probe blew up" in result.error_envelope.detail
            # State is UNHEALTHY
            reader = ModuleStateReader(module_name="probe_raise")
            try:
                entry = reader.read()
                assert entry is not None
                assert entry.state == "unhealthy"
                assert entry.last_error is not None
                assert entry.last_error.error_code == "PROBE_FAILED"
            finally:
                reader.close()
        finally:
            writer.close()

    def test_probe_fn_returning_wrong_type_treated_as_failed(self, shm_root):
        writer = ModuleStateWriter(
            module_name="probe_wrongtype", layer="L2",
            boot_priority=BootPriority.MANDATORY,
        )
        try:
            writer.write_state("booted")
            q = _FakeSendQueue()
            def weird_probe(bus):
                return "I'm a string, not a ProbeResult"
            msg = {
                "type": MODULE_PROBE_REQUEST, "src": "titan_hcl",
                "dst": "probe_wrongtype", "rid": "probe-4",
                "payload": {"probe_id": "probe-4"},
            }
            result = handle_module_probe_request(
                msg, probe_fn=weird_probe, send_queue=q,
                module_name="probe_wrongtype", state_writer=writer,
            )
            assert result.ok is False
            assert result.error_envelope is not None
        finally:
            writer.close()


# ─── Orchestrator-side: ProbeDispatcher ──────────────────────────────────────

class _FakeBusOk:
    """Bus that always replies with a successful ProbeResult."""
    def __init__(self):
        self.requested: list[dict] = []
    async def request_async(self, src, dst, payload, timeout=10.0):
        self.requested.append({"src": src, "dst": dst, "payload": payload, "timeout": timeout})
        return {
            "type": MODULE_PROBE_RESPONSE,
            "src": dst, "dst": src,
            "rid": payload["rid"],
            "ts": time.time(),
            "payload": {
                "probe_id": payload["rid"],
                "name": dst,
                "result": ProbeResult.ok_(latency_ms=42.0).as_wire_dict(),
            },
        }


class _FakeBusTimeout:
    """Bus that always times out the RPC."""
    async def request_async(self, src, dst, payload, timeout=10.0):
        await asyncio.sleep(timeout + 1)
        return None


class _FakeBusFail:
    """Bus that replies with a failed ProbeResult."""
    async def request_async(self, src, dst, payload, timeout=10.0):
        err = ModuleError(
            module_name=dst, subsystem="probe",
            error_code="PROBE_FAILED",
            severity=Severity.ERROR, message="probe failed inside worker",
        )
        return {
            "type": MODULE_PROBE_RESPONSE,
            "src": dst, "dst": src,
            "rid": payload["rid"],
            "ts": time.time(),
            "payload": {
                "probe_id": payload["rid"],
                "name": dst,
                "result": ProbeResult.fail(error=err, latency_ms=1500.0).as_wire_dict(),
            },
        }


class TestProbeDispatcher:
    def test_dispatch_happy_path(self):
        bus = _FakeBusOk()
        dispatcher = ProbeDispatcher(bus)
        result = asyncio.run(dispatcher.dispatch_probe("agno_worker"))
        assert result.ok is True
        # Verify the request was a proper MODULE_PROBE_REQUEST.
        assert len(bus.requested) == 1
        req = bus.requested[0]
        assert req["src"] == "titan_hcl"
        assert req["dst"] == "agno_worker"
        assert req["payload"]["type"] == MODULE_PROBE_REQUEST

    def test_dispatch_failed_probe(self):
        bus = _FakeBusFail()
        dispatcher = ProbeDispatcher(bus)
        result = asyncio.run(dispatcher.dispatch_probe("memory_worker"))
        assert result.ok is False
        assert result.error_envelope is not None
        assert result.error_envelope.error_code == "PROBE_FAILED"

    def test_dispatch_timeout(self):
        bus = _FakeBusTimeout()
        dispatcher = ProbeDispatcher(bus)
        result = asyncio.run(dispatcher.dispatch_probe("agno_worker", timeout_s=0.1))
        assert result.ok is False
        assert result.error_envelope is not None
        assert result.error_envelope.error_code == "PROBE_TIMEOUT"
