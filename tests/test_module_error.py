"""Tests for Phase 11 Chunk 11B: ModuleError envelope + MODULE_ERROR bus topic.

Per RFP_phase_c_enhancements.md §3H.2 chunk 11B verification:
  - Envelope serializes/deserializes
  - publish ≤10µs p99 (smoke-checked here; full perf in load suite)

Per SPEC §11.I.4 (D-SPEC-141):
  - Single structured-error path between workers/orchestrator/supervisor
  - Severity enum drives guardian_hcl restart-strategy selector
  - Length limits enforced (message ≤200, detail ≤1024, traceback ≤10 frames)
  - Rate-gated to 100/s per (module_name, error_code); MODULE_ERROR_FLOOD on excess
"""
from __future__ import annotations

import time

import pytest

from titan_hcl import bus as bus_module
from titan_hcl.bus import (
    MODULE_ERROR,
    MODULE_ERROR_FLOOD,
    MODULE_PROBE_REQUEST,
    MODULE_PROBE_RESPONSE,
    publish_module_error,
)
from titan_hcl.errors import (
    DETAIL_MAX_LEN,
    MESSAGE_MAX_LEN,
    TRACEBACK_MAX_FRAMES,
    ModuleError,
    ModuleErrorCode,
    Severity,
)


# ─── Phase 11 bus-constant presence ───────────────────────────────────────────

class TestPhase11BusConstants:
    """Chunks 11B + 11D add three new bus topics. Verify the strings are
    importable + match the SPEC §11.I wire contract."""

    def test_module_error_topic_string(self):
        assert MODULE_ERROR == "MODULE_ERROR"
        assert MODULE_ERROR_FLOOD == "MODULE_ERROR_FLOOD"

    def test_module_probe_topic_strings(self):
        assert MODULE_PROBE_REQUEST == "MODULE_PROBE_REQUEST"
        assert MODULE_PROBE_RESPONSE == "MODULE_PROBE_RESPONSE"


# ─── Severity / ModuleErrorCode enums ─────────────────────────────────────────

class TestSeverityEnum:
    def test_severity_values(self):
        # Order is significant for restart-strategy selector; document via test.
        assert Severity.DEBUG.value == "DEBUG"
        assert Severity.INFO.value == "INFO"
        assert Severity.WARN.value == "WARN"
        assert Severity.ERROR.value == "ERROR"
        assert Severity.FATAL.value == "FATAL"

    def test_severity_is_str_subclass(self):
        # Required for clean msgpack serialization without custom encoder.
        assert isinstance(Severity.FATAL, str)
        assert Severity.FATAL == "FATAL"


class TestModuleErrorCodeRegistry:
    def test_canonical_codes_present(self):
        # Spot-check the codes the SPEC §11.I.4 examples use.
        assert ModuleErrorCode.UNCAUGHT_EXCEPTION.value == "UNCAUGHT_EXCEPTION"
        assert ModuleErrorCode.PROBE_TIMEOUT.value == "PROBE_TIMEOUT"
        assert ModuleErrorCode.OVG_TIMECHAIN_OPEN_FAILED.value == "OVG_TIMECHAIN_OPEN_FAILED"

    def test_codes_accept_raw_strings(self):
        # Per SPEC: error_code field is `str`, callers may pass raw strings
        # for codes not yet registered. The enum is a registry, not a constraint.
        err = ModuleError(
            module_name="custom_worker",
            subsystem="unregistered.path",
            error_code="WHOLLY_NEW_CODE_NOT_IN_ENUM",
            severity=Severity.ERROR,
            message="raw-string code accepted",
        )
        assert err.error_code == "WHOLLY_NEW_CODE_NOT_IN_ENUM"


# ─── ModuleError dataclass ────────────────────────────────────────────────────

class TestModuleErrorConstruction:
    def test_minimal_envelope(self):
        err = ModuleError(
            module_name="agno_worker",
            subsystem="boot",
            error_code=str(ModuleErrorCode.BOOT_TIMEOUT.value),
            severity=Severity.FATAL,
            message="agno_worker failed to boot within budget",
        )
        assert err.module_name == "agno_worker"
        assert err.severity is Severity.FATAL
        assert err.detail == ""
        assert err.traceback_top10 == []
        assert err.context == {}
        assert err.suggested_remediation is None
        assert err.ts > 0
        assert err.correlation_id is None

    def test_message_truncation_enforced(self):
        long_msg = "x" * 5000
        err = ModuleError(
            module_name="m", subsystem="s", error_code="C",
            severity=Severity.INFO, message=long_msg,
        )
        assert len(err.message) <= MESSAGE_MAX_LEN
        assert err.message.endswith("…[truncated]")

    def test_detail_truncation_enforced(self):
        long_detail = "y" * 5000
        err = ModuleError(
            module_name="m", subsystem="s", error_code="C",
            severity=Severity.INFO, message="m", detail=long_detail,
        )
        assert len(err.detail) <= DETAIL_MAX_LEN
        assert err.detail.endswith("…[truncated]")

    def test_traceback_frame_count_clamped(self):
        too_many = [f"frame {i}" for i in range(50)]
        err = ModuleError(
            module_name="m", subsystem="s", error_code="C",
            severity=Severity.INFO, message="m", traceback_top10=too_many,
        )
        assert len(err.traceback_top10) == TRACEBACK_MAX_FRAMES

    def test_frozen_dataclass_immutable(self):
        err = ModuleError(
            module_name="m", subsystem="s", error_code="C",
            severity=Severity.INFO, message="m",
        )
        with pytest.raises((AttributeError, Exception)):
            err.message = "different"  # type: ignore[misc]


class TestModuleErrorFromException:
    def test_captures_exception_message(self):
        try:
            raise ValueError("a real exception happened")
        except ValueError as e:
            err = ModuleError.from_exception(
                e, module_name="agno_worker", subsystem="llm.client",
            )
        assert "ValueError" in err.message or "a real exception" in err.message
        assert err.severity is Severity.ERROR  # default
        assert err.error_code == "UNCAUGHT_EXCEPTION"
        assert len(err.traceback_top10) >= 1
        # detail is the full formatted traceback
        assert "ValueError" in err.detail
        assert "a real exception happened" in err.detail

    def test_custom_severity_and_code(self):
        try:
            raise RuntimeError("ovg failed")
        except RuntimeError as e:
            err = ModuleError.from_exception(
                e,
                module_name="agno_worker",
                subsystem="ovg.warmup",
                error_code=ModuleErrorCode.OVG_TIMECHAIN_OPEN_FAILED,
                severity=Severity.FATAL,
                suggested_remediation="Check timechain fork-state cache integrity",
            )
        assert err.severity is Severity.FATAL
        assert err.error_code == "OVG_TIMECHAIN_OPEN_FAILED"
        assert err.suggested_remediation is not None


class TestWireSerialization:
    def test_round_trip_preserves_all_fields(self):
        original = ModuleError(
            module_name="memory_worker",
            subsystem="faiss.query",
            error_code="MEMORY_INDEX_NOT_READY",
            severity=Severity.WARN,
            message="zero-vector query on un-warmed index",
            detail="warmup elapsed 12.4s; expected ≤8s",
            traceback_top10=["frame A", "frame B"],
            context={"index_size": 0, "warmup_elapsed_s": 12.4},
            suggested_remediation="raise memory_warmup_timeout_s",
            correlation_id="probe-7c3",
        )
        wire = original.as_wire_dict()
        # Severity must be a plain string on the wire (msgpack compat).
        assert wire["severity"] == "WARN"
        assert isinstance(wire["severity"], str)
        # Round-trip:
        round_tripped = ModuleError.from_wire_dict(wire)
        assert round_tripped.module_name == original.module_name
        assert round_tripped.subsystem == original.subsystem
        assert round_tripped.error_code == original.error_code
        assert round_tripped.severity is Severity.WARN
        assert round_tripped.message == original.message
        assert round_tripped.detail == original.detail
        assert round_tripped.traceback_top10 == original.traceback_top10
        assert round_tripped.context == original.context
        assert round_tripped.suggested_remediation == original.suggested_remediation
        assert round_tripped.correlation_id == original.correlation_id

    def test_wire_dict_is_msgpack_safe(self):
        import msgpack
        err = ModuleError(
            module_name="m", subsystem="s", error_code="C",
            severity=Severity.ERROR, message="m",
        )
        packed = msgpack.packb(err.as_wire_dict(), use_bin_type=True)
        unpacked = msgpack.unpackb(packed, raw=False)
        restored = ModuleError.from_wire_dict(unpacked)
        assert restored.message == "m"
        assert restored.severity is Severity.ERROR


# ─── publish_module_error: duck-typed sender dispatch + rate-gate ────────────

class _FakeSendQueue:
    """Mimics multiprocessing.Queue interface used by worker subprocesses."""
    def __init__(self):
        self.sent: list[dict] = []
    def put_nowait(self, msg: dict) -> None:
        self.sent.append(msg)


class _FakeBus:
    """Mimics DivineBus.publish interface used in main-process callsites."""
    def __init__(self):
        self.published: list[dict] = []
    def publish(self, msg: dict) -> int:
        self.published.append(msg)
        return 1


def _reset_rate_gate():
    """Reset the per-(module, code) rate-gate state between tests."""
    bus_module._module_error_last_window_ts.clear()
    bus_module._module_error_window_count.clear()
    bus_module._module_error_flood_last_emit.clear()


class TestPublishModuleErrorDispatch:
    def setup_method(self):
        _reset_rate_gate()

    def test_publish_via_put_nowait_sender(self):
        q = _FakeSendQueue()
        err = ModuleError(
            module_name="agno_worker", subsystem="boot",
            error_code="BOOT_TIMEOUT", severity=Severity.FATAL,
            message="boot timed out",
        )
        ok = publish_module_error(q, err)
        assert ok is True
        assert len(q.sent) == 1
        msg = q.sent[0]
        assert msg["type"] == MODULE_ERROR
        assert msg["src"] == "agno_worker"
        assert msg["dst"] == "all"
        assert msg["payload"]["error_code"] == "BOOT_TIMEOUT"
        assert msg["payload"]["severity"] == "FATAL"

    def test_publish_via_bus_publish_sender(self):
        b = _FakeBus()
        err = ModuleError(
            module_name="memory_worker", subsystem="faiss",
            error_code="SHM_READ_FAILED", severity=Severity.ERROR,
            message="failed reading memory_state.bin",
            correlation_id="probe-99",
        )
        ok = publish_module_error(b, err)
        assert ok is True
        assert len(b.published) == 1
        msg = b.published[0]
        assert msg["type"] == MODULE_ERROR
        assert msg["rid"] == "probe-99"

    def test_invalid_sender_returns_false(self):
        class _Bad:
            pass
        err = ModuleError(
            module_name="m", subsystem="s", error_code="C",
            severity=Severity.INFO, message="m",
        )
        assert publish_module_error(_Bad(), err) is False

    def test_invalid_error_arg_returns_false(self):
        q = _FakeSendQueue()
        # Passing a raw dict instead of a ModuleError is a callsite bug.
        assert publish_module_error(q, {"not": "a moduleerror"}) is False
        assert len(q.sent) == 0


class TestPublishModuleErrorRateGate:
    def setup_method(self):
        _reset_rate_gate()

    def test_under_limit_all_emitted(self):
        q = _FakeSendQueue()
        for i in range(50):  # well under 100/s
            err = ModuleError(
                module_name="rate_test", subsystem="s",
                error_code="RATE_TEST_CODE", severity=Severity.ERROR,
                message=f"msg {i}",
            )
            assert publish_module_error(q, err) is True
        assert len(q.sent) == 50

    def test_over_limit_dropped_and_flood_notified(self):
        q = _FakeSendQueue()
        emitted = 0
        for i in range(150):  # 50 over the 100/window limit
            err = ModuleError(
                module_name="flood_test", subsystem="s",
                error_code="FLOOD_CODE", severity=Severity.ERROR,
                message=f"msg {i}",
            )
            if publish_module_error(q, err):
                emitted += 1
        assert emitted == 100  # exactly the limit
        # The MODULE_ERROR_FLOOD notification should appear at least once.
        flood_msgs = [m for m in q.sent if m["type"] == MODULE_ERROR_FLOOD]
        assert len(flood_msgs) >= 1
        flood = flood_msgs[0]
        assert flood["payload"]["module_name"] == "flood_test"
        assert flood["payload"]["error_code"] == "FLOOD_CODE"
        assert flood["payload"]["limit_per_window"] == 100

    def test_independent_modules_dont_share_quota(self):
        q = _FakeSendQueue()
        # 100 from module A — fills A's quota.
        for i in range(100):
            err = ModuleError(
                module_name="mod_a", subsystem="s",
                error_code="C", severity=Severity.ERROR, message="a",
            )
            publish_module_error(q, err)
        # 1 from module B — should still go through (its own quota).
        err_b = ModuleError(
            module_name="mod_b", subsystem="s",
            error_code="C", severity=Severity.ERROR, message="b",
        )
        assert publish_module_error(q, err_b) is True

    def test_window_rolls_after_interval(self):
        q = _FakeSendQueue()
        # Fill the window for a single (module, code) tuple.
        for i in range(100):
            err = ModuleError(
                module_name="window_test", subsystem="s",
                error_code="WIN_CODE", severity=Severity.ERROR, message="m",
            )
            publish_module_error(q, err)
        # 101st in the same window: dropped.
        over = ModuleError(
            module_name="window_test", subsystem="s",
            error_code="WIN_CODE", severity=Severity.ERROR, message="over",
        )
        assert publish_module_error(q, over) is False
        # Simulate window roll by advancing the stored window-start past the limit.
        key = ("window_test", "WIN_CODE")
        bus_module._module_error_last_window_ts[key] = time.time() - 2.0  # >1s ago
        bus_module._module_error_window_count[key] = 0
        # Now next emission should succeed.
        after = ModuleError(
            module_name="window_test", subsystem="s",
            error_code="WIN_CODE", severity=Severity.ERROR, message="after",
        )
        assert publish_module_error(q, after) is True


class TestPublishLatency:
    """Smoke check that publish itself is fast (target ≤10µs p99 per RFP §3H.2)."""

    def setup_method(self):
        _reset_rate_gate()

    def test_per_call_latency_smoke(self):
        q = _FakeSendQueue()
        err = ModuleError(
            module_name="perf_test", subsystem="s",
            error_code="PERF_CODE", severity=Severity.ERROR, message="m",
        )
        # One warmup, then measure 1000 calls — well under rate limit because
        # we reset the gate between each call to isolate publish-latency from
        # gate-drop behaviour.
        publish_module_error(q, err)
        N = 1000
        latencies = []
        for _ in range(N):
            _reset_rate_gate()
            t0 = time.perf_counter()
            publish_module_error(q, err)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
        latencies.sort()
        p99 = latencies[int(0.99 * N)]
        # Generous CI-safe ceiling — local typical is <5µs; we target ≤10µs
        # but bump to 500µs to absorb cold caches + CI noise without flakes.
        # If this fails consistently, investigate the publish path.
        assert p99 < 500e-6, f"p99 publish latency {p99*1e6:.1f}µs exceeds 500µs ceiling"
