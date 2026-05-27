"""Tests for Phase 11 Chunk 11C: `@with_error_envelope` decorator.

Per RFP_phase_c_enhancements.md §3H.2 chunk 11C verification:
  - Decorator preserves exception semantics
  - Envelope published before re-raise
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from titan_hcl import bus as bus_module
from titan_hcl.bus import MODULE_ERROR
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import ModuleError, ModuleErrorCode, Severity


def _reset_rate_gate():
    bus_module._module_error_last_window_ts.clear()
    bus_module._module_error_window_count.clear()
    bus_module._module_error_flood_last_emit.clear()


class _FakeSendQueue:
    def __init__(self):
        self.sent: list[dict] = []
    def put_nowait(self, msg: dict) -> None:
        self.sent.append(msg)


class _FakeBus:
    def __init__(self):
        self.published: list[dict] = []
    def publish(self, msg: dict) -> int:
        self.published.append(msg)
        return 1


# ─── Function-attribute preservation ──────────────────────────────────────────

class TestWrapsPreservation:
    def test_name_and_doc_preserved(self):
        @with_error_envelope(module_name="m", subsystem="s")
        def my_worker(recv_queue, send_queue):
            """My worker docstring."""
            return 1

        assert my_worker.__name__ == "my_worker"
        assert my_worker.__doc__ == "My worker docstring."

    def test_signature_preserved(self):
        @with_error_envelope(module_name="m", subsystem="s")
        def f(a, b, send_queue=None):
            return a + b

        sig = inspect.signature(f)
        assert list(sig.parameters) == ["a", "b", "send_queue"]

    def test_async_wrapped_returns_coroutine_function(self):
        @with_error_envelope(module_name="m", subsystem="s")
        async def af(send_queue):
            return 42

        assert inspect.iscoroutinefunction(af)


# ─── Happy-path: no exception, no envelope ────────────────────────────────────

class TestHappyPath:
    def setup_method(self):
        _reset_rate_gate()

    def test_sync_returns_value_no_envelope(self):
        q = _FakeSendQueue()

        @with_error_envelope(module_name="agno_worker", subsystem="boot")
        def fn(recv_queue, send_queue):
            return "all good"

        result = fn(None, q)
        assert result == "all good"
        assert q.sent == []  # no envelope on happy path

    def test_async_returns_value_no_envelope(self):
        q = _FakeSendQueue()

        @with_error_envelope(module_name="m", subsystem="s")
        async def afn(send_queue):
            return "async ok"

        result = asyncio.run(afn(q))
        assert result == "async ok"
        assert q.sent == []


# ─── Exception path: envelope before re-raise ─────────────────────────────────

class TestExceptionPath:
    def setup_method(self):
        _reset_rate_gate()

    def test_sync_publishes_envelope_then_reraises(self):
        q = _FakeSendQueue()

        @with_error_envelope(module_name="agno_worker", subsystem="ovg.warmup")
        def fn(recv_queue, send_queue):
            raise ValueError("OVG failed to open timechain")

        with pytest.raises(ValueError, match="OVG failed to open timechain"):
            fn(None, q)

        # Envelope was published before re-raise
        assert len(q.sent) == 1
        msg = q.sent[0]
        assert msg["type"] == MODULE_ERROR
        assert msg["src"] == "agno_worker"
        payload = msg["payload"]
        assert payload["module_name"] == "agno_worker"
        assert payload["subsystem"] == "ovg.warmup"
        assert payload["error_code"] == "UNCAUGHT_EXCEPTION"  # default
        assert payload["severity"] == "ERROR"  # default
        assert "ValueError" in payload["message"]
        assert "OVG failed to open timechain" in payload["detail"]
        assert len(payload["traceback_top10"]) >= 1

    def test_async_publishes_envelope_then_reraises(self):
        q = _FakeSendQueue()

        @with_error_envelope(module_name="memory_worker", subsystem="faiss")
        async def afn(send_queue):
            raise RuntimeError("FAISS index unavailable")

        with pytest.raises(RuntimeError, match="FAISS index unavailable"):
            asyncio.run(afn(q))

        assert len(q.sent) == 1
        payload = q.sent[0]["payload"]
        assert payload["module_name"] == "memory_worker"
        assert payload["subsystem"] == "faiss"

    def test_custom_severity_and_code_propagate_to_envelope(self):
        q = _FakeSendQueue()

        @with_error_envelope(
            module_name="agno_worker",
            subsystem="boot",
            severity=Severity.FATAL,
            error_code=ModuleErrorCode.OVG_TIMECHAIN_OPEN_FAILED,
            suggested_remediation="Check timechain fork-state cache integrity",
        )
        def fn(recv_queue, send_queue):
            raise OSError("disk read error")

        with pytest.raises(OSError):
            fn(None, q)

        payload = q.sent[0]["payload"]
        assert payload["severity"] == "FATAL"
        assert payload["error_code"] == "OVG_TIMECHAIN_OPEN_FAILED"
        assert payload["suggested_remediation"] == "Check timechain fork-state cache integrity"


# ─── Sender resolution ────────────────────────────────────────────────────────

class TestSenderResolution:
    def setup_method(self):
        _reset_rate_gate()

    def test_sender_resolved_from_positional_send_queue(self):
        q = _FakeSendQueue()

        @with_error_envelope(module_name="m", subsystem="s")
        def fn(recv_queue, send_queue, name, config):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            fn("recv", q, "myname", {})
        assert len(q.sent) == 1

    def test_sender_resolved_from_kwarg(self):
        q = _FakeSendQueue()

        @with_error_envelope(module_name="m", subsystem="s")
        def fn(recv_queue, send_queue=None):
            raise RuntimeError("kw test")

        with pytest.raises(RuntimeError):
            fn("recv", send_queue=q)
        assert len(q.sent) == 1

    def test_custom_sender_arg_name_for_helpers(self):
        # Helpers that take `bus` instead of `send_queue` work via sender_arg=.
        bus = _FakeBus()

        @with_error_envelope(module_name="cognitive_worker", subsystem="reason",
                             sender_arg="bus")
        def helper(bus, payload):
            raise ValueError("bad payload")

        with pytest.raises(ValueError):
            helper(bus, {"x": 1})
        assert len(bus.published) == 1
        assert bus.published[0]["src"] == "cognitive_worker"

    def test_no_sender_available_still_reraises(self):
        # Helper without any matching arg name — envelope skipped, but raise survives.
        @with_error_envelope(module_name="m", subsystem="s")
        def lonely(x, y):
            raise IndexError("no sender here")

        with pytest.raises(IndexError):
            lonely(1, 2)


# ─── Re-raise semantics ───────────────────────────────────────────────────────

class TestReraiseSemantics:
    def setup_method(self):
        _reset_rate_gate()

    def test_keyboard_interrupt_not_caught(self):
        q = _FakeSendQueue()

        @with_error_envelope(module_name="m", subsystem="s")
        def fn(recv_queue, send_queue):
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            fn(None, q)
        # KeyboardInterrupt is not Exception; decorator does NOT publish envelope.
        assert q.sent == []

    def test_system_exit_not_caught(self):
        q = _FakeSendQueue()

        @with_error_envelope(module_name="m", subsystem="s")
        def fn(recv_queue, send_queue):
            raise SystemExit(0)

        with pytest.raises(SystemExit):
            fn(None, q)
        assert q.sent == []

    def test_traceback_preserved(self):
        q = _FakeSendQueue()

        @with_error_envelope(module_name="m", subsystem="s")
        def inner(send_queue):
            raise ValueError("deepest")

        def outer():
            inner(q)

        try:
            outer()
        except ValueError as exc:
            # Traceback should include both outer() and inner() frames.
            import traceback as _tb
            tb_str = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
            assert "inner" in tb_str
            assert "outer" in tb_str
        else:
            pytest.fail("expected ValueError")


# ─── Bus-failure double-fault guard ───────────────────────────────────────────

class TestDoubleFaultGuard:
    def setup_method(self):
        _reset_rate_gate()

    def test_broken_bus_does_not_mask_original_exception(self):
        # Sender that explodes on put_nowait simulates a broken bus.
        class _BrokenQueue:
            def put_nowait(self, msg):
                raise OSError("bus pipe broken")

        @with_error_envelope(module_name="m", subsystem="s")
        def fn(recv_queue, send_queue):
            raise ValueError("real error")

        # Must surface the real ValueError, NOT the bus OSError.
        with pytest.raises(ValueError, match="real error"):
            fn(None, _BrokenQueue())


# ─── Integration: ModuleError envelope shape matches Chunk 11B contract ──────

class TestEnvelopeShapeContract:
    def setup_method(self):
        _reset_rate_gate()

    def test_envelope_is_round_trippable_via_from_wire_dict(self):
        q = _FakeSendQueue()

        @with_error_envelope(
            module_name="output_verifier",
            subsystem="timechain.scan",
            severity=Severity.WARN,
            error_code=ModuleErrorCode.TIMECHAIN_SCAN_TIMEOUT,
        )
        def fn(send_queue):
            raise TimeoutError("timechain scan exceeded budget")

        with pytest.raises(TimeoutError):
            fn(q)

        # The decorator's published payload should reconstruct cleanly via
        # the Chunk 11B from_wire_dict contract — guards against silent
        # divergence between the two surfaces.
        payload = q.sent[0]["payload"]
        restored = ModuleError.from_wire_dict(payload)
        assert restored.module_name == "output_verifier"
        assert restored.subsystem == "timechain.scan"
        assert restored.severity is Severity.WARN
        assert restored.error_code == "TIMECHAIN_SCAN_TIMEOUT"
