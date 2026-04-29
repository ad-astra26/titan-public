"""
L3 Phase A.8.5 — Reflex subprocess extraction regression tests.

Covers:
  - ReflexCollector refactor: collect_and_fire still produces same output
    (back-compat for legacy in-parent path)
  - _aggregate_inline pure function: same selection as old logic
  - _execute_selected step: still runs registered executors + writes
    cooldowns
  - reflex_worker handles QUERY(action="aggregate") + emits MODULE_READY
  - reflex_worker uses caller-supplied cooldowns, not its own state
  - ReflexProxy translates collect_and_fire → bus.request(aggregate)
  - ReflexProxy reconstructs selected from worker's serial response
  - ReflexProxy returns empty PerceptualField on bus timeout
  - ReflexProxy executes executors locally + cooldowns are written on parent
  - ModuleSpec for "reflex" registered + autostart bound to flag
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
import threading
import time
import unittest
from queue import Empty, Queue
from unittest.mock import MagicMock

_HERE = os.path.dirname(__file__)
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Test fixtures ─────────────────────────────────────────────────────

def _signal(reflex, source, confidence, reason=""):
    return {"reflex": reflex, "source": source,
            "confidence": float(confidence), "reason": reason}


def _convergent_signals(reflex_name, conf=0.6):
    """Body+mind+spirit converge — passes 2-of-3 partial-convergence rule."""
    return [
        _signal(reflex_name, "body", conf),
        _signal(reflex_name, "mind", conf),
        _signal(reflex_name, "spirit", conf),
    ]


# ── Refactor back-compat: ReflexCollector.collect_and_fire ────────────

class TestReflexCollectorBackCompat(unittest.IsolatedAsyncioTestCase):
    """Verify the refactor preserves collect_and_fire behavior."""

    def _build_collector(self):
        from titan_plugin.logic.reflexes import ReflexCollector, ReflexType

        cfg = {
            "fire_threshold": 0.05, "action_threshold": 0.20,
            "public_action_threshold": 0.40, "session_cooldown": 30.0,
            "max_parallel_reflexes": 4, "guardian_threat_threshold": 0.5,
            "focus_boost_threshold": 1.0,  # disable focus boost in tests
        }
        c = ReflexCollector(cfg)

        # Register a fake executor for IDENTITY_CHECK
        async def _fake_identity(stim):
            return {"identity_verified": True, "summary": "ok"}

        c.register_executor(ReflexType.IDENTITY_CHECK, _fake_identity)
        return c

    async def test_collect_and_fire_produces_perceptual_field(self):
        from titan_plugin.logic.reflexes import PerceptualField
        c = self._build_collector()
        signals = _convergent_signals("identity_check", conf=0.7)
        pf = await c.collect_and_fire(
            signals=signals, stimulus_features={"threat_level": 0.0},
            focus_magnitude=0.0, trinity_state={"body": [0.5]*5},
        )
        self.assertIsInstance(pf, PerceptualField)
        self.assertEqual(len(pf.fired_reflexes), 1)
        self.assertEqual(pf.fired_reflexes[0].reflex_type.value,
                         "identity_check")

    async def test_collect_and_fire_writes_cooldown_on_fire(self):
        c = self._build_collector()
        signals = _convergent_signals("identity_check", conf=0.7)
        await c.collect_and_fire(signals=signals,
                                 stimulus_features={"threat_level": 0.0})
        self.assertIn("identity_check", c._cooldowns)
        self.assertGreater(c._cooldowns["identity_check"], 0.0)

    async def test_collect_and_fire_respects_cooldown_on_second_call(self):
        c = self._build_collector()
        signals = _convergent_signals("identity_check", conf=0.7)
        # First call fires
        await c.collect_and_fire(signals=signals,
                                 stimulus_features={"threat_level": 0.0})
        first_cd = c._cooldowns.get("identity_check", 0.0)
        await asyncio.sleep(0.05)  # cooldown 30s; not enough to expire
        # Second call must NOT fire (cooldown active)
        pf2 = await c.collect_and_fire(
            signals=signals, stimulus_features={"threat_level": 0.0})
        self.assertEqual(len(pf2.fired_reflexes), 0)
        # Cooldown timestamp unchanged
        self.assertEqual(c._cooldowns.get("identity_check", 0.0), first_cd)


class TestAggregateInline(unittest.TestCase):
    """Pure-function aggregation behavior (_aggregate_inline)."""

    def _build_collector(self):
        from titan_plugin.logic.reflexes import ReflexCollector
        return ReflexCollector({
            "fire_threshold": 0.05, "action_threshold": 0.20,
            "public_action_threshold": 0.40, "session_cooldown": 30.0,
            "max_parallel_reflexes": 4, "guardian_threat_threshold": 0.5,
            "focus_boost_threshold": 1.0,
        })

    def test_aggregate_inline_returns_selected_tuples(self):
        from titan_plugin.logic.reflexes import ReflexType
        c = self._build_collector()
        signals = _convergent_signals("identity_check", conf=0.7)
        selected = c._aggregate_inline(
            signals, {"threat_level": 0.0}, focus_magnitude=0.0)
        self.assertEqual(len(selected), 1)
        rt, conf, sigs = selected[0]
        self.assertEqual(rt, ReflexType.IDENTITY_CHECK)
        self.assertGreater(conf, 0.0)
        self.assertEqual(len(sigs), 3)

    def test_aggregate_inline_below_threshold_filtered(self):
        c = self._build_collector()
        # Confidence so low combined < fire_threshold (0.05)
        signals = _convergent_signals("identity_check", conf=0.1)
        # combined = 0.1 * 0.1 * 0.1 = 0.001 < 0.05 → filtered
        selected = c._aggregate_inline(
            signals, {"threat_level": 0.0}, focus_magnitude=0.0)
        self.assertEqual(selected, [])

    def test_aggregate_inline_guardian_fast_path(self):
        from titan_plugin.logic.reflexes import ReflexType
        c = self._build_collector()
        # Threat level high → guardian fires regardless of signal convergence
        selected = c._aggregate_inline(
            [], {"threat_level": 0.9}, focus_magnitude=0.0)
        self.assertEqual(len(selected), 1)
        rt, conf, _ = selected[0]
        self.assertEqual(rt, ReflexType.GUARDIAN_SHIELD)
        self.assertGreaterEqual(conf, 0.9)

    def test_aggregate_inline_respects_cooldown(self):
        c = self._build_collector()
        c._cooldowns["identity_check"] = time.time()  # just fired
        signals = _convergent_signals("identity_check", conf=0.7)
        selected = c._aggregate_inline(
            signals, {"threat_level": 0.0}, focus_magnitude=0.0)
        self.assertEqual(selected, [])

    def test_aggregate_inline_top_n_truncation(self):
        c = self._build_collector()
        c.max_parallel = 2
        signals = []
        # Build 3 reflexes that would all fire
        for r in ["identity_check", "metabolism_check", "infra_check"]:
            signals.extend(_convergent_signals(r, conf=0.7))
        selected = c._aggregate_inline(
            signals, {"threat_level": 0.0}, focus_magnitude=0.0)
        self.assertEqual(len(selected), 2)


# ── Reflex worker behavior ────────────────────────────────────────────

class TestReflexWorkerLifecycle(unittest.TestCase):
    """Boot signals + shutdown."""

    def _run_worker(self, recv, send):
        from titan_plugin.modules.reflex_worker import reflex_worker_main

        def _run():
            reflex_worker_main(
                recv, send, "reflex",
                {"info_banner": {"titan_id": "T-test"}})

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def _drain(self, q, timeout=0.5):
        msgs = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msgs.append(q.get(timeout=0.05))
            except Empty:
                if msgs:
                    break
        return msgs

    def test_worker_emits_module_ready_and_reflex_ready(self):
        recv, send = Queue(), Queue()
        t = self._run_worker(recv, send)
        try:
            time.sleep(0.5)
            msgs = self._drain(send, timeout=1.0)
            types = [m["type"] for m in msgs]
            self.assertIn("MODULE_READY", types)
            self.assertIn("REFLEX_READY", types)
        finally:
            recv.put({"type": "MODULE_SHUTDOWN"})
            t.join(timeout=2.0)

    def test_worker_exits_on_shutdown(self):
        recv, send = Queue(), Queue()
        t = self._run_worker(recv, send)
        time.sleep(0.5)
        self._drain(send, timeout=0.5)
        recv.put({"type": "MODULE_SHUTDOWN"})
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive())


class TestReflexWorkerAggregate(unittest.TestCase):
    """QUERY(action="aggregate") → RESPONSE round-trip."""

    def _run_and_query(self, query_payload):
        from titan_plugin.modules.reflex_worker import reflex_worker_main
        recv, send = Queue(), Queue()

        def _run():
            reflex_worker_main(
                recv, send, "reflex",
                {"info_banner": {"titan_id": "T-test"}})

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        try:
            time.sleep(0.5)
            # Drain boot signals
            while True:
                try:
                    send.get(timeout=0.1)
                except Empty:
                    break

            recv.put({
                "type": "QUERY", "src": "reflex_proxy", "dst": "reflex",
                "rid": "test-rid",
                "payload": query_payload, "ts": time.time(),
            })
            deadline = time.time() + 3.0
            while time.time() < deadline:
                try:
                    msg = send.get(timeout=0.2)
                except Empty:
                    continue
                if msg["type"] == "RESPONSE":
                    return msg
            return None
        finally:
            recv.put({"type": "MODULE_SHUTDOWN"})
            t.join(timeout=2.0)

    def test_aggregate_returns_selected_serial(self):
        signals = _convergent_signals("identity_check", conf=0.7)
        reply = self._run_and_query({
            "action": "aggregate", "signals": signals,
            "stimulus_features": {"threat_level": 0.0},
            "focus_magnitude": 0.0, "cooldowns": {},
        })
        self.assertIsNotNone(reply)
        self.assertEqual(reply["rid"], "test-rid")
        body = reply["payload"]
        self.assertIn("selected_serial", body)
        selected = body["selected_serial"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["reflex_type"], "identity_check")
        self.assertGreater(selected[0]["combined_confidence"], 0.0)

    def test_aggregate_uses_caller_cooldowns(self):
        signals = _convergent_signals("identity_check", conf=0.7)
        # Caller passes a fresh cooldown for identity_check → worker
        # must filter it out.
        reply = self._run_and_query({
            "action": "aggregate", "signals": signals,
            "stimulus_features": {"threat_level": 0.0},
            "focus_magnitude": 0.0,
            "cooldowns": {"identity_check": time.time()},
        })
        self.assertIsNotNone(reply)
        self.assertEqual(reply["payload"]["selected_serial"], [])

    def test_aggregate_unknown_action_returns_error(self):
        reply = self._run_and_query({"action": "bogus"})
        self.assertIsNotNone(reply)
        body = reply["payload"]
        self.assertIn("error", body)


# ── ReflexProxy behavior (mocked bus) ─────────────────────────────────

class TestReflexProxyCore(unittest.IsolatedAsyncioTestCase):
    """Proxy's collect_and_fire override + executor-side preserved."""

    def _make_bus_mock(self, response_payload=None, return_none=False):
        bus = MagicMock()
        bus.subscribe.return_value = Queue()
        if return_none:
            bus.request.return_value = None
        else:
            bus.request.return_value = {
                "type": "RESPONSE", "src": "reflex", "dst": "reflex_proxy",
                "rid": "test-rid",
                "payload": response_payload or {"selected_serial": []},
                "ts": time.time(),
            }
        return bus

    def _build_proxy(self, bus):
        from titan_plugin.proxies.reflex_proxy import ReflexProxy
        cfg = {
            "fire_threshold": 0.05, "action_threshold": 0.20,
            "public_action_threshold": 0.40, "session_cooldown": 30.0,
            "max_parallel_reflexes": 4, "guardian_threat_threshold": 0.5,
            "focus_boost_threshold": 1.0,
        }
        return ReflexProxy(bus, cfg, request_timeout_s=1.0)

    def test_proxy_subscribes_to_bus_at_construction(self):
        bus = self._make_bus_mock()
        self._build_proxy(bus)
        bus.subscribe.assert_called_once_with("reflex_proxy", reply_only=True)

    def test_proxy_inherits_register_executor(self):
        from titan_plugin.logic.reflexes import ReflexType
        bus = self._make_bus_mock()
        proxy = self._build_proxy(bus)

        async def _exec(stim):
            return {"ok": True}

        proxy.register_executor(ReflexType.IDENTITY_CHECK, _exec)
        self.assertIn(ReflexType.IDENTITY_CHECK, proxy._executors)

    async def test_proxy_collect_and_fire_routes_via_bus(self):
        bus = self._make_bus_mock(response_payload={
            "selected_serial": [
                {"reflex_type": "identity_check",
                 "combined_confidence": 0.5,
                 "signals": [_signal("identity_check", "body", 0.7)]},
            ],
            "notices": [],
        })
        proxy = self._build_proxy(bus)

        async def _fake_identity(stim):
            return {"identity_verified": True, "summary": "ok"}

        from titan_plugin.logic.reflexes import ReflexType
        proxy.register_executor(ReflexType.IDENTITY_CHECK, _fake_identity)

        pf = await proxy.collect_and_fire(
            signals=[_signal("identity_check", "body", 0.7)],
            stimulus_features={"threat_level": 0.0},
            focus_magnitude=0.0, trinity_state=None,
        )
        # bus.request was called with action="aggregate"
        call_kwargs = bus.request.call_args.kwargs
        self.assertEqual(call_kwargs["dst"], "reflex")
        self.assertEqual(call_kwargs["payload"]["action"], "aggregate")
        # Executor ran locally → fired_reflexes populated
        self.assertEqual(len(pf.fired_reflexes), 1)
        self.assertEqual(pf.fired_reflexes[0].reflex_type, ReflexType.IDENTITY_CHECK)
        # Cooldown written locally on proxy
        self.assertIn("identity_check", proxy._cooldowns)

    async def test_proxy_collect_and_fire_returns_empty_on_timeout(self):
        bus = self._make_bus_mock(return_none=True)
        proxy = self._build_proxy(bus)
        pf = await proxy.collect_and_fire(
            signals=[_signal("identity_check", "body", 0.7)],
            stimulus_features={"threat_level": 0.0},
            focus_magnitude=0.0,
        )
        self.assertEqual(len(pf.fired_reflexes), 0)
        self.assertIn("reflex_aggregate_timeout", pf.reflex_notices)

    async def test_proxy_sends_cooldowns_to_worker(self):
        bus = self._make_bus_mock()
        proxy = self._build_proxy(bus)
        proxy._cooldowns["identity_check"] = 12345.0
        await proxy.collect_and_fire(
            signals=[_signal("identity_check", "body", 0.7)],
            stimulus_features={"threat_level": 0.0},
        )
        call_kwargs = bus.request.call_args.kwargs
        self.assertEqual(
            call_kwargs["payload"]["cooldowns"]["identity_check"], 12345.0)

    async def test_proxy_handles_unknown_reflex_type_in_response(self):
        bus = self._make_bus_mock(response_payload={
            "selected_serial": [
                {"reflex_type": "made_up_reflex",
                 "combined_confidence": 0.5, "signals": []},
            ],
            "notices": [],
        })
        proxy = self._build_proxy(bus)
        pf = await proxy.collect_and_fire(
            signals=[], stimulus_features={"threat_level": 0.0})
        self.assertEqual(len(pf.fired_reflexes), 0)
        self.assertTrue(any("reflex_unknown_type" in n for n in pf.reflex_notices))


# ── Plugin flag-routing surface ───────────────────────────────────────

class TestPluginFlagRouting(unittest.TestCase):
    """Verify worker function signature + ReflexProxy is a ReflexCollector."""

    def test_worker_main_signature(self):
        from titan_plugin.modules.reflex_worker import reflex_worker_main
        sig = inspect.signature(reflex_worker_main)
        params = list(sig.parameters.keys())
        self.assertEqual(params[:4], ["recv_queue", "send_queue", "name", "config"])

    def test_proxy_is_reflex_collector_subclass(self):
        from titan_plugin.logic.reflexes import ReflexCollector
        from titan_plugin.proxies.reflex_proxy import ReflexProxy
        self.assertTrue(issubclass(ReflexProxy, ReflexCollector))


if __name__ == "__main__":
    unittest.main()
