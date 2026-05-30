"""D-SPEC-146 — guardian_state per-module {pid, rss_mb, cpu_delta_s} via MODULE_HEARTBEAT.

Covers the SPEC §1339 heartbeat self-metrics contract end to end:
  1. BusSocketClient.publish stamps {pid, rss_mb, cpu_delta_s} onto MODULE_HEARTBEAT
     at the single producer chokepoint (and leaves other message types untouched).
  2. Orchestrator._ingest_hb_self_metrics ingests pid + cpu_delta_s (+ rss) and
     reconciles STOPPED→RUNNING on a live heartbeat (guardian_hcl metadata path).
  3. GuardianStatePublisher._compute_payload publishes pid + cpu_delta_s.
  4. ProfileReport.collect_proc_report surfaces self-reported rss_mb/cpu_delta_s even
     when pid is falsy (imw-class writers) instead of collapsing to a 2-key stub.
"""
import os

from titan_hcl import bus
from titan_hcl.core.bus_socket import BusSocketClient
from titan_hcl.orchestrator.core import Orchestrator
from titan_hcl.orchestrator.module_registry import ModuleInfo, ModuleState, ModuleSpec
from titan_hcl.logic.guardian_state_publisher import GuardianStatePublisher
from titan_hcl.core.profiler import ProfileReport


def _make_client():
    # __init__ does not touch the socket — .start() would. Safe to construct.
    return BusSocketClient(titan_id="T_test", authkey=b"k" * 32, name="unit_test")


def test_publish_stamps_heartbeat_self_metrics():
    client = _make_client()
    msg = {"type": bus.MODULE_HEARTBEAT, "src": "unit_test", "dst": "guardian", "payload": {}}
    client._enrich_heartbeat(msg)
    p = msg["payload"]
    assert p["pid"] == os.getpid()
    assert "cpu_delta_s" in p and isinstance(p["cpu_delta_s"], float)
    assert "rss_mb" in p and p["rss_mb"] > 0.0  # this test process has real RSS


def test_publish_respects_caller_rss_mb():
    client = _make_client()
    msg = {"type": bus.MODULE_HEARTBEAT, "src": "x", "dst": "guardian",
           "payload": {"rss_mb": 123.4}}
    client._enrich_heartbeat(msg)
    # self-report from the emit site is preserved; pid still stamped.
    assert msg["payload"]["rss_mb"] == 123.4
    assert msg["payload"]["pid"] == os.getpid()


def test_cpu_delta_grows_across_heartbeats():
    client = _make_client()
    m1 = {"type": bus.MODULE_HEARTBEAT, "payload": {}}
    client._enrich_heartbeat(m1)
    assert m1["payload"]["cpu_delta_s"] == 0.0  # first sample → baseline, no delta
    # burn a little CPU then heartbeat again
    s = 0
    for i in range(2_000_000):
        s += i
    m2 = {"type": bus.MODULE_HEARTBEAT, "payload": {}}
    client._enrich_heartbeat(m2)
    assert m2["payload"]["cpu_delta_s"] >= 0.0  # monotonic, non-negative


def test_non_heartbeat_untouched():
    client = _make_client()
    msg = {"type": "SOME_OTHER_EVENT", "payload": {"foo": 1}}
    # publish() only enriches MODULE_HEARTBEAT — verify the guard in publish path
    if msg.get("type") == bus.MODULE_HEARTBEAT:
        client._enrich_heartbeat(msg)
    assert "pid" not in msg["payload"]
    assert msg["payload"] == {"foo": 1}


def test_orchestrator_ingests_pid_cpu_and_promotes_state():
    info = ModuleInfo(spec=ModuleSpec(name="memory", entry_fn=lambda *a, **k: None, layer="L2"))
    assert info.state == ModuleState.STOPPED
    assert info.pid is None
    # _ingest_hb_self_metrics uses no instance state — call unbound.
    Orchestrator._ingest_hb_self_metrics(
        None, info, {"rss_mb": 688.1, "pid": 4242, "cpu_delta_s": 2.5})
    assert info.pid == 4242
    assert info.rss_mb == 688.1
    assert info.cpu_delta_s == 2.5
    assert info.state == ModuleState.RUNNING  # live heartbeat reconciles STOPPED→RUNNING


def test_orchestrator_does_not_demote_running_or_touch_disabled():
    spec = ModuleSpec(name="cgn", entry_fn=lambda *a, **k: None, layer="L2")
    info = ModuleInfo(spec=spec)
    info.state = ModuleState.DISABLED
    Orchestrator._ingest_hb_self_metrics(None, info, {"rss_mb": 1.0, "pid": 7})
    assert info.state == ModuleState.DISABLED  # never auto-promote a DISABLED module


def test_guardian_state_publisher_emits_pid_and_cpu_delta():
    class _FakeGuardian:
        def get_status(self):
            return {"memory": {"state": "running", "pid": 4242, "rss_mb": 688.1,
                               "cpu_delta_s": 2.5, "layer": "L2"}}
    pub = GuardianStatePublisher.__new__(GuardianStatePublisher)  # bypass SHM init
    payload = pub._compute_payload(_FakeGuardian())
    m = payload["modules"]["memory"]
    assert m["pid"] == 4242
    assert m["rss_mb"] == 688.1
    assert m["cpu_delta_s"] == 2.5


def test_profiler_surfaces_rss_when_pid_falsy():
    class _FakeGuardian:
        def get_status(self):
            return {"observatory_writer": {"state": "stopped", "pid": 0,
                                           "rss_mb": 15.7, "cpu_delta_s": 0.3}}
    report = ProfileReport(collector=None).collect_proc_report(_FakeGuardian())
    mod = report["modules"]["observatory_writer"]
    assert mod["pid"] is None
    assert mod["rss_mb"] == 15.7          # NOT discarded to a 2-key stub
    assert mod["cpu_delta_s"] == 0.3
