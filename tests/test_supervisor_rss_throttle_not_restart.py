"""RFP_supervision_lifecycle §7.B/§7.F (INV-SUP-1/2/8) — a RESOURCE condition
(RssAnon over rss_limit) THROTTLES + is surfaced; it NEVER drives a kill→respawn.
Only genuine critical faults (dead pid / hung heartbeat / FATAL ModuleError)
restart. The legacy rss→restart stays behind a kill-switch for emergency revert.

This is the fix for the load→restart→load cascade (agno 82×/40min on T1 mainnet
2026-06-16): a single rss-over-limit must not detonate a restart storm.

Run isolated: python -m pytest tests/test_supervisor_rss_throttle_not_restart.py -v -p no:anchorpy
"""
import os
import time
from unittest.mock import MagicMock

from titan_hcl.bus import DivineBus, MODULE_RESTART_REQUEST
from titan_hcl.guardian_hcl import Guardian, ModuleSpec, ModuleState
from titan_hcl.supervisor.core import Supervisor


def _harness(rss_mb: float, *, limit: int = 700, throttle: bool = True):
    g = Guardian(DivineBus())
    g.register(ModuleSpec(
        name="heavy", layer="L2", entry_fn=lambda *a, **k: None,
        autostart=True, restart_on_crash=True, rss_limit_mb=limit))
    info = g._modules["heavy"]
    info.state = ModuleState.RUNNING
    g._rss_over_throttles = throttle
    # over/under-limit RssAnon (the #1 fix already makes _get_rss_mb read RssAnon)
    g._get_rss_mb = MagicMock(return_value=rss_mb)
    # SHM slot: a live, freshly-heartbeating RUNNING worker (real pid for os.kill)
    entry = MagicMock(state="running", pid=os.getpid(), last_heartbeat=time.time())
    bank = MagicMock(); bank.read = MagicMock(return_value=entry)
    g._ensure_module_state_reader_bank = MagicMock(return_value=bank)
    sup = Supervisor(g.bus, g)
    published = []
    g.bus.publish = lambda msg: published.append(msg)
    return g, info, sup, published


def _restart_requested(published) -> bool:
    # restart requests are make_msg dicts routed to the lifecycle subscriber.
    for m in published:
        d = m if isinstance(m, dict) else getattr(m, "__dict__", {}) or {}
        if d.get("dst") == "guardian_hcl_lifecycle":
            return True
        payload = d.get("payload", {}) or {}
        if isinstance(payload, dict) and str(payload.get("reason", "")).startswith("rss_"):
            return True
    return False


def test_rss_over_limit_throttles_no_restart():
    # RssAnon 900 > 700 limit, throttle mode (default) → NO restart, counter ticks.
    g, info, sup, published = _harness(900.0, limit=700, throttle=True)
    sup.monitor_tick()
    assert info.consecutive_rss_over_cycles == 1
    assert not _restart_requested(published), "resource condition must NOT restart"


def test_sustained_over_limit_still_no_restart():
    # Many cycles over-limit → still no respawn (throttle beats respawn).
    g, info, sup, published = _harness(900.0, limit=700, throttle=True)
    for _ in range(40):
        sup.monitor_tick()
    assert info.consecutive_rss_over_cycles >= 40
    assert not _restart_requested(published)


def test_under_limit_resets_counter():
    g, info, sup, published = _harness(900.0, limit=700, throttle=True)
    sup.monitor_tick()
    assert info.consecutive_rss_over_cycles == 1
    g._get_rss_mb = MagicMock(return_value=300.0)   # recovered, under limit
    sup.monitor_tick()
    assert info.consecutive_rss_over_cycles == 0


def test_kill_switch_restores_legacy_rss_restart():
    # rss_over_limit_throttles_not_restarts=false → legacy behavior (rss → restart).
    g, info, sup, published = _harness(900.0, limit=700, throttle=False)
    sup.monitor_tick()
    assert _restart_requested(published), "kill-switch must restore rss→restart"
