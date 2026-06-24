"""AdaptiveRouter — Phase B (RFP_load_adaptive_inference_routing §7.B).

Run:  python -m pytest tests/test_adaptive_router.py -v -p no:anchorpy
"""
import os
import tempfile

from titan_hcl.inference.adaptive_router import (
    AdaptiveRouter, InferenceLatencyMonitor, load_bucket)


class _Provider:
    """Stub — resolve_model_class maps the chat class to the managed model."""
    def resolve_model_class(self, model_class):
        return {"heavy": "gemma4:31b", "fast": "gemma3:4b"}.get(model_class, "gemma4:31b")


def _cfg(**over):
    c = {"router_enabled": True, "min_samples": 4, "explore_eps": 0.0,
         "in_flight_ceiling": 8, "enter_latency_s": 12.0, "exit_latency_s": 7.0,
         "min_dwell_s": 0.0, "model_ladder":
         ["gemma4:31b", "ministral-3:14b", "gemini-3-flash-preview"]}
    c.update(over)
    return c


# ── 1. PASSTHROUGH safety — zero change unless enabled + chat + managed ────────
def test_passthrough_when_disabled_or_nonchat_or_nonmanaged():
    p = _Provider()
    # disabled → plain resolution
    r = AdaptiveRouter(_cfg(router_enabled=False))
    assert r.choose(p, "heavy", in_flight=20, is_chat=True) == "gemma4:31b"
    # enabled but non-chat → passthrough
    r = AdaptiveRouter(_cfg())
    assert r.choose(p, "heavy", in_flight=20, is_chat=False) == "gemma4:31b"
    # enabled, chat, but the class doesn't resolve to the managed model → passthrough
    assert r.choose(p, "fast", in_flight=20, is_chat=True) == "gemma3:4b"


# ── 2. Cold-start heuristic — offload when gemma is saturated ───────────────────
def test_cold_start_heuristic_offloads_under_load():
    p = _Provider()
    r = AdaptiveRouter(_cfg())
    # low load → keep gemma
    assert r.choose(p, "heavy", in_flight=1, is_chat=True) == "gemma4:31b"
    # over the ceiling → first fallback (cold-start, no samples yet)
    assert r.choose(p, "heavy", in_flight=12, is_chat=True) == "ministral-3:14b"


# ── 3. The bandit LEARNS — a fallback that scores well under load gets chosen ───
def test_bandit_learns_to_prefer_better_arm():
    p = _Provider()
    r = AdaptiveRouter(_cfg(min_samples=2, explore_eps=0.0))
    # under high load, feed: gemma slow (bad reward), ministral fast (good reward)
    for _ in range(6):
        r.feedback("gemma4:31b", latency_s=40.0, in_flight=10, is_chat=True)
        r.feedback("ministral-3:14b", latency_s=4.0, in_flight=10, is_chat=True)
    # now past cold-start, greedy → the faster arm wins for that bucket
    assert r.choose(p, "heavy", in_flight=10, is_chat=True) == "ministral-3:14b"


# ── 4. Flap guardrail — min-dwell + confirmed-warm before reverting to gemma ────
def test_revert_requires_dwell_and_confirmed_warm():
    p = _Provider()
    r = AdaptiveRouter(_cfg(min_dwell_s=999.0))  # long dwell → can't revert yet
    # force a switch to a fallback
    assert r.choose(p, "heavy", in_flight=12, is_chat=True) == "ministral-3:14b"
    # even if load drops, min_dwell holds the fallback
    assert r.choose(p, "heavy", in_flight=0, is_chat=True) == "ministral-3:14b"
    # with dwell satisfied but gemma still HOT (ema>exit) → still hold the fallback
    r2 = AdaptiveRouter(_cfg(min_dwell_s=0.0))
    r2.choose(p, "heavy", in_flight=12, is_chat=True)        # switch to fallback
    r2.monitor.record("gemma4:31b", 30.0)                    # gemma confirmed COLD
    assert r2.choose(p, "heavy", in_flight=0, is_chat=True) == "ministral-3:14b"
    # gemma now confirmed WARM (ema pulled well below exit) → revert allowed
    for _ in range(12):
        r2.monitor.record("gemma4:31b", 2.0)
    assert r2.monitor.ema("gemma4:31b") < 7.0, "gemma EMA must be confirmed-warm"
    assert r2.choose(p, "heavy", in_flight=0, is_chat=True) == "gemma4:31b"


# ── 5. Composite reward — lower latency + higher quality → higher reward ───────
def test_reward_rewards_speed_and_quality():
    r = AdaptiveRouter(_cfg())
    fast = r._reward("ministral-3:14b", latency_s=3.0, is_chat=True)
    slow = r._reward("ministral-3:14b", latency_s=40.0, is_chat=True)
    assert fast > slow, "lower latency → higher reward"
    # at equal (fast) latency, gemma's quality prior beats a fallback's
    g = r._reward("gemma4:31b", latency_s=3.0, is_chat=True)
    m = r._reward("ministral-3:14b", latency_s=3.0, is_chat=True)
    assert g > m, "gemma's higher quality prior wins at equal speed"


# ── 6. Persistence — learned table round-trips ─────────────────────────────────
def test_state_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "inference_router_state.json")
        r = AdaptiveRouter(_cfg(router_state_path=sp))
        for _ in range(5):
            r.feedback("ministral-3:14b", 4.0, in_flight=10, is_chat=True)
        r.save()
        r2 = AdaptiveRouter(_cfg(router_state_path=sp))
        assert r2._table, "loaded a non-empty learned table"
        assert "high|chat" in r2._table and "ministral-3:14b" in r2._table["high|chat"]


def test_monitor_ema_and_bucket():
    m = InferenceLatencyMonitor(alpha=0.5)
    m.record("x", 10.0)
    m.record("x", 20.0)
    assert 10.0 < m.ema("x") < 20.0
    assert load_bucket(1) == "low" and load_bucket(4) == "mid" and load_bucket(9) == "high"
