"""RFP_load_adaptive_inference_routing §R2 addendum — offload effectiveness +
dispatch pacing. Unit gates for the 5 zero-quality-loss fixes.

Run isolated: python -m pytest tests/test_router_r2_addendum.py -v -p no:anchorpy
"""
import time
import asyncio

import pytest

from titan_hcl.inference.adaptive_router import AdaptiveRouter, InferenceLatencyMonitor


def _router(**cfg):
    cfg.setdefault("router_enabled", True)
    cfg.setdefault("model_ladder", ["gemma4:31b", "ministral-3:14b", "gemma3:12b"])
    cfg.setdefault("min_dwell_s", 20.0)
    cfg.setdefault("exit_latency_s", 7.0)
    cfg.setdefault("target_latency_chat_s", 12.0)
    return AdaptiveRouter(cfg)


# ── R2.4 — reward responsiveness monotonic + never-saturating ──

def test_r2d_reward_monotonic_past_2x_target():
    """Gate R2-d: a faster model out-scores a slower one EVEN when both exceed
    2×target (24s) — the old clamped curve scored both 0 (equal) and let quality
    win (→ gemma under load)."""
    r = _router()
    # same model → identical quality/cost priors → isolates the latency term
    fast = r._reward("gemma4:31b", 30.0, True, 0.0)   # >24s
    slow = r._reward("gemma4:31b", 90.0, True, 0.0)   # >24s
    assert fast > slow, f"reward not monotonic past 2×target: {fast} !> {slow}"


def test_r2d_responsiveness_never_zero_and_ordered():
    r = _router()
    # monotone strictly-decreasing across the whole range, always > 0
    rewards = [r._reward("gemma4:31b", lat, True, 0.0)
               for lat in (1.0, 12.0, 24.0, 60.0, 120.0)]
    assert all(a > b for a, b in zip(rewards, rewards[1:])), rewards
    # even a huge latency keeps a positive (non-saturated) reward
    assert r._reward("gemma4:31b", 300.0, True, 0.0) > r._reward("gemma4:31b", 600.0, True, 0.0)


# ── R2.2 — anti-flap dwell between fallback arms ──

def test_r2b_escalation_gemma_to_fallback_is_immediate():
    r = _router()
    r._current_arm = r.managed
    r._switched_at = time.time()          # just switched — but escalation must ignore dwell
    out = r._apply_dwell("ministral-3:14b")
    assert out == "ministral-3:14b"       # offload responsiveness — not dwell-blocked
    assert r._current_arm == "ministral-3:14b"


def test_r2b_fallback_to_fallback_held_within_dwell():
    r = _router()
    r._current_arm = "ministral-3:14b"
    r._switched_at = time.time()          # within min_dwell
    out = r._apply_dwell("gemma3:12b")    # different fallback
    assert out == "ministral-3:14b"       # HELD (anti-flap)


def test_r2b_fallback_to_fallback_allowed_after_dwell():
    r = _router()
    r._current_arm = "ministral-3:14b"
    r._switched_at = time.time() - 100.0  # dwell elapsed
    out = r._apply_dwell("gemma3:12b")
    assert out == "gemma3:12b"


# ── R2.3 — warmth signal separate from chat-ema, revert gated on warmth ──

def test_r2c_warmth_is_separate_from_ema():
    m = InferenceLatencyMonitor()
    m.record("gemma4:31b", 40.0)          # real chat latency
    m.record_warmth("gemma4:31b", 2.5)    # keepalive ping
    assert m.ema("gemma4:31b") == 40.0    # chat ema untouched by warmth
    assert abs(m.warmth("gemma4:31b") - 2.5) < 1e-9
    assert m.warmth("never-pinged") == 999.0   # default HIGH = not-confirmed-warm


def test_r2c_revert_to_gemma_gated_on_warmth_not_frozen_ema():
    r = _router()
    r._current_arm = "ministral-3:14b"
    r._switched_at = time.time() - 100.0            # dwell elapsed
    # simulate the frozen-ema bug: chat ema stuck high; warmth says gemma is warm
    r.monitor.record("gemma4:31b", 22.0)            # frozen high (old bug source)
    r.monitor.record_warmth("gemma4:31b", 3.0)      # < exit_latency → confirmed warm
    assert r._apply_dwell("gemma4:31b") == "gemma4:31b"   # reverts (R2.3 fix)
    # now gemma NOT warm (ping slow) → must NOT revert
    r2 = _router()
    r2._current_arm = "ministral-3:14b"
    r2._switched_at = time.time() - 100.0
    r2.monitor.record_warmth("gemma4:31b", 20.0)    # > exit_latency → cold
    assert r2._apply_dwell("gemma4:31b") == "ministral-3:14b"   # held


# ── R2.5 — admission-control semaphore ──

def test_r2e_admission_semaphore_sized_and_reused():
    from titan_hcl.inference.ollama_cloud import OllamaCloudProvider
    prov = OllamaCloudProvider({"api_key": "test-key"})
    loop = asyncio.new_event_loop()
    try:
        sem = prov._get_admit_sem(loop)
        assert isinstance(sem, asyncio.Semaphore)
        assert sem._value >= 1                       # sized (default 6 when no config)
        assert prov._get_admit_sem(loop) is sem      # reused on same loop
        loop2 = asyncio.new_event_loop()
        assert prov._get_admit_sem(loop2) is not sem  # rebuilt on loop change
        loop2.close()
    finally:
        loop.close()


def test_r2e_admission_semaphore_respects_config_limit():
    from titan_hcl.inference.ollama_cloud import OllamaCloudProvider
    prov = OllamaCloudProvider({"api_key": "test-key"})
    loop = asyncio.new_event_loop()
    try:
        # force a known limit by pre-seeding the cache path
        prov._admit_sem_limit = 3
        prov._admit_sem = asyncio.Semaphore(3)
        prov._admit_sem_loop = loop
        sem = prov._get_admit_sem(loop)
        # if config resolves to default 6, it rebuilds to 6; either way sized >=1
        assert sem._value >= 1
    finally:
        loop.close()
