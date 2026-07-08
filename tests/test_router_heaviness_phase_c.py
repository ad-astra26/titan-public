"""Phase C — session-aware heaviness routing (fully emergent).

RFP_load_adaptive_inference_routing §7.C. Asserts the in-memory heaviness tracker,
the composite scalar, and the router's two coupling channels: the light|heavy context
BUCKET dimension and the REWARD-modulation (heavier ⇒ quality weighted up + latency
relaxed ⇒ the bandit LEARNS to keep gemma for deep sessions). No deterministic pin.
"""
from titan_hcl.inference import session_heaviness as sh
from titan_hcl.inference.adaptive_router import AdaptiveRouter


def _router(**cfg):
    cfg.setdefault("router_enabled", True)
    return AdaptiveRouter(cfg)


# ── the tracker ──

def test_note_turn_accumulates_and_raw_reports():
    t = sh.SessionHeaviness()
    t.note_turn("s", 100, now=1000.0)
    t.note_turn("s", 300, now=1030.0)
    turns, dur, toks = t.raw("s", now=1030.0)
    assert turns == 2 and abs(dur - 30.0) < 1e-6 and toks == 400


def test_unknown_session_is_light():
    t = sh.SessionHeaviness()
    assert t.raw("nope") == (0, 0.0, 0)


def test_ttl_expired_session_reads_fresh():
    """A returning user after a long gap is NOT treated as a deep session."""
    t = sh.SessionHeaviness(ttl_s=60.0)
    t.note_turn("s", 100, now=1000.0)
    assert t.raw("s", now=1000.0 + 61.0) == (0, 0.0, 0)


def test_compute_heaviness_monotonic_and_bounded():
    base = sh.compute_heaviness(1, 10.0, 100, "", {})
    more_turns = sh.compute_heaviness(8, 10.0, 100, "", {})
    more_tokens = sh.compute_heaviness(1, 10.0, 6000, "", {})
    assert 0.0 <= base <= 1.0
    assert more_turns > base and more_tokens > base
    # saturates at the cap (never exceeds 1.0)
    assert sh.compute_heaviness(9999, 1e9, 1e9, "", {}) <= 1.0


def test_goal_class_weight_contributes_when_configured():
    cfg = {"heaviness_goal_class_weights": {"deep": 1.0}, "heaviness_w_goal_class": 0.5}
    with_gc = sh.compute_heaviness(1, 0.0, 0, "deep", cfg)
    without = sh.compute_heaviness(1, 0.0, 0, "greeting", cfg)
    assert with_gc > without


# ── the router coupling ──

def test_bucket_carries_heaviness_dimension():
    r = _router(heaviness_threshold=0.5)
    assert r._bucket(3, True, 0.2) == "mid|light|chat"
    assert r._bucket(3, True, 0.8) == "mid|heavy|chat"


def test_heaviness_zero_reward_is_identical_to_pre_c():
    """At heaviness=0 both modulation scales vanish → the reward is the base
    (no-modulation) formula. NB: R2.4 (§R2 2026-07-08) replaced the responsiveness
    curve with the monotonic non-saturating `target/(target+lat)`; this parity test
    tracks that base curve — its subject is the heaviness-modulation vanishing at
    heaviness=0, not the specific responsiveness shape."""
    r = _router()
    # recompute the base composite by hand for a mid-latency gemma turn
    wl, wq, wc, target = 0.6, 0.3, 0.1, 12.0
    lat = 6.0
    resp = max(1e-6, target) / (max(1e-6, target) + max(0.0, lat))   # R2.4 curve
    q = r._qprior["gemma4:31b"]
    c = r._cprior["gemma4:31b"]
    expected = wl * resp + wq * q + wc * c
    assert abs(r._reward("gemma4:31b", lat, True, 0.0) - expected) < 1e-9


def test_heavy_session_rewards_slow_gemma_higher_than_light():
    """The crux: a SLOW gemma (past the latency target) earns a higher composite in a
    heavy session than a light one — so the bandit keeps gemma for depth, offloads light."""
    r = _router()
    light = r._reward("gemma4:31b", latency_s=30.0, is_chat=True, heaviness=0.0)
    heavy = r._reward("gemma4:31b", latency_s=30.0, is_chat=True, heaviness=1.0)
    assert heavy > light


def test_exploration_clamped_for_heavy_session_under_high_load():
    """INV-AR-EXPLORE-BOUNDED (§7.C) — even with explore_eps=1.0 (always explore), a
    heavy session under high load never explores off the greedy-best arm."""
    r = _router(explore_eps=1.0, min_samples=1, min_dwell_s=0.0,
                in_flight_ceiling=8, heaviness_threshold=0.5)
    # warm the heavy|high bucket so gemma is the clear greedy winner
    bucket = r._bucket(20, True, 1.0)
    r._table[bucket] = {"gemma4:31b": {"r": 1.0, "n": 50},
                        "ministral-3:14b": {"r": 0.1, "n": 50},
                        "gemma3:12b": {"r": 0.1, "n": 50}}
    picks = {r._bandit_choose(20, True, heaviness=1.0) for _ in range(40)}
    assert picks == {"gemma4:31b"}      # clamp held — never explored away


def test_light_session_under_load_still_explores():
    """Control: the clamp is specific to heavy+high-load — a light session still explores."""
    r = _router(explore_eps=1.0, min_samples=1, min_dwell_s=0.0, in_flight_ceiling=8)
    bucket = r._bucket(20, True, 0.0)
    r._table[bucket] = {"gemma4:31b": {"r": 1.0, "n": 50},
                        "ministral-3:14b": {"r": 0.1, "n": 50},
                        "gemma3:12b": {"r": 0.1, "n": 50}}
    picks = {r._bandit_choose(20, True, heaviness=0.0) for _ in range(60)}
    assert len(picks) > 1                # explored (eps=1.0 not clamped)
