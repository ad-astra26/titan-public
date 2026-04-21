"""
Unit tests for F-phase Session 1 — Meta-Reasoning Consumer Service Layer.

Covers all 8 Session 1 commits end-to-end:
    1. Bus types + [meta_service_interface] config
    2. meta_service_client.py — public helper + schema validation
    3. meta_service.py — queue/rate-limiter/cache/dry-run handler
    4. meta_recruitment.py — catalog + β-posterior Thompson sampler
    5. SUB_MODES compositional extension (13 new sub-modes)
    6. meta_dynamic_rewards.py — signed accumulator + α ramp
    7. timechain_v2 SIMILAR primitive
    8. Social consumer wire (§16.1) — end-to-end

Test isolation: each test creates fresh instances; no process-wide state
bleeds between tests. meta_service_client has process-local handler registry
— each test that touches it calls _clear_handlers_for_testing first.

Run with:
    pytest tests/test_meta_service_session1.py -v -p no:anchorpy --tb=short
"""
from __future__ import annotations

import random

import pytest


# ═══════════════════════════════════════════════════════════════════════
# Commit 1 — Bus types + config
# ═══════════════════════════════════════════════════════════════════════

def test_bus_constants_present():
    from titan_plugin import bus
    for name in (
        "META_REASON_REQUEST",
        "META_REASON_RESPONSE",
        "META_REASON_OUTCOME",
        "TIMECHAIN_SIMILAR",
        "TIMECHAIN_SIMILAR_RESP",
    ):
        val = getattr(bus, name, None)
        assert val == name, f"bus.{name} missing or wrong value: {val!r}"


def test_meta_service_interface_config_loads():
    import tomllib
    from pathlib import Path
    cfg_path = (Path(__file__).parent.parent
                / "titan_plugin" / "titan_params.toml")
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    msi = cfg.get("meta_service_interface")
    assert msi is not None, "[meta_service_interface] section missing"
    # Required keys
    for k in (
        "per_consumer_requests_per_min",
        "global_requests_per_min",
        "queue_max_depth",
        "backpressure_threshold",
        "delegate_recursion_max_depth",
        "cache_enabled",
        "cache_max_entries",
        "cache_ttl_seconds",
        "cache_cosine_match_threshold",
        "alpha_ramp_enabled",
        "alpha_phase_0_end",
        "outcome_reward_min",
        "outcome_reward_max",
        "timechain_queries_per_min",
        "faiss_index_path",
        "similar_threshold_default",
        "similar_limit_default",
    ):
        assert k in msi, f"missing config key: {k}"
    # Session 1 hard-wire: α ramp DISABLED
    assert msi["alpha_ramp_enabled"] is False, \
        "Session 1 MUST ship with alpha_ramp_enabled=false"
    # Signed outcome bounds
    assert msi["outcome_reward_min"] == -1.0
    assert msi["outcome_reward_max"] == 1.0
    # Sub-tables
    assert "time_budget_ms_hint" in msi
    assert "consumer_home_worker" in msi
    assert msi["consumer_home_worker"]["social"] == "spirit"
    assert msi["consumer_home_worker"]["language"] == "language"


# ═══════════════════════════════════════════════════════════════════════
# Commit 2 — meta_service_client
# ═══════════════════════════════════════════════════════════════════════

def test_client_request_happy_path():
    from titan_plugin.logic import meta_service_client as msc

    class _Q:
        def __init__(self): self.msgs = []
        def put_nowait(self, m): self.msgs.append(m)

    q = _Q()
    rid = msc.send_meta_request(
        consumer_id="social",
        question_type="formulate_strategy",
        context_vector=[0.0] * 30,
        time_budget_ms=300,
        send_queue=q,
        src="spirit",
    )
    assert isinstance(rid, str) and len(rid) > 20
    assert len(q.msgs) == 1
    m = q.msgs[0]
    assert m["type"] == "META_REASON_REQUEST"
    assert m["dst"] == "spirit"
    assert m["src"] == "spirit"
    assert m["payload"]["request_id"] == rid
    assert len(m["payload"]["context_vector"]) == 30
    assert m["payload"]["time_budget_ms"] == 300


@pytest.mark.parametrize("kwargs,needle", [
    (dict(consumer_id="nope", question_type="formulate_strategy",
          context_vector=[0.0]*30, time_budget_ms=300), "unknown consumer"),
    (dict(consumer_id="social", question_type="bad_qt",
          context_vector=[0.0]*30, time_budget_ms=300), "unknown question_type"),
    (dict(consumer_id="social", question_type="formulate_strategy",
          context_vector=[0.0]*29, time_budget_ms=300), "length 30"),
    (dict(consumer_id="social", question_type="formulate_strategy",
          context_vector=[0.0]*30, time_budget_ms=0), "positive int"),
    (dict(consumer_id="social", question_type="formulate_strategy",
          context_vector=[0.0]*30, time_budget_ms=-1), "positive int"),
])
def test_client_request_schema_violations(kwargs, needle):
    from titan_plugin.logic import meta_service_client as msc
    with pytest.raises(ValueError, match=needle):
        msc.send_meta_request(**kwargs)


def test_client_outcome_signed_bounds():
    from titan_plugin.logic import meta_service_client as msc

    class _Q:
        def __init__(self): self.msgs = []
        def put_nowait(self, m): self.msgs.append(m)

    # Happy paths — edges
    for r in (-1.0, -0.5, 0.0, 0.5, 1.0):
        q = _Q()
        msc.send_meta_outcome("rid-1", "social", r, send_queue=q)
        assert q.msgs[0]["payload"]["outcome_reward"] == r

    # Rejections
    for bad in (1.5, -1.5, "abc", None):
        with pytest.raises(ValueError):
            msc.send_meta_outcome("rid-1", "social", bad)


def test_client_dispatch_response():
    from titan_plugin.logic import meta_service_client as msc
    msc._clear_handlers_for_testing()
    bucket = []
    msc.register_response_handler("social", bucket.append)
    ok = msc.dispatch_meta_response({
        "type": "META_REASON_RESPONSE",
        "payload": {"consumer_id": "social", "request_id": "r1",
                    "insight": {"suggested_action": "test"}},
    })
    assert ok and len(bucket) == 1
    assert bucket[0]["insight"]["suggested_action"] == "test"
    msc._clear_handlers_for_testing()


def test_client_dispatch_handles_exception_without_crash():
    from titan_plugin.logic import meta_service_client as msc
    msc._clear_handlers_for_testing()

    def _broken(p): raise RuntimeError("boom")
    msc.register_response_handler("social", _broken)
    ok = msc.dispatch_meta_response({
        "payload": {"consumer_id": "social", "request_id": "r1"}})
    assert ok is False  # exception caught, False returned
    msc._clear_handlers_for_testing()


# ═══════════════════════════════════════════════════════════════════════
# Commit 3 — MetaService queue / rate-limiter / cache / dry-run handler
# ═══════════════════════════════════════════════════════════════════════

def _make_request_msg(consumer="social", qt="formulate_strategy",
                      request_id="r-1", ctx_dim=30):
    return {
        "type": "META_REASON_REQUEST",
        "src": "spirit",
        "dst": "spirit",
        "ts": 0.0,
        "rid": None,
        "payload": {
            "consumer_id": consumer,
            "question_type": qt,
            "context_vector": [0.1] * ctx_dim,
            "time_budget_ms": 300,
            "constraints": {},
            "payload_snippet": "test",
            "request_id": request_id,
        },
    }


def test_service_dry_run_resolves_request():
    from titan_plugin.logic.meta_service import MetaService
    out = []
    svc = MetaService(response_emitter=out.append)
    svc.handle_request(_make_request_msg(request_id="r-a"))
    assert len(out) == 1
    m = out[0]
    assert m["type"] == "META_REASON_RESPONSE"
    assert m["dst"] == "spirit"  # social → spirit per home_worker_map
    assert m["payload"]["failure_mode"] == "not_yet_implemented"
    assert m["payload"]["request_id"] == "r-a"


def test_service_rate_limit_per_consumer():
    from titan_plugin.logic.meta_service import MetaService
    out = []
    svc = MetaService(response_emitter=out.append)
    # Per-consumer default = 10/min; 12 requests → 10 dry-run + 2 rate-limited
    for i in range(12):
        svc.handle_request(_make_request_msg(request_id=f"r-{i}"))
    modes = [m["payload"]["failure_mode"] for m in out]
    assert modes.count("not_yet_implemented") == 10
    assert modes.count("rate_limited") == 2


def test_service_rejects_schema_invalid():
    from titan_plugin.logic.meta_service import MetaService
    svc = MetaService(response_emitter=lambda m: None)
    bad_ctx = _make_request_msg(request_id="bad-1", ctx_dim=29)
    assert svc.handle_request(bad_ctx) == "schema_invalid"
    # Unknown consumer
    msg = _make_request_msg(request_id="bad-2")
    msg["payload"]["consumer_id"] = "UNKNOWN"
    assert svc.handle_request(msg) == "schema_invalid"
    # Unknown question type
    msg = _make_request_msg(request_id="bad-3")
    msg["payload"]["question_type"] = "UNKNOWN"
    assert svc.handle_request(msg) == "schema_invalid"


def test_service_outcome_ingestion_bounded():
    from titan_plugin.logic.meta_service import MetaService
    svc = MetaService(response_emitter=lambda m: None)
    # Happy
    assert svc.handle_outcome({"payload": {
        "consumer_id": "social", "request_id": "r-1",
        "outcome_reward": 0.5,
    }}) is True
    # Out of range
    assert svc.handle_outcome({"payload": {
        "consumer_id": "social", "request_id": "r-1",
        "outcome_reward": 1.5,
    }}) is False
    # Unknown consumer
    assert svc.handle_outcome({"payload": {
        "consumer_id": "BAD", "request_id": "r-1",
        "outcome_reward": 0.5,
    }}) is False
    # Status reflects both (invalid counter advanced)
    status = svc.get_status()
    assert status["counters"]["outcomes_received"] == 1
    assert status["counters"]["outcomes_invalid"] == 2


def test_service_status_export_shape():
    from titan_plugin.logic.meta_service import MetaService
    svc = MetaService(response_emitter=lambda m: None)
    status = svc.get_status()
    # Required top-level keys
    for k in ("session_phase", "alpha_ramp_enabled", "uptime_seconds",
              "queue_depth", "counters", "home_worker_map",
              "cache", "recruitment", "rewards"):
        assert k in status, f"missing status key: {k}"
    assert status["session_phase"] == "session_1_dry_run"
    assert status["alpha_ramp_enabled"] is False
    # Home worker map sanity
    assert status["home_worker_map"]["social"] == "spirit"


# ═══════════════════════════════════════════════════════════════════════
# Commit 4 — MetaRecruitment catalog + β-selector
# ═══════════════════════════════════════════════════════════════════════

def test_recruitment_catalog_complete():
    from titan_plugin.logic.meta_recruitment import RECRUITMENT_CATALOG
    # 47 keys per rFP §5.1 (9 primitives × varying sub-modes)
    assert len(RECRUITMENT_CATALOG) == 47
    # Every key has at least one recruiter
    for k, v in RECRUITMENT_CATALOG.items():
        assert isinstance(v, list) and len(v) >= 1, f"{k} has no recruiters"


def test_recruitment_health_check_no_unknowns():
    from titan_plugin.logic.meta_recruitment import MetaRecruitment
    mr = MetaRecruitment()
    health = mr.catalog_health_check()
    assert health["unknown_recruiter_count"] == 0, \
        f"unknown recruiters: {health['unknown_sample']}"
    assert health["orphan_keys"] == []


def test_recruitment_thompson_converges_to_best():
    from titan_plugin.logic.meta_recruitment import MetaRecruitment
    mr = MetaRecruitment()
    # Train: DECOMPOSE strongly positive, others negative
    for _ in range(50):
        mr.update_outcome("FORMULATE", "define",
                          "reasoning.DECOMPOSE", 0.9)
        mr.update_outcome("FORMULATE", "define",
                          "language_reasoner.formulate_query", -0.5)
        mr.update_outcome("FORMULATE", "define",
                          "pattern_primitives.extract_structure", 0.0)
    # After training, Thompson should overwhelmingly pick DECOMPOSE
    rng = random.Random(99)
    picks = [mr.select_recruiter("FORMULATE", "define", rng=rng)
             for _ in range(200)]
    assert picks.count("reasoning.DECOMPOSE") > 180, \
        f"DECOMPOSE should dominate: {picks.count('reasoning.DECOMPOSE')}/200"


def test_recruitment_resolver_registration_reduces_stale():
    from titan_plugin.logic.meta_recruitment import MetaRecruitment
    mr = MetaRecruitment()
    stale_before = mr.catalog_health_check()["stale_recruiter_count"]
    mr.register_resolver("reasoning", lambda name, ctx: {"success": True})
    stale_after = mr.catalog_health_check()["stale_recruiter_count"]
    assert stale_after < stale_before
    assert "reasoning" in mr.get_stats()["resolvers_registered"]


# ═══════════════════════════════════════════════════════════════════════
# Commit 5 — SUB_MODES compositional extension
# ═══════════════════════════════════════════════════════════════════════

def test_sub_modes_expanded_to_47():
    from titan_plugin.logic.meta_reasoning import SUB_MODES, STEP_REWARDS
    total = sum(len(v) for v in SUB_MODES.values())
    assert total == 47
    # F-phase new sub-modes
    assert "compose_intersection" in SUB_MODES["FORMULATE"]
    assert "compose_union" in SUB_MODES["FORMULATE"]
    assert "compose_difference" in SUB_MODES["FORMULATE"]
    assert "narrow_to_subset" in SUB_MODES["FORMULATE"]
    assert "generalize_from_instance" in SUB_MODES["FORMULATE"]
    assert "episodic_specific" in SUB_MODES["RECALL"]
    assert "semantic_neighbors" in SUB_MODES["RECALL"]
    assert "procedural_matching" in SUB_MODES["RECALL"]
    assert "autobiographical_relevant" in SUB_MODES["RECALL"]
    assert "analogize_from" in SUB_MODES["HYPOTHESIZE"]
    assert "contrast_with" in SUB_MODES["HYPOTHESIZE"]
    assert "propose_by_inversion" in SUB_MODES["HYPOTHESIZE"]
    assert "extend_pattern" in SUB_MODES["HYPOTHESIZE"]
    # All 47 have STEP_REWARDS
    for p, modes in SUB_MODES.items():
        for m in modes:
            assert f"{p}.{m}" in STEP_REWARDS, f"{p}.{m} missing STEP_REWARDS"


def test_recruitment_catalog_covers_all_sub_modes():
    from titan_plugin.logic.meta_reasoning import SUB_MODES
    from titan_plugin.logic.meta_recruitment import RECRUITMENT_CATALOG
    sm_keys = {f"{p}.{m}" for p, modes in SUB_MODES.items() for m in modes}
    cat_keys = set(RECRUITMENT_CATALOG.keys())
    gap = sm_keys - cat_keys
    assert not gap, f"SUB_MODES uncovered by catalog: {gap}"


# ═══════════════════════════════════════════════════════════════════════
# Commit 6 — DynamicRewardAccumulator + α ramp
# ═══════════════════════════════════════════════════════════════════════

def test_accumulator_alpha_disabled_forces_zero():
    from titan_plugin.logic.meta_dynamic_rewards import DynamicRewardAccumulator
    acc = DynamicRewardAccumulator(alpha_ramp_enabled=False)
    # Even after many outcomes, α stays 0
    for _ in range(5000):
        acc.record_single_step("social", "FORMULATE", "define", 0.5)
    assert acc.current_alpha() == 0.0
    assert acc.current_phase() == "disabled"


def test_accumulator_alpha_ramp_boundaries():
    from titan_plugin.logic.meta_dynamic_rewards import DynamicRewardAccumulator
    acc = DynamicRewardAccumulator(
        alpha_ramp_enabled=True,
        phase_0_end=5, phase_1_end=10, phase_2_end=15, phase_3_end=20,
    )
    assert acc.current_alpha() == 0.0
    for _ in range(5):
        acc.record_single_step("social", "FORMULATE", "define", 0.1)
    assert acc.current_alpha() == 0.25
    for _ in range(5):
        acc.record_single_step("social", "FORMULATE", "define", 0.1)
    assert acc.current_alpha() == 0.50
    for _ in range(5):
        acc.record_single_step("social", "FORMULATE", "define", 0.1)
    assert acc.current_alpha() == 0.75
    for _ in range(5):
        acc.record_single_step("social", "FORMULATE", "define", 0.1)
    assert acc.current_alpha() == 1.0


def test_accumulator_cold_start_returns_static():
    from titan_plugin.logic.meta_dynamic_rewards import DynamicRewardAccumulator
    acc = DynamicRewardAccumulator(
        alpha_ramp_enabled=True,
        phase_0_end=5, phase_1_end=100, phase_2_end=200, phase_3_end=300,
        cold_start_n=10,
    )
    # Below cold_start_n=10 → static
    for _ in range(5):
        acc.record_single_step("social", "FORMULATE", "define", 1.0)
    b = acc.blend_step_reward(
        static_reward=0.05, primitive="FORMULATE",
        sub_mode="define", consumer_context="social")
    assert b == 0.05

    # Above cold_start_n with α=0.25, dynamic=1.0, static=0.05
    for _ in range(15):
        acc.record_single_step("social", "FORMULATE", "define", 1.0)
    b = acc.blend_step_reward(
        static_reward=0.05, primitive="FORMULATE",
        sub_mode="define", consumer_context="social")
    # (1 - 0.25) * 0.05 + 0.25 * 1.0 = 0.2875
    assert abs(b - 0.2875) < 1e-6


def test_accumulator_self_driven_chains_unchanged():
    from titan_plugin.logic.meta_dynamic_rewards import DynamicRewardAccumulator
    acc = DynamicRewardAccumulator(alpha_ramp_enabled=True)
    for _ in range(1000):
        acc.record_single_step("social", "FORMULATE", "define", 1.0)
    # consumer_context=None → static unchanged regardless of α
    assert acc.blend_step_reward(
        static_reward=0.05, primitive="FORMULATE",
        sub_mode="define", consumer_context=None) == 0.05


# ═══════════════════════════════════════════════════════════════════════
# Commit 7 — TIMECHAIN_SIMILAR primitive
# ═══════════════════════════════════════════════════════════════════════

def test_similar_query_dataclass_defaults():
    from titan_plugin.logic.timechain_v2 import SimilarQuery
    q = SimilarQuery(query_vector=[0.1] * 132)
    assert q.threshold == 0.75
    assert q.limit == 10
    assert q.since_hours == 72
    assert q.embedding_version == 0


def test_similar_ranks_by_cosine_and_skips_missing():
    from types import MethodType
    from titan_plugin.logic.timechain_v2 import (
        TimeChainOrchestrator, SimilarQuery,
    )

    class _Mock:
        def recall(self, q):
            return [
                {"block_hash": "b1",
                 "payload": {"context_embedding": [1.0, 0, 0, 0],
                             "summary": "exact"}},
                {"block_hash": "b2",
                 "payload": {"context_embedding": [0.9, 0.1, 0, 0],
                             "summary": "near"}},
                {"block_hash": "b3",
                 "payload": {"context_embedding": [0, 1.0, 0, 0],
                             "summary": "ortho"}},
                {"block_hash": "b4",
                 "payload": {"summary": "no-embedding"}},
            ]

    m = _Mock()
    m.similar = MethodType(TimeChainOrchestrator.similar, m)
    r = m.similar(SimilarQuery(query_vector=[1.0, 0, 0, 0], threshold=0.5))
    assert [x["block_hash"] for x in r] == ["b1", "b2"]
    assert r[0]["similarity"] > r[1]["similarity"]


def test_similar_empty_query_returns_empty():
    from types import MethodType
    from titan_plugin.logic.timechain_v2 import (
        TimeChainOrchestrator, SimilarQuery,
    )

    class _Mock:
        def recall(self, q): return []

    m = _Mock()
    m.similar = MethodType(TimeChainOrchestrator.similar, m)
    assert m.similar(SimilarQuery(query_vector=[])) == []


# ═══════════════════════════════════════════════════════════════════════
# Commit 8 — Social consumer wire
# ═══════════════════════════════════════════════════════════════════════

def test_social_context_builder_30d():
    from titan_plugin.logic.social_narrator import build_social_meta_context_30d
    # With realistic inputs
    vec = build_social_meta_context_30d(
        neuromods={"DA": 0.7, "5HT": 0.5, "NE": 0.6, "GABA": 0.4},
        hormones={"CURIOSITY": 0.3, "CREATIVITY": 0.6},
        chi={"total": 0.7},
        persona_qualities=[0.4, 0.35, 0.38],
        vocab_stats={"vocab_size": 388, "productive": 322},
        pressure={"engagement_1h": 0.3},
        last_post_age_s=1800.0,
    )
    assert len(vec) == 30
    assert all(0.0 <= x <= 1.0 for x in vec)


def test_social_context_builder_defaults_are_valid():
    from titan_plugin.logic.social_narrator import build_social_meta_context_30d
    vec = build_social_meta_context_30d()
    assert len(vec) == 30
    assert all(0.0 <= x <= 1.0 for x in vec)


def test_end_to_end_social_request_outcome_roundtrip():
    """Full social flow: build context → send request → service dry-runs →
    response emitted → outcome sent → counters reflect traffic."""
    from titan_plugin.logic import meta_service_client as msc
    from titan_plugin.logic.meta_service import MetaService
    from titan_plugin.logic.social_narrator import build_social_meta_context_30d

    responses = []
    svc = MetaService(response_emitter=responses.append)

    class _Q:
        def __init__(self): self.msgs = []
        def put_nowait(self, m):
            self.msgs.append(m)
            # Simulate bus delivery
            if m["type"] == "META_REASON_REQUEST":
                svc.handle_request(m)
            elif m["type"] == "META_REASON_OUTCOME":
                svc.handle_outcome(m)

    q = _Q()
    ctx = build_social_meta_context_30d(
        neuromods={"DA": 0.6}, chi={"total": 0.5},
    )
    rid = msc.send_meta_request(
        consumer_id="social",
        question_type="formulate_strategy",
        context_vector=ctx,
        time_budget_ms=300,
        send_queue=q,
        src="spirit",
    )
    msc.send_meta_outcome(
        request_id=rid, consumer_id="social",
        outcome_reward=0.0,  # Session 1: neutral
        send_queue=q, src="spirit",
    )
    status = svc.get_status()
    assert status["counters"]["requests_received"] == 1
    assert status["counters"]["requests_dry_run_resolved"] == 1
    assert status["counters"]["outcomes_received"] == 1
    assert len(responses) == 1
    assert responses[0]["dst"] == "spirit"
    assert responses[0]["payload"]["failure_mode"] == "not_yet_implemented"
