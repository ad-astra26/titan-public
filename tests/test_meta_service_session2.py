"""
F-phase Session 2 test suite — validates the pieces shipped in Session 2:

- 10 resolver categories register cleanly + stale count drops
- 7 context builders (language/knowledge/reasoning/coding/self_model/
  emotional/dreaming) return exactly 30 floats in [0, 1]
- 7 signed outcome computers respect [-1, +1] bounds + formulas
- /v4/meta-service/{queue,recruitment,rewards,timechain} endpoints registered

These tests complement test_meta_service_session1.py (which covers the
Session 1 infra) and must not break any existing tests.
"""
from __future__ import annotations

import math

import pytest

from titan_plugin.logic.meta_consumer_contexts import (
    build_language_meta_context_30d,
    build_knowledge_meta_context_30d,
    build_reasoning_meta_context_30d,
    build_coding_meta_context_30d,
    build_self_model_meta_context_30d,
    build_emotional_meta_context_30d,
    build_dreaming_meta_context_30d,
    compute_outcome_language,
    compute_outcome_knowledge,
    compute_outcome_reasoning,
    compute_outcome_coding,
    compute_outcome_self_model,
    compute_outcome_emotional,
    compute_outcome_dreaming,
)
from titan_plugin.logic.meta_recruitment import (
    KNOWN_RESOLVER_CATEGORIES, MetaRecruitment,
)
from titan_plugin.logic.meta_resolvers import (
    register_default_resolvers, get_supported_categories,
)


# ──────────────────────────────────────────────────────────────────────
# Resolvers — Session 2 Commit 1
# ──────────────────────────────────────────────────────────────────────

def test_resolver_supported_categories_cover_known():
    """All KNOWN_RESOLVER_CATEGORIES have a factory in meta_resolvers."""
    supported = set(get_supported_categories())
    assert supported == set(KNOWN_RESOLVER_CATEGORIES)


def test_resolver_registration_drops_stale_to_zero():
    """Registering all 10 default resolvers eliminates stale recruiters."""
    mr = MetaRecruitment()
    before = mr.catalog_health_check()
    assert before["stale_recruiter_count"] > 0  # 25 expected in Session 1 shape

    registered = register_default_resolvers(mr)
    assert all(registered.values()), \
        f"Some resolvers failed to register: {registered}"

    after = mr.catalog_health_check()
    assert after["stale_recruiter_count"] == 0
    assert after["resolver_categories_missing"] == []


def test_resolver_returns_shell_success_for_known_name():
    """reasoning.DECOMPOSE resolver returns success=True + action hint."""
    mr = MetaRecruitment()
    register_default_resolvers(mr)
    resolver = mr._resolvers.get("reasoning")
    assert resolver is not None
    result = resolver("DECOMPOSE", {})
    assert result["success"] is True
    assert "DECOMPOSE" in result["recruiter"]
    assert result["output"]["primitive_hint"] == "DECOMPOSE"


def test_resolver_returns_failure_for_unknown_name():
    """reasoning resolver with unknown op returns success=False without raising."""
    mr = MetaRecruitment()
    register_default_resolvers(mr)
    resolver = mr._resolvers.get("reasoning")
    result = resolver("FROBULATE", {})
    assert result["success"] is False
    assert "unknown" in result["reason"].lower()


def test_resolver_timechain_ops():
    """timechain resolver accepts all 5 bus ops per rFP §9."""
    mr = MetaRecruitment()
    register_default_resolvers(mr)
    resolver = mr._resolvers.get("timechain")
    for op in ("recall", "check", "compare", "aggregate", "similar"):
        r = resolver(op, {})
        assert r["success"] is True
        assert r["output"]["timechain_op"] == op


def test_resolver_idempotent_registration():
    """Calling register_default_resolvers twice doesn't double-bind or error."""
    mr = MetaRecruitment()
    register_default_resolvers(mr)
    first = set(mr._resolvers.keys())
    register_default_resolvers(mr)
    second = set(mr._resolvers.keys())
    assert first == second
    assert mr.catalog_health_check()["stale_recruiter_count"] == 0


# ──────────────────────────────────────────────────────────────────────
# Context builders — Session 2 Commit 1 (§16.2-§16.8)
# ──────────────────────────────────────────────────────────────────────

_BUILDERS = [
    ("language", build_language_meta_context_30d),
    ("knowledge", build_knowledge_meta_context_30d),
    ("reasoning", build_reasoning_meta_context_30d),
    ("coding", build_coding_meta_context_30d),
    ("self_model", build_self_model_meta_context_30d),
    ("emotional", build_emotional_meta_context_30d),
    ("dreaming", build_dreaming_meta_context_30d),
]


@pytest.mark.parametrize("name,builder", _BUILDERS)
def test_context_builders_return_exactly_30_floats(name, builder):
    vec = builder()
    assert len(vec) == 30, f"{name} returned {len(vec)}"
    assert all(isinstance(x, float) for x in vec), f"{name} non-float"
    assert all(0.0 <= x <= 1.0 for x in vec), f"{name} out-of-range"


@pytest.mark.parametrize("name,builder", _BUILDERS)
def test_context_builders_accept_partial_inputs(name, builder):
    """Builders tolerate missing dicts (neutral 0.5 defaults)."""
    # reasoning uses policy_input, not neuromods. Others accept neuromods.
    if name == "reasoning":
        vec = builder(policy_input=[0.5] * 18)
    else:
        vec = builder(neuromods={"DA": 0.7, "5HT": 0.4})
    assert len(vec) == 30


def test_reasoning_builder_leverages_policy_input():
    """reasoning builder folds in first 18 dims of policy_input."""
    pi = [0.1, 0.2, 0.3] + [0.5] * 15 + [0.9, 0.9]
    vec = build_reasoning_meta_context_30d(policy_input=pi)
    assert len(vec) == 30
    assert vec[0] == pytest.approx(0.1, abs=0.01)
    assert vec[1] == pytest.approx(0.2, abs=0.01)
    assert vec[2] == pytest.approx(0.3, abs=0.01)


def test_emotional_builder_uses_all_8_anchors():
    """emotional builder first 8 dims = V for each EMOT_ANCHOR."""
    anchors = {
        "FLOW":              {"V": 0.8},
        "PEACE":             {"V": 0.3},
        "CURIOSITY":         {"V": 0.6},
        "GRIEF":             {"V": 0.1},
        "WONDER":            {"V": 0.7},
        "IMPASSE_TENSION":   {"V": 0.4},
        "RESOLUTION":        {"V": 0.5},
        "LOVE":              {"V": 0.9},
    }
    vec = build_emotional_meta_context_30d(anchors=anchors)
    assert vec[0] == pytest.approx(0.8, abs=0.01)
    assert vec[7] == pytest.approx(0.9, abs=0.01)


def test_knowledge_builder_primitive_name_flag():
    """knowledge builder flags topics containing internal primitive names."""
    vec_internal = build_knowledge_meta_context_30d(topic="FORMULATE depth")
    vec_external = build_knowledge_meta_context_30d(topic="tea ceremony")
    # Second dim is contains_primitive_name flag per §16.3
    assert vec_internal[1] == 1.0
    assert vec_external[1] == 0.0


# ──────────────────────────────────────────────────────────────────────
# Outcome computers — Session 2 Commit 1
# ──────────────────────────────────────────────────────────────────────

def test_outcome_language_regression_is_negative():
    r = compute_outcome_language({"concept_regressed": True})
    assert r == pytest.approx(-0.7, abs=0.01)


def test_outcome_language_fast_grounding_is_high():
    r = compute_outcome_language({
        "concept_grounded": True,
        "grounding_strength": 0.9,
        "time_to_ground_s": 5,
    })
    assert 0.5 < r <= 1.0


def test_outcome_knowledge_contradicts_prior_is_negative():
    r = compute_outcome_knowledge({"contradicts_prior": True})
    assert r == pytest.approx(-0.6, abs=0.01)


def test_outcome_knowledge_strong_gain_penalizes_bandwidth():
    r_cheap = compute_outcome_knowledge({
        "pre_confidence": 0.2, "post_confidence": 0.8,
        "bytes_used": 1_000_000,  # 1 MB cheap
    })
    r_expensive = compute_outcome_knowledge({
        "pre_confidence": 0.2, "post_confidence": 0.8,
        "bytes_used": 100_000_000,  # 100 MB expensive
    })
    assert r_cheap > r_expensive
    assert r_expensive >= 0.6  # still positive


def test_outcome_reasoning_maps_0_1_to_signed():
    # task_success=1.0, conf=1.0, steps=5 → all high → should be near +1
    r_win = compute_outcome_reasoning({
        "task_success": 1.0, "final_confidence": 1.0, "steps": 5,
    })
    r_loss = compute_outcome_reasoning({
        "task_success": 0.0, "final_confidence": 0.0, "steps": 20,
    })
    assert r_win > 0.8
    assert r_loss < -0.8
    assert -1.0 <= r_win <= 1.0
    assert -1.0 <= r_loss <= 1.0


def test_outcome_coding_regression_is_strongly_negative():
    r = compute_outcome_coding({"regressed_from_baseline": True})
    assert r == pytest.approx(-0.8, abs=0.01)


def test_outcome_self_model_coherence_delta_sign_propagates():
    r_up = compute_outcome_self_model(
        pre={"chi_coherence": 0.4, "self_prediction_accuracy": 0.5},
        post={"chi_coherence": 0.6, "self_prediction_accuracy": 0.6},
    )
    r_down = compute_outcome_self_model(
        pre={"chi_coherence": 0.7, "self_prediction_accuracy": 0.7},
        post={"chi_coherence": 0.3, "self_prediction_accuracy": 0.3},
    )
    assert r_up > 0
    assert r_down < 0
    assert -1.0 <= r_up <= 1.0
    assert -1.0 <= r_down <= 1.0


def test_outcome_emotional_anchor_stabilized_is_positive():
    r = compute_outcome_emotional(
        pre={"anchors": {"FLOW": {"V": 0.3, "variance": 0.4}}},
        post={"anchors": {"FLOW": {"V": 0.5, "variance": 0.2}}},
        anchor="FLOW",
    )
    assert r > 0


def test_outcome_emotional_anchor_destabilized_is_negative():
    r = compute_outcome_emotional(
        pre={"anchors": {"FLOW": {"V": 0.6, "variance": 0.2}}},
        post={"anchors": {"FLOW": {"V": 0.2, "variance": 0.5}}},
        anchor="FLOW",
    )
    assert r < 0


def test_outcome_dreaming_better_distillation_is_positive():
    r = compute_outcome_dreaming(
        pre_sleep={
            "distill_pass_rate": 0.5, "variance_samples_count": 200,
            "creativity_signal": 0.4,
        },
        post_wake={
            "distill_pass_rate": 0.7, "variance_samples_count": 100,
            "creativity_signal": 0.6,
        },
    )
    assert r > 0


@pytest.mark.parametrize("computer,args", [
    (compute_outcome_language, ({},)),
    (compute_outcome_knowledge, ({},)),
    (compute_outcome_reasoning, ({},)),
    (compute_outcome_coding, ({},)),
    (compute_outcome_self_model, ({}, {})),
    (compute_outcome_emotional, ({}, {})),
    (compute_outcome_dreaming, ({}, {})),
])
def test_all_outcome_computers_respect_signed_bounds(computer, args):
    """Any outcome computer, any input, stays in [-1, +1]."""
    r = computer(*args)
    assert -1.0 <= r <= 1.0


# ──────────────────────────────────────────────────────────────────────
# Dashboard endpoints — Session 2 Commit 5
# ──────────────────────────────────────────────────────────────────────

def test_four_new_meta_service_endpoints_registered():
    """All 4 Session 2 endpoints registered on the dashboard router."""
    from titan_plugin.api.dashboard import router
    paths = {r.path for r in router.routes if hasattr(r, "path")}
    for p in (
        "/v4/meta-service",           # Session 1
        "/v4/meta-service/queue",      # Session 2
        "/v4/meta-service/recruitment", # Session 2
        "/v4/meta-service/rewards",    # Session 2
        "/v4/meta-service/timechain",  # Session 2
    ):
        assert p in paths, f"Missing endpoint: {p}"


# ──────────────────────────────────────────────────────────────────────
# Integration sanity — consumer_id ↔ home_worker ↔ client
# ──────────────────────────────────────────────────────────────────────

def test_consumer_home_worker_map_covers_all_known_consumers():
    """consumer_home_worker in titan_params.toml maps every KNOWN_CONSUMER."""
    import tomllib
    from pathlib import Path
    from titan_plugin.logic.meta_service_client import KNOWN_CONSUMERS

    cfg_path = (Path(__file__).parent.parent
                / "titan_plugin" / "titan_params.toml")
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    mapping = cfg.get("meta_service_interface", {}).get(
        "consumer_home_worker", {})
    for c in KNOWN_CONSUMERS:
        assert c in mapping, f"Consumer {c!r} has no home_worker mapping"


def test_meta_consumer_contexts_covers_all_7_session2_consumers():
    """Every Session 2 consumer has a build_*_30d helper importable."""
    from titan_plugin.logic import meta_consumer_contexts as mcc
    for fn_name in (
        "build_language_meta_context_30d",
        "build_knowledge_meta_context_30d",
        "build_reasoning_meta_context_30d",
        "build_coding_meta_context_30d",
        "build_self_model_meta_context_30d",
        "build_emotional_meta_context_30d",
        "build_dreaming_meta_context_30d",
    ):
        assert callable(getattr(mcc, fn_name)), f"{fn_name} missing"


def test_end_to_end_language_request_roundtrip():
    """language consumer sends a META_REASON_REQUEST via the same client
    path social uses; request_id returned; validators accept the context."""
    from queue import Queue
    from titan_plugin.logic.meta_service_client import send_meta_request
    from titan_plugin.logic.meta_consumer_contexts import (
        build_language_meta_context_30d)
    q = Queue()
    ctx = build_language_meta_context_30d(
        neuromods={"DA": 0.5, "5HT": 0.6, "NE": 0.4, "GABA": 0.5},
        vocab_stats={"vocab_size": 389, "productive": 318})
    req_id = send_meta_request(
        consumer_id="language",
        question_type="recall_context",
        context_vector=ctx,
        time_budget_ms=500,
        send_queue=q,
        src="test",
    )
    assert req_id  # UUID4 string
    assert q.qsize() == 1
    msg = q.get_nowait()
    assert msg["type"] == "META_REASON_REQUEST"
    assert msg["dst"] == "spirit"
    assert msg["payload"]["consumer_id"] == "language"
    assert len(msg["payload"]["context_vector"]) == 30
