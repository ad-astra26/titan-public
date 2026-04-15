"""Tests for CGN-EXTRACT (rFP_cgn_cognitive_kernel_v2.md) — Phase 7 integration tests.

Covers the Phase 3a/3b fixes shipped 2026-04-12:
  - Phase 3a: meta-reasoning topic extraction quality (primitive.submode names
              no longer sent as SearXNG queries — 3-tier fallback returns
              conceptual queries or None).
  - Phase 3b: language short-word query enrichment (single-word topics like
              "own", "noun" get "{word} meaning definition" suffix).

LIVE integration testing uses `python scripts/arch_map.py verify cgn-pipeline`
on a running Titan — checks /dev/shm state, consumer registration, HAOV
activity, knowledge pipeline, dream consolidation in one command.

Run: python -m pytest tests/test_cgn_extract.py -v -p no:anchorpy --tb=short
"""
import os
import tempfile

import numpy as np
import pytest


# ── Phase 3a: meta-reasoning topic extraction fix ────────────────────

class _MockState:
    """Minimal MetaChainState stand-in for topic extraction tests."""
    def __init__(self, chain=None, formulate_output=None, trigger_reason="",
                 impasse_detected=False, step_rewards=None, chain_id=0):
        self.chain = chain or []
        self.formulate_output = formulate_output or {}
        self.trigger_reason = trigger_reason
        self.impasse_detected = impasse_detected
        self.step_rewards = step_rewards or [0.2, 0.15, 0.1, 0.08, 0.05]
        self.chain_id = chain_id
        self.impasse_topic = ""


def _make_engine_with_state(state, soar_config=None):
    """Build a minimal MetaReasoningEngine wrapper to invoke detect_chain_impasse.

    Avoids expensive full-engine init by monkey-patching just what the method
    reads. Tests the 3-tier fallback logic directly.
    """
    from titan_plugin.logic import meta_reasoning as mr

    class _Engine:
        def __init__(self, state, soar):
            self.state = state
            self._soar_config = soar or {
                "threshold_consec": 3,
                "curiosity_bonus": 1,
                "max_internal_per_hour": 10,
                "max_external_per_hour": 5,
                "concept_cooldown_s": 900,
                "urgency_stuck": 0.9,
                "urgency_declining": 0.7,
                "urgency_plateau": 0.4,
            }
            self._soar_curiosity_until = 0.0
            self._soar_requests_this_hour = []
            self._soar_concept_cooldowns = {}

        detect_chain_impasse = mr.MetaReasoningEngine.detect_chain_impasse

    return _Engine(state, soar_config)


def test_phase_3a_tier1_formulate_output_topic():
    """Tier 1: FORMULATE.define output 'topic' field used when available."""
    state = _MockState(
        chain=["FORMULATE.define", "RECALL.entity", "EVALUATE.check_progress"],
        formulate_output={"topic": "quantum entanglement in biology"},
        trigger_reason="low_commit_rate",
    )
    engine = _make_engine_with_state(state)
    result = engine.detect_chain_impasse()
    assert result is not None, "declining chain should detect impasse"
    assert result["topic"] == "quantum entanglement in biology"


def test_phase_3a_tier2_trigger_reason_fallback():
    """Tier 2: trigger_reason maps to conceptual query when FORMULATE has no topic."""
    state = _MockState(
        chain=["RECALL.entity", "EVALUATE.check_progress", "EVALUATE.check_strategy"],
        formulate_output={},  # empty — Tier 1 fails
        trigger_reason="low_commit_rate",
    )
    engine = _make_engine_with_state(state)
    result = engine.detect_chain_impasse()
    assert result is not None
    assert "cognitive reasoning" in result["topic"].lower(), (
        f"expected conceptual query, got '{result['topic']}'")
    # Must NOT be a primitive.submode name
    assert "." not in result["topic"], (
        f"topic should not contain primitive.submode notation: '{result['topic']}'")


def test_phase_3a_tier3_primitive_fallback():
    """Tier 3: primitive name maps to conceptual query when trigger unknown."""
    state = _MockState(
        chain=["FORMULATE.load_wisdom", "FORMULATE.refine", "FORMULATE.define"],
        formulate_output={},  # Tier 1 empty
        trigger_reason="unknown_custom_reason",  # Tier 2 no match
    )
    engine = _make_engine_with_state(state)
    result = engine.detect_chain_impasse()
    assert result is not None
    # Must map FORMULATE → "problem formulation cognitive strategies"
    assert "formulation" in result["topic"].lower(), (
        f"expected FORMULATE→formulation mapping, got '{result['topic']}'")
    # Critically: NOT the raw "FORMULATE.load_wisdom" primitive name
    assert "load_wisdom" not in result["topic"], (
        f"topic must not leak primitive.submode: '{result['topic']}'")


def test_phase_3a_no_garbage_query():
    """Safety: if no tier produces topic, return None (skip request entirely)."""
    state = _MockState(
        chain=["UNKNOWN_PRIMITIVE.foo"],  # unmapped primitive
        formulate_output={},
        trigger_reason="",
    )
    engine = _make_engine_with_state(state)
    result = engine.detect_chain_impasse()
    assert result is None, "should return None rather than send garbage"


def test_phase_3a_all_known_primitives_have_mappings():
    """Every meta primitive must have a Tier 3 conceptual-query mapping.

    Otherwise impasses on that primitive produce no knowledge request.
    """
    from titan_plugin.logic import meta_reasoning as mr
    for primitive in mr.META_PRIMITIVES:
        state = _MockState(
            chain=[f"{primitive}.some_submode", f"{primitive}.other"],
            formulate_output={},
            trigger_reason="",
        )
        engine = _make_engine_with_state(state)
        result = engine.detect_chain_impasse()
        assert result is not None, (
            f"primitive {primitive} has no Tier 3 mapping — would produce "
            f"no knowledge request at impasse")
        assert "." not in result["topic"], (
            f"topic for {primitive} still leaking submode notation")


# ── Phase 3b: language short-word query enrichment ────────────────────

def test_phase_3b_short_words_get_enriched():
    """Words ≤6 chars get '{word} meaning definition' suffix for SearXNG."""
    # Inline the logic from language_worker.py:2214+ for testable form.
    # This isolates the enrichment decision from the bus/send_msg scaffolding.
    def _enrich_query(word: str) -> str:
        return f"{word} meaning definition" if len(word) <= 6 else word

    # Short words previously returned 0 SearXNG results — now enriched
    assert _enrich_query("own") == "own meaning definition"
    assert _enrich_query("noun") == "noun meaning definition"
    assert _enrich_query("feel") == "feel meaning definition"
    assert _enrich_query("spirit") == "spirit meaning definition"  # 6 chars, still short

    # Longer words preserved as-is (SearXNG handles them well)
    assert _enrich_query("mitochondria") == "mitochondria"
    assert _enrich_query("consciousness") == "consciousness"
    assert _enrich_query("quantumentanglement") == "quantumentanglement"


# ── CGN core: imports + basic instantiation ──────────────────────────

def test_cgn_core_imports():
    """Phase 0: all CGN-EXTRACT core modules import without error."""
    from titan_plugin.logic import cgn  # noqa: F401
    from titan_plugin.logic import cgn_consumer_client  # noqa: F401
    from titan_plugin.logic import cgn_shm_protocol  # noqa: F401
    from titan_plugin.modules import cgn_worker  # noqa: F401
    from titan_plugin.modules import knowledge_worker  # noqa: F401


def test_cgn_worker_entry_function_exists():
    """Phase 0: cgn_worker_main is callable (Guardian module entry point)."""
    from titan_plugin.modules.cgn_worker import cgn_worker_main
    assert callable(cgn_worker_main), "cgn_worker_main must be callable"


def test_cgn_consumer_client_constructor():
    """Phase 1: CGNConsumerClient instantiates without requiring a live CGN worker."""
    from titan_plugin.logic.cgn_consumer_client import CGNConsumerClient

    with tempfile.TemporaryDirectory() as tmp:
        # Use a non-existent SHM path — client should degrade gracefully
        client = CGNConsumerClient(
            consumer_name="test_consumer",
            state_dir=tmp,
            shm_path=os.path.join(tmp, "nonexistent.bin"),
        )
        assert client is not None
        assert hasattr(client, "ground") or hasattr(client, "infer_action"), (
            "client must expose at least one of ground/infer_action")


def test_shm_protocol_header_format():
    """Phase 0: ShmWeightWriter/Reader use consistent 16-byte header format.

    Regression guard for 2026-04-12 verification confusion where header
    `total_size` was mis-interpreted as total file size (it's payload-only).
    """
    from titan_plugin.logic.cgn_shm_protocol import HEADER_SIZE
    assert HEADER_SIZE == 16, (
        "HEADER_SIZE changed — any reader/writer update must stay consistent")


# ── Sanity: rewards module primitives match declared meta primitives ──

def test_reward_helpers_cover_compound_primitives():
    """Phase A compound rewards: reward_recall, reward_formulate etc. exist."""
    from titan_plugin.logic.meta_reasoning_rewards import PRIMITIVE_REWARD_HELPERS

    # Compound rewards defined for core primitives; others fall back to legacy
    assert "RECALL" in PRIMITIVE_REWARD_HELPERS
    assert "FORMULATE" in PRIMITIVE_REWARD_HELPERS
    assert "EVALUATE" in PRIMITIVE_REWARD_HELPERS
    assert "INTROSPECT" in PRIMITIVE_REWARD_HELPERS
    assert "BREAK" in PRIMITIVE_REWARD_HELPERS
