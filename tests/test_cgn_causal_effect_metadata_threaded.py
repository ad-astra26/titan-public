"""BUG-CGN-CAUSAL-EFFECT-METADATA-DROPPED (Fix 1) — record_experience must thread
the emit's effect metadata onto the pending CGNTransition so the causal effect
extractor (haov_causal_generator.py:48-107) sees the real domain-effect delta
instead of falling back to the degenerate reward-bucket.

Before the fix: record_experience neither accepted nor preserved `metadata`, so all
experience-path consumers (emotional/meta/self_model/reasoning_strategy/knowledge)
reached observe_for with empty transition.metadata (CausalDBG md_keys=[]).
"""
import warnings
warnings.filterwarnings("ignore")

from titan_hcl.logic.cgn import ConceptGroundingNetwork, CGNConsumerConfig


def _mk_cgn(tmp_path):
    return ConceptGroundingNetwork(state_dir=str(tmp_path))


def test_record_experience_threads_effect_metadata_onto_transition(tmp_path):
    """THE FIX: effect-delta metadata supplied to record_experience reaches the
    buffered transition (record_outcome self-match preserves it)."""
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="emotional"))

    cgn.record_experience(
        consumer="emotional", concept_id="joy", reward=0.6,
        action=3, state=[0.0] * 30,
        metadata={"urgency_delta": 0.7, "action_name": "joy"},
    )

    txs = [t for t in cgn._buffer._buffer
           if t.consumer == "emotional" and t.concept_id == "joy"]
    assert txs, "no transition buffered for emotional/joy"
    t = txs[-1]
    # record_outcome self-matched the pending → real reward applied.
    assert abs(t.reward - 0.6) < 1e-6, f"reward not applied on self-match: {t.reward}"
    # THE REGRESSION GUARD: the effect-delta reached the transition (was dropped).
    assert t.metadata.get("urgency_delta") == 0.7, (
        f"effect metadata dropped — md_keys={list(t.metadata.keys())}")


def test_outcome_context_still_merges_into_metadata(tmp_path):
    """The existing outcome_context channel still merges onto the same dict
    (record_outcome:597) — both channels co-exist."""
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="language"))

    cgn.record_experience(
        consumer="language", concept_id="word1", reward=0.4,
        metadata={"conf_delta": 0.2},
        outcome_context={"extra_signal": 1.0},
    )

    txs = [t for t in cgn._buffer._buffer if t.concept_id == "word1"]
    assert txs
    md = txs[-1].metadata
    assert md.get("conf_delta") == 0.2          # from emit metadata (the fix)
    assert md.get("extra_signal") == 1.0        # from outcome_context (existing)


def test_no_metadata_defaults_empty_no_crash(tmp_path):
    """Backward-compat: omitting metadata yields an empty dict, prior behavior."""
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="reasoning_strategy"))
    cgn.record_experience(consumer="reasoning_strategy", concept_id="c1", reward=0.5)
    txs = [t for t in cgn._buffer._buffer if t.concept_id == "c1"]
    assert txs and isinstance(txs[-1].metadata, dict)


def test_meta_chain_success_yields_real_effect_not_reward_bucket():
    """Fix 2 (meta): once chain_success is in the metadata (now emitted by
    meta_reasoning), the meta extractor forms a real chain-outcome effect
    instead of the degenerate reward-bucket — combined with Fix 1 this makes
    meta a fully-working causal consumer."""
    import numpy as np
    from titan_hcl.logic.haov_causal_generator import extract_effect
    from titan_hcl.logic.cgn_types import CGNTransition

    def _t(md, reward):
        return CGNTransition(
            consumer="meta", concept_id="p",
            state=np.zeros(30, dtype=np.float32), action=0,
            action_params=np.zeros(4, dtype=np.float32),
            reward=reward, metadata=md)

    assert extract_effect("meta", _t({"chain_success": True}, 0.5), 0.5) == "chain_success"
    assert extract_effect("meta", _t({"chain_success": False}, 0.1), 0.1) == "chain_failure"
    # pre-Fix-2 (no key) degenerates to reward-bucket — the bug we fixed:
    assert extract_effect("meta", _t({"chain_id": 1}, 0.5), 0.5) == "strong_positive"
