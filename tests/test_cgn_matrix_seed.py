"""AUDIT_cgn_trigger_wiring_design_20260610 M3 (SPEC §5.4 matrix-seed) — every
chain walks a real concept.

The weeks-old monoculture root cause: mechanical triggers (experience_pressure
89.7%) fire with NO concept → chains collapse to FORMULATE.define (§5.3) →
0/22495 chains carried a grounding_concept. M3 seeds such chains from a recent
matured CGN-matrix concept (_pending_concept_grounded, ≥2-consumer-grounded) so
they walk a real concept (§1.3). These tests pin:
  1. flag default-on, configurable off (§5 rollback).
  2. a matrix-seeded chain gets grounding_concept + NO hardcoded entry primitive
     (the first step EMERGES from grounded V per-Titan — INV-EMERGENCE; a fixed
     entry would suppress sovereign divergence + reinforce T2's RECALL) and NO
     grounding_request_id → the ARC-4 outcome emit stays gated.
  3. round-robin walks the buffer for breadth.
  4. an active consumer-grounding (Path#0) is NOT displaced by matrix-seed.

Run: python -m pytest tests/test_cgn_matrix_seed.py -v -p no:anchorpy
"""
import warnings
warnings.filterwarnings("ignore")

from titan_hcl.logic.meta_reasoning import MetaReasoningEngine, META_PRIMITIVES


def _make_engine(**cfg):
    return MetaReasoningEngine(config=cfg, send_queue=None)


def _seed_payload_from_buffer(eng):
    """Replicate tick()'s matrix-seed pick (round-robin over the matured-concept
    buffer) so the seeding mechanism is unit-testable without the full tick."""
    buf = eng._pending_concept_grounded
    seed = buf[eng._matrix_seed_idx % len(buf)]
    eng._matrix_seed_idx += 1
    cid = str((seed or {}).get("concept_id", ""))[:128]
    return {"consumer": "", "concept_id": cid, "entry_primitive": ""}


def test_matrix_seed_flag_default_on_and_configurable():
    assert _make_engine()._matrix_seed_enabled is True
    assert _make_engine(matrix_seed_enabled=False)._matrix_seed_enabled is False


def test_matrix_seeded_chain_walks_concept_no_arc4_emit():
    eng = _make_engine()
    eng._pending_concept_grounded = [
        {"concept_id": "epigenetic inheritance", "consumers": ["knowledge", "language"]},
    ]
    eng._active_grounding = _seed_payload_from_buffer(eng)
    eng._start_chain("experience_pressure(3230)+matrix_seed(epigenetic inheritance)",
                     [0.0] * 132)

    # The chain now WALKS the concept (§1.3) instead of collapsing to FORMULATE.
    assert eng.state.grounding_concept == "epigenetic inheritance"
    assert eng.state.entity_refs.get("current_topic") == "epigenetic inheritance"
    # NO hardcoded entry primitive — the walk (incl. step 1) emerges from
    # grounded V per-Titan (INV-EMERGENCE; a fixed entry would suppress
    # sovereign divergence + reinforce a Titan's own monoculture).
    assert eng.state.entry_primitive == ""
    # No consumer request → grounding_request_id empty → ARC-4 emit stays gated
    # (matrix-seed breaks monoculture; it does NOT feed the α-ramp).
    assert eng.state.grounding_request_id == ""
    assert "matrix_seed" in eng.state.trigger_reason


def test_matrix_seed_round_robin_walks_buffer():
    eng = _make_engine()
    eng._pending_concept_grounded = [
        {"concept_id": "alpha"}, {"concept_id": "beta"}, {"concept_id": "gamma"},
    ]
    picks = [_seed_payload_from_buffer(eng)["concept_id"] for _ in range(4)]
    assert picks == ["alpha", "beta", "gamma", "alpha"]   # round-robin breadth


def test_consumer_grounding_not_displaced_by_matrix_seed():
    """Path#0 (a real consumer learning event) keeps priority + its request_id
    (so its ARC-4 outcome still emits); matrix-seed only fills the contentless
    mechanical-trigger void."""
    eng = _make_engine()
    eng._active_grounding = {
        "consumer": "knowledge", "concept_id": "warmth",
        "entry_primitive": "HYPOTHESIZE", "request_id": "req-7",
        "question_type": "hypothesize_cause",
    }
    eng._start_chain("concept_grounding(knowledge:warmth)", [0.0] * 132)
    assert eng.state.grounding_concept == "warmth"
    assert eng.state.grounding_consumer == "knowledge"
    assert eng.state.grounding_request_id == "req-7"        # ARC-4 emit WILL fire
    assert eng.state.entry_primitive == "HYPOTHESIZE"


if __name__ == "__main__":
    test_matrix_seed_flag_default_on_and_configurable()
    test_matrix_seeded_chain_walks_concept_no_arc4_emit()
    test_matrix_seed_round_robin_walks_buffer()
    test_consumer_grounding_not_displaced_by_matrix_seed()
    print("OK — M3 matrix-seed checks passed")
