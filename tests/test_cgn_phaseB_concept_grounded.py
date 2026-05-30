"""Phase B (RFP_cgn_enhancements §9.2) — B-1: CGN_CONCEPT_GROUNDED maturity.

The central ConceptGroundingNetwork (cgn_worker) sees every consumer's outcomes
via record_outcome. When a concept_id has been grounded across >= 2 distinct
consumers it has "matured" cross-consumer and is queued for cgn_worker to emit
CGN_CONCEPT_GROUNDED (the Level-B trigger). Each concept fires at most once
(deduped via _matured_emitted, persisted across restarts).

Run: python -m pytest tests/test_cgn_phaseB_concept_grounded.py -v -p no:anchorpy
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from titan_hcl.logic.cgn import ConceptGroundingNetwork, CGNConsumerConfig
from titan_hcl.logic.cgn_types import CGNTransition


def _mk_cgn(tmp_path):
    return ConceptGroundingNetwork(state_dir=str(tmp_path))


def _outcome(cgn, consumer, concept, reward=0.6):
    """Add one unrewarded transition + record its outcome (grows the journey)."""
    cgn._buffer.add(CGNTransition(
        consumer=consumer, concept_id=concept,
        state=np.zeros(30, dtype=np.float32), action=0,
        action_params=np.zeros(8, dtype=np.float32), reward=0.0,
        metadata={"action_name": "reinforce"},
    ))
    cgn.record_outcome(consumer, concept, reward)


def test_single_consumer_does_not_mature(tmp_path):
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="reasoning"))
    _outcome(cgn, "reasoning", "warmth")
    # One consumer only → not cross-consumer mature → nothing queued.
    assert cgn.pop_matured_concepts() == []


def test_two_consumers_mature_and_emit_once(tmp_path):
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="reasoning"))
    cgn.register_consumer(CGNConsumerConfig(name="language"))

    _outcome(cgn, "reasoning", "warmth")
    assert cgn.pop_matured_concepts() == []  # 1 consumer

    _outcome(cgn, "language", "warmth")      # 2nd distinct consumer → mature
    matured = cgn.pop_matured_concepts()
    assert len(matured) == 1
    m = matured[0]
    assert m["concept_id"] == "warmth"
    assert set(m["consumers"]) == {"reasoning", "language"}
    assert m["first_consumer"] == "reasoning"

    # Dedup: pop again returns nothing; a 3rd consumer does NOT re-fire.
    assert cgn.pop_matured_concepts() == []
    cgn.register_consumer(CGNConsumerConfig(name="social"))
    _outcome(cgn, "social", "warmth")
    assert cgn.pop_matured_concepts() == []


def test_matured_emitted_persists_across_restart(tmp_path):
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="reasoning"))
    cgn.register_consumer(CGNConsumerConfig(name="language"))
    _outcome(cgn, "reasoning", "warmth")
    _outcome(cgn, "language", "warmth")
    assert len(cgn.pop_matured_concepts()) == 1
    cgn._save_state()

    # Fresh instance loads matured_emitted → 'warmth' must NOT re-fire.
    fresh = ConceptGroundingNetwork(state_dir=str(tmp_path))
    assert "warmth" in fresh._matured_emitted
    fresh.register_consumer(CGNConsumerConfig(name="reasoning"))
    fresh.register_consumer(CGNConsumerConfig(name="language"))
    _outcome(fresh, "reasoning", "warmth")
    _outcome(fresh, "language", "warmth")
    assert fresh.pop_matured_concepts() == []  # already emitted before restart


if __name__ == "__main__":
    import tempfile, pathlib
    for fn in (test_single_consumer_does_not_mature,
               test_two_consumers_mature_and_emit_once,
               test_matured_emitted_persists_across_restart):
        with tempfile.TemporaryDirectory() as td:
            fn(pathlib.Path(td))
    print("OK — Phase B B-1 maturity checks passed")
