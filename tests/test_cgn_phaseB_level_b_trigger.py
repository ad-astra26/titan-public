"""Phase B B-3 — Level-B abstraction trigger (RFP_cgn_enhancements §9.2).

CGN_CONCEPT_GROUNDED events accumulate in the meta engine; when ≥ threshold
multi-consumer concepts have matured AND a neuromod gate opens (CURIOSITY or
REFLECTION high), synthesize_level_b() builds a cross-consumer composition into
the store ABOVE CGN (not a CGN row).

Run: python -m pytest tests/test_cgn_phaseB_level_b_trigger.py -v -p no:anchorpy
"""
import warnings
warnings.filterwarnings("ignore")


def _make_engine(tmp_path):
    from titan_hcl.logic.meta_reasoning import MetaReasoningEngine
    from titan_hcl.logic.cross_consumer_composition import (
        CrossConsumerCompositionStore)
    eng = MetaReasoningEngine(config={"level_b_concept_threshold": 3}, send_queue=None)
    # Isolate the store to a temp path (don't touch the real data/ file).
    eng._composition_store = CrossConsumerCompositionStore(
        save_path=str(tmp_path / "ccc.json"))
    return eng


def test_note_accumulates(tmp_path):
    eng = _make_engine(tmp_path)
    assert eng._concept_grounded_since_levelb == 0
    eng.note_concept_grounded({"concept_id": "COLD", "consumers": ["language", "reasoning"]})
    eng.note_concept_grounded({"concept_id": "edge", "consumers": ["reasoning", "social"]})
    assert eng._concept_grounded_since_levelb == 2
    eng.note_concept_grounded({"concept_id": "", "consumers": []})  # ignored (no id)
    assert eng._concept_grounded_since_levelb == 2


def test_gate_requires_count_and_neuromod(tmp_path):
    eng = _make_engine(tmp_path)
    eng.note_concept_grounded({"concept_id": "a", "consumers": ["language"]})
    eng.note_concept_grounded({"concept_id": "b", "consumers": ["reasoning"]})
    # Below threshold (2 < 3) → no trigger even with high curiosity.
    assert eng.should_trigger_level_b({"CURIOSITY": 0.9}) is False
    eng.note_concept_grounded({"concept_id": "c", "consumers": ["social"]})
    # At threshold but LOW neuromods → still no trigger.
    assert eng.should_trigger_level_b({"CURIOSITY": 0.1, "REFLECTION": 0.1}) is False
    # At threshold + high curiosity → trigger.
    assert eng.should_trigger_level_b({"CURIOSITY": 0.8}) is True
    # REFLECTION gate also opens it.
    assert eng.should_trigger_level_b({"REFLECTION": 0.8}) is True


def test_synthesize_stores_composition(tmp_path):
    eng = _make_engine(tmp_path)
    for cid, cons in [("COLD", ["language", "reasoning"]),
                      ("edge", ["reasoning", "social"]),
                      ("distance", ["social", "language"])]:
        eng.note_concept_grounded({"concept_id": cid, "consumers": cons})
    assert eng.should_trigger_level_b({"CURIOSITY": 0.8})
    bid = eng.synthesize_level_b()
    assert bid and bid.startswith("ccc_")
    assert eng._composition_store.count() == 1
    comp = eng._composition_store.get(bid)
    assert set(comp.member_concepts) == {"COLD", "edge", "distance"}
    assert set(comp.member_consumers) == {"language", "reasoning", "social"}
    # Counter reset + pending drained after synthesis.
    assert eng._concept_grounded_since_levelb == 0
    assert eng._pending_concept_grounded == []


def test_synthesize_needs_two_distinct_concepts(tmp_path):
    eng = _make_engine(tmp_path)
    # Same concept twice → only 1 distinct → no composition.
    eng.note_concept_grounded({"concept_id": "x", "consumers": ["language"]})
    eng.note_concept_grounded({"concept_id": "x", "consumers": ["reasoning"]})
    eng.note_concept_grounded({"concept_id": "x", "consumers": ["social"]})
    assert eng.should_trigger_level_b({"CURIOSITY": 0.8})
    assert eng.synthesize_level_b() is None
    assert eng._composition_store.count() == 0


if __name__ == "__main__":
    import tempfile, pathlib
    for fn in (test_note_accumulates, test_gate_requires_count_and_neuromod,
               test_synthesize_stores_composition,
               test_synthesize_needs_two_distinct_concepts):
        with tempfile.TemporaryDirectory() as td:
            fn(pathlib.Path(td))
    print("OK — Phase B B-3 trigger checks passed")
