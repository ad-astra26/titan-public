"""Phase B B-2 — CrossConsumerCompositionStore (RFP_cgn_enhancements §9.2).

Run: python -m pytest tests/test_cross_consumer_composition.py -v -p no:anchorpy
"""
import warnings
warnings.filterwarnings("ignore")

import os

from titan_hcl.logic.cross_consumer_composition import (
    CrossConsumerCompositionStore, _binding_id)


def test_add_and_query(tmp_path):
    s = CrossConsumerCompositionStore(save_path=str(tmp_path / "ccc.json"))
    bid = s.add(["COLD", "edge", "distance"],
                ["language", "reasoning", "social"],
                abstraction_label="boundary", confidence=0.3,
                lineage=["chain_1"], now=1000.0)
    assert bid.startswith("ccc_")
    assert s.count() == 1
    c = s.get(bid)
    assert c.abstraction_label == "boundary"
    assert set(c.member_consumers) == {"language", "reasoning", "social"}
    assert "COLD" in c.member_concepts


def test_idempotent_reinforce(tmp_path):
    s = CrossConsumerCompositionStore(save_path=str(tmp_path / "ccc.json"))
    bid1 = s.add(["a", "b"], ["language"], "x", confidence=0.3, now=1.0)
    bid2 = s.add(["b", "a"], ["reasoning"], "x", confidence=0.3,  # same SET
                 lineage=["chain_2"], now=2.0)
    assert bid1 == bid2          # same concept set → same binding
    assert s.count() == 1        # not duplicated
    c = s.get(bid1)
    assert c.n_reinforced == 1
    assert c.confidence > 0.3    # reinforced upward
    assert set(c.member_consumers) == {"language", "reasoning"}  # merged
    assert "chain_2" in c.lineage


def test_recent_ordering(tmp_path):
    s = CrossConsumerCompositionStore(save_path=str(tmp_path / "ccc.json"))
    s.add(["a"], ["l"], "first", now=1.0)
    s.add(["b"], ["r"], "second", now=2.0)
    s.add(["c"], ["s"], "third", now=3.0)
    labels = [c.abstraction_label for c in s.recent(2)]
    assert labels == ["third", "second"]


def test_persistence_roundtrip(tmp_path):
    p = str(tmp_path / "ccc.json")
    s = CrossConsumerCompositionStore(save_path=p)
    bid = s.add(["COLD", "edge"], ["language", "reasoning"], "boundary",
                confidence=0.5, now=10.0)
    assert s.save()
    assert os.path.exists(p)

    fresh = CrossConsumerCompositionStore(save_path=p)
    assert fresh.count() == 1
    c = fresh.get(bid)
    assert c.abstraction_label == "boundary"
    assert c.confidence == 0.5
    assert set(c.member_concepts) == {"COLD", "edge"}


def test_eviction_bounded(tmp_path):
    s = CrossConsumerCompositionStore(save_path=str(tmp_path / "ccc.json"), max_size=3)
    for i in range(6):
        s.add([f"concept_{i}"], ["language"], f"label_{i}",
              confidence=0.1 + i * 0.1, now=float(i))
    assert s.count() == 3   # bounded


if __name__ == "__main__":
    import tempfile, pathlib
    for fn in (test_add_and_query, test_idempotent_reinforce, test_recent_ordering,
               test_persistence_roundtrip, test_eviction_bounded):
        with tempfile.TemporaryDirectory() as td:
            fn(pathlib.Path(td))
    print("OK — Phase B B-2 store checks passed")
