"""§7.E — offline learned grounding combiner (self-gating + integration).

Covers:
  • inactive by default → EngramStore falls back to the §7.D blend
  • the self-gating guard: all-positive / too-few-samples data NEVER activates
    (the live near-degenerate data must keep the blend — short-circuits before
    any sklearn import)
  • separable data → the logistic ACTIVATES, beats the blend on held-out AUC,
    and discriminates at inference
  • save/load round-trips the active model; inference is dependency-free O(1)
  • EngramStore.recompute_population_groundedness uses combiner.score when active
"""
from __future__ import annotations

import json
import math
import queue

import pytest

from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.engram_store import (
    EngramStore,
    reduce_population_to_scalars,
    _BlendParams,
)
from titan_hcl.synthesis.grounding_combiner import GroundingCombiner, AXES
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter


def _blend_scorer(axes_list):
    """The real §7.D percentile-blend baseline (what the learned model replaces)."""
    rows = [{"key": i, "used": a[0], "verified": a[1], "felt": a[2], "fluent": a[3]}
            for i, a in enumerate(axes_list)]
    sc = reduce_population_to_scalars(rows, _BlendParams())
    return [sc.get(i, 0.5) for i in range(len(axes_list))]


def _separable_events(n=260):
    """cited ⇔ fluent>0.5; used/felt are noise wrt the label. The logistic should
    learn fluent (AUC≈1) and clearly beat the used-dominated §7.D blend."""
    ev = []
    for i in range(n):
        cited = (i % 2 == 0)
        fluent = 0.8 if cited else 0.2
        used = (i % 7) / 7.0     # varies but uncorrelated with parity
        felt = (i % 5) / 5.0
        ev.append(((used, 0.0, felt, fluent), cited))
    return ev


# ── Self-gating ──────────────────────────────────────────────────────────────
def test_fresh_combiner_inactive_and_scores_zero():
    c = GroundingCombiner()
    assert c.is_active() is False
    assert c.score({"used": 0.9, "verified": 0, "felt": 0, "fluent": 0.9}) == 0.0
    assert AXES == ("used", "verified", "felt", "fluent")


def test_all_positive_data_stays_inactive():
    # The live failure mode: every event cited=True → no negative class.
    ev = [((0.5, 0.0, 0.0, 0.5), True) for _ in range(300)]
    c = GroundingCombiner()
    m = c.train(ev, blend_scorer=lambda xs: [0.5] * len(xs))
    assert m["activated"] is False
    assert m["reason"] == "insufficient_data"
    assert m["neg"] == 0
    assert not c.is_active()


def test_too_few_samples_stays_inactive():
    ev = [((0.5, 0.0, 0.0, 0.8 if i % 2 == 0 else 0.2), i % 2 == 0)
          for i in range(100)]   # 100 < _MIN_SAMPLES (200)
    c = GroundingCombiner()
    m = c.train(ev, blend_scorer=_blend_scorer)
    assert m["reason"] == "insufficient_data"
    assert not c.is_active()


def test_separable_data_activates_and_beats_blend():
    c = GroundingCombiner()
    m = c.train(_separable_events(), blend_scorer=_blend_scorer, clock=lambda: 42.0)
    assert m["activated"] is True
    assert c.is_active()
    assert m["auc_learned"] >= 0.55
    assert m["auc_learned"] >= m["auc_blend"] + 0.02
    # Discriminates at inference: high fluent scores above low fluent.
    hi = c.score({"used": 0.5, "verified": 0, "felt": 0.5, "fluent": 0.8})
    lo = c.score({"used": 0.5, "verified": 0, "felt": 0.5, "fluent": 0.2})
    assert hi > lo
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
    assert c.meta["trained_at"] == 42.0


# ── Persistence + dependency-free inference ──────────────────────────────────
def test_save_load_roundtrips_active_model(tmp_path):
    c = GroundingCombiner()
    c.train(_separable_events(), blend_scorer=_blend_scorer)
    assert c.is_active()
    p = str(tmp_path / "gc.json")
    assert c.save(p) is True
    c2 = GroundingCombiner.load(p)
    assert c2.is_active()
    s = {"used": 0.4, "verified": 0, "felt": 0.3, "fluent": 0.7}
    assert c2.score(s) == pytest.approx(c.score(s), abs=1e-9)


def test_load_missing_file_is_inactive(tmp_path):
    c = GroundingCombiner.load(str(tmp_path / "absent.json"))
    assert not c.is_active()
    assert c.score({"used": 1, "verified": 1, "felt": 1, "fluent": 1}) == 0.0


def test_score_matches_manual_sigmoid():
    c = GroundingCombiner(weights=[10.0, 0.0, 0.0, 0.0], bias=-5.0, active=True)
    assert c.is_active()
    assert c.score({"used": 0.9, "verified": 0, "felt": 0, "fluent": 0}) == pytest.approx(
        1.0 / (1.0 + math.exp(-(10 * 0.9 - 5))), abs=1e-9)


# ── EngramStore integration (reduce uses combiner when active) ───────────────
def test_recompute_uses_combiner_when_active(tmp_path):
    p = str(tmp_path / "gc.json")
    # Pre-save an ACTIVE combiner: groundedness = sigmoid(10·used − 5).
    GroundingCombiner(weights=[10.0, 0.0, 0.0, 0.0], bias=-5.0, active=True).save(p)
    g = TitanKnowledgeGraph(str(tmp_path / "f.kuzu"))
    w = OuterMemoryWriter(send_queue=queue.Queue(), src="combiner_int_test")
    store = EngramStore(g, w, clock=lambda: 1.0, combiner_path=p)
    assert store._combiner.is_active()

    store.create_concept("hi", "Hi", memory_type="meta")
    store.create_concept("lo", "Lo", memory_type="meta")
    g.spine_update_groundedness("hi", 1, 0.5,
                                axes={"used": 0.9, "verified": 0, "felt": 0, "fluent": 0})
    g.spine_update_groundedness("lo", 1, 0.5,
                                axes={"used": 0.1, "verified": 0, "felt": 0, "fluent": 0})

    store.recompute_population_groundedness()
    snap = str(tmp_path / "s.json")
    store.export_snapshot(snap)
    rows = {r["concept_id"]: r["groundedness"]
            for r in json.load(open(snap))["concepts"]}
    sig = lambda z: 1.0 / (1.0 + math.exp(-z))
    assert rows["hi"] == pytest.approx(sig(10 * 0.9 - 5), abs=1e-6)   # ≈0.982
    assert rows["lo"] == pytest.approx(sig(10 * 0.1 - 5), abs=1e-6)   # ≈0.018


def test_train_grounding_combiner_via_store_activates_and_persists(tmp_path):
    p = str(tmp_path / "gc.json")
    g = TitanKnowledgeGraph(str(tmp_path / "f.kuzu"))
    w = OuterMemoryWriter(send_queue=queue.Queue(), src="combiner_train_test")
    store = EngramStore(g, w, clock=lambda: 7.0, combiner_path=p)
    assert not store._combiner.is_active()   # blend until trained

    m = store.train_grounding_combiner(_separable_events())
    assert m["activated"] is True
    assert store._combiner.is_active()        # hot-swapped in place
    assert GroundingCombiner.load(p).is_active()   # persisted
