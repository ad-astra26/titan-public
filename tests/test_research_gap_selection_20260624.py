"""RFP_titan_research_agent §1.4 step 1 — autonomous research-curiosity gap selector.

Unit-tests `rank_research_gaps` (pure ranking) + `EngramStore.lowest_grounded_concepts`
(spine read → rank). The gap selector picks declarative concepts Titan has ENCOUNTERED
but understands LEAST (high `used` × low `groundedness`), each carrying its current
groundedness as the verifiable-research BASELINE.
"""
from __future__ import annotations

from titan_hcl.synthesis.engram_store import EngramStore, rank_research_gaps


def _c(cid, *, used, g, version=1, name="", domain="general"):
    return {"concept_id": cid, "version": version, "name": name or cid,
            "used": used, "groundedness": g, "domain_hint": domain}


# ── rank_research_gaps (pure) ────────────────────────────────────────────────

def test_high_used_low_grounded_ranks_first():
    rows = [
        _c("well_understood", used=0.9, g=0.9),    # met a lot, grasped → salience 0.09
        _c("the_gap", used=0.8, g=0.1),            # met a lot, NOT grasped → salience 0.72
        _c("rare_known", used=0.1, g=0.05),        # barely met → salience 0.095
    ]
    out = rank_research_gaps(rows, n=3)
    assert [r["concept_id"] for r in out][0] == "the_gap"
    assert out[0]["salience"] > out[1]["salience"] >= out[2]["salience"]


def test_baseline_groundedness_is_carried():
    out = rank_research_gaps([_c("z", used=0.5, g=0.27)], n=1)
    assert out[0]["groundedness"] == 0.27  # the before-value the verifier reads
    assert out[0]["domain_hint"] == "general"


def test_latest_version_wins_not_stale_low_g():
    # v1 ungrounded (g=0.05) but v2 grounded (g=0.8): the LATEST (v2) must be used,
    # so this concept is NOT surfaced as a big gap.
    rows = [
        _c("evolving", used=0.7, g=0.05, version=1),
        _c("evolving", used=0.7, g=0.80, version=2),
        _c("real_gap", used=0.6, g=0.10, version=1),
    ]
    out = rank_research_gaps(rows, n=2)
    assert out[0]["concept_id"] == "real_gap"
    evolving = [r for r in out if r["concept_id"] == "evolving"]
    assert evolving and evolving[0]["groundedness"] == 0.80  # latest, not the stale 0.05


def test_min_used_drops_never_encountered():
    rows = [_c("noise", used=0.0, g=0.0), _c("met", used=0.3, g=0.1)]
    out = rank_research_gaps(rows, n=5)
    assert [r["concept_id"] for r in out] == ["met"]  # used=0 noise excluded


def test_n_limit_and_empty():
    assert rank_research_gaps([], n=5) == []
    rows = [_c(f"c{i}", used=0.5, g=i / 10.0) for i in range(10)]
    assert len(rank_research_gaps(rows, n=3)) == 3


def test_clamps_out_of_range_groundedness():
    out = rank_research_gaps([_c("x", used=1.0, g=1.5)], n=1)
    assert out[0]["groundedness"] == 1.0 and out[0]["salience"] == 0.0


# ── EngramStore.lowest_grounded_concepts (spine read → rank) ──────────────────

class _FakeGraph:
    def __init__(self, rows):
        self._rows = rows

    def spine_research_gap_candidates(self):
        return list(self._rows)


class _InlineWriter:
    """@on_writer routes through `_db_writer.submit_sync` — run it inline for tests."""
    def submit_sync(self, fn):
        return fn()


def _store_with(rows):
    store = EngramStore.__new__(EngramStore)   # bypass __init__ (no Kuzu/FAISS needed)
    store._graph = _FakeGraph(rows)
    store._db_writer = _InlineWriter()
    return store


def test_lowest_grounded_concepts_reads_and_ranks():
    store = _store_with([
        _c("grounded", used=0.9, g=0.95),
        _c("gap_a", used=0.8, g=0.10),
        _c("gap_b", used=0.4, g=0.05),
    ])
    out = store.lowest_grounded_concepts(n=2)
    assert [r["concept_id"] for r in out] == ["gap_a", "gap_b"]
    assert out[0]["groundedness"] == 0.10


def test_lowest_grounded_concepts_soft_fails_to_empty():
    class _Boom:
        def spine_research_gap_candidates(self):
            raise RuntimeError("kuzu down")
    store = EngramStore.__new__(EngramStore)
    store._graph = _Boom()
    store._db_writer = _InlineWriter()
    assert store.lowest_grounded_concepts(n=3) == []
