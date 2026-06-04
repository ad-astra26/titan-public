"""Phase D — decomposed axes + population-percentile reduction (RFP §7.D / G4).

Pure-function coverage of the discrimination engine: compute_axes (the 4 absolute
axes), the empirical-CDF percentile ranks, and the variance-gated population blend
that turns the compressed [0.1,0.3] scalar into a discriminating one. The store
integration (recompute_population_groundedness over Kuzu) is exercised by the
consolidation suite.
"""
import math

import pytest

from titan_hcl.synthesis.engram_store import (
    EngramStore,
    _BlendParams,
    _percentile_ranks,
    compute_axes,
    reduce_population_to_scalars,
)


# ── compute_axes (option-1: used = the provisional grounding blend) ──

def test_compute_axes_used_is_the_provisional():
    ax = compute_axes(provisional=0.27)
    assert ax["used"] == pytest.approx(0.27)


def test_compute_axes_verified_nars_c():
    # n/(n+k): 0 evidence → 0; grows with oracle evidence
    assert compute_axes(oracle_evidence=0, nars_k=1.0)["verified"] == 0.0
    assert compute_axes(oracle_evidence=3, nars_k=1.0)["verified"] == pytest.approx(0.75)


def test_compute_axes_felt_and_fluent_clamped():
    ax = compute_axes(felt_coverage=1.5, fluent=-0.2)
    assert ax["felt"] == 1.0 and ax["fluent"] == 0.0


# ── percentile ranks (empirical CDF, floor 1/N — never 0) ──

def test_percentile_ranks_monotonic_cdf():
    assert _percentile_ranks([10.0, 20.0, 30.0, 40.0]) == [0.25, 0.5, 0.75, 1.0]


def test_percentile_ranks_floor_is_one_over_n_not_zero():
    # the minimum value maps to >0 (avoids a 0 scalar that would kill name-match recall)
    r = _percentile_ranks([5.0, 6.0, 7.0])
    assert min(r) == pytest.approx(1 / 3) and max(r) == 1.0


def test_percentile_ranks_empty():
    assert _percentile_ranks([]) == []


# ── reduce: the G4 discrimination property ──

def _rows(used_vals):
    return [{"key": i, "used": u, "verified": 0.0, "felt": 0.0, "fluent": 0.0}
            for i, u in enumerate(used_vals)]


def test_reduce_used_only_widens_spread_out_of_compressed_band():
    # Simulate the live diagnosis: counts that the OLD scalar compressed into
    # [0.1,0.3]. The percentile blend (used the only varying axis) must spread them.
    bp = _BlendParams()
    rows = _rows([math.log1p(2), math.log1p(3), math.log1p(5), math.log1p(9), math.log1p(20)])
    sc = reduce_population_to_scalars(rows, bp)
    vals = [sc[i] for i in range(len(rows))]
    assert max(vals) - min(vals) > 0.3      # demonstrably wider than the old ~0.2 band
    assert max(vals) == pytest.approx(1.0)
    assert vals == sorted(vals)             # ordering tracks `used` (monotonic)
    assert min(vals) > 0.0                  # no zero scalar


def test_reduce_all_flat_population_returns_empty_keeps_provisional():
    # No axis varies → percentile can't discriminate → return {} so the caller
    # KEEPS each Engram's provisional scalar (never zeroes a real Engram).
    bp = _BlendParams()
    flat = [{"key": i, "used": 0.0, "verified": 0.0, "felt": 0.0, "fluent": 0.0}
            for i in range(4)]
    assert reduce_population_to_scalars(flat, bp) == {}


def test_reduce_single_engram_returns_empty():
    # A 1-Engram population has no variance → {} (provisional kept, not zeroed).
    assert reduce_population_to_scalars(
        [{"key": "solo", "used": 1.0, "verified": 0.0, "felt": 0.0, "fluent": 0.0}],
        _BlendParams()) == {}


def test_reduce_variance_gates_flat_axes():
    # used flat, felt VARIES → blend must be driven by felt alone (used excluded),
    # so ordering tracks felt even though w_used > w_felt.
    bp = _BlendParams()
    rows = [
        {"key": "a", "used": 1.0, "verified": 0.0, "felt": 0.1, "fluent": 0.0},
        {"key": "b", "used": 1.0, "verified": 0.0, "felt": 0.5, "fluent": 0.0},
        {"key": "c", "used": 1.0, "verified": 0.0, "felt": 0.9, "fluent": 0.0},
    ]
    sc = reduce_population_to_scalars(rows, bp)
    assert sc["a"] < sc["b"] < sc["c"]      # felt drives ordering (used is flat → gated out)


def test_reduce_multi_axis_blend_when_both_vary():
    bp = _BlendParams(w_used=0.5, w_verified=0.5, w_felt=0.0, w_fluent=0.0, nars_k=1.0)
    rows = [
        {"key": "lo", "used": 0.1, "verified": 0.1, "felt": 0.0, "fluent": 0.0},
        {"key": "hi", "used": 0.9, "verified": 0.9, "felt": 0.0, "fluent": 0.0},
    ]
    sc = reduce_population_to_scalars(rows, bp)
    assert sc["hi"] > sc["lo"]              # both axes agree → hi ranks above lo


def test_reduce_empty():
    assert reduce_population_to_scalars([], _BlendParams()) == {}


# ── population pass: pre-D backfill + whole-population discrimination (option-1) ──

class _StubGraph:
    """Minimal graph exposing the two methods the population pass calls."""

    def __init__(self, rows):
        self._rows = rows
        self.writes = {}  # (cid, ver) -> {"g": scalar, "axes": dict|None}

    def spine_read_engram_axes(self):
        return [dict(r) for r in self._rows]

    def spine_update_groundedness(self, cid, ver, g, *, axes=None):
        self.writes[(cid, int(ver))] = {"g": g, "axes": axes}
        return True


def test_population_backfills_pre_d_engrams_and_discriminates():
    # 4 pre-D Engrams: axis_used=0 (B default) but a real provisional groundedness
    # in the compressed [0.1,0.3] band. The pass must backfill axis_used from
    # groundedness AND spread the population out of [0.1,0.3].
    rows = [
        {"concept_id": f"c{i}", "version": 1, "used": 0.0, "verified": 0.0,
         "felt": 0.0, "fluent": 0.0, "groundedness": g}
        for i, g in enumerate([0.12, 0.18, 0.24, 0.30])
    ]
    graph = _StubGraph(rows)
    store = EngramStore(graph, None, db_writer=None)
    n = store.recompute_population_groundedness()
    assert n == 4
    # axis_used backfilled from the provisional (same scale, no magic constant)
    assert graph.writes[("c0", 1)]["axes"] == {"used": 0.12}
    # new scalars discriminate: ordered by grounding + spread well out of [0.1,0.3]
    scalars = [graph.writes[(f"c{i}", 1)]["g"] for i in range(4)]
    assert scalars == sorted(scalars)
    assert max(scalars) - min(scalars) > 0.3
    assert min(scalars) > 0.0  # never zeroes a real Engram


def test_population_single_engram_keeps_provisional():
    # 1 Engram → no variance → provisional kept (NOT zeroed, NOT written).
    rows = [{"concept_id": "solo", "version": 1, "used": 0.0, "verified": 0.0,
             "felt": 0.0, "fluent": 0.0, "groundedness": 0.2}]
    graph = _StubGraph(rows)
    store = EngramStore(graph, None, db_writer=None)
    n = store.recompute_population_groundedness()
    assert n == 0 and graph.writes == {}  # nothing overwritten — provisional stands
