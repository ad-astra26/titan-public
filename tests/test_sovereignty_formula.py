"""P3 — the ONE sovereignty metric `S = 0.7·E + 0.3·V` (RFP_synthesis_decision_authority).

Covers: E (substrate-cited token share), the refined V (verification share —
timechain-anchored cited fraction, lifted to 1.0 by a passing tool/oracle
verdict, zeroed by an OVG consistency failure), the 0.7/0.3 weights, the [0,1]
clamp, and the degenerate cases (nothing cited / surfaced / empty → honest S=0).
"""

import math

from titan_hcl.synthesis.cited_use import CitedUseDetector, SurfacedItem
from titan_hcl.synthesis.sovereignty_score import (
    WEIGHT_E,
    WEIGHT_V,
    TX_ANCHORED_SOURCES,
    compute_sovereignty_score,
)


def _score(response, items, cited, **kw):
    return compute_sovereignty_score(
        response_text=response,
        surfaced_items=items,
        cited_item_ids=cited,
        detector=CitedUseDetector(min_token_len=4),
        **kw,
    )


def test_weights_are_the_canonical_07_03():
    assert WEIGHT_E == 0.7
    assert WEIGHT_V == 0.3
    assert math.isclose(WEIGHT_E + WEIGHT_V, 1.0)


def test_E_counts_cited_substrate_V_zero_when_unverified():
    # `vcb` (live inner state) grounds E ("was it his") but is NOT on the
    # tx_hash spine → not verified → V=0.
    response = "octopus distributed intelligence is fascinating"  # 4 salient
    item = SurfacedItem(
        item_id="vcb:1", title="Octopus distributed intelligence",
        content_snippet="octopus arms compute", source="vcb",
    )
    sc = _score(response, [item], ["vcb:1"])
    # attributed salient ∩ resp = {octopus, distributed, intelligence} = 3 of 4.
    assert sc.response_token_count == 4
    assert math.isclose(sc.e, 3.0 / 4.0)
    assert sc.cited_count == 1
    assert sc.verified_cited_count == 0
    assert math.isclose(sc.v, 0.0)
    assert math.isclose(sc.s, WEIGHT_E * (3.0 / 4.0))


def test_V_is_timechain_anchored_fraction_of_cited():
    response = "octopus distributed intelligence fascinating wonderful"
    anchored = SurfacedItem(item_id="tx:9", title="intelligence",
                            content_snippet="fascinating", source="recall")
    unverified = SurfacedItem(item_id="vcb:mood", title="octopus",
                              content_snippet="distributed", source="vcb")
    sc = _score(response, [anchored, unverified], ["tx:9", "vcb:mood"])
    assert sc.cited_count == 2
    assert sc.verified_cited_count == 1   # only the recall item is anchored
    assert math.isclose(sc.v, 0.5)
    assert math.isclose(sc.s, WEIGHT_E * sc.e + WEIGHT_V * 0.5)


def test_every_tx_anchored_source_counts_as_verified():
    resp = "alpha bravo charlie delta echo foxtrot"
    items = [SurfacedItem(item_id=f"i{i}", content_snippet="alpha", source=s)
             for i, s in enumerate(sorted(TX_ANCHORED_SOURCES))]
    sc = _score(resp, items, [f"i{i}" for i in range(len(items))])
    assert sc.verified_cited_count == len(TX_ANCHORED_SOURCES)
    assert math.isclose(sc.v, 1.0)


def test_oracle_verdict_lifts_V_to_one_even_with_no_cited_substrate():
    # A sandbox/oracle-computed answer is fully provable even if it cited nothing.
    sc = _score("the factorial answer is 479001600", [], [], oracle_verified=True)
    assert math.isclose(sc.v, 1.0)
    assert math.isclose(sc.e, 0.0)           # nothing cited
    assert math.isclose(sc.s, WEIGHT_V * 1.0)


def test_consistency_failure_zeroes_V():
    # A reply checked and found to contradict ground truth is not verified.
    item = SurfacedItem(item_id="tx:1", content_snippet="alpha bravo",
                        source="recall")
    sc = _score("alpha bravo charlie delta", [item], ["tx:1"],
                consistency_ok=False)
    assert math.isclose(sc.v, 0.0)
    assert math.isclose(sc.s, WEIGHT_E * sc.e)   # only E survives


def test_consistency_failure_beats_oracle_verified():
    sc = _score("alpha bravo charlie delta", [], [],
                oracle_verified=True, consistency_ok=False)
    assert math.isclose(sc.v, 0.0)


def test_S_combines_E_and_V_with_weights():
    # one cited, tx-anchored item supplying all 4 tokens → E=1.0, V=1.0 → S=1.0.
    response = "alpha bravo charlie delta"
    item = SurfacedItem(item_id="tx:1",
                        content_snippet="alpha bravo charlie delta",
                        source="recall")
    sc = _score(response, [item], ["tx:1"])
    assert math.isclose(sc.e, 1.0)
    assert math.isclose(sc.v, 1.0)
    assert math.isclose(sc.s, 1.0)


def test_clamped_to_unit_interval():
    response = "alpha bravo charlie delta"
    item = SurfacedItem(item_id="tx:1",
                        content_snippet="alpha bravo charlie delta",
                        source="recall")
    sc = compute_sovereignty_score(
        response_text=response, surfaced_items=[item], cited_item_ids=["tx:1"],
        detector=CitedUseDetector(min_token_len=4), w_e=5.0, w_v=5.0,
    )
    assert sc.s == 1.0
    assert 0.0 <= sc.e <= 1.0 and 0.0 <= sc.v <= 1.0


def test_nothing_cited_is_zero():
    response = "octopus distributed intelligence fascinating"
    item = SurfacedItem(item_id="tx:1", content_snippet="octopus", source="recall")
    sc = _score(response, [item], [])  # surfaced but not cited
    assert sc.s == 0.0 and sc.e == 0.0 and sc.v == 0.0
    assert sc.cited_count == 0


def test_nothing_surfaced_is_zero():
    sc = _score("a full response here friend", [], [])
    assert sc.s == 0.0


def test_empty_response_is_zero():
    # Empty reply grounded nothing → S=0, even though it "cited" an anchored item.
    item = SurfacedItem(item_id="tx:1", content_snippet="octopus", source="recall")
    sc = _score("", [item], ["tx:1"])
    assert sc.s == 0.0
    assert sc.v == 0.0
    assert sc.response_token_count == 0


def test_never_raises_on_garbage():
    # None / wrong types must degrade to S=0, never raise (hot-path safety).
    sc = compute_sovereignty_score(
        response_text=None, surfaced_items=[None, "nope", 42],
        cited_item_ids=None,
    )
    assert sc.s == 0.0
