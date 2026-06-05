"""P3 — the ONE sovereignty metric `S = 0.7·E + 0.3·V` (RFP_synthesis_decision_authority).

Covers: E (substrate-cited token share), V (VCB inner-state share of cited
substrate), the 0.7/0.3 weights, the [0,1] clamp, and the degenerate cases
(nothing cited / nothing surfaced / empty response → honest S=0).
"""

import math

from titan_hcl.synthesis.cited_use import CitedUseDetector, SurfacedItem
from titan_hcl.synthesis.sovereignty_score import (
    WEIGHT_E,
    WEIGHT_V,
    compute_sovereignty_score,
)


def _score(response, items, cited):
    return compute_sovereignty_score(
        response_text=response,
        surfaced_items=items,
        cited_item_ids=cited,
        detector=CitedUseDetector(min_token_len=4),
    )


def test_weights_are_the_canonical_07_03():
    assert WEIGHT_E == 0.7
    assert WEIGHT_V == 0.3
    assert math.isclose(WEIGHT_E + WEIGHT_V, 1.0)


def test_E_is_cited_token_overlap_share():
    # Response has 4 salient tokens (len>=4): octopus, distributed,
    # intelligence, fascinating. ("is" is len 2 → not salient.)
    response = "octopus distributed intelligence is fascinating"
    item = SurfacedItem(
        item_id="tx:1", title="Octopus distributed intelligence",
        content_snippet="octopus arms compute", source="recall",
    )
    sc = _score(response, [item], ["tx:1"])
    # attributed salient ∩ resp = {octopus, distributed, intelligence} = 3
    # of 4 salient response tokens.
    assert sc.response_token_count == 4
    assert math.isclose(sc.e, 3.0 / 4.0)
    assert sc.cited_count == 1
    assert sc.vcb_cited_count == 0
    assert math.isclose(sc.v, 0.0)
    assert math.isclose(sc.s, WEIGHT_E * (3.0 / 4.0))


def test_V_is_vcb_fraction_of_cited():
    response = "octopus distributed intelligence fascinating wonderful"
    vcb = SurfacedItem(item_id="vcb:mood", title="octopus",
                       content_snippet="distributed", source="vcb")
    recall = SurfacedItem(item_id="tx:9", title="intelligence",
                          content_snippet="fascinating", source="recall")
    sc = _score(response, [vcb, recall], ["vcb:mood", "tx:9"])
    assert sc.cited_count == 2
    assert sc.vcb_cited_count == 1
    assert math.isclose(sc.v, 0.5)
    assert math.isclose(sc.s, WEIGHT_E * sc.e + WEIGHT_V * 0.5)


def test_S_combines_E_and_V_with_weights():
    response = "alpha bravo charlie delta"  # 4 salient tokens
    # one cited VCB item supplying all 4 → E=1.0, V=1.0 → S=1.0
    item = SurfacedItem(item_id="vcb:1", content_snippet="alpha bravo charlie delta",
                        source="vcb")
    sc = _score(response, [item], ["vcb:1"])
    assert math.isclose(sc.e, 1.0)
    assert math.isclose(sc.v, 1.0)
    assert math.isclose(sc.s, 1.0)


def test_clamped_to_unit_interval():
    response = "alpha bravo charlie delta"
    item = SurfacedItem(item_id="vcb:1", content_snippet="alpha bravo charlie delta",
                        source="vcb")
    # absurd weights → S would exceed 1 without the clamp.
    sc = compute_sovereignty_score(
        response_text=response, surfaced_items=[item], cited_item_ids=["vcb:1"],
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
    item = SurfacedItem(item_id="tx:1", content_snippet="octopus", source="recall")
    sc = _score("", [item], ["tx:1"])
    assert sc.s == 0.0
    assert sc.response_token_count == 0


def test_never_raises_on_garbage():
    # None / wrong types must degrade to S=0, never raise (hot-path safety).
    sc = compute_sovereignty_score(
        response_text=None, surfaced_items=[None, "nope", 42],
        cited_item_ids=None,
    )
    assert sc.s == 0.0
