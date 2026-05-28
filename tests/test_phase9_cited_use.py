"""Phase 9 — CitedUseDetector (strict cited gate, INV-Syn-23)."""

from titan_hcl.synthesis.cited_use import CitedUseDetector, SurfacedItem


def _det(**kw):
    return CitedUseDetector(**kw)


def test_substring_match_cites_item():
    d = _det()
    items = [
        SurfacedItem(item_id="mem:1", title="Metaplex minting", content_snippet="bubblegum tree"),
        SurfacedItem(item_id="mem:2", title="Unrelated weather", content_snippet="raining today"),
    ]
    cited = d.detect(
        response_text="To fix the metaplex minting bug, inspect the bubblegum config.",
        surfaced_items=items,
    )
    assert cited == ["mem:1"]


def test_concept_overlap_match():
    d = _det(concept_overlap_min=1)
    items = [SurfacedItem(item_id="tx:abc", concept_ids=["c_solana", "c_nft"])]
    cited = d.detect(
        response_text="here is an answer with no shared tokens whatsoever",
        surfaced_items=items,
        response_concept_ids=["c_nft", "c_other"],
    )
    assert cited == ["tx:abc"]


def test_concept_overlap_threshold_not_met():
    d = _det(concept_overlap_min=2)
    items = [SurfacedItem(item_id="tx:abc", concept_ids=["c_solana", "c_nft"])]
    cited = d.detect(
        response_text="zzz",
        surfaced_items=items,
        response_concept_ids=["c_nft"],  # only 1 overlap, need 2
    )
    assert cited == []


def test_no_match_returns_empty():
    d = _det()
    items = [SurfacedItem(item_id="mem:1", title="quantum chromodynamics")]
    cited = d.detect(response_text="completely different sentence", surfaced_items=items)
    assert cited == []


def test_multi_item_partition():
    d = _det()
    items = [
        SurfacedItem(item_id="a", title="alpha keyword"),
        SurfacedItem(item_id="b", title="beta keyword"),
        SurfacedItem(item_id="c", title="gamma keyword"),
    ]
    cited = d.detect(response_text="I used alpha and gamma here.", surfaced_items=items)
    assert set(cited) == {"a", "c"}


def test_empty_response_soft_path():
    d = _det()
    items = [SurfacedItem(item_id="a", title="alpha")]
    assert d.detect(response_text="   ", surfaced_items=items) == []
    assert d.detect(response_text="", surfaced_items=items) == []


def test_empty_items():
    d = _det()
    assert d.detect(response_text="anything", surfaced_items=[]) == []


def test_deterministic_rerun():
    d = _det()
    items = [SurfacedItem(item_id="x", title="deterministic token")]
    r1 = d.detect(response_text="deterministic output", surfaced_items=items)
    r2 = d.detect(response_text="deterministic output", surfaced_items=items)
    assert r1 == r2 == ["x"]


def test_min_token_len_filters_short_tokens():
    # "is" is shorter than the default min_token_len (4) → not a salient match.
    d = _det(min_token_len=4)
    items = [SurfacedItem(item_id="a", title="is to")]
    assert d.detect(response_text="this is to be", surfaced_items=items) == []


def test_dedup_preserves_order():
    d = _det()
    items = [
        SurfacedItem(item_id="dup", title="shared"),
        SurfacedItem(item_id="dup", title="shared"),
    ]
    # Same id surfaced twice → returned once.
    assert d.detect(response_text="shared content", surfaced_items=items) == ["dup"]


def test_skips_items_without_id():
    d = _det()
    items = [SurfacedItem(item_id="", title="orphan keyword")]
    assert d.detect(response_text="orphan keyword here", surfaced_items=items) == []
