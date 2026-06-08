"""Tests for synthesis/context_assembler.py (RFP_synthesis_decision_authority P4).

Covers the correctness core: source-tagging (activates the sovereignty V-term),
content-hash de-dup across partitions (gate G6 — no content twice), de-dup
collision priority (display==count), item_id assignment, and the SurfacedItem
projection the cited-use / sovereignty path consumes.
"""

import pytest

from titan_hcl.synthesis.context_assembler import (
    AssembledItem,
    SOURCE_PRIORITY,
    assemble,
    content_hash,
)
from titan_hcl.synthesis.cited_use import SurfacedItem, CitedUseDetector
from titan_hcl.synthesis.sovereignty_score import compute_sovereignty_score, VCB_SOURCE


# ── content_hash: normalisation ────────────────────────────────────────────

def test_content_hash_normalises_case_and_whitespace():
    assert content_hash("Hello   World") == content_hash("hello world")
    assert content_hash("  go-karts at Brno ") == content_hash("go-karts at brno")
    assert content_hash("a") != content_hash("b")


def test_content_hash_empty_is_stable():
    assert content_hash("") == content_hash("   ")


# ── source tagging ─────────────────────────────────────────────────────────

def test_each_partition_tagged_at_its_source():
    out = assemble(
        vcb=[{"content": "vocab word raffinesse"}],
        memory=[{"content": "legacy memory note"}],
        recall=[{"content": "spine thought about karts"}],
    )
    by_source = {it.source: it for it in out}
    assert set(by_source) == {"vcb", "memory", "recall"}
    assert by_source["vcb"].content == "vocab word raffinesse"


def test_vcb_item_drives_sovereignty_v_term():
    """The whole point of P4: a cited vcb-sourced item lifts V above 0."""
    out = assemble(vcb=[{"content": "Jirka plays go-karts on Saturdays"}])
    assert len(out) == 1
    surfaced = [it.to_surfaced() for it in out]
    assert surfaced[0].source == VCB_SOURCE
    score = compute_sovereignty_score(
        response_text="We talked about how Jirka plays go-karts on Saturdays.",
        surfaced_items=surfaced,
        cited_item_ids=[surfaced[0].item_id],
    )
    assert score.vcb_cited_count == 1
    assert score.v == 1.0          # the only cited item is inner-state
    assert score.s > 0.0           # V-term now contributes (was structurally 0)


# ── de-dup across partitions (G6) ──────────────────────────────────────────

def test_same_content_deduped_across_partitions():
    out = assemble(
        vcb=[{"content": "go-karts at the Brno circuit"}],
        recall=[{"content": "Go-Karts   at the BRNO circuit"}],  # same fact, diff format
    )
    assert len(out) == 1
    # vcb has priority → the survivor is the wholesale-injected one (display==count)
    assert out[0].source == "vcb"


def test_dedup_priority_follows_source_priority():
    assert SOURCE_PRIORITY[0] == "vcb"
    out = assemble(
        memory=[{"content": "shared fact"}],
        recall=[{"content": "shared fact"}],
    )
    assert len(out) == 1
    assert out[0].source == "memory"   # memory precedes recall in SOURCE_PRIORITY


def test_distinct_content_all_kept():
    out = assemble(
        vcb=[{"content": "fact A"}, {"content": "fact B"}],
        recall=[{"content": "fact C"}],
    )
    assert {it.content for it in out} == {"fact A", "fact B", "fact C"}


# ── item_id assignment ─────────────────────────────────────────────────────

def test_native_item_id_preserved():
    out = assemble(recall=[{"item_id": "tx:deadbeef", "content": "anchored thought"}])
    assert out[0].item_id == "tx:deadbeef"


def test_synthetic_item_id_when_absent():
    out = assemble(vcb=[{"content": "no native id here"}])
    iid = out[0].item_id
    assert iid.startswith("vcb:")
    assert iid.endswith(content_hash("no native id here")[:16])


# ── robustness (soft + total) ──────────────────────────────────────────────

def test_empty_inputs_yield_empty():
    assert assemble() == []
    assert assemble(vcb=[], memory=[], recall=[]) == []


def test_blank_content_rows_skipped():
    out = assemble(vcb=[{"content": "   "}, {"content": "real"}, {"title": "no content"}])
    assert len(out) == 1
    assert out[0].content == "real"


def test_malformed_rows_skipped_not_fatal():
    out = assemble(recall=[None, "not a dict", 42, {"content": "survivor"}])
    assert len(out) == 1
    assert out[0].content == "survivor"


def test_max_items_caps_output():
    rows = [{"content": f"fact {i}"} for i in range(100)]
    out = assemble(vcb=rows, max_items=10)
    assert len(out) == 10


def test_content_snippet_key_accepted():
    """The recall stash dicts use 'content_snippet', not 'content'."""
    out = assemble(recall=[{"item_id": "tx:1", "content_snippet": "from the spine"}])
    assert len(out) == 1
    assert out[0].content == "from the spine"


def test_weight_and_concept_ids_coerced():
    out = assemble(vcb=[{"content": "x", "weight": "0.7", "concept_ids": ["c1", "", "c2"]}])
    assert out[0].weight == pytest.approx(0.7)
    assert out[0].concept_ids == ("c1", "c2")
    out2 = assemble(vcb=[{"content": "y", "weight": None}])
    assert out2[0].weight == 0.0


def test_to_stash_dict_shape_matches_posthook_reader():
    """PostHook reads s.get('item_id'/'title'/'content_snippet'/'concept_ids'/'source')."""
    it = AssembledItem(
        item_id="vcb:abc", source="vcb", content="c" * 600,
        content_hash="h", title="t", concept_ids=("k1",),
    )
    d = it.to_stash_dict()
    assert d["item_id"] == "vcb:abc"
    assert d["source"] == "vcb"
    assert d["concept_ids"] == ["k1"]
    assert len(d["content_snippet"]) == 512   # capped


def test_cited_use_detector_consumes_assembled_surfaced_items():
    """End-to-end: assembled → SurfacedItem → CitedUseDetector marks the cited one."""
    out = assemble(
        vcb=[{"content": "the capital is Reykjavik"}],
        recall=[{"content": "unrelated spine note"}],
    )
    surfaced = [it.to_surfaced() for it in out]
    cited = CitedUseDetector().detect(
        response_text="As I recall, the capital is Reykjavik.",
        surfaced_items=surfaced,
    )
    vcb_item = next(it for it in out if it.source == "vcb")
    assert vcb_item.item_id in cited
