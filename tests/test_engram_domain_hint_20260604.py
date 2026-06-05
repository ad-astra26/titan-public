"""RFP §7.F — advisory domain_hint (parse → persist → export → report).

Covers:
  • `_parse_llm_response` extracts + normalizes the new DOMAIN line (and drops
    placeholders / "unclear")
  • `create_concept(domain_hint=…)` persists it on the Kuzu Engram node and
    `export_snapshot` round-trips it (the lock-free read surface the report uses)
  • the offline clustering report groups hints (latest-version dedup) into
    candidate BRAIN_DOMAIN_* buckets
"""
from __future__ import annotations

import json
import queue

import pytest

from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.consolidation_defaults import _parse_llm_response
from titan_hcl.synthesis.engram_store import EngramStore
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter
from scripts.synthesis_domain_hints_report import build_report, _brain_domain_id


# ── Parser ───────────────────────────────────────────────────────────────
def test_parse_extracts_and_normalizes_domain():
    resp = (
        "ACTION: new_concept\n"
        "CONCEPT_ID: glacier_microbes\n"
        "NAME: Glacier Microbial Ecosystems\n"
        "MEMORY_TYPE: declarative\n"
        "DOMAIN:  Biology \n"
        "REASON: recurring theme\n"
    )
    p = _parse_llm_response(resp)
    assert p.action == "new_concept"
    assert p.domain_hint == "biology"  # normalized: stripped + lowercased


def test_parse_drops_placeholder_and_unclear():
    for raw in ("<broad knowledge domain>", "unclear", "empty", "none", ""):
        resp = (f"ACTION: new_concept\nCONCEPT_ID: x\nNAME: X\n"
                f"MEMORY_TYPE: meta\nDOMAIN: {raw}\nREASON: r\n")
        assert _parse_llm_response(resp).domain_hint == ""


def test_parse_missing_domain_line_is_empty():
    # The §7.F line is additive — a response without it still parses (lenient).
    resp = ("ACTION: new_concept\nCONCEPT_ID: y\nNAME: Y\n"
            "MEMORY_TYPE: meta\nREASON: r\n")
    assert _parse_llm_response(resp).domain_hint == ""


def test_parse_reject_has_empty_domain():
    assert _parse_llm_response("ACTION: reject\nREASON: noise\n").domain_hint == ""


# ── Persist + export round-trip ──────────────────────────────────────────
def test_domain_hint_persists_and_exports(tmp_path):
    g = TitanKnowledgeGraph(str(tmp_path / "f.kuzu"))
    w = OuterMemoryWriter(send_queue=queue.Queue(), src="domain_hint_test")
    store = EngramStore(g, w, clock=lambda: 1000.0)

    store.create_concept("glacier_microbes", "Glacier Microbes",
                         memory_type="declarative", domain_hint="biology")
    store.create_concept("noise_cluster", "Noise", memory_type="meta")  # no hint

    snap = str(tmp_path / "snap.json")
    store.export_snapshot(snap)
    with open(snap, encoding="utf-8") as f:
        rows = {r["concept_id"]: r for r in json.load(f)["concepts"]}
    assert rows["glacier_microbes"]["domain_hint"] == "biology"
    assert rows["noise_cluster"]["domain_hint"] == ""


# ── Report ───────────────────────────────────────────────────────────────
def _snapshot(rows):
    return {"version": 1, "concepts": rows}


def test_report_clusters_and_dedups_to_latest_version():
    snap = _snapshot([
        {"concept_id": "a", "version": 1, "name": "A v1", "domain_hint": "biology"},
        {"concept_id": "a", "version": 2, "name": "A v2", "domain_hint": "biology"},
        {"concept_id": "b", "version": 1, "name": "B", "domain_hint": "Biology"},
        {"concept_id": "c", "version": 1, "name": "C", "domain_hint": "mathematics"},
        {"concept_id": "d", "version": 1, "name": "D", "domain_hint": ""},
    ])
    rep = build_report(snap)
    assert rep["total_engrams"] == 4          # a counted once (latest version)
    assert rep["engrams_with_hint"] == 3      # a, b, c (d has none)
    top = rep["candidate_domains"][0]
    assert top["domain_hint"] == "biology"    # a + b (normalized "Biology"→"biology")
    assert top["engram_count"] == 2
    assert top["brain_domain_id"] == "BRAIN_DOMAIN_BIOLOGY"


def test_report_empty_when_no_hints():
    rep = build_report(_snapshot([
        {"concept_id": "a", "version": 1, "name": "A", "domain_hint": ""},
    ]))
    assert rep["engrams_with_hint"] == 0
    assert rep["candidate_domains"] == []


def test_brain_domain_id_slug():
    assert _brain_domain_id("self") == "BRAIN_DOMAIN_SELF"
    assert _brain_domain_id("music theory") == "BRAIN_DOMAIN_MUSIC_THEORY"
    assert _brain_domain_id("") == "BRAIN_DOMAIN_UNSPECIFIED"
