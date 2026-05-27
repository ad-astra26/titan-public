"""Phase 8.Y — CGN lexicon snapshot exporter + agno loader (D-SPEC-PHASE8 fold-in).

Closes the P7 follow-up gap: agno's _ground_for_goal_hook reads
plugin.cgn_lexicon but no loader populated it → concept_ids stuck at [].

Covers:
- tokenize: minimum length, case-folding, punctuation strip
- build_lexicon_from_db: empty DB safe, basic mapping, confidence ordering
- write_snapshot: atomic tmp+rename, payload schema
- load_lexicon_snapshot: round-trip, missing file safe, corrupt JSON safe
- export_lexicon: end-to-end
- _ground_for_goal_hook now returns non-empty when plugin.cgn_lexicon populated
- agno _load_cgn_lexicon populates plugin.cgn_lexicon from snapshot
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest


# ── tokenize ───────────────────────────────────────────────────────────


def test_tokenize_basic():
    from titan_hcl.cgn.lexicon_exporter import tokenize
    assert tokenize("Linux Terminal") == ["linux", "terminal"]


def test_tokenize_skips_short_tokens():
    from titan_hcl.cgn.lexicon_exporter import tokenize
    # Tokens < 3 chars dropped
    assert tokenize("a b") == []
    assert tokenize("at no on it") == []  # 2-char skipped
    assert tokenize("hello AI") == ["hello"]  # "AI" too short


def test_tokenize_strips_punctuation():
    from titan_hcl.cgn.lexicon_exporter import tokenize
    out = tokenize("Hello, world! It's cosmetic.")
    assert "hello" in out
    assert "world" in out
    assert "cosmetic" in out


# ── build_lexicon_from_db ──────────────────────────────────────────────


def _seed_knowledge_concepts(tmp_path: Path, topics: list[tuple[str, float]]) -> Path:
    p = tmp_path / "inner_memory.db"
    conn = sqlite3.connect(str(p))
    conn.execute("""
        CREATE TABLE knowledge_concepts (
            topic TEXT, summary TEXT, confidence REAL, source TEXT,
            encounter_count INTEGER, created_at REAL
        )
    """)
    for topic, conf in topics:
        conn.execute(
            "INSERT INTO knowledge_concepts (topic, confidence) VALUES (?, ?)",
            (topic, conf),
        )
    conn.commit()
    conn.close()
    return p


def test_build_lexicon_missing_db_returns_empty(tmp_path):
    from titan_hcl.cgn.lexicon_exporter import build_lexicon_from_db
    out = build_lexicon_from_db(inner_memory_db=str(tmp_path / "nonexistent.db"))
    assert out == {}


def test_build_lexicon_basic(tmp_path):
    from titan_hcl.cgn.lexicon_exporter import build_lexicon_from_db
    db = _seed_knowledge_concepts(tmp_path, [
        ("Linux Terminal", 0.9),
        ("Solana NFT Minting", 0.7),
    ])
    out = build_lexicon_from_db(inner_memory_db=str(db))
    # Lowercase keys + topic-as-concept-id
    assert out["linux"] == "Linux Terminal"
    assert out["terminal"] == "Linux Terminal"
    assert out["solana"] == "Solana NFT Minting"
    assert out["minting"] == "Solana NFT Minting"
    # 2-char tokens like "of" skipped
    assert "of" not in out


def test_build_lexicon_higher_confidence_wins_collision(tmp_path):
    """When two topics share a token, the higher-confidence one wins."""
    from titan_hcl.cgn.lexicon_exporter import build_lexicon_from_db
    db = _seed_knowledge_concepts(tmp_path, [
        ("Solana NFT", 0.9),
        ("Cardano NFT", 0.4),
    ])
    out = build_lexicon_from_db(inner_memory_db=str(db))
    # "nft" appears in both — high-confidence "Solana NFT" wins
    assert out["nft"] == "Solana NFT"


def test_build_lexicon_caps_at_max_entries(tmp_path):
    from titan_hcl.cgn.lexicon_exporter import build_lexicon_from_db
    db = _seed_knowledge_concepts(
        tmp_path,
        [(f"Concept_{i}_unique", 0.5) for i in range(100)],
    )
    out = build_lexicon_from_db(inner_memory_db=str(db), max_entries=10)
    assert len(out) <= 10


# ── write_snapshot ──────────────────────────────────────────────────────


def test_write_snapshot_atomic(tmp_path):
    from titan_hcl.cgn.lexicon_exporter import write_snapshot
    target = tmp_path / "cgn_lexicon_snapshot.json"
    ok, payload = write_snapshot(
        {"linux": "Linux Terminal", "nft": "Solana NFT"},
        str(target),
    )
    assert ok is True
    assert target.exists()
    assert not (tmp_path / "cgn_lexicon_snapshot.json.tmp").exists()
    data = json.loads(target.read_text())
    assert data["lexicon_size"] == 2
    assert data["lexicon"]["linux"] == "Linux Terminal"


def test_write_snapshot_payload_schema(tmp_path):
    from titan_hcl.cgn.lexicon_exporter import write_snapshot
    target = tmp_path / "s.json"
    ok, payload = write_snapshot({}, str(target))
    assert ok is True
    for key in ("version", "ts", "lexicon_size", "lexicon"):
        assert key in payload


# ── load_lexicon_snapshot ──────────────────────────────────────────────


def test_load_lexicon_round_trip(tmp_path):
    from titan_hcl.cgn.lexicon_exporter import write_snapshot, load_lexicon_snapshot
    target = tmp_path / "s.json"
    write_snapshot({"abc": "AbcConcept", "xyz": "XyzConcept"}, str(target))
    loaded = load_lexicon_snapshot(str(target))
    assert loaded == {"abc": "AbcConcept", "xyz": "XyzConcept"}


def test_load_lexicon_missing_file_returns_empty(tmp_path):
    from titan_hcl.cgn.lexicon_exporter import load_lexicon_snapshot
    assert load_lexicon_snapshot(str(tmp_path / "absent.json")) == {}


def test_load_lexicon_corrupt_returns_empty(tmp_path):
    from titan_hcl.cgn.lexicon_exporter import load_lexicon_snapshot
    p = tmp_path / "corrupt.json"
    p.write_text("{not json")
    assert load_lexicon_snapshot(str(p)) == {}


def test_load_lexicon_filters_non_string_values(tmp_path):
    from titan_hcl.cgn.lexicon_exporter import load_lexicon_snapshot
    p = tmp_path / "weird.json"
    p.write_text(json.dumps({"version": 1, "lexicon": {
        "ok": "GoodConcept", "bad": 42, "also_bad": None,
    }}))
    loaded = load_lexicon_snapshot(str(p))
    assert loaded == {"ok": "GoodConcept"}


# ── export_lexicon ──────────────────────────────────────────────────────


def test_export_lexicon_end_to_end(tmp_path):
    from titan_hcl.cgn.lexicon_exporter import export_lexicon
    db = _seed_knowledge_concepts(tmp_path, [("Cosmetic Website", 0.8)])
    snap = tmp_path / "cgn_lexicon_snapshot.json"
    payload = export_lexicon(
        inner_memory_db=str(db),
        snapshot_path=str(snap),
    )
    assert payload is not None
    assert snap.exists()
    assert payload["lexicon_size"] > 0
    assert payload["lexicon"]["cosmetic"] == "Cosmetic Website"


# ── _ground_for_goal_hook integration ──────────────────────────────────


def test_ground_for_goal_hook_returns_real_concept_ids():
    """The P7 hook now returns real concept_ids when plugin.cgn_lexicon
    is populated (was [] pre-P8.Y)."""
    from titan_hcl.modules.agno_worker import _ground_for_goal_hook
    plugin = SimpleNamespace(cgn_lexicon={
        "cosmetic": "Cosmetic Website",
        "website": "Cosmetic Website",
        "solana": "Solana NFT",
    })
    out = _ground_for_goal_hook(plugin, "Can you build my cosmetic website on Solana?")
    assert "Cosmetic Website" in out
    assert "Solana NFT" in out


def test_ground_for_goal_hook_empty_when_lexicon_missing():
    """Regression — when lexicon is None / missing, the hook still returns []."""
    from titan_hcl.modules.agno_worker import _ground_for_goal_hook
    plugin = SimpleNamespace()  # no cgn_lexicon attr
    out = _ground_for_goal_hook(plugin, "Cosmetic website setup")
    assert out == []


# ── agno _load_cgn_lexicon ─────────────────────────────────────────────


def test_load_cgn_lexicon_populates_plugin(tmp_path, monkeypatch):
    """End-to-end: write snapshot → _load_cgn_lexicon → plugin.cgn_lexicon set."""
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    from titan_hcl.cgn.lexicon_exporter import write_snapshot
    write_snapshot({"hello": "HelloConcept"}, str(tmp_path / "cgn_lexicon_snapshot.json"))
    from titan_hcl.modules.agno_worker import _load_cgn_lexicon
    plugin = SimpleNamespace()
    n = _load_cgn_lexicon(plugin)
    assert n == 1
    assert plugin.cgn_lexicon == {"hello": "HelloConcept"}


def test_load_cgn_lexicon_missing_snapshot_safe(tmp_path, monkeypatch):
    """No snapshot → returns 0; plugin.cgn_lexicon untouched."""
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    from titan_hcl.modules.agno_worker import _load_cgn_lexicon
    plugin = SimpleNamespace()
    n = _load_cgn_lexicon(plugin)
    assert n == 0
    # plugin.cgn_lexicon not necessarily set on miss; hook handles both None + missing.


# ── cgn_worker imports CGN_LEXICON_UPDATED ─────────────────────────────


def test_cgn_worker_emits_lexicon_event_constant():
    """Regression: bus event constant + cgn_worker references it."""
    import inspect
    from titan_hcl.modules import cgn_worker
    from titan_hcl import bus
    assert hasattr(bus, "CGN_LEXICON_UPDATED")
    src = inspect.getsource(cgn_worker)
    assert "CGN_LEXICON_UPDATED" in src
    assert "_maybe_export_lexicon_snapshot" in src


def test_agno_worker_handles_lexicon_updated_event():
    """Regression: agno_worker has the recv-loop handler for CGN_LEXICON_UPDATED."""
    import inspect
    from titan_hcl.modules import agno_worker
    src = inspect.getsource(agno_worker)
    assert "msg_type == CGN_LEXICON_UPDATED" in src
    assert "_load_cgn_lexicon" in src
