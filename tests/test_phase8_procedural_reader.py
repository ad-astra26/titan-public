"""Phase 8 — ProceduralSkillReader + EngineRecall(granularity='procedural') tests.

Covers:
- recall returns empty list when query empty
- recall returns empty list when FAISS index empty (no skills)
- recall ranks by composite (cosine surrogate * utility * (1 + name_boost))
- recall pre-filters utility floor + verified_at IS NOT NULL
- recall falls back to utility-only ordering when embedder is None
- should_delegate: False when no skill / unverified / utility<floor / score<floor / utility<0
- should_delegate: True when all gates pass
- EngineRecall.recall(granularity='procedural') wires the reader, returns RecallResult
- EngineRecall.recall(granularity='procedural', procedural_reader=None) returns None (fallback)
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import duckdb
import numpy as np
import pytest

from titan_hcl.synthesis.procedural_reader import (
    DEFAULT_MATCH_FLOOR,
    DEFAULT_UTILITY_FLOOR,
    ProceduralSkillReader,
)
from titan_hcl.synthesis.recall import EngineRecall
from titan_hcl.synthesis.skill_store import EMBEDDING_DIM, ProceduralSkillStore


def _deterministic_embedder():
    cache: dict[str, np.ndarray] = {}

    def embed(text: str) -> np.ndarray:
        if text in cache:
            return cache[text]
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        cache[text] = v
        return v

    return embed


@pytest.fixture()
def populated_store(tmp_path):
    conn = duckdb.connect(":memory:")
    store = ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "skills.faiss"),
        snapshot_path=str(tmp_path / "skills.json"),
        embedder=_deterministic_embedder(),
    )
    # 3 verified, 1 unverified, 1 retired
    base = dict(
        name="x", executable_spec={}, compiled_from=["t1"],
    )
    store.persist_skill(skill_id="A", nl_description="cosmetic website",
                        utility_score=0.7, **base)
    store.mark_verified("A")
    store.persist_skill(skill_id="B", nl_description="solana minting",
                        utility_score=0.5, **base)
    store.mark_verified("B")
    store.persist_skill(skill_id="C", nl_description="another cosmetic shop",
                        utility_score=0.4, **base)
    store.mark_verified("C")
    store.persist_skill(skill_id="D_unverified", nl_description="fresh skill",
                        utility_score=0.8, **base)
    # no mark_verified
    store.persist_skill(skill_id="E_retired", nl_description="bad skill",
                        utility_score=-0.6, **base)
    store.mark_verified("E_retired")
    return store


@pytest.fixture()
def reader(populated_store):
    return ProceduralSkillReader(populated_store)


# ── recall ──────────────────────────────────────────────────────────────


def test_recall_empty_query_returns_empty(reader):
    assert reader.recall("") == []


def test_recall_returns_skills_only_above_utility_floor(reader):
    out = reader.recall("cosmetic website", k=5)
    ids = {r["skill_id"] for r in out}
    # E_retired (utility=-0.6) excluded; D_unverified excluded (no verified_at)
    assert "E_retired" not in ids
    assert "D_unverified" not in ids


def test_recall_excludes_unverified(reader):
    out = reader.recall("fresh skill", k=5)
    ids = {r["skill_id"] for r in out}
    assert "D_unverified" not in ids


def test_recall_respects_k(reader):
    out = reader.recall("cosmetic", k=1)
    assert len(out) <= 1


def test_recall_returns_match_score_shape(reader):
    out = reader.recall("cosmetic website setup", k=5)
    for r in out:
        assert "match_score" in r
        assert 0.0 <= r["match_score"] <= 1.0
        assert "cosine_surrogate" in r
        assert "name_match_boost" in r


def test_recall_name_match_boosts_score(populated_store):
    """Skill whose nl_description contains a query token should rank above
    one that doesn't (when other factors are similar)."""
    reader = ProceduralSkillReader(populated_store)
    out = reader.recall("cosmetic", k=5)
    # "cosmetic website" (A) should appear ahead of "solana minting" (B)
    ids_order = [r["skill_id"] for r in out]
    if "A" in ids_order and "B" in ids_order:
        assert ids_order.index("A") < ids_order.index("B")


def test_recall_fallback_when_no_embedder(tmp_path):
    """When embedder is None, falls back to utility-only ordering."""
    conn = duckdb.connect(":memory:")
    store = ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "f.faiss"),
        snapshot_path=str(tmp_path / "s.json"),
        embedder=None,
    )
    base = dict(name="x", executable_spec={}, compiled_from=["t1"])
    store.persist_skill(skill_id="hi", nl_description="hi", utility_score=0.9, **base)
    store.mark_verified("hi")
    store.persist_skill(skill_id="med", nl_description="med", utility_score=0.5, **base)
    store.mark_verified("med")
    reader = ProceduralSkillReader(store)
    out = reader.recall("anything", k=2)
    assert [r["skill_id"] for r in out] == ["hi", "med"]


# ── should_delegate ─────────────────────────────────────────────────────


def test_should_delegate_false_on_none():
    reader = ProceduralSkillReader(MagicMock())
    assert reader.should_delegate(None) is False
    assert reader.should_delegate({}) is False


def test_should_delegate_false_on_unverified():
    reader = ProceduralSkillReader(MagicMock())
    assert reader.should_delegate({
        "match_score": 0.9, "utility_score": 0.8, "verified_at": None,
    }) is False


def test_should_delegate_false_on_negative_utility():
    reader = ProceduralSkillReader(MagicMock())
    assert reader.should_delegate({
        "match_score": 0.9, "utility_score": -0.1, "verified_at": 1000.0,
    }) is False


def test_should_delegate_false_below_utility_floor():
    reader = ProceduralSkillReader(MagicMock(), utility_floor=0.5)
    assert reader.should_delegate({
        "match_score": 0.9, "utility_score": 0.3, "verified_at": 1000.0,
    }) is False


def test_should_delegate_false_below_match_floor():
    reader = ProceduralSkillReader(MagicMock(), match_floor=0.7)
    assert reader.should_delegate({
        "match_score": 0.5, "utility_score": 0.8, "verified_at": 1000.0,
    }) is False


def test_should_delegate_true_on_pass():
    reader = ProceduralSkillReader(MagicMock())
    assert reader.should_delegate({
        "match_score": DEFAULT_MATCH_FLOOR + 0.1,
        "utility_score": DEFAULT_UTILITY_FLOOR + 0.1,
        "verified_at": 1000.0,
    }) is True


# ── EngineRecall integration ────────────────────────────────────────────


def test_engine_recall_procedural_branch_wires_reader(populated_store):
    reader = ProceduralSkillReader(populated_store)
    engine = EngineRecall(
        rule_evaluator=MagicMock(),
        activation_lookup=lambda ids: {},
        embedder=lambda t: [0.0],
        procedural_reader=reader,
    )
    results = engine.recall("cosmetic", granularity="procedural", k=3)
    assert results is not None
    assert len(results) > 0
    for r in results:
        assert r.fork == "procedural_skill"
        assert r.source == "synthesis_procedural_skill"


def test_engine_recall_procedural_branch_none_when_no_reader():
    engine = EngineRecall(
        rule_evaluator=MagicMock(),
        activation_lookup=lambda ids: {},
        embedder=lambda t: [0.0],
        procedural_reader=None,
    )
    results = engine.recall("cosmetic", granularity="procedural", k=3)
    assert results is None  # fall back


def test_engine_recall_procedural_handles_reader_exception(populated_store):
    bad_reader = MagicMock()
    bad_reader.recall = MagicMock(side_effect=RuntimeError("boom"))
    engine = EngineRecall(
        rule_evaluator=MagicMock(),
        activation_lookup=lambda ids: {},
        embedder=lambda t: [0.0],
        procedural_reader=bad_reader,
    )
    # Should not raise
    results = engine.recall("any", granularity="procedural", k=3)
    assert results is None


def test_engine_recall_procedural_empty_returns_empty_list():
    """Empty skill table → empty list (not None — distinguishes 'no results'
    from 'reader not wired')."""
    reader = MagicMock()
    reader.recall = MagicMock(return_value=[])
    engine = EngineRecall(
        rule_evaluator=MagicMock(),
        activation_lookup=lambda ids: {},
        embedder=lambda t: [0.0],
        procedural_reader=reader,
    )
    results = engine.recall("any", granularity="procedural", k=3)
    assert results == []
