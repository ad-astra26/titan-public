"""Phase 8 / EEL B1 — ProceduralSkillReader + EngineRecall('procedural') tests.

Re-pointed 2026-06-09 to the EEL B1 outcome × cells model (SPEC v0.29.0 /
D-SPEC-153): skills are built via the per-oracle-verified-use path
(enqueue_score_event + drain_score_events) rather than the retired `persist_skill`;
read_for_match returns POSITIVE cells only (the INV-EEL-5 polarity guard at
source); delegate-eligibility is `promoted OR verified_at` (a per-use-promoted
cell is oracle-verified by construction — INV-EEL-2).
"""
from __future__ import annotations

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
from titan_hcl.synthesis.skill_store import (
    EMBEDDING_DIM, ProceduralSkillStore, compute_skill_id,
)


def _deterministic_embedder():
    cache: dict[str, np.ndarray] = {}

    def embed(text: str) -> np.ndarray:
        if text in cache:
            return cache[text]
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:4], "big"))
        v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        cache[text] = v
        return v

    return embed


def _promote(store, *, goal_class, desc, n_success=1, n_fail=0, verify=True,
             oracle_id="o"):
    """Build an OUTCOME with a scored positive (or mixed) cell via the per-use
    path. Returns its skill_id (= sha256(oracle_id|goal_class))."""
    sid = store.ensure_outcome(oracle_id=oracle_id, goal_class=goal_class,
                               nl_description=desc)
    ts = 0.0
    task_shape = f"informational|tool-{goal_class}|"
    for i in range(n_success):
        store.enqueue_score_event(oracle_id=oracle_id, goal_class=goal_class,
                                  task_shape=task_shape, success=True,
                                  parent_tool_call_tx=f"s-{goal_class}-{i}", ts=ts)
        ts += 1.0
    for i in range(n_fail):
        store.enqueue_score_event(oracle_id=oracle_id, goal_class=goal_class,
                                  task_shape=task_shape, success=False,
                                  parent_tool_call_tx=f"f-{goal_class}-{i}", ts=ts)
        ts += 1.0
    store.drain_score_events()
    if verify:
        store.mark_verified(sid)
    return sid


@pytest.fixture()
def populated_store(tmp_path):
    conn = duckdb.connect(":memory:")
    store = ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "skills.faiss"),
        snapshot_path=str(tmp_path / "skills.json"),
        embedder=_deterministic_embedder(),
    )
    ids = {
        "A": _promote(store, goal_class="cosmetic-lookup", desc="cosmetic website"),
        "B": _promote(store, goal_class="solana-mint", desc="solana minting"),
        "C": _promote(store, goal_class="cosmetic-shop", desc="another cosmetic shop"),
    }
    # NOT-promoted (bare outcome, no positive cell) — excluded from match.
    ids["D_unpromoted"] = store.ensure_outcome(
        oracle_id="o", goal_class="fresh", nl_description="fresh skill")
    # Negative (failure-dominant) — excluded (polarity guard).
    ids["E_negative"] = _promote(store, goal_class="bad", desc="bad skill",
                                 n_success=0, n_fail=1, verify=True)
    store._ids = ids  # stash for tests
    return store


@pytest.fixture()
def reader(populated_store):
    return ProceduralSkillReader(populated_store)


# ── recall ──────────────────────────────────────────────────────────────


def test_recall_empty_query_returns_empty(reader):
    assert reader.recall("") == []


def test_recall_excludes_negative_and_unpromoted(reader, populated_store):
    ids = populated_store._ids
    out = reader.recall("cosmetic website", k=5)
    got = {r["skill_id"] for r in out}
    assert ids["E_negative"] not in got      # negative — polarity guard
    assert ids["D_unpromoted"] not in got     # no positive cell


def test_recall_respects_k(reader):
    assert len(reader.recall("cosmetic", k=1)) <= 1


def test_recall_returns_match_score_shape(reader):
    for r in reader.recall("cosmetic website setup", k=5):
        assert 0.0 <= r["match_score"] <= 1.0
        assert "cosine_surrogate" in r and "name_match_boost" in r


def test_recall_name_match_boosts_score(populated_store):
    ids = populated_store._ids
    reader = ProceduralSkillReader(populated_store)
    order = [r["skill_id"] for r in reader.recall("cosmetic", k=5)]
    # "cosmetic website" (A) ranks ahead of "solana minting" (B) via name boost.
    if ids["A"] in order and ids["B"] in order:
        assert order.index(ids["A"]) < order.index(ids["B"])


def test_recall_fallback_when_no_embedder(tmp_path):
    conn = duckdb.connect(":memory:")
    store = ProceduralSkillStore(
        duckdb_conn=conn, faiss_path=str(tmp_path / "f.faiss"),
        snapshot_path=str(tmp_path / "s.json"), embedder=None)
    hi = _promote(store, goal_class="hi", desc="hi", n_success=1)          # time_cost 1.0
    med = _promote(store, goal_class="med", desc="med", n_success=3, n_fail=1)  # 0.5625
    reader = ProceduralSkillReader(store)
    out = reader.recall("anything", k=2)
    assert [r["skill_id"] for r in out] == [hi, med]  # utility-only DESC ordering


# ── should_delegate (dict-based; back-compat + the B1 promoted/polarity gates) ──


def test_should_delegate_false_on_none():
    reader = ProceduralSkillReader(MagicMock())
    assert reader.should_delegate(None) is False
    assert reader.should_delegate({}) is False


def test_should_delegate_false_when_neither_promoted_nor_verified():
    reader = ProceduralSkillReader(MagicMock())
    assert reader.should_delegate({
        "match_score": 0.9, "utility_score": 0.8, "verified_at": None,
    }) is False


def test_should_delegate_true_when_promoted_even_if_unverified():
    # EEL B1: per-use promotion is delegate-eligible without SkillVerifier.
    reader = ProceduralSkillReader(MagicMock())
    assert reader.should_delegate({
        "match_score": DEFAULT_MATCH_FLOOR + 0.1,
        "utility_score": DEFAULT_UTILITY_FLOOR + 0.1,
        "verified_at": None, "promoted": True,
    }) is True


def test_should_delegate_false_on_negative_polarity():
    reader = ProceduralSkillReader(MagicMock())
    assert reader.should_delegate({
        "match_score": 0.9, "utility_score": 0.8, "promoted": True,
        "polarity": "negative",
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
        rule_evaluator=MagicMock(), activation_lookup=lambda ids: {},
        embedder=lambda t: [0.0], procedural_reader=reader)
    results = engine.recall("cosmetic", granularity="procedural", k=3)
    assert results is not None and len(results) > 0
    for r in results:
        assert r.fork == "procedural_skill"
        assert r.source == "synthesis_procedural_skill"


def test_engine_recall_procedural_branch_none_when_no_reader():
    engine = EngineRecall(
        rule_evaluator=MagicMock(), activation_lookup=lambda ids: {},
        embedder=lambda t: [0.0], procedural_reader=None)
    assert engine.recall("cosmetic", granularity="procedural", k=3) is None


def test_engine_recall_procedural_handles_reader_exception():
    bad_reader = MagicMock()
    bad_reader.recall = MagicMock(side_effect=RuntimeError("boom"))
    engine = EngineRecall(
        rule_evaluator=MagicMock(), activation_lookup=lambda ids: {},
        embedder=lambda t: [0.0], procedural_reader=bad_reader)
    assert engine.recall("any", granularity="procedural", k=3) is None


def test_engine_recall_procedural_empty_returns_empty_list():
    reader = MagicMock()
    reader.recall = MagicMock(return_value=[])
    engine = EngineRecall(
        rule_evaluator=MagicMock(), activation_lookup=lambda ids: {},
        embedder=lambda t: [0.0], procedural_reader=reader)
    assert engine.recall("any", granularity="procedural", k=3) == []
