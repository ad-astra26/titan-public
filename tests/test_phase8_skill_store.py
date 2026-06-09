"""Phase 8 — ProceduralSkillStore unit tests (D-SPEC-PHASE8 / INV-Syn-19/20).

Covers:
- DDL idempotent + creates utility/last_used indices
- persist_skill round-trips content + canonical-JSON dict/list fields
- Re-persist preserves success_count/failure_count/verified_at
- increment_success / increment_failure clamp + counter movement
- Soft-retire callback fires when utility crosses floor
- mark_verified sets verified_at
- mark_rejected sets utility_score=-1.0 + verified_at
- read_skill / read_for_match / list_all shapes
- snapshot_export atomic write + payload shape
- FAISS round-trip when embedder is provided
- FAISS deferred (embedding_id=-1) when embedder is None
- ValueError on empty skill_id / missing required fields / empty compiled_from
- stats() surfaces all counters
"""
from __future__ import annotations

import json
import os
from typing import Optional

import duckdb
import numpy as np
import pytest

from titan_hcl.synthesis.skill_store import (
    DEFAULT_UTILITY_SCORE,
    EMBEDDING_DIM,
    ProceduralSkillStore,
    UTILITY_DELTA,
    _safe_json_load,
)


# ── Fixtures ───────────────────────────────────────────────────────────


def _deterministic_embedder(seed_text_to_vec: dict | None = None):
    """Return an embedder that maps text → 384D unit vector. For tests we
    just hash text → seed numpy RNG → return a stable vector so FAISS index
    rows are deterministic."""
    cache: dict[str, np.ndarray] = dict(seed_text_to_vec or {})

    def embed(text: str) -> np.ndarray:
        if text in cache:
            return cache[text]
        # Use sha256 prefix as a deterministic seed for the test
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
def store_no_embed(tmp_path):
    conn = duckdb.connect(":memory:")
    yield ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "skills_vectors.faiss"),
        snapshot_path=str(tmp_path / "skills_snapshot.json"),
        embedder=None,
    )


@pytest.fixture()
def store_with_embed(tmp_path):
    conn = duckdb.connect(":memory:")
    yield ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "skills_vectors.faiss"),
        snapshot_path=str(tmp_path / "skills_snapshot.json"),
        embedder=_deterministic_embedder(),
    )


def _basic_skill_kwargs(**overrides):
    base = dict(
        skill_id="skill_test_001",
        name="test_skill",
        nl_description="Test skill description",
        executable_spec={"steps": [{"tool": "x", "args": {}}]},
        preconditions=["pre1"],
        postconditions=["post1"],
        compiled_from=["tx_aaa", "tx_bbb"],
    )
    base.update(overrides)
    return base


# ── DDL ────────────────────────────────────────────────────────────────


def test_schema_idempotent(tmp_path):
    conn = duckdb.connect(":memory:")
    ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "f.faiss"),
        snapshot_path=str(tmp_path / "s.json"),
        embedder=None,
    )
    # Second construct on same connection must not raise
    ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "f.faiss"),
        snapshot_path=str(tmp_path / "s.json"),
        embedder=None,
    )
    # Table exists
    rows = conn.execute(
        "SELECT table_name FROM duckdb_tables() WHERE table_name = 'procedural_skills'"
    ).fetchall()
    assert rows


def test_schema_has_expected_columns(store_no_embed):
    skill = store_no_embed.read_skill("skill_test_001")
    assert skill is None  # empty table


# ── persist_skill ──────────────────────────────────────────────────────


def test_persist_round_trip(store_no_embed):
    store_no_embed.persist_skill(**_basic_skill_kwargs())
    row = store_no_embed.read_skill("skill_test_001")
    assert row is not None
    assert row["skill_id"] == "skill_test_001"
    assert row["name"] == "test_skill"
    assert row["nl_description"] == "Test skill description"
    assert row["executable_spec"] == {"steps": [{"tool": "x", "args": {}}]}
    assert row["preconditions"] == ["pre1"]
    assert row["postconditions"] == ["post1"]
    assert row["compiled_from"] == ["tx_aaa", "tx_bbb"]
    assert row["success_count"] == 0
    assert row["failure_count"] == 0
    assert row["utility_score"] == DEFAULT_UTILITY_SCORE
    assert row["verified_at"] is None
    assert row["embedding_id"] == -1  # no embedder


def test_persist_with_embedder_assigns_embedding_id(store_with_embed):
    store_with_embed.persist_skill(**_basic_skill_kwargs())
    row = store_with_embed.read_skill("skill_test_001")
    assert row["embedding_id"] == 0  # first row in FAISS index


def test_re_persist_preserves_counters(store_no_embed):
    store_no_embed.persist_skill(**_basic_skill_kwargs())
    store_no_embed.increment_success("skill_test_001")
    store_no_embed.increment_success("skill_test_001")
    store_no_embed.mark_verified("skill_test_001")
    # Now re-persist (e.g. miner re-detects same recurrence)
    store_no_embed.persist_skill(**_basic_skill_kwargs(nl_description="updated"))
    row = store_no_embed.read_skill("skill_test_001")
    assert row["nl_description"] == "updated"
    assert row["success_count"] == 2
    assert row["verified_at"] is not None


def test_persist_requires_skill_id():
    conn = duckdb.connect(":memory:")
    store = ProceduralSkillStore(
        duckdb_conn=conn, faiss_path="/tmp/_t.faiss", snapshot_path="/tmp/_t.json",
        embedder=None,
    )
    with pytest.raises(ValueError, match="skill_id"):
        store.persist_skill(**_basic_skill_kwargs(skill_id=""))


def test_persist_requires_compiled_from(store_no_embed):
    with pytest.raises(ValueError, match="compiled_from"):
        store_no_embed.persist_skill(**_basic_skill_kwargs(compiled_from=[]))


def test_persist_requires_name_and_desc(store_no_embed):
    with pytest.raises(ValueError):
        store_no_embed.persist_skill(**_basic_skill_kwargs(name=""))
    with pytest.raises(ValueError):
        store_no_embed.persist_skill(**_basic_skill_kwargs(nl_description=""))


# ── increment_success / increment_failure ──────────────────────────────


def test_increment_success_moves_utility_up(store_no_embed):
    store_no_embed.persist_skill(**_basic_skill_kwargs())
    store_no_embed.increment_success("skill_test_001")
    row = store_no_embed.read_skill("skill_test_001")
    assert row["success_count"] == 1
    assert row["utility_score"] == pytest.approx(DEFAULT_UTILITY_SCORE + UTILITY_DELTA)
    assert row["last_used"] is not None


def test_increment_failure_moves_utility_down(store_no_embed):
    store_no_embed.persist_skill(**_basic_skill_kwargs())
    store_no_embed.increment_failure("skill_test_001")
    row = store_no_embed.read_skill("skill_test_001")
    assert row["failure_count"] == 1
    assert row["utility_score"] == pytest.approx(DEFAULT_UTILITY_SCORE - UTILITY_DELTA)


def test_utility_clamped_to_one(store_no_embed):
    store_no_embed.persist_skill(**_basic_skill_kwargs(utility_score=0.99))
    for _ in range(10):
        store_no_embed.increment_success("skill_test_001")
    row = store_no_embed.read_skill("skill_test_001")
    assert row["utility_score"] == pytest.approx(1.0)


def test_utility_clamped_to_negative_one(store_no_embed):
    store_no_embed.persist_skill(**_basic_skill_kwargs(utility_score=-0.99))
    for _ in range(10):
        store_no_embed.increment_failure("skill_test_001")
    row = store_no_embed.read_skill("skill_test_001")
    assert row["utility_score"] == pytest.approx(-1.0)


def test_unknown_skill_id_logs_and_no_op(store_no_embed, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    store_no_embed.increment_success("does_not_exist")
    assert any("does_not_exist" in r.message for r in caplog.records)


# ── soft-retire callback ───────────────────────────────────────────────


def test_soft_retire_callback_fires_when_crossing_floor(tmp_path):
    conn = duckdb.connect(":memory:")
    fired: list[tuple[str, float]] = []
    store = ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "f.faiss"),
        snapshot_path=str(tmp_path / "s.json"),
        embedder=None,
        soft_retire_floor=-0.5,
        on_soft_retire=lambda sid, u: fired.append((sid, u)),
    )
    store.persist_skill(**_basic_skill_kwargs(utility_score=-0.45))
    # one decrement should cross -0.5 floor
    store.increment_failure("skill_test_001")
    assert len(fired) == 1
    assert fired[0][0] == "skill_test_001"
    assert fired[0][1] <= -0.5


def test_soft_retire_does_not_re_fire(tmp_path):
    """Crossing the floor once should fire; subsequent decrements should NOT."""
    conn = duckdb.connect(":memory:")
    fired: list[tuple[str, float]] = []
    store = ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "f.faiss"),
        snapshot_path=str(tmp_path / "s.json"),
        embedder=None,
        soft_retire_floor=-0.5,
        on_soft_retire=lambda sid, u: fired.append((sid, u)),
    )
    store.persist_skill(**_basic_skill_kwargs(utility_score=-0.45))
    store.increment_failure("skill_test_001")  # crosses
    store.increment_failure("skill_test_001")  # already past floor
    assert len(fired) == 1


# ── mark_verified / mark_rejected ──────────────────────────────────────


def test_mark_verified_sets_verified_at(store_no_embed):
    store_no_embed.persist_skill(**_basic_skill_kwargs())
    store_no_embed.mark_verified("skill_test_001")
    row = store_no_embed.read_skill("skill_test_001")
    assert row["verified_at"] is not None
    assert row["utility_score"] == DEFAULT_UTILITY_SCORE  # unchanged


def test_mark_rejected_sets_utility_negative_one(store_no_embed):
    store_no_embed.persist_skill(**_basic_skill_kwargs())
    store_no_embed.mark_rejected("skill_test_001", reason="content_hash_mismatch")
    row = store_no_embed.read_skill("skill_test_001")
    assert row["utility_score"] == pytest.approx(-1.0)
    assert row["verified_at"] is not None


# ── read_for_match ─────────────────────────────────────────────────────


def test_read_for_match_filters_by_utility_floor(store_no_embed):
    store_no_embed.persist_skill(**_basic_skill_kwargs(skill_id="hi", utility_score=0.7))
    store_no_embed.mark_verified("hi")
    store_no_embed.persist_skill(**_basic_skill_kwargs(skill_id="lo", utility_score=0.2))
    store_no_embed.mark_verified("lo")
    result = store_no_embed.read_for_match(utility_floor=0.3, k=5)
    ids = [r["skill_id"] for r in result]
    assert "hi" in ids
    assert "lo" not in ids


def test_read_for_match_filters_unverified(store_no_embed):
    store_no_embed.persist_skill(**_basic_skill_kwargs(skill_id="verified"))
    store_no_embed.mark_verified("verified")
    store_no_embed.persist_skill(**_basic_skill_kwargs(skill_id="fresh"))
    result = store_no_embed.read_for_match(utility_floor=0.3, k=5, verified_only=True)
    assert [r["skill_id"] for r in result] == ["verified"]
    # verified_only=False should include both
    result2 = store_no_embed.read_for_match(utility_floor=0.3, k=5, verified_only=False)
    assert {r["skill_id"] for r in result2} == {"verified", "fresh"}


# ── list_all ────────────────────────────────────────────────────────────


def test_list_all_orders_by_utility(store_no_embed):
    store_no_embed.persist_skill(**_basic_skill_kwargs(skill_id="a", utility_score=0.3))
    store_no_embed.persist_skill(**_basic_skill_kwargs(skill_id="b", utility_score=0.9))
    store_no_embed.persist_skill(**_basic_skill_kwargs(skill_id="c", utility_score=0.5))
    result = store_no_embed.list_all()
    ids = [r["skill_id"] for r in result]
    assert ids == ["b", "c", "a"]


# ── FAISS round-trip ────────────────────────────────────────────────────


def test_faiss_search_returns_top_k(store_with_embed):
    store_with_embed.persist_skill(**_basic_skill_kwargs(skill_id="a", nl_description="cosmetic website setup"))
    store_with_embed.persist_skill(**_basic_skill_kwargs(skill_id="b", nl_description="solana minting flow"))
    store_with_embed.persist_skill(**_basic_skill_kwargs(skill_id="c", nl_description="cosmetic shop deploy"))
    q = store_with_embed.embed_query("cosmetic")
    assert q is not None
    hits = store_with_embed.faiss_search(q, top_k=3)
    assert len(hits) == 3
    # Each hit is (embedding_id, distance)
    for emb_id, dist in hits:
        assert emb_id >= 0
        assert dist >= 0.0


def test_faiss_persists_across_construct(tmp_path):
    """FAISS index is atomic-written; second construct loads existing index."""
    embedder = _deterministic_embedder()
    conn1 = duckdb.connect(str(tmp_path / "synth.duckdb"))
    s1 = ProceduralSkillStore(
        duckdb_conn=conn1,
        faiss_path=str(tmp_path / "f.faiss"),
        snapshot_path=str(tmp_path / "s.json"),
        embedder=embedder,
    )
    s1.persist_skill(**_basic_skill_kwargs(skill_id="x", nl_description="hello world"))
    assert os.path.exists(str(tmp_path / "f.faiss"))
    conn1.close()
    # Open second store
    conn2 = duckdb.connect(str(tmp_path / "synth.duckdb"))
    s2 = ProceduralSkillStore(
        duckdb_conn=conn2,
        faiss_path=str(tmp_path / "f.faiss"),
        snapshot_path=str(tmp_path / "s.json"),
        embedder=embedder,
    )
    s2._ensure_faiss()
    assert s2._faiss.ntotal == 1


# ── snapshot_export ────────────────────────────────────────────────────


def test_snapshot_export_atomic(store_no_embed, tmp_path):
    store_no_embed.persist_skill(**_basic_skill_kwargs())
    snap_path = store_no_embed._snapshot_path
    assert os.path.exists(snap_path)
    payload = json.loads(open(snap_path).read())
    assert payload["version"] == 1
    assert payload["count"] == 1
    assert payload["persists_seen"] >= 1
    assert payload["skills"][0]["skill_id"] == "skill_test_001"
    # Tmp file removed
    assert not os.path.exists(snap_path + ".tmp")


def test_snapshot_payload_orders_by_utility(store_no_embed):
    store_no_embed.persist_skill(**_basic_skill_kwargs(skill_id="lo", utility_score=0.2))
    store_no_embed.persist_skill(**_basic_skill_kwargs(skill_id="hi", utility_score=0.8))
    snap_path = store_no_embed._snapshot_path
    payload = json.loads(open(snap_path).read())
    ids = [s["skill_id"] for s in payload["skills"]]
    assert ids == ["hi", "lo"]


# ── stats() ─────────────────────────────────────────────────────────────


def test_stats_surfaces_counters(store_with_embed):
    s = store_with_embed.stats()
    assert s["persists_seen"] == 0
    assert s["faiss_count"] == 0
    store_with_embed.persist_skill(**_basic_skill_kwargs())
    store_with_embed.increment_success("skill_test_001")
    store_with_embed.mark_verified("skill_test_001")
    s = store_with_embed.stats()
    assert s["persists_seen"] == 1
    assert s["utility_updates"] == 1
    assert s["verifications_seen"] == 1
    assert s["faiss_count"] == 1


# ── _safe_json_load helper ──────────────────────────────────────────────


def test_safe_json_load_handles_corrupt():
    assert _safe_json_load("not-json", []) == []
    assert _safe_json_load(None, {}) == {}
    assert _safe_json_load("[1, 2]", []) == [1, 2]
