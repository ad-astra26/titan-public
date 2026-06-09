"""Phase 8 / EEL B1 — ProceduralSkillStore unit tests (INV-Syn-19/20/28).

Re-pointed 2026-06-09 to the EEL B1 outcome × task-shape model (SPEC v0.29.0 /
D-SPEC-153): a skill is the OUTCOME row `(oracle_id, goal_class)` (skill_id =
sha256(oracle_id|goal_class)) × per-task-shape `skill_cells` carrying the §3.4
triple; scoring is per oracle-verified use via the `skill_score_events` queue +
`drain_score_events`. The end-to-end gates (single-success promote, polarity
guard, legacy migration) live in `test_skill_cells_b1_20260609.py`; this file
covers the store's other surfaces (outcome lifecycle, FAISS-per-outcome, soft-
retire in [0,1], verify/reject, read_for_match / list_all / snapshot / stats).
"""
from __future__ import annotations

import json
import os

import duckdb
import numpy as np
import pytest

from titan_hcl.synthesis.skill_store import (
    DEFAULT_PROMOTE_FLOOR,
    EMBEDDING_DIM,
    ProceduralSkillStore,
    compute_skill_id,
    _safe_json_load,
)


# ── Fixtures ───────────────────────────────────────────────────────────


def _deterministic_embedder(seed_text_to_vec: dict | None = None):
    cache: dict[str, np.ndarray] = dict(seed_text_to_vec or {})

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


def _score(store, *, oracle_id, goal_class, task_shape, success, tx, ts=None):
    """Enqueue one oracle-verified use + drain → upsert the cell."""
    store.enqueue_score_event(
        oracle_id=oracle_id, goal_class=goal_class, task_shape=task_shape,
        success=success, parent_tool_call_tx=tx, ts=ts)
    return store.drain_score_events()


# ── DDL ────────────────────────────────────────────────────────────────


def test_schema_idempotent(tmp_path):
    conn = duckdb.connect(":memory:")
    kw = dict(faiss_path=str(tmp_path / "f.faiss"),
              snapshot_path=str(tmp_path / "s.json"), embedder=None)
    ProceduralSkillStore(duckdb_conn=conn, **kw)
    ProceduralSkillStore(duckdb_conn=conn, **kw)  # second construct must not raise
    tables = {r[0] for r in conn.execute(
        "SELECT table_name FROM duckdb_tables()").fetchall()}
    assert {"procedural_skills", "skill_cells", "skill_score_events"} <= tables


def test_empty_table_read_skill_none(store_no_embed):
    assert store_no_embed.read_skill(compute_skill_id("web_api_oracle", "defi-lookup")) is None


# ── outcome lifecycle (ensure_outcome) ──────────────────────────────────


def test_ensure_outcome_round_trip(store_no_embed):
    sid = store_no_embed.ensure_outcome(oracle_id="web_api_oracle", goal_class="defi-lookup")
    assert sid == compute_skill_id("web_api_oracle", "defi-lookup")
    sk = store_no_embed.read_skill(sid)
    assert sk["oracle_id"] == "web_api_oracle" and sk["goal_class"] == "defi-lookup"
    assert sk["promoted"] is False and sk["verified_at"] is None
    assert sk["embedding_id"] == -1  # no embedder


def test_ensure_outcome_idempotent(store_no_embed):
    a = store_no_embed.ensure_outcome(oracle_id="o", goal_class="g")
    b = store_no_embed.ensure_outcome(oracle_id="o", goal_class="g", name="other")
    assert a == b
    n = store_no_embed._db.execute(
        "SELECT COUNT(*) FROM procedural_skills WHERE skill_id = ?", [a]).fetchall()[0][0]
    assert n == 1


def test_ensure_outcome_requires_keys(store_no_embed):
    with pytest.raises(ValueError):
        store_no_embed.ensure_outcome(oracle_id="", goal_class="g")
    with pytest.raises(ValueError):
        store_no_embed.ensure_outcome(oracle_id="o", goal_class="")


def test_enqueue_requires_fields(store_no_embed):
    with pytest.raises(ValueError):
        store_no_embed.enqueue_score_event(
            oracle_id="", goal_class="g", task_shape="t", success=True)
    with pytest.raises(ValueError):
        store_no_embed.enqueue_score_event(
            oracle_id="o", goal_class="g", task_shape="", success=True)


# ── cell scoring across drains ───────────────────────────────────────────


def test_cells_accumulate_across_drains(store_no_embed):
    sid = compute_skill_id("web_api_oracle", "defi-lookup")
    for i, tx in enumerate(("t1", "t2", "t3")):
        _score(store_no_embed, oracle_id="web_api_oracle", goal_class="defi-lookup",
               task_shape="informational|searxng|defi", success=True, tx=tx, ts=float(i))
    cell = store_no_embed.read_skill(sid)["cells"][0]
    assert cell["b_i"] == 3 and cell["success_count"] == 3 and cell["polarity"] == "positive"


def test_failure_makes_negative_cell(store_no_embed):
    sid = compute_skill_id("o", "market-lookup")
    _score(store_no_embed, oracle_id="o", goal_class="market-lookup",
           task_shape="informational|flaky|market", success=False, tx="tf")
    cell = store_no_embed.read_skill(sid)["cells"][0]
    assert cell["polarity"] == "negative" and cell["time_cost"] == 0.0


# ── FAISS (per-outcome embedding) ────────────────────────────────────────


def test_ensure_outcome_with_embedder_assigns_embedding_id(store_with_embed):
    sid = store_with_embed.ensure_outcome(
        oracle_id="o", goal_class="g", nl_description="cosmetic website setup")
    assert store_with_embed.read_skill(sid)["embedding_id"] == 0  # first FAISS row


def test_faiss_search_returns_top_k(store_with_embed):
    for gc, desc in (("defi-lookup", "solana defi tvl"),
                     ("market-lookup", "token market price"),
                     ("code-compute", "python hash compute")):
        store_with_embed.ensure_outcome(oracle_id="o", goal_class=gc, nl_description=desc)
    q = store_with_embed.embed_query("defi")
    assert q is not None
    hits = store_with_embed.faiss_search(q, top_k=3)
    assert len(hits) == 3
    for emb_id, dist in hits:
        assert emb_id >= 0 and dist >= 0.0


def test_faiss_persists_across_construct(tmp_path):
    embedder = _deterministic_embedder()
    conn1 = duckdb.connect(str(tmp_path / "synth.duckdb"))
    s1 = ProceduralSkillStore(
        duckdb_conn=conn1, faiss_path=str(tmp_path / "f.faiss"),
        snapshot_path=str(tmp_path / "s.json"), embedder=embedder)
    s1.ensure_outcome(oracle_id="o", goal_class="g", nl_description="hello world")
    assert os.path.exists(str(tmp_path / "f.faiss"))
    conn1.close()
    conn2 = duckdb.connect(str(tmp_path / "synth.duckdb"))
    s2 = ProceduralSkillStore(
        duckdb_conn=conn2, faiss_path=str(tmp_path / "f.faiss"),
        snapshot_path=str(tmp_path / "s.json"), embedder=embedder)
    s2._ensure_faiss()
    assert s2._faiss.ntotal == 1


# ── soft-retire (time_cost crossing the floor downward, [0,1] space) ─────


def test_soft_retire_callback_fires_when_crossing_floor(tmp_path):
    conn = duckdb.connect(":memory:")
    fired: list[tuple[str, float]] = []
    store = ProceduralSkillStore(
        duckdb_conn=conn, faiss_path=str(tmp_path / "f.faiss"),
        snapshot_path=str(tmp_path / "s.json"), embedder=None,
        soft_retire_floor=0.5,
        on_soft_retire=lambda sid, tc: fired.append((sid, tc)))
    sid = compute_skill_id("o", "g")
    # success (time_cost 1.0) then failure (time_cost 0.25) — crosses 0.5 down.
    store.enqueue_score_event(oracle_id="o", goal_class="g", task_shape="t",
                              success=True, parent_tool_call_tx="s", ts=1.0)
    store.enqueue_score_event(oracle_id="o", goal_class="g", task_shape="t",
                              success=False, parent_tool_call_tx="f", ts=2.0)
    store.drain_score_events()
    assert len(fired) == 1 and fired[0][0] == sid and fired[0][1] <= 0.5


# ── mark_verified / mark_rejected ────────────────────────────────────────


def test_mark_verified_sets_verified_at(store_no_embed):
    sid = store_no_embed.ensure_outcome(oracle_id="o", goal_class="g")
    store_no_embed.mark_verified(sid)
    assert store_no_embed.read_skill(sid)["verified_at"] is not None


def test_mark_rejected_flips_cells_negative(store_no_embed):
    sid = compute_skill_id("o", "g")
    _score(store_no_embed, oracle_id="o", goal_class="g", task_shape="t",
           success=True, tx="s")  # makes a positive, promoted cell
    assert sid in [r["skill_id"] for r in store_no_embed.read_for_match()]
    store_no_embed.mark_rejected(sid, reason="content_hash_mismatch")
    sk = store_no_embed.read_skill(sid)
    assert sk["verified_at"] is not None and sk["promoted"] is False
    assert all(c["polarity"] == "negative" for c in sk["cells"])
    assert sid not in [r["skill_id"] for r in store_no_embed.read_for_match()]


# ── read_for_match (positives only — polarity guard at source) ───────────


def test_read_for_match_positives_only_and_floor(store_no_embed):
    # a promoted positive (time_cost 1.0)
    _score(store_no_embed, oracle_id="o", goal_class="hi", task_shape="t1",
           success=True, tx="s")
    # a negative (failure-dominant) — must be excluded
    _score(store_no_embed, oracle_id="o", goal_class="lo", task_shape="t2",
           success=False, tx="f")
    ids = [r["skill_id"] for r in store_no_embed.read_for_match(utility_floor=0.3)]
    assert compute_skill_id("o", "hi") in ids
    assert compute_skill_id("o", "lo") not in ids
    # a high floor excludes the positive too
    assert store_no_embed.read_for_match(utility_floor=1.01) == []


def test_read_for_match_verified_only_accepts_promoted(store_no_embed):
    # per-use-promoted (verified_at is None) is delegate-eligible by `promoted`.
    sid = compute_skill_id("o", "g")
    _score(store_no_embed, oracle_id="o", goal_class="g", task_shape="t",
           success=True, tx="s")
    assert store_no_embed.read_skill(sid)["verified_at"] is None
    assert sid in [r["skill_id"] for r in store_no_embed.read_for_match(verified_only=True)]


# ── list_all ──────────────────────────────────────────────────────────────


def test_list_all_orders_promoted_first(store_no_embed):
    _score(store_no_embed, oracle_id="o", goal_class="promoted", task_shape="t",
           success=True, tx="s")  # promoted
    store_no_embed.ensure_outcome(oracle_id="o", goal_class="bare")  # no cells
    ids = [r["skill_id"] for r in store_no_embed.list_all()]
    assert ids[0] == compute_skill_id("o", "promoted")
    assert compute_skill_id("o", "bare") in ids


# ── snapshot ──────────────────────────────────────────────────────────────


def test_snapshot_export_atomic_v2(store_no_embed):
    _score(store_no_embed, oracle_id="web_api_oracle", goal_class="defi-lookup",
           task_shape="informational|searxng|defi", success=True, tx="s")
    snap_path = store_no_embed._snapshot_path
    assert os.path.exists(snap_path) and not os.path.exists(snap_path + ".tmp")
    payload = json.loads(open(snap_path).read())
    assert payload["version"] == 2 and payload["count"] == 1
    assert payload["skills"][0]["goal_class"] == "defi-lookup"


def test_snapshot_orders_by_best_time_cost(store_no_embed):
    _score(store_no_embed, oracle_id="o", goal_class="hi", task_shape="t1",
           success=True, tx="s1")              # time_cost 1.0
    # a present-but-weaker outcome: 1 success + 1 failure → time_cost 0.25
    store_no_embed.enqueue_score_event(oracle_id="o", goal_class="lo",
                                       task_shape="t2", success=True, parent_tool_call_tx="s2", ts=1.0)
    store_no_embed.enqueue_score_event(oracle_id="o", goal_class="lo",
                                       task_shape="t2", success=False, parent_tool_call_tx="f2", ts=2.0)
    store_no_embed.drain_score_events()
    payload = json.loads(open(store_no_embed._snapshot_path).read())
    ids = [s["skill_id"] for s in payload["skills"]]
    assert ids[0] == compute_skill_id("o", "hi")


# ── stats ─────────────────────────────────────────────────────────────────


def test_stats_surfaces_counters(store_with_embed):
    s = store_with_embed.stats()
    assert s["events_enqueued"] == 0 and s["faiss_count"] == 0
    _score(store_with_embed, oracle_id="o", goal_class="g",
           task_shape="t", success=True, tx="s")
    store_with_embed.mark_verified(compute_skill_id("o", "g"))
    s = store_with_embed.stats()
    assert s["events_enqueued"] == 1 and s["events_drained"] == 1
    assert s["promotions_seen"] == 1 and s["verifications_seen"] == 1
    assert s["faiss_count"] == 1  # one outcome embedded


# ── persist_negative_skill (miner surface) ──────────────────────────────


def test_persist_negative_skill_never_matchable(store_no_embed):
    sid = store_no_embed.persist_negative_skill(
        oracle_id="miner_recurrence", goal_class="code-compute",
        task_shape="procedural|sandbox|", name="[negative] x",
        nl_description="failed", compiled_from=["t1"])
    assert store_no_embed.read_skill(sid)["cells"][0]["polarity"] == "negative"
    assert sid not in [r["skill_id"] for r in store_no_embed.read_for_match()]
    with pytest.raises(ValueError):
        store_no_embed.persist_negative_skill(
            oracle_id="o", goal_class="g", task_shape="t", name="n",
            nl_description="d", compiled_from=[])  # empty lineage gate


# ── _safe_json_load helper (unchanged) ───────────────────────────────────


def test_safe_json_load_handles_corrupt():
    assert _safe_json_load("not-json", []) == []
    assert _safe_json_load(None, {}) == {}
    assert _safe_json_load("[1, 2]", []) == [1, 2]
