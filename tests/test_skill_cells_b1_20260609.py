"""EEL Pillar B1 — skill_store re-key: outcome × task-shape cells + per-use
scoring + polarity guard + legacy migration (RFP §7.B1 / SPEC §25.1 INV-Syn-29 /
arch §8.1 v0.29.0). Also asserts the D-SPEC-154 / INV-Syn-30 PK-only invariant
(no secondary ART indexes on the B1 tables) + delete-on-drain queue bounding.

Covers the gates: EEL-G2 (positive skill from a single oracle-verified success →
promoted → delegatable) + EEL-G4 (a [negative] cell is never delegated).
"""
import duckdb
import pytest

from titan_hcl.synthesis.skill_store import (
    ProceduralSkillStore, compute_skill_id, reduce_time_cost,
)
from titan_hcl.synthesis.procedural_reader import ProceduralSkillReader


def _store(tmp_path, conn=None):
    conn = conn or duckdb.connect(":memory:")
    return ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "skills.faiss"),
        snapshot_path=str(tmp_path / "skills_snapshot.json"),
        embedder=None,
    ), conn


# ── skill_id identity (re-derived from the outcome — INV-Syn-29) ─────────────

def test_skill_id_stable_and_outcome_keyed():
    a = compute_skill_id("web_api_oracle", "defi-lookup")
    b = compute_skill_id("web_api_oracle", "defi-lookup")
    assert a == b and a.startswith("skill_")
    # different outcome → different id
    assert compute_skill_id("web_api_oracle", "market-lookup") != a
    assert compute_skill_id("coding_sandbox", "defi-lookup") != a


# ── reduce_time_cost (the synthesis-reduction proficiency — NOT IQL) ─────────

def test_reduce_time_cost_single_success_promotable():
    # one oracle-verified success → proficiency 1.0 (≥ promote_floor 0.7) → EEL-G2
    assert reduce_time_cost(1, 0, 1.0) == 1.0


def test_reduce_time_cost_failure_dominant_low():
    assert reduce_time_cost(1, 2, 1.0 / 3.0) < 0.5
    assert reduce_time_cost(0, 1, 0.0) == 0.0
    assert reduce_time_cost(0, 0, 0.0) == 0.0


# ── EEL-G2: single oracle-verified success → promoted ────────────────────────

def test_single_success_promotes_and_is_matchable(tmp_path):
    store, _ = _store(tmp_path)
    store.enqueue_score_event(
        oracle_id="web_api_oracle", goal_class="defi-lookup",
        task_shape="informational|searxng-search|defi", success=True,
        parent_tool_call_tx="tx_abc")
    summary = store.drain_score_events()
    assert summary["drained"] == 1 and summary["promoted"] == 1

    sid = compute_skill_id("web_api_oracle", "defi-lookup")
    sk = store.read_skill(sid)
    assert sk is not None and sk["promoted"] is True
    assert sk["utility_score"] == 1.0
    assert sk["cells"][0]["polarity"] == "positive"
    assert sk["cells"][0]["b_i"] == 1 and sk["cells"][0]["success_count"] == 1

    match_ids = [r["skill_id"] for r in store.read_for_match()]
    assert sid in match_ids


def test_entity_generalization_same_skill_reinforced(tmp_path):
    store, _ = _store(tmp_path)
    # same goal_class + task_shape, different entities (the args differ, not the key)
    for tx in ("tx1", "tx2", "tx3"):
        store.enqueue_score_event(
            oracle_id="web_api_oracle", goal_class="defi-lookup",
            task_shape="informational|searxng-search|defi", success=True,
            parent_tool_call_tx=tx)
    store.drain_score_events()
    sid = compute_skill_id("web_api_oracle", "defi-lookup")
    sk = store.read_skill(sid)
    assert sk["cells"][0]["b_i"] == 3 and sk["cells"][0]["success_count"] == 3


# ── EEL-G4: polarity guard — a [negative] cell is never delegated ────────────

def test_failure_dominant_is_negative_and_excluded_from_match(tmp_path):
    store, _ = _store(tmp_path)
    store.enqueue_score_event(
        oracle_id="web_api_oracle", goal_class="market-lookup",
        task_shape="informational|flaky|market", success=False,
        parent_tool_call_tx="txf")
    store.drain_score_events()
    sid_neg = compute_skill_id("web_api_oracle", "market-lookup")
    neg = store.read_skill(sid_neg)
    assert neg["cells"][0]["polarity"] == "negative"
    assert neg["promoted"] is False
    assert sid_neg not in [r["skill_id"] for r in store.read_for_match()]


def test_miner_negative_skill_never_delegated(tmp_path):
    store, _ = _store(tmp_path)
    store.persist_negative_skill(
        oracle_id="miner_recurrence", goal_class="code-compute",
        task_shape="procedural|sandbox+sandbox|", name="[negative] failed compute",
        nl_description="repeatedly failed compute", compiled_from=["t1", "t2"])
    sid = compute_skill_id("miner_recurrence", "code-compute")
    sk = store.read_skill(sid)
    assert sk["cells"][0]["polarity"] == "negative"
    assert sid not in [r["skill_id"] for r in store.read_for_match()]


def test_reader_should_delegate_polarity_and_promotion(tmp_path):
    store, _ = _store(tmp_path)
    reader = ProceduralSkillReader(skill_store=store)
    # a promoted positive (no embedder → recall falls back to read_for_match)
    store.enqueue_score_event(
        oracle_id="coding_sandbox", goal_class="code-compute",
        task_shape="computational|coding_sandbox|code", success=True,
        parent_tool_call_tx="txp")
    store.drain_score_events()
    results = reader.recall("compute the hash", k=3)
    assert results, "promoted positive must be recallable"
    top = results[0]
    assert reader.should_delegate(top) is True
    # explicit polarity guard: a negative row is never delegatable
    neg_row = dict(top)
    neg_row["polarity"] = "negative"
    assert reader.should_delegate(neg_row) is False
    # un-promoted + unverified → not delegatable
    unp = dict(top)
    unp["promoted"] = False
    unp["verified_at"] = None
    assert reader.should_delegate(unp) is False


# ── idempotent enqueue (a replayed bus message never double-counts) ──────────

def test_enqueue_idempotent(tmp_path):
    store, conn = _store(tmp_path)
    for _ in range(3):
        store.enqueue_score_event(
            oracle_id="web_api_oracle", goal_class="defi-lookup",
            task_shape="informational|searxng-search|defi", success=True,
            parent_tool_call_tx="dup", ts=1.0)
    n = conn.execute(
        "SELECT COUNT(*) FROM skill_score_events WHERE parent_tool_call_tx='dup'"
    ).fetchall()[0][0]
    assert n == 1


def test_drain_marks_processed(tmp_path):
    store, conn = _store(tmp_path)
    store.enqueue_score_event(
        oracle_id="web_api_oracle", goal_class="defi-lookup",
        task_shape="informational|searxng-search|defi", success=True,
        parent_tool_call_tx="t")
    store.drain_score_events()
    unprocessed = conn.execute(
        "SELECT COUNT(*) FROM skill_score_events WHERE processed = FALSE"
    ).fetchall()[0][0]
    assert unprocessed == 0
    # a second drain is a no-op
    assert store.drain_score_events()["drained"] == 0


# ── legacy migration (never-delete §8.4 / INV-3) ─────────────────────────────

def test_legacy_migration_preserves_pre_b1_table(tmp_path):
    conn = duckdb.connect(":memory:")
    # Simulate a pre-B1 procedural_skills (skill_id PK, NO oracle_id column) WITH
    # its secondary indexes — those dependents are what blocked the RENAME live
    # on T3 (2026-06-09): DuckDB errors "Cannot alter entry … entries depend on
    # it" unless they're dropped first. This reproduces that exact failure.
    conn.execute(
        "CREATE TABLE procedural_skills ("
        " skill_id TEXT PRIMARY KEY, name TEXT, nl_description TEXT, "
        " utility_score DOUBLE, last_used DOUBLE)")
    conn.execute("CREATE INDEX idx_procedural_skills_utility ON procedural_skills(utility_score DESC)")
    conn.execute("CREATE INDEX idx_procedural_skills_last_used ON procedural_skills(last_used DESC)")
    conn.execute(
        "INSERT INTO procedural_skills VALUES ('skill_oldneg', '[negative] old', 'x', 0.7, NULL)")
    # Booting the B1 store must rename the old table aside (never drop it).
    store, _ = _store(tmp_path, conn=conn)
    cols = {r[0].lower() for r in conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='procedural_skills'").fetchall()}
    assert "oracle_id" in cols, "new procedural_skills must be B1-shaped"
    legacy = conn.execute(
        "SELECT skill_id FROM procedural_skills_legacy").fetchall()
    assert legacy and legacy[0][0] == "skill_oldneg", "pre-B1 row must be preserved"
    # migration is idempotent — re-init does not error or re-migrate
    store2, _ = _store(tmp_path, conn=conn)
    assert conn.execute(
        "SELECT COUNT(*) FROM procedural_skills_legacy").fetchall()[0][0] == 1


def test_fresh_install_no_legacy(tmp_path):
    store, conn = _store(tmp_path)
    legacy = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_name='procedural_skills_legacy'").fetchall()
    assert not legacy, "fresh install must not create a legacy table"


# ── PK-only — no secondary ART indexes (INV-Syn-30 / D-SPEC-154) ─────────────

def test_b1_tables_are_pk_only(tmp_path):
    """The three B1 tables carry ONLY their PRIMARY KEY — no secondary ART
    indexes (the runtime-corruption source behind the actr_buffers-class FATAL
    crash-loop). `duckdb_indexes()` lists only explicit secondary indexes; the
    PK index is implicit + unlisted → expect EMPTY for every table."""
    store, conn = _store(tmp_path)
    for table in ("procedural_skills", "skill_cells", "skill_score_events"):
        rows = conn.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = ?",
            [table]).fetchall()
        names = {r[0] for r in rows}
        assert names == set(), f"{table} must be PK-only, found secondary {names}"


def test_b1_boot_self_heals_stray_secondary_indexes(tmp_path):
    """A B1-shaped DB that picked up stray secondary indexes from the initial
    deploy is self-healed on the next store boot (DROP IF EXISTS — no operator
    script, D-SPEC-154)."""
    conn = duckdb.connect(":memory:")
    # First boot creates the B1 tables PK-only.
    _store(tmp_path, conn=conn)
    # Simulate a deploy that had created the now-banned secondary indexes.
    conn.execute("CREATE INDEX idx_skill_cells_time_cost ON skill_cells(time_cost DESC)")
    conn.execute("CREATE INDEX idx_skill_events_unprocessed ON skill_score_events(processed)")
    before = {r[0] for r in conn.execute(
        "SELECT index_name FROM duckdb_indexes() "
        "WHERE table_name IN ('skill_cells','skill_score_events')").fetchall()}
    assert before == {"idx_skill_cells_time_cost", "idx_skill_events_unprocessed"}
    # Re-boot the store → self-heal drops them.
    _store(tmp_path, conn=conn)
    after = {r[0] for r in conn.execute(
        "SELECT index_name FROM duckdb_indexes() "
        "WHERE table_name IN ('skill_cells','skill_score_events')").fetchall()}
    assert after == set(), f"self-heal must drop stray B1 secondary indexes, left {after}"


# ── delete-on-drain — bound the queue (INV-Syn-30 / D-SPEC-154) ──────────────

def test_drain_purges_processed_events_past_window(tmp_path):
    """Processed score events older than PROCESSED_PURGE_WINDOW_S are DELETEd on
    drain so the queue (which is intentionally unindexed) cannot grow unbounded
    and re-introduce ART churn. A within-window processed row is retained for the
    idempotent-enqueue dedup."""
    from titan_hcl.synthesis.skill_store import PROCESSED_PURGE_WINDOW_S

    now = [10_000.0]

    def _clock():
        return now[0]

    conn = duckdb.connect(":memory:")
    store = ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "skills.faiss"),
        snapshot_path=str(tmp_path / "skills_snapshot.json"),
        embedder=None,
        clock=_clock,
    )
    # Event 1 at t0 → drain marks it processed (ts = t0).
    store.enqueue_score_event(
        oracle_id="web_api_oracle", goal_class="defi-lookup",
        task_shape="informational|searxng-search|defi", success=True,
        parent_tool_call_tx="tx1")
    store.drain_score_events()
    assert conn.execute("SELECT COUNT(*) FROM skill_score_events").fetchone()[0] == 1

    # Advance the clock PAST the purge window, enqueue + drain a 2nd event.
    # The 2nd drain purges the now-stale event 1 (processed AND ts < now-window)
    # but keeps the freshly-processed event 2.
    now[0] = 10_000.0 + PROCESSED_PURGE_WINDOW_S + 100.0
    store.enqueue_score_event(
        oracle_id="web_api_oracle", goal_class="market-lookup",
        task_shape="informational|searxng-search|market", success=True,
        parent_tool_call_tx="tx2")
    store.drain_score_events()
    remaining = conn.execute(
        "SELECT parent_tool_call_tx FROM skill_score_events").fetchall()
    assert remaining == [("tx2",)], f"stale processed event must be purged, got {remaining}"


# ── snapshot shape (cross-process readers) ───────────────────────────────────

def test_snapshot_v2_shape(tmp_path):
    import json
    store, _ = _store(tmp_path)
    store.enqueue_score_event(
        oracle_id="web_api_oracle", goal_class="defi-lookup",
        task_shape="informational|searxng-search|defi", success=True,
        parent_tool_call_tx="t")
    store.drain_score_events()
    snap = json.load(open(tmp_path / "skills_snapshot.json"))
    assert snap["version"] == 2
    assert snap["count"] == 1
    assert snap["promotions_seen"] == 1
    sk = snap["skills"][0]
    assert sk["oracle_id"] == "web_api_oracle" and sk["goal_class"] == "defi-lookup"
    assert sk["promoted"] is True
