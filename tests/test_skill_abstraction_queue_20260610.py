"""Dream-hardening (2026-06-10): the off-hot-path negative-skill abstraction
queue — enqueue (idempotent) → fetch pending → record result (persist+done OR
retry/failed) — and PERSISTENCE across restarts/crashes (save/resume).

The dream's miner enqueues recurrent failure clusters here (fast, no LLM); the
synthesis_worker abstraction daemon makes the LLM call OFF the dream/writer thread
and records the result. A slow/timed-out LLM must never lose work or block the
dream — these tests pin enqueue/drain semantics + that a pending entry survives a
restart (a NEW store instance on the SAME DuckDB still sees it).
"""
import duckdb
import pytest

from titan_hcl.synthesis.skill_store import (
    ProceduralSkillStore, ABSTRACTION_MAX_ATTEMPTS,
)


def _store(tmp_path, conn=None, dbfile=None):
    conn = conn or duckdb.connect(dbfile or ":memory:")
    return ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "skills.faiss"),
        snapshot_path=str(tmp_path / "skills_snapshot.json"),
        embedder=None,
    ), conn


def _prepared(cluster_id="absc_test01", seq=None):
    seq = seq or [["coding_sandbox", "h1"], ["coding_sandbox", "h2"]]
    return {
        "cluster_id": cluster_id,
        "sequence": seq,
        "occurrence_count": 3,
        "kind": "negative",
        "members_summary": [{"tool": "coding_sandbox"}],
        "compiled_from": ["tx_a", "tx_b", "tx_c"],
    }


def _valid_proposal():
    return {
        "nl_description": "Approach iterative sandbox retry fails for convergence tasks.",
        "executable_spec": {"steps": [{"tool": "coding_sandbox"}]},
        "preconditions": ["pre"], "postconditions": ["post"],
    }


# ── enqueue (idempotent) ─────────────────────────────────────────────────────

def test_enqueue_is_idempotent(tmp_path):
    store, _ = _store(tmp_path)
    assert store.enqueue_abstraction(_prepared(), ts=1.0) is True
    assert store.enqueue_abstraction(_prepared(), ts=2.0) is False   # same cluster_id
    pending = store.fetch_pending_abstractions(limit=100)
    assert len(pending) == 1 and pending[0]["cluster_id"] == "absc_test01"


# ── record: valid proposal → done + negative skill persisted ─────────────────

def test_record_valid_proposal_persists_and_marks_done(tmp_path):
    store, conn = _store(tmp_path)
    store.enqueue_abstraction(_prepared(), ts=1.0)
    skill_id = store.record_abstraction_result("absc_test01", _valid_proposal())
    assert skill_id, "valid proposal should persist a negative skill + return skill_id"
    # queue entry marked done; no longer pending
    assert store.fetch_pending_abstractions(limit=100) == []
    status = conn.execute(
        "SELECT status, skill_id FROM skill_abstraction_queue WHERE cluster_id='absc_test01'"
    ).fetchone()
    assert status[0] == "done" and status[1] == skill_id
    # a [negative] skill now exists
    assert any(str(r.get("name", "")).startswith("[negative]") for r in store.list_all())


# ── record: failure → retry (stays pending) → max_attempts → failed ──────────

def test_record_failure_retries_then_fails(tmp_path):
    store, conn = _store(tmp_path)
    store.enqueue_abstraction(_prepared(), ts=1.0)
    # None proposal = LLM timeout/unparseable → retry, stays pending until max.
    for attempt in range(1, ABSTRACTION_MAX_ATTEMPTS):
        assert store.record_abstraction_result("absc_test01", None) is None
        pend = store.fetch_pending_abstractions(limit=100)
        assert len(pend) == 1, f"should still be pending after attempt {attempt}"
        assert pend[0]["attempts"] == attempt
    # final attempt crosses the cap → marked failed (no longer pending, not lost)
    assert store.record_abstraction_result("absc_test01", None) is None
    assert store.fetch_pending_abstractions(limit=100) == []
    st = conn.execute(
        "SELECT status, attempts FROM skill_abstraction_queue WHERE cluster_id='absc_test01'"
    ).fetchone()
    assert st[0] == "failed" and st[1] == ABSTRACTION_MAX_ATTEMPTS


def test_record_recovers_after_transient_failures(tmp_path):
    """A few timeouts then a success → still compiles (work not lost)."""
    store, _ = _store(tmp_path)
    store.enqueue_abstraction(_prepared(), ts=1.0)
    assert store.record_abstraction_result("absc_test01", None) is None   # timeout
    assert store.record_abstraction_result("absc_test01", None) is None   # timeout
    skill_id = store.record_abstraction_result("absc_test01", _valid_proposal())  # LLM recovers
    assert skill_id and store.fetch_pending_abstractions(limit=100) == []


# ── SAVE/RESUME — pending entries survive a restart (new store, same DuckDB) ──

def test_pending_survives_restart(tmp_path):
    dbfile = str(tmp_path / "synth.duckdb")
    store1, conn1 = _store(tmp_path, dbfile=dbfile)
    store1.enqueue_abstraction(_prepared("absc_resume"), ts=1.0)
    assert len(store1.fetch_pending_abstractions(limit=100)) == 1
    conn1.close()   # simulate process death / restart

    # Fresh store instance on the SAME durable DuckDB — must resume the pending work.
    store2, conn2 = _store(tmp_path, dbfile=dbfile)
    pend = store2.fetch_pending_abstractions(limit=100)
    assert len(pend) == 1 and pend[0]["cluster_id"] == "absc_resume", \
        "pending abstraction must survive a restart (durable queue / save-resume)"
    # and it can still be completed post-restart
    skill_id = store2.record_abstraction_result("absc_resume", _valid_proposal())
    assert skill_id
    conn2.close()


def test_done_entry_not_reprocessed_after_restart(tmp_path):
    """A completed abstraction stays 'done' across a restart (no duplicate work)."""
    dbfile = str(tmp_path / "synth2.duckdb")
    store1, conn1 = _store(tmp_path, dbfile=dbfile)
    store1.enqueue_abstraction(_prepared("absc_done"), ts=1.0)
    sid = store1.record_abstraction_result("absc_done", _valid_proposal())
    assert sid
    conn1.close()
    store2, conn2 = _store(tmp_path, dbfile=dbfile)
    assert store2.fetch_pending_abstractions(limit=100) == []   # not re-queued
    conn2.close()
