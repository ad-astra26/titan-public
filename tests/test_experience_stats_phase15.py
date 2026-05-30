"""
test_experience_stats_phase15 — §3L Phase 15 chunk 15.1 (D-SPEC-PHASE15).

Covers the ExperienceMemory→ExperienceOrchestrator retirement:
  - ExperienceOrchestrator.get_experience_stats_payload() — by_domain aggregate
    from the INCREMENTAL action_stats table (attempt-weighted avg_score;
    success_rate = successes / attempts); never a per-read GROUP BY over records.
  - ExperienceOrchestrator.recall_similar() — live recall from experience_records
    (successor to the retired ExperienceMemory.recall_similar), incl. optional
    cosine re-rank on inner_state.
  - ExperienceStatsPublisher._compute_payload / _stub — slot payload shape +
    schema_version, oversize-safe.
"""
import time

import pytest

from titan_hcl.logic.experience_orchestrator import ExperienceOrchestrator
from titan_hcl.logic.experience_stats_publisher import ExperienceStatsPublisher
from titan_hcl._phase_c_constants import (
    EXPERIENCE_STATS_SCHEMA_VERSION,
    EXPERIENCE_STATS_MAX_BYTES,
)


@pytest.fixture
def orch(tmp_path):
    db = str(tmp_path / "experience_orchestrator.db")
    o = ExperienceOrchestrator(db_path=db)
    yield o
    o._conn.close()


def _seed_action_stats(o, rows):
    """rows: list of (domain, action, attempts, successes, avg_score)."""
    with o._lock:
        for dom, act, att, succ, avg in rows:
            o._conn.execute(
                "INSERT INTO action_stats (domain, action, total_attempts, "
                "total_successes, avg_score, last_updated) VALUES (?,?,?,?,?,?)",
                (dom, act, att, succ, avg, time.time()),
            )
        o._conn.commit()


def _seed_record(o, domain, action, score, inner_state, created_at):
    with o._lock:
        o._conn.execute(
            "INSERT INTO experience_records (domain, perception_key, "
            "inner_state_hash, inner_state, hormonal_snapshot, action_taken, "
            "outcome_score, context, epoch_id, distilled, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (domain, "[]", "h", str(inner_state).replace("'", '"'),
             "{}", action, score, "{}", 0, 0, created_at),
        )
        o._conn.commit()


def test_stats_payload_by_domain_attempt_weighted(orch):
    # communication: two actions → attempt-weighted avg, summed success_rate.
    _seed_action_stats(orch, [
        ("communication", "kin_sense", 10, 6, 0.6),
        ("communication", "social_post", 30, 15, 0.4),
        ("creative", "art_generate", 20, 18, 0.9),
    ])
    payload = orch.get_experience_stats_payload()
    bd = payload["by_domain"]
    assert set(bd) == {"communication", "creative"}
    # communication: attempts 40, successes 21 → 0.525; avg (0.6*10+0.4*30)/40=0.45
    assert bd["communication"]["count"] == 40
    assert bd["communication"]["success_rate"] == round(21 / 40, 4)
    assert bd["communication"]["avg_score"] == round((0.6 * 10 + 0.4 * 30) / 40, 4)
    assert bd["creative"]["count"] == 20
    assert bd["creative"]["success_rate"] == round(18 / 20, 4)


def test_stats_payload_empty_is_safe(orch):
    payload = orch.get_experience_stats_payload()
    assert payload["total_records"] == 0
    assert payload["by_domain"] == {}


def test_stats_payload_skips_zero_attempt_rows(orch):
    _seed_action_stats(orch, [("communication", "noop", 0, 0, 0.0)])
    payload = orch.get_experience_stats_payload()
    assert "communication" not in payload["by_domain"]


def test_recall_similar_recency(orch):
    _seed_record(orch, "communication", "a1", 0.5, [0.1] * 4, 100.0)
    _seed_record(orch, "communication", "a2", 0.7, [0.2] * 4, 200.0)
    _seed_record(orch, "creative", "a3", 0.9, [0.3] * 4, 300.0)
    res = orch.recall_similar("communication", top_k=5)
    assert len(res) == 2
    # recency-ordered (most recent first) when no current_inner
    assert res[0]["action_taken"] == "a2"
    assert all(r["domain"] == "communication" for r in res)
    assert "similarity" in res[0]


def test_recall_similar_cosine_rerank(orch):
    _seed_record(orch, "communication", "near", 0.5, [1.0, 0.0, 0.0, 0.0], 100.0)
    _seed_record(orch, "communication", "far", 0.5, [0.0, 1.0, 0.0, 0.0], 200.0)
    # current_inner aligned with "near" → it should rank first despite older ts
    res = orch.recall_similar("communication",
                              current_inner=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert res[0]["action_taken"] == "near"
    assert res[0]["similarity"] > res[1]["similarity"]


def test_recall_similar_no_match_empty(orch):
    assert orch.recall_similar("nonexistent_domain") == []


def test_publisher_payload_shape(orch):
    _seed_action_stats(orch, [("communication", "kin_sense", 4, 2, 0.5)])
    pub = ExperienceStatsPublisher(titan_id="T_test")
    payload = pub._compute_payload(orch)
    assert payload["schema_version"] == EXPERIENCE_STATS_SCHEMA_VERSION
    assert "ts" in payload
    assert payload["by_domain"]["communication"]["count"] == 4
    # never exceeds the slot cap
    import msgpack
    assert len(msgpack.packb(payload, use_bin_type=True)) <= EXPERIENCE_STATS_MAX_BYTES


def test_publisher_stub_on_none(orch):
    pub = ExperienceStatsPublisher(titan_id="T_test")
    stub = pub._compute_payload(None)
    assert stub["total_records"] == 0
    assert stub["by_domain"] == {}
    assert stub["schema_version"] == EXPERIENCE_STATS_SCHEMA_VERSION


def test_orchestrator_no_longer_takes_ex_mem():
    import inspect
    sig = inspect.signature(ExperienceOrchestrator.__init__)
    assert "ex_mem" not in sig.parameters
    assert "e_mem" in sig.parameters
