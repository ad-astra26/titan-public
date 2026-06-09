"""Phase 8 / EEL B1 — SkillVerifier unit tests (INV-Syn-20).

Re-pointed 2026-06-09 to the EEL B1 outcome × cells model (SPEC v0.29.0 /
D-SPEC-153): the skill under test is built via the per-oracle-verified-use path
(its positive cell's `compiled_from` = the scoring tool-call TXs), `read_skill`
exposes that merged lineage for the INV-Syn-20 re-verify, and "rejected" now
flips the cells negative (utility_score → 0.0) rather than setting -1.0.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import duckdb
import pytest

from titan_hcl.synthesis.skill_store import ProceduralSkillStore, compute_skill_id
from titan_hcl.synthesis.skill_verifier import SkillVerifier


SKILL_A = compute_skill_id("o", "skill-a")  # built with lineage [tx_1, tx_2]


@pytest.fixture()
def store(tmp_path):
    conn = duckdb.connect(":memory:")
    s = ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "f.faiss"),
        snapshot_path=str(tmp_path / "s.json"),
        embedder=None,
    )
    # A promoted positive whose cell lineage (compiled_from) = [tx_1, tx_2].
    s.enqueue_score_event(oracle_id="o", goal_class="skill-a", task_shape="t",
                          success=True, parent_tool_call_tx="tx_1", ts=1.0)
    s.enqueue_score_event(oracle_id="o", goal_class="skill-a", task_shape="t",
                          success=True, parent_tool_call_tx="tx_2", ts=2.0)
    s.drain_score_events()
    return s


@pytest.fixture()
def good_chain_reader():
    reader = MagicMock()
    reader.read_tx_by_content_hash = lambda h: {"content_hash": h, "fork": "procedural"}
    return reader


@pytest.fixture()
def writer():
    w = MagicMock()
    w.write_skill_lifecycle_tx = MagicMock(return_value="lc_tx")
    return w


# ── verify_skill paths ─────────────────────────────────────────────────


def test_verify_unknown_skill_returns_false(store, good_chain_reader, writer):
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader,
                      outer_memory_writer=writer)
    assert v.verify_skill("nonexistent") is False


def test_verify_skill_empty_id_returns_false(store, good_chain_reader, writer):
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader,
                      outer_memory_writer=writer)
    assert v.verify_skill("") is False


def test_verify_skill_already_verified_short_circuits(store, good_chain_reader, writer):
    store.mark_verified(SKILL_A)
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader,
                      outer_memory_writer=writer)
    good_chain_reader.read_tx_by_content_hash = MagicMock(
        side_effect=AssertionError("should not be called"))
    assert v.verify_skill(SKILL_A) is True  # verified + positive cell (utility 1.0)
    writer.write_skill_lifecycle_tx.assert_not_called()


def test_verify_skill_empty_compiled_from_rejects(store, writer, good_chain_reader):
    # A bare outcome (no cells) → empty merged compiled_from → malformed.
    sid = store.ensure_outcome(oracle_id="o", goal_class="empty-lineage")
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader,
                      outer_memory_writer=writer)
    assert v.verify_skill(sid) is False
    assert store.read_skill(sid)["verified_at"] is not None  # rejected → verified_at set


def test_verify_skill_miss_rejects(store, writer):
    reader = MagicMock()
    reader.read_tx_by_content_hash = lambda h: None if h == "tx_2" else {"content_hash": h}
    v = SkillVerifier(skill_store=store, chain_reader=reader, outer_memory_writer=writer)
    assert v.verify_skill(SKILL_A) is False
    row = store.read_skill(SKILL_A)
    assert row["verified_at"] is not None
    assert row["utility_score"] == 0.0  # all cells flipped negative on reject
    assert all(c["polarity"] == "negative" for c in row["cells"])
    writer.write_skill_lifecycle_tx.assert_called_once()
    kwargs = writer.write_skill_lifecycle_tx.call_args.kwargs
    assert kwargs["event_kind"] == "rejected"
    assert "chain_resolve_miss" in kwargs["reason"]


def test_verify_skill_chain_exception_rejects(store, writer):
    reader = MagicMock()
    reader.read_tx_by_content_hash = MagicMock(side_effect=RuntimeError("disk gone"))
    v = SkillVerifier(skill_store=store, chain_reader=reader, outer_memory_writer=writer)
    assert v.verify_skill(SKILL_A) is False
    assert "chain_reader_exception" in writer.write_skill_lifecycle_tx.call_args.kwargs["reason"]


def test_verify_skill_hash_mismatch_rejects(store, writer):
    reader = MagicMock()
    reader.read_tx_by_content_hash = lambda h: {"content_hash": "DIFFERENT", "fork": "procedural"}
    v = SkillVerifier(skill_store=store, chain_reader=reader, outer_memory_writer=writer)
    assert v.verify_skill(SKILL_A) is False
    assert "content_hash_mismatch" in writer.write_skill_lifecycle_tx.call_args.kwargs["reason"]


def test_verify_skill_accepts_when_all_hashes_resolve(store, good_chain_reader, writer):
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader,
                      outer_memory_writer=writer)
    assert v.verify_skill(SKILL_A) is True
    row = store.read_skill(SKILL_A)
    assert row["verified_at"] is not None and row["utility_score"] > 0.0
    writer.write_skill_lifecycle_tx.assert_called_once()
    assert writer.write_skill_lifecycle_tx.call_args.kwargs["event_kind"] == "verified"


def test_verify_skill_accepts_when_reader_returns_block_without_hash_field(store, writer):
    reader = MagicMock()
    reader.read_tx_by_content_hash = lambda h: {"fork": "procedural", "height": 100}
    v = SkillVerifier(skill_store=store, chain_reader=reader, outer_memory_writer=writer)
    assert v.verify_skill(SKILL_A) is True


# ── is_eligible ────────────────────────────────────────────────────────


def test_is_eligible_false_when_unknown(store, good_chain_reader, writer):
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader, outer_memory_writer=writer)
    assert v.is_eligible("missing") is False


def test_is_eligible_false_when_not_verified(store, good_chain_reader, writer):
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader, outer_memory_writer=writer)
    assert v.is_eligible(SKILL_A) is False  # has a positive cell but not yet verified


def test_is_eligible_false_when_rejected(store, good_chain_reader, writer):
    store.mark_rejected(SKILL_A, reason="test")  # flips cells negative, utility 0.0
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader, outer_memory_writer=writer)
    assert v.is_eligible(SKILL_A) is False


def test_is_eligible_true_when_verified_and_positive(store, good_chain_reader, writer):
    store.mark_verified(SKILL_A)
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader, outer_memory_writer=writer)
    assert v.is_eligible(SKILL_A) is True  # verified + positive cell (utility 1.0)


# ── Bus events ─────────────────────────────────────────────────────────


def test_meta_skill_verified_emitted(store, good_chain_reader, writer):
    emitted = []
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader,
                      outer_memory_writer=writer,
                      bus_emit=lambda ev, p: emitted.append((ev, p)))
    v.verify_skill(SKILL_A)
    assert any(ev == "META_SKILL_VERIFIED" for ev, _ in emitted)


def test_meta_skill_rejected_emitted(store, writer):
    reader = MagicMock()
    reader.read_tx_by_content_hash = lambda h: None
    emitted = []
    v = SkillVerifier(skill_store=store, chain_reader=reader, outer_memory_writer=writer,
                      bus_emit=lambda ev, p: emitted.append((ev, p)))
    v.verify_skill(SKILL_A)
    assert any(ev == "META_SKILL_REJECTED" for ev, _ in emitted)


def test_writer_exception_is_swallowed(store, good_chain_reader):
    writer = MagicMock()
    writer.write_skill_lifecycle_tx = MagicMock(side_effect=RuntimeError("disk"))
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader,
                      outer_memory_writer=writer)
    assert v.verify_skill(SKILL_A) is True
    assert store.read_skill(SKILL_A)["verified_at"] is not None
