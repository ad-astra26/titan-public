"""Phase 8 — SkillVerifier unit tests (D-SPEC-PHASE8 / INV-Syn-20).

Covers:
- verify_skill returns False when skill_id missing / unknown
- verify_skill returns True when already verified
- verify_skill rejects skill with empty compiled_from
- verify_skill rejects when any compiled_from tx_hash misses chain
- verify_skill rejects when chain_reader raises
- verify_skill rejects when content_hash mismatch (corrupted block)
- verify_skill accepts when all compiled_from hashes resolve + match
- is_eligible: True only when verified_at AND utility_score >= 0.0
- META_SKILL_VERIFIED / META_SKILL_REJECTED bus events emitted
- write_skill_lifecycle_tx called on both accept + reject
"""
from __future__ import annotations

from unittest.mock import MagicMock

import duckdb
import pytest

from titan_hcl.synthesis.skill_store import ProceduralSkillStore
from titan_hcl.synthesis.skill_verifier import SkillVerifier


@pytest.fixture()
def store(tmp_path):
    conn = duckdb.connect(":memory:")
    s = ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "f.faiss"),
        snapshot_path=str(tmp_path / "s.json"),
        embedder=None,
    )
    s.persist_skill(
        skill_id="skill_A", name="a", nl_description="a desc",
        executable_spec={}, compiled_from=["tx_1", "tx_2"],
    )
    return s


@pytest.fixture()
def good_chain_reader():
    # All hashes resolve; content_hash matches
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
    # Pre-mark verified
    store.mark_verified("skill_A")
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader,
                      outer_memory_writer=writer)
    # Re-verify should NOT touch chain_reader (already verified)
    good_chain_reader.read_tx_by_content_hash = MagicMock(side_effect=AssertionError("should not be called"))
    assert v.verify_skill("skill_A") is True
    writer.write_skill_lifecycle_tx.assert_not_called()


def test_verify_skill_empty_compiled_from_rejects(tmp_path, writer, good_chain_reader):
    """If somehow compiled_from is empty in production, treat as malformed."""
    conn = duckdb.connect(":memory:")
    s = ProceduralSkillStore(
        duckdb_conn=conn, faiss_path=str(tmp_path / "f.faiss"),
        snapshot_path=str(tmp_path / "s.json"), embedder=None,
    )
    # Bypass the constructor's compiled_from non-empty guard by direct INSERT
    s._db.execute(
        "INSERT INTO procedural_skills (skill_id, name, nl_description, "
        "executable_spec, compiled_from, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ["skill_E", "e", "e desc", "{}", "[]", 1.0],
    )
    v = SkillVerifier(skill_store=s, chain_reader=good_chain_reader, outer_memory_writer=writer)
    assert v.verify_skill("skill_E") is False
    row = s.read_skill("skill_E")
    assert row["utility_score"] == pytest.approx(-1.0)


def test_verify_skill_miss_rejects(store, writer):
    """Any source tx_hash missing from chain → reject."""
    reader = MagicMock()
    reader.read_tx_by_content_hash = lambda h: None if h == "tx_2" else {"content_hash": h}
    v = SkillVerifier(skill_store=store, chain_reader=reader, outer_memory_writer=writer)
    assert v.verify_skill("skill_A") is False
    row = store.read_skill("skill_A")
    assert row["utility_score"] == pytest.approx(-1.0)
    assert row["verified_at"] is not None
    # Lifecycle TX written with kind=rejected
    writer.write_skill_lifecycle_tx.assert_called_once()
    kwargs = writer.write_skill_lifecycle_tx.call_args.kwargs
    assert kwargs["event_kind"] == "rejected"
    assert "chain_resolve_miss" in kwargs["reason"]


def test_verify_skill_chain_exception_rejects(store, writer):
    reader = MagicMock()
    reader.read_tx_by_content_hash = MagicMock(side_effect=RuntimeError("disk gone"))
    v = SkillVerifier(skill_store=store, chain_reader=reader, outer_memory_writer=writer)
    assert v.verify_skill("skill_A") is False
    kwargs = writer.write_skill_lifecycle_tx.call_args.kwargs
    assert "chain_reader_exception" in kwargs["reason"]


def test_verify_skill_hash_mismatch_rejects(store, writer):
    reader = MagicMock()
    # Return a block with content_hash != expected
    reader.read_tx_by_content_hash = lambda h: {"content_hash": "DIFFERENT", "fork": "procedural"}
    v = SkillVerifier(skill_store=store, chain_reader=reader, outer_memory_writer=writer)
    assert v.verify_skill("skill_A") is False
    kwargs = writer.write_skill_lifecycle_tx.call_args.kwargs
    assert "content_hash_mismatch" in kwargs["reason"]


def test_verify_skill_accepts_when_all_hashes_resolve(store, good_chain_reader, writer):
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader,
                      outer_memory_writer=writer)
    assert v.verify_skill("skill_A") is True
    row = store.read_skill("skill_A")
    assert row["verified_at"] is not None
    assert row["utility_score"] >= 0.0
    writer.write_skill_lifecycle_tx.assert_called_once()
    assert writer.write_skill_lifecycle_tx.call_args.kwargs["event_kind"] == "verified"


def test_verify_skill_accepts_when_reader_returns_block_without_hash_field(store, writer):
    """Reader may return {fork, ...} without explicit content_hash — treat
    non-None as authoritative (the reader's contract is to return None on miss)."""
    reader = MagicMock()
    reader.read_tx_by_content_hash = lambda h: {"fork": "procedural", "height": 100}
    v = SkillVerifier(skill_store=store, chain_reader=reader, outer_memory_writer=writer)
    assert v.verify_skill("skill_A") is True


# ── is_eligible ────────────────────────────────────────────────────────


def test_is_eligible_false_when_unknown(store, good_chain_reader, writer):
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader, outer_memory_writer=writer)
    assert v.is_eligible("missing") is False


def test_is_eligible_false_when_not_verified(store, good_chain_reader, writer):
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader, outer_memory_writer=writer)
    assert v.is_eligible("skill_A") is False


def test_is_eligible_false_when_rejected(store, good_chain_reader, writer):
    # Rejected skill = utility_score=-1.0 + verified_at set
    store.mark_rejected("skill_A", reason="test")
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader, outer_memory_writer=writer)
    assert v.is_eligible("skill_A") is False


def test_is_eligible_true_when_verified_and_positive(store, good_chain_reader, writer):
    store.mark_verified("skill_A")
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader, outer_memory_writer=writer)
    assert v.is_eligible("skill_A") is True


# ── Bus events ─────────────────────────────────────────────────────────


def test_meta_skill_verified_emitted(store, good_chain_reader, writer):
    emitted = []
    v = SkillVerifier(
        skill_store=store, chain_reader=good_chain_reader,
        outer_memory_writer=writer,
        bus_emit=lambda ev, p: emitted.append((ev, p)),
    )
    v.verify_skill("skill_A")
    assert any(ev == "META_SKILL_VERIFIED" for ev, _ in emitted)


def test_meta_skill_rejected_emitted(store, writer):
    reader = MagicMock()
    reader.read_tx_by_content_hash = lambda h: None
    emitted = []
    v = SkillVerifier(
        skill_store=store, chain_reader=reader, outer_memory_writer=writer,
        bus_emit=lambda ev, p: emitted.append((ev, p)),
    )
    v.verify_skill("skill_A")
    assert any(ev == "META_SKILL_REJECTED" for ev, _ in emitted)


def test_writer_exception_is_swallowed(store, good_chain_reader):
    writer = MagicMock()
    writer.write_skill_lifecycle_tx = MagicMock(side_effect=RuntimeError("disk"))
    v = SkillVerifier(skill_store=store, chain_reader=good_chain_reader,
                      outer_memory_writer=writer)
    # Should not raise
    assert v.verify_skill("skill_A") is True
    row = store.read_skill("skill_A")
    # Store mutation still happened
    assert row["verified_at"] is not None
