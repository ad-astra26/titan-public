"""Phase 4 — consolidation default mine + LLM-propose tests (§P4.G prod wiring).

Covers `titan_hcl/synthesis/consolidation_defaults.py`:
- LLM-response parser handles well-formed + malformed input
- LLM provider exception → reject with diagnostic reason
- default_mine_recent_txs reads from a real SQLite block_index table
- mine excludes excluded_forks
- mine handles missing DB file gracefully
- prompt builder produces a compact prompt
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile

import pytest

from titan_hcl.synthesis.consolidation import Cluster, LLMProposal, TxCandidate
from titan_hcl.synthesis.consolidation_defaults import (
    _build_cluster_prompt,
    _parse_llm_response,
    default_mine_recent_txs,
    make_default_llm_propose,
)


# ── LLM-response parser ─────────────────────────────────────────────


def test_parse_new_concept_response():
    resp = """ACTION: new_concept
CONCEPT_ID: linux_terminal
NAME: Linux terminal
MEMORY_TYPE: declarative
REASON: cluster centered on terminal-use experience"""
    p = _parse_llm_response(resp)
    assert p.action == "new_concept"
    assert p.concept_id == "linux_terminal"
    assert p.proposed_name == "Linux terminal"
    assert p.memory_type == "declarative"
    assert "terminal-use" in p.reason


def test_parse_version_bump_response():
    resp = """ACTION: version_bump
CONCEPT_ID: solana_rpc
NAME: Solana RPC
MEMORY_TYPE: procedural
REASON: enrichment from 5 new TXs"""
    p = _parse_llm_response(resp)
    assert p.action == "version_bump"
    assert p.concept_id == "solana_rpc"
    assert p.memory_type == "procedural"


def test_parse_reject_response():
    resp = """ACTION: reject
REASON: noise — TXs not coherent"""
    p = _parse_llm_response(resp)
    assert p.action == "reject"
    assert "noise" in p.reason


def test_parse_concept_id_sanitized():
    """Spaces + dashes → underscores; case → lowercase."""
    resp = """ACTION: new_concept
CONCEPT_ID: Metaplex NFT-Minting
NAME: Metaplex NFT minting
MEMORY_TYPE: procedural"""
    p = _parse_llm_response(resp)
    assert p.concept_id == "metaplex_nft_minting"


def test_parse_unknown_action_falls_back_to_reject():
    resp = "ACTION: maybe_someday\nCONCEPT_ID: x"
    p = _parse_llm_response(resp)
    assert p.action == "reject"


def test_parse_empty_concept_id_falls_back_to_reject():
    resp = "ACTION: new_concept\nCONCEPT_ID:\nNAME: X"
    p = _parse_llm_response(resp)
    assert p.action == "reject"
    assert p.reason == "llm_returned_empty_concept_id"


def test_parse_missing_memory_type_defaults_to_meta():
    resp = "ACTION: new_concept\nCONCEPT_ID: x\nNAME: X"
    p = _parse_llm_response(resp)
    assert p.memory_type == "meta"


def test_parse_malformed_response_safe():
    """Non-protocol response → reject without raising."""
    for bad in ("", "just some text without prefixes", "ACTION", "::::"):
        p = _parse_llm_response(bad)
        assert p.action == "reject"


# ── LLM-propose provider integration ────────────────────────────────


class _FakeProvider:
    """Mimics inference.base.InferenceProvider's async surface."""

    def __init__(self, response: str = "", raise_exc: Exception | None = None):
        self._response = response
        self._raise = raise_exc

    async def complete(self, **_kw) -> str:
        if self._raise is not None:
            raise self._raise
        return self._response


def test_llm_propose_calls_provider_and_parses():
    fake = _FakeProvider(response="""ACTION: new_concept
CONCEPT_ID: linux_basics
NAME: Linux basics
MEMORY_TYPE: declarative
REASON: emergent""")
    propose = make_default_llm_propose(fake)
    cluster = Cluster(members=[
        TxCandidate(tx_hash="t1", fork="declarative", tags=("topic:linux",), embedding=None),
    ])
    p = propose(cluster)
    assert p.action == "new_concept"
    assert p.concept_id == "linux_basics"


def test_llm_propose_provider_exception_returns_reject():
    fake = _FakeProvider(raise_exc=RuntimeError("ollama down"))
    propose = make_default_llm_propose(fake)
    cluster = Cluster(members=[
        TxCandidate(tx_hash="t1", fork="declarative", tags=("topic:x",), embedding=None),
    ])
    p = propose(cluster)
    assert p.action == "reject"
    assert "RuntimeError" in p.reason


def test_llm_propose_empty_response_returns_reject():
    fake = _FakeProvider(response="")
    propose = make_default_llm_propose(fake)
    cluster = Cluster(members=[
        TxCandidate(tx_hash="t1", fork="declarative", tags=(), embedding=None),
    ])
    p = propose(cluster)
    assert p.action == "reject"


# ── Cluster prompt builder ─────────────────────────────────────────


def test_cluster_prompt_includes_size_forks_tags():
    cluster = Cluster(members=[
        TxCandidate(
            tx_hash="abcdef1234567890" + "0" * 48,
            fork="declarative",
            tags=("topic:linux", "topic:terminal"),
            embedding=None,
        ),
        TxCandidate(
            tx_hash="cafef00d" + "0" * 56,
            fork="declarative",
            tags=("topic:linux",),
            embedding=None,
        ),
    ])
    prompt = _build_cluster_prompt(cluster)
    assert "Cluster size: 2" in prompt
    assert "declarative" in prompt
    assert "topic:linux" in prompt
    # Sample hash prefixes use only first 12 chars (per builder).
    assert "abcdef123456" in prompt
    assert "cafef00d0000" in prompt


# ── Default mine ────────────────────────────────────────────────────


def _seed_block_index(db_path: str, rows: list[dict]) -> None:
    """Create a minimal block_index table matching the timechain_v2 schema
    + insert the given rows."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS block_index ("
            "tx_hash BLOB PRIMARY KEY, fork_name TEXT, tx_type TEXT, "
            "source TEXT, significance REAL, tags TEXT, ts REAL)"
        )
        for r in rows:
            conn.execute(
                "INSERT OR REPLACE INTO block_index "
                "(tx_hash, fork_name, tx_type, source, significance, tags, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    bytes.fromhex(r["tx_hash"]),
                    r["fork_name"], r.get("tx_type", "thought"),
                    r.get("source", "test"),
                    r.get("significance", 0.5),
                    json.dumps(r.get("tags", [])),
                    r["ts"],
                ),
            )
        conn.commit()
    finally:
        conn.close()


def test_default_mine_returns_tx_candidates():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "index.db")
        _seed_block_index(db, [
            {"tx_hash": "11" * 32, "fork_name": "declarative",
             "tags": ["topic:linux"], "ts": 1000.0},
            {"tx_hash": "22" * 32, "fork_name": "procedural",
             "tags": ["topic:ssh"], "ts": 1100.0},
        ])
        txs = default_mine_recent_txs(
            since_ts=500.0, exclude_forks=set(),
            index_db_path=db,
        )
        assert len(txs) == 2
        assert all(isinstance(t, TxCandidate) for t in txs)
        # Ordered by ts DESC.
        assert txs[0].tx_hash.startswith("22")
        assert txs[1].tx_hash.startswith("11")
        # Tag parsing.
        assert "topic:ssh" in txs[0].tags
        # Embedding None per P4 contract.
        assert txs[0].embedding is None


def test_default_mine_excludes_forks():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "index.db")
        _seed_block_index(db, [
            {"tx_hash": "aa" * 32, "fork_name": "meta",
             "tags": ["topic:noise"], "ts": 1000.0},
            {"tx_hash": "bb" * 32, "fork_name": "declarative",
             "tags": ["topic:signal"], "ts": 1100.0},
            {"tx_hash": "cc" * 32, "fork_name": "conversation",
             "tags": ["chat"], "ts": 1200.0},
        ])
        txs = default_mine_recent_txs(
            since_ts=0.0,
            exclude_forks={"meta", "conversation"},
            index_db_path=db,
        )
        forks = [t.fork for t in txs]
        assert forks == ["declarative"]


def test_default_mine_since_ts_filter():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "index.db")
        _seed_block_index(db, [
            {"tx_hash": "11" * 32, "fork_name": "declarative",
             "tags": [], "ts": 100.0},
            {"tx_hash": "22" * 32, "fork_name": "declarative",
             "tags": [], "ts": 2000.0},
        ])
        txs = default_mine_recent_txs(
            since_ts=1000.0, exclude_forks=set(), index_db_path=db,
        )
        assert len(txs) == 1
        assert txs[0].tx_hash.startswith("22")


def test_default_mine_missing_db_returns_empty():
    """Production path: chain might not have written index.db yet on first
    boot. Soft-fail to empty result so the consolidation pass just records
    a 'skipped' summary and the next pass tries again."""
    txs = default_mine_recent_txs(
        since_ts=0.0, exclude_forks=set(),
        index_db_path="/nonexistent/path/index.db",
    )
    assert txs == []
