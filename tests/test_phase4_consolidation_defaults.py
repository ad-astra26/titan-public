"""Consolidation default mine + LLM-propose tests (§P4.G prod wiring; mine
re-targeted to the sidecar by RFP_synthesis_spine_reads_real_data §7.D).

Covers `titan_hcl/synthesis/consolidation_defaults.py`:
- LLM-response parser handles well-formed + malformed input
- LLM provider exception → reject with diagnostic reason
- default_mine_recent_thoughts reads PROMOTED THOUGHTS from the content sidecar
  (real content, keyed by per-TX hash) — newest-first, since_ts-windowed,
  fork-exclusion defensive, missing-sidecar soft-fail
- prompt builder carries REAL thought content (D3), not hash prefixes
"""
from __future__ import annotations

import os
import tempfile

from titan_hcl.synthesis.consolidation import Cluster, LLMProposal, TxCandidate
from titan_hcl.synthesis.consolidation_defaults import (
    _build_cluster_prompt,
    _parse_llm_response,
    default_mine_recent_thoughts,
    derive_domain_hint,
    make_default_llm_propose,
)
from titan_hcl.synthesis.thought_sidecar import ThoughtSidecar


# ── domain split: self (Titan-about-himself) vs social (interpersonal) ──
# RFP_titan_authored_soul_diary §7.P2 — `self` was lumped with social; split
# 2026-06-10 so the narrative-SELF partition stays clean (INV-SD-16).


def test_domain_self_for_titan_about_himself():
    assert derive_domain_hint("Daily Soul-Diary Reflection") == "self"
    assert derive_domain_hint("Titan Sovereignty Journey") == "self"
    assert derive_domain_hint("Self-Refactor Patterns") == "self"
    assert derive_domain_hint("Introspective Self-Inspection") == "self"


def test_domain_social_for_interpersonal_content():
    # The formerly-"self"-mislabeled interpersonal content now routes to social.
    assert derive_domain_hint("Seaside Philosophical Dialogue") == "social"
    assert derive_domain_hint("Interpersonal Dialogue Fragments") == "social"
    assert derive_domain_hint("User Interpersonal Dynamics") == "social"


def test_domain_self_wins_over_social_first_match():
    # A name carrying BOTH a self marker and a social cue → self (most-specific).
    assert derive_domain_hint("Self-Reflection on a Dialogue") == "self"


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
        TxCandidate(tx_hash="t1", fork="declarative", tags=(), embedding=None,
                    content_summary="I use the Linux terminal daily."),
    ])
    p = propose(cluster)
    assert p.action == "new_concept"
    assert p.concept_id == "linux_basics"


def test_llm_propose_provider_exception_returns_reject():
    fake = _FakeProvider(raise_exc=RuntimeError("ollama down"))
    propose = make_default_llm_propose(fake)
    cluster = Cluster(members=[
        TxCandidate(tx_hash="t1", fork="declarative", tags=(), embedding=None),
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


# ── Cluster prompt builder (now carries REAL content — D3) ──────────


def test_cluster_prompt_includes_size_forks_and_real_content():
    cluster = Cluster(members=[
        TxCandidate(
            tx_hash="abcdef1234567890" + "0" * 48,
            fork="declarative",
            tags=("topic:linux",),
            embedding=None,
            content_summary="I prefer the zsh shell on Linux for the terminal.",
        ),
        TxCandidate(
            tx_hash="cafef00d" + "0" * 56,
            fork="declarative",
            tags=("topic:linux",),
            embedding=None,
            content_summary="The Linux terminal is where I do most of my work.",
        ),
    ])
    prompt = _build_cluster_prompt(cluster)
    assert "Cluster size: 2" in prompt
    assert "declarative" in prompt
    assert "topic:linux" in prompt
    # The D3 fix: real content drives the prompt (was hash prefixes).
    assert "zsh shell" in prompt
    assert "Linux terminal" in prompt


def test_cluster_prompt_truncates_long_content():
    long_text = "x" * 1000
    cluster = Cluster(members=[
        TxCandidate(tx_hash="aa" * 32, fork="episodic", tags=(),
                    embedding=None, content_summary=long_text),
    ])
    prompt = _build_cluster_prompt(cluster, max_chars=240)
    assert "…" in prompt          # truncation marker
    assert "x" * 1000 not in prompt  # not the full blob


def test_cluster_prompt_handles_empty_content():
    cluster = Cluster(members=[
        TxCandidate(tx_hash="aa" * 32, fork="episodic", tags=(),
                    embedding=None, content_summary=""),
    ])
    prompt = _build_cluster_prompt(cluster)
    assert "no content available" in prompt  # never crashes on contentless rows


# ── Default mine — sidecar (promoted thoughts) ──────────────────────


def _seed_sidecar(data_dir: str, rows: list[dict]) -> None:
    """Write rows into a real thought_sidecar.db via the production writer."""
    sc = ThoughtSidecar(data_dir)
    try:
        for r in rows:
            sc.put(
                tx_hash=r["tx_hash"], node_id=r.get("node_id", 1),
                user_prompt=r.get("user_prompt", ""),
                agent_response=r.get("agent_response", ""),
                memory_type=r.get("memory_type", "episodic"),
                fork=r.get("fork", "episodic"), ts=r["ts"],
            )
    finally:
        sc.close()


def test_mine_thoughts_returns_candidates_newest_first():
    with tempfile.TemporaryDirectory() as tmp:
        _seed_sidecar(tmp, [
            {"tx_hash": "11" * 32, "user_prompt": "I race karts",
             "fork": "episodic", "ts": 1000.0},
            {"tx_hash": "22" * 32, "user_prompt": "I love the Brno circuit",
             "fork": "declarative", "ts": 1100.0},
        ])
        txs = default_mine_recent_thoughts(
            since_ts=500.0, exclude_forks=set(), data_dir=tmp)
        assert len(txs) == 2
        assert all(isinstance(t, TxCandidate) for t in txs)
        # Ordered by ts DESC.
        assert txs[0].tx_hash.startswith("22")
        assert txs[1].tx_hash.startswith("11")
        # Real content is carried (the crux fix), tags empty, embedding deferred.
        assert "Brno" in txs[0].content_summary
        assert txs[0].tags == ()
        assert txs[0].embedding is None
        assert txs[0].fork == "declarative"


def test_mine_thoughts_concatenates_prompt_and_response():
    with tempfile.TemporaryDirectory() as tmp:
        _seed_sidecar(tmp, [
            {"tx_hash": "33" * 32, "user_prompt": "what is my hobby?",
             "agent_response": "You race go-karts on weekends.",
             "fork": "episodic", "ts": 1000.0},
        ])
        txs = default_mine_recent_thoughts(
            since_ts=0.0, exclude_forks=set(), data_dir=tmp)
        assert len(txs) == 1
        assert "hobby" in txs[0].content_summary
        assert "go-karts" in txs[0].content_summary


def test_mine_thoughts_since_ts_filter():
    with tempfile.TemporaryDirectory() as tmp:
        _seed_sidecar(tmp, [
            {"tx_hash": "11" * 32, "user_prompt": "old", "ts": 100.0},
            {"tx_hash": "22" * 32, "user_prompt": "new", "ts": 2000.0},
        ])
        txs = default_mine_recent_thoughts(
            since_ts=1000.0, exclude_forks=set(), data_dir=tmp)
        assert len(txs) == 1
        assert txs[0].tx_hash.startswith("22")


def test_mine_thoughts_excludes_forks_defensively():
    with tempfile.TemporaryDirectory() as tmp:
        _seed_sidecar(tmp, [
            {"tx_hash": "aa" * 32, "user_prompt": "noise",
             "fork": "meta", "ts": 1000.0},
            {"tx_hash": "bb" * 32, "user_prompt": "signal",
             "fork": "declarative", "ts": 1100.0},
        ])
        txs = default_mine_recent_thoughts(
            since_ts=0.0, exclude_forks={"meta", "conversation"}, data_dir=tmp)
        assert [t.fork for t in txs] == ["declarative"]


def test_mine_thoughts_missing_sidecar_returns_empty():
    """First boot before any promotion: no sidecar yet → soft-fail to []."""
    with tempfile.TemporaryDirectory() as tmp:
        txs = default_mine_recent_thoughts(
            since_ts=0.0, exclude_forks=set(), data_dir=tmp)
        assert txs == []
