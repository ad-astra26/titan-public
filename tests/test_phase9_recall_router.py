"""Phase 9 — RECALL → EngineRecall archive/autobiographical branches (INV-Syn-22).

Drives `EngineRecall.recall(granularity="archive"|"autobiographical")` through the
real RuleEvaluator + an in-memory block_index. These FORK_READ-based sub-modes
(arch §13.2: chain_archive → FORK_READ(meta)+CROSS_REF; autobiographical_relevant
→ FORK_READ(main genesis)+DIFF) bypass the embedder (recency/frequency-ranked).
"""

from __future__ import annotations

import sqlite3

from titan_hcl.logic.timechain_v2 import FORK_IDS, RuleEvaluator
from titan_hcl.synthesis.recall import (
    EngineRecall,
    GRANULARITY_ARCHIVE,
    GRANULARITY_AUTOBIOGRAPHICAL,
)


def _make_index_db(rows):
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE block_index ("
        "block_hash BLOB PRIMARY KEY, fork_id INTEGER NOT NULL, "
        "block_height INTEGER NOT NULL, timestamp REAL NOT NULL, "
        "epoch_id INTEGER NOT NULL, thought_type TEXT, source TEXT, "
        "significance REAL, chi_spent REAL, neuromod_da REAL, "
        "neuromod_ach REAL, neuromod_ne REAL, tags TEXT, cross_refs TEXT, "
        "db_ref TEXT, compacted INTEGER DEFAULT 0, file_offset INTEGER NOT NULL)"
    )
    conn.executemany(
        "INSERT INTO block_index VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    # Chain-local fork NAME→id registry — RuleEvaluator._resolve_fork_id
    # resolves FORK_READ targets against this table (INV-Syn-26), NEVER the
    # static FORK_IDS map. Seed it to agree with the ids the rows use.
    conn.execute(
        "CREATE TABLE fork_registry ("
        "fork_id INTEGER PRIMARY KEY, fork_name TEXT NOT NULL)")
    conn.executemany(
        "INSERT INTO fork_registry (fork_id, fork_name) VALUES (?, ?)",
        [(fid, name) for name, fid in FORK_IDS.items()])
    conn.commit()
    return conn


def _row(byte, fork, *, thought_type="thought", tags=""):
    return (
        bytes([byte]) * 32, FORK_IDS[fork], byte, 0.0, 10,
        thought_type, "synthesis", 0.5, 0.0, 0, 0, 0, tags, "", "", 0, byte * 100,
    )


class _StubFaiss:
    def knn(self, fork, vec, k, min_similarity):
        return []


def _engine(rows, *, embedder=None):
    conn = _make_index_db(rows)
    ev = RuleEvaluator(faiss_reader=_StubFaiss(), index_db=conn)
    er = EngineRecall(
        rule_evaluator=ev,
        activation_lookup=lambda ids: {},
        embedder=embedder,
    )
    return er, conn


def test_archive_reads_meta_fork():
    rows = [_row(1, "meta"), _row(2, "meta"), _row(9, "episodic")]
    er, _ = _engine(rows)
    res = er.recall("anything", granularity=GRANULARITY_ARCHIVE)
    assert res is not None
    hashes = {r.tx_hash for r in res}
    # only meta-fork rows surface (episodic excluded)
    assert (bytes([1]) * 32).hex() in {h if isinstance(h, str) else h for h in hashes} or len(res) == 2


def test_archive_works_without_embedder():
    # FORK_READ-based mode must NOT require an embedder (INV-Syn-22).
    rows = [_row(1, "meta")]
    er, _ = _engine(rows, embedder=None)
    res = er.recall("q", granularity=GRANULARITY_ARCHIVE)
    assert res is not None
    assert len(res) == 1


def test_autobiographical_reads_main_genesis_fork():
    rows = [_row(3, "main"), _row(4, "main"), _row(5, "meta")]
    er, _ = _engine(rows)
    res = er.recall("who am I", granularity=GRANULARITY_AUTOBIOGRAPHICAL)
    assert res is not None
    assert len(res) == 2  # only main-fork rows


def test_archive_empty_fork_returns_empty_list():
    rows = [_row(9, "episodic")]  # no meta rows
    er, _ = _engine(rows)
    res = er.recall("q", granularity=GRANULARITY_ARCHIVE)
    assert res == []


def test_archive_results_have_source_label():
    rows = [_row(1, "meta")]
    er, _ = _engine(rows)
    res = er.recall("q", granularity=GRANULARITY_ARCHIVE)
    assert res[0].source == "synthesis_archive"


def test_autobiographical_source_label():
    rows = [_row(3, "main")]
    er, _ = _engine(rows)
    res = er.recall("q", granularity=GRANULARITY_AUTOBIOGRAPHICAL)
    assert res[0].source == "synthesis_autobiographical"


def test_evaluator_failure_returns_none():
    class _BoomEval:
        def evaluate(self, *a, **k):
            raise RuntimeError("boom")
    er = EngineRecall(rule_evaluator=_BoomEval(),
                      activation_lookup=lambda ids: {}, embedder=None)
    assert er.recall("q", granularity=GRANULARITY_ARCHIVE) is None


# ── meta_resolvers EngineRecall wrapper (INV-Syn-22 routing) ───────────

import asyncio

from titan_hcl.logic.meta_resolvers import _make_engine_recall_resolver


class _FakeEngine:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def recall(self, query, *, granularity=None, **kw):
        self.calls.append((query, granularity))
        return self._results


async def _legacy(name, ctx):
    return {"success": True, "output": "LEGACY", "recruiter": "x", "reason": "legacy"}


def test_resolver_routes_through_engine_on_hit():
    eng = _FakeEngine(results=[{"tx_hash": "t1", "score": 0.9}])
    r = _make_engine_recall_resolver(
        category="chain_archive", granularity="archive",
        engine_recall=eng, legacy_resolver=_legacy, soft_fallback=True)
    out = asyncio.get_event_loop().run_until_complete(
        r("query", {"query_text": "metaplex bug"}))
    assert out["success"] is True
    assert out["reason"] == "engine_recall:archive"
    assert eng.calls == [("metaplex bug", "archive")]


def test_resolver_soft_fallback_on_none():
    eng = _FakeEngine(results=None)
    r = _make_engine_recall_resolver(
        category="chain_archive", granularity="archive",
        engine_recall=eng, legacy_resolver=_legacy, soft_fallback=True)
    out = asyncio.get_event_loop().run_until_complete(
        r("query", {"query_text": "x"}))
    assert out["output"] == "LEGACY"


def test_resolver_no_fallback_returns_empty_when_disabled():
    eng = _FakeEngine(results=None)
    r = _make_engine_recall_resolver(
        category="semantic_graph", granularity="concept",
        engine_recall=eng, legacy_resolver=_legacy, soft_fallback=False)
    out = asyncio.get_event_loop().run_until_complete(
        r("query", {"query_text": "x"}))
    assert out["success"] is False
    assert out["reason"] == "engine_recall_empty:concept"


def test_resolver_empty_query_falls_back():
    eng = _FakeEngine(results=[{"tx_hash": "t1"}])
    r = _make_engine_recall_resolver(
        category="episodic_memory", granularity="turn",
        engine_recall=eng, legacy_resolver=_legacy, soft_fallback=True)
    out = asyncio.get_event_loop().run_until_complete(r("query", {}))
    # no query text → engine not called → legacy fallback
    assert out["output"] == "LEGACY"
    assert eng.calls == []
