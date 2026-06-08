"""Conn-pool tests for StoreRouter / VerifiedContextBuilder
(RFP_synthesis_decision_authority P4).

Pooled thread-local sqlite connections + a _PooledConn proxy whose close() is a
no-op (so the 14 handlers' `finally: conn.close()` doesn't tear down a reused
connection), backed by a PERSISTENT query pool so the thread-local cache
survives across turns while parallelism (one conn per pool thread) is preserved.
"""

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from titan_hcl.logic.verified_context_builder import (
    ParsedQuery,
    StoreRouter,
    VerifiedContextBuilder,
    _PooledConn,
)


# ── P4 per-session TTL cache — session-stable stores only (identity + profile) ──

def _row(content="r"):
    return {"content": content, "source": "s", "timestamp": 0, "db_ref": "d"}


def test_self_insights_cached_query_independent(tmp_path):
    """self_insights is query-INDEPENDENT → cached by (store, limit); a 2nd
    query with a DIFFERENT parse still hits (handler invoked once)."""
    r = StoreRouter(data_dir=str(tmp_path))
    calls = []
    r._query_self_insights = lambda parsed, limit: (calls.append(1) or [_row("identity")])
    a = r.query_store("self_insights", ParsedQuery(entities=["alpha"]), limit=5)
    b = r.query_store("self_insights", ParsedQuery(entities=["beta", "gamma"]), limit=5)
    assert a == b == [_row("identity")]
    assert len(calls) == 1   # cache hit despite a different parse


def test_social_graph_cached_by_person_entity(tmp_path):
    """social_graph (profile) is keyed by the person entity: same person hits,
    a different person misses."""
    r = StoreRouter(data_dir=str(tmp_path))
    calls = []
    r._query_social_graph = lambda parsed, limit: (calls.append(1) or [_row("profile")])
    alice = ParsedQuery(entities=["alice"], entity_types={"alice": "person"})
    alice2 = ParsedQuery(entities=["alice"], entity_types={"alice": "person"})
    bob = ParsedQuery(entities=["bob"], entity_types={"bob": "person"})
    r.query_store("social_graph", alice, limit=5)
    r.query_store("social_graph", alice2, limit=5)    # same person → hit
    assert len(calls) == 1
    r.query_store("social_graph", bob, limit=5)        # different person → miss
    assert len(calls) == 2


def test_prompt_dependent_store_not_cached(tmp_path):
    """vocabulary is prompt-dependent → NOT in the session-stable allowlist →
    re-queried every turn (the cache can never serve stale prompt-specific recall)."""
    r = StoreRouter(data_dir=str(tmp_path))
    calls = []
    r._query_vocabulary = lambda parsed, limit: (calls.append(1) or [_row("v")])
    p = ParsedQuery(entities=["x"])
    r.query_store("vocabulary", p, limit=5)
    r.query_store("vocabulary", p, limit=5)
    assert len(calls) == 2   # not cached


def test_session_cache_ttl_expiry(tmp_path):
    """An expired entry is not served — the store re-queries."""
    r = StoreRouter(data_dir=str(tmp_path))
    calls = []
    r._query_self_insights = lambda parsed, limit: (calls.append(1) or [_row("id")])
    p = ParsedQuery()
    r.query_store("self_insights", p, limit=5)
    assert len(calls) == 1
    key = r._session_cache_key("self_insights", p, 5)
    _expiry, results = r._session_cache[key]
    r._session_cache[key] = (time.time() - 1.0, results)   # force-expire
    r.query_store("self_insights", p, limit=5)
    assert len(calls) == 2   # expired → re-query


def _make_db(tmp_path, name="inner_memory.db"):
    p = tmp_path / name
    c = sqlite3.connect(str(p))
    c.execute("CREATE TABLE t (x INTEGER)")
    c.execute("INSERT INTO t VALUES (1)")
    c.commit()
    c.close()
    return p


def test_connect_pools_same_conn_same_thread(tmp_path):
    _make_db(tmp_path)
    r = StoreRouter(data_dir=str(tmp_path))
    a = r._connect("inner_memory.db")
    b = r._connect("inner_memory.db")
    assert a is b  # pooled — same wrapper reused (no reconnect)
    assert isinstance(a, _PooledConn)


def test_pooled_close_is_noop(tmp_path):
    _make_db(tmp_path)
    r = StoreRouter(data_dir=str(tmp_path))
    conn = r._connect("inner_memory.db")
    conn.close()  # the handlers' `finally: conn.close()` — must NOT really close
    assert conn.execute("SELECT x FROM t").fetchone()[0] == 1
    assert r._connect("inner_memory.db") is conn  # still the same pooled conn


def test_connect_missing_db_returns_none(tmp_path):
    r = StoreRouter(data_dir=str(tmp_path))
    assert r._connect("does_not_exist.db") is None


def test_missing_db_not_cached_then_appears(tmp_path):
    """A db that doesn't exist yet returns None and is NOT cached, so once it is
    created a later _connect picks it up."""
    r = StoreRouter(data_dir=str(tmp_path))
    assert r._connect("inner_memory.db") is None
    _make_db(tmp_path)
    conn = r._connect("inner_memory.db")
    assert isinstance(conn, _PooledConn)


def test_thread_local_distinct_conns(tmp_path):
    _make_db(tmp_path)
    r = StoreRouter(data_dir=str(tmp_path))
    main_conn = r._connect("inner_memory.db")
    captured = {}

    def worker():
        captured["c"] = r._connect("inner_memory.db")

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert captured["c"] is not main_conn  # each thread owns its own conn
    assert isinstance(captured["c"], _PooledConn)


def test_pooled_conn_forwards_row_factory_and_attrs(tmp_path):
    _make_db(tmp_path)
    r = StoreRouter(data_dir=str(tmp_path))
    conn = r._connect("inner_memory.db")
    row = conn.execute("SELECT x FROM t").fetchone()
    assert row["x"] == 1  # sqlite3.Row factory forwarded through the proxy
    assert conn  # proxy is truthy (handlers' `if not conn` guard)


def test_query_store_works_over_pooled_conn(tmp_path):
    """The real handler path (_query_vocabulary) over a pooled conn, called twice,
    returns rows both times (proves close()-no-op keeps the conn alive)."""
    p = tmp_path / "inner_memory.db"
    c = sqlite3.connect(str(p))
    c.execute(
        "CREATE TABLE vocabulary (word TEXT, word_type TEXT, confidence REAL, "
        "times_produced INTEGER, learning_phase TEXT, created_at REAL)"
    )
    c.execute(
        "INSERT INTO vocabulary VALUES ('raffinesse','noun',0.9,3,'mastered',1.0)"
    )
    c.commit()
    c.close()
    from titan_hcl.logic.verified_context_builder import QueryParser

    r = StoreRouter(data_dir=str(tmp_path))
    parsed = QueryParser().parse("what word did I learn")
    out1 = r.query_store("vocabulary", parsed, limit=5)
    out2 = r.query_store("vocabulary", parsed, limit=5)  # reuse pooled conn
    assert out1 and out2
    assert "raffinesse" in out1[0]["content"]


def test_build_uses_one_persistent_pool(tmp_path):
    v = VerifiedContextBuilder(data_dir=str(tmp_path))
    assert isinstance(v._query_pool, ThreadPoolExecutor)
    pool_id = id(v._query_pool)
    ctx = v.build(query="hello there", user_id="u")
    assert ctx.total_records == 0  # empty data dir → empty (valid) context
    v.build(query="again", user_id="u")
    assert id(v._query_pool) == pool_id  # SAME persistent pool across builds
