"""ThoughtSidecar (RFP_synthesis_spine_reads_real_data Phase B / B1) — lock-free
tx_hash → promoted-thought content store. Writer (memory_worker) + R/O reader
(recall deref consumers)."""
from titan_hcl.synthesis.thought_sidecar import (
    ThoughtSidecar,
    ThoughtSidecarReader,
)


def test_put_get_roundtrip(tmp_path):
    d = str(tmp_path)
    w = ThoughtSidecar(d)
    w.put(tx_hash="abc123", node_id=42, user_prompt="I race karts",
          agent_response="Cool!", memory_type="declarative", fork="declarative",
          ts=100.0)
    r = ThoughtSidecarReader(d)
    got = r.get("abc123")
    assert got is not None
    assert got["user_prompt"] == "I race karts"
    assert got["agent_response"] == "Cool!"
    assert got["node_id"] == 42
    assert got["memory_type"] == "declarative"
    assert got["fork"] == "declarative"
    assert abs(got["ts"] - 100.0) < 1e-6
    w.close()
    r.close()


def test_get_missing_returns_none(tmp_path):
    w = ThoughtSidecar(str(tmp_path))
    r = ThoughtSidecarReader(str(tmp_path))
    assert r.get("nonexistent") is None
    assert r.get("") is None
    w.close()
    r.close()


def test_reader_before_any_write_is_none(tmp_path):
    # No sidecar file created yet → reader soft-fails to None (never crashes).
    r = ThoughtSidecarReader(str(tmp_path / "sub_not_created"))
    assert r.get("abc") is None
    r.close()


def test_put_idempotent_replace(tmp_path):
    d = str(tmp_path)
    w = ThoughtSidecar(d)
    w.put(tx_hash="h1", node_id=1, user_prompt="v1", agent_response="r1",
          memory_type="episodic", fork="episodic", ts=1.0)
    w.put(tx_hash="h1", node_id=1, user_prompt="v2", agent_response="r2",
          memory_type="episodic", fork="episodic", ts=2.0)
    r = ThoughtSidecarReader(d)
    got = r.get("h1")
    assert got["user_prompt"] == "v2"   # replaced, not duplicated
    assert got["agent_response"] == "r2"
    w.close()
    r.close()


def test_empty_tx_hash_put_is_noop(tmp_path):
    d = str(tmp_path)
    w = ThoughtSidecar(d)
    w.put(tx_hash="", node_id=1, user_prompt="x", agent_response="y",
          memory_type="episodic", fork="episodic")
    r = ThoughtSidecarReader(d)
    assert r.get("") is None
    w.close()
    r.close()


def test_iter_all(tmp_path):
    d = str(tmp_path)
    w = ThoughtSidecar(d)
    w.put(tx_hash="a" * 64, node_id=1, user_prompt="p1", agent_response="r1",
          memory_type="declarative", fork="declarative", ts=1.0)
    w.put(tx_hash="b" * 64, node_id=2, user_prompt="p2", agent_response="r2",
          memory_type="episodic", fork="episodic", ts=2.0)
    r = ThoughtSidecarReader(d)
    rows = r.iter_all(limit=10)
    assert len(rows) == 2
    assert {row["tx_hash"] for row in rows} == {"a" * 64, "b" * 64}
    # newest-first ordering (ts DESC)
    assert rows[0]["tx_hash"] == "b" * 64
    w.close()
    r.close()


def test_iter_all_empty_when_no_file(tmp_path):
    r = ThoughtSidecarReader(str(tmp_path / "absent"))
    assert r.iter_all() == []
    r.close()


def test_tx_content_deref_reads_sidecar(tmp_path):
    # Phase C end-to-end: a promoted thought in the sidecar derefs to its REAL
    # content (not the chain envelope) via TxContentDeref.
    from titan_hcl.synthesis.tx_index_builder import TxContentDeref
    d = str(tmp_path)
    w = ThoughtSidecar(d)
    w.put(tx_hash="f" * 64, node_id=7,
          user_prompt="I love spaghetti carbonara",
          agent_response="Great choice!", memory_type="declarative",
          fork="declarative", ts=1.0)
    w.close()
    deref = TxContentDeref(data_dir=d)   # no timechain index.db present → sidecar-only
    snip = deref.snippet("f" * 64)
    assert snip is not None
    assert "spaghetti carbonara" in snip
    # a hash absent from the sidecar (and no chain) → None, no crash
    assert deref.snippet("0" * 64) is None
    deref.close()
