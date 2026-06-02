"""Regression test for the procedural tool_call index query (2026-06-01).

Both `block_index` and `fork_registry` carry a `fork_id` column. The reader's
SELECT listed `fork_id` unqualified → sqlite3 "ambiguous column name", which the
reader swallowed to []. Result: oracle coverage read 0 for the entire Phase-6
lifetime even when tool_call TXs were on chain. This pins the table-qualified
fix at the SQL level (no chain-file fixture needed)."""
import sqlite3

from titan_hcl.synthesis.procedural_tx_reader import (
    _TOOL_CALL_INDEX_SQL,
    _scored_by_from_tags,
    _tool_id_from_tags,
)


def _build_index(conn):
    # Minimal faithful schema: both tables carry fork_id (the ambiguity source).
    conn.execute(
        "CREATE TABLE fork_registry (fork_id INTEGER, fork_name TEXT, "
        "fork_type TEXT)")
    conn.execute(
        "CREATE TABLE block_index (block_hash TEXT, fork_id INTEGER, "
        "block_height INTEGER, file_offset INTEGER, thought_type TEXT, "
        "tags TEXT, timestamp REAL)")
    conn.execute("INSERT INTO fork_registry VALUES (7, 'procedural', 'proc')")
    conn.execute("INSERT INTO fork_registry VALUES (1, 'conversation', 'conv')")
    # one tool_call on the procedural fork, one non-tool_call, one wrong-fork
    conn.execute("INSERT INTO block_index VALUES "
                 "('h_tc', 7, 10, 0, 'tool_call', '[]', 1000.0)")
    conn.execute("INSERT INTO block_index VALUES "
                 "('h_other', 7, 11, 1, 'concept', '[]', 1001.0)")
    conn.execute("INSERT INTO block_index VALUES "
                 "('h_conv', 1, 12, 2, 'tool_call', '[]', 1002.0)")
    conn.commit()


def test_index_sql_no_ambiguity_and_filters():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _build_index(conn)
    # Must NOT raise "ambiguous column name: fork_id".
    rows = list(conn.execute(_TOOL_CALL_INDEX_SQL, ["procedural", 0.0, 100]))
    # Only the procedural-fork tool_call row.
    assert len(rows) == 1
    assert rows[0]["block_hash"] == "h_tc"
    assert rows[0]["fork_id"] == 7
    assert rows[0]["thought_type"] == "tool_call"


def test_index_sql_respects_since_ts():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _build_index(conn)
    # since_ts above the row's timestamp → excluded.
    rows = list(conn.execute(_TOOL_CALL_INDEX_SQL, ["procedural", 5000.0, 100]))
    assert rows == []


def test_unqualified_fork_id_would_be_ambiguous():
    """Document the original bug: the unqualified form raises."""
    conn = sqlite3.connect(":memory:")
    _build_index(conn)
    bad_sql = _TOOL_CALL_INDEX_SQL.replace("bi.fork_id", "fork_id", 1)
    with __import__("pytest").raises(sqlite3.OperationalError):
        conn.execute(bad_sql, ["procedural", 0.0, 100]).fetchall()


# ── v2 tx_summaries tag parsing (the reader extracts scored_by/tool_id from
#    the batch-block tx_summaries[].tags, since v2 stores no per-TX content) ──
def test_scored_by_from_tags():
    assert _scored_by_from_tags(["tool_call", "tool:coding_sandbox",
                                 "scored_by:oracle"]) == "oracle"
    assert _scored_by_from_tags(["scored_by:llm"]) == "llm"
    # scored_by:none → None (unscored, not the literal string)
    assert _scored_by_from_tags(["scored_by:none"]) is None
    assert _scored_by_from_tags(["tool:coding_sandbox"]) is None
    assert _scored_by_from_tags([]) is None


def test_tool_id_from_tags():
    assert _tool_id_from_tags(["tool_call", "tool:coding_sandbox",
                               "scored_by:none"]) == "coding_sandbox"
    assert _tool_id_from_tags(["tool:x_research"]) == "x_research"
    assert _tool_id_from_tags(["tool_call"]) == ""


# ── G1 (AUDIT §5.3): per-call tool_call_score overlay ──────────────────────
# The LLM judge anchors a tool_call_score TX on the PROCEDURAL fork per scored
# call. Its tags survive v2 batch-sealing (content does not), so the reader
# overlays scored_by onto the matching unscored tool_call record — letting the
# miner + coverage see llm-scored calls (the meta scored_by_patch TX can't be
# read back because v2 sealing drops its per-entry content).

def _build_overlay_index(idx_path, *, parent_scored="none"):
    """index.db with one tool_call block (offset 0) + one tool_call_score
    block (offset 1), both on the procedural fork."""
    conn = sqlite3.connect(str(idx_path))
    conn.execute("CREATE TABLE fork_registry (fork_id INTEGER, fork_name TEXT, "
                 "fork_type TEXT)")
    conn.execute("CREATE TABLE block_index (block_hash TEXT, fork_id INTEGER, "
                 "block_height INTEGER, file_offset INTEGER, thought_type TEXT, "
                 "tags TEXT, timestamp REAL)")
    conn.execute("INSERT INTO fork_registry VALUES (7, 'procedural', 'proc')")
    conn.execute("INSERT INTO block_index VALUES "
                 "('h_tc', 7, 10, 0, 'tool_call', '[]', 1000.0)")
    conn.execute("INSERT INTO block_index VALUES "
                 "('h_score', 7, 11, 1, 'tool_call_score', '[]', 1001.0)")
    conn.commit()
    conn.close()


def _fake_read_factory(parent, *, parent_scored="none", score_verdict="llm"):
    def fake_read(data_dir, fork_id, file_offset):
        if file_offset == 0:  # the tool_call block
            return {"v2": True, "tx_count": 1, "tx_summaries": [
                {"hash": parent, "type": "tool_call",
                 "tags": ["tool_call", "tool:coding_sandbox",
                          f"scored_by:{parent_scored}"]},
            ]}
        # the tool_call_score block (carries verdict + FULL parent tx in tags)
        return {"v2": True, "tx_count": 1, "tx_summaries": [
            {"hash": "c" * 64, "type": "tool_call_score",
             "tags": ["tool_call_score", f"scored_by:{score_verdict}",
                      f"parent:{parent}"]},
        ]}
    return fake_read


def test_parent_from_tags():
    from titan_hcl.synthesis.procedural_tx_reader import _parent_from_tags
    full = "a" * 64
    assert _parent_from_tags(["tool_call_score", "scored_by:llm",
                              f"parent:{full}"]) == full
    # full hash preserved (not truncated) — the exact overlay join key
    assert len(_parent_from_tags([f"parent:{full}"])) == 64
    assert _parent_from_tags(["scored_by:llm"]) == ""
    assert _parent_from_tags([]) == ""


def test_index_sql_includes_tool_call_score():
    """G1 SQL filter must fetch tool_call_score blocks too, but still exclude
    unrelated thought_types (e.g. concept)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE fork_registry (fork_id INTEGER, fork_name TEXT, "
                 "fork_type TEXT)")
    conn.execute("CREATE TABLE block_index (block_hash TEXT, fork_id INTEGER, "
                 "block_height INTEGER, file_offset INTEGER, thought_type TEXT, "
                 "tags TEXT, timestamp REAL)")
    conn.execute("INSERT INTO fork_registry VALUES (7, 'procedural', 'proc')")
    conn.execute("INSERT INTO block_index VALUES "
                 "('h_tc', 7, 10, 0, 'tool_call', '[]', 1000.0)")
    conn.execute("INSERT INTO block_index VALUES "
                 "('h_score', 7, 11, 1, 'tool_call_score', '[]', 1001.0)")
    conn.execute("INSERT INTO block_index VALUES "
                 "('h_concept', 7, 12, 2, 'concept', '[]', 1002.0)")
    conn.commit()
    rows = list(conn.execute(_TOOL_CALL_INDEX_SQL, ["procedural", 0.0, 100]))
    types = sorted(r["thought_type"] for r in rows)
    assert types == ["tool_call", "tool_call_score"]  # concept excluded


def test_overlay_fills_scored_by_from_tool_call_score(tmp_path, monkeypatch):
    """End-to-end: a tool_call_score TX overlays its verdict onto the matching
    unscored tool_call record; the score TX itself is not emitted as a record."""
    import titan_hcl.synthesis.chain_reader as chain_reader
    from titan_hcl.synthesis.procedural_tx_reader import (
        default_procedural_tool_call_reader,
    )
    parent = "b" * 64
    idx = tmp_path / "index.db"
    _build_overlay_index(idx)
    monkeypatch.setattr(chain_reader, "read_block_content_at",
                        _fake_read_factory(parent))
    out = default_procedural_tool_call_reader(
        0.0, 100, index_db_path=str(idx), chain_dir=str(tmp_path / "timechain"))
    assert len(out) == 1  # the tool_call_score summary is NOT a record
    rec = out[0]
    assert rec["tx_hash"] == parent
    assert rec["scored_by"] == "llm"             # overlaid
    assert rec["content"]["scored_by"] == "llm"


def test_overlay_disabled_leaves_scored_by_none(tmp_path, monkeypatch):
    import titan_hcl.synthesis.chain_reader as chain_reader
    from titan_hcl.synthesis.procedural_tx_reader import (
        default_procedural_tool_call_reader,
    )
    parent = "d" * 64
    idx = tmp_path / "index.db"
    _build_overlay_index(idx)
    monkeypatch.setattr(chain_reader, "read_block_content_at",
                        _fake_read_factory(parent))
    out = default_procedural_tool_call_reader(
        0.0, 100, index_db_path=str(idx),
        chain_dir=str(tmp_path / "timechain"), overlay=False)
    assert len(out) == 1
    assert out[0]["scored_by"] is None           # overlay suppressed


def test_overlay_does_not_overwrite_write_time_oracle_score(tmp_path, monkeypatch):
    """A call already scored_by:oracle at write-time must NOT be overwritten by
    a later llm overlay — oracle write-time wins, no double-count."""
    import titan_hcl.synthesis.chain_reader as chain_reader
    from titan_hcl.synthesis.procedural_tx_reader import (
        default_procedural_tool_call_reader,
    )
    parent = "f" * 64
    idx = tmp_path / "index.db"
    _build_overlay_index(idx)
    monkeypatch.setattr(
        chain_reader, "read_block_content_at",
        _fake_read_factory(parent, parent_scored="oracle", score_verdict="llm"))
    out = default_procedural_tool_call_reader(
        0.0, 100, index_db_path=str(idx), chain_dir=str(tmp_path / "timechain"))
    assert len(out) == 1
    assert out[0]["scored_by"] == "oracle"       # write-time score preserved
