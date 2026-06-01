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
