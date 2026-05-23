"""Phase 1 schema tests — memory_type column + activation_state table +
backfill correctness. D-SPEC-123 (SPEC v1.56.0 §25).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest

from titan_hcl.core.direct_memory import TitanDuckDB


@pytest.fixture
def fresh_db(tmp_path):
    """A brand-new titan_memory.duckdb opened through TitanDuckDB (so the
    schema lands exactly as production would init it)."""
    db_path = tmp_path / "titan_memory.duckdb"
    db = TitanDuckDB(str(db_path))
    yield db, str(db_path)
    db._conn.close()


# ─────────────────────────────────────────────────────────────────────────
# memory_type column
# ─────────────────────────────────────────────────────────────────────────

def test_memory_type_column_present(fresh_db):
    db, _ = fresh_db
    cols = db._conn.execute(
        "PRAGMA table_info('memory_nodes')"
    ).fetchall()
    col_names = [c[1] for c in cols]
    assert "memory_type" in col_names, (
        f"memory_type column missing from memory_nodes; have: {col_names}")


def test_memory_type_default_episodic(fresh_db):
    db, _ = fresh_db
    db._conn.execute(
        "INSERT INTO memory_nodes (id) VALUES (1)"
    )
    mt = db._conn.execute(
        "SELECT memory_type FROM memory_nodes WHERE id = 1"
    ).fetchone()[0]
    assert mt == "episodic"


def test_alter_table_idempotent_on_reopen(tmp_path):
    """Calling _init_schema again on the same DB MUST NOT raise (ADD COLUMN
    IF NOT EXISTS is silent-skip)."""
    db_path = tmp_path / "titan_memory.duckdb"
    db1 = TitanDuckDB(str(db_path))
    db1._conn.close()
    # Reopen — _init_schema runs again. Must be a no-op.
    db2 = TitanDuckDB(str(db_path))
    cols = db2._conn.execute("PRAGMA table_info('memory_nodes')").fetchall()
    assert "memory_type" in [c[1] for c in cols]
    db2._conn.close()


# ─────────────────────────────────────────────────────────────────────────
# activation_state table (sole-writer G21 / INV-Syn-3 surface — schema only)
# ─────────────────────────────────────────────────────────────────────────

def test_activation_state_schema_present(fresh_db):
    db, _ = fresh_db
    cols = db._conn.execute(
        "PRAGMA table_info('activation_state')"
    ).fetchall()
    col_names = [c[1] for c in cols]
    expected = {
        "item_id", "last_access", "access_log", "access_count",
        "first_access", "base_level", "last_recompute",
    }
    assert expected.issubset(set(col_names)), (
        f"missing: {expected - set(col_names)}; have: {col_names}")


def test_activation_state_primary_key_item_id(fresh_db):
    db, _ = fresh_db
    cols = db._conn.execute(
        "PRAGMA table_info('activation_state')"
    ).fetchall()
    pk_cols = [c[1] for c in cols if c[5] != 0]   # column 5 = 'pk' index
    assert pk_cols == ["item_id"]


def test_activation_state_insert_roundtrip(fresh_db):
    db, _ = fresh_db
    import time
    now = time.time()
    db._conn.execute(
        "INSERT INTO activation_state "
        "(item_id, last_access, access_count, first_access, base_level, "
        " last_recompute) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("kuzu:NODE_42", now, 1, now, 0.5, now),
    )
    row = db._conn.execute(
        "SELECT item_id, base_level FROM activation_state "
        "WHERE item_id = ?", ("kuzu:NODE_42",)
    ).fetchone()
    assert row[0] == "kuzu:NODE_42"
    assert row[1] == pytest.approx(0.5)


def test_activation_state_idempotent_on_reopen(tmp_path):
    db_path = tmp_path / "titan_memory.duckdb"
    db1 = TitanDuckDB(str(db_path))
    db1._conn.execute(
        "INSERT INTO activation_state (item_id, base_level) "
        "VALUES (?, ?)", ("tc:abc", 0.7))
    db1._conn.close()
    # Reopen — schema CREATE IF NOT EXISTS is a no-op.
    db2 = TitanDuckDB(str(db_path))
    n = db2._conn.execute(
        "SELECT COUNT(*) FROM activation_state"
    ).fetchone()[0]
    assert n == 1
    db2._conn.close()


# ─────────────────────────────────────────────────────────────────────────
# Backfill script
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def db_with_mixed_source_ids(tmp_path):
    db_path = tmp_path / "titan_memory.duckdb"
    db = TitanDuckDB(str(db_path))
    # Pre-Phase-1 rows where memory_type is NULL (simulate legacy data by
    # explicitly nulling it).
    db._conn.execute(
        "INSERT INTO memory_nodes (id, source_id, memory_type) VALUES "
        "(1, 'identity_personality', NULL), "
        "(2, 'identity_user_profile', NULL), "
        "(3, 'chat_session_abc', NULL), "
        "(4, NULL, NULL), "
        "(5, 'identity_personality', 'meta')"    # already-set NON-default
    )
    db._conn.close()
    yield str(db_path)


def _run_backfill(db_path: str, apply: bool) -> tuple[int, str]:
    """Returns (returncode, combined stdout/stderr)."""
    cmd = [sys.executable, "scripts/migrate_memory_type_backfill.py",
           "--db-path", db_path]
    if apply:
        cmd.append("--apply")
    else:
        cmd.append("--dry-run")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=repo_root, timeout=30)
    return proc.returncode, (proc.stdout + proc.stderr)


def test_backfill_dry_run_writes_nothing(db_with_mixed_source_ids):
    import duckdb
    rc, output = _run_backfill(db_with_mixed_source_ids, apply=False)
    assert rc == 0, f"dry-run exited {rc}: {output}"
    assert "DRY-RUN" in output
    # Verify rows untouched.
    con = duckdb.connect(db_with_mixed_source_ids)
    rows = con.execute(
        "SELECT id, memory_type FROM memory_nodes ORDER BY id"
    ).fetchall()
    con.close()
    types = {r[0]: r[1] for r in rows}
    assert types[1] is None
    assert types[2] is None
    assert types[3] is None
    assert types[4] is None
    assert types[5] == "meta"


def test_backfill_apply_correctness(db_with_mixed_source_ids):
    import duckdb
    rc, output = _run_backfill(db_with_mixed_source_ids, apply=True)
    assert rc == 0, f"apply exited {rc}: {output}"
    con = duckdb.connect(db_with_mixed_source_ids)
    rows = con.execute(
        "SELECT id, memory_type FROM memory_nodes ORDER BY id"
    ).fetchall()
    con.close()
    types = {r[0]: r[1] for r in rows}
    # identity_* → declarative (rows 1, 2)
    assert types[1] == "declarative"
    assert types[2] == "declarative"
    # other source_id → episodic (row 3)
    assert types[3] == "episodic"
    # NULL source_id → episodic (row 4)
    assert types[4] == "episodic"
    # Row 5 was identity_personality (would normally → declarative) but
    # already had memory_type='meta' — UPDATE filter excludes it because
    # the predicate "memory_type <> 'declarative'" matched, so it WILL
    # change to 'declarative'. This is correct behavior: the backfill
    # rewrites identity_* rows to declarative regardless of prior value.
    # If we wanted to preserve future phases' explicit settings, we'd
    # filter on memory_type IS NULL only. Current behavior: source_id
    # provenance dominates. This test pins that behavior.
    assert types[5] == "declarative", (
        "backfill rewrites identity_* rows to declarative even if they "
        "had a prior memory_type — pin this behavior")


def test_backfill_apply_is_idempotent(db_with_mixed_source_ids):
    # First apply.
    rc1, _ = _run_backfill(db_with_mixed_source_ids, apply=True)
    assert rc1 == 0
    # Second apply — should report "nothing to do".
    rc2, output2 = _run_backfill(db_with_mixed_source_ids, apply=True)
    assert rc2 == 0
    assert "nothing to do" in output2
