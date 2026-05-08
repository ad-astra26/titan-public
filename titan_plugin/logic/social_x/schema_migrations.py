"""Phase 1 X-voice enrichment — schema migrations.

rFP §4.5 — additive, non-blocking, idempotent migrations:

* `vocabulary`            +grounded_at, +grounded_felt_summary
* `mention_tracking`      +reply_emotion, +reply_felt_summary, +reply_neuromods_json
* `Kuzu Person`           +last_felt_emotion, +last_felt_summary, +last_felt_neuromods_json
* `archetype_pool_scores` NEW table (adaptive scoring substrate, §4.7)
* `distilled_wisdom`      verify created_at exists (no-op if present)

All ALTER TABLE statements are wrapped in PRAGMA-aware idempotent helpers so
applying twice is a no-op. Existing rows degrade gracefully — selectors skip
rows with empty backfill columns until natural backfill catches up.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Iterable

logger = logging.getLogger(__name__)


# ── Generic SQLite helper ────────────────────────────────────────────

def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, decl: str
) -> bool:
    """Idempotent ADD COLUMN. Returns True if added, False if already present."""
    if not _table_exists(conn, table):
        logger.debug("[schema_migrations] table %s missing — skipping %s", table, column)
        return False
    if _column_exists(conn, table, column):
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    logger.info("[schema_migrations] %s ADD COLUMN %s %s", table, column, decl)
    return True


# ── social_x.db (mention_tracking + archetype_pool_scores) ───────────

def apply_social_x_migrations(db_path: str) -> dict:
    """Apply mention_tracking ADDs + archetype_pool_scores CREATE.

    Returns a summary dict for logging/test assertions.
    """
    out = {"db": db_path, "added": [], "created": []}
    conn = sqlite3.connect(db_path, timeout=5)
    try:
        # mention_tracking — felt-state-at-reply backfill columns
        for col, decl in (
            ("reply_emotion",         "TEXT DEFAULT ''"),
            ("reply_felt_summary",    "TEXT DEFAULT ''"),
            ("reply_neuromods_json",  "TEXT DEFAULT ''"),
        ):
            if _add_column_if_missing(conn, "mention_tracking", col, decl):
                out["added"].append(f"mention_tracking.{col}")

        # archetype_pool_scores — adaptive scoring substrate (§4.7)
        if not _table_exists(conn, "archetype_pool_scores"):
            conn.execute(
                """
                CREATE TABLE archetype_pool_scores (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  titan_id TEXT NOT NULL,
                  archetype TEXT NOT NULL,
                  pool TEXT NOT NULL,
                  score INTEGER NOT NULL,
                  source_id TEXT,
                  engagement_signals TEXT,
                  ts REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX idx_pool_scores_lookup "
                "ON archetype_pool_scores(titan_id, archetype, pool, ts)"
            )
            out["created"].append("archetype_pool_scores")
            logger.info("[schema_migrations] created archetype_pool_scores in %s", db_path)
        conn.commit()
    finally:
        conn.close()
    return out


# ── inner_memory.db (vocabulary) ─────────────────────────────────────

def apply_inner_memory_migrations(db_path: str) -> dict:
    """vocabulary +grounded_at, +grounded_felt_summary."""
    out = {"db": db_path, "added": []}
    conn = sqlite3.connect(db_path, timeout=5)
    try:
        for col, decl in (
            ("grounded_at",            "REAL DEFAULT 0"),
            ("grounded_felt_summary",  "TEXT DEFAULT ''"),
        ):
            if _add_column_if_missing(conn, "vocabulary", col, decl):
                out["added"].append(f"vocabulary.{col}")
        conn.commit()
    finally:
        conn.close()
    return out


def verify_distilled_wisdom(db_path: str) -> bool:
    """rFP §4.5: distilled_wisdom.created_at must exist. (Already does in
    experience_orchestrator.py:174 — this is a runtime sanity check.)"""
    conn = sqlite3.connect(db_path, timeout=5)
    try:
        if not _table_exists(conn, "distilled_wisdom"):
            return False
        return _column_exists(conn, "distilled_wisdom", "created_at")
    finally:
        conn.close()


# ── Kuzu Person ──────────────────────────────────────────────────────

_KUZU_PERSON_COLUMNS = (
    ("last_felt_emotion",       "STRING DEFAULT ''"),
    ("last_felt_summary",       "STRING DEFAULT ''"),
    ("last_felt_neuromods_json","STRING DEFAULT ''"),
)


def _kuzu_person_has_column(conn, column: str) -> bool:
    """CALL TABLE_INFO returns columns for a Kuzu node table."""
    try:
        qr = conn.execute("CALL TABLE_INFO('Person') RETURN *")
        while qr.has_next():
            row = qr.get_next()
            # Kuzu TABLE_INFO row shape: [property_id, name, type, default, primary_key]
            if len(row) >= 2 and row[1] == column:
                return True
    except Exception as e:
        logger.debug("[schema_migrations] kuzu TABLE_INFO probe failed: %s", e)
    return False


def apply_kuzu_person_migrations(graph) -> dict:
    """Add last_felt_* columns to Kuzu Person node table.

    `graph` is a TitanKnowledgeGraph instance (or anything exposing `_conn`
    that runs Cypher). All ADDs are idempotent.
    """
    out = {"added": []}
    conn = getattr(graph, "_conn", None)
    if conn is None:
        logger.warning("[schema_migrations] kuzu graph has no _conn — skipping Person migration")
        return out
    for col, decl in _KUZU_PERSON_COLUMNS:
        if _kuzu_person_has_column(conn, col):
            continue
        try:
            conn.execute(f"ALTER TABLE Person ADD {col} {decl}")
            out["added"].append(f"Person.{col}")
            logger.info("[schema_migrations] Person ADD %s %s", col, decl)
        except Exception as e:
            # Common: the table doesn't exist yet (fresh install) — _init_schema
            # will create it WITHOUT the new columns; we run again on next boot.
            logger.warning("[schema_migrations] Person ADD %s failed: %s", col, e)
    return out


# ── Orchestrator ─────────────────────────────────────────────────────

def apply_all(
    *,
    social_x_db: str = "./data/social_x.db",
    inner_memory_db: str = "./data/inner_memory.db",
    distilled_wisdom_db: str | None = None,
    kuzu_graph=None,
) -> dict:
    """Apply every Phase 1 schema migration. Idempotent.

    Args:
        social_x_db: path to gateway's actions/mention_tracking SQLite DB
        inner_memory_db: path to inner_memory's vocabulary SQLite DB
        distilled_wisdom_db: optional path to verify created_at exists
        kuzu_graph: optional TitanKnowledgeGraph instance for Person ALTERs
    """
    summary: dict = {}
    summary["social_x"]    = apply_social_x_migrations(social_x_db)
    summary["inner_memory"] = apply_inner_memory_migrations(inner_memory_db)
    if distilled_wisdom_db:
        summary["distilled_wisdom_ok"] = verify_distilled_wisdom(distilled_wisdom_db)
    if kuzu_graph is not None:
        summary["kuzu_person"] = apply_kuzu_person_migrations(kuzu_graph)
    return summary


__all__ = (
    "apply_all",
    "apply_social_x_migrations",
    "apply_inner_memory_migrations",
    "apply_kuzu_person_migrations",
    "verify_distilled_wisdom",
)
