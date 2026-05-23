"""One-time idempotent backfill of memory_nodes.memory_type from source_id.

Synthesis Engine Phase 1 (D-SPEC-123, SPEC v1.56.0 §25). Per arch §6 +
rFP_outer_memory_enhancement.md §17.3, every memory_nodes row carries a
memory_type ∈ {'declarative','procedural','episodic','meta'}. The DDL in
titan_hcl/core/direct_memory.py:_init_schema defaults new rows to 'episodic'
(overwhelming majority is chat-turn content). This script rewrites existing
rows from their source_id provenance.

Mapping rules (derived from the two sole source_id assigner sites in
titan_hcl/core/memory.py:
  - line 615: source_id = identity_node_id      → chat conversation =
    'episodic'
  - line 785: source_id = f"identity_{source}"  → identity injection
    (personality / user_profile / ...) = 'declarative'
Any other historical pattern → 'episodic' (safe default — chat is the
overwhelming class).

DISCIPLINE: dry-run first (--dry-run prints the per-bucket count + writes
nothing). Idempotent: re-running the live pass on already-backfilled rows
is a no-op (every UPDATE filters on the current value).

NEVER deletes / never mutates existing memory_type values set by future
phases (procedural / meta). Per directive_memory_preservation.md.

Usage:
    python scripts/migrate_memory_type_backfill.py --dry-run
    python scripts/migrate_memory_type_backfill.py --apply

If --apply is not passed, defaults to dry-run.
"""
from __future__ import annotations

import argparse
import os
import sys

# Default DB path — same resolution as TitanDuckDB(db_path=...).
# Users can override via --db-path for non-default installs.
DEFAULT_DB_PATH = os.path.join("data", "titan_memory.duckdb")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to titan_memory.duckdb (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes. Omit for dry-run (the default).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run (no writes). Same as omitting --apply.",
    )
    args = parser.parse_args()

    apply_changes = args.apply and not args.dry_run

    import duckdb

    if not os.path.exists(args.db_path):
        print(f"FAIL: {args.db_path} does not exist", file=sys.stderr)
        return 2

    # Use a normal (read-write) connection — we may need to ALTER TABLE if
    # the column hasn't been created yet (fresh DB pre-Phase-1 schema).
    con = duckdb.connect(args.db_path)

    # Ensure column exists (idempotent — mirrors _init_schema's ALTER).
    # If a fully-fresh install ran TitanDuckDB._init_schema already, this is
    # a no-op; if someone runs the backfill against a pre-D-SPEC-123 DB
    # without booting the kernel first, it self-heals the schema.
    con.execute(
        "ALTER TABLE memory_nodes "
        "ADD COLUMN IF NOT EXISTS memory_type TEXT DEFAULT 'episodic'"
    )

    # Snapshot the current distribution.
    print(f"== Backfill memory_nodes.memory_type from source_id "
          f"(db={args.db_path}, apply={apply_changes}) ==")
    print()

    total = con.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0]
    print(f"Total rows: {total:,}")

    rows = con.execute(
        "SELECT COALESCE(memory_type, 'NULL') AS mt, COUNT(*) AS n "
        "FROM memory_nodes GROUP BY mt ORDER BY n DESC"
    ).fetchall()
    print("Current memory_type distribution:")
    for mt, n in rows:
        print(f"  {mt}: {n:,}")
    print()

    # Identify the backfill buckets.
    #   1. identity_*  -> 'declarative'  (identity injections)
    #   2. everything else with a non-null source_id -> 'episodic' (chat)
    # We only flip rows whose memory_type does NOT already match the target,
    # so re-runs are no-ops.

    identity_to_declarative = con.execute(
        "SELECT COUNT(*) FROM memory_nodes "
        "WHERE source_id LIKE 'identity_%' "
        "AND (memory_type IS NULL OR memory_type <> 'declarative')"
    ).fetchone()[0]
    chat_to_episodic = con.execute(
        "SELECT COUNT(*) FROM memory_nodes "
        "WHERE source_id IS NOT NULL "
        "AND source_id NOT LIKE 'identity_%' "
        "AND (memory_type IS NULL OR memory_type <> 'episodic')"
    ).fetchone()[0]
    null_source_to_episodic = con.execute(
        "SELECT COUNT(*) FROM memory_nodes "
        "WHERE source_id IS NULL "
        "AND (memory_type IS NULL OR memory_type <> 'episodic')"
    ).fetchone()[0]

    print("Backfill plan (rows that WILL change):")
    print(f"  identity_* source_id  -> declarative : {identity_to_declarative:,}")
    print(f"  other source_id       -> episodic    : {chat_to_episodic:,}")
    print(f"  NULL source_id        -> episodic    : {null_source_to_episodic:,}")
    print()

    total_to_change = (identity_to_declarative + chat_to_episodic +
                       null_source_to_episodic)
    if total_to_change == 0:
        print("OK: nothing to do (already backfilled).")
        return 0

    if not apply_changes:
        print(f"DRY-RUN: would write {total_to_change:,} rows. Re-run with "
              "--apply to commit.")
        return 0

    print(f"APPLYING {total_to_change:,} updates...")

    con.execute("BEGIN")
    try:
        con.execute(
            "UPDATE memory_nodes SET memory_type = 'declarative' "
            "WHERE source_id LIKE 'identity_%' "
            "AND (memory_type IS NULL OR memory_type <> 'declarative')"
        )
        con.execute(
            "UPDATE memory_nodes SET memory_type = 'episodic' "
            "WHERE source_id IS NOT NULL "
            "AND source_id NOT LIKE 'identity_%' "
            "AND (memory_type IS NULL OR memory_type <> 'episodic')"
        )
        con.execute(
            "UPDATE memory_nodes SET memory_type = 'episodic' "
            "WHERE source_id IS NULL "
            "AND (memory_type IS NULL OR memory_type <> 'episodic')"
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    # Re-print distribution to confirm.
    print()
    print("Post-backfill memory_type distribution:")
    for mt, n in con.execute(
        "SELECT COALESCE(memory_type, 'NULL') AS mt, COUNT(*) AS n "
        "FROM memory_nodes GROUP BY mt ORDER BY n DESC"
    ).fetchall():
        print(f"  {mt}: {n:,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
