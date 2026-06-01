#!/usr/bin/env python3
"""Repair the corrupted `actr_buffers` (and defensively `procedural_skills`) ART
index in a synthesis.duckdb (2026-06-01).

DuckDB's `INSERT OR REPLACE` on a table with secondary indexes is DELETE+INSERT,
which can corrupt the ART index — a later commit/revert then throws a FATAL
"duplicate key in PRIMARY_actr_buffers" that ABORTS the whole synthesis_worker
(observed crash-loop on the devnet soak DBs). The write path is fixed to
ON CONFLICT DO UPDATE (buffer_store.py / skill_store.py), but an already-corrupted
on-disk index keeps crashing until rebuilt. This script rebuilds the affected
tables: read rows → dedup by PK (keep latest) → drop+recreate table+indexes →
reinsert. Working-memory buffers are transient (repopulate per chat), so even a
total loss is low-impact; we preserve via dedup anyway.

MUST run with the synthesis_worker STOPPED (it holds the DB open + is crashing).

Usage:  python scripts/repair_actr_buffers_corruption.py /path/to/synthesis.duckdb
"""
import sys

import duckdb

BUFFER_NAMES = ("goal", "retrieval", "imaginal", "manual")


def _rebuild_actr_buffers(con) -> int:
    rows = con.execute(
        "SELECT chat_id, buffer_name, content, concept_ids, embedding_hash, "
        "updated_at FROM actr_buffers"
    ).fetchall()
    # Dedup by (chat_id, buffer_name), keep the newest updated_at.
    best: dict[tuple, tuple] = {}
    for r in rows:
        key = (r[0], r[1])
        if key not in best or (r[5] or 0) >= (best[key][5] or 0):
            best[key] = r
    con.execute("DROP TABLE IF EXISTS actr_buffers")
    con.execute(
        "CREATE TABLE actr_buffers ("
        "  chat_id TEXT NOT NULL, buffer_name TEXT NOT NULL, content TEXT,"
        "  concept_ids TEXT, embedding_hash TEXT, updated_at DOUBLE NOT NULL,"
        "  PRIMARY KEY (chat_id, buffer_name))"
    )
    con.execute("CREATE INDEX idx_actr_buffers_chat ON actr_buffers(chat_id)")
    con.execute(
        "CREATE INDEX idx_actr_buffers_updated ON actr_buffers(updated_at DESC)")
    for r in best.values():
        con.execute(
            "INSERT INTO actr_buffers (chat_id, buffer_name, content, "
            "concept_ids, embedding_hash, updated_at) VALUES (?,?,?,?,?,?)",
            list(r),
        )
    return len(best)


def _rebuild_procedural_skills(con) -> int:
    # Only rebuild if the table exists + has rows (defensive — same index class).
    try:
        cols = [c[0] for c in con.execute(
            "SELECT * FROM procedural_skills LIMIT 0").description]
    except Exception:
        return -1  # table absent
    rows = con.execute(f"SELECT {', '.join(cols)} FROM procedural_skills").fetchall()
    if not rows:
        return 0
    best: dict[str, tuple] = {}
    sid_i = cols.index("skill_id")
    for r in rows:
        best[r[sid_i]] = r  # last wins; skill_id is PK
    con.execute("DROP TABLE IF EXISTS procedural_skills")
    con.execute(
        "CREATE TABLE procedural_skills ("
        "  skill_id TEXT PRIMARY KEY, name TEXT, nl_description TEXT,"
        "  embedding_id INTEGER, executable_spec TEXT, preconditions TEXT,"
        "  postconditions TEXT, compiled_from TEXT, success_count INTEGER,"
        "  failure_count INTEGER, last_used DOUBLE, created_at DOUBLE,"
        "  utility_score DOUBLE, verified_at DOUBLE)"
    )
    con.execute("CREATE INDEX idx_procedural_skills_utility "
                "ON procedural_skills(utility_score DESC)")
    con.execute("CREATE INDEX idx_procedural_skills_last_used "
                "ON procedural_skills(last_used DESC)")
    ph = ", ".join("?" for _ in cols)
    for r in best.values():
        con.execute(
            f"INSERT INTO procedural_skills ({', '.join(cols)}) VALUES ({ph})",
            list(r))
    return len(best)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: repair_actr_buffers_corruption.py <synthesis.duckdb>")
        return 2
    path = sys.argv[1]
    con = duckdb.connect(path)
    try:
        con.execute("BEGIN")
        n_buf = _rebuild_actr_buffers(con)
        n_skill = _rebuild_procedural_skills(con)
        con.execute("COMMIT")
        con.execute("CHECKPOINT")
    except Exception as e:  # noqa: BLE001
        con.execute("ROLLBACK")
        print(f"REPAIR FAILED: {e}")
        return 1
    finally:
        con.close()
    print(f"REPAIR OK — actr_buffers rebuilt ({n_buf} rows), "
          f"procedural_skills rebuilt ({n_skill} rows; -1=absent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
