#!/usr/bin/env python3
"""
One-time migration: agno_sessions.db → memory_nodes.db

Backfills Titan's memory_nodes database with conversation history from
77 sessions (2128 turns) accumulated during V2/V3 runtime. This enriches
the frontend memory topology and gives Titan access to his full history.

ADDITIVE ONLY — Memory Preservation Protocol. Never deletes existing nodes.

Usage:
    source test_env/bin/activate
    python scripts/backfill_memory_nodes.py [--dry-run] [--limit N]
"""

import json
import logging
import os
import sqlite3
import sys
import time

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

SESSIONS_DB = os.path.join(DATA_DIR, "agno_sessions.db")
MEMORY_DB = os.path.join(DATA_DIR, "memory_nodes.db")

# Minimum content length to be worth persisting (skip empty/trivial turns)
MIN_RESPONSE_LEN = 50
# Skip system prompt prefixes
SYSTEM_PREFIXES = ("You are Titan", "SYSTEM:", "[SYSTEM]", "<<SYS>>")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backfill")


def ensure_identity_node(mem_conn: sqlite3.Connection, user_id: str, created_at: float) -> str:
    """
    Ensure an identity_node exists for this user. Returns the identity node ID.
    Additive only — never updates existing nodes.
    """
    identity_id = f"identity_{user_id}"
    existing = mem_conn.execute(
        "SELECT id FROM identity_nodes WHERE id = ?", (identity_id,)
    ).fetchone()
    if existing:
        return identity_id

    mem_conn.execute(
        "INSERT INTO identity_nodes (id, identifier, created_at) VALUES (?, ?, ?)",
        (identity_id, user_id, created_at),
    )
    return identity_id


def extract_turns(runs_raw: str) -> list[dict]:
    """
    Parse a session's runs field into structured conversation turns.

    Each turn has: user_prompt, agent_response, created_at, model
    """
    try:
        runs = json.loads(runs_raw)
        if isinstance(runs, str):
            runs = json.loads(runs)  # double-encoded
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(runs, list):
        return []

    turns = []
    for run in runs:
        if not isinstance(run, dict):
            continue

        # Extract Titan's response
        response = run.get("content", "")
        if not response or not isinstance(response, str):
            continue
        if len(response) < MIN_RESPONSE_LEN:
            continue

        # Skip system prompt outputs
        if any(response.startswith(prefix) for prefix in SYSTEM_PREFIXES):
            continue

        # Extract user input
        input_data = run.get("input", {})
        if isinstance(input_data, dict):
            user_prompt = input_data.get("input_content", "")
        elif isinstance(input_data, str):
            user_prompt = input_data
        else:
            user_prompt = ""

        # Skip if this is a function call artifact, not a real response
        if response.startswith("<function=") or response.startswith("```tool"):
            continue

        created_at = run.get("created_at", 0)
        if isinstance(created_at, (int, float)) and created_at > 0:
            # Already epoch timestamp
            pass
        else:
            created_at = time.time()

        turns.append({
            "user_prompt": user_prompt[:2000] if user_prompt else "",  # cap prompt length
            "agent_response": response[:5000],  # cap response length
            "created_at": float(created_at),
            "model": run.get("model", "unknown"),
        })

    return turns


def get_existing_node_count(mem_conn: sqlite3.Connection) -> int:
    """Get current max ID to avoid conflicts."""
    row = mem_conn.execute("SELECT MAX(id) FROM memory_nodes").fetchone()
    return row[0] if row[0] else 0


def backfill(dry_run: bool = False, limit: int = 0) -> dict:
    """
    Main migration: read agno_sessions → write memory_nodes.

    Returns stats dict.
    """
    if not os.path.exists(SESSIONS_DB):
        logger.error("Source database not found: %s", SESSIONS_DB)
        return {"error": "sessions db not found"}

    if not os.path.exists(MEMORY_DB):
        logger.error("Target database not found: %s", MEMORY_DB)
        return {"error": "memory db not found"}

    # Connect to both databases
    sess_conn = sqlite3.connect(SESSIONS_DB)
    mem_conn = sqlite3.connect(MEMORY_DB)

    # Stats
    stats = {
        "sessions_processed": 0,
        "turns_extracted": 0,
        "nodes_created": 0,
        "identities_created": 0,
        "skipped_short": 0,
        "skipped_system": 0,
        "existing_nodes_before": get_existing_node_count(mem_conn),
    }

    # Track existing identity nodes
    existing_identities = set(
        row[0] for row in mem_conn.execute("SELECT id FROM identity_nodes").fetchall()
    )

    # Read all sessions
    query = "SELECT session_id, user_id, runs, created_at FROM titan_sessions ORDER BY created_at ASC"
    sessions = sess_conn.execute(query).fetchall()
    if limit > 0:
        sessions = sessions[:limit]

    logger.info(
        "Starting migration: %d sessions, %d existing nodes, dry_run=%s",
        len(sessions), stats["existing_nodes_before"], dry_run,
    )

    next_id = stats["existing_nodes_before"] + 1

    for session_id, user_id, runs_raw, session_created_at in sessions:
        if not user_id:
            user_id = "anonymous"

        # Clean user_id (remove @ prefix for identity node naming)
        clean_uid = user_id.lstrip("@") if user_id.startswith("@") else user_id

        turns = extract_turns(runs_raw)
        stats["sessions_processed"] += 1

        if not turns:
            continue

        # Ensure identity node exists
        identity_id = f"identity_{clean_uid}"
        if identity_id not in existing_identities:
            if not dry_run:
                ensure_identity_node(
                    mem_conn, clean_uid,
                    turns[0]["created_at"] if turns else time.time(),
                )
            existing_identities.add(identity_id)
            stats["identities_created"] += 1

        # Create memory nodes for each meaningful turn
        for turn in turns:
            stats["turns_extracted"] += 1

            if dry_run:
                stats["nodes_created"] += 1
                continue

            mem_conn.execute(
                """INSERT INTO memory_nodes (
                    id, type, user_prompt, agent_response, source_id,
                    status, score, base_weight, anchor_bonus,
                    reinforcement_count, emotional_intensity,
                    created_at, last_accessed, last_reinforced,
                    mempool_reinforcements, mempool_weight, effective_weight
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    next_id,
                    "MemoryNode",
                    turn["user_prompt"],
                    turn["agent_response"],
                    identity_id,
                    "persistent",  # survived Titan's lifetime — earned persistent status
                    0.5,  # neutral initial score
                    1.0,
                    0.0,
                    0,
                    0,
                    turn["created_at"],
                    turn["created_at"],  # last_accessed = created
                    None,
                    0,
                    1.0,
                    1.0,
                ),
            )
            next_id += 1
            stats["nodes_created"] += 1

        if stats["sessions_processed"] % 10 == 0:
            logger.info(
                "Progress: %d/%d sessions, %d nodes created",
                stats["sessions_processed"], len(sessions), stats["nodes_created"],
            )

    if not dry_run:
        mem_conn.commit()
        logger.info("Committed %d new nodes to memory_nodes.db", stats["nodes_created"])

    stats["existing_nodes_after"] = (
        get_existing_node_count(mem_conn) if not dry_run else stats["existing_nodes_before"]
    )

    sess_conn.close()
    mem_conn.close()

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backfill memory_nodes from agno_sessions")
    parser.add_argument("--dry-run", action="store_true", help="Count without writing")
    parser.add_argument("--limit", type=int, default=0, help="Max sessions to process (0=all)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Titan Memory Backfill — agno_sessions → memory_nodes")
    logger.info("=" * 60)
    logger.info("Source: %s", SESSIONS_DB)
    logger.info("Target: %s", MEMORY_DB)

    # Backup before migration (Memory Preservation Protocol)
    if not args.dry_run:
        backup_path = MEMORY_DB + f".pre_migration_{int(time.time())}"
        import shutil
        shutil.copy2(MEMORY_DB, backup_path)
        logger.info("Backup created: %s", backup_path)

    stats = backfill(dry_run=args.dry_run, limit=args.limit)

    logger.info("=" * 60)
    logger.info("Migration %s", "DRY RUN complete" if args.dry_run else "COMPLETE")
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
