"""MakerProfile — dialogue history + bond health for the Maker-Titan bond layer.

Storage: SQLite tables in the existing `data/maker_proposals.db` (already
in AUXILIARY_BACKUP_PATHS for Arweave backup — no new wiring needed).

The dialogue log is append-only. Bond health metrics are computed on-demand
from the log entries (no separate materialized state to get stale).

This is the DECISION profile — tracking what was proposed, approved, declined,
and Titan's reflections. Distinct from MakerRelationshipEngine which tracks
the Maker's PERSONALITY from conversation analysis. Both are wired together
via MAKER_DIALOGUE_COMPLETE events.
"""
import json
import logging
import sqlite3
import threading
import time
from typing import Optional

logger = logging.getLogger("MakerProfile")


class MakerProfile:
    """Maker dialogue history + bond health queries."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.RLock()
        self._init_tables()

    def _init_tables(self) -> None:
        """Create maker_dialogue table if not exists."""
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path, timeout=5.0)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS maker_dialogue (
                        dialogue_id TEXT PRIMARY KEY,
                        proposal_id TEXT NOT NULL,
                        proposal_type TEXT NOT NULL,
                        response TEXT NOT NULL,
                        maker_reason TEXT NOT NULL,
                        titan_narration TEXT DEFAULT '',
                        titan_reasoning TEXT DEFAULT '',
                        neuromod_snapshot TEXT DEFAULT '{}',
                        grounded_words TEXT DEFAULT '[]',
                        created_at REAL NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_dialogue_created
                    ON maker_dialogue(created_at DESC)
                """)
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning("[MakerProfile] Table init error: %s", e)

    def add_dialogue_entry(
        self,
        *,
        proposal_id: str,
        proposal_type: str,
        response: str,
        maker_reason: str,
        titan_narration: str = "",
        titan_reasoning: str = "",
        neuromod_snapshot: Optional[dict] = None,
        grounded_words: Optional[list] = None,
    ) -> str:
        """Add a dialogue entry. Returns dialogue_id."""
        import hashlib
        dialogue_id = hashlib.sha256(
            f"{proposal_id}:{response}:{time.time()}".encode()
        ).hexdigest()[:16]

        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path, timeout=5.0)
                conn.execute(
                    """INSERT OR IGNORE INTO maker_dialogue
                       (dialogue_id, proposal_id, proposal_type, response,
                        maker_reason, titan_narration, titan_reasoning,
                        neuromod_snapshot, grounded_words, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        dialogue_id, proposal_id, proposal_type, response,
                        maker_reason, titan_narration, titan_reasoning,
                        json.dumps(neuromod_snapshot or {}),
                        json.dumps(grounded_words or []),
                        time.time(),
                    ),
                )
                conn.commit()
                conn.close()
                logger.info(
                    "[MakerProfile] Dialogue entry: id=%s type=%s response=%s",
                    dialogue_id, proposal_type, response)
            except Exception as e:
                logger.warning("[MakerProfile] add_dialogue_entry error: %s", e)
        return dialogue_id

    def get_recent_dialogue(self, n: int = 10) -> list[dict]:
        """Return last N dialogue entries, newest first."""
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path, timeout=5.0)
                rows = conn.execute(
                    """SELECT dialogue_id, proposal_id, proposal_type, response,
                              maker_reason, titan_narration, titan_reasoning,
                              neuromod_snapshot, grounded_words, created_at
                       FROM maker_dialogue ORDER BY created_at DESC LIMIT ?""",
                    (n,),
                ).fetchall()
                conn.close()
                return [
                    {
                        "dialogue_id": r[0], "proposal_id": r[1],
                        "proposal_type": r[2], "response": r[3],
                        "maker_reason": r[4], "titan_narration": r[5],
                        "titan_reasoning": r[6],
                        "neuromod_snapshot": json.loads(r[7] or "{}"),
                        "grounded_words": json.loads(r[8] or "[]"),
                        "created_at": r[9],
                    }
                    for r in rows
                ]
            except Exception as e:
                logger.warning("[MakerProfile] get_recent_dialogue error: %s", e)
                return []

    def get_bond_health(self) -> dict:
        """Compute bond health metrics from dialogue history."""
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path, timeout=5.0)
                # Total interaction count
                total = conn.execute(
                    "SELECT COUNT(*) FROM maker_dialogue"
                ).fetchone()[0]
                # Approve / decline counts
                approves = conn.execute(
                    "SELECT COUNT(*) FROM maker_dialogue WHERE response='approve'"
                ).fetchone()[0]
                declines = conn.execute(
                    "SELECT COUNT(*) FROM maker_dialogue WHERE response='decline'"
                ).fetchone()[0]
                # Average reason depth (character length)
                avg_reason = conn.execute(
                    "SELECT AVG(LENGTH(maker_reason)) FROM maker_dialogue"
                ).fetchone()[0] or 0
                # Topic diversity (distinct proposal types)
                types = conn.execute(
                    "SELECT COUNT(DISTINCT proposal_type) FROM maker_dialogue"
                ).fetchone()[0]
                # Last interaction timestamp
                last_ts = conn.execute(
                    "SELECT MAX(created_at) FROM maker_dialogue"
                ).fetchone()[0] or 0
                # Agreement trajectory (last 10 vs previous 10)
                recent_10 = conn.execute(
                    """SELECT response FROM maker_dialogue
                       ORDER BY created_at DESC LIMIT 10"""
                ).fetchall()
                prev_10 = conn.execute(
                    """SELECT response FROM maker_dialogue
                       ORDER BY created_at DESC LIMIT 10 OFFSET 10"""
                ).fetchall()
                conn.close()

                recent_approval_rate = (
                    sum(1 for r in recent_10 if r[0] == "approve") / max(1, len(recent_10))
                )
                prev_approval_rate = (
                    sum(1 for r in prev_10 if r[0] == "approve") / max(1, len(prev_10))
                    if prev_10 else recent_approval_rate
                )
                trajectory = recent_approval_rate - prev_approval_rate

                return {
                    "interaction_count": total,
                    "approves": approves,
                    "declines": declines,
                    "avg_reason_depth": round(avg_reason, 1),
                    "topic_diversity": types,
                    "last_interaction_ts": last_ts,
                    "agreement_trajectory": round(trajectory, 3),
                    "recent_approval_rate": round(recent_approval_rate, 3),
                }
            except Exception as e:
                logger.warning("[MakerProfile] get_bond_health error: %s", e)
                return {
                    "interaction_count": 0, "approves": 0, "declines": 0,
                    "avg_reason_depth": 0, "topic_diversity": 0,
                    "last_interaction_ts": 0, "agreement_trajectory": 0,
                    "recent_approval_rate": 0.5,
                }

    def get_dialogue_for_introspect(self, n: int = 5) -> str:
        """Format recent dialogue for INTROSPECT sub-mode consumption.

        Returns a human-readable summary of the last N decisions that
        meta-reasoning can use as context during the maker_alignment sub-mode.
        """
        recent = self.get_recent_dialogue(n)
        if not recent:
            return "No Maker dialogue history yet."
        lines = []
        for d in recent:
            action = "approved" if d["response"] == "approve" else "declined"
            reason_preview = d["maker_reason"][:80] if d["maker_reason"] else ""
            narration_preview = d["titan_narration"][:80] if d["titan_narration"] else ""
            line = f"- Maker {action} '{d['proposal_type']}': \"{reason_preview}\""
            if narration_preview:
                line += f" → I reflected: \"{narration_preview}\""
            lines.append(line)
        return "\n".join(lines)
