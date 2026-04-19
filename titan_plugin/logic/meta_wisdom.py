"""
titan_plugin/logic/meta_wisdom.py — Distilled reasoning strategy store.

Stores proven reasoning strategies from successful meta-reasoning chains.
Strategies have a lifecycle: creation → reuse → crystallization or decay.

Used by: FORMULATE.load_wisdom, dream consolidation (decay/prune),
         KIN_SENSE (cross-instance transfer).
"""

import json
import logging
import sqlite3
import time
from typing import Optional

logger = logging.getLogger("titan.meta_wisdom")


class MetaWisdomStore:
    """Distilled reasoning strategies with decay/crystallization lifecycle."""

    def __init__(self, db_path: str = "./data/inner_memory.db",
                 decay_rate: float = 0.95,
                 crystallize_reuses: int = 10,
                 crystallize_success: float = 0.60,
                 prune_threshold: float = 0.10):
        self._db_path = db_path
        self._decay_rate = decay_rate
        self._crystallize_reuses = crystallize_reuses
        self._crystallize_success = crystallize_success
        self._prune_threshold = prune_threshold
        self._init_tables()

    def _get_db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._db_path, timeout=10.0)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=5000")
        return db

    def _init_tables(self) -> None:
        try:
            db = self._get_db()
            db.execute("""
                CREATE TABLE IF NOT EXISTS meta_wisdom (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    problem_pattern TEXT NOT NULL,
                    strategy_sequence TEXT NOT NULL,
                    problem_embedding BLOB,
                    strategy_embedding BLOB,
                    outcome_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    times_reused INTEGER DEFAULT 0,
                    times_successful INTEGER DEFAULT 0,
                    crystallized INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'self',
                    source_kin TEXT,
                    created_at REAL NOT NULL,
                    last_used REAL
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS idx_mw_confidence ON meta_wisdom(confidence)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_mw_crystallized ON meta_wisdom(crystallized)")
            db.commit()
            db.close()
        except Exception as e:
            logger.warning("[MetaWisdom] Init tables error: %s", e)

    def store_wisdom(
        self,
        problem_pattern: str,
        strategy_sequence: list,
        outcome_score: float,
        problem_embedding: list = None,
        strategy_embedding: list = None,
        source: str = "self",
        source_kin: str = None,
    ) -> int:
        try:
            confidence = outcome_score
            if source == "kin":
                confidence *= 0.5
            db = self._get_db()
            cursor = db.execute(
                "INSERT INTO meta_wisdom "
                "(problem_pattern, strategy_sequence, problem_embedding, strategy_embedding, "
                "outcome_score, confidence, source, source_kin, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    problem_pattern,
                    json.dumps(strategy_sequence),
                    json.dumps(problem_embedding) if problem_embedding else None,
                    json.dumps(strategy_embedding) if strategy_embedding else None,
                    outcome_score,
                    confidence,
                    source,
                    source_kin,
                    time.time(),
                ),
            )
            row_id = cursor.lastrowid
            db.commit()
            db.close()
            return row_id
        except Exception as e:
            logger.warning("[MetaWisdom] Store error: %s", e)
            return -1

    def query_by_pattern(
        self,
        problem_pattern: str,
        min_confidence: float = 0.3,
        limit: int = 5,
    ) -> list:
        try:
            db = self._get_db()
            # Split pattern into keywords for LIKE matching
            keywords = [w.strip() for w in problem_pattern.lower().split() if len(w.strip()) > 2][:5]
            if not keywords:
                db.close()
                return []
            # Match any keyword in problem_pattern
            conditions = " OR ".join(["LOWER(problem_pattern) LIKE ?" for _ in keywords])
            params = [f"%{kw}%" for kw in keywords[:5]]
            params.extend([min_confidence, limit])
            rows = db.execute(
                f"SELECT id, problem_pattern, strategy_sequence, outcome_score, confidence, "
                f"times_reused, times_successful, crystallized, source, created_at, last_used "
                f"FROM meta_wisdom WHERE ({conditions}) AND confidence >= ? "
                f"ORDER BY confidence DESC LIMIT ?",
                params,
            ).fetchall()
            db.close()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.warning("[MetaWisdom] Query by pattern error: %s", e)
            return []

    def query_by_embedding(
        self,
        problem_embedding: list,
        min_confidence: float = 0.3,
        top_k: int = 5,
    ) -> list:
        try:
            db = self._get_db()
            rows = db.execute(
                "SELECT id, problem_pattern, strategy_sequence, outcome_score, confidence, "
                "times_reused, times_successful, crystallized, source, created_at, last_used, "
                "problem_embedding "
                "FROM meta_wisdom WHERE problem_embedding IS NOT NULL AND confidence >= ?",
                (min_confidence,),
            ).fetchall()
            db.close()

            if not rows:
                return []

            import math
            scored = []
            for r in rows:
                emb_blob = r[11]
                if not emb_blob:
                    continue
                stored_emb = json.loads(emb_blob) if isinstance(emb_blob, str) else list(emb_blob)
                sim = _cosine_sim(problem_embedding, stored_emb)
                scored.append((sim, r))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {**self._row_to_dict(r[:11]), "similarity": round(sim, 4)}
                for sim, r in scored[:top_k]
            ]
        except Exception as e:
            logger.warning("[MetaWisdom] Query by embedding error: %s", e)
            return []

    def record_reuse(self, wisdom_id: int, success: bool) -> None:
        try:
            db = self._get_db()
            if success:
                db.execute(
                    "UPDATE meta_wisdom SET "
                    "times_reused = times_reused + 1, "
                    "times_successful = times_successful + 1, "
                    "confidence = MIN(1.0, confidence * 1.10), "
                    "last_used = ? WHERE id = ?",
                    (time.time(), wisdom_id),
                )
            else:
                db.execute(
                    "UPDATE meta_wisdom SET "
                    "times_reused = times_reused + 1, "
                    "confidence = MAX(0.0, confidence * 0.85), "
                    "last_used = ? WHERE id = ?",
                    (time.time(), wisdom_id),
                )
            # Check crystallization
            row = db.execute(
                "SELECT times_reused, times_successful FROM meta_wisdom WHERE id = ?",
                (wisdom_id,),
            ).fetchone()
            if row and row[0] >= self._crystallize_reuses:
                success_rate = row[1] / row[0] if row[0] > 0 else 0
                if success_rate >= self._crystallize_success:
                    db.execute(
                        "UPDATE meta_wisdom SET crystallized = 1 WHERE id = ?",
                        (wisdom_id,),
                    )
                    logger.info("[MetaWisdom] Strategy %d crystallized (reuses=%d, success=%.0f%%)",
                                wisdom_id, row[0], success_rate * 100)
            db.commit()
            db.close()
        except Exception as e:
            logger.warning("[MetaWisdom] Record reuse error: %s", e)

    def force_crystallize(self, wisdom_id: int) -> None:
        """Immediately crystallize a wisdom entry (M9: EUREKA event)."""
        try:
            db = self._get_db()
            db.execute(
                "UPDATE meta_wisdom SET crystallized = 1, "
                "confidence = MAX(confidence, 0.8) WHERE id = ?",
                (wisdom_id,),
            )
            db.commit()
            db.close()
            logger.info("[MetaWisdom] Wisdom %d EUREKA-crystallized", wisdom_id)
        except Exception as e:
            logger.warning("[MetaWisdom] Force crystallize error: %s", e)

    def dream_decay(self) -> dict:
        try:
            db = self._get_db()
            # Decay non-crystallized entries
            db.execute(
                "UPDATE meta_wisdom SET confidence = confidence * ? "
                "WHERE crystallized = 0",
                (self._decay_rate,),
            )
            # Prune below threshold
            cursor = db.execute(
                "DELETE FROM meta_wisdom WHERE confidence < ? AND crystallized = 0",
                (self._prune_threshold,),
            )
            pruned = cursor.rowcount
            # Stats
            remaining = db.execute("SELECT COUNT(*) FROM meta_wisdom").fetchone()[0]
            avg_conf = db.execute("SELECT AVG(confidence) FROM meta_wisdom").fetchone()[0]
            crystallized = db.execute(
                "SELECT COUNT(*) FROM meta_wisdom WHERE crystallized = 1"
            ).fetchone()[0]
            db.commit()
            db.close()
            return {
                "pruned": pruned,
                "remaining": remaining,
                "avg_confidence": round(avg_conf or 0, 4),
                "crystallized": crystallized,
            }
        except Exception as e:
            logger.warning("[MetaWisdom] Dream decay error: %s", e)
            return {"pruned": 0, "remaining": 0, "avg_confidence": 0, "crystallized": 0}

    def get_crystallized(self, limit: int = 20) -> list:
        try:
            db = self._get_db()
            rows = db.execute(
                "SELECT id, problem_pattern, strategy_sequence, outcome_score, confidence, "
                "times_reused, times_successful, crystallized, source, created_at, last_used "
                "FROM meta_wisdom WHERE crystallized = 1 "
                "ORDER BY confidence DESC LIMIT ?",
                (limit,),
            ).fetchall()
            db.close()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.warning("[MetaWisdom] Get crystallized error: %s", e)
            return []

    def import_kin_wisdom(self, wisdom_entries: list, source_kin: str) -> int:
        imported = 0
        for entry in wisdom_entries:
            result = self.store_wisdom(
                problem_pattern=entry.get("problem_pattern", ""),
                strategy_sequence=entry.get("strategy_sequence", []),
                outcome_score=entry.get("outcome_score", 0.5),
                problem_embedding=entry.get("problem_embedding"),
                strategy_embedding=entry.get("strategy_embedding"),
                source="kin",
                source_kin=source_kin,
            )
            if result > 0:
                imported += 1
        return imported

    def prune_to_cap(self, max_entries: int = 500) -> int:
        """Prune low-confidence wisdom to prevent embedding space saturation.

        Keeps the top *max_entries* by confidence, deleting the rest.
        Crystallized entries get a 0.2 confidence bonus for ranking.
        Returns the number of entries deleted.
        """
        try:
            db = self._get_db()
            total = db.execute("SELECT COUNT(*) FROM meta_wisdom").fetchone()[0]
            if total <= max_entries:
                db.close()
                return 0
            # Keep highest effective confidence (crystallized get bonus)
            # Use ROWID ordering as tiebreaker to prefer newer entries
            db.execute("""
                DELETE FROM meta_wisdom WHERE id NOT IN (
                    SELECT id FROM meta_wisdom
                    ORDER BY (confidence + CASE WHEN crystallized = 1 THEN 0.2 ELSE 0.0 END) DESC,
                             id DESC
                    LIMIT ?
                )
            """, (max_entries,))
            deleted = db.execute("SELECT changes()").fetchone()[0]
            db.commit()
            db.close()
            if deleted > 0:
                logger.info("[MetaWisdom] Pruned %d entries (kept top %d by confidence, was %d)",
                            deleted, max_entries, total)
            return deleted
        except Exception as e:
            logger.warning("[MetaWisdom] Prune error: %s", e)
            return 0

    def get_stats(self) -> dict:
        try:
            db = self._get_db()
            total = db.execute("SELECT COUNT(*) FROM meta_wisdom").fetchone()[0]
            crystallized = db.execute(
                "SELECT COUNT(*) FROM meta_wisdom WHERE crystallized = 1"
            ).fetchone()[0]
            avg_conf = db.execute("SELECT AVG(confidence) FROM meta_wisdom").fetchone()[0]
            by_source = dict(
                db.execute("SELECT source, COUNT(*) FROM meta_wisdom GROUP BY source").fetchall()
            )
            db.close()
            return {
                "total": total,
                "crystallized": crystallized,
                "avg_confidence": round(avg_conf or 0, 4),
                "by_source": by_source,
            }
        except Exception as e:
            logger.warning("[MetaWisdom] Stats error: %s", e)
            return {"total": 0}

    @staticmethod
    def _row_to_dict(row) -> dict:
        return {
            "id": row[0],
            "problem_pattern": row[1],
            "strategy_sequence": json.loads(row[2]) if row[2] else [],
            "outcome_score": row[3],
            "confidence": row[4],
            "times_reused": row[5],
            "times_successful": row[6],
            "crystallized": bool(row[7]),
            "source": row[8],
            "created_at": row[9],
            "last_used": row[10],
        }


def _cosine_sim(a: list, b: list) -> float:
    import math
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-8 or mag_b < 1e-8:
        return 0.0
    return dot / (mag_a * mag_b)
