"""
titan_plugin/logic/chain_archive.py — Persistent reasoning chain archive.

Stores completed main reasoning chains (high-scored) and meta-reasoning chains.
Queryable by domain, outcome score, and embedding similarity (once autoencoder trained).

Used by: meta-reasoning RECALL, FORMULATE.load_wisdom, dream consolidation.
"""

import json
import logging
import math
import os
import sqlite3
import time
from typing import Optional

logger = logging.getLogger("titan.chain_archive")


class ChainArchive:
    """Persistent archive of reasoning chains for meta-reasoning recall."""

    def __init__(self, db_path: str = "./data/inner_memory.db"):
        self._db_path = db_path
        self._init_tables()

    def _get_db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._db_path, timeout=30.0)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=15000")
        return db

    def _retry_write(self, operation, max_retries: int = 3):
        """Execute a write operation with retry + exponential backoff."""
        for attempt in range(max_retries):
            try:
                return operation()
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    wait = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
                    time.sleep(wait)
                    continue
                raise

    def _init_tables(self) -> None:
        try:
            db = self._get_db()
            db.execute("""
                CREATE TABLE IF NOT EXISTS chain_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    chain_sequence TEXT NOT NULL,
                    chain_length INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    gut_agreement REAL DEFAULT 0.5,
                    outcome_score REAL NOT NULL,
                    domain TEXT DEFAULT 'general',
                    strategy_label TEXT,
                    problem_type TEXT,
                    observation_snapshot TEXT,
                    observation_embedding BLOB,
                    reasoning_plan TEXT,
                    meta_chain_id INTEGER,
                    epoch_id INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    dream_consolidated INTEGER DEFAULT 0,
                    FOREIGN KEY (meta_chain_id) REFERENCES chain_archive(id)
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS idx_ca_source ON chain_archive(source)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_ca_domain ON chain_archive(domain)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_ca_outcome ON chain_archive(outcome_score)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_ca_consolidated ON chain_archive(dream_consolidated)")
            # Composite index for get_unconsolidated() — satisfies
            # WHERE dream_consolidated=0 + ORDER BY created_at DESC in one seek.
            # Before this: full sort of ~60K rows per call (~1M comparisons).
            # Added 2026-04-16 after observing autoencoder dream phase stalls
            # (see titan-docs/INVESTIGATION_spirit_hang_root_cause.md).
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_ca_unconsol_recent "
                "ON chain_archive(dream_consolidated, created_at DESC)"
            )
            db.commit()
            db.close()
        except Exception as e:
            logger.warning("[ChainArchive] Init tables error: %s", e)

    def record_main_chain(
        self,
        chain_sequence: list,
        confidence: float,
        gut_agreement: float,
        outcome_score: float,
        domain: str,
        observation_snapshot: list,
        epoch_id: int,
        chain_results: list = None,
        reasoning_plan: dict = None,
    ) -> int:
        def _do_insert():
            db = self._get_db()
            try:
                cursor = db.execute(
                    "INSERT INTO chain_archive "
                    "(source, chain_sequence, chain_length, confidence, gut_agreement, "
                    "outcome_score, domain, observation_snapshot, reasoning_plan, "
                    "epoch_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "main",
                        json.dumps(chain_sequence),
                        len(chain_sequence),
                        confidence,
                        gut_agreement,
                        outcome_score,
                        domain,
                        json.dumps(observation_snapshot) if observation_snapshot else None,
                        json.dumps(reasoning_plan) if reasoning_plan else None,
                        epoch_id,
                        time.time(),
                    ),
                )
                row_id = cursor.lastrowid
                db.commit()
                return row_id
            finally:
                db.close()
        try:
            return self._retry_write(_do_insert)
        except Exception as e:
            logger.warning("[ChainArchive] Record main chain error: %s", e)
            return -1

    def record_meta_chain(
        self,
        chain_sequence: list,
        confidence: float,
        outcome_score: float,
        problem_type: str,
        strategy_label: str,
        observation_snapshot: list,
        epoch_id: int,
        chain_results: list = None,
        sub_chain_ids: list = None,
    ) -> int:
        def _do_insert():
            db = self._get_db()
            try:
                cursor = db.execute(
                    "INSERT INTO chain_archive "
                    "(source, chain_sequence, chain_length, confidence, "
                    "outcome_score, domain, strategy_label, problem_type, "
                    "observation_snapshot, reasoning_plan, epoch_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "meta",
                        json.dumps(chain_sequence),
                        len(chain_sequence),
                        confidence,
                        outcome_score,
                        "meta",
                        strategy_label,
                        problem_type,
                        json.dumps(observation_snapshot) if observation_snapshot else None,
                        json.dumps({"sub_chain_ids": sub_chain_ids}) if sub_chain_ids else None,
                        epoch_id,
                        time.time(),
                    ),
                )
                row_id = cursor.lastrowid
                db.commit()
                return row_id
            finally:
                db.close()
        try:
            return self._retry_write(_do_insert)
        except Exception as e:
            logger.warning("[ChainArchive] Record meta chain error: %s", e)
            return -1

    def query_by_domain(
        self,
        domain: str,
        source: str = "main",
        min_outcome: float = 0.0,
        limit: int = 20,
    ) -> list:
        try:
            db = self._get_db()
            rows = db.execute(
                "SELECT id, chain_sequence, chain_length, confidence, gut_agreement, "
                "outcome_score, domain, strategy_label, epoch_id, created_at "
                "FROM chain_archive "
                "WHERE source = ? AND domain = ? AND outcome_score >= ? "
                "ORDER BY outcome_score DESC LIMIT ?",
                (source, domain, min_outcome, limit),
            ).fetchall()
            db.close()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.warning("[ChainArchive] Query by domain error: %s", e)
            return []

    def query_high_scoring(
        self,
        min_outcome: float = 0.6,
        source: str = "main",
        limit: int = 10,
    ) -> list:
        try:
            db = self._get_db()
            rows = db.execute(
                "SELECT id, chain_sequence, chain_length, confidence, gut_agreement, "
                "outcome_score, domain, strategy_label, epoch_id, created_at "
                "FROM chain_archive "
                "WHERE source = ? AND outcome_score >= ? "
                "ORDER BY outcome_score DESC LIMIT ?",
                (source, min_outcome, limit),
            ).fetchall()
            db.close()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.warning("[ChainArchive] Query high scoring error: %s", e)
            return []

    def query_by_embedding(
        self,
        embedding: list,
        source: str = None,
        top_k: int = 5,
    ) -> list:
        try:
            db = self._get_db()
            query = "SELECT id, chain_sequence, chain_length, confidence, gut_agreement, " \
                    "outcome_score, domain, strategy_label, epoch_id, created_at, " \
                    "observation_embedding FROM chain_archive WHERE observation_embedding IS NOT NULL"
            params = []
            if source:
                query += " AND source = ?"
                params.append(source)
            rows = db.execute(query, params).fetchall()
            db.close()

            if not rows:
                return []

            scored = []
            for r in rows:
                emb_blob = r[10]
                if not emb_blob:
                    continue
                stored_emb = json.loads(emb_blob) if isinstance(emb_blob, str) else list(emb_blob)
                sim = self._cosine_sim(embedding, stored_emb)
                scored.append((sim, r))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {**self._row_to_dict(r), "similarity": round(sim, 4)}
                for sim, r in scored[:top_k]
            ]
        except Exception as e:
            logger.warning("[ChainArchive] Query by embedding error: %s", e)
            return []

    def get_unconsolidated(self, limit: int = 50) -> list:
        try:
            db = self._get_db()
            rows = db.execute(
                "SELECT id, chain_sequence, chain_length, confidence, gut_agreement, "
                "outcome_score, domain, strategy_label, epoch_id, created_at, "
                "observation_snapshot "
                "FROM chain_archive WHERE dream_consolidated = 0 "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            db.close()
            result = []
            for r in rows:
                d = self._row_to_dict(r)
                if r[10]:
                    d["observation_snapshot"] = json.loads(r[10])
                result.append(d)
            return result
        except Exception as e:
            logger.warning("[ChainArchive] Get unconsolidated error: %s", e)
            return []

    def mark_consolidated(self, chain_ids: list) -> None:
        if not chain_ids:
            return
        try:
            db = self._get_db()
            placeholders = ",".join("?" * len(chain_ids))
            db.execute(
                f"UPDATE chain_archive SET dream_consolidated = 1 WHERE id IN ({placeholders})",
                chain_ids,
            )
            db.commit()
            db.close()
        except Exception as e:
            logger.warning("[ChainArchive] Mark consolidated error: %s", e)

    def update_embedding(self, chain_id: int, embedding: list) -> None:
        try:
            db = self._get_db()
            db.execute(
                "UPDATE chain_archive SET observation_embedding = ? WHERE id = ?",
                (json.dumps(embedding), chain_id),
            )
            db.commit()
            db.close()
        except Exception as e:
            logger.debug("[ChainArchive] Update embedding error: %s", e)

    def update_embeddings_batch(self, pairs: list) -> int:
        """Batch-update embeddings in a single transaction (one fsync).

        ``pairs`` is a list of ``(chain_id, embedding_list)``. Uses the same
        retry/backoff pattern as other writes. Returns the number of rows
        queued for update (not necessarily the number actually persisted if
        some ids don't exist — SQLite's UPDATE is silent on no-match).

        Introduced 2026-04-16 to fix the autoencoder dream-cycle bottleneck:
        previously backfilling N embeddings did N separate open+execute+commit+close
        cycles, each subject to SQLite's 15s busy_timeout. On a contended DB
        this produced 17-minute dream cycles on T2 (68 rows × ~15s each).
        Single-transaction executemany drops that to one lock acquisition.
        """
        if not pairs:
            return 0

        def _do_batch():
            db = self._get_db()
            try:
                db.executemany(
                    "UPDATE chain_archive SET observation_embedding = ? WHERE id = ?",
                    [(json.dumps(emb), cid) for cid, emb in pairs],
                )
                db.commit()
                return len(pairs)
            finally:
                db.close()

        try:
            return self._retry_write(_do_batch)
        except Exception as e:
            logger.warning("[ChainArchive] Batch update embeddings error: %s", e)
            return 0

    def get_chains_without_embedding(self, limit: int = 100) -> list:
        try:
            db = self._get_db()
            rows = db.execute(
                "SELECT id, observation_snapshot FROM chain_archive "
                "WHERE observation_embedding IS NULL AND observation_snapshot IS NOT NULL "
                "LIMIT ?",
                (limit,),
            ).fetchall()
            db.close()
            result = []
            for row_id, snap_json in rows:
                if snap_json:
                    snap = json.loads(snap_json)
                    if len(snap) >= 65:
                        result.append({"id": row_id, "observation_snapshot": snap})
            return result
        except Exception as e:
            logger.debug("[ChainArchive] Get without embedding error: %s", e)
            return []

    def prune_old(self, max_age_days: int = 14, keep_min: int = 100) -> int:
        try:
            db = self._get_db()
            total = db.execute("SELECT COUNT(*) FROM chain_archive").fetchone()[0]
            if total <= keep_min:
                db.close()
                return 0
            cutoff = time.time() - (max_age_days * 86400)
            cursor = db.execute(
                "DELETE FROM chain_archive WHERE created_at < ? AND outcome_score < 0.5 "
                "AND dream_consolidated = 1",
                (cutoff,),
            )
            pruned = cursor.rowcount
            db.commit()
            db.close()
            return pruned
        except Exception as e:
            logger.warning("[ChainArchive] Prune error: %s", e)
            return 0

    def get_stats(self) -> dict:
        try:
            db = self._get_db()
            total = db.execute("SELECT COUNT(*) FROM chain_archive").fetchone()[0]
            by_source = dict(
                db.execute(
                    "SELECT source, COUNT(*) FROM chain_archive GROUP BY source"
                ).fetchall()
            )
            by_domain = dict(
                db.execute(
                    "SELECT domain, COUNT(*) FROM chain_archive GROUP BY domain"
                ).fetchall()
            )
            avg_outcome = db.execute(
                "SELECT AVG(outcome_score) FROM chain_archive"
            ).fetchone()[0]
            unconsolidated = db.execute(
                "SELECT COUNT(*) FROM chain_archive WHERE dream_consolidated = 0"
            ).fetchone()[0]
            db.close()
            return {
                "total": total,
                "by_source": by_source,
                "by_domain": by_domain,
                "avg_outcome": round(avg_outcome or 0, 4),
                "unconsolidated": unconsolidated,
            }
        except Exception as e:
            logger.warning("[ChainArchive] Stats error: %s", e)
            return {"total": 0}

    @staticmethod
    def _row_to_dict(row) -> dict:
        return {
            "id": row[0],
            "chain_sequence": json.loads(row[1]) if row[1] else [],
            "chain_length": row[2],
            "confidence": row[3],
            "gut_agreement": row[4],
            "outcome_score": row[5],
            "domain": row[6],
            "strategy_label": row[7],
            "epoch_id": row[8],
            "created_at": row[9],
        }

    @staticmethod
    def _cosine_sim(a: list, b: list) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a < 1e-8 or mag_b < 1e-8:
            return 0.0
        return dot / (mag_a * mag_b)
