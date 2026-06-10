"""ReasoningStore — the graphed outer chain-of-thought under SELF → LEARNING → REASONING.

RFP_synthesis_self_learning_meta_reasoning v1.1 (§1.2 C1 / S2; INV-OML-11). Every
reasoning EPISODE (a per-tool-use leaf, or a distilled macro-strategy) is persisted
as ONE deref-able record across the three substrates, written at verdict-time
through the single SynthesisWriter (INV-Syn-19/28):

  • DuckDB `reasoning_records` — the real recallable scalars (features, reward, the
    BRAIN §3.4 triple {b_i, c, time_cost, use_count}). PK-only, NO secondary ART
    index (the actr_buffers/skill_cells crash-class rule — INSERT-heavy table).
  • FAISS  `reasoning_vectors.faiss` — the goal/context signature, so SC `search`
    finds the record (IndexFlatL2; position = embedding_id, mapped to reasoning_id).
  • Kuzu   `Reasoning` node + `LEARNING_HAS_REASONING` edge under the `Learning`
    hub under `Self` — the graph structure ("what have I reasoned / learned?").

The timechain holds ONLY the `tx_hash` pointer (`reasoning_id` = the procedural-fork
tool_call_tx); the real data lives here (SPEC ENGINE MECHANIC: SC SEARCH → tx_hash →
DEREF → real node). Every write is soft-fail — a record-write failure must NEVER
break the verdict path.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional

from titan_hcl.synthesis.writer import resolve_writer

logger = logging.getLogger(__name__)

EMBEDDING_DIM: int = 384


class ReasoningStore:
    """Sole writer of `reasoning_records` + `reasoning_vectors.faiss` + the Kuzu
    Reasoning nodes. Synthesis-worker-side; all mutations on the writer thread."""

    def __init__(
        self,
        duckdb_conn: Any,
        *,
        faiss_path: str,
        graph: Any = None,
        embedder: Optional[Callable[[str], Any]] = None,
        writer: Any = None,
        clock: Callable[[], float] = time.time,
    ):
        self._db = duckdb_conn
        self._faiss_path = str(faiss_path)
        self._graph = graph                 # core.direct_memory KnowledgeGraph (Kuzu)
        self._embedder = embedder
        self._writer = resolve_writer(writer)
        self._clock = clock
        self._faiss = None
        self._faiss_dim = EMBEDDING_DIM
        self.records_written = 0
        self.macros_written = 0
        self._init_schema()

    # ── schema (PK-only, NO secondary index — crash-class rule) ──────────
    def _init_schema(self) -> None:
        def _create() -> None:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS reasoning_records ("
                "  reasoning_id  TEXT    PRIMARY KEY,"   # = the procedural-fork tool_call_tx
                "  kind          TEXT    NOT NULL,"      # 'tool_use' | 'macro_strategy'
                "  goal_class    TEXT,"
                "  action        TEXT,"
                "  oracle_id     TEXT,"
                "  verdict       TEXT,"
                "  reward        DOUBLE,"
                "  features_json TEXT,"
                "  b_i           INTEGER DEFAULT 1,"
                "  c             DOUBLE  DEFAULT 1.0,"
                "  time_cost     DOUBLE  DEFAULT 1.0,"
                "  use_count     INTEGER DEFAULT 1,"
                "  embedding_id  INTEGER DEFAULT -1,"
                "  anchor_tx     TEXT,"
                "  created_at    DOUBLE  NOT NULL"
                ")")
        try:
            self._writer.submit_sync(_create)
        except Exception as e:  # noqa: BLE001
            logger.warning("[ReasoningStore] schema init failed: %s", e)

    # ── FAISS (mirror skill_store; atomic persist) ───────────────────────
    def _ensure_faiss(self) -> None:
        if self._faiss is not None:
            return
        import faiss  # local import — cold-boot RSS
        if os.path.exists(self._faiss_path):
            try:
                self._faiss = faiss.read_index(self._faiss_path)
                if self._faiss.d != self._faiss_dim:
                    self._faiss = faiss.IndexFlatL2(self._faiss_dim)
            except Exception:  # noqa: BLE001 — defensive
                self._faiss = faiss.IndexFlatL2(self._faiss_dim)
        else:
            self._faiss = faiss.IndexFlatL2(self._faiss_dim)

    def _save_faiss(self) -> None:
        if self._faiss is None:
            return
        import faiss
        try:
            os.makedirs(os.path.dirname(self._faiss_path) or ".", exist_ok=True)
            tmp = self._faiss_path + ".tmp"
            faiss.write_index(self._faiss, tmp)
            os.replace(tmp, self._faiss_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("[ReasoningStore] FAISS save failed: %s", e)

    def _embed_vec(self, text: str):
        if self._embedder is None or not text:
            return None
        try:
            import numpy as np
            vec = self._embedder(text)
            if vec is None:
                return None
            vec = np.asarray(vec, dtype=np.float32)
            if vec.ndim == 1:
                vec = vec.reshape(1, -1)
            if vec.shape[1] != self._faiss_dim:
                return None
            return vec
        except Exception as e:  # noqa: BLE001
            logger.debug("[ReasoningStore] embed failed: %s", e)
            return None

    # ── WRITE: one per-use leaf (C1) — all three substrates, writer-thread ──
    def record_tool_use(
        self, *, reasoning_id: str, goal_class: str, action: str, oracle_id: str,
        verdict: str, reward: float, features: list, signature_text: str,
        b_i: float = 1.0, c: float = 1.0, time_cost: float = 1.0,
    ) -> bool:
        """Persist a `Reasoning(kind='tool_use')` episode. Idempotent on
        reasoning_id (the tool_call_tx). Soft-fail."""
        return self._write_record(
            reasoning_id=reasoning_id, kind="tool_use", goal_class=goal_class,
            action=action, oracle_id=oracle_id, verdict=verdict, reward=reward,
            features=features, signature_text=signature_text,
            b_i=b_i, c=c, time_cost=time_cost, anchor_tx=reasoning_id,
            composed_from=None)

    def write_macro(
        self, *, reasoning_id: str, goal_class: str, action: str,
        signature: list, b_i: float, c: float, time_cost: float, use_count: int,
        anchor_tx: str = "", composed_from: Optional[list] = None,
    ) -> bool:
        """Persist a distilled `Reasoning(kind='macro_strategy')` (S2). The signature
        is the mean leaf feature vector; `composed_from` = the leaf reasoning_ids."""
        ok = self._write_record(
            reasoning_id=reasoning_id, kind="macro_strategy", goal_class=goal_class,
            action=action, oracle_id="", verdict="true", reward=1.0,
            features=list(signature or []), signature_text=str(goal_class or ""),
            b_i=b_i, c=c, time_cost=time_cost, use_count=int(use_count),
            anchor_tx=anchor_tx, composed_from=composed_from)
        if ok:
            self.macros_written += 1
        return ok

    def _write_record(
        self, *, reasoning_id, kind, goal_class, action, oracle_id, verdict,
        reward, features, signature_text, b_i, c, time_cost, use_count=1,
        anchor_tx="", composed_from=None,
    ) -> bool:
        if not reasoning_id:
            return False
        import json
        vec = self._embed_vec(signature_text)

        def _do_write() -> bool:
            # 1) FAISS signature (position = embedding_id)
            emb_id = -1
            if vec is not None:
                try:
                    self._ensure_faiss()
                    emb_id = int(self._faiss.ntotal)
                    self._faiss.add(vec)
                    self._save_faiss()
                except Exception as e:  # noqa: BLE001
                    logger.debug("[ReasoningStore] faiss add failed: %s", e)
                    emb_id = -1
            # 2) DuckDB scalars (the real recallable data) — idempotent on PK
            try:
                self._db.execute(
                    "INSERT INTO reasoning_records (reasoning_id, kind, goal_class, "
                    "action, oracle_id, verdict, reward, features_json, b_i, c, "
                    "time_cost, use_count, embedding_id, anchor_tx, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT (reasoning_id) DO NOTHING",
                    [str(reasoning_id), str(kind), str(goal_class or ""),
                     str(action or ""), str(oracle_id or ""), str(verdict or ""),
                     float(reward), json.dumps(list(features or [])),
                     int(b_i), float(c), float(time_cost), int(use_count),
                     int(emb_id), str(anchor_tx or ""), float(self._clock())])
            except Exception as e:  # noqa: BLE001
                logger.warning("[ReasoningStore] duckdb insert failed: %s", e)
                return False
            # 3) Kuzu node + edges (graph structure under SELF → LEARNING)
            if self._graph is not None:
                try:
                    self._graph.spine_create_reasoning_node(
                        reasoning_id=str(reasoning_id), kind=str(kind),
                        goal_class=str(goal_class or ""), action=str(action or ""),
                        oracle_id=str(oracle_id or ""), verdict=str(verdict or ""),
                        anchor_tx=str(anchor_tx or ""), created_at=float(self._clock()))
                    self._graph.spine_link_learning_reasoning(str(reasoning_id))
                    for leaf in (composed_from or []):
                        self._graph.spine_link_reasoning_composed_from(
                            str(reasoning_id), str(leaf))
                except Exception as e:  # noqa: BLE001
                    logger.debug("[ReasoningStore] kuzu write soft-fail: %s", e)
            return True

        try:
            ok = bool(self._writer.submit_sync(_do_write))
            if ok and kind == "tool_use":
                self.records_written += 1
            return ok
        except Exception as e:  # noqa: BLE001
            logger.warning("[ReasoningStore] record write failed: %s", e)
            return False

    # ── READ / SC-search (DEREF) ─────────────────────────────────────────
    def faiss_search(self, query_vec: Any, top_k: int = 10) -> list[tuple[int, float]]:
        try:
            import numpy as np
        except ImportError:
            return []
        if self._faiss is None:
            self._ensure_faiss()
        if self._faiss is None or self._faiss.ntotal == 0:
            return []
        try:
            vec = np.asarray(query_vec, dtype=np.float32)
            if vec.ndim == 1:
                vec = vec.reshape(1, -1)
            k = min(int(top_k), self._faiss.ntotal)
            dists, ids = self._faiss.search(vec, k)
            return [(int(ids[0][i]), float(dists[0][i]))
                    for i in range(k) if ids[0][i] >= 0]
        except Exception as e:  # noqa: BLE001
            logger.warning("[ReasoningStore] faiss_search failed: %s", e)
            return []

    def get_record(self, reasoning_id: str) -> Optional[dict]:
        """DEREF a reasoning_id → the real record (the verifiable recall)."""
        try:
            row = self._db.execute(
                "SELECT reasoning_id, kind, goal_class, action, oracle_id, verdict, "
                "reward, features_json, b_i, c, time_cost, use_count, anchor_tx, "
                "created_at FROM reasoning_records WHERE reasoning_id = ?",
                [str(reasoning_id)]).fetchone()
            if not row:
                return None
            import json
            return {
                "reasoning_id": row[0], "kind": row[1], "goal_class": row[2],
                "action": row[3], "oracle_id": row[4], "verdict": row[5],
                "reward": float(row[6]) if row[6] is not None else 0.0,
                "features": json.loads(row[7]) if row[7] else [],
                "b_i": row[8], "c": row[9], "time_cost": row[10],
                "use_count": row[11], "anchor_tx": row[12], "created_at": row[13],
            }
        except Exception as e:  # noqa: BLE001
            logger.debug("[ReasoningStore] get_record failed: %s", e)
            return None

    def search(self, query_text: str, k: int = 5) -> list[dict]:
        """SC-search convenience: embed → FAISS knn → DEREF the real records."""
        vec = self._embed_vec(query_text)
        if vec is None:
            return []
        hits = self.faiss_search(vec, top_k=k * 2)
        if not hits:
            return []
        try:
            emb_ids = [h[0] for h in hits]
            placeholders = ",".join("?" for _ in emb_ids)
            rows = self._db.execute(
                f"SELECT reasoning_id, embedding_id FROM reasoning_records "
                f"WHERE embedding_id IN ({placeholders})", emb_ids).fetchall()
            by_emb = {int(r[1]): r[0] for r in rows}
            out = []
            for emb_id, dist in hits:
                rid = by_emb.get(int(emb_id))
                if rid is None:
                    continue
                rec = self.get_record(rid)
                if rec is not None:
                    rec["_match_dist"] = dist
                    out.append(rec)
                if len(out) >= k:
                    break
            return out
        except Exception as e:  # noqa: BLE001
            logger.debug("[ReasoningStore] search deref failed: %s", e)
            return []

    def count(self) -> int:
        try:
            row = self._db.execute("SELECT COUNT(*) FROM reasoning_records").fetchone()
            return int(row[0]) if row else 0
        except Exception:  # noqa: BLE001
            return 0


__all__ = ["ReasoningStore", "EMBEDDING_DIM"]
