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
        snapshot_path: Optional[str] = None,
        clock: Callable[[], float] = time.time,
    ):
        self._db = duckdb_conn
        self._faiss_path = str(faiss_path)
        self._graph = graph                 # core.direct_memory KnowledgeGraph (Kuzu)
        self._embedder = embedder
        self._writer = resolve_writer(writer)
        self._snapshot_path = str(snapshot_path) if snapshot_path else os.path.join(
            os.path.dirname(self._faiss_path) or ".", "reasoning_snapshot.json")
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
                # §7.D-knowledge DK.5 finisher (M6) — the matured research source
                # on a crystallized `research::{gc}` macro ("research via <src>");
                # '' on non-research reasoning. Additive (the anchor_tx/idea_type
                # additive-column precedent).
                "  source        TEXT    DEFAULT '',"
                "  created_at    DOUBLE  NOT NULL"
                ")")
            # §7.B (B.3) — the Maker↔Titan bond scalars (the MakerAssessment Kuzu
            # node carries the graph identity; the rating scalars live here). PK-only.
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS maker_assessments ("
                "  reasoning_id TEXT PRIMARY KEY,"
                "  score        DOUBLE,"
                "  scale        TEXT,"
                "  reward       DOUBLE,"
                "  turn_summary TEXT,"
                "  created_at   DOUBLE NOT NULL"
                ")")
            # §7.D-knowledge DK.5 — the research-path-as-skill recipe (Axis-2).
            # Keyed (goal_class, source); `success_count` reinforces on every
            # successful research-confirm (the recurring source rises). This is the
            # reinforced learning substrate (a) that SURVIVES the volatile data
            # decay — the "how I find X" know-how. A matured recipe crystallizes
            # the (goal_class, action='research') macro (b). `last_used_epoch` is
            # Titan's emergent epoch (not wall-clock). Synthesis-owned, no cross-DB.
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS research_recipes ("
                "  goal_class           TEXT NOT NULL,"
                "  source               TEXT NOT NULL,"
                "  success_count        INTEGER DEFAULT 0,"
                "  last_used_epoch      DOUBLE  DEFAULT 0,"
                # §7.D-knowledge DK.5 finisher (M5) — the success_count at which
                # this recipe last (re-)crystallized its (goal_class,'research')
                # macro; the next re-version fires at crystallized_at_count +
                # macro_compose_min (the recipe keeps GROWING past maturity).
                "  crystallized_at_count INTEGER DEFAULT 0,"
                "  created_at           DOUBLE  NOT NULL,"
                "  PRIMARY KEY (goal_class, source)"
                ")")
            # Additive column migrations for already-existing tables (deployed
            # synthesis.duckdb predates M5/M6). DuckDB ALTER ... ADD is idempotent
            # via IF NOT EXISTS; tolerated best-effort.
            for _table, _col, _decl in (
                ("reasoning_records", "source", "TEXT DEFAULT ''"),
                ("research_recipes", "crystallized_at_count", "INTEGER DEFAULT 0"),
            ):
                try:
                    self._db.execute(
                        f"ALTER TABLE {_table} ADD COLUMN IF NOT EXISTS "
                        f"{_col} {_decl}")
                except Exception as _alter_e:  # noqa: BLE001
                    logger.debug("[ReasoningStore] ALTER %s.%s soft-fail: %s",
                                 _table, _col, _alter_e)
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

    def record_turn(
        self, *, reasoning_id: str, goal_class: str, action: str,
        features: list, signature_text: str,
    ) -> bool:
        """§7.B (C1′) — persist a `Reasoning(kind='turn')` episode for a NON-
        verifiable turn (direct/research/IDK). `reward` is NULL (pending) until the
        turn-judge (B.2) or a user/Maker rating (B.3) scores it; the record is the
        deref-able graphed thought (INV-OML-11). Idempotent on reasoning_id. Soft."""
        return self._write_record(
            reasoning_id=reasoning_id, kind="turn", goal_class=goal_class,
            action=action, oracle_id="", verdict="", reward=None,
            features=features, signature_text=signature_text,
            b_i=1, c=0.0, time_cost=1.0, anchor_tx=reasoning_id, composed_from=None)

    def record_maker_assessment(
        self, *, reasoning_id: str, score: float, scale: str, reward: float,
        turn_summary: str = "",
    ) -> bool:
        """§7.B (B.3) — persist a Maker assessment (the Maker↔Titan bond): DuckDB
        scalars + a `MakerAssessment` Kuzu node under `Self` (SELF_HAS_MAKER_
        ASSESSMENT). Maker-only (ordinary-user feedback is reward-only). Idempotent
        on reasoning_id. Single-writer (INV-Syn-19/28). Soft."""
        if not reasoning_id:
            return False

        def _do() -> bool:
            try:
                self._db.execute(
                    "INSERT INTO maker_assessments (reasoning_id, score, scale, "
                    "reward, turn_summary, created_at) VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT (reasoning_id) DO NOTHING",
                    [str(reasoning_id), float(score), str(scale or ""), float(reward),
                     str(turn_summary or "")[:280], float(self._clock())])
            except Exception as e:  # noqa: BLE001
                logger.warning("[ReasoningStore] maker_assessment insert failed: %s", e)
                return False
            if self._graph is not None:
                try:
                    self._graph.spine_create_maker_assessment_node(
                        reasoning_id=str(reasoning_id), score=float(score),
                        scale=str(scale or ""), reward=float(reward),
                        turn_summary=str(turn_summary or ""),
                        created_at=float(self._clock()))
                    self._graph.spine_link_self_maker_assessment(str(reasoning_id))
                except Exception as e:  # noqa: BLE001
                    logger.debug("[ReasoningStore] maker_assessment kuzu soft-fail: %s", e)
            return True

        try:
            return bool(self._writer.submit_sync(_do))
        except Exception as e:  # noqa: BLE001
            logger.warning("[ReasoningStore] maker_assessment write failed: %s", e)
            return False

    def write_macro(
        self, *, reasoning_id: str, goal_class: str, action: str,
        signature: list, b_i: float, c: float, time_cost: float, use_count: int,
        anchor_tx: str = "", composed_from: Optional[list] = None,
        source: str = "",
    ) -> bool:
        """Persist a distilled `Reasoning(kind='macro_strategy')` (S2). The signature
        is the mean leaf feature vector; `composed_from` = the leaf reasoning_ids.
        `source` (§7.D-knowledge DK.5 / M6) = the matured research source on a
        `research::{gc}` macro ("research via <src>"), '' on a strategy macro."""
        ok = self._write_record(
            reasoning_id=reasoning_id, kind="macro_strategy", goal_class=goal_class,
            action=action, oracle_id="", verdict="true", reward=1.0,
            features=list(signature or []), signature_text=str(goal_class or ""),
            b_i=b_i, c=c, time_cost=time_cost, use_count=int(use_count),
            anchor_tx=anchor_tx, composed_from=composed_from,
            idea_type="procedural",   # §7.D D.3 / FC-8 — a composite IS procedural Idea
            source=source)
        if ok:
            self.macros_written += 1
        return ok

    def record_research_recipe(
        self, *, goal_class: str, source: str, epoch: float, mature_at: int = 2,
        macro_compose_min: int = 2,
    ) -> tuple[int, str]:
        """§7.D-knowledge DK.5 (a) + finisher (M5) — reinforce the research recipe
        for a successful (goal_class, source). Upsert: `success_count += 1`,
        `last_used_epoch = epoch`. The recipe is the LIVING reinforced skill — it
        keeps growing past maturity, surviving the volatile data's decay.

        Returns `(success_count, crystallize_signal)`:
          • ``"initial"`` — exactly on the bump that first REACHES `mature_at`
            (crystallized_at_count was 0) → the caller crystallizes the base
            `research::{gc}` macro once. Sets `crystallized_at_count`.
          • ``"reversion"`` (M5) — on a later bump where `success_count ≥
            crystallized_at_count + macro_compose_min` → the caller mints the
            SUCCESSOR `research::{gc}::v{n+1}` (mutate-not-update lineage). Re-bumps
            `crystallized_at_count`.
          • ``""`` — no crystallize event this bump.
        Soft → (0, "")."""
        gc = str(goal_class or "").strip()
        src = str(source or "").strip()
        if not gc or not src:
            return (0, "")

        def _do() -> tuple[int, str]:
            self._db.execute(
                "INSERT INTO research_recipes "
                "(goal_class, source, success_count, last_used_epoch, "
                " crystallized_at_count, created_at) "
                "VALUES (?,?,1,?,0,?) "
                "ON CONFLICT (goal_class, source) DO UPDATE SET "
                "  success_count = research_recipes.success_count + 1, "
                "  last_used_epoch = excluded.last_used_epoch",
                [gc, src, float(epoch), float(self._clock())])
            row = self._db.execute(
                "SELECT success_count, crystallized_at_count FROM research_recipes "
                "WHERE goal_class=? AND source=?", [gc, src]).fetchone()
            cnt = int(row[0]) if row else 0
            crystallized_at = int(row[1]) if row and row[1] is not None else 0
            signal = ""
            if crystallized_at == 0 and cnt >= int(mature_at):
                signal = "initial"
            elif (crystallized_at > 0
                  and cnt >= crystallized_at + int(macro_compose_min)):
                signal = "reversion"
            if signal:
                self._db.execute(
                    "UPDATE research_recipes SET crystallized_at_count = ? "
                    "WHERE goal_class=? AND source=?", [cnt, gc, src])
            return (cnt, signal)

        try:
            return self._writer.submit_sync(_do)
        except Exception as e:  # noqa: BLE001
            logger.debug("[ReasoningStore] record_research_recipe soft-fail: %s", e)
            return (0, "")

    def research_macro_lineage(self, goal_class: str) -> tuple[str, int]:
        """§7.D-knowledge DK.5 finisher (M5) — the research-macro version chain for
        `goal_class`. Returns `(prior_label, next_version)`:
          • no macro yet           → ``("", 1)``         (mint base `research::{gc}`)
          • only base exists       → ``("research::{gc}", 2)``
          • `…::v{k}` is the max    → ``("research::{gc}::v{k}", k+1)``
        The successor's `composed_from=[prior_label]` is the D.4c mutate-not-update
        lineage. Soft → ("", 1)."""
        gc = str(goal_class or "").strip()
        if not gc:
            return ("", 1)
        base = f"research::{gc}"

        def _do() -> tuple[str, int]:
            rows = self._db.execute(
                "SELECT reasoning_id FROM reasoning_records "
                "WHERE kind='macro_strategy' AND reasoning_id LIKE ?",
                [base + "%"]).fetchall()
            ids = {str(r[0]) for r in (rows or [])}
            if base not in ids:
                return ("", 1)
            max_v = 1
            prior = base
            for rid in ids:
                if rid == base:
                    continue
                suffix = rid[len(base):]
                m = suffix.startswith("::v") and suffix[3:].isdigit()
                if m:
                    v = int(suffix[3:])
                    if v > max_v:
                        max_v = v
                        prior = rid
            return (prior, max_v + 1)

        try:
            return self._writer.submit_sync(_do)
        except Exception as e:  # noqa: BLE001
            logger.debug("[ReasoningStore] research_macro_lineage soft-fail: %s", e)
            return ("", 1)

    def _write_record(
        self, *, reasoning_id, kind, goal_class, action, oracle_id, verdict,
        reward, features, signature_text, b_i, c, time_cost, use_count=1,
        anchor_tx="", composed_from=None, idea_type="", source="",
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
                    "time_cost, use_count, embedding_id, anchor_tx, source, "
                    "created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT (reasoning_id) DO NOTHING",
                    [str(reasoning_id), str(kind), str(goal_class or ""),
                     str(action or ""), str(oracle_id or ""), str(verdict or ""),
                     (None if reward is None else float(reward)),  # §7.B turn = NULL (pending)
                     json.dumps(list(features or [])),
                     int(b_i), float(c), float(time_cost), int(use_count),
                     int(emb_id), str(anchor_tx or ""), str(source or ""),
                     float(self._clock())])
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
                        anchor_tx=str(anchor_tx or ""), created_at=float(self._clock()),
                        idea_type=str(idea_type or ""), source=str(source or ""))
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
            if ok:
                self.snapshot_export()   # bounded payload — readable past the writer-lock
            return ok
        except Exception as e:  # noqa: BLE001
            logger.warning("[ReasoningStore] record write failed: %s", e)
            return False

    # ── §7.D D.1 — the macro→leaf provenance join (synthesis-side; the worker
    # has no leaf reasoning_ids — they live HERE). Verified tool_use leaves of a
    # (goal_class, action) become the macro's REASONING_COMPOSED_FROM evidence.
    def leaf_reasoning_ids(
        self, goal_class: str, action: Optional[str] = None, limit: int = 16,
    ) -> list[str]:
        """Return the reasoning_ids of VERIFIED (reward>0) `tool_use` leaves for a
        (goal_class[, action]), most-recent first — the composite's evidence."""
        if not goal_class:
            return []
        try:
            params: list = [str(goal_class)]
            sql = ("SELECT reasoning_id FROM reasoning_records "
                   "WHERE kind='tool_use' AND goal_class=? AND reward>0")
            if action:
                sql += " AND action=?"
                params.append(str(action))
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(int(limit))
            rows = self._db.execute(sql, params).fetchall()
            return [str(r[0]) for r in rows if r and r[0]]
        except Exception as e:  # noqa: BLE001
            logger.debug("[ReasoningStore] leaf_reasoning_ids failed: %s", e)
            return []

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

    # ── Snapshot (atomic JSON; the api/analysis reads this past the writer-lock,
    # mirroring skills_snapshot.json — INV-Syn-8: readers never open the db) ──
    def _build_snapshot_payload(self) -> dict:
        import json
        try:
            total = self.count()
            by_kind = self._db.execute(
                "SELECT kind, COUNT(*) FROM reasoning_records GROUP BY kind").fetchall()
            by_goal = self._db.execute(
                "SELECT goal_class, COUNT(*), SUM(CASE WHEN reward>0 THEN 1 ELSE 0 END) "
                "FROM reasoning_records GROUP BY goal_class ORDER BY 2 DESC LIMIT 30"
            ).fetchall()
            recent = self._db.execute(
                "SELECT reasoning_id, kind, goal_class, action, oracle_id, verdict, "
                "reward, created_at FROM reasoning_records ORDER BY created_at DESC "
                "LIMIT 50").fetchall()
            # Phase-C piece 7b — the macro retrieval-prior map. The agno hot-path
            # decide reads this lock-free snapshot (DuckDB holds the exclusive lock
            # even read_only — the SPEC-canonical cross-process pattern) + the faiss
            # FILE (read-only-safe) to SC-search the prompt against macro composites.
            # embedding_id = the faiss position (set at write time); action = the
            # routed action; only macro_strategy rows with a real embedding.
            # §7.D D.4a — `reasoning_id` added so the worker's OuterCompositeReader
            # can NAME a matched composite (macro-of-macros provenance + reuse) and
            # the agno reader can deref it; `use_count` ranks library entries.
            macros = self._db.execute(
                "SELECT embedding_id, action, goal_class, reasoning_id, use_count "
                "FROM reasoning_records "
                "WHERE kind='macro_strategy' AND embedding_id >= 0").fetchall()
            # §24.12 Track 2 — the EMERGENT retrieval prior. The composite reader
            # should match a prompt not only against the rare hand-distilled
            # macro_strategy composites, but against Titan's OWN oracle-VERIFIED
            # tool_use experience (kind='tool_use', reward>0 = a verified win) —
            # the self-learning, oracle-scored tool-intent the Maker envisioned,
            # built from data that already accumulates at verdict-time. Bounded +
            # fresh (recent wins) so the matchable set stays small as records grow;
            # the reader reward-weights the match. macro_strategy still takes
            # precedence (it is the refined distillate).
            verified = self._db.execute(
                "SELECT embedding_id, action, goal_class, reward FROM reasoning_records "
                "WHERE kind='tool_use' AND embedding_id >= 0 AND reward > 0 "
                "ORDER BY created_at DESC LIMIT 256").fetchall()
            return {
                "version": 1, "ts": float(self._clock()), "count": int(total),
                "records_written": int(self.records_written),
                "macros_written": int(self.macros_written),
                "by_kind": {str(k): int(n) for k, n in by_kind},
                "by_goal_class": [
                    {"goal_class": g, "count": int(n), "wins": int(w or 0)}
                    for g, n, w in by_goal],
                "recent": [
                    {"reasoning_id": r[0], "kind": r[1], "goal_class": r[2],
                     "action": r[3], "oracle_id": r[4], "verdict": r[5],
                     "reward": float(r[6]) if r[6] is not None else 0.0,
                     "created_at": r[7]}
                    for r in recent],
                "macros": [
                    {"embedding_id": int(m[0]), "action": str(m[1] or ""),
                     "goal_class": str(m[2] or ""), "reasoning_id": str(m[3] or ""),
                     "use_count": int(m[4] or 1)}
                    for m in macros],
                "verified_priors": [
                    {"embedding_id": int(v[0]), "action": str(v[1] or ""),
                     "goal_class": str(v[2] or ""),
                     "reward": float(v[3]) if v[3] is not None else 0.0}
                    for v in verified],
            }
        except Exception as e:  # noqa: BLE001
            logger.debug("[ReasoningStore] snapshot payload failed: %s", e)
            return {"version": 1, "ts": float(self._clock()), "count": 0,
                    "by_kind": {}, "by_goal_class": [], "recent": []}

    def snapshot_export(self) -> str:
        """Atomically write the read-snapshot (on the writer thread)."""
        import json

        def _export() -> str:
            payload = self._build_snapshot_payload()
            try:
                os.makedirs(os.path.dirname(self._snapshot_path) or ".", exist_ok=True)
                tmp = self._snapshot_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
                os.replace(tmp, self._snapshot_path)
            except Exception as e:  # noqa: BLE001
                logger.debug("[ReasoningStore] snapshot write failed: %s", e)
            return self._snapshot_path
        try:
            return str(self._writer.submit_sync(_export))
        except Exception:  # noqa: BLE001
            return self._snapshot_path


__all__ = ["ReasoningStore", "EMBEDDING_DIM"]
