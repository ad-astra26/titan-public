"""ProceduralSkillStore — sole writer of `procedural_skills` + `skills_vectors.faiss`.

INV-Syn-19 (Phase 8, D-SPEC-PHASE8): synthesis_worker is the only process that
writes to `procedural_skills` on `synthesis.duckdb` AND to the companion FAISS
index at `data/skills_vectors.faiss`. Cross-process consumers read
`data/skills_snapshot.json` (atomic tmp+rename, mirrors P4 spine_snapshot /
P5 forks_snapshot / P7 buffers_snapshot).

Schema (matches SPEC §25.7 + arch §8.1):

    CREATE TABLE procedural_skills (
        skill_id        TEXT    PRIMARY KEY,
        name            TEXT    NOT NULL,
        nl_description  TEXT    NOT NULL,
        embedding_id    INTEGER,                     -- FAISS row id (-1 = not embedded yet)
        executable_spec TEXT    NOT NULL,            -- canonical JSON
        preconditions   TEXT,                        -- canonical JSON list
        postconditions  TEXT,                        -- canonical JSON list
        compiled_from   TEXT    NOT NULL,            -- canonical JSON list of tx_hashes
        success_count   INTEGER DEFAULT 0,
        failure_count   INTEGER DEFAULT 0,
        last_used       DOUBLE,
        created_at      DOUBLE  NOT NULL,
        utility_score   DOUBLE  DEFAULT 0.7,         -- P8 ship default per rFP §11.4
        verified_at     DOUBLE                       -- NULL until INV-Syn-20 first-invocation re-verify
    );

INV-Syn-20 first-invocation re-verification is owned by the companion
SkillVerifier (P8.F) — this store only exposes `mark_verified()` /
`mark_rejected()` mutators.

Skills are NEVER deleted (Q4 / arch §8.4 "Skills never deleted; soft-retire
via utility decay"). `utility_score` below `soft_retire_floor` (default -0.5)
triggers a META_SKILL_SOFT_RETIRED log + bus event but leaves the row intact —
Timechain provenance is preserved end-to-end.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Embedding model dimensionality (BAAI/bge-small-en-v1.5 — same as P4 spine
# concepts so cross-substrate analogy queries stay shape-compatible).
EMBEDDING_DIM: int = 384

# Soft default per rFP §11.4 / arch §8.1: freshly-compiled skills ship at 0.7
# utility — high enough to be matchable, low enough that a couple of failures
# drop them below the delegate gate.
DEFAULT_UTILITY_SCORE: float = 0.7

# Per arch §23 added knobs (D-SPEC-PHASE8). Each tick of success/failure
# moves utility_score by this delta (clamped to [-1.0, 1.0]).
UTILITY_DELTA: float = 0.05

# Utility floor below which the skill is considered soft-retired and emits a
# META_SKILL_SOFT_RETIRED log. The row stays in the table per Q4.
DEFAULT_SOFT_RETIRE_FLOOR: float = -0.5


class _DuckDBConnLike:
    """Structural protocol for the DuckDB connection (tests inject fakes)."""
    def execute(self, sql: str, params: Any = None) -> Any: ...
    def fetchall(self) -> list: ...


class ProceduralSkillStore:
    """Sole writer of `procedural_skills` + `skills_vectors.faiss`. INV-Syn-19.

    Constructor params:
      duckdb_conn: open DuckDB connection to `synthesis.duckdb` (sole-writer).
      faiss_path: path to `data/skills_vectors.faiss` (created lazily; atomic
                  tmp+rename on every persist that embeds).
      snapshot_path: path to `data/skills_snapshot.json` (atomic tmp+rename
                     after every persist / utility update).
      embedder: optional callable `(text: str) -> np.ndarray` returning a
                384D L2-normalized embedding. When None, persist_skill stores
                the row with embedding_id=-1 and skips the FAISS write —
                the dream-time mining pass can re-embed later.
      clock: time source (overridable for tests).
      soft_retire_floor: utility_score below which META_SKILL_SOFT_RETIRED
                         fires (callback `on_soft_retire(skill_id, utility)`).
      on_soft_retire: optional callback invoked when utility crosses the floor.
                      Bus emission is wired here by the worker so the store
                      itself stays free of bus imports.
    """

    def __init__(
        self,
        *,
        duckdb_conn: _DuckDBConnLike,
        faiss_path: str | os.PathLike,
        snapshot_path: str | os.PathLike,
        embedder: Optional[Any] = None,
        clock: Any = time.time,
        soft_retire_floor: float = DEFAULT_SOFT_RETIRE_FLOOR,
        on_soft_retire: Optional[Any] = None,
    ):
        self._db = duckdb_conn
        self._faiss_path = str(faiss_path)
        self._snapshot_path = str(snapshot_path)
        self._embedder = embedder
        self._clock = clock
        self._soft_retire_floor = float(soft_retire_floor)
        self._on_soft_retire = on_soft_retire
        # FAISS index is loaded lazily on first embedding call.
        self._faiss = None
        self._faiss_dim = EMBEDDING_DIM
        # Single lock guards every DuckDB + FAISS mutation. Sole-writer
        # contract is preserved by the synthesis_worker process boundary;
        # the lock makes write/snapshot-export atomic against any in-process
        # reader (recompute_loop reading counters while recv_loop persists).
        self._lock = threading.Lock()
        # Counters for /v6/synthesis/skills/coverage + Observatory readouts.
        self._persists_seen = 0
        self._utility_updates = 0
        self._verifications_seen = 0
        self._rejections_seen = 0
        self._soft_retires_seen = 0
        self._init_schema()

    # ── Schema bootstrap ────────────────────────────────────────────────

    def _init_schema(self) -> None:
        """CREATE TABLE IF NOT EXISTS procedural_skills (idempotent)."""
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS procedural_skills ("
                "  skill_id        TEXT    PRIMARY KEY,"
                "  name            TEXT    NOT NULL,"
                "  nl_description  TEXT    NOT NULL,"
                "  embedding_id    INTEGER,"
                "  executable_spec TEXT    NOT NULL,"
                "  preconditions   TEXT,"
                "  postconditions  TEXT,"
                "  compiled_from   TEXT    NOT NULL,"
                "  success_count   INTEGER DEFAULT 0,"
                "  failure_count   INTEGER DEFAULT 0,"
                "  last_used       DOUBLE,"
                "  created_at      DOUBLE  NOT NULL,"
                f"  utility_score   DOUBLE  DEFAULT {DEFAULT_UTILITY_SCORE},"
                "  verified_at     DOUBLE"
                ")"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_procedural_skills_utility "
                "ON procedural_skills(utility_score DESC)"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_procedural_skills_last_used "
                "ON procedural_skills(last_used DESC)"
            )

    # ── FAISS lifecycle (lazy load / save) ──────────────────────────────

    def _ensure_faiss(self) -> None:
        """Open or create the FAISS index on first use."""
        if self._faiss is not None:
            return
        import faiss  # local import — keeps cold-boot RSS down per Phase 11 D-SPEC-138 lesson
        if os.path.exists(self._faiss_path):
            try:
                self._faiss = faiss.read_index(self._faiss_path)
                if self._faiss.d != self._faiss_dim:
                    logger.warning(
                        "[ProceduralSkillStore] FAISS index at %s has dim=%d, expected %d "
                        "— resetting empty index",
                        self._faiss_path, self._faiss.d, self._faiss_dim,
                    )
                    self._faiss = faiss.IndexFlatL2(self._faiss_dim)
            except Exception as e:  # pragma: no cover — defensive
                logger.warning(
                    "[ProceduralSkillStore] failed to load FAISS index (%s) — "
                    "starting empty: %s", self._faiss_path, e,
                )
                self._faiss = faiss.IndexFlatL2(self._faiss_dim)
        else:
            self._faiss = faiss.IndexFlatL2(self._faiss_dim)

    def _save_faiss(self) -> None:
        """Atomic tmp+rename save of the FAISS index."""
        if self._faiss is None:
            return
        import faiss
        try:
            os.makedirs(os.path.dirname(self._faiss_path) or ".", exist_ok=True)
            tmp_path = self._faiss_path + ".tmp"
            faiss.write_index(self._faiss, tmp_path)
            os.replace(tmp_path, self._faiss_path)
        except Exception as e:
            logger.warning(
                "[ProceduralSkillStore] FAISS save failed (%s): %s",
                self._faiss_path, e,
            )

    def _embed_and_add(self, text: str) -> int:
        """Embed `text` + add to FAISS. Returns the new row id (FAISS ntotal-1).
        Returns -1 if no embedder is configured OR the embed call fails."""
        if self._embedder is None or not text:
            return -1
        try:
            import numpy as np
            vec = self._embedder(text)
            if vec is None:
                return -1
            vec = np.asarray(vec, dtype=np.float32)
            if vec.ndim == 1:
                vec = vec.reshape(1, -1)
            if vec.shape[1] != self._faiss_dim:
                logger.warning(
                    "[ProceduralSkillStore] embedder returned dim=%d, expected %d — skipping FAISS add",
                    vec.shape[1], self._faiss_dim,
                )
                return -1
            self._ensure_faiss()
            row_id = self._faiss.ntotal
            self._faiss.add(vec)
            self._save_faiss()
            return row_id
        except Exception as e:
            logger.warning("[ProceduralSkillStore] embed_and_add failed: %s", e)
            return -1

    # ── Write surface (INV-Syn-19 sole-writer) ──────────────────────────

    def persist_skill(
        self,
        *,
        skill_id: str,
        name: str,
        nl_description: str,
        executable_spec: dict,
        preconditions: Optional[list] = None,
        postconditions: Optional[list] = None,
        compiled_from: list[str],
        utility_score: float = DEFAULT_UTILITY_SCORE,
        ts: Optional[float] = None,
    ) -> int:
        """INSERT OR REPLACE one skill row + (best-effort) embed into FAISS +
        snapshot export. Returns the FAISS embedding_id (-1 if unembedded).

        Raises ValueError on empty skill_id / missing required fields — caller
        bugs surface loudly. Idempotent: re-persisting the same skill_id keeps
        success_count/failure_count/verified_at (the lifecycle counters move
        through `increment_success` / `increment_failure` / `mark_*` only)."""
        if not skill_id:
            raise ValueError("skill_id must be non-empty")
        if not name or not nl_description:
            raise ValueError("name and nl_description required")
        if not compiled_from:
            raise ValueError("compiled_from must be non-empty (lineage gate)")
        ts_val = float(ts) if ts is not None else float(self._clock())
        spec_json = json.dumps(executable_spec or {}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        pre_json = json.dumps(list(preconditions or []), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        post_json = json.dumps(list(postconditions or []), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        cf_json = json.dumps(list(compiled_from), ensure_ascii=False, separators=(",", ":"))
        utility = max(-1.0, min(1.0, float(utility_score)))
        with self._lock:
            # Check if skill exists — preserve counters on re-persist
            existing = self._db.execute(
                "SELECT embedding_id, success_count, failure_count, verified_at, created_at "
                "FROM procedural_skills WHERE skill_id = ?",
                [skill_id],
            ).fetchall()
            if existing:
                emb_id, succ, fail, ver_at, created_at = existing[0]
                success_count = int(succ or 0)
                failure_count = int(fail or 0)
                verified_at = ver_at  # preserve None or timestamp
                created_at_val = float(created_at) if created_at is not None else ts_val
                # Skip re-embedding if already embedded (FAISS is append-only;
                # re-embedding would create a stale row id).
                embedding_id = int(emb_id) if emb_id is not None and int(emb_id) >= 0 else -1
            else:
                success_count = 0
                failure_count = 0
                verified_at = None
                created_at_val = ts_val
                embedding_id = -1
        # Embed AFTER releasing the lock (FAISS+embedder calls can be slow).
        if embedding_id < 0:
            embedding_id = self._embed_and_add(nl_description)
        with self._lock:
            # True in-place UPSERT (NOT `INSERT OR REPLACE`). procedural_skills
            # has two secondary indexes (idx_procedural_skills_utility /
            # _last_used); DuckDB's OR REPLACE is DELETE+INSERT, which can corrupt
            # those ART indexes → a later commit aborts the synthesis_worker with a
            # FATAL "duplicate key" (same class of crash as actr_buffers, 2026-06-01).
            # ON CONFLICT DO UPDATE mutates in place; behaviour matches OR REPLACE
            # (a re-persist of an existing skill_id overwrites every column).
            self._db.execute(
                "INSERT INTO procedural_skills "
                "(skill_id, name, nl_description, embedding_id, executable_spec, "
                " preconditions, postconditions, compiled_from, success_count, "
                " failure_count, last_used, created_at, utility_score, verified_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (skill_id) DO UPDATE SET "
                "name = excluded.name, nl_description = excluded.nl_description, "
                "embedding_id = excluded.embedding_id, "
                "executable_spec = excluded.executable_spec, "
                "preconditions = excluded.preconditions, "
                "postconditions = excluded.postconditions, "
                "compiled_from = excluded.compiled_from, "
                "success_count = excluded.success_count, "
                "failure_count = excluded.failure_count, "
                "last_used = excluded.last_used, created_at = excluded.created_at, "
                "utility_score = excluded.utility_score, "
                "verified_at = excluded.verified_at",
                [
                    skill_id, name, nl_description, embedding_id, spec_json,
                    pre_json, post_json, cf_json,
                    success_count, failure_count, None, created_at_val,
                    utility, verified_at,
                ],
            )
            self._persists_seen += 1
        self.snapshot_export()
        return embedding_id

    def increment_success(self, skill_id: str) -> None:
        """utility_score += UTILITY_DELTA (clamped); success_count += 1; last_used=now."""
        self._adjust_utility(skill_id, success=True)

    def increment_failure(self, skill_id: str) -> None:
        """utility_score -= UTILITY_DELTA (clamped); failure_count += 1; last_used=now."""
        self._adjust_utility(skill_id, success=False)

    def apply_utility_delta(self, skill_id: str, delta: float) -> Optional[float]:
        """Apply an arbitrary utility delta (clamped to [-1,1]) WITHOUT bumping
        success/failure counts. Phase 9 / INV-Syn-24: the Tier-2 user-feedback
        override uses a larger delta (default 0.15) than the 0.05 invocation
        delta — and it is an override signal, not an invocation outcome, so the
        counts stay untouched. Returns the new utility (or None for an unknown
        skill_id). Snapshot-exported."""
        if not skill_id:
            raise ValueError("skill_id must be non-empty")
        with self._lock:
            rows = self._db.execute(
                "SELECT utility_score FROM procedural_skills WHERE skill_id = ?",
                [skill_id],
            ).fetchall()
            if not rows:
                logger.warning(
                    "[ProceduralSkillStore] apply_utility_delta for unknown "
                    "skill_id=%s", skill_id,
                )
                return None
            current = float(rows[0][0] if rows[0][0] is not None else DEFAULT_UTILITY_SCORE)
            new_utility = max(-1.0, min(1.0, current + float(delta)))
            self._db.execute(
                "UPDATE procedural_skills SET utility_score = ? WHERE skill_id = ?",
                [new_utility, skill_id],
            )
            self._utility_updates += 1
        self.snapshot_export()
        return new_utility

    def _adjust_utility(self, skill_id: str, *, success: bool) -> None:
        if not skill_id:
            raise ValueError("skill_id must be non-empty")
        now = float(self._clock())
        delta = UTILITY_DELTA if success else -UTILITY_DELTA
        with self._lock:
            rows = self._db.execute(
                "SELECT utility_score FROM procedural_skills WHERE skill_id = ?",
                [skill_id],
            ).fetchall()
            if not rows:
                logger.warning(
                    "[ProceduralSkillStore] %s_count called for unknown skill_id=%s",
                    "success" if success else "failure", skill_id,
                )
                return
            current = float(rows[0][0] if rows[0][0] is not None else DEFAULT_UTILITY_SCORE)
            new_utility = max(-1.0, min(1.0, current + delta))
            if success:
                self._db.execute(
                    "UPDATE procedural_skills SET success_count = success_count + 1, "
                    "last_used = ?, utility_score = ? WHERE skill_id = ?",
                    [now, new_utility, skill_id],
                )
            else:
                self._db.execute(
                    "UPDATE procedural_skills SET failure_count = failure_count + 1, "
                    "last_used = ?, utility_score = ? WHERE skill_id = ?",
                    [now, new_utility, skill_id],
                )
            self._utility_updates += 1
            crossed_floor = (
                current > self._soft_retire_floor
                and new_utility <= self._soft_retire_floor
            )
            if crossed_floor:
                self._soft_retires_seen += 1
        if crossed_floor:
            logger.info(
                "[ProceduralSkillStore] skill %s soft-retired (utility=%.3f ≤ floor=%.3f)",
                skill_id, new_utility, self._soft_retire_floor,
            )
            if self._on_soft_retire is not None:
                try:
                    self._on_soft_retire(skill_id, new_utility)
                except Exception as e:  # pragma: no cover — defensive
                    logger.warning(
                        "[ProceduralSkillStore] on_soft_retire callback raised: %s", e,
                    )
        self.snapshot_export()

    def mark_verified(self, skill_id: str) -> None:
        """INV-Syn-20: skill passed first-invocation chain-resolve + content-hash check."""
        now = float(self._clock())
        with self._lock:
            self._db.execute(
                "UPDATE procedural_skills SET verified_at = ? WHERE skill_id = ?",
                [now, skill_id],
            )
            self._verifications_seen += 1
        self.snapshot_export()

    def mark_rejected(self, skill_id: str, reason: str) -> None:
        """INV-Syn-20: skill failed first-invocation re-verification.
        Sets utility_score=-1.0 (hard ineligible) + verified_at=now (so the
        skill never re-enters the verifier loop). reason is logged + carried
        on the META_SKILL_REJECTED bus event by the worker."""
        now = float(self._clock())
        with self._lock:
            self._db.execute(
                "UPDATE procedural_skills SET utility_score = ?, verified_at = ? "
                "WHERE skill_id = ?",
                [-1.0, now, skill_id],
            )
            self._rejections_seen += 1
        logger.info(
            "[ProceduralSkillStore] skill %s rejected (reason=%s)", skill_id, reason,
        )
        self.snapshot_export()

    # ── Read surface ────────────────────────────────────────────────────

    def read_skill(self, skill_id: str) -> Optional[dict]:
        """Returns the full skill row as a dict, or None if missing."""
        if not skill_id:
            return None
        with self._lock:
            rows = self._db.execute(
                "SELECT skill_id, name, nl_description, embedding_id, executable_spec, "
                "preconditions, postconditions, compiled_from, success_count, "
                "failure_count, last_used, created_at, utility_score, verified_at "
                "FROM procedural_skills WHERE skill_id = ?",
                [skill_id],
            ).fetchall()
        if not rows:
            return None
        return self._row_to_dict(rows[0])

    def read_for_match(
        self, *, utility_floor: float = 0.3, k: int = 5, verified_only: bool = True,
    ) -> list[dict]:
        """Return skills eligible for the agno match path:
        utility_score ≥ utility_floor, optionally verified_only.

        Caller (ProceduralSkillReader) will FAISS-rank the result + apply
        match_floor. This is the broad pre-filter."""
        with self._lock:
            if verified_only:
                rows = self._db.execute(
                    "SELECT skill_id, name, nl_description, embedding_id, executable_spec, "
                    "preconditions, postconditions, compiled_from, success_count, "
                    "failure_count, last_used, created_at, utility_score, verified_at "
                    "FROM procedural_skills "
                    "WHERE utility_score >= ? AND verified_at IS NOT NULL "
                    "ORDER BY utility_score DESC LIMIT ?",
                    [utility_floor, max(1, int(k) * 4)],
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT skill_id, name, nl_description, embedding_id, executable_spec, "
                    "preconditions, postconditions, compiled_from, success_count, "
                    "failure_count, last_used, created_at, utility_score, verified_at "
                    "FROM procedural_skills "
                    "WHERE utility_score >= ? "
                    "ORDER BY utility_score DESC LIMIT ?",
                    [utility_floor, max(1, int(k) * 4)],
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_all(self, *, limit: int = 100) -> list[dict]:
        """For Observatory list_skills route. Ordered by utility_score DESC."""
        with self._lock:
            rows = self._db.execute(
                "SELECT skill_id, name, nl_description, embedding_id, executable_spec, "
                "preconditions, postconditions, compiled_from, success_count, "
                "failure_count, last_used, created_at, utility_score, verified_at "
                "FROM procedural_skills "
                "ORDER BY utility_score DESC, created_at DESC LIMIT ?",
                [max(1, int(limit))],
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def faiss_search(self, query_vec: Any, top_k: int = 20) -> list[tuple[int, float]]:
        """Returns [(embedding_id, distance), ...] for the top-k FAISS hits.
        Empty list when no FAISS index exists or it's empty."""
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
            return [(int(ids[0][i]), float(dists[0][i])) for i in range(k) if ids[0][i] >= 0]
        except Exception as e:
            logger.warning("[ProceduralSkillStore] faiss_search failed: %s", e)
            return []

    def embed_query(self, text: str) -> Optional[Any]:
        """Embed a query string for FAISS search. Returns None if no embedder."""
        if self._embedder is None or not text:
            return None
        try:
            return self._embedder(text)
        except Exception as e:
            logger.warning("[ProceduralSkillStore] embed_query failed: %s", e)
            return None

    def _row_to_dict(self, row: tuple) -> dict:
        skill_id, name, nl, emb_id, spec_json, pre_json, post_json, cf_json, succ, fail, last_used, created_at, utility, verified_at = row
        return {
            "skill_id": skill_id,
            "name": name,
            "nl_description": nl,
            "embedding_id": int(emb_id) if emb_id is not None else -1,
            "executable_spec": _safe_json_load(spec_json, {}),
            "preconditions": _safe_json_load(pre_json, []),
            "postconditions": _safe_json_load(post_json, []),
            "compiled_from": _safe_json_load(cf_json, []),
            "success_count": int(succ or 0),
            "failure_count": int(fail or 0),
            "last_used": float(last_used) if last_used is not None else None,
            "created_at": float(created_at) if created_at is not None else 0.0,
            "utility_score": float(utility) if utility is not None else DEFAULT_UTILITY_SCORE,
            "verified_at": float(verified_at) if verified_at is not None else None,
        }

    # ── Snapshot export ────────────────────────────────────────────────

    def _build_snapshot_payload(self) -> dict:
        with self._lock:
            rows = self._db.execute(
                "SELECT skill_id, name, nl_description, success_count, failure_count, "
                "last_used, created_at, utility_score, verified_at "
                "FROM procedural_skills "
                "ORDER BY utility_score DESC, created_at DESC"
            ).fetchall()
            persists = self._persists_seen
            updates = self._utility_updates
            verifications = self._verifications_seen
            rejections = self._rejections_seen
            soft_retires = self._soft_retires_seen
        skills = [
            {
                "skill_id": r[0],
                "name": r[1],
                "nl_description": r[2],
                "success_count": int(r[3] or 0),
                "failure_count": int(r[4] or 0),
                "last_used": float(r[5]) if r[5] is not None else None,
                "created_at": float(r[6]) if r[6] is not None else 0.0,
                "utility_score": float(r[7]) if r[7] is not None else DEFAULT_UTILITY_SCORE,
                "verified_at": float(r[8]) if r[8] is not None else None,
            }
            for r in rows
        ]
        return {
            "version": 1,
            "ts": float(self._clock()),
            "count": len(skills),
            "persists_seen": persists,
            "utility_updates": updates,
            "verifications_seen": verifications,
            "rejections_seen": rejections,
            "soft_retires_seen": soft_retires,
            "skills": skills,
        }

    def snapshot_export(self) -> str:
        """Atomic tmp+rename write of `skills_snapshot.json`.

        Soft-fails on filesystem errors (logs WARN; returns intended path)
        so persist surfaces never raise on transient disk issues."""
        payload = self._build_snapshot_payload()
        target = self._snapshot_path
        try:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            tmp_path = target + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, target)
        except Exception as e:
            logger.warning(
                "[ProceduralSkillStore] snapshot export failed (%s): %s", target, e,
            )
        return target

    # ── Observability ───────────────────────────────────────────────────

    def stats(self) -> dict:
        """Counters surfaced via Observatory + the A.4/A.6 acceptance gates."""
        with self._lock:
            faiss_count = self._faiss.ntotal if self._faiss is not None else 0
            return {
                "persists_seen": self._persists_seen,
                "utility_updates": self._utility_updates,
                "verifications_seen": self._verifications_seen,
                "rejections_seen": self._rejections_seen,
                "soft_retires_seen": self._soft_retires_seen,
                "faiss_count": int(faiss_count),
                "faiss_path": self._faiss_path,
                "snapshot_path": self._snapshot_path,
            }


def _safe_json_load(raw: Optional[str], default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default
