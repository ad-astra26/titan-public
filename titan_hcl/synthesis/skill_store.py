"""ProceduralSkillStore — sole writer of `procedural_skills` + `skill_cells` +
`skill_score_events` + `skills_vectors.faiss`.

INV-Syn-19 (Phase 8, D-SPEC-PHASE8): synthesis_worker is the only process that
writes the procedural-skill DuckDB tables on `synthesis.duckdb` AND the companion
FAISS index at `data/skills_vectors.faiss`. Cross-process consumers read
`data/skills_snapshot.json` (atomic tmp+rename).

EEL Pillar B1 (2026-06-09 / D-SPEC-153 / INV-Syn-29): a skill is keyed on its
**outcome** `(oracle_id, goal_class)` and is a *policy over the task-shapes* that
reach it. `skill_id` stays the canonical cross-substrate PK (this table + the
Kuzu `Production` mirror, arch §3.6 + `parent_skill_id` provenance + INV-Syn-20),
**re-derived** from the outcome: `skill_id = "skill_"+sha256(oracle_id|goal_class)[:16]`.

Schema (matches SPEC §25.1 INV-Syn-29 + arch §8.1 v0.29.0):

    procedural_skills   — the OUTCOME / skill row (one per (oracle_id, goal_class));
                          skill_id PK, UNIQUE(oracle_id, goal_class); FAISS embedding per-OUTCOME.
    skill_cells         — per task-shape CELL carrying the §3.4 triple {b_i, c, time_cost}
                          (INDEPENDENT — INV-EEL-9), + per-cell recipe + polarity (INV-EEL-5).
    skill_score_events  — the off-hot-path capture queue (Option C): appended at verdict-time
                          (synthesis sole-writer), drained by the resource-gated 60s tick → cells.

Scoring: per oracle-verified tool-use (INV-EEL-2; never an LLM self-claim), single
successes count. Proficiency `time_cost` EMERGES via a synthesis-reduction over
the cell stats (NOT IQL — INV-EEL-4); `±1` is the bootstrap. **Semantics note:**
`time_cost` is stored as a normalized **proficiency in [0,1] (HIGH = proficient)**
so it composes with the existing utility-gated delegate path + the Engram-style
"high = grounded" reduction; the BRAIN §3.4 `time_cost` is an ACT-R *cost*
(low = proficient) — BRAIN ingest inverts proficiency→cost. Promote/delegate when
`time_cost ≥ promote_floor`.

Skills are NEVER deleted (Q4 / arch §8.4 / INV-3); soft-retire via `time_cost`
decay. The pre-B1 `sha256(sequence,kind)`-keyed table is preserved as
`procedural_skills_legacy` by a one-time idempotent migration.

INV-Syn-20 first-invocation re-verification is owned by the companion
SkillVerifier (P8.F) — this store exposes `mark_verified()` / `mark_rejected()`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Optional

from titan_hcl.synthesis.writer import on_writer, resolve_writer

logger = logging.getLogger(__name__)


# Embedding model dimensionality (BAAI/bge-small-en-v1.5 — same as P4 spine
# concepts so cross-substrate analogy queries stay shape-compatible).
EMBEDDING_DIM: int = 384

# Promote / delegate floor on the cell `time_cost` proficiency (high=proficient).
# Conservative bootstrap per RFP §7.B1; emergence (the reduction) refines.
# A single oracle-verified success yields time_cost=1.0 → promotes (EEL-G2).
DEFAULT_PROMOTE_FLOOR: float = 0.7

# A cell is delegatable only if success-dominant. success_ratio ≥ this → positive.
POSITIVE_RATIO_FLOOR: float = 0.5

# time_cost below which a previously-positive cell soft-retires (emits a log +
# bus event; the row stays per Q4 / INV-3).
DEFAULT_SOFT_RETIRE_FLOOR: float = 0.1

# Max events drained per pass (bounds the off-hot-path tick budget).
DEFAULT_DRAIN_LIMIT: int = 500

# Drained events are marked processed (so the idempotent enqueue can dedup a
# replayed bus message) then PURGED once older than this window — bounding the
# queue so a full `WHERE processed=FALSE` scan stays cheap with NO secondary
# index (D-SPEC-154). Comfortably exceeds any realistic bus redelivery latency.
PROCESSED_PURGE_WINDOW_S: float = 600.0

# Boot-critical schema bootstrap timeout. _init_schema routes through the
# SynthesisWriter (single-writer, INV-Syn-19), which at boot can be backed up
# with the recompute + dream sequence (~30s+ of GIL-held compute). The 30s
# `submit_sync` default TIMED OUT under load → ProceduralSkillStore construction
# raised → B1 silently unwired (2026-06-10 soak finding). A generous budget lets
# the trivial DDL through once the writer drains its boot backlog.
SCHEMA_INIT_TIMEOUT_S: float = 180.0

# Dream-hardening (2026-06-10): the negative-skill LLM-abstraction queue. The
# off-hot-path daemon drains up to this many clusters/tick and RETRIES a failed
# (timed-out / unparseable) LLM abstraction up to MAX_ATTEMPTS before marking it
# 'failed' (give up gracefully — never retry forever, never silently lose it).
ABSTRACTION_DRAIN_LIMIT: int = 8
ABSTRACTION_MAX_ATTEMPTS: int = 5


def compute_skill_id(oracle_id: str, goal_class: str) -> str:
    """Deterministic skill_id from the OUTCOME `(oracle_id, goal_class)` (EEL B1).
    Same outcome → same skill_id on every call (mirrors the pre-B1
    `compute_skill_id(sequence_tuple, kind)`; the identity mechanism is unchanged,
    only its defining content moved from the sequence-shape to the outcome)."""
    payload = json.dumps(
        {"oracle_id": oracle_id or "", "goal_class": goal_class or ""},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return "skill_" + hashlib.sha256(payload).hexdigest()[:16]


def _event_id(oracle_id: str, goal_class: str, task_shape: str,
              parent_tool_call_tx: str, ts: float) -> str:
    """Content-hash id for a score event — idempotent enqueue (a replayed bus
    message never double-counts)."""
    payload = json.dumps(
        {"o": oracle_id or "", "g": goal_class or "", "t": task_shape or "",
         "p": parent_tool_call_tx or "", "ts": round(float(ts), 6)},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return "ev_" + hashlib.sha256(payload).hexdigest()[:24]


def reduce_time_cost(success_count: int, failure_count: int, c: float) -> float:
    """Synthesis-reduction proficiency in [0,1] (HIGH=proficient) — NOT IQL.

    Every counted use is oracle-verified (INV-EEL-2), so the verified-success
    fraction IS the bootstrap proficiency. A single oracle-verified success
    (1 success, 0 failure, c=1.0) → 1.0 → promotable (EEL-G2). A failure-dominant
    cell → low → not delegatable + flips polarity negative. `c` (the verified
    confidence) carries through as the dominant term; the ±1 / counts are the
    bootstrap the reduction refines (the §6.2 Engram percentile-blend is the
    forward enrichment — INV-EEL-4)."""
    total = int(success_count) + int(failure_count)
    if total <= 0:
        return 0.0
    success_ratio = float(success_count) / float(total)
    return max(0.0, min(1.0, float(c) * success_ratio))


def _polarity_of(success_count: int, failure_count: int) -> str:
    """'positive' (success-dominant, delegatable) · 'negative' (failure-dominant,
    INV-EEL-5 never delegated) · 'neutral' (no evidence yet)."""
    total = int(success_count) + int(failure_count)
    if total <= 0:
        return "neutral"
    if (float(success_count) / float(total)) >= POSITIVE_RATIO_FLOOR and success_count > 0:
        return "positive"
    return "negative"


class _DuckDBConnLike:
    """Structural protocol for the DuckDB connection (tests inject fakes)."""
    def execute(self, sql: str, params: Any = None) -> Any: ...
    def fetchall(self) -> list: ...


class ProceduralSkillStore:
    """Sole writer of the procedural-skill substrate. INV-Syn-19 / INV-Syn-29.

    Constructor params (unchanged from Phase 8):
      duckdb_conn, faiss_path, snapshot_path, embedder, clock, soft_retire_floor,
      on_soft_retire, writer. See the prior Phase-8 docstring; behaviour preserved.
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
        promote_floor: float = DEFAULT_PROMOTE_FLOOR,
        on_soft_retire: Optional[Any] = None,
        writer: Any = None,
    ):
        self._db = duckdb_conn
        self._faiss_path = str(faiss_path)
        self._snapshot_path = str(snapshot_path)
        self._embedder = embedder
        self._clock = clock
        self._soft_retire_floor = float(soft_retire_floor)
        self._promote_floor = float(promote_floor)
        self._on_soft_retire = on_soft_retire
        self._faiss = None
        self._faiss_dim = EMBEDDING_DIM
        # Single-writer-thread (Option C): every DuckDB + FAISS op runs on the
        # one SynthesisWriter thread (@on_writer) → the writer is the serializer.
        self._writer = resolve_writer(writer)
        self._lock = threading.RLock()
        # Counters for /v6/synthesis/skills/coverage + Observatory readouts.
        self._persists_seen = 0
        self._events_enqueued = 0
        self._events_drained = 0
        self._abstractions_enqueued = 0
        self._abstractions_compiled = 0
        self._cell_updates = 0
        self._promotions_seen = 0
        self._verifications_seen = 0
        self._rejections_seen = 0
        self._soft_retires_seen = 0
        self._init_schema()

    # ── Schema bootstrap + one-time migration ───────────────────────────

    def _init_schema(self) -> None:
        """Boot-critical schema bootstrap (migration + CREATE + index self-heal).
        Routed through the writer with a GENEROUS timeout (SCHEMA_INIT_TIMEOUT_S):
        at boot the SynthesisWriter can be backed up with the recompute/dream
        sequence, and the 30s @on_writer default TIMED OUT under load → construction
        raised → B1 silently unwired (2026-06-10 soak finding). submit_sync runs
        inline if already on the writer thread (tests / InlineWriter)."""
        self._writer.submit_sync(self._init_schema_body, timeout=SCHEMA_INIT_TIMEOUT_S)

    def _init_schema_body(self) -> None:
        """The actual DDL — ALWAYS runs on the writer thread (via _init_schema's
        submit_sync). Migrate the pre-B1 table out of the way (never delete —
        §8.4/INV-3), then CREATE the B1 tables (idempotent)."""
        with self._lock:
            self._migrate_legacy_if_needed()
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS procedural_skills ("
                "  skill_id        TEXT    PRIMARY KEY,"          # = sha256(oracle_id|goal_class)
                "  oracle_id       TEXT    NOT NULL,"
                "  goal_class      TEXT    NOT NULL,"
                "  name            TEXT    NOT NULL,"
                "  nl_description  TEXT    NOT NULL,"
                "  embedding_id    INTEGER,"                       # FAISS row id, per-OUTCOME (-1 = unembedded)
                "  created_at      DOUBLE  NOT NULL,"
                "  promoted        BOOLEAN DEFAULT FALSE,"
                "  promoted_at     DOUBLE,"
                "  verified_at     DOUBLE,"                        # INV-Syn-20 first-invocation re-verify
                "  UNIQUE (oracle_id, goal_class)"
                ")"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS skill_cells ("
                "  skill_id        TEXT    NOT NULL,"
                "  task_shape      TEXT    NOT NULL,"              # (task_type|tool_id|domain_hint)
                "  b_i             INTEGER DEFAULT 0,"             # §3.4 use count
                "  c               DOUBLE  DEFAULT 0.0,"           # §3.4 verified-success / confidence
                "  time_cost       DOUBLE  DEFAULT 0.0,"           # §3.4 proficiency (high=proficient)
                "  success_count   INTEGER DEFAULT 0,"
                "  failure_count   INTEGER DEFAULT 0,"
                "  last_used       DOUBLE,"
                "  executable_spec TEXT,"
                "  preconditions   TEXT,"
                "  postconditions  TEXT,"
                "  compiled_from   TEXT,"                          # source tx_hashes (INV-Syn-20 re-verifies the CELL)
                "  polarity        TEXT    DEFAULT 'neutral',"     # 'positive'|'negative'|'neutral' (INV-EEL-5)
                "  PRIMARY KEY (skill_id, task_shape)"
                ")"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS skill_score_events ("
                "  event_id            TEXT    PRIMARY KEY,"
                "  ts                  DOUBLE  NOT NULL,"
                "  oracle_id           TEXT    NOT NULL,"
                "  goal_class          TEXT    NOT NULL,"
                "  task_shape          TEXT    NOT NULL,"
                "  success             BOOLEAN NOT NULL,"
                "  parent_tool_call_tx TEXT,"
                "  processed           BOOLEAN DEFAULT FALSE"
                ")"
            )
            # Dream-hardening (2026-06-10): the negative-skill LLM ABSTRACTION is
            # decoupled from the dream into this DURABLE queue + an off-hot-path
            # daemon. The dream's miner ENQUEUES recurrent failure clusters here
            # (fast, no LLM); the abstraction daemon drains them, making the LLM
            # call OFF the dream/writer thread, with retry — so a slow/timed-out
            # LLM never blocks dreaming and the work is never lost. PERSISTENT
            # across restarts/crashes: the table is durable, entries are marked
            # 'done' only AFTER the skill persists, and the daemon RESUMES from
            # status='pending' on every boot (save/resume). PK-only (INV-Syn-30).
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS skill_abstraction_queue ("
                "  cluster_id   TEXT    PRIMARY KEY,"          # sha256(sequence|kind)
                "  payload      TEXT    NOT NULL,"             # JSON: sequence/occurrence_count/kind/members_summary/compiled_from
                "  enqueued_at  DOUBLE  NOT NULL,"
                "  attempts     INTEGER DEFAULT 0,"
                "  status       TEXT    DEFAULT 'pending',"    # pending | done | failed
                "  last_attempt DOUBLE,"
                "  skill_id     TEXT"                          # set on 'done'
                ")"
            )
            # Root-cause fix (D-SPEC-154 / INV-Syn-30): NO secondary ART indexes
            # on these high-churn UPSERT'd tables. DuckDB secondary indexes degrade
            # pathologically under sustained UPSERT churn (reproduced 2026-06-09:
            # ~2.5× slowdown over a 6000-cycle stress, while the PK-only path stays
            # clean + fast) — the precursor to the actr_buffers-class FATAL
            # crash-loop. These tables are small; the PK index covers every query
            # (skill_id is the PK prefix; time_cost/promoted sort over dozens of
            # rows; the events queue is purged-on-drain). DROP IF EXISTS self-heals
            # fleet DBs that got the indexes from the initial B1 deploy.
            for _idx in ("idx_skill_cells_skill", "idx_skill_cells_time_cost",
                         "idx_skill_events_unprocessed", "idx_procedural_skills_promoted"):
                try:
                    self._db.execute(f"DROP INDEX IF EXISTS {_idx}")
                except Exception:  # pragma: no cover — defensive
                    pass

    def _migrate_legacy_if_needed(self) -> None:
        """If a pre-B1 `procedural_skills` (skill_id PK, no `oracle_id` column)
        exists, rename it → `procedural_skills_legacy` (one-time, idempotent;
        NEVER drops a canonical skill row — §8.4/INV-3). No-op once migrated or on
        a fresh install."""
        try:
            cols = self._db.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'procedural_skills'"
            ).fetchall()
        except Exception:
            return  # information_schema unavailable (fake conn) — nothing to migrate
        col_names = {str(r[0]).lower() for r in (cols or [])}
        if not col_names:
            return  # table doesn't exist yet — fresh install
        if "oracle_id" in col_names:
            return  # already B1-shaped
        # Pre-B1 shape detected → preserve by rename (only if legacy not present).
        legacy = self._db.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'procedural_skills_legacy'"
        ).fetchall()
        # DuckDB refuses `ALTER TABLE … RENAME` while dependent objects (the
        # pre-B1 secondary indexes) still reference the table — so drop them
        # first. The legacy table is archival (never queried hot), so it needs
        # no indexes. (Caught live on T3 2026-06-09: the rename threw "Cannot
        # alter entry … entries that depend on it" → store init failed.)
        self._drop_pre_b1_indexes()
        if legacy:
            # legacy already exists from a prior migration attempt — leave it
            # intact + rename the stray pre-B1 table aside with a suffix so the
            # CREATE below proceeds.
            logger.warning(
                "[ProceduralSkillStore] both procedural_skills (pre-B1) and "
                "procedural_skills_legacy exist — preserving legacy; renaming the "
                "stray pre-B1 table to procedural_skills_legacy_dup")
            self._db.execute(
                "ALTER TABLE procedural_skills RENAME TO procedural_skills_legacy_dup")
            return
        logger.info(
            "[ProceduralSkillStore] EEL B1 migration: preserving pre-B1 "
            "procedural_skills → procedural_skills_legacy (never-delete §8.4/INV-3)")
        self._db.execute(
            "ALTER TABLE procedural_skills RENAME TO procedural_skills_legacy")

    def _drop_pre_b1_indexes(self) -> None:
        """Drop the pre-B1 secondary indexes on `procedural_skills` so the table
        can be renamed (DuckDB blocks RENAME while indexes depend on it)."""
        for idx in ("idx_procedural_skills_utility", "idx_procedural_skills_last_used"):
            try:
                self._db.execute(f"DROP INDEX IF EXISTS {idx}")
            except Exception as e:  # pragma: no cover — defensive
                logger.debug("[ProceduralSkillStore] drop pre-B1 index %s: %s", idx, e)

    # ── FAISS lifecycle (lazy load / save) — preserved verbatim ──────────

    def _ensure_faiss(self) -> None:
        if self._faiss is not None:
            return
        import faiss  # local import — keeps cold-boot RSS down (Phase 11 D-SPEC-138)
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
        """Embed `text` + add to FAISS. Returns the new row id, -1 on no-embedder/fail."""
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
            def _add() -> int:
                self._ensure_faiss()
                row_id = self._faiss.ntotal
                self._faiss.add(vec)
                self._save_faiss()
                return row_id
            return self._writer.submit_sync(_add)
        except Exception as e:
            logger.warning("[ProceduralSkillStore] embed_and_add failed: %s", e)
            return -1

    # ── Outcome row (the skill) ──────────────────────────────────────────

    @on_writer
    def ensure_outcome(
        self,
        *,
        oracle_id: str,
        goal_class: str,
        name: Optional[str] = None,
        nl_description: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> str:
        """Idempotently create the OUTCOME row for `(oracle_id, goal_class)` and
        return its `skill_id`. Embeds `nl_description` (per-OUTCOME) on first
        create. Re-calling preserves promoted/verified/embedding."""
        if not oracle_id or not goal_class:
            raise ValueError("oracle_id and goal_class required")
        skill_id = compute_skill_id(oracle_id, goal_class)
        ts_val = float(ts) if ts is not None else float(self._clock())
        nm = name or goal_class
        desc = nl_description or f"{goal_class} via {oracle_id}"
        with self._lock:
            existing = self._db.execute(
                "SELECT embedding_id FROM procedural_skills WHERE skill_id = ?",
                [skill_id],
            ).fetchall()
        if existing:
            return skill_id  # already present — preserve all lifecycle state
        embedding_id = self._embed_and_add(desc)
        with self._lock:
            self._db.execute(
                "INSERT INTO procedural_skills "
                "(skill_id, oracle_id, goal_class, name, nl_description, embedding_id, "
                " created_at, promoted, promoted_at, verified_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, FALSE, NULL, NULL) "
                "ON CONFLICT (skill_id) DO NOTHING",
                [skill_id, oracle_id, goal_class, nm, desc, embedding_id, ts_val],
            )
            self._persists_seen += 1
        return skill_id

    # ── Capture queue (Option C) ─────────────────────────────────────────

    @on_writer
    def enqueue_score_event(
        self,
        *,
        oracle_id: str,
        goal_class: str,
        task_shape: str,
        success: bool,
        parent_tool_call_tx: str = "",
        ts: Optional[float] = None,
    ) -> str:
        """Append a per-oracle-verified-use score event (verdict-time, synthesis
        sole-writer). Idempotent on the content-hash `event_id` (a replayed bus
        message never double-counts). Returns the event_id. Cheap — the cell
        upsert happens off-hot-path in `drain_score_events`."""
        if not oracle_id or not goal_class or not task_shape:
            raise ValueError("oracle_id, goal_class, task_shape required")
        ts_val = float(ts) if ts is not None else float(self._clock())
        eid = _event_id(oracle_id, goal_class, task_shape, parent_tool_call_tx, ts_val)
        with self._lock:
            self._db.execute(
                "INSERT INTO skill_score_events "
                "(event_id, ts, oracle_id, goal_class, task_shape, success, "
                " parent_tool_call_tx, processed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, FALSE) "
                "ON CONFLICT (event_id) DO NOTHING",
                [eid, ts_val, oracle_id, goal_class, task_shape, bool(success),
                 parent_tool_call_tx or None],
            )
            self._events_enqueued += 1
        return eid

    @on_writer
    def drain_score_events(self, *, limit: int = DEFAULT_DRAIN_LIMIT) -> dict:
        """Drain unprocessed score events → UPSERT the `(skill_id, task_shape)`
        cells → recompute `time_cost` + `polarity` → promote outcomes whose best
        positive cell crosses `promote_floor` → mark events processed.

        OFF-HOT-PATH (INV-EEL-1): the resource-gate (cpu-load back-off) is applied
        by the *caller* (the 60s tick); this method does the bounded DB work.
        Returns a summary dict. Snapshot-exported once at the end."""
        with self._lock:
            rows = self._db.execute(
                "SELECT event_id, oracle_id, goal_class, task_shape, success, "
                "parent_tool_call_tx, ts FROM skill_score_events "
                "WHERE processed = FALSE ORDER BY ts ASC LIMIT ?",
                [max(1, int(limit))],
            ).fetchall()
        if not rows:
            return {"drained": 0, "cells_touched": 0, "promoted": 0,
                    "successes": 0, "failures": 0}

        touched_cells: set[tuple] = set()
        outcomes_seen: set[str] = set()
        drained_ids: list[str] = []
        # Affective Grounding Loop §7.A: tally this pass's outcomes (only the
        # ones that applied cleanly) so the caller can derive a competence-delta
        # affective nudge. Purely additive — no effect on cell/promotion logic.
        successes = 0
        failures = 0
        for (eid, oracle_id, goal_class, task_shape, success,
             parent_tx, ts) in rows:
            try:
                skill_id = self.ensure_outcome(
                    oracle_id=oracle_id, goal_class=goal_class, ts=ts)
                self._apply_event_to_cell(
                    skill_id=skill_id, task_shape=task_shape,
                    success=bool(success), parent_tool_call_tx=parent_tx or "",
                    ts=float(ts))
                touched_cells.add((skill_id, task_shape))
                outcomes_seen.add(skill_id)
                drained_ids.append(eid)
                if bool(success):
                    successes += 1
                else:
                    failures += 1
            except Exception as e:
                logger.warning(
                    "[ProceduralSkillStore] drain: event %s failed: %s", eid, e)
        # Mark drained processed (only the ones that applied cleanly), then purge
        # processed events older than the redelivery window — bounds the queue so
        # the unindexed `WHERE processed=FALSE` scan stays cheap (D-SPEC-154).
        if drained_ids:
            with self._lock:
                placeholders = ",".join("?" for _ in drained_ids)
                self._db.execute(
                    f"UPDATE skill_score_events SET processed = TRUE "
                    f"WHERE event_id IN ({placeholders})",
                    list(drained_ids),
                )
                self._events_drained += len(drained_ids)
                self._db.execute(
                    "DELETE FROM skill_score_events WHERE processed = TRUE AND ts < ?",
                    [float(self._clock()) - PROCESSED_PURGE_WINDOW_S],
                )
        promoted = 0
        for skill_id in outcomes_seen:
            if self._reconcile_promotion(skill_id):
                promoted += 1
        self.snapshot_export()
        return {
            "drained": len(drained_ids),
            "cells_touched": len(touched_cells),
            "promoted": promoted,
            "successes": successes,
            "failures": failures,
        }

    def _apply_event_to_cell(
        self, *, skill_id: str, task_shape: str, success: bool,
        parent_tool_call_tx: str, ts: float,
    ) -> None:
        """UPSERT one cell from a single oracle-verified use (writer thread)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT b_i, c, success_count, failure_count, compiled_from, "
                "time_cost, polarity FROM skill_cells "
                "WHERE skill_id = ? AND task_shape = ?",
                [skill_id, task_shape],
            ).fetchall()
            if rows:
                b_i, c_old, succ, fail, cf_json, _tc, _pol = rows[0]
                b_i = int(b_i or 0)
                succ = int(succ or 0)
                fail = int(fail or 0)
                cf = _safe_json_load(cf_json, [])
            else:
                b_i = succ = fail = 0
                cf = []
            b_i += 1
            if success:
                succ += 1
            else:
                fail += 1
            # c = verified-success fraction (every counted use is oracle-verified).
            c_new = float(succ) / float(succ + fail) if (succ + fail) > 0 else 0.0
            time_cost = reduce_time_cost(succ, fail, c_new)
            polarity = _polarity_of(succ, fail)
            if parent_tool_call_tx and parent_tool_call_tx not in cf:
                cf.append(parent_tool_call_tx)
                cf = cf[-32:]  # bound the lineage list
            cf_out = json.dumps(cf, ensure_ascii=False, separators=(",", ":"))
            prev_time_cost = float(_tc) if rows and _tc is not None else 0.0
            self._db.execute(
                "INSERT INTO skill_cells "
                "(skill_id, task_shape, b_i, c, time_cost, success_count, "
                " failure_count, last_used, compiled_from, polarity) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (skill_id, task_shape) DO UPDATE SET "
                "b_i = excluded.b_i, c = excluded.c, time_cost = excluded.time_cost, "
                "success_count = excluded.success_count, "
                "failure_count = excluded.failure_count, last_used = excluded.last_used, "
                "compiled_from = excluded.compiled_from, polarity = excluded.polarity",
                [skill_id, task_shape, b_i, c_new, time_cost, succ, fail, ts,
                 cf_out, polarity],
            )
            self._cell_updates += 1
            crossed_down = (
                prev_time_cost > self._soft_retire_floor
                and time_cost <= self._soft_retire_floor
            )
            if crossed_down:
                self._soft_retires_seen += 1
        if crossed_down and self._on_soft_retire is not None:
            try:
                self._on_soft_retire(skill_id, time_cost)
            except Exception as e:  # pragma: no cover — defensive
                logger.warning(
                    "[ProceduralSkillStore] on_soft_retire callback raised: %s", e)

    @on_writer
    def _reconcile_promotion(self, skill_id: str) -> bool:
        """Promote the outcome iff its best POSITIVE cell's time_cost ≥
        promote_floor. Returns True on a fresh promotion. Mirrors the readout the
        Kuzu `Production` node reflects (utility_score = best-cell time_cost)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT MAX(time_cost) FROM skill_cells "
                "WHERE skill_id = ? AND polarity = 'positive'",
                [skill_id],
            ).fetchall()
            best = float(rows[0][0]) if (rows and rows[0][0] is not None) else 0.0
            prev = self._db.execute(
                "SELECT promoted FROM procedural_skills WHERE skill_id = ?",
                [skill_id],
            ).fetchall()
            was_promoted = bool(prev[0][0]) if prev else False
            should = best >= self._promote_floor
            if should and not was_promoted:
                self._db.execute(
                    "UPDATE procedural_skills SET promoted = TRUE, promoted_at = ? "
                    "WHERE skill_id = ?",
                    [float(self._clock()), skill_id],
                )
                self._promotions_seen += 1
                return True
            if not should and was_promoted:
                # soft-demote (never delete) — the cell decayed below floor.
                self._db.execute(
                    "UPDATE procedural_skills SET promoted = FALSE WHERE skill_id = ?",
                    [skill_id],
                )
        return False

    # ── Negative-skill compilation (miner, negative-only — RFP §3) ───────

    @on_writer
    def persist_negative_skill(
        self,
        *,
        oracle_id: str,
        goal_class: str,
        task_shape: str,
        name: str,
        nl_description: str,
        compiled_from: list[str],
        executable_spec: Optional[dict] = None,
        preconditions: Optional[list] = None,
        postconditions: Optional[list] = None,
        ts: Optional[float] = None,
    ) -> str:
        """The negative-only miner's write surface (RFP §3 — negatives kept as
        replay fuel + avoidance). Creates the outcome + a `polarity='negative'`
        cell (never delegated — INV-EEL-5). Idempotent on (skill_id, task_shape).
        Returns the skill_id."""
        if not compiled_from:
            raise ValueError("compiled_from must be non-empty (lineage gate)")
        ts_val = float(ts) if ts is not None else float(self._clock())
        skill_id = self.ensure_outcome(
            oracle_id=oracle_id, goal_class=goal_class,
            name=name, nl_description=nl_description, ts=ts_val)
        spec_json = json.dumps(executable_spec or {}, sort_keys=True,
                               ensure_ascii=False, separators=(",", ":"))
        pre_json = json.dumps(list(preconditions or []), sort_keys=True,
                              ensure_ascii=False, separators=(",", ":"))
        post_json = json.dumps(list(postconditions or []), sort_keys=True,
                               ensure_ascii=False, separators=(",", ":"))
        cf_json = json.dumps(list(compiled_from), ensure_ascii=False,
                             separators=(",", ":"))
        with self._lock:
            self._db.execute(
                "INSERT INTO skill_cells "
                "(skill_id, task_shape, b_i, c, time_cost, success_count, "
                " failure_count, last_used, executable_spec, preconditions, "
                " postconditions, compiled_from, polarity) "
                "VALUES (?, ?, 0, 0.0, 0.0, 0, 0, ?, ?, ?, ?, ?, 'negative') "
                "ON CONFLICT (skill_id, task_shape) DO UPDATE SET "
                "executable_spec = excluded.executable_spec, "
                "preconditions = excluded.preconditions, "
                "postconditions = excluded.postconditions, "
                "compiled_from = excluded.compiled_from, polarity = 'negative'",
                [skill_id, task_shape, ts_val, spec_json, pre_json, post_json, cf_json],
            )
            self._persists_seen += 1
        self.snapshot_export()
        return skill_id

    # ── Dream-hardening: off-hot-path negative-skill abstraction queue ───
    # (2026-06-10) The dream ENQUEUES recurrent failure clusters here (fast, no
    # LLM); the synthesis_worker abstraction daemon FETCHES pending entries,
    # makes the LLM call OFF the writer/dream thread, then RECORDS the result
    # (persist + 'done', or attempts++/'failed'). Durable + save/resume: the
    # daemon resumes from status='pending' on every boot.

    @on_writer
    def enqueue_abstraction(self, prepared: dict, ts: Optional[float] = None) -> bool:
        """Idempotent enqueue of a prepared recurrent cluster (no LLM here).
        `prepared` = {cluster_id, sequence, occurrence_count, kind,
        members_summary, compiled_from}. Returns True if NEWLY enqueued (False if
        already queued/done/failed — ON CONFLICT DO NOTHING keeps it idempotent
        across re-mines)."""
        cid = prepared.get("cluster_id")
        if not cid:
            return False
        payload = json.dumps(prepared, ensure_ascii=False, separators=(",", ":"))
        ts_val = float(ts) if ts is not None else float(self._clock())
        with self._lock:
            before = self._db.execute(
                "SELECT COUNT(*) FROM skill_abstraction_queue WHERE cluster_id = ?",
                [cid]).fetchone()[0]
            self._db.execute(
                "INSERT INTO skill_abstraction_queue (cluster_id, payload, enqueued_at) "
                "VALUES (?, ?, ?) ON CONFLICT (cluster_id) DO NOTHING",
                [cid, payload, ts_val])
            self._abstractions_enqueued += int(before == 0)
        return before == 0

    @on_writer
    def fetch_pending_abstractions(
        self, *, limit: int = ABSTRACTION_DRAIN_LIMIT,
        max_attempts: int = ABSTRACTION_MAX_ATTEMPTS,
    ) -> list[dict]:
        """Return up to `limit` pending clusters (attempts < max), oldest first —
        the daemon then makes the LLM call OFF this thread. Resumes naturally on
        boot (the table is durable). Returns [{cluster_id, payload(dict)}]."""
        with self._lock:
            rows = self._db.execute(
                "SELECT cluster_id, payload, attempts FROM skill_abstraction_queue "
                "WHERE status = 'pending' AND attempts < ? "
                "ORDER BY enqueued_at ASC LIMIT ?",
                [int(max_attempts), max(1, int(limit))]).fetchall()
        out: list[dict] = []
        for cid, payload_json, attempts in rows:
            try:
                out.append({"cluster_id": cid, "attempts": int(attempts),
                            "payload": json.loads(payload_json)})
            except (TypeError, ValueError):
                continue
        return out

    @on_writer
    def record_abstraction_result(
        self, cluster_id: str, proposal: Optional[dict], *,
        max_attempts: int = ABSTRACTION_MAX_ATTEMPTS,
    ) -> Optional[str]:
        """Record the outcome of a daemon LLM-abstraction attempt. On a VALID
        proposal → derive the outcome + persist the negative skill + mark 'done'
        (returns skill_id). On failure → attempts++ and mark 'failed' once it hits
        max_attempts, else leave 'pending' for retry (returns None). The LLM call
        already happened OFF this thread; this is only the fast DB write."""
        row = None
        with self._lock:
            r = self._db.execute(
                "SELECT payload, attempts FROM skill_abstraction_queue "
                "WHERE cluster_id = ? AND status = 'pending'", [cluster_id]).fetchall()
            row = r[0] if r else None
        if row is None:
            return None
        try:
            p = json.loads(row[0])
        except (TypeError, ValueError):
            p = {}
        attempts = int(row[1])
        nl = ((proposal or {}).get("nl_description") or "").strip()
        spec = (proposal or {}).get("executable_spec")
        valid = bool(nl) and isinstance(spec, dict) and bool(p.get("compiled_from"))
        if not valid:
            new_attempts = attempts + 1
            new_status = "failed" if new_attempts >= int(max_attempts) else "pending"
            with self._lock:
                self._db.execute(
                    "UPDATE skill_abstraction_queue SET attempts = ?, status = ?, "
                    "last_attempt = ? WHERE cluster_id = ?",
                    [new_attempts, new_status, float(self._clock()), cluster_id])
            return None
        # Valid → derive (oracle_id sentinel marks the recurrence origin) + persist.
        from titan_hcl.synthesis.goal_class import (
            goal_class as _derive_goal_class, make_task_shape as _derive_task_shape)
        seq = p.get("sequence")
        tool_sig = ("+".join(str(s) for s in seq)
                    if isinstance(seq, (list, tuple)) else str(seq))[:80]
        try:
            skill_id = self.persist_negative_skill(
                oracle_id="miner_recurrence", goal_class=_derive_goal_class(nl),
                task_shape=_derive_task_shape("procedural", tool_sig, ""),
                name=f"[negative] {nl.split('.')[0][:64] or 'compiled_skill'}",
                nl_description=nl, compiled_from=list(p.get("compiled_from") or []),
                executable_spec=spec,
                preconditions=(proposal or {}).get("preconditions") or [],
                postconditions=(proposal or {}).get("postconditions") or [],
                ts=self._clock())
        except Exception as e:
            # persist failed → treat as a retryable attempt (don't lose the cluster).
            logger.warning("[ProceduralSkillStore] abstraction persist failed: %s", e)
            with self._lock:
                self._db.execute(
                    "UPDATE skill_abstraction_queue SET attempts = ?, last_attempt = ? "
                    "WHERE cluster_id = ?",
                    [attempts + 1, float(self._clock()), cluster_id])
            return None
        with self._lock:
            self._db.execute(
                "UPDATE skill_abstraction_queue SET status = 'done', skill_id = ?, "
                "last_attempt = ? WHERE cluster_id = ?",
                [skill_id, float(self._clock()), cluster_id])
            self._abstractions_compiled += 1
        return skill_id

    # ── INV-Syn-20 verifier mutators (on the outcome row) ────────────────

    @on_writer
    def mark_verified(self, skill_id: str) -> None:
        """INV-Syn-20: the outcome's cell lineage passed first-invocation
        chain-resolve + content-hash check."""
        now = float(self._clock())
        with self._lock:
            self._db.execute(
                "UPDATE procedural_skills SET verified_at = ? WHERE skill_id = ?",
                [now, skill_id],
            )
            self._verifications_seen += 1
        self.snapshot_export()

    @on_writer
    def mark_rejected(self, skill_id: str, reason: str) -> None:
        """INV-Syn-20: first-invocation re-verification failed. Marks the outcome
        verified_at=now (won't re-enter the verifier) + flips all its cells
        negative (hard-ineligible for delegation — INV-EEL-5). Row stays (Q4)."""
        now = float(self._clock())
        with self._lock:
            self._db.execute(
                "UPDATE procedural_skills SET verified_at = ? WHERE skill_id = ?",
                [now, skill_id],
            )
            self._db.execute(
                "UPDATE skill_cells SET polarity = 'negative', time_cost = 0.0 "
                "WHERE skill_id = ?",
                [skill_id],
            )
            self._db.execute(
                "UPDATE procedural_skills SET promoted = FALSE WHERE skill_id = ?",
                [skill_id],
            )
            self._rejections_seen += 1
        logger.info(
            "[ProceduralSkillStore] skill %s rejected (reason=%s)", skill_id, reason)
        self.snapshot_export()

    @on_writer
    def apply_utility_delta(self, skill_id: str, delta: float) -> Optional[float]:
        """Phase 9 / INV-Syn-24 Tier-2 user-feedback override (coarse outcome-level
        nudge). Adjusts the outcome's POSITIVE cells' time_cost by `delta`
        (clamped [0,1]); re-reconciles promotion. Returns the new best time_cost,
        or None for an unknown skill_id. (A user override is not an invocation
        outcome, so b_i/success/failure counts stay untouched.)"""
        if not skill_id:
            raise ValueError("skill_id must be non-empty")
        with self._lock:
            rows = self._db.execute(
                "SELECT task_shape, time_cost FROM skill_cells "
                "WHERE skill_id = ? AND polarity = 'positive'",
                [skill_id],
            ).fetchall()
            if not rows:
                logger.warning(
                    "[ProceduralSkillStore] apply_utility_delta for unknown/"
                    "non-positive skill_id=%s", skill_id)
                return None
            best = 0.0
            for task_shape, tc in rows:
                new_tc = max(0.0, min(1.0, float(tc or 0.0) + float(delta)))
                self._db.execute(
                    "UPDATE skill_cells SET time_cost = ? "
                    "WHERE skill_id = ? AND task_shape = ?",
                    [new_tc, skill_id, task_shape],
                )
                best = max(best, new_tc)
            self._cell_updates += 1
        self._reconcile_promotion(skill_id)
        self.snapshot_export()
        return best

    # ── Read surface ─────────────────────────────────────────────────────

    @on_writer
    def read_skill(self, skill_id: str) -> Optional[dict]:
        """Full outcome row + its cells (best positive cell summarized as the
        delegatable recipe). None if missing."""
        if not skill_id:
            return None
        with self._lock:
            head = self._db.execute(
                "SELECT skill_id, oracle_id, goal_class, name, nl_description, "
                "embedding_id, created_at, promoted, promoted_at, verified_at "
                "FROM procedural_skills WHERE skill_id = ?",
                [skill_id],
            ).fetchall()
            if not head:
                return None
            cells = self._db.execute(
                "SELECT task_shape, b_i, c, time_cost, success_count, failure_count, "
                "last_used, executable_spec, preconditions, postconditions, "
                "compiled_from, polarity FROM skill_cells WHERE skill_id = ? "
                "ORDER BY time_cost DESC",
                [skill_id],
            ).fetchall()
        return self._row_to_dict(head[0], cells)

    @on_writer
    def read_for_match(
        self, *, utility_floor: float = 0.3, k: int = 5, verified_only: bool = True,
    ) -> list[dict]:
        """Outcomes eligible for the agno match path — each joined to its BEST
        POSITIVE cell whose time_cost ≥ utility_floor. **Polarity guard at the
        source (INV-EEL-5):** negative/neutral-only outcomes are never returned,
        so `match_procedural_skill` can never surface a `[negative]` as a recipe.
        `verified_only` additionally requires the outcome to be promoted OR
        verified_at IS NOT NULL (a per-use-promoted cell was oracle-verified by
        construction — INV-EEL-2). Returns rows shaped for ProceduralSkillReader
        (`utility_score` = best-cell time_cost; `verified_at`/`embedding_id`
        carried)."""
        floor = float(utility_floor)
        with self._lock:
            verified_clause = (
                " AND (s.promoted = TRUE OR s.verified_at IS NOT NULL)"
                if verified_only else "")
            sql = (
                "SELECT s.skill_id, s.oracle_id, s.goal_class, s.name, s.nl_description, "
                "s.embedding_id, s.created_at, s.promoted, s.verified_at, "
                "c.task_shape, c.time_cost, c.executable_spec, c.preconditions, "
                "c.postconditions, c.b_i, c.success_count, c.failure_count "
                "FROM procedural_skills s "
                "JOIN skill_cells c ON c.skill_id = s.skill_id "
                "WHERE c.polarity = 'positive' AND c.time_cost >= ?"
                + verified_clause +
                # one (best) cell per outcome — DuckDB QUALIFY over the join
                " QUALIFY ROW_NUMBER() OVER (PARTITION BY s.skill_id ORDER BY c.time_cost DESC) = 1 "
                "ORDER BY c.time_cost DESC LIMIT ?"
            )
            rows = self._db.execute(sql, [floor, max(1, int(k) * 4)]).fetchall()
        return [self._match_row_to_dict(r) for r in rows]

    @on_writer
    def list_all(self, *, limit: int = 100) -> list[dict]:
        """For Observatory. Outcomes + best-cell summary, ordered by best time_cost."""
        with self._lock:
            head = self._db.execute(
                "SELECT skill_id, oracle_id, goal_class, name, nl_description, "
                "embedding_id, created_at, promoted, promoted_at, verified_at "
                "FROM procedural_skills ORDER BY promoted DESC, created_at DESC LIMIT ?",
                [max(1, int(limit))],
            ).fetchall()
            out = []
            for h in head:
                cells = self._db.execute(
                    "SELECT task_shape, b_i, c, time_cost, success_count, failure_count, "
                    "last_used, executable_spec, preconditions, postconditions, "
                    "compiled_from, polarity FROM skill_cells WHERE skill_id = ? "
                    "ORDER BY time_cost DESC",
                    [h[0]],
                ).fetchall()
                out.append(self._row_to_dict(h, cells))
        return out

    @on_writer
    def list_unverified(self, *, limit: int = 500) -> list[str]:
        """Outcome skill_ids with verified_at IS NULL that have a positive cell —
        the candidates the dream-cycle SkillVerifier re-verifies (INV-Syn-20)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT s.skill_id FROM procedural_skills s "
                "JOIN skill_cells c ON c.skill_id = s.skill_id "
                "WHERE s.verified_at IS NULL AND c.polarity = 'positive' "
                "ORDER BY s.created_at DESC LIMIT ?",
                [max(1, int(limit))],
            ).fetchall()
        return [r[0] for r in rows]

    @on_writer
    def faiss_search(self, query_vec: Any, top_k: int = 20) -> list[tuple[int, float]]:
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
        if self._embedder is None or not text:
            return None
        try:
            return self._embedder(text)
        except Exception as e:
            logger.warning("[ProceduralSkillStore] embed_query failed: %s", e)
            return None

    def _row_to_dict(self, head: tuple, cells: list) -> dict:
        (skill_id, oracle_id, goal_class, name, nl, emb_id, created_at,
         promoted, promoted_at, verified_at) = head
        cell_dicts = [
            {
                "task_shape": c[0], "b_i": int(c[1] or 0), "c": float(c[2] or 0.0),
                "time_cost": float(c[3] or 0.0), "success_count": int(c[4] or 0),
                "failure_count": int(c[5] or 0),
                "last_used": float(c[6]) if c[6] is not None else None,
                "executable_spec": _safe_json_load(c[7], {}),
                "preconditions": _safe_json_load(c[8], []),
                "postconditions": _safe_json_load(c[9], []),
                "compiled_from": _safe_json_load(c[10], []),
                "polarity": c[11] or "neutral",
            }
            for c in (cells or [])
        ]
        positives = [c for c in cell_dicts if c["polarity"] == "positive"]
        best = max(positives, key=lambda c: c["time_cost"], default=None)
        # INV-Syn-20: the SkillVerifier re-verifies the outcome's lineage at first
        # invocation — expose the union of the cells' compiled_from tx_hashes (the
        # real on-chain tool-call TXs that scored the cells) so the check resolves.
        merged_cf: list = []
        for c in cell_dicts:
            for h in c.get("compiled_from") or []:
                if h not in merged_cf:
                    merged_cf.append(h)
        return {
            "skill_id": skill_id,
            "oracle_id": oracle_id,
            "goal_class": goal_class,
            "name": name,
            "nl_description": nl,
            "embedding_id": int(emb_id) if emb_id is not None else -1,
            "created_at": float(created_at) if created_at is not None else 0.0,
            "promoted": bool(promoted),
            "promoted_at": float(promoted_at) if promoted_at is not None else None,
            "verified_at": float(verified_at) if verified_at is not None else None,
            "compiled_from": merged_cf,
            # Summed across cells (back-compat for the agno match display).
            "success_count": sum(c["success_count"] for c in cell_dicts),
            "failure_count": sum(c["failure_count"] for c in cell_dicts),
            # Best-positive-cell summary = the delegatable recipe (None if no positive).
            "utility_score": float(best["time_cost"]) if best else 0.0,
            "best_task_shape": best["task_shape"] if best else None,
            "executable_spec": best["executable_spec"] if best else {},
            "cells": cell_dicts,
        }

    def _match_row_to_dict(self, r: tuple) -> dict:
        (skill_id, oracle_id, goal_class, name, nl, emb_id, created_at, promoted,
         verified_at, task_shape, time_cost, spec, pre, post, b_i, succ, fail) = r
        return {
            "skill_id": skill_id,
            "oracle_id": oracle_id,
            "goal_class": goal_class,
            "name": name,
            "nl_description": nl,
            "embedding_id": int(emb_id) if emb_id is not None else -1,
            "created_at": float(created_at) if created_at is not None else 0.0,
            "promoted": bool(promoted),
            "verified_at": float(verified_at) if verified_at is not None else None,
            "best_task_shape": task_shape,
            # `utility_score` = best-cell proficiency, for the reader's composite + gate.
            "utility_score": float(time_cost) if time_cost is not None else 0.0,
            "time_cost": float(time_cost) if time_cost is not None else 0.0,
            "executable_spec": _safe_json_load(spec, {}),
            "preconditions": _safe_json_load(pre, []),
            "postconditions": _safe_json_load(post, []),
            "b_i": int(b_i or 0),
            "success_count": int(succ or 0),
            "failure_count": int(fail or 0),
            "polarity": "positive",  # read_for_match returns positives only (guard at source)
        }

    # ── Snapshot export ────────────────────────────────────────────────

    @on_writer
    def _build_snapshot_payload(self) -> dict:
        with self._lock:
            rows = self._db.execute(
                "SELECT s.skill_id, s.oracle_id, s.goal_class, s.name, s.nl_description, "
                "s.promoted, s.verified_at, "
                "COALESCE(MAX(CASE WHEN c.polarity='positive' THEN c.time_cost END), 0.0) AS best_tc, "
                "COALESCE(SUM(c.success_count), 0) AS succ, "
                "COALESCE(SUM(c.failure_count), 0) AS fail, "
                "COUNT(c.task_shape) AS n_cells, "
                # Break F (RFP_synthesis_reuse_and_routing_revival) — the FAISS row
                # id so the cross-process SnapshotProceduralReader (agno side) can
                # join skills_vectors.faiss hits back to skill rows.
                "s.embedding_id "
                "FROM procedural_skills s "
                "LEFT JOIN skill_cells c ON c.skill_id = s.skill_id "
                "GROUP BY s.skill_id, s.oracle_id, s.goal_class, s.name, "
                "s.nl_description, s.promoted, s.verified_at, s.created_at, "
                "s.embedding_id "
                "ORDER BY best_tc DESC, s.created_at DESC"
            ).fetchall()
            counters = (
                self._persists_seen, self._events_enqueued, self._events_drained,
                self._cell_updates, self._promotions_seen, self._verifications_seen,
                self._rejections_seen, self._soft_retires_seen,
            )
        skills = [
            {
                "skill_id": r[0], "oracle_id": r[1], "goal_class": r[2],
                "name": r[3], "nl_description": r[4],
                "promoted": bool(r[5]),
                "verified_at": float(r[6]) if r[6] is not None else None,
                # `utility_score` retained for snapshot back-compat = best-cell time_cost.
                "utility_score": float(r[7] or 0.0),
                "success_count": int(r[8] or 0),
                "failure_count": int(r[9] or 0),
                "task_shapes": int(r[10] or 0),
                # Break F — FAISS row id for the agno-side snapshot reader (-1 = unembedded).
                "embedding_id": int(r[11]) if r[11] is not None else -1,
            }
            for r in rows
        ]
        (persists, enq, drained, cell_updates, promos, verifs, rejs, retires) = counters
        return {
            "version": 2,  # B1 schema (outcome × cells)
            "ts": float(self._clock()),
            "count": len(skills),
            "persists_seen": persists,
            "events_enqueued": enq,
            "events_drained": drained,
            "cell_updates": cell_updates,
            "promotions_seen": promos,
            "verifications_seen": verifs,
            "rejections_seen": rejs,
            "soft_retires_seen": retires,
            "skills": skills,
        }

    def snapshot_export(self) -> str:
        """Atomic tmp+rename write of `skills_snapshot.json`. Soft-fails on FS errors."""
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
                "[ProceduralSkillStore] snapshot export failed (%s): %s", target, e)
        return target

    # ── Observability ───────────────────────────────────────────────────

    @on_writer
    def stats(self) -> dict:
        with self._lock:
            faiss_count = self._faiss.ntotal if self._faiss is not None else 0
            return {
                "persists_seen": self._persists_seen,
                "events_enqueued": self._events_enqueued,
                "events_drained": self._events_drained,
                "cell_updates": self._cell_updates,
                "promotions_seen": self._promotions_seen,
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
