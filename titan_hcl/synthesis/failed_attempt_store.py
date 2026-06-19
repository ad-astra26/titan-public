"""FailedAttemptStore — sole writer of the `failed_attempts` table on
`synthesis.duckdb` (the unified EEL-B2 / mastery-P9 failure-replay store).

INV-Syn-19 / INV-OML-8: synthesis_worker is the only process that writes this
table; the IQL `self_learning` worker stays numpy-only and owns only its own
duckdb. The failure-replay loop adds NO new learning authority — it is the
existing P8.2 corrector + OracleRouter + skill_store, wired around a durable
unresolved-problem queue (the 2026-06-19 VC-audit; RFP_emergent_mastery_curriculum
§7.P9, the canonical mechanic; EEL RFP §7.B2 points there).

Lifecycle of one row (status):
    pending      — enqueued (a P8.2 solve-until-correct EXHAUSTED, or a Phase-2
                   procedural_miner negative cluster); awaiting an idle revisit.
    in_progress  — handed to a revisit (IMPULSE dispatched to the agency corrector).
                   Durable save/resume: reset to `pending` on every boot — an
                   in-flight revisit interrupted by a restart is simply retried.
    resolved     — the corrector solved it; a positive skill cell was anchored
                   (Sink 1, EEL-G3) and a boosted IQL reward emitted (Sink 2, P9).
    abandoned    — `revisit_count` reached `max_revisits` without a solve (→ the
                   P10 ask-the-Maker escalation hook; not built here).

PK-only (INV-Syn-30 — no secondary ART index on a churn table; this one is small
and every query rides the PK or a bounded `WHERE status=` scan over few rows).
Never deleted (audit trail / re-feed); resolved/abandoned rows are terminal.

`problem_id` is a content hash of the normalized `(problem, goal_class)` so the
SAME unsolved problem re-enqueued (a later exhaustion of the same task) bumps the
existing row's `enqueue_count` instead of forking a duplicate — idempotent enqueue.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any, Optional

from titan_hcl.synthesis.writer import on_writer, resolve_writer

logger = logging.getLogger(__name__)

# Boot-critical schema bootstrap timeout — same rationale as skill_store
# (_init_schema can race the recompute/dream backlog on the writer thread at boot;
# the 30s @on_writer default TIMED OUT under load → silent unwire). Generous budget.
SCHEMA_INIT_TIMEOUT_S: float = 180.0

# Status constants (single source — never inline a literal).
ST_PENDING = "pending"
ST_IN_PROGRESS = "in_progress"
ST_RESOLVED = "resolved"
ST_ABANDONED = "abandoned"

# Default cap on revisits before a problem is abandoned (config overrides it).
DEFAULT_MAX_REVISITS: int = 3


def compute_problem_id(problem: str, goal_class: str) -> str:
    """Deterministic id from the normalized `(problem, goal_class)` — idempotent
    enqueue (the same unsolved problem never forks a duplicate row)."""
    payload = json.dumps(
        {"p": (problem or "").strip().lower(), "g": (goal_class or "").strip()},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return "fa_" + hashlib.sha256(payload).hexdigest()[:24]


class FailedAttemptStore:
    """Sole-writer store for the failure-replay queue. Every DuckDB op runs on the
    one SynthesisWriter thread (@on_writer) — the writer is the serializer, exactly
    like ProceduralSkillStore (Option C / INV-Syn-19)."""

    def __init__(
        self,
        *,
        duckdb_conn: Any,
        clock: Any = time.time,
        max_revisits: int = DEFAULT_MAX_REVISITS,
        writer: Any = None,
    ):
        self._db = duckdb_conn
        self._clock = clock
        self._max_revisits = int(max_revisits)
        self._writer = resolve_writer(writer)
        self._lock = threading.RLock()
        # Readout counters (Observatory / coverage endpoints).
        self._enqueued = 0
        self._revisits_handed = 0
        self._resolved = 0
        self._abandoned = 0
        self._init_schema()

    # ── Schema bootstrap + boot save/resume ──────────────────────────────

    def _init_schema(self) -> None:
        """Boot-critical DDL, writer-routed with a generous timeout (matches
        skill_store). Also performs the save/resume reset of any `in_progress`
        rows left by a restart mid-revisit → `pending` (durable, never lost)."""
        self._writer.submit_sync(self._init_schema_body, timeout=SCHEMA_INIT_TIMEOUT_S)

    def _init_schema_body(self) -> None:
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS failed_attempts ("
                "  problem_id      TEXT    PRIMARY KEY,"           # sha256(norm(problem)|goal_class)
                "  problem         TEXT    NOT NULL,"              # the task text the corrector re-runs
                "  goal_class      TEXT    NOT NULL,"
                "  action          TEXT,"                          # the routing action that failed (tool/research/…)
                "  helper          TEXT,"                          # the agency helper to re-run
                "  features        TEXT,"                          # JSON φ vector at the original attempt
                "  attempt_history TEXT,"                          # JSON list of {evidence, verdict, correction}
                "  why_failed      TEXT,"
                "  known_target    TEXT,"                          # NULL unless a deterministic expected result exists (P9 high-value)
                "  status          TEXT    DEFAULT 'pending',"     # pending|in_progress|resolved|abandoned
                "  enqueue_count   INTEGER DEFAULT 1,"
                "  revisit_count   INTEGER DEFAULT 0,"
                "  created_at      DOUBLE  NOT NULL,"
                "  last_attempt_at DOUBLE,"
                "  resolved_at     DOUBLE,"
                "  skill_id        TEXT"                            # set on resolve (the anchored corrected skill)
                ")"
            )
            # Save/resume: a revisit interrupted by a restart is simply retried.
            self._db.execute(
                "UPDATE failed_attempts SET status = ? WHERE status = ?",
                [ST_PENDING, ST_IN_PROGRESS],
            )

    # ── Writes ───────────────────────────────────────────────────────────

    @on_writer
    def enqueue(
        self,
        *,
        problem: str,
        goal_class: str,
        action: str = "",
        helper: str = "",
        features: Optional[list] = None,
        attempt_history: Optional[list] = None,
        why_failed: str = "",
        known_target: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> str:
        """Append (or idempotently re-touch) an unresolved problem. Returns the
        problem_id. A re-enqueue of the SAME `(problem, goal_class)` that is still
        pending/in_progress bumps `enqueue_count` (it failed again) rather than
        forking a duplicate; a re-enqueue of a resolved/abandoned row REOPENS it
        to pending (it regressed — worth revisiting again)."""
        if not problem or not goal_class:
            raise ValueError("problem and goal_class required")
        pid = compute_problem_id(problem, goal_class)
        now = float(ts) if ts is not None else float(self._clock())
        feats_json = json.dumps(features) if features is not None else None
        hist_json = json.dumps(attempt_history) if attempt_history is not None else None
        with self._lock:
            row = self._db.execute(
                "SELECT status FROM failed_attempts WHERE problem_id = ?", [pid]
            ).fetchone()
            if row is None:
                self._db.execute(
                    "INSERT INTO failed_attempts "
                    "(problem_id, problem, goal_class, action, helper, features, "
                    " attempt_history, why_failed, known_target, status, "
                    " enqueue_count, revisit_count, created_at, last_attempt_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,1,0,?,?)",
                    [pid, problem, goal_class, action or None, helper or None,
                     feats_json, hist_json, why_failed or None, known_target,
                     ST_PENDING, now, now],
                )
            else:
                # Re-touch: bump the count, refresh the evidence, REOPEN if terminal.
                self._db.execute(
                    "UPDATE failed_attempts SET enqueue_count = enqueue_count + 1, "
                    "status = ?, attempt_history = COALESCE(?, attempt_history), "
                    "features = COALESCE(?, features), "
                    "why_failed = COALESCE(?, why_failed), last_attempt_at = ? "
                    "WHERE problem_id = ?",
                    [ST_PENDING, hist_json, feats_json, why_failed or None, now, pid],
                )
            self._enqueued += 1
        return pid

    @on_writer
    def next_unresolved(self, *, limit: int = 1) -> list[dict]:
        """Claim the next `limit` pending problems for a revisit (oldest-first =
        fair), marking them `in_progress` so a concurrent tick never double-hands
        the same one. Returns full row dicts (the driver builds the IMPULSE from
        them). ONE-at-a-time by default (the driver's bounded discipline)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT problem_id, problem, goal_class, action, helper, features, "
                "known_target, revisit_count FROM failed_attempts "
                "WHERE status = ? ORDER BY last_attempt_at ASC NULLS FIRST, created_at ASC "
                "LIMIT ?",
                [ST_PENDING, int(limit)],
            ).fetchall()
            now = float(self._clock())
            out: list[dict] = []
            for r in rows:
                pid = r[0]
                self._db.execute(
                    "UPDATE failed_attempts SET status = ?, last_attempt_at = ? "
                    "WHERE problem_id = ?",
                    [ST_IN_PROGRESS, now, pid],
                )
                self._revisits_handed += 1
                out.append({
                    "problem_id": pid, "problem": r[1], "goal_class": r[2],
                    "action": r[3] or "", "helper": r[4] or "",
                    "features": json.loads(r[5]) if r[5] else None,
                    "known_target": r[6], "revisit_count": int(r[7] or 0),
                })
            return out

    @on_writer
    def mark_resolved(self, problem_id: str, *, skill_id: str = "",
                      ts: Optional[float] = None) -> bool:
        """The corrector solved it → terminal `resolved` + the anchored skill_id.
        Returns False if the row is gone/already terminal (never re-resolve)."""
        now = float(ts) if ts is not None else float(self._clock())
        with self._lock:
            cur = self._db.execute(
                "UPDATE failed_attempts SET status = ?, resolved_at = ?, "
                "last_attempt_at = ?, skill_id = ? WHERE problem_id = ? "
                "AND status NOT IN (?, ?)",
                [ST_RESOLVED, now, now, skill_id or None, problem_id,
                 ST_RESOLVED, ST_ABANDONED],
            )
        changed = bool(getattr(cur, "rowcount", 0) or 0)
        if changed:
            self._resolved += 1
        return changed

    @on_writer
    def bump_attempt(self, problem_id: str, *, correction: str = "",
                     verdict: str = "", ts: Optional[float] = None) -> str:
        """A revisit did NOT solve it → record the correction/verdict, increment
        `revisit_count`, and return the resulting status: back to `pending` for
        another try, or `abandoned` once `revisit_count` reaches `max_revisits`."""
        now = float(ts) if ts is not None else float(self._clock())
        with self._lock:
            row = self._db.execute(
                "SELECT attempt_history, revisit_count FROM failed_attempts "
                "WHERE problem_id = ?", [problem_id]
            ).fetchone()
            if row is None:
                return ""
            history = json.loads(row[0]) if row[0] else []
            history.append({"correction": correction, "verdict": verdict, "ts": now})
            new_count = int(row[1] or 0) + 1
            terminal = new_count >= self._max_revisits
            new_status = ST_ABANDONED if terminal else ST_PENDING
            self._db.execute(
                "UPDATE failed_attempts SET attempt_history = ?, revisit_count = ?, "
                "status = ?, last_attempt_at = ? WHERE problem_id = ?",
                [json.dumps(history), new_count, new_status, now, problem_id],
            )
            if terminal:
                self._abandoned += 1
        return new_status

    # ── Reads (Observatory / tests) ──────────────────────────────────────

    @on_writer
    def get(self, problem_id: str) -> Optional[dict]:
        with self._lock:
            r = self._db.execute(
                "SELECT problem_id, problem, goal_class, action, helper, status, "
                "enqueue_count, revisit_count, known_target, skill_id "
                "FROM failed_attempts WHERE problem_id = ?", [problem_id]
            ).fetchone()
        if r is None:
            return None
        return {
            "problem_id": r[0], "problem": r[1], "goal_class": r[2],
            "action": r[3] or "", "helper": r[4] or "", "status": r[5],
            "enqueue_count": int(r[6] or 0), "revisit_count": int(r[7] or 0),
            "known_target": r[8], "skill_id": r[9] or "",
        }

    @on_writer
    def coverage(self) -> dict:
        """Counts by status + lifetime counters (Observatory readout)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT status, COUNT(*) FROM failed_attempts GROUP BY status"
            ).fetchall()
        by_status = {r[0]: int(r[1]) for r in rows}
        return {
            "by_status": by_status,
            "pending": by_status.get(ST_PENDING, 0),
            "in_progress": by_status.get(ST_IN_PROGRESS, 0),
            "resolved": by_status.get(ST_RESOLVED, 0),
            "abandoned": by_status.get(ST_ABANDONED, 0),
            "enqueued_total": self._enqueued,
            "revisits_handed": self._revisits_handed,
            "resolved_total": self._resolved,
            "abandoned_total": self._abandoned,
        }
