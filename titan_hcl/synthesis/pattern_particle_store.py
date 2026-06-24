"""PatternParticleStore — the proto-BRAIN verified-transition + particle store.

RFP_pattern_logic.md §1.2 / §7.1 (LOCKED 2026-06-24). Owner / G21 sole-writer:
the `pattern_logic_worker` process. Storage: `data/pattern_logic.duckdb` — its OWN
dedicated DB (DuckDB 1.5+ holds an exclusive file lock; no other process opens it,
arch §0 v0.5.0 lesson 1). The worker is single-threaded over its recv loop, so this
store is single-threaded too — no writer-thread machinery needed (cf.
StandingBundleStore's cross-process snapshot dance, which we do NOT need: the OFFER
path reaches synthesis/CGN through reasoning-composites + a φ(s) feature, NOT by a
cross-process read of this DB).

Two tables:
  • `transitions` — append-only log of normalized verified-transitions
        (signature × operation × frame × verdict × substrate). The raw evidence.
  • `particles`  — PATTERN (low-c clustered hunch) / MODEL (high-c oracle-verified
        rule), BRAIN §3.4-aligned: signature(proto-hv) · operation(proto-link) ·
        frame · Beta(α,β)→(f,c) · time_cost · use_count · lineage · verdict_history.

Lifecycle (BRAIN §4 mutate-not-update):
  • evidence merge moves the SCALARS (α,β,f,c) in place — BRAIN §3.4 "c rises on new
    evidence" is scalar movement, not a new particle.
  • PROMOTION (PATTERN→MODEL) is a LIFECYCLE mutation → a SUCCESSOR row (new id,
    parent_id=old), parent marked SUPERSEDED. The chain + tombstone are preserved.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import duckdb

logger = logging.getLogger(__name__)

__all__ = [
    "PatternParticleStore",
    "beta_to_f",
    "beta_to_c",
    "PARTICLE_SCHEMA_VERSION",
]

PARTICLE_SCHEMA_VERSION = 1

# Beta priors (α0=β0=1 — uniform). Evidence mass n = (α+β) − (α0+β0).
_ALPHA0 = 1.0
_BETA0 = 1.0


def beta_to_f(alpha: float, beta: float) -> float:
    """Truth fraction f = α/(α+β) — the posterior mean of Beta(α,β)."""
    denom = alpha + beta
    return float(alpha / denom) if denom > 0 else 0.0


def beta_to_c(alpha: float, beta: float, c0: float) -> float:
    """Confidence c = how much total evidence backs it (BRAIN §3.4 semantics).

    c = n/(n + c0), where n = (α+β) − (α0+β0) is the observed-evidence mass beyond
    the uniform prior. Saturating in [0,1): more evidence → higher c. With c0=1,
    n=5 → c≈0.83, n=6 → c≈0.86 (pairs with the Q2 ≥5-transitions / c≥0.85 gate).
    """
    n = max(0.0, (alpha + beta) - (_ALPHA0 + _BETA0))
    return float(n / (n + c0)) if (n + c0) > 0 else 0.0


def _vec_to_blob(vec: Sequence[float]) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _blob_to_vec(blob: Optional[bytes]) -> np.ndarray:
    if not blob:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32)


class PatternParticleStore:
    """Sole-writer (pattern_logic_worker, G21) persistence for verified-transitions
    + pattern/model particles.

    `c0` (confidence saturation constant), `promote_floor`, `min_transitions`,
    `f_floor` are passed from config (`[pattern_logic]`) — NOT hardcoded floors
    (INV: emergence over determinism). Defaults match RFP §6/Q2.
    """

    def __init__(
        self,
        db_path: str,
        *,
        conn: Any = None,
        c0: float = 1.0,
        promote_floor: float = 0.85,
        min_transitions: int = 5,
        f_floor: float = 0.7,
    ) -> None:
        self._db_path = db_path
        self._c0 = float(c0)
        self._promote_floor = float(promote_floor)
        self._min_transitions = int(min_transitions)
        self._f_floor = float(f_floor)
        self._owns_conn = conn is None

        if conn is not None:
            self._conn = conn
        else:
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
            self._conn = duckdb.connect(db_path)
        self._init_schema()

        # Monotonic id counters seeded from the DB (single-writer → race-free).
        self._next_tx_id = self._max_id("transitions") + 1
        self._particle_seq = self._max_particle_seq() + 1

    # ── schema ────────────────────────────────────────────────────────────
    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transitions (
                id           BIGINT PRIMARY KEY,
                signature    BLOB,
                operation    TEXT NOT NULL,
                frame        TEXT NOT NULL,
                verdict      BOOLEAN NOT NULL,
                substrate    TEXT NOT NULL,   -- 'outer' | 'inner'
                source       TEXT NOT NULL,   -- producing stream label
                context_label TEXT,           -- human-readable context (debug/G-OUT)
                particle_id  TEXT,            -- which particle it fed (NULL until clustered)
                ts           DOUBLE NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS particles (
                id           TEXT PRIMARY KEY,
                parent_id    TEXT,
                kind         TEXT NOT NULL,    -- 'PATTERN' | 'MODEL'
                status       TEXT NOT NULL,    -- 'ACTIVE' | 'SUPERSEDED'
                signature    BLOB,             -- centroid embedding (proto-hv)
                operation    TEXT NOT NULL,
                frame        TEXT NOT NULL,
                alpha        DOUBLE NOT NULL,
                beta         DOUBLE NOT NULL,
                f            DOUBLE NOT NULL,
                c            DOUBLE NOT NULL,
                time_cost    DOUBLE NOT NULL DEFAULT 0.0,
                use_count    INTEGER NOT NULL DEFAULT 0,
                n_sources    INTEGER NOT NULL DEFAULT 0,
                verdict_history BLOB,          -- json: [[source, verdict, ts], ...]
                created_ts   DOUBLE NOT NULL,
                updated_ts   DOUBLE NOT NULL
            )
            """
        )

    def _max_id(self, table: str) -> int:
        row = self._conn.execute(f"SELECT MAX(id) FROM {table}").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def _max_particle_seq(self) -> int:
        rows = self._conn.execute("SELECT id FROM particles").fetchall()
        best = 0
        for (pid,) in rows:
            # id shape: 'pl_<seq>[_v<version>]'
            try:
                seq = int(str(pid).split("_")[1])
                best = max(best, seq)
            except (IndexError, ValueError):
                continue
        return best

    # ── transitions (OBSERVE) ───────────────────────────────────────────────
    def record_transition(
        self,
        *,
        signature: Sequence[float],
        operation: str,
        frame: str,
        verdict: bool,
        substrate: str,
        source: str,
        ts: Optional[float] = None,
        context_label: str = "",
    ) -> int:
        """Append one normalized verified-transition. Returns its row id."""
        tx_id = self._next_tx_id
        self._next_tx_id += 1
        self._conn.execute(
            "INSERT INTO transitions (id, signature, operation, frame, verdict, "
            "substrate, source, context_label, particle_id, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
            (tx_id, _vec_to_blob(signature), operation, frame, bool(verdict),
             substrate, source, context_label, float(ts if ts is not None else time.time())),
        )
        return tx_id

    def recent_transitions(self, *, limit: Optional[int] = None,
                           only_unclustered: bool = False) -> List[Dict[str, Any]]:
        """Read transitions newest-first for the RECOGNISE clustering pass."""
        q = ("SELECT id, signature, operation, frame, verdict, substrate, source, "
             "context_label, particle_id, ts FROM transitions")
        if only_unclustered:
            q += " WHERE particle_id IS NULL"
        q += " ORDER BY id DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        rows = self._conn.execute(q).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "id": r[0], "signature": _blob_to_vec(r[1]), "operation": r[2],
                "frame": r[3], "verdict": bool(r[4]), "substrate": r[5],
                "source": r[6], "context_label": r[7], "particle_id": r[8], "ts": r[9],
            })
        return out

    def _attach_transitions(self, tx_ids: Sequence[int], particle_id: str) -> None:
        if not tx_ids:
            return
        self._conn.executemany(
            "UPDATE transitions SET particle_id = ? WHERE id = ?",
            [(particle_id, int(i)) for i in tx_ids],
        )

    # ── particles (RECOGNISE → CONSTRUCT) ───────────────────────────────────
    def propose_pattern(
        self,
        *,
        signature: Sequence[float],
        operation: str,
        frame: str,
        evidence: Sequence[Dict[str, Any]],
        n_sources: int,
        tx_ids: Optional[Sequence[int]] = None,
    ) -> str:
        """Create a PATTERN (low-c) from a recognised cluster.

        `evidence` = the contributing transitions [{verdict, source, ts}, ...]; the
        Beta(α,β) is seeded by tallying TRUE→α, FALSE→β over the cluster.
        """
        alpha, beta = _ALPHA0, _BETA0
        history: List[list] = []
        for e in evidence:
            if e["verdict"]:
                alpha += 1.0
            else:
                beta += 1.0
            history.append([e.get("source", ""), bool(e["verdict"]), float(e.get("ts", 0.0))])
        pid = f"pl_{self._particle_seq}"
        self._particle_seq += 1
        now = time.time()
        f = beta_to_f(alpha, beta)
        c = beta_to_c(alpha, beta, self._c0)
        self._conn.execute(
            "INSERT INTO particles (id, parent_id, kind, status, signature, operation, "
            "frame, alpha, beta, f, c, time_cost, use_count, n_sources, verdict_history, "
            "created_ts, updated_ts) VALUES (?, NULL, 'PATTERN', 'ACTIVE', ?, ?, ?, ?, ?, "
            "?, ?, 0.0, 0, ?, ?, ?, ?)",
            (pid, _vec_to_blob(signature), operation, frame, alpha, beta, f, c,
             int(n_sources), json.dumps(history), now, now),
        )
        self._attach_transitions(tx_ids or [], pid)
        return pid

    def merge_evidence(self, particle_id: str, *, verdict: bool, source: str,
                       ts: Optional[float] = None) -> Dict[str, Any]:
        """Fold one new verified-transition into a particle's Beta(α,β) IN PLACE
        (scalar movement — BRAIN §3.4). Returns the updated {f, c, alpha, beta}."""
        row = self._conn.execute(
            "SELECT alpha, beta, verdict_history FROM particles WHERE id = ?",
            (particle_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"particle {particle_id} not found")
        alpha, beta = float(row[0]), float(row[1])
        history = json.loads(row[2]) if row[2] else []
        if verdict:
            alpha += 1.0
        else:
            beta += 1.0
        ts = float(ts if ts is not None else time.time())
        history.append([source, bool(verdict), ts])
        f = beta_to_f(alpha, beta)
        c = beta_to_c(alpha, beta, self._c0)
        self._conn.execute(
            "UPDATE particles SET alpha = ?, beta = ?, f = ?, c = ?, "
            "verdict_history = ?, updated_ts = ? WHERE id = ?",
            (alpha, beta, f, c, json.dumps(history), time.time(), particle_id),
        )
        return {"f": f, "c": c, "alpha": alpha, "beta": beta}

    def eligible_for_promotion(self, particle_id: str) -> bool:
        """Q2 gate: c ≥ promote_floor AND evidence-mass ≥ min_transitions AND the op
        is reliably TRUE (f ≥ f_floor — you only 'apply' an op that works)."""
        row = self._conn.execute(
            "SELECT kind, alpha, beta, f, c FROM particles WHERE id = ?",
            (particle_id,),
        ).fetchone()
        if row is None or row[0] != "PATTERN":
            return False
        alpha, beta, f, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        n = (alpha + beta) - (_ALPHA0 + _BETA0)
        return (c >= self._promote_floor and n >= self._min_transitions
                and f >= self._f_floor)

    def promote_to_model(self, particle_id: str) -> str:
        """Mutate-not-update: insert a MODEL successor (new id, parent_id=old),
        mark the PATTERN parent SUPERSEDED. Returns the new model id."""
        row = self._conn.execute(
            "SELECT signature, operation, frame, alpha, beta, f, c, time_cost, "
            "use_count, n_sources, verdict_history FROM particles WHERE id = ?",
            (particle_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"particle {particle_id} not found")
        seq = particle_id.split("_")[1]
        model_id = f"pl_{seq}_v{int(time.time())}"
        now = time.time()
        self._conn.execute(
            "INSERT INTO particles (id, parent_id, kind, status, signature, operation, "
            "frame, alpha, beta, f, c, time_cost, use_count, n_sources, verdict_history, "
            "created_ts, updated_ts) VALUES (?, ?, 'MODEL', 'ACTIVE', ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?)",
            (model_id, particle_id, row[0], row[1], row[2], float(row[3]), float(row[4]),
             float(row[5]), float(row[6]), float(row[7]), int(row[8]), int(row[9]),
             row[10], now, now),
        )
        self._conn.execute(
            "UPDATE particles SET status = 'SUPERSEDED', updated_ts = ? WHERE id = ?",
            (now, particle_id),
        )
        return model_id

    def cite_model(self, particle_id: str) -> int:
        """Record a reuse (G-REUSE). Increments use_count, returns the new count."""
        self._conn.execute(
            "UPDATE particles SET use_count = use_count + 1, updated_ts = ? WHERE id = ?",
            (time.time(), particle_id),
        )
        row = self._conn.execute(
            "SELECT use_count FROM particles WHERE id = ?", (particle_id,)).fetchone()
        return int(row[0]) if row else 0

    # ── reads (OFFER / cache / observability) ───────────────────────────────
    def active_particles(self, *, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT id, kind, signature, operation, frame, f, c, use_count, n_sources " \
            "FROM particles WHERE status = 'ACTIVE'"
        params: List[Any] = []
        if kind:
            q += " AND kind = ?"
            params.append(kind)
        rows = self._conn.execute(q, params).fetchall()
        return [{
            "id": r[0], "kind": r[1], "signature": _blob_to_vec(r[2]), "operation": r[3],
            "frame": r[4], "f": r[5], "c": r[6], "use_count": r[7], "n_sources": r[8],
        } for r in rows]

    def get_models(self, *, min_c: Optional[float] = None) -> List[Dict[str, Any]]:
        """Active MODEL particles — feeds the OFFER + the inner model-sig cache."""
        models = [p for p in self.active_particles(kind="MODEL")
                  if min_c is None or p["c"] >= min_c]
        return models

    def get_particle(self, particle_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT id, parent_id, kind, status, signature, operation, frame, alpha, "
            "beta, f, c, time_cost, use_count, n_sources, verdict_history, created_ts, "
            "updated_ts FROM particles WHERE id = ?", (particle_id,)).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "parent_id": row[1], "kind": row[2], "status": row[3],
            "signature": _blob_to_vec(row[4]), "operation": row[5], "frame": row[6],
            "alpha": row[7], "beta": row[8], "f": row[9], "c": row[10],
            "time_cost": row[11], "use_count": row[12], "n_sources": row[13],
            "verdict_history": json.loads(row[14]) if row[14] else [],
            "created_ts": row[15], "updated_ts": row[16],
        }

    def get_stats(self) -> Dict[str, Any]:
        def _count(where: str) -> int:
            r = self._conn.execute(f"SELECT COUNT(*) FROM particles WHERE {where}").fetchone()
            return int(r[0]) if r else 0
        tx = self._conn.execute("SELECT COUNT(*) FROM transitions").fetchone()
        return {
            "transitions": int(tx[0]) if tx else 0,
            "patterns_active": _count("kind='PATTERN' AND status='ACTIVE'"),
            "models_active": _count("kind='MODEL' AND status='ACTIVE'"),
            "models_cited": _count("kind='MODEL' AND use_count>0"),
            "superseded": _count("status='SUPERSEDED'"),
        }

    def close(self) -> None:
        if self._owns_conn:
            try:
                self._conn.close()
            except Exception:
                logger.exception("[PatternParticleStore] close failed")
