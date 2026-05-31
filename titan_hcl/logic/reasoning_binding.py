"""reasoning_binding.py — Meta-teacher → policy ReasoningBinding store (RFP_cgn_enhancements Phase G).

The meta-teacher produces high-quality diagnostic prose, but pre-G only a *scalar*
``reward_bonus`` (≤0.05) + a soft template-bias crossed to the meta-reasoning policy —
ε-greedy noise (0.25) dwarfed it and teacher quality scores declined (0.391→0.365).

Phase G (proto-SPEC §9.5c, LOCKED 2026-05-30) is the language-teacher analog: the teacher
accumulates a *curriculum of bindings* — "when the inner context looks like X, the proper
next primitive is Y" — and each matching binding contributes a **logit bias** to the policy's
primitive selection, exactly as the language teacher biases the language head in felt states
matching a taught word.

Cross-process design (no bus in the hot loop, no sync RPC — §11.5):
  * meta_teacher_worker is the SOLE writer (§G21 one-writer) → ``reasoning_bindings.db`` (WAL).
  * MetaReasoningEngine (cognitive_worker) is a cached read-only consumer: it recomputes the
    current ``context_signature`` each pre-step and retrieves top-k bindings by cosine sim.

The ``context_signature`` is **numeric only** (§11.4 non-linguistic inner loop — no
sentence-transformer): a normalized primitive-histogram of the chain-so-far concatenated with
process-stable hashed one-hots of (trigger_reason, dominant_emotion, domain). Both sides call
the same ``build_context_signature`` so mint-time and retrieval-time vectors live in one space.

Pure-logic module — no bus, no worker imports — so it is unit-testable in isolation.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Canonical signature geometry ────────────────────────────────────────────
# BINDING_PRIMITIVES MUST mirror meta_reasoning.META_PRIMITIVES (same order).
# Kept as a local copy to avoid importing the heavy meta_reasoning module into
# the teacher worker; test_cgn_phaseG asserts the two stay in lockstep.
BINDING_PRIMITIVES: Tuple[str, ...] = (
    "FORMULATE", "RECALL", "HYPOTHESIZE", "DELEGATE",
    "SYNTHESIZE", "EVALUATE", "BREAK", "SPIRIT_SELF", "INTROSPECT",
)
_PRIM_INDEX = {p: i for i, p in enumerate(BINDING_PRIMITIVES)}
N_PRIM = len(BINDING_PRIMITIVES)            # 9
N_TRIG_BUCKETS = 8
N_EMOT_BUCKETS = 8
N_DOM_BUCKETS = 8
# Concept-aware (Maker 2026-05-31): the grounding_concept the inner Titan is
# reasoning about (Phase A puts it on every learning-event chain) keys the
# binding too — so the teacher learns "when reasoning about THIS concept, use
# primitive Y", not just "in this context-shape". Highest-cardinality categorical
# → the most buckets. Empty concept (non-grounding chains) → bucket 0, so those
# chains behave exactly as the pre-concept-aware context-only binding.
N_CONCEPT_BUCKETS = 24
SIG_DIM = (N_PRIM + N_TRIG_BUCKETS + N_EMOT_BUCKETS + N_DOM_BUCKETS
           + N_CONCEPT_BUCKETS)             # 57

DEFAULT_DB_PATH = "data/meta_teacher/reasoning_bindings.db"

# Mint behaviour: a new binding MERGES into an existing one when the contexts are
# near-identical AND the recommended primitive matches — otherwise the curriculum
# would fragment into thousands of near-duplicates.
MINT_MERGE_THRESHOLD = 0.92
BASE_CONFIDENCE = 0.30
MAX_BINDINGS = 2000

# Curriculum (§9.5c / RFP G.iv): promote when productive ≥ 50% of receptive.
PRODUCED_PROMOTE_RATIO = 0.50
MAX_LEVEL = 9


def _stable_bucket(label: str, n_buckets: int) -> int:
    """Process-stable hash → bucket. Python's builtin hash() is PYTHONHASHSEED-salted
    and differs across processes — fatal here because the teacher (writer) and the
    policy (reader) run in separate processes and MUST agree on the signature."""
    if not label:
        return 0
    digest = hashlib.blake2b(label.strip().lower().encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % n_buckets


def build_context_signature(
    trigger_reason: str,
    dominant_emotion: str,
    chain_so_far: List[str],
    domain: str,
    grounding_concept: str = "",
) -> np.ndarray:
    """Deterministic numeric context signature (L2-normalized, float32, dim=SIG_DIM).

    Context + chain-prefix scope (Maker decision 2026-05-31): bindings can fire mid-chain
    (e.g. "after a FORMULATE loop, recommend EVALUATE") not just at entry — matches the
    §9.5c ``post_formulate_loop_breaker`` example. ``chain_so_far`` is the partial chain at
    retrieval-time and the full chain at mint-time; cosine sim degrades gracefully on the
    prefix overlap.

    Concept-aware (Maker 2026-05-31): ``grounding_concept`` (the concept the inner Titan is
    reasoning about — Phase A puts it on every learning-event chain) keys the binding too,
    via a process-stable hashed one-hot. Empty concept → bucket 0 (non-grounding chains
    behave exactly as the pre-concept-aware signature).
    """
    vec = np.zeros(SIG_DIM, dtype=np.float32)

    # [0:9] normalized primitive histogram of chain-so-far (base primitive only —
    # chain entries are compound "PRIMITIVE.subtype"; strip before lookup).
    if chain_so_far:
        counts = np.zeros(N_PRIM, dtype=np.float32)
        for step in chain_so_far:
            base = step.split(".", 1)[0] if step else ""
            idx = _PRIM_INDEX.get(base)
            if idx is not None:
                counts[idx] += 1.0
        total = counts.sum()
        if total > 0:
            vec[0:N_PRIM] = counts / total

    # trigger / emotion / domain / concept — process-stable hashed one-hots.
    off = N_PRIM
    vec[off + _stable_bucket(trigger_reason, N_TRIG_BUCKETS)] = 1.0
    off += N_TRIG_BUCKETS
    vec[off + _stable_bucket(dominant_emotion, N_EMOT_BUCKETS)] = 1.0
    off += N_EMOT_BUCKETS
    vec[off + _stable_bucket(domain, N_DOM_BUCKETS)] = 1.0
    off += N_DOM_BUCKETS
    # Empty concept lands in bucket 0 (shared across all non-grounding chains).
    vec[off + (_stable_bucket(grounding_concept, N_CONCEPT_BUCKETS)
               if grounding_concept else 0)] = 1.0

    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return vec


@dataclass
class ReasoningBinding:
    """One curriculum entry: in a given inner context, the taught next primitive."""
    binding_id: int
    context_signature: np.ndarray          # (SIG_DIM,) float32, L2-normalized
    recommended_primitive: str             # one of BINDING_PRIMITIVES
    recommended_sub_action: str            # e.g. "combine"; "" if unspecified
    principle_label: str                   # compact tag e.g. "post_formulate_loop_breaker"
    confidence: float                      # [0,1] — builds with corroboration
    n_taught: int = 0
    n_recognized: int = 0                  # matched context AND chose recommended
    n_produced: int = 0                    # chose recommended BEFORE teacher prompted
    level: int = 0                         # L0..L9 curriculum
    ts_created: float = 0.0
    ts_last_reinforced: float = 0.0

    @property
    def primitive_index(self) -> Optional[int]:
        return _PRIM_INDEX.get(self.recommended_primitive)


def _sig_to_blob(sig: np.ndarray) -> bytes:
    return np.asarray(sig, dtype=np.float32).tobytes()


def _blob_to_sig(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.shape[0] != SIG_DIM:
        # Defensive: a schema/geometry change invalidates old blobs — pad/truncate
        # so a stale row can't crash retrieval (it will simply match poorly).
        out = np.zeros(SIG_DIM, dtype=np.float32)
        out[: min(SIG_DIM, arr.shape[0])] = arr[:SIG_DIM]
        return out
    return arr.copy()


class ReasoningBindingStore:
    """Bounded, WAL-persisted curriculum of ReasoningBindings.

    ``read_only=True`` (the policy side) opens no writer client and serves cached,
    periodically-refreshed reads. The teacher side (``read_only=False``) routes writes
    through the universal writer when configured, else direct sqlite.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        writer_client=None,
        read_only: bool = False,
        refresh_interval_s: float = 60.0,
        max_bindings: int = MAX_BINDINGS,
    ):
        self._db_path = db_path
        self._read_only = read_only
        self._refresh_interval_s = float(refresh_interval_s)
        self._max_bindings = int(max_bindings)
        self._writer = writer_client

        # In-memory cache for cosine retrieval: parallel matrix + binding list.
        self._cache: List[ReasoningBinding] = []
        self._matrix: Optional[np.ndarray] = None     # (N, SIG_DIM)
        self._last_refresh_ts: float = 0.0

        self._ensure_dir()
        if not read_only:
            self._init_db()
            self._attach_writer(writer_client)
        self.refresh(force=True)

    # ── infra ───────────────────────────────────────────────────────────
    def _ensure_dir(self) -> None:
        d = os.path.dirname(self._db_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def _attach_writer(self, writer_client) -> None:
        if writer_client is not None:
            self._writer = writer_client
            return
        try:
            from titan_hcl.persistence.config import IMWConfig
            from titan_hcl.persistence.writer_client import get_client
            cfg = IMWConfig.from_titan_config_section("persistence_meta_teacher")
            if cfg.enabled and cfg.mode != "disabled":
                if cfg.db_path:
                    if os.path.realpath(cfg.db_path) != os.path.realpath(self._db_path):
                        return
                self._writer = get_client("reasoning_bindings", cfg=cfg)
                logger.info("[ReasoningBindingStore] routed via meta_teacher writer (mode=%s)", cfg.mode)
        except Exception as e:  # noqa: BLE001
            logger.debug("[ReasoningBindingStore] writer client unavailable, direct writes: %s", e)
            self._writer = None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS reasoning_bindings (
                    binding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_signature BLOB NOT NULL,
                    recommended_primitive TEXT NOT NULL,
                    recommended_sub_action TEXT DEFAULT '',
                    principle_label TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.3,
                    n_taught INTEGER DEFAULT 0,
                    n_recognized INTEGER DEFAULT 0,
                    n_produced INTEGER DEFAULT 0,
                    level INTEGER DEFAULT 0,
                    ts_created REAL DEFAULT 0.0,
                    ts_last_reinforced REAL DEFAULT 0.0
                );
                CREATE INDEX IF NOT EXISTS idx_rb_primitive
                    ON reasoning_bindings(recommended_primitive);
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _route_write(self, sql: str, params, *, table: str = "reasoning_bindings"):
        if self._writer is not None:
            result = self._writer.write(sql, params, table=table)
            return getattr(result, "last_row_id", None)
        conn = self._connect()
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    # ── cache / retrieval ────────────────────────────────────────────────
    def refresh(self, force: bool = False) -> None:
        """Reload the in-memory matrix from disk (rate-limited unless ``force``)."""
        now = time.time()
        if not force and (now - self._last_refresh_ts) < self._refresh_interval_s:
            return
        self._last_refresh_ts = now
        if not os.path.exists(self._db_path):
            self._cache, self._matrix = [], None
            return
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM reasoning_bindings ORDER BY binding_id"
                ).fetchall()
            finally:
                conn.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("[ReasoningBindingStore] refresh failed: %s", e)
            return
        cache: List[ReasoningBinding] = []
        for r in rows:
            cache.append(ReasoningBinding(
                binding_id=int(r["binding_id"]),
                context_signature=_blob_to_sig(r["context_signature"]),
                recommended_primitive=str(r["recommended_primitive"]),
                recommended_sub_action=str(r["recommended_sub_action"] or ""),
                principle_label=str(r["principle_label"] or ""),
                confidence=float(r["confidence"] or 0.0),
                n_taught=int(r["n_taught"] or 0),
                n_recognized=int(r["n_recognized"] or 0),
                n_produced=int(r["n_produced"] or 0),
                level=int(r["level"] or 0),
                ts_created=float(r["ts_created"] or 0.0),
                ts_last_reinforced=float(r["ts_last_reinforced"] or 0.0),
            ))
        self._cache = cache
        self._matrix = (
            np.stack([b.context_signature for b in cache]).astype(np.float32)
            if cache else None
        )

    def retrieve_topk(
        self, query_signature: np.ndarray, k: int = 3, sim_floor: float = 0.6,
    ) -> List[Tuple[ReasoningBinding, float]]:
        """Top-k bindings by cosine similarity (signatures are L2-normalized → dot product)."""
        if self._matrix is None or not self._cache:
            return []
        q = np.asarray(query_signature, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 1e-8:
            q = q / qn
        sims = self._matrix @ q   # (N,)
        if sims.size == 0:
            return []
        order = np.argsort(sims)[::-1][:k]
        out: List[Tuple[ReasoningBinding, float]] = []
        for i in order:
            s = float(sims[i])
            if s >= sim_floor:
                out.append((self._cache[int(i)], s))
        return out

    # ── mint / refine (writer side) ──────────────────────────────────────
    def mint_or_refine(
        self,
        context_signature: np.ndarray,
        recommended_primitive: str,
        recommended_sub_action: str = "",
        principle_label: str = "",
    ) -> int:
        """Mint a new binding or refine a near-identical existing one. Returns binding_id.

        Refine = same recommended primitive AND cosine sim ≥ MINT_MERGE_THRESHOLD:
        bump n_taught, corroborate confidence (asymptotic→1), refresh timestamp."""
        if recommended_primitive not in _PRIM_INDEX:
            raise ValueError(f"unknown primitive: {recommended_primitive}")
        self.refresh(force=True)
        sig = np.asarray(context_signature, dtype=np.float32)

        # Find a merge candidate among same-primitive bindings.
        best_id, best_sim = None, -1.0
        for b in self._cache:
            if b.recommended_primitive != recommended_primitive:
                continue
            s = float(np.dot(b.context_signature, sig))
            if s > best_sim:
                best_sim, best_id = s, b.binding_id

        now = time.time()
        if best_id is not None and best_sim >= MINT_MERGE_THRESHOLD:
            existing = next(b for b in self._cache if b.binding_id == best_id)
            new_conf = min(1.0, existing.confidence + (1.0 - existing.confidence) * 0.20)
            self._route_write(
                "UPDATE reasoning_bindings SET n_taught = n_taught + 1, "
                "confidence = ?, ts_last_reinforced = ?, recommended_sub_action = ?, "
                "principle_label = ? WHERE binding_id = ?",
                (round(new_conf, 4), now,
                 recommended_sub_action or existing.recommended_sub_action,
                 principle_label or existing.principle_label, best_id),
            )
            self.refresh(force=True)
            return best_id

        new_id = self._route_write(
            "INSERT INTO reasoning_bindings (context_signature, recommended_primitive, "
            "recommended_sub_action, principle_label, confidence, n_taught, n_recognized, "
            "n_produced, level, ts_created, ts_last_reinforced) "
            "VALUES (?, ?, ?, ?, ?, 1, 0, 0, 0, ?, ?)",
            (_sig_to_blob(sig), recommended_primitive, recommended_sub_action,
             principle_label, BASE_CONFIDENCE, now, now),
        )
        self._evict_if_needed()
        self.refresh(force=True)
        return int(new_id) if new_id is not None else -1

    def _evict_if_needed(self) -> None:
        """Bound the table: drop the lowest-value bindings (low confidence, stale, L0)."""
        try:
            conn = self._connect()
            try:
                (count,) = conn.execute(
                    "SELECT COUNT(*) FROM reasoning_bindings").fetchone()
            finally:
                conn.close()
        except Exception:
            return
        if count <= self._max_bindings:
            return
        overflow = count - self._max_bindings
        # Value = confidence + level + recent reinforcement; evict the weakest.
        self._route_write(
            "DELETE FROM reasoning_bindings WHERE binding_id IN ("
            "SELECT binding_id FROM reasoning_bindings "
            "ORDER BY (confidence + level * 0.1) ASC, ts_last_reinforced ASC LIMIT ?)",
            (int(overflow),),
        )

    # ── outcome counters + curriculum (G.iv) ─────────────────────────────
    def record_recognized(self, binding_id: int) -> None:
        """Chain matched this binding's context AND chose the recommended primitive."""
        self._route_write(
            "UPDATE reasoning_bindings SET n_recognized = n_recognized + 1, "
            "confidence = MIN(1.0, confidence + (1.0 - confidence) * 0.05), "
            "ts_last_reinforced = ? WHERE binding_id = ?",
            (time.time(), int(binding_id)),
        )
        self._maybe_promote(int(binding_id))

    def record_produced(self, binding_id: int) -> None:
        """Chain chose the recommended primitive BEFORE any teacher prompt (spontaneous)."""
        self._route_write(
            "UPDATE reasoning_bindings SET n_produced = n_produced + 1, "
            "n_recognized = n_recognized + 1, "
            "confidence = MIN(1.0, confidence + (1.0 - confidence) * 0.08), "
            "ts_last_reinforced = ? WHERE binding_id = ?",
            (time.time(), int(binding_id)),
        )
        self._maybe_promote(int(binding_id))

    def _maybe_promote(self, binding_id: int) -> None:
        """Promote L_k→L_{k+1} when n_produced ≥ 0.5 × n_recognized (and ≥1 produced)."""
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT n_recognized, n_produced, level FROM reasoning_bindings "
                    "WHERE binding_id = ?", (binding_id,)).fetchone()
            finally:
                conn.close()
        except Exception:
            return
        if row is None:
            return
        n_rec, n_prod, level = int(row[0]), int(row[1]), int(row[2])
        if level >= MAX_LEVEL or n_prod < 1:
            return
        if n_prod >= PRODUCED_PROMOTE_RATIO * max(n_rec, 1):
            self._route_write(
                "UPDATE reasoning_bindings SET level = MIN(?, level + 1) "
                "WHERE binding_id = ?", (MAX_LEVEL, binding_id))

    # ── introspection ────────────────────────────────────────────────────
    def count(self) -> int:
        return len(self._cache)

    def all(self) -> List[ReasoningBinding]:
        return list(self._cache)

    def telemetry(self) -> dict:
        if not self._cache:
            return {"n_bindings": 0, "avg_confidence": 0.0, "avg_level": 0.0,
                    "total_taught": 0, "total_recognized": 0, "total_produced": 0}
        n = len(self._cache)
        return {
            "n_bindings": n,
            "avg_confidence": round(sum(b.confidence for b in self._cache) / n, 4),
            "avg_level": round(sum(b.level for b in self._cache) / n, 2),
            "total_taught": sum(b.n_taught for b in self._cache),
            "total_recognized": sum(b.n_recognized for b in self._cache),
            "total_produced": sum(b.n_produced for b in self._cache),
            "max_level": max(b.level for b in self._cache),
        }
