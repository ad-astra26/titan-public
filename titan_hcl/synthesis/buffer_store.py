"""ActrBufferStore — sole writer of `actr_buffers` on `synthesis.duckdb`.

INV-Syn-16 (Phase 7, D-SPEC-PHASE7): the `synthesis_worker` process is the
only writer to `actr_buffers`. The agno_worker holds a `BufferCache` for fast
per-chat read + write during an active chat and emits `SYNTHESIS_BUFFER_COMMAND`
bus events on every write; this module's `persist()` / `clear()` are the
sole landing surface for those commands inside synthesis_worker.

Schema (matches SPEC §25.6):

    CREATE TABLE actr_buffers (
        chat_id        TEXT    NOT NULL,
        buffer_name    TEXT    NOT NULL,
        content        TEXT,
        concept_ids    TEXT,              -- JSON array
        embedding_hash TEXT,              -- sha256(canonical_json({content, concept_ids}))
        updated_at     DOUBLE  NOT NULL,
        PRIMARY KEY (chat_id, buffer_name)
    );

Cross-process surface = `data/buffers_snapshot.json` (atomic tmp+rename;
mirrors P4 spine_snapshot / P5 forks_snapshot). The api process reads the
snapshot for `/v6/synthesis/buffers/*`; agno_worker reads it on first cold
access via `BufferCache._hydrate(chat_id)`.

In-process spreading-activation reader: `buffer_entities(chat_id)` returns
the union of `concept_ids` across the 4 rows for that chat (INV-Syn-18).
Replaces `BufferStub.current_entities()` in P7.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

from titan_hcl.synthesis.writer import on_writer, resolve_writer

logger = logging.getLogger(__name__)


BUFFER_NAMES: tuple[str, ...] = ("goal", "retrieval", "imaginal", "perception")

DEFAULT_MAX_ENTITIES: int = 20

# Content fields capped to a reasonable ceiling to keep DuckDB rows compact and
# the snapshot file under the watermark cap. 8 KiB matches the user-message
# truncation applied in the pre-LLM goal hook.
MAX_CONTENT_BYTES: int = 8192

# Degradation observability (D-SPEC-154). A healthy actr_buffers persist is
# sub-millisecond; sustained latency growth is the early signature of DuckDB
# ART-index churn re-accruing (the precursor — days ahead — of the FATAL
# crash-loop the index-removal fix targets). When the rolling-avg persist
# latency crosses the threshold we fire the `on_degraded` callback → the
# synthesis_worker emits a WARN MODULE_ERROR onto the kernel journal (the
# SPEC error cascade, INV-SDA-12). Rate-limited so the journal isn't flooded.
PERSIST_LATENCY_WINDOW: int = 128
PERSIST_DEGRADE_AVG_MS: float = 50.0      # ~10× a healthy persist — a clear creep
DEGRADE_EMIT_COOLDOWN_S: float = 300.0    # at most one degradation alert / 5 min
PERSIST_DEGRADE_MIN_SAMPLES: int = 20


class _DuckDBConnLike:
    """Structural protocol for the DuckDB connection injected by tests."""
    def execute(self, sql: str, params: Any = None) -> Any: ...
    def fetchall(self) -> list: ...


def _canonical_payload_hash(content: str, concept_ids: list[str]) -> str:
    """sha256(canonical_json({content, concept_ids})). Content fingerprint —
    NOT a FAISS vector hash (FAISS embeddings are computed on demand from
    content; storing a vector hash here would force a FAISS call per write)."""
    payload = json.dumps(
        {"content": content, "concept_ids": list(concept_ids)},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _truncate(content: str) -> str:
    """Cap content at MAX_CONTENT_BYTES (UTF-8 safe)."""
    if not content:
        return ""
    raw = content.encode("utf-8")
    if len(raw) <= MAX_CONTENT_BYTES:
        return content
    return raw[:MAX_CONTENT_BYTES].decode("utf-8", errors="ignore")


class ActrBufferStore:
    """Sole writer of `actr_buffers`. INV-Syn-16.

    Constructor params:
      duckdb_conn: open connection to `synthesis.duckdb` (sole-writer).
      snapshot_path: path to `data/buffers_snapshot.json` (atomic
                     tmp+rename written after every persist/clear).
      clock: time source (overridable for tests).
    """

    def __init__(
        self,
        *,
        duckdb_conn: _DuckDBConnLike,
        snapshot_path: str | os.PathLike,
        clock: Any = time.time,
        writer: Any = None,
        on_degraded: Optional[Any] = None,
    ):
        self._db = duckdb_conn
        self._snapshot_path = str(snapshot_path)
        self._clock = clock
        # Degradation observability (D-SPEC-154): rolling persist-latency window;
        # a sustained rise → `on_degraded(avg_ms, max_ms, samples)` → the worker
        # emits a WARN MODULE_ERROR (SPEC cascade). None (tests/default) = off.
        self._on_degraded = on_degraded
        self._persist_latencies: deque = deque(maxlen=PERSIST_LATENCY_WINDOW)
        self._max_persist_ms = 0.0
        self._slow_persists = 0
        self._last_degrade_emit = 0.0
        # Single-writer-thread (Option C): every DuckDB op runs on the one
        # SynthesisWriter thread (the @on_writer methods below) → the writer IS
        # the serializer. INV-Syn-16 sole writer preserved. Tests inject no
        # writer → InlineWriter runs ops inline.
        self._writer = resolve_writer(writer)
        # Vestigial after Option C (the writer thread already serializes); kept
        # as a reentrant no-contention guard so the method bodies are unchanged
        # and a nested store call on the writer thread never self-deadlocks.
        self._lock = threading.RLock()
        # Cheap counter exposed via `stats()` for the A.7 acceptance gate
        # (synthesis_worker counter cross-checks agno emit counter).
        self._writes_seen = 0
        self._clears_seen = 0
        self._init_schema()

    # ── Schema bootstrap ────────────────────────────────────────────────

    @on_writer
    def _init_schema(self) -> None:
        """CREATE TABLE IF NOT EXISTS actr_buffers (idempotent)."""
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS actr_buffers ("
                "  chat_id        TEXT    NOT NULL,"
                "  buffer_name    TEXT    NOT NULL,"
                "  content        TEXT,"
                "  concept_ids    TEXT,"
                "  embedding_hash TEXT,"
                "  updated_at     DOUBLE  NOT NULL,"
                "  PRIMARY KEY (chat_id, buffer_name)"
                ")"
            )
            # Root-cause fix (D-SPEC-154 / INV-Syn-30): NO secondary ART indexes
            # on this high-churn UPSERT'd table. `idx_actr_buffers_updated` churned
            # the ART tree on EVERY write (updated_at changes each persist) and was
            # never queried; `idx_actr_buffers_chat` is redundant with the PK prefix
            # (PK = (chat_id, buffer_name)). DuckDB secondary indexes degrade
            # pathologically under sustained UPSERT churn → the runtime FATAL
            # crash-loop (reproduced 2026-06-09: ~2.5× slowdown WITH indexes, clean
            # + fast WITHOUT). The 2026-06-01 INSERT-OR-REPLACE→ON-CONFLICT fix only
            # closed the PK churn, not these. actr_buffers is tiny (4 buffers ×
            # active chats) → the PK index covers every query. DROP IF EXISTS
            # self-heals existing fleet DBs (removes the corruption source on boot).
            for _idx in ("idx_actr_buffers_chat", "idx_actr_buffers_updated"):
                try:
                    self._db.execute(f"DROP INDEX IF EXISTS {_idx}")
                except Exception:  # pragma: no cover — defensive
                    pass

    # ── Write surface (INV-Syn-16) ──────────────────────────────────────

    @on_writer
    def persist(
        self,
        *,
        chat_id: str,
        buffer_name: str,
        content: str,
        concept_ids: list[str],
        ts: Optional[float] = None,
    ) -> None:
        """INSERT OR REPLACE one row + re-export the snapshot.

        Raises ValueError on unknown buffer_name or empty chat_id —
        agno's BufferCache validates ahead of time, so a raise here is a
        programmer-error signal worth surfacing (caller logs at WARN).
        """
        if not chat_id:
            raise ValueError("chat_id must be non-empty")
        if buffer_name not in BUFFER_NAMES:
            raise ValueError(
                f"buffer_name must be one of {BUFFER_NAMES}; got {buffer_name!r}"
            )
        content = _truncate(content or "")
        concept_ids = list(concept_ids or [])
        emb_hash = _canonical_payload_hash(content, concept_ids)
        ts_val = float(ts) if ts is not None else float(self._clock())
        concepts_json = json.dumps(concept_ids, ensure_ascii=False)
        _t0 = time.perf_counter()
        with self._lock:
            # In-place UPSERT (ON CONFLICT DO UPDATE, NOT `INSERT OR REPLACE` —
            # OR REPLACE is DELETE+INSERT, re-touching the PK index every write).
            # The table also carries NO secondary ART indexes (removed D-SPEC-154 /
            # INV-Syn-30): they churned on every UPSERT (updated_at changes each
            # write) and were the runtime corruption source behind the
            # actr_buffers FATAL crash-loop. PK-only + in-place update = no ART
            # churn. INV-Syn-16 (sole writer) preserved.
            self._db.execute(
                "INSERT INTO actr_buffers "
                "(chat_id, buffer_name, content, concept_ids, embedding_hash, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (chat_id, buffer_name) DO UPDATE SET "
                "content = excluded.content, "
                "concept_ids = excluded.concept_ids, "
                "embedding_hash = excluded.embedding_hash, "
                "updated_at = excluded.updated_at",
                [chat_id, buffer_name, content, concepts_json, emb_hash, ts_val],
            )
            self._writes_seen += 1
            # Degradation observability (D-SPEC-154) — record persist latency.
            _lat_ms = (time.perf_counter() - _t0) * 1000.0
            self._persist_latencies.append(_lat_ms)
            if _lat_ms > self._max_persist_ms:
                self._max_persist_ms = _lat_ms
            if _lat_ms > PERSIST_DEGRADE_AVG_MS:
                self._slow_persists += 1
        self._maybe_emit_degraded()
        self.snapshot_export()

    def _maybe_emit_degraded(self) -> None:
        """Fire `on_degraded` when the rolling-avg persist latency crosses the
        threshold (rate-limited). The early-warning signature of ART-index churn
        re-accruing — days ahead of the FATAL crash (D-SPEC-154). Soft: never
        raises into the persist path."""
        if self._on_degraded is None:
            return
        with self._lock:
            n = len(self._persist_latencies)
            if n < PERSIST_DEGRADE_MIN_SAMPLES:
                return
            avg = sum(self._persist_latencies) / n
            now = float(self._clock())
            if avg <= PERSIST_DEGRADE_AVG_MS or (now - self._last_degrade_emit) < DEGRADE_EMIT_COOLDOWN_S:
                return
            self._last_degrade_emit = now
            mx = self._max_persist_ms
        try:
            self._on_degraded(avg_ms=avg, max_ms=mx, samples=n)
        except Exception as e:  # pragma: no cover — defensive
            logger.debug("[ActrBufferStore] on_degraded callback raised: %s", e)

    @on_writer
    def clear(self, *, chat_id: str, buffer_name: str) -> None:
        """DELETE one row + re-export the snapshot. Idempotent — clearing
        an absent row is not an error (agno may emit clear without prior set)."""
        if not chat_id:
            raise ValueError("chat_id must be non-empty")
        if buffer_name not in BUFFER_NAMES:
            raise ValueError(
                f"buffer_name must be one of {BUFFER_NAMES}; got {buffer_name!r}"
            )
        with self._lock:
            self._db.execute(
                "DELETE FROM actr_buffers WHERE chat_id = ? AND buffer_name = ?",
                [chat_id, buffer_name],
            )
            self._clears_seen += 1
        self.snapshot_export()

    # ── Read surface ────────────────────────────────────────────────────

    @on_writer
    def read_all_for_chat(self, chat_id: str) -> dict[str, dict]:
        """Return {buffer_name: {content, concept_ids, embedding_hash, updated_at}}.

        Missing buffers are simply absent from the dict (no stub rows)."""
        if not chat_id:
            return {}
        with self._lock:
            rows = self._db.execute(
                "SELECT buffer_name, content, concept_ids, embedding_hash, updated_at "
                "FROM actr_buffers WHERE chat_id = ?",
                [chat_id],
            ).fetchall()
        out: dict[str, dict] = {}
        for buf, content, concepts_json, emb_hash, updated_at in rows:
            try:
                cids = json.loads(concepts_json) if concepts_json else []
            except (TypeError, ValueError):
                cids = []
            out[buf] = {
                "content": content or "",
                "concept_ids": list(cids) if isinstance(cids, list) else [],
                "embedding_hash": emb_hash or "",
                "updated_at": float(updated_at) if updated_at is not None else 0.0,
            }
        return out

    def buffer_entities(
        self,
        chat_id: str,
        *,
        cap: int = DEFAULT_MAX_ENTITIES,
    ) -> list[str]:
        """INV-Syn-18: union of `concept_ids` across the 4 buffer rows for
        `chat_id`, dedup'd preserving first-seen order, capped at `cap`.

        Order = goal → perception → retrieval → imaginal (matches ACT-R
        spreading intuition: explicit goal beats incidental observation;
        retrieved memories beat freeform scratchpad). Replaces
        `BufferStub.current_entities()` (deleted in P7)."""
        rows = self.read_all_for_chat(chat_id)
        ordered = ("goal", "perception", "retrieval", "imaginal")
        out: list[str] = []
        seen: set[str] = set()
        for buf in ordered:
            row = rows.get(buf)
            if not row:
                continue
            for cid in row.get("concept_ids") or []:
                if not cid or not isinstance(cid, str):
                    continue
                if cid in seen:
                    continue
                seen.add(cid)
                out.append(cid)
                if len(out) >= cap:
                    return out
        return out

    # ── Snapshot export ────────────────────────────────────────────────

    @on_writer
    def _build_snapshot_payload(self) -> dict:
        """Build the full snapshot dict. SELECT every row, group by chat_id."""
        with self._lock:
            rows = self._db.execute(
                "SELECT chat_id, buffer_name, content, concept_ids, "
                "embedding_hash, updated_at FROM actr_buffers "
                "ORDER BY chat_id, buffer_name"
            ).fetchall()
            writes_seen = self._writes_seen
            clears_seen = self._clears_seen
        chats: dict[str, dict] = {}
        for chat_id, buf, content, concepts_json, emb_hash, updated_at in rows:
            try:
                cids = json.loads(concepts_json) if concepts_json else []
            except (TypeError, ValueError):
                cids = []
            chats.setdefault(chat_id, {})[buf] = {
                "content": content or "",
                "concept_ids": list(cids) if isinstance(cids, list) else [],
                "embedding_hash": emb_hash or "",
                "updated_at": float(updated_at) if updated_at is not None else 0.0,
            }
        return {
            "version": 1,
            "ts": float(self._clock()),
            "writes_seen": writes_seen,
            "clears_seen": clears_seen,
            "chat_count": len(chats),
            "chats": chats,
        }

    def snapshot_export(self) -> str:
        """Atomic tmp+rename write of `buffers_snapshot.json`.

        Returns the path actually written. Soft-fail (logs WARN + returns
        the intended path) on filesystem errors so the persist surface
        does not raise on transient disk issues — the next persist will
        retry the export automatically."""
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
                "[ActrBufferStore] snapshot export failed (%s): %s",
                target, e,
            )
        return target

    # ── Observability ───────────────────────────────────────────────────

    def stats(self) -> dict:
        """Counters for the A.7 acceptance gate + Observatory."""
        with self._lock:
            n = len(self._persist_latencies)
            avg = (sum(self._persist_latencies) / n) if n else 0.0
            return {
                "writes_seen": self._writes_seen,
                "clears_seen": self._clears_seen,
                "snapshot_path": self._snapshot_path,
                # D-SPEC-154 degradation observability — flat = healthy; a rising
                # avg signals ART-index churn re-accruing (the FATAL precursor).
                "persist_avg_ms": round(avg, 3),
                "persist_max_ms": round(self._max_persist_ms, 3),
                "slow_persists": self._slow_persists,
            }


__all__ = (
    "ActrBufferStore",
    "BUFFER_NAMES",
    "DEFAULT_MAX_ENTITIES",
    "MAX_CONTENT_BYTES",
)
