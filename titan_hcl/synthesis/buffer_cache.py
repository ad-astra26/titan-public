"""BufferCache — agno_worker-side per-chat in-mem buffer cache.

SPEC §25.6 / D-SPEC-PHASE7 (v1.66.0): the agno_worker process holds an
in-process dict of `{chat_id: {buffer_name: {content, concept_ids, ts}}}`
for fast read + write on the active chat. Sole DuckDB writer remains
`synthesis_worker` (INV-Syn-16) — every cache write publishes a
`SYNTHESIS_BUFFER_COMMAND` bus event so persistence is durable.

Read path:
  - **Active chat**: `BufferCache.get(chat_id, buf)` reads from the dict;
    if `chat_id` has never been seen, the cache lazily hydrates from
    `data/buffers_snapshot.json` (atomic JSON, written by synthesis_worker).
  - **Cold chat (post-restart)**: same path; the snapshot read happens
    once per `chat_id` per process lifetime.
  - **Stale chat** (last touched > `hydration_warm_threshold_s`): hydration
    skipped — treat as a fresh chat, no spurious DuckDB-snapshot reads
    for week-old session_ids.

Write path:
  - `BufferCache.set(chat_id, buf, content=, concept_ids=)` → updates the
    local dict + emits `SYNTHESIS_BUFFER_COMMAND op="set"` via the
    bus_emit callable.
  - `BufferCache.clear(chat_id, buf)` → drops the dict entry + emits
    `SYNTHESIS_BUFFER_COMMAND op="clear"`.
  - Bus-emit failures soft-fail (logged at DEBUG, no raise — INV-Syn-17).

Snapshot read is best-effort: missing / corrupt snapshot → cache stays
empty for that chat (returns None on get); next chat tick re-tries.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


BUFFER_NAMES: tuple[str, ...] = ("goal", "retrieval", "imaginal", "perception")
DEFAULT_HYDRATION_WARM_THRESHOLD_S: float = 3600.0


# Type alias for the bus_emit callable. Receives the SYNTHESIS_BUFFER_COMMAND
# payload (already shaped with `op` field); the binder lives in agno_worker
# and wraps `bus.publish(make_msg(SYNTHESIS_BUFFER_COMMAND, src, "synthesis",
# payload))`.
BusEmit = Callable[[dict], None]


class BufferCache:
    """Per-chat in-mem cache + write-through bus emit + lazy hydrate.

    Constructor params:
      bus_emit: callable that publishes the SYNTHESIS_BUFFER_COMMAND payload.
                Pass `lambda payload: None` to disable persistence (tests).
      snapshot_path: path to `data/buffers_snapshot.json` (written by
                synthesis_worker; agno reads on first cold access).
      hydration_warm_threshold_s: chats whose snapshot-recorded
                `updated_at` is older than `now - this` skip hydration
                (treated as fresh).
      clock: time source (overridable for tests).
    """

    def __init__(
        self,
        *,
        bus_emit: BusEmit,
        snapshot_path: str | os.PathLike,
        hydration_warm_threshold_s: float = DEFAULT_HYDRATION_WARM_THRESHOLD_S,
        clock: Callable[[], float] = time.time,
    ):
        self._emit = bus_emit
        self._snapshot_path = str(snapshot_path)
        self._warm_s = float(hydration_warm_threshold_s)
        self._clock = clock
        # chat_id -> {buffer_name -> {content, concept_ids, ts}}
        self._cache: dict[str, dict[str, dict]] = {}
        # chat_ids whose hydration has been attempted (success or skip)
        self._hydrated: set[str] = set()
        self._lock = threading.Lock()
        self._writes_emitted = 0
        self._clears_emitted = 0

    # ── Hydration ──────────────────────────────────────────────────────

    def _hydrate(self, chat_id: str) -> None:
        """Read `buffers_snapshot.json` once + pull this chat's rows.

        Skips silently on missing / corrupt snapshot. Skips per-chat
        hydration if `updated_at` is older than warm threshold (the chat
        is treated as fresh — no spurious DuckDB reads for week-old
        session_ids)."""
        if chat_id in self._hydrated:
            return
        self._hydrated.add(chat_id)
        try:
            with open(self._snapshot_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            logger.debug(
                "[BufferCache] snapshot read failed (%s): %s",
                self._snapshot_path, e,
            )
            return
        chats = payload.get("chats") if isinstance(payload, dict) else None
        if not isinstance(chats, dict):
            return
        chat_payload = chats.get(chat_id)
        if not isinstance(chat_payload, dict) or not chat_payload:
            return
        # Determine freshness from max updated_at across the buffers.
        latest_ts = 0.0
        for row in chat_payload.values():
            if isinstance(row, dict):
                ts = row.get("updated_at") or 0
                try:
                    latest_ts = max(latest_ts, float(ts))
                except (TypeError, ValueError):
                    pass
        if latest_ts and (self._clock() - latest_ts) > self._warm_s:
            return  # stale — treat as fresh chat
        # Adopt the snapshot rows into the local cache.
        with self._lock:
            local = self._cache.setdefault(chat_id, {})
            for buf, row in chat_payload.items():
                if buf not in BUFFER_NAMES or not isinstance(row, dict):
                    continue
                local[buf] = {
                    "content": row.get("content") or "",
                    "concept_ids": list(row.get("concept_ids") or []),
                    "ts": float(row.get("updated_at") or 0.0),
                }

    # ── Read surface ────────────────────────────────────────────────────

    def get(self, chat_id: str, buffer_name: str) -> Optional[dict]:
        """Return {content, concept_ids, ts} or None if unset.

        Triggers lazy hydration on first access of an unseen chat_id."""
        if not chat_id or buffer_name not in BUFFER_NAMES:
            return None
        self._hydrate(chat_id)
        with self._lock:
            row = self._cache.get(chat_id, {}).get(buffer_name)
            if row is None:
                return None
            return {
                "content": row.get("content") or "",
                "concept_ids": list(row.get("concept_ids") or []),
                "ts": float(row.get("ts") or 0.0),
            }

    def get_all(self, chat_id: str) -> dict[str, dict]:
        """Return {buffer_name: {content, concept_ids, ts}} (empty buffers absent)."""
        if not chat_id:
            return {}
        self._hydrate(chat_id)
        with self._lock:
            local = self._cache.get(chat_id, {})
            return {
                buf: {
                    "content": row.get("content") or "",
                    "concept_ids": list(row.get("concept_ids") or []),
                    "ts": float(row.get("ts") or 0.0),
                }
                for buf, row in local.items()
            }

    # ── Write surface (write-through) ───────────────────────────────────

    def set(
        self,
        chat_id: str,
        buffer_name: str,
        *,
        content: str,
        concept_ids: Optional[list[str]] = None,
    ) -> None:
        """Update local dict + emit SYNTHESIS_BUFFER_COMMAND(op="set").

        Soft-fails on bus emit error (INV-Syn-17 — chat NEVER fails because
        of a buffer-write failure; the local cache stays correct in-process,
        durable persistence is lost for this single write but logged)."""
        if not chat_id:
            raise ValueError("chat_id must be non-empty")
        if buffer_name not in BUFFER_NAMES:
            raise ValueError(
                f"buffer_name must be one of {BUFFER_NAMES}; got {buffer_name!r}"
            )
        cids = list(concept_ids or [])
        ts = float(self._clock())
        with self._lock:
            self._cache.setdefault(chat_id, {})[buffer_name] = {
                "content": content or "",
                "concept_ids": cids,
                "ts": ts,
            }
            # Mark hydrated — we know our local state is authoritative now.
            self._hydrated.add(chat_id)
        payload = {
            "op": "set",
            "chat_id": chat_id,
            "buffer_name": buffer_name,
            "content": content or "",
            "concept_ids": cids,
            "ts": ts,
        }
        try:
            self._emit(payload)
            self._writes_emitted += 1
        except Exception as e:
            logger.debug(
                "[BufferCache] bus emit (set) failed for %s/%s: %s",
                chat_id, buffer_name, e,
            )

    def clear(self, chat_id: str, buffer_name: str) -> None:
        """Drop the local entry + emit SYNTHESIS_BUFFER_COMMAND(op="clear").

        Idempotent — clearing an absent buffer is not an error."""
        if not chat_id:
            raise ValueError("chat_id must be non-empty")
        if buffer_name not in BUFFER_NAMES:
            raise ValueError(
                f"buffer_name must be one of {BUFFER_NAMES}; got {buffer_name!r}"
            )
        ts = float(self._clock())
        with self._lock:
            local = self._cache.get(chat_id)
            if local is not None:
                local.pop(buffer_name, None)
            self._hydrated.add(chat_id)
        payload = {
            "op": "clear",
            "chat_id": chat_id,
            "buffer_name": buffer_name,
            "ts": ts,
        }
        try:
            self._emit(payload)
            self._clears_emitted += 1
        except Exception as e:
            logger.debug(
                "[BufferCache] bus emit (clear) failed for %s/%s: %s",
                chat_id, buffer_name, e,
            )

    # ── Observability ───────────────────────────────────────────────────

    def stats(self) -> dict:
        """Counters for the A.7 acceptance gate cross-check."""
        with self._lock:
            return {
                "writes_emitted": self._writes_emitted,
                "clears_emitted": self._clears_emitted,
                "chats_cached": len(self._cache),
                "chats_hydrated": len(self._hydrated),
            }


__all__ = (
    "BufferCache",
    "BUFFER_NAMES",
    "DEFAULT_HYDRATION_WARM_THRESHOLD_S",
)
