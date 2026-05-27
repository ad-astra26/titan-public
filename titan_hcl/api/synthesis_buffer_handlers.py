"""API handlers for `/v6/synthesis/buffers/*` (Phase 7 §P7.G).

Backs the Observatory's ACT-R working-memory panel: list active chats,
read a per-chat buffer, list recent writes, dump the full snapshot.

**Read source:** mirrors the Phase 4 FU-1 / P5 snapshot pattern —
synthesis_worker (sole writer per INV-Syn-16) exports
`data/buffers_snapshot.json` atomically after every persist/clear. The
api process reads this JSON only; it never opens `synthesis.duckdb`
directly (DuckDB 1.5+ exclusive-lock against the active sole writer).

Snapshot schema (see `ActrBufferStore._build_snapshot_payload`):

    {
      "version": 1,
      "ts": <wall-clock seconds>,
      "writes_seen": int,
      "clears_seen": int,
      "chat_count": int,
      "chats": {
        "<chat_id>": {
          "goal":       {content, concept_ids, embedding_hash, updated_at},
          "retrieval":  {...},
          "imaginal":   {...},
          "perception": {...},
        },
        ...
      }
    }

Soft-fail contract: missing / unparseable snapshot → 200 with
`{"ok": true, "snapshot": "missing|stale|corrupt", ...empty...}`. The
frontend renders an empty state instead of a 500 cascade.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from fastapi import Request

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_NAME = "buffers_snapshot.json"
SNAPSHOT_STALENESS_SECONDS = 600  # mirrors P5 forks handler


# ── Snapshot cache (mtime-keyed) ──────────────────────────────────


_SNAPSHOT_CACHE: dict[str, dict] = {}


def _resolve_snapshot_path() -> str:
    data_dir = os.environ.get("TITAN_DATA_DIR", "data")
    return os.path.join(data_dir, DEFAULT_SNAPSHOT_NAME)


def _load_snapshot(path: Optional[str] = None) -> Optional[dict]:
    if path is None:
        path = _resolve_snapshot_path()
    if not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached = _SNAPSHOT_CACHE.get(path)
    if cached is not None and cached.get("mtime") == mtime:
        return cached["data"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(
            "[synthesis_buffer_handlers] snapshot parse failed (%s): %s",
            path, e,
        )
        return None
    if not isinstance(data, dict):
        return None
    _SNAPSHOT_CACHE[path] = {"mtime": mtime, "data": data}
    return data


def _load_snapshot_with_status(
    path: Optional[str] = None,
) -> tuple[Optional[dict], str]:
    """Return (payload, status). status ∈ {"ok","missing","stale","corrupt"}."""
    if path is None:
        path = _resolve_snapshot_path()
    if not os.path.exists(path):
        return None, "missing"
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None, "missing"
    age = time.time() - mtime
    payload = _load_snapshot(path)
    if payload is None:
        return None, "corrupt"
    if age > SNAPSHOT_STALENESS_SECONDS:
        return payload, "stale"
    return payload, "ok"


# ── GET handlers ──────────────────────────────────────────────────


async def get_v6_synthesis_buffers_list_chats(request: Request):
    """Return list of chat_ids with any buffer rows.

    Response shape:
        {ok, snapshot, ts, chat_count, chats: [chat_id, ...]}
    """
    payload, status = _load_snapshot_with_status()
    if payload is None:
        return {
            "ok": True, "snapshot": status, "ts": 0.0,
            "chat_count": 0, "chats": [],
        }
    chats = payload.get("chats") if isinstance(payload, dict) else None
    if not isinstance(chats, dict):
        chats = {}
    return {
        "ok": True,
        "snapshot": status,
        "ts": float(payload.get("ts") or 0.0),
        "chat_count": len(chats),
        "chats": sorted(chats.keys()),
    }


async def get_v6_synthesis_buffers_read(request: Request):
    """Return a specific buffer for a specific chat.

    Query params:
        chat_id: required, e.g. "alice:default"
        buffer:  required, one of {goal, retrieval, imaginal, perception}

    Response shape:
        {ok, snapshot, chat_id, buffer_name, content, concept_ids,
         embedding_hash, updated_at}
    """
    chat_id = request.query_params.get("chat_id", "")
    buffer_name = request.query_params.get("buffer", "")
    valid_buffers = ("goal", "retrieval", "imaginal", "perception")
    if not chat_id:
        return {"ok": False, "error": "chat_id query param required"}
    if buffer_name not in valid_buffers:
        return {
            "ok": False,
            "error": (
                f"buffer query param must be one of {valid_buffers}; "
                f"got {buffer_name!r}"
            ),
        }
    payload, status = _load_snapshot_with_status()
    base = {
        "ok": True,
        "snapshot": status,
        "chat_id": chat_id,
        "buffer_name": buffer_name,
        "content": "",
        "concept_ids": [],
        "embedding_hash": "",
        "updated_at": 0.0,
    }
    if payload is None:
        return base
    chats = payload.get("chats") if isinstance(payload, dict) else None
    if not isinstance(chats, dict):
        return base
    row = (chats.get(chat_id) or {}).get(buffer_name)
    if not isinstance(row, dict):
        return base
    base["content"] = row.get("content") or ""
    base["concept_ids"] = list(row.get("concept_ids") or [])
    base["embedding_hash"] = row.get("embedding_hash") or ""
    try:
        base["updated_at"] = float(row.get("updated_at") or 0.0)
    except (TypeError, ValueError):
        base["updated_at"] = 0.0
    return base


async def get_v6_synthesis_buffers_recent_writes(request: Request):
    """Return most recently updated buffers across all chats.

    Query params:
        limit: int, default 20, max 200

    Response shape:
        {ok, snapshot, writes_seen, clears_seen,
         writes: [{chat_id, buffer_name, updated_at}, ...]}
    """
    try:
        limit = int(request.query_params.get("limit", 20))
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 200))

    payload, status = _load_snapshot_with_status()
    if payload is None:
        return {
            "ok": True, "snapshot": status,
            "writes_seen": 0, "clears_seen": 0, "writes": [],
        }
    chats = payload.get("chats") if isinstance(payload, dict) else None
    writes: list[dict] = []
    if isinstance(chats, dict):
        for chat_id, chat_row in chats.items():
            if not isinstance(chat_row, dict):
                continue
            for buf_name, row in chat_row.items():
                if not isinstance(row, dict):
                    continue
                try:
                    ts = float(row.get("updated_at") or 0.0)
                except (TypeError, ValueError):
                    ts = 0.0
                writes.append({
                    "chat_id": chat_id,
                    "buffer_name": buf_name,
                    "updated_at": ts,
                })
    writes.sort(key=lambda w: w["updated_at"], reverse=True)
    return {
        "ok": True,
        "snapshot": status,
        "writes_seen": int(payload.get("writes_seen") or 0),
        "clears_seen": int(payload.get("clears_seen") or 0),
        "writes": writes[:limit],
    }


async def get_v6_synthesis_buffers_snapshot(request: Request):
    """Return the full buffers snapshot (cap response by limiting chats).

    Query params:
        limit: int, default 50, max 500 (number of chats to include)

    Response shape: full snapshot payload (see module docstring).
    """
    try:
        limit = int(request.query_params.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, 500))

    payload, status = _load_snapshot_with_status()
    if payload is None:
        return {
            "ok": True, "snapshot": status,
            "version": 1, "ts": 0.0,
            "writes_seen": 0, "clears_seen": 0,
            "chat_count": 0, "chats": {},
        }
    chats = payload.get("chats") if isinstance(payload, dict) else None
    if not isinstance(chats, dict):
        chats = {}
    # Cap by sorted chat_id for determinism + freshness via most-recent
    # buffer write across rows.
    def _chat_max_ts(items: dict) -> float:
        max_ts = 0.0
        for row in items.values():
            if isinstance(row, dict):
                try:
                    max_ts = max(max_ts, float(row.get("updated_at") or 0.0))
                except (TypeError, ValueError):
                    pass
        return max_ts

    sorted_chats = sorted(
        chats.items(),
        key=lambda kv: _chat_max_ts(kv[1] if isinstance(kv[1], dict) else {}),
        reverse=True,
    )
    capped = dict(sorted_chats[:limit])
    return {
        "ok": True,
        "snapshot": status,
        "version": int(payload.get("version") or 1),
        "ts": float(payload.get("ts") or 0.0),
        "writes_seen": int(payload.get("writes_seen") or 0),
        "clears_seen": int(payload.get("clears_seen") or 0),
        "chat_count": len(chats),
        "chats": capped,
    }


__all__ = (
    "get_v6_synthesis_buffers_list_chats",
    "get_v6_synthesis_buffers_read",
    "get_v6_synthesis_buffers_recent_writes",
    "get_v6_synthesis_buffers_snapshot",
)
