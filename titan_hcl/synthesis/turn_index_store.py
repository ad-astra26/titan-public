"""Per-session turn-index counter — arch §7 chat-TX `turn_index` field.

Phase 3 (rFP §18 Phase 3 — episode model, D-SPEC-127).

Replaces the Phase 2 closure placeholder `turn_index=0` with a real
per-chat-session monotonic counter. Used by:
  - `actr_episodic_recall_helper` (P3.D granularity-aware retrieval) —
    `granularity=turn` needs an ordering signal to compose
    "the Nth-most-recent turn for chat_id".
  - Observatory chat-view scrolling — stable ordering even when TX
    block-height isn't 1:1 with turn order (Merkle batching can
    reshuffle TXs within a batch).
  - Downstream §10 Concept-spine composition — the "Nth turn within
    session" temporal coordinate enables `composed_into` edge ordering.

Design (chosen after considering alternatives):
  - **Single JSON file** `data/conversation_turn_index.json` — agno_worker
    is the sole writer (P1 lesson 1: sole-writer per state file).
  - **Atomic tmp+rename** for crash-safe persistence.
  - **In-process LRU cache** capped at `MAX_TRACKED_SESSIONS` (default
    5000) to bound memory under traffic bursts; eviction is just "drop
    least-recently-touched key" — losing a counter means the next turn
    restarts at 0 (acceptable — chat sessions rarely outlive 5000
    distinct sessions in agno's lifetime, and the JSON keeps eviction
    visible via key disappearance).
  - **Cross-process safe (read-only)**: other processes reading the JSON
    snapshot for analytics get an eventually-consistent view; only
    agno_worker writes.
  - **Restart preserves continuity**: file persists on disk; next boot
    loads + resumes. Same chat_id continuing after restart picks up at
    `last_index + 1`.

Surfaced concerns from the Phase 3 plan (revisit after live soak):
  3. turn_index reset semantics on chat-resume — current design assumes
     a chat_id is a stable session key. If clients reuse a chat_id
     across long gaps (e.g. days), the counter keeps growing. This is
     correct per arch §7 ("turn_index within the chat session") since
     "session" = the entire chat_id lifespan; if Maker decides to
     bucket by time-window instead, the cache + JSON shape stays
     compatible (just add a `gap_threshold_s` flush).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


# Cap on tracked sessions in the in-process cache. The JSON file may
# carry more (older sessions get loaded lazily as they recur); the cache
# evicts least-recently-touched once the cap is hit.
MAX_TRACKED_SESSIONS = 5000

# Default state-file location. Caller (synthesis_worker boot or
# agno_hooks) may override via `set_state_path`.
_DEFAULT_STATE_PATH = os.path.join("data", "conversation_turn_index.json")


# ── Module-level singleton state (process-local) ─────────────────────

# OrderedDict for LRU semantics — move_to_end on each access.
_cache: "OrderedDict[str, int]" = OrderedDict()
_cache_lock = threading.Lock()
_state_path: str = _DEFAULT_STATE_PATH
_loaded: bool = False


def set_state_path(path: str) -> None:
    """Override the state-file path (test injection / titan-local paths).

    Re-loads the cache from the new path on next access. Safe to call
    multiple times. Used by tests via a tmp_path fixture.
    """
    global _state_path, _loaded
    with _cache_lock:
        _state_path = path
        _loaded = False
        _cache.clear()


def _ensure_loaded() -> None:
    """Lazy-load the JSON state file into the in-process cache.

    First call after process start (or after `set_state_path`) reads the
    file. Subsequent calls are no-ops. The cache is the source of truth
    in-process; the file is the durable backing store.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    if not os.path.exists(_state_path):
        return
    try:
        with open(_state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(
                "[turn_index_store] state file %s not a dict — starting fresh",
                _state_path)
            return
        # Preserve insertion order (post-Python 3.7 dict). Cap at
        # MAX_TRACKED_SESSIONS by dropping oldest entries; the file may
        # have grown beyond the cap, so trim on load.
        for chat_id, last_idx in list(data.items())[-MAX_TRACKED_SESSIONS:]:
            try:
                _cache[str(chat_id)] = int(last_idx)
            except (TypeError, ValueError):
                continue
    except Exception as exc:
        logger.warning(
            "[turn_index_store] failed to load state file %s: %s — starting "
            "fresh", _state_path, exc)


def _persist() -> None:
    """Atomic tmp+rename write of the cache to the JSON state file.

    Caller must hold `_cache_lock`. Writes the FULL cache each tick
    (acceptable — even at MAX_TRACKED_SESSIONS the dict is <200 KB
    serialized; chat traffic is ≤ a few turns/sec).
    """
    parent = os.path.dirname(_state_path) or "."
    try:
        os.makedirs(parent, exist_ok=True)
    except Exception as exc:
        logger.warning(
            "[turn_index_store] mkdir(%s) failed: %s", parent, exc)
        return
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=".turn_index.", suffix=".tmp", dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(dict(_cache), f, separators=(",", ":"))
            os.replace(tmp_path, _state_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as exc:
        logger.warning(
            "[turn_index_store] persist to %s failed: %s "
            "(in-process cache still correct; restart loses delta)",
            _state_path, exc)


def next_turn_index(chat_id: str) -> int:
    """Return the next turn_index for `chat_id` (0-based first turn).

    First call for a chat_id returns 0 and seeds the counter at 0. Each
    subsequent call returns the previous value + 1.

    Args:
        chat_id: The session identifier. Empty string → returns 0 with
            no caching (caller should not have requested an index without
            a session key). Non-string keys are coerced to str.

    Returns:
        Monotonic int per (chat_id), starting at 0. Soft-fails to 0 on
        any persistence error (the chat path NEVER breaks because of
        turn-index bookkeeping).
    """
    cid = str(chat_id or "")
    if not cid:
        return 0
    with _cache_lock:
        _ensure_loaded()
        if cid in _cache:
            prev = _cache[cid]
            new = prev + 1
        else:
            new = 0
            # Evict LRU if cache is full (drop oldest = first key).
            if len(_cache) >= MAX_TRACKED_SESSIONS:
                _cache.popitem(last=False)
        _cache[cid] = new
        _cache.move_to_end(cid)
        _persist()
        return new


def peek_turn_index(chat_id: str) -> Optional[int]:
    """Return the LAST issued turn_index for `chat_id` without incrementing.

    Returns None if the chat_id has never been seen. Useful for tests +
    diagnostics. No file writes.
    """
    cid = str(chat_id or "")
    if not cid:
        return None
    with _cache_lock:
        _ensure_loaded()
        return _cache.get(cid)


def clear_cache_for_test() -> None:
    """Reset module state — test fixtures only.

    Production code MUST NOT call this; turn_index continuity is
    load-bearing for the episode model.
    """
    global _loaded
    with _cache_lock:
        _cache.clear()
        _loaded = False


__all__ = [
    "next_turn_index",
    "peek_turn_index",
    "set_state_path",
    "clear_cache_for_test",
    "MAX_TRACKED_SESSIONS",
]
