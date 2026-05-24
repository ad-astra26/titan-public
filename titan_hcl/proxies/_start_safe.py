"""
Shared helper for proxy _ensure_started — detects asyncio context and
avoids blocking the event loop.

Why: Observed 2026-04-14 after Phase A landed — Guardian.start() was
still being called synchronously from FastAPI async endpoints via
MindProxy._ensure_started → guardian.start(). Blocks event loop for
the full startup duration (seconds). This is the same CLASS of bug
Phase A fixed for Guardian.monitor_tick (moved to asyncio.to_thread),
but Phase A didn't cover the on-demand lazy-start path in proxies.

Fix: detect if we're in an asyncio event loop. If yes, kick off the
start in a background thread (with a lock so only ONE thread races),
let the current call return quickly, and the next call will find the
module ready. If no loop (i.e. running inside a worker process), block
as before — worker code expects synchronous semantics.

Tolerable side effect: the first API call that hits a not-yet-started
module may get a default / empty response. The second call a few
seconds later will find the module running. This is far better than
hanging the entire API event loop for 5-15 seconds per boot.
"""
import asyncio
import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)

# One lock per (proxy_instance, module_name) so multiple concurrent
# endpoint calls don't spawn duplicate start threads.
_start_locks: dict[tuple, threading.Lock] = {}
_start_locks_lock = threading.Lock()


def _get_lock(proxy_id: int, module: str) -> threading.Lock:
    key = (proxy_id, module)
    with _start_locks_lock:
        lock = _start_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _start_locks[key] = lock
        return lock


def ensure_started_async_safe(
    guardian,
    module: str,
    proxy_id: int,
    proxy_label: str = "Proxy",
) -> bool:
    """Start the module if not running, safe to call from both sync
    (worker) and async (FastAPI) contexts.

    Returns:
        True if the module is running at the end of this call.
        False if we're in an asyncio context and kicked off an async
        start (module not guaranteed ready yet; caller should tolerate
        a default/empty response and retry).
    """
    # D-SPEC-78 (2026-05-18) NoneType-tolerance was REVERTED then because
    # restoring proxy routing surfaced downstream worker bugs (schema
    # errors / 20s waits / 90s chat timeouts). HOWEVER the prior
    # implementation crashed on `guardian.is_running` when guardian was
    # None (agno_worker_plugin.py:146 passes None) — surfacing as
    # `[PreHook] Memory recall failed: 'NoneType' object has no attribute
    # 'is_running'` and BLOCKING MemoryProxy.query() from agno entirely.
    #
    # 2026-05-23 D-SPEC-123 follow-up — restore tolerance for the
    # guardian=None case specifically: skip the start-check (optimistic —
    # assume the module is running; if it isn't, the downstream
    # bus.request_async hits its G19 timeout cap, MemoryProxy.query
    # catches+degrades to []). The downstream-bug concerns the D-SPEC-78
    # comment cited are bounded by:
    #   - MemoryProxy.query: 5s G19 cap (line 299) + except+return-[]
    #     fallback (line 301)
    #   - SocialGraphProxy / RLProxy: their callsite hygiene is a
    #     separate concern (D-SPEC-78's "follow-up rFP")
    # This change UNBLOCKS the Phase 1 synthesis wire end-to-end (chat
    # memory recall reaches _cognee_search → emits MEMORY_RETRIEVAL_USED →
    # activation_state populates).
    if guardian is None:
        return True
    is_alive = getattr(guardian, "is_started", guardian.is_running)
    if is_alive(module):
        return True
    try:
        asyncio.get_running_loop()
        in_event_loop = True
    except RuntimeError:
        in_event_loop = False

    if not in_event_loop:
        # Sync caller (worker process) — safe to block.
        logger.info("[%s] First use — starting %s module (sync)...",
                    proxy_label, module)
        guardian.start(module)
        return True

    # Async caller (FastAPI event loop) — must NOT block the loop.
    # Spawn a thread to run start() in the background; only one thread
    # per (proxy, module) in flight. Return False so caller can use
    # default/empty value; next request will find it ready.
    lock = _get_lock(proxy_id, module)
    if lock.acquire(blocking=False):
        def _go():
            try:
                logger.info(
                    "[%s] First use from async context — starting %s "
                    "in background thread (event loop unblocked)...",
                    proxy_label, module,
                )
                guardian.start(module)
            except Exception as e:
                logger.warning("[%s] Async start for %s failed: %s",
                               proxy_label, module, e)
            finally:
                lock.release()
        threading.Thread(
            target=_go, name=f"{proxy_label}-start-{module}", daemon=True
        ).start()
    return False
