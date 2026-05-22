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
    # D-SPEC-78 (Phase 2 Chunk α 2026-05-18) — REVERTED proxy-None-tolerance.
    # An earlier version of this code (commit 1d3f80cc) tolerated
    # guardian=None on the assumption that "fixing" the silent-NoneType
    # path in worker-context proxies (MemoryProxy / SocialGraphProxy /
    # RLProxy built with guardian=None by agno_worker's WorkerPlugin shim)
    # would improve chat behavior. PRODUCTION OBSERVATION 2026-05-18 on T3:
    # the silent-NoneType path was the FAST path. Once the proxies actually
    # routed bus requests, downstream workers either timed out (MemoryProxy
    # query → memory_worker 5s G19 timeout) or raised schema errors
    # (SocialGraphProxy.save_profile → KeyError 'user_id' in social_graph
    # worker → 20s wait). PreHook stages that previously took 2ms now took
    # 25s. CHAT_REQUEST timed out at 90s on T3.
    #
    # The right architectural fix is to bring these proxies' worker-context
    # callsites up to spec (correct kwargs, proper schemas, async/timeout
    # hygiene). That is OUT-OF-SCOPE for Chunk α — a follow-up rFP
    # ("agno worker plugin proxy hygiene") will revive each one properly.
    # Until then, the silent-NoneType path stays (chat works; ERROR logs
    # are a known nuisance, not a regression).
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
