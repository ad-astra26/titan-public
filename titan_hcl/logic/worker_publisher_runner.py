"""
worker_publisher_runner — shared 1 Hz periodic publisher thread for
Phase C SHM publishers running inside Guardian-supervised worker
processes (agency_worker, rl_worker, timechain_worker,
output_verifier_worker).

Each worker calls ``run_worker_publisher`` once at boot with its
publisher instance(s) + state-fetcher callable. The runner owns a
daemon thread that:

  - calls ``state_fetcher()`` to get a fresh state ref each tick
  - calls ``publisher.publish(state)`` (BaseStatePublisher handles all
    encode/oversize/write resilience + heartbeat logging)
  - sleeps ``cadence_s`` between ticks
  - terminates cleanly with the parent worker (daemon=True)

Failure isolation:
  - state_fetcher raises → caught + WARN-logged + skip publish this tick
  - publisher.publish raises → caught at top level inside this runner
    (BaseStatePublisher already guards internally; this is belt-and-
    braces to ensure the thread never dies)

Per Preamble G20 hot-path safety: this thread is parallel to the
worker's recv loop, so RPC handling continues unaffected by publisher
work.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def run_worker_publisher(
    *,
    publisher: Any,
    state_fetcher: Callable[[], Any],
    worker_name: str,
    cadence_s: float = 1.0,
    publish_args: Callable[[Any], tuple] = lambda s: (s,),
) -> threading.Thread:
    """
    Start a daemon thread that runs publisher.publish(...) at cadence_s.

    Args:
      publisher        — instance with .publish(*args) method (typically
                         a BaseStatePublisher subclass)
      state_fetcher    — callable returning current state (called each tick)
      worker_name      — for log prefixes (e.g., "agency_worker")
      cadence_s        — seconds between ticks (default 1.0 — SPEC §7.1)
      publish_args     — callable that takes state_fetcher's output and
                         returns positional args for publisher.publish.
                         Default: pass state directly. Use this to extract
                         multiple publisher inputs from one state object
                         (e.g., (recorder, gatekeeper) for rl_worker).

    Returns:
      The started Thread (daemon=True).
    """
    pub_class = publisher.__class__.__name__

    def _loop() -> None:
        consecutive_errors = 0
        try:
            while True:
                t0 = time.time()
                try:
                    state = state_fetcher()
                except Exception as fetch_err:
                    consecutive_errors += 1
                    if (consecutive_errors == 1
                            or consecutive_errors % 60 == 0):
                        logger.warning(
                            "[%s] %s state_fetcher raised "
                            "(#%d consecutive): %s",
                            worker_name, pub_class,
                            consecutive_errors, fetch_err, exc_info=True)
                    state = None

                if state is not None:
                    try:
                        args = publish_args(state)
                        publisher.publish(*args)
                        consecutive_errors = 0
                    except Exception as pub_err:
                        consecutive_errors += 1
                        if (consecutive_errors == 1
                                or consecutive_errors % 60 == 0):
                            logger.warning(
                                "[%s] %s publish raised at top level "
                                "(#%d consecutive): %s",
                                worker_name, pub_class,
                                consecutive_errors, pub_err, exc_info=True)

                elapsed = time.time() - t0
                sleep_for = max(0.0, cadence_s - elapsed)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except BaseException as fatal:
            logger.error(
                "[%s] %s publisher thread exited unexpectedly — "
                "SHM will go stale: %s",
                worker_name, pub_class, fatal, exc_info=True)

    thread_name = f"{worker_name}-{publisher.slot_name}-publisher"
    t = threading.Thread(target=_loop, daemon=True, name=thread_name)
    t.start()
    logger.info(
        "[%s] %s publisher thread started — slot=%s cadence=%.2fs",
        worker_name, pub_class, publisher.slot_name, cadence_s)
    return t


def run_multi_slot_worker_publisher(
    *,
    publishers: list,
    state_fetcher: Callable[[], Any],
    worker_name: str,
    cadence_s: float = 1.0,
    publish_args: Callable[[Any], tuple] = lambda s: (s,),
) -> threading.Thread:
    """
    Start ONE daemon thread that fans out to multiple publishers per
    tick (used when one worker owns multiple slots — e.g.,
    agency_worker owns agency_state + assessment_state).

    Each publisher.publish failure is isolated; remaining publishers
    still tick.

    Returns:
      The started Thread (daemon=True).
    """
    pub_classes = [p.__class__.__name__ for p in publishers]
    pub_slots = [p.slot_name for p in publishers]

    def _loop() -> None:
        consecutive_errors = 0
        try:
            while True:
                t0 = time.time()
                try:
                    state = state_fetcher()
                except Exception as fetch_err:
                    consecutive_errors += 1
                    if (consecutive_errors == 1
                            or consecutive_errors % 60 == 0):
                        logger.warning(
                            "[%s] multi-publisher state_fetcher raised "
                            "(#%d consecutive): %s",
                            worker_name, consecutive_errors, fetch_err,
                            exc_info=True)
                    state = None

                if state is not None:
                    try:
                        args = publish_args(state)
                    except Exception as args_err:
                        logger.warning(
                            "[%s] multi-publisher publish_args raised: %s",
                            worker_name, args_err, exc_info=True)
                        args = None
                    if args is not None:
                        for pub in publishers:
                            try:
                                pub.publish(*args)
                            except Exception as pub_err:
                                logger.warning(
                                    "[%s] %s publish raised "
                                    "(slot=%s): %s",
                                    worker_name,
                                    pub.__class__.__name__,
                                    pub.slot_name, pub_err,
                                    exc_info=True)
                        consecutive_errors = 0

                elapsed = time.time() - t0
                sleep_for = max(0.0, cadence_s - elapsed)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except BaseException as fatal:
            logger.error(
                "[%s] multi-publisher thread exited unexpectedly — "
                "SHM slots will go stale (%s): %s",
                worker_name, pub_slots, fatal, exc_info=True)

    thread_name = f"{worker_name}-multi-publisher"
    t = threading.Thread(target=_loop, daemon=True, name=thread_name)
    t.start()
    logger.info(
        "[%s] multi-publisher thread started — publishers=%s "
        "slots=%s cadence=%.2fs",
        worker_name, pub_classes, pub_slots, cadence_s)
    return t
