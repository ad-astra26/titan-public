"""
sensor_cache — shared substrate for §L1 Trinity Daemon Internal Design.

Microkernel v2 Phase A §A.7 / §L1 (S7, 2026-04-26). Provides the
3-layer pattern that lets body / mind / future Trinity daemons keep
the Schumann-rhythmic tick layer fully isolated from sensor I/O:

    ┌──────────────────────┐
    │ 1. Sensor refresh    │  background daemon thread per sense,
    │    layer             │  refreshing at native cadence (file
    │    (per-sense thread)│  reads ~1s, network ~10s, http ~30s)
    └──────────┬───────────┘
               ▼
    ┌──────────────────────┐
    │ 2. Cache substrate   │  this module — SensorCache class,
    │    (locked dict)     │  always last-known-good value per
    │                      │  sense, lock-protected dict reads.
    └──────────┬───────────┘
               ▼
    ┌──────────────────────┐
    │ 3. Tensor tick layer │  caller-defined Schumann-rhythmic
    │    (Schumann)        │  thread reads cache, computes tensor,
    │                      │  writes shm slot. NO I/O on hot path.
    └──────────────────────┘

Reference implementation: ``spirit_worker``'s S3b 70.47 Hz writer
already validates the pattern at zero refactor cost (spirit tick
has no sensor I/O — pure compute). S7 brings the pattern to body
(524 ms → 100 μs target, 5,240×) and mind (73 ms → 100 μs, 730×).

Spec: ``titan-docs/rFP_microkernel_v2_shadow_core.md`` §L1 +
``titan-docs/PLAN_microkernel_phase_a_s7.md``.

Threading model
───────────────
Each Trinity daemon worker runs as its own multiprocessing
subprocess. Inside that subprocess, S7 adds:

  - one background daemon thread per sense (5 for body, 5 for mind)
  - one Schumann-rate shm writer daemon thread

The main worker loop (already serving the recv_queue / send_queue
bus interface) is unchanged in shape — it just calls helper functions
from this module to start / stop the threads, and reads from the
``SensorCache`` instead of calling sense functions inline.

All threads check a shared ``threading.Event`` (``stop_event``) and
exit cleanly on MODULE_SHUTDOWN. Daemon-flag ensures the subprocess
itself exits even if the event is missed.

Failure semantics
─────────────────
- Sense fn raises → cache keeps last-known-good (initial value if
  never refreshed). NEVER blocks the tick.
- Cache read during write → lock serializes (microsecond-scale
  contention; tick is at 7.83-23.49 Hz, refresh is 0.03-1 Hz).
- Tick path → cache read returns a shallow dict copy; caller can
  mutate freely without stomping cache state.
- shm writer fails → swallowed; chronic failures surface via
  ``arch_map shm-status age_seconds``. Logging at Schumann rate
  would flood. Same pattern as S3b spirit-fast writer.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Cache substrate ─────────────────────────────────────────────────


@dataclass
class SensorCache:
    """
    Per-sense last-known-good cache for a Trinity daemon.

    Stores ``dict`` readings keyed by sense name. Initial values come
    from the caller (typically a synchronous warmup pass at boot) so
    that the tick path never sees an empty cache.

    Thread-safety: a single ``threading.Lock`` guards all reads /
    writes. At 7.83-23.49 Hz tick rate vs 0.03-1 Hz refresh rate
    contention is negligible; lock-free atomic-pointer-swap would
    add complexity for zero measurable win.
    """

    initial: dict[str, dict[str, Any]] = field(default_factory=dict)
    _data: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        # Seed from initial (each value gets a ts of 0 unless caller
        # specifies one; tick path can then check age_seconds()).
        for name, reading in self.initial.items():
            seeded = dict(reading)
            seeded.setdefault("ts", 0.0)
            self._data[name] = seeded

    def set(self, name: str, reading: dict[str, Any]) -> None:
        """Atomic per-sense update. ts auto-stamped to time.time()."""
        snapshot = dict(reading)
        snapshot["ts"] = time.time()
        with self._lock:
            self._data[name] = snapshot

    def get(self, name: str) -> dict[str, Any] | None:
        """Return a shallow copy of the named sense reading, or None."""
        with self._lock:
            current = self._data.get(name)
            return dict(current) if current is not None else None

    def get_all(self) -> dict[str, dict[str, Any]]:
        """Return a shallow-copied snapshot of all senses (single lock)."""
        with self._lock:
            return {name: dict(reading) for name, reading in self._data.items()}

    def age_seconds(self, name: str) -> float | None:
        """Seconds since last refresh of named sense (None if absent)."""
        with self._lock:
            current = self._data.get(name)
            if current is None:
                return None
            ts = current.get("ts", 0.0)
        if ts <= 0.0:
            return None
        return max(0.0, time.time() - ts)


# ── Refresh-thread spec + helpers ───────────────────────────────────


@dataclass(frozen=True)
class RefreshSpec:
    """
    Declarative spec for a single per-sense refresh thread.

    name:        cache key + thread-name suffix (e.g. "interoception")
    refresh_fn:  zero-arg callable returning a dict — typically a
                 closure over the existing _sense_X(thresholds) fn
    period_s:    refresh cadence (matches sensor's natural timescale)
    """

    name: str
    refresh_fn: Callable[[], dict[str, Any]]
    period_s: float


def _refresh_loop(
    spec: RefreshSpec,
    cache: SensorCache,
    stop_event: threading.Event,
) -> None:
    """Body of one refresh thread. Sleeps in stop_event-aware chunks."""
    # Initial refresh ASAP so the cache transitions from "boot warmup"
    # to "live data" without waiting one period_s.
    try:
        cache.set(spec.name, spec.refresh_fn())
    except Exception as exc:  # noqa: BLE001 — swallow per failure semantics
        logger.warning("[sensor_cache] %s initial refresh failed: %s", spec.name, exc)

    while not stop_event.wait(spec.period_s):
        try:
            cache.set(spec.name, spec.refresh_fn())
        except Exception as exc:  # noqa: BLE001
            logger.warning("[sensor_cache] %s refresh failed: %s", spec.name, exc)


def start_refresh_threads(
    specs: list[RefreshSpec],
    cache: SensorCache,
    stop_event: threading.Event,
    thread_name_prefix: str = "sensor_refresh",
) -> list[threading.Thread]:
    """
    Spawn one daemon thread per RefreshSpec. Returns the thread list
    so the caller can join() at shutdown if desired.

    Daemon-flag ensures threads die when the worker subprocess exits
    even if stop_event is missed (defense in depth).
    """
    threads: list[threading.Thread] = []
    for spec in specs:
        t = threading.Thread(
            target=_refresh_loop,
            args=(spec, cache, stop_event),
            daemon=True,
            name=f"{thread_name_prefix}_{spec.name}",
        )
        t.start()
        threads.append(t)
    return threads


# ── Schumann shm writer helper ──────────────────────────────────────


def _shm_writer_loop(
    tick_fn: Callable[[], None],
    period_s: float,
    stop_event: threading.Event,
    log_label: str,
) -> None:
    """
    Body of the Schumann-rate shm writer thread. Uses a target-time
    scheduler so the cadence stays accurate even when ticks run a
    few μs slow — without busy-looping when they run a few μs fast.

    tick_fn captures the worker's tensor-compute + shm-write logic.
    Failures inside tick_fn are swallowed at this level; the caller
    is expected to handle its own per-tick exceptions cleanly so a
    transient sensor cache miss doesn't kill the writer thread.
    """
    next_tick = time.time()
    while not stop_event.is_set():
        try:
            tick_fn()
        except Exception as exc:  # noqa: BLE001 — never kill the writer
            # Throttle log: only one per 60s per writer.
            now = time.time()
            last = _last_writer_log_ts.get(log_label, 0.0)
            if now - last >= 60.0:
                logger.warning("[sensor_cache] %s tick raised: %s", log_label, exc)
                _last_writer_log_ts[log_label] = now

        next_tick += period_s
        sleep_for = next_tick - time.time()
        if sleep_for > 0:
            # stop_event.wait returns True if signaled; either way we
            # re-check the loop condition, so this is safe.
            stop_event.wait(sleep_for)
        else:
            # We're behind schedule (system jitter / GC pause / etc.).
            # Reset the schedule rather than firing back-to-back ticks.
            next_tick = time.time()


# Per-writer-label last log ts, for 60s log throttle. Module-global
# is fine — there are at most 3 writers per worker (one per Trinity
# daemon) and lookup is keyed by label.
_last_writer_log_ts: dict[str, float] = {}


def start_shm_writer_thread(
    tick_fn: Callable[[], None],
    period_s: float,
    stop_event: threading.Event,
    thread_name: str,
) -> threading.Thread:
    """
    Spawn the Schumann-rate writer thread. Returns it so caller can
    join() at shutdown.

    period_s should be the Schumann-derived period (1/freq):
      - body:   1/7.83  ≈ 0.1277 s  (Schumann fundamental)
      - mind:   1/23.49 ≈ 0.0426 s  (Schumann × 3)
      - spirit: 1/70.47 ≈ 0.0142 s  (Schumann × 9)
    """
    t = threading.Thread(
        target=_shm_writer_loop,
        args=(tick_fn, period_s, stop_event, thread_name),
        daemon=True,
        name=thread_name,
    )
    t.start()
    return t


# ── Convenience: graceful shutdown of an S7 thread set ─────────────


def stop_threads(
    stop_event: threading.Event,
    threads: list[threading.Thread],
    timeout_s: float = 2.0,
) -> None:
    """
    Signal stop_event + join each thread up to timeout_s. Used at
    MODULE_SHUTDOWN so we can verify thread cleanup in tests.
    Daemon flag ensures the subprocess exits even if a join times
    out — but a timeout in normal operation is a bug worth surfacing.
    """
    stop_event.set()
    for t in threads:
        t.join(timeout=timeout_s)
        if t.is_alive():
            logger.warning(
                "[sensor_cache] thread %s failed to stop within %.1fs",
                t.name, timeout_s,
            )
