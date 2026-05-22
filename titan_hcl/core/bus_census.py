"""
Bus census — opt-in, low-overhead instrumentation for diagnosing the
heartbeat-cascade root cause (Phase E.1 of bus + Guardian hardening).

Purpose: measure WHO is producing what at what rate, who is dropping
what, and how queue depth evolves over time. Without this measurement
we can only *guess* at the cascade root cause; with it, we can identify
the actual bottleneck (slow drain vs. burst producer vs. queue-too-small
vs. heartbeat-priority-inversion).

Disabled by default. Enable via env var: TITAN_BUS_CENSUS=1

Design constraints:
  - Zero overhead when disabled (single boolean check per hook)
  - No new bus messages (writes only to a separate log file, never to bus)
  - Lock-protected counters (worker processes increment in-process counts,
    parent samples its own bus stats — fork-state aware)
  - Periodic flush every CENSUS_FLUSH_S seconds via background thread
  - Output is TSV at /tmp/titan_bus_census.log for easy awk/sort analysis

Hook points in bus.py:
  1. publish() entry  → record_emission(msg_type, dst)
  2. _try_put() drop  → record_drop(subscriber, msg_type)
  3. drain() return   → record_drain(subscriber, n)
  4. background tick  → sample_queue_depths(subscribers_dict)

Output schema (TSV):
  ts<TAB>kind<TAB>key<TAB>value
  where:
    kind ∈ {EMIT, DROP, DRAIN, DEPTH, TICK}
    key  = "msg_type|dst" or "subscriber|msg_type" or "subscriber"
    value = count (EMIT/DROP/DRAIN), depth (DEPTH), tick_seq (TICK)

Codified after 2026-04-14 incident (1,420 restarts across T1/T2/T3 in
12 days, all linked to bus queue saturation that we could not pinpoint
because we lacked rate/drain/depth measurement).
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────

ENABLED = os.environ.get("TITAN_BUS_CENSUS", "0") == "1"
# Phase C C-S2 (D11): per-Titan census log path. Default resolves the active
# Titan ID from canonical TITAN_KERNEL_TITAN_ID env var (with legacy TITAN_ID
# fallback) so T1/T2/T3 each get a distinct path. Per SPEC §3 D11 +
# PLAN_microkernel_phase_c_s2_kernel.md §12.5.
_TITAN_ID_FOR_CENSUS = (
    os.environ.get("TITAN_KERNEL_TITAN_ID") or os.environ.get("TITAN_ID", "T1")
)
CENSUS_LOG_PATH = os.environ.get(
    "TITAN_BUS_CENSUS_LOG", f"/tmp/titan_{_TITAN_ID_FOR_CENSUS}_bus_census.log")
CENSUS_FLUSH_S = float(os.environ.get("TITAN_BUS_CENSUS_FLUSH_S", "10"))


class _BusCensus:
    """Per-process census aggregator. One instance per worker process.

    Counters are reset every flush to make rate computation trivial
    (count_in_window / window_seconds = msg/s).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._emit: dict[tuple[str, str], int] = defaultdict(int)
        self._drop: dict[tuple[str, str], int] = defaultdict(int)
        self._drain: dict[str, int] = defaultdict(int)
        self._depth_samples: list[tuple[float, str, int]] = []
        self._last_flush_ts = time.time()
        self._tick_seq = 0
        self._writer_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._pid = os.getpid()
        # Optional depth sampler callable — called on every flush tick if set.
        # DivineBus registers itself here so the writer thread can sample
        # all subscriber queues without the bus needing its own thread.
        self._depth_sampler: Optional[callable] = None

    def register_depth_sampler(self, fn) -> None:
        self._depth_sampler = fn

    def start(self):
        if not ENABLED:
            return
        if self._writer_thread is not None and self._writer_thread.is_alive():
            return
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="bus-census-writer", daemon=True)
        self._writer_thread.start()

    def stop(self):
        self._stop.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2.0)

    def _ensure_writer(self) -> None:
        # Lazy-start writer the first time any record fires. Fork-safe:
        # workers that inherited a parent's writer thread will start their
        # own here because pid changes after fork.
        cur_pid = os.getpid()
        if self._writer_thread is None or cur_pid != self._pid:
            self._pid = cur_pid
            self.start()

    def record_emission(self, msg_type: str, dst: str) -> None:
        if not ENABLED:
            return
        self._ensure_writer()
        with self._lock:
            self._emit[(msg_type, dst)] += 1

    def record_drop(self, subscriber: str, msg_type: str) -> None:
        if not ENABLED:
            return
        self._ensure_writer()
        with self._lock:
            self._drop[(subscriber, msg_type)] += 1

    def record_drain(self, subscriber: str, n: int) -> None:
        if not ENABLED or n <= 0:
            return
        self._ensure_writer()
        with self._lock:
            self._drain[subscriber] += n

    def sample_queue_depths(self, subscribers: dict) -> None:
        """Caller passes the bus._subscribers dict; we sample qsize() of each.

        Safe to call from the parent process that owns the queue endpoints.
        Worker processes will only see their own per-worker queues.
        """
        if not ENABLED:
            return
        self._ensure_writer()
        ts = time.time()
        samples = []
        for mod_name, queues in subscribers.items():
            for q in queues:
                try:
                    depth = q.qsize()
                except Exception:
                    depth = -1  # qsize() not supported on macOS MPQueue
                samples.append((ts, mod_name, depth))
        with self._lock:
            self._depth_samples.extend(samples)

    def _writer_loop(self):
        while not self._stop.wait(CENSUS_FLUSH_S):
            try:
                if self._depth_sampler is not None:
                    try:
                        self._depth_sampler()
                    except Exception:
                        pass
                self._flush()
            except Exception:
                pass

    def _flush(self):
        with self._lock:
            emit = dict(self._emit)
            drop = dict(self._drop)
            drain = dict(self._drain)
            depths = list(self._depth_samples)
            self._emit.clear()
            self._drop.clear()
            self._drain.clear()
            self._depth_samples.clear()
            self._tick_seq += 1
            tick = self._tick_seq

        ts = time.time()
        rows = []
        rows.append(f"{ts:.3f}\tTICK\tpid={self._pid}\t{tick}")
        for (mtype, dst), n in emit.items():
            rows.append(f"{ts:.3f}\tEMIT\t{mtype}|{dst}\t{n}")
        for (sub, mtype), n in drop.items():
            rows.append(f"{ts:.3f}\tDROP\t{sub}|{mtype}\t{n}")
        for sub, n in drain.items():
            rows.append(f"{ts:.3f}\tDRAIN\t{sub}\t{n}")
        for d_ts, mod, depth in depths:
            rows.append(f"{d_ts:.3f}\tDEPTH\t{mod}\t{depth}")

        if not rows:
            return
        try:
            with open(CENSUS_LOG_PATH, "a") as f:
                f.write("\n".join(rows) + "\n")
        except OSError:
            pass


_global = _BusCensus()


def get_census() -> _BusCensus:
    return _global


def record_emission(msg_type: str, dst: str) -> None:
    if ENABLED:
        _global.record_emission(msg_type, dst)


def record_drop(subscriber: str, msg_type: str) -> None:
    if ENABLED:
        _global.record_drop(subscriber, msg_type)


def record_drain(subscriber: str, n: int) -> None:
    if ENABLED:
        _global.record_drain(subscriber, n)


def sample_queue_depths(subscribers: dict) -> None:
    if ENABLED:
        _global.sample_queue_depths(subscribers)


def start_writer():
    if ENABLED:
        _global.start()


def stop_writer():
    if ENABLED:
        _global.stop()


def register_depth_sampler(fn) -> None:
    if ENABLED:
        _global.register_depth_sampler(fn)
