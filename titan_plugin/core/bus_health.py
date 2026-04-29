"""
Bus health monitoring — observability bundle for META-CGN v3 rewire.

Built 2026-04-14 post-incident (rFP_meta_cgn_v3_clean_rewire.md § 10 Phase A).

Tracks the two canonical health indicators the architecture rests on:
  - Bus-clean (queue depths ~0 at steady state)
  - Emission rate against per-producer rate budget

The 2026-04-14 incident proved that without surfaced observability, bus
pressure is invisible until it cascades (184 670 'Failed to send:' lines
accumulated on T1 before anyone noticed). This module makes bus pressure
visible at minute 1 of any session, via /v4/bus-health endpoint and
/health embedding.

Three metrics families:
  1. Emission rates per (producer, event_type) — detects rate-budget
     violations before they flood.
  2. Queue depth per worker inbox — detects consumer back-pressure.
  3. Orphan signals (rejected_unknown) — detects producer wiring that
     has no SIGNAL_TO_PRIMITIVE entry. Logged WARN on first occurrence
     per tuple so an operator sees it immediately, not hours later.

Edge-detected events emitted to the cognitive bus (respecting the
architecture — discrete state transitions only, no continuous telemetry):
  BUS_BACKPRESSURE — fired once when any worker queue crosses > 30% full,
                     cleared when all queues drop below 20% (hysteresis).
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Callable, Optional
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)

# Backpressure thresholds on queue fill fraction
BACKPRESSURE_ENTER = 0.30   # > 30% full → WARN + event
BACKPRESSURE_EXIT = 0.20    # < 20% full → clear (hysteresis)

# Rolling window size for emission-rate counts (seconds)
RATE_WINDOW_S = 60.0
RATE_WINDOW_5MIN_S = 300.0

# Max size for per-tuple emission history deque (ring buffer by time)
RATE_HISTORY_MAX = 2000


class BusHealthMonitor:
    """Single instance lives on TitanCore. Thread-safe.

    Collect-only (no side effects on the bus) except for the one edge-
    detected BUS_BACKPRESSURE transition which is published via the
    injected publish_fn. Never modifies producer or consumer state."""

    def __init__(
        self,
        publish_fn: Optional[Callable[[str, dict], None]] = None,
    ):
        self._lock = threading.Lock()
        self._publish_fn = publish_fn
        self._start_time = time.time()

        # Emission tracking: (producer_src, consumer, event_type) →
        # deque[(timestamp, intensity)]
        self._emissions: dict[tuple, collections.deque] = {}

        # Orphan tracking: set of tuples (consumer, event_type) that had
        # no SIGNAL_TO_PRIMITIVE entry when received. Tuple added on first
        # occurrence; we emit exactly one WARN per tuple then go silent.
        self._orphan_tuples: set[tuple] = set()
        self._orphan_count_total = 0

        # Rate-gated drops — producer sent faster than min_interval_s.
        # (producer_src, consumer, event_type) → count
        self._rate_drops: dict[tuple, int] = {}

        # Queue depth snapshot — written by external poller (Guardian has
        # the worker queues; it calls update_queue_depths() periodically).
        # worker_name → (depth, maxsize, fraction)
        self._queue_depths: dict[str, tuple[int, int, float]] = {}

        # Backpressure state (edge-detected)
        self._backpressure_active = False

    # ------------------------------------------------------------------
    # Emission recording
    # ------------------------------------------------------------------
    def record_emission(
        self,
        producer_src: str,
        consumer: str,
        event_type: str,
        intensity: float = 1.0,
    ) -> None:
        """Record a successful META_CGN_SIGNAL emission."""
        key = (producer_src, consumer, event_type)
        now = time.time()
        with self._lock:
            deque = self._emissions.get(key)
            if deque is None:
                deque = collections.deque(maxlen=RATE_HISTORY_MAX)
                self._emissions[key] = deque
            deque.append((now, float(intensity)))

    def record_rate_drop(
        self,
        producer_src: str,
        consumer: str,
        event_type: str,
    ) -> None:
        """Record that an emission was dropped due to rate-gate violation.
        (producer emitted faster than its min_interval_s budget.)"""
        key = (producer_src, consumer, event_type)
        with self._lock:
            self._rate_drops[key] = self._rate_drops.get(key, 0) + 1

    def record_orphan(self, consumer: str, event_type: str) -> None:
        """Record a META_CGN_SIGNAL received with no SIGNAL_TO_PRIMITIVE
        entry. Logs WARN exactly once per (consumer, event_type) so the
        operator sees the problem immediately without log spam."""
        key = (consumer, event_type)
        first_time = False
        with self._lock:
            self._orphan_count_total += 1
            if key not in self._orphan_tuples:
                self._orphan_tuples.add(key)
                first_time = True
        if first_time:
            logger.warning(
                "[BusHealth] ORPHAN META_CGN_SIGNAL: (%s, %s) has no "
                "SIGNAL_TO_PRIMITIVE mapping — producer fires but consumer "
                "silently discards. Add a mapping or remove the producer. "
                "Subsequent orphans for this tuple will be counted but "
                "not logged.",
                consumer, event_type,
            )

    # ------------------------------------------------------------------
    # Queue depth tracking (external poller pushes snapshots)
    # ------------------------------------------------------------------
    def update_queue_depths(
        self,
        depths: dict[str, tuple[int, int]],
    ) -> None:
        """Update per-worker queue depth snapshot.

        Args:
            depths: worker_name → (current_depth, maxsize). maxsize=0 means
                    unbounded (fraction always 0.0 in that case).
        """
        now_max_fraction = 0.0
        with self._lock:
            new_snapshot: dict[str, tuple[int, int, float]] = {}
            for name, (depth, maxsize) in depths.items():
                frac = (depth / maxsize) if maxsize > 0 else 0.0
                new_snapshot[name] = (depth, maxsize, frac)
                if frac > now_max_fraction:
                    now_max_fraction = frac
            self._queue_depths = new_snapshot

            # Edge-detect backpressure
            was_active = self._backpressure_active
            if not was_active and now_max_fraction > BACKPRESSURE_ENTER:
                self._backpressure_active = True
                transition = "enter"
            elif was_active and now_max_fraction < BACKPRESSURE_EXIT:
                self._backpressure_active = False
                transition = "exit"
            else:
                transition = None

        if transition and self._publish_fn is not None:
            try:
                self._publish_fn(
                    "BUS_BACKPRESSURE",
                    {
                        "transition": transition,
                        "max_queue_fraction": round(now_max_fraction, 3),
                        "queues": {
                            name: {"depth": d, "maxsize": m, "fraction": round(f, 3)}
                            for name, (d, m, f) in new_snapshot.items()
                            if f > 0.1  # Only include non-trivial
                        },
                    },
                )
            except Exception as e:
                from titan_plugin.utils.silent_swallow import swallow_warn
                swallow_warn("[BusHealth] publish_fn error", e,
                             key="bus_health.publish_fn")

    # ------------------------------------------------------------------
    # Snapshot — read-only summary for /v4/bus-health endpoint
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Build a JSON-safe snapshot. Holds the lock briefly."""
        now = time.time()
        with self._lock:
            # Compute per-producer rates over 1-min and 5-min windows
            producers = []
            for key, deque in self._emissions.items():
                producer_src, consumer, event_type = key
                count_1min = sum(
                    1 for (ts, _) in deque if (now - ts) <= RATE_WINDOW_S
                )
                count_5min = sum(
                    1 for (ts, _) in deque if (now - ts) <= RATE_WINDOW_5MIN_S
                )
                rate_1min_hz = count_1min / RATE_WINDOW_S
                rate_5min_hz = count_5min / RATE_WINDOW_5MIN_S
                rate_drops = self._rate_drops.get(key, 0)
                producers.append({
                    "src": producer_src,
                    "consumer": consumer,
                    "event_type": event_type,
                    "count_1min": count_1min,
                    "count_5min": count_5min,
                    "rate_1min_hz": round(rate_1min_hz, 4),
                    "rate_5min_hz": round(rate_5min_hz, 4),
                    "total_emissions": len(deque),
                    "rate_drops": rate_drops,
                })

            total_rate_1min = sum(p["rate_1min_hz"] for p in producers)

            # Queue depths
            queues = {
                name: {"depth": d, "maxsize": m, "fraction": round(f, 3)}
                for name, (d, m, f) in self._queue_depths.items()
            }
            max_queue_fraction = max(
                (info["fraction"] for info in queues.values()), default=0.0
            )

            return {
                "ts": now,
                "uptime_s": round(now - self._start_time, 1),
                "overall_state": self._overall_state_locked(
                    max_queue_fraction, total_rate_1min
                ),
                "rate_budget_hz": 0.5,  # rFP § 3 hard budget
                "total_emission_rate_1min_hz": round(total_rate_1min, 4),
                "producers": sorted(
                    producers, key=lambda p: -p["rate_1min_hz"]
                ),
                "queues": queues,
                "max_queue_fraction": round(max_queue_fraction, 3),
                "backpressure_active": self._backpressure_active,
                "orphans": {
                    "total_count": self._orphan_count_total,
                    # Cast tuples → lists for JSON-safety across the wire.
                    "unique_tuples": sorted(list(t) for t in self._orphan_tuples),
                },
            }

    def _overall_state_locked(
        self, max_q_frac: float, total_rate: float
    ) -> str:
        """Caller must hold the lock. Returns overall bus-health state string."""
        if self._orphan_tuples:
            return "warning"   # any orphan is a design violation to surface
        if total_rate > 0.5:
            return "critical"  # over-budget emission rate
        if max_q_frac > BACKPRESSURE_ENTER:
            return "critical"  # consumers can't keep up
        if total_rate > 0.4 or max_q_frac > 0.15:
            return "warning"   # approaching limits
        return "healthy"


# ----------------------------------------------------------------------
# Module-level singleton accessor — bus.py imports from here
# ----------------------------------------------------------------------
_singleton: Optional[BusHealthMonitor] = None
_singleton_lock = threading.Lock()


def set_global_monitor(monitor: BusHealthMonitor) -> None:
    """Wire a BusHealthMonitor as the global singleton. Called once at
    TitanCore boot. Before this is called, emission recording is a no-op."""
    global _singleton
    with _singleton_lock:
        _singleton = monitor


def get_global_monitor() -> Optional[BusHealthMonitor]:
    """Fetch the current global monitor, or None if not wired yet."""
    return _singleton
