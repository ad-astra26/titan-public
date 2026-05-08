"""
titan_plugin/logic/outer_spirit_history.py — Phase 2 outer-spirit producers.

Tracker classes for the five outer-spirit history-dependent dims that
require plugin-side state (see SPEC §23.9):

  * ``EnvironmentalAdaptationTracker`` — SAT[11] environmental_adaptation.
    Variance of assessment.score during high-cpu_thermal windows;
    high stability under load → high adaptation. Threshold cpu_thermal>0.6.

  * ``GracefulRestTracker`` — ANANDA[40] graceful_rest.
    Min assessment.score during low-cpu_spike + low-circadian periods.
    Thresholds cpu_spike<0.3 AND circadian<0.3 (night-phase + low load).

  * ``CircadianAlignmentTracker`` — CHIT[26] circadian_alignment.
    Mean of sin(2π·hour/24) across last 200 action timestamps; high
    when actions cluster around the sin peak (sunrise hour). Linear
    remap (mean+1)/2 → [0,1] (0.5=random, 1.0=fully aligned).

  * ``SelfTrajectoryTracker`` — CHIT[29] self_trajectory.
    L2 distance between outer_spirit_45d_now and outer_spirit_45d_1h_ago
    (deque maxlen=120 × 30s = 60min). Normalized via /5.0 then clamped.
    Worker-local in the implementation (outer_spirit_worker holds a
    module-level deque, mirroring Phase 1 `_DELTA_HISTORY`); class
    provided here for testability.

  * ``DreamRecallProducer`` — CHIT[25] dream_recall (SPEC refinement
    2026-05-07). Reads ``experiential_memory.get_recall_ratio()`` —
    fraction of stored insights with ``recall_count >= 1``. The recall
    mechanism is cosine-similarity via ``recall_by_state`` (called
    from spirit_worker:5966 + 10200), already live.

Plus an ``OuterSpiritHistory`` aggregator owned by the plugin process.
Producers update via direct method calls from ``_gather_outer_sources``;
the aggregator's ``get_stats()`` is consumed by ``outer_spirit_worker``
via OUTER_SOURCES_SNAPSHOT (one-way publish, G18-G22 compliant).

ANANDA[41] creative_tension and ANANDA[44] flow_state do not need
trackers — both are pure functions of cache values and computed inline
in outer_spirit_worker (see SPEC §23.9).
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

# Maker-locked thresholds (2026-05-07; see Phase 2 design discussion):
HIGH_THERMAL_THRESHOLD: float = 0.6   # cpu_thermal > this → "load" window
LOW_SPIKE_THRESHOLD: float = 0.3      # cpu_spike_rate < this → "calm"
LOW_CIRCADIAN_THRESHOLD: float = 0.3  # circadian_phase < this → "night"

# Window sizes (Maker-locked 2026-05-07):
ENV_ADAPT_DEQUE: int = 30        # last 30 high-thermal assessments
ENV_ADAPT_MIN_N: int = 5
GRACEFUL_REST_DEQUE: int = 30    # last 30 low-load assessments
GRACEFUL_REST_MIN_N: int = 3
CIRCADIAN_DEQUE: int = 200       # SPEC §23.9: last 200 action timestamps
CIRCADIAN_MIN_N: int = 10
SELF_TRAJ_DEQUE: int = 120       # 120 × 30s = 60min, SPEC §23.9


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


class EnvironmentalAdaptationTracker:
    """SAT[11] environmental_adaptation. ``1 - clamped(variance(scores))``
    over the rolling window of assessments collected while
    cpu_thermal > HIGH_THERMAL_THRESHOLD. Cold-start (n<MIN_N) → 0.5.
    """

    def __init__(self):
        self._scores: deque[float] = deque(maxlen=ENV_ADAPT_DEQUE)
        self._lock = threading.Lock()

    def record(self, assessment_score: float, cpu_thermal: float) -> None:
        if cpu_thermal is None or assessment_score is None:
            return
        if float(cpu_thermal) > HIGH_THERMAL_THRESHOLD:
            with self._lock:
                self._scores.append(float(assessment_score))

    def compute(self) -> float:
        with self._lock:
            n = len(self._scores)
            if n < ENV_ADAPT_MIN_N:
                return 0.5  # cold-start: no measurement yet (see SPEC §23.9 SAT[11])
            mean = sum(self._scores) / n
            var = sum((s - mean) ** 2 for s in self._scores) / n
        # Variance ∈ [0, 0.25] for scores ∈ [0,1]; multiply by 4 to span
        # [0,1] before subtracting from 1 — high stability → 1.0.
        return _clamp(1.0 - min(1.0, var * 4.0))


class GracefulRestTracker:
    """ANANDA[40] graceful_rest. Min assessment score during
    cpu_spike < LOW_SPIKE_THRESHOLD AND circadian < LOW_CIRCADIAN_THRESHOLD
    periods. Cold-start (n<MIN_N) → 0.5.
    """

    def __init__(self):
        self._scores: deque[float] = deque(maxlen=GRACEFUL_REST_DEQUE)
        self._lock = threading.Lock()

    def record(
        self,
        assessment_score: float,
        cpu_spike_rate: float,
        circadian_phase: float,
    ) -> None:
        if (assessment_score is None or cpu_spike_rate is None
                or circadian_phase is None):
            return
        if (float(cpu_spike_rate) < LOW_SPIKE_THRESHOLD
                and float(circadian_phase) < LOW_CIRCADIAN_THRESHOLD):
            with self._lock:
                self._scores.append(float(assessment_score))

    def compute(self) -> float:
        with self._lock:
            n = len(self._scores)
            if n < GRACEFUL_REST_MIN_N:
                return 0.5
            return _clamp(min(self._scores))


class CircadianAlignmentTracker:
    """CHIT[26] circadian_alignment. Mean of ``sin(2π·hour/24)`` across
    the last 200 action timestamps, linearly remapped to [0,1] via
    ``(mean+1)/2``. Cold-start (n<MIN_N) → 0.5 (random distribution = no
    alignment).

    Each timestamp's hour is derived as ``(t mod 86400) / 86400 * 24``.
    sin(2π·6/24)=1 → peak at sunrise hour; sin(2π·18/24)=-1 → trough at
    sunset. The (+1)/2 remap means: 0.5 = no preference, 1.0 = fully
    aligned with sunrise window, 0.0 = fully anti-aligned.
    """

    def __init__(self):
        self._timestamps: deque[float] = deque(maxlen=CIRCADIAN_DEQUE)
        # Watermark: only timestamps strictly greater than this are appended.
        # Prevents duplicate ingestion when ``record_many`` is called every
        # 10 s gather cycle with the same ``agency.recent_actions_detail``
        # slice (the slice doesn't change between gathers unless new actions
        # fire). Without this, deque(maxlen=200) fills with ~30 unique
        # timestamps repeated ~7×, skewing the mean toward the most recent
        # cluster instead of spreading across the real action-history window.
        self._last_ts_seen: float = 0.0
        self._lock = threading.Lock()

    def record(self, ts: float) -> None:
        with self._lock:
            t = float(ts)
            if t > self._last_ts_seen:
                self._timestamps.append(t)
                self._last_ts_seen = t

    def record_many(self, timestamps) -> None:
        """Bulk-add timestamps (used to seed from ``recent_actions_detail``).
        Filtered against ``_last_ts_seen`` watermark for idempotent ingestion.
        """
        with self._lock:
            new_high_water = self._last_ts_seen
            for t in timestamps:
                if t is None:
                    continue
                tf = float(t)
                if tf > self._last_ts_seen:
                    self._timestamps.append(tf)
                    if tf > new_high_water:
                        new_high_water = tf
            self._last_ts_seen = new_high_water

    def compute(self) -> float:
        with self._lock:
            n = len(self._timestamps)
            if n < CIRCADIAN_MIN_N:
                return 0.5
            xs = [
                math.sin(2.0 * math.pi * ((t % 86400.0) / 86400.0))
                for t in self._timestamps
            ]
        mean_x = sum(xs) / len(xs)
        return _clamp(0.5 + mean_x * 0.5)


class SelfTrajectoryTracker:
    """CHIT[29] self_trajectory. L2 distance between current
    outer_spirit_45d and 1h-ago snapshot, normalized /5.0 and clamped [0,1].
    Cold-start (n<2) → 0.0 (no trajectory yet — SPEC-correct).
    """

    def __init__(self, deque_size: int = SELF_TRAJ_DEQUE):
        self._snapshots: deque[tuple[float, list[float]]] = deque(
            maxlen=deque_size)
        self._lock = threading.Lock()

    def record_snapshot(self, vec_45d) -> None:
        try:
            v = [float(x) for x in vec_45d]
        except Exception:
            return
        if len(v) != 45:
            return
        with self._lock:
            self._snapshots.append((time.time(), v))

    def compute(self) -> float:
        with self._lock:
            n = len(self._snapshots)
            if n < 2:
                return 0.0
            old = self._snapshots[0][1]
            new = self._snapshots[-1][1]
        # Pad / truncate defensively in case schema ever drifts (45 fixed).
        L = min(len(old), len(new))
        dist_sq = sum((old[i] - new[i]) ** 2 for i in range(L))
        return _clamp(math.sqrt(dist_sq) / 5.0)


class DreamRecallProducer:
    """CHIT[25] dream_recall (SPEC refinement 2026-05-07).

    Wraps ``experiential_memory.get_recall_ratio()`` — fraction of stored
    insights with ``recall_count >= 1``. The recall mechanism is the
    cosine-similarity ``recall_by_state`` (see experiential_memory.py:202)
    which is invoked from spirit_worker.py:5966 + 10200 in production
    and auto-increments recall_count.

    The original SPEC §23.9 row 25 formula assumed text content in
    ``dreaming_history``; the actual schema is stats-only. Refining to
    the existing live producer; SPEC §23.9 updated in this commit.

    Heavy-stats integration: ``get_ratio()`` does a SQL COUNT — must be
    refreshed via the existing ``_heavy_stats_cache`` 60s background
    pattern, NEVER inline in ``_gather_outer_sources``.
    """

    def __init__(self, e_mem_lookup):
        """e_mem_lookup: callable() returning the ExperientialMemory instance, or None."""
        self._lookup = e_mem_lookup
        self._cached_ratio: float = 0.0  # cold-start: 0.0 (no recalls yet, SPEC-valid)

    def refresh(self) -> None:
        """Called from the heavy-stats refresher thread (60s cadence)."""
        try:
            e_mem = self._lookup()
            if e_mem is not None and hasattr(e_mem, "get_recall_ratio"):
                self._cached_ratio = float(e_mem.get_recall_ratio())
        except Exception as e:
            logger.debug("[DreamRecallProducer] refresh error: %s", e)

    def get_value(self) -> float:
        return _clamp(self._cached_ratio)


class OuterSpiritHistory:
    """Plugin-side aggregator for the 4 history-dependent outer_spirit
    dims that require plugin-process state. Owned by ``TitanPlugin``.

    ``self_trajectory`` is computed worker-local in
    ``outer_spirit_worker._collect_tick`` (the worker holds the 45D it
    just produced); the SelfTrajectoryTracker class is exposed here for
    unit testing only.
    """

    def __init__(self, e_mem_lookup):
        self.environmental_adaptation = EnvironmentalAdaptationTracker()
        self.graceful_rest = GracefulRestTracker()
        self.circadian_alignment = CircadianAlignmentTracker()
        self.dream_recall = DreamRecallProducer(e_mem_lookup)
        # Track which assessment scores we've already consumed (by index)
        # so each score lands at most once in env_adapt + graceful_rest.
        self._last_assessment_consumed: float = 0.0

    def ingest_assessments(
        self,
        recent_assessments: list[dict],
        cpu_thermal: float,
        cpu_spike_rate: float,
        circadian_phase: float,
    ) -> None:
        """Consume new assessment-result dicts. Each must carry ``score``
        and ``ts``; older-than-last-consumed entries are skipped (idempotent
        across multiple gather cycles).
        """
        new_high_water = self._last_assessment_consumed
        for a in recent_assessments or []:
            ts = float(a.get("ts", 0.0))
            score = a.get("score", a.get("avg_score"))
            if ts <= self._last_assessment_consumed or score is None:
                continue
            self.environmental_adaptation.record(score, cpu_thermal)
            self.graceful_rest.record(score, cpu_spike_rate, circadian_phase)
            if ts > new_high_water:
                new_high_water = ts
        self._last_assessment_consumed = new_high_water

    def ingest_action_timestamps(self, timestamps) -> None:
        """Bulk-add action timestamps for circadian_alignment."""
        self.circadian_alignment.record_many(timestamps)

    def refresh_dream_recall(self) -> None:
        """Called from heavy-stats refresher thread (60s)."""
        self.dream_recall.refresh()

    def get_stats(self) -> dict:
        """Snapshot consumed by outer_spirit_worker via OUTER_SOURCES_SNAPSHOT.
        SPEC §23.9 dim names preserved.
        """
        return {
            "environmental_adaptation": self.environmental_adaptation.compute(),
            "graceful_rest": self.graceful_rest.compute(),
            "circadian_alignment": self.circadian_alignment.compute(),
            "dream_recall_ratio": self.dream_recall.get_value(),
        }
