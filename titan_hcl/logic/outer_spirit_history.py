"""
titan_hcl/logic/outer_spirit_history.py — Phase 2 outer-spirit producers.

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
# v1.32.0 / D-SPEC-94 — snapshot throttle for outer_spirit_trajectory.
# Phase C parent process ingests outer_spirit_45d from SHM via
# OuterSpiritHistory.ingest_outer_spirit_45d at every _gather_outer_sources
# call (~10s cadence) but trackers only need 30s snapshots to cover the
# full 60min trajectory window with the 120-slot deque.
SELF_TRAJ_SNAPSHOT_INTERVAL_S: float = 30.0


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
    """CHIT[26] circadian_alignment — RE-GROUNDED (rFP_trinity_dim_resonance,
    greenlit 2026-05-20).

    Measures the regularity of the Titan's *internal* clock rather than
    alignment to Earth's sunrise (arbitrary for a digital being). The signal is
    the cadence of π-cluster boundaries (``pi_heartbeat`` CLUSTER_START): a
    steady average spacing of epochs between clusters over a rolling 24h window
    means the internal circadian rhythm is in alignment; erratic spacing means
    misalignment.

        alignment = EMA( 1 − min(1, CV(inter_cluster_epoch_intervals_24h)) )

    where CV = stddev / mean (coefficient of variation). Steady cadence → low CV
    → high alignment. Cold-start (< MIN intervals) → 0.5.
    """

    _WINDOW_S: float = 86400.0      # 24h rolling window
    _MIN_INTERVALS: int = 3
    _EMA_ALPHA: float = 0.2

    def __init__(self):
        # (wall_ts, interval_epochs) sampled per gather (~10s) over the last 24h.
        # 8640 = 24h / 10s; the _WINDOW_S prune is the authoritative bound.
        # The "cluster" cadence signal is pi_heartbeat.pulse_count (the kernel's
        # π-rhythm pulse counter exposed in SHM); steady epochs-per-pulse → aligned.
        self._intervals: deque[tuple[float, float]] = deque(maxlen=8640)
        self._last_cluster_count: int = -1
        self._last_cluster_epoch: int = 0
        self._ema: Optional[float] = None
        self._lock = threading.Lock()

    def record_cluster(self, cluster_count, current_epoch) -> None:
        """Observe π-cluster state each gather. On a new cluster (count
        increments), record the inter-cluster epoch interval. Idempotent when
        called repeatedly with the same cluster_count between gathers.
        """
        if cluster_count is None or current_epoch is None:
            return
        with self._lock:
            cc = int(cluster_count)
            ep = int(current_epoch)
            if self._last_cluster_count < 0:
                # First observation — seed baseline, no interval yet.
                self._last_cluster_count = cc
                self._last_cluster_epoch = ep
                return
            if cc > self._last_cluster_count:
                interval = float(ep - self._last_cluster_epoch)
                if interval > 0:
                    self._intervals.append((time.time(), interval))
                self._last_cluster_count = cc
                self._last_cluster_epoch = ep

    def compute(self) -> float:
        now = time.time()
        with self._lock:
            while self._intervals and now - self._intervals[0][0] > self._WINDOW_S:
                self._intervals.popleft()
            vals = [iv for _, iv in self._intervals]
            n = len(vals)
            if n < self._MIN_INTERVALS:
                return 0.5
            mean = sum(vals) / n
            if mean <= 0.0:
                return 0.5
            var = sum((v - mean) ** 2 for v in vals) / n
            cv = (var ** 0.5) / mean
            raw = _clamp(1.0 - min(1.0, cv))
            if self._ema is None:
                self._ema = raw
            else:
                self._ema = (self._EMA_ALPHA * raw
                             + (1.0 - self._EMA_ALPHA) * self._ema)
            return _clamp(self._ema)


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
    dims that require plugin-process state. Owned by ``TitanHCL``.

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
        # v1.32.0 / D-SPEC-94 — CHIT[14] self_trajectory: 60min L2 trajectory
        # of outer_spirit_45d. SelfTrajectoryTracker held here so the parent
        # process (which has ShmReaderBank) can ingest 45D snapshots from
        # the SHM slot written by titan-outer-spirit-rs. Replaces the
        # worker-local _OUTER_SPIRIT_SNAPSHOTS deque from the Phase A+B
        # outer_spirit_worker (dead under Phase C; the Rust daemon writes
        # the 45D but never computed self_trajectory, leaving the field
        # missing from outer_spirit_history_stats fleet-wide).
        self.self_trajectory = SelfTrajectoryTracker()
        # Throttle 45D snapshot ingest to SELF_TRAJ_SNAPSHOT_INTERVAL_S so
        # the 120-slot deque covers ~60min (matches Phase A+B cadence).
        self._self_traj_last_ingest_ts: float = 0.0
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

    def ingest_pi_cluster(self, cluster_count, current_epoch) -> None:
        """Feed π-cluster cadence to circadian_alignment (rFP_trinity_dim_resonance).

        circadian_alignment now measures internal-clock regularity via the
        spacing of π-cluster boundaries, not Earth-sunrise action timestamps.
        """
        self.circadian_alignment.record_cluster(cluster_count, current_epoch)

    def ingest_outer_spirit_45d(self, vec_45d) -> None:
        """v1.32.0 / D-SPEC-94 — snapshot outer_spirit_45d for self_trajectory.

        Throttled to ``SELF_TRAJ_SNAPSHOT_INTERVAL_S`` (30s) so the
        ``SELF_TRAJ_DEQUE`` (120 slots) covers ~60min of trajectory —
        matches the Phase A+B `_OUTER_SPIRIT_SNAPSHOTS` cadence. Called
        from ``plugin._gather_outer_sources`` with the live 45D read
        from ``outer_spirit_45d.bin`` SHM (G18 SHM-direct).
        """
        now = time.time()
        if now - self._self_traj_last_ingest_ts < SELF_TRAJ_SNAPSHOT_INTERVAL_S:
            return
        self.self_trajectory.record_snapshot(vec_45d)
        self._self_traj_last_ingest_ts = now

    def refresh_dream_recall(self) -> None:
        """Called from heavy-stats refresher thread (60s)."""
        self.dream_recall.refresh()

    def get_stats(self) -> dict:
        """Snapshot consumed by titan-outer-spirit-rs via OUTER_SOURCES_SNAPSHOT
        → sensor_cache_outer_spirit.bin → tick_loop.rs source dict
        ``outer_spirit_history_stats``. SPEC §23.9 dim names preserved.

        ``outer_spirit_trajectory`` added v1.32.0 / D-SPEC-94 — closes the
        missing-field gap that left CHIT[14] self_trajectory locked at
        the Rust ``field_or_default`` default (0.0) fleet-wide.
        """
        return {
            "environmental_adaptation": self.environmental_adaptation.compute(),
            "graceful_rest": self.graceful_rest.compute(),
            "circadian_alignment": self.circadian_alignment.compute(),
            "dream_recall_ratio": self.dream_recall.get_value(),
            "outer_spirit_trajectory": self.self_trajectory.compute(),
        }
