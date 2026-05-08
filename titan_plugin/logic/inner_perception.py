"""
titan_plugin/logic/inner_perception.py — Phase 2 perception producers.

Three trackers feeding the inner_mind 15D feeling block (SPEC §23.5):

  * ``AudioPerception`` — feeds inner_mind[5] inner_hearing.
    Counts audio/music creates over a rolling 1h window. Combined in
    mind_tensor with ``sense_hearing_ambient`` (mind_worker.py:564).

  * ``VisualPerception`` — feeds inner_mind[7] inner_sight.
    Symmetric to AudioPerception, for art creates. Combined with
    ``sense_vision_ambient`` (mind_worker.py:519).

  * ``AmbientChangeMonitor`` — feeds inner_mind[9] inner_smell.
    Rolling stddev of (cpu_thermal + circadian_phase) over a 60-sample
    window, clipped to [0, 1]. Sampled at 1 Hz from a daemon thread.

Plus an ``InnerPerceptionState`` aggregator that owns the three trackers
plus ``_last_create_ts`` (a single float updated on every art/audio/text/
music create — feeds outer_spirit ANANDA[41] creative_tension via
``creative_tension = hormone_levels.CREATIVITY * min(1, dt/600)``).

Design notes (G18-G22 compliant):
  * All state is in-process (deques + ints + floats). ``get_*`` reads are
    O(1). No DB I/O on the hot path.
  * Ambient ticker is a single daemon thread, 1 Hz; ``stop()`` for clean
    shutdown.
  * No bus.request anywhere. Producers update via direct method calls;
    consumers read via plugin._gather_outer_sources → OUTER_SOURCES_SNAPSHOT.

SPEC §23.5 formulas (locked 2026-05-06):
  * inner_hearing (feeling[0]): 0.5*min(1,audio_creates_recent/5) + 0.5*sense_hearing_ambient
  * inner_sight   (feeling[2]): 0.5*min(1,art_creates_recent/5)   + 0.5*sense_vision_ambient
  * inner_smell   (feeling[4]): rolling stddev of (cpu_thermal + circadian) over 60s, clipped [0,1]

The mind_tensor module computes the final dim values from the dicts this
module exposes; the producers here only own the rolling-window state.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

# 1h rolling window for "creates_recent" counts. Aligns with the 1h cadence
# used elsewhere (posts_last_hour, actions_this_hour) and with the SPEC
# §23.5 formula scaling (5 creates/hr → max sensitivity).
_CREATES_WINDOW_S = 3600.0

# AmbientChangeMonitor window: 60 samples × 1Hz tick = 60s rolling window
# per SPEC §23.5 inner_smell ("rolling stddev ... over 60s").
_AMBIENT_SAMPLES = 60
_AMBIENT_TICK_S = 1.0


class _CreateTracker:
    """Rolling timestamp deque + count-in-window helper."""

    __slots__ = ("_timestamps", "_max_keep", "_lock")

    def __init__(self, max_keep: int = 200):
        self._timestamps: deque[float] = deque(maxlen=max_keep)
        self._max_keep = max_keep
        self._lock = threading.Lock()

    def record(self, ts: Optional[float] = None) -> None:
        with self._lock:
            self._timestamps.append(time.time() if ts is None else ts)

    def count_in_last(self, window_s: float = _CREATES_WINDOW_S) -> int:
        cutoff = time.time() - window_s
        with self._lock:
            return sum(1 for t in self._timestamps if t >= cutoff)


class AudioPerception:
    """Audio/music create tracker for inner_mind[5] inner_hearing."""

    def __init__(self):
        self._tracker = _CreateTracker()

    def record_create(self, ts: Optional[float] = None) -> None:
        self._tracker.record(ts)

    def get_state(self, window_s: float = _CREATES_WINDOW_S) -> dict:
        return {"creates_recent": self._tracker.count_in_last(window_s)}


class VisualPerception:
    """Art-create tracker for inner_mind[7] inner_sight."""

    def __init__(self):
        self._tracker = _CreateTracker()

    def record_create(self, ts: Optional[float] = None) -> None:
        self._tracker.record(ts)

    def get_state(self, window_s: float = _CREATES_WINDOW_S) -> dict:
        return {"creates_recent": self._tracker.count_in_last(window_s)}


class AmbientChangeMonitor:
    """1 Hz sampler + 60 s rolling stddev for inner_mind[9] inner_smell.

    The combined signal is ``cpu_thermal + circadian_phase``. Rolling
    stddev over the deque, normalized so a stddev of 0.25 maps to 1.0
    (anything above is fully saturated). Returns 0.0 cold-start (n<5)
    — the SPEC-correct value when no environmental change has been
    observed yet.
    """

    def __init__(self, sampler):
        """sampler: callable() → (cpu_thermal_float, circadian_float)."""
        self._sampler = sampler
        self._history: deque[float] = deque(maxlen=_AMBIENT_SAMPLES)
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="ambient-change-monitor")
        self._thread.start()
        logger.info("[AmbientChangeMonitor] sampler started "
                    "(samples=%d, tick=%.1fs)", _AMBIENT_SAMPLES, _AMBIENT_TICK_S)

    def stop(self) -> None:
        self._stop_evt.set()

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                cpu, circ = self._sampler()
                with self._lock:
                    self._history.append(float(cpu) + float(circ))
            except Exception as e:
                logger.debug("[AmbientChangeMonitor] sample error: %s", e)
            self._stop_evt.wait(_AMBIENT_TICK_S)

    def get_value(self) -> float:
        """Rolling stddev clipped to [0, 1]. Cold-start (n<5) → 0.0."""
        with self._lock:
            n = len(self._history)
            if n < 5:
                return 0.0
            mean = sum(self._history) / n
            var = sum((v - mean) ** 2 for v in self._history) / n
        # Normalize: stddev of 0.25 (combined cpu+circ) ~= max change.
        # 0.25 chosen because cpu_thermal and circadian each span [0,1],
        # so the sum spans [0,2]; a stddev of 0.25 over 60s is genuinely
        # turbulent. Above that → saturated at 1.0.
        return min(1.0, math.sqrt(var) * 4.0)


class InnerPerceptionState:
    """Aggregates AudioPerception + VisualPerception + AmbientChangeMonitor
    plus _last_create_ts. One owner per plugin instance.
    """

    def __init__(self, ambient_sampler):
        self.audio = AudioPerception()
        self.visual = VisualPerception()
        self.ambient = AmbientChangeMonitor(ambient_sampler)
        self._last_create_ts: float = 0.0

    def start(self) -> None:
        self.ambient.start()

    def stop(self) -> None:
        self.ambient.stop()

    def notify_create(self, type_: str, ts: Optional[float] = None) -> None:
        """Hook called from every observatory_db.record_expressive site.

        type_ ∈ {"art", "audio", "music", "text", "haiku", "x_post", ...}.
        Anything that registers as a creative emission updates
        ``_last_create_ts``. Audio/music creates also feed AudioPerception;
        art creates feed VisualPerception.
        """
        now = time.time() if ts is None else ts
        self._last_create_ts = now
        t = (type_ or "").lower()
        if t in ("audio", "music"):
            self.audio.record_create(now)
        elif t == "art":
            self.visual.record_create(now)

    def get_stats(self) -> dict:
        """Plugin-side snapshot consumed by mind_worker via OUTER_SOURCES_SNAPSHOT.

        ``audio_state`` / ``visual_state`` are the dicts the SPEC §23.5
        formulas read from. ``ambient_change`` is the precomputed rolling-
        stddev value (mind_tensor uses it directly per SPEC). ``last_create_ts``
        feeds outer_spirit ANANDA[41] creative_tension.
        """
        return {
            "audio_state": self.audio.get_state(),
            "visual_state": self.visual.get_state(),
            "ambient_change": self.ambient.get_value(),
            "last_create_ts": self._last_create_ts,
        }
