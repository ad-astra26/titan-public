"""
titan_plugin/logic/resonance.py — V4 Time Awareness: Resonance Detector.

Detects resonance between inner↔outer counterpart sphere clock pairs.
Resonance is the Proof of Harmony (PoH) — evidence that corresponding
Trinity components work in unison.

Three pairs are tracked:
  - Body pair:   inner_body ↔ outer_body
  - Mind pair:   inner_mind ↔ outer_mind
  - Spirit pair: inner_spirit ↔ outer_spirit

Resonance detection uses phase alignment from sphere clock pulses:
  - When both clocks in a pair pulse within a phase window (π/6 = 30°),
    the pair scores a "resonant cycle"
  - After N consecutive resonant cycles, the pair achieves RESONANCE
  - A resonant pair emits a BIG PULSE

When ALL 3 pairs achieve resonance simultaneously, a GREAT PULSE can be
generated (handled by Phase 3 UnifiedSpirit).

Phase-based detection mechanics:
  - Each pair tracks recent pulses from both inner and outer clocks
  - When a pulse arrives, check if the counterpart pulsed recently
  - If phase difference < threshold → resonant cycle recorded
  - If streak of resonant cycles >= required → BIG PULSE emitted
"""
import logging
import math
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Defaults (overridden by config) ──────────────────────────────────

DEFAULT_PHASE_THRESHOLD = math.pi / 6  # 30° — max phase difference for resonance
DEFAULT_RESONANCE_CYCLES = 3           # Consecutive resonant cycles needed for BIG PULSE
DEFAULT_PULSE_WINDOW = 120.0           # Seconds — max time between counterpart pulses

# Pair definitions
PAIRS = ("body", "mind", "spirit")


class ResonancePair:
    """
    Tracks resonance state for one inner↔outer pair.

    Records pulse events from both sides. When both sides pulse within
    the time window AND their phases are aligned, it counts as a
    resonant cycle. After enough consecutive resonant cycles, the
    pair achieves resonance and emits a BIG PULSE.
    """

    def __init__(
        self,
        name: str,
        phase_threshold: float = DEFAULT_PHASE_THRESHOLD,
        required_cycles: int = DEFAULT_RESONANCE_CYCLES,
        pulse_window: float = DEFAULT_PULSE_WINDOW,
    ):
        self.name = name
        self.phase_threshold = phase_threshold
        self.required_cycles = required_cycles
        self.pulse_window = pulse_window

        # Last pulse timestamps and phases
        self._inner_last_pulse_ts: float = 0.0
        self._inner_last_phase: float = 0.0
        self._inner_pulse_count: int = 0
        self._inner_fresh: bool = False  # New pulse since last resonance check

        self._outer_last_pulse_ts: float = 0.0
        self._outer_last_phase: float = 0.0
        self._outer_pulse_count: int = 0
        self._outer_fresh: bool = False  # New pulse since last resonance check

        # Resonance tracking
        self._consecutive_resonant: int = 0
        self._total_resonant_cycles: int = 0
        self._total_checks: int = 0
        self._big_pulse_count: int = 0
        self._last_big_pulse_ts: float = 0.0

        # Current resonance state
        self._is_resonant: bool = False

    def record_pulse(self, component: str, pulse_event: dict) -> Optional[dict]:
        """
        Record a sphere clock pulse and check for resonance.

        Args:
            component: Full component name (e.g., "inner_body", "outer_mind")
            pulse_event: Pulse event dict from SphereClock

        Returns:
            BIG PULSE event dict if resonance threshold reached, None otherwise.
        """
        pulse_ts = pulse_event.get("ts", time.time())
        is_inner = component.startswith("inner_")

        if is_inner:
            self._inner_last_pulse_ts = pulse_ts
            self._inner_last_phase = pulse_event.get("pulse_count", 0) * 0.1  # Approximate phase
            self._inner_pulse_count += 1
            self._inner_fresh = True
        else:
            self._outer_last_pulse_ts = pulse_ts
            self._outer_last_phase = pulse_event.get("pulse_count", 0) * 0.1
            self._outer_pulse_count += 1
            self._outer_fresh = True

        # Only check resonance when BOTH sides have a fresh pulse
        if self._inner_fresh and self._outer_fresh:
            return self._check_resonance(pulse_ts)
        return None

    def record_pulse_with_phase(
        self, component: str, phase: float, pulse_ts: float
    ) -> Optional[dict]:
        """
        Record a pulse with explicit phase information from SphereClockEngine.

        This is the preferred method when sphere clock phases are available.
        """
        is_inner = component.startswith("inner_")

        if is_inner:
            self._inner_last_pulse_ts = pulse_ts
            self._inner_last_phase = phase
            self._inner_pulse_count += 1
            self._inner_fresh = True
        else:
            self._outer_last_pulse_ts = pulse_ts
            self._outer_last_phase = phase
            self._outer_pulse_count += 1
            self._outer_fresh = True

        if self._inner_fresh and self._outer_fresh:
            return self._check_resonance(pulse_ts)
        return None

    def _check_resonance(self, now: float) -> Optional[dict]:
        """
        Check if the inner↔outer pair is resonating.

        Both must have pulsed within the time window, and their
        phase difference must be below the threshold.
        """
        self._total_checks += 1

        # Consume fresh flags — one check per pair of pulses
        self._inner_fresh = False
        self._outer_fresh = False

        # Both sides must have pulsed at least once
        if self._inner_last_pulse_ts == 0.0 or self._outer_last_pulse_ts == 0.0:
            return None

        # Check time proximity — both must have pulsed within the window
        time_diff = abs(self._inner_last_pulse_ts - self._outer_last_pulse_ts)
        if time_diff > self.pulse_window:
            # Too far apart temporally — break resonance streak
            self._consecutive_resonant = 0
            self._is_resonant = False
            return None

        # Check phase alignment
        phase_diff = self._phase_difference(self._inner_last_phase, self._outer_last_phase)

        if phase_diff <= self.phase_threshold:
            # Resonant cycle!
            self._consecutive_resonant += 1
            self._total_resonant_cycles += 1
            self._is_resonant = True

            logger.debug(
                "[Resonance:%s] Resonant cycle #%d (phase_diff=%.3f < %.3f, time_diff=%.1fs)",
                self.name, self._consecutive_resonant, phase_diff,
                self.phase_threshold, time_diff,
            )

            # Check if we've reached the threshold for BIG PULSE
            if self._consecutive_resonant >= self.required_cycles:
                return self._generate_big_pulse(phase_diff, time_diff)
        else:
            # Phase misalignment — break streak
            self._consecutive_resonant = 0
            self._is_resonant = False

        return None

    @staticmethod
    def _phase_difference(phase_a: float, phase_b: float) -> float:
        """
        Compute the angular difference between two phases.

        Returns value in [0, π] — handles wrapping correctly.
        """
        diff = abs(phase_a - phase_b) % (2.0 * math.pi)
        if diff > math.pi:
            diff = 2.0 * math.pi - diff
        return diff

    def _generate_big_pulse(self, phase_diff: float, time_diff: float) -> dict:
        """Generate a BIG PULSE event — proof of harmony for this pair."""
        self._big_pulse_count += 1
        self._last_big_pulse_ts = time.time()

        # Reset streak (start fresh for next BIG PULSE)
        self._consecutive_resonant = 0

        big_pulse = {
            "pair": self.name,
            "big_pulse_count": self._big_pulse_count,
            "phase_diff": round(phase_diff, 4),
            "time_diff": round(time_diff, 2),
            "inner_pulse_count": self._inner_pulse_count,
            "outer_pulse_count": self._outer_pulse_count,
            "total_resonant_cycles": self._total_resonant_cycles,
            "ts": time.time(),
        }

        logger.info(
            "[Resonance:%s] BIG PULSE #%d — PoH achieved! "
            "phase_diff=%.3f time_diff=%.1fs resonant_cycles=%d",
            self.name, self._big_pulse_count, phase_diff, time_diff,
            self._total_resonant_cycles,
        )

        return big_pulse

    @property
    def is_resonant(self) -> bool:
        """Whether this pair is currently in resonance."""
        return self._is_resonant

    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "is_resonant": self._is_resonant,
            "consecutive_resonant": self._consecutive_resonant,
            "required_cycles": self.required_cycles,
            "total_resonant_cycles": self._total_resonant_cycles,
            "total_checks": self._total_checks,
            "big_pulse_count": self._big_pulse_count,
            "inner_pulse_count": self._inner_pulse_count,
            "outer_pulse_count": self._outer_pulse_count,
            "last_big_pulse_ts": self._last_big_pulse_ts,
            "phase_threshold": round(self.phase_threshold, 4),
            "pulse_window": self.pulse_window,
        }

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "name": self.name,
            "_consecutive_resonant": self._consecutive_resonant,
            "_total_resonant_cycles": self._total_resonant_cycles,
            "_total_checks": self._total_checks,
            "_big_pulse_count": self._big_pulse_count,
            "_last_big_pulse_ts": self._last_big_pulse_ts,
            "_inner_pulse_count": self._inner_pulse_count,
            "_outer_pulse_count": self._outer_pulse_count,
            "_is_resonant": self._is_resonant,
        }

    def from_dict(self, data: dict) -> None:
        """Restore from persistence."""
        self._consecutive_resonant = data.get("_consecutive_resonant", 0)
        self._total_resonant_cycles = data.get("_total_resonant_cycles", 0)
        self._total_checks = data.get("_total_checks", 0)
        self._big_pulse_count = data.get("_big_pulse_count", 0)
        self._last_big_pulse_ts = data.get("_last_big_pulse_ts", 0.0)
        self._inner_pulse_count = data.get("_inner_pulse_count", 0)
        self._outer_pulse_count = data.get("_outer_pulse_count", 0)
        self._is_resonant = data.get("_is_resonant", False)


class ResonanceDetector:
    """
    Orchestrates resonance detection for all 3 inner↔outer pairs.

    Consumes SPHERE_PULSE events from the bus and feeds them to the
    appropriate ResonancePair. Tracks which pairs are currently resonant
    and whether all 3 are resonant simultaneously (GREAT PULSE condition).

    Usage from Spirit worker:
        detector = ResonanceDetector(config)
        # On each SPHERE_PULSE event:
        big_pulse = detector.record_pulse(pulse_event)
        if big_pulse:
            publish BIG_PULSE to bus
        # Check for GREAT PULSE condition:
        if detector.all_resonant():
            # Phase 3 UnifiedSpirit will handle GREAT PULSE generation
    """

    def __init__(self, config: Optional[dict] = None, data_dir: str = "./data"):
        import json
        import os

        cfg = config or {}
        self._phase_threshold = float(cfg.get("phase_threshold", DEFAULT_PHASE_THRESHOLD))
        self._required_cycles = int(cfg.get("resonance_cycles", DEFAULT_RESONANCE_CYCLES))
        self._pulse_window = float(cfg.get("pulse_window", DEFAULT_PULSE_WINDOW))
        self._data_dir = data_dir
        self._state_path = os.path.join(data_dir, "resonance_state.json")

        # Create 3 resonance pairs
        self.pairs: dict[str, ResonancePair] = {}
        for pair_name in PAIRS:
            self.pairs[pair_name] = ResonancePair(
                name=pair_name,
                phase_threshold=self._phase_threshold,
                required_cycles=self._required_cycles,
                pulse_window=self._pulse_window,
            )

        # GREAT PULSE tracking (for Phase 3)
        self._great_pulse_count = 0
        self._last_great_pulse_ts = 0.0

        # Load persisted state
        self._load_state()

        logger.info(
            "[ResonanceDetector] Initialized 3 pairs (threshold=%.2f° cycles=%d window=%.0fs)",
            math.degrees(self._phase_threshold), self._required_cycles, self._pulse_window,
        )

    def record_pulse(self, pulse_event: dict) -> Optional[dict]:
        """
        Record a SPHERE_PULSE event and check for resonance.

        Args:
            pulse_event: Dict with at least "component" key
                         (e.g., "inner_body", "outer_mind")

        Returns:
            BIG PULSE event dict if a pair achieved resonance, None otherwise.
        """
        component = pulse_event.get("component", "")

        # Determine which pair this pulse belongs to
        pair_name = self._component_to_pair(component)
        if pair_name is None:
            return None

        pair = self.pairs.get(pair_name)
        if pair is None:
            return None

        big_pulse = pair.record_pulse(component, pulse_event)

        # Check for GREAT PULSE condition after any BIG PULSE
        if big_pulse and self.all_resonant():
            self._great_pulse_count += 1
            self._last_great_pulse_ts = time.time()
            big_pulse["great_pulse_ready"] = True
            big_pulse["great_pulse_count"] = self._great_pulse_count
            logger.info(
                "[ResonanceDetector] GREAT PULSE CONDITION MET #%d — all 3 pairs resonant!",
                self._great_pulse_count,
            )

        return big_pulse

    def record_pulse_with_phases(
        self, pulse_event: dict, inner_phase: float, outer_phase: float,
    ) -> Optional[dict]:
        """
        Record a pulse with explicit phase data from SphereClockEngine.

        This gives more accurate resonance detection than pulse timing alone.
        """
        component = pulse_event.get("component", "")
        pair_name = self._component_to_pair(component)
        if pair_name is None or pair_name not in self.pairs:
            return None

        pair = self.pairs[pair_name]
        pulse_ts = pulse_event.get("ts", time.time())

        big_pulse = pair.record_pulse_with_phase(component,
            inner_phase if component.startswith("inner_") else outer_phase,
            pulse_ts)

        if big_pulse and self.all_resonant():
            self._great_pulse_count += 1
            self._last_great_pulse_ts = time.time()
            big_pulse["great_pulse_ready"] = True
            big_pulse["great_pulse_count"] = self._great_pulse_count
            logger.info(
                "[ResonanceDetector] GREAT PULSE CONDITION MET #%d — all 3 pairs resonant!",
                self._great_pulse_count,
            )

        return big_pulse

    def all_resonant(self) -> bool:
        """Check if all 3 pairs are currently resonant (GREAT PULSE condition)."""
        return all(pair.is_resonant for pair in self.pairs.values())

    def resonant_count(self) -> int:
        """How many of the 3 pairs are currently resonant."""
        return sum(1 for pair in self.pairs.values() if pair.is_resonant)

    @staticmethod
    def _component_to_pair(component: str) -> Optional[str]:
        """Map a component name to its pair name."""
        for pair_name in PAIRS:
            if pair_name in component:
                return pair_name
        return None

    def get_stats(self) -> dict:
        return {
            "pairs": {n: p.get_stats() for n, p in self.pairs.items()},
            "resonant_count": self.resonant_count(),
            "all_resonant": self.all_resonant(),
            "great_pulse_count": self._great_pulse_count,
            "last_great_pulse_ts": self._last_great_pulse_ts,
            "config": {
                "phase_threshold_deg": round(math.degrees(self._phase_threshold), 1),
                "required_cycles": self._required_cycles,
                "pulse_window": self._pulse_window,
            },
        }

    def get_state(self) -> dict:
        """Serialize all mutable state for hot-reload."""
        return {
            "pairs": {n: p.to_dict() for n, p in self.pairs.items()},
            "great_pulse_count": self._great_pulse_count,
            "last_great_pulse_ts": self._last_great_pulse_ts,
        }

    def restore_state(self, state: dict) -> None:
        """Restore all mutable state from hot-reload snapshot."""
        for pair_name, pair_data in state.get("pairs", {}).items():
            if pair_name in self.pairs:
                self.pairs[pair_name].from_dict(pair_data)
        self._great_pulse_count = state.get("great_pulse_count", 0)
        self._last_great_pulse_ts = state.get("last_great_pulse_ts", 0.0)
        total_bigs = sum(p._big_pulse_count for p in self.pairs.values())
        logger.info(
            "[ResonanceDetector] Hot-reload restored: %d BIG PULSEs, %d GREAT PULSEs",
            total_bigs, self._great_pulse_count,
        )

    def save_state(self) -> None:
        """Persist resonance state to disk."""
        import json
        import os
        from pathlib import Path
        try:
            Path(self._state_path).parent.mkdir(parents=True, exist_ok=True)
            state = {
                "pairs": {n: p.to_dict() for n, p in self.pairs.items()},
                "great_pulse_count": self._great_pulse_count,
                "last_great_pulse_ts": self._last_great_pulse_ts,
            }
            with open(self._state_path, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.warning("[ResonanceDetector] Save failed: %s", e)

    def _load_state(self) -> None:
        """Restore resonance state from disk."""
        import json
        import os
        try:
            if not os.path.exists(self._state_path):
                return
            with open(self._state_path) as f:
                state = json.load(f)
            for pair_name, pair_data in state.get("pairs", {}).items():
                if pair_name in self.pairs:
                    self.pairs[pair_name].from_dict(pair_data)
            self._great_pulse_count = state.get("great_pulse_count", 0)
            self._last_great_pulse_ts = state.get("last_great_pulse_ts", 0.0)
            total_bigs = sum(p._big_pulse_count for p in self.pairs.values())
            logger.info("[ResonanceDetector] Restored state: %d BIG PULSEs, %d GREAT PULSEs",
                       total_bigs, self._great_pulse_count)
        except Exception as e:
            logger.warning("[ResonanceDetector] Load failed: %s", e)
