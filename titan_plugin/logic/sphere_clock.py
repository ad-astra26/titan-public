"""
titan_plugin/logic/sphere_clock.py — V4 Time Awareness: Sphere Clock Engine.

Each of the 6 Trinity components (3 inner + 3 outer) has a sphere clock:
  - Expansion: After a pulse, sphere expands from center (radius resets)
  - Contraction: Scalar journeys toward center, velocity proportional to
    Middle Path adherence (closer to center = faster contraction)
  - Pulse: When scalar reaches center → sphere collapses → local pulse generated
  - Reset: Sphere immediately re-expands, scalar starts new journey

Sphere clocks produce natural, emergent pulses based on component health.
Phase 2 will pair inner↔outer clocks for resonance detection (Proof of Harmony).

The contraction velocity is the key mechanism:
  - Well-balanced component (low delta from center) → fast contraction → frequent pulses
  - Imbalanced component (high delta) → slow contraction → infrequent pulses
  - Over time, IQL teaches each component that staying balanced speeds up its clock

IQL scoring hooks are prepared here but NOT active until Phase 3 wires UnifiedSpirit.
"""
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Defaults (overridden by [sphere_clock] config) ────────────────────

DEFAULT_BASE_SPEED = 0.05       # Scalar units per tick
DEFAULT_MIN_RADIUS = 0.3        # Minimum sphere radius
DEFAULT_PULSE_SHRINK_RATE = 0.02  # Radius shrink per balanced pulse
DEFAULT_BALANCE_THRESHOLD = 0.20  # 20% delta = "balanced"
DEFAULT_MIN_VELOCITY_FACTOR = 0.15  # Floor: even max-imbalanced clocks contract at 15% speed
DEFAULT_TICK_INTERVAL = 60.0    # Seconds between ticks

# Component names — canonical ordering matches 30DT SPIRIT tensor
INNER_COMPONENTS = ("inner_body", "inner_mind", "inner_spirit")
OUTER_COMPONENTS = ("outer_body", "outer_mind", "outer_spirit")
ALL_COMPONENTS = INNER_COMPONENTS + OUTER_COMPONENTS


class SphereClock:
    """
    Single sphere clock for one Trinity component.

    The sphere contracts toward its center as the component stays balanced.
    When the scalar reaches the center, a pulse is generated and the sphere
    re-expands for the next cycle.
    """

    def __init__(
        self,
        name: str,
        base_speed: float = DEFAULT_BASE_SPEED,
        min_radius: float = DEFAULT_MIN_RADIUS,
        pulse_shrink_rate: float = DEFAULT_PULSE_SHRINK_RATE,
        balance_threshold: float = DEFAULT_BALANCE_THRESHOLD,
        min_velocity_factor: float = DEFAULT_MIN_VELOCITY_FACTOR,
    ):
        self.name = name
        self.base_speed = base_speed
        self.min_radius = min_radius
        self.pulse_shrink_rate = pulse_shrink_rate
        self.balance_threshold = balance_threshold
        self.min_velocity_factor = min_velocity_factor

        # State
        self.radius = 1.0          # Current sphere radius (1.0 = fully expanded)
        self.scalar_position = 1.0  # 1.0 = edge, 0.0 = center (pulse point)
        self.phase = 0.0           # 0.0 to 2π (for resonance detection)
        self.contraction_velocity = 0.0  # Current tick's velocity
        self.pulse_count = 0       # Total pulses generated
        self.last_pulse_ts = 0.0   # Monotonic timestamp of last pulse (stats only)

        # Stats for IQL scoring hooks (Phase 3)
        self._consecutive_balanced = 0  # Ticks within balance threshold
        self._total_ticks = 0

    def tick(self, coherence: float) -> Optional[dict]:
        """
        Advance the sphere clock by one tick.

        Args:
            coherence: Coherence of the component's tensor (0.0–1.0).
                       High coherence → fast contraction → frequent pulses.
                       Low coherence → slow contraction → infrequent pulses.
                       Typically from middle_path.layer_coherence(tensor) or
                       observables.BodyPartObserver.observe()["coherence"].

                       For backward compatibility, also accepts legacy
                       delta_from_center (layer_loss) — the velocity formula
                       is the same: max(min_velocity_factor, coherence).

        Returns:
            Pulse event dict if a pulse was generated, None otherwise.
        """
        self._total_ticks += 1

        # Clamp to [0, 1]
        coh = max(0.0, min(1.0, coherence))

        # Contraction velocity: proportional to coherence
        # High coherence → full speed
        # Low coherence → minimum speed (floor)
        # "Even sick hearts beat" — min_velocity_factor ensures clocks always pulse
        velocity_factor = max(self.min_velocity_factor, coh)
        self.contraction_velocity = self.base_speed * velocity_factor

        # Move scalar toward center
        self.scalar_position -= self.contraction_velocity
        self.scalar_position = max(0.0, self.scalar_position)

        # Advance phase (wraps at 2π)
        # Phase advancement proportional to contraction progress
        if self.radius > 0:
            phase_step = (self.contraction_velocity / self.radius) * math.pi
            self.phase = (self.phase + phase_step) % (2.0 * math.pi)

        # Track balance streaks (high coherence = balanced)
        is_balanced = coh >= (1.0 - self.balance_threshold)
        if is_balanced:
            self._consecutive_balanced += 1
        else:
            self._consecutive_balanced = 0

        # Check for pulse (scalar reached center)
        if self.scalar_position <= 0.0:
            return self._generate_pulse(is_balanced)

        return None

    def _generate_pulse(self, is_balanced: bool) -> dict:
        """Generate a pulse event and reset the sphere for the next cycle."""
        self.pulse_count += 1
        self.last_pulse_ts = time.monotonic()

        # Sphere shrinks for consistently balanced components
        # (faster future pulses as a natural reward)
        if is_balanced:
            new_radius = max(
                self.min_radius,
                self.radius - self.pulse_shrink_rate,
            )
        else:
            # Imbalanced pulse: sphere stays same or grows slightly
            new_radius = min(1.0, self.radius + self.pulse_shrink_rate * 0.5)

        old_radius = self.radius
        self.radius = new_radius

        # Reset scalar to edge of (new) sphere for next cycle
        self.scalar_position = self.radius

        # Phase continues naturally (no reset) — already wraps at 2π in tick().
        # Continuous phase allows inner↔outer clocks to drift in/out of alignment,
        # creating natural resonance cycling that gates GREAT PULSE (~292s body rate).
        # Previously reset to 0.0 here, which made phase_diff always ≈0 → permanent resonance.

        pulse_event = {
            "component": self.name,
            "pulse_count": self.pulse_count,
            "radius_before": round(old_radius, 4),
            "radius_after": round(new_radius, 4),
            "balanced": is_balanced,
            "consecutive_balanced": self._consecutive_balanced,
            "ts": time.time(),
        }

        logger.info(
            "[SphereClock:%s] PULSE #%d — radius %.3f→%.3f balanced=%s streak=%d",
            self.name, self.pulse_count, old_radius, new_radius,
            is_balanced, self._consecutive_balanced,
        )

        return pulse_event

    def get_iql_score(self, coherence: float) -> int:
        """
        IQL scoring hook (prepared for Phase 3 — UnifiedSpirit wiring).

        Args:
            coherence: Coherence value (0.0–1.0). High = balanced.

        Returns:
            +1 if coherence above threshold AND pulse was recently generated
            -1 if coherence below threshold
             0 otherwise (neutral — making progress but no pulse yet)
        """
        is_balanced = coherence >= (1.0 - self.balance_threshold)

        if is_balanced and self.scalar_position <= 0.01:
            return 1
        elif not is_balanced:
            return -1
        else:
            return 0

    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "radius": round(self.radius, 4),
            "scalar_position": round(self.scalar_position, 4),
            "phase": round(self.phase, 4),
            "contraction_velocity": round(self.contraction_velocity, 4),
            "pulse_count": self.pulse_count,
            "total_ticks": self._total_ticks,
            "consecutive_balanced": self._consecutive_balanced,
        }

    def get_state(self) -> dict:
        """Capture mutable state for hot-reload."""
        return {
            "name": self.name,
            "radius": self.radius,
            "scalar_position": self.scalar_position,
            "phase": self.phase,
            "contraction_velocity": self.contraction_velocity,
            "pulse_count": self.pulse_count,
            "last_pulse_ts": self.last_pulse_ts,
            "_consecutive_balanced": self._consecutive_balanced,
            "_total_ticks": self._total_ticks,
        }

    def restore_state(self, state: dict) -> None:
        """Restore mutable state after hot-reload."""
        self.radius = state.get("radius", 1.0)
        self.scalar_position = state.get("scalar_position", 1.0)
        self.phase = state.get("phase", 0.0)
        self.contraction_velocity = state.get("contraction_velocity", 0.0)
        self.pulse_count = state.get("pulse_count", 0)
        self.last_pulse_ts = state.get("last_pulse_ts", 0.0)
        self._consecutive_balanced = state.get("_consecutive_balanced", 0)
        self._total_ticks = state.get("_total_ticks", 0)

    def to_dict(self) -> dict:
        """Serialize state for persistence."""
        return {
            "name": self.name,
            "radius": self.radius,
            "scalar_position": self.scalar_position,
            "phase": self.phase,
            "pulse_count": self.pulse_count,
            "last_pulse_ts": self.last_pulse_ts,
            "_consecutive_balanced": self._consecutive_balanced,
            "_total_ticks": self._total_ticks,
        }

    def from_dict(self, data: dict) -> None:
        """Restore state from persistence."""
        self.radius = data.get("radius", 1.0)
        self.scalar_position = data.get("scalar_position", 1.0)
        self.phase = data.get("phase", 0.0)
        self.pulse_count = data.get("pulse_count", 0)
        self.last_pulse_ts = data.get("last_pulse_ts", 0.0)
        self._consecutive_balanced = data.get("_consecutive_balanced", 0)
        self._total_ticks = data.get("_total_ticks", 0)


class SphereClockEngine:
    """
    Orchestrates 6 sphere clocks — one per Trinity component.

    Inner Trinity clocks are ticked from Spirit worker (has Body/Mind/Spirit tensors).
    Outer Trinity clocks are ticked from Core (has Outer Trinity collector data).
    Both publish SPHERE_PULSE events on the bus when pulses fire.

    Phase 2 will add ResonanceDetector that consumes pulse events from pairs.
    """

    def __init__(self, config: Optional[dict] = None, data_dir: str = "./data"):
        cfg = config or {}
        self._base_speed = float(cfg.get("base_contraction_speed", DEFAULT_BASE_SPEED))
        self._min_radius = float(cfg.get("min_radius", DEFAULT_MIN_RADIUS))
        self._shrink_rate = float(cfg.get("pulse_shrink_rate", DEFAULT_PULSE_SHRINK_RATE))
        self._balance_threshold = float(cfg.get("balance_threshold", DEFAULT_BALANCE_THRESHOLD))
        self._min_velocity_factor = float(cfg.get("min_velocity_factor", DEFAULT_MIN_VELOCITY_FACTOR))
        self._data_dir = data_dir
        self._state_path = os.path.join(data_dir, "sphere_clock_state.json")

        # Create 6 sphere clocks
        self.clocks: dict[str, SphereClock] = {}
        for name in ALL_COMPONENTS:
            self.clocks[name] = SphereClock(
                name=name,
                base_speed=self._base_speed,
                min_radius=self._min_radius,
                pulse_shrink_rate=self._shrink_rate,
                balance_threshold=self._balance_threshold,
                min_velocity_factor=self._min_velocity_factor,
            )

        # Try to restore persisted state
        self._load_state()

        logger.info(
            "[SphereClockEngine] Initialized 6 clocks (speed=%.3f threshold=%.2f)",
            self._base_speed, self._balance_threshold,
        )

    def tick_inner(
        self,
        body_tensor: list[float],
        mind_tensor: list[float],
        spirit_tensor: list[float],
        coherences: dict[str, float] = None,
    ) -> list[dict]:
        """
        Tick the 3 inner sphere clocks using Inner Trinity tensors.

        Args:
            body_tensor: Inner Body 5DT values
            mind_tensor: Inner Mind 5DT values
            spirit_tensor: Inner Spirit 5DT values (3DT+2)
            coherences: Optional pre-computed coherences from ObservableEngine.
                        If None, computes coherence from tensors directly.

        Returns:
            List of pulse events (may be empty if no pulses fired)
        """
        from .middle_path import layer_coherence

        pulses = []
        for name, tensor in [
            ("inner_body", body_tensor),
            ("inner_mind", mind_tensor),
            ("inner_spirit", spirit_tensor),
        ]:
            coh = (coherences or {}).get(name) if coherences else None
            if coh is None:
                coh = layer_coherence(tensor)
            pulse = self.clocks[name].tick(coh)
            if pulse:
                pulses.append(pulse)

        return pulses

    def tick_outer(
        self,
        outer_body: list[float],
        outer_mind: list[float],
        outer_spirit: list[float],
        coherences: dict[str, float] = None,
    ) -> list[dict]:
        """
        Tick the 3 outer sphere clocks using Outer Trinity tensors.

        Args:
            outer_body: Outer Body 5DT values
            outer_mind: Outer Mind 5DT values
            outer_spirit: Outer Lower Spirit 5DT values
            coherences: Optional pre-computed coherences from ObservableEngine.
                        If None, computes coherence from tensors directly.

        Returns:
            List of pulse events (may be empty)
        """
        from .middle_path import layer_coherence

        pulses = []
        for name, tensor in [
            ("outer_body", outer_body),
            ("outer_mind", outer_mind),
            ("outer_spirit", outer_spirit),
        ]:
            coh = (coherences or {}).get(name) if coherences else None
            if coh is None:
                coh = layer_coherence(tensor)
            pulse = self.clocks[name].tick(coh)
            if pulse:
                pulses.append(pulse)

        return pulses

    def get_all_phases(self) -> dict[str, float]:
        """
        Get current phase of all 6 clocks.

        Returns dict like {"inner_body": 1.23, "outer_mind": 4.56, ...}
        Used by Phase 2 ResonanceDetector for pair phase matching.
        """
        return {name: clock.phase for name, clock in self.clocks.items()}

    def get_paired_phases(self) -> dict[str, tuple[float, float]]:
        """
        Get phase pairs for resonance detection (Phase 2).

        Returns:
            {"body": (inner_phase, outer_phase),
             "mind": (inner_phase, outer_phase),
             "spirit": (inner_phase, outer_phase)}
        """
        return {
            "body": (self.clocks["inner_body"].phase, self.clocks["outer_body"].phase),
            "mind": (self.clocks["inner_mind"].phase, self.clocks["outer_mind"].phase),
            "spirit": (self.clocks["inner_spirit"].phase, self.clocks["outer_spirit"].phase),
        }

    def get_stats(self) -> dict:
        """Full engine stats."""
        return {
            "clocks": {n: c.get_stats() for n, c in self.clocks.items()},
            "total_pulses": sum(c.pulse_count for c in self.clocks.values()),
            "config": {
                "base_speed": self._base_speed,
                "min_radius": self._min_radius,
                "shrink_rate": self._shrink_rate,
                "balance_threshold": self._balance_threshold,
            },
        }

    def get_state(self) -> dict:
        """Capture full engine state for hot-reload."""
        return {
            "clocks": {name: clock.get_state() for name, clock in self.clocks.items()},
        }

    def restore_state(self, state: dict) -> None:
        """Restore full engine state after hot-reload."""
        clocks_state = state.get("clocks", {})
        for name, clock_state in clocks_state.items():
            if name in self.clocks:
                self.clocks[name].restore_state(clock_state)
        total_pulses = sum(c.pulse_count for c in self.clocks.values())
        logger.info(
            "[SphereClockEngine] Hot-reload restored: %d total pulses across %d clocks",
            total_pulses, len(clocks_state),
        )

    def save_state(self) -> None:
        """Persist all clock states to disk."""
        try:
            Path(self._state_path).parent.mkdir(parents=True, exist_ok=True)
            state = {name: clock.to_dict() for name, clock in self.clocks.items()}
            with open(self._state_path, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.warning("[SphereClockEngine] Save failed: %s", e)

    def _load_state(self) -> None:
        """Restore clock states from disk."""
        try:
            if not os.path.exists(self._state_path):
                return
            with open(self._state_path) as f:
                state = json.load(f)
            for name, data in state.items():
                if name in self.clocks:
                    self.clocks[name].from_dict(data)
            total_pulses = sum(c.pulse_count for c in self.clocks.values())
            logger.info("[SphereClockEngine] Restored state: %d total pulses", total_pulses)
        except Exception as e:
            logger.warning("[SphereClockEngine] Load failed: %s", e)
