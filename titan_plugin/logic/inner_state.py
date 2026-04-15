"""
titan_plugin/logic/inner_state.py — InnerState: subconscious state registry (T2).

In-process state inside spirit_worker. No bus needed — same process.
Always running, even when OuterState is paused (dreaming).

Stores:
  - Observables (6 parts × 5 values, from T1 ObservableEngine)
  - Topology (volume, curvature, clusters — populated by T5)
  - Fatigue/readiness scores (populated by T6)
  - Dynamic thresholds (learned during operation)
  - Experience buffer (outer snapshots queued for dreaming processing)
  - Cycle tracking (conscious→dreaming transitions)
"""
import time
from typing import Any


class InnerState:
    """Subconscious state — always running, in-process."""

    def __init__(self):
        # T1: 6 parts × 5 observables (coherence, magnitude, velocity, direction, polarity)
        self.observables: dict[str, dict] = {}

        # T5: Space topology (volume, curvature, clusters)
        self.topology: dict = {}

        # T6: Composite fatigue/readiness scores
        self.fatigue: float = 0.0
        self.readiness: float = 1.0  # start fully rested

        # Dynamic thresholds (learned from experience, populated by T6/T7)
        self.thresholds: dict[str, float] = {}

        # Experience buffer: outer snapshots queued for dreaming processing
        self._experience_buffer: list[dict] = []
        self._max_buffer_size: int = 100  # cap to prevent unbounded growth

        # Cycle tracking
        self.cycle_count: int = 0
        self.is_dreaming: bool = False
        self.last_cycle_ts: float = 0.0

        # General metadata
        self._created_ts: float = time.time()
        self._update_count: int = 0

    def update_observables(self, observables: dict[str, dict]) -> None:
        """Store latest observables from ObservableEngine."""
        self.observables = observables
        self._update_count += 1

    def update_topology(self, topology: dict) -> None:
        """Store latest topology computation (T5)."""
        self.topology = topology

    def buffer_experience(self, snapshot: dict) -> None:
        """Buffer an outer state snapshot for dreaming processing (T6)."""
        if len(self._experience_buffer) >= self._max_buffer_size:
            self._experience_buffer.pop(0)  # drop oldest
        self._experience_buffer.append(snapshot)

    def drain_experience_buffer(self) -> list[dict]:
        """Drain and return all buffered experiences (called during dreaming)."""
        buffer = self._experience_buffer
        self._experience_buffer = []
        return buffer

    def get(self, key: str, default: Any = None) -> Any:
        """Generic getter for any attribute."""
        return getattr(self, key, default)

    def snapshot(self) -> dict:
        """Full state snapshot."""
        return {
            "observables": dict(self.observables),
            "topology": dict(self.topology),
            "fatigue": self.fatigue,
            "readiness": self.readiness,
            "thresholds": dict(self.thresholds),
            "experience_buffer_size": len(self._experience_buffer),
            "cycle_count": self.cycle_count,
            "is_dreaming": self.is_dreaming,
            "update_count": self._update_count,
        }
