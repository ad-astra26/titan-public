"""
titan_plugin/shared_blackboard.py — Latest-Value State Store.

High-frequency state propagation for Schumann-rate processing.
Producers write at their natural frequency. Consumers read when
THEY are ready. Each key stores only the LATEST value.

This replaces queue-based state message routing for periodic
tensor updates (BODY_STATE, MIND_STATE, SPIRIT_STATE). Event
messages (IMPULSE, REFLEX_SIGNAL, etc.) remain on queues.

Thread-safe via RLock for concurrent read/write from main process.
Worker subprocesses access via Guardian relay (not direct access).
"""
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class SharedBlackboard:
    """
    Latest-value state store for high-frequency state propagation.

    Producers write at their natural frequency (Schumann rates).
    Consumers read when THEY are ready (no queue backpressure).
    Each key stores only the LATEST value — no accumulation.
    """

    def __init__(self):
        self._data: dict[str, dict] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = threading.RLock()
        self._write_count: int = 0
        self._read_count: int = 0

    def write(self, key: str, value: dict) -> None:
        """Write latest state. O(1), never blocks producers."""
        with self._lock:
            self._data[key] = value
            self._timestamps[key] = time.time()
            self._write_count += 1

    def read(self, key: str) -> tuple[Optional[dict], float]:
        """Read latest state + timestamp. O(1), never blocks."""
        with self._lock:
            self._read_count += 1
            return self._data.get(key), self._timestamps.get(key, 0.0)

    def read_if_newer(self, key: str, since: float) -> Optional[dict]:
        """Read only if updated since given timestamp. Returns None if stale."""
        with self._lock:
            ts = self._timestamps.get(key, 0.0)
            if ts > since:
                self._read_count += 1
                return self._data.get(key)
            return None

    def read_all(self) -> dict[str, dict]:
        """Read all current state (snapshot)."""
        with self._lock:
            self._read_count += 1
            return dict(self._data)

    def keys(self) -> list[str]:
        """List all stored keys."""
        with self._lock:
            return list(self._data.keys())

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "keys": list(self._data.keys()),
                "key_count": len(self._data),
                "writes": self._write_count,
                "reads": self._read_count,
            }
