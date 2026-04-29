"""
titan_plugin/utils/system_sensor.py — CPU + thermal + circadian rich signals.

Produces three independent rich sources for outer_body V6 5DT composites:
- CPU load (normalized per-core, 1.0 = saturated)
- CPU thermal (best-effort via /sys/class/thermal; 0.5 when unavailable)
- Circadian phase (time-of-day, 0.2 night → 0.9 peak)
- CPU spike rate (rolling window of load surges)

Used by outer_trinity._collect_outer_body for dims [2] somatosensation
(via spike rate) and [4] thermal (via CPU thermal + circadian).

All getters are sample-on-demand with a 30s-TTL cache; no background
thread. Thread-safe for concurrent readers.
"""
import logging
import math
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Cache TTL — outer_trinity_loop polls every 60s, so 30s gives us fresh
# readings at each tick without over-sampling /sys files.
_CACHE_TTL_S = 30.0

# Thermal normalization range
# Typical VPS CPU temps: 35°C cool, 50°C warm, 70°C hot, 80°C danger
_THERMAL_COOL_C = 30.0
_THERMAL_HOT_C = 80.0

# Spike-rate: rolling window of 20 samples × 30s TTL = ~10 min of history
_SPIKE_WINDOW_MAX = 20
_SPIKE_THRESHOLD_NORMALIZED = 0.75  # load > 0.75 per core counts as a spike


class _Cache:
    """Thread-safe single-value TTL cache."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Optional[float] = None
        self._ts: float = 0.0

    def get(self) -> Optional[float]:
        with self._lock:
            if self._value is None or time.time() - self._ts > _CACHE_TTL_S:
                return None
            return self._value

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value
            self._ts = time.time()


_cpu_load_cache = _Cache()
_cpu_thermal_cache = _Cache()

# Spike-rate rolling buffer: (ts, normalized_load)
_spike_buffer: deque = deque(maxlen=_SPIKE_WINDOW_MAX)
_spike_lock = threading.Lock()

# Log thermal-unavailable only once to avoid noise
_thermal_unavailable_logged = False


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v) if v == v else 0.5))


def get_cpu_load() -> float:
    """1-minute load average normalized by CPU count.

    Returns [0, 1] where 1.0 = all cores fully loaded.
    Cached for 30s.
    """
    cached = _cpu_load_cache.get()
    if cached is not None:
        return cached

    try:
        load1, _, _ = os.getloadavg()
        cpu_count = os.cpu_count() or 1
        normalized = _clamp(load1 / cpu_count)
    except (OSError, AttributeError):
        normalized = 0.5  # Neutral fallback if getloadavg unavailable

    _cpu_load_cache.set(normalized)

    with _spike_lock:
        _spike_buffer.append((time.time(), normalized))

    return normalized


def get_cpu_thermal() -> float:
    """CPU thermal reading normalized to [0, 1].

    Reads /sys/class/thermal/thermal_zone*/temp (millicelsius on Linux).
    Takes the max across zones to capture the hottest core/package.
    Returns 0.5 if /sys/class/thermal is unavailable (e.g. container without
    thermal passthrough — common on low-tier VPS).
    """
    global _thermal_unavailable_logged

    cached = _cpu_thermal_cache.get()
    if cached is not None:
        return cached

    thermal_dir = "/sys/class/thermal"
    try:
        if not os.path.isdir(thermal_dir):
            raise FileNotFoundError(thermal_dir)

        max_temp_c = None
        for entry in os.listdir(thermal_dir):
            if not entry.startswith("thermal_zone"):
                continue
            temp_path = os.path.join(thermal_dir, entry, "temp")
            if not os.path.isfile(temp_path):
                continue
            try:
                with open(temp_path) as f:
                    milli = int(f.read().strip())
                temp_c = milli / 1000.0
                if max_temp_c is None or temp_c > max_temp_c:
                    max_temp_c = temp_c
            except (OSError, ValueError):
                continue

        if max_temp_c is None:
            raise FileNotFoundError("no readable thermal zones")

        normalized = _clamp(
            (max_temp_c - _THERMAL_COOL_C) / (_THERMAL_HOT_C - _THERMAL_COOL_C)
        )

    except (FileNotFoundError, OSError, PermissionError):
        if not _thermal_unavailable_logged:
            logger.warning(
                "[SystemSensor] CPU thermal unavailable (no /sys/class/thermal); "
                "returning 0.5 neutral. This is expected on VPS without thermal passthrough."
            )
            _thermal_unavailable_logged = True
        normalized = 0.5

    _cpu_thermal_cache.set(normalized)
    return normalized


def get_circadian_phase(now: Optional[datetime] = None) -> float:
    """Time-of-day circadian phase.

    Returns [0, 1] where 1.0 = peak alertness, 0.2 = night trough.
    Peak 8am-6pm, trough 2am-5am, transition evening/early-morning.

    Accepts `now` for deterministic testing.
    """
    if now is None:
        now = datetime.now()

    hour = now.hour + now.minute / 60.0

    if 8.0 <= hour <= 18.0:
        # Daytime: sinusoidal ramp, peak ~1pm
        return _clamp(0.7 + 0.2 * math.sin(math.pi * (hour - 8.0) / 10.0))
    elif 2.0 <= hour < 5.0:
        # Deep trough
        return 0.2
    elif 5.0 <= hour < 8.0:
        # Dawn ramp-up
        return _clamp(0.2 + 0.5 * ((hour - 5.0) / 3.0))
    elif 18.0 < hour <= 23.0:
        # Evening wind-down
        return _clamp(0.7 - 0.4 * ((hour - 18.0) / 5.0))
    else:
        # 23-02 late night
        return _clamp(0.3 - 0.1 * ((hour - 23.0) % 24.0 / 3.0))


def get_cpu_spike_rate() -> float:
    """Rate of recent CPU spikes within the rolling sample window.

    Returns fraction of buffered samples whose normalized load exceeded
    _SPIKE_THRESHOLD_NORMALIZED (0.75 = 75% per-core saturation).

    Requires get_cpu_load() to have been called recently (each call
    appends a sample). Returns 0.0 on empty buffer.
    """
    with _spike_lock:
        samples = list(_spike_buffer)

    if not samples:
        return 0.0

    now = time.time()
    # Consider only samples from last 5 min (stale samples don't count)
    recent = [load for ts, load in samples if now - ts < 300.0]
    if not recent:
        return 0.0

    spikes = sum(1 for load in recent if load > _SPIKE_THRESHOLD_NORMALIZED)
    return _clamp(spikes / len(recent))


def get_all_stats() -> dict:
    """Single-call snapshot of all system sensor signals.

    Convenient for outer_trinity.py's _gather_outer_trinity_sources.
    """
    load = get_cpu_load()
    thermal = get_cpu_thermal()
    circadian = get_circadian_phase()
    spike_rate = get_cpu_spike_rate()
    return {
        "cpu_load": load,
        "cpu_thermal": thermal,
        "circadian_phase": circadian,
        "cpu_spike_rate": spike_rate,
    }


def _reset_for_testing() -> None:
    """Clear caches + buffer. Test-only helper."""
    global _thermal_unavailable_logged
    _cpu_load_cache._value = None
    _cpu_thermal_cache._value = None
    with _spike_lock:
        _spike_buffer.clear()
    _thermal_unavailable_logged = False
