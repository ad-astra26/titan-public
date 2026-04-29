"""Unit tests for titan_plugin.utils.system_sensor.

Runs in its own pytest process per project convention (TorchRL mmap).
"""
import os
from datetime import datetime
from unittest.mock import patch

import pytest

from titan_plugin.utils import system_sensor as ss


@pytest.fixture(autouse=True)
def _reset_state():
    ss._reset_for_testing()
    yield
    ss._reset_for_testing()


def test_cpu_load_returns_normalized_value():
    with patch("os.getloadavg", return_value=(2.0, 1.5, 1.0)), \
         patch("os.cpu_count", return_value=4):
        load = ss.get_cpu_load()
    assert 0.0 <= load <= 1.0
    # 2.0 load on 4 cores = 0.5
    assert abs(load - 0.5) < 0.01


def test_cpu_load_saturates_at_one():
    with patch("os.getloadavg", return_value=(10.0, 5.0, 2.0)), \
         patch("os.cpu_count", return_value=4):
        load = ss.get_cpu_load()
    assert load == 1.0


def test_cpu_load_cached_within_ttl():
    """Repeated calls within TTL return cached value without re-sampling."""
    call_count = {"n": 0}

    def fake_loadavg():
        call_count["n"] += 1
        return (1.0, 1.0, 1.0)

    with patch("os.getloadavg", side_effect=fake_loadavg), \
         patch("os.cpu_count", return_value=2):
        load1 = ss.get_cpu_load()
        load2 = ss.get_cpu_load()
        load3 = ss.get_cpu_load()

    assert load1 == load2 == load3 == 0.5
    assert call_count["n"] == 1, "loadavg should have been called only once"


def test_cpu_load_fallback_when_getloadavg_raises():
    with patch("os.getloadavg", side_effect=OSError("not supported")):
        load = ss.get_cpu_load()
    assert load == 0.5  # Neutral fallback


def test_cpu_thermal_reads_sys_class():
    """Simulate a readable thermal zone; verify normalization."""
    thermal_content = "55000\n"  # 55°C in millicelsius

    def fake_isdir(path):
        return path == "/sys/class/thermal"

    def fake_isfile(path):
        return path.endswith("/temp")

    def fake_listdir(path):
        return ["thermal_zone0", "thermal_zone1"]

    from unittest.mock import mock_open
    with patch("os.path.isdir", side_effect=fake_isdir), \
         patch("os.path.isfile", side_effect=fake_isfile), \
         patch("os.listdir", side_effect=fake_listdir), \
         patch("builtins.open", mock_open(read_data=thermal_content)):
        thermal = ss.get_cpu_thermal()

    # 55°C in [30, 80] range → (55-30)/(80-30) = 0.5
    assert 0.45 <= thermal <= 0.55


def test_cpu_thermal_fallback_when_unavailable():
    """No /sys/class/thermal → return 0.5 neutral."""
    with patch("os.path.isdir", return_value=False):
        thermal = ss.get_cpu_thermal()
    assert thermal == 0.5


def test_circadian_phase_peak_midday():
    noon = datetime(2026, 4, 23, 13, 0)  # 1pm
    phase = ss.get_circadian_phase(noon)
    assert phase >= 0.85, f"Expected peak near 1pm, got {phase}"


def test_circadian_phase_trough_3am():
    predawn = datetime(2026, 4, 23, 3, 0)  # 3am
    phase = ss.get_circadian_phase(predawn)
    assert phase == 0.2, f"Expected trough 0.2 at 3am, got {phase}"


def test_circadian_phase_evening_transition():
    evening = datetime(2026, 4, 23, 20, 0)  # 8pm
    phase = ss.get_circadian_phase(evening)
    # Wind-down: 0.7 - 0.4*(2/5) = 0.54
    assert 0.4 <= phase <= 0.6, f"Expected evening 0.4-0.6, got {phase}"


def test_circadian_phase_dawn_ramp():
    dawn = datetime(2026, 4, 23, 6, 30)
    phase = ss.get_circadian_phase(dawn)
    # Dawn ramp 5-8am: at 6:30 → 0.2 + 0.5*(1.5/3) = 0.45
    assert 0.35 <= phase <= 0.55, f"Expected dawn ramp 0.35-0.55, got {phase}"


def test_circadian_phase_stays_in_unit_range_across_24h():
    for hour in range(24):
        for minute in (0, 30):
            t = datetime(2026, 4, 23, hour, minute)
            phase = ss.get_circadian_phase(t)
            assert 0.0 <= phase <= 1.0, f"t={t}: phase={phase} out of range"


def test_spike_rate_empty_buffer():
    assert ss.get_cpu_spike_rate() == 0.0


def test_spike_rate_counts_threshold_breaches():
    """Feed load samples via get_cpu_load(), verify spike detection."""
    # Low load samples — no spikes
    with patch("os.getloadavg", return_value=(0.3, 0.3, 0.3)), \
         patch("os.cpu_count", return_value=1):
        for _ in range(5):
            ss._cpu_load_cache._value = None  # Force re-sample each call
            ss.get_cpu_load()
    assert ss.get_cpu_spike_rate() == 0.0  # 0.3 < 0.75 threshold

    # High load samples — all spikes
    ss._reset_for_testing()
    with patch("os.getloadavg", return_value=(0.9, 0.9, 0.9)), \
         patch("os.cpu_count", return_value=1):
        for _ in range(5):
            ss._cpu_load_cache._value = None
            ss.get_cpu_load()
    rate = ss.get_cpu_spike_rate()
    assert rate == 1.0, f"Expected 100% spike rate, got {rate}"


def test_get_all_stats_returns_all_keys():
    with patch("os.getloadavg", return_value=(0.5, 0.5, 0.5)), \
         patch("os.cpu_count", return_value=1), \
         patch("os.path.isdir", return_value=False):
        stats = ss.get_all_stats()
    assert set(stats.keys()) == {
        "cpu_load", "cpu_thermal", "circadian_phase", "cpu_spike_rate"
    }
    for v in stats.values():
        assert 0.0 <= v <= 1.0
