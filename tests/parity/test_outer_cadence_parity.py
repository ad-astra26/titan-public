"""
Cross-language cadence parity test for outer-trinity daemons (post-A.S8 D2).

Verifies the Schumann frequencies + bus publish throttle intervals match
between Python (`titan_hcl._phase_c_constants`) and Rust
(`titan-rust/crates/titan-core/src/constants.rs`) regenerated from the
single SPEC TOML source-of-truth.

Closes rFP_phase_c_definitive_runtime_closure D2 — outer-rs Schumann cadence
migration. SPEC §13 G13 LOCKED + body-slowest invariant at the bus-publish
layer (45s body > 15s mind > 5s spirit).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from titan_hcl._phase_c_constants import (
    OUTER_BODY_BUS_PUBLISH_INTERVAL_S,
    OUTER_BODY_TICK_BASE_S,
    OUTER_MIND_BUS_PUBLISH_INTERVAL_S,
    OUTER_MIND_TICK_BASE_S,
    OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S,
    OUTER_SPIRIT_TICK_BASE_S,
    SCHUMANN_BODY_HZ,
    SCHUMANN_MIND_HZ,
    SCHUMANN_SPIRIT_HZ,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUST_CONSTANTS = REPO_ROOT / "titan-rust" / "crates" / "titan-core" / "src" / "constants.rs"


def _rust_const(name: str) -> float:
    """Extract `pub const NAME: TYPE = VALUE;` from generated constants.rs."""
    pattern = re.compile(rf"pub const {name}:\s*\w+\s*=\s*([0-9eE.+-]+)\s*;")
    text = RUST_CONSTANTS.read_text()
    m = pattern.search(text)
    if m is None:
        raise AssertionError(f"const {name!r} not found in {RUST_CONSTANTS}")
    return float(m.group(1))


# ── Schumann frequencies (G13 LOCKED) ────────────────────────────────


def test_schumann_body_hz_locked() -> None:
    """SPEC §13 G13: body Schumann fundamental is 7.83 Hz."""
    assert SCHUMANN_BODY_HZ == 7.83
    assert _rust_const("SCHUMANN_BODY_HZ") == 7.83


def test_schumann_mind_hz_locked() -> None:
    """SPEC §13 G13: mind Schumann is body × 3 = 23.49 Hz."""
    assert SCHUMANN_MIND_HZ == 23.49
    assert _rust_const("SCHUMANN_MIND_HZ") == 23.49


def test_schumann_spirit_hz_locked() -> None:
    """SPEC §13 G13: spirit Schumann is body × 9 = 70.47 Hz."""
    assert SCHUMANN_SPIRIT_HZ == 70.47
    assert _rust_const("SCHUMANN_SPIRIT_HZ") == 70.47


def test_schumann_ratios_canonical() -> None:
    """body × 3 = mind, body × 9 = spirit (G13 derivation)."""
    assert pytest.approx(SCHUMANN_BODY_HZ * 3, abs=1e-6) == SCHUMANN_MIND_HZ
    assert pytest.approx(SCHUMANN_BODY_HZ * 9, abs=1e-6) == SCHUMANN_SPIRIT_HZ


# ── Bus publish intervals (D2 new constants) ─────────────────────────


def test_outer_body_bus_publish_interval_python_rust_parity() -> None:
    assert OUTER_BODY_BUS_PUBLISH_INTERVAL_S == 45.0
    assert _rust_const("OUTER_BODY_BUS_PUBLISH_INTERVAL_S") == 45.0


def test_outer_mind_bus_publish_interval_python_rust_parity() -> None:
    assert OUTER_MIND_BUS_PUBLISH_INTERVAL_S == 15.0
    assert _rust_const("OUTER_MIND_BUS_PUBLISH_INTERVAL_S") == 15.0


def test_outer_spirit_bus_publish_interval_python_rust_parity() -> None:
    assert OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S == 5.0
    assert _rust_const("OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S") == 5.0


# ── Body-slowest G13 invariant at bus-publish layer ──────────────────


def test_body_slowest_publish_interval_invariant() -> None:
    """G13 body-slowest at the bus publish cadence layer:
    body publishes LEAST FREQUENTLY (longest interval)."""
    assert OUTER_BODY_BUS_PUBLISH_INTERVAL_S > OUTER_MIND_BUS_PUBLISH_INTERVAL_S
    assert OUTER_MIND_BUS_PUBLISH_INTERVAL_S > OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S


def test_publish_intervals_monotone_decreasing_body_to_spirit() -> None:
    """body 45 > mind 15 > spirit 5 — strictly monotone decreasing."""
    intervals = [
        OUTER_BODY_BUS_PUBLISH_INTERVAL_S,
        OUTER_MIND_BUS_PUBLISH_INTERVAL_S,
        OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S,
    ]
    for i in range(len(intervals) - 1):
        assert intervals[i] > intervals[i + 1], (
            f"intervals[{i}]={intervals[i]} not > intervals[{i+1}]={intervals[i+1]}"
        )


# ── Sensor sidecar refresh cadences (kept post-A.S8 — define stale threshold) ──


def test_outer_sidecar_cadence_python_rust_parity() -> None:
    """OUTER_*_TICK_BASE_S defines sensor sidecar source-refresh cadence
    (and stale threshold = 3× this), NOT daemon tick rate. Daemon ticks
    at Schumann via SchumannGenerator. G13 1:3:9 (spirit fastest, body
    slowest) per D-SPEC-100 — mirrors OUTER_*_BUS_PUBLISH_INTERVAL_S."""
    assert OUTER_BODY_TICK_BASE_S == 45.0
    assert OUTER_MIND_TICK_BASE_S == 15.0
    assert OUTER_SPIRIT_TICK_BASE_S == 5.0
    assert _rust_const("OUTER_BODY_TICK_BASE_S") == 45.0
    assert _rust_const("OUTER_MIND_TICK_BASE_S") == 15.0
    assert _rust_const("OUTER_SPIRIT_TICK_BASE_S") == 5.0


def test_stale_thresholds_3x_sidecar_cadence() -> None:
    """SPEC §18.1 + D-SPEC-100 stale rule: 3× sidecar refresh cadence."""
    assert OUTER_BODY_TICK_BASE_S * 3 == 135.0
    assert OUTER_MIND_TICK_BASE_S * 3 == 45.0
    assert OUTER_SPIRIT_TICK_BASE_S * 3 == 15.0


# ── Schumann period derivations (informational) ──────────────────────


def test_schumann_period_seconds_human_readable() -> None:
    """Per-role period (1 / Hz) — sanity check the Hz values produce
    reasonable tick periods."""
    body_period_ms = 1000.0 / SCHUMANN_BODY_HZ
    mind_period_ms = 1000.0 / SCHUMANN_MIND_HZ
    spirit_period_ms = 1000.0 / SCHUMANN_SPIRIT_HZ
    # Body ~127.7ms, mind ~42.6ms, spirit ~14.2ms
    assert pytest.approx(body_period_ms, abs=0.5) == 127.7
    assert pytest.approx(mind_period_ms, abs=0.5) == 42.6
    assert pytest.approx(spirit_period_ms, abs=0.5) == 14.2
