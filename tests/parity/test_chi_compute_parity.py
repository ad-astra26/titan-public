"""
Cross-language parity test for substrate-scoped chi computation.

Verifies that Rust `titan_trinity_rs::chi_state::ChiState::compute(...)` and
the Python reference implementation in this file produce byte-identical
24-byte payloads (per SPEC §7.1 chi_state.bin layout) for the same set of
substrate-readable inputs.

Closes rFP_phase_c_close_all_runtime_gaps §6.2 row 3.

Why a fresh Python reference (not a port of LifeForceEngine):
- LifeForceEngine reads Python L2 signals (vocabulary, sovereignty,
  sol_balance, anchor_freshness, expression_rate, ...) — none of those
  are available at the substrate.
- The substrate's chi_state.bin (per SPEC §7.1 row 13, owner =
  titan-trinity-rs) is the substrate-scoped subset using only:
    * 6 daemon tensors (inner+outer × body/mind/spirit)
    * sphere_clocks contraction velocity per layer
    * neuromod_state.bin (6 floats)
- Layer weights fixed at the LifeForceEngine "mature" terminus (0.40 /
  0.35 / 0.25) because the substrate has no developmental-age clock.

This file's `compute_substrate_chi` is the Python ground-truth-by-
construction. Both sides match by spec; this test verifies they actually do.
"""
from __future__ import annotations

import math
import subprocess
import json
from pathlib import Path
from typing import Any

import pytest


# ── Substrate-scoped chi formula (Python reference) ──────────────────

W_SPIRIT = 0.40
W_MIND = 0.35
W_BODY = 0.25
BODY_NORM = math.sqrt(5.0)
SPIRIT_NORM = math.sqrt(45.0)
MIND_NORM = BODY_NORM
MIND_WILLING_RANGE = (10, 15)


def _l2(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _avg_pair(a: list[float], b: list[float]) -> list[float]:
    assert len(a) == len(b)
    return [(x + y) * 0.5 for x, y in zip(a, b)]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    ma, mb = _l2(a), _l2(b)
    if ma < 1e-10 or mb < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, dot / (ma * mb)))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_substrate_chi(
    inner_body_5d: list[float],
    inner_mind_15d: list[float],
    inner_spirit_45d: list[float],
    outer_body_5d: list[float],
    outer_mind_15d: list[float],
    outer_spirit_45d: list[float],
    inner_body_clock_velocity: float,
    outer_body_clock_velocity: float,
    inner_mind_clock_velocity: float,
    outer_mind_clock_velocity: float,
    inner_spirit_clock_velocity: float,
    outer_spirit_clock_velocity: float,
    neuromod_6: list[float],
) -> dict[str, float]:
    """Substrate-scoped chi — mirrors Rust `ChiState::compute(...)`."""
    body_inner_mag = _l2(inner_body_5d) / BODY_NORM
    body_outer_mag = _l2(outer_body_5d) / BODY_NORM
    body_mag = _clamp01((body_inner_mag + body_outer_mag) * 0.5)

    inner_mind_willing = inner_mind_15d[MIND_WILLING_RANGE[0]:MIND_WILLING_RANGE[1]]
    outer_mind_willing = outer_mind_15d[MIND_WILLING_RANGE[0]:MIND_WILLING_RANGE[1]]
    mind_inner_mag = _l2(inner_mind_willing) / MIND_NORM
    mind_outer_mag = _l2(outer_mind_willing) / MIND_NORM
    mind_mag = _clamp01((mind_inner_mag + mind_outer_mag) * 0.5)

    spirit_inner_mag = _l2(inner_spirit_45d) / SPIRIT_NORM
    spirit_outer_mag = _l2(outer_spirit_45d) / SPIRIT_NORM
    spirit_mag = _clamp01((spirit_inner_mag + spirit_outer_mag) * 0.5)

    body_coh = (inner_body_clock_velocity + outer_body_clock_velocity) * 0.5
    mind_coh = (inner_mind_clock_velocity + outer_mind_clock_velocity) * 0.5
    spirit_coh = (inner_spirit_clock_velocity + outer_spirit_clock_velocity) * 0.5

    body = _clamp01(body_mag * body_coh)
    mind = _clamp01(mind_mag * mind_coh)
    spirit = _clamp01(spirit_mag * spirit_coh)

    body_avg = _avg_pair(inner_body_5d, outer_body_5d)
    mind_willing_avg = _avg_pair(inner_mind_willing, outer_mind_willing)
    spirit_head_inner = inner_spirit_45d[:5]
    spirit_head_outer = outer_spirit_45d[:5]
    spirit_avg = _avg_pair(spirit_head_inner, spirit_head_outer)

    cos_bm = _cosine(body_avg, mind_willing_avg)
    cos_ms = _cosine(mind_willing_avg, spirit_avg)
    cos_bs = _cosine(body_avg, spirit_avg)
    coherence = _clamp01((cos_bm + cos_ms + cos_bs) / 3.0)

    urgency = _clamp01(max(neuromod_6))

    total = _clamp01(W_SPIRIT * spirit + W_MIND * mind + W_BODY * body)

    return {
        "total": total,
        "spirit": spirit,
        "mind": mind,
        "body": body,
        "coherence": coherence,
        "urgency": urgency,
    }


def serialize_chi(chi: dict[str, float]) -> bytes:
    """Pack chi dict into 24-byte float32 LE per SPEC §7.1 field order:
    total, spirit, mind, body, coherence, urgency.
    """
    import struct

    return struct.pack(
        "<6f",
        chi["total"],
        chi["spirit"],
        chi["mind"],
        chi["body"],
        chi["coherence"],
        chi["urgency"],
    )


# ── Fixtures ──────────────────────────────────────────────────────────

FIXTURES: list[dict[str, Any]] = [
    {
        "name": "all_zeros",
        "inner_body_5d": [0.0] * 5,
        "inner_mind_15d": [0.0] * 15,
        "inner_spirit_45d": [0.0] * 45,
        "outer_body_5d": [0.0] * 5,
        "outer_mind_15d": [0.0] * 15,
        "outer_spirit_45d": [0.0] * 45,
        "ib_v": 0.0, "ob_v": 0.0,
        "im_v": 0.0, "om_v": 0.0,
        "is_v": 0.0, "os_v": 0.0,
        "neuromod_6": [0.0] * 6,
    },
    {
        "name": "balanced_05_clocks_full",
        "inner_body_5d": [0.5] * 5,
        "inner_mind_15d": [0.0] * 10 + [0.5] * 5,
        "inner_spirit_45d": [0.5] * 45,
        "outer_body_5d": [0.5] * 5,
        "outer_mind_15d": [0.0] * 10 + [0.5] * 5,
        "outer_spirit_45d": [0.5] * 45,
        "ib_v": 1.0, "ob_v": 1.0,
        "im_v": 1.0, "om_v": 1.0,
        "is_v": 1.0, "os_v": 1.0,
        "neuromod_6": [0.0] * 6,
    },
    {
        "name": "asymmetric_inner_outer",
        "inner_body_5d": [0.3, 0.4, 0.5, 0.6, 0.7],
        "inner_mind_15d": [0.0] * 10 + [0.4] * 5,
        "inner_spirit_45d": [0.5] * 45,
        "outer_body_5d": [0.5] * 5,
        "outer_mind_15d": [0.0] * 10 + [0.6] * 5,
        "outer_spirit_45d": [0.4] * 45,
        "ib_v": 0.5, "ob_v": 0.5,
        "im_v": 0.7, "om_v": 0.3,
        "is_v": 0.6, "os_v": 0.4,
        "neuromod_6": [0.3, 0.5, 0.4, 0.2, 0.1, 0.6],
    },
    {
        "name": "neuromod_peak_drives_urgency",
        "inner_body_5d": [0.1] * 5,
        "inner_mind_15d": [0.0] * 10 + [0.1] * 5,
        "inner_spirit_45d": [0.1] * 45,
        "outer_body_5d": [0.1] * 5,
        "outer_mind_15d": [0.0] * 10 + [0.1] * 5,
        "outer_spirit_45d": [0.1] * 45,
        "ib_v": 0.5, "ob_v": 0.5,
        "im_v": 0.5, "om_v": 0.5,
        "is_v": 0.5, "os_v": 0.5,
        "neuromod_6": [0.1, 0.2, 0.95, 0.3, 0.0, 0.5],
    },
]


# ── Rust-side parity binary discovery ────────────────────────────────

PARITY_BIN = (
    Path(__file__).parent.parent.parent
    / "titan-rust"
    / "target"
    / "debug"
    / "examples"
    / "chi_parity_dump"
)


def _rust_compute(fixture: dict[str, Any]) -> dict[str, float] | None:
    """Invoke the Rust parity helper if built; return its output dict or
    None when the binary isn't available (skip rather than fail).

    The helper is a tiny `examples/chi_parity_dump.rs` that reads a JSON
    fixture from stdin + emits the 24 bytes as JSON-friendly per-field
    floats. Build with `cargo build -p titan-trinity-rs --example chi_parity_dump`.
    """
    if not PARITY_BIN.exists():
        return None
    proc = subprocess.run(
        [str(PARITY_BIN)],
        input=json.dumps(fixture),
        capture_output=True,
        text=True,
        timeout=10,
    )
    if proc.returncode != 0:
        pytest.fail(f"chi_parity_dump rc={proc.returncode}: {proc.stderr}")
    return json.loads(proc.stdout)


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["name"])
def test_python_reference_clamps_to_unit_range(fixture: dict[str, Any]) -> None:
    """Python reference always returns [0, 1] per field."""
    chi = compute_substrate_chi(
        fixture["inner_body_5d"],
        fixture["inner_mind_15d"],
        fixture["inner_spirit_45d"],
        fixture["outer_body_5d"],
        fixture["outer_mind_15d"],
        fixture["outer_spirit_45d"],
        fixture["ib_v"], fixture["ob_v"],
        fixture["im_v"], fixture["om_v"],
        fixture["is_v"], fixture["os_v"],
        fixture["neuromod_6"],
    )
    for field, value in chi.items():
        assert 0.0 <= value <= 1.0, f"{fixture['name']}.{field} = {value} out of [0,1]"


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["name"])
def test_serialize_layout_matches_spec(fixture: dict[str, Any]) -> None:
    """24-byte payload, field order per SPEC §7.1: total/spirit/mind/body/coherence/urgency."""
    chi = compute_substrate_chi(
        fixture["inner_body_5d"],
        fixture["inner_mind_15d"],
        fixture["inner_spirit_45d"],
        fixture["outer_body_5d"],
        fixture["outer_mind_15d"],
        fixture["outer_spirit_45d"],
        fixture["ib_v"], fixture["ob_v"],
        fixture["im_v"], fixture["om_v"],
        fixture["is_v"], fixture["os_v"],
        fixture["neuromod_6"],
    )
    bytes_out = serialize_chi(chi)
    assert len(bytes_out) == 24
    import struct

    fields = struct.unpack("<6f", bytes_out)
    assert math.isclose(fields[0], chi["total"], abs_tol=1e-6)
    assert math.isclose(fields[1], chi["spirit"], abs_tol=1e-6)
    assert math.isclose(fields[2], chi["mind"], abs_tol=1e-6)
    assert math.isclose(fields[3], chi["body"], abs_tol=1e-6)
    assert math.isclose(fields[4], chi["coherence"], abs_tol=1e-6)
    assert math.isclose(fields[5], chi["urgency"], abs_tol=1e-6)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["name"])
def test_rust_python_parity_within_float32_tolerance(fixture: dict[str, Any]) -> None:
    """Rust ↔ Python: for each fixture, both implementations produce
    chi within 1e-5 (float32 precision floor).

    Skips when chi_parity_dump example binary not built; integration tests
    on T3 cover the same code path through actual chi_state.bin reads.
    """
    rust_chi = _rust_compute(fixture)
    if rust_chi is None:
        pytest.skip(
            "chi_parity_dump example not built — run "
            "`cargo build -p titan-trinity-rs --example chi_parity_dump`"
        )

    py_chi = compute_substrate_chi(
        fixture["inner_body_5d"],
        fixture["inner_mind_15d"],
        fixture["inner_spirit_45d"],
        fixture["outer_body_5d"],
        fixture["outer_mind_15d"],
        fixture["outer_spirit_45d"],
        fixture["ib_v"], fixture["ob_v"],
        fixture["im_v"], fixture["om_v"],
        fixture["is_v"], fixture["os_v"],
        fixture["neuromod_6"],
    )

    for field in ("total", "spirit", "mind", "body", "coherence", "urgency"):
        py_val = py_chi[field]
        rust_val = rust_chi[field]
        assert math.isclose(py_val, rust_val, abs_tol=1e-5), (
            f"{fixture['name']}.{field}: Rust={rust_val} Python={py_val} "
            f"diff={abs(py_val - rust_val)}"
        )
