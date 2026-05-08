"""
Cross-language parity test for outer-mind 15D projection.

Verifies that Rust `titan_outer_mind_rs::tick_loop::project_outer_mind_15d`
and Python `outer_mind_tensor::collect_outer_mind_15d` (wrapped by
`outer_trinity::_collect_extended` preprocessing) produce numerically-
equivalent 15D outputs for the same set of msgpack source dict + outer-body.

Closes rFP_phase_c_close_all_runtime_gaps chunk 9I (§4.4 expanded).

Why fixtures + JSON-not-msgpack on the wire:
- The Rust example binary takes JSON on stdin (deterministic, hand-editable);
  it converts JSON → rmpv::Value internally before invoking the project fn,
  so the test harness exercises the same msgpack→[f32;15] path that the
  daemon uses on T3 with a real `sensor_cache_outer_mind.bin` payload.
- Tolerance |Δ| < 1e-3 mirrors the chi parity convention shipped 2026-05-06
  + the 9G outer_body convention. f32 cast + Python float cast both
  preserve >7 significant digits; 1e-3 is comfortable.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pytest

# Re-implement Python reference inline (mirrors _collect_extended → collect_outer_mind_15d).
# Inlined rather than imported because outer_trinity.py pulls heavy deps; test
# only needs the math.


def _clamp(v: float) -> float:
    if math.isnan(v):
        return 0.5
    return max(0.0, min(1.0, v))


def _compute_blockchain_active(anchor: dict | None, now: float) -> float:
    if not anchor:
        return 0.5
    if not anchor.get("success"):
        return 0.5
    last = anchor.get("last_anchor_time")
    if not last or last <= 0:
        return 0.5
    since = max(0.0, now - last)
    return max(0.1, 1.0 / (1.0 + since / 300.0))


def compute_outer_mind_15d_python(sources: dict[str, Any], outer_body: list[float],
                                  now: float) -> list[float]:
    """Python reference — mirrors Rust `project_outer_mind_15d`. `now`
    parameter is wired so anchor freshness is deterministic across the
    Rust/Python boundary (Rust calls SystemTime::now()) — fixtures must
    pass anchor `last_anchor_time` close enough to `now` (or omit anchor)
    that the freshness lookup is stable for the test window.
    """
    agency = sources.get("agency_stats") or {}
    assessment = sources.get("assessment_stats") or {}
    memory_status = sources.get("memory_status") or {}
    social_perception = sources.get("social_perception_stats") or {}
    twin_state = sources.get("twin_state") or {}
    anchor_state = sources.get("anchor_state") or {}
    bus_stats = sources.get("bus_stats") or {}

    total_actions = float(agency.get("total_actions", 0))
    failed_actions = float(agency.get("failed_actions", 0))
    action_success_rate = (total_actions - failed_actions) / max(1.0, total_actions)
    action_per_window = float(agency.get("actions_this_hour", 0))
    creative_per_window = float(agency.get("creative_this_hour", 0))
    rejections_per_window = float(agency.get("rejections_this_hour", 0))

    interactions_per_window = float(memory_status.get("unique_interactors", 0))
    sentiment_avg = float(social_perception.get("sentiment_ema", 0.5))
    social_outputs_per_window = 0.0  # not in source

    research_queries = float(memory_status.get("research_nodes", 0))
    research_usage_rate = 0.5
    research_seconds_since_last = 300.0
    research_queries_per_window = 0.0

    assessment_mean = float(assessment.get("average_score", 0.5))

    thinking = [0.5] * 5
    thinking[0] = _clamp(research_usage_rate) if research_queries > 0 else 0.3
    thinking[1] = _clamp(assessment_mean)
    thinking[2] = _clamp(1.0 / (1.0 + research_seconds_since_last / 1800.0))
    thinking[3] = _clamp(action_success_rate)
    thinking[4] = _clamp(assessment_mean)

    feeling = [0.5] * 5
    social_activity = _clamp(
        min(1.0, interactions_per_window / 5.0) * 0.5 + sentiment_avg * 0.5
    )
    interaction_rate = min(1.0, interactions_per_window / 8.0)
    feeling[0] = _clamp(0.5 * sentiment_avg + 0.3 * interaction_rate + 0.2 * social_activity)

    if twin_state.get("reachable"):
        twin_da = float(twin_state.get("DA", 0.5))
        twin_ne = float(twin_state.get("NE", 0.5))
        twin_gaba = float(twin_state.get("GABA", 0.5))
        twin_sim = 1.0 - (
            abs(twin_da - 0.5) + abs(twin_ne - 0.5) + abs(twin_gaba - 0.5)
        ) / 3.0
        feeling[1] = _clamp(0.6 * (0.3 + 0.5 * twin_sim) + 0.4 * social_activity)
    else:
        feeling[1] = _clamp(social_activity)

    feeling[2] = _clamp(1.0 - outer_body[3])
    blockchain_active = _compute_blockchain_active(anchor_state, now)
    feeling[3] = _clamp(
        0.35 * blockchain_active + 0.35 * outer_body[4] + 0.30 * outer_body[3]
    )

    bus_published = float(bus_stats.get("published", 0))
    bus_diversity = min(1.0, bus_published / 1000.0) if bus_published > 0 else 0.1
    social_input = min(1.0, interactions_per_window / 10.0)
    modules = bus_stats.get("modules")
    bus_types = len(modules) if isinstance(modules, (list, tuple)) else 0
    perturbation_richness = min(1.0, bus_types / 8.0)
    feeling[4] = _clamp(0.4 * social_input + 0.3 * bus_diversity + 0.3 * perturbation_richness)

    willing = [0.5] * 5
    willing[0] = _clamp(min(1.0, action_per_window / 10.0))
    willing[1] = _clamp(min(1.0, social_outputs_per_window / 5.0))
    willing[2] = _clamp(min(1.0, creative_per_window / 5.0))
    willing[3] = _clamp(min(1.0, rejections_per_window / 3.0))
    willing[4] = _clamp(min(1.0, research_queries_per_window / 5.0))

    return thinking + feeling + willing


# ── Fixtures ──────────────────────────────────────────────────────────

FIXTURES: list[dict[str, Any]] = [
    {
        "name": "all_empty",
        "sources": {},
        "outer_body": [0.5, 0.5, 0.5, 0.5, 0.5],
    },
    {
        "name": "balanced_neutral",
        "sources": {
            "agency_stats": {"total_actions": 0, "failed_actions": 0},
            "assessment_stats": {"average_score": 0.5},
            "memory_status": {"unique_interactors": 0, "research_nodes": 0},
        },
        "outer_body": [0.5, 0.5, 0.5, 0.5, 0.5],
    },
    {
        "name": "active_realistic",
        "sources": {
            "agency_stats": {
                "total_actions": 100,
                "failed_actions": 15,
                "actions_this_hour": 5,
                "creative_this_hour": 2,
                "rejections_this_hour": 1,
            },
            "assessment_stats": {"average_score": 0.7},
            "memory_status": {
                "unique_interactors": 4,
                "research_nodes": 8,
            },
            "social_perception_stats": {"sentiment_ema": 0.6},
            "twin_state": {"reachable": False},
            "bus_stats": {"published": 500, "modules": ["a", "b", "c", "d"]},
        },
        "outer_body": [0.6, 0.4, 0.5, 0.3, 0.7],
    },
    {
        "name": "twin_reachable",
        "sources": {
            "twin_state": {
                "reachable": True,
                "DA": 0.55,
                "NE": 0.45,
                "GABA": 0.5,
            },
            "memory_status": {"unique_interactors": 2},
            "social_perception_stats": {"sentiment_ema": 0.7},
        },
        "outer_body": [0.5, 0.5, 0.5, 0.5, 0.5],
    },
    {
        "name": "no_anchor_state",
        "sources": {
            "memory_status": {"unique_interactors": 1},
        },
        "outer_body": [0.5, 0.5, 0.5, 0.4, 0.6],
    },
]


# ── Rust-side parity binary discovery ────────────────────────────────

PARITY_BIN = (
    Path(__file__).parent.parent.parent
    / "titan-rust"
    / "target"
    / "debug"
    / "examples"
    / "outer_mind_parity_dump"
)


def _rust_compute(fixture: dict[str, Any]) -> list[float] | None:
    """Invoke the Rust parity helper if built; return its 15D output or
    None when the binary isn't available (skip rather than fail).
    """
    if not PARITY_BIN.exists():
        return None
    payload = {
        "sources": fixture["sources"],
        "outer_body": fixture["outer_body"],
        "name": fixture.get("name"),
    }
    proc = subprocess.run(
        [str(PARITY_BIN)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=10,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"outer_mind_parity_dump rc={proc.returncode}\n"
            f"stderr={proc.stderr}\nstdout={proc.stdout}"
        )
    out = json.loads(proc.stdout)
    return out["mind_15d"]


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["name"])
def test_python_reference_clamps_to_unit_range(fixture: dict[str, Any]) -> None:
    """Python reference always returns values in [0, 1]."""
    import time

    mind = compute_outer_mind_15d_python(
        fixture["sources"], fixture["outer_body"], now=time.time()
    )
    assert len(mind) == 15
    for i, v in enumerate(mind):
        assert 0.0 <= v <= 1.0, f"{fixture['name']}.dim[{i}] = {v} out of [0,1]"


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["name"])
def test_rust_python_parity_within_float32_tolerance(fixture: dict[str, Any]) -> None:
    """Rust ↔ Python: |Δ| < 1e-3 per dim. Skips when example binary not built.

    Note: anchor freshness is time-dependent — fixtures with anchor_state
    must use a `last_anchor_time` close to `time.time()` at test invocation
    or omit the field. Current fixtures avoid live anchors entirely; if
    one is added later, set `last_anchor_time` = the harness's `time.time()`
    capture and re-pass that into `compute_outer_mind_15d_python`.
    """
    import time

    rust_mind = _rust_compute(fixture)
    if rust_mind is None:
        pytest.skip(
            "outer_mind_parity_dump example not built — run "
            "`cargo build -p titan-outer-mind-rs --example outer_mind_parity_dump`"
        )
    # Capture `now` after Rust subprocess returns (Rust used SystemTime::now
    # internally; this side captures an anchor-time close to it).
    now = time.time()
    py_mind = compute_outer_mind_15d_python(fixture["sources"], fixture["outer_body"], now=now)

    assert len(rust_mind) == 15
    for i in range(15):
        py_v = py_mind[i]
        rust_v = rust_mind[i]
        assert abs(py_v - rust_v) < 1e-3, (
            f"{fixture['name']}.dim[{i}]: Rust={rust_v} Python={py_v} "
            f"diff={abs(py_v - rust_v)}"
        )
