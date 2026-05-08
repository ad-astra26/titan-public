"""
Cross-language parity test for outer-spirit 45D projection.

Verifies that Rust ``titan_outer_spirit_rs::tick_loop::project_outer_spirit_45d``
and Python ``outer_spirit_tensor::collect_outer_spirit_45d`` (wrapped by
``outer_trinity::_collect_extended`` preprocessing) produce numerically-
equivalent 45D outputs for the same set of msgpack source dict + outer-body
+ outer-mind.

Closes rFP_phase_c_definitive_runtime_closure D1 (GAP-CS6-001).

Why fixtures + JSON-not-msgpack on the wire:
- The Rust example binary takes JSON on stdin (deterministic, hand-editable);
  it converts JSON → rmpv::Value internally before invoking the project fn,
  so the test harness exercises the same msgpack→[f32;45] path that the
  daemon uses on T3 with a real ``sensor_cache_outer_spirit.bin`` payload.
- Tolerance |Δ| < 1e-3 mirrors the chi / outer_mind parity convention.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pytest


# ── Python reference (inlined mirror of _collect_extended → collect_outer_spirit_45d) ──
#
# Inlined rather than imported because outer_trinity.py pulls heavy deps;
# test only needs the math.


def _clamp(v: float) -> float:
    if math.isnan(v):
        return 0.5
    return max(0.0, min(1.0, v))


def compute_outer_spirit_45d_python(
    sources: dict[str, Any], outer_body: list[float], outer_mind: list[float]
) -> list[float]:
    """Python reference — mirrors Rust ``project_outer_spirit_45d``.

    Implements both the ``_collect_extended`` preprocessing
    (``outer_trinity.py:600-697``) and ``collect_outer_spirit_45d``
    formulas (``outer_spirit_tensor.py:53-302``) inline.
    """
    agency = sources.get("agency_stats") or {}
    assessment = sources.get("assessment_stats") or {}
    memory_status = sources.get("memory_status") or {}
    social_perception = sources.get("social_perception_stats") or {}
    recovery = sources.get("recovery_stats") or {}
    hormone_levels = sources.get("hormone_levels") or {}
    solana = sources.get("solana_stats") or {}
    history = sources.get("history") or {}

    art_count = float(sources.get("art_count_500") or 0)
    audio_count = float(sources.get("audio_count_500") or 0)
    uptime = max(1.0, float(sources.get("uptime_seconds") or 1.0))

    # Preprocess
    total_actions = float(agency.get("total_actions", 0) or 0)
    failed_actions = float(agency.get("failed_actions", 0) or 0)
    action_success_rate = (total_actions - failed_actions) / max(1.0, total_actions)
    actions_per_hour = total_actions / max(0.01, uptime / 3600.0)
    failed_retry_rate = float(agency.get("failed_retry_rate", 0.0) or 0.0)
    burst_frequency = float(agency.get("burst_frequency", 0.0) or 0.0)
    action_error_rate = 1.0 - action_success_rate

    creative_total = art_count + audio_count
    creative_unique_types = float(min(2, (1 if art_count > 0 else 0) + (1 if audio_count > 0 else 0)))
    creative_mean_assessment = float(assessment.get("average_score", 0.5))

    threats_detected = float(agency.get("threats_detected", 0) or 0)
    rejections = float(agency.get("rejections", 0) or 0)
    sovereignty_ratio = float(agency.get("sovereignty_ratio", 0.0) or 0.0)

    social_mean_conversation_quality = float(assessment.get("average_score", 0.5))

    assessment_mean = float(assessment.get("average_score", 0.5))
    assessment_trend = float(assessment.get("trend", 0.0))
    assessment_score_variance = float(assessment.get("score_variance", 0.3))

    mem_persistent_nodes = float(memory_status.get("persistent_count", 0) or 0)
    mem_growth_per_epoch = float(memory_status.get("growth_per_epoch", 0) or 0)

    uptime_ratio = min(1.0, uptime / max(1.0, uptime + 60.0))

    # Coherence
    body_coh = sum(outer_body) / len(outer_body) if outer_body else 0.5
    mind_coh = sum(outer_mind) / len(outer_mind) if outer_mind else 0.5
    combined_coh = (body_coh + mind_coh) / 2.0

    # ── SAT (15D) ──
    sat = [0.5] * 15
    sat[0] = _clamp(float(solana.get("identity_verified", 0.5)))
    sat[1] = _clamp(0.5)  # inner_outer_coherence — never set
    sat[2] = _clamp(sovereignty_ratio)
    sat[3] = _clamp(rejections / max(1.0, threats_detected)) if threats_detected > 0 else 0.8
    sat[4] = _clamp(uptime_ratio)
    sat[5] = _clamp(float(solana.get("genesis_nft_exists", 0.5)))
    sat[6] = _clamp(0.5 + assessment_trend)
    tx_count = float(solana.get("tx_count", 0) or 0)
    total_outputs = total_actions + creative_total + tx_count
    sat[7] = _clamp(min(1.0, total_outputs / 200.0))
    sat[8] = _clamp(1.0 - assessment_score_variance)
    sat[9] = _clamp(assessment_mean * action_success_rate * 2.0)
    mean_recovery_s = float(recovery.get("mean_recovery_seconds", 60.0))
    sat[10] = _clamp(1.0 / (1.0 + mean_recovery_s / 30.0))
    sat[11] = _clamp(1.0 - 0.3)  # load_variance — never set
    sat[12] = _clamp(min(1.0, creative_unique_types / 5.0))
    sat[13] = _clamp(float(solana.get("tx_success_rate", 0.5)))
    sat[14] = _clamp(min(1.0, actions_per_hour / 20.0) * uptime_ratio)

    # ── CHIT (15D) ──
    chit = [0.5] * 15
    chit[0] = _clamp(min(1.0, mem_persistent_nodes / 2000.0))
    chit[1] = _clamp(combined_coh)
    # confirmed_threats never plumbed → 0
    chit[2] = _clamp(0.0 / max(1.0, threats_detected)) if threats_detected > 0 else 0.8
    chit[3] = 0.5  # multi_source_success — never set
    chit[4] = _clamp(body_coh * mind_coh * 2.0)
    chit[5] = 0.5  # pattern_reuse_rate — never set
    chit[6] = _clamp(min(1.0, mem_growth_per_epoch / 10.0))
    chit[7] = 0.5  # research_usage_rate — fixed 0.5
    chit[8] = _clamp(social_mean_conversation_quality)
    chit[9] = _clamp(0.5 + assessment_trend)
    dream_recall = float(history.get("dream_recall_ratio", 0.0))
    chit[10] = _clamp(dream_recall * 0.6 + body_coh * 0.4)
    chit[11] = _clamp(float(history.get("circadian_alignment", 0.5)))
    chit[12] = _clamp(body_coh)
    chit[13] = 0.5  # correlation_strength — never set
    chit[14] = _clamp(float(history.get("outer_spirit_trajectory", 0.5)))

    # ── ANANDA (15D) ──
    ananda = [0.5] * 15
    ananda[0] = _clamp(action_success_rate)
    ananda[1] = _clamp(social_mean_conversation_quality)
    ananda[2] = _clamp(creative_mean_assessment)
    ananda[3] = _clamp(1.0 - 0.1)  # cross_module_error_rate default 0.1
    ananda[4] = _clamp(creative_mean_assessment)
    ananda[5] = 0.5  # research_accuracy — never set
    ananda[6] = _clamp(min(1.0, 0.0 / 5.0))  # new_connections — never set
    ananda[7] = _clamp(min(1.0, 0.0 / 3.0))  # novel_types — never set
    ananda[8] = 0.5  # creative_engagement — never set
    ananda[9] = 0.5  # research_to_action_rate — never set
    ananda[10] = _clamp(float(history.get("rest_performance_floor", 0.5)))
    creativity_level = float(hormone_levels.get("CREATIVITY", 0.0))
    time_since_create = float(history.get("seconds_since_last_create", 300.0))
    ananda[11] = _clamp(creativity_level * min(1.0, time_since_create / 600.0))
    resource_depletion = 1.0 - body_coh
    surrender = 1.0 - _clamp((failed_retry_rate + resource_depletion + burst_frequency) / 3.0)
    ananda[12] = _clamp(surrender)
    ananda[13] = _clamp(float(history.get("resource_efficiency", 0.5)))
    min_coherence = min(body_coh, mind_coh)
    error_factor = 1.0 - action_error_rate
    ananda[14] = _clamp(min_coherence * error_factor * assessment_mean * ananda[12])

    return sat + chit + ananda


# ── Fixtures ──────────────────────────────────────────────────────────

FIXTURES: list[dict[str, Any]] = [
    {
        "name": "all_empty",
        "sources": {},
        "outer_body": [0.5, 0.5, 0.5, 0.5, 0.5],
        "outer_mind": [0.5] * 15,
    },
    {
        "name": "balanced_neutral",
        "sources": {
            "agency_stats": {"total_actions": 0, "failed_actions": 0},
            "assessment_stats": {"average_score": 0.5},
            "memory_status": {"persistent_count": 0, "unique_interactors": 0},
            "uptime_seconds": 60.0,
        },
        "outer_body": [0.5] * 5,
        "outer_mind": [0.5] * 15,
    },
    {
        "name": "active_realistic",
        "sources": {
            "agency_stats": {
                "total_actions": 100,
                "failed_actions": 15,
                "actions_this_hour": 8,
                "creative_this_hour": 3,
                "rejections": 4,
                "threats_detected": 5,
                "sovereignty_ratio": 0.65,
                "failed_retry_rate": 0.05,
                "burst_frequency": 0.1,
            },
            "assessment_stats": {
                "average_score": 0.7,
                "trend": 0.04,
                "score_variance": 0.2,
            },
            "memory_status": {
                "persistent_count": 250,
                "growth_per_epoch": 3.0,
                "unique_interactors": 4,
            },
            "social_perception_stats": {"sentiment_ema": 0.6},
            "uptime_seconds": 7200.0,
            "art_count_500": 4,
            "audio_count_500": 2,
        },
        "outer_body": [0.6, 0.5, 0.55, 0.4, 0.65],
        "outer_mind": [0.5, 0.55, 0.6, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    },
    {
        "name": "high_history",
        "sources": {
            "history": {
                "dream_recall_ratio": 0.8,
                "circadian_alignment": 0.7,
                "outer_spirit_trajectory": 0.55,
                "rest_performance_floor": 0.85,
                "seconds_since_last_create": 240.0,
                "resource_efficiency": 0.9,
            },
            "hormone_levels": {"CREATIVITY": 0.7},
        },
        "outer_body": [0.5] * 5,
        "outer_mind": [0.5] * 15,
    },
    {
        "name": "high_coherence",
        "sources": {
            "agency_stats": {
                "total_actions": 50,
                "failed_actions": 2,
                "actions_this_hour": 12,
            },
            "assessment_stats": {"average_score": 0.85, "score_variance": 0.1},
            "uptime_seconds": 14400.0,
        },
        "outer_body": [0.85, 0.85, 0.85, 0.85, 0.85],
        "outer_mind": [0.85] * 15,
    },
    {
        "name": "solana_engaged",
        "sources": {
            "solana_stats": {
                "identity_verified": 1.0,
                "genesis_nft_exists": 1.0,
                "tx_success_rate": 0.95,
                "tx_count": 25,
            },
        },
        "outer_body": [0.5] * 5,
        "outer_mind": [0.5] * 15,
    },
]


# ── Rust-side parity binary discovery ────────────────────────────────

PARITY_BIN = (
    Path(__file__).parent.parent.parent
    / "titan-rust"
    / "target"
    / "debug"
    / "examples"
    / "outer_spirit_parity_dump"
)


def _rust_compute(fixture: dict[str, Any]) -> list[float] | None:
    if not PARITY_BIN.exists():
        return None
    payload = {
        "sources": fixture["sources"],
        "outer_body": fixture["outer_body"],
        "outer_mind": fixture["outer_mind"],
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
            f"outer_spirit_parity_dump rc={proc.returncode}\n"
            f"stderr={proc.stderr}\nstdout={proc.stdout}"
        )
    out = json.loads(proc.stdout)
    return out["spirit_45d"]


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["name"])
def test_python_reference_clamps_to_unit_range(fixture: dict[str, Any]) -> None:
    """Python reference always returns 45D values in [0, 1]."""
    spirit = compute_outer_spirit_45d_python(
        fixture["sources"], fixture["outer_body"], fixture["outer_mind"]
    )
    assert len(spirit) == 45
    for i, v in enumerate(spirit):
        assert 0.0 <= v <= 1.0, f"{fixture['name']}.dim[{i}] = {v} out of [0,1]"


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["name"])
def test_rust_python_parity_within_float32_tolerance(fixture: dict[str, Any]) -> None:
    """Rust ↔ Python: |Δ| < 1e-3 per dim. Skips when example binary not built."""
    rust_spirit = _rust_compute(fixture)
    if rust_spirit is None:
        pytest.skip(
            "outer_spirit_parity_dump example not built — run "
            "`cargo build -p titan-outer-spirit-rs --example outer_spirit_parity_dump`"
        )
    py_spirit = compute_outer_spirit_45d_python(
        fixture["sources"], fixture["outer_body"], fixture["outer_mind"]
    )

    assert len(rust_spirit) == 45
    for i in range(45):
        py_v = py_spirit[i]
        rust_v = rust_spirit[i]
        assert abs(py_v - rust_v) < 1e-3, (
            f"{fixture['name']}.dim[{i}]: Rust={rust_v} Python={py_v} "
            f"diff={abs(py_v - rust_v)}"
        )


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["name"])
def test_distinct_dim_count_under_realistic_input(fixture: dict[str, Any]) -> None:
    """Under realistic input, 45D output must contain ≥10 distinct values
    (closes the GAP-CS6-001 stub that produced only 4 distinct values)."""
    if fixture["name"] != "active_realistic":
        pytest.skip("only checks the realistic fixture")
    rust_spirit = _rust_compute(fixture)
    if rust_spirit is None:
        pytest.skip("outer_spirit_parity_dump example not built")
    distinct = {round(v, 3) for v in rust_spirit}
    assert len(distinct) >= 10, (
        f"Expected ≥10 distinct dims under realistic input, got {len(distinct)}: "
        f"{sorted(distinct)}"
    )
