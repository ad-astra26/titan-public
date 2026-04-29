"""Tests for Option A — silent META consumer fixes (2026-04-28).

Closes the 3-silent-consumer hole found in OBS-meta-service-session2-soak:
EmotMeta + DreamingMeta + CodingMeta all firing 0 across all 3 Titans
because each was wired only behind a rare-edge-case trigger.

A.1 — EmotMeta routine pulse (120s) added to emot_cgn_worker.py main loop.
A.2 — DreamingMeta routine hook added to BEGIN_DREAMING in spirit_worker.py.
A.3 — CodingMeta hook added to Layer B time-fallback in spirit_worker.py.

These tests use static source inspection because the hooks live in long
worker loops that can't be unit-tested without spawning subprocesses.
The schema/shape of the requests they emit is exercised via the
existing context-builders (already tested by test_meta_service_session2.py).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EMOT_CGN_WORKER = REPO_ROOT / "titan_plugin/modules/emot_cgn_worker.py"
SPIRIT_WORKER = REPO_ROOT / "titan_plugin/modules/spirit_worker.py"
TITAN_PARAMS = REPO_ROOT / "titan_plugin/titan_params.toml"


def _read(p: Path) -> str:
    return p.read_text()


# ─────────────────────────── A.1 EmotMeta ────────────────────────────


def test_a1_emot_meta_pulse_interval_constant_declared():
    src = _read(EMOT_CGN_WORKER)
    assert "EMOT_META_PULSE_INTERVAL_S" in src, (
        "Routine-pulse cadence constant must be declared")
    assert 'cfg.get("emot_meta_pulse_interval_s", 120.0)' in src, (
        "Constant must be cfg-overridable with 120.0 default")
    assert "_last_emot_meta_pulse_ts" in src, (
        "Tracking timer must be declared")


def test_a1_emot_meta_routine_pulse_block_present():
    src = _read(EMOT_CGN_WORKER)
    # Sentinel comment + emit
    assert "F-phase routine emotional pulse" in src
    assert 'question_type="evaluate_trajectory"' in src, (
        "Routine pulse must use evaluate_trajectory question_type "
        "(distinct from ambig-anchor branch's spirit_self_nudge)")
    assert "routine_pulse" in src, (
        "Routine pulse context tag must appear in payload_snippet/context")


def test_a1_emot_meta_routine_pulse_does_not_replace_ambig_branch():
    src = _read(EMOT_CGN_WORKER)
    # Both call sites must coexist
    assert 'question_type="spirit_self_nudge"' in src, (
        "Existing ambig-anchor branch must remain (complementary trigger)")
    assert "cluster_conf < 0.5" in src, (
        "Ambig-anchor gate must remain")


def test_a1_emot_meta_pulse_param_in_titan_params_toml():
    src = _read(TITAN_PARAMS)
    assert "emot_meta_pulse_interval_s = 120.0" in src, (
        "120s default must be visible/tunable in titan_params.toml [emot_cgn]")


# ─────────────────────────── A.2 DreamingMeta ─────────────────────────


def test_a2_dreaming_meta_routine_hook_in_begin_dreaming():
    src = _read(SPIRIT_WORKER)
    # Find the BEGIN_DREAMING block
    begin_idx = src.find('elif coord_event == "BEGIN_DREAMING":')
    end_idx = src.find('elif coord_event == "END_DREAMING":')
    assert begin_idx > 0 and end_idx > begin_idx, (
        "BEGIN_DREAMING block must exist in spirit_worker")
    block = src[begin_idx:end_idx]
    assert "F-phase routine DreamingMeta" in block, (
        "Routine DreamingMeta hook must be inside BEGIN_DREAMING block")
    assert 'question_type="consolidate_themes"' in block, (
        "Routine dream entry uses consolidate_themes question_type "
        "(distinct from emergency branch's recall_context)")


def test_a2_dreaming_meta_emergency_branch_preserved():
    src = _read(SPIRIT_WORKER)
    assert 'question_type="recall_context"' in src, (
        "Emergency-fainting DreamingMeta branch must remain (complementary)")
    assert "_drain > 0.75 and _chi_total < 0.2" in src, (
        "Emergency gate must remain")


def test_a2_dreaming_meta_routine_uses_smaller_time_budget():
    """Regular dream entry is non-urgent vs emergency fainting."""
    src = _read(SPIRIT_WORKER)
    # Both budgets should appear; routine should be < emergency.
    assert "time_budget_ms=10000" in src, "Emergency budget present"
    assert "time_budget_ms=5000" in src, "Routine budget present"


def test_a2_dreaming_meta_routine_pending_dict_used():
    src = _read(SPIRIT_WORKER)
    assert "_dreaming_meta_pending[_dr_rt_meta_req_id]" in src, (
        "Routine hook must register req in _dreaming_meta_pending")


# ─────────────────────────── A.3 CodingMeta ──────────────────────────


def test_a3_coding_meta_layer_a_hook_preserved():
    """Layer A (novelty/introspect gap) hook must remain — A+B linkage."""
    src = _read(SPIRIT_WORKER)
    assert "F-phase (rFP §16.5): consult meta on trigger" in src, (
        "Layer A hook comment must remain")
    # Distinguishing context: Layer A uses _ce_send/_ce_meta_req_id,
    # Layer B (new) uses _ce_fb_send/_ce_fb_meta_req_id.
    assert "_ce_meta_req_id = _ce_send(" in src, (
        "Layer A call site must remain")


def test_a3_coding_meta_layer_b_hook_added():
    src = _read(SPIRIT_WORKER)
    assert "F-phase CodingMeta on Layer B time-fallback" in src, (
        "Layer B sentinel comment must be present")
    assert "_ce_fb_meta_req_id = _ce_fb_send(" in src, (
        "Layer B fallback call site must be present")


def test_a3_coding_meta_layer_b_uses_same_question_type_as_layer_a():
    """A+B linkage: same META_REASON_REQUEST shape so meta sees full stream."""
    src = _read(SPIRIT_WORKER)
    # formulate_strategy occurs in BOTH layers — count must be ≥ 2
    count = src.count('question_type="formulate_strategy"')
    assert count >= 2, (
        f"formulate_strategy must appear in both Layer A and Layer B "
        f"(count={count})")


def test_a3_coding_meta_layer_b_outcome_closed_dry_run():
    src = _read(SPIRIT_WORKER)
    # Verify the outcome close exists in fallback branch
    assert "session_2_dry fallback" in src, (
        "Layer B outcome close must use session_2_dry context tag")
    assert "_coding_meta_pending.pop(\n                                _ce_fb_meta_req_id, None)" in src or \
           "_coding_meta_pending.pop(_ce_fb_meta_req_id, None)" in src, (
        "Layer B outcome must pop the pending dict entry")


# ────────────────────── A.* cross-cutting invariants ──────────────────


def test_a_total_question_types_distinct_per_consumer():
    """Each consumer has unique question_types per call site for telemetry."""
    emot = _read(EMOT_CGN_WORKER)
    spirit = _read(SPIRIT_WORKER)

    # EmotMeta: routine=evaluate_trajectory, ambig=spirit_self_nudge
    emot_qts = set(re.findall(r'question_type="([^"]+)"', emot))
    assert "evaluate_trajectory" in emot_qts
    assert "spirit_self_nudge" in emot_qts

    # DreamingMeta: emergency=recall_context, routine=consolidate_themes
    spirit_qts = set(re.findall(r'question_type="([^"]+)"', spirit))
    assert "recall_context" in spirit_qts
    assert "consolidate_themes" in spirit_qts


def test_a_consumer_id_stays_canonical():
    """consumer_id strings must match meta_service_interface.consumer_home_worker."""
    emot = _read(EMOT_CGN_WORKER)
    spirit = _read(SPIRIT_WORKER)
    # emotional → emot_cgn_worker
    assert 'consumer_id="emotional"' in emot
    # dreaming + coding → spirit_worker
    assert 'consumer_id="dreaming"' in spirit
    assert 'consumer_id="coding"' in spirit
