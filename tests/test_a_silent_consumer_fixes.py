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
EMOT_CGN_WORKER = REPO_ROOT / "titan_hcl/modules/emot_cgn_worker.py"
SPIRIT_WORKER = REPO_ROOT / "titan_hcl/modules/spirit_worker.py"
TITAN_PARAMS = REPO_ROOT / "titan_hcl/titan_params.toml"


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


# ─────────────── A.2 DreamingMeta / A.3 CodingMeta — RETIRED ──────────
# D-SPEC-116 (2026-05-22): these guarded the DreamingMeta routine-pulse +
# CodingMeta layer-B hooks as SOURCE STRINGS inside spirit_worker.py's
# ~10.4k-LOC body. That body was deleted at D8-3 (the engines moved to
# cognitive_worker — see its consumer_home_worker map `"coding"`/`"dreaming"`),
# and spirit_worker.py itself is now fully DELETED. These tests were already
# red on the heartbeat stub (the strings left with the body). The capability is
# covered by test_meta_service_session2.py + cognitive_worker's own suite.
# A.1 (EmotMeta in emot_cgn_worker) above is unaffected.
#
# AUDIT CONCLUSION (D-SPEC-116, 2026-05-22 — both hooks traced to ground):
#  • A.2 DreamingMeta routine pulse: the consult used
#    question_type="consolidate_themes", which was NEVER in
#    meta_service_client.KNOWN_QUESTION_TYPES nor meta_service
#    .QUESTION_TYPE_TO_PRIMITIVE → send_meta_request raised ValueError →
#    caught+skipped. So it was DEAD-ON-ARRIVAL (same failure class as the
#    documented evaluate_trajectory "dormant 14 days" rejection); it never
#    actually fired even before the §4.I dream_state_worker extraction
#    (commit 4aead6fb / D-SPEC-56) deleted the BEGIN_DREAMING block.
#    Faithful restore = dead code; a meaningful restore needs a question-type
#    decision (whitelist consolidate_themes + map to a primitive, OR reuse the
#    valid synthesize_insight → SYNTHESIZE). DEFERRED to Maker contract call.
#  • A.3 CodingMeta layer-B: used question_type="formulate_strategy" +
#    consumer_id="coding" (BOTH valid). It was a real consult, dropped when
#    coding_explorer migrated to self_reflection_worker. BUT it fires only when
#    coding_explorer.explore() runs, which is DELIBERATELY LATENT
#    (rFP_coding_explorer_activation — gap conditions rarely fire on stable
#    Titans; the 6h Layer-B fallback exists to force it). Re-home belongs WITH
#    the coding-explorer activation work (needs neuromod context plumbing).
# Tracked: project_spirit_cognitive_migration_dropped_orchestration_loops.

def test_a2_a3_meta_hooks_migrated_off_deleted_spirit_worker():
    """spirit_worker.py (which hosted the A.2/A.3 hook source) is deleted; the
    dreaming/coding meta consumers now live in cognitive_worker."""
    import importlib.util
    assert importlib.util.find_spec("titan_hcl.modules.spirit_worker") is None
    from titan_hcl.modules import cognitive_worker
    import inspect
    src = inspect.getsource(cognitive_worker)
    # consumer_home_worker map carries the migrated dreaming + coding consumers.
    assert '"dreaming"' in src and '"coding"' in src
