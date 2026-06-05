"""EEL Pillar 0 — Phase 0a: grounded execution-mode router decision table.

Offline unit tests for titan_hcl/logic/sage/grounded_router.py (pure logic,
no agno/torch/plugin). Verifies the locked §7.0 lane→strength cascade, the
metabolic gate, graceful skill-lane degradation (empty until B1), IQL passive
pass-through, the config load, and the dispatch-string contract.
"""

import pytest

from titan_hcl.logic.sage.grounded_router import (
    GroundedReadout,
    RouterThresholds,
    RouterDecision,
    grounded_route,
    load_router_thresholds,
    is_informational_query,
    recall_score_from_memories,
    MODE_SOVEREIGN,
    MODE_COLLABORATIVE,
    MODE_RESEARCH,
    MODE_SHADOW,
    MODE_SKILL_DELEGATE,
    MODE_TOOL_ORACLE,
    LANE_COMPUTATIONAL,
    LANE_SKILL,
    LANE_INFORMATIONAL,
    LANE_CONVERSATIONAL,
)

T = RouterThresholds()  # in-code defaults (== the toml bootstrap)


# ── Lane 1: computational / verifiable ───────────────────────────────────────

def test_computational_routes_to_tool_oracle():
    d = grounded_route(GroundedReadout(requires_tool=True), T)
    assert d.mode == MODE_TOOL_ORACLE
    assert d.lane == LANE_COMPUTATIONAL


def test_computational_wins_over_skill_and_recall():
    # A deterministic oracle is the most reliable path → beats a strong recall
    # AND a delegatable skill (lane-1 precedence).
    d = grounded_route(
        GroundedReadout(
            requires_tool=True,
            recall_score=0.99,
            skill_utility=0.99,
            is_informational=True,
        ),
        T,
    )
    assert d.mode == MODE_TOOL_ORACLE


# ── Lane 2: skill / procedural ───────────────────────────────────────────────

def test_skill_above_floor_routes_to_skill_delegate():
    d = grounded_route(GroundedReadout(skill_utility=0.80), T)
    assert d.mode == MODE_SKILL_DELEGATE
    assert d.lane == LANE_SKILL


def test_skill_wins_over_memory_lane():
    # §1 ambiguous-case resolution: procedural task with BOTH strong recall and
    # a verified skill → skill-lane wins (checked before the memory lane).
    d = grounded_route(
        GroundedReadout(skill_utility=0.90, recall_score=0.99, is_informational=True),
        T,
    )
    assert d.mode == MODE_SKILL_DELEGATE


def test_skill_below_floor_falls_through():
    # Weak skill is not delegatable → falls through to the informational lane.
    d = grounded_route(
        GroundedReadout(skill_utility=0.50, is_informational=True, recall_score=0.0),
        T,
    )
    assert d.mode == MODE_RESEARCH
    assert d.lane == LANE_INFORMATIONAL


def test_skill_empty_degrades_gracefully():
    # Until B1 populates skills, skill_utility is None → the skill lane is inert
    # and routing still works (locked sequencing: Phase 0 ships skill-readout empty).
    d = grounded_route(
        GroundedReadout(skill_utility=None, is_informational=True, recall_score=0.0),
        T,
    )
    assert d.mode == MODE_RESEARCH


# ── Lane 3: informational / memory ───────────────────────────────────────────

def test_informational_strong_recall_to_sovereign():
    d = grounded_route(GroundedReadout(is_informational=True, recall_score=0.80), T)
    assert d.mode == MODE_SOVEREIGN
    assert d.lane == LANE_INFORMATIONAL


def test_informational_strong_engram_to_sovereign():
    # A grounded Engram counts as strong substrate even with zero episodic recall.
    d = grounded_route(
        GroundedReadout(is_informational=True, recall_score=0.0, engram_ground=0.40),
        T,
    )
    assert d.mode == MODE_SOVEREIGN


def test_informational_partial_recall_to_collaborative():
    d = grounded_route(GroundedReadout(is_informational=True, recall_score=0.45), T)
    assert d.mode == MODE_COLLABORATIVE


def test_informational_no_memory_affordable_to_research():
    d = grounded_route(
        GroundedReadout(is_informational=True, recall_score=0.0, can_afford_research=True),
        T,
    )
    assert d.mode == MODE_RESEARCH


def test_informational_no_memory_unaffordable_to_shadow():
    # Metabolic gate: no memory + research unaffordable (low SOL) → honest IDK.
    d = grounded_route(
        GroundedReadout(is_informational=True, recall_score=0.0, can_afford_research=False),
        T,
    )
    assert d.mode == MODE_SHADOW


def test_boundary_recall_known_floor_inclusive():
    # recall_score exactly at the floor → Sovereign (>= is inclusive).
    d = grounded_route(GroundedReadout(is_informational=True, recall_score=T.recall_known_floor), T)
    assert d.mode == MODE_SOVEREIGN


def test_boundary_recall_present_floor_inclusive():
    d = grounded_route(GroundedReadout(is_informational=True, recall_score=T.recall_present_floor), T)
    assert d.mode == MODE_COLLABORATIVE


# ── Lane 4: conversational / creative ─────────────────────────────────────────

def test_conversational_defaults_to_sovereign():
    # No tool / skill / informational signal → answer directly (no oracle → never
    # skill-scored, §1.0 oracle roster).
    d = grounded_route(GroundedReadout(), T)
    assert d.mode == MODE_SOVEREIGN
    assert d.lane == LANE_CONVERSATIONAL


def test_conversational_strong_substrate_still_sovereign():
    d = grounded_route(GroundedReadout(recall_score=0.90, is_informational=False), T)
    assert d.mode == MODE_SOVEREIGN
    assert d.lane == LANE_CONVERSATIONAL


# ── IQL refinement is PASSIVE in 0a/0b ───────────────────────────────────────

def test_iql_advantage_is_carried_but_does_not_change_mode():
    base = grounded_route(GroundedReadout(is_informational=True, recall_score=0.80), T)
    with_iql = grounded_route(
        GroundedReadout(is_informational=True, recall_score=0.80), T,
        iql_advantage=-0.99,  # strongly disagrees — would veto IF active (0c)
    )
    assert base.mode == with_iql.mode == MODE_SOVEREIGN   # passive: mode unchanged
    assert with_iql.iql_advantage == -0.99               # but recorded for 0c / telemetry
    assert base.iql_advantage is None


# ── Determinism / purity ─────────────────────────────────────────────────────

def test_route_is_deterministic():
    ro = GroundedReadout(is_informational=True, recall_score=0.45, engram_ground=0.1)
    assert grounded_route(ro, T) == grounded_route(ro, T)


def test_decision_carries_a_reason():
    d = grounded_route(GroundedReadout(requires_tool=True), T)
    assert isinstance(d, RouterDecision) and d.reason


# ── Config load ──────────────────────────────────────────────────────────────

def test_load_thresholds_from_toml_matches_bootstrap():
    # Reads the real titan_params.toml [gatekeeper.grounded_router]; also proves
    # the section parses (catches toml syntax errors).
    loaded = load_router_thresholds()
    assert loaded.recall_known_floor == pytest.approx(0.65)
    assert loaded.recall_present_floor == pytest.approx(0.30)
    assert loaded.engram_ground_floor == pytest.approx(0.30)
    assert loaded.skill_promote_floor == pytest.approx(0.70)
    assert loaded.iql_veto_margin == pytest.approx(0.25)


def test_default_thresholds_present():
    d = RouterThresholds()
    assert d.recall_known_floor > d.recall_present_floor  # ordering invariant


# ── Dispatch-string contract (must match existing agno_hooks branches) ────────

def test_mode_strings_match_existing_dispatch():
    assert MODE_SOVEREIGN == "Sovereign"
    assert MODE_COLLABORATIVE == "Collaborative"
    assert MODE_RESEARCH == "STATE_NEED_RESEARCH"
    assert MODE_SHADOW == "Shadow"


# ── Readout helpers (0b): informational classifier + recall extraction ───────

def test_is_informational_query_detects_realtime():
    assert is_informational_query("what is the SOL price today")
    assert is_informational_query("latest news on the launch")
    assert is_informational_query("what's the current TVL of Jupiter?")


def test_is_informational_query_false_for_creative():
    assert not is_informational_query("write me a poem about the sea")
    assert not is_informational_query("tell me about your day")
    assert not is_informational_query("")


def test_is_informational_query_is_case_insensitive():
    assert is_informational_query("BREAKING: ANNOUNCEMENT")


def test_recall_score_from_dict_memories():
    mems = [{"effective_weight": 0.3}, {"effective_weight": 0.82}, {"effective_weight": 0.5}]
    assert recall_score_from_memories(mems) == pytest.approx(0.82)


def test_recall_score_empty_is_zero():
    assert recall_score_from_memories([]) == 0.0
    assert recall_score_from_memories(None) == 0.0


def test_recall_score_falls_back_to_mempool_weight():
    assert recall_score_from_memories([{"mempool_weight": 0.6}]) == pytest.approx(0.6)


def test_recall_score_handles_missing_and_bad_weights():
    # No weight key → 0 contribution; non-numeric → ignored; max wins.
    mems = [{"no_weight": 1}, {"effective_weight": "bad"}, {"effective_weight": 0.4}]
    assert recall_score_from_memories(mems) == pytest.approx(0.4)


def test_recall_score_from_object_attr():
    class _Rec:
        effective_weight = 0.7
    assert recall_score_from_memories([_Rec()]) == pytest.approx(0.7)


def test_recall_extraction_feeds_router_to_research_when_empty():
    # End-to-end: empty recall + informational → research (the EEL-G0 core).
    ro = GroundedReadout(
        recall_score=recall_score_from_memories([]),
        is_informational=is_informational_query("what's the latest SOL price?"),
    )
    assert grounded_route(ro, T).mode == MODE_RESEARCH
