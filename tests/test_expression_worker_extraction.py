"""Tests for §4.B expression_worker extraction (D-SPEC-53, v1.7.4).

Closure-tier tests per `rFP_titan_hcl_l2_separation_strategy.md §8`:
  1. SPEC parity — Changelog + glossary + §9.B block + D-SPEC-53
  2. Bus constants — NS_REWARD + EXPRESSION_WORKER_READY exist
  3. ExpressionManager moved — cognitive_worker no longer instantiates
  4. HormonalShmReader smoke + canonical hormone-order parity
  5. Worker entry-fn + subscribe-topics surface
  6. ModuleSpec registration in core/plugin.py
  7. cognitive_worker subscribes to SPEAK_REQUEST_PENDING + NS_REWARD
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─── 1. SPEC parity ───────────────────────────────────────────────────


SPEC_PATH = REPO_ROOT / "titan-docs" / "specs" / "SPEC_titan_architecture.md"


def test_spec_version_bumped_to_1_7_4():
    """v1.7.4 PATCH shipped expression_worker per D-SPEC-53. The bump must
    remain in the SPEC Changelog as historical evidence even after subsequent
    bumps update the frontmatter (test refactored 2026-05-15 during v1.8.2
    §4.I dream_state_worker carve — frontmatter check became too narrow once
    SPEC_VERSION advanced past 1.7.4)."""
    text = SPEC_PATH.read_text()
    assert "v1.7.4 (PATCH)" in text, (
        "SPEC Changelog must carry v1.7.4 (PATCH) row per D-SPEC-53")


def test_spec_changelog_has_v1_7_4_row():
    text = SPEC_PATH.read_text()
    assert "v1.7.4 (PATCH)" in text, (
        "SPEC Changelog must carry a v1.7.4 (PATCH) row per "
        "feedback_spec_changelog_mandatory.md")
    assert "expression_worker" in text and "D-SPEC-53" in text


def test_spec_section_1_glossary_has_expression_worker_row():
    text = SPEC_PATH.read_text()
    assert "| **expression_worker** | (none — new in v1.7.4)" in text, (
        "§1 Glossary must carry an expression_worker row")


def test_spec_section_9b_has_expression_worker_block():
    text = SPEC_PATH.read_text()
    assert "#### expression_worker (Python L2 module" in text, (
        "§9.B must contain a dedicated expression_worker block")
    assert "D-SPEC-53" in text


def test_spec_section_21_has_dspec_53_entry():
    text = SPEC_PATH.read_text()
    assert "**D-SPEC-53" in text, "§21 Decision Log must contain D-SPEC-53"
    # Maker Q1/Q2/Q3 decisions captured inline per
    # feedback_spec_changes_need_maker_greenlight_first.md
    assert "Q1 hormone-levels" in text
    assert "Q2 tick cadence" in text
    assert "Q3 Tier-1 SPEAK" in text


def test_spec_cognitive_worker_owns_list_no_longer_lists_expression_manager():
    """cognitive_worker §1 glossary + §9.B Owns list must reflect the
    extraction (ExpressionManager moved out)."""
    text = SPEC_PATH.read_text()
    # Glossary row (line ~303) explicitly notes the removal.
    assert ("ExpressionManager removed in v1.7.4" in text), (
        "cognitive_worker glossary row must annotate the removal")
    # §9.B tree must NOT show ExpressionManager in the Owns list anymore.
    cognitive_block_start = text.find("#### cognitive_worker (Python L2")
    assert cognitive_block_start != -1
    cognitive_block_end = text.find("#### ", cognitive_block_start + 1)
    cognitive_block = text[cognitive_block_start:cognitive_block_end]
    # The owned-engines bullet line previously read
    # "PiHeartbeatMonitor, ObservableEngine, ExpressionManager" —
    # ExpressionManager must be gone (the new line lists only the
    # first two).
    assert "PiHeartbeatMonitor, ObservableEngine, ExpressionManager" not in \
        cognitive_block, (
            "§9.B cognitive_worker Owns list still claims ExpressionManager")


# ─── 2. Bus constants ─────────────────────────────────────────────────


def test_bus_has_ns_reward_constant():
    from titan_hcl import bus
    assert hasattr(bus, "NS_REWARD")
    assert bus.NS_REWARD == "NS_REWARD"


def test_bus_has_expression_worker_ready_constant():
    from titan_hcl import bus
    assert hasattr(bus, "EXPRESSION_WORKER_READY")
    assert bus.EXPRESSION_WORKER_READY == "EXPRESSION_WORKER_READY"


def test_bus_has_speak_request_pending_constant():
    from titan_hcl import bus
    assert hasattr(bus, "SPEAK_REQUEST_PENDING")
    assert bus.SPEAK_REQUEST_PENDING == "SPEAK_REQUEST_PENDING"


# ─── 3. ExpressionManager extraction (cognitive_worker delta) ─────────


COGNITIVE_WORKER_PATH = (
    REPO_ROOT / "titan_hcl" / "modules" / "cognitive_worker.py")


def test_cognitive_worker_no_longer_instantiates_expression_manager():
    """cognitive_worker.py must NOT contain `ExpressionManager()` invocation
    nor the 6 `create_<composite>()` registration lines anymore."""
    text = COGNITIVE_WORKER_PATH.read_text()
    assert "ExpressionManager()" not in text, (
        "cognitive_worker still instantiates ExpressionManager — "
        "extraction incomplete")
    for create_fn in (
        "create_speak()", "create_art()", "create_music()",
        "create_social()", "create_kin_sense()", "create_longing()",
    ):
        assert create_fn not in text, (
            f"cognitive_worker still registers {create_fn} — "
            "extraction incomplete")


def test_cognitive_worker_subscribes_to_speak_request_pending_and_ns_reward():
    """cognitive_worker's subscribe-topics list must include the two new
    bus events from expression_worker."""
    from titan_hcl.modules.cognitive_worker import (
        _COGNITIVE_WORKER_SUBSCRIBE_TOPICS,
    )
    from titan_hcl import bus
    assert bus.SPEAK_REQUEST_PENDING in _COGNITIVE_WORKER_SUBSCRIBE_TOPICS
    assert bus.NS_REWARD in _COGNITIVE_WORKER_SUBSCRIBE_TOPICS


# ─── 4. HormonalShmReader ─────────────────────────────────────────────


def test_hormonal_shm_reader_module_imports_clean():
    from titan_hcl.logic.hormonal_shm_reader import (
        HormonalShmReader, HORMONE_NAMES,
    )
    assert callable(HormonalShmReader)
    # Hormone order must mirror NS_PROGRAMS canonical row order per SPEC
    # §3.1 D05 hormonal_state.bin schema. 5 inner + 6 outer = 11 total.
    assert len(HORMONE_NAMES) == 11
    assert HORMONE_NAMES[:5] == (
        "REFLEX", "FOCUS", "INTUITION", "IMPULSE", "METABOLISM")
    assert HORMONE_NAMES[5:] == (
        "CREATIVITY", "CURIOSITY", "EMPATHY",
        "REFLECTION", "INSPIRATION", "VIGILANCE")


def test_hormonal_shm_reader_canonical_order_matches_hormonal_worker():
    """The reader's hormone order must mirror the writer's
    (hormonal_worker.HORMONE_NAMES) — single source of truth is the
    NS_PROGRAMS row order per SPEC §3.1 D05."""
    from titan_hcl.logic.hormonal_shm_reader import (
        HORMONE_NAMES as READER_NAMES,
    )
    from titan_hcl.modules.hormonal_worker import (
        HORMONE_NAMES as WRITER_NAMES,
    )
    assert tuple(READER_NAMES) == tuple(WRITER_NAMES), (
        "HormonalShmReader.HORMONE_NAMES drifted from "
        "hormonal_worker.HORMONE_NAMES — slot byte layout would be "
        "misaligned. Both must follow NS_PROGRAMS canonical order.")


def test_hormonal_shm_reader_cold_boot_returns_empty_dict():
    """When the slot is absent (synthetic titan_id, no writer), the
    reader returns empty dict — the evaluator short-circuits cleanly."""
    from titan_hcl.logic.hormonal_shm_reader import HormonalShmReader
    r = HormonalShmReader(titan_id="T_EXPRESSION_WORKER_TEST_COLD_BOOT")
    out = r.get_hormone_levels()
    assert out == {}, "Cold-boot must return empty dict, not raise"


# ─── 5. expression_worker module surface ──────────────────────────────


def test_expression_worker_entry_function_callable():
    from titan_hcl.modules.expression_worker import expression_worker_main
    assert callable(expression_worker_main)


def test_expression_worker_subscribe_topics_includes_kernel_epoch_tick():
    from titan_hcl.modules.expression_worker import (
        _EXPRESSION_WORKER_SUBSCRIBE_TOPICS,
    )
    from titan_hcl import bus
    # Per Maker Q2 — KERNEL_EPOCH_TICK drives evaluate_all.
    assert bus.KERNEL_EPOCH_TICK in _EXPRESSION_WORKER_SUBSCRIBE_TOPICS
    assert bus.MODULE_SHUTDOWN in _EXPRESSION_WORKER_SUBSCRIBE_TOPICS
    assert bus.SAVE_NOW in _EXPRESSION_WORKER_SUBSCRIBE_TOPICS


def test_expression_worker_init_expression_manager_registers_six_composites():
    from titan_hcl.modules.expression_worker import (
        _init_expression_manager,
    )
    em = _init_expression_manager()
    assert em is not None, "_init_expression_manager must not return None"
    assert set(em.composites.keys()) == {
        "SPEAK", "ART", "MUSIC", "SOCIAL", "KIN_SENSE", "LONGING",
    }


def test_expression_worker_strong_composition_thresholds_match_spec():
    """The catalyst-site #8 closure (D8-3 prereq) gates `strong_composition`
    SOCIAL_CATALYST emit on `level ≥ 7` AND `confidence ≥ 0.8`. Both
    constants exposed for any downstream tuning rFP."""
    from titan_hcl.modules.expression_worker import (
        _STRONG_COMPOSITION_LEVEL_GATE,
        _STRONG_COMPOSITION_CONFIDENCE_GATE,
    )
    assert _STRONG_COMPOSITION_LEVEL_GATE == 7
    assert _STRONG_COMPOSITION_CONFIDENCE_GATE == 0.8


# ─── 6. ModuleSpec registration ───────────────────────────────────────


PLUGIN_PATH = REPO_ROOT / "titan_hcl" / "core" / "plugin.py"


def test_plugin_registers_expression_worker_module_spec():
    text = PLUGIN_PATH.read_text()
    assert "from titan_hcl.modules.expression_worker import" in text
    assert "expression_worker_main" in text
    assert 'name="expression_worker"' in text
    # rss_limit must reflect the lightweight composite-ledger footprint.
    assert "rss_limit_mb=400" in text


def test_plugin_gates_off_expression_state_publisher_under_l0_rust_true():
    """G21 single-writer ownership transfer — main plugin's
    `_expression_state_publish_loop` thread MUST be gated off when
    `l0_rust_enabled=true` (expression_worker owns the slot)."""
    text = PLUGIN_PATH.read_text()
    # String is split across a logger.info call; check the load-bearing
    # phrases on both sides of the split.
    assert "SKIPPED under l0_rust_enabled=true" in text, (
        "Parent's expression_state_publish_loop must be gated off "
        "under l0_rust_enabled=true to preserve G21 single-writer")
    assert "ownership transferred " in text
    assert "expression_worker per §4.B" in text


# ─── 7. Catalyst-site #8 closure marker ──────────────────────────────


EXPRESSION_WORKER_PATH = (
    REPO_ROOT / "titan_hcl" / "modules" / "expression_worker.py")


def test_expression_worker_publishes_strong_composition_catalyst():
    """D8-3 catalyst-producer site #8 closure — expression_worker
    publishes `SOCIAL_CATALYST(type="strong_composition", ...)` per the
    "D8 RETIREMENT PREREQUISITE" block at the top of
    rFP_titan_hcl_l2_separation_strategy.md.
    """
    text = EXPRESSION_WORKER_PATH.read_text()
    assert 'bus.SOCIAL_CATALYST' in text
    assert '"type": "strong_composition"' in text
    assert "D8-3 catalyst-site #8" in text


def test_cognitive_worker_dispatches_ns_reward_to_record_outcome():
    """cognitive_worker's NS_REWARD subscriber must dispatch payload to
    neural_nervous_system.record_outcome — the cross-process replacement
    for the in-proc record_outcome call that was inside Block 8."""
    text = COGNITIVE_WORKER_PATH.read_text()
    assert "msg_type == bus.NS_REWARD" in text
    assert "nns.record_outcome(" in text
