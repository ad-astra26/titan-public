"""D-SPEC-66 v1.11.0 / PLAN_catalyst_producer_migrations_d8_3.md tests.

Per rFP closure criterion #3 — verify each of the 6 migrated catalyst
producers publishes the correct payload at the correct trigger condition.

The 6 migrations are:
  #1 onchain_anchor   → agency_worker (post-memo_inscribe success)
  #2 eureka(_spirit)  → cognitive_worker (meta_engine.tick eureka)
  #3 vulnerability    → cognitive_worker (meta_engine.tick BREAK)
  #4 emotion_shift    → social_worker (ShmReaderBank tick on transition)
  #6 maker_force      → social_worker (X_FORCE_POST handler)
  #7 kin_resonance    → cognitive_worker (KIN_SIGNAL handler)

Plus:
  - detect_emotion_from_levels — pure helper extracted from
    NeuromodulatorSystem._detect_emotion (byte-identical math).
  - _emit_x_catalyst legacy spirit_worker helper is now NO-OP
    (deletion deferred to D8-3 spirit_worker.py retirement).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────
# Pure helper — detect_emotion_from_levels
# Byte-identical extraction from NeuromodulatorSystem._detect_emotion.
# ─────────────────────────────────────────────────────────────────────


class TestDetectEmotionFromLevels:
    def test_joy_template_returns_joy(self):
        """Levels matching `joy` template return 'joy'."""
        from titan_hcl.logic.neuromodulator import (
            detect_emotion_from_levels)
        levels = {"DA": 0.8, "5HT": 0.7, "NE": 0.3, "ACh": 0.5,
                  "Endorphin": 0.8, "GABA": 0.3}
        emotion, conf = detect_emotion_from_levels(levels)
        assert emotion == "joy"
        assert 0.9 < conf <= 1.0  # cosine sim with self ~= 1.0

    def test_fear_template_returns_fear(self):
        from titan_hcl.logic.neuromodulator import (
            detect_emotion_from_levels)
        levels = {"DA": 0.2, "5HT": 0.2, "NE": 0.9, "ACh": 0.7,
                  "Endorphin": 0.2, "GABA": 0.1}
        emotion, _ = detect_emotion_from_levels(levels)
        assert emotion == "fear"

    def test_neutral_levels_return_some_emotion(self):
        """All 0.5 levels — pick whichever template is closest."""
        from titan_hcl.logic.neuromodulator import (
            detect_emotion_from_levels)
        levels = {"DA": 0.5, "5HT": 0.5, "NE": 0.5, "ACh": 0.5,
                  "Endorphin": 0.5, "GABA": 0.5}
        emotion, conf = detect_emotion_from_levels(levels)
        assert emotion in {"joy", "peace", "curiosity", "fear", "love",
                           "anger", "sadness", "wonder", "flow", "calm",
                           "neutral"}
        assert -1.0 <= conf <= 1.0

    def test_missing_keys_default_to_05(self):
        """Missing keys default to 0.5 per docstring contract."""
        from titan_hcl.logic.neuromodulator import (
            detect_emotion_from_levels)
        emotion, _ = detect_emotion_from_levels({})  # all missing
        # Should still return something valid
        assert isinstance(emotion, str)
        assert len(emotion) > 0

    def test_byte_identical_to_in_class_method(self):
        """Extracted helper produces byte-identical results to the
        in-class _detect_emotion method (refactor preserves behavior).
        """
        from titan_hcl.logic.neuromodulator import (
            NeuromodulatorSystem, detect_emotion_from_levels)
        sys = NeuromodulatorSystem()
        # Set known levels
        sys.modulators["DA"].level = 0.8
        sys.modulators["5HT"].level = 0.7
        sys.modulators["NE"].level = 0.3
        sys.modulators["ACh"].level = 0.5
        sys.modulators["Endorphin"].level = 0.8
        sys.modulators["GABA"].level = 0.3
        emo_class, conf_class = sys._detect_emotion()
        levels = {name: mod.level for name, mod in sys.modulators.items()}
        emo_helper, conf_helper = detect_emotion_from_levels(levels)
        assert emo_class == emo_helper
        assert conf_class == conf_helper


# ─────────────────────────────────────────────────────────────────────
# Site #1 onchain_anchor — agency_worker._maybe_emit_onchain_anchor_catalyst
# ─────────────────────────────────────────────────────────────────────


class TestSite1OnchainAnchor:
    def test_emits_catalyst_on_memo_inscribe_success(self):
        from titan_hcl.modules.agency_worker import (
            _maybe_emit_onchain_anchor_catalyst)
        from titan_hcl import bus
        sq = MagicMock()
        action_result = {
            "helper": "memo_inscribe",
            "success": True,
            "result": "Inscribed epoch 42 on Solana (tx=abc123)",
            "enrichment_data": {"balance": 1.234},
            "action_id": 100, "impulse_id": 200,
        }
        _maybe_emit_onchain_anchor_catalyst(sq, "agency_worker", action_result)
        assert sq.put.call_count == 1
        msg = sq.put.call_args[0][0]
        assert msg["type"] == bus.SOCIAL_CATALYST
        assert msg["dst"] == "social"
        assert msg["payload"]["type"] == "onchain_anchor"
        assert msg["payload"]["significance"] == 0.4
        assert msg["payload"]["data"]["balance"] == 1.234

    def test_no_emit_on_failure(self):
        from titan_hcl.modules.agency_worker import (
            _maybe_emit_onchain_anchor_catalyst)
        sq = MagicMock()
        _maybe_emit_onchain_anchor_catalyst(sq, "agency_worker", {
            "helper": "memo_inscribe", "success": False,
        })
        sq.put.assert_not_called()

    def test_no_emit_on_wrong_helper(self):
        from titan_hcl.modules.agency_worker import (
            _maybe_emit_onchain_anchor_catalyst)
        sq = MagicMock()
        _maybe_emit_onchain_anchor_catalyst(sq, "agency_worker", {
            "helper": "art_generate", "success": True,
        })
        sq.put.assert_not_called()

    def test_no_emit_on_none_action_result(self):
        from titan_hcl.modules.agency_worker import (
            _maybe_emit_onchain_anchor_catalyst)
        sq = MagicMock()
        _maybe_emit_onchain_anchor_catalyst(sq, "agency_worker", None)
        sq.put.assert_not_called()


# ─────────────────────────────────────────────────────────────────────
# Legacy spirit_worker _emit_x_catalyst — verify NO-OP behavior
# ─────────────────────────────────────────────────────────────────────


class TestLegacyEmitXCatalystNoOp:
    def test_spirit_worker_module_deleted(self):
        """D-SPEC-116 (2026-05-22): the interim D8-3 NO-OP guard is superseded
        — spirit_worker.py (which hosted the legacy _emit_x_catalyst helper +
        callsites) was fully DELETED. The ultimate invariant: module is gone.
        """
        import importlib.util
        assert importlib.util.find_spec(
            "titan_hcl.modules.spirit_worker") is None, \
            "spirit_worker.py must stay deleted (D-SPEC-116)"


# ─────────────────────────────────────────────────────────────────────
# SPEC contract checks — §8.7 SOCIAL_CATALYST row + §21 D-SPEC-66
# ─────────────────────────────────────────────────────────────────────


class TestSpecV1100Contract:
    def test_spec_changelog_has_v1100_row(self):
        from pathlib import Path
        spec_path = (Path(__file__).parent.parent
                     / "titan-docs" / "SPEC_titan_architecture.md")
        src = spec_path.read_text(encoding="utf-8")
        assert "v1.11.0 (MINOR)" in src, "SPEC v1.11.0 Changelog row missing"
        assert "D-SPEC-66" in src, "D-SPEC-66 entry missing"
        assert ("**6 catalyst-producer migrations close D8-3 "
                "prerequisite**") in src

    def test_spec_frontmatter_version_1100_or_later(self):
        """Frontmatter must be at v1.11.x (or higher) — v1.11.0 was the
        D-SPEC-66 introduction MINOR bump; v1.11.1 PATCH adds producer-
        field correction per PLAN Q4. (Renumbered from v1.10.x at merge
        time per parallel-session v1.9.5 D-SPEC-64 + v1.9.6 D-SPEC-65
        collisions.)"""
        from pathlib import Path
        spec_path = (Path(__file__).parent.parent
                     / "titan-docs" / "SPEC_titan_architecture.md")
        src = spec_path.read_text(encoding="utf-8")
        # Match spec_version: 1.11.* or higher (PATCH-bump tolerant).
        import re
        m = re.search(
            r"^spec_version:\s*1\.(\d+)\.(\d+)\s*$", src, re.MULTILINE)
        assert m is not None, "SPEC frontmatter spec_version not parseable"
        minor = int(m.group(1))
        assert minor >= 11, (
            f"SPEC frontmatter spec_version 1.{minor}.* below required "
            f"v1.11.x for D-SPEC-66 — closure regression")

    def test_spec_section_87_has_social_catalyst_row(self):
        """SPEC §8.7 must include the NEW SOCIAL_CATALYST table row
        (was previously narrative-only)."""
        from pathlib import Path
        spec_path = (Path(__file__).parent.parent
                     / "titan-docs" / "SPEC_titan_architecture.md")
        src = spec_path.read_text(encoding="utf-8")
        assert "| `SOCIAL_CATALYST` | P2 | none |" in src, (
            "§8.7 SOCIAL_CATALYST table row missing")
        # Producer set must include all 5 producers
        for producer in ("backup_worker", "cognitive_worker",
                         "social_worker", "meditation_worker",
                         "expression_worker"):
            assert producer in src

    def test_plan_doc_exists(self):
        """Post-D8-3 (2026-05-16): PLAN was moved to titan-docs/finished/
        after verification all §3 sequencing steps SHIPPED."""
        from pathlib import Path
        plan_path = (Path(__file__).parent.parent / "titan-docs"
                     / "finished" / "PLAN_catalyst_producer_migrations_d8_3.md")
        assert plan_path.exists(), (
            "PLAN_catalyst_producer_migrations_d8_3.md missing from finished/")
        src = plan_path.read_text(encoding="utf-8")
        for site in ("§1.1", "§1.2", "§1.3", "§1.4", "§1.5", "§1.6"):
            assert site in src


# ─────────────────────────────────────────────────────────────────────
# Source-contract assertions for the 5 cross-worker migrations
# ─────────────────────────────────────────────────────────────────────


class TestMigrationSourceContracts:
    """Verify the 6 migration insertion sites exist in target workers.
    Source-level check — cheap to run + catches accidental reverts.
    """

    def test_cognitive_worker_emits_meta_eureka_plus_catalyst(self):
        from pathlib import Path
        cw_path = (Path(__file__).parent.parent / "titan_hcl"
                   / "modules" / "cognitive_worker.py")
        src = cw_path.read_text(encoding="utf-8")
        # #2 eureka — META_EUREKA publish + SOCIAL_CATALYST
        assert "bus.META_EUREKA" in src
        assert "_send_msg(send_queue, bus.META_EUREKA" in src
        assert "\"type\": \"eureka_spirit\"" in src
        # #3 vulnerability — BREAK primitive check + SOCIAL_CATALYST
        assert "\"primitive\") == \"BREAK\"" in src
        assert "\"type\": \"vulnerability\"" in src
        # #7 kin_resonance — bus.KIN_SIGNAL subscribed + handler
        assert "bus.KIN_SIGNAL" in src
        assert "\"type\": \"kin_resonance\"" in src

    def test_cognitive_worker_subscribes_kin_signal(self):
        from pathlib import Path
        cw_path = (Path(__file__).parent.parent / "titan_hcl"
                   / "modules" / "cognitive_worker.py")
        src = cw_path.read_text(encoding="utf-8")
        # _COGNITIVE_WORKER_SUBSCRIBE_TOPICS must include KIN_SIGNAL
        # (D-SPEC-66 v1.11.0 added).
        topic_block_start = src.find(
            "_COGNITIVE_WORKER_SUBSCRIBE_TOPICS")
        assert topic_block_start > 0
        # The closing bracket of the list is the next "\n]\n" after
        # the start. Slice the full subscription block.
        end_marker = src.find("\n]\n", topic_block_start)
        assert end_marker > topic_block_start
        topic_block = src[topic_block_start:end_marker]
        assert "bus.KIN_SIGNAL" in topic_block, (
            "KIN_SIGNAL subscription missing from _COGNITIVE_WORKER_"
            "SUBSCRIBE_TOPICS — D-SPEC-66 #7 closure regression")

    def test_social_worker_subscribes_x_force_post(self):
        from pathlib import Path
        sw_path = (Path(__file__).parent.parent / "titan_hcl"
                   / "modules" / "social_worker.py")
        src = sw_path.read_text(encoding="utf-8")
        topic_block_start = src.find(
            "_SOCIAL_WORKER_SUBSCRIBE_TOPICS")
        assert topic_block_start > 0
        topic_block = src[topic_block_start:topic_block_start + 2000]
        assert "bus.X_FORCE_POST" in topic_block, (
            "X_FORCE_POST subscription missing from _SOCIAL_WORKER_"
            "SUBSCRIBE_TOPICS — D-SPEC-66 #6 closure regression")

    def test_social_worker_handles_x_force_post(self):
        from pathlib import Path
        sw_path = (Path(__file__).parent.parent / "titan_hcl"
                   / "modules" / "social_worker.py")
        src = sw_path.read_text(encoding="utf-8")
        assert "msg_type == bus.X_FORCE_POST" in src, (
            "X_FORCE_POST dispatch handler missing")
        assert ("\"type\": _fp_type" in src
                or "_fp_type," in src), (
            "X_FORCE_POST handler doesn't forward catalyst_type")

    def test_social_worker_has_emotion_shift_tick(self):
        from pathlib import Path
        sw_path = (Path(__file__).parent.parent / "titan_hcl"
                   / "modules" / "social_worker.py")
        src = sw_path.read_text(encoding="utf-8")
        assert "detect_emotion_from_levels" in src
        assert "_prev_emotion" in src
        assert "\"type\": \"emotion_shift\"" in src

    def test_agency_worker_has_onchain_anchor_helper(self):
        from pathlib import Path
        aw_path = (Path(__file__).parent.parent / "titan_hcl"
                   / "modules" / "agency_worker.py")
        src = aw_path.read_text(encoding="utf-8")
        assert "_maybe_emit_onchain_anchor_catalyst" in src
        assert "\"type\": \"onchain_anchor\"" in src

    def test_api_maker_dst_is_social(self):
        """api/maker.py X_FORCE_POST publish must target 'social' dst
        (was 'spirit' pre-D-SPEC-66; dead handler since fleet-wide
        Phase C cascade 2026-05-14)."""
        from pathlib import Path
        m_path = (Path(__file__).parent.parent / "titan_hcl"
                  / "api" / "maker.py")
        src = m_path.read_text(encoding="utf-8")
        # maker.py imports the constant directly (`from titan_hcl.bus import
        # make_msg, X_FORCE_POST`) and publishes it bare, so match the publish
        # call site (`X_FORCE_POST, "maker_api", ...`), not a `bus.` prefix.
        idx = src.find('X_FORCE_POST, "maker_api"')
        assert idx > 0, "X_FORCE_POST publish call not found in api/maker.py"
        next_lines = src[idx:idx + 500]
        assert "\"social\"" in next_lines or "'social'" in next_lines
        assert ", \"spirit\", {" not in next_lines, (
            "api/maker.py X_FORCE_POST still targets 'spirit' — "
            "D-SPEC-66 #6 dst-flip regression")
