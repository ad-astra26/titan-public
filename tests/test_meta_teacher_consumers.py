"""Consumer-wiring tests for Meta-Teacher feedback paths.

Covers rFP §10 checklist:
  - MetaCGNConsumer.handle_teacher_grounding applies Beta posterior nudge
  - chain_iql.apply_external_reward receives teacher reward correctly
  - Regression: enabled=false path produces zero META_TEACHER_* traffic
    (covered in test_meta_teacher_worker.py::TestDisabledPath)
"""
import numpy as np
import pytest

from titan_plugin.logic.meta_cgn import MetaCGNConsumer, BETA_PARAM_FLOOR
from titan_plugin.logic.chain_iql import ChainIQL


@pytest.fixture
def meta_cgn(tmp_path):
    # Minimal config for a testable MetaCGNConsumer
    mcgn = MetaCGNConsumer(
        send_queue=None,
        titan_id="TEST",
        save_dir=str(tmp_path / "meta_cgn"),
        shm_path=str(tmp_path / "cgn_weights.bin"),
    )
    return mcgn


class TestHandleTeacherGrounding:
    def test_high_quality_boosts_alpha(self, meta_cgn):
        prim_id = "FORMULATE"
        p = meta_cgn._primitives[prim_id]
        alpha_before = p.alpha
        beta_before = p.beta
        meta_cgn.handle_teacher_grounding({
            "chain_id": 1,
            "primitive_id": prim_id,
            "label_quality": 0.9,
            "ctx_fingerprint": "social|WONDER|none",
            "grounding_weight": 0.15,
        })
        assert p.alpha > alpha_before, "α should grow on high-quality label"
        assert p.beta >= beta_before  # with label=0.9, β gets 0.15*0.1=0.015
        assert p.alpha - alpha_before == pytest.approx(0.135, abs=1e-3)

    def test_low_quality_boosts_beta(self, meta_cgn):
        prim_id = "RECALL"
        p = meta_cgn._primitives[prim_id]
        alpha_before = p.alpha
        beta_before = p.beta
        meta_cgn.handle_teacher_grounding({
            "chain_id": 1,
            "primitive_id": prim_id,
            "label_quality": 0.1,
            "grounding_weight": 0.15,
        })
        assert p.beta > beta_before
        assert p.beta - beta_before == pytest.approx(0.135, abs=1e-3)

    def test_unknown_primitive_no_crash(self, meta_cgn):
        # Should silently no-op
        meta_cgn.handle_teacher_grounding({
            "chain_id": 1,
            "primitive_id": "NOT_A_REAL_PRIMITIVE",
            "label_quality": 0.5,
            "grounding_weight": 0.15,
        })

    def test_malformed_payload_no_crash(self, meta_cgn):
        # Missing label_quality defaults to 0.5 via .get — valid update
        meta_cgn.handle_teacher_grounding({
            "chain_id": 1,
            "primitive_id": "FORMULATE",
            "grounding_weight": 0.15,
        })

    def test_zero_weight_noop(self, meta_cgn):
        prim_id = "FORMULATE"
        p = meta_cgn._primitives[prim_id]
        alpha_before = p.alpha
        meta_cgn.handle_teacher_grounding({
            "chain_id": 1,
            "primitive_id": prim_id,
            "label_quality": 0.9,
            "grounding_weight": 0.0,
        })
        assert p.alpha == alpha_before

    def test_grounding_counter_increments(self, meta_cgn):
        for _ in range(3):
            meta_cgn.handle_teacher_grounding({
                "chain_id": 1,
                "primitive_id": "FORMULATE",
                "label_quality": 0.7,
                "grounding_weight": 0.1,
            })
        assert meta_cgn._teacher_groundings_applied == 3


class TestChainIQLTeacherReward:
    """Verify the chain_iql.apply_external_reward integration used by
    spirit_worker's META_TEACHER_FEEDBACK handler."""

    def test_apply_external_reward_blends_task_success(self, tmp_path):
        iql = ChainIQL(dna={}, save_dir=str(tmp_path))
        # Seed one chain outcome in the buffer — chain must be strings
        task_emb = np.zeros(iql._task_dim, dtype=np.float32)
        iql.record_chain_outcome(
            task_emb=task_emb,
            chain=["FORMULATE.define", "RECALL.chain_archive", "EVALUATE.check"],
            task_success=0.4,
            primitives=["FORMULATE", "RECALL", "EVALUATE"],
            domain="social", chain_id=123,
        )
        assert iql.buffer[-1]["task_success"] == pytest.approx(0.4)

        # Teacher says this chain was high-quality (0.9)
        ok = iql.apply_external_reward(
            chain_id=123, external_reward=0.9, alpha=0.05)
        assert ok is True
        # Blended: 0.4 * 0.95 + 0.9 * 0.05 = 0.38 + 0.045 = 0.425
        assert iql.buffer[-1]["task_success"] == pytest.approx(0.425)
        assert iql.buffer[-1].get("external_applied") is True

    def test_apply_external_reward_alpha_clamped(self, tmp_path):
        iql = ChainIQL(dna={}, save_dir=str(tmp_path))
        task_emb = np.zeros(iql._task_dim, dtype=np.float32)
        iql.record_chain_outcome(
            task_emb=task_emb, chain=["FORMULATE.define"], task_success=0.5,
            primitives=["FORMULATE"], domain="general", chain_id=1,
        )
        # Try out-of-range alpha — should clamp to [0, 1]
        ok = iql.apply_external_reward(
            chain_id=1, external_reward=1.0, alpha=5.0)
        assert ok
        # α clamps to 1.0 → fully replaces task_success with external
        assert iql.buffer[-1]["task_success"] == pytest.approx(1.0)

    def test_missing_chain_returns_false(self, tmp_path):
        iql = ChainIQL(dna={}, save_dir=str(tmp_path))
        ok = iql.apply_external_reward(
            chain_id=99999, external_reward=0.8, alpha=0.1)
        assert ok is False
        # Late-drop counter should increment
        assert iql._external_reward_late_drops >= 1
