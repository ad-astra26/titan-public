"""rFP β Stage 4 — Regression CI tests.

Lock-in tests for the Phase 2 + Phase 3 foundation. These tests would
have caught the original collapse-to-zero bug and several other failure
modes that the rFP β diagnoses. Run as part of pre-commit / CI gate.

Coverage:
  § 4e (persistence regression):
    - 11 program files all persisted after warmup
    - record_outcome(program=X) routes to correct buffer
    - Audit log entries are well-formed JSON

  § 4f (learning verification):
    - Soft-fire propagation moves not-fired program toward threshold
    - Stratified sampling prevents zero-target collapse
    - Per-program reward routing isolates updates correctly
    - Z-score normalization equalizes magnitudes
"""
import json
import os
import tempfile

import numpy as np
import pytest


def _make_test_config(n_programs: int = 3) -> dict:
    """Minimal config for NeuralNervousSystem instantiation."""
    progs = {}
    for i, name in enumerate(["TEST_A", "TEST_B", "TEST_C"][:n_programs]):
        progs[name] = {
            "enabled": True,
            "input_features": "standard",  # 55D
            "fire_threshold": 0.3,
            "buffer_max": 200,
        }
    return {
        "warmup_steps": 50,
        "train_every_n": 5,
        "batch_size": 8,
        "save_every_n": 100,
        "programs": progs,
    }


# ── § 4e PERSISTENCE REGRESSION TESTS ─────────────────────────────


class TestPersistenceRegression:
    def test_all_programs_persisted_after_record(self, tmp_path):
        """Every enabled program must have weights + buffer files after save_all."""
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        nns = NeuralNervousSystem(config=_make_test_config(3),
                                   data_dir=str(tmp_path), vm_nervous_system=None)
        # Force a save
        nns.save_all()
        for prog in ["test_a", "test_b", "test_c"]:
            assert (tmp_path / f"{prog}_weights.json").exists()
            assert (tmp_path / f"{prog}_buffer.json").exists()
        assert (tmp_path / "training_state.json").exists()

    def test_audit_log_entries_well_formed(self, tmp_path):
        """Every record_outcome event writes a JSON-parseable audit line."""
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        nns = NeuralNervousSystem(config=_make_test_config(2),
                                   data_dir=str(tmp_path), vm_nervous_system=None)
        # Add a transition that fires so reward update has a target
        nns.buffers["TEST_A"].add(
            observation=[0.5] * 55, urgency=0.4, vm_baseline=0.3, fired=True)
        nns.record_outcome(reward=0.7, program="TEST_A", source="test")
        log_path = tmp_path / "reward_log.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        for required in ["ts", "program", "reward_raw", "reward_z", "source", "k", "fired"]:
            assert required in entry, f"Missing field: {required}"
        assert entry["program"] == "TEST_A"
        assert entry["source"] == "test"


# ── § 4f LEARNING VERIFICATION TESTS ──────────────────────────────


class TestSoftFirePropagation:
    def test_soft_fire_with_meaningful_urgency_applies_reward(self):
        """Not-fired transition with urgency near threshold gets soft reward."""
        from titan_plugin.logic.neural_reflex_net import NervousTransitionBuffer
        buf = NervousTransitionBuffer(max_size=10)
        # Simulate a transition that DIDN'T fire but had high urgency (close to threshold)
        buf.add(observation=[0.5] * 10, urgency=0.25, vm_baseline=0.2, fired=False)
        # Threshold = 0.3 → urgency_ratio = 0.25/0.3 = 0.833
        # soft_reward = 0.6 × 0.833 × 0.5 = 0.25
        applied = buf.update_soft_reward(reward=0.6, fire_threshold=0.3, soft_factor=0.5)
        assert applied is True
        assert buf._rewards[-1] != 0.0
        assert abs(buf._rewards[-1] - 0.25) < 0.01

    def test_soft_fire_below_threshold_skipped(self):
        """Urgency ratio < 0.1 (true restraint) → no soft reward."""
        from titan_plugin.logic.neural_reflex_net import NervousTransitionBuffer
        buf = NervousTransitionBuffer(max_size=10)
        buf.add(observation=[0.5] * 10, urgency=0.01, vm_baseline=0.0, fired=False)
        applied = buf.update_soft_reward(reward=0.7, fire_threshold=0.3)
        assert applied is False
        assert buf._rewards[-1] == 0.0


class TestStratifiedSampling:
    def test_stratified_returns_balanced_classes(self):
        """When both classes have samples, sample_stratified returns ~50/50."""
        from titan_plugin.logic.neural_reflex_net import NervousTransitionBuffer
        buf = NervousTransitionBuffer(max_size=200)
        # 90% not-fired, 10% fired (extreme imbalance)
        for i in range(180):
            buf.add(observation=[0.5] * 10, urgency=0.1, vm_baseline=0.0, fired=False)
        for i in range(20):
            buf.add(observation=[0.5] * 10, urgency=0.5, vm_baseline=0.4, fired=True)
        # Sample batch of 16
        obs, urg, vm, rew, fired = buf.sample_stratified(16)
        n_fired = sum(fired)
        # Should be 8 fired + 8 not-fired (half/half), allowing some leeway if
        # one class is smaller than half (here both have ≥ 8 so should be exact)
        assert n_fired == 8, f"Expected 8 fired in stratified batch, got {n_fired}"

    def test_stratified_falls_back_when_one_class_empty(self):
        """If only one class has samples, fall back to uniform."""
        from titan_plugin.logic.neural_reflex_net import NervousTransitionBuffer
        buf = NervousTransitionBuffer(max_size=50)
        # Only not-fired samples
        for i in range(30):
            buf.add(observation=[0.5] * 10, urgency=0.1, vm_baseline=0.0, fired=False)
        obs, urg, vm, rew, fired = buf.sample_stratified(8)
        # All should be not-fired (no fired available) — no crash
        assert sum(fired) == 0
        assert obs.shape == (8, 10)


class TestPerProgramRouting:
    def test_record_outcome_program_kwarg_isolates_update(self, tmp_path):
        """record_outcome(reward, program=X) updates ONLY X's buffer."""
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        nns = NeuralNervousSystem(config=_make_test_config(3),
                                   data_dir=str(tmp_path), vm_nervous_system=None)
        # Add fired transition to all 3 buffers
        for name in ["TEST_A", "TEST_B", "TEST_C"]:
            nns.buffers[name].add(
                observation=[0.5] * 55, urgency=0.4, vm_baseline=0.3, fired=True)
        # Reward only TEST_A
        nns.record_outcome(reward=0.8, program="TEST_A", source="test")
        # TEST_A's last reward should be non-zero (z-normalized but non-zero)
        assert nns.buffers["TEST_A"]._rewards[-1] != 0.0
        # TEST_B and TEST_C should remain at 0
        assert nns.buffers["TEST_B"]._rewards[-1] == 0.0
        assert nns.buffers["TEST_C"]._rewards[-1] == 0.0

    def test_record_outcome_firehose_updates_all_fired(self, tmp_path):
        """record_outcome(reward) WITHOUT program kwarg = legacy firehose."""
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        nns = NeuralNervousSystem(config=_make_test_config(3),
                                   data_dir=str(tmp_path), vm_nervous_system=None)
        for name in ["TEST_A", "TEST_B", "TEST_C"]:
            nns.buffers[name].add(
                observation=[0.5] * 55, urgency=0.4, vm_baseline=0.3, fired=True)
        nns.record_outcome(0.6)  # firehose mode
        # All 3 should have non-zero last reward
        for name in ["TEST_A", "TEST_B", "TEST_C"]:
            assert nns.buffers[name]._rewards[-1] != 0.0, (
                f"{name} reward not updated in firehose mode")

    def test_unknown_program_warns_no_crash(self, tmp_path):
        """record_outcome(program="NONEXISTENT") logs WARNING, doesn't crash."""
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        nns = NeuralNervousSystem(config=_make_test_config(2),
                                   data_dir=str(tmp_path), vm_nervous_system=None)
        # Add a fired transition to each known program so we can verify they're untouched
        for name in ["TEST_A", "TEST_B"]:
            nns.buffers[name].add(observation=[0.5] * 55, urgency=0.4,
                                   vm_baseline=0.3, fired=True)
        # Should not raise
        nns.record_outcome(reward=0.5, program="NONEXISTENT_PROGRAM")
        # Known programs untouched
        for name in ["TEST_A", "TEST_B"]:
            assert nns.buffers[name]._rewards[-1] == 0.0


class TestEligibilityTraces:
    def test_recent_rewards_with_decay(self):
        """update_recent_rewards applies exponential decay to last K fires."""
        from titan_plugin.logic.neural_reflex_net import NervousTransitionBuffer
        buf = NervousTransitionBuffer(max_size=20)
        # Add 5 fired transitions interleaved with not-fired
        for i in range(10):
            buf.add(observation=[0.5] * 10, urgency=0.5, vm_baseline=0.4,
                    fired=(i % 2 == 0))
        # Now we have fired at indices 0, 2, 4, 6, 8 (5 fires)
        # Apply reward=1.0 to last K=3 fires with decay=0.5
        # Should write: idx 8 → 1.0, idx 6 → 0.5, idx 4 → 0.25
        n = buf.update_recent_rewards(reward=1.0, k=3, decay=0.5)
        assert n == 3
        assert abs(buf._rewards[8] - 1.0) < 0.001
        assert abs(buf._rewards[6] - 0.5) < 0.001
        assert abs(buf._rewards[4] - 0.25) < 0.001
        # Older fires (idx 0, 2) untouched
        assert buf._rewards[0] == 0.0
        assert buf._rewards[2] == 0.0

    def test_k_larger_than_available_fires(self):
        """K=10 with only 2 fires available → updates 2, returns 2."""
        from titan_plugin.logic.neural_reflex_net import NervousTransitionBuffer
        buf = NervousTransitionBuffer(max_size=20)
        for i in range(5):
            buf.add(observation=[0.5] * 10, urgency=0.5, vm_baseline=0.4,
                    fired=(i < 2))  # Only first 2 fired
        n = buf.update_recent_rewards(reward=0.8, k=10, decay=0.5)
        assert n == 2


class TestZScoreNormalization:
    def test_z_score_stabilizes_to_zero_for_constant_reward(self, tmp_path):
        """Constant reward stream → z-score converges toward 0 (zero variance)."""
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        nns = NeuralNervousSystem(config=_make_test_config(1),
                                   data_dir=str(tmp_path), vm_nervous_system=None)
        # Feed 100 constant rewards
        for _ in range(100):
            z = nns._z_normalize_reward("TEST_A", 0.5)
        # Final z-score should be near 0 (mean ≈ 0.5, deviation = 0)
        assert abs(z) < 0.5  # Close to zero (some lag from EMA)

    def test_z_score_amplifies_outlier(self, tmp_path):
        """Outlier reward against established baseline → high |z|."""
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        nns = NeuralNervousSystem(config=_make_test_config(1),
                                   data_dir=str(tmp_path), vm_nervous_system=None)
        # Establish baseline of 0.3 ± small noise
        for _ in range(200):
            nns._z_normalize_reward("TEST_A", 0.3 + 0.01 * np.random.randn())
        # Now an outlier
        z_outlier = nns._z_normalize_reward("TEST_A", 1.0)
        assert z_outlier > 1.5, f"Outlier should produce high z, got {z_outlier}"
        assert z_outlier <= 3.0, "Should be clipped at +3"
