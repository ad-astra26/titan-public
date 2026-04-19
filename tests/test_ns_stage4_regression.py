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


# ── 2026-04-19 _compute_targets refactor (autonomous-collapse fix) ──
# Locks in the residual-learning formula that replaced the discrete-case
# classifier. These tests would have caught the sigmoid(-10) collapse.


class TestResidualTargetFormula:
    def _ns(self, tmp_path):
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        return NeuralNervousSystem(
            config=_make_test_config(1),
            data_dir=str(tmp_path), vm_nervous_system=None)

    def test_zero_reward_preserves_current_urgency(self, tmp_path):
        """r=0 → target = urgency (no collapse pressure when no signal)."""
        nns = self._ns(tmp_path)
        urgencies = np.array([0.3, 0.5, 0.7])
        rewards = np.array([0.0, 0.0, 0.0])
        fired = np.array([False, True, False])
        targets = nns._compute_targets(
            vm_baselines=np.zeros(3), rewards=rewards,
            fired=fired, urgencies=urgencies,
            supervision_weight=0.0)  # autonomous phase
        np.testing.assert_allclose(targets, urgencies, atol=1e-6)

    def test_positive_reward_bumps_target_up(self, tmp_path):
        """Positive reward + fired → target climbs above current urgency."""
        nns = self._ns(tmp_path)
        urgencies = np.array([0.3])
        rewards = np.array([1.0])
        fired = np.array([True])
        targets = nns._compute_targets(
            vm_baselines=np.zeros(1), rewards=rewards,
            fired=fired, urgencies=urgencies, supervision_weight=0.0)
        assert targets[0] > urgencies[0], (
            f"Positive reward didn't bump target: {urgencies[0]} → {targets[0]}")
        # Bounded by fire_gain via tanh
        assert targets[0] <= urgencies[0] + nns._nn_target_fire_gain + 1e-6

    def test_negative_reward_pushes_target_down(self, tmp_path):
        """Negative reward + fired → target drops below current urgency
        (the prior discrete formula silently lost negative-reward signal)."""
        nns = self._ns(tmp_path)
        urgencies = np.array([0.5])
        rewards = np.array([-1.0])
        fired = np.array([True])
        targets = nns._compute_targets(
            vm_baselines=np.zeros(1), rewards=rewards,
            fired=fired, urgencies=urgencies, supervision_weight=0.0)
        assert targets[0] < urgencies[0], (
            f"Negative reward didn't discourage: {urgencies[0]} → {targets[0]}")

    def test_fired_has_stronger_gain_than_not_fired(self, tmp_path):
        """Same reward magnitude: fired transitions shift more than not-fired
        (fire means the NN's decision caused the outcome)."""
        nns = self._ns(tmp_path)
        urgencies = np.array([0.3, 0.3])
        rewards = np.array([1.0, 1.0])
        fired = np.array([True, False])
        targets = nns._compute_targets(
            vm_baselines=np.zeros(2), rewards=rewards,
            fired=fired, urgencies=urgencies, supervision_weight=0.0)
        assert targets[0] > targets[1], (
            "Fired gain not stronger than not-fired gain")

    def test_sparse_reward_regression_no_collapse(self, tmp_path):
        """Regression: 98%+ reward=0 transitions must NOT drive targets to 0.
        Prior discrete formula: all r=0 → target=0 → NN collapse to
        sigmoid(-10) = 4.5e-05. Residual formula keeps target = urgency."""
        nns = self._ns(tmp_path)
        n = 100
        urgencies = np.full(n, 0.4)  # Current healthy urgency
        rewards = np.zeros(n)
        rewards[0] = 0.8  # Single sparse positive reward
        fired = np.zeros(n, dtype=bool)
        fired[0] = True
        targets = nns._compute_targets(
            vm_baselines=np.zeros(n), rewards=rewards,
            fired=fired, urgencies=urgencies, supervision_weight=0.0)
        # 99 non-reward transitions preserve urgency (no collapse toward 0)
        np.testing.assert_allclose(targets[1:], 0.4, atol=1e-6)
        # The 1 positive signal raises its target
        assert targets[0] > 0.4

    def test_tanh_bounds_extreme_rewards(self, tmp_path):
        """Very large |r| doesn't cause target spike beyond ±gain —
        tanh saturates smoothly."""
        nns = self._ns(tmp_path)
        urgencies = np.array([0.5])
        rewards = np.array([100.0])  # Extreme outlier
        fired = np.array([True])
        targets = nns._compute_targets(
            vm_baselines=np.zeros(1), rewards=rewards,
            fired=fired, urgencies=urgencies, supervision_weight=0.0)
        # Should cap at urgency + fire_gain (tanh(100) ≈ 1.0)
        expected_cap = min(1.0, 0.5 + nns._nn_target_fire_gain)
        assert abs(targets[0] - expected_cap) < 1e-6

    def test_warmup_phase_still_uses_vm_supervision(self, tmp_path):
        """During warmup (supervision_weight=1.0), VM baseline dominates —
        outcome component is fully suppressed."""
        nns = self._ns(tmp_path)
        urgencies = np.array([0.1])  # Low urgency
        rewards = np.array([1.0])  # High reward
        fired = np.array([True])
        vm_baselines = np.array([0.7])  # VM says 0.7
        targets = nns._compute_targets(
            vm_baselines=vm_baselines, rewards=rewards, fired=fired,
            urgencies=urgencies, supervision_weight=1.0)
        np.testing.assert_allclose(targets, vm_baselines)


# ── 2026-04-19 Sigmoid-trap liberation (residual-learning rescue) ──
# Residual target formula alone can't escape sigmoid saturation because
# σ'(−10) ≈ 4.5e-5 makes gradients vanishingly small. Liberation resets
# the output layer when saved b3 indicates saturation.


class TestSigmoidTrapLiberation:
    def _ns(self, tmp_path):
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        return NeuralNervousSystem(
            config=_make_test_config(1),
            data_dir=str(tmp_path), vm_nervous_system=None)

    def test_liberation_resets_saturated_output_layer(self, tmp_path):
        """Program with b3 < -5 gets w3 + b3 reset; escape sigmoid trap."""
        nns = self._ns(tmp_path)
        net = nns.programs["TEST_A"]
        # Simulate saturation: push b3 deeply negative
        net.b3 = np.array([-10.0])
        net.w3 = np.random.randn(net.hidden_2, 1) * 0.01  # near-dead weights
        pre_b3 = float(net.b3[0])
        pre_w3_std = float(np.std(net.w3))
        # Trigger liberation
        nns._liberate_saturated_output_layers()
        assert float(net.b3[0]) == 0.0, (
            f"b3 not zeroed: {net.b3[0]}")
        # w3 should be reinitialized (std different from pre)
        post_w3_std = float(np.std(net.w3))
        assert post_w3_std != pre_w3_std, (
            "w3 unchanged — reinit didn't happen")

    def test_liberation_preserves_hidden_layers(self, tmp_path):
        """Hidden layer weights (w1, w2, b1, b2) must NOT be touched —
        they carry learned feature representations."""
        nns = self._ns(tmp_path)
        net = nns.programs["TEST_A"]
        # Snapshot hidden layers
        w1_pre = net.w1.copy()
        w2_pre = net.w2.copy()
        b1_pre = net.b1.copy()
        b2_pre = net.b2.copy()
        # Force saturation + liberate
        net.b3 = np.array([-10.0])
        nns._liberate_saturated_output_layers()
        # Hidden layers unchanged
        np.testing.assert_array_equal(net.w1, w1_pre)
        np.testing.assert_array_equal(net.w2, w2_pre)
        np.testing.assert_array_equal(net.b1, b1_pre)
        np.testing.assert_array_equal(net.b2, b2_pre)

    def test_liberation_idempotent_on_healthy_nn(self, tmp_path):
        """Healthy NN (b3 near 0) passes through liberation unchanged."""
        nns = self._ns(tmp_path)
        net = nns.programs["TEST_A"]
        # Fresh NN has b3 = 0 (Xavier init)
        w3_pre = net.w3.copy()
        b3_pre = net.b3.copy()
        nns._liberate_saturated_output_layers()
        np.testing.assert_array_equal(net.w3, w3_pre)
        np.testing.assert_array_equal(net.b3, b3_pre)

    def test_liberation_threshold_boundary(self, tmp_path):
        """b3 = exactly −5 is not liberated (boundary case); b3 = −5.1 is."""
        nns = self._ns(tmp_path)
        net = nns.programs["TEST_A"]
        # Exactly at boundary — not liberated
        net.b3 = np.array([-5.0])
        nns._liberate_saturated_output_layers()
        assert float(net.b3[0]) == -5.0, "Boundary case (-5.0) was liberated"
        # Just past boundary — liberated
        net.b3 = np.array([-5.1])
        nns._liberate_saturated_output_layers()
        assert float(net.b3[0]) == 0.0, "b3=-5.1 should have been liberated"
