"""
Tests for V5 Neural NervousSystem — NeuralReflexNet, ObservationSpace,
NervousTransitionBuffer, and NeuralNervousSystem registry.

Covers phases N1 + N2 + N3 of the V5 implementation plan.
"""
import json
import math
import os
import tempfile

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════
# N1: NeuralReflexNet Tests
# ═══════════════════════════════════════════════════════════════════

class TestNeuralReflexNet:
    """Tests for the core micro-network."""

    def _make_net(self, input_dim=55):
        from titan_plugin.logic.neural_reflex_net import NeuralReflexNet
        return NeuralReflexNet("test_net", input_dim=input_dim)

    def test_forward_output_range(self):
        """Output must be in [0, 1] (sigmoid)."""
        net = self._make_net()
        for _ in range(20):
            x = np.random.randn(55)
            out = net.forward(x)
            assert 0.0 <= out <= 1.0, f"Output {out} outside [0, 1]"

    def test_forward_zeros(self):
        """All-zeros input should produce valid output."""
        net = self._make_net()
        out = net.forward(np.zeros(55))
        assert 0.0 <= out <= 1.0

    def test_forward_ones(self):
        """All-ones input should produce valid output."""
        net = self._make_net()
        out = net.forward(np.ones(55))
        assert 0.0 <= out <= 1.0

    def test_batch_forward_matches_single(self):
        """Batch forward should match individual forwards."""
        net = self._make_net()
        inputs = np.random.randn(5, 55)
        batch_out = net.forward_batch(inputs)
        for i in range(5):
            single_out = net.forward(inputs[i])
            assert abs(batch_out[i, 0] - single_out) < 1e-10

    def test_train_step_reduces_loss(self):
        """Training should reduce loss over multiple steps."""
        net = self._make_net(input_dim=10)
        inputs = np.random.randn(32, 10)
        targets = np.random.rand(32, 1) * 0.5 + 0.25  # targets in [0.25, 0.75]

        losses = []
        for _ in range(50):
            loss = net.train_step(inputs, targets)
            losses.append(loss)

        # Loss should decrease over 50 steps
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.6f} → {losses[-1]:.6f}"
        assert net.total_updates == 50

    def test_save_load_roundtrip(self):
        """Save and load should preserve weights exactly."""
        net = self._make_net(input_dim=10)
        test_input = np.random.randn(10)
        out_before = net.forward(test_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_weights.json")
            net.save(path)

            from titan_plugin.logic.neural_reflex_net import NeuralReflexNet
            net2 = NeuralReflexNet("test_net", input_dim=10)
            loaded = net2.load(path)
            assert loaded is True

            out_after = net2.forward(test_input)
            assert abs(out_before - out_after) < 1e-12

    def test_param_count(self):
        """Verify parameter count for 55→48→24→1."""
        net = self._make_net(input_dim=55)
        expected = (55 * 48 + 48) + (48 * 24 + 24) + (24 * 1 + 1)
        assert net.param_count() == expected
        assert expected == 3889  # 55*48+48 + 48*24+24 + 24+1 = 2688+1200+25

    def test_different_input_dims(self):
        """Network should work with various input dimensions."""
        for dim in [30, 55, 75, 88]:
            net = self._make_net(input_dim=dim)
            out = net.forward(np.random.randn(dim))
            assert 0.0 <= out <= 1.0

    def test_get_stats(self):
        """Stats should include all expected fields."""
        net = self._make_net()
        stats = net.get_stats()
        assert stats["name"] == "test_net"
        assert stats["type"] == "neural"
        assert stats["input_dim"] == 55
        assert "param_count" in stats
        assert "total_updates" in stats
        assert "last_loss" in stats


# ═══════════════════════════════════════════════════════════════════
# N1: NervousTransitionBuffer Tests
# ═══════════════════════════════════════════════════════════════════

class TestNervousTransitionBuffer:
    """Tests for per-program transition storage."""

    def test_add_and_len(self):
        from titan_plugin.logic.neural_reflex_net import NervousTransitionBuffer
        buf = NervousTransitionBuffer(max_size=100)
        assert len(buf) == 0

        buf.add([1.0, 2.0], 0.5, 0.3, 0.8, True)
        assert len(buf) == 1

        buf.add([3.0, 4.0], 0.2, 0.1, 0.0, False)
        assert len(buf) == 2

    def test_max_size_eviction(self):
        from titan_plugin.logic.neural_reflex_net import NervousTransitionBuffer
        buf = NervousTransitionBuffer(max_size=5)
        for i in range(10):
            buf.add([float(i)], 0.5, 0.3, 0.0, False)
        assert len(buf) == 5

    def test_sample(self):
        from titan_plugin.logic.neural_reflex_net import NervousTransitionBuffer
        buf = NervousTransitionBuffer()
        for i in range(20):
            buf.add([float(i), float(i) * 2], 0.5, 0.3, 0.1, i % 3 == 0)

        obs, urg, vm, rew, fired = buf.sample(8)
        assert obs.shape == (8, 2)
        assert len(urg) == 8
        assert len(vm) == 8
        assert len(rew) == 8
        assert len(fired) == 8

    def test_update_last_reward(self):
        from titan_plugin.logic.neural_reflex_net import NervousTransitionBuffer
        buf = NervousTransitionBuffer()
        buf.add([1.0], 0.5, 0.3, 0.0, True)
        buf.add([2.0], 0.2, 0.1, 0.0, False)

        assert buf.last_fired is True  # First transition was fired
        buf.update_last_reward(0.9)
        # The fired transition's reward should be updated
        assert buf._rewards[0] == 0.9

    def test_save_load_roundtrip(self):
        from titan_plugin.logic.neural_reflex_net import NervousTransitionBuffer
        buf = NervousTransitionBuffer()
        for i in range(10):
            buf.add([float(i)], 0.5, 0.3, float(i) / 10, i % 2 == 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "buffer.json")
            buf.save(path)

            buf2 = NervousTransitionBuffer()
            loaded = buf2.load(path)
            assert loaded is True
            assert len(buf2) == 10


# ═══════════════════════════════════════════════════════════════════
# N1: ObservationSpace Tests
# ═══════════════════════════════════════════════════════════════════

class TestObservationSpace:
    """Tests for the centralized input builder."""

    def _make_observables(self):
        """Create mock observables dict."""
        parts = ["inner_body", "inner_mind", "inner_spirit",
                 "outer_body", "outer_mind", "outer_spirit"]
        return {
            p: {"coherence": 0.9, "magnitude": 0.7, "velocity": 0.05,
                "direction": 0.95, "polarity": 0.1}
            for p in parts
        }

    def _make_space(self):
        from titan_plugin.logic.observation_space import ObservationSpace
        space = ObservationSpace()
        space.update(observables=self._make_observables())
        return space

    def test_tier1_dimensions(self):
        """Tier 1 (core) should be 30D."""
        space = self._make_space()
        vec = space.build_input("core")
        assert len(vec) == 30

    def test_standard_dimensions(self):
        """Standard should be 55D (30 + 25)."""
        space = self._make_space()
        vec = space.build_input("standard")
        assert len(vec) == 55

    def test_extended_dimensions(self):
        """Extended should be 75D (30 + 25 + 20)."""
        space = self._make_space()
        vec = space.build_input("extended")
        assert len(vec) == 75

    def test_full_dimensions(self):
        """Full should be 88D (30 + 25 + 20 + 13)."""
        space = self._make_space()
        vec = space.build_input("full")
        assert len(vec) == 88

    def test_tier1_values(self):
        """Tier 1 should contain observable values in correct order."""
        space = self._make_space()
        vec = space.build_input("core")
        # First 5 values = inner_body: coherence, magnitude, velocity, direction, polarity
        assert vec[0] == 0.9   # coherence
        assert vec[1] == 0.7   # magnitude
        assert vec[2] == 0.05  # velocity
        assert vec[3] == 0.95  # direction
        assert vec[4] == 0.1   # polarity

    def test_feature_names_count(self):
        """Feature names should match dimension count."""
        from titan_plugin.logic.observation_space import ObservationSpace
        space = ObservationSpace()
        for fset, expected_dim in [("core", 30), ("standard", 55),
                                    ("extended", 75), ("full", 88)]:
            names = space.get_feature_names(fset)
            assert len(names) == expected_dim, f"{fset}: {len(names)} names != {expected_dim}D"

    def test_get_dim(self):
        """get_dim should return correct dimensions."""
        from titan_plugin.logic.observation_space import ObservationSpace
        assert ObservationSpace.get_dim("core") == 30
        assert ObservationSpace.get_dim("standard") == 55
        assert ObservationSpace.get_dim("extended") == 75
        assert ObservationSpace.get_dim("full") == 88

    def test_no_nan_with_empty_data(self):
        """ObservationSpace should produce finite values with empty/missing data."""
        from titan_plugin.logic.observation_space import ObservationSpace
        space = ObservationSpace()
        space.update()  # No data at all
        vec = space.build_input("full")
        assert len(vec) == 88
        assert np.all(np.isfinite(vec))


# ═══════════════════════════════════════════════════════════════════
# N2: NeuralNervousSystem Registry Tests
# ═══════════════════════════════════════════════════════════════════

class TestNeuralNervousSystem:
    """Tests for the config-driven registry + training."""

    def _make_config(self):
        return {
            "warmup_steps": 100,
            "train_every_n": 5,
            "batch_size": 8,
            "save_every_n": 50,
            "programs": {
                "REFLEX": {"enabled": True, "fire_threshold": 0.3, "input_features": "standard"},
                "FOCUS": {"enabled": True, "fire_threshold": 0.25, "input_features": "standard"},
                "IMPULSE": {"enabled": True, "fire_threshold": 0.3, "input_features": "standard"},
                "DISABLED_PROG": {"enabled": False},
            },
        }

    def _make_observables(self):
        parts = ["inner_body", "inner_mind", "inner_spirit",
                 "outer_body", "outer_mind", "outer_spirit"]
        return {
            p: {"coherence": 0.9, "magnitude": 0.7, "velocity": 0.05,
                "direction": 0.95, "polarity": 0.1}
            for p in parts
        }

    def test_loads_enabled_programs(self):
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = NeuralNervousSystem(self._make_config(), data_dir=tmpdir)
            assert len(ns.programs) == 3  # DISABLED_PROG excluded
            assert "REFLEX" in ns.programs
            assert "FOCUS" in ns.programs
            assert "IMPULSE" in ns.programs
            assert "DISABLED_PROG" not in ns.programs

    def test_evaluate_returns_signals(self):
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = NeuralNervousSystem(self._make_config(), data_dir=tmpdir)
            ns.update_observation_space(observables=self._make_observables())
            signals = ns.evaluate(self._make_observables())
            # Signals is a list of dicts with system/urgency keys
            assert isinstance(signals, list)
            for sig in signals:
                assert "system" in sig
                assert "urgency" in sig
                assert 0.0 <= sig["urgency"] <= 1.0

    def test_supervision_weight_decay(self):
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = NeuralNervousSystem(self._make_config(), data_dir=tmpdir)
            assert ns._get_supervision_weight() == 1.0  # Start

            ns._total_transitions = 50
            assert abs(ns._get_supervision_weight() - 0.5) < 0.01  # Mid

            ns._total_transitions = 100
            assert ns._get_supervision_weight() == 0.0  # End

    def test_training_phase_transitions(self):
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = NeuralNervousSystem(self._make_config(), data_dir=tmpdir)
            assert ns.training_phase == "bootstrap"

            ns._total_transitions = 50
            assert ns.training_phase == "blending"

            ns._total_transitions = 200
            assert ns.training_phase == "autonomous"

    def test_save_load_roundtrip(self):
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = NeuralNervousSystem(self._make_config(), data_dir=tmpdir)
            ns._total_transitions = 42
            ns._total_train_steps = 7
            ns.save_all()

            ns2 = NeuralNervousSystem(self._make_config(), data_dir=tmpdir)
            assert ns2._total_transitions == 42
            assert ns2._total_train_steps == 7

    def test_get_stats(self):
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = NeuralNervousSystem(self._make_config(), data_dir=tmpdir)
            stats = ns.get_stats()
            assert stats["version"] == "v5_neural"
            assert "training_phase" in stats
            assert "programs" in stats
            assert len(stats["programs"]) == 3

    def test_training_reduces_loss(self):
        """After multiple evaluate cycles, training should reduce loss."""
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config()
            config["train_every_n"] = 2
            config["batch_size"] = 4
            ns = NeuralNervousSystem(config, data_dir=tmpdir)

            obs = self._make_observables()
            # Run 30 evaluate cycles to accumulate transitions + trigger training
            # Each evaluate records 3 transitions (3 enabled programs)
            for _ in range(30):
                ns.update_observation_space(observables=obs)
                ns.evaluate(obs)

            # 30 cycles × 3 programs = 90 transitions total
            assert ns._total_transitions >= 30, f"Expected >=30 transitions, got {ns._total_transitions}"
            # Training should have occurred (every 2 transitions)
            assert ns._total_train_steps > 0, f"Expected >0 train steps, got {ns._total_train_steps}"

    def test_record_outcome(self):
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = NeuralNervousSystem(self._make_config(), data_dir=tmpdir)
            obs = self._make_observables()
            ns.update_observation_space(observables=obs)
            ns.evaluate(obs)  # Record transitions
            # Should not crash
            ns.record_outcome(0.8)
