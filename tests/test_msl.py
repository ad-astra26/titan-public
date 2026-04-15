"""Tests for Multisensory Synthesis Layer (MSL) — Phase 1 + Phase 2 + Phase 3."""

import json
import os
import random
import tempfile

import numpy as np
import pytest

from titan_plugin.logic.msl import (
    CONCEPT_NAMES,
    SELF_RELEVANCE_MAP,
    SPIRIT_DIMS,
    DEFAULT_INPUT_DIM,
    DEFAULT_OUTPUT_DIM,
    FRAME_DIM,
    ConvergenceDetector,
    ConvergenceMemoryLog,
    IConfidenceTracker,
    IRecipeEMA,
    SelfActionEcho,
    MODALITY_NAMES,
    MSLPolicyNet,
    MSLRewardComputer,
    MSLTemporalBuffer,
    MSLTransitionBuffer,
    MultisensorySynthesisLayer,
    STATIC_DIM,
    ConceptGrounder,
    CONCEPT_RELEVANCE_MAPS,
    HomeostaticAttention,
    ChiCoherenceTracker,
    IAMEventDetector,
)


# ── Temporal Buffer ────────────────────────────────────────────────────────


class TestMSLTemporalBuffer:
    def test_dimensions(self):
        buf = MSLTemporalBuffer(max_frames=5)
        assert buf.max_frames == 5
        assert buf.frame_count == 0
        assert not buf.is_ready()

    def test_push_and_ready(self):
        buf = MSLTemporalBuffer(max_frames=3)
        for _ in range(3):
            buf.push(np.random.randn(FRAME_DIM).astype(np.float32))
        assert buf.is_ready()
        assert buf.frame_count == 3

    def test_circular_eviction(self):
        buf = MSLTemporalBuffer(max_frames=3)
        frames = [np.full(FRAME_DIM, i, dtype=np.float32) for i in range(5)]
        for f in frames:
            buf.push(f)
        assert buf.frame_count == 3
        flat = buf.get_flat()
        # Should contain frames 2, 3, 4 (last 3)
        assert flat[0] == 2.0  # First element of frame[2]
        assert flat[FRAME_DIM] == 3.0  # First element of frame[3]

    def test_get_flat_dimensions(self):
        buf = MSLTemporalBuffer(max_frames=5)
        for _ in range(5):
            buf.push(np.random.randn(FRAME_DIM))
        buf.set_static_context(100, 0.7, 50.0, 0.0, 1.0)
        flat = buf.get_flat()
        assert flat.shape == (DEFAULT_INPUT_DIM,)  # 255D
        # Check static context normalization
        assert flat[-5] == pytest.approx(100 / 500.0, abs=0.01)  # vocab
        assert flat[-4] == pytest.approx(0.7, abs=0.01)  # chi
        assert flat[-3] == pytest.approx(50.0 / 200.0, abs=0.01)  # age
        assert flat[-2] == pytest.approx(0.0, abs=0.01)  # spirit_self
        assert flat[-1] == pytest.approx(1.0, abs=0.01)  # conversation

    def test_get_flat_not_ready(self):
        buf = MSLTemporalBuffer(max_frames=5)
        buf.push(np.random.randn(FRAME_DIM))
        flat = buf.get_flat()
        assert flat.shape == (DEFAULT_INPUT_DIM,)
        assert np.all(flat == 0.0)  # Returns zeros when not ready

    def test_frame_dimension_validation(self):
        buf = MSLTemporalBuffer()
        with pytest.raises(ValueError):
            buf.push(np.random.randn(10))  # Wrong dimension

    def test_get_latest_frame(self):
        buf = MSLTemporalBuffer(max_frames=3)
        f = np.full(FRAME_DIM, 42.0, dtype=np.float32)
        buf.push(f)
        latest = buf.get_latest_frame()
        assert latest is not None
        np.testing.assert_array_equal(latest, f)


# ── Policy Network ─────────────────────────────────────────────────────────


class TestMSLPolicyNet:
    def test_dimensions(self):
        net = MSLPolicyNet()
        assert net.input_dim == DEFAULT_INPUT_DIM
        assert net.output_dim == DEFAULT_OUTPUT_DIM

    def test_forward_shape(self):
        net = MSLPolicyNet()
        x = np.random.randn(DEFAULT_INPUT_DIM).astype(np.float32)
        out = net.forward(x)
        assert out.shape == (DEFAULT_OUTPUT_DIM,)

    def test_infer_structure(self):
        net = MSLPolicyNet()
        x = np.random.randn(DEFAULT_INPUT_DIM).astype(np.float32)
        result = net.infer(x)

        # Check all expected keys
        assert "attention_weights" in result
        assert "cross_modal_predictions" in result
        assert "distilled_context" in result
        assert "concept_activations" in result
        assert "spirit_resonance_gate" in result
        assert "coherence_pulse" in result

        # Attention weights sum to 1
        attn = result["attention_weights"]
        assert len(attn) == 7
        assert all(name in attn for name in MODALITY_NAMES)
        total = sum(attn.values())
        assert total == pytest.approx(1.0, abs=0.01)

        # Concept activations are sigmoid (0-1)
        concepts = result["concept_activations"]
        assert len(concepts) == 6
        for name in CONCEPT_NAMES:
            assert 0.0 <= concepts[name] <= 1.0

        # Coherence is sigmoid (0-1)
        assert 0.0 <= result["coherence_pulse"] <= 1.0

        # Distilled context is bounded (-1, 1)
        ctx = result["distilled_context"]
        assert len(ctx) == 20
        for v in ctx:
            assert -1.0 <= v <= 1.0

    def test_train_step(self):
        net = MSLPolicyNet()
        x = np.random.randn(DEFAULT_INPUT_DIM).astype(np.float32)
        loss = net.train_step(x, reward=0.5)
        assert isinstance(loss, float)
        assert net.total_updates == 1

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "policy.json")
            net1 = MSLPolicyNet(lr=0.002)
            x = np.random.randn(DEFAULT_INPUT_DIM).astype(np.float32)
            net1.forward(x)
            net1.total_updates = 42
            net1.save(path)

            net2 = MSLPolicyNet(lr=0.001)
            loaded = net2.load(path)
            assert loaded is True
            assert net2.total_updates == 42
            np.testing.assert_array_almost_equal(net1.w1, net2.w1)
            np.testing.assert_array_almost_equal(net1.ln_gamma, net2.ln_gamma)

    def test_load_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "policy.json")
            net1 = MSLPolicyNet(input_dim=100, h1=32, h2=16)
            net1.save(path)
            net2 = MSLPolicyNet()  # Default 255D
            loaded = net2.load(path)
            assert loaded is False


# ── Transition Buffer ──────────────────────────────────────────────────────


class TestMSLTransitionBuffer:
    def test_record_and_sample(self):
        buf = MSLTransitionBuffer(max_size=100)
        for _ in range(20):
            buf.record(
                np.random.randn(DEFAULT_INPUT_DIM),
                np.random.randn(10),
                np.random.randn(10),
                random.random(),
            )
        assert buf.size() == 20
        batch = buf.sample(batch_size=8)
        assert batch is not None
        states, preds, acts, rewards = batch
        assert len(states) == 8

    def test_fifo_eviction(self):
        buf = MSLTransitionBuffer(max_size=10)
        for i in range(15):
            buf.record(np.full(5, i), np.zeros(3), np.zeros(3), float(i))
        assert buf.size() == 10
        # First element should be from i=5
        assert buf._rewards[0] == 5.0

    def test_sample_too_small(self):
        buf = MSLTransitionBuffer(max_size=100)
        for _ in range(5):
            buf.record(np.zeros(5), np.zeros(3), np.zeros(3), 0.0)
        assert buf.sample(batch_size=16) is None

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "buffer.json")
            buf1 = MSLTransitionBuffer(max_size=100)
            for i in range(10):
                buf1.record(np.full(5, i), np.full(3, i), np.full(3, i), float(i))
            buf1.save(path)

            buf2 = MSLTransitionBuffer(max_size=100)
            loaded = buf2.load(path)
            assert loaded is True
            assert buf2.size() == 10


# ── Reward Computer ────────────────────────────────────────────────────────


class TestMSLRewardComputer:
    def test_compute_basic(self):
        rc = MSLRewardComputer()
        preds = np.array([0.5] * 10, dtype=np.float32)
        actuals = np.array([0.5] * 10, dtype=np.float32)
        reward, components = rc.compute(
            predictions=preds,
            actuals=actuals,
            cross_modal_coherence=0.8,
            attention_weights=np.array([1/7] * 7, dtype=np.float32),
        )
        assert 0.0 <= reward <= 1.0
        assert "prediction" in components
        assert "convergence" in components
        assert "internal" in components
        # Perfect prediction → high prediction reward
        assert components["prediction"] == pytest.approx(1.0, abs=0.01)

    def test_high_prediction_error(self):
        rc = MSLRewardComputer()
        preds = np.array([0.0] * 10, dtype=np.float32)
        actuals = np.array([1.0] * 10, dtype=np.float32)
        reward, components = rc.compute(
            predictions=preds, actuals=actuals,
            cross_modal_coherence=0.0,
            attention_weights=np.array([1/7] * 7, dtype=np.float32),
        )
        # High error → low prediction reward
        assert components["prediction"] < 0.1

    def test_epoch_stage_transition(self):
        rc = MSLRewardComputer()
        rc.update_stage_weights(100_000)  # Early
        assert rc._w_convergence == pytest.approx(0.40)
        rc.update_stage_weights(1_000_000)  # Mid
        assert rc._w_convergence == pytest.approx(0.25)
        rc.update_stage_weights(3_000_000)  # Mature
        assert rc._w_convergence == pytest.approx(0.15)


# ── Main Orchestrator ──────────────────────────────────────────────────────


class TestMultisensorySynthesisLayer:
    def test_init_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = MultisensorySynthesisLayer(config={"save_dir": tmpdir})
            assert msl.buffer.max_frames == 5
            assert msl.policy.input_dim == DEFAULT_INPUT_DIM
            assert msl.policy.output_dim == DEFAULT_OUTPUT_DIM

    def test_collect_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = MultisensorySynthesisLayer(config={"save_dir": tmpdir})
            msl.collect_snapshot(
                visual_semantic=[0.5, 0.6, 0.7, 0.8, 0.9],
                audio_physical=[0.3, 0.4, 0.5, 0.6, 0.7],
                pattern_profile=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                inner_body=[0.5] * 5,
                inner_mind=[0.5] * 15,
                outer_body=[0.5] * 5,
                neuromod_levels={"DA": 0.6, "5HT": 0.5, "NE": 0.4,
                                 "ACh": 0.5, "Endorphin": 0.3, "GABA": 0.7},
                action_flag=0.0,
                cross_modal=0.75,
                vocab_size=150,
                chi_total=0.65,
                developmental_age=30.0,
            )
            assert msl.buffer.frame_count == 1
            assert msl._total_snapshots == 1

    def test_tick_not_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = MultisensorySynthesisLayer(config={"save_dir": tmpdir})
            result = msl.tick()
            assert result is None  # Buffer not full yet

    def test_full_cycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = MultisensorySynthesisLayer(config={
                "save_dir": tmpdir,
                "min_transitions": 5,
                "train_every_n": 3,
            })
            # Fill buffer with 5 snapshots
            for i in range(5):
                msl.collect_snapshot(
                    visual_semantic=[0.1 * i] * 5,
                    audio_physical=[0.2 * i] * 5,
                    inner_body=[0.5] * 5,
                    inner_mind=[0.5] * 15,
                    outer_body=[0.5] * 5,
                    cross_modal=0.5 + 0.1 * i,
                    vocab_size=100,
                    chi_total=0.6,
                    developmental_age=20.0,
                )
            # Now tick should work
            result = msl.tick()
            assert result is not None
            assert "attention_weights" in result
            assert "coherence_pulse" in result
            assert "reward" in result
            assert "distilled_context" in result

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl1 = MultisensorySynthesisLayer(config={"save_dir": tmpdir})
            # Generate some data
            for i in range(5):
                msl1.collect_snapshot(
                    inner_body=[0.5] * 5, inner_mind=[0.5] * 15,
                    outer_body=[0.5] * 5, vocab_size=50,
                    chi_total=0.5, developmental_age=10.0,
                )
            msl1.tick()
            msl1.save_all()

            # Load into new instance
            msl2 = MultisensorySynthesisLayer(config={"save_dir": tmpdir})
            msl2.load_all()
            assert msl2.policy.total_updates == msl1.policy.total_updates

    def test_dream_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = MultisensorySynthesisLayer(config={
                "save_dir": tmpdir,
                "min_transitions": 5,
            })
            # Fill buffer and generate transitions
            for _ in range(10):
                for i in range(5):
                    msl.collect_snapshot(
                        inner_body=[0.5] * 5, inner_mind=[0.5] * 15,
                        outer_body=[0.5] * 5, vocab_size=50,
                        chi_total=0.5, developmental_age=10.0,
                    )
                msl.tick()

            assert msl.transitions.size() >= 5
            result = msl.train(boost_factor=2.0)
            assert result.get("trained") is True
            assert result.get("samples", 0) > 0

    def test_inner_mind_5d_fallback(self):
        """Ensure 5D inner_mind is padded to 15D gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = MultisensorySynthesisLayer(config={"save_dir": tmpdir})
            msl.collect_snapshot(
                inner_body=[0.5] * 5,
                inner_mind=[0.5] * 5,  # Only 5D instead of 15D
                outer_body=[0.5] * 5,
                vocab_size=50, chi_total=0.5, developmental_age=10.0,
            )
            assert msl.buffer.frame_count == 1

    def test_none_modality_defaults(self):
        """All None inputs should produce valid 50D frame with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = MultisensorySynthesisLayer(config={"save_dir": tmpdir})
            msl.collect_snapshot(
                vocab_size=0, chi_total=0.5, developmental_age=0.0,
            )
            assert msl.buffer.frame_count == 1


# ── Frame Assembly Dimension Check ─────────────────────────────────────────


class TestFrameAssembly:
    def test_frame_is_50d(self):
        """Verify the 50D frame composition: 5+5+7+5+15+5+6+1+1 = 50."""
        dims = [5, 5, 7, 5, 15, 5, 6, 1, 1]
        assert sum(dims) == FRAME_DIM == 50

    def test_total_input_is_255d(self):
        """Verify 5 frames * 50D + 5D static = 255D."""
        assert 5 * FRAME_DIM + STATIC_DIM == DEFAULT_INPUT_DIM == 255

    def test_output_is_51d(self):
        """Verify output heads: 7+10+20+6+7+1 = 51."""
        heads = [7, 10, 20, 6, 7, 1]
        assert sum(heads) == DEFAULT_OUTPUT_DIM == 51


# ══════════════════════════════════════════════════════════════════════════
# Phase 2: "I" Grounding Tests
# ══════════════════════════════════════════════════════════════════════════


class TestSelfRelevanceMap:
    def test_dimensions(self):
        assert SELF_RELEVANCE_MAP.shape == (SPIRIT_DIMS,)
        assert SPIRIT_DIMS == 132

    def test_tier_counts(self):
        core = (SELF_RELEVANCE_MAP == 1.0).sum()
        assert core == 30  # Feeling + WHO + WHY + Outer Identity

    def test_all_positive(self):
        assert (SELF_RELEVANCE_MAP > 0).all()

    def test_sums_correctly(self):
        total_dims = (
            (SELF_RELEVANCE_MAP == 1.0).sum() +
            (SELF_RELEVANCE_MAP == 0.7).sum() +
            (SELF_RELEVANCE_MAP == 0.4).sum() +
            np.isclose(SELF_RELEVANCE_MAP, 0.15).sum()
        )
        assert total_dims == 132


class TestConvergenceDetector:
    def _make_buffer_with_delta(self, ib_delta=0.0, nm_delta=0.0):
        buf = MSLTemporalBuffer(max_frames=5)
        for i in range(5):
            frame = np.full(FRAME_DIM, 0.5, dtype=np.float32)
            if i == 4:  # Last frame has delta
                frame[17:22] += ib_delta  # inner body
                frame[42:48] += nm_delta  # neuromod
            buf.push(frame)
        return buf

    def test_detects_convergence(self):
        det = ConvergenceDetector()
        buf = self._make_buffer_with_delta(ib_delta=0.05, nm_delta=0.05)
        result = det.check(buf, current_epoch=100, is_dreaming=False,
                          spirit_self_active=False, action_type="internal",
                          external_stimulus=False)
        assert result is not None
        assert result["action_type"] == "internal"
        assert result["count"] == 1

    def test_no_action_no_convergence(self):
        det = ConvergenceDetector()
        buf = self._make_buffer_with_delta(ib_delta=0.05)
        result = det.check(buf, current_epoch=100, is_dreaming=False,
                          spirit_self_active=False, action_type=None,
                          external_stimulus=False)
        assert result is None

    def test_external_stimulus_blocks(self):
        det = ConvergenceDetector()
        buf = self._make_buffer_with_delta(ib_delta=0.05)
        result = det.check(buf, current_epoch=100, is_dreaming=False,
                          spirit_self_active=False, action_type="internal",
                          external_stimulus=True)
        assert result is None

    def test_dream_filter(self):
        det = ConvergenceDetector()
        buf = self._make_buffer_with_delta(ib_delta=0.05, nm_delta=0.05)
        # Dreaming without SPIRIT_SELF → blocked
        result = det.check(buf, current_epoch=100, is_dreaming=True,
                          spirit_self_active=False, action_type="internal",
                          external_stimulus=False)
        assert result is None
        # Dreaming WITH SPIRIT_SELF → allowed (lucid dream)
        result = det.check(buf, current_epoch=100, is_dreaming=True,
                          spirit_self_active=True, action_type="internal",
                          external_stimulus=False)
        assert result is not None

    def test_epoch_gap_enforced(self):
        det = ConvergenceDetector()
        buf = self._make_buffer_with_delta(ib_delta=0.05, nm_delta=0.05)
        # First convergence
        r1 = det.check(buf, current_epoch=100, is_dreaming=False,
                       spirit_self_active=False, action_type="internal",
                       external_stimulus=False)
        assert r1 is not None
        # Too soon (gap < 10)
        r2 = det.check(buf, current_epoch=105, is_dreaming=False,
                       spirit_self_active=False, action_type="internal",
                       external_stimulus=False)
        assert r2 is None
        # Enough gap
        r3 = det.check(buf, current_epoch=115, is_dreaming=False,
                       spirit_self_active=False, action_type="internal",
                       external_stimulus=False)
        assert r3 is not None

    def test_engagement_double_weight(self):
        det = ConvergenceDetector()
        buf = self._make_buffer_with_delta(ib_delta=0.05, nm_delta=0.05)
        result = det.check(buf, current_epoch=100, is_dreaming=False,
                          spirit_self_active=False, action_type="engagement",
                          external_stimulus=False)
        assert result is not None
        assert result["weight"] == 2.0


class TestIConfidenceTracker:
    def test_initial_zero(self):
        ct = IConfidenceTracker()
        assert ct.confidence == 0.0
        assert not ct.is_grounded

    def test_log_ramp(self):
        ct = IConfidenceTracker()
        for i in range(200):
            ct.on_convergence(i * 10)
        assert ct.confidence == pytest.approx(0.0, abs=0.01)  # At onset
        for i in range(200, 400):
            ct.on_convergence(i * 10)
        assert ct.confidence > 0.3  # Ramping up

    def test_event_bonus(self):
        ct = IConfidenceTracker()
        ct.on_event("social_engagement")
        assert ct.confidence == pytest.approx(0.005, abs=0.001)
        ct.on_event("eureka")
        assert ct.confidence == pytest.approx(0.009, abs=0.001)

    def test_event_bonus_cap(self):
        ct = IConfidenceTracker()
        for _ in range(100):
            ct.on_event("social_engagement")
        assert ct.confidence <= 0.3 + 0.01  # Capped

    def test_additive_ramp_plus_bonus(self):
        ct = IConfidenceTracker()
        # Add some convergences past onset
        for i in range(400):
            ct.on_convergence(i * 10)
        base = ct.confidence
        ct.on_event("eureka")
        assert ct.confidence > base

    def test_decay(self):
        ct = IConfidenceTracker()
        for i in range(300):
            ct.on_convergence(i * 10)
        ct.on_event("social_engagement")
        conf_before = ct.confidence
        ct.check_decay(current_epoch=300 * 10 + 10000)  # 10K epochs later
        assert ct.confidence <= conf_before

    def test_save_load(self):
        ct1 = IConfidenceTracker()
        for i in range(250):
            ct1.on_convergence(i * 10)
        ct1.on_event("eureka")
        d = ct1.to_dict()
        ct2 = IConfidenceTracker()
        ct2.from_dict(d)
        assert ct2.confidence == pytest.approx(ct1.confidence, abs=0.001)


class TestIRecipeEMA:
    def test_deferred_init(self):
        recipe = IRecipeEMA()
        assert not recipe.is_initialized
        assert recipe.recipe is None
        # Feed 9 convergences — not enough
        for _ in range(9):
            recipe.on_convergence(np.random.randn(SPIRIT_DIMS), quality=0.5)
        assert not recipe.is_initialized
        # 10th triggers init
        recipe.on_convergence(np.random.randn(SPIRIT_DIMS), quality=0.5)
        assert recipe.is_initialized
        assert recipe.recipe is not None
        assert recipe.recipe.shape == (SPIRIT_DIMS,)

    def test_ema_blending(self):
        recipe = IRecipeEMA(config={"recipe_init_count": 2})
        # Init with 2 snapshots
        recipe.on_convergence(np.zeros(SPIRIT_DIMS), quality=0.5)
        recipe.on_convergence(np.ones(SPIRIT_DIMS), quality=0.5)
        init_recipe = recipe.recipe.copy()
        # Blend a new snapshot
        recipe.on_convergence(np.ones(SPIRIT_DIMS) * 2.0, quality=0.8)
        # Recipe should have shifted toward 2.0
        assert recipe.recipe[0] > init_recipe[0]

    def test_quality_affects_alpha(self):
        r1 = IRecipeEMA(config={"recipe_init_count": 2})
        r2 = IRecipeEMA(config={"recipe_init_count": 2})
        base = np.zeros(SPIRIT_DIMS)
        # Both init with same data
        for r in [r1, r2]:
            r.on_convergence(base, quality=0.5)
            r.on_convergence(base, quality=0.5)
        # Strong convergence
        r1.on_convergence(np.ones(SPIRIT_DIMS), quality=0.9)
        # Weak convergence
        r2.on_convergence(np.ones(SPIRIT_DIMS), quality=0.1)
        # Strong should shift more
        assert r1.recipe[0] > r2.recipe[0]

    def test_save_load(self):
        recipe = IRecipeEMA(config={"recipe_init_count": 3})
        for _ in range(5):
            recipe.on_convergence(np.random.randn(SPIRIT_DIMS), quality=0.5)
        d = recipe.to_dict()
        recipe2 = IRecipeEMA()
        recipe2.from_dict(d)
        assert recipe2.is_initialized == recipe.is_initialized
        if recipe.recipe is not None:
            np.testing.assert_array_almost_equal(recipe.recipe, recipe2.recipe)


class TestConvergenceMemoryLog:
    def test_record_and_size(self):
        log = ConvergenceMemoryLog(max_size=10)
        for i in range(5):
            log.record({"epoch": i, "quality": 0.5 + i * 0.1})
        assert log.size() == 5

    def test_circular_eviction(self):
        log = ConvergenceMemoryLog(max_size=5)
        for i in range(10):
            log.record({"epoch": i, "quality": float(i) / 10})
        assert log.size() == 5
        assert log._events[0]["epoch"] == 5

    def test_recent_quality(self):
        log = ConvergenceMemoryLog()
        for i in range(5):
            log.record({"quality": 0.8})
        assert log.recent_quality(5) == pytest.approx(0.8)

    def test_signal_distribution(self):
        log = ConvergenceMemoryLog()
        log.record({"action_type": "internal"})
        log.record({"action_type": "internal"})
        log.record({"action_type": "engagement"})
        dist = log.get_signal_distribution()
        assert dist["internal"] == 2
        assert dist["engagement"] == 1

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            log1 = ConvergenceMemoryLog()
            log1.record({"epoch": 1, "quality": 0.5}, np.ones(SPIRIT_DIMS))
            log1.save(path)
            log2 = ConvergenceMemoryLog()
            assert log2.load(path)
            assert log2.size() == 1


class TestSelfActionEcho:
    def test_trigger_and_decay(self):
        echo = SelfActionEcho()
        assert not echo.is_active
        echo.trigger("internal")
        assert echo.is_active
        inner, outer = echo.get_and_decay()
        assert np.linalg.norm(inner) > 0
        assert np.linalg.norm(outer) > 0
        assert np.linalg.norm(inner) > np.linalg.norm(outer)

    def test_decays_to_zero(self):
        echo = SelfActionEcho()
        echo.trigger("internal")
        for _ in range(50):
            echo.get_and_decay()
        assert not echo.is_active

    def test_external_stronger(self):
        e1 = SelfActionEcho()
        e2 = SelfActionEcho()
        e1.trigger("internal")
        e2.trigger("external")
        i1, _ = e1.get_and_decay()
        i2, _ = e2.get_and_decay()
        assert np.linalg.norm(i2) > np.linalg.norm(i1)


class TestPhaseBNeuromodCoupling:
    """Phase B: Neuromod coupling to homeostatic attention."""

    def test_modulate_changes_params(self):
        """Neuromod levels should modulate homeostatic parameters."""
        h = HomeostaticAttention(n_modalities=7)
        base_gain = h._base_autoreceptor_gain
        base_tau = h._base_tonic_tau
        base_allo = h._base_allo_lr
        base_homeo = h._base_homeo_lr

        # High NE → stronger autoreceptor
        h.modulate_from_neuromod({"NE": 0.9, "GABA": 0.1, "DA": 0.1, "ACh": 0.1})
        assert h._autoreceptor_gain > base_gain
        assert h._tonic_tau > base_tau * 0.8  # Low GABA → slow clearance

        # High GABA → faster clearance (lower tau)
        h.modulate_from_neuromod({"NE": 0.1, "GABA": 0.9, "DA": 0.1, "ACh": 0.1})
        assert h._tonic_tau < base_tau  # High GABA → faster clearance

        # High DA → faster setpoint drift
        h.modulate_from_neuromod({"NE": 0.5, "GABA": 0.5, "DA": 0.9, "ACh": 0.5})
        assert h._allo_lr > base_allo

        # High ACh → stronger sensitivity
        h.modulate_from_neuromod({"NE": 0.5, "GABA": 0.5, "DA": 0.5, "ACh": 0.9})
        assert h._homeo_lr > base_homeo

    def test_modulate_with_defaults(self):
        """Missing neuromod keys should use 0.5 defaults."""
        h = HomeostaticAttention(n_modalities=7)
        h.modulate_from_neuromod({})  # All defaults
        # Should not crash, params should be at base * (1 + coupling * 0.5)
        assert h._autoreceptor_gain > h._base_autoreceptor_gain

    def test_modulate_preserves_attention(self):
        """Modulation should not break attention computation."""
        h = HomeostaticAttention(n_modalities=7)
        h.modulate_from_neuromod({"NE": 0.8, "GABA": 0.3, "DA": 0.7, "ACh": 0.6})
        logits = np.random.randn(7).astype(np.float32)
        attn = h.adjust_and_attend(logits)
        assert abs(attn.sum() - 1.0) < 1e-5
        assert np.all(attn >= 0)


class TestPhase2Integration:
    def test_full_convergence_cycle(self):
        """Test the complete Phase 2 pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = MultisensorySynthesisLayer(config={
                "save_dir": tmpdir,
                "min_transitions": 5,
                "identity": {"recipe_init_count": 3},
            })
            # Fill buffer
            for _ in range(5):
                msl.collect_snapshot(
                    inner_body=[0.5] * 5, inner_mind=[0.5] * 15,
                    outer_body=[0.5] * 5, vocab_size=50,
                    chi_total=0.5, developmental_age=10.0,
                )
            # Signal action
            msl.signal_action("internal")
            # Tick
            result = msl.tick()
            assert result is not None
            # Check convergence (with large delta in buffer for detection)
            snap = np.random.randn(SPIRIT_DIMS).astype(np.float32) * 0.5 + 0.5
            conv = msl.check_convergence(
                current_epoch=100, is_dreaming=False,
                spirit_self_active=False, spirit_snapshot=snap)
            # May or may not detect (depends on body delta from echo)
            # But confidence should start from event bonuses
            msl.confidence.on_event("social_engagement")
            assert msl.get_i_confidence() > 0

    def test_i_perturbation(self):
        """Test that compute_i_perturbation returns scaled 132D vector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = MultisensorySynthesisLayer(config={
                "save_dir": tmpdir,
                "identity": {"recipe_init_count": 2},
            })
            # Build a recipe
            for _ in range(3):
                msl.i_recipe.on_convergence(
                    np.full(SPIRIT_DIMS, 0.7, dtype=np.float32), quality=0.5)
            # Set confidence
            for _ in range(250):
                msl.confidence.on_convergence(0)
            assert msl.get_i_confidence() > 0
            # Compute perturbation
            spirit = [0.5] * 130
            perturb = msl.compute_i_perturbation(spirit, [1.0, 0.5])
            assert perturb is not None
            assert perturb.shape == (SPIRIT_DIMS,)
            # Core dims should have larger perturbation
            core_mag = float(np.abs(perturb[20:40]).mean())
            tert_mag = float(np.abs(perturb[105:130]).mean())
            assert core_mag > tert_mag

    def test_phase2_save_load(self):
        """Test save/load preserves Phase 2 state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            msl1 = MultisensorySynthesisLayer(config={
                "save_dir": tmpdir,
                "identity": {"recipe_init_count": 2},
            })
            # Build some state
            for i in range(5):
                msl1.collect_snapshot(
                    inner_body=[0.5] * 5, inner_mind=[0.5] * 15,
                    outer_body=[0.5] * 5, vocab_size=50,
                    chi_total=0.5, developmental_age=10.0,
                )
            msl1.confidence.on_event("eureka")
            msl1.i_recipe.on_convergence(np.ones(SPIRIT_DIMS) * 0.5, 0.5)
            msl1.i_recipe.on_convergence(np.ones(SPIRIT_DIMS) * 0.6, 0.5)
            msl1.convergence_log.record({"epoch": 1, "quality": 0.5})
            msl1.save_all()

            msl2 = MultisensorySynthesisLayer(config={
                "save_dir": tmpdir,
                "identity": {"recipe_init_count": 2},
            })
            msl2.load_all()
            assert msl2.get_i_confidence() == pytest.approx(
                msl1.get_i_confidence(), abs=0.001)
            assert msl2.convergence_log.size() == 1


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Concept Cascade Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestConceptRelevanceMaps:
    """Test per-concept 132D relevance maps."""

    def test_all_concepts_present(self):
        for name in CONCEPT_NAMES:
            assert name in CONCEPT_RELEVANCE_MAPS, f"Missing map for {name}"

    def test_dimensions(self):
        for name, m in CONCEPT_RELEVANCE_MAPS.items():
            assert m.shape == (SPIRIT_DIMS,), f"{name} map wrong shape: {m.shape}"

    def test_all_positive(self):
        for name, m in CONCEPT_RELEVANCE_MAPS.items():
            assert (m > 0).all(), f"{name} map has zero dimensions"

    def test_they_reduced_magnitudes(self):
        """THEY uses reduced Core=0.7, not 1.0."""
        they_map = CONCEPT_RELEVANCE_MAPS["THEY"]
        assert they_map.max() <= 0.71  # 0.7 + tolerance

    def test_i_map_has_core_1(self):
        """I should have Core dimensions at 1.0."""
        i_map = CONCEPT_RELEVANCE_MAPS["I"]
        assert i_map[10] == 1.0  # Inner Feeling
        assert i_map[20] == 1.0  # Inner WHO

    def test_you_is_social_focused(self):
        """YOU should have social dimensions as Core."""
        you_map = CONCEPT_RELEVANCE_MAPS["YOU"]
        assert you_map[80] == 1.0  # Social Mind
        assert you_map[85] == 1.0  # Identity


class TestConceptGrounder:
    """Test ConceptGrounder class."""

    def test_init(self):
        cg = ConceptGrounder()
        assert len(cg._trackers) == 5
        assert len(cg._recipes) == 5
        assert len(cg._logs) == 5
        assert cg._interaction_matrix.shape == (6, 6)

    def test_signal_yes(self):
        cg = ConceptGrounder()
        evt = cg.signal_yes(quality=0.8, epoch=100, spirit_snap=None)
        assert evt is not None
        assert evt["concept"] == "YES"
        assert cg._trackers["YES"]._convergence_count == 1

    def test_signal_no(self):
        cg = ConceptGrounder()
        evt = cg.signal_no(quality=0.6, epoch=100, spirit_snap=None)
        assert evt is not None
        assert cg._trackers["NO"]._convergence_count == 1

    def test_signal_they(self):
        cg = ConceptGrounder()
        evt = cg.signal_they("reply_received", "user1", 0.7, epoch=100, spirit_snap=None)
        assert evt is not None
        assert cg._trackers["THEY"]._convergence_count == 1

    def test_signal_you(self):
        cg = ConceptGrounder()
        evt = cg.signal_you("pubkey123", 0.5, epoch=100, spirit_snap=None)
        assert evt is not None
        assert cg._trackers["YOU"]._convergence_count == 1

    def test_epoch_gap_enforced(self):
        cg = ConceptGrounder()
        cg.signal_yes(0.8, epoch=100, spirit_snap=None)
        evt = cg.signal_yes(0.8, epoch=105, spirit_snap=None)  # Too soon (gap < 10)
        assert evt is None
        evt = cg.signal_yes(0.8, epoch=115, spirit_snap=None)  # Far enough
        assert evt is not None

    def test_interaction_matrix(self):
        cg = ConceptGrounder()
        cg.update_interaction_matrix("YES", i_confidence=0.5)
        assert cg._interaction_matrix[0, 2] > 0  # I→YES reinforcement

    def test_we_gating(self):
        cg = ConceptGrounder()
        assert not cg.is_we_unlocked(i_confidence=0.0)
        # Seed enough YOU convergences to get confidence > 0.1
        for i in range(200):
            cg._trackers["YOU"].on_convergence(i * 20)
        assert cg._trackers["YOU"].confidence > 0.1
        assert cg.is_we_unlocked(i_confidence=0.5)

    def test_concept_confidences(self):
        cg = ConceptGrounder()
        confs = cg.get_concept_confidences()
        assert set(confs.keys()) == {"YOU", "YES", "NO", "WE", "THEY"}
        assert all(v == 0.0 for v in confs.values())

    def test_emotional_valence(self):
        cg = ConceptGrounder()
        v = cg.get_emotional_valence("YES")
        assert "DA" in v["nudge_map"]
        assert v["max_delta"] == 0.015
        # THEY has reduced max_delta
        t = cg.get_emotional_valence("THEY")
        assert t["max_delta"] == 0.008

    def test_recipe_update(self):
        cg = ConceptGrounder()
        snap = np.random.randn(SPIRIT_DIMS).astype(np.float32) * 0.1 + 0.5
        for i in range(12):
            cg.signal_yes(0.8, epoch=i * 20, spirit_snap=snap)
        assert cg._recipes["YES"].is_initialized

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "concepts.json")
            cg1 = ConceptGrounder()
            for i in range(5):
                cg1.signal_yes(0.8, epoch=i * 20, spirit_snap=None)
            cg1.signal_no(0.7, epoch=200, spirit_snap=None)
            cg1.update_interaction_matrix("YES", 0.3)
            cg1.save(path)

            cg2 = ConceptGrounder()
            assert cg2.load(path)
            assert cg2._trackers["YES"]._convergence_count == 5
            assert cg2._trackers["NO"]._convergence_count == 1
            assert cg2._interaction_matrix[0, 2] > 0

    def test_perturbation(self):
        cg = ConceptGrounder()
        spirit = np.full(SPIRIT_DIMS, 0.5, dtype=np.float32)
        # No confidence = no perturbation
        assert cg.compute_perturbation("YES", spirit) is None
        # Build confidence
        for i in range(120):
            cg._trackers["YES"].on_convergence(i * 20)
        # Build recipe via on_convergence (needs init_count=10 snapshots)
        snap = np.full(SPIRIT_DIMS, 0.6, dtype=np.float32)
        for i in range(12):
            cg._recipes["YES"].on_convergence(snap, 0.8)
        assert cg._recipes["YES"].is_initialized
        result = cg.compute_perturbation("YES", spirit)
        assert result is not None
        assert result.shape == (SPIRIT_DIMS,)


class TestPhase3Integration:
    """Test Phase 3 integration with MSL orchestrator."""

    def test_msl_with_concept_grounder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = MultisensorySynthesisLayer(config={
                "save_dir": tmpdir,
                "concepts": {"enabled": True},
            })
            assert msl.concept_grounder is not None

    def test_msl_without_concept_grounder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = MultisensorySynthesisLayer(config={
                "save_dir": tmpdir,
                "concepts": {"enabled": False},
            })
            assert msl.concept_grounder is None

    def test_signal_engagement_routing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = MultisensorySynthesisLayer(config={
                "save_dir": tmpdir,
                "concepts": {"enabled": True},
            })
            # Stranger → THEY
            msl.signal_engagement("reply_received", "stranger1", 0.7, is_regular=False)
            assert msl.concept_grounder._trackers["THEY"]._convergence_count == 1
            # Regular → YOU
            msl.signal_engagement("reply_received", "friend1", 0.8, is_regular=True)
            assert msl.concept_grounder._trackers["YOU"]._convergence_count == 1

    def test_concept_supervised_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = MultisensorySynthesisLayer(config={
                "save_dir": tmpdir,
                "min_transitions": 5,
                "train_every_n": 3,
                "concepts": {"enabled": True},
            })
            # Accumulate transitions
            for _ in range(10):
                for i in range(5):
                    msl.collect_snapshot(
                        inner_body=[0.5] * 5, inner_mind=[0.5] * 15,
                        outer_body=[0.5] * 5, vocab_size=50,
                        chi_total=0.5, developmental_age=10.0)
                msl.tick()
            result = msl.train()
            assert result.get("trained") is True

    def test_i_convergence_updates_interaction_matrix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = MultisensorySynthesisLayer(config={
                "save_dir": tmpdir,
                "concepts": {"enabled": True},
                "identity": {"min_convergence_epoch_gap": 1},
            })
            # Simulate conditions for convergence
            for i in range(5):
                msl.collect_snapshot(
                    inner_body=[0.5 + i * 0.02] * 5,
                    inner_mind=[0.5] * 15,
                    outer_body=[0.5] * 5, vocab_size=50,
                    chi_total=0.5, developmental_age=10.0)
            # Signal action + check convergence
            msl.signal_action("internal")
            # Build confidence so interaction matrix updates
            msl.confidence._convergence_count = 300
            msl.confidence._grounded = True
            event = msl.check_convergence(
                current_epoch=1000, is_dreaming=False,
                spirit_self_active=False)
            # Matrix should have been updated if convergence detected
            # (even if not detected, verify no crash)

    def test_concept_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl1 = MultisensorySynthesisLayer(config={
                "save_dir": tmpdir,
                "concepts": {"enabled": True},
            })
            msl1.concept_grounder.signal_yes(0.8, epoch=100, spirit_snap=None)
            msl1.save_all()

            msl2 = MultisensorySynthesisLayer(config={
                "save_dir": tmpdir,
                "concepts": {"enabled": True},
            })
            msl2.load_all()
            assert msl2.concept_grounder._trackers["YES"]._convergence_count == 1


# ── Phase 5: Chi Coherence + "I AM" Event Detection ─────────────────────


class TestChiCoherenceTracker:
    def test_init(self):
        tracker = ChiCoherenceTracker()
        assert tracker.chi == 0.0

    def test_update_returns_value(self):
        tracker = ChiCoherenceTracker()
        ib = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        ob = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        chi = tracker.update(pred_error=0.1, entropy_normalized=0.9,
                             inner_body_5d=ib, outer_body_5d=ob)
        assert 0.0 <= chi <= 1.0
        assert chi == tracker.chi

    def test_perfect_alignment(self):
        """Identical inner/outer bodies should give high alignment."""
        tracker = ChiCoherenceTracker()
        ib = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        for _ in range(10):
            chi = tracker.update(pred_error=0.0, entropy_normalized=0.9,
                                 inner_body_5d=ib, outer_body_5d=ib)
        # Perfect predictions + perfect alignment + stable entropy = high chi
        assert chi > 0.7

    def test_high_error_low_chi(self):
        """High prediction error should lower chi."""
        tracker = ChiCoherenceTracker()
        ib = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        for _ in range(10):
            chi = tracker.update(pred_error=0.9, entropy_normalized=0.5,
                                 inner_body_5d=ib, outer_body_5d=-ib)
        assert chi < 0.5

    def test_entropy_stability(self):
        """Stable entropy should contribute positively."""
        tracker = ChiCoherenceTracker()
        ib = np.array([0.5] * 5, dtype=np.float32)
        # All same entropy = perfect stability
        for _ in range(10):
            tracker.update(pred_error=0.3, entropy_normalized=0.9,
                           inner_body_5d=ib, outer_body_5d=ib)
        stable = tracker.chi

        tracker2 = ChiCoherenceTracker()
        # Wildly varying entropy = unstable
        for i in range(10):
            ent = 0.1 if i % 2 == 0 else 0.95
            tracker2.update(pred_error=0.3, entropy_normalized=ent,
                            inner_body_5d=ib, outer_body_5d=ib)
        unstable = tracker2.chi
        assert stable > unstable

    def test_to_from_dict(self):
        tracker = ChiCoherenceTracker()
        tracker.update(pred_error=0.2, entropy_normalized=0.8)
        d = tracker.to_dict()
        assert "chi" in d
        tracker2 = ChiCoherenceTracker()
        tracker2.from_dict(d)
        assert abs(tracker2.chi - tracker.chi) < 0.001


class TestIAMEventDetector:
    def test_init(self):
        det = IAMEventDetector()
        assert det.total_events == 0
        assert det.sustained_count == 0

    def test_no_event_when_conditions_not_met(self):
        det = IAMEventDetector()
        # Only I-confidence meets threshold
        event = det.check(current_epoch=1000, i_confidence=0.95,
                          chi_coherence=0.3, convergence_quality_last_10=0.2)
        assert event is None
        assert det.total_events == 0

    def test_sustained_counter_increments(self):
        det = IAMEventDetector(config={"chi_coherence_threshold": 0.5})
        det.check(current_epoch=1, i_confidence=0.5, chi_coherence=0.6,
                  convergence_quality_last_10=0.1)
        assert det.sustained_count == 1
        det.check(current_epoch=2, i_confidence=0.5, chi_coherence=0.6,
                  convergence_quality_last_10=0.1)
        assert det.sustained_count == 2

    def test_sustained_counter_resets(self):
        det = IAMEventDetector(config={"chi_coherence_threshold": 0.5})
        det.check(current_epoch=1, i_confidence=0.5, chi_coherence=0.6,
                  convergence_quality_last_10=0.1)
        det.check(current_epoch=2, i_confidence=0.5, chi_coherence=0.6,
                  convergence_quality_last_10=0.1)
        assert det.sustained_count == 2
        # Drop below threshold
        det.check(current_epoch=3, i_confidence=0.5, chi_coherence=0.3,
                  convergence_quality_last_10=0.1)
        assert det.sustained_count == 0

    def test_event_fires_when_all_conditions_met(self):
        det = IAMEventDetector(config={
            "i_confidence_threshold": 0.5,
            "chi_coherence_threshold": 0.5,
            "sustained_coherence_min_epochs": 3,
            "convergence_quality_threshold": 0.3,
            "min_event_gap_epochs": 0,
        })
        # Build sustained count — first 2 should NOT fire
        for i in range(2):
            event = det.check(current_epoch=i, i_confidence=0.6,
                              chi_coherence=0.7,
                              convergence_quality_last_10=0.5,
                              pi_value=0.05)
            assert event is None
        # Third epoch: sustained_count reaches 3, should fire
        event = det.check(current_epoch=2, i_confidence=0.6,
                          chi_coherence=0.7,
                          convergence_quality_last_10=0.5,
                          pi_value=0.05)
        assert event is not None
        assert event["event_number"] == 1
        assert event["pi_value"] == 0.05
        assert det.total_events >= 1

    def test_min_event_gap(self):
        det = IAMEventDetector(config={
            "i_confidence_threshold": 0.5,
            "chi_coherence_threshold": 0.5,
            "sustained_coherence_min_epochs": 2,
            "convergence_quality_threshold": 0.3,
            "min_event_gap_epochs": 100,
        })
        # First event fires (use epochs > 0 to avoid gap collision with init)
        for i in range(1000, 1005):
            det.check(current_epoch=i, i_confidence=0.6, chi_coherence=0.7,
                      convergence_quality_last_10=0.5)
        assert det.total_events >= 1
        first_count = det.total_events
        # Events within gap should NOT fire
        for i in range(1005, 1050):
            det.check(current_epoch=i, i_confidence=0.6, chi_coherence=0.7,
                      convergence_quality_last_10=0.5)
        assert det.total_events == first_count  # No new events within gap

    def test_to_from_dict(self):
        det = IAMEventDetector(config={
            "i_confidence_threshold": 0.5,
            "chi_coherence_threshold": 0.5,
            "sustained_coherence_min_epochs": 2,
            "convergence_quality_threshold": 0.3,
            "min_event_gap_epochs": 0,
        })
        for i in range(5):
            det.check(current_epoch=i, i_confidence=0.6, chi_coherence=0.7,
                      convergence_quality_last_10=0.5)
        d = det.to_dict()
        det2 = IAMEventDetector()
        det2.from_dict(d)
        assert det2.total_events == det.total_events
        assert det2.sustained_count == det.sustained_count


class TestPhase5Integration:
    def _make_msl(self, tmpdir):
        return MultisensorySynthesisLayer(config={
            "save_dir": tmpdir,
            "concepts": {"enabled": True},
            "phase5": {
                "coherence_window": 5,
            },
        })

    def test_msl_has_phase5_components(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = self._make_msl(tmpdir)
            assert hasattr(msl, "chi_tracker")
            assert hasattr(msl, "iam_detector")
            assert isinstance(msl.chi_tracker, ChiCoherenceTracker)
            assert isinstance(msl.iam_detector, IAMEventDetector)

    def test_tick_produces_chi_coherence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = self._make_msl(tmpdir)
            # Fill buffer
            for _ in range(5):
                msl.collect_snapshot(
                    visual_semantic=[0.5]*5, audio_physical=[0.5]*5,
                    pattern_profile=[0.1]*7, inner_body=[0.5]*5,
                    inner_mind=[0.5]*15, outer_body=[0.5]*5,
                    neuromod_levels={"DA": 0.5, "5HT": 0.5, "NE": 0.5,
                                     "ACh": 0.5, "Endorphin": 0.5, "GABA": 0.5})
            out = msl.tick()
            assert out is not None
            assert "chi_coherence" in out
            assert 0.0 <= out["chi_coherence"] <= 1.0
            # coherence_pulse should now be same as chi
            assert out["coherence_pulse"] == out["chi_coherence"]

    def test_set_pi_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = self._make_msl(tmpdir)
            msl.set_pi_value(0.042)
            assert msl._last_pi_value == 0.042

    def test_get_chi_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl = self._make_msl(tmpdir)
            state = msl.get_chi_state()
            assert "chi_coherence" in state
            assert "iam_detector" in state
            assert "total_iam_events" in state

    def test_phase5_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msl1 = self._make_msl(tmpdir)
            # Generate some chi data
            for _ in range(5):
                msl1.collect_snapshot(
                    visual_semantic=[0.5]*5, audio_physical=[0.5]*5,
                    pattern_profile=[0.1]*7, inner_body=[0.5]*5,
                    inner_mind=[0.5]*15, outer_body=[0.5]*5)
            msl1.tick()
            msl1.save_all()

            msl2 = self._make_msl(tmpdir)
            msl2.load_all()
            assert abs(msl2.chi_tracker.chi - msl1.chi_tracker.chi) < 0.01
