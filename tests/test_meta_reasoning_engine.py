"""
Tests for Meta-Reasoning Engine (M4-M6):
  - M4: 6 primitives with sub-modes
  - M5: MetaPolicy + SubModePolicy selection
  - M6: Meta-reward computation + transition buffer
"""

import os
import random
import tempfile

import numpy as np
import pytest


class MockReasoningEngine:
    """Mock for testing DELEGATE integration."""
    def __init__(self):
        self._total_chains = 100
        self._total_conclusions = 45
        self.confidence = 0.6
        self.gut_agreement = 0.5
        self._strategy_bias = None

    def set_strategy_bias(self, bias):
        self._strategy_bias = bias

    def clear_strategy_bias(self):
        self._strategy_bias = None


class MockPiMonitor:
    def __init__(self, age=10):
        self.developmental_age = age


class MockCoordinator:
    def __init__(self):
        self.inner = type('obj', (object,), {'cycle_count': 5})()


@pytest.fixture
def engine(tmp_path):
    from titan_plugin.logic.meta_reasoning import MetaReasoningEngine
    return MetaReasoningEngine(config={
        "save_dir": str(tmp_path),
        "max_chain_length": 10,
        "gate_reasoning_chains": 10,
        "gate_pi_clusters": 1,
        "gate_dream_cycles": 1,
        "trigger_periodic_interval": 10,
    })


@pytest.fixture
def state_132d():
    return [random.random() for _ in range(132)]


@pytest.fixture
def neuromods():
    return {"DA": 0.6, "5HT": 0.5, "NE": 0.4, "ACh": 0.5, "Endorphin": 0.5, "GABA": 0.3}


# ── M5: Policy Networks ──────────────────────────────────────────

class TestMetaPolicy:
    def test_select_action_valid_range(self):
        from titan_plugin.logic.meta_reasoning import MetaPolicy, NUM_META_ACTIONS
        policy = MetaPolicy()
        inp = [random.random() for _ in range(80)]
        action = policy.select_action(inp)
        # Original test asserted `< 6`, but NUM_META_ACTIONS grew to 9 when
        # INTROSPECT was added (commit 63c26f6). The assert needs to track
        # the constant, not a hardcoded number.
        assert 0 <= action < NUM_META_ACTIONS

    def test_train_step(self):
        from titan_plugin.logic.meta_reasoning import MetaPolicy
        policy = MetaPolicy()
        inp = [random.random() for _ in range(80)]
        loss = policy.train_step(np.array(inp), 2, 0.5)
        assert loss >= 0
        assert policy.total_updates == 1

    def test_save_load_roundtrip(self, tmp_path):
        from titan_plugin.logic.meta_reasoning import MetaPolicy
        p1 = MetaPolicy()
        inp = [random.random() for _ in range(80)]
        a1 = p1.forward(np.array(inp, dtype=np.float32))
        p1.save(str(tmp_path / "test_policy.json"))

        p2 = MetaPolicy()
        p2.load(str(tmp_path / "test_policy.json"))
        a2 = p2.forward(np.array(inp, dtype=np.float32))
        np.testing.assert_allclose(a1, a2, atol=1e-5)


class TestSubModePolicy:
    def test_select_action(self):
        from titan_plugin.logic.meta_reasoning import SubModePolicy
        sp = SubModePolicy(n_modes=4)
        inp = [random.random() for _ in range(30)]
        action = sp.select_action(inp)
        assert 0 <= action < 4

    def test_train_step(self):
        from titan_plugin.logic.meta_reasoning import SubModePolicy
        sp = SubModePolicy(n_modes=3)
        inp = [random.random() for _ in range(30)]
        loss = sp.train_step(inp, 1, 0.3)
        assert loss >= 0

    def test_to_from_dict(self):
        from titan_plugin.logic.meta_reasoning import SubModePolicy
        sp1 = SubModePolicy(n_modes=3)
        d = sp1.to_dict()
        sp2 = SubModePolicy(n_modes=3)
        sp2.from_dict(d)
        np.testing.assert_allclose(sp1.w1, sp2.w1)


# ── M6: Transition Buffer ────────────────────────────────────────

class TestMetaTransitionBuffer:
    def test_record_and_sample(self):
        from titan_plugin.logic.meta_reasoning import MetaTransitionBuffer
        buf = MetaTransitionBuffer(max_size=100)
        for i in range(20):
            buf.record([0.5] * 80, i % 6, i % 3, 0.1 * i)
        assert buf.size() == 20
        batch = buf.sample(batch_size=10)
        assert batch is not None
        assert len(batch[0]) == 10  # states

    def test_max_size(self):
        from titan_plugin.logic.meta_reasoning import MetaTransitionBuffer
        buf = MetaTransitionBuffer(max_size=10)
        for i in range(20):
            buf.record([0.5] * 80, 0, 0, 0.1)
        assert buf.size() == 10

    def test_save_load(self, tmp_path):
        from titan_plugin.logic.meta_reasoning import MetaTransitionBuffer
        buf = MetaTransitionBuffer()
        for i in range(5):
            buf.record([0.1 * i] * 80, i % 6, i % 3, 0.1)
        buf.save(str(tmp_path / "buf.json"))

        buf2 = MetaTransitionBuffer()
        buf2.load(str(tmp_path / "buf.json"))
        assert buf2.size() == 5


# ── M4: Meta-Reasoning Engine ────────────────────────────────────

class TestMetaReasoningEngine:
    def test_gates_passed(self, engine):
        mock_re = MockReasoningEngine()
        mock_pi = MockPiMonitor(age=10)
        mock_coord = MockCoordinator()
        assert engine.gates_passed(mock_pi, mock_re, mock_coord) is True

    def test_gates_fail_low_chains(self, engine):
        mock_re = MockReasoningEngine()
        mock_re._total_chains = 5  # Below gate
        assert engine.gates_passed(MockPiMonitor(), mock_re, MockCoordinator()) is False

    def test_tick_idle_no_trigger(self, engine, state_132d, neuromods):
        mock_re = MockReasoningEngine()
        mock_re._total_chains = 97  # Not a multiple of periodic interval
        mock_re._total_conclusions = 80  # High commit rate = no trigger
        result = engine.tick(state_132d, neuromods, mock_re, None, None, None, None)
        assert result["action"] == "IDLE"

    def test_tick_triggers_on_low_commit(self, engine, state_132d, neuromods, tmp_path):
        from titan_plugin.logic.chain_archive import ChainArchive
        archive = ChainArchive(db_path=str(tmp_path / "test.db"))

        mock_re = MockReasoningEngine()
        mock_re._total_chains = 100
        mock_re._total_conclusions = 20  # 20% commit rate → trigger

        result = engine.tick(state_132d, neuromods, mock_re, archive, None, None, None)
        assert result["action"] == "CONTINUE"
        assert engine.state.is_active is True
        assert "low_commit_rate" in engine.state.trigger_reason

    def test_full_chain_lifecycle(self, engine, state_132d, neuromods, tmp_path):
        from titan_plugin.logic.chain_archive import ChainArchive
        from titan_plugin.logic.meta_wisdom import MetaWisdomStore

        archive = ChainArchive(db_path=str(tmp_path / "test.db"))
        wisdom = MetaWisdomStore(db_path=str(tmp_path / "test.db"))
        mock_re = MockReasoningEngine()
        mock_re._total_chains = 100
        mock_re._total_conclusions = 20

        # Run chain until conclusion
        results = []
        for _ in range(25):  # max_steps + buffer
            result = engine.tick(state_132d, neuromods, mock_re, archive, wisdom, None, None)
            results.append(result)
            if result["action"] in ("CONCLUDE", "IDLE"):
                break
            # If DELEGATE waiting, simulate main reasoning completing
            if result.get("action") == "WAITING":
                mock_re._total_chains += 1

        # Should have concluded
        actions = [r["action"] for r in results]
        assert "CONCLUDE" in actions or len(results) >= 10

    def test_delegate_sets_bias(self, engine, state_132d, neuromods):
        mock_re = MockReasoningEngine()
        # Manually start chain and set up for DELEGATE
        engine._start_chain("test", state_132d)
        engine.state.formulate_output = {"domain": "general", "problem_template": "test"}
        engine.state.hypotheses = [{"strategy": ["DECOMPOSE", "COMPARE"], "predicted_confidence": 0.7}]

        result = engine._prim_delegate("full_chain", mock_re)
        assert result["delegated"] is True
        assert mock_re._strategy_bias is not None
        assert engine.state.awaiting_delegate is True

    def test_consolidate_training(self, engine, state_132d, neuromods, tmp_path):
        from titan_plugin.logic.chain_archive import ChainArchive
        archive = ChainArchive(db_path=str(tmp_path / "test.db"))
        mock_re = MockReasoningEngine()
        mock_re._total_chains = 100
        mock_re._total_conclusions = 20

        # Run a few steps to populate buffer
        for _ in range(5):
            engine.tick(state_132d, neuromods, mock_re, archive, None, None, None)
            if engine.state.awaiting_delegate:
                mock_re._total_chains += 1

        # Need enough samples for training
        for _ in range(20):
            engine.buffer.record([random.random() for _ in range(80)],
                                 random.randint(0, 5), random.randint(0, 2), 0.1)

        result = engine.consolidate_training(boost_factor=2.0)
        assert result["trained"] is True
        assert result["samples"] > 0

    def test_save_load_all(self, engine, tmp_path):
        engine._total_meta_chains = 5
        engine._total_meta_steps = 42
        engine.save_all()

        from titan_plugin.logic.meta_reasoning import MetaReasoningEngine
        engine2 = MetaReasoningEngine(config={"save_dir": str(tmp_path)})
        assert engine2._total_meta_chains == 5
        assert engine2._total_meta_steps == 42


class TestTriggerConditions:
    def test_low_commit_rate(self):
        from titan_plugin.logic.meta_reasoning import should_trigger_meta
        mock_re = MockReasoningEngine()
        mock_re._total_chains = 100
        mock_re._total_conclusions = 20
        should, reason = should_trigger_meta(mock_re, {}, None, {})
        assert should is True
        assert "low_commit_rate" in reason

    def test_high_reflection(self):
        from titan_plugin.logic.meta_reasoning import should_trigger_meta
        mock_re = MockReasoningEngine()
        mock_re._total_chains = 100
        mock_re._total_conclusions = 80
        should, reason = should_trigger_meta(mock_re, {"REFLECTION": 0.85}, None,
                                             {"trigger_reflection_threshold": 0.75})
        assert should is True
        assert "high_reflection" in reason

    def test_periodic(self):
        from titan_plugin.logic.meta_reasoning import should_trigger_meta
        mock_re = MockReasoningEngine()
        mock_re._total_chains = 50
        mock_re._total_conclusions = 40
        should, reason = should_trigger_meta(mock_re, {}, None,
                                             {"trigger_periodic_interval": 50})
        assert should is True
        assert "periodic" in reason

    def test_no_trigger(self):
        from titan_plugin.logic.meta_reasoning import should_trigger_meta
        mock_re = MockReasoningEngine()
        mock_re._total_chains = 97  # Not a multiple of 50 (periodic interval)
        mock_re._total_conclusions = 80
        should, _ = should_trigger_meta(mock_re, {}, None, {})
        assert should is False


class TestMetaReward:
    def test_compute_reward_with_delegation(self, engine):
        engine.state.is_active = True
        engine.state.chain = ["FORMULATE.define", "RECALL.chain_archive", "DELEGATE.full_chain"]
        engine.state.delegate_results = [{"confidence": 0.8, "gut_agreement": 0.7}]
        engine.state.hypotheses = [{"predicted_confidence": 0.75, "strategy": ["COMPARE"]}]
        engine.state.synthesized = {"insight": "test insight"}
        engine.state.confidence = 0.7
        engine._baseline_confidence = 0.5

        reward = engine._compute_meta_reward()
        assert reward > 0.0
        assert reward <= 1.0

    def test_empty_chain_reward(self, engine):
        engine.state.is_active = True
        engine.state.chain = ["FORMULATE.define"]
        reward = engine._compute_meta_reward()
        assert reward >= 0.0


class TestStrategyBiasIntegration:
    def test_reasoning_policy_accepts_bias(self):
        from titan_plugin.logic.reasoning import ReasoningPolicyNet
        policy = ReasoningPolicyNet(input_dim=115)
        x = np.random.randn(115).astype(np.float32)
        bias = np.array([2.0, 0, 0, 0, 1.0, 0, 0, -1.0], dtype=np.float32)

        # Without bias
        action1_scores = policy.forward(x).copy()

        # With bias — scores should shift
        action2 = policy.select_action(x, strategy_bias=bias)
        # Can't deterministically test action selection, but verify no crash
        assert 0 <= action2 < 8


# ── Phase D.1 — External Reward Routing (META_LANGUAGE loop) ─────────


class _FakeChainIQL:
    """Minimal chain_iql stand-in for routing tests.

    Records calls to apply_external_reward and returns configurable result.
    """

    def __init__(self, return_value=True):
        self.enabled = True
        self.calls = []
        self._return_value = return_value

    def apply_external_reward(self, chain_id, external_reward, alpha):
        self.calls.append({
            "chain_id": chain_id,
            "external_reward": external_reward,
            "alpha": alpha,
        })
        return self._return_value


class TestExternalRewardRouting:
    """Phase D.1: meta_engine.add_external_reward forwards to chain_iql."""

    def test_chain_id_assigned_monotonically(self, engine, state_132d):
        """Two consecutive _start_chain calls → chain_id 1 then 2."""
        engine._start_chain("reason_a", state_132d)
        id_a = engine.state.chain_id
        # Simulate conclude (reset state)
        engine.state = engine.state.__class__()
        engine._start_chain("reason_b", state_132d)
        id_b = engine.state.chain_id
        assert id_a == 1
        assert id_b == 2
        assert engine._next_chain_id == 3

    def test_dna_external_reward_blend_alpha_loaded(self, tmp_path):
        """external_reward_blend_alpha from DNA lands on the engine."""
        from titan_plugin.logic.meta_reasoning import MetaReasoningEngine
        eng = MetaReasoningEngine(config={
            "save_dir": str(tmp_path),
            "dna": {"external_reward_blend_alpha": 0.7},
        })
        assert abs(eng._external_reward_blend_alpha - 0.7) < 1e-9

    def test_dna_external_reward_blend_alpha_defaults_to_half(self, tmp_path):
        from titan_plugin.logic.meta_reasoning import MetaReasoningEngine
        eng = MetaReasoningEngine(config={"save_dir": str(tmp_path)})
        assert abs(eng._external_reward_blend_alpha - 0.5) < 1e-9

    def test_add_external_reward_routes_to_iql_with_dna_alpha(self, engine):
        """Engine passes DNA alpha when forwarding to chain_iql."""
        engine._external_reward_blend_alpha = 0.42
        fake = _FakeChainIQL(return_value=True)
        engine._chain_iql = fake
        applied = engine.add_external_reward(
            chain_id=17, external_reward=0.75)
        assert applied is True
        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert call["chain_id"] == 17
        assert abs(call["external_reward"] - 0.75) < 1e-9
        assert abs(call["alpha"] - 0.42) < 1e-9

    def test_add_external_reward_late_drop_returns_false(self, engine):
        """When chain_iql reports late-drop, engine returns False cleanly."""
        fake = _FakeChainIQL(return_value=False)
        engine._chain_iql = fake
        applied = engine.add_external_reward(
            chain_id=99, external_reward=0.5)
        assert applied is False
        # Still routed the call
        assert len(fake.calls) == 1

    def test_add_external_reward_no_iql_returns_false(self, engine):
        """If chain_iql missing/disabled, add_external_reward is a no-op False."""
        engine._chain_iql = None
        assert engine.add_external_reward(
            chain_id=1, external_reward=0.5) is False

    def test_chain_id_appears_in_state(self, engine, state_132d):
        """After _start_chain, the state carries the chain_id."""
        engine._start_chain("trigger", state_132d)
        assert engine.state.chain_id > 0
        assert engine.state.is_active is True
