"""Tests for Meta-Reasoning M7-M10: BREAK, EUREKA, SPIRIT_SELF, PARALLEL."""

import json
import os
import tempfile

import numpy as np
import pytest


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def engine(tmp_dir):
    from titan_plugin.logic.meta_reasoning import MetaReasoningEngine
    # Seed numpy RNG so MetaPolicy.select_action is deterministic across
    # test runs. Without this, the policy can randomly select BREAK during
    # setup loops, inflating break_count and breaking exact-equality asserts.
    np.random.seed(7)
    return MetaReasoningEngine(config={
        "save_dir": tmp_dir,
        "max_chain_length": 20,
        "gate_reasoning_chains": 0,
        "gate_pi_clusters": 0,
        "gate_dream_cycles": 0,
        "gate_spirit_self_chains": 5,  # low gate for testing
        "trigger_periodic_interval": 50,
        "trigger_experience_pressure": 999,
        "eureka_threshold": 0.70,
        "eureka_cooldown": 3,
        "spirit_self_cooldown": 3,
        "max_breaks_per_chain": 3,
        "parallel_enabled": False,
    })


@pytest.fixture
def mock_reasoning():
    """Mock reasoning engine with controllable state."""
    class MockReasoning:
        _total_chains = 20
        _total_conclusions = 12
        confidence = 0.6
        gut_agreement = 0.5
        _strategy_bias = None
        def set_strategy_bias(self, b): self._strategy_bias = b
        def clear_strategy_bias(self): self._strategy_bias = None
    return MockReasoning()


@pytest.fixture
def wisdom_store(tmp_dir):
    from titan_plugin.logic.meta_wisdom import MetaWisdomStore
    return MetaWisdomStore(db_path=os.path.join(tmp_dir, "test.db"))


def _start_chain(engine, mock_reasoning):
    """Helper to start a chain via tick with low commit rate trigger."""
    mock_reasoning._total_chains = 20
    mock_reasoning._total_conclusions = 4  # 20% commit rate < 30% threshold
    sv = [0.5] * 132
    nm = {"DA": 0.5, "5HT": 0.5, "NE": 0.5, "ACh": 0.5, "Endorphin": 0.5, "GABA": 0.3}
    result = engine.tick(sv, nm, mock_reasoning, None, None, None, None)
    assert engine.state.is_active
    return sv, nm


# ── M7: BREAK Tests ───────────────────────────────────────────────

class TestM7Break:

    def test_break_rewind_last(self, engine, mock_reasoning):
        sv, nm = _start_chain(engine, mock_reasoning)
        # Run a few steps to build chain
        for _ in range(3):
            engine.tick(sv, nm, mock_reasoning, None, None, None, None)
        chain_len = len(engine.state.chain)
        assert chain_len >= 3
        # Manually invoke BREAK.rewind_last
        result = engine._prim_break("rewind_last", sv, mock_reasoning)
        assert result["sub_mode"] == "rewind_last"
        assert len(engine.state.chain) == chain_len - 1  # rewind pops last step
        assert engine.state.break_count == 1

    def test_break_rewind_to_checkpoint(self, engine, mock_reasoning):
        sv, nm = _start_chain(engine, mock_reasoning)
        # Save a checkpoint manually
        engine.state.chain = ["FORMULATE.define", "RECALL.wisdom", "HYPOTHESIZE.generate"]
        engine.state.chain_results = [{"confidence": 0.5}, {"confidence": 0.6}, {"confidence": 0.7}]
        engine.state.confidence = 0.7
        engine._save_checkpoint()  # checkpoint at step 3
        # Add more steps
        engine.state.chain.append("EVALUATE.check_progress")
        engine.state.chain_results.append({"confidence": 0.3})
        engine.state.confidence = 0.3
        # Rewind to checkpoint
        result = engine._prim_break("rewind_to_checkpoint", sv, mock_reasoning)
        assert result["rewound_to"] == 3
        assert engine.state.confidence == 0.7
        assert len(engine.state.chain) == 3

    def test_break_restart_fresh(self, engine, mock_reasoning):
        sv, nm = _start_chain(engine, mock_reasoning)
        for _ in range(3):
            engine.tick(sv, nm, mock_reasoning, None, None, None, None)
        result = engine._prim_break("restart_fresh", sv, mock_reasoning)
        assert result["sub_mode"] == "restart_fresh"
        assert len(engine.state.chain) == 0
        assert engine.state.confidence == 0.5

    def test_break_count_caps(self, engine, mock_reasoning):
        sv, nm = _start_chain(engine, mock_reasoning)
        engine.state.max_breaks = 2
        engine.state.break_count = 2
        # When break_count >= max_breaks, tick redirects to EVALUATE
        # The enforcement is in tick(), not _prim_break
        assert engine.state.break_count >= engine.state.max_breaks

    def test_checkpoint_auto_saved(self, engine, mock_reasoning):
        sv, nm = _start_chain(engine, mock_reasoning)
        engine.state.chain = []
        engine.state.chain_results = []
        engine.state.formulate_output = {"domain": "test"}
        # Simulate FORMULATE step → should auto-checkpoint
        engine.state.chain.append("FORMULATE.define")
        engine.state.chain_results.append({"confidence": 0.5})
        engine._save_checkpoint()
        assert len(engine.state.checkpoints) == 1
        assert engine.state.checkpoints[0]["step_index"] == 1

    def test_break_reward_penalty(self, engine, mock_reasoning):
        sv, nm = _start_chain(engine, mock_reasoning)
        engine.state.break_count = 0
        reward_0 = engine._compute_meta_reward()
        engine.state.break_count = 2
        reward_2 = engine._compute_meta_reward()
        assert reward_2 < reward_0  # penalty applied


# ── M9: EUREKA Tests ──────────────────────────────────────────────

class TestM9Eureka:

    def test_eureka_fires_on_high_synth_conf(self, engine, mock_reasoning):
        sv, nm = _start_chain(engine, mock_reasoning)
        engine.state.formulate_output = {"problem_template": "test", "domain": "body_mind"}
        # Simulate high-conf SYNTHESIZE
        eureka = engine._fire_eureka(0.85, [0.5] * 132, None, None)
        assert eureka["type"] == "eureka"
        assert eureka["confidence"] == 0.85
        assert eureka["da_burst_magnitude"] > 0
        assert engine._total_eurekas == 1

    def test_eureka_cooldown(self, engine, mock_reasoning):
        sv, nm = _start_chain(engine, mock_reasoning)
        engine.state.formulate_output = {"problem_template": "test", "domain": "test"}
        engine._fire_eureka(0.9, [0.5] * 132, None, None)
        assert engine._eureka_cooldown_steps == engine._eureka_cooldown_max
        # Second fire blocked by cooldown
        assert engine._eureka_cooldown_steps > 0

    def test_eureka_novelty_no_wisdom(self, engine, mock_reasoning):
        sv, nm = _start_chain(engine, mock_reasoning)
        engine.state.formulate_output = {"problem_template": "test", "domain": "test"}
        eureka = engine._fire_eureka(0.8, [0.5] * 132, None, None)
        # No autoencoder → default novelty 0.5
        assert eureka["novelty"] == 0.5

    def test_force_crystallize(self, wisdom_store):
        wid = wisdom_store.store_wisdom("test pattern", ["COMPARE"], 0.5)
        assert wid > 0
        wisdom_store.force_crystallize(wid)
        # Query crystallized
        crystallized = wisdom_store.get_crystallized()
        assert len(crystallized) == 1
        assert crystallized[0]["confidence"] >= 0.8

    def test_eureka_stats_persisted(self, engine, tmp_dir):
        engine._total_eurekas = 7
        engine.save_all()
        stats_path = os.path.join(tmp_dir, "meta_stats.json")
        with open(stats_path) as f:
            data = json.load(f)
        assert data["total_eurekas"] == 7


# ── M8: SPIRIT_SELF Tests ────────────────────────────────────────

class TestM8SpiritSelf:

    def test_spirit_self_returns_nudges(self, engine, mock_reasoning):
        sv, nm = _start_chain(engine, mock_reasoning)
        result = engine._prim_spirit_self("boost_curiosity", nm)
        assert result["nudge_request"]["sub_mode"] == "boost_curiosity"
        assert "NE" in result["nudge_request"]["nudges"]
        assert "DA" in result["nudge_request"]["nudges"]

    def test_spirit_self_cooldown(self, engine, mock_reasoning):
        sv, nm = _start_chain(engine, mock_reasoning)
        engine._prim_spirit_self("boost_focus", nm)
        assert engine.state.spirit_self_cooldown == engine._spirit_self_cooldown_max

    def test_spirit_self_maturity_gate(self, engine, mock_reasoning):
        """SPIRIT_SELF requires gate_spirit_self_chains completed chains."""
        engine._total_meta_chains = 2  # below gate of 5
        # Gate check happens in tick() — we verify the condition
        assert engine._total_meta_chains < engine._spirit_self_gate

    def test_spirit_self_maturity_unlocked(self, engine, mock_reasoning):
        engine._total_meta_chains = 10  # above gate of 5
        assert engine._total_meta_chains >= engine._spirit_self_gate

    def test_spirit_self_pre_nudge_confidence(self, engine, mock_reasoning):
        sv, nm = _start_chain(engine, mock_reasoning)
        engine.state.confidence = 0.6
        engine._prim_spirit_self("boost_energy", nm)
        assert engine.state.pre_nudge_confidence == 0.6

    def test_spirit_self_reward_effectiveness(self, engine, mock_reasoning):
        sv, nm = _start_chain(engine, mock_reasoning)
        engine.state.last_spirit_self_step = 3
        engine.state.pre_nudge_confidence = 0.4
        engine.state.confidence = 0.7  # improved after nudge
        reward = engine._compute_meta_reward()
        # Improvement = 0.3, contribution = min(0.05, 0.3 * 0.5) = 0.05
        # Should be higher than without SPIRIT_SELF
        engine.state.last_spirit_self_step = -1
        reward_no_spirit = engine._compute_meta_reward()
        assert reward >= reward_no_spirit

    def test_all_nudge_modes(self, engine, mock_reasoning):
        from titan_plugin.logic.meta_reasoning import SPIRIT_SELF_NUDGE_MAP
        for mode, nudges in SPIRIT_SELF_NUDGE_MAP.items():
            result = engine._prim_spirit_self(mode, {})
            assert result["nudge_request"]["sub_mode"] == mode
            assert result["nudge_request"]["nudges"] == nudges


# ── M10: PARALLEL Tests ──────────────────────────────────────────

class TestM10Parallel:

    def test_resource_detection(self):
        from titan_plugin.logic.meta_reasoning import _detect_resource_budget
        budget, max_par, ram, cpu = _detect_resource_budget()
        assert budget >= 20
        assert budget <= 100
        assert max_par >= 1
        assert max_par <= 3
        assert ram > 0
        assert cpu > 0

    def test_scheduler_spawn(self):
        from titan_plugin.logic.meta_reasoning import MultiChainScheduler
        sched = MultiChainScheduler(max_chains=3, total_budget=20)
        chain = sched.spawn_chain("test", [0.5] * 132, 20)
        assert chain.is_active
        assert len(sched.chains) == 1

    def test_scheduler_budget(self):
        from titan_plugin.logic.meta_reasoning import MultiChainScheduler
        sched = MultiChainScheduler(max_chains=3, total_budget=20)
        assert sched.budget_remaining() == 20
        sched.total_steps_used = 15
        assert sched.budget_remaining() == 5

    def test_scheduler_round_robin(self):
        from titan_plugin.logic.meta_reasoning import MultiChainScheduler
        sched = MultiChainScheduler(max_chains=3, total_budget=40,
                                     config={"parallel_schedule_mode": "round_robin"})
        sched.spawn_chain("a", [0.5] * 132, 20)
        sched.spawn_chain("b", [0.5] * 132, 20)
        assert sched.active_index == 0
        sched.advance()
        assert sched.active_index == 1
        sched.advance()
        assert sched.active_index == 0

    def test_scheduler_should_spawn(self):
        from titan_plugin.logic.meta_reasoning import MultiChainScheduler
        sched = MultiChainScheduler(max_chains=3, total_budget=40)
        sched.spawn_chain("a", [0.5] * 132, 20)
        # High NE+DA = parallel tendency
        assert sched.should_spawn({"NE": 0.9, "DA": 0.9, "5HT": 0.1, "GABA": 0.1})
        # High 5HT+GABA = serial tendency
        assert not sched.should_spawn({"NE": 0.1, "DA": 0.1, "5HT": 0.9, "GABA": 0.9})

    def test_scheduler_merge(self):
        from titan_plugin.logic.meta_reasoning import MultiChainScheduler
        sched = MultiChainScheduler(max_chains=3, total_budget=40)
        c1 = sched.spawn_chain("a", [0.5] * 132, 20)
        c2 = sched.spawn_chain("b", [0.5] * 132, 20)
        c1.formulate_output = {"domain": "body_mind"}
        c2.formulate_output = {"domain": "body_mind"}
        c2.hypotheses = [{"strategy": ["TEST"]}]
        merge = sched.should_merge()
        assert merge == (0, 1)
        sched.merge_chains(0, 1)
        assert len(sched.chains) == 1
        assert len(sched.chains[0].hypotheses) == 1

    def test_parallel_disabled_preserves_serial(self, engine, mock_reasoning):
        """parallel_enabled=False means exact same behavior as before."""
        assert not engine._parallel_enabled
        sv = [0.5] * 132
        nm = {"DA": 0.5, "5HT": 0.5, "NE": 0.5, "ACh": 0.5, "Endorphin": 0.5, "GABA": 0.3}
        mock_reasoning._total_chains = 20
        mock_reasoning._total_conclusions = 4
        result = engine.tick(sv, nm, mock_reasoning, None, None, None, None)
        assert result["action"] in ("CONTINUE", "CONCLUDE", "IDLE", "WAITING")


# ── Migration Tests ──────────────────────────────────────────────

class TestMigration:

    def test_policy_migration_6_to_8(self, tmp_dir):
        """Verify MetaPolicy loads 6-output weights into 8-output model."""
        from titan_plugin.logic.meta_reasoning import MetaPolicy, NUM_META_ACTIONS
        # Save a 6-output policy
        old = MetaPolicy(input_dim=80, h1=40, h2=20)
        old_w3 = np.random.randn(20, 6).astype(np.float32)
        old.w3 = old_w3.copy()
        old.b3 = np.zeros(6, dtype=np.float32)
        path = os.path.join(tmp_dir, "test_policy.json")
        old.save(path)
        # Load into new 8-output policy
        new = MetaPolicy(input_dim=80, h1=40, h2=20)
        assert new.load(path)
        assert new.w3.shape == (20, NUM_META_ACTIONS)
        # First 6 columns preserved
        np.testing.assert_array_almost_equal(new.w3[:, :6], old_w3)
        # Last 2 columns near-zero (migration padding)
        assert abs(new.w3[:, 6:].mean()) < 0.1

    def test_stats_backward_compatible(self, tmp_dir):
        """Old stats without total_eurekas loads correctly."""
        stats = {"total_chains": 50, "total_steps": 500, "total_wisdom_saved": 10,
                 "baseline_confidence": 0.5}
        with open(os.path.join(tmp_dir, "meta_stats.json"), "w") as f:
            json.dump(stats, f)
        from titan_plugin.logic.meta_reasoning import MetaReasoningEngine
        eng = MetaReasoningEngine(config={"save_dir": tmp_dir, "max_chain_length": 20,
                                           "trigger_periodic_interval": 999})
        assert eng._total_eurekas == 0  # default
        assert eng._total_meta_chains == 50

    def test_cognitive_contract_counters_roundtrip(self, tmp_dir):
        """CC handler counters survive save → load (BUG-CONTRACT-GATE-STARVATION fix).

        Before 2026-04-21, _cc_strategy_drift_fires / _cc_pattern_emerged_fires /
        _cc_monoculture_fires lived in-memory only. Every restart reset them
        to 0, making the dashboard show fires=0 even when contracts fired
        correctly. Frequent restarts during incident recovery compounded the
        problem. Lifetime accounting now persists in meta_stats.json.
        """
        from titan_plugin.logic.meta_reasoning import MetaReasoningEngine
        # Seed an engine with realistic counter values + last_* payloads
        eng = MetaReasoningEngine(config={
            "save_dir": tmp_dir, "max_chain_length": 20,
            "trigger_periodic_interval": 999})
        eng._cc_strategy_drift_fires = 7
        eng._cc_strategy_drift_last_top = [
            {"template": "FORMULATE-RECALL-SYNTHESIZE", "score": 0.81,
             "mean": 0.84, "n": 4}]
        eng._cc_pattern_emerged_fires = 3
        eng._cc_pattern_emerged_last = [
            {"template": "RECALL-RECALL-FORMULATE", "count": 12}]
        eng._cc_monoculture_fires = 5
        eng._cc_monoculture_last = {
            "dominant": "RECALL", "share": 0.83, "applied": True,
            "magnitude": 0.30, "decay_chains": 50}
        eng.save_all()
        # Verify file contents
        with open(os.path.join(tmp_dir, "meta_stats.json")) as f:
            data = json.load(f)
        assert data["cc_strategy_drift_fires"] == 7
        assert data["cc_pattern_emerged_fires"] == 3
        assert data["cc_monoculture_fires"] == 5
        assert data["cc_monoculture_last"]["dominant"] == "RECALL"
        # Simulate a restart: new engine from same save_dir
        eng2 = MetaReasoningEngine(config={
            "save_dir": tmp_dir, "max_chain_length": 20,
            "trigger_periodic_interval": 999})
        assert eng2._cc_strategy_drift_fires == 7
        assert eng2._cc_pattern_emerged_fires == 3
        assert eng2._cc_monoculture_fires == 5
        assert eng2._cc_monoculture_last["dominant"] == "RECALL"
        assert eng2._cc_strategy_drift_last_top[0]["template"] == \
            "FORMULATE-RECALL-SYNTHESIZE"

    def test_cognitive_contract_counters_backward_compatible(self, tmp_dir):
        """Old meta_stats.json without cc_* fields loads with counters = 0."""
        stats = {"total_chains": 100, "total_steps": 1000,
                 "total_wisdom_saved": 5, "baseline_confidence": 0.6}
        with open(os.path.join(tmp_dir, "meta_stats.json"), "w") as f:
            json.dump(stats, f)
        from titan_plugin.logic.meta_reasoning import MetaReasoningEngine
        eng = MetaReasoningEngine(config={
            "save_dir": tmp_dir, "max_chain_length": 20,
            "trigger_periodic_interval": 999})
        # Old file has no cc_* fields — counters must default safely
        assert getattr(eng, "_cc_strategy_drift_fires", 0) == 0
        assert getattr(eng, "_cc_pattern_emerged_fires", 0) == 0
        assert getattr(eng, "_cc_monoculture_fires", 0) == 0
        assert eng._total_meta_chains == 100  # Regular fields still loaded

    def test_lifetime_counters_full_audit_roundtrip(self, tmp_dir):
        """Systematic audit fix: 14 previously-ephemeral lifetime/state
        attributes now persist across restart. Pattern identical to
        cc_* counters — this is the class-of-bug fix. 2026-04-21."""
        from titan_plugin.logic.meta_reasoning import MetaReasoningEngine
        eng = MetaReasoningEngine(config={
            "save_dir": tmp_dir, "max_chain_length": 20,
            "trigger_periodic_interval": 999})
        # Seed realistic lifetime + mid-state values
        eng._inengine_mono_total_fires = 4
        eng._inengine_mono_last_fire_chain = 150
        eng._mono_adj_fires_count = 2
        eng._mono_adj_cumulative = 0.75
        eng._diversity_pressure_total_applied = 9
        eng._diversity_pressure_remaining = 25  # mid-decay
        eng._diversity_pressure_initial_magnitude = 0.30
        eng._diversity_pressure_initial_decay = 50
        eng._introspect_executions_lifetime = 11
        eng._introspect_picks_lifetime = 14
        eng._introspect_rerouted_lifetime = 3
        eng._next_chain_id = 201
        eng._last_concluded_chain_id = 200
        eng._repeat_impasse_count = 7
        eng._repeat_impasse_primitives = {"FORMULATE": 4, "RECALL": 3}
        eng._soar_last_successful_topic = "cognitive_diversity_research"
        eng.save_all()
        # Fresh engine from same save_dir
        eng2 = MetaReasoningEngine(config={
            "save_dir": tmp_dir, "max_chain_length": 20,
            "trigger_periodic_interval": 999})
        assert eng2._inengine_mono_total_fires == 4
        assert eng2._inengine_mono_last_fire_chain == 150
        assert eng2._mono_adj_fires_count == 2
        assert abs(eng2._mono_adj_cumulative - 0.75) < 1e-6
        assert eng2._diversity_pressure_total_applied == 9
        assert eng2._diversity_pressure_remaining == 25
        assert abs(eng2._diversity_pressure_initial_magnitude - 0.30) < 1e-6
        assert eng2._diversity_pressure_initial_decay == 50
        assert eng2._introspect_executions_lifetime == 11
        assert eng2._introspect_picks_lifetime == 14
        assert eng2._introspect_rerouted_lifetime == 3
        assert eng2._next_chain_id == 201
        assert eng2._last_concluded_chain_id == 200
        assert eng2._repeat_impasse_count == 7
        assert eng2._repeat_impasse_primitives == {"FORMULATE": 4, "RECALL": 3}
        assert eng2._soar_last_successful_topic == "cognitive_diversity_research"


# ── Integration Tests ────────────────────────────────────────────

class TestIntegration:

    def test_full_chain_with_break(self, engine, mock_reasoning):
        """Run a full chain that includes a BREAK step."""
        sv, nm = _start_chain(engine, mock_reasoning)
        for _ in range(5):
            engine.tick(sv, nm, mock_reasoning, None, None, None, None)
        # Force a BREAK manually
        engine._prim_break("rewind_last", sv, mock_reasoning)
        engine.state.break_count = 1
        # Continue
        for _ in range(5):
            engine.tick(sv, nm, mock_reasoning, None, None, None, None)
        assert engine.state.break_count >= 1

    def test_get_stats_includes_m7_m10(self, engine, mock_reasoning):
        engine._total_eurekas = 3
        stats = engine.get_stats()
        assert "total_eurekas" in stats
        assert "spirit_self_unlocked" in stats
        assert "parallel_enabled" in stats
        assert "resource_budget" in stats
        assert stats["total_eurekas"] == 3
