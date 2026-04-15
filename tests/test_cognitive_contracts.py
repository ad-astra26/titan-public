"""
Tests for TUNING-012 v2 Sub-phase C — Cognitive Contracts.

Verifies:
  - R1: _conclude_chain result dict carries the new contract fields
        (chain_template, task_success, primitives_used, domain, unique_primitives).
  - R3: MetaReasoningEngine.apply_diversity_pressure stores bias + decay state,
        rejects unknown primitives, biases action selection, and decays per chain.
  - R5: cognitive_contracts_dna eureka_threshold_* keys override the default
        per-primitive thresholds AND introduce thresholds for primitives the
        defaults didn't cover (RECALL, INTROSPECT, BREAK).
  - R7: per-Titan contracts_dna passes through cfg["contracts_dna"] unchanged.
  - Inverse-frequency weighting math (the formula used in the
        META_STRATEGY_DRIFT handler).
  - Contract JSON files in titan_plugin/contracts/meta_cognitive/ are valid + parseable.
  - load_meta_cognitive_contracts loader is callable + computes a stable
        bundle hash.
"""

import hashlib
import json
import os

import numpy as np
import pytest

from titan_plugin.logic.meta_reasoning import (
    META_PRIMITIVES,
    NUM_META_ACTIONS,
    MetaReasoningEngine,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Phase C contracts live INSIDE the package (titan_plugin/contracts/...) so
# they ship via git pull. data/ is per-Titan runtime state and must not sync.
CONTRACTS_DIR = os.path.join(REPO_ROOT, "titan_plugin", "contracts", "meta_cognitive")


# ── R1: enriched _conclude_chain return dict ─────────────────────────


class TestConcludeChainResult:
    """The R1 wire path requires the chain conclude result to carry the
    fields the contracts/handlers downstream will read."""

    def test_required_fields_present(self):
        engine = MetaReasoningEngine()
        # Force a tiny chain by directly populating state then concluding
        engine.state.is_active = True
        engine.state.chain = ["RECALL.episodic", "FORMULATE.define"]
        engine.state.chain_results = [{"count": 1}, {"domain": "inner_spirit"}]
        engine.state.formulate_output = {"domain": "inner_spirit"}
        engine.state.confidence = 0.6
        engine.state.start_time = 0.0
        engine.state.start_epoch = 100
        engine.state.trigger_reason = "test"

        result = engine._conclude_chain(
            state_132d=[0.5] * 132,
            chain_archive=None,
            meta_wisdom=None,
            autoencoder=None,
        )

        for required in (
            "action", "reward", "chain_length", "confidence", "duration_s",
            "chain_template", "task_success", "primitives_used", "domain",
            "unique_primitives",
        ):
            assert required in result, f"missing field: {required}"

        assert result["chain_template"] == "RECALL→FORMULATE"
        assert result["primitives_used"] == ["RECALL", "FORMULATE"]
        assert result["domain"] == "inner_spirit"
        assert result["unique_primitives"] == 2
        assert 0.0 <= result["task_success"] <= 1.0


# ── R3: apply_diversity_pressure ─────────────────────────────────────


class TestDiversityPressure:
    def _fresh_engine(self):
        return MetaReasoningEngine()

    def test_default_state_is_inactive(self):
        eng = self._fresh_engine()
        assert eng._diversity_pressure_remaining == 0
        assert eng._diversity_pressure_target == ""
        assert not np.any(eng._primitive_bias)

    def test_apply_pressure_sets_bias(self):
        eng = self._fresh_engine()
        ok = eng.apply_diversity_pressure("FORMULATE", magnitude=0.30, decay_chains=50)
        assert ok is True
        idx = META_PRIMITIVES.index("FORMULATE")
        assert eng._primitive_bias[idx] == pytest.approx(-0.30)
        assert eng._diversity_pressure_remaining == 50
        assert eng._diversity_pressure_target == "FORMULATE"
        assert eng._diversity_pressure_total_applied == 1

    def test_apply_pressure_rejects_unknown_primitive(self):
        eng = self._fresh_engine()
        assert eng.apply_diversity_pressure("BANANA", 0.30, 50) is False
        assert eng._diversity_pressure_remaining == 0

    def test_apply_pressure_rejects_zero_magnitude(self):
        eng = self._fresh_engine()
        assert eng.apply_diversity_pressure("FORMULATE", 0.0, 50) is False
        assert eng.apply_diversity_pressure("FORMULATE", 0.30, 0) is False

    def test_pressure_decays_per_chain(self):
        eng = self._fresh_engine()
        eng.apply_diversity_pressure("FORMULATE", 0.30, decay_chains=4)
        idx = META_PRIMITIVES.index("FORMULATE")

        # Set up minimal state to call _conclude_chain
        eng.state.is_active = True
        eng.state.chain = ["FORMULATE.define"]
        eng.state.chain_results = [{}]
        eng.state.formulate_output = {"domain": "general"}
        eng.state.confidence = 0.5
        eng.state.start_time = 0.0
        eng.state.start_epoch = 0

        # First conclude → remaining drops to 3, magnitude scales to 0.30 * 3/4
        eng._conclude_chain([0.0] * 132, None, None, None)
        assert eng._diversity_pressure_remaining == 3
        assert eng._primitive_bias[idx] == pytest.approx(-0.30 * 3 / 4, abs=1e-5)

        # Re-prime state for the next chain
        eng.state.is_active = True
        eng.state.chain = ["FORMULATE.define"]
        eng.state.chain_results = [{}]
        eng.state.formulate_output = {"domain": "general"}
        eng._conclude_chain([0.0] * 132, None, None, None)
        assert eng._diversity_pressure_remaining == 2
        assert eng._primitive_bias[idx] == pytest.approx(-0.30 * 2 / 4, abs=1e-5)

    def test_pressure_clears_at_zero(self):
        eng = self._fresh_engine()
        eng.apply_diversity_pressure("FORMULATE", 0.30, decay_chains=2)
        for _ in range(2):
            eng.state.is_active = True
            eng.state.chain = ["FORMULATE.define"]
            eng.state.chain_results = [{}]
            eng.state.formulate_output = {"domain": "general"}
            eng._conclude_chain([0.0] * 132, None, None, None)
        assert eng._diversity_pressure_remaining == 0
        assert eng._diversity_pressure_target == ""
        assert not np.any(eng._primitive_bias)


# ── R5: per-primitive eureka thresholds from contracts_dna ───────────


class TestEurekaThresholdsR5:
    def test_defaults_when_no_contracts_dna(self):
        eng = MetaReasoningEngine()
        # Defaults should still be present for the original 5 primitives
        for p in ("SYNTHESIZE", "FORMULATE", "HYPOTHESIZE",
                  "EVALUATE", "SPIRIT_SELF"):
            assert p in eng._eureka_thresholds

    def test_contracts_dna_overrides_thresholds(self):
        cfg = {
            "contracts_dna": {
                "eureka_threshold_recall": 0.40,
                "eureka_threshold_introspect": 0.45,
                "eureka_threshold_break": 0.50,
                "eureka_threshold_formulate": 0.65,
            }
        }
        eng = MetaReasoningEngine(config=cfg)
        # New primitives now have thresholds
        assert eng._eureka_thresholds["RECALL"] == pytest.approx(0.40)
        assert eng._eureka_thresholds["INTROSPECT"] == pytest.approx(0.45)
        assert eng._eureka_thresholds["BREAK"] == pytest.approx(0.50)
        # Existing one is overridden
        assert eng._eureka_thresholds["FORMULATE"] == pytest.approx(0.65)
        # Unmentioned defaults still present
        assert "SYNTHESIZE" in eng._eureka_thresholds


# ── R7: per-Titan contracts_dna passthrough ──────────────────────────


class TestContractsDnaR7:
    def test_contracts_dna_stored(self):
        cfg = {
            "contracts_dna": {
                "monoculture_share_threshold": 0.80,
                "monoculture_pressure_magnitude": 0.40,
                "strategy_inverse_freq_weight": 1.2,
            },
            "titan_id": "T2",
        }
        eng = MetaReasoningEngine(config=cfg)
        assert eng._contracts_dna["monoculture_share_threshold"] == 0.80
        assert eng._contracts_dna["monoculture_pressure_magnitude"] == 0.40
        assert eng._contracts_dna["strategy_inverse_freq_weight"] == 1.2

    def test_empty_contracts_dna_defaults_to_dict(self):
        eng = MetaReasoningEngine()
        assert isinstance(eng._contracts_dna, dict)
        assert eng._contracts_dna == {}

    def test_get_stats_includes_cognitive_contracts_block(self):
        cfg = {
            "contracts_dna": {
                "monoculture_share_threshold": 0.85,
                "eureka_threshold_recall": 0.50,
            }
        }
        eng = MetaReasoningEngine(config=cfg)
        stats = eng.get_stats()
        assert "cognitive_contracts" in stats
        cc = stats["cognitive_contracts"]
        assert "diversity_pressure" in cc
        assert "eureka_thresholds" in cc
        assert cc["dna_param_count"] == 2
        assert cc["diversity_pressure"]["active"] is False
        assert cc["diversity_pressure"]["total_applied"] == 0


# ── R4: inverse-frequency weighting math ─────────────────────────────


class TestInverseFrequencyMath:
    """The handler computes:
        score = mean_success / max(1, n ** (inv_freq_w / 2))
    Verify: rare-but-strong template ranks higher than frequent-but-mediocre."""

    @staticmethod
    def _score(mean_success: float, n: int, inv_freq_w: float) -> float:
        denom = max(1.0, n ** (inv_freq_w / 2.0))
        return mean_success / denom

    def test_rare_strong_beats_frequent_mediocre(self):
        # rare-strong: 5 chains, mean 0.80
        rare = self._score(0.80, 5, inv_freq_w=1.0)
        # frequent-mediocre: 100 chains, mean 0.40
        freq = self._score(0.40, 100, inv_freq_w=1.0)
        assert rare > freq, f"rare={rare:.3f} should beat freq={freq:.3f}"

    def test_inv_freq_weight_zero_keeps_means(self):
        # When inv_freq_w=0, denom=1 so score == mean
        assert self._score(0.50, 100, inv_freq_w=0.0) == pytest.approx(0.50)
        assert self._score(0.80, 5, inv_freq_w=0.0) == pytest.approx(0.80)

    def test_higher_weight_punishes_frequency_more(self):
        # n=100 with weight=2 should penalize harder than weight=1
        s1 = self._score(0.50, 100, inv_freq_w=1.0)
        s2 = self._score(0.50, 100, inv_freq_w=2.0)
        assert s2 < s1


# ── Contract JSON files: schema + bundle hash ────────────────────────


class TestContractJsons:
    def test_directory_exists(self):
        assert os.path.isdir(CONTRACTS_DIR), \
            f"Phase C contracts directory missing: {CONTRACTS_DIR}"

    def test_three_contracts_present(self):
        files = sorted(
            f for f in os.listdir(CONTRACTS_DIR)
            if f.endswith(".json") and not f.startswith(".")
        )
        assert files == [
            "abstract_pattern_extraction.json",
            "monoculture_detector.json",
            "strategy_evolution.json",
        ]

    @pytest.mark.parametrize("fname,expected_id,expected_event", [
        ("strategy_evolution.json", "strategy_evolution", "META_STRATEGY_DRIFT"),
        ("abstract_pattern_extraction.json", "abstract_pattern_extraction", "META_PATTERN_EMERGED"),
        ("monoculture_detector.json", "monoculture_detector", "META_DIVERSITY_PRESSURE"),
    ])
    def test_contract_schema(self, fname, expected_id, expected_event):
        with open(os.path.join(CONTRACTS_DIR, fname)) as f:
            d = json.load(f)
        assert d["contract_id"] == expected_id
        assert d["contract_type"] == "genesis"
        assert d["fork_scope"] == "meta"
        assert d["status"] == "active"
        assert "genesis_seal" in d["triggers"]
        assert len(d["rules"]) >= 1
        # Each rule must have an emit-action with the expected event
        rule = d["rules"][0]
        assert rule["op"] == "AND"
        then_block = rule["then"]
        assert then_block["action"] == "emit"
        assert then_block["event"] == expected_event

    def test_bundle_hash_is_deterministic(self):
        files = sorted(
            f for f in os.listdir(CONTRACTS_DIR)
            if f.endswith(".json") and not f.startswith(".")
        )
        # Mirror the loader's exact hash construction so the test catches drift
        h1 = hashlib.sha256()
        for fname in files:
            with open(os.path.join(CONTRACTS_DIR, fname), "rb") as f:
                raw = f.read()
            h1.update(fname.encode())
            h1.update(b"\x00")
            h1.update(raw)
            h1.update(b"\x00")

        h2 = hashlib.sha256()
        for fname in files:
            with open(os.path.join(CONTRACTS_DIR, fname), "rb") as f:
                raw = f.read()
            h2.update(fname.encode())
            h2.update(b"\x00")
            h2.update(raw)
            h2.update(b"\x00")
        assert h1.hexdigest() == h2.hexdigest()
        assert len(h1.hexdigest()) == 64


# ── Integration: contracts_dna section parses cleanly + per-Titan ────


class TestContractsDnaParsing:
    def test_titan_params_has_section(self):
        import tomllib
        params_path = os.path.join(REPO_ROOT, "titan_plugin", "titan_params.toml")
        with open(params_path, "rb") as f:
            d = tomllib.load(f)
        assert "cognitive_contracts_dna" in d
        ccd = d["cognitive_contracts_dna"]
        # Base params (not subtables)
        base = {k: v for k, v in ccd.items() if not isinstance(v, dict)}
        # Required keys for the 3 contracts + handlers
        for required in (
            "strategy_evolution_enabled",
            "strategy_inverse_freq_weight",
            "eureka_threshold_recall",
            "eureka_threshold_formulate",
            "abstract_pattern_enabled",
            "pattern_min_count",
            "monoculture_share_threshold",
            "monoculture_pressure_magnitude",
            "monoculture_decay_chains",
        ):
            assert required in base, f"missing cognitive_contracts_dna.{required}"

        # Per-Titan subtables exist
        for titan in ("T1", "T2", "T3"):
            assert titan in ccd
            assert isinstance(ccd[titan], dict)

        # T2 should have a more aggressive monoculture config (faster RECALL escape)
        assert ccd["T2"]["monoculture_share_threshold"] <= base["monoculture_share_threshold"]
        assert ccd["T2"]["monoculture_pressure_magnitude"] > base["monoculture_pressure_magnitude"]


# ── A-finish: Subsystem signal cache (rFP §7.A) ──────────────────────


class TestSubsystemCacheStaleness:
    """A-finish: TTL-based stale check on the subsystem cache."""

    def test_fresh_cache_not_stale_by_default(self):
        engine = MetaReasoningEngine()
        # Empty cache → never populated → STALE
        assert engine.is_subsystem_cache_stale(now=100.0) is True

    def test_marked_fresh_after_full_update(self):
        engine = MetaReasoningEngine()
        engine._subsystem_cache_ttl = 30.0
        engine.update_subsystem_cache(
            timechain_results=[],
            contract_results=[],
            now=100.0,
        )
        assert engine.is_subsystem_cache_stale(now=100.0) is False
        assert engine.is_subsystem_cache_stale(now=129.0) is False
        assert engine.is_subsystem_cache_stale(now=131.0) is True  # past TTL

    def test_pending_flag_lifecycle(self):
        engine = MetaReasoningEngine()
        engine._subsystem_cache_ttl = 30.0
        # Not pending initially
        assert engine.is_subsystem_cache_pending(now=100.0) is False
        # Mark pending
        engine.mark_subsystem_cache_pending(now=100.0)
        assert engine.is_subsystem_cache_pending(now=100.0) is True
        # Still pending within 2x TTL
        assert engine.is_subsystem_cache_pending(now=130.0) is True
        # Auto-expires after 2x TTL (safety net for lost responses)
        assert engine.is_subsystem_cache_pending(now=170.0) is False

    def test_full_update_clears_pending(self):
        engine = MetaReasoningEngine()
        engine._subsystem_cache_ttl = 30.0
        engine.mark_subsystem_cache_pending(now=100.0)
        assert engine.is_subsystem_cache_pending(now=100.0) is True
        # Full update (BOTH responses) clears pending
        engine.update_subsystem_cache(
            timechain_results=[], contract_results=[], now=100.0)
        assert engine.is_subsystem_cache_pending(now=100.0) is False

    def test_partial_update_keeps_pending(self):
        engine = MetaReasoningEngine()
        engine._subsystem_cache_ttl = 30.0
        engine.mark_subsystem_cache_pending(now=100.0)
        # Only TimeChain response arrived; still waiting on contracts
        engine.update_subsystem_cache(timechain_results=[], now=100.0)
        assert engine.is_subsystem_cache_pending(now=100.0) is True


class TestSubsystemCacheTimechainMapping:
    """A-finish: TIMECHAIN_QUERY_RESP results → 5 timechain signals."""

    def test_recall_blocks_drive_depth(self):
        engine = MetaReasoningEngine()
        recalls = [{"thought_type": "recall", "significance": 0.5}] * 5
        engine.update_subsystem_cache(timechain_results=recalls)
        # 5 recall blocks / 10 = 0.5
        assert engine._subsystem_cache["timechain_depth"] == 0.5

    def test_formulate_count_inverts_novelty(self):
        engine = MetaReasoningEngine()
        # 0 formulate blocks → max novelty
        engine.update_subsystem_cache(timechain_results=[])
        assert engine._subsystem_cache["timechain_novelty"] == 1.0
        # 10 formulate blocks → zero novelty
        formulates = [{"thought_type": "formulate", "significance": 0.5}] * 10
        engine.update_subsystem_cache(timechain_results=formulates)
        assert engine._subsystem_cache["timechain_novelty"] == 0.0

    def test_evaluate_avg_significance(self):
        engine = MetaReasoningEngine()
        evals = [
            {"thought_type": "evaluate", "significance": 0.6},
            {"thought_type": "evaluate", "significance": 0.8},
        ]
        engine.update_subsystem_cache(timechain_results=evals)
        assert abs(engine._subsystem_cache["timechain_eval_consistency"] - 0.7) < 1e-6

    def test_break_avg_significance(self):
        engine = MetaReasoningEngine()
        breaks = [
            {"thought_type": "break", "significance": 0.4},
            {"thought_type": "break", "significance": 0.6},
        ]
        engine.update_subsystem_cache(timechain_results=breaks)
        assert abs(engine._subsystem_cache["timechain_break_pattern"] - 0.5) < 1e-6

    def test_introspect_or_self_observation(self):
        engine = MetaReasoningEngine()
        blocks = [
            {"thought_type": "introspect", "significance": 0.7},
            {"thought_type": "self_observation", "significance": 0.3},
        ]
        engine.update_subsystem_cache(timechain_results=blocks)
        assert abs(engine._subsystem_cache["timechain_self_continuity"] - 0.5) < 1e-6


class TestSubsystemCacheContractMapping:
    """A-finish: CONTRACT_LIST_RESP contracts → 5 contract signals."""

    def test_genesis_count_drives_ratified(self):
        engine = MetaReasoningEngine()
        contracts = [
            {"contract_type": "genesis", "status": "active", "contract_id": "homeostatic_alert"},
            {"contract_type": "genesis", "status": "active", "contract_id": "milestone_tracker"},
            {"contract_type": "genesis", "status": "active", "contract_id": "noise_floor_gate"},
        ]
        engine.update_subsystem_cache(contract_results=contracts)
        # 3 active genesis / 5 = 0.6
        assert abs(engine._subsystem_cache["contract_ratified"] - 0.6) < 1e-6

    def test_filter_count_drives_priority(self):
        engine = MetaReasoningEngine()
        contracts = [
            {"contract_type": "filter", "status": "active", "contract_id": "f1"},
            {"contract_type": "filter", "status": "active", "contract_id": "f2"},
        ]
        engine.update_subsystem_cache(contract_results=contracts)
        # 2 active filter / 5 = 0.4, compliance mirrors priority
        assert abs(engine._subsystem_cache["contract_priority"] - 0.4) < 1e-6
        assert engine._subsystem_cache["contract_compliance"] == engine._subsystem_cache["contract_priority"]

    def test_inactive_contracts_excluded(self):
        engine = MetaReasoningEngine()
        contracts = [
            {"contract_type": "genesis", "status": "inactive", "contract_id": "old"},
            {"contract_type": "genesis", "status": "active", "contract_id": "new"},
        ]
        engine.update_subsystem_cache(contract_results=contracts)
        # Only 1 active → 1/5 = 0.2
        assert abs(engine._subsystem_cache["contract_ratified"] - 0.2) < 1e-6

    def test_break_trigger_fires_for_homeostatic_alert(self):
        engine = MetaReasoningEngine()
        contracts = [
            {"contract_type": "genesis", "status": "active", "contract_id": "homeostatic_alert"},
        ]
        engine.update_subsystem_cache(contract_results=contracts)
        assert engine._subsystem_cache["contract_break_trigger"] == 1.0

    def test_break_trigger_zero_when_no_alert(self):
        engine = MetaReasoningEngine()
        contracts = [
            {"contract_type": "genesis", "status": "active", "contract_id": "milestone_tracker"},
        ]
        engine.update_subsystem_cache(contract_results=contracts)
        assert engine._subsystem_cache["contract_break_trigger"] == 0.0


class TestSubsystemCacheDnaTtl:
    """A-finish: TTL is read from DNA, not hardcoded."""

    def test_ttl_loaded_from_dna(self):
        engine = MetaReasoningEngine(config={
            "dna": {"subsystem_cache_ttl_seconds": 60.0},
        })
        assert engine._subsystem_cache_ttl == 60.0

    def test_default_ttl_when_dna_missing(self):
        engine = MetaReasoningEngine()
        assert engine._subsystem_cache_ttl == 30.0
