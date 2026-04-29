"""Tests for TitanVM v2 Phase 3 — cross-program coupling + eligibility flag.

Covers rFP_titan_vm_v2 Phase 3 acceptance (§3.13):
  - Cross-program coupling fires for all 6 biological pairs
  - Coupling bonus scales by supervision_weight (decays to 0 post-warmup)
  - coupling_enabled=false disables coupling cleanly (no effect)
  - Partner lookup is symmetric (EMPATHY↔REFLECTION both ways)
  - eligibility_traces_enabled=false reverts to K=1, decay=1.0
  - eligibility_traces_enabled=true (default) uses Stage-0.5 params
"""
from __future__ import annotations

import pytest


# ── IMW disable fixture ─────────────────────────────────────────────
#
# NeuralNervousSystem instantiates InnerMemoryStore which pulls a
# module-level IMW client singleton configured from config.toml
# (persistence.enabled=true + mode="shadow" on this host). In a fresh
# test process with no IMW daemon socket present, client construction
# raises WriterError. Patch IMWConfig before any NNS construction.

@pytest.fixture(autouse=True)
def _disable_imw_for_tests(monkeypatch):
    from titan_plugin.persistence.config import IMWConfig
    from titan_plugin.persistence import writer_client as _wc

    def _disabled(cls):
        return cls(enabled=False, mode="disabled")

    monkeypatch.setattr(
        IMWConfig, "from_titan_config", classmethod(_disabled)
    )
    _wc.reset_client()
    yield
    _wc.reset_client()


from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def base_config(tmp_path):
    """Minimal NNS config with 11 programs pointing at tmp data_dir."""
    return {
        "enabled": True,
        "warmup_steps": 500,
        "train_every_n": 5,
        "batch_size": 16,
        "save_every_n": 999999,
        "data_dir": str(tmp_path),
        "programs": {
            "REFLEX": {"input_features": "standard"},
            "FOCUS": {"input_features": "standard"},
            "INTUITION": {"input_features": "standard"},
            "IMPULSE": {"input_features": "standard"},
            "INSPIRATION": {"input_features": "standard"},
            "METABOLISM": {"input_features": "standard"},
            "CREATIVITY": {"input_features": "standard"},
            "CURIOSITY": {"input_features": "standard"},
            "EMPATHY": {"input_features": "standard"},
            "REFLECTION": {"input_features": "standard"},
            "VIGILANCE": {"input_features": "standard"},
        },
    }


def _make_nns(config, coupling_enabled=False, eligibility_enabled=True,
              coupling_overrides=None):
    cfg = dict(config)
    cfg["coupling"] = {
        "enabled": coupling_enabled,
        "strength": 0.1,
        "fire_threshold": 0.3,
        **(coupling_overrides or {}),
    }
    cfg["eligibility"] = {"enabled": eligibility_enabled}
    return NeuralNervousSystem(config=cfg)


# ── Coupling: enabled / disabled behavior ───────────────────────────

class TestCouplingFlag:

    def test_coupling_disabled_returns_zero_bonus(self, base_config):
        nns = _make_nns(base_config, coupling_enabled=False)
        # Even with the partner firing, disabled coupling = 0 bonus
        nns._latest_vm_baselines = {"REFLEX": 0.8, "IMPULSE": 0.8}
        bonus = nns._compute_coupling_bonus("IMPULSE", supervision_weight=1.0)
        assert bonus == 0.0

    def test_coupling_enabled_but_partner_silent(self, base_config):
        nns = _make_nns(base_config, coupling_enabled=True)
        nns._latest_vm_baselines = {"REFLEX": 0.0}  # partner below threshold
        bonus = nns._compute_coupling_bonus("IMPULSE", supervision_weight=1.0)
        assert bonus == 0.0

    def test_coupling_enabled_and_partner_firing(self, base_config):
        nns = _make_nns(base_config, coupling_enabled=True)
        nns._latest_vm_baselines = {"REFLEX": 0.8, "IMPULSE": 0.8}
        bonus = nns._compute_coupling_bonus("IMPULSE", supervision_weight=1.0)
        # 0.1 (strength) × 0.8 (partner) × 1.0 (sup_weight) = 0.08
        assert bonus == pytest.approx(0.08, abs=1e-6)

    def test_coupling_partner_below_fire_threshold(self, base_config):
        nns = _make_nns(base_config, coupling_enabled=True)
        # REFLEX fires at 0.25 — below fire_threshold=0.3, no bonus
        nns._latest_vm_baselines = {"REFLEX": 0.25}
        bonus = nns._compute_coupling_bonus("IMPULSE", supervision_weight=1.0)
        assert bonus == 0.0


# ── Coupling: sup_weight decay ──────────────────────────────────────

class TestCouplingWarmupDecay:

    def test_bonus_scales_linearly_with_sup_weight(self, base_config):
        nns = _make_nns(base_config, coupling_enabled=True)
        nns._latest_vm_baselines = {"REFLEX": 0.8}
        b1 = nns._compute_coupling_bonus("IMPULSE", supervision_weight=1.0)
        b_half = nns._compute_coupling_bonus("IMPULSE", supervision_weight=0.5)
        b_zero = nns._compute_coupling_bonus("IMPULSE", supervision_weight=0.0)
        assert b1 == pytest.approx(2 * b_half, abs=1e-6)
        assert b_zero == 0.0


# ── Coupling: all 6 biological pairs ────────────────────────────────

class TestCouplingPairs:

    @pytest.mark.parametrize("prog_a,prog_b", [
        ("EMPATHY", "REFLECTION"),
        ("CURIOSITY", "INSPIRATION"),
        ("CREATIVITY", "INSPIRATION"),
        ("FOCUS", "INTUITION"),
        ("METABOLISM", "VIGILANCE"),
        ("IMPULSE", "REFLEX"),
    ])
    def test_pair_couples_symmetrically(self, base_config, prog_a, prog_b):
        nns = _make_nns(base_config, coupling_enabled=True)
        # A fires → B gets bonus
        nns._latest_vm_baselines = {prog_a: 0.7}
        assert nns._compute_coupling_bonus(prog_b, 1.0) > 0
        # B fires → A gets bonus
        nns._latest_vm_baselines = {prog_b: 0.7}
        assert nns._compute_coupling_bonus(prog_a, 1.0) > 0

    def test_unpaired_program_gets_no_bonus(self, base_config):
        nns = _make_nns(base_config, coupling_enabled=True)
        # REFLEX's partner is IMPULSE — unrelated programs don't couple
        nns._latest_vm_baselines = {"INSPIRATION": 0.9}
        # INSPIRATION pairs with CURIOSITY + CREATIVITY only
        assert nns._compute_coupling_bonus("REFLEX", 1.0) == 0.0


# ── Coupling: multiple partners ─────────────────────────────────────

class TestCouplingMultiPartner:

    def test_inspiration_sums_bonuses_from_both_partners(self, base_config):
        # INSPIRATION pairs with CURIOSITY AND CREATIVITY — both firing
        # should sum their individual contributions
        nns = _make_nns(base_config, coupling_enabled=True)
        nns._latest_vm_baselines = {"CURIOSITY": 0.8, "CREATIVITY": 0.6}
        bonus = nns._compute_coupling_bonus("INSPIRATION", 1.0)
        expected = 0.1 * 0.8 + 0.1 * 0.6  # 0.14
        assert bonus == pytest.approx(expected, abs=1e-6)


# ── Coupling: custom pairs override ─────────────────────────────────

class TestCouplingCustomPairs:

    def test_custom_pairs_replace_defaults(self, base_config):
        nns = _make_nns(
            base_config,
            coupling_enabled=True,
            coupling_overrides={"pairs": [("REFLEX", "FOCUS")]},
        )
        # Only the custom pair is active — default pairs are replaced
        nns._latest_vm_baselines = {"FOCUS": 0.7}
        assert nns._compute_coupling_bonus("REFLEX", 1.0) > 0
        # IMPULSE/REFLEX (default pair) is no longer active
        nns._latest_vm_baselines = {"REFLEX": 0.7}
        assert nns._compute_coupling_bonus("IMPULSE", 1.0) == 0.0


# ── Eligibility flag ────────────────────────────────────────────────

class TestEligibilityFlag:

    def test_eligibility_disabled_reverts_to_single_fire(self, base_config):
        nns = _make_nns(base_config, eligibility_enabled=False)
        assert nns._eligibility_traces_enabled is False
        # Flag is plumbed — runtime effect is K=1, decay=1.0 in record_outcome
        # (unit test verifies init-time flag; full behavior tested at
        # integration level since buf.update_recent_rewards is internal)

    def test_eligibility_enabled_by_default(self, base_config):
        nns = _make_nns(base_config, eligibility_enabled=True)
        assert nns._eligibility_traces_enabled is True

    def test_default_nns_has_eligibility_enabled(self, tmp_path):
        # Constructing NNS without explicit eligibility config → default true
        cfg = {
            "enabled": True,
            "warmup_steps": 500,
            "data_dir": str(tmp_path),
            "programs": {"REFLEX": {"input_features": "standard"}},
        }
        nns = NeuralNervousSystem(config=cfg)
        assert nns._eligibility_traces_enabled is True

    def test_default_nns_has_coupling_disabled(self, tmp_path):
        cfg = {
            "enabled": True,
            "warmup_steps": 500,
            "data_dir": str(tmp_path),
            "programs": {"REFLEX": {"input_features": "standard"}},
        }
        nns = NeuralNervousSystem(config=cfg)
        assert nns._coupling_enabled is False


# ── Coupling partners map structure ─────────────────────────────────

class TestCouplingPartnersMap:

    def test_partners_map_built_symmetrically(self, base_config):
        nns = _make_nns(base_config, coupling_enabled=True)
        # EMPATHY↔REFLECTION: both directions should be in the map
        assert "REFLECTION" in nns._coupling_partners.get("EMPATHY", [])
        assert "EMPATHY" in nns._coupling_partners.get("REFLECTION", [])
        # INSPIRATION has TWO partners
        assert set(nns._coupling_partners["INSPIRATION"]) == {
            "CURIOSITY", "CREATIVITY"
        }
