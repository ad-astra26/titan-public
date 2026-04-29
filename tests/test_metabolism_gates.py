"""tests/test_metabolism_gates.py — Mainnet Lifecycle Wiring rFP (2026-04-20)

Unit tests for MetabolismController.evaluate_gate + ring buffer + tier-
parametrized gate decisions. Covers the 5 active tiers. HIBERNATION has
identical gate behavior to SURVIVAL/EMERGENCY (all features disabled,
rate_factor=0.0) so one representative tier is sufficient.

Run:
    source test_env/bin/activate
    python -m pytest tests/test_metabolism_gates.py -v -p no:anchorpy --tb=short
"""
import pytest

from titan_plugin.core.metabolism import MetabolismController


class _MockNetwork:
    """Minimal network stub for MetabolismController construction."""

    def __init__(self, balance: float = 0.5):
        self._balance = balance

    async def get_balance(self) -> float:
        return self._balance


def _make_controller(balance: float, gates_enforced: bool = False) -> MetabolismController:
    """Construct a controller at a specific tier + kill-switch state."""
    c = MetabolismController(soul=None, network=_MockNetwork(balance))
    c._last_balance = balance
    c._update_tier(balance)
    c._gates_enforced = gates_enforced
    return c


@pytest.mark.parametrize("balance,expected_tier,expected_features_enabled", [
    (2.0, "THRIVING", True),
    (0.5, "HEALTHY", True),
    (0.2, "CONSERVING", True),   # features ENABLED but rate_factor=0.5
    (0.1, "SURVIVAL", False),
    (0.03, "EMERGENCY", False),
    (0.005, "HIBERNATION", False),
])
def test_tier_thresholds(balance, expected_tier, expected_features_enabled):
    """Tier is correctly derived from balance and feature flags align."""
    c = _make_controller(balance)
    assert c.get_metabolic_tier() == expected_tier
    for feature in ("memos", "nfts", "expression", "research", "social"):
        assert c.can_use_feature(feature) == expected_features_enabled


def test_evaluate_gate_observation_mode_never_blocks():
    """With gates_enforced=False, evaluate_gate always returns (True, 1.0)."""
    c = _make_controller(0.005, gates_enforced=False)  # HIBERNATION
    # Even at the deepest starvation tier, observation mode proceeds.
    proceed, rate = c.evaluate_gate("memos", "UnitTest")
    assert proceed is True
    assert rate == 1.0


def test_evaluate_gate_enforcement_mode_blocks_at_survival():
    """With gates_enforced=True, SURVIVAL tier closes all features."""
    c = _make_controller(0.1, gates_enforced=True)  # SURVIVAL
    for feature in ("memos", "nfts", "expression", "research", "social"):
        proceed, rate = c.evaluate_gate(feature, f"UnitTest.{feature}")
        assert proceed is False
        assert rate == 0.0


def test_evaluate_gate_enforcement_mode_at_conserving_throttles():
    """With gates_enforced=True, CONSERVING returns allowed=True with rate=0.5."""
    c = _make_controller(0.2, gates_enforced=True)  # CONSERVING
    proceed, rate = c.evaluate_gate("research", "UnitTest")
    assert proceed is True
    assert rate == 0.5  # half-rate throttle


def test_evaluate_gate_enforcement_mode_at_healthy_fullrate():
    """With gates_enforced=True, HEALTHY tier passes cleanly at rate=1.0."""
    c = _make_controller(0.5, gates_enforced=True)  # HEALTHY
    proceed, rate = c.evaluate_gate("social", "UnitTest")
    assert proceed is True
    assert rate == 1.0


def test_ring_buffer_records_decisions():
    """Every evaluate_gate call appends to the ring buffer."""
    c = _make_controller(0.5)
    assert len(c._gate_decisions) == 0
    c.evaluate_gate("memos", "Test.a")
    c.evaluate_gate("nfts", "Test.b")
    c.evaluate_gate("expression", "Test.c")
    assert len(c._gate_decisions) == 3
    # Entries carry expected fields
    last = c._gate_decisions[-1]
    for key in ("ts", "feature", "caller", "tier", "allowed", "rate", "reason", "enforced"):
        assert key in last


def test_gate_decision_summary_structure():
    """get_gate_decision_summary returns the telemetry shape used by /v4."""
    c = _make_controller(0.1, gates_enforced=True)  # SURVIVAL, closures expected
    c.evaluate_gate("memos", "Test.alpha")
    c.evaluate_gate("memos", "Test.alpha")
    c.evaluate_gate("social", "Test.beta")
    summary = c.get_gate_decision_summary()
    assert summary["gates_enforced"] is True
    assert summary["current_tier"] == "SURVIVAL"
    assert summary["total_evaluations"] == 3
    assert summary["decisions_buffered"] == 3
    # All 3 are closures at SURVIVAL
    assert summary["window_10min_closures"] == 3
    by_caller = summary["by_caller"]
    assert by_caller["Test.alpha"]["total"] == 2
    assert by_caller["Test.alpha"]["closed"] == 2
    assert by_caller["Test.beta"]["feature"] == "social"


def test_ring_buffer_bounded():
    """Ring buffer respects _GATE_DECISION_RING_SIZE cap."""
    from titan_plugin.core.metabolism import _GATE_DECISION_RING_SIZE
    c = _make_controller(0.5)
    for i in range(_GATE_DECISION_RING_SIZE + 50):
        c.evaluate_gate("memos", f"Test.{i}")
    assert len(c._gate_decisions) == _GATE_DECISION_RING_SIZE
    # But total_evaluations tracks lifetime count
    assert sum(c._gate_decision_counts.values()) == _GATE_DECISION_RING_SIZE + 50
