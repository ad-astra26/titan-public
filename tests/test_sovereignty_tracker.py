"""tests/test_sovereignty_tracker.py — Mainnet Lifecycle Wiring rFP (2026-04-20)

Unit tests for SovereigntyTracker lifecycle: record_epoch, convergence
bookkeeping, state save/load round-trip, confirm_maker, and
increment_great_cycle.

NOTE: This file is distinct from tests/test_sovereignty.py which tests
the jailbreak-defense suite (private-key extraction attacks).

Run:
    source test_env/bin/activate
    python -m pytest tests/test_sovereignty_tracker.py -v -p no:anchorpy --tb=short
"""
import json
import os
import tempfile
import pytest

from titan_plugin.logic import sovereignty as sov_module
from titan_plugin.logic.sovereignty import SovereigntyTracker


@pytest.fixture
def tmp_state_file(tmp_path, monkeypatch):
    """Redirect PERSISTENCE_FILE to a tmp path for test isolation."""
    tmp_file = tmp_path / "sovereignty_state.json"
    monkeypatch.setattr(sov_module, "PERSISTENCE_FILE", str(tmp_file))
    return tmp_file


def test_initial_state(tmp_state_file):
    """Fresh tracker starts in ENFORCING mode with zeros."""
    t = SovereigntyTracker()
    assert t._sovereignty_mode == "ENFORCING"
    assert t._great_cycle == 0
    assert t._total_great_pulses == 0
    assert t._maker_confirmed is False


def test_record_epoch_accumulates_modulators(tmp_state_file):
    """record_epoch fills modulator deques and tracks violations."""
    t = SovereigntyTracker()
    # Normal convergence epochs
    for i in range(50):
        t.record_epoch(i, {"DA": 0.6, "5HT": 0.55, "NE": 0.5}, developmental_age=i)
    assert len(t._modulator_history["DA"]) == 50
    assert t._saturation_violations == 0
    assert t._collapse_violations == 0


def test_record_epoch_flags_saturation(tmp_state_file):
    """High modulator levels (>0.95) increment saturation_violations."""
    t = SovereigntyTracker()
    for i in range(20):
        t.record_epoch(i, {"DA": 0.97, "5HT": 0.5, "NE": 0.5})
    assert t._saturation_violations == 20
    assert t._collapse_violations == 0


def test_record_epoch_flags_collapse(tmp_state_file):
    """Low modulator levels (<0.05) increment collapse_violations."""
    t = SovereigntyTracker()
    for i in range(20):
        t.record_epoch(i, {"DA": 0.5, "5HT": 0.02, "NE": 0.5})
    assert t._collapse_violations == 20
    assert t._saturation_violations == 0


def test_record_epoch_counts_great_pulses(tmp_state_file):
    """great_pulse_fired=True increments the total_great_pulses counter."""
    t = SovereigntyTracker()
    for i in range(10):
        t.record_epoch(i, {"DA": 0.5}, great_pulse_fired=(i % 3 == 0))
    # Fires at i=0, 3, 6, 9 → 4 pulses
    assert t._total_great_pulses == 4


def test_check_transition_criteria_not_met_with_low_data(tmp_state_file):
    """Criteria evaluator reports all_met=False when data is insufficient."""
    t = SovereigntyTracker()
    for i in range(10):
        t.record_epoch(i, {"DA": 0.5}, developmental_age=i)
    c = t.check_transition_criteria()
    assert c["all_met"] is False
    assert c["great_cycle_met"] is False
    assert c["developmental_age_met"] is False
    assert c["convergence_met"] is False
    assert c["great_pulses_met"] is False


def test_confirm_maker_flips_flag(tmp_state_file):
    """confirm_maker() flips _maker_confirmed and persists state."""
    t = SovereigntyTracker()
    assert t._maker_confirmed is False
    t.confirm_maker()
    assert t._maker_confirmed is True
    # State file should now exist
    assert tmp_state_file.exists()


def test_increment_great_cycle(tmp_state_file):
    """increment_great_cycle advances the counter and persists."""
    t = SovereigntyTracker()
    assert t._great_cycle == 0
    t.increment_great_cycle()
    t.increment_great_cycle()
    assert t._great_cycle == 2
    data = json.loads(tmp_state_file.read_text())
    assert data["great_cycle"] == 2


def test_state_save_load_round_trip(tmp_state_file):
    """Persist → load returns the same tracker state (rFP acceptance).

    Every field that _save_state writes MUST round-trip through _load_state.
    This test caught: (1) _maker_confirmed not persisted originally,
    (2) _developmental_age written with wrong attr name in _load_state.
    """
    t1 = SovereigntyTracker()
    t1.confirm_maker()
    t1.increment_great_cycle()
    for i in range(15):
        t1.record_epoch(i, {"DA": 0.5}, developmental_age=1234, great_pulse_fired=True)
    t1._save_state()

    t2 = SovereigntyTracker()  # fresh instance, should _load_state
    assert t2._maker_confirmed is True
    assert t2._great_cycle == 1
    assert t2._total_great_pulses == 15
    assert t2._sovereignty_mode == "ENFORCING"
    # Every persisted field must actually populate the expected attribute.
    assert t2._developmental_age == 1234, (
        "developmental_age must round-trip — "
        "regression guard for 2026-04-23 _total_developmental_age typo")


def test_transition_to_advisory_rejects_unmet_criteria(tmp_state_file):
    """transition_to_advisory returns False when criteria aren't met."""
    t = SovereigntyTracker()
    ok = t.transition_to_advisory(epoch_id=1)
    assert ok is False
    assert t._sovereignty_mode == "ENFORCING"


def test_get_stats_shape(tmp_state_file):
    """get_stats() returns the shape expected by /v4/sovereignty/status."""
    t = SovereigntyTracker()
    stats = t.get_stats()
    for key in ("sovereignty_mode", "great_cycle", "total_great_pulses",
                "developmental_age", "saturation_violations",
                "collapse_violations", "convergence_window",
                "transition_epoch", "maker_confirmed"):
        assert key in stats
