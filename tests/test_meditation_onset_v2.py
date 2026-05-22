"""
tests/test_meditation_onset_v2.py — rFP_meditation_emergent_onset_redesign / D-SPEC-107.

Sleep-symmetric meditation onset: emergent drive competition on the homeostatic-
balance axis (neuromod gain-deviation) + a hard cadence floor. Tests the pure
decision function `evaluate_meditation_onset`.
"""
import pytest

from titan_hcl.modules.meditation_worker import (
    evaluate_meditation_onset,
    _read_all_gains,
    _read_arousal,
)

# Default-ish config for tests.
CFG = dict(
    balance_band=0.25,
    balance_half_life=50.0,
    k_agit=1.0,
    debt_onset=1500,
    debt_ramp=900,
    max_interval_s=21600.0,
    min_epochs=1500,
)


def _eval(**over):
    base = dict(
        gains=[1.0] * 6, arousal=0.5, epoch_gap=2000, time_since_last=600.0,
        balance_ema=None, last_epoch=100, meditation_count=5, **CFG,
    )
    base.update(over)
    return evaluate_meditation_onset(**base)


class TestNaturalBalanceTrigger:
    def test_balanced_sustained_fires(self):
        # All gains at 1.0 → agitation 0 → balanced; past min_epochs; calm held.
        d = _eval(gains=[1.0] * 6, balance_ema=1.0, epoch_gap=2000)
        assert d["agitation"] == pytest.approx(0.0)
        assert d["balance"] == pytest.approx(1.0)
        assert d["fire"] is True
        assert d["reason"] == "homeostatic_balance_sustained"

    def test_agitated_does_not_fire_on_balance(self):
        # gains far from 1.0 → high agitation → not balanced → no natural fire.
        d = _eval(gains=[0.3, 1.8, 0.4, 1.7, 0.3, 0.4], balance_ema=0.1,
                  epoch_gap=2000, time_since_last=600.0)
        assert d["agitation"] > 0.25
        assert d["reason"] != "homeostatic_balance_sustained"
        assert d["fire"] is False  # not balanced, not overdue, not first-ever

    def test_min_epochs_gate_blocks_early_fire(self):
        # Perfectly balanced but min interval not elapsed → no natural fire.
        d = _eval(gains=[1.0] * 6, balance_ema=1.0, epoch_gap=1000,
                  time_since_last=600.0)
        assert d["fire"] is False

    def test_arousal_raises_agitation_drive(self):
        lo = _eval(arousal=0.0, gains=[1.2] * 6, balance_ema=0.5)
        hi = _eval(arousal=1.0, gains=[1.2] * 6, balance_ema=0.5)
        assert hi["agitation_drive"] > lo["agitation_drive"]


class TestHardCadenceFloor:
    def test_hard_floor_forces_even_when_agitated(self):
        # Heavily agitated + low calm, but past max_interval → MUST fire.
        d = _eval(gains=[0.3, 1.9, 0.3, 1.9, 0.3, 0.3], balance_ema=0.0,
                  epoch_gap=5000, time_since_last=22000.0)
        assert d["fire"] is True
        assert d["reason"] == "hard_floor_max_interval"

    def test_below_floor_not_forced(self):
        d = _eval(gains=[0.3] * 6, balance_ema=0.0, epoch_gap=2000,
                  time_since_last=10000.0)  # < max_interval 21600
        assert not (d["reason"] == "hard_floor_max_interval")


class TestFirstEverAndDebt:
    def test_first_ever_fires_after_min_interval(self):
        d = _eval(meditation_count=0, last_epoch=0, gains=[0.4] * 6,
                  balance_ema=0.0, epoch_gap=2000, time_since_last=600.0)
        assert d["fire"] is True
        assert d["reason"] == "first_ever"

    def test_debt_ramps_with_epoch_gap(self):
        early = _eval(epoch_gap=1500)   # at onset → debt 0
        later = _eval(epoch_gap=2400)   # onset+900 → debt 1.0
        assert early["debt"] == pytest.approx(0.0)
        assert later["debt"] == pytest.approx(1.0)
        assert later["meditate_drive"] > early["meditate_drive"]


class TestBalanceEmaAndReadFailure:
    def test_balance_ema_advances_toward_balance(self):
        # Starting from agitated EMA, a balanced sample pulls sustain up.
        d = _eval(gains=[1.0] * 6, balance_ema=0.0)
        assert 0.0 < d["balance_sustain"] <= 1.0

    def test_gains_none_is_max_agitation(self):
        d = _eval(gains=None, balance_ema=0.5, epoch_gap=2000,
                  time_since_last=600.0)
        assert d["agitation"] == pytest.approx(1.0)
        assert d["balance"] == pytest.approx(0.0)
        # Not balanced → only the hard floor / first-ever could fire it (neither here).
        assert d["fire"] is False


class TestReaders:
    class _FakeReader:
        def __init__(self, arr):
            self._arr = arr

        def read(self):
            return self._arr

    def test_read_all_gains_extracts_field1(self):
        import numpy as np
        arr = np.zeros((6, 4), dtype=np.float32)
        for i in range(6):
            arr[i, 1] = 0.5 + i * 0.1  # gain field
        gains = _read_all_gains({"neuromod": self._FakeReader(arr)})
        assert gains is not None
        assert gains[0] == pytest.approx(0.5, abs=1e-5)
        assert gains[5] == pytest.approx(1.0, abs=1e-5)

    def test_read_all_gains_none_on_bad_shape(self):
        import numpy as np
        arr = np.zeros((6,), dtype=np.float32)
        assert _read_all_gains({"neuromod": self._FakeReader(arr)}) is None
        assert _read_all_gains({}) is None

    def test_read_arousal_mean_ne_da(self):
        import numpy as np
        arr = np.zeros((6, 4), dtype=np.float32)
        arr[0, 0] = 0.6  # DA level
        arr[2, 0] = 0.8  # NE level
        a = _read_arousal({"neuromod": self._FakeReader(arr)})
        assert a == pytest.approx(0.7, abs=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-p", "no:anchorpy"])
