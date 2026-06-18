"""Phase 0 — Titan-time spine. Proves the §7.0.5 DONE-contract for
`titan_hcl/logic/titan_time.py` (RFP_verifiable_autobiographical_presence_memory):

  (a) the counter SEEDS cycle_start_epoch from a real get_age_epochs();
  (b) latch_if_trough is edge-triggered — exactly ONCE per trough, with re-arm;
  (c) state round-trips through a persist + reload (survives restart);
  (d) the translator derives a LIVE per-Titan rate from a 2-point (epoch,ts) anchor
      and buckets epochs→human phrase; cold-start hedges honestly to "recently"
      (never a hardcoded rate).

Pure-unit: a fake age reader injects (age_epochs, ts) so no SHM/worker is needed.
"""
import os

import pytest

from titan_hcl.logic.titan_time import CircadianCycleCounter, TitanTimeTranslator


class _FakeAgeReader:
    """Stand-in for ConsciousnessAgeReader — controllable (age_epochs, ts)."""

    def __init__(self, epochs: int = 0, ts: float = 0.0):
        self.epochs = epochs
        self.ts = ts

    def get_age_epochs(self) -> int:
        return self.epochs

    def get_age_snapshot(self):
        return self.epochs, self.ts


# ── (a) seed ────────────────────────────────────────────────────────────────
def test_counter_seeds_from_age(tmp_path):
    r = _FakeAgeReader(epochs=812300)
    c = CircadianCycleCounter(save_dir=str(tmp_path), age_reader=r, config={})
    assert c.cycle_id == 0
    assert c.cycle_start_epoch == 812300        # seeded from live age, never pre-feature
    assert c.armed is False                     # seed disarmed → no boot-artifact latch
    assert os.path.exists(os.path.join(str(tmp_path), "cycle_state.json"))


# ── (b) edge-triggered latch: once per trough + re-arm ───────────────────────
def test_latch_once_per_trough_and_rearm(tmp_path):
    r = _FakeAgeReader(epochs=800000)
    cfg = {"trough_threshold": 0.30, "rearm_threshold": 0.45}
    c = CircadianCycleCounter(save_dir=str(tmp_path), age_reader=r, config=cfg)

    # seeded disarmed → a trough alone does NOT fire (needs a daytime re-arm first,
    # so the inaugural boundary is a genuine post-daytime trough)
    assert c.latch_if_trough(phase=0.20, age_epochs=800000) is None
    assert c.cycle_id == 0

    # daytime → re-arm
    assert c.latch_if_trough(phase=0.90, age_epochs=805000) is None
    assert c.armed is True

    # trough → fires exactly once, opens cycle 1 at the current age
    assert c.latch_if_trough(phase=0.20, age_epochs=810000) == 1
    assert c.cycle_id == 1
    assert c.cycle_start_epoch == 810000
    assert c.armed is False

    # still in trough → idempotent, no double-fire
    assert c.latch_if_trough(phase=0.15, age_epochs=810500) is None
    assert c.cycle_id == 1

    # hysteresis band (between thresholds) while disarmed → neither fires nor re-arms
    assert c.latch_if_trough(phase=0.35, age_epochs=811000) is None
    assert c.armed is False

    # next daytime re-arms; next trough opens cycle 2
    assert c.latch_if_trough(phase=0.90, age_epochs=820000) is None
    assert c.armed is True
    assert c.latch_if_trough(phase=0.20, age_epochs=830000) == 2
    assert c.cycle_id == 2
    assert c.cycle_start_epoch == 830000


# ── (c) persistence round-trip (survives restart) ────────────────────────────
def test_state_roundtrip(tmp_path):
    r = _FakeAgeReader(epochs=900000)
    c = CircadianCycleCounter(save_dir=str(tmp_path), age_reader=r, config={})
    c.latch_if_trough(phase=0.90, age_epochs=901000)   # arm
    assert c.latch_if_trough(phase=0.20, age_epochs=905000) == 1

    # a fresh instance (= a restart) restores from cycle_state.json
    c2 = CircadianCycleCounter(
        save_dir=str(tmp_path), age_reader=_FakeAgeReader(epochs=910000), config={})
    assert c2.cycle_id == 1
    assert c2.cycle_start_epoch == 905000
    assert c2.armed is False


# ── (d) translator: live per-Titan rate + honest cold-start ──────────────────
def test_translator_live_rate(tmp_path):
    # anchor captured at construction = (epoch0=1000, ts0=10_000.0)
    r = _FakeAgeReader(epochs=1000, ts=10_000.0)
    t = TitanTimeTranslator(save_dir=str(tmp_path), age_reader=r)

    # advance: 8348 epochs elapsed over 86_400 s → rate ≈ 10.35 s/epoch (measured, live)
    r.epochs = 1000 + 8348
    r.ts = 10_000.0 + 86_400.0
    rate = t.measured_sec_per_epoch()
    assert rate == pytest.approx(86_400.0 / 8348.0, rel=1e-6)

    # gap of ~2 days worth of epochs → "~2 days ago" (the RFP §1.3 worked phrasing)
    assert t.to_human(2 * 8348) == "~2 days ago"
    # a small gap → minutes
    assert "minutes ago" in t.to_human(100)
    # zero gap → present
    assert t.to_human(0) == "moments ago"


def test_translator_cold_start_hedges(tmp_path):
    # degenerate: the snapshot has not advanced since the anchor → no live rate yet
    r = _FakeAgeReader(epochs=5000, ts=50_000.0)
    t = TitanTimeTranslator(save_dir=str(tmp_path), age_reader=r)
    assert t.measured_sec_per_epoch() is None
    assert t.to_human(8348) == "recently"        # honest hedge, NO hardcoded precision
    assert t.to_human(0) == "moments ago"


def test_translator_anchor_persists(tmp_path):
    r = _FakeAgeReader(epochs=2000, ts=20_000.0)
    TitanTimeTranslator(save_dir=str(tmp_path), age_reader=r)
    # a fresh instance restores the SAME baseline anchor (rate stays continuous
    # across restarts; it does not re-anchor to "now")
    r2 = _FakeAgeReader(epochs=99_999, ts=999_999.0)
    t2 = TitanTimeTranslator(save_dir=str(tmp_path), age_reader=r2)
    rate = t2.measured_sec_per_epoch()
    expected = (999_999.0 - 20_000.0) / (99_999 - 2000)
    assert rate == pytest.approx(expected, rel=1e-6)
