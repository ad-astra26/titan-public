"""P3 — MasteryLevel primitive (ARCHITECTURE_mastery_leveling.md §2.2).

Pins the load-bearing behaviors:
  - the ratchet is GATED by the scale-free competence_rate (a value rise WITHOUT
    competence — reward-scale inflation — never advances the level: INV-ML-2/3);
  - the ratchet is monotone (a value dip never lowers the level: INV-ML-2);
  - higher grades demand more competence (rising floor);
  - chunk-graduation fires milestone ticks but is NOT the primary driver (INV-ML-5);
  - persistence round-trips; schema/grade mismatch → safe relearn.
"""
import numpy as np
import pytest

from titan_hcl.synthesis.mastery_level import MasteryLevel


def _ml(**kw):
    return MasteryLevel(n_grades=10, grade_lo=-5.0, grade_hi=5.0,
                        ema_alpha=0.5, competence_ema_alpha=0.5,
                        competence_floor_base=0.55, competence_floor_slope=0.02, **kw)


def test_competence_gate_blocks_scale_inflation_level_up():
    # High value (would be a high grade) but ZERO competence ⇒ level must NOT rise.
    ml = _ml()
    out = {}
    for _ in range(50):
        out = ml.update(v_symlog=4.5, competence_rate=0.0, n_chunks=0)
    assert out["grade"] == 0, "no competence ⇒ ratchet stays at floor (anti scale-inflation)"
    assert out["value_milestones"] == 0


def test_competence_confirmed_value_rise_levels_up():
    # Same high value, but with genuine competence ⇒ level ratchets up.
    ml = _ml()
    out = {}
    for _ in range(50):
        out = ml.update(v_symlog=4.5, competence_rate=0.9, n_chunks=0)
    assert out["grade"] > 0, "competence-confirmed value rise must level up"
    assert out["value_milestones"] >= 1


def test_ratchet_is_monotone_under_value_dip():
    ml = _ml()
    for _ in range(50):
        ml.update(v_symlog=4.5, competence_rate=0.95)
    high = ml.readout()["grade"]
    assert high > 0
    # now a sustained value collapse — grade must NOT fall (running-max ratchet)
    for _ in range(50):
        ml.update(v_symlog=-4.5, competence_rate=0.95)
    after = ml.readout()
    assert after["grade"] == high, "ratchet must not un-learn competence on a dip"
    assert after["level"] >= float(high)  # level never drops below the ratcheted grade floor


def test_floor_rises_with_grade():
    ml = _ml()
    f0 = ml.competence_floor(0)
    f5 = ml.competence_floor(5)
    assert f5 > f0, "higher grades demand more proven competence"
    assert ml.competence_floor(0) == pytest.approx(0.55)


def test_chunk_graduation_fires_milestone_not_primary_level():
    ml = _ml()
    # value flat at floor, but chunks accrue → chunk milestones tick, grade stays.
    out = ml.update(v_symlog=-5.0, competence_rate=0.2, n_chunks=3)
    assert "chunk" in out["milestones"]
    assert out["chunk_milestones"] == 3
    assert out["grade"] == 0, "chunks do NOT drive the primary grade (INV-ML-5)"
    # no new chunks → no new chunk milestone
    out2 = ml.update(v_symlog=-5.0, competence_rate=0.2, n_chunks=3)
    assert "chunk" not in out2["milestones"]
    assert out2["chunk_milestones"] == 3


def test_persistence_roundtrip():
    ml = _ml()
    for _ in range(30):
        ml.update(v_symlog=3.0, competence_rate=0.8, n_chunks=2)
    snap = ml.to_dict()
    ml2 = _ml()
    assert ml2.load_dict(snap) is True
    assert ml2.readout()["grade"] == ml.readout()["grade"]
    assert ml2.readout()["n_chunks"] == ml.readout()["n_chunks"]
    assert ml2._competence_ema == pytest.approx(ml._competence_ema)


def test_load_dict_rejects_schema_or_grade_mismatch():
    ml = _ml()
    assert ml.load_dict({"schema_version": 999, "n_grades": 10}) is False
    assert ml.load_dict({"schema_version": 1, "n_grades": 7}) is False  # grade-ladder mismatch
    assert ml.load_dict("not a dict") is False
