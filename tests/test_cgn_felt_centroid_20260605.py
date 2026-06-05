"""Phase A — CGN per-concept felt centroid (RFP_cgn_felt_state_exposure §7.A).

Covers the torch-free shared helpers (normalize_neuromods / felt_ema / felt_distance,
cgn_types.py) AND the central materialization in ConceptGroundingNetwork.record_outcome
+ the concept_felt_centroid accessor + persistence round-trip.

Run: python -m pytest tests/test_cgn_felt_centroid_20260605.py -v -p no:anchorpy
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from titan_hcl.logic.cgn_types import (
    CGNTransition, FELT_EMA_ALPHA, felt_distance, felt_ema, normalize_neuromods,
)


# ── Shared helpers (torch-free) ─────────────────────────────────────────────

def test_normalize_collapses_5ht_keeps_endorphin_drops_meta():
    out = normalize_neuromods({
        "5-HT": 0.4, "DA": 0.6, "Endorphin": 0.3,
        "emotion": "joy", "emotion_confidence": 0.9, "ts": 123.0, "dream_cycle": 7,
    })
    assert out == {"5HT": 0.4, "DA": 0.6, "Endorphin": 0.3}


def test_normalize_flattens_nested_level():
    assert normalize_neuromods({"DA": {"level": 0.7, "setpoint": 0.5}}) == {"DA": 0.7}


def test_normalize_drops_bool_and_nonnumeric():
    # bool is an int subclass — must be excluded; strings dropped.
    assert normalize_neuromods({"DA": True, "NE": "x", "GABA": 0.5}) == {"GABA": 0.5}


def test_normalize_non_dict_and_empty():
    assert normalize_neuromods(None) == {}
    assert normalize_neuromods("nope") == {}
    assert normalize_neuromods({}) == {}


def test_felt_ema_first_observation_copies():
    assert felt_ema(None, {"DA": 0.6}) == {"DA": 0.6}
    assert felt_ema({}, {"DA": 0.6}) == {"DA": 0.6}


def test_felt_ema_recency_weighted():
    # 0.3*0.5 + 0.7*0.6 = 0.57
    out = felt_ema({"DA": 0.6}, {"DA": 0.5}, 0.3)
    assert abs(out["DA"] - 0.57) < 1e-9


def test_felt_ema_preserves_prior_only_keys():
    out = felt_ema({"DA": 0.6, "NE": 0.4}, {"DA": 0.5}, 0.3)
    assert abs(out["DA"] - 0.57) < 1e-9
    assert out["NE"] == 0.4  # untouched — this outcome carried no NE


def test_felt_distance_rms_two_keys():
    # gaps DA 0.25, NE 0.30 over 2 keys → sqrt((0.0625+0.09)/2) = 0.2762
    d = felt_distance({"DA": 0.80, "NE": 0.75}, {"DA": 0.55, "NE": 0.45})
    assert abs(d - 0.27613) < 1e-4


def test_felt_distance_rms_diluted_by_neutral_keys():
    # The §1.3 worked example — same DA/NE gaps but 5 carried keys → 0.1746
    lived = {"DA": 0.80, "5HT": 0.50, "NE": 0.75, "GABA": 0.50, "ACh": 0.60}
    centroid = {"DA": 0.55, "5HT": 0.50, "NE": 0.45, "GABA": 0.50, "ACh": 0.60}
    d = felt_distance(lived, centroid)
    assert abs(d - 0.17464) < 1e-4


def test_felt_distance_missing_key_counts_neutral():
    # b lacks DA → DA compares 1.0 vs 0.5 (neutral); NE matches → sqrt((0.25)/2)=0.3536
    d = felt_distance({"DA": 1.0, "NE": 0.5}, {"NE": 0.5})
    assert abs(d - 0.35355) < 1e-4


def test_felt_distance_empty_side_is_zero():
    # Absence of signal is NOT a conflict.
    assert felt_distance({}, {"DA": 0.8}) == 0.0
    assert felt_distance({"DA": 0.8}, {}) == 0.0


def test_felt_distance_rejects_cosine_blindness():
    # Same direction, different magnitude — cosine ≈ 1 (distance ≈ 0) would MISS this;
    # RMS must report a real gap (the magnitude IS the felt frame).
    d = felt_distance({"DA": 0.8, "NE": 0.8}, {"DA": 0.4, "NE": 0.4})
    assert d > 0.3  # sqrt((0.16+0.16)/2) = 0.4


def test_felt_distance_collapses_dash_naming_before_compare():
    # "5-HT" (CGN) vs "5HT" (lived) must be treated as the SAME axis → 0 distance.
    assert felt_distance({"5-HT": 0.7}, {"5HT": 0.7}) == 0.0


# ── Central materialization (ConceptGroundingNetwork.record_outcome) ─────────

def _mk_cgn(tmp_path):
    from titan_hcl.logic.cgn import ConceptGroundingNetwork
    return ConceptGroundingNetwork(state_dir=str(tmp_path))


def _outcome(cgn, consumer, concept, reward=0.6, felt=None):
    """Add one unrewarded transition + record its outcome (the central path)."""
    cgn._buffer.add(CGNTransition(
        consumer=consumer, concept_id=concept,
        state=np.zeros(30, dtype=np.float32), action=0,
        action_params=np.zeros(8, dtype=np.float32), reward=0.0,
        metadata={"action_name": "reinforce"},
    ))
    ctx = {"felt_state": felt} if felt is not None else None
    cgn.record_outcome(consumer, concept, reward, outcome_context=ctx)


def test_record_outcome_accumulates_centroid(tmp_path):
    from titan_hcl.logic.cgn import CGNConsumerConfig
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="felt_teaching"))
    _outcome(cgn, "felt_teaching", "microbe", felt={"DA": 0.6, "5-HT": 0.5})
    c1 = cgn.concept_felt_centroid("microbe")
    assert c1 == {"DA": 0.6, "5HT": 0.5}  # first obs → normalized copy (dash collapsed)

    _outcome(cgn, "felt_teaching", "microbe", felt={"DA": 0.5, "5HT": 0.5})
    c2 = cgn.concept_felt_centroid("microbe")
    assert abs(c2["DA"] - (FELT_EMA_ALPHA * 0.5 + (1 - FELT_EMA_ALPHA) * 0.6)) < 1e-9
    assert abs(c2["5HT"] - 0.5) < 1e-9


def test_centroid_empty_when_no_felt(tmp_path):
    from titan_hcl.logic.cgn import CGNConsumerConfig
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="reasoning"))
    _outcome(cgn, "reasoning", "warmth")  # no felt_state
    assert cgn.concept_felt_centroid("warmth") == {}
    assert cgn.concept_felt_centroid("never_seen") == {}


def test_centroid_ignores_empty_felt_dict(tmp_path):
    from titan_hcl.logic.cgn import CGNConsumerConfig
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="felt_teaching"))
    _outcome(cgn, "felt_teaching", "microbe", felt={"emotion": "joy", "ts": 1.0})
    # only metadata → normalized to {} → no centroid
    assert cgn.concept_felt_centroid("microbe") == {}


def test_centroid_persists_across_restart(tmp_path):
    from titan_hcl.logic.cgn import ConceptGroundingNetwork, CGNConsumerConfig
    cgn = _mk_cgn(tmp_path)
    cgn.register_consumer(CGNConsumerConfig(name="felt_teaching"))
    _outcome(cgn, "felt_teaching", "microbe", felt={"DA": 0.6, "NE": 0.4})
    saved = cgn.concept_felt_centroid("microbe")
    assert saved  # non-empty
    cgn._save_state()

    fresh = ConceptGroundingNetwork(state_dir=str(tmp_path))
    assert fresh.concept_felt_centroid("microbe") == saved  # rode cgn_state.pt


if __name__ == "__main__":
    import tempfile
    import pathlib
    # pure helpers
    for fn in (test_normalize_collapses_5ht_keeps_endorphin_drops_meta,
               test_normalize_flattens_nested_level,
               test_normalize_drops_bool_and_nonnumeric,
               test_normalize_non_dict_and_empty,
               test_felt_ema_first_observation_copies,
               test_felt_ema_recency_weighted,
               test_felt_ema_preserves_prior_only_keys,
               test_felt_distance_rms_two_keys,
               test_felt_distance_rms_diluted_by_neutral_keys,
               test_felt_distance_missing_key_counts_neutral,
               test_felt_distance_empty_side_is_zero,
               test_felt_distance_rejects_cosine_blindness,
               test_felt_distance_collapses_dash_naming_before_compare):
        fn()
    # central path
    for fn in (test_record_outcome_accumulates_centroid,
               test_centroid_empty_when_no_felt,
               test_centroid_ignores_empty_felt_dict,
               test_centroid_persists_across_restart):
        with tempfile.TemporaryDirectory() as td:
            fn(pathlib.Path(td))
    print("OK — Phase A felt-centroid checks passed")
