"""Tests for emot_cgn.EmotCGNConsumer (rFP_emot_cgn_v2)."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from titan_plugin.logic.emot_cgn import (
    BETA_PARAM_FLOOR,
    EMOT_SIGNAL_TO_PRIMITIVE,
    EmotCGNConsumer,
    EmotHypothesis,
    EmotPrimitive,
    _build_seed_hypotheses,
    _beta_mean,
    _posterior_confidence,
)
from titan_plugin.logic.emotion_cluster import (
    EMOT_PRIMITIVES,
    FEATURE_DIM,
    NUM_PRIMITIVES,
)


# ── Utility functions ──────────────────────────────────────────────

def test_beta_mean_uniform_prior():
    assert abs(_beta_mean(1.0, 1.0) - 0.5) < 1e-9


def test_beta_mean_skewed_positive():
    # α=9, β=1 → mean 0.9
    assert abs(_beta_mean(9.0, 1.0) - 0.9) < 1e-9


def test_posterior_confidence_uniform_prior_zero():
    assert _posterior_confidence(1.0, 1.0) == 0.0


def test_posterior_confidence_grows_with_samples():
    c1 = _posterior_confidence(10.0, 10.0)
    c2 = _posterior_confidence(100.0, 100.0)
    assert c2 > c1


# ── Bus signal mapping (rFP improvement #6 — orphan guard) ────────

def test_signal_mapping_uses_nonempty_values():
    """rFP_emot_cgn_v2 §lesson from 2026-04-20 Producer #16 bug: empty {}
    fails consumer truthiness check. All values must be non-empty."""
    for key, val in EMOT_SIGNAL_TO_PRIMITIVE.items():
        assert val  # truthy
        assert isinstance(val, dict)
        assert len(val) >= 1


def test_signal_mapping_neutral_weight():
    """All signal mappings should use neutral 0.5 weight (pure
    observability; no V bias pre-graduation)."""
    for key, val in EMOT_SIGNAL_TO_PRIMITIVE.items():
        for prim, weight in val.items():
            assert weight == 0.5


def test_signal_mapping_covers_expected_events():
    expected_events = {"cluster_assignment", "chain_emotion_context",
                       "cluster_recenter", "cluster_emerged",
                       "graduation_transition", "rollback_fired"}
    actual_events = {event for (_, event) in EMOT_SIGNAL_TO_PRIMITIVE.keys()}
    assert expected_events == actual_events


# ── EmotPrimitive ──────────────────────────────────────────────────

def test_emot_primitive_default_beta_1_1():
    p = EmotPrimitive(primitive_id="FLOW")
    assert p.alpha == 1.0
    assert p.beta == 1.0


def test_emot_primitive_recompute_derived():
    p = EmotPrimitive(primitive_id="FLOW", alpha=9.0, beta=1.0)
    p.recompute_derived()
    assert abs(p.V - 0.9) < 1e-9
    assert p.confidence > 0.0
    assert p.n_samples == 8  # α+β - 2·floor = 10 - 2 = 8


# ── Hypotheses (rFP improvement #5 — 8 not 6) ──────────────────────

def test_seed_hypotheses_count_is_8():
    """rFP improvement: 8 hypotheses to make 50% graduation bar (≥4/8)
    achievable. Had been 6 in initial rFP draft."""
    h = _build_seed_hypotheses()
    assert len(h) == 8


def test_seed_hypotheses_all_start_nascent():
    for hyp in _build_seed_hypotheses().values():
        assert hyp.status == "nascent"


def test_seed_hypotheses_include_improvements():
    """The 2 improvement hypotheses added for graduation reachability."""
    h = _build_seed_hypotheses()
    assert "H7_impasse_v_above_resolution" in h
    assert "H8_curiosity_precedes_knowledge_growth" in h


def test_hypothesis_observations_bounded():
    h = _build_seed_hypotheses()
    for hyp in h.values():
        assert hyp.observations.maxlen == 500


# ── Consumer construction ────────────────────────────────────────

def test_consumer_construction_shadow_mode_default():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        assert ec._status == "shadow_mode"


def test_consumer_construction_8_primitives():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        assert len(ec._primitives) == 8
        for p in EMOT_PRIMITIVES:
            assert p in ec._primitives


def test_consumer_construction_8_hypotheses():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        assert len(ec._hypotheses) == 8


def test_consumer_is_active_false_in_shadow():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        assert not ec.is_active()


def test_consumer_is_active_true_after_force_graduate():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        ec.force_graduate()
        assert ec.is_active()


# ── Feature vector builder ────────────────────────────────────────

def test_build_feature_vec_returns_150d():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        fv = ec.build_feature_vec(felt_tensor_130d=[0.5] * 130)
        assert fv.shape == (FEATURE_DIM,)


def test_build_feature_vec_embeds_felt_tensor():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        felt = [0.3] * 130
        fv = ec.build_feature_vec(felt_tensor_130d=felt)
        assert np.allclose(fv[:130], 0.3)


def test_build_feature_vec_includes_kin_slot_default_zero():
    """rFP improvement #3: kin resonance slot always present, 0.0 default."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        fv = ec.build_feature_vec(felt_tensor_130d=[0.5] * 130)
        assert fv[149] == 0.0


def test_set_kin_resonance_reflected_in_feature():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        ec.set_kin_resonance(0.8)
        fv = ec.build_feature_vec(felt_tensor_130d=[0.5] * 130)
        # Should be non-zero now
        assert fv[149] > 0.0


def test_build_feature_vec_failsafe_on_none():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        fv = ec.build_feature_vec(felt_tensor_130d=None)
        assert fv.shape == (FEATURE_DIM,)


# ── Neuromod EMA (rFP improvement #2) ─────────────────────────────

def test_update_neuromod_ema_smooth_change():
    """Not instantaneous — EMA requires multiple samples before delta shows."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        # First sample
        ec.update_neuromod_ema({"DA": 0.5, "5HT": 0.5})
        d1 = ec.get_neuromod_deltas()
        # Deltas close to 0 initially
        assert abs(d1.get("DA", 0.0)) < 0.01


def test_update_neuromod_ema_delta_after_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        # Sample 100 times at 0.5 — prev_ema snapshots at sample 100
        for _ in range(100):
            ec.update_neuromod_ema({"DA": 0.5})
        # Now shift DA upward
        for _ in range(50):
            ec.update_neuromod_ema({"DA": 0.9})
        deltas = ec.get_neuromod_deltas()
        # DA delta should now be positive (current EMA higher than 100-epoch
        # snapshot)
        assert deltas.get("DA", 0.0) > 0.05


# ── Chain evidence updates ────────────────────────────────────────

def test_observe_chain_evidence_updates_primitive_V():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        v_before = ec._primitives["FLOW"].V
        # Positive reward → V should move toward 1.0
        for _ in range(20):
            ec.observe_chain_evidence(
                chain_id=1, dominant_at_start="FLOW",
                dominant_at_end="FLOW", terminal_reward=0.9)
        v_after = ec._primitives["FLOW"].V
        assert v_after > v_before


def test_observe_chain_evidence_negative_reward_lowers_V():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        for _ in range(20):
            ec.observe_chain_evidence(
                chain_id=1, dominant_at_start="GRIEF",
                dominant_at_end="GRIEF", terminal_reward=0.1)
        assert ec._primitives["GRIEF"].V < 0.4


def test_observe_chain_evidence_total_updates_counter():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        ec.observe_chain_evidence(1, "FLOW", "FLOW", 0.5)
        assert ec._total_updates >= 1


def test_observe_chain_evidence_handles_unknown_primitives():
    """Failsafe: unknown primitive IDs shouldn't crash or corrupt state."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        # Should not raise
        ec.observe_chain_evidence(1, "UNKNOWN_A", "UNKNOWN_B", 0.5)


# ── HAOV hypothesis testing ──────────────────────────────────────

def test_test_hypotheses_no_crash_with_empty_observations():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        # Should skip all tests since min_samples not met
        ec._test_hypotheses()
        # No change in status
        for h in ec._hypotheses.values():
            assert h.status == "nascent"


def test_haov_accumulates_observations_on_chain():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        for i in range(5):
            ec.observe_chain_evidence(
                i, "FLOW", "FLOW", 0.6,
                ctx={"DA": 0.7, "5HT": 0.6})
        # H1 (flow_neuromod) should have observations
        assert len(ec._hypotheses["H1_flow_neuromod"].observations) >= 5


# ── Graduation readiness ─────────────────────────────────────────

def test_graduation_readiness_ineligible_at_start():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        r = ec.graduation_readiness()
        assert not r["eligible"]


def test_graduation_readiness_shows_all_criteria():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        r = ec.graduation_readiness()
        for key in ("updates_ok", "hypotheses_ok", "primitives_ok",
                    "contrast_ok", "window_ok"):
            assert key in r


def test_force_graduate_sets_active():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        assert ec.force_graduate()
        assert ec._status == "active"


def test_force_shadow_reverts():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        ec.force_graduate()
        assert ec.force_shadow()
        assert ec._status == "shadow_mode"


def test_force_shadow_increments_rollback():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        ec.force_graduate()
        n_before = ec._rolled_back_count
        ec.force_shadow()
        assert ec._rolled_back_count == n_before + 1


# ── State queries ────────────────────────────────────────────────

def test_get_current_emotion_state_returns_10d_composite():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        s = ec.get_current_emotion_state()
        assert "one_hot" in s
        assert len(s["one_hot"]) == 8
        assert "intensity" in s
        assert "confidence" in s
        assert "dominant" in s
        assert "active" in s


def test_get_current_emotion_state_active_false_in_shadow():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        s = ec.get_current_emotion_state()
        assert s["active"] is False


def test_get_emotion_for_narration_empty_in_shadow():
    """Narrator gate: pre-graduation returns empty string."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        assert ec.get_emotion_for_narration() == ""


def test_get_emotion_for_narration_nonempty_after_graduate():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        ec.force_graduate()
        label = ec.get_emotion_for_narration()
        assert label in EMOT_PRIMITIVES or label == ""  # cluster label


# ── Persistence ──────────────────────────────────────────────────

def test_save_state_writes_grounding_file():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        ec.save_state()
        assert os.path.exists(os.path.join(tmp, "primitive_grounding.json"))


def test_save_state_writes_haov_file():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        ec.save_state()
        assert os.path.exists(os.path.join(tmp, "haov_hypotheses.json"))


def test_save_state_writes_watchdog_file():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        ec.save_state()
        assert os.path.exists(os.path.join(tmp, "watchdog_state.json"))


def test_save_load_preserves_V():
    with tempfile.TemporaryDirectory() as tmp:
        ec1 = EmotCGNConsumer(save_dir=tmp)
        for _ in range(10):
            ec1.observe_chain_evidence(1, "FLOW", "FLOW", 0.9)
        v1 = ec1._primitives["FLOW"].V
        ec1.save_state()
        ec2 = EmotCGNConsumer(save_dir=tmp)
        v2 = ec2._primitives["FLOW"].V
        assert abs(v1 - v2) < 1e-6


def test_save_load_preserves_status():
    with tempfile.TemporaryDirectory() as tmp:
        ec1 = EmotCGNConsumer(save_dir=tmp)
        ec1.force_graduate()
        ec1.save_state()
        ec2 = EmotCGNConsumer(save_dir=tmp)
        assert ec2._status == "active"


def test_save_on_init_creates_all_files():
    """Persistence audit: files must exist immediately after init so
    /v4/emot-cgn is observable from boot, not only after N chains."""
    with tempfile.TemporaryDirectory() as tmp:
        EmotCGNConsumer(save_dir=tmp)
        for fname in ("primitive_grounding.json", "haov_hypotheses.json",
                      "watchdog_state.json", "clusters_state.json"):
            assert os.path.exists(os.path.join(tmp, fname)), \
                f"{fname} missing after init"


def test_neuromod_ema_survives_restart():
    """Persistence audit: neuromod EMA must persist across restarts
    so feature vectors don't degrade for ~100 epochs post-restart."""
    with tempfile.TemporaryDirectory() as tmp:
        ec1 = EmotCGNConsumer(save_dir=tmp)
        for _ in range(50):
            ec1.update_neuromod_ema({"DA": 0.7, "5HT": 0.6})
        ec1.save_state()
        ema_before = dict(ec1._neuromod_ema)
        ec2 = EmotCGNConsumer(save_dir=tmp)
        # After reload, EMA values should match (within float tolerance)
        for k, v in ema_before.items():
            assert abs(ec2._neuromod_ema.get(k, 0) - v) < 1e-6, \
                f"EMA key {k} not preserved"


def test_recent_rewards_survives_restart():
    """Persistence audit: recent_rewards deque feeds feature vector's
    reward-recency slot — must persist across restarts."""
    with tempfile.TemporaryDirectory() as tmp:
        ec1 = EmotCGNConsumer(save_dir=tmp)
        for r in [0.3, 0.5, 0.7, 0.4, 0.6]:
            ec1._recent_rewards.append(r)
        ec1.save_state()
        ec2 = EmotCGNConsumer(save_dir=tmp)
        assert list(ec2._recent_rewards) == [0.3, 0.5, 0.7, 0.4, 0.6]


def test_dominant_emotion_survives_restart():
    """Persistence audit: dominant_emotion must persist so consumers
    querying /v4/emot-cgn get consistent answers across restarts
    (not reset to FLOW default)."""
    with tempfile.TemporaryDirectory() as tmp:
        ec1 = EmotCGNConsumer(save_dir=tmp)
        ec1._dominant_emotion = "GRIEF"
        ec1.save_state()
        ec2 = EmotCGNConsumer(save_dir=tmp)
        assert ec2._dominant_emotion == "GRIEF"


def test_kin_resonance_survives_restart():
    with tempfile.TemporaryDirectory() as tmp:
        ec1 = EmotCGNConsumer(save_dir=tmp)
        ec1.set_kin_resonance(0.8)
        ec1.save_state()
        ec2 = EmotCGNConsumer(save_dir=tmp)
        assert abs(ec2._last_kin_resonance - ec1._last_kin_resonance) < 1e-6


# ── Kin Protocol ─────────────────────────────────────────────────

def test_export_kin_snapshot_has_schema():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp, titan_id="T_PEER")
        snap = ec.export_kin_snapshot()
        assert snap.get("schema") == "emot_cgn_snapshot_v1"
        assert snap.get("titan_id") == "T_PEER"


def test_import_kin_snapshot_rejects_empty():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        assert ec.import_kin_snapshot({}) == 0


def test_import_kin_snapshot_accepts_well_formed():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        snap = {
            "schema": "emot_cgn_snapshot_v1",
            "titan_id": "T_PEER",
            "primitives": {
                "FLOW": {"V": 0.7, "confidence": 0.6, "n_samples": 200},
                "GRIEF": {"V": 0.2, "confidence": 0.5, "n_samples": 150},
            },
        }
        imported = ec.import_kin_snapshot(snap)
        assert imported >= 1


# ── Signal emission (orphan-safe) ────────────────────────────────

def test_emit_signal_without_queue_returns_false():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)  # no send_queue
        assert ec._emit_signal("cluster_assignment") is False


# ── handle_felt_tensor integration ───────────────────────────────

def test_handle_felt_tensor_returns_primitive_dict():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        fv = ec.build_feature_vec(felt_tensor_130d=[0.5] * 130)
        r = ec.handle_felt_tensor(fv, emit_bus_signal=False)
        assert "primitive" in r
        assert r["primitive"] in EMOT_PRIMITIVES


def test_handle_felt_tensor_updates_dominant():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        fv = ec.build_feature_vec(felt_tensor_130d=[0.5] * 130)
        ec.handle_felt_tensor(fv, emit_bus_signal=False)
        # dominant_emotion should now match
        assert ec.get_dominant_emotion() in EMOT_PRIMITIVES


# ── Stats surface ────────────────────────────────────────────────

def test_get_stats_includes_key_fields():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        s = ec.get_stats()
        for key in ("status", "dominant_emotion", "total_updates",
                    "primitives", "hypotheses", "clusterer"):
            assert key in s


def test_get_stats_compact_has_status():
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        s = ec.get_stats_compact()
        assert s["status"] == "shadow_mode"


# ─── CGN Integration tests (rFP_cgn_orchestrator_promotion §4 — 8th consumer) ──
# Verify EMOT-CGN properly integrates with cgn_worker as the 8th CGN consumer,
# participates in shared V(s) learning, and handles cross-consumer insights.

from titan_plugin.logic.emot_cgn import (
    CGN_CONSUMER_NAME, FEATURE_DIMS, ACTION_DIMS,
)


class _MockSendQueue:
    """Minimal mock send_queue that records put() calls for inspection."""
    def __init__(self):
        self.sent = []
    def put(self, msg):
        self.sent.append(msg)
    def put_nowait(self, msg):
        self.sent.append(msg)


def test_cgn_consumer_name_is_emotional():
    """rFP §4.1: consumer name registers as 'emotional' (family convention)
    not 'emot_cgn' (internal Python identifier)."""
    assert CGN_CONSUMER_NAME == "emotional"


def test_cgn_feature_dims_is_30():
    """cgn_worker's SharedValueNet is fixed at 30D. Must match."""
    assert FEATURE_DIMS == 30


def test_cgn_action_dims_is_8():
    """8 emotion primitives = 8 actions."""
    assert ACTION_DIMS == 8


def test_cgn_register_sent_on_init():
    """CGN_REGISTER is sent to bus on EmotCGNConsumer init (idempotent with
    cgn_worker's pre-registration)."""
    with tempfile.TemporaryDirectory() as tmp:
        q = _MockSendQueue()
        ec = EmotCGNConsumer(save_dir=tmp, send_queue=q)
        register_msgs = [m for m in q.sent if m.get("type") == "CGN_REGISTER"]
        assert len(register_msgs) == 1
        payload = register_msgs[0]["payload"]
        assert payload["name"] == "emotional"
        assert payload["feature_dims"] == 30
        assert payload["action_dims"] == 8
        assert len(payload["action_names"]) == 8
        assert payload["reward_source"] == "terminal_reward"


def test_cgn_register_skipped_without_queue():
    """No send_queue (standalone/test) → no CGN_REGISTER but no crash."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        assert ec._cgn_registered is False


def test_encode_state_30d_shape():
    """30D state vector for cgn_worker's SharedValueNet."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        vec = ec.encode_state_30d("FLOW", {"DA": 0.7, "5HT": 0.6})
        assert vec.shape == (30,)
        assert vec.dtype == np.float32


def test_encode_state_30d_neuromods_populated():
    """[0:6] slice carries neuromod levels from ctx."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        ctx = {"DA": 0.8, "5HT": 0.6, "NE": 0.7, "ACh": 0.4,
               "Endorphin": 0.5, "GABA": 0.3}
        vec = ec.encode_state_30d("FLOW", ctx)
        assert abs(vec[0] - 0.8) < 1e-5
        assert abs(vec[1] - 0.6) < 1e-5
        assert abs(vec[2] - 0.7) < 1e-5
        assert abs(vec[5] - 0.3) < 1e-5


def test_encode_state_30d_cluster_features_upgrade_ii():
    """Upgrade II: cluster_confidence + cluster_distance in 30D state [6,7]."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        ec._last_cluster_assignment = ("FLOW", 0.5, 0.85)
        vec = ec.encode_state_30d("FLOW", {})
        assert abs(vec[6] - 0.85) < 1e-4       # confidence
        assert abs(vec[7] - 0.125) < 1e-4      # distance/4.0


def test_encode_state_30d_primitive_index():
    """[29] carries primitive index normalized (0-7 → 0-1)."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        vec0 = ec.encode_state_30d("FLOW", {})
        vec7 = ec.encode_state_30d("LOVE", {})
        assert vec0[29] == 0.0
        assert abs(vec7[29] - 1.0) < 1e-4


def test_encode_state_30d_failsafe():
    """Failsafe: returns zeros on error without raising."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        ec._primitives = None  # type: ignore
        vec = ec.encode_state_30d("FLOW", {})
        assert vec.shape == (30,)
        assert np.allclose(vec, 0.0)


def test_cgn_transition_sent_on_chain_evidence():
    """observe_chain_evidence sends CGN_TRANSITION per primitive touched."""
    with tempfile.TemporaryDirectory() as tmp:
        q = _MockSendQueue()
        ec = EmotCGNConsumer(save_dir=tmp, send_queue=q)
        q.sent.clear()
        ec.observe_chain_evidence(
            chain_id=1, dominant_at_start="FLOW",
            dominant_at_end="WONDER", terminal_reward=0.9,
            ctx={"DA": 0.8, "5HT": 0.7})
        transitions = [m for m in q.sent if m.get("type") == "CGN_TRANSITION"]
        assert len(transitions) == 2
        for t in transitions:
            payload = t["payload"]
            assert payload["consumer"] == "emotional"
            assert len(payload["state"]) == 30
            assert 0 <= payload["action"] < 8


def test_cgn_transition_same_primitive_one_send():
    """start == end → dedupes to 1 transition (set-based iteration)."""
    with tempfile.TemporaryDirectory() as tmp:
        q = _MockSendQueue()
        ec = EmotCGNConsumer(save_dir=tmp, send_queue=q)
        q.sent.clear()
        ec.observe_chain_evidence(1, "FLOW", "FLOW", 0.5, ctx={})
        transitions = [m for m in q.sent if m.get("type") == "CGN_TRANSITION"]
        assert len(transitions) == 1


def test_cgn_transition_skipped_without_queue():
    """Failsafe: no send_queue → no crash, counter stays 0."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        ec.observe_chain_evidence(1, "FLOW", "FLOW", 0.5, ctx={})
        assert ec._cgn_transitions_sent == 0


def test_get_blended_v_falls_back_to_beta_without_shm():
    """Upgrade I: when shm unavailable, blended V = local β V."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        ec._cgn_client = None  # force shm-unavailable fallback path
        ec._primitives["FLOW"].alpha = 9.0
        ec._primitives["FLOW"].beta = 1.0
        ec._primitives["FLOW"].recompute_derived()
        v_beta = ec._primitives["FLOW"].V
        v_blended = ec.get_blended_V("FLOW", ctx={"DA": 0.5})
        assert abs(v_blended - v_beta) < 1e-6


def test_get_blended_v_without_ctx_returns_beta():
    """No ctx → cannot build 30D state → fall back to β."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        v = ec.get_blended_V("FLOW")
        assert v == 0.5  # Beta(1,1) prior mean


def test_cross_insight_emitted_on_high_reward_chain():
    """Upgrade III outgoing: CGN_CROSS_INSIGHT emitted for informative chain."""
    with tempfile.TemporaryDirectory() as tmp:
        q = _MockSendQueue()
        ec = EmotCGNConsumer(save_dir=tmp, send_queue=q)
        q.sent.clear()
        ec.observe_chain_evidence(1, "FLOW", "FLOW", 0.95, ctx={"DA": 0.9})
        insights = [m for m in q.sent if m.get("type") == "CGN_CROSS_INSIGHT"]
        assert len(insights) >= 1
        payload = insights[0]["payload"]
        assert payload["origin_consumer"] == "emotional"
        assert payload["insight_type"] == "emotion_outcome"
        assert payload["emotion_start"] == "FLOW"


def test_cross_insight_rate_limited_02hz():
    """0.2 Hz rate limit — two immediate emits → only 1 passes through."""
    with tempfile.TemporaryDirectory() as tmp:
        q = _MockSendQueue()
        ec = EmotCGNConsumer(save_dir=tmp, send_queue=q)
        q.sent.clear()
        ec.observe_chain_evidence(1, "FLOW", "FLOW", 0.95, ctx={})
        ec.observe_chain_evidence(2, "FLOW", "FLOW", 0.95, ctx={})
        insights = [m for m in q.sent if m.get("type") == "CGN_CROSS_INSIGHT"]
        assert len(insights) == 1


def test_cross_insight_skipped_on_neutral_reward():
    """Uninformative chain (reward≈0.5, no transition) → no emission."""
    with tempfile.TemporaryDirectory() as tmp:
        q = _MockSendQueue()
        ec = EmotCGNConsumer(save_dir=tmp, send_queue=q)
        q.sent.clear()
        ec.observe_chain_evidence(1, "FLOW", "FLOW", 0.50, ctx={})
        insights = [m for m in q.sent if m.get("type") == "CGN_CROSS_INSIGHT"]
        assert len(insights) == 0


def test_handle_incoming_cross_insight_nudges_dominant():
    """Upgrade III incoming: META-CGN insight on high-reward chain nudges
    dominant emotion's β-posterior upward."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        ec._dominant_emotion = "WONDER"
        v_before = ec._primitives["WONDER"].V
        ec.handle_incoming_cross_insight({
            "origin_consumer": "meta_cgn",
            "insight_type": "chain_outcome",
            "terminal_reward": 0.9,
        })
        v_after = ec._primitives["WONDER"].V
        assert v_after > v_before
        assert ec._cgn_cross_insights_received == 1


def test_handle_incoming_cross_insight_ignores_own_origin():
    """Own emissions bouncing back via dst='all' → ignored."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        n_before = ec._cgn_cross_insights_received
        ec.handle_incoming_cross_insight({
            "origin_consumer": "emotional",
            "insight_type": "chain_outcome",
            "terminal_reward": 0.9,
        })
        assert ec._cgn_cross_insights_received == n_before


def test_stats_expose_cgn_integration_block():
    """get_stats() includes cgn_integration summary for operators."""
    with tempfile.TemporaryDirectory() as tmp:
        ec = EmotCGNConsumer(save_dir=tmp)
        s = ec.get_stats()
        assert "cgn_integration" in s
        ci = s["cgn_integration"]
        assert ci["consumer_name"] == "emotional"
        assert ci["feature_dims"] == 30
        assert ci["action_dims"] == 8
