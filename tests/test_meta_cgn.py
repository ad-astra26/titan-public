"""META-CGN Phase 1 tests — consumer lifecycle, encoding, persistence, failsafe.

Run isolated (separate pytest process) per TorchRL mmap convention:
    python -m pytest tests/test_meta_cgn.py -v -p no:anchorpy --tb=short
"""
import json
import os
import tempfile
from queue import Queue

import numpy as np
import pytest


# ── Basic construction + registration ─────────────────────────────────

def test_construction_without_send_queue():
    """Consumer must init safely with no bus (test / standalone mode)."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer, PRIMITIVES
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp, titan_id="T1")
        # No crash, all primitives present, none registered (no bus)
        assert c._registered is False
        assert len(c._primitives) == len(PRIMITIVES)
        for p in PRIMITIVES:
            assert p in c._primitives
            assert c._primitives[p].V == 0.5
            assert c._primitives[p].n_samples == 0


def test_register_sends_bus_message():
    """With a send_queue, consumer must emit exactly one CGN_REGISTER."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, PRIMITIVES, FEATURE_DIMS, ACTION_DIMS)
    q = Queue()
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=q, save_dir=tmp)
        assert c._registered is True
        msgs = []
        while not q.empty():
            msgs.append(q.get_nowait())
        register_msgs = [m for m in msgs if m.get("type") == "CGN_REGISTER"]
        assert len(register_msgs) == 1
        payload = register_msgs[0]["payload"]
        assert payload["name"] == "meta"
        assert payload["feature_dims"] == FEATURE_DIMS
        assert payload["action_dims"] == ACTION_DIMS
        assert payload["action_names"] == list(PRIMITIVES)


# ── State encoding ────────────────────────────────────────────────────

def test_encode_state_shape_and_one_hot():
    """encode_state must return 30D float32 with one-hot primitive index."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, FEATURE_DIMS, PRIMITIVE_INDEX)
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        vec = c.encode_state("FORMULATE", {})
        assert vec.shape == (FEATURE_DIMS,)
        assert vec.dtype == np.float32
        assert vec[PRIMITIVE_INDEX["FORMULATE"]] == 1.0
        # All non-primitive one-hot slots are zero
        for p, idx in PRIMITIVE_INDEX.items():
            if p != "FORMULATE":
                assert vec[idx] == 0.0


def test_encode_state_unknown_primitive_no_crash():
    """Unknown primitive name must return zero vector (no one-hot set)."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer, FEATURE_DIMS
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        vec = c.encode_state("NONEXISTENT_PRIMITIVE", {})
        assert vec.shape == (FEATURE_DIMS,)
        # First 9 slots (primitive one-hot) all zero
        assert np.all(vec[:9] == 0.0)


def test_encode_state_clips_out_of_range():
    """encode_state must clip extreme values to [0, 1]."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        ctx = {
            "chain_len": 9999,       # clips to 1.0 (div 20)
            "DA": -5.0,              # clips to 0.0
            "monoculture_share": 2.5,  # clips to 1.0
            "terminal_reward_ema": float("inf"),  # NaN/inf safety
        }
        vec = c.encode_state("RECALL", ctx)
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)
        assert np.all(np.isfinite(vec))


# ── Grounding updates ────────────────────────────────────────────────

def test_update_primitive_V_ema_moves_toward_target():
    """Repeated quality=1.0 updates must raise V toward 1.0.

    2026-04-21 B-phase: under per-update EMA (γ<1.0, see COMPOSITION_DEFAULTS),
    n_samples reflects geometric-series effective-sample-size not raw count.
    At γ=0.9999 with 30 updates, n_samples ≈ 29.996 → int = 29 or 30 depending
    on float rounding. Use range assertion to accept either."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        p = c._primitives["FORMULATE"]
        initial_V = p.V
        for _ in range(30):
            c.update_primitive_V("FORMULATE", quality=1.0, chain_id=1)
        assert p.V > initial_V
        assert p.V > 0.8          # approaching 1.0
        # EMA at γ close to 1 yields n_samples very close to raw count.
        # Allow ±2 to survive any γ in (0.99, 1.0] without re-updating.
        assert 28 <= p.n_samples <= 30
        assert p.confidence > 0.0


def test_update_primitive_V_confidence_grows_with_samples():
    """Confidence must grow as samples accumulate."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        conf_at = []
        for step in range(200):
            c.update_primitive_V("EVALUATE", quality=0.7)
            if step in (10, 50, 100, 199):
                conf_at.append(c._primitives["EVALUATE"].confidence)
        # Monotonically non-decreasing after initial warmup
        assert conf_at[-1] > conf_at[0]


def test_update_unknown_primitive_is_no_op():
    """Unknown primitive ID must not raise, must not mutate any concept."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        before = {p: c._primitives[p].V for p in c._primitives}
        c.update_primitive_V("FAKE_PRIM", quality=1.0)
        after = {p: c._primitives[p].V for p in c._primitives}
        assert before == after


# ── Transitions ──────────────────────────────────────────────────────

def test_send_transition_bus_payload():
    """send_transition must enqueue CGN_TRANSITION with correct schema."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer, PRIMITIVE_INDEX
    q = Queue()
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=q, save_dir=tmp)
        # Drain registration
        while not q.empty():
            q.get_nowait()
        state = np.zeros(30, dtype=np.float32)
        c.send_transition("SYNTHESIZE", state, 0.75, chain_id=42,
                          metadata={"epoch": 100})
        msg = q.get_nowait()
        assert msg["type"] == "CGN_TRANSITION"
        assert msg["dst"] == "cgn"
        payload = msg["payload"]
        assert payload["consumer"] == "meta"
        assert payload["action"] == PRIMITIVE_INDEX["SYNTHESIZE"]
        assert payload["action_name"] == "SYNTHESIZE"
        assert payload["reward"] == 0.75
        assert len(payload["state"]) == 30


def test_send_transition_no_queue_is_silent():
    """Without a bus, send_transition must not raise."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c.send_transition("BREAK", np.zeros(30), 0.5)


# ── Shadow-mode composition ──────────────────────────────────────────

def test_compose_template_score_returns_shadow_only():
    """Phase 1 composition must mark shadow_only=True and never raise."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        result = c.compose_template_score(
            template_id="FORMULATE→RECALL→EVALUATE",
            state_ctx={"chain_len": 3},
            chain_iql_score=0.7,
            chain_iql_confidence=0.6,
            template_primitives=["FORMULATE", "RECALL", "EVALUATE"],
        )
        assert result["shadow_only"] is True
        assert "direct_Q" in result
        assert "composed_V" in result
        assert "lambda_used" in result
        assert 0.1 <= result["lambda_used"] <= 0.9
        assert "disagreement" in result


def test_compose_logs_disagreement_only_when_confident():
    """Disagreements should only log when V_confidence is sufficient."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # First call — primitives have 0 confidence, disagreement should NOT log
        c.compose_template_score(
            template_id="T1",
            state_ctx={},
            chain_iql_score=0.9,
            chain_iql_confidence=0.9,
            template_primitives=["FORMULATE"],
        )
        assert c._total_disagreements == 0
        # Ground FORMULATE until confidence > 0.3
        # B.2 (2026-04-21): iterations increased 200 → 500 so assertion holds
        # under BOTH old hard-cap `min(1.0, n/500)` (gives 1.0 at n=500) AND
        # new asymptotic `n/(n+500)` (gives 0.5 at n=500). Both >= 0.3.
        for _ in range(500):
            c.update_primitive_V("FORMULATE", quality=0.1)
        assert c._primitives["FORMULATE"].confidence > 0.3
        # Now disagreement SHOULD log (direct_Q=0.9, composed_V near 0.1)
        c.compose_template_score(
            template_id="T2",
            state_ctx={},
            chain_iql_score=0.9,
            chain_iql_confidence=0.9,
            template_primitives=["FORMULATE"],
        )
        assert c._total_disagreements >= 1


# ── Persistence ──────────────────────────────────────────────────────

def test_save_and_reload_roundtrip():
    """save_state -> new consumer -> _load_state must preserve primitive state."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c1 = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        for _ in range(25):
            c1.update_primitive_V("HYPOTHESIZE", quality=0.8)
        c1.save_state()
        V_saved = c1._primitives["HYPOTHESIZE"].V
        n_saved = c1._primitives["HYPOTHESIZE"].n_samples

        # New consumer from same dir — should reload
        c2 = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        assert abs(c2._primitives["HYPOTHESIZE"].V - V_saved) < 1e-4
        assert c2._primitives["HYPOTHESIZE"].n_samples == n_saved


def test_save_file_is_valid_json_with_expected_shape():
    """Persistence file must be well-formed JSON with all 9 primitives."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer, PRIMITIVES
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c.update_primitive_V("FORMULATE", 1.0)
        c.save_state()
        path = os.path.join(tmp, "primitive_grounding.json")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "primitives" in data
        assert set(data["primitives"].keys()) == set(PRIMITIVES)
        assert data["version"] >= 1


# ── Telemetry ────────────────────────────────────────────────────────

def test_get_stats_schema():
    """get_stats must return all documented fields (contract for audit endpoint)."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        stats = c.get_stats()
        for key in ("status", "registered", "consumer_name", "feature_dims",
                    "action_dims", "primitives_total", "primitives_grounded",
                    "primitive_V_summary", "transitions_sent",
                    "updates_applied", "compositions_computed",
                    "disagreements_logged", "ready_to_graduate"):
            assert key in stats, f"missing stat: {key}"
        assert stats["status"] == "shadow_mode"
        assert stats["primitives_total"] == 9
        assert stats["feature_dims"] == 30
        assert stats["action_dims"] == 9


def test_not_ready_to_graduate_at_init():
    """Fresh consumer must not be ready to graduate."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        assert c.get_stats()["ready_to_graduate"] is False


# ── Phase 2: HAOV hypothesis tests ──────────────────────────────────

def test_seed_hypotheses_loaded_at_init():
    """5 seed hypotheses must be registered at construction."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        assert len(c._hypotheses) == 6   # P7: added H6_advisor_disagreement
        for hid in ("H1_monoculture", "H2_domain_affinity",
                    "H3_position_effect", "H4_mono_context_v_drop",
                    "H5_impasse_primitives"):
            assert hid in c._hypotheses
            assert c._hypotheses[hid].status == "nascent"


def test_observe_chain_evidence_populates_hypotheses():
    """observe_chain_evidence must fan evidence out to all 5 hypotheses."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c.observe_chain_evidence({
            "chain_id": 1,
            "primitives": ["FORMULATE", "RECALL", "BREAK"],
            "quality": 0.4,
            "domain": "problem_solving",
            "monoculture_share": 0.8,
            "is_in_soar_impasse": True,
            "dominant_primitive": "FORMULATE",
            "pop_avg_V": 0.5,
            "per_primitive_V": {"FORMULATE": 0.3, "RECALL": 0.5, "BREAK": 0.4},
        })
        # H1, H4 each get 1 obs (dominant primitive–centric)
        assert len(c._hypotheses["H1_monoculture"].observations) == 1
        assert len(c._hypotheses["H4_mono_context_v_drop"].observations) == 1
        # H2 gets one per distinct primitive
        assert len(c._hypotheses["H2_domain_affinity"].observations) == 3
        # H3 gets one per position
        assert len(c._hypotheses["H3_position_effect"].observations) == 3
        # H5 only gets BREAK (one primitive in {BREAK, HYPOTHESIZE})
        assert len(c._hypotheses["H5_impasse_primitives"].observations) == 1


def test_h1_monoculture_confirms_when_dominant_V_is_low():
    """H1 must confirm when dominant primitive V is meaningfully below pop avg."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Push 35 observations where dominant V is 0.3 but pop avg is 0.5
        for _ in range(35):
            c.observe_chain_evidence({
                "chain_id": 1,
                "primitives": ["FORMULATE"],
                "quality": 0.3,
                "domain": "x",
                "monoculture_share": 0.5,
                "dominant_primitive": "FORMULATE",
                "pop_avg_V": 0.5,
                "per_primitive_V": {"FORMULATE": 0.3},
            })
        c._run_due_tests()
        h = c._hypotheses["H1_monoculture"]
        assert h.status == "confirmed"
        # Effect should be ≈ 0.2
        assert h.effect_size > 0.1
        # Confirmed hypothesis tagged the dominant primitive
        assert "H1_monoculture" in c._primitives["FORMULATE"].haov_rules


def test_h5_impasse_confirms_when_break_helps_in_impasse():
    """H5 must confirm when BREAK V in impasse > non-impasse."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # 10 impasse BREAK observations at quality 0.7
        for _ in range(10):
            c.observe_chain_evidence({
                "primitives": ["BREAK"],
                "quality": 0.7,
                "is_in_soar_impasse": True,
                "dominant_primitive": "BREAK",
                "pop_avg_V": 0.5,
                "per_primitive_V": {"BREAK": 0.5},
                "monoculture_share": 0.3,
            })
        # 15 non-impasse BREAK observations at quality 0.3
        for _ in range(15):
            c.observe_chain_evidence({
                "primitives": ["BREAK"],
                "quality": 0.3,
                "is_in_soar_impasse": False,
                "dominant_primitive": "BREAK",
                "pop_avg_V": 0.5,
                "per_primitive_V": {"BREAK": 0.5},
                "monoculture_share": 0.3,
            })
        c._run_due_tests()
        h = c._hypotheses["H5_impasse_primitives"]
        assert h.status == "confirmed"
        assert h.effect_size > 0.2


def test_hypothesis_nascent_when_below_min_samples():
    """Hypothesis must stay nascent if it doesn't have min_samples."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Only 5 observations — H1 needs 30
        for _ in range(5):
            c.observe_chain_evidence({
                "primitives": ["FORMULATE"], "quality": 0.3,
                "dominant_primitive": "FORMULATE", "pop_avg_V": 0.5,
                "per_primitive_V": {"FORMULATE": 0.3},
                "monoculture_share": 0.5, "is_in_soar_impasse": False,
            })
        c._run_due_tests()
        assert c._hypotheses["H1_monoculture"].status == "nascent"


def test_haov_state_persists_across_reload():
    """save_state + reload must preserve hypothesis status + effect_size."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c1 = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        for _ in range(35):
            c1.observe_chain_evidence({
                "primitives": ["FORMULATE"], "quality": 0.3,
                "dominant_primitive": "FORMULATE", "pop_avg_V": 0.5,
                "per_primitive_V": {"FORMULATE": 0.3},
                "monoculture_share": 0.5, "is_in_soar_impasse": False,
            })
        c1._run_due_tests()
        assert c1._hypotheses["H1_monoculture"].status == "confirmed"
        saved_effect = c1._hypotheses["H1_monoculture"].effect_size
        c1.save_state()

        # Reload into new instance
        c2 = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        assert c2._hypotheses["H1_monoculture"].status == "confirmed"
        assert abs(c2._hypotheses["H1_monoculture"].effect_size - saved_effect) < 1e-4


def test_get_haov_stats_schema():
    """get_haov_stats must return expected structure."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        stats = c.get_haov_stats()
        assert "total" in stats
        assert "by_status" in stats
        assert "details" in stats
        assert stats["total"] == 6   # P7: added H6_advisor_disagreement
        assert stats["by_status"]["nascent"] == 6


def test_confirmed_hypothesis_raises_confidence_multiplier():
    """Confirmed hypothesis must set confidence_multiplier > 1."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        for _ in range(35):
            c.observe_chain_evidence({
                "primitives": ["RECALL"], "quality": 0.2,
                "dominant_primitive": "RECALL", "pop_avg_V": 0.6,
                "per_primitive_V": {"RECALL": 0.2},
                "monoculture_share": 0.5, "is_in_soar_impasse": False,
            })
        c._run_due_tests()
        h = c._hypotheses["H1_monoculture"]
        assert h.status == "confirmed"
        assert h.confidence_multiplier > 1.0


# ── Phase 3: composition aggregation + HAOV multiplier ─────────────

def test_composition_confidence_weighted_arithmetic_mean():
    """P6: Compose still uses confidence-weighted arithmetic (not geometric).

    With F anti-monoculture bonus, over-sampled primitives receive equal tanh-
    bounded penalty (≈ −κ_explore). So both primitives' contributions shift
    down by the same constant; the weighted mean remains arithmetic, shifted.
    Geometric mean of (0.8, 0.2) would be √0.16 ≈ 0.4 pre-shift → ≈ 0.25 shifted.
    Arithmetic mean ≈ 0.5 pre-shift → ≈ 0.35 shifted. Assertion: arithmetic-
    distinctive range, clearly above geometric post-shift floor.
    """
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        for _ in range(200):
            c.update_primitive_V("FORMULATE", quality=0.8)
        for _ in range(200):
            c.update_primitive_V("RECALL", quality=0.2)
        result = c.compose_template_score(
            template_id="FORMULATE→RECALL",
            state_ctx={},
            chain_iql_score=0.5,
            chain_iql_confidence=0.5,
            template_primitives=["FORMULATE", "RECALL"],
        )
        # Arithmetic-shifted ≈ 0.35; geometric-shifted ≈ 0.25. Bound allows both
        # tuning headroom and clearly rules out dominant-primitive collapse.
        assert 0.20 < result["composed_V"] < 0.55, result["composed_V"]


def test_composition_skips_ungrounded_primitives_gracefully():
    """Primitives with confidence 0 should contribute minimally."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Only ground FORMULATE
        for _ in range(100):
            c.update_primitive_V("FORMULATE", quality=0.9)
        # RECALL is untouched (V=0.5, conf=0)
        result = c.compose_template_score(
            template_id="FORMULATE→RECALL",
            state_ctx={},
            chain_iql_score=0.5,
            chain_iql_confidence=0.5,
            template_primitives=["FORMULATE", "RECALL"],
        )
        # V_confidence = min — RECALL has 0
        assert result["V_confidence"] == 0.0
        # Composed V should be between 0.5 and 0.9 (weighted toward grounded)
        assert result["composed_V"] > 0.5
        assert result["shadow_only"] is True


def test_confirmed_haov_boosts_composition_weight():
    """Confirmed HAOV rule on a primitive should raise its composition weight."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Ground two primitives to equal V + confidence
        for _ in range(100):
            c.update_primitive_V("FORMULATE", quality=0.3)
            c.update_primitive_V("RECALL", quality=0.7)
        # Without HAOV — composed is weighted-arithmetic (~0.5)
        r1 = c.compose_template_score(
            template_id="FORMULATE→RECALL",
            state_ctx={}, chain_iql_score=0.5, chain_iql_confidence=0.5,
            template_primitives=["FORMULATE", "RECALL"])
        # Force a confirmed HAOV rule on RECALL
        c._primitives["RECALL"].haov_rules.append("H1_monoculture")
        c._hypotheses["H1_monoculture"].status = "confirmed"
        c._hypotheses["H1_monoculture"].confidence_multiplier = 1.5
        r2 = c.compose_template_score(
            template_id="FORMULATE→RECALL",
            state_ctx={}, chain_iql_score=0.5, chain_iql_confidence=0.5,
            template_primitives=["FORMULATE", "RECALL"])
        # With RECALL boosted by 1.5×, composed V should shift toward 0.7
        assert r2["composed_V"] > r1["composed_V"]


def test_graduation_requires_hypothesis_confirmations():
    """Phase 2-tightened graduation requires ≥3 confirmed hypotheses."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer, PRIMITIVES
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Force all primitives to have many samples but no confirmed hypotheses
        for p in PRIMITIVES:
            for _ in range(60):
                c.update_primitive_V(p, quality=0.5)
        c._total_updates_applied = 3000   # surpass old threshold
        stats = c.get_stats()
        assert stats["primitives_well_sampled"] >= 5
        # Should NOT graduate: 0 confirmed hypotheses
        assert stats["ready_to_graduate"] is False


# ── Phase 4: Graduation state machine + rollback ───────────────────

def test_boot_selftest_passes_on_fresh_consumer():
    """P5 I-8: fresh consumer must pass boot self-test."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        assert c._status == "shadow_mode"


def test_boot_selftest_self_heals_from_stale_disabled_state():
    """2026-04-22: when watchdog state loads from disk as
    `disabled_boot_selftest_failed` (e.g. stuck from a prior code bug),
    boot selftest must RETRY and, if passing, flip status to `shadow_mode`
    and persist. `disabled_failsafe` (recent failsafe trip) remains sticky.
    """
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    import json
    import os
    with tempfile.TemporaryDirectory() as tmp:
        # Seed a stale disabled_boot_selftest_failed watchdog state — as if
        # a prior titan_main left behind the stuck flag before code fix.
        watchdog = {
            "version": 1, "saved_ts": 0,
            "status": "disabled_boot_selftest_failed",
            "cooldown_remaining": 0, "disabled_reason": "",
            "total_failures": 0, "failsafe_trip_count": 0,
            "last_failure_ts": 0.0, "window": [],
            "graduation_progress": 0, "graduation_ts": 0,
            "rolled_back_count": 0, "total_updates_applied": 0,
            "pre_graduation_baseline": {},
        }
        with open(os.path.join(tmp, "watchdog_state.json"), "w") as f:
            json.dump(watchdog, f)
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Self-heal: selftest retried on boot, passed, flipped to shadow_mode.
        assert c._status == "shadow_mode", (
            f"Self-healing retry expected, got status={c._status!r}. Check "
            f"meta_cgn.py disabled_boot_selftest_failed branch.")
        # Persisted state also cleared (next boot reloads shadow_mode clean).
        with open(os.path.join(tmp, "watchdog_state.json")) as f:
            persisted = json.load(f)
        assert persisted["status"] == "shadow_mode", (
            f"Self-heal must persist the cleared status to disk. "
            f"Got {persisted['status']!r}.")


def test_boot_selftest_failsafe_state_stays_sticky():
    """Complement to self-heal test: disabled_failsafe (recent failsafe trip)
    must NOT retry on boot — cooldown machinery handles clearing it."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    import json
    import os
    with tempfile.TemporaryDirectory() as tmp:
        watchdog = {
            "version": 1, "saved_ts": 0,
            "status": "disabled_failsafe",
            "cooldown_remaining": 500, "disabled_reason": "v_flatline",
            "total_failures": 5, "failsafe_trip_count": 1,
            "last_failure_ts": 0.0, "window": [],
            "graduation_progress": 0, "graduation_ts": 0,
            "rolled_back_count": 0, "total_updates_applied": 0,
            "pre_graduation_baseline": {},
        }
        with open(os.path.join(tmp, "watchdog_state.json"), "w") as f:
            json.dump(watchdog, f)
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        assert c._status == "disabled_failsafe"


def test_boot_selftest_passes_with_gamma_decay_on_saturated_posterior():
    """2026-04-22 regression: selftest must pass for persistence_loaded
    consumers even when γ<1.0 and n_samples is so large that the decay
    term (1-γ)·n dominates the +1 evidence → n_samples can DECREASE on
    update. Previous `n_samples == pre+1` check was mathematically
    impossible for saturated primitives and persistently failed boot
    after γ=0.9999/0.999 activation (commits 3427dfe, 64a92be).
    """
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    import json
    import os
    with tempfile.TemporaryDirectory() as tmp:
        # Pre-seed a saturated RECALL posterior — α+β ≫ 1/(1-γ) = 1000.
        # At these values, one update with w·q=0.5, w·(1-q)=0.5 yields
        # new (α+β) ≈ γ·(α+β) + 1 < α+β, and n_samples derived from
        # int(α+β-2·FLOOR) drops by ~(1-γ)·n_pre − 1 ≈ 5.
        grounding = {
            "version": 3,
            "titan_id": "T1",
            "saved_ts": 0,
            "primitives": {
                "RECALL": {
                    "primitive_id": "RECALL", "alpha": 2000.0, "beta": 4000.0,
                    "V": 0.333, "confidence": 0.98, "n_samples": 5998,
                    "variance": 0.0, "last_updated_ts": 0.0,
                    "last_updated_chain": 0, "haov_rules": [], "by_domain": {},
                },
                "FORMULATE": {
                    "primitive_id": "FORMULATE", "alpha": 1.0, "beta": 1.0,
                    "V": 0.5, "confidence": 0.0, "n_samples": 0,
                    "variance": 0.25, "last_updated_ts": 0.0,
                    "last_updated_chain": 0, "haov_rules": [], "by_domain": {},
                },
            },
        }
        with open(os.path.join(tmp, "primitive_grounding.json"), "w") as f:
            json.dump(grounding, f)
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Fresh save_dir → no watchdog state → selftest runs fresh.
        # With saturated RECALL + γ=0.999, this must still PASS, meaning
        # `c._status` stays "shadow_mode" (not flipped to disabled_*).
        assert c._status == "shadow_mode", (
            f"Boot selftest failed on saturated posterior with γ<1.0 — "
            f"status={c._status!r}. Fix (meta_cgn.py:1918-1930) checks α/β "
            f"change directly instead of derived n_samples."
        )


def test_force_graduate_transitions_to_graduating():
    """force_graduate() must flip status shadow → graduating."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        assert c._status == "shadow_mode"
        c.force_graduate()
        assert c._status == "graduating"
        assert c._graduation_progress == 0
        assert c._graduation_ts > 0
        assert "reward_mean" in c._pre_graduation_baseline
        assert "reward_std" in c._pre_graduation_baseline


def test_evaluate_graduation_ramps_progress():
    """After force_graduate, each evaluate_graduation call advances ramp."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c.force_graduate()
        for _ in range(50):
            c.evaluate_graduation()
        assert c._status == "graduating"
        assert c._graduation_progress == 50


def test_ramp_completes_at_100_chains():
    """Graduating → active after 100 evaluate_graduation calls."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c.force_graduate()
        for _ in range(100):
            c.evaluate_graduation()
        assert c._status == "active"
        assert c._graduation_progress == 100


def test_force_shadow_rollback():
    """force_shadow must revert from any state to shadow_mode."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c.force_graduate()
        # Ramp to active
        for _ in range(100):
            c.evaluate_graduation()
        assert c._status == "active"
        # Force rollback
        c.force_shadow()
        assert c._status == "shadow_mode"
        assert c._rolled_back_count == 1


def test_rerank_templates_in_shadow_preserves_top1():
    """In shadow mode, rerank MUST return chain_iql's top pick unchanged."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Ground RECALL high, FORMULATE low
        for _ in range(100):
            c.update_primitive_V("RECALL", 0.9)
            c.update_primitive_V("FORMULATE", 0.1)
        candidates = [
            ("FORMULATE→SYNTHESIZE", 0.8),   # chain_iql's pick (higher Q)
            ("RECALL→SYNTHESIZE", 0.6),       # β might prefer (RECALL well-grounded)
            ("BREAK→SYNTHESIZE", 0.4),
        ]
        best, _, info = c.rerank_templates(candidates, {"chain_len": 0})
        # Shadow mode: always top-1
        assert best == "FORMULATE→SYNTHESIZE"
        assert info["mode"] == "shadow"


def test_rerank_templates_in_active_can_reorder():
    """In active mode with full ramp, β CAN pick different from chain_iql top."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Ground primitives strongly so composition confidence is high
        for _ in range(200):
            c.update_primitive_V("RECALL", 1.0)
            c.update_primitive_V("SYNTHESIZE", 1.0)
            c.update_primitive_V("FORMULATE", 0.0)
            c.update_primitive_V("EVALUATE", 0.0)
        # Force active mode
        c.force_graduate()
        for _ in range(100):
            c.evaluate_graduation()
        assert c._status == "active"
        candidates = [
            ("FORMULATE→EVALUATE", 0.55),    # chain_iql top but primitives weak
            ("RECALL→SYNTHESIZE", 0.50),     # β loves these primitives
        ]
        best, score, info = c.rerank_templates(candidates, {"chain_len": 0})
        assert info["mode"] == "active"
        assert info["ramp"] == 1.0


def test_rerank_templates_disabled_bypasses():
    """Disabled consumer must bypass rerank, return top-1."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c._status = "disabled_failsafe"
        candidates = [("A→B", 0.7), ("C→D", 0.5)]
        best, score, info = c.rerank_templates(candidates, {})
        assert best == "A→B"
        assert info["mode"] == "bypass"


def test_graduation_readiness_exposes_blockers():
    """get_graduation_readiness shows specific blockers."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        r = c.get_graduation_readiness()
        assert r["ready"] is False
        assert any("primitives_well_sampled" in b for b in r["blockers"])
        assert any("total_updates" in b for b in r["blockers"])
        assert any("confirmed_hypotheses" in b for b in r["blockers"])


def test_shadow_quality_metric_reports_health():
    """shadow_quality_metric classifies disagreement rate."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        assert c.shadow_quality_metric()["health"] == "no_data"
        c._total_compositions = 100
        c._total_disagreements = 15  # 15% → healthy
        assert c.shadow_quality_metric()["health"] == "healthy"
        c._total_disagreements = 2   # 2% → too low
        assert c.shadow_quality_metric()["health"] == "too_low_beta_adds_little"


# ── Phase 5: Failsafe + impasse detection ──────────────────────────

def test_failsafe_trips_on_severity_9():
    """P5 I-5: 3 composition_errors (severity 3 each) → severity_sum=9 → trip."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Feed 3 DISTINCT composition errors (dedup counts unique signatures)
        for i in range(3):
            c._record_failure("composition_error",
                               ValueError(f"unique error {i}"), chain_id=i)
        assert c._status == "disabled_failsafe"
        assert c._cooldown_remaining == 1000
        assert c._failsafe_trip_count == 1


def test_failsafe_dedup_prevents_false_trip():
    """P5 I-6: 100 identical failures should count once — no trip."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Same error 50 times — should dedup to 1 signature = severity 3
        for _ in range(50):
            c._record_failure("composition_error",
                               ValueError("same error"), chain_id=0)
        assert c._status == "shadow_mode"   # not tripped
        assert c._failsafe_trip_count == 0


def test_failsafe_benign_noise_does_not_trip():
    """8 encoding_errors (severity 1 each, 8 unique) → severity=8 → below threshold."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        for i in range(8):
            c._record_failure("encoding_error", ValueError(f"noise {i}"), i)
        assert c._status == "shadow_mode"
        # 9th unique encoding_error would trip (severity reaches 9)
        c._record_failure("encoding_error", ValueError("noise 8"), 8)
        assert c._status == "disabled_failsafe"


def test_failsafe_disabled_short_circuits_operations():
    """Disabled consumer must no-op all public operations."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c._status = "disabled_failsafe"
        pre_n = c._primitives["FORMULATE"].n_samples
        c.update_primitive_V("FORMULATE", 0.9)
        assert c._primitives["FORMULATE"].n_samples == pre_n   # untouched
        pre_obs = len(c._hypotheses["H1_monoculture"].observations)
        c.observe_chain_evidence({
            "primitives": ["FORMULATE"], "quality": 0.5,
            "dominant_primitive": "FORMULATE", "pop_avg_V": 0.5,
            "per_primitive_V": {"FORMULATE": 0.5},
        })
        assert len(c._hypotheses["H1_monoculture"].observations) == pre_obs


def test_reset_watchdog_recovers_from_failsafe():
    """reset_watchdog() must return to shadow from disabled."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Trip failsafe
        for i in range(3):
            c._record_failure("composition_error",
                               ValueError(f"err {i}"), i)
        assert c._status == "disabled_failsafe"
        # Manual reset
        c.reset_watchdog()
        assert c._status == "shadow_mode"
        assert c._cooldown_remaining == 0


def test_watchdog_state_persists_across_reload():
    """P5 I-9: watchdog state (failure count, status) survives reboot."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c1 = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        for i in range(3):
            c1._record_failure("composition_error",
                                ValueError(f"err {i}"), i)
        assert c1._status == "disabled_failsafe"
        # Reload
        c2 = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Watchdog state restored: status still disabled, cooldown preserved
        assert c2._status == "disabled_failsafe"
        assert c2._cooldown_remaining > 0


def test_failure_log_written():
    """P5 I-7: failures append to failure_log.jsonl."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c._record_failure("encoding_error", ValueError("test"), 42)
        path = os.path.join(tmp, "failure_log.jsonl")
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) >= 1
        rec = json.loads(lines[-1])
        assert rec["kind"] == "encoding_error"
        assert rec["chain_id"] == 42
        assert rec["severity"] == 1


def test_impasse_v_flatline_triggers():
    """F8: V values unchanging for 500 chains → v_flatline."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Populate 500 chains of constant V
        for _ in range(500):
            c.check_impasse()
        assert c._impasse_state == "v_flatline"
        assert c._impasse_total_fires == 1
        assert c._impasse_alpha_boost_remaining == 100


def test_impasse_status_healthy_when_V_moves():
    """F8: active V changes AND HAOV evidence prevent all impasse signals."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Populate 500 chains where V is moving AND HAOV observations grow
        for i in range(500):
            c.update_primitive_V("FORMULATE", 0.1 if i % 2 else 0.9)
            # Push HAOV evidence so obs_rate doesn't trigger stagnation
            c.observe_chain_evidence({
                "primitives": ["FORMULATE", "RECALL", "BREAK"],
                "quality": 0.1 if i % 2 else 0.9,
                "dominant_primitive": "FORMULATE",
                "pop_avg_V": 0.5,
                "per_primitive_V": {"FORMULATE": 0.1 if i % 2 else 0.9},
                "monoculture_share": 0.5,
                "is_in_soar_impasse": False,
                "domain": "test",
            })
            c.check_impasse()
        assert c._impasse_state == "healthy"
        assert c._impasse_total_fires == 0


def test_get_failsafe_status_schema():
    """Failsafe status endpoint data shape."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        s = c.get_failsafe_status()
        for key in ("status", "total_failures", "failsafe_trip_count",
                    "cooldown_remaining", "severity_sum_in_window",
                    "severity_trip_threshold"):
            assert key in s


def test_record_chain_outcome_feeds_rollback_detector():
    """record_chain_outcome splits raw reward floats into pre-grad vs post-grad
    deques. Scale-invariant after Option B refactor (2026-04-19)."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Pre-graduation
        for _ in range(5):
            c.record_chain_outcome(0.7)
        assert len(c._pre_grad_rewards) == 5
        assert len(c._post_grad_rewards) == 0
        # Force to active; rewards now feed post-grad
        c.force_graduate()
        for _ in range(100):
            c.evaluate_graduation()
        assert c._status == "active"
        c.record_chain_outcome(0.2)
        assert len(c._post_grad_rewards) == 1


def test_rollback_detector_scale_invariant_no_drop():
    """Scale-invariance regression: if rewards stay at the same scale as
    pre-grad (no actual drop), rollback must NOT fire — even if the absolute
    reward level is well below 0.5. Bug: pre-Option-B the hardcoded 0.5
    threshold would rollback any scale < 0.5."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Populate pre-grad baseline with low-scale rewards (post-COMPLETE-5
        # compound-reward scale: mean ~0.27, tight σ).
        for v in [0.22, 0.25, 0.27, 0.28, 0.30, 0.26, 0.24, 0.29, 0.27, 0.25] * 10:
            c.record_chain_outcome(v)
        # Graduate — captures baseline_mean ~0.263, std ~0.023
        c.force_graduate()
        for _ in range(100):
            c.evaluate_graduation()
        assert c._status == "active"
        baseline = c._pre_graduation_baseline
        assert 0.24 < baseline["reward_mean"] < 0.29
        # Now post-graduation: feed 60 chains at SAME reward scale (no drop)
        # — drives chains_since_graduation past the 50-chain guard so rollback
        # check actually fires.
        for _ in range(60):
            c.record_chain_outcome(0.26)
            c.evaluate_graduation()
        assert c._status == "active", (
            f"Rollback fired on stable-scale rewards (baseline_mean="
            f"{baseline['reward_mean']:.3f}, post-grad=0.26) — "
            f"scale-invariance broken")


def test_rollback_detector_fires_on_real_regression():
    """Positive case: if post-grad rewards really drop below baseline − k·σ,
    rollback DOES fire. Prevents over-loosening the detector."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Baseline ~0.60 with std ~0.05
        for v in [0.55, 0.58, 0.60, 0.62, 0.65, 0.57, 0.61, 0.59, 0.63, 0.56] * 10:
            c.record_chain_outcome(v)
        c.force_graduate()
        for _ in range(100):
            c.evaluate_graduation()
        assert c._status == "active"
        # Now feed 60 chains at severely dropped scale (0.25 — well below
        # baseline_mean − 2σ ≈ 0.60 − 0.10 = 0.50). Drive evaluate_graduation
        # to tick chains_since_graduation past the 50-chain guard.
        for _ in range(60):
            c.record_chain_outcome(0.25)
            c.evaluate_graduation()
        assert c._status == "shadow_mode", (
            "Rollback did NOT fire on real regression — detector too loose")
        assert c._rolled_back_count == 1


# ══════════════════════════════════════════════════════════════════════
# P6 — Bayesian Beta + CI + decay + anti-monoculture + per-domain V
# ══════════════════════════════════════════════════════════════════════

def test_beta_posterior_update_accumulates_alpha_beta():
    """Beta update: α += quality; β += (1 − quality). n_eff grows with samples.

    With γ=0.999 (shipped 2026-04-21, commit 64a92be), each update also decays
    existing excess by 0.1%. At 100 updates the γ-loss is ~5% of accumulated
    n, so n_samples converges below the nominal 100.
    """
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer, BETA_PARAM_FLOOR
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # 100 high-quality updates → α grows ~85, β grows ~10 under γ=0.999
        for _ in range(100):
            c.update_primitive_V("FORMULATE", quality=0.9)
        p = c._primitives["FORMULATE"]
        # α started at BETA_PARAM_FLOOR=1; each step adds ~0.9 (minus γ decay)
        assert p.alpha > BETA_PARAM_FLOOR + 80
        assert p.beta > BETA_PARAM_FLOOR + 5
        # Posterior mean ≈ quality
        assert 0.85 < p.V < 0.95
        # n_samples (derived from α+β - 2*floor). At γ=0.999, 100 updates
        # land in the 90–100 band (γ-decay shaves ~5 off the raw 100).
        assert 90 <= p.n_samples <= 100


def test_beta_ci_narrows_with_more_samples():
    """CI width shrinks as evidence accumulates — confidence becomes 'tight'."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer, _beta_ci_width
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Early CI (wide)
        for _ in range(5):
            c.update_primitive_V("FORMULATE", quality=0.7)
        p = c._primitives["FORMULATE"]
        _, _, w_early = _beta_ci_width(p.alpha, p.beta, 0.05, 0.95)
        # Late CI (narrow)
        for _ in range(500):
            c.update_primitive_V("FORMULATE", quality=0.7)
        _, _, w_late = _beta_ci_width(p.alpha, p.beta, 0.05, 0.95)
        assert w_late < w_early * 0.5, (w_early, w_late)


def test_migration_v2_to_v3_preserves_v_ordering_and_caps_n_eff():
    """Converted bootstrap: α=V·n_eff+1, β=(1−V)·n_eff+1, n_eff≤200.
    Preserves V ordering but caps FORMULATE's prior strength so it stays
    learning-responsive (no α≈0.01 lock-in)."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, MIGRATION_N_EFF_CAP)
    import json, os
    with tempfile.TemporaryDirectory() as tmp:
        # Simulate a v2 file: FORMULATE with extreme n=1187, SPIRIT_SELF n=34
        v2_path = os.path.join(tmp, "primitive_grounding.json")
        v2_data = {
            "version": 2,
            "titan_id": "T1",
            "saved_ts": 1.0,
            "primitives": {
                p: {
                    "primitive_id": p,
                    "V": 0.5, "confidence": 0.1, "n_samples": 10,
                    "variance": 0.1, "last_updated_ts": 0.0,
                    "last_updated_chain": 0, "cross_consumer_signals": {},
                    "haov_rules": []}
                for p in ["FORMULATE", "RECALL", "HYPOTHESIZE", "DELEGATE",
                          "SYNTHESIZE", "EVALUATE", "BREAK", "SPIRIT_SELF",
                          "INTROSPECT"]
            },
        }
        # Overwrite FORMULATE (1187 samples, V=0.235) and SPIRIT_SELF (34, V=0.33)
        v2_data["primitives"]["FORMULATE"].update(
            {"V": 0.235, "n_samples": 1187})
        v2_data["primitives"]["SPIRIT_SELF"].update(
            {"V": 0.328, "n_samples": 34})
        with open(v2_path, "w") as f:
            json.dump(v2_data, f)
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        f = c._primitives["FORMULATE"]
        s = c._primitives["SPIRIT_SELF"]
        # V preserved within ~0.02 (uniform prior pulls slightly toward 0.5
        # when n_eff is small; that's expected Beta posterior behavior).
        assert abs(f.V - 0.235) < 0.02, f.V
        assert abs(s.V - 0.328) < 0.02, s.V
        # n_eff capped at 200 for FORMULATE (was 1187)
        n_eff_F = f.n_samples
        assert n_eff_F <= MIGRATION_N_EFF_CAP + 1, n_eff_F
        # SPIRIT_SELF kept its smaller n
        assert 30 < s.n_samples < 40, s.n_samples


def test_f_novelty_bounded_by_kappa_explore():
    """F anti-monoculture bonus uses tanh, so magnitude is bounded by κ.
    Even a primitive with n=10000 vs pool n_mean=0 can't drag the score more
    than κ_explore below the V_effective + UCB shift."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, COMPOSITION_DEFAULTS)
    kappa = COMPOSITION_DEFAULTS["kappa_explore"]
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Ground only FORMULATE to extreme over-sampling
        for _ in range(1000):
            c.update_primitive_V("FORMULATE", quality=0.8)
        r = c.compose_template_score(
            template_id="FORMULATE",
            state_ctx={},
            template_primitives=["FORMULATE"],
        )
        # F penalty magnitude ≤ kappa_explore (bounded by tanh)
        novelty = r["per_primitive"][0]["novelty"]
        assert abs(novelty) <= kappa + 1e-6, novelty


def test_ucb_composition_flips_at_n_anchor():
    """D4: under-sampled primitives get optimistic shift (+κ·(hi−V)),
    over-sampled get pessimistic shift (−κ·(V−lo)). Flip happens at N_ANCHOR."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, COMPOSITION_DEFAULTS)
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Under-sampled: n < N_ANCHOR → optimistic (positive shift)
        for _ in range(30):
            c.update_primitive_V("RECALL", quality=0.5)
        r = c.compose_template_score(
            template_id="RECALL", state_ctx={},
            template_primitives=["RECALL"])
        ucb_low_n = r["per_primitive"][0]["ucb"]
        assert ucb_low_n > 0, ucb_low_n
        # Over-sampled: n ≥ N_ANCHOR → pessimistic (negative shift)
        for _ in range(200):
            c.update_primitive_V("FORMULATE", quality=0.5)
        r = c.compose_template_score(
            template_id="FORMULATE", state_ctx={},
            template_primitives=["FORMULATE"])
        ucb_high_n = r["per_primitive"][0]["ucb"]
        assert ucb_high_n <= 0, ucb_high_n


def test_chain_decay_reduces_n_but_preserves_V(monkeypatch):
    """D2: Batch decay shrinks (α, β) proportionally → n drops, V unchanged,
    confidence re-opens. Skips primitives under skip_n_min floor.

    2026-04-21 B-phase update: batch decay is now GATED to only fire when
    `ema_decay_gamma == 1.0` (per-update EMA takes over otherwise). This
    test forces γ=1.0 to exercise the legacy batch path that remains the
    active decay mechanism whenever EMA is disabled."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, COMPOSITION_DEFAULTS)
    monkeypatch.setitem(COMPOSITION_DEFAULTS, "ema_decay_gamma", 1.0)
    cadence = COMPOSITION_DEFAULTS["decay_cadence_chains"]
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Accumulate heavy evidence on FORMULATE
        for _ in range(300):
            c.update_primitive_V("FORMULATE", quality=0.7)
        p = c._primitives["FORMULATE"]
        n_before = p.n_samples
        V_before = p.V
        # Trigger cadence via record_chain_outcome
        for _ in range(cadence):
            c.record_chain_outcome(1.0)
        n_after = p.n_samples
        V_after = p.V
        assert n_after < n_before, (n_before, n_after)
        assert abs(V_after - V_before) < 0.02, (V_before, V_after)


def test_chain_decay_skips_low_n_primitives(monkeypatch):
    """D2 skip_n_min: under-sampled primitives must be spared decay.

    2026-04-21 B-phase: forces γ=1.0 to exercise the batch-decay path."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, COMPOSITION_DEFAULTS)
    monkeypatch.setitem(COMPOSITION_DEFAULTS, "ema_decay_gamma", 1.0)
    cadence = COMPOSITION_DEFAULTS["decay_cadence_chains"]
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Ground only lightly — should remain below skip_n_min
        for _ in range(5):
            c.update_primitive_V("RECALL", quality=0.5)
        p = c._primitives["RECALL"]
        n_before = p.n_samples
        for _ in range(cadence):
            c.record_chain_outcome(1.0)
        assert c._primitives["RECALL"].n_samples == n_before


# ══════════════════════════════════════════════════════════════════════
# B-phase γ activation (2026-04-21 afternoon) — per-update EMA decay
# replaces batch decay once ema_decay_gamma < 1.0.
# ══════════════════════════════════════════════════════════════════════

def test_batch_decay_gated_off_when_ema_active(monkeypatch):
    """B-phase gate: `_maybe_apply_decay` must early-return when
    `ema_decay_gamma < 1.0`. Prevents compound decay from batch × EMA."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, COMPOSITION_DEFAULTS, BETA_PARAM_FLOOR)
    # Activate EMA — any value < 1.0
    monkeypatch.setitem(COMPOSITION_DEFAULTS, "ema_decay_gamma", 0.9999)
    cadence = COMPOSITION_DEFAULTS["decay_cadence_chains"]
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Ground heavy, above skip_n_min
        for _ in range(300):
            c.update_primitive_V("FORMULATE", quality=0.7)
        # Snapshot α,β after the last per-update EMA tick (accumulated
        # already via update_primitive_V)
        p = c._primitives["FORMULATE"]
        alpha_before_batch_tick = p.alpha
        beta_before_batch_tick = p.beta
        # Now fire the batch trigger (cadence chain outcomes)
        # record_chain_outcome only updates _chains_since_decay and calls
        # _maybe_apply_decay; no per-update EMA ticks happen here.
        for _ in range(cadence):
            c.record_chain_outcome(1.0)
        # EMA is active → batch decay must have been gated off. α,β
        # unchanged from the cadence's record_chain_outcome calls alone.
        assert p.alpha == alpha_before_batch_tick, \
            f"α changed during batch-gated cadence: "\
            f"{alpha_before_batch_tick} → {p.alpha}"
        assert p.beta == beta_before_batch_tick
        # Gate should also reset the cadence counter
        assert c._chains_since_decay == 0


def test_ema_decay_shrinks_posterior_when_active(monkeypatch):
    """B-phase: per-update EMA with γ<1.0 multiplies (α-FLOOR, β-FLOOR)
    by γ each observation. Demonstrates that the Beta posterior forgets
    stale evidence at a rate determined by γ — the whole point of B-phase.

    Test setup: at γ=0.9 (extreme value for visibility), 50 identical
    observations should yield α+β ≈ weight × 1/(1-γ) ≈ 10 (not the 50
    raw accumulation we'd see at γ=1.0)."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, COMPOSITION_DEFAULTS, BETA_PARAM_FLOOR)
    monkeypatch.setitem(COMPOSITION_DEFAULTS, "ema_decay_gamma", 0.9)
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # At γ=0.9, steady-state (α-FLOOR) + (β-FLOOR) ≈ 1 × 1/0.1 = 10
        # (weight=1 per update, geometric series)
        for _ in range(100):
            c.update_primitive_V("FORMULATE", quality=0.5)
        p = c._primitives["FORMULATE"]
        excess = (p.alpha - BETA_PARAM_FLOOR) + (p.beta - BETA_PARAM_FLOOR)
        # Steady-state is 1/(1-0.9) = 10. Allow ±20% tolerance (series
        # converges but isn't exact after finite samples).
        assert 8 <= excess <= 12, \
            f"EMA steady-state α+β-2*FLOOR expected ~10; got {excess:.2f}"


def test_ema_gamma_one_is_mathematical_identity(monkeypatch):
    """Regression: γ=1.0 must be the exact no-op identity case. At γ=1.0,
    α+β grows linearly with the number of observations (no decay). This
    was the SHIPPED NEUTRAL behavior from B.1 (commit c17ee12)."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, COMPOSITION_DEFAULTS, BETA_PARAM_FLOOR)
    monkeypatch.setitem(COMPOSITION_DEFAULTS, "ema_decay_gamma", 1.0)
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        for _ in range(100):
            c.update_primitive_V("FORMULATE", quality=0.5)
        p = c._primitives["FORMULATE"]
        excess = (p.alpha - BETA_PARAM_FLOOR) + (p.beta - BETA_PARAM_FLOOR)
        # At γ=1.0, 100 updates with weight=1 → excess should be ~100
        assert 99 <= excess <= 101, \
            f"γ=1.0 should be linear: 100 updates → ~100; got {excess:.2f}"


def test_per_domain_V_diverges_from_pooled_when_domain_grounded():
    """I3: when a domain has ≥ threshold observations, composition uses its
    specific V posterior. Verifies the domain routing is effective."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, COMPOSITION_DEFAULTS)
    threshold = COMPOSITION_DEFAULTS["domain_obs_threshold"]
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # BREAK in "reasoning" domain: strongly positive (frees impasse)
        for _ in range(threshold * 3):
            c.update_primitive_V("BREAK", quality=0.9, domain="reasoning")
        # BREAK in "social" domain: strongly negative (bad fit)
        for _ in range(threshold * 3):
            c.update_primitive_V("BREAK", quality=0.1, domain="social")
        # Compose in each domain
        r_reason = c.compose_template_score(
            template_id="BREAK",
            state_ctx={"domain": "reasoning"},
            template_primitives=["BREAK"])
        r_social = c.compose_template_score(
            template_id="BREAK",
            state_ctx={"domain": "social"},
            template_primitives=["BREAK"])
        # Domain-specific V(reasoning) should be >> V(social)
        V_reason = r_reason["per_primitive"][0]["V_eff"]
        V_social = r_social["per_primitive"][0]["V_eff"]
        assert V_reason - V_social > 0.4, (V_reason, V_social)
        assert r_reason["per_primitive"][0]["domain_used"] is True
        assert r_social["per_primitive"][0]["domain_used"] is True


def test_per_domain_fallback_below_threshold():
    """I3: below threshold, composition falls back to pooled V."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, COMPOSITION_DEFAULTS)
    threshold = COMPOSITION_DEFAULTS["domain_obs_threshold"]
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Only 2 domain observations — below threshold
        for _ in range(2):
            c.update_primitive_V("BREAK", quality=0.9, domain="sparse")
        r = c.compose_template_score(
            template_id="BREAK",
            state_ctx={"domain": "sparse"},
            template_primitives=["BREAK"])
        assert r["per_primitive"][0]["domain_used"] is False


def test_impasse_2x_weighting_accelerates_convergence():
    """Impasse α-boost port: during impasse_alpha_boost_remaining, Beta
    updates use 2× observation weight → posterior moves twice as fast."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Normal update: +1 weight each
        for _ in range(10):
            c.update_primitive_V("FORMULATE", quality=1.0)
        normal_alpha = c._primitives["FORMULATE"].alpha
        # Reset
        c._primitives["FORMULATE"].alpha = 1.0
        c._primitives["FORMULATE"].beta = 1.0
        # With impasse boost
        c._impasse_alpha_boost_remaining = 100
        for _ in range(10):
            c.update_primitive_V("FORMULATE", quality=1.0)
        boosted_alpha = c._primitives["FORMULATE"].alpha
        # Boosted should be ~2× delta of normal
        assert (boosted_alpha - 1.0) > (normal_alpha - 1.0) * 1.8


def test_beta_dispersion_ema_moves_when_templates_differ():
    """I1: β-dispersion EMA should rise when top-K candidates produce
    meaningfully different composed_V scores."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Ground primitives with divergent V
        for _ in range(200):
            c.update_primitive_V("FORMULATE", quality=0.9)
        for _ in range(200):
            c.update_primitive_V("RECALL", quality=0.1)
        # Call rerank repeatedly with different top-K candidates
        for _ in range(50):
            c.rerank_templates(
                [("FORMULATE", 0.7), ("RECALL", 0.5)],
                state_ctx={})
        # Dispersion EMA must have risen above 0
        assert c._beta_dispersion_ema > 0.05


def test_usage_gini_signals_monoculture():
    """I2 Gini: 0 when usage uniform, ≈ 1 when one primitive dominates."""
    from titan_plugin.logic.meta_cgn import _gini
    uniform = [100] * 9
    assert _gini(uniform) < 0.05
    monoc = [1000] + [10] * 8
    assert _gini(monoc) > 0.6


def test_schema_v3_round_trip_persists_alpha_beta_and_by_domain():
    """v3 save/load preserves α, β, and per-domain dict."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    import json
    with tempfile.TemporaryDirectory() as tmp:
        c1 = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        for _ in range(50):
            c1.update_primitive_V("FORMULATE", quality=0.8, domain="reasoning")
        c1.save_state()
        # Inspect raw file
        with open(f"{tmp}/primitive_grounding.json") as f:
            data = json.load(f)
        assert data["version"] == 3
        fp = data["primitives"]["FORMULATE"]
        assert "alpha" in fp and "beta" in fp and "by_domain" in fp
        assert "reasoning" in fp["by_domain"]
        # Reload into fresh consumer
        c2 = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        p = c2._primitives["FORMULATE"]
        assert p.alpha > 30 and p.beta > 1
        assert "reasoning" in p.by_domain


# ══════════════════════════════════════════════════════════════════════
# P7 — EUREKA accelerator + advisor disagreement signal
# ══════════════════════════════════════════════════════════════════════

def test_eureka_weight_amplifies_beta_posterior():
    """EUREKA chain with trigger primitive applies 5× weight — α/β deltas
    grow 5× vs baseline chain."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Baseline: 1× weight on FORMULATE with quality=1.0 → α += 1
        a0 = c._primitives["FORMULATE"].alpha
        c.update_primitive_V("FORMULATE", quality=1.0, weight=1.0)
        a_baseline = c._primitives["FORMULATE"].alpha - a0
        # Reset + apply 5× weight
        c._primitives["FORMULATE"].alpha = a0
        c.update_primitive_V("FORMULATE", quality=1.0, weight=5.0)
        a_trigger = c._primitives["FORMULATE"].alpha - a0
        # Trigger delta should be ≈ 5× baseline
        assert a_trigger > a_baseline * 4.5
        # Accelerated update counter should have incremented exactly once
        assert c._eureka_accelerated_updates == 1


def test_non_eureka_chain_does_not_increment_accelerated_counter():
    """Regression: when weight=1, no accelerated updates are counted."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        for _ in range(10):
            c.update_primitive_V("FORMULATE", quality=0.5, weight=1.0)
        assert c._eureka_accelerated_updates == 0


def test_eureka_trigger_counts_tracked_per_primitive():
    """observe_chain_evidence counts per-trigger primitive via chain_info."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        base_chain = {
            "ts": 0.0, "chain_id": 1, "primitives": ["FORMULATE"],
            "quality": 0.9, "domain": "general", "monoculture_share": 0.0,
            "is_in_soar_impasse": False, "dominant_primitive": "FORMULATE",
            "pop_avg_V": 0.5, "per_primitive_V": {},
        }
        # Non-EUREKA
        c.observe_chain_evidence(dict(base_chain, eureka_fired=False))
        assert c._eureka_trigger_counts["FORMULATE"] == 0
        # EUREKA triggered by SYNTHESIZE
        c.observe_chain_evidence(dict(
            base_chain, eureka_fired=True, eureka_trigger="SYNTHESIZE"))
        assert c._eureka_trigger_counts["SYNTHESIZE"] == 1
        # EUREKA triggered by SYNTHESIZE again
        c.observe_chain_evidence(dict(
            base_chain, eureka_fired=True, eureka_trigger="SYNTHESIZE"))
        assert c._eureka_trigger_counts["SYNTHESIZE"] == 2


def test_advisor_conflict_emits_bus_event():
    """Strong disagreement → META_CGN_ADVISOR_CONFLICT on the send_queue."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    from queue import Queue
    q = Queue()
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=q, save_dir=tmp)
        # Ground FORMULATE well so V_confidence > 0.3
        # B.2 (2026-04-21): 200 → 500 for new asymptotic confidence function
        for _ in range(500):
            c.update_primitive_V("FORMULATE", quality=0.9)
        # Drain any queue events from registration
        while not q.empty():
            q.get_nowait()
        # Compose with very low chain_iql score → large disagreement
        c.compose_template_score(
            template_id="FORMULATE",
            state_ctx={"domain": "reasoning", "chain_id": 42},
            chain_iql_score=0.0,       # far from V_effective ≈ 0.9
            chain_iql_confidence=0.5,
            template_primitives=["FORMULATE"],
        )
        # Drain and check for META_CGN_ADVISOR_CONFLICT
        seen_conflict = False
        while not q.empty():
            msg = q.get_nowait()
            if msg.get("type") == "META_CGN_ADVISOR_CONFLICT":
                seen_conflict = True
                assert msg["payload"]["template_id"] == "FORMULATE"
                assert msg["payload"]["domain"] == "reasoning"
                assert abs(msg["payload"]["disagreement"]) > 0.3
        assert seen_conflict
        assert c._conflict_bus_events_emitted == 1


def test_advisor_conflict_throttle_suppresses_duplicate_signature():
    """Two identical-signature conflicts within 100 chains → only one bus emission."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    from queue import Queue
    q = Queue()
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=q, save_dir=tmp)
        # B.2 (2026-04-21): 200 → 500 for new asymptotic confidence function
        for _ in range(500):
            c.update_primitive_V("FORMULATE", quality=0.9)
        while not q.empty():
            q.get_nowait()
        ctx = {"domain": "reasoning", "chain_id": 1}
        c.compose_template_score(
            template_id="FORMULATE", state_ctx=ctx,
            chain_iql_score=0.0, chain_iql_confidence=0.5,
            template_primitives=["FORMULATE"])
        # Second identical conflict immediately — throttle should suppress
        ctx["chain_id"] = 2
        c.compose_template_score(
            template_id="FORMULATE", state_ctx=ctx,
            chain_iql_score=0.0, chain_iql_confidence=0.5,
            template_primitives=["FORMULATE"])
        assert c._conflict_bus_events_emitted == 1, \
            c._conflict_bus_events_emitted
        assert c._conflict_sigs_throttled == 1


def test_advisor_conflict_throttle_clears_after_cooldown():
    """After 100 chain ticks, the same signature can emit again."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    from queue import Queue
    q = Queue()
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=q, save_dir=tmp)
        # B.2 (2026-04-21): 200 → 500 for new asymptotic confidence function
        for _ in range(500):
            c.update_primitive_V("FORMULATE", quality=0.9)
        ctx = {"domain": "reasoning", "chain_id": 1}
        c.compose_template_score(
            template_id="FORMULATE", state_ctx=ctx,
            chain_iql_score=0.0, chain_iql_confidence=0.5,
            template_primitives=["FORMULATE"])
        emits1 = c._conflict_bus_events_emitted
        # Advance chain counter 100 chains (past cooldown)
        for _ in range(100):
            c.record_chain_outcome(1.0)
        ctx["chain_id"] = 2
        c.compose_template_score(
            template_id="FORMULATE", state_ctx=ctx,
            chain_iql_score=0.0, chain_iql_confidence=0.5,
            template_primitives=["FORMULATE"])
        assert c._conflict_bus_events_emitted == emits1 + 1


def test_h6_advisor_disagreement_observation_and_test():
    """H6 accumulates disagreement→quality pairs; confirms when high-disagree
    chains under-perform low-disagree chains by ≥ threshold."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        h = c._hypotheses["H6_advisor_disagreement"]
        # Inject synthetic H6 observations: high disagreement → quality 0.3,
        # low disagreement → quality 0.8. Effect = 0.5, well above 0.1.
        for _ in range(15):
            h.observations.append({
                "ts": 0.0, "chain_id": 0,
                "disagreement": 0.4, "quality": 0.3})
        for _ in range(15):
            h.observations.append({
                "ts": 0.0, "chain_id": 0,
                "disagreement": 0.05, "quality": 0.8})
        effect, confirmed = c._test_h6_advisor_disagreement(h)
        assert effect > 0.3
        assert confirmed is True


# ══════════════════════════════════════════════════════════════════════
# P8 — SOAR-via-CGN full protocol
# ══════════════════════════════════════════════════════════════════════

def _force_impasse(c, signal="v_flatline", diagnostic="test"):
    """Helper — drive MetaCGNConsumer into a given impasse state."""
    c._enter_impasse(signal, diagnostic)


def test_p8_impasse_emits_knowledge_req_with_request_id_and_broadcast():
    """D8.1 + I-P8.2: impasse → CGN_KNOWLEDGE_REQ with dst='all' + request_id."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    from queue import Queue
    q = Queue()
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=q, save_dir=tmp)
        while not q.empty():
            q.get_nowait()
        _force_impasse(c, "v_flatline", "max|ΔV|=0.018 over 500 chains")
        # Expect META_CGN_IMPASSE + CGN_KNOWLEDGE_REQ (broadcast)
        msgs = []
        while not q.empty():
            msgs.append(q.get_nowait())
        req_msgs = [m for m in msgs if m.get("type") == "CGN_KNOWLEDGE_REQ"]
        assert len(req_msgs) == 1
        m = req_msgs[0]
        assert m["dst"] == "all"               # D8.1 broadcast
        assert "request_id" in m["payload"]    # I-P8.2 correlation ID
        assert len(m["payload"]["request_id"]) >= 8
        assert c._knowledge_requests_emitted == 1


def test_p8_request_deduplication_same_signature():
    """I-P8.1: second impasse entry with same signature does NOT re-emit request."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    from queue import Queue
    q = Queue()
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=q, save_dir=tmp)
        _force_impasse(c, "v_flatline", "max|ΔV|=0.018 over 500 chains")
        emitted_1 = c._knowledge_requests_emitted
        # Same signal + same diagnostic → dedup should fire
        _force_impasse(c, "v_flatline", "max|ΔV|=0.018 over 500 chains")
        assert c._knowledge_requests_emitted == emitted_1
        assert c._knowledge_requests_deduped == 1


def test_p8_aggregation_window_finalizes_after_timeout():
    """D8.2: after 2s window closes, pending request is finalized."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    from queue import Queue
    q = Queue()
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=q, save_dir=tmp)
        _force_impasse(c, "v_flatline", "test diagnostic")
        rid = next(iter(c._pending_knowledge_requests))
        # No responses arrive, window not yet expired → no finalize
        winners = c.finalize_expired_requests()
        assert winners == []
        assert rid in c._pending_knowledge_requests
        # Backdate start_ts to force expiration
        c._pending_knowledge_requests[rid]["start_ts"] -= 3.0
        winners = c.finalize_expired_requests()
        # 0 responses → no winner but request is cleaned up
        assert winners == []
        assert rid not in c._pending_knowledge_requests
        assert c._knowledge_requests_empty == 1
        assert c._knowledge_requests_finalized == 1


def test_p8_b_hybrid_ranks_source_affinity_wins_over_confidence():
    """D8.3: B-hybrid uses source affinity as primary. A high-confidence
    response from the wrong source loses to a modestly-confident response
    from the canonical source for the impasse signal."""
    from titan_plugin.logic.meta_cgn import _rank_hybrid
    # v_flatline canonical source = knowledge (affinity 1.0)
    # social has affinity 0.3 — lower
    resp_knowledge = {"source": "knowledge", "confidence": 0.5,
                      "summary": "patterns of reasoning", "domain": "general"}
    resp_social = {"source": "social", "confidence": 0.9,
                   "summary": "friendly chat", "domain": "general"}
    score_k = _rank_hybrid(resp_knowledge, "v_flatline",
                            "max|ΔV|=0.018 patterns", "general")
    score_s = _rank_hybrid(resp_social, "v_flatline",
                            "max|ΔV|=0.018 patterns", "general")
    # Even though social has higher confidence, knowledge wins on signal affinity
    assert score_k > score_s, (score_k, score_s)


def test_p8_response_aggregation_picks_highest_ranked():
    """D8.3: winner of a multi-response aggregation is the highest-ranked."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    from queue import Queue
    q = Queue()
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=q, save_dir=tmp)
        _force_impasse(c, "haov_stagnant", "all nascent")
        rid = next(iter(c._pending_knowledge_requests))
        # self_model is canonical for haov_stagnant (affinity 1.0)
        c.handle_knowledge_response({
            "request_id": rid, "source": "self_model",
            "confidence": 0.6, "summary": "self pattern",
            "topic": "introspection", "domain": "general",
        })
        c.handle_knowledge_response({
            "request_id": rid, "source": "social",
            "confidence": 0.9, "summary": "chat",
            "topic": "banter", "domain": "general",
        })
        # Backdate to force finalize
        c._pending_knowledge_requests[rid]["start_ts"] -= 3.0
        winners = c.finalize_expired_requests()
        assert len(winners) == 1
        assert winners[0]["source"] == "self_model"
        assert c._knowledge_responses_received == 2


def test_p8_meta_cgn_responds_to_external_knowledge_request():
    """D8.4: META-CGN responds with primitive grounding summary when another
    consumer broadcasts CGN_KNOWLEDGE_REQ."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Ground a couple of primitives so META-CGN has something to share
        for _ in range(50):
            c.update_primitive_V("SYNTHESIZE", quality=0.8, domain="reasoning")
        resp = c.handle_knowledge_request({
            "request_id": "ext-1",
            "requestor": "language",
            "query": "which primitives help reasoning?",
            "context": {"domain": "reasoning"},
        })
        assert resp is not None
        assert resp["source"] == "meta_cgn_grounding"
        assert resp["request_id"] == "ext-1"
        assert "SYNTHESIZE" in resp["primitives"]
        assert c._knowledge_responses_sent == 1


def test_p8_meta_cgn_ignores_own_requests():
    """D8.4: META-CGN must not respond to its own broadcast requests."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        resp = c.handle_knowledge_request({
            "request_id": "self-1",
            "requestor": "meta_reasoning",
            "query": "anything",
        })
        assert resp is None


def test_p8_source_credit_tracking():
    """D8.5: provided_by_source increments on response; helpful_by_source on inject."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    from queue import Queue
    q = Queue()
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=q, save_dir=tmp)
        _force_impasse(c, "v_flatline", "diag")
        rid = next(iter(c._pending_knowledge_requests))
        c.handle_knowledge_response({
            "request_id": rid, "source": "knowledge",
            "confidence": 0.7, "summary": "ok",
            "topic": "t", "domain": "general",
        })
        assert c._knowledge_provided_by_source["knowledge"] == 1
        # Simulate injection
        c.mark_helpful("knowledge")
        assert c._knowledge_helpful_by_source["knowledge"] == 1


# ══════════════════════════════════════════════════════════════════════
# P9 — Q-i reward blending (r_legacy + r_compound + r_grounded)
# ══════════════════════════════════════════════════════════════════════

def test_p9_blend_weights_shadow_mode_zero_grounded():
    """Bootstrap/shadow_mode → w_grounded = 0 (reward blending starts clean)."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        w_leg, w_comp, w_grd = c.compute_blend_weights("general")
        assert w_grd == 0.0
        assert 0.99 <= (w_leg + w_comp + w_grd) <= 1.01


def test_p9_blend_weights_active_stage_half_grounded():
    """Active stage → w_grounded ≈ 0.5 (rFP §7 Mature), subject to E3 gate."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Satisfy E3 β-dispersion gate explicitly
        c._beta_dispersion_ema = 0.1
        c._status = "active"
        c._graduation_progress = 100
        w_leg, w_comp, w_grd = c.compute_blend_weights("general")
        assert 0.4 < w_grd < 0.6, w_grd
        assert 0.99 <= (w_leg + w_comp + w_grd) <= 1.01


def test_p9_e3_beta_dispersion_gate_caps_w_grounded():
    """E3: β-dispersion EMA below 0.05 caps w_grounded at ~0.05 regardless
    of graduation stage."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c._status = "active"
        c._beta_dispersion_ema = 0.001  # silent β
        _, _, w_grd = c.compute_blend_weights("general")
        assert w_grd <= 0.06, w_grd


def test_p9_d4_disabled_auto_zero_grounded():
    """D4 safety: disabled states → w_grounded = 0."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c._status = "disabled_failsafe"
        w_leg, w_comp, w_grd = c.compute_blend_weights("general")
        assert w_grd == 0.0


def test_p9_e1_pessimistic_ci_lowers_r_grounded():
    """E1: r_grounded uses pessimistic shift — primitive with wide CI
    contributes less than its posterior mean would suggest."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Modest grounding → wide CI still
        for _ in range(10):
            c.update_primitive_V("FORMULATE", quality=0.9)
        r_narrow_cap, _ = c.compute_grounded_reward(
            ["FORMULATE"], "general", kappa_ci_reward=0.0)
        r_pessimistic, _ = c.compute_grounded_reward(
            ["FORMULATE"], "general", kappa_ci_reward=0.5)
        # Pessimistic shift must reduce r_grounded vs zero-kappa baseline
        assert r_pessimistic < r_narrow_cap, (r_pessimistic, r_narrow_cap)


def test_p9_e4_per_domain_bonus_activates_when_domain_well_grounded():
    """E4: when 2+ primitives have n_domain ≥ threshold, w_grounded gets bonus."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, COMPOSITION_DEFAULTS)
    dom_thresh = COMPOSITION_DEFAULTS["domain_obs_threshold"]
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c._status = "active"
        c._beta_dispersion_ema = 0.1
        for _ in range(dom_thresh * 2):
            c.update_primitive_V("RECALL", quality=0.5, domain="coding")
            c.update_primitive_V("HYPOTHESIZE", quality=0.5, domain="coding")
        _, _, w_grd_domain = c.compute_blend_weights("coding")
        _, _, w_grd_other = c.compute_blend_weights("zzzz_untouched_domain")
        # With bonus + renorm, coding-domain w_grounded should be strictly
        # larger than the same stage in an untouched domain
        assert w_grd_domain > w_grd_other, (w_grd_domain, w_grd_other)


def test_p9_e5_blend_weights_history_written():
    """E5: log_blend_weights appends JSONL rows for audit."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    import json
    import os
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c.log_blend_weights(
            chain_id=42, domain="reasoning",
            w_leg=0.4, w_comp=0.4, w_grd=0.2,
            r_leg=0.7, r_comp=0.6, r_grd=0.5, terminal=0.65)
        path = os.path.join(tmp, "blend_weights_history.jsonl")
        assert os.path.exists(path)
        with open(path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        assert len(rows) == 1
        assert rows[0]["chain_id"] == 42
        assert rows[0]["w_grounded"] == 0.2


# ══════════════════════════════════════════════════════════════════════
# P10 — Cross-consumer signal flow (Layer 1 only; narrative bridge = sep rFP)
# ══════════════════════════════════════════════════════════════════════

def test_p10_known_signal_applies_pseudo_observation():
    """D10.1 + D10.2: a (language, concept_grounded) signal nudges FORMULATE
    and RECALL's Beta posterior by P10_SIGNAL_WEIGHT × intensity."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer, P10_SIGNAL_WEIGHT
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        a0_form = c._primitives["FORMULATE"].alpha
        b0_form = c._primitives["FORMULATE"].beta
        ok = c.handle_cross_consumer_signal(
            consumer="language", event_type="concept_grounded",
            intensity=1.0, domain="language")
        assert ok is True
        # α+β increased by approximately P10_SIGNAL_WEIGHT (intensity=1.0)
        d_form = (c._primitives["FORMULATE"].alpha - a0_form
                  + c._primitives["FORMULATE"].beta - b0_form)
        assert 0.5 * P10_SIGNAL_WEIGHT <= d_form <= 1.5 * P10_SIGNAL_WEIGHT
        assert c._signals_applied == 1


def test_p10_unknown_signal_rejected_cleanly():
    """Unknown (consumer, event_type) increments rejected counter, no crash."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        ok = c.handle_cross_consumer_signal(
            consumer="chaos", event_type="nonsense",
            intensity=1.0)
        assert ok is False
        assert c._signals_rejected_unknown == 1
        assert c._signals_applied == 0


def test_p10_intensity_scales_signal_weight():
    """Intensity × P10_SIGNAL_WEIGHT controls the pseudo-observation magnitude."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        a0 = c._primitives["HYPOTHESIZE"].alpha
        c.handle_cross_consumer_signal(
            consumer="knowledge", event_type="concept_grounded",
            intensity=0.5)
        mid_delta = c._primitives["HYPOTHESIZE"].alpha - a0
        # Reset + double intensity
        c._primitives["HYPOTHESIZE"].alpha = a0
        c._primitives["HYPOTHESIZE"].beta = a0
        c.handle_cross_consumer_signal(
            consumer="knowledge", event_type="concept_grounded",
            intensity=1.0)
        full_delta = c._primitives["HYPOTHESIZE"].alpha - a0
        assert full_delta > mid_delta * 1.5, (full_delta, mid_delta)


def test_p10_narrative_hook_triggers_when_intensity_above_threshold():
    """P10 Layer 2 stub: high-intensity signals with narrative_context record
    a counter (deferred to standalone narrative rFP for real bridging)."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, P10_NARRATIVE_TRIGGER_INTENSITY)
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Low intensity — no hook
        c.handle_cross_consumer_signal(
            consumer="self_model", event_type="reflection_depth",
            intensity=0.3, narrative_context={"subject": "maker-dialogue"})
        assert c._narrative_hooks_deferred == 0
        # High intensity WITH narrative_context — hook triggers
        c.handle_cross_consumer_signal(
            consumer="self_model", event_type="reflection_depth",
            intensity=max(P10_NARRATIVE_TRIGGER_INTENSITY, 0.9),
            narrative_context={"subject": "maker-dialogue"})
        assert c._narrative_hooks_deferred == 1


def test_p10_signal_respects_disabled_failsafe():
    """P10 must be a no-op when META-CGN is disabled (safety)."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c._status = "disabled_failsafe"
        ok = c.handle_cross_consumer_signal(
            consumer="language", event_type="concept_grounded",
            intensity=1.0)
        assert ok is False
        assert c._signals_received == 0  # short-circuited before counter bump


def test_p10_per_consumer_counter_tracked():
    """signals_by_consumer tallies per-source counts for observability."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c.handle_cross_consumer_signal("language", "concept_grounded", 1.0)
        c.handle_cross_consumer_signal("language", "vocab_expanded", 1.0)
        c.handle_cross_consumer_signal("knowledge", "concept_grounded", 1.0)
        assert c._signals_by_consumer == {"language": 2, "knowledge": 1}


# ══════════════════════════════════════════════════════════════════════
# SOAR Phase 3 — primitive_repeat_impasse dynamic mapping (2026-04-21)
# Consolidated tracker §2.3; replaces Phase 2 static {"FORMULATE": 0.5}
# observability-only mapping with payload-driven learning nudge.
# ══════════════════════════════════════════════════════════════════════

def test_soar_phase3_dynamic_mapping_nudges_repeated_primitive():
    """Phase 3: when narrative_context.repeated_primitive is present, the
    signal nudges THAT primitive (not the static FORMULATE fallback)."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Baseline α+β for RECALL and FORMULATE before signal
        a0_recall = c._primitives["RECALL"].alpha
        b0_recall = c._primitives["RECALL"].beta
        a0_formulate = c._primitives["FORMULATE"].alpha
        b0_formulate = c._primitives["FORMULATE"].beta
        # Signal says "RECALL is the one that got stuck"
        ok = c.handle_cross_consumer_signal(
            consumer="meta", event_type="primitive_repeat_impasse",
            intensity=1.0,
            narrative_context={"repeated_primitive": "RECALL",
                               "repeat_count": 5, "chain_id": 42})
        assert ok is True
        # RECALL's posterior moved (either α or β changed)
        recall_delta = (c._primitives["RECALL"].alpha - a0_recall
                        + c._primitives["RECALL"].beta - b0_recall)
        assert recall_delta > 0, \
            f"RECALL should have been nudged; delta={recall_delta}"
        # FORMULATE's posterior did NOT move (Phase 2 static mapping
        # was overridden by Phase 3 dynamic mapping)
        formulate_delta = (c._primitives["FORMULATE"].alpha - a0_formulate
                           + c._primitives["FORMULATE"].beta - b0_formulate)
        assert abs(formulate_delta) < 1e-6, \
            f"FORMULATE should NOT be nudged; delta={formulate_delta}"


def test_soar_phase3_quality_below_neutral_penalizes_repetition():
    """Phase 3: dynamic mapping uses quality=0.35 (below 0.5 neutral),
    shifting the Beta posterior toward penalty (β grows more than α)."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        a0 = c._primitives["HYPOTHESIZE"].alpha
        b0 = c._primitives["HYPOTHESIZE"].beta
        # Fire 20 repeat signals to accumulate enough delta to measure
        for _ in range(20):
            c.handle_cross_consumer_signal(
                consumer="meta", event_type="primitive_repeat_impasse",
                intensity=1.0,
                narrative_context={"repeated_primitive": "HYPOTHESIZE"})
        a1 = c._primitives["HYPOTHESIZE"].alpha
        b1 = c._primitives["HYPOTHESIZE"].beta
        # Quality 0.35 < 0.5 means β grows MORE than α (penalty bias)
        da, db = a1 - a0, b1 - b0
        assert db > da, f"β should grow more than α (penalty); da={da} db={db}"


def test_soar_phase3_fallback_to_static_when_no_payload():
    """Phase 3 preserves backwards compatibility: if narrative_context is
    missing or repeated_primitive absent, the static {"FORMULATE": 0.5}
    fallback from Phase 2 applies (neutral observability nudge)."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        a0 = c._primitives["FORMULATE"].alpha
        b0 = c._primitives["FORMULATE"].beta
        # Signal with NO narrative_context — Phase 2 static mapping applies
        ok = c.handle_cross_consumer_signal(
            consumer="meta", event_type="primitive_repeat_impasse",
            intensity=1.0)
        assert ok is True
        # FORMULATE posterior moved (static fallback applied)
        assert c._primitives["FORMULATE"].alpha != a0 or \
               c._primitives["FORMULATE"].beta != b0


def test_soar_phase3_ignores_invalid_repeated_primitive():
    """Phase 3: if payload has an unrecognized primitive name, fall back
    to static mapping rather than silently succeeding with no effect."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        a0_formulate = c._primitives["FORMULATE"].alpha
        # Garbage primitive name → falls back to static mapping → FORMULATE nudged
        ok = c.handle_cross_consumer_signal(
            consumer="meta", event_type="primitive_repeat_impasse",
            intensity=1.0,
            narrative_context={"repeated_primitive": "GARBAGE_PRIM"})
        assert ok is True
        assert c._primitives["FORMULATE"].alpha != a0_formulate


# ══════════════════════════════════════════════════════════════════════
# META-CGN drift mechanism (§2.6, 2026-04-21)
# Magnitude-capped nudge drift from analyzer hints.
# ══════════════════════════════════════════════════════════════════════

def test_drift_apply_increases_nudge_within_cap():
    """Positive-direction hint with sample count ≥ DRIFT_MIN_SAMPLES
    increases the quality_nudge for that (consumer, event, primitive) tuple,
    capped at DRIFT_MAX_PER_APPLY per call."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, SIGNAL_TO_PRIMITIVE, DRIFT_MAX_PER_APPLY)
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Pick a real tuple that exists in SIGNAL_TO_PRIMITIVE
        tup = ("meta_reasoning", "eureka")
        assert tup in SIGNAL_TO_PRIMITIVE, \
            "test assumes this tuple exists in static mapping"
        baseline = SIGNAL_TO_PRIMITIVE[tup]["SYNTHESIZE"]
        hints = [{
            "consumer": "meta_reasoning", "event": "eureka",
            "prim": "SYNTHESIZE", "N": 500, "dir": "increase",
            "hint": 0.15,  # > cap of 0.05 → gets capped
        }]
        result = c.apply_drift_hints(hints)
        assert result["applied"] == 1
        assert result["skipped_low_n"] == 0
        override = c._drift_overrides[tup]["SYNTHESIZE"]
        # Capped at DRIFT_MAX_PER_APPLY above baseline
        assert abs(override - (baseline + DRIFT_MAX_PER_APPLY)) < 1e-4


def test_drift_apply_decreases_nudge():
    """Direction=decrease reduces the quality_nudge."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer, SIGNAL_TO_PRIMITIVE
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        tup = ("meta_reasoning", "eureka")
        baseline = SIGNAL_TO_PRIMITIVE[tup]["SYNTHESIZE"]
        hints = [{
            "consumer": "meta_reasoning", "event": "eureka",
            "prim": "SYNTHESIZE", "N": 150, "dir": "decrease",
            "hint": 0.02,  # within cap
        }]
        c.apply_drift_hints(hints)
        override = c._drift_overrides[tup]["SYNTHESIZE"]
        assert override < baseline
        assert abs(override - (baseline - 0.02)) < 1e-4


def test_drift_apply_skips_low_sample_hints():
    """Hints with N < DRIFT_MIN_SAMPLES are skipped (noise filter)."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        hints = [{
            "consumer": "meta_reasoning", "event": "eureka",
            "prim": "SYNTHESIZE", "N": 50, "dir": "increase", "hint": 0.03,
        }]
        result = c.apply_drift_hints(hints)
        assert result["applied"] == 0
        assert result["skipped_low_n"] == 1
        assert len(c._drift_overrides) == 0


def test_drift_apply_rejects_unknown_tuples_and_primitives():
    """Hints referencing tuples/primitives absent from SIGNAL_TO_PRIMITIVE
    are rejected cleanly (no crash, skipped counter increments)."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        hints = [
            # Unknown tuple
            {"consumer": "unknown", "event": "nonsense", "prim": "FORMULATE",
             "N": 200, "dir": "increase", "hint": 0.01},
            # Known tuple, unknown primitive for that tuple
            {"consumer": "meta_reasoning", "event": "eureka",
             "prim": "NOT_A_PRIM", "N": 200, "dir": "increase", "hint": 0.01},
            # Invalid direction
            {"consumer": "meta_reasoning", "event": "eureka",
             "prim": "SYNTHESIZE", "N": 200, "dir": "sideways", "hint": 0.01},
        ]
        result = c.apply_drift_hints(hints)
        assert result["applied"] == 0
        assert result["skipped_unknown"] == 3
        assert len(c._drift_overrides) == 0


def test_drift_overrides_consulted_in_signal_handling():
    """After apply_drift_hints, subsequent handle_cross_consumer_signal
    uses the drifted quality_nudge instead of the static one."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, SIGNAL_TO_PRIMITIVE, P10_SIGNAL_WEIGHT)
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Apply drift: push SYNTHESIZE nudge DOWN by max cap
        hints = [{"consumer": "meta_reasoning", "event": "eureka",
                  "prim": "SYNTHESIZE", "N": 200, "dir": "decrease",
                  "hint": 0.05}]
        c.apply_drift_hints(hints)
        drifted_val = c._drift_overrides[("meta_reasoning", "eureka")][
            "SYNTHESIZE"]
        original_val = SIGNAL_TO_PRIMITIVE[
            ("meta_reasoning", "eureka")]["SYNTHESIZE"]
        assert drifted_val < original_val
        # Baseline state
        a0 = c._primitives["SYNTHESIZE"].alpha
        b0 = c._primitives["SYNTHESIZE"].beta
        # Fire signal — drifted quality should show in β/α ratio
        c.handle_cross_consumer_signal(
            consumer="meta_reasoning", event_type="eureka", intensity=1.0)
        da = c._primitives["SYNTHESIZE"].alpha - a0
        db = c._primitives["SYNTHESIZE"].beta - b0
        # drifted = original - 0.05; so da+db should = P10_SIGNAL_WEIGHT
        total = da + db
        expected_total = P10_SIGNAL_WEIGHT
        assert abs(total - expected_total) < 1e-4
        # α portion should equal drifted_val × expected_total
        expected_da = drifted_val * expected_total
        assert abs(da - expected_da) < 1e-4


def test_drift_overrides_persist_across_restart():
    """drift_overrides + applies_total + last_applied_ts survive
    save_state → fresh-consumer → _load_state cycle."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c1 = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c1.apply_drift_hints([{
            "consumer": "language", "event": "concept_grounded",
            "prim": "FORMULATE", "N": 500, "dir": "increase",
            "hint": 0.03}])
        applies_before = c1._drift_applies_total
        override_before = c1._drift_overrides[
            ("language", "concept_grounded")]["FORMULATE"]
        c1.save_state()
        # Fresh consumer — loads from disk
        c2 = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        assert c2._drift_applies_total == applies_before
        assert ("language", "concept_grounded") in c2._drift_overrides
        override_after = c2._drift_overrides[
            ("language", "concept_grounded")]["FORMULATE"]
        assert abs(override_after - override_before) < 1e-4


def test_drift_overrides_bounded_at_dmin_dmax():
    """Overrides clamp to [DRIFT_BOUND_MIN, DRIFT_BOUND_MAX] even under
    many successive applies pushing same direction."""
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, DRIFT_BOUND_MIN, DRIFT_BOUND_MAX)
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Push decrease 50 times at max cap = 2.5 total shift
        # Starting from wherever static is, should clamp at DRIFT_BOUND_MIN
        for _ in range(50):
            c.apply_drift_hints([{
                "consumer": "meta_reasoning", "event": "eureka",
                "prim": "SYNTHESIZE", "N": 200, "dir": "decrease",
                "hint": 0.05}])
        override = c._drift_overrides[("meta_reasoning", "eureka")][
            "SYNTHESIZE"]
        assert override >= DRIFT_BOUND_MIN, \
            f"override {override} below floor {DRIFT_BOUND_MIN}"
        # Now push increase 50 times
        for _ in range(50):
            c.apply_drift_hints([{
                "consumer": "meta_reasoning", "event": "eureka",
                "prim": "SYNTHESIZE", "N": 200, "dir": "increase",
                "hint": 0.05}])
        override = c._drift_overrides[("meta_reasoning", "eureka")][
            "SYNTHESIZE"]
        assert override <= DRIFT_BOUND_MAX, \
            f"override {override} above ceiling {DRIFT_BOUND_MAX}"


def test_drift_get_stats_snapshot_shape():
    """get_drift_stats returns JSON-safe snapshot for /v4/meta-cgn/drift."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c.apply_drift_hints([{
            "consumer": "meta_reasoning", "event": "eureka",
            "prim": "SYNTHESIZE", "N": 200, "dir": "increase",
            "hint": 0.02}])
        snap = c.get_drift_stats()
        assert snap["overrides_count"] == 1
        assert snap["applies_total"] == 1
        assert snap["last_applied_ts"] > 0
        assert "meta_reasoning|eureka" in snap["overrides"]
        import json
        json.dumps(snap)  # roundtrip through json to confirm JSON-safe


# ══════════════════════════════════════════════════════════════════════
# P11 — Kin Protocol v1 + cross-Titan grounding transfer
# ══════════════════════════════════════════════════════════════════════

def test_p11_export_kin_snapshot_has_correct_schema():
    """export_kin_snapshot returns v1 schema with primitives, hypotheses,
    protocol version."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp, titan_id="T1")
        # Give some grounding
        for _ in range(20):
            c.update_primitive_V("FORMULATE", quality=0.7, domain="general")
        snap = c.export_kin_snapshot()
        assert snap["kin_protocol_version"] == 1
        assert snap["schema"] == "meta_cgn_snapshot_v1"
        assert snap["titan_id"] == "T1"
        assert "FORMULATE" in snap["primitives"]
        assert "alpha" in snap["primitives"]["FORMULATE"]
        assert "by_domain" in snap["primitives"]["FORMULATE"]
        assert "H1_monoculture" in snap["hypotheses"]


def test_p11_import_kin_snapshot_adds_priors_at_scaled_strength():
    """Imported (α,β) contributions scaled by confidence_scale, merged into
    local posterior. V moves toward peer's V proportionally."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer, BETA_PARAM_FLOOR
    with tempfile.TemporaryDirectory() as tmp1, \
            tempfile.TemporaryDirectory() as tmp2:
        # T1 builds grounding
        t1 = MetaCGNConsumer(send_queue=None, save_dir=tmp1, titan_id="T1")
        for _ in range(100):
            t1.update_primitive_V("SPIRIT_SELF", quality=0.9)
        snap = t1.export_kin_snapshot()
        # T2 starts fresh, imports T1's priors at 0.5× confidence
        t2 = MetaCGNConsumer(send_queue=None, save_dir=tmp2, titan_id="T2")
        a_before = t2._primitives["SPIRIT_SELF"].alpha
        b_before = t2._primitives["SPIRIT_SELF"].beta
        result = t2.import_kin_snapshot(snap, confidence_scale=0.5)
        assert result["imported"] is True
        assert result["peer_titan"] == "T1"
        assert result["primitives_imported"] == 9
        # SPIRIT_SELF on T2 should now have α roughly:
        # before + 0.5 · (t1.alpha − floor). T1 spent ~90 updates at q=0.9 →
        # t1.alpha ≈ 1 + 90 = 91 → delta ≈ 0.5 × 90 = 45
        a_after = t2._primitives["SPIRIT_SELF"].alpha
        assert a_after > a_before + 30, (a_before, a_after)
        # V_effective should have moved meaningfully toward 0.9
        assert t2._primitives["SPIRIT_SELF"].V > 0.75


def test_p11_import_kin_snapshot_rejects_unsupported_version():
    """Version mismatch aborts import cleanly."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp, titan_id="T2")
        result = c.import_kin_snapshot(
            {"kin_protocol_version": 99, "titan_id": "T1"},
            confidence_scale=0.5)
        assert result["imported"] is False
        assert "99" in result["reason"] or "unsupported" in result["reason"]


def test_p11_import_kin_snapshot_rejects_self_import():
    """Titan must not import its own snapshot (would double-count native evidence)."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp, titan_id="T1")
        snap = c.export_kin_snapshot()
        result = c.import_kin_snapshot(snap, confidence_scale=0.5)
        assert result["imported"] is False
        assert "self" in result["reason"].lower()


def test_p11_import_kin_snapshot_respects_disabled_failsafe():
    """Disabled consumer refuses import (safety)."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp, titan_id="T2")
        c._status = "disabled_failsafe"
        result = c.import_kin_snapshot(
            {"kin_protocol_version": 1, "titan_id": "T1",
             "primitives": {}, "hypotheses": {}},
            confidence_scale=0.5)
        assert result["imported"] is False
        assert "disabled" in result["reason"]


def test_p11_import_kin_snapshot_merges_per_domain_entries():
    """by_domain entries from peer are merged at scaled strength."""
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp1, \
            tempfile.TemporaryDirectory() as tmp2:
        t1 = MetaCGNConsumer(send_queue=None, save_dir=tmp1, titan_id="T1")
        # Grow "reasoning" domain on FORMULATE
        for _ in range(30):
            t1.update_primitive_V("FORMULATE", quality=0.8, domain="reasoning")
        snap = t1.export_kin_snapshot()
        t2 = MetaCGNConsumer(send_queue=None, save_dir=tmp2, titan_id="T2")
        assert "reasoning" not in t2._primitives["FORMULATE"].by_domain
        result = t2.import_kin_snapshot(snap, confidence_scale=0.5)
        assert result["imported"] is True
        assert "reasoning" in t2._primitives["FORMULATE"].by_domain
        # Native domain-n counter stays 0 (priors don't claim to be own evidence)
        assert t2._primitives["FORMULATE"].by_domain["reasoning"][2] == 0


# ══════════════════════════════════════════════════════════════════════
# COMPLETE-9 HAOV signal↔chain correlation telemetry (2026-04-19)
# ══════════════════════════════════════════════════════════════════════

def test_haov_signal_entry_written():
    """handle_cross_consumer_signal writes an entry to the HAOV log."""
    import json as _json
    from titan_plugin.logic.meta_cgn import (
        MetaCGNConsumer, SIGNAL_TO_PRIMITIVE)
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        # Pick any real signal mapping
        (consumer, event_type), _ = next(iter(SIGNAL_TO_PRIMITIVE.items()))
        c.handle_cross_consumer_signal(
            consumer=consumer, event_type=event_type, intensity=0.8)
        log_path = os.path.join(tmp, "haov_signal_outcomes.jsonl")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            lines = [_json.loads(ln) for ln in f if ln.strip()]
        assert len(lines) == 1
        e = lines[0]
        assert e["kind"] == "signal"
        assert e["consumer"] == consumer
        assert e["event_type"] == event_type
        assert e["intensity"] == 0.8
        assert isinstance(e["primitives_nudged"], dict)
        assert "ts" in e


def test_haov_chain_entry_written():
    """log_haov_chain writes a chain-outcome entry."""
    import json as _json
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c.log_haov_chain(
            chain_id=42, primitives=["FORMULATE", "RECALL"],
            dominant="FORMULATE", terminal_reward=0.27, domain="meta")
        log_path = os.path.join(tmp, "haov_signal_outcomes.jsonl")
        with open(log_path) as f:
            lines = [_json.loads(ln) for ln in f if ln.strip()]
        assert len(lines) == 1
        e = lines[0]
        assert e["kind"] == "chain"
        assert e["chain_id"] == 42
        assert e["primitives"] == ["FORMULATE", "RECALL"]
        assert e["dominant"] == "FORMULATE"
        assert e["terminal_reward"] == 0.27


def test_haov_log_rotates_at_cap():
    """Log rotation: when line cap hit, current becomes .archive, new file
    starts fresh. Prevents unbounded disk growth."""
    import json as _json
    from titan_plugin.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp)
        c._haov_log_max_lines = 5  # Force early rotation for test
        for i in range(12):
            c.log_haov_chain(
                chain_id=i, primitives=["FORMULATE"],
                dominant="FORMULATE", terminal_reward=0.2)
        log_path = os.path.join(tmp, "haov_signal_outcomes.jsonl")
        archive_path = log_path + ".archive"
        assert os.path.exists(log_path)
        assert os.path.exists(archive_path), "Archive should have been created"
        # Current file has fewer lines than cap + 1 cycle
        with open(log_path) as f:
            current = sum(1 for _ in f)
        assert current <= c._haov_log_max_lines + 1
