"""Tests for titan_hcl.modules.outer_interface_worker (Track 2 of
rFP_phase_c_self_improvement_subsystem_migration, SPEC v1.2.1 §9.B
D-SPEC-38).

Bus-independent tests covering chunks A4 + A6 + A7:
- Module identity (MODULE_NAME, subscribe topics list)
- A6 dispatcher routing — each handler is called with correct payload
- A6 handlers — state_refs mutations + OuterInterface method calls
- A7 SPEAK_REQUEST_PENDING → WORD_PERTURBATION_HINT publish chain
- A7 advisor refractory state hash + change detection

Note: subprocess boot + full Guardian stack is covered by the integration
suite at session-close gate. Here we exercise the pure-Python handler
logic with mock OuterInterface + a list-backed send_queue.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from titan_hcl import bus
from titan_hcl.modules import outer_interface_worker as oiw


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_outer_interface():
    """Minimal mock with the methods + attributes the handlers touch."""
    oi = MagicMock()
    # Sub-engines used by handlers + publisher.
    oi.decoder = MagicMock()
    oi.advisor = MagicMock()
    oi.narrator = MagicMock()
    oi.advisor._base_refractory = {"SPEAK": 60.0, "SOCIAL": 30.0, "ART": 90.0}
    oi.advisor._last_action_time = {}
    oi._cooldown_multiplier = 9.0
    # get_stats returns a structured snapshot
    oi.get_stats.return_value = {
        "mode": "SELF_EXPLORE",
        "total_actions_processed": 5,
        "narrator": {"total_words_reinforced": 12},
    }
    # narrator.get_word_perturbation returns a float per word (some words)
    def _pert(w):
        return {"perturbation": 0.42} if w in {"hello", "joy"} else None
    oi.narrator.get_word_perturbation.side_effect = _pert
    # Strip the auto-mock _composition_engine so the hasattr-guarded patch
    # in _handle_reasoning_stats is a no-op (mirroring the canonical case
    # where composition_engine lives in language_worker, not OuterInterface).
    del oi._composition_engine
    return oi


@pytest.fixture
def state_refs(mock_outer_interface):
    return {
        "outer_interface": mock_outer_interface,
        "_last_neuromod_gaba": 0.5,
        "_lang_boosts": {},
        "_lang_bias": {},
    }


@pytest.fixture
def send_queue():
    """List-backed fake send_queue — .put appends; tests assert on contents."""
    class _Q:
        def __init__(self):
            self.items: list = []
        def put(self, msg):
            self.items.append(msg)
    return _Q()


# ── A4 — module identity ────────────────────────────────────────────────────


def test_module_name_matches_spec():
    """SPEC v1.2.1 §9.B outer_interface_worker row — guardian/bus routing key."""
    assert oiw.MODULE_NAME == "outer_interface_worker"


def test_entry_function_present_and_callable():
    """Guardian-spawned L2 worker entry is `outer_interface_worker_main`."""
    assert hasattr(oiw, "outer_interface_worker_main")
    assert callable(oiw.outer_interface_worker_main)


def test_subscribe_topics_canonical_set():
    """SPEC v1.2.1 §9.B outer_interface_worker Bus subscriptions row +
    rFP §2.A.3 handler table. Drift = subscriber misses events.

    D1 (chi placeholder fix) added CHI_UPDATED → 9 → 10 topics.
    """
    topics = oiw._OUTER_INTERFACE_WORKER_SUBSCRIBE_TOPICS
    assert len(topics) >= 10
    # Critical ones — SPEAK gating + lifecycle + chi (D1).
    for t in ("REASONING_STATS_UPDATED", "NEUROMOD_STATS_UPDATED",
              "CHI_UPDATED", "KERNEL_EPOCH_TICK", "EXPRESSION_FIRED",
              "CONVERSATION_STIMULUS", "SPEAK_REQUEST_PENDING",
              "GREAT_KIN_PULSE", "MODULE_SHUTDOWN", "SAVE_NOW"):
        assert t in topics, f"Missing subscribe topic {t}"


def test_d1_chi_updated_handler_caches_state(state_refs, send_queue):
    """D1 — CHI_UPDATED handler caches chi state for _tick_self_exploration
    to consume. Closes the chi={0.5,0.5} placeholder per Prime Directive #1.
    """
    msg = {"type": bus.CHI_UPDATED, "payload": {
        "circulation": 0.42, "total": 0.67,
    }}
    oiw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "outer_interface_worker")
    assert state_refs["_chi_state"]["circulation"] == pytest.approx(0.42)
    assert state_refs["_chi_state"]["total"] == pytest.approx(0.67)


def test_d1_tick_self_exploration_uses_cached_chi(state_refs):
    """_tick_self_exploration must pass the cached chi state to
    outer_interface.tick_self_exploration, NOT the 0.5/0.5 placeholder."""
    state_refs["_chi_state"] = {"circulation": 0.31, "total": 0.89}
    oiw._tick_self_exploration(state_refs["outer_interface"], state_refs)
    call_kwargs = state_refs["outer_interface"].tick_self_exploration.call_args.kwargs
    assert call_kwargs["chi"]["circulation"] == pytest.approx(0.31)
    assert call_kwargs["chi"]["total"] == pytest.approx(0.89)


def test_default_cadences_in_range():
    """Defaults are sane — none zero (would busy-loop) nor too sparse."""
    assert 1.0 <= oiw.PUBLISHER_DEFAULT_S <= 10.0
    assert 10.0 <= oiw.SELF_EXPLORATION_DEFAULT_S <= 120.0
    assert 60.0 <= oiw.SAVE_RECIPES_DEFAULT_S <= 3600.0
    assert oiw.HEARTBEAT_INTERVAL_S == 10.0  # SPEC §10.B MODULE_HEARTBEAT_INTERVAL_S


# ── A6 — dispatcher routing ─────────────────────────────────────────────────


def test_dispatcher_routes_reasoning_stats(state_refs, send_queue):
    msg = {"type": bus.REASONING_STATS_UPDATED, "payload": {
        "word_boost": ["alpha", "beta"],
        "template_bias": "expressive",
    }}
    oiw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "outer_interface_worker")
    assert state_refs["_lang_boosts"] == ["alpha", "beta"]
    assert state_refs["_lang_bias"] == "expressive"


def test_dispatcher_routes_neuromod_stats(state_refs, send_queue):
    msg = {"type": bus.NEUROMOD_STATS_UPDATED, "payload": {
        "neuromods": {"GABA": {"level": 0.31}, "DA": {"level": 0.65}},
    }}
    oiw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "outer_interface_worker")
    assert state_refs["_last_neuromod_gaba"] == pytest.approx(0.31)
    # check_resume called inline with fresh GABA
    state_refs["outer_interface"].check_resume.assert_called_once()


def test_dispatcher_routes_kernel_epoch_tick(state_refs, send_queue):
    msg = {"type": bus.KERNEL_EPOCH_TICK, "payload": {"circadian_phase": 0.42}}
    oiw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "outer_interface_worker")
    assert state_refs["_circadian_phase"] == pytest.approx(0.42)


def test_dispatcher_expression_fired_filters_composite(state_refs, send_queue):
    """Composites not in {SPEAK, SOCIAL, ART, MUSIC, KIN} are silently dropped."""
    # Non-matching composite — no method calls.
    msg_skip = {"type": bus.EXPRESSION_FIRED, "payload": {
        "composite": "BREATH", "action_helper": "breathe"}}
    oiw._dispatch_msg(msg_skip, msg_skip["type"], state_refs, send_queue,
                     "outer_interface_worker")
    state_refs["outer_interface"].on_external_interaction.assert_not_called()

    # Matching composite — on_external_interaction + process_action_result called.
    msg_speak = {"type": bus.EXPRESSION_FIRED, "payload": {
        "composite": "SPEAK", "action_helper": "speak", "outcome": "ok"}}
    oiw._dispatch_msg(msg_speak, msg_speak["type"], state_refs, send_queue,
                     "outer_interface_worker")
    state_refs["outer_interface"].on_external_interaction.assert_called_once()
    state_refs["outer_interface"].process_action_result.assert_called_once()


def test_dispatcher_conversation_stimulus_pre_warms_narrator(state_refs, send_queue):
    """on_external_interaction + narrator.get_word_perturbation called per word."""
    msg = {"type": bus.CONVERSATION_STIMULUS, "payload": {
        "text": "hello joy world",
    }}
    oiw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "outer_interface_worker")
    state_refs["outer_interface"].on_external_interaction.assert_called_once()
    # 3 words — narrator.get_word_perturbation called 3 times
    assert state_refs["outer_interface"].narrator.get_word_perturbation.call_count == 3


def test_dispatcher_great_kin_pulse_caches_state(state_refs, send_queue):
    msg = {"type": bus.GREAT_KIN_PULSE, "payload": {
        "resonance_score": 0.87, "peer": "T2", "ts": 12345.6}}
    oiw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "outer_interface_worker")
    assert state_refs["_last_kin_pulse"]["resonance_score"] == 0.87
    assert state_refs["_last_kin_pulse"]["peer"] == "T2"
    assert state_refs["_last_kin_pulse"]["ts"] == 12345.6


# ── A7 — SPEAK_REQUEST_PENDING → WORD_PERTURBATION_HINT publisher chain ─────


def test_speak_request_pending_emits_word_perturbation_hint(state_refs, send_queue):
    """The load-bearing handler closing the T3 SPEAK quality regression.

    Given cognitive_worker's PENDING with 3 candidate words, outer_interface_worker
    must:
      1. Call narrator.get_word_perturbation(w) per word.
      2. Emit WORD_PERTURBATION_HINT with the same request_id.
      3. Include only words with a numeric perturbation (skip None returns).
    """
    msg = {"type": bus.SPEAK_REQUEST_PENDING, "payload": {
        "request_id": "abc123",
        "candidate_words": ["hello", "joy", "unknown_word"],  # mock returns dict for hello+joy, None for unknown
        "epoch_id": 42,
    }}
    oiw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "outer_interface_worker")
    # 1 hint message published
    hint_msgs = [m for m in send_queue.items
                 if m.get("type") == bus.WORD_PERTURBATION_HINT]
    assert len(hint_msgs) == 1
    hint = hint_msgs[0]
    # Correct request_id
    assert hint["payload"]["request_id"] == "abc123"
    # Only the 2 known words present (unknown_word stripped)
    assert set(hint["payload"]["words"]) == {"hello", "joy"}
    assert hint["payload"]["perturbations"]["hello"] == pytest.approx(0.42)
    assert hint["payload"]["perturbations"]["joy"] == pytest.approx(0.42)
    assert "unknown_word" not in hint["payload"]["perturbations"]


def test_speak_request_pending_with_no_request_id_silently_drops(state_refs, send_queue):
    """No request_id → no WORD_PERTURBATION_HINT (silent drop, not crash)."""
    msg = {"type": bus.SPEAK_REQUEST_PENDING, "payload": {
        "candidate_words": ["hello"],
    }}
    oiw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "outer_interface_worker")
    hint_msgs = [m for m in send_queue.items
                 if m.get("type") == bus.WORD_PERTURBATION_HINT]
    assert hint_msgs == []


def test_speak_request_pending_caps_word_count(state_refs, send_queue):
    """Pathological inputs — 200 candidate words — capped to 64."""
    msg = {"type": bus.SPEAK_REQUEST_PENDING, "payload": {
        "request_id": "x",
        "candidate_words": [f"w_{i}" for i in range(200)],
    }}
    oiw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "outer_interface_worker")
    # narrator.get_word_perturbation called at most 64 times
    assert state_refs["outer_interface"].narrator.get_word_perturbation.call_count <= 64


# ── A7 — periodic publishers ────────────────────────────────────────────────


def test_publish_outer_interface_stats_emits_expected_payload(
        state_refs, send_queue):
    oiw._publish_outer_interface_stats(
        state_refs["outer_interface"], state_refs, send_queue,
        "outer_interface_worker", "T1")
    msgs = [m for m in send_queue.items
            if m.get("type") == bus.OUTER_INTERFACE_STATS_UPDATED]
    assert len(msgs) == 1
    payload = msgs[0]["payload"]
    assert payload["titan_id"] == "T1"
    assert payload["stats"]["mode"] == "SELF_EXPLORE"
    assert payload["stats"]["total_actions_processed"] == 5


def test_advisor_state_hash_stable_when_unchanged(mock_outer_interface):
    """Hash is stable when advisor state doesn't change — no spurious
    ADVISOR_REFRACTORY_STATE re-publish."""
    h1 = oiw._advisor_state_hash(mock_outer_interface)
    h2 = oiw._advisor_state_hash(mock_outer_interface)
    assert h1 == h2


def test_advisor_state_hash_changes_when_last_action_time_updates(
        mock_outer_interface):
    """Hash changes when advisor records an action — triggers publish."""
    h1 = oiw._advisor_state_hash(mock_outer_interface)
    mock_outer_interface.advisor._last_action_time = {"SPEAK": 1234567.0}
    h2 = oiw._advisor_state_hash(mock_outer_interface)
    assert h1 != h2


def test_publish_advisor_refractory_state_schema(state_refs, send_queue):
    """Per SPEC §8.5 schema:
        {action_refractory: {action_type: {next_allowed_ts, base_refractory_s}},
         cooldown_multiplier: float, titan_id, ts}
    """
    oi = state_refs["outer_interface"]
    oi.advisor._last_action_time = {"SPEAK": 1000.0}
    oiw._publish_advisor_refractory_state(
        oi, send_queue, "outer_interface_worker", "T3")
    msgs = [m for m in send_queue.items
            if m.get("type") == bus.ADVISOR_REFRACTORY_STATE]
    assert len(msgs) == 1
    p = msgs[0]["payload"]
    assert p["titan_id"] == "T3"
    assert p["cooldown_multiplier"] == 9.0
    assert "SPEAK" in p["action_refractory"]
    assert "next_allowed_ts" in p["action_refractory"]["SPEAK"]
    assert "base_refractory_s" in p["action_refractory"]["SPEAK"]
    # SPEAK base 60s × cooldown_multiplier 9 = next_allowed_ts = 1000 + 540
    assert p["action_refractory"]["SPEAK"]["next_allowed_ts"] == pytest.approx(1540.0)


# ── Bus + spec audit consistency ────────────────────────────────────────────


def test_new_bus_constants_consistent_with_specs():
    """Every new constant ships in bus.py + bus_specs.py per
    feedback_bus_emit_use_constants.md + the §2.4 canonicalization rules."""
    from titan_hcl.bus_specs import MSG_SPECS
    for const in ("ADVISOR_REFRACTORY_STATE", "WORD_PERTURBATION_HINT",
                  "SPEAK_REQUEST_PENDING", "OUTER_INTERFACE_STATS_UPDATED",
                  "KIN_SIGNATURE_UPDATED", "KIN_SOCIETY_UPDATED"):
        assert hasattr(bus, const), f"bus.{const} missing"
        assert getattr(bus, const) == const, f"bus.{const} value drift"
        assert const in MSG_SPECS, f"bus_specs.MSG_SPECS missing {const}"


def test_bus_specs_audit_clean():
    """audit_against_bus_constants must be clean after Track 2 additions."""
    from titan_hcl.bus_specs import audit_against_bus_constants, all_priorities_in_range
    assert audit_against_bus_constants() == []
    assert all_priorities_in_range() == []


# NOTE: cache_key_registry was RETIRED in Phase D D-SPEC-80 (the entire
# bus-cache → CachedState pipeline was deleted). Track 2 cache keys are
# preserved as bus event constants; SHM-direct reads replace the cache
# fallback per Preamble G18.
