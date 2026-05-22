"""Tests for observatory_worker (RFP_phase_c_titan_hcl_cleanup Phase A+B).

Covers:
  • V4 event-bridge translation — every broadcast type → correct
    OBSERVATORY_EVENT (dst="api") frontend shape (WebSocket contract parity
    with the retired parent _v4_event_bridge_loop).
  • DREAM_INBOX_REPLAY → WS mirror only (observatory_worker does NOT emit
    CHAT_REQUEST — that orchestration moved to agno_worker).
  • Snapshot builder — all reads SHM-direct (no proxy-RPC); correct
    ObservatoryDB record_* calls with values derived from SHM payloads.

Run: python -m pytest tests/test_observatory_worker.py -v -p no:anchorpy
"""
from __future__ import annotations

from unittest.mock import MagicMock

from titan_hcl.bus import (
    BIG_PULSE,
    DREAM_INBOX_REPLAY,
    DREAM_STATE_CHANGED,
    EXPRESSION_FIRED,
    GREAT_PULSE,
    HORMONE_FIRED,
    NEUROMOD_UPDATE,
    OBSERVATORY_EVENT,
    SPHERE_PULSE,
)
from titan_hcl.modules import observatory_worker as ow


class _FakeQueue:
    """Minimal mp.Queue stand-in capturing put_nowait payloads."""

    def __init__(self):
        self.msgs = []

    def put_nowait(self, m):
        self.msgs.append(m)


def _emit_one(msg_type, payload):
    q = _FakeQueue()
    handled = ow._translate_and_emit(q, "observatory", msg_type, payload)
    return handled, q.msgs


# ── Event-bridge translation ────────────────────────────────────────────

def test_sphere_pulse_translation():
    handled, msgs = _emit_one(SPHERE_PULSE, {
        "clock": "alpha", "pulse_count": 7, "radius": 1.2, "phase": 0.3,
    })
    assert handled is True
    assert len(msgs) == 1
    m = msgs[0]
    assert m["type"] == OBSERVATORY_EVENT
    assert m["dst"] == "api"
    assert m["payload"]["event_type"] == "sphere_pulse"
    assert m["payload"]["data"]["clock"] == "alpha"
    assert m["payload"]["data"]["pulse_count"] == 7
    assert m["payload"]["data"]["radius"] == 1.2


def test_big_and_great_pulse_translation():
    _, big = _emit_one(BIG_PULSE, {
        "pair": "BM", "big_pulse_count": 3, "consecutive": 2})
    assert big[0]["payload"]["event_type"] == "big_pulse"
    assert big[0]["payload"]["data"]["big_pulse_count"] == 3

    _, great = _emit_one(GREAT_PULSE, {"pair": "BMS", "great_pulse_count": 1})
    assert great[0]["payload"]["event_type"] == "great_pulse"
    assert great[0]["payload"]["data"]["great_pulse_count"] == 1


def test_dream_state_translation():
    _, msgs = _emit_one(DREAM_STATE_CHANGED, {
        "is_dreaming": True, "state": "dream_start", "recovery_pct": 12.0,
        "remaining_epochs": 30, "wake_transition": False,
    })
    d = msgs[0]["payload"]
    assert d["event_type"] == "dream_state"
    assert d["data"]["is_dreaming"] is True
    assert d["data"]["state"] == "dream_start"


def test_neuromod_hormone_expression_passthrough():
    for mt, ev in ((NEUROMOD_UPDATE, "neuromod_update"),
                   (HORMONE_FIRED, "hormone_fired"),
                   (EXPRESSION_FIRED, "expression_fired")):
        _, msgs = _emit_one(mt, {"foo": "bar"})
        assert msgs[0]["payload"]["event_type"] == ev
        assert msgs[0]["payload"]["data"] == {"foo": "bar"}


def test_dream_inbox_replay_is_ws_mirror_only():
    """observatory_worker emits a dream_inbox_replay WS event with counts ONLY
    — it must NOT re-emit CHAT_REQUEST (that moved to agno_worker)."""
    _, msgs = _emit_one(DREAM_INBOX_REPLAY, {
        "messages": [{"message": "hi"}, {"message": "there"}],
        "dream_duration_s": 42.0,
    })
    assert len(msgs) == 1
    m = msgs[0]
    assert m["type"] == OBSERVATORY_EVENT
    assert m["payload"]["event_type"] == "dream_inbox_replay"
    assert m["payload"]["data"] == {"replayed": 2, "total": 2}
    # No CHAT_REQUEST emitted from the observatory worker.
    assert all(x["type"] == OBSERVATORY_EVENT for x in msgs)


def test_unknown_type_not_handled():
    handled, msgs = _emit_one("SOME_OTHER_TYPE", {})
    assert handled is False
    assert msgs == []


# ── Snapshot builder (SHM-direct) ────────────────────────────────────────

def _make_shm_bank():
    bank = MagicMock()
    bank.compose_trinity.return_value = {
        "body_values": [0.1, 0.2, 0.3, 0.4, 0.5],
        "mind_values": [0.5, 0.5, 0.5, 0.5, 0.5],
        "spirit_tensor": [0.6, 0.6, 0.6, 0.7, 0.6],
        "middle_path_loss": 0.12,
        "body_center_dist": 0.05,
        "mind_center_dist": 0.06,
        "consciousness": {"state_vector": [0.0, 0.1, 0.2, 0.3, 0.44,
                                           0.55, 0.66],
                          "epoch_id": 1611},
        "sphere_clock": {"alpha": 1},
        "unified_spirit": {"x": 1},
        "resonance": {"r": 1},
        "filter_down": {"f": 1},
    }
    bank.read_mind_state.return_value = {
        "mood_label": "calm", "mood_valence": 0.4}
    bank.read_neuromod.return_value = {
        "modulators": {"DA": {"level": 0.7}, "5HT": {"level": 0.3}}}
    bank.read_chi.return_value = {"total": 0.539}
    bank.read_body_state.return_value = {"sol_balance": 12.3}
    bank.read_memory_state.return_value = {"persistent_count": 143230}
    bank.read_life_force_state.return_value = {"state": "HIGH"}
    return bank


def test_snapshot_records_all_four_with_shm_values():
    bank = _make_shm_bank()
    obs_db = MagicMock()

    ow._build_and_record_snapshot(bank, obs_db)

    # Trinity snapshot uses compose_trinity body/mind/spirit tensors.
    obs_db.record_trinity_snapshot.assert_called_once()
    t_kw = obs_db.record_trinity_snapshot.call_args.kwargs
    assert t_kw["body_tensor"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert t_kw["spirit_tensor"] == [0.6, 0.6, 0.6, 0.7, 0.6]
    assert t_kw["middle_path_loss"] == 0.12

    # Growth snapshot derives from consciousness.state_vector.
    g_kw = obs_db.record_growth_snapshot.call_args.kwargs
    assert g_kw["learning_velocity"] == 0.55   # sv[5]
    assert g_kw["social_density"] == 0.66      # sv[6]
    assert g_kw["directive_alignment"] == 0.44  # sv[4]
    assert g_kw["metabolic_health"] == 0.7     # spirit_tensor[3]

    # V4 snapshot recorded because sphere_clock present.
    obs_db.record_v4_snapshot.assert_called_once()

    # Vital snapshot: SHM-direct mood (neuromod DA=0.7 > 0.5 overrides
    # mind_state calm/0.4), chi, sol_balance, persistent_count, energy_state.
    v_kw = obs_db.record_vital_snapshot.call_args.kwargs
    assert v_kw["mood_label"] == "DA"
    assert v_kw["mood_score"] == 0.7
    assert round(v_kw["sovereignty_pct"], 1) == 53.9
    assert v_kw["sol_balance"] == 12.3
    assert v_kw["persistent_count"] == 143230
    assert v_kw["energy_state"] == "HIGH"
    assert v_kw["epoch_counter"] == 1611


def test_snapshot_mood_falls_back_to_mind_state_when_neuromod_low():
    bank = _make_shm_bank()
    bank.read_neuromod.return_value = {
        "modulators": {"DA": {"level": 0.2}, "5HT": {"level": 0.1}}}
    obs_db = MagicMock()

    ow._build_and_record_snapshot(bank, obs_db)

    v_kw = obs_db.record_vital_snapshot.call_args.kwargs
    # No modulator exceeds 0.5 → keep mind_state mood.
    assert v_kw["mood_label"] == "calm"
    assert v_kw["mood_score"] == 0.4


def test_snapshot_skips_v4_when_no_sphere_or_unified():
    bank = _make_shm_bank()
    trinity = bank.compose_trinity.return_value
    trinity.pop("sphere_clock", None)
    trinity.pop("unified_spirit", None)
    obs_db = MagicMock()

    ow._build_and_record_snapshot(bank, obs_db)

    obs_db.record_v4_snapshot.assert_not_called()
    obs_db.record_trinity_snapshot.assert_called_once()
