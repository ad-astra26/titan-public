"""INV-LOOP-7 — verified-causal prior on the consumer learned policy.

RFP_cgn_loop_closure §7.D (the corrected apply mechanic). Tests the four pure
units that close the symbolic learning→behaviour loop without live SHM:
  1. cgn-side serialize of action-bearing verified rules → `extra` blob
  2. consumer-side parse of `extra` → self._verified_action_rules
  3. the bounded prior (nudges, never overrides; correct attribution)
  4. the throttled CGN_HAOV_RULE_APPLIED emit
"""
import time
from types import SimpleNamespace

import numpy as np
import msgpack
import pytest

from titan_hcl.logic.cgn_consumer_client import (
    CGNConsumerClient, _HAOV_PRIOR_CAP, _HAOV_EMIT_MIN_INTERVAL_S,
)
from titan_hcl.modules.cgn_worker import _serialize_verified_action_rules


def _client(name="coding", send_queue=None):
    """Build a client WITHOUT init side effects — we test pure methods only."""
    c = object.__new__(CGNConsumerClient)
    c._name = name
    c._module_name = name
    c._send_queue = send_queue
    c._verified_action_rules = []
    c._last_haov_emit_ts = 0.0
    return c


def _hyp(action, conf, mag, rule):
    return SimpleNamespace(action_context={"action": action}, confidence=conf,
                           predicted_magnitude=mag, rule=rule)


# ── 1. cgn-side serialize ──────────────────────────────────────────────────

def test_serialize_only_action_bearing_high_conf_rules():
    tracker = SimpleNamespace(_verified_rules=[
        _hyp(3, 0.71, 0.5, "coding_action_3_causes_strong_positive"),   # keep
        _hyp(1, 0.40, 0.9, "coding_action_1_low_conf"),                  # drop: conf<=0.5
        SimpleNamespace(action_context={"impasse_type": "plateau"},      # drop: no action
                        confidence=0.8, predicted_magnitude=0.5,
                        rule="coding_impasse_plateau"),
    ])
    cgn = SimpleNamespace(_haov_trackers={"coding": tracker})
    blob = _serialize_verified_action_rules(cgn)
    decoded = msgpack.unpackb(blob, raw=False)
    assert list(decoded.keys()) == ["coding"]
    assert len(decoded["coding"]) == 1
    assert decoded["coding"][0][0] == 3 and decoded["coding"][0][3].endswith("strong_positive")


def test_serialize_empty_when_nothing_qualifies():
    tracker = SimpleNamespace(_verified_rules=[
        SimpleNamespace(action_context={"impasse_type": "stuck"},
                        confidence=0.9, predicted_magnitude=0.5, rule="x_impasse"),
    ])
    cgn = SimpleNamespace(_haov_trackers={"x": tracker})
    assert _serialize_verified_action_rules(cgn) == b""


def test_serialize_top_k_by_conf_times_magnitude():
    rules = [_hyp(i, 0.6 + i * 0.01, 0.5, f"r{i}") for i in range(12)]
    cgn = SimpleNamespace(_haov_trackers={"coding":
                          SimpleNamespace(_verified_rules=rules)})
    decoded = msgpack.unpackb(_serialize_verified_action_rules(cgn), raw=False)
    assert len(decoded["coding"]) == 8           # capped
    # highest conf*mag first
    assert decoded["coding"][0][0] == 11


# ── 2. consumer-side parse ─────────────────────────────────────────────────

def test_parse_round_trip_keeps_own_slice():
    c = _client("knowledge")
    blob = msgpack.packb({"knowledge": [[2, 0.7, 0.5, "knowledge_action_2_x"]],
                          "coding": [[1, 0.9, 0.5, "coding_action_1_y"]]},
                         use_bin_type=True)
    c._parse_verified_action_rules(blob)
    assert c._verified_action_rules == [(2, 0.7, 0.5, "knowledge_action_2_x")]


def test_parse_empty_and_garbage_degrade_to_no_prior():
    c = _client()
    c._verified_action_rules = [(1, 0.9, 0.5, "stale")]
    c._parse_verified_action_rules(b"")
    assert c._verified_action_rules == []
    c._parse_verified_action_rules(b"\xff\xff not-msgpack")
    assert c._verified_action_rules == []


# ── 3. the bounded prior ───────────────────────────────────────────────────

def test_no_rules_is_plain_argmax():
    c = _client()
    probs = np.array([0.1, 0.6, 0.3])
    idx, endorsing = c._apply_verified_prior(probs)
    assert idx == 1 and endorsing is None


def test_prior_flips_a_near_tie_and_attributes():
    c = _client()
    c._verified_action_rules = [(2, 0.8, 0.5, "rule_a2")]   # boost ~0.16
    probs = np.array([0.1, 0.45, 0.45])                     # 1 and 2 tied
    idx, endorsing = c._apply_verified_prior(probs)
    assert idx == 2
    assert endorsing == ("rule_a2", 0.8)


def test_prior_cannot_override_a_dominant_action():
    c = _client()
    c._verified_action_rules = [(0, 1.0, 1.0, "rule_a0")]   # max boost = cap (0.2)
    probs = np.array([0.1, 0.7, 0.2])                       # action 1 dominates by 0.6 >> cap
    idx, endorsing = c._apply_verified_prior(probs)
    assert idx == 1                                          # policy still wins
    assert endorsing is None                                 # chosen action not rule-endorsed


def test_boost_is_bounded_by_cap():
    c = _client()
    c._verified_action_rules = [(0, 2.0, 9.0, "rule_huge")]  # conf/mag absurd
    probs = np.array([0.40, 0.55, 0.05])                     # gap 0.15 < cap 0.2
    idx, endorsing = c._apply_verified_prior(probs)
    # boost capped at _HAOV_PRIOR_CAP (0.2) → 0.40+0.2=0.60 > 0.55 → flips
    assert idx == 0 and endorsing[0] == "rule_huge"
    # but a gap larger than the cap would NOT flip:
    probs2 = np.array([0.40, 0.65, 0.05])                    # gap 0.25 > cap
    idx2, _ = c._apply_verified_prior(probs2)
    assert idx2 == 1
    assert _HAOV_PRIOR_CAP == 0.2


# ── 4. the throttled emit ──────────────────────────────────────────────────

class _Q:
    def __init__(self):
        self.msgs = []
    def put_nowait(self, m):
        self.msgs.append(m)


def test_emit_payload_shape_and_throttle():
    q = _Q()
    c = _client("emotional", send_queue=q)
    c._emit_haov_applied("emotional_action_5_x")
    assert len(q.msgs) == 1
    m = q.msgs[0]
    assert m["type"] == "CGN_HAOV_RULE_APPLIED" and m["dst"] == "cgn"
    assert m["payload"]["source_consumer"] == "emotional"
    assert m["payload"]["rule"] == "emotional_action_5_x"
    assert m["payload"]["count"] == 1
    # immediate second emit is throttled
    c._emit_haov_applied("emotional_action_5_x")
    assert len(q.msgs) == 1
    # past the throttle window it emits again
    c._last_haov_emit_ts = time.time() - (_HAOV_EMIT_MIN_INTERVAL_S + 0.01)
    c._emit_haov_applied("emotional_action_5_x")
    assert len(q.msgs) == 2


def test_emit_noop_without_queue():
    c = _client(send_queue=None)
    c._emit_haov_applied("r")  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
