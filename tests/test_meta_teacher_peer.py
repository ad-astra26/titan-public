"""Tests for titan_plugin.logic.meta_teacher_peer — Phase D.1 + D.2 of
rFP_meta_teacher_v2_content_awareness_memory.md.

Covers:
  - validate_outbound_envelope — whitelist, size cap, PII guard, schema
  - validate_inbound_response — whitelist match, rid round-trip, free-text
    cap, size cap
  - build_response_for_query — quality_trajectory / adoption_rate /
    still_needs_push_similar_topics / voice_summary against on-disk state
  - PeerQueryPolicy.record_peer_query / consume_chain_marker / record_outcome
  - PeerQueryPolicy.should_peer_query gate (off by default; on when
    reward_learning_enabled)
  - PeerExchangeClient rate limit, per-topic cooldown, log append + prune
  - PeerExchangeClient.handle_inbound_query end-to-end
"""
from __future__ import annotations

import json
import os
import time

import pytest

from titan_plugin.logic.meta_teacher_peer import (
    PeerExchangeClient,
    PeerQueryPolicy,
    ALLOWED_QUESTION_TYPES,
    OUTBOUND_ENVELOPE_MAX_BYTES,
    INBOUND_RESPONSE_MAX_BYTES,
    RESPONSE_FREE_TEXT_MAX_CHARS,
    validate_outbound_envelope,
    validate_inbound_response,
    build_response_for_query,
)


def _client_cfg(**over):
    base = {
        "peer_exchange_enabled": True,
        "peer_query_rate_limit_per_hour": 5,
        "peer_query_topic_cooldown_seconds": 86400.0,
        "peer_query_min_still_needs_push_count": 3,
        "peer_query_http_timeout_seconds": 5.0,
        "peer_query_log_retention_days": 30,
        # Policy
        "peer_query_feature_logging": True,
        "peer_query_reward_learning_enabled": False,
        "peer_query_reward_adopted": 0.05,
        "peer_query_reward_unadopted": -0.03,
        "peer_query_ema_alpha": 0.2,
        "peer_query_min_ema_to_query": 0.3,
    }
    base.update(over)
    return base


@pytest.fixture
def tmp_data_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def policy(tmp_data_dir):
    p = PeerQueryPolicy(_client_cfg(), data_dir=tmp_data_dir)
    p.load()
    return p


@pytest.fixture
def peers():
    return {"t1": "http://10.0.0.1:7777", "t2": "http://10.0.0.2:7777"}


@pytest.fixture
def client(tmp_data_dir, peers):
    c = PeerExchangeClient(
        _client_cfg(), data_dir=tmp_data_dir,
        my_titan_id="t3", peer_endpoints=peers)
    c.load()
    return c


# ── validate_outbound_envelope ─────────────────────────────────────────────

class TestValidateOutbound:
    def _env(self, **over):
        base = {
            "src_titan": "t1", "target_titan": "t2",
            "question_type": "still_needs_push_similar_topics",
            "topic_key": "AI development|person=@abc",
            "rid": "rid-1", "ts": time.time(),
        }
        base.update(over)
        return base

    def test_well_formed_passes(self):
        ok, reason = validate_outbound_envelope(self._env())
        assert ok is True, reason

    def test_unknown_question_type_rejects(self):
        ok, reason = validate_outbound_envelope(
            self._env(question_type="malicious_payload"))
        assert ok is False
        assert "whitelist" in reason

    def test_missing_field_rejects(self):
        ok, reason = validate_outbound_envelope(self._env(rid=""))
        assert ok is False
        assert "missing" in reason

    def test_email_in_topic_rejects(self):
        ok, reason = validate_outbound_envelope(
            self._env(topic_key="user@evil.com"))
        assert ok is False
        assert "email" in reason

    def test_handle_in_topic_passes(self):
        ok, reason = validate_outbound_envelope(
            self._env(topic_key="topic|person=@jkacrpto"))
        assert ok is True, reason

    def test_size_cap_rejects(self):
        env = self._env(topic_key="x" * (OUTBOUND_ENVELOPE_MAX_BYTES + 100))
        ok, reason = validate_outbound_envelope(env)
        assert ok is False
        assert "size" in reason

    def test_dict_required(self):
        ok, reason = validate_outbound_envelope("not a dict")
        assert ok is False


# ── validate_inbound_response ─────────────────────────────────────────────

class TestValidateInbound:
    def _resp(self, **over):
        base = {
            "answering_titan": "t2",
            "question_type": "quality_trajectory",
            "rid": "rid-1", "ts": time.time(),
            "data": {"observed": False},
        }
        base.update(over)
        return base

    def test_well_formed_passes(self):
        ok, reason = validate_inbound_response(self._resp())
        assert ok is True, reason

    def test_rid_mismatch_rejects(self):
        ok, reason = validate_inbound_response(
            self._resp(rid="rid-2"), expected_rid="rid-1")
        assert ok is False
        assert "rid mismatch" in reason

    def test_question_type_mismatch_rejects(self):
        ok, reason = validate_inbound_response(
            self._resp(),
            expected_question_type="adoption_rate")
        assert ok is False
        assert "match expected" in reason

    def test_unsolicited_free_text_rejects(self):
        # Long string in non-note field
        long_str = "x" * (RESPONSE_FREE_TEXT_MAX_CHARS + 50)
        ok, reason = validate_inbound_response(
            self._resp(data={"observed": False, "extra_text": long_str}))
        assert ok is False
        assert "longer than" in reason

    def test_note_field_capped(self):
        long_note = "x" * (RESPONSE_FREE_TEXT_MAX_CHARS + 50)
        ok, reason = validate_inbound_response(
            self._resp(data={"observed": False, "note": long_note}))
        assert ok is False

    def test_size_cap_rejects(self):
        # data contains a large list of small strings
        data = {"matches": [{"k": "x" * 50} for _ in range(1000)]}
        ok, reason = validate_inbound_response(self._resp(data=data))
        assert ok is False
        assert "size" in reason

    def test_data_must_be_dict(self):
        env = {
            "answering_titan": "t2",
            "question_type": "quality_trajectory",
            "rid": "rid-1", "ts": time.time(),
            "data": "not a dict",
        }
        ok, reason = validate_inbound_response(env)
        assert ok is False
        assert "data must be a dict" in reason


# ── build_response_for_query ──────────────────────────────────────────────

class TestBuildResponse:
    def _seed_journal(self, tmp_data_dir):
        d = os.path.join(tmp_data_dir, "meta_teacher")
        os.makedirs(d, exist_ok=True)
        rows = [
            {
                "topic_key": "AI development|person=@abc",
                "first_seen": 1000.0, "last_seen": 2000.0,
                "critique_count": 4,
                "adoption_trajectory": [
                    {"ts": 1100.0, "adopted_bool": True,
                     "suggested_list": ["INTROSPECT"]},
                    {"ts": 1200.0, "adopted_bool": False,
                     "suggested_list": ["RECALL"]},
                    {"ts": 1300.0, "adopted_bool": True,
                     "suggested_list": ["RECALL"]},
                ],
                "quality_trajectory": [
                    {"ts": 1100.0, "chain_quality": 0.6},
                    {"ts": 1200.0, "chain_quality": 0.5},
                    {"ts": 1300.0, "chain_quality": 0.4},
                    {"ts": 1400.0, "chain_quality": 0.3},
                ],
                "quality_delta": -0.25,
                "still_needs_push": True,
                "summary_cache": "shallow chain",
            },
            {
                "topic_key": "AI development|person=@xyz",
                "first_seen": 1100.0, "last_seen": 2100.0,
                "critique_count": 5,
                "adoption_trajectory": [],
                "quality_trajectory": [
                    {"ts": 1500.0, "chain_quality": 0.55}],
                "quality_delta": 0.0,
                "still_needs_push": True,
                "summary_cache": "",
            },
        ]
        with open(os.path.join(d, "teaching_journal.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return d

    def test_quality_trajectory(self, tmp_data_dir):
        self._seed_journal(tmp_data_dir)
        ok, data, reason = build_response_for_query(
            "quality_trajectory", "AI development|person=@abc",
            data_dir=tmp_data_dir)
        assert ok is True, reason
        assert data["observed"] is True
        assert data["critique_count"] == 4
        assert data["still_needs_push"] is True
        assert len(data["trajectory_tail"]) <= 20

    def test_quality_trajectory_unobserved(self, tmp_data_dir):
        self._seed_journal(tmp_data_dir)
        ok, data, reason = build_response_for_query(
            "quality_trajectory", "unknown_topic", data_dir=tmp_data_dir)
        assert ok is True, reason
        assert data["observed"] is False

    def test_adoption_rate(self, tmp_data_dir):
        self._seed_journal(tmp_data_dir)
        ok, data, reason = build_response_for_query(
            "adoption_rate", "AI development|person=@abc",
            data_dir=tmp_data_dir)
        assert ok is True, reason
        # 2 adopted out of 3 total
        assert data["n"] == 3
        assert data["adopted_n"] == 2
        assert abs(data["adoption_rate"] - 0.667) < 0.01

    def test_still_needs_push_similar_topics(self, tmp_data_dir):
        self._seed_journal(tmp_data_dir)
        ok, data, reason = build_response_for_query(
            "still_needs_push_similar_topics",
            "AI development|person=@unknown",
            data_dir=tmp_data_dir)
        assert ok is True, reason
        # Both seeded topics share "ai development" prefix
        assert data["match_count"] == 2

    def test_voice_summary_default(self, tmp_data_dir):
        ok, data, reason = build_response_for_query(
            "voice_summary", "any", data_dir=tmp_data_dir)
        assert ok is True, reason
        assert data["applied_count"] == 0
        assert data["domain_biases"] == {}

    def test_voice_summary_seeded(self, tmp_data_dir):
        d = os.path.join(tmp_data_dir, "meta_teacher")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "voice_state.json"), "w") as f:
            json.dump({
                "applied_count": 5,
                "domain_biases": {"social": {"INTROSPECT": 0.2}},
                "topic_suppressions": [],
            }, f)
        ok, data, reason = build_response_for_query(
            "voice_summary", "any", data_dir=tmp_data_dir)
        assert ok is True
        assert data["applied_count"] == 5
        assert data["domain_biases"] == {"social": {"INTROSPECT": 0.2}}

    def test_unknown_question_rejects(self, tmp_data_dir):
        ok, data, reason = build_response_for_query(
            "telepathy", "x", data_dir=tmp_data_dir)
        assert ok is False
        assert "unknown" in reason


# ── PeerQueryPolicy ───────────────────────────────────────────────────────

class TestPolicy:
    def test_default_neutral_ema(self, policy):
        ok, reason, ema = policy.should_peer_query("social")
        assert ok is True   # learning off → always allow
        assert ema == policy.DEFAULT_EMA_NEUTRAL
        # gate counter incremented
        assert policy.snapshot()["gate_allow"] >= 1

    def test_record_peer_query_marks(self, policy):
        ok = policy.record_peer_query("social", chain_id=42)
        assert ok is True
        assert policy.peek_chain_marker(42) is True
        # Consume removes
        assert policy.consume_chain_marker(42) is True
        assert policy.peek_chain_marker(42) is False

    def test_record_outcome_adopted(self, policy):
        policy.record_peer_query("social", chain_id=10)
        applied, ema = policy.record_outcome(
            chain_id=10, domain="social", adopted=True, quality_delta=0.1)
        assert applied is True
        assert ema > policy.DEFAULT_EMA_NEUTRAL  # nudged toward 1
        # Marker consumed defensively
        assert policy.peek_chain_marker(10) is False

    def test_record_outcome_unadopted_negative_delta(self, policy):
        policy.record_peer_query("social", chain_id=11)
        applied, ema = policy.record_outcome(
            chain_id=11, domain="social", adopted=False, quality_delta=-0.05)
        assert applied is True
        assert ema < policy.DEFAULT_EMA_NEUTRAL

    def test_record_outcome_unadopted_positive_skips(self, policy):
        policy.record_peer_query("social", chain_id=12)
        applied, ema = policy.record_outcome(
            chain_id=12, domain="social", adopted=False, quality_delta=+0.05)
        assert applied is False  # neither credit nor debit
        assert ema == policy.DEFAULT_EMA_NEUTRAL

    def test_should_peer_query_with_learning_on(self, tmp_data_dir):
        p = PeerQueryPolicy(
            _client_cfg(peer_query_reward_learning_enabled=True),
            data_dir=tmp_data_dir)
        p.load()
        # Default 0.5 ≥ min 0.3 → allow
        ok, reason, ema = p.should_peer_query("social")
        assert ok is True
        # Drag EMA below threshold via repeated unadopted outcomes
        for cid in range(20):
            p.record_peer_query("social", cid)
            p.record_outcome(cid, "social", adopted=False,
                              quality_delta=-0.05)
        ok, reason, ema = p.should_peer_query("social")
        assert ema < 0.3
        assert ok is False
        assert "ema" in reason

    def test_feature_logging_off_disables_record(self, tmp_data_dir):
        p = PeerQueryPolicy(
            _client_cfg(peer_query_feature_logging=False),
            data_dir=tmp_data_dir)
        p.load()
        ok = p.record_peer_query("x", 99)
        assert ok is False
        applied, _ = p.record_outcome(99, "x", True, 0.1)
        assert applied is False

    def test_persistence_round_trip(self, tmp_data_dir):
        p1 = PeerQueryPolicy(_client_cfg(), data_dir=tmp_data_dir)
        p1.load()
        p1.record_peer_query("k", 5)
        p1.record_outcome(5, "k", True, 0.1)
        # Reload
        p2 = PeerQueryPolicy(_client_cfg(), data_dir=tmp_data_dir)
        p2.load()
        snap = p2.snapshot()
        assert "k" in snap["domain_ema"]
        assert snap["outcomes_applied"] >= 1


# ── PeerExchangeClient ────────────────────────────────────────────────────

class TestClient:
    def test_self_filtered_from_peers(self, client):
        snap = client.snapshot()
        # client is t3, peer dict had t1 + t2
        assert "t3" not in snap["peer_endpoints"]
        assert set(snap["peer_endpoints"].keys()) == {"t1", "t2"}

    def test_disabled_no_op(self, tmp_data_dir, peers):
        c = PeerExchangeClient(
            _client_cfg(peer_exchange_enabled=False),
            data_dir=tmp_data_dir, my_titan_id="t1",
            peer_endpoints=peers)
        c.load()
        assert c.enabled is False

    def test_format_no_observation(self, client):
        assert client.format_recent_observation_for_topic("nope") is None

    def test_handle_inbound_query_well_formed(self, client, tmp_data_dir):
        # Seed a journal entry
        d = os.path.join(tmp_data_dir, "meta_teacher")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "teaching_journal.jsonl"), "w") as f:
            f.write(json.dumps({
                "topic_key": "x",
                "critique_count": 5,
                "adoption_trajectory": [],
                "quality_trajectory": [{"ts": 0, "chain_quality": 0.5}],
                "quality_delta": -0.1, "still_needs_push": True,
                "first_seen": 0, "last_seen": 0,
            }) + "\n")
        envelope = {
            "src_titan": "t1", "target_titan": "t3",
            "question_type": "quality_trajectory",
            "topic_key": "x", "rid": "rr-1",
        }
        ok, response, reason = client.handle_inbound_query(
            envelope, data_dir=tmp_data_dir)
        assert ok is True, reason
        assert response["answering_titan"] == "t3"
        assert response["data"]["observed"] is True
        assert response["data"]["critique_count"] == 5

    def test_handle_inbound_target_mismatch(self, client, tmp_data_dir):
        envelope = {
            "src_titan": "t1", "target_titan": "t99",
            "question_type": "quality_trajectory",
            "topic_key": "x", "rid": "rr-2",
        }
        ok, response, reason = client.handle_inbound_query(
            envelope, data_dir=tmp_data_dir)
        assert ok is False
        assert "target_titan" in reason

    def test_handle_inbound_bad_question_type(self, client, tmp_data_dir):
        envelope = {
            "src_titan": "t1", "target_titan": "t3",
            "question_type": "evil_query",
            "topic_key": "x", "rid": "rr-3",
        }
        ok, response, reason = client.handle_inbound_query(
            envelope, data_dir=tmp_data_dir)
        assert ok is False
        assert "whitelist" in reason

    def test_log_appends_on_inbound(self, client, tmp_data_dir):
        envelope = {
            "src_titan": "t2", "target_titan": "t3",
            "question_type": "voice_summary",
            "topic_key": "any", "rid": "rr-4",
        }
        client.handle_inbound_query(envelope, data_dir=tmp_data_dir)
        log_path = os.path.join(
            tmp_data_dir, "meta_teacher", "peer_query_log.jsonl")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert any(r.get("kind") == "inbound_handled" for r in rows)

    def test_prune_old_logs(self, client, tmp_data_dir):
        log_path = os.path.join(
            tmp_data_dir, "meta_teacher", "peer_query_log.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        old_ts = time.time() - (40 * 86400.0)
        recent_ts = time.time() - 60.0
        with open(log_path, "w") as f:
            f.write(json.dumps({"ts": old_ts, "rid": "old"}) + "\n")
            f.write(json.dumps({"ts": recent_ts, "rid": "new"}) + "\n")
        kept = client.prune_old_logs(now=time.time())
        assert kept == 1
        with open(log_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 1
        assert rows[0]["rid"] == "new"

    def test_consume_marker_via_client(self, client):
        client.policy.record_peer_query("x", 7)
        assert client.consume_peer_query_marker_for_chain(7) is True
        assert client.consume_peer_query_marker_for_chain(7) is False
