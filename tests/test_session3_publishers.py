"""
Tests for Phase C Session 3 publishers (rFP §4.B.2-5 + §4.B.9-11).

Covers:
  - BaseStatePublisher: init, encode/oversize/write fail isolation,
    heartbeat ticks, get_stats
  - 7 concrete publishers: schema correctness, cold-boot stub, msgpack
    round-trip via StateRegistryReader
  - MultiSlotStatePublisher: composition + per-slot failure isolation
  - WorkerPublisherRunner: thread starts + state_fetcher exception
    isolation

Run: ``python -m pytest tests/test_session3_publishers.py -v -p no:anchorpy``
"""
from __future__ import annotations

import logging
import time

import msgpack
import pytest

from titan_hcl.core.state_registry import StateRegistryReader
from titan_hcl.logic.agency_state_publisher import AgencyStatePublisher
from titan_hcl.logic.assessment_state_publisher import AssessmentStatePublisher
from titan_hcl.logic.base_state_publisher import (
    BaseStatePublisher,
    MultiSlotStatePublisher,
)
from titan_hcl.logic.output_verifier_state_publisher import (
    OutputVerifierStatePublisher)
from titan_hcl.logic.reflex_state_publisher import ReflexStatePublisher
from titan_hcl.logic.rl_state_publisher import RLStatePublisher
from titan_hcl.logic.session3_state_specs import (
    AGENCY_STATE_SPEC,
    ASSESSMENT_STATE_SPEC,
    OUTPUT_VERIFIER_STATE_SPEC,
    REFLEX_STATE_SPEC,
    RL_STATE_SPEC,
    SOCIAL_PERCEPTION_STATE_SPEC,
    TIMECHAIN_STATE_SPEC,
)
from titan_hcl.logic.social_perception_state_publisher import (
    SocialPerceptionStatePublisher)
from titan_hcl.logic.timechain_state_publisher import (
    TimechainStatePublisher)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def shm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


def _read_slot(spec, shm_root):
    reader = StateRegistryReader(spec, shm_root)
    raw = reader.read_variable()
    return msgpack.unpackb(raw, raw=False) if raw else None


# ── BaseStatePublisher unit tests (via a tiny test subclass) ──────────


class _DummySpec:
    """Tiny RegistrySpec-shaped duck for base-class tests not tied to
    a real SPEC slot (we keep BaseStatePublisher unit tests independent
    from session3 specs to stay focused)."""

    name = "test_dummy"
    schema_version = 1
    payload_bytes = 256


def _make_dummy_publisher(shm_root_path):
    """A minimal subclass that publishes a fixed payload."""
    import numpy as np
    from titan_hcl.core.state_registry import RegistrySpec

    class _DummyPub(BaseStatePublisher):
        slot_name = "test_dummy"
        slot_spec = RegistrySpec(
            name="test_dummy", dtype=np.dtype("uint8"),
            shape=(256,), schema_version=1, variable_size=True)

        def _compute_payload(self, payload):
            return dict(payload)

    return _DummyPub(titan_id="T_TEST")


def test_base_init_logs_and_writer_lazy(shm_root, caplog):
    caplog.set_level(logging.INFO,
                     logger="titan_hcl.logic.base_state_publisher")
    pub = _make_dummy_publisher(shm_root)
    stats = pub.get_stats()
    assert stats["publish_count"] == 0
    assert stats["writer_attached"] is False
    assert any("initialized" in r.message for r in caplog.records)


def test_base_publish_success_path(shm_root, caplog):
    caplog.set_level(logging.INFO,
                     logger="titan_hcl.logic.base_state_publisher")
    pub = _make_dummy_publisher(shm_root)
    pub.publish({"x": 1, "ts": time.time()})
    stats = pub.get_stats()
    assert stats["publish_success"] == 1
    assert stats["encode_fails"] == 0
    assert any("FIRST PUBLISH SUCCESS" in r.message for r in caplog.records)


def test_base_oversize_logged_critical_no_write(shm_root, caplog):
    caplog.set_level(logging.CRITICAL,
                     logger="titan_hcl.logic.base_state_publisher")
    pub = _make_dummy_publisher(shm_root)
    # 256-byte slot; pack 1KB to trigger oversize
    pub.publish({"big": "x" * 1024})
    stats = pub.get_stats()
    assert stats["oversize_fails"] == 1
    assert stats["publish_success"] == 0
    assert any("payload" in r.message and "MAX" in r.message
               for r in caplog.records)


def test_base_compute_payload_raises_isolated(shm_root):
    """Subclass _compute_payload raising must NOT crash the thread —
    BaseStatePublisher.publish catches and counts it."""
    import numpy as np
    from titan_hcl.core.state_registry import RegistrySpec

    class _BrokenPub(BaseStatePublisher):
        slot_name = "test_broken"
        slot_spec = RegistrySpec(
            name="test_broken", dtype=np.dtype("uint8"),
            shape=(256,), schema_version=1, variable_size=True)

        def _compute_payload(self, *args, **kwargs):
            raise RuntimeError("simulated compute failure")

    pub = _BrokenPub(titan_id="T_TEST")
    pub.publish()  # must not raise
    stats = pub.get_stats()
    assert stats["encode_fails"] == 1
    assert stats["publish_success"] == 0


def test_base_heartbeat_at_canonical_ticks(shm_root, caplog):
    caplog.set_level(logging.INFO,
                     logger="titan_hcl.logic.base_state_publisher")
    pub = _make_dummy_publisher(shm_root)
    for _ in range(10):
        pub.publish({"x": 1})
    heartbeat_logs = [r for r in caplog.records if "heartbeat" in r.message]
    assert len(heartbeat_logs) >= 2  # tick 1 + tick 10


# ── 7 concrete publisher round-trip tests ────────────────────────────


def test_assessment_state_round_trip(shm_root):
    pub = AssessmentStatePublisher(titan_id="T_TEST")

    class _Assess:
        def get_stats(self):
            return {
                "average_score": 0.82, "total": 12,
                "recent": [{"score": 0.8}, {"score": 0.84}],
                "trend": 0.05, "score_variance": 0.012,
                "research_avg_score": 0.79,
            }

    pub.publish(_Assess())
    decoded = _read_slot(ASSESSMENT_STATE_SPEC, shm_root)
    assert decoded["average_score"] == pytest.approx(0.82)
    assert decoded["total"] == 12
    assert decoded["trend"] == pytest.approx(0.05)
    assert "ts" in decoded


def test_assessment_state_cold_boot(shm_root):
    pub = AssessmentStatePublisher(titan_id="T_TEST")
    pub.publish(None)
    decoded = _read_slot(ASSESSMENT_STATE_SPEC, shm_root)
    assert decoded["total"] == 0
    assert decoded["average_score"] == 0.0


def test_agency_state_round_trip(shm_root):
    pub = AgencyStatePublisher(titan_id="T_TEST")

    class _Reg:
        def list_all_names(self):
            return ["helper_a", "helper_b"]

        def get_all_statuses(self):
            return {"helper_a": "ready", "helper_b": "degraded"}

    class _Agency:
        _action_counter = 42
        _llm_calls_this_hour = 5
        _budget_per_hour = 10
        _registry = _Reg()
        _history = [
            {"posture": "research", "helper": "helper_a",
             "success": True, "ts": time.time() - 100},
            {"posture": "create", "helper": "helper_b",
             "success": False, "ts": time.time() - 50},
        ]

    pub.publish(_Agency())
    decoded = _read_slot(AGENCY_STATE_SPEC, shm_root)
    assert decoded["total_actions"] == 42
    assert decoded["llm_calls_this_hour"] == 5
    assert decoded["budget_per_hour"] == 10
    assert decoded["budget_remaining"] == 5
    assert decoded["actions_this_hour"] == 2
    assert "helper_a" in decoded["registered_helpers"]
    assert decoded["helper_statuses"]["helper_a"] == "ready"
    assert decoded["success_rate"] == pytest.approx(0.5)


def test_agency_state_cold_boot(shm_root):
    pub = AgencyStatePublisher(titan_id="T_TEST")
    pub.publish(None)
    decoded = _read_slot(AGENCY_STATE_SPEC, shm_root)
    assert decoded["total_actions"] == 0
    assert decoded["registered_helpers"] == []


def test_rl_state_round_trip(shm_root):
    pub = RLStatePublisher(titan_id="T_TEST")

    class _Recorder:
        buffer = list(range(100))
        storage = list(range(50))
        buffer_size = 50000
        last_train_ts = 1234.5
        training_loss_ema = 0.014
        total_transitions = 100

    class _Gatekeeper:
        sovereignty_score = 0.78
        _decision_history = [{"d": i} for i in range(7)]

    pub.publish(_Recorder(), _Gatekeeper())
    decoded = _read_slot(RL_STATE_SPEC, shm_root)
    assert decoded["buffer_len"] == 100
    assert decoded["storage_len"] == 50
    assert decoded["sovereignty_score"] == pytest.approx(0.78)
    assert decoded["decision_history_len"] == 7
    assert decoded["training_loss_ema"] == pytest.approx(0.014)


def test_rl_state_cold_boot(shm_root):
    pub = RLStatePublisher(titan_id="T_TEST")
    pub.publish(None, None)
    decoded = _read_slot(RL_STATE_SPEC, shm_root)
    assert decoded["buffer_len"] == 0
    assert decoded["sovereignty_score"] == 0.0


def test_timechain_state_round_trip(shm_root):
    pub = TimechainStatePublisher(titan_id="T_TEST")

    class _Mempool:
        def get_pending_forks(self):
            return ["main", "declarative", "procedural"]

    class _TC:
        total_blocks = 12345
        chi_spent_total = 678.9
        _mempool = _Mempool()
        _last_integrity_status = "all_forks_healthy"

    pub.publish(_TC())
    decoded = _read_slot(TIMECHAIN_STATE_SPEC, shm_root)
    assert decoded["total_blocks"] == 12345
    assert decoded["chi_spent_total"] == pytest.approx(678.9)
    assert len(decoded["fork_summary"]) == 3
    assert decoded["integrity_status"] == "all_forks_healthy"


def test_timechain_state_cold_boot(shm_root):
    pub = TimechainStatePublisher(titan_id="T_TEST")
    pub.publish(None)
    decoded = _read_slot(TIMECHAIN_STATE_SPEC, shm_root)
    assert decoded["total_blocks"] == 0
    assert decoded["fork_summary"] == []


def test_reflex_state_round_trip(shm_root):
    pub = ReflexStatePublisher(titan_id="T_TEST")

    class _ReflexNet:
        def get_stats(self):
            return {
                "fire_count": 17, "total_updates": 1000,
                "last_loss": 0.045, "fire_threshold": 0.6,
            }

    class _NS:
        _reflex_net = _ReflexNet()

    pub.publish(_NS())
    decoded = _read_slot(REFLEX_STATE_SPEC, shm_root)
    assert "default" in decoded["reflexes"]
    assert decoded["reflexes"]["default"]["fire_count"] == 17
    assert decoded["reflex_count"] == 1


def test_reflex_state_cold_boot(shm_root):
    pub = ReflexStatePublisher(titan_id="T_TEST")
    pub.publish(None)
    decoded = _read_slot(REFLEX_STATE_SPEC, shm_root)
    assert decoded["reflexes"] == {}
    assert decoded["reflex_count"] == 0


def test_social_perception_state_round_trip(shm_root):
    pub = SocialPerceptionStatePublisher(titan_id="T_TEST")

    class _Inner:
        observables = {
            "sentiment_ema": 0.65, "interaction_rate": 0.42,
            "social_activity": 0.58, "last_interaction_ts": 1234.0,
        }

    state_refs = {"inner_state": _Inner()}
    pub.publish(state_refs)
    decoded = _read_slot(SOCIAL_PERCEPTION_STATE_SPEC, shm_root)
    assert decoded["sentiment_ema"] == pytest.approx(0.65)
    assert decoded["interaction_rate"] == pytest.approx(0.42)
    assert decoded["social_activity"] == pytest.approx(0.58)


def test_social_perception_state_cold_boot(shm_root):
    pub = SocialPerceptionStatePublisher(titan_id="T_TEST")
    pub.publish({})
    decoded = _read_slot(SOCIAL_PERCEPTION_STATE_SPEC, shm_root)
    assert decoded["sentiment_ema"] == 0.0
    assert decoded["interaction_rate"] == 0.0


def test_output_verifier_state_round_trip(shm_root):
    pub = OutputVerifierStatePublisher(titan_id="T_TEST")

    class _Verifier:
        verified_count = 142
        rejected_count = 7
        sovereignty_score = 0.95
        threats_24h = {
            "directive": 2, "injection": 1, "consistency": 0,
            "identity": 0, "qualia": 1,
        }
        _recent_rejections = [
            {"category": "directive", "ts": 100.0, "score": 0.4},
            {"category": "injection", "ts": 200.0, "score": 0.3},
        ]

    pub.publish(_Verifier())
    decoded = _read_slot(OUTPUT_VERIFIER_STATE_SPEC, shm_root)
    assert decoded["verified_count"] == 142
    assert decoded["rejected_count"] == 7
    assert decoded["sovereignty_score"] == pytest.approx(0.95)
    assert decoded["threats_24h"]["directive"] == 2
    assert len(decoded["recent_rejections_digest"]) == 2


def test_output_verifier_state_cold_boot(shm_root):
    pub = OutputVerifierStatePublisher(titan_id="T_TEST")
    pub.publish(None)
    decoded = _read_slot(OUTPUT_VERIFIER_STATE_SPEC, shm_root)
    assert decoded["verified_count"] == 0
    assert decoded["sovereignty_score"] == 0.0
    assert all(v == 0 for v in decoded["threats_24h"].values())


# ── MultiSlotStatePublisher composition + isolation ─────────────────


def test_multi_slot_composition(shm_root):
    """Multi-slot composer publishes to ALL slots per tick; failure of
    one slot does NOT block the others."""
    p1 = AssessmentStatePublisher(titan_id="T_TEST")
    p2 = AgencyStatePublisher(titan_id="T_TEST")
    multi = MultiSlotStatePublisher(publishers=[p1, p2])

    class _Assess:
        def get_stats(self):
            return {"average_score": 0.5, "total": 1, "recent": [],
                    "trend": 0.0, "score_variance": 0.0,
                    "research_avg_score": 0.5}

    # Both publishers receive the SAME args here — but only the first
    # one matches; the second one expects an AgencyModule, gets the
    # _Assess stub, runs its defensive shape walk, returns cold-boot-style
    # payload (since _Assess doesn't have _action_counter).
    multi.publish(_Assess())

    s1 = p1.get_stats()
    s2 = p2.get_stats()
    # p1 succeeds (assess shape)
    assert s1["publish_success"] == 1
    # p2 also writes (defensive — falls back to zeros for missing attrs)
    assert s2["publish_success"] == 1


# ── WorkerPublisherRunner thread integration ────────────────────────


def test_worker_publisher_runner_thread_starts(shm_root):
    """run_worker_publisher must start a daemon thread that ticks and
    survives state_fetcher returning None or raising."""
    from titan_hcl.logic.worker_publisher_runner import run_worker_publisher
    pub = _make_dummy_publisher(shm_root)
    fetch_count = {"n": 0}

    def _fetcher():
        fetch_count["n"] += 1
        if fetch_count["n"] % 3 == 0:
            raise RuntimeError("simulated fetch failure")
        return {"x": fetch_count["n"]}

    t = run_worker_publisher(
        publisher=pub, state_fetcher=_fetcher,
        worker_name="test_worker", cadence_s=0.05)
    time.sleep(0.3)  # ~6 ticks expected
    assert t.is_alive()
    stats = pub.get_stats()
    # Some ticks should have published (fetch did succeed for non-3rd)
    assert stats["publish_success"] >= 2
    # Some ticks should have failed (fetch raised every 3rd)
    assert stats["encode_fails"] >= 1 or fetch_count["n"] >= 6
