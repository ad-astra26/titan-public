"""P1 (RFP_emergent_mastery_curriculum) — developmental_age cross-process publish.

Verify-mode finding: the RFP §7.P1 assumed "no canonical SHM source post-v1.19.0"
for developmental_age (the stale dashboard.py:674 TODO). That assumption is FALSE —
cognitive_worker already co-publishes pi_monitor.get_stats() (incl.
developmental_age = π-cluster count, and heartbeat_ratio) into
meta_reasoning_state.bin under the `pi_heartbeat` key via
MetaReasoningStatePublisher. So P1 needs no new SHM slot: only the read-side
fix in the dashboard /status.lifetime composer (which had hardcoded 0/0.0).

These tests pin the producer contract (the source publishes developmental_age +
heartbeat_ratio) and the read parity the dashboard composer now relies on,
including the flag-off / stub case (empty pi_heartbeat → 0 / 0.0).
"""

import os
import tempfile
import types

import pytest


@pytest.fixture
def tmp_state_path():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "pi_heartbeat_state.json")


def _drive_cluster(monitor, n_pi=5):
    """Feed n_pi consecutive π-epochs (curvature in the π band) to ratchet
    at least one completed cluster (min_cluster_size default 3)."""
    for epoch in range(1, n_pi + 1):
        monitor.observe(3.14159, epoch)  # 2.9 < x < 3.3 ⇒ π-epoch
    return monitor


def test_pi_heartbeat_get_stats_carries_developmental_age(tmp_state_path):
    from titan_hcl.logic.pi_heartbeat import PiHeartbeatMonitor

    pi = PiHeartbeatMonitor(state_path=tmp_state_path)
    _drive_cluster(pi, n_pi=5)

    stats = pi.get_stats()
    assert stats["developmental_age"] == pi._cluster_count
    assert stats["developmental_age"] >= 1, (
        "5 consecutive π-epochs (min_cluster_size 3) must ratchet ≥1 cluster")
    assert "heartbeat_ratio" in stats
    assert 0.0 <= stats["heartbeat_ratio"] <= 1.0


def test_publisher_copublishes_developmental_age(tmp_state_path):
    """MetaReasoningStatePublisher._compute_payload must emit the rich
    pi_heartbeat block (developmental_age + heartbeat_ratio) — the
    cross-process source the dashboard composer reads. No SHM write here:
    _compute_payload is pure (publish() is what touches the slot)."""
    from titan_hcl.logic.pi_heartbeat import PiHeartbeatMonitor
    from titan_hcl.logic.meta_reasoning_state_publisher import (
        MetaReasoningStatePublisher,
    )

    pi = PiHeartbeatMonitor(state_path=tmp_state_path)
    _drive_cluster(pi, n_pi=5)
    expected_age = pi._cluster_count

    pub = MetaReasoningStatePublisher(titan_id="test_p1")
    # Minimal meta_engine stub: no get_audit_stats / accumulators → safe
    # defaults; we only assert the co-located pi_heartbeat block.
    payload = pub._compute_payload(
        meta_engine=types.SimpleNamespace(), pi_monitor=pi)

    assert isinstance(payload.get("pi_heartbeat"), dict)
    ph = payload["pi_heartbeat"]
    assert ph.get("developmental_age") == expected_age
    assert ph.get("developmental_age", 0) >= 1
    assert "heartbeat_ratio" in ph


def test_stub_payload_has_empty_pi_heartbeat():
    """meta_engine=None ⇒ stub: pi_heartbeat is {} (the flag-off / cold-boot
    case the dashboard read must default to 0 / 0.0 for)."""
    from titan_hcl.logic.meta_reasoning_state_publisher import (
        MetaReasoningStatePublisher,
    )

    pub = MetaReasoningStatePublisher(titan_id="test_p1_stub")
    payload = pub._compute_payload(meta_engine=None)
    assert payload.get("pi_heartbeat") == {}


def _dashboard_lifetime_read(meta_reasoning_slot):
    """Mirror the dashboard /status.lifetime read of developmental_age +
    heartbeat_ratio from coord['meta_reasoning']['pi_heartbeat']
    (dashboard.py ~670-680). Pinned here so the read contract is regression-
    guarded without spinning the whole /status endpoint + SHM stack."""
    _mr = meta_reasoning_slot or {}
    _pi_rich = _mr.get("pi_heartbeat", {}) if isinstance(_mr, dict) else {}
    if not isinstance(_pi_rich, dict):
        _pi_rich = {}
    return (
        int(_pi_rich.get("developmental_age", 0) or 0),
        float(_pi_rich.get("heartbeat_ratio", 0.0) or 0.0),
    )


def test_dashboard_read_surfaces_published_age():
    age, ratio = _dashboard_lifetime_read(
        {"pi_heartbeat": {"developmental_age": 11425, "heartbeat_ratio": 0.42}})
    assert age == 11425
    assert ratio == pytest.approx(0.42)


def test_dashboard_read_defaults_when_empty():
    # flag-off / stub / cold-boot: empty pi_heartbeat ⇒ 0 / 0.0 (legacy parity).
    assert _dashboard_lifetime_read({"pi_heartbeat": {}}) == (0, 0.0)
    assert _dashboard_lifetime_read({}) == (0, 0.0)
    assert _dashboard_lifetime_read(None) == (0, 0.0)
