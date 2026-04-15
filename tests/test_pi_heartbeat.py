"""Tests for π-Heartbeat Monitor — READ-ONLY curvature observer."""
import math
import pytest

from titan_plugin.logic.pi_heartbeat import PiHeartbeatMonitor


def _fresh(tmp_path, name, min_cluster=3, min_gap=2):
    return PiHeartbeatMonitor(
        min_cluster_size=min_cluster, min_gap_size=min_gap,
        state_path=str(tmp_path / f"{name}.json"))


class TestPiDetection:

    def test_single_pi_not_cluster(self, tmp_path):
        mon = _fresh(tmp_path, "t1")
        assert mon.observe(math.pi, 1) is None
        assert not mon.in_cluster

    def test_cluster_start_at_min_size(self, tmp_path):
        mon = _fresh(tmp_path, "t2")
        assert mon.observe(3.14, 1) is None
        assert mon.observe(3.14, 2) is None
        assert mon.observe(3.14, 3) == "CLUSTER_START"
        assert mon.in_cluster

    def test_cluster_end_at_min_gap(self, tmp_path):
        mon = _fresh(tmp_path, "t3", min_cluster=2, min_gap=2)
        mon.observe(3.14, 1)
        mon.observe(3.14, 2)  # CLUSTER_START
        mon.observe(0.5, 3)
        assert mon.observe(0.5, 4) == "CLUSTER_END"
        assert not mon.in_cluster

    def test_single_zero_does_not_end_cluster(self, tmp_path):
        mon = _fresh(tmp_path, "t4", min_cluster=2, min_gap=3)
        mon.observe(3.14, 1)
        mon.observe(3.14, 2)
        assert mon.observe(0.0, 3) is None
        assert mon.in_cluster

    def test_cluster_count_increments(self, tmp_path):
        mon = _fresh(tmp_path, "t5", min_cluster=2, min_gap=2)
        mon.observe(3.14, 1)
        mon.observe(3.14, 2)  # START cluster 1
        mon.observe(0.0, 3)
        mon.observe(0.0, 4)   # END cluster 1
        assert mon.developmental_age == 1
        mon.observe(3.14, 5)
        mon.observe(3.14, 6)  # START cluster 2
        assert mon.developmental_age == 2

    def test_developmental_age(self, tmp_path):
        mon = _fresh(tmp_path, "t6", min_cluster=2, min_gap=2)
        assert mon.developmental_age == 0
        mon.observe(3.14, 1)
        mon.observe(3.14, 2)
        assert mon.developmental_age == 1

    def test_heartbeat_ratio(self, tmp_path):
        mon = _fresh(tmp_path, "t7", min_cluster=2, min_gap=2)
        mon.observe(3.14, 1)
        mon.observe(3.14, 2)
        mon.observe(0.0, 3)
        mon.observe(0.0, 4)
        assert mon.heartbeat_ratio == pytest.approx(0.5)

    def test_stats_complete(self, tmp_path):
        mon = _fresh(tmp_path, "t8", min_cluster=2, min_gap=2)
        mon.observe(3.14, 1)
        mon.observe(3.14, 2)
        stats = mon.get_stats()
        assert stats["cluster_count"] == 1
        assert stats["total_pi_epochs"] == 2
        assert "developmental_age" in stats
        assert "heartbeat_ratio" in stats

    def test_pi_boundary_values(self, tmp_path):
        m1 = _fresh(tmp_path, "t9a", min_cluster=1, min_gap=1)
        assert m1.observe(2.91, 1) == "CLUSTER_START"
        m2 = _fresh(tmp_path, "t9b", min_cluster=1, min_gap=1)
        assert m2.observe(3.29, 1) == "CLUSTER_START"
        m3 = _fresh(tmp_path, "t9c", min_cluster=1, min_gap=1)
        assert m3.observe(2.89, 1) is None
        m4 = _fresh(tmp_path, "t9d", min_cluster=1, min_gap=1)
        assert m4.observe(3.31, 1) is None

    def test_sustained_4_1_is_one_cluster(self, tmp_path):
        """4π+1zero repeated: gaps of 1 < min_gap=2, so it's one sustained cluster."""
        mon = _fresh(tmp_path, "t10")
        for cycle in range(4):
            base = cycle * 5
            for i in range(4):
                mon.observe(3.1416, base + i + 1)
            mon.observe(0.0, base + 5)
        assert mon.in_cluster
        assert mon.developmental_age == 1

    def test_real_cluster_boundary(self, tmp_path):
        """4π, 3 zeros (>min_gap=2), 4π → two clusters."""
        mon = _fresh(tmp_path, "t11", min_cluster=3, min_gap=2)
        for i in range(4):
            mon.observe(3.14, i + 1)
        for i in range(3):
            mon.observe(0.0, 5 + i)
        for i in range(4):
            mon.observe(3.14, 8 + i)
        assert mon.developmental_age == 2
        assert len(mon._cluster_sizes) == 1  # First cluster recorded

    def test_avg_cluster_size(self, tmp_path):
        mon = _fresh(tmp_path, "t12", min_cluster=2, min_gap=2)
        # Cluster 1: 3 π-epochs
        for i in range(3):
            mon.observe(3.14, i + 1)
        for i in range(2):
            mon.observe(0.0, 4 + i)
        # Cluster 2: 5 π-epochs
        for i in range(5):
            mon.observe(3.14, 6 + i)
        for i in range(2):
            mon.observe(0.0, 11 + i)
        assert len(mon._cluster_sizes) == 2
        assert mon.avg_cluster_size == pytest.approx(4.0)

    def test_zero_curvature_not_pi(self, tmp_path):
        mon = _fresh(tmp_path, "t13", min_cluster=1, min_gap=1)
        assert mon.observe(0.0, 1) is None
        assert not mon.in_cluster
