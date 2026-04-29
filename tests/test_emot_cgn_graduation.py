"""Tests for Phase B v3 graduation state machine (rFP §23.5).

Covers:
  - SHADOW → OBSERVING transition when a region persists ≥4 reclusters
  - OBSERVING → GRADUATED blocked by age gate + naming gate
  - Dissolution drops a region out of persistent_regions on next pass
  - Persistence bookkeeping survives save_state / _load_state roundtrip
  - graduation_snapshot blockers list matches the gate state
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from titan_plugin.logic.emot_bundle_protocol import (
    GRAD_SHADOW, GRAD_OBSERVING, GRAD_GRADUATED,
)
from titan_plugin.logic.emot_region_clusterer import (
    RegionClusterer, STATE_DIM,
)


def _make_clusterer(tmp_path, min_cluster_size=5):
    return RegionClusterer(save_dir=str(tmp_path),
                           min_cluster_size=min_cluster_size)


def _seed_two_clusters(rc, seed=0, n_per=60):
    """Seed two well-separated clusters of synthetic observations."""
    rng = np.random.default_rng(seed)
    for _ in range(n_per):
        rc.observe(rng.normal(0.2, 0.01, STATE_DIM).astype(np.float32))
    for _ in range(n_per):
        rc.observe(rng.normal(0.8, 0.01, STATE_DIM).astype(np.float32))


def _feed_consistent_points(rc, seed=0, n=30):
    """Add more observations near the two established clusters so
    the centroids stay stable across reclusters."""
    rng = np.random.default_rng(seed + 100)
    for _ in range(n):
        rc.observe(rng.normal(0.2, 0.01, STATE_DIM).astype(np.float32))
    for _ in range(n):
        rc.observe(rng.normal(0.8, 0.01, STATE_DIM).astype(np.float32))


class TestGraduationStateMachine:

    def test_fresh_boot_is_shadow(self, tmp_path):
        """A clusterer with no reclusters yet is always SHADOW."""
        rc = _make_clusterer(tmp_path)
        assert rc.graduation_status() == GRAD_SHADOW
        assert rc._recluster_count == 0

    def test_first_emergence_not_yet_observing(self, tmp_path):
        """A region that appeared in ONE recluster hasn't persisted yet —
        status stays SHADOW until PERSISTENCE_THRESHOLD is met."""
        rc = _make_clusterer(tmp_path)
        _seed_two_clusters(rc, seed=1)
        rc.recluster()  # 1st recluster — regions appear for first time
        assert len(rc._regions) >= 1
        # After just one pass, consecutive_reclusters = 1 < threshold=4
        assert rc.graduation_status() == GRAD_SHADOW
        assert len(rc.persistent_regions()) == 0

    def test_persistence_promotes_to_observing(self, tmp_path):
        """After 4 consecutive reclusters with same signature → OBSERVING."""
        rc = _make_clusterer(tmp_path)
        _seed_two_clusters(rc, seed=2)
        # Run 4 consecutive reclusters with consistent additional data
        for i in range(4):
            _feed_consistent_points(rc, seed=i)
            rc.recluster()
        # At least one region should now have persisted 4 cycles
        persistent = rc.persistent_regions()
        assert len(persistent) >= 1, \
            f"expected persistent regions after 4 reclusters, got {len(persistent)}"
        # 1-2 persistent regions but neither named + age < 14 days →
        # blocked on multiple gates, stays at OBSERVING
        assert rc.graduation_status() == GRAD_OBSERVING

    def test_graduation_requires_named_and_age_and_count(self, tmp_path):
        """Full GRAD_GRADUATED requires: ≥3 persistent regions AND
        ≥14 days age AND ≥1 named region. Test each gate blocks."""
        rc = _make_clusterer(tmp_path)
        _seed_two_clusters(rc, seed=3)
        for i in range(4):
            _feed_consistent_points(rc, seed=i)
            rc.recluster()
        persistent = rc.persistent_regions()
        # Force the age gate open so only region-count + naming remain.
        import time as _t
        rc._first_boot_ts = _t.time() - (15 * 86400)
        # Only 2 regions → still blocks on count gate.
        assert rc.graduation_status() == GRAD_OBSERVING
        # Name one of them (still only 2 regions so count gate fails).
        if persistent:
            persistent[0].label = "serene"
        assert rc.graduation_status() == GRAD_OBSERVING
        snap = rc.graduation_snapshot()
        assert any("persistent regions" in b for b in snap["blocking"])

    def test_dissolution_drops_from_persistent(self, tmp_path):
        """If a region's signature fails to match next recluster, it drops
        out of persistent_regions even if it had persisted before."""
        rc = _make_clusterer(tmp_path)
        _seed_two_clusters(rc, seed=4)
        rc.recluster()
        initial_regions = list(rc._regions.values())
        assert len(initial_regions) >= 1
        # Inject FAR-away point cloud → old signatures won't match
        rng = np.random.default_rng(999)
        for _ in range(60):
            rc.observe(rng.normal(5.0, 0.1, STATE_DIM).astype(np.float32))
        rc.recluster()
        # Old signatures dropped — their last_seen_recluster stays at the
        # earlier pass, so they don't survive the "alive on current" filter.
        persistent = rc.persistent_regions()
        assert len(persistent) == 0


class TestGraduationSnapshot:

    def test_snapshot_fields_present(self, tmp_path):
        rc = _make_clusterer(tmp_path)
        snap = rc.graduation_snapshot()
        for k in ("status", "recluster_count", "age_seconds", "age_days",
                  "persistent_regions", "total_regions", "named_regions",
                  "gates", "blocking"):
            assert k in snap, f"missing {k}"
        # Fresh clusterer: status=SHADOW, 0 reclusters, 0 persistent
        assert snap["status"] == GRAD_SHADOW
        assert snap["recluster_count"] == 0
        assert snap["persistent_regions"] == 0

    def test_blockers_enumerated_before_graduation(self, tmp_path):
        """Before graduation, snapshot.blocking names exactly which gate
        fails. Useful for UI and operator diagnosis."""
        rc = _make_clusterer(tmp_path)
        _seed_two_clusters(rc, seed=5)
        for i in range(4):
            _feed_consistent_points(rc, seed=i)
            rc.recluster()
        snap = rc.graduation_snapshot()
        # 2 persistent regions < min=3, age < 14d, no named → 3 blockers
        assert len(snap["blocking"]) >= 2
        assert any("persistent" in b for b in snap["blocking"])
        assert any("observation window" in b for b in snap["blocking"])


class TestGraduationPersistence:

    def test_persistence_bookkeeping_survives_reload(self, tmp_path):
        """save_state → _load_state roundtrip preserves recluster_count,
        first_boot_ts, and per-region first_seen/last_seen counters."""
        sd = str(tmp_path)
        rc1 = _make_clusterer(tmp_path, min_cluster_size=5)
        _seed_two_clusters(rc1, seed=6)
        for i in range(3):
            _feed_consistent_points(rc1, seed=i)
            rc1.recluster()
        persistent_before = rc1.persistent_regions()
        counters_before = {rid: (r.first_seen_recluster, r.last_seen_recluster)
                           for rid, r in rc1._regions.items()}
        rc1.save_state()

        rc2 = RegionClusterer(save_dir=sd, min_cluster_size=5)
        # Counters restored.
        assert rc2._recluster_count == rc1._recluster_count
        assert abs(rc2._first_boot_ts - rc1._first_boot_ts) < 1e-3
        counters_after = {rid: (r.first_seen_recluster, r.last_seen_recluster)
                          for rid, r in rc2._regions.items()}
        assert counters_after == counters_before

    def test_persistence_threshold_constant(self):
        """Pin the graduation constants — changing these shifts the
        meaning of 'emergent emotion' across Titans. Changes require
        rFP §23.5 amendment."""
        assert RegionClusterer.PERSISTENCE_THRESHOLD_RECLUSTERS == 4
        assert RegionClusterer.GRADUATION_MIN_PERSISTENT_REGIONS == 3
        assert RegionClusterer.GRADUATION_MIN_AGE_S == 14 * 86400


class TestBufferPersistence:
    """Buffer persistence (rFP §23.6+, 2026-04-22): rolling trajectory
    buffer survives restarts so dev-cycle restarts don't wipe ~50 min
    of HDBSCAN warmup each time.
    """

    def test_buffer_saves_and_restores(self, tmp_path):
        """End-to-end: seed, save, recreate, verify buffer restored."""
        sd = str(tmp_path)
        rc1 = _make_clusterer(tmp_path, min_cluster_size=5)
        _seed_two_clusters(rc1, seed=100, n_per=40)
        assert len(rc1._buffer) == 80
        rc1.save_state()  # save_state now also writes the buffer

        rc2 = RegionClusterer(save_dir=sd, min_cluster_size=5)
        assert len(rc2._buffer) == 80, \
            f"expected 80 obs restored, got {len(rc2._buffer)}"
        # Spot-check values
        assert np.allclose(rc2._buffer[0], rc1._buffer[0], atol=1e-5)
        assert np.allclose(rc2._buffer[-1], rc1._buffer[-1], atol=1e-5)

    def test_buffer_shape_mismatch_discarded(self, tmp_path):
        """Wrong STATE_DIM (e.g. schema migration) → discard buffer."""
        sd = str(tmp_path)
        wrong_dim = STATE_DIM - 2  # simulate v1 208D remnant
        fake = np.random.rand(50, wrong_dim).astype(np.float32)
        np.save(os.path.join(sd, "regions_buffer.npy"),
                fake, allow_pickle=False)

        rc = RegionClusterer(save_dir=sd, min_cluster_size=5)
        assert len(rc._buffer) == 0, (
            f"wrong-shape buffer should be discarded, got {len(rc._buffer)}")

    def test_buffer_age_cap_discards_stale(self, tmp_path):
        """Buffer files > BUFFER_MAX_AGE_S must be discarded."""
        sd = str(tmp_path)
        rc1 = _make_clusterer(tmp_path, min_cluster_size=5)
        _seed_two_clusters(rc1, seed=200, n_per=30)
        rc1.save_state()
        buf_path = os.path.join(sd, "regions_buffer.npy")
        assert os.path.exists(buf_path)

        # Backdate mtime to 8 days ago
        import time as _t
        old_ts = _t.time() - 8 * 86400
        os.utime(buf_path, (old_ts, old_ts))

        rc2 = RegionClusterer(save_dir=sd, min_cluster_size=5)
        assert len(rc2._buffer) == 0, (
            f"stale buffer should be discarded, got {len(rc2._buffer)}")

    def test_empty_buffer_removes_file(self, tmp_path):
        """Saving an empty buffer should remove any stale .npy file."""
        sd = str(tmp_path)
        buf_path = os.path.join(sd, "regions_buffer.npy")
        fake = np.zeros((5, STATE_DIM), dtype=np.float32)
        np.save(buf_path, fake, allow_pickle=False)
        assert os.path.exists(buf_path)

        rc = RegionClusterer(save_dir=sd, min_cluster_size=5)
        rc._buffer.clear()  # force empty
        rc.save_state()
        assert not os.path.exists(buf_path), (
            "empty-buffer save should remove stale file")

    def test_restored_buffer_enables_fast_recluster(self, tmp_path):
        """Integration benefit: restart → full buffer immediately available
        → HDBSCAN can recluster right away, not wait for 50 fresh samples."""
        sd = str(tmp_path)
        rc1 = _make_clusterer(tmp_path, min_cluster_size=5)
        _seed_two_clusters(rc1, seed=300, n_per=60)
        rc1.save_state()

        rc2 = RegionClusterer(save_dir=sd, min_cluster_size=5)
        assert len(rc2._buffer) >= 100
        n_regions = rc2.recluster()
        assert n_regions >= 1, (
            f"immediate recluster post-restart should find ≥1 region "
            f"with restored buffer; got {n_regions}")
