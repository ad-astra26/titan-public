"""EMOT-CGN dead-cluster rescue + soft-responsibility recenter (2026-05-31, Maker).

Fixes the k-means dead-cluster trap behind the LOVE-never-fires + WONDER/IMPASSE
monoculture: the old recenter only updated clusters that hard-won ≥5 obs, so a
slot that never wins (LOVE on T1) was never moved → frozen at its RNG-seed →
never wins. The fix: soft-responsibility recenter (every centroid drifts) + rescue
of still-starved EMERGENT slots to the worst-served observation (anchors keep
their semantic seed).

Run: python -m pytest tests/test_emot_cgn_dead_cluster_rescue.py -v -p no:anchorpy
"""
import os
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from titan_hcl.logic.emotion_cluster import (
    EmotionClusterer, FEATURE_DIM, EMOT_PRIMITIVES,
)


def _clusterer():
    tmp = tempfile.mkdtemp()
    # recenter_interval_s=0 so force isn't needed for the time-gate, but we still
    # pass force=True to bypass the buffer-min on a fresh instance deterministically.
    return EmotionClusterer(save_dir=tmp, recenter_interval_s=0.0)


def _obs(center, n, jitter=0.01):
    out = []
    for _ in range(n):
        v = center + np.random.normal(0.0, jitter, FEATURE_DIM).astype(np.float32)
        out.append(v.astype(np.float32))
    return out


def test_starved_emergent_slot_is_rescued():
    clst = _clusterer()
    # Dense region A (most obs) + a small under-served region B far away.
    region_a = np.full(FEATURE_DIM, 0.5, dtype=np.float32); region_a[5:20] += 0.30
    region_b = np.full(FEATURE_DIM, 0.5, dtype=np.float32); region_b[60:80] += 0.40
    for v in _obs(region_a, 70):
        clst.observe(v)
    for v in _obs(region_b, 12):
        clst.observe(v)

    # An emergent slot that never wins is the dead one.
    love_before = clst._clusters["LOVE"].centroid.copy()
    assert clst._recenter_rescues == 0

    clst.maybe_recenter(force=True)

    # At least one starved emergent slot was rescued.
    assert clst._recenter_rescues >= 1
    # LOVE (a starved emergent slot) moved off its frozen seed.
    moved = float(np.linalg.norm(clst._clusters["LOVE"].centroid - love_before))
    assert moved > 0.05, f"LOVE centroid did not move (drift={moved})"


def test_rescued_slot_can_then_win():
    """After rescue toward an under-served region, observations in that region
    should start assigning to a (previously dead) emergent slot."""
    clst = _clusterer()
    region_a = np.full(FEATURE_DIM, 0.5, dtype=np.float32); region_a[5:20] += 0.30
    region_b = np.full(FEATURE_DIM, 0.5, dtype=np.float32); region_b[60:80] += 0.40
    for v in _obs(region_a, 70):
        clst.observe(v)
    for v in _obs(region_b, 12):
        clst.observe(v)
    clst.maybe_recenter(force=True)

    # A fresh observation in region B should now win SOME emergent slot (3..7),
    # not collapse to an anchor — the rescue placed a slot there.
    probe = region_b + np.random.normal(0.0, 0.01, FEATURE_DIM).astype(np.float32)
    p_id, _, _ = clst.assign(probe.astype(np.float32))
    assert p_id in EMOT_PRIMITIVES
    # The winner for region B should be an emergent slot (the rescued one),
    # since region B was under-served by the anchors.
    assert p_id in EMOT_PRIMITIVES[3:], f"region B won by {p_id}, expected emergent slot"


def test_starved_anchor_keeps_seed_not_rescued():
    """A STARVED anchor (wins nothing) keeps its semantic seed — it is never
    reseeded to a worst-served outlier (only emergent slots are rescued).
    Anchors that DO win obs may hard-drift toward them (that's k-means) — this
    test isolates the starved case by feeding obs at an emergent slot's centroid."""
    clst = _clusterer()
    flow_seed = clst._clusters["FLOW"].centroid.copy()
    impasse_seed = clst._clusters["IMPASSE_TENSION"].centroid.copy()
    # All obs sit on WONDER's centroid → WONDER wins them all; anchors win 0.
    wonder_c = clst._clusters["WONDER"].centroid.copy()
    for v in _obs(wonder_c, 60, jitter=0.004):
        clst.observe(v)
    clst.maybe_recenter(force=True)
    # Starved anchors are untouched (not updated, not rescued).
    assert np.allclose(clst._clusters["FLOW"].centroid, flow_seed, atol=1e-5)
    assert np.allclose(clst._clusters["IMPASSE_TENSION"].centroid, impasse_seed, atol=1e-5)


def test_emergent_soft_update_moves_winner():
    """An emergent slot that wins observations soft-updates toward them."""
    clst = _clusterer()
    wonder_before = clst._clusters["WONDER"].centroid.copy()
    # Obs offset from WONDER's centroid, small enough that WONDER stays nearest.
    target = wonder_before.copy(); target[100:115] += 0.05
    for v in _obs(target, 60, jitter=0.004):
        clst.observe(v)
    clst.maybe_recenter(force=True)
    drift = float(np.linalg.norm(clst._clusters["WONDER"].centroid - wonder_before))
    assert drift > 0.0, "WONDER did not soft-update toward its observations"
    # And the target region is now well-represented (small assignment distance).
    _, dist, _ = clst.assign(target.astype(np.float32))
    assert dist < 0.5


def test_buffer_persists_across_restart():
    """The observation buffer survives a 'restart' (new clusterer, same save_dir)
    so the recenter can actually reach its obs threshold — the reachability fix."""
    tmp = tempfile.mkdtemp()
    c1 = EmotionClusterer(save_dir=tmp, recenter_interval_s=0.0)
    region = np.full(FEATURE_DIM, 0.5, dtype=np.float32); region[5:20] += 0.2
    for v in _obs(region, 80):
        c1.observe(v)
    c1._save_buffer()
    buffered_before = len(c1._observation_buffer)
    assert buffered_before > 0
    # "Restart": brand-new clusterer over the same save_dir.
    c2 = EmotionClusterer(save_dir=tmp, recenter_interval_s=0.0)
    assert len(c2._observation_buffer) > 0, "buffer did not survive restart"
    assert len(c2._observation_buffer) == min(buffered_before, 600)


def test_adaptive_heal_cadence_while_dead():
    """While an emergent slot is dead (LOVE n=0), the effective recenter interval
    is the daily heal interval, not the weekly one — so the heal fires in days."""
    tmp = tempfile.mkdtemp()
    clst = EmotionClusterer(save_dir=tmp, recenter_interval_s=7 * 86400)
    # Fresh init: LOVE n_obs=0, not emerged → has_dead True → daily cadence.
    assert clst._clusters["LOVE"].n_observations == 0
    assert clst._heal_recenter_interval_s < clst._recenter_interval_s
    # Simulate a recenter ~2 days ago: with weekly interval it'd be skipped, but
    # the daily heal cadence (dead slot present) makes it due.
    import time as _t
    clst._last_recenter_ts = _t.time() - 2 * 86400
    region = np.full(FEATURE_DIM, 0.5, dtype=np.float32); region[5:20] += 0.2
    for v in _obs(region, 60):
        clst.observe(v)
    fired = clst.maybe_recenter(force=False)   # NOT forced — relies on cadence
    assert fired is True, "heal cadence did not make the recenter due"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("OK — dead-cluster rescue checks passed")
