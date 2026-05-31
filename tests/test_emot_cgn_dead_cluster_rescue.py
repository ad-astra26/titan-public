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


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("OK — dead-cluster rescue checks passed")
