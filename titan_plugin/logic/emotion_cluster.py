"""Emotion Cluster — k-means clustering of felt-state vectors for EMOT-CGN.

⚠ TRANSITIONAL (audit 2026-04-23 Q3): The 8 hand-seeded primitive slots
below are transitional display labels pending the v3 attractor model
(`rFP_titan_emotion_attractor_model.md`, in draft). Root cause of the
WONDER monoculture + LOVE-never-fires symptoms observed in audit:

  - FLOW / IMPASSE_TENSION / RESOLUTION have meaningful centroid
    perturbations (inner_mind +0.12, DA +0.10, etc.)
  - PEACE / CURIOSITY / GRIEF / WONDER / LOVE are initialized with pure
    rng.normal(0, 0.02, 150D) — the RNG lottery (seed 20260420) happens
    to place WONDER nearest to typical neutral observations, so WONDER
    wins ~all non-anchored assignments (70-95% in 3-day archaeology).
    LOVE's random position is never the nearest for any observation.

DO NOT TUNE individual primitive seeds or add new ones. The legacy
design (hardcoded categories + nearest-centroid assignment) is being
replaced by the v3 attractor-state model (emotion as multi-level
equilibrium over composite state, named after stable recurrence).
§16 Options A-D in rFP_emot_cgn_v2.md are superseded by that work.

Each cluster is a candidate emotion primitive. Seeded from hand-crafted
anchors (FLOW, IMPASSE_TENSION, RESOLUTION) + 5 emergent slots (PEACE,
CURIOSITY, GRIEF, WONDER, LOVE). Centroids drift as experience
accumulates; re-centered periodically (default: weekly during dreams).

Feature vector layout (150D total):
    [0:130]   130D felt tensor (inner_body[5] + inner_mind[15] +
              inner_spirit[45] + outer_body[5] + outer_mind[15] +
              outer_spirit[45])
    [130:136] 6D neuromod EMA deltas (DA, 5HT, NE, ACh, Endorphin, GABA)
              — computed over ~100-epoch window, NOT instantaneous
    [136:144] 8D cluster-membership history (last 8 emits × one-hot)
    [144:145] 1D terminal-reward recency (mean of last 5)
    [145:149] 4D sphere-clock / π-heartbeat context
    [149:150] 1D kin emotional resonance (0.0 if no kin signal yet — always
              present so post-kin-maturation we don't need a schema change)

See: titan-docs/rFP_emot_cgn_v2.md §4.1
See: titan-docs/rFP_emot_cgn_wiring_audit_20260423.md (audit finding)
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

logger = logging.getLogger("titan.emot_cluster")

FEATURE_DIM = 150
NUM_PRIMITIVES = 8
CLUSTER_HISTORY_LEN = 8
NEUMOD_EMA_WINDOW = 100  # epoch window for neuromod rate-of-change EMA

EMOT_PRIMITIVES = [
    "FLOW",           # hand-seeded anchor
    "IMPASSE_TENSION", # hand-seeded anchor
    "RESOLUTION",     # hand-seeded anchor
    "PEACE",          # emergent
    "CURIOSITY",      # emergent
    "GRIEF",          # emergent
    "WONDER",         # emergent
    "LOVE",           # emergent — Heart (Maker addition 2026-04-19)
]
EMOT_PRIMITIVE_INDEX = {p: i for i, p in enumerate(EMOT_PRIMITIVES)}


def _neutral_centroid() -> np.ndarray:
    """All-neutral (0.5 for felt-tensor, 0.0 deltas) centroid. 150D."""
    c = np.full(FEATURE_DIM, 0.5, dtype=np.float32)
    c[130:136] = 0.0  # neuromod deltas neutral
    c[136:144] = 0.0  # cluster history empty
    c[144:145] = 0.5  # reward recency neutral
    c[145:149] = 0.5  # sphere-clock neutral
    c[149:150] = 0.0  # kin resonance absent by default
    return c


def _seed_centroids() -> np.ndarray:
    """Initialize 8 centroids. Hand-seed 3 anchors with distinct signatures;
    5 emergent initialized near-neutral with small perturbation so k-means
    can differentiate them as experience arrives.

    Signatures (felt-tensor substructure indices roughly mirror perturbation
    patterns seen in existing dream clusters):
      - FLOW: inner_mind slightly elevated (positive cognitive flow),
              inner_spirit elevated, low GABA delta
      - IMPASSE_TENSION: inner_body elevated (somatic stress), NE delta +,
                         GABA delta -
      - RESOLUTION: inner_mind spike, DA delta +, ACh delta +
    """
    centroids = np.stack([_neutral_centroid() for _ in range(NUM_PRIMITIVES)])
    # FLOW (idx=0)
    centroids[0, 5:20] += 0.12      # inner_mind elevated
    centroids[0, 20:65] += 0.08     # inner_spirit elevated
    centroids[0, 130] = 0.10        # DA delta +
    centroids[0, 131] = 0.08        # 5HT delta +
    centroids[0, 135] = -0.05       # GABA delta –
    # IMPASSE_TENSION (idx=1)
    centroids[1, 0:5] += 0.15       # inner_body elevated (somatic tension)
    centroids[1, 132] = 0.12        # NE delta +
    centroids[1, 135] = 0.10        # GABA delta + (inhibited)
    centroids[1, 130] = -0.05       # DA delta –
    # RESOLUTION (idx=2)
    centroids[2, 5:20] += 0.18      # inner_mind strong spike
    centroids[2, 130] = 0.15        # DA spike (eureka)
    centroids[2, 133] = 0.12        # ACh delta + (salience)
    centroids[2, 149] = 0.0         # kin resonance neutral
    # PEACE/CURIOSITY/GRIEF/WONDER/LOVE (3..7) — small random perturbations
    # so they're distinguishable from neutral at cold-start. Real drift
    # happens via k-means updates on live data.
    rng = np.random.default_rng(seed=20260420)  # deterministic seed
    for i in range(3, NUM_PRIMITIVES):
        centroids[i] += rng.normal(0.0, 0.02, size=FEATURE_DIM).astype(np.float32)
    return centroids.astype(np.float32)


@dataclass
class EmotionCluster:
    """A single k-means cluster — represents one emotion primitive.

    Centroid drifts with experience; a "label" (proposed name) is set by
    Language Teacher + Titan's naming gate (ref rFP §7.2). Initial labels
    are the hand-seeded / emergent slot names; Titan may rename later.
    """
    primitive_id: str                   # slot id (FLOW, LOVE, …)
    label: str = ""                     # Titan's chosen name (may be same
                                        # as primitive_id or different if
                                        # Titan/LLM propose alternative)
    centroid: np.ndarray = field(default_factory=_neutral_centroid)
    n_observations: int = 0
    last_updated_ts: float = 0.0
    # Rolling mean of distances at assignment (smaller = tighter cluster)
    mean_assignment_distance: float = 0.0
    # Whether this cluster has "emerged" — i.e. accumulated enough
    # observations to be considered real (threshold = 100 per rFP)
    is_emerged: bool = False


# PERSISTENCE_BY_DESIGN: EmotionClusterer._clusters is saved as a dict and
# reloaded via indexed assignment (self._clusters[p_id] = EmotionCluster(...))
# rather than a single attribute-set. Dead-wiring's asymmetry detector reads
# this as "saved not loaded" — suppressed here.
class EmotionClusterer:
    """K-means clusterer over 150D emotion feature vectors.

    Seeded with 3 anchor centroids (FLOW, IMPASSE_TENSION, RESOLUTION) + 5
    emergent slots. Observations accumulate into the nearest cluster;
    centroids are re-centered periodically (during dream cycles).

    Persistence: cluster snapshots written to
    `data/emot_cgn/cluster_snapshots.jsonl` + live state at
    `data/emot_cgn/clusters_state.json` (atomic write).
    """

    def __init__(self, save_dir: str = "data/emot_cgn",
                 recenter_interval_s: float = 7 * 86400):
        self._save_dir = save_dir
        self._recenter_interval_s = float(recenter_interval_s)

        os.makedirs(save_dir, exist_ok=True)
        self._state_path = os.path.join(save_dir, "clusters_state.json")
        self._snapshot_path = os.path.join(save_dir, "cluster_snapshots.jsonl")

        seeds = _seed_centroids()
        self._clusters: dict[str, EmotionCluster] = {}
        for i, p in enumerate(EMOT_PRIMITIVES):
            self._clusters[p] = EmotionCluster(
                primitive_id=p,
                label=p,
                centroid=seeds[i],
                is_emerged=(i < 3),  # anchors are immediately "emerged"
            )

        self._last_recenter_ts = 0.0
        self._emergence_threshold = 100   # n_observations to emerge
        self._recent_assignments: deque = deque(maxlen=CLUSTER_HISTORY_LEN)
        self._observation_buffer: deque = deque(maxlen=2000)  # for recenter

        self._load_state()

    # ── Assignment ─────────────────────────────────────────────────

    def assign(self, feature_vec: np.ndarray) -> tuple[str, float, float]:
        """Assign a feature vector to the nearest cluster.

        Returns (primitive_id, distance, confidence). Confidence derived
        from margin vs second-nearest cluster (softmax over neg-distances).

        Failsafe: returns ('FLOW', 0.0, 0.0) on any error (FLOW as default
        since it's the most neutrally-positive anchor).
        """
        try:
            if feature_vec is None or len(feature_vec) < FEATURE_DIM:
                return ("FLOW", 0.0, 0.0)
            v = np.asarray(feature_vec[:FEATURE_DIM], dtype=np.float32)
            distances = []
            for p in EMOT_PRIMITIVES:
                c = self._clusters[p].centroid
                d = float(np.linalg.norm(v - c))
                distances.append((p, d))
            distances.sort(key=lambda x: x[1])
            best_id, best_d = distances[0]
            second_d = distances[1][1] if len(distances) > 1 else best_d + 1.0
            # Margin-based confidence: bigger margin = more confident
            margin = second_d - best_d
            confidence = float(1.0 / (1.0 + math.exp(-2.0 * margin)))
            return (best_id, best_d, confidence)
        except Exception as e:
            logger.debug("[EmotCluster] assign failed: %s", e)
            return ("FLOW", 0.0, 0.0)

    def observe(self, feature_vec: np.ndarray) -> tuple[str, float, float]:
        """Assign + accumulate for future recentering.

        Called each time EMOT-CGN sees a new felt-tensor emit. Updates
        assignment history + observation buffer. Returns assignment tuple.
        """
        p_id, dist, conf = self.assign(feature_vec)
        try:
            cluster = self._clusters[p_id]
            cluster.n_observations += 1
            cluster.last_updated_ts = time.time()
            # EMA of assignment distance
            alpha = 0.05
            cluster.mean_assignment_distance = (
                (1.0 - alpha) * cluster.mean_assignment_distance
                + alpha * dist
            )
            if (not cluster.is_emerged
                    and cluster.n_observations >= self._emergence_threshold):
                cluster.is_emerged = True
                logger.info("[EmotCluster] '%s' emerged after %d observations "
                            "(mean_dist=%.3f)",
                            p_id, cluster.n_observations,
                            cluster.mean_assignment_distance)
            self._recent_assignments.append(p_id)
            # Store every ~5th observation for recenter (reduces memory)
            if cluster.n_observations % 5 == 0:
                self._observation_buffer.append(
                    (p_id, np.asarray(feature_vec[:FEATURE_DIM],
                                      dtype=np.float32).copy()))
        except Exception as e:
            logger.debug("[EmotCluster] observe bookkeeping failed: %s", e)
        return (p_id, dist, conf)

    def get_cluster_history_onehot(self) -> np.ndarray:
        """Return 8D vector of last-CLUSTER_HISTORY_LEN assignment frequencies."""
        counts = np.zeros(NUM_PRIMITIVES, dtype=np.float32)
        for p_id in self._recent_assignments:
            idx = EMOT_PRIMITIVE_INDEX.get(p_id, -1)
            if idx >= 0:
                counts[idx] += 1.0
        denom = max(1.0, float(len(self._recent_assignments)))
        return (counts / denom).astype(np.float32)

    # ── Re-centering ───────────────────────────────────────────────

    def maybe_recenter(self, force: bool = False) -> bool:
        """Re-center cluster centroids from accumulated observations.

        Invoked during dream cycles (natural fit — Titan already clusters
        then). Uses simple k-means update: new_centroid = mean of
        observations assigned to that cluster in the last recenter window.

        Returns True if recentering happened, False if skipped.
        """
        try:
            now = time.time()
            if not force and (now - self._last_recenter_ts) < self._recenter_interval_s:
                return False
            # BUG #10 follow-up (2026-04-24): observation_buffer is in-memory
            # only (not persisted across restarts). Post-restart, it fills at
            # ~1/5 of the observation rate — at typical T1 chain-conclude rate
            # (~2/3min), reaching the normal 50-obs threshold takes ~6 hours.
            # For the FIRST recenter after a fresh start (last_recenter_ts
            # still at legacy 0.0 sentinel), lower the threshold to 10 so the
            # initial centroid unfreeze happens within ~1 hour. After that,
            # the full 50-obs threshold applies per the original design.
            min_buffer = 10 if self._last_recenter_ts == 0.0 else 50
            if len(self._observation_buffer) < min_buffer:
                return False
            by_cluster: dict[str, list] = {p: [] for p in EMOT_PRIMITIVES}
            for p_id, vec in self._observation_buffer:
                if p_id in by_cluster:
                    by_cluster[p_id].append(vec)
            updates = 0
            for p_id, vecs in by_cluster.items():
                if len(vecs) < 5:
                    continue  # too few to move confidently
                stack = np.stack(vecs).astype(np.float32)
                new_centroid = stack.mean(axis=0)
                old_centroid = self._clusters[p_id].centroid
                # Blend 70% old + 30% new to avoid violent drift
                blended = 0.7 * old_centroid + 0.3 * new_centroid
                drift = float(np.linalg.norm(blended - old_centroid))
                self._clusters[p_id].centroid = blended.astype(np.float32)
                updates += 1
                logger.debug("[EmotCluster] recentered '%s' drift=%.4f n=%d",
                             p_id, drift, len(vecs))
            self._last_recenter_ts = now
            # Snapshot + clear buffer
            self._snapshot()
            self._observation_buffer.clear()
            self.save_state()
            if updates:
                logger.info("[EmotCluster] recentered %d clusters (force=%s)",
                            updates, force)
            return True
        except Exception as e:
            logger.warning("[EmotCluster] recenter failed: %s", e)
            return False

    def seed_from_dream_clusters(self, dream_clusters: list) -> int:
        """Optionally seed initial centroids from existing dream clusters
        (rFP §7.3). Each dream_cluster is expected to be a dict with a
        'tensor' key (130D vector). We pad/truncate to 150D and assign to
        emergent slots (indices 3..7) round-robin. Returns count seeded.

        Idempotent-ish: only seeds if an emergent slot has n_observations=0.
        """
        seeded = 0
        try:
            if not dream_clusters:
                return 0
            emergent_slots = [p for p in EMOT_PRIMITIVES[3:]  # 5 emergent
                              if self._clusters[p].n_observations == 0]
            if not emergent_slots:
                return 0
            for i, dc in enumerate(dream_clusters[:len(emergent_slots)]):
                tensor = dc.get("tensor") if isinstance(dc, dict) else None
                if tensor is None or len(tensor) < 130:
                    continue
                # Build 150D padded vector
                padded = _neutral_centroid()
                padded[:130] = np.asarray(tensor[:130], dtype=np.float32)
                slot = emergent_slots[i]
                # Blend 50/50 with existing seed (don't fully overwrite)
                self._clusters[slot].centroid = (
                    0.5 * self._clusters[slot].centroid + 0.5 * padded
                ).astype(np.float32)
                seeded += 1
            if seeded:
                logger.info("[EmotCluster] seeded %d emergent slots from "
                            "dream clusters", seeded)
        except Exception as e:
            logger.warning("[EmotCluster] seed_from_dream_clusters failed: %s", e)
        return seeded

    # ── Persistence ────────────────────────────────────────────────

    def save_state(self) -> None:
        try:
            data = {
                "version": 1,
                "saved_ts": time.time(),
                "last_recenter_ts": self._last_recenter_ts,
                "clusters": {
                    p: {
                        "primitive_id": c.primitive_id,
                        "label": c.label,
                        "centroid": c.centroid.tolist(),
                        "n_observations": c.n_observations,
                        "last_updated_ts": c.last_updated_ts,
                        "mean_assignment_distance": c.mean_assignment_distance,
                        "is_emerged": c.is_emerged,
                    }
                    for p, c in self._clusters.items()
                },
                "recent_assignments": list(self._recent_assignments),
            }
            tmp = self._state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self._state_path)
        except Exception as e:
            logger.warning("[EmotCluster] save_state failed: %s", e)

    def _load_state(self) -> None:
        try:
            if not os.path.exists(self._state_path):
                return
            with open(self._state_path) as f:
                data = json.load(f)
            self._last_recenter_ts = float(data.get("last_recenter_ts", 0.0))
            for p_id, c_data in data.get("clusters", {}).items():
                if p_id not in self._clusters:
                    continue
                try:
                    centroid = np.asarray(c_data.get("centroid"),
                                          dtype=np.float32)
                    if centroid.shape[0] != FEATURE_DIM:
                        continue
                    self._clusters[p_id] = EmotionCluster(
                        primitive_id=p_id,
                        label=str(c_data.get("label", p_id)),
                        centroid=centroid,
                        n_observations=int(c_data.get("n_observations", 0)),
                        last_updated_ts=float(
                            c_data.get("last_updated_ts", 0.0)),
                        mean_assignment_distance=float(
                            c_data.get("mean_assignment_distance", 0.0)),
                        is_emerged=bool(c_data.get("is_emerged", False)),
                    )
                except Exception as _e:
                    continue
            self._recent_assignments = deque(
                data.get("recent_assignments", []),
                maxlen=CLUSTER_HISTORY_LEN)
            logger.info("[EmotCluster] Loaded cluster state from %s",
                        self._state_path)
        except Exception as e:
            logger.warning("[EmotCluster] _load_state failed: %s", e)

    def _snapshot(self) -> None:
        """Append a compact snapshot to cluster_snapshots.jsonl for audit."""
        try:
            line = {
                "ts": time.time(),
                "clusters": {
                    p: {
                        "n_obs": c.n_observations,
                        "emerged": c.is_emerged,
                        "mean_dist": round(c.mean_assignment_distance, 4),
                        # Compact centroid digest: mean of inner_spirit region
                        "spirit_mean": float(np.mean(c.centroid[20:65])),
                    }
                    for p, c in self._clusters.items()
                },
            }
            with open(self._snapshot_path, "a") as f:
                f.write(json.dumps(line) + "\n")
        except Exception as e:
            logger.debug("[EmotCluster] snapshot failed: %s", e)

    # ── Introspection ──────────────────────────────────────────────

    def get_summary(self) -> dict:
        return {
            "last_recenter_ts": self._last_recenter_ts,
            "recent_assignments": list(self._recent_assignments),
            "clusters": {
                p: {
                    "label": c.label,
                    "n_observations": c.n_observations,
                    "emerged": c.is_emerged,
                    "mean_assignment_distance": round(
                        c.mean_assignment_distance, 4),
                }
                for p, c in self._clusters.items()
            },
        }

    def get_cluster(self, primitive_id: str) -> Optional[EmotionCluster]:
        return self._clusters.get(primitive_id)

    # UNUSED_PUBLIC_API: operator-facing helper for external audit scripts
    # and future bus query handlers. Kept deliberately — accessing
    # _clusters directly would bypass the defensive copy.
    def all_clusters(self) -> dict[str, EmotionCluster]:
        return dict(self._clusters)

    def set_label(self, primitive_id: str, label: str) -> bool:
        """Rename a cluster (Titan-led naming — rFP §7.2).

        Returns True if renamed, False if primitive_id unknown. Called
        when Language Teacher + Titan converge on a name for a novel
        cluster.
        """
        if primitive_id not in self._clusters or not label:
            return False
        old = self._clusters[primitive_id].label
        self._clusters[primitive_id].label = str(label)[:40]
        logger.info("[EmotCluster] Renamed cluster '%s': '%s' → '%s'",
                    primitive_id, old, label)
        self.save_state()
        return True
