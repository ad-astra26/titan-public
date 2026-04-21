"""EMOT-CGN Region Clusterer — density-based region discovery over
Titan's emotional trajectory (rFP §19, replaces v2 k-means monoculture).

PHILOSOPHY (Maker directive 2026-04-21): emotion emerges from density.
Under HDBSCAN, a region exists where Titan's state trajectory is densely
self-connected. Early in life, nothing is dense enough → everything is
NOISE → no false labels. Over time, dense regions earn region_ids
(arbitrary integers); names come later from Language Teacher translating
density centroids. No k to pre-specify, no hand-seeded primitives, no
winner-take-all.

Contrast with v2 k-means:
  - k-means: must pick k=8, forces every observation into one of 8
    bins; WONDER's random-seed perturbation lottery dominated 70-85%
    of assignments on all 3 Titans.
  - HDBSCAN: discovers cluster count from data; accepts "too-sparse
    to be a region" as NOISE; no self-reinforcing collapse.

Input: 210D "emotional state vector" (schema v2) =
    168D native consciousness (felt 130 + trajectory 2 + space 30 + neuromod 6)
  + 42D side channels (hormones 11 + NS urg 11 + CGN β 8 + MSL 6 + π 6)

Between re-clusterings (dream-cycle cadence), incoming observations are
assigned to the nearest known region centroid via k-NN (or NOISE if all
regions are farther than the HDBSCAN-determined core-distance at the
time of re-cluster). Cheap (< 1 ms per assignment).

Region IDs are STABLE across re-clusterings via centroid signature
matching — a region that persists keeps the same region_id; genuinely
new regions get fresh IDs.

Persistence: `data/emot_cgn/regions_state.json` — centroids + signatures +
core distances + residence history.

See: titan-docs/rFP_emot_cgn_v2.md §19.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .emot_bundle_protocol import (
    REGION_NOISE, REGION_UNCLUSTERED,
    FELT_TENSOR_DIM, TRAJECTORY_DIM, SPACE_TOPOLOGY_DIM, NEUROMOD_DIM,
    HORMONE_DIM, NS_URGENCY_DIM, CGN_BETA_DIM, MSL_ACT_DIM, PI_PHASE_DIM,
)

logger = logging.getLogger("titan.emot_region_clusterer")

# Dimensionality — must match concat order in assemble_state_vec().
NATIVE_CORE_DIM = (FELT_TENSOR_DIM + TRAJECTORY_DIM + SPACE_TOPOLOGY_DIM
                   + NEUROMOD_DIM)  # 130 + 2 + 30 + 6 = 168
SIDE_CHANNEL_DIM = (HORMONE_DIM + NS_URGENCY_DIM + CGN_BETA_DIM
                    + MSL_ACT_DIM + PI_PHASE_DIM)  # 11+11+8+6+6 = 42 (schema v2)
STATE_DIM = NATIVE_CORE_DIM + SIDE_CHANNEL_DIM  # 168 + 42 = 210 (was 208 in v1)


def assemble_state_vec(encoded: dict) -> np.ndarray:
    """Concat an encoder's output dict into the STATE_DIM state vector
    consumed by RegionClusterer.observe(). Order is fixed — changing
    it here without matching a stored regions_state.json breaks
    centroid comparisons across reloads, so this must stay stable.

    Schema v2 (2026-04-21): pi_phase 4D → 6D; STATE_DIM 208 → 210.
    Existing v1 regions_state.json is skipped on load by centroid-shape
    check in RegionClusterer._load_state — no data loss since v3
    shipped with 0 regions emerged.
    """
    def _as(k, n):
        v = encoded.get(k)
        if v is None:
            return np.zeros(n, dtype=np.float32)
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.size < n:
            out = np.zeros(n, dtype=np.float32)
            out[: arr.size] = arr
            return out
        return arr[:n]
    return np.concatenate([
        _as("felt_tensor_130d", FELT_TENSOR_DIM),
        _as("trajectory_2d", TRAJECTORY_DIM),
        _as("space_topology_30d", SPACE_TOPOLOGY_DIM),
        _as("neuromod_state_6d", NEUROMOD_DIM),
        _as("hormone_levels_11d", HORMONE_DIM),
        _as("ns_urgencies_11d", NS_URGENCY_DIM),
        _as("cgn_beta_states_8d", CGN_BETA_DIM),
        _as("msl_activations_6d", MSL_ACT_DIM),
        _as("pi_phase_6d", PI_PHASE_DIM),
    ], axis=0).astype(np.float32)

# HDBSCAN hyperparameters — 2 knobs total, intentionally minimal.
# min_cluster_size: smallest group of observations that can form a region.
#   Too small → noise mistaken for regions. Too large → genuine small
#   regions missed. 20 chosen to match v2's emergence_threshold=30 soft
#   intent with HDBSCAN's typical recommendation (2% of dataset).
DEFAULT_MIN_CLUSTER_SIZE = 20
# min_samples: how dense the neighborhood must be for a point to be
#   "core." Higher → stricter density requirement → more NOISE; lower
#   → more permissive. Matches HDBSCAN default when None.
DEFAULT_MIN_SAMPLES = None

# Rolling buffer size — trades memory for trajectory recency.
# At ~1 bundle/10s, 4096 ≈ 11 hours of state — covers 1-2 full dream cycles.
DEFAULT_BUFFER_SIZE = 4096


def _signature_from_centroid(centroid: np.ndarray, bucket: float = 0.05) -> int:
    """Compute a stable 64-bit signature from a cluster centroid.

    Quantizes each dim to `bucket` precision so small centroid drifts
    (expected as trajectory grows) produce the SAME signature; large
    drifts (genuinely a different region) produce a DIFFERENT signature.
    SHA-256 for collision resistance; truncated to 64 bits for bundle
    storage (`region_signature` u64 slot).
    """
    q = np.round(np.asarray(centroid, dtype=np.float32) / bucket) * bucket
    b = q.tobytes()
    h = hashlib.sha256(b).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False)


@dataclass
class Region:
    """A discovered density region in emotional state space."""
    region_id: int                          # local ID (arbitrary int, stable across reclusters via sig)
    signature: int                          # 64-bit fingerprint of centroid
    centroid: np.ndarray                    # 208D mean of assigned observations
    core_distance: float                    # typical intra-region distance
    n_observations: int = 0                 # lifetime observation count
    first_seen_ts: float = field(default_factory=time.time)
    last_seen_ts: float = field(default_factory=time.time)
    label: str = ""                         # Titan/LLM-assigned name (empty until named)


class RegionClusterer:
    """Density-based emotional region discovery with stable IDs.

    Lifecycle:
      - observe(state_vec)  → (region_id, confidence, residence_s, signature)
      - recluster()         → full HDBSCAN pass + signature re-mapping
      - save_state() / _load_state()
    """

    def __init__(self,
                 save_dir: str = "data/emot_cgn",
                 min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
                 min_samples: Optional[int] = DEFAULT_MIN_SAMPLES,
                 buffer_size: int = DEFAULT_BUFFER_SIZE):
        self._save_dir = save_dir
        self._min_cluster_size = int(min_cluster_size)
        self._min_samples = min_samples
        self._buffer: deque = deque(maxlen=buffer_size)
        self._regions: dict[int, Region] = {}  # region_id → Region
        self._next_region_id: int = 0
        self._current_region_id: int = REGION_UNCLUSTERED
        self._current_region_entered_ts: float = time.time()
        os.makedirs(save_dir, exist_ok=True)
        self._state_path = os.path.join(save_dir, "regions_state.json")
        self._load_state()

    # ── Assignment path (hot, per-bundle-write) ─────────────────────

    def observe(self, state_vec: np.ndarray) -> tuple[int, float, float, int]:
        """Append observation to buffer; assign to nearest known region.

        Returns (region_id, confidence, residence_s, signature).
        region_id: >=0 (known region), REGION_NOISE (too far), or
                   REGION_UNCLUSTERED (no regions discovered yet).
        confidence: 0..1, 1 = right at a centroid, 0 = exactly at core-dist.
        residence_s: seconds since Titan entered the current region.
        signature: 64-bit stable ID; 0 if NOISE/UNCLUSTERED.
        """
        try:
            v = np.asarray(state_vec, dtype=np.float32).reshape(-1)
            if v.size != STATE_DIM:
                # Pad or truncate defensively — bundle may drop side channels
                # if an input was None. Zero-pad keeps the geometry sensible.
                padded = np.zeros(STATE_DIM, dtype=np.float32)
                padded[: min(v.size, STATE_DIM)] = v[: STATE_DIM]
                v = padded
            self._buffer.append(v.copy())

            if not self._regions:
                self._current_region_id = REGION_UNCLUSTERED
                return (REGION_UNCLUSTERED, 0.0, 0.0, 0)

            # k-NN to nearest centroid.
            best_id = REGION_NOISE
            best_dist = float("inf")
            best_sig = 0
            best_core_dist = 1.0
            for rid, region in self._regions.items():
                d = float(np.linalg.norm(v - region.centroid))
                if d < best_dist:
                    best_dist = d
                    best_id = rid
                    best_sig = region.signature
                    best_core_dist = max(1e-6, region.core_distance)

            # Accept only if within 1.5 × core-distance; else NOISE.
            if best_dist > 1.5 * best_core_dist:
                self._current_region_id = REGION_NOISE
                return (REGION_NOISE, 0.0, 0.0, 0)

            # Confidence: 1 at centroid, 0 at 1.5× core-distance.
            confidence = float(max(0.0, min(1.0,
                1.0 - (best_dist / (1.5 * best_core_dist)))))

            # Residence tracking — only reset when region_id changes.
            now = time.time()
            if best_id != self._current_region_id:
                self._current_region_id = best_id
                self._current_region_entered_ts = now
            residence_s = float(now - self._current_region_entered_ts)

            # Update region bookkeeping.
            region = self._regions[best_id]
            region.n_observations += 1
            region.last_seen_ts = now

            return (best_id, confidence, residence_s, best_sig)
        except Exception as e:
            logger.debug("[RegionClusterer] observe failed: %s", e)
            return (REGION_UNCLUSTERED, 0.0, 0.0, 0)

    # ── Re-clustering (cold, dream-cycle cadence) ───────────────────

    def recluster(self) -> int:
        """Run HDBSCAN over buffered trajectory. Returns number of regions.

        Preserves region_id across re-runs by matching new cluster
        centroids to existing region signatures. New regions get fresh IDs.
        """
        try:
            if len(self._buffer) < max(self._min_cluster_size * 2, 50):
                return len(self._regions)
            from sklearn.cluster import HDBSCAN
            X = np.stack(list(self._buffer)).astype(np.float32)
            kwargs = {"min_cluster_size": self._min_cluster_size}
            if self._min_samples is not None:
                kwargs["min_samples"] = int(self._min_samples)
            clusterer = HDBSCAN(**kwargs)
            labels = clusterer.fit_predict(X)

            # Build new region set from labels (-1 = HDBSCAN noise, skip).
            new_regions: dict[int, Region] = {}
            existing_sigs = {r.signature: rid
                             for rid, r in self._regions.items()}
            unique_labels = sorted(set(int(l) for l in labels) - {-1})
            for lbl in unique_labels:
                mask = labels == lbl
                members = X[mask]
                centroid = members.mean(axis=0).astype(np.float32)
                # Core distance: median pairwise dist within cluster
                # (cheap stand-in for HDBSCAN's internal core-distance).
                if len(members) > 1:
                    dists = np.linalg.norm(members - centroid, axis=1)
                    core_dist = float(np.median(dists))
                else:
                    core_dist = 1.0
                sig = _signature_from_centroid(centroid)
                # Re-use existing region_id if signature matches.
                if sig in existing_sigs:
                    rid = existing_sigs[sig]
                    old = self._regions[rid]
                    new_regions[rid] = Region(
                        region_id=rid, signature=sig,
                        centroid=centroid, core_distance=core_dist,
                        n_observations=old.n_observations + int(mask.sum()),
                        first_seen_ts=old.first_seen_ts,
                        last_seen_ts=time.time(),
                        label=old.label,
                    )
                else:
                    rid = self._next_region_id
                    self._next_region_id += 1
                    new_regions[rid] = Region(
                        region_id=rid, signature=sig,
                        centroid=centroid, core_distance=core_dist,
                        n_observations=int(mask.sum()),
                    )

            self._regions = new_regions
            n_noise = int((labels == -1).sum())
            logger.info("[RegionClusterer] reclustered: %d regions from "
                        "%d observations (min_cluster_size=%d, n_noise=%d)",
                        len(new_regions), len(X), self._min_cluster_size,
                        n_noise)

            # ── A4: per-group variance scan + dead-dim WARNs + telemetry. ──
            # Every field group in the bundle has a known dim span; if any
            # group shows <1e-6 std across the whole buffer, it's almost
            # certainly a silent producer-wiring gap (rFP §23.6+ class of
            # bug). WARN at recluster cadence (once per 15 min) — not spammy.
            try:
                self._emit_recluster_telemetry(X, labels, new_regions, n_noise)
            except Exception as _tel_err:
                logger.debug("[RegionClusterer] telemetry emit failed: %s",
                             _tel_err)

            self.save_state()
            return len(new_regions)
        except Exception as e:
            logger.warning("[RegionClusterer] recluster failed: %s", e)
            return len(self._regions)

    # ── A4: dead-dim detector + recluster telemetry ─────────────────

    # Field-group layout of the 210D state vector (schema v2). Must match
    # the concat order in assemble_state_vec() — see module docstring.
    # (group_name, start, end) — end is exclusive.
    _FIELD_GROUPS = (
        ("felt_tensor",      0,                                                          FELT_TENSOR_DIM),
        ("trajectory",       FELT_TENSOR_DIM,                                            FELT_TENSOR_DIM + TRAJECTORY_DIM),
        ("space_topology",   FELT_TENSOR_DIM + TRAJECTORY_DIM,                           FELT_TENSOR_DIM + TRAJECTORY_DIM + SPACE_TOPOLOGY_DIM),
        ("neuromod_state",   FELT_TENSOR_DIM + TRAJECTORY_DIM + SPACE_TOPOLOGY_DIM,      NATIVE_CORE_DIM),
        ("hormone_levels",   NATIVE_CORE_DIM,                                            NATIVE_CORE_DIM + HORMONE_DIM),
        ("ns_urgencies",     NATIVE_CORE_DIM + HORMONE_DIM,                              NATIVE_CORE_DIM + HORMONE_DIM + NS_URGENCY_DIM),
        ("cgn_beta_states",  NATIVE_CORE_DIM + HORMONE_DIM + NS_URGENCY_DIM,             NATIVE_CORE_DIM + HORMONE_DIM + NS_URGENCY_DIM + CGN_BETA_DIM),
        ("msl_activations",  NATIVE_CORE_DIM + HORMONE_DIM + NS_URGENCY_DIM + CGN_BETA_DIM,
                             NATIVE_CORE_DIM + HORMONE_DIM + NS_URGENCY_DIM + CGN_BETA_DIM + MSL_ACT_DIM),
        ("pi_phase",         STATE_DIM - PI_PHASE_DIM,                                   STATE_DIM),
    )
    # Groups known to be unwired until the upstream producer ships.
    # Appearing here does NOT suppress the WARN — it just adds an
    # explanatory annotation so a human reading the telemetry knows
    # the dead-dim is expected (not a regression).
    _KNOWN_DEFERRED_GROUPS = frozenset(["cgn_beta_states"])

    def _emit_recluster_telemetry(self, X: np.ndarray, labels: np.ndarray,
                                  new_regions: dict, n_noise: int) -> None:
        """Compute per-group variance + pairwise-distance stats, WARN on
        zero-variance groups, append one line to recluster_telemetry.jsonl.
        Non-blocking — any failure degrades silently.
        """
        per_group = {}
        dead = []
        for name, start, end in self._FIELD_GROUPS:
            slice_ = X[:, start:end]
            std = float(slice_.std())
            per_group[name] = {
                "dim": int(end - start),
                "std": round(std, 6),
                "mean": round(float(slice_.mean()), 6),
                "nonzero_pct": round(100.0 * float((slice_ != 0).mean()), 1),
            }
            if std < 1e-6:
                dead.append(name)
                if name in self._KNOWN_DEFERRED_GROUPS:
                    logger.warning(
                        "[RegionClusterer] DEAD-DIM group=%s std=%.2e "
                        "(KNOWN deferred — producer not wired yet)",
                        name, std)
                else:
                    logger.warning(
                        "[RegionClusterer] DEAD-DIM group=%s std=%.2e "
                        "over %d samples — likely producer-wiring regression. "
                        "Inspect /v4/emot-cgn/audit.bundle_snapshot.",
                        name, std, len(X))

        # Pairwise-distance stats — high-dim curse indicator.
        # (CoV = std/mean; low CoV means all points roughly equidistant.)
        cov = 0.0
        d_mean = 0.0
        d_std = 0.0
        try:
            # Sample subset if buffer is large — pdist is O(N²).
            sample = X if len(X) <= 256 else X[
                np.random.choice(len(X), 256, replace=False)]
            from scipy.spatial.distance import pdist
            d = pdist(sample)
            d_mean = float(d.mean())
            d_std = float(d.std())
            cov = d_std / max(d_mean, 1e-6)
        except Exception:
            pass

        entry = {
            "ts": time.time(),
            "n_samples": int(len(X)),
            "n_clusters": int(len(new_regions)),
            "n_noise": int(n_noise),
            "min_cluster_size": int(self._min_cluster_size),
            "pairwise_dist": {
                "mean": round(d_mean, 4),
                "std": round(d_std, 4),
                "coefficient_of_variation": round(cov, 4),
            },
            "per_group": per_group,
            "dead_groups": dead,
        }
        try:
            path = os.path.join(self._save_dir, "recluster_telemetry.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    # ── Introspection ───────────────────────────────────────────────

    def regions_count(self) -> int:
        return len(self._regions)

    def get_region(self, region_id: int) -> Optional[Region]:
        return self._regions.get(region_id)

    # ── Persistence ─────────────────────────────────────────────────

    def save_state(self) -> None:
        try:
            data = {
                "version": 1,
                "saved_ts": time.time(),
                "next_region_id": self._next_region_id,
                "regions": {
                    str(rid): {
                        "signature": r.signature,
                        "centroid": r.centroid.tolist(),
                        "core_distance": r.core_distance,
                        "n_observations": r.n_observations,
                        "first_seen_ts": r.first_seen_ts,
                        "last_seen_ts": r.last_seen_ts,
                        "label": r.label,
                    }
                    for rid, r in self._regions.items()
                },
            }
            tmp = self._state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self._state_path)
        except Exception as e:
            logger.warning("[RegionClusterer] save_state failed: %s", e)

    def _load_state(self) -> None:
        try:
            if not os.path.exists(self._state_path):
                return
            with open(self._state_path) as f:
                data = json.load(f)
            self._next_region_id = int(data.get("next_region_id", 0))
            for rid_str, rd in data.get("regions", {}).items():
                rid = int(rid_str)
                centroid = np.asarray(rd.get("centroid"), dtype=np.float32)
                if centroid.shape[0] != STATE_DIM:
                    continue
                self._regions[rid] = Region(
                    region_id=rid,
                    signature=int(rd.get("signature", 0)),
                    centroid=centroid,
                    core_distance=float(rd.get("core_distance", 1.0)),
                    n_observations=int(rd.get("n_observations", 0)),
                    first_seen_ts=float(rd.get("first_seen_ts", time.time())),
                    last_seen_ts=float(rd.get("last_seen_ts", time.time())),
                    label=str(rd.get("label", "")),
                )
            logger.info("[RegionClusterer] loaded %d regions from %s",
                        len(self._regions), self._state_path)
        except Exception as e:
            logger.warning("[RegionClusterer] _load_state failed: %s", e)
