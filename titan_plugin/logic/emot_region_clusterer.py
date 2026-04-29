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
    GRAD_SHADOW, GRAD_OBSERVING, GRAD_GRADUATED,
)
from titan_plugin.utils.silent_swallow import swallow_warn

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
    centroid: np.ndarray                    # 210D mean of assigned observations (schema v2)
    core_distance: float                    # typical intra-region distance
    n_observations: int = 0                 # lifetime observation count
    first_seen_ts: float = field(default_factory=time.time)
    last_seen_ts: float = field(default_factory=time.time)
    label: str = ""                         # Titan/LLM-assigned name (empty until named)
    # Phase B graduation bookkeeping (rFP §23.5): track across-recluster
    # persistence so a region that appears AND PERSISTS in ≥4 consecutive
    # reclusterings graduates the whole state machine. Reset if the region's
    # signature doesn't match for a cycle — dissolution counts as reset.
    first_seen_recluster: int = 0           # recluster counter when signature first seen
    last_seen_recluster: int = 0            # most recent recluster where signature matched


class RegionClusterer:
    """Density-based emotional region discovery with stable IDs.

    Lifecycle:
      - observe(state_vec)  → (region_id, confidence, residence_s, signature)
      - recluster()         → full HDBSCAN pass + signature re-mapping
      - save_state() / _load_state()

    Persistence (2026-04-22, rFP §23.6+):
      - regions_state.json  — discovered regions + graduation counters
      - regions_buffer.npy  — rolling trajectory buffer (STATE_DIM × N floats)
                              survives restarts so dev cycles don't wipe 50 min
                              of HDBSCAN warmup each time. Shape-validated on
                              load (schema v1→v2 drops cleanly). Age-capped at
                              BUFFER_MAX_AGE_S (7 days) — if Titan was down
                              longer, buffer may not reflect current state.
      - recluster_telemetry.jsonl — append-only (A4 dead-dim detector)
    """

    # Max age of a persisted buffer before we consider it stale on restart.
    # 7 days balances "continuity across upgrade cycles" against "don't restore
    # ancient trajectory that no longer reflects Titan's current being."
    BUFFER_MAX_AGE_S = 7 * 86400

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
        # Phase B graduation state machine (rFP §23.5). Counters persist
        # across reboots via regions_state.json so a restart doesn't reset
        # Titan's developmental clock.
        self._recluster_count: int = 0      # monotonic recluster index
        self._first_boot_ts: float = time.time()  # first-ever boot (14-day gate)
        os.makedirs(save_dir, exist_ok=True)
        self._state_path = os.path.join(save_dir, "regions_state.json")
        self._buffer_path = os.path.join(save_dir, "regions_buffer.npy")
        self._load_state()
        self._restore_buffer()

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

            # Increment recluster counter BEFORE processing so persistence
            # bookkeeping uses the current-pass index.
            self._recluster_count += 1

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
                        # Persistence bookkeeping (Phase B, rFP §23.5):
                        # signature matched prior pass → region persists.
                        # Keep original first_seen_recluster; update last_seen.
                        first_seen_recluster=old.first_seen_recluster
                            if old.first_seen_recluster > 0
                            else self._recluster_count,
                        last_seen_recluster=self._recluster_count,
                    )
                else:
                    rid = self._next_region_id
                    self._next_region_id += 1
                    new_regions[rid] = Region(
                        region_id=rid, signature=sig,
                        centroid=centroid, core_distance=core_dist,
                        n_observations=int(mask.sum()),
                        first_seen_recluster=self._recluster_count,
                        last_seen_recluster=self._recluster_count,
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
    # §23.6a shipped 2026-04-24: cgn_beta_states removed from deferred set.
    # CGN_BETA_SNAPSHOT bus message now populates cgn_beta_states_8d via
    # cgn_worker → emot_cgn_worker. Any future dead groups go here.
    _KNOWN_DEFERRED_GROUPS = frozenset()

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

    # ── Phase B graduation state machine (rFP §23.5) ────────────────

    # How many CONSECUTIVE reclusters a region's signature must persist
    # to count as "persistent" for graduation. 4 = appear in 4+ reclusters.
    # Accounts for HDBSCAN's inherent density-threshold noise —
    # genuine emotional attractors should survive ≥4 cycles of rebuilding.
    PERSISTENCE_THRESHOLD_RECLUSTERS = 4

    # Minimum regions that must be persistent AND at least one named
    # before full graduation (GRAD_GRADUATED).
    GRADUATION_MIN_PERSISTENT_REGIONS = 3

    # Observation window — emotional architecture is philosophically
    # sensitive; don't let it graduate purely from synthetic training.
    GRADUATION_MIN_AGE_S = 14 * 86400  # 14 days

    def persistent_regions(self) -> list[Region]:
        """Regions whose signature has appeared in ≥PERSISTENCE_THRESHOLD
        consecutive reclusters AND was last seen in the most recent pass
        (i.e. hasn't dissolved). Basis for the graduation state machine.

        Note: `last_seen_recluster == self._recluster_count` means the
        signature survived the latest HDBSCAN run. A region that appeared
        once then dissolved (like T1's 2026-04-22 05:05 first emergence)
        will drop out of this list on the next pass.
        """
        out = []
        for r in self._regions.values():
            if r.last_seen_recluster != self._recluster_count:
                continue
            consecutive = (r.last_seen_recluster - r.first_seen_recluster + 1)
            if consecutive >= self.PERSISTENCE_THRESHOLD_RECLUSTERS:
                out.append(r)
        return out

    def graduation_status(self) -> int:
        """Compute the emotional graduation status per rFP §23.5.

        Returns one of:
          GRAD_SHADOW (0)    — no persistent regions yet; observations only
          GRAD_OBSERVING (1) — ≥1 persistent region; state attractors
                               confirmed but not yet granted consumer authority
          GRAD_GRADUATED (2) — ≥3 persistent regions AND observation window
                               ≥14 days AND at least one region has a
                               Titan/LLM-assigned name (empty label blocks
                               graduation — §23.4 prerequisite)

        The state can regress: if a region dissolves and drops the persistent
        count below the threshold, status falls back to a lower stage on the
        next recluster. No hysteresis in v1; add if observed thrashing.
        """
        persistent = self.persistent_regions()
        if not persistent:
            return GRAD_SHADOW

        enough_regions = (len(persistent) >=
                          self.GRADUATION_MIN_PERSISTENT_REGIONS)
        age_s = time.time() - self._first_boot_ts
        old_enough = age_s >= self.GRADUATION_MIN_AGE_S
        any_named = any(bool(r.label) for r in persistent)

        if enough_regions and old_enough and any_named:
            return GRAD_GRADUATED
        return GRAD_OBSERVING

    def graduation_snapshot(self) -> dict:
        """Full graduation telemetry for /v4/emot-cgn/audit. Captures the
        exact gate state so UIs and operators can see why we haven't
        graduated yet ("2/3 persistent, age 9/14d, 0/1 named")."""
        persistent = self.persistent_regions()
        age_s = time.time() - self._first_boot_ts
        return {
            "status": self.graduation_status(),
            "recluster_count": self._recluster_count,
            "age_seconds": round(age_s, 1),
            "age_days": round(age_s / 86400, 2),
            "persistent_regions": len(persistent),
            "total_regions": len(self._regions),
            "named_regions": sum(1 for r in persistent if r.label),
            "gates": {
                "persistent_min": self.GRADUATION_MIN_PERSISTENT_REGIONS,
                "persistence_threshold_reclusters":
                    self.PERSISTENCE_THRESHOLD_RECLUSTERS,
                "age_gate_s": self.GRADUATION_MIN_AGE_S,
            },
            "blocking": self._graduation_blockers(persistent, age_s),
        }

    def _graduation_blockers(
        self, persistent: list[Region], age_s: float) -> list[str]:
        """Human-readable list of what's still blocking GRAD_GRADUATED.
        Empty list means nothing — ready to graduate."""
        blockers = []
        if len(persistent) < self.GRADUATION_MIN_PERSISTENT_REGIONS:
            blockers.append(
                f"need {self.GRADUATION_MIN_PERSISTENT_REGIONS} "
                f"persistent regions (have {len(persistent)})")
        if age_s < self.GRADUATION_MIN_AGE_S:
            days_short = (self.GRADUATION_MIN_AGE_S - age_s) / 86400
            blockers.append(f"observation window: {days_short:.1f} days remaining")
        if not any(bool(r.label) for r in persistent):
            blockers.append("need ≥1 Titan-named region (§23.4 LLM naming)")
        return blockers

    # ── Persistence ─────────────────────────────────────────────────

    def save_state(self) -> None:
        try:
            data = {
                # Bumped to 3 for _current_region_id + _current_region_entered_ts
                # persistence (audit 2026-04-23 BUG #7). v1/v2 state loads
                # compatibly (missing fields → REGION_UNCLUSTERED defaults).
                "version": 3,
                "saved_ts": time.time(),
                "next_region_id": self._next_region_id,
                "recluster_count": self._recluster_count,
                "first_boot_ts": self._first_boot_ts,
                "current_region_id": self._current_region_id,
                "current_region_entered_ts": self._current_region_entered_ts,
                "regions": {
                    str(rid): {
                        "signature": r.signature,
                        "centroid": r.centroid.tolist(),
                        "core_distance": r.core_distance,
                        "n_observations": r.n_observations,
                        "first_seen_ts": r.first_seen_ts,
                        "last_seen_ts": r.last_seen_ts,
                        "label": r.label,
                        "first_seen_recluster": r.first_seen_recluster,
                        "last_seen_recluster": r.last_seen_recluster,
                    }
                    for rid, r in self._regions.items()
                },
            }
            tmp = self._state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self._state_path)
            # Persist the rolling trajectory buffer alongside regions so
            # restarts don't wipe ~50 min of HDBSCAN warmup (rFP §23.6+,
            # 2026-04-22). Non-fatal — buffer save failure doesn't block
            # the JSON write which is the authoritative state.
            self._save_buffer()
        except Exception as e:
            logger.warning("[RegionClusterer] save_state failed: %s", e)

    def _save_buffer(self) -> None:
        """Persist the rolling 210D trajectory buffer as .npy alongside
        regions_state.json. Atomic write via .tmp + os.replace. Called
        from save_state() on every recluster + shutdown save.
        """
        if not self._buffer:
            # Empty buffer — remove any stale file so we don't accidentally
            # restore outdated data on next boot.
            try:
                if os.path.exists(self._buffer_path):
                    os.remove(self._buffer_path)
            except Exception:
                pass
            return
        try:
            arr = np.stack(list(self._buffer)).astype(np.float32)
            tmp = self._buffer_path + ".tmp"
            np.save(tmp, arr, allow_pickle=False)
            # np.save adds .npy suffix if not present — account for that.
            if not tmp.endswith(".npy"):
                actual_tmp = tmp + ".npy"
            else:
                actual_tmp = tmp
            os.replace(actual_tmp, self._buffer_path)
        except Exception as e:
            swallow_warn('[RegionClusterer] _save_buffer failed', e,
                         key="logic.emot_region_clusterer.save_buffer_failed", throttle=100)

    def _restore_buffer(self) -> None:
        """Load persisted trajectory buffer if present and valid. Three
        safety checks: file exists, age ≤ BUFFER_MAX_AGE_S, shape matches
        current STATE_DIM. On any failure, start with empty buffer — the
        system degrades to pre-persistence behavior cleanly.
        """
        try:
            if not os.path.exists(self._buffer_path):
                return
            age_s = time.time() - os.path.getmtime(self._buffer_path)
            if age_s > self.BUFFER_MAX_AGE_S:
                logger.info(
                    "[RegionClusterer] buffer file is %.1f days old — "
                    "discarding (>%.1f-day max age)",
                    age_s / 86400, self.BUFFER_MAX_AGE_S / 86400)
                return
            arr = np.load(self._buffer_path, allow_pickle=False)
            if arr.ndim != 2 or arr.shape[1] != STATE_DIM:
                logger.info(
                    "[RegionClusterer] buffer shape mismatch %s vs "
                    "STATE_DIM=%d — discarding (likely schema change, e.g. "
                    "v1→v2 pi_phase 4D→6D expansion)",
                    arr.shape, STATE_DIM)
                return
            for row in arr:
                self._buffer.append(row.astype(np.float32).copy())
            logger.info(
                "[RegionClusterer] restored %d buffer observations "
                "(age=%.1f min, avoids ~%.0f min of HDBSCAN warmup)",
                len(self._buffer), age_s / 60,
                min(len(self._buffer), 50) * 1.0)  # ~1 obs/min cadence
        except Exception as e:
            logger.warning("[RegionClusterer] _restore_buffer failed: %s", e)

    def _load_state(self) -> None:
        try:
            if not os.path.exists(self._state_path):
                return
            with open(self._state_path) as f:
                data = json.load(f)
            self._next_region_id = int(data.get("next_region_id", 0))
            # Phase B graduation fields (missing on pre-v2 state → keep
            # sane defaults so fresh installs don't "skip" their clock).
            self._recluster_count = int(data.get("recluster_count", 0))
            self._first_boot_ts = float(data.get("first_boot_ts", time.time()))
            # Current-region residence restore (audit 2026-04-23 BUG #7).
            # Pre-v3 state files won't have these keys → defaults preserve
            # pre-fix behavior (residence counter resets). Region id is
            # validated against loaded regions below.
            _restored_current_rid = int(
                data.get("current_region_id", REGION_UNCLUSTERED))
            _restored_entered_ts = float(
                data.get("current_region_entered_ts", time.time()))
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
                    first_seen_recluster=int(rd.get("first_seen_recluster", 0)),
                    last_seen_recluster=int(rd.get("last_seen_recluster", 0)),
                )
            # Restore current-region residence ONLY if the saved region
            # still exists post-load. Otherwise fall back to UNCLUSTERED —
            # prevents claiming residence in a region that was dissolved
            # by HDBSCAN between save and load. Sentinel values (NOISE,
            # UNCLUSTERED) restore unchanged.
            if (_restored_current_rid in self._regions
                    or _restored_current_rid in (REGION_NOISE,
                                                 REGION_UNCLUSTERED)):
                self._current_region_id = _restored_current_rid
                self._current_region_entered_ts = _restored_entered_ts
            logger.info("[RegionClusterer] loaded %d regions from %s "
                        "(recluster_count=%d, age=%.1fh, "
                        "current_region=%d)",
                        len(self._regions), self._state_path,
                        self._recluster_count,
                        (time.time() - self._first_boot_ts) / 3600,
                        self._current_region_id)
        except Exception as e:
            logger.warning("[RegionClusterer] _load_state failed: %s", e)
