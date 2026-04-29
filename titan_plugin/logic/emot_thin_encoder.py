"""EMOT-CGN thin encoder — assembles Titan's native state for the bundle
(rFP §19 bridge encoder).

PHILOSOPHY: NO PROJECTION TODAY. Titan's native Trinity dimensions
(130D felt + 2D trajectory + 30D space topology + 6D neuromod = 168D
consciousness) are the bundle's primary representation. The "thin
encoder" is pure assembly — it gathers inputs from callers and packs
them losslessly. HDBSCAN runs over the native representation; nothing
is thrown away before density discovery.

This matches Maker's architectural principle: Titan IS a 130+2+30D
entity. Squeezing him through a 96D bottleneck just to satisfy an L5
convention meant for NS action heads would be false compression.

L5 forward compatibility
------------------------
When rFP_titan_unified_learning_v1.md Phase 0 ships its trained AIF
trunk/inner/outer encoder, a new encoder class populates the
3×32D z_trunk/z_inner/z_outer slots IN ADDITION to native fields
(not instead of). `encoder_id` advances from ENCODER_THIN_ASSEMBLY=0
to ENCODER_L5_PHASE0=1. Consumers that want the compressed view read
z_*; consumers that want the full state read native fields. Bundle
bytes unchanged; no schema bump.

Inputs (all optional; missing → zeros):
    felt_tensor_130d       — 130D felt (IB+IM+IS + OB+OM+OS)
    trajectory_2d          — 2D journey (T3+T4)
    space_topology_30d     — 30D space/chi context
    neuromod_state_6d      — DA, 5HT, NE, ACh, Endorphin, GABA
    hormone_levels_11d     — per-NS-program hormonal accumulators
    ns_urgencies_11d       — per-NS-program urgency
    cgn_beta_states_8d     — per-CGN-consumer dominant V
    msl_activations_6d     — I, YOU, ME, WE, YES, NO
    pi_phase_6d            — sphere-clock phases (6 — Trinity × Inner/Outer)

Outputs (dict ready for BundleWriter.write()):
    All native + side-channel fields, passed through unchanged
    (coerced + padded to expected dims).
    valence    f32    — Russell circumplex [-1,+1] (reward-EMA based)
    arousal    f32    — Russell circumplex [-1,+1] (neuromod-based)
    novelty    f32    — inverse-density proxy [0,1] (cosine-to-centroid)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .emot_bundle_protocol import (
    FELT_TENSOR_DIM, TRAJECTORY_DIM, SPACE_TOPOLOGY_DIM, NEUROMOD_DIM,
    HORMONE_DIM, NS_URGENCY_DIM, CGN_BETA_DIM, MSL_ACT_DIM, PI_PHASE_DIM,
    ENCODER_THIN_ASSEMBLY,
)

logger = logging.getLogger(__name__)


class ThinEmotEncoder:
    """Pure-assembly encoder — no projection, no training, no cold-start.

    Packs Titan's native state into the bundle dict shape. Also computes
    derived scalar projections (valence / arousal / novelty) from simple
    reward/neuromod/centroid-distance proxies. All cheap (< 0.1 ms).

    When L5 Phase 0 ships, a new `L5PhaseZeroEncoder` class with the
    same `encode(...)` signature will ALSO populate z_trunk/z_inner/
    z_outer — the worker swaps the instance, the bundle writer is
    unchanged.
    """

    # Novelty tier-1 constants (rFP §19.3 novelty proxy, v2 formulation 2026-04-22):
    # the original cosine-to-running-mean flat-lined near 0 because the running
    # mean had no decay and cosine on mostly-positive vectors is compressed to
    # 0.98-1.00. Tier-1 replaces it with an EMA-based rescaled deviation that
    # self-calibrates to Titan's observed variability. Tier-2 (future) replaces
    # this with HDBSCAN-membership distance once ≥1 region is persistent.
    NOVELTY_EMA_MEAN_ALPHA = 0.05    # ~20-observation half-life for trajectory baseline
    NOVELTY_EMA_DEV_ALPHA = 0.02     # ~50-observation half-life for deviation baseline
    NOVELTY_WARMUP_OBS = 10          # first N observations return 0.5 (neutral)
    NOVELTY_FLOOR_DEV = 0.01         # prevent tiny-variance amplification when state is very still

    def __init__(self, titan_id: str = "T1"):
        self._titan_id = str(titan_id)

        # Novelty proxy — EMA-based rescaled deviation. Cheap, self-calibrating,
        # produces real dynamic range (0.0-1.0) even before HDBSCAN regions
        # graduate. Replaced by region-membership distance at Tier-2.
        self._ema_mean: Optional[np.ndarray] = None    # exponential moving average of core 168D state
        self._ema_dev: float = 0.1                      # EMA of |core - ema_mean| (scalar, normalized by sqrt(dim))
        self._running_count: int = 0                    # observation count (for warmup gate)

        # EMA of recent terminal rewards → valence proxy.
        self._reward_ema: float = 0.5
        self._reward_ema_alpha: float = 0.05

    def encoder_id(self) -> int:
        return ENCODER_THIN_ASSEMBLY

    def encode(
        self,
        *,
        felt_tensor_130d=None,
        trajectory_2d=None,
        space_topology_30d=None,
        neuromod_state_6d=None,
        hormone_levels_11d=None,
        ns_urgencies_11d=None,
        cgn_beta_states_8d=None,
        msl_activations_6d=None,
        pi_phase_6d=None,
        last_terminal_reward: Optional[float] = None,
    ) -> dict:
        """Assemble native state + compute derived scalars.

        Returns a dict suitable as kwargs to BundleWriter.write()
        (plus valence/arousal/novelty keys).
        """
        def _vec(x, n: int) -> np.ndarray:
            if x is None:
                return np.zeros(n, dtype=np.float32)
            arr = np.asarray(x, dtype=np.float32).reshape(-1)
            if arr.size < n:
                out = np.zeros(n, dtype=np.float32)
                out[:arr.size] = arr
                return out
            return arr[:n]

        felt = _vec(felt_tensor_130d, FELT_TENSOR_DIM)
        traj = _vec(trajectory_2d, TRAJECTORY_DIM)
        space = _vec(space_topology_30d, SPACE_TOPOLOGY_DIM)
        neuromod = _vec(neuromod_state_6d, NEUROMOD_DIM)
        hormones = _vec(hormone_levels_11d, HORMONE_DIM)
        ns_urg = _vec(ns_urgencies_11d, NS_URGENCY_DIM)
        cgn_b = _vec(cgn_beta_states_8d, CGN_BETA_DIM)
        msl_act = _vec(msl_activations_6d, MSL_ACT_DIM)
        pi_ph = _vec(pi_phase_6d, PI_PHASE_DIM)

        # Valence: reward EMA remapped from [0,1] to [-1,+1].
        if last_terminal_reward is not None:
            r = float(max(0.0, min(1.0, last_terminal_reward)))
            self._reward_ema = (
                (1.0 - self._reward_ema_alpha) * self._reward_ema
                + self._reward_ema_alpha * r
            )
        valence = float(max(-1.0, min(1.0, 2.0 * (self._reward_ema - 0.5))))

        # Arousal: mean of DA (idx 0) + NE (idx 2), remapped [0,1]→[-1,+1].
        if neuromod.size >= 3:
            nm_arousal = float((neuromod[0] + neuromod[2]) * 0.5)
        else:
            nm_arousal = 0.5
        arousal = float(max(-1.0, min(1.0, 2.0 * nm_arousal - 1.0)))

        # Novelty proxy: cosine distance from running centroid over the
        # CORE consciousness state (native 168D). Warm-up: <10 obs → 0.5.
        core = np.concatenate([felt, traj, space, neuromod], axis=0)
        novelty = self._update_novelty(core)

        return {
            "felt_tensor_130d": felt,
            "trajectory_2d": traj,
            "space_topology_30d": space,
            "neuromod_state_6d": neuromod,
            "hormone_levels_11d": hormones,
            "ns_urgencies_11d": ns_urg,
            "cgn_beta_states_8d": cgn_b,
            "msl_activations_6d": msl_act,
            "pi_phase_6d": pi_ph,
            "valence": valence,
            "arousal": arousal,
            "novelty": novelty,
        }

    def _update_novelty(self, core: np.ndarray) -> float:
        """EMA-based rescaled-deviation novelty (rFP §19.3, Tier-1 2026-04-22).

        Returns a value in [0, 1] where:
          - 0.0  Titan's current state exactly matches his recent trajectory
                (static state, low interest)
          - 0.5  current deviation ≈ typical deviation (baseline variability)
          - 1.0  deviation is ≥2× the typical magnitude (a genuinely surprising
                moment — a persona session firing, a kin exchange landing, an
                ARC puzzle shifting Titan into novel state space)

        Why EMA and not cumulative mean:
        Original v1 implementation averaged ALL past observations without
        decay. After ~100 observations the mean went rigid and cosine
        similarity on mostly-positive 168D vectors stayed >0.99 → novelty
        compressed to <0.01 regardless of actual state variation. Tier-1
        uses two EMAs — one for the trajectory baseline (α=0.05, ~20-obs
        half-life), one for typical deviation (α=0.02, ~50-obs half-life
        so single events still pop through). This yields a self-calibrating
        novelty that responds to genuine state shifts.

        Tier-2 (future, once HDBSCAN regions persist): replace with
        distance-to-nearest-centroid normalized by core_distance. Until
        then, this proxy provides real dynamic range during the soak.
        """
        try:
            import math as _m

            # First call — initialize and return neutral 0.5.
            if self._ema_mean is None:
                self._ema_mean = core.copy()
                self._running_count = 1
                return 0.5

            # Step 1: compute current deviation from EMA baseline BEFORE updating.
            # Normalize by sqrt(dim) so the magnitude is comparable across
            # different core sizes (if we ever change the schema).
            delta = core - self._ema_mean
            current_dev = float(np.linalg.norm(delta)) / _m.sqrt(max(1, len(core)))

            # Step 2: compute novelty using the OLD dev baseline. Floor the
            # baseline so a genuinely still state doesn't amplify tiny
            # numerical noise into fake novelty.
            self._running_count += 1
            if self._running_count < self.NOVELTY_WARMUP_OBS:
                novelty = 0.5
            else:
                typical_dev = max(self._ema_dev, self.NOVELTY_FLOOR_DEV)
                # current ≈ typical → novelty ≈ 0.5
                # current ≈ 2×typical → novelty ≈ 1.0 (saturates)
                # current ≈ 0 → novelty ≈ 0.0 (state matches baseline exactly)
                novelty = float(min(1.0, current_dev / (2.0 * typical_dev)))

            # Step 3: update both EMAs for next call.
            a_mean = self.NOVELTY_EMA_MEAN_ALPHA
            a_dev = self.NOVELTY_EMA_DEV_ALPHA
            self._ema_mean = (
                (1.0 - a_mean) * self._ema_mean + a_mean * core
            ).astype(np.float32)
            self._ema_dev = (1.0 - a_dev) * self._ema_dev + a_dev * current_dev

            return float(max(0.0, min(1.0, novelty)))
        except Exception:
            return 0.5
