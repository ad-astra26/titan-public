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

    def __init__(self, titan_id: str = "T1"):
        self._titan_id = str(titan_id)

        # Running centroid for novelty proxy — cosine distance from mean
        # of recent native felt state. Cheap, gives sane values before
        # HDBSCAN produces membership probabilities. Replaced by HDBSCAN
        # membership once the region consumer is running.
        self._running_mean: Optional[np.ndarray] = None
        self._running_count: int = 0

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
        try:
            if self._running_mean is None:
                self._running_mean = core.copy()
                self._running_count = 1
                return 0.5
            c = self._running_count
            self._running_mean = (
                (c * self._running_mean + core) / (c + 1)
            ).astype(np.float32)
            self._running_count = c + 1
            if c < 10:
                return 0.5
            n1 = float(np.linalg.norm(core))
            n2 = float(np.linalg.norm(self._running_mean))
            if n1 < 1e-9 or n2 < 1e-9:
                return 0.5
            cos = float(np.dot(core, self._running_mean) / (n1 * n2))
            nov = 0.5 * (1.0 - cos)
            return float(max(0.0, min(1.0, nov)))
        except Exception:
            return 0.5
