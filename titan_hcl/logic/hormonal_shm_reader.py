"""HormonalShmReader — sub-ms reader for cognitive_worker's NNS-hosted
HormonalSystem state.

Per SPEC §7.1 `nns_hormonal_state.bin` (D-SPEC-53, v1.7.4). Reads the
11×4 float32 LE matrix written by cognitive_worker (snapshot of
`neural_nervous_system._hormonal` after each NS evaluate tick — the
authoritative source for ExpressionManager.evaluate_all per the §4.B
Track 3 migration-preserving fix). Decodes back to the familiar
`{hormone_name: level}` dict shape used by composites.

Migration context — IMPORTANT. The hormonal_module-owned
`hormonal_state.bin` slot exists separately and reflects a DIFFERENT
HormonalSystem instance (hormonal_worker's, which accumulates only the
cross-worker HORMONE_STIMULUS bus stream). Pre-§4.B Track 3,
ExpressionManager.evaluate_all read from cognitive_worker's in-proc
HormonalSystem (NeuralNervousSystem._hormonal) which receives the full
env_stimuli stream from NS evaluate at neural_nervous_system.py:371.
That instance's live levels are what made composites fire at urge=0.69
within 33s of boot pre-extraction. Reading hormonal_state.bin would
read the OTHER instance with different stimulus inputs and break the
firing dynamics. Hence the new dedicated `nns_hormonal_state.bin` slot
+ this reader pointing at it.

Used by:
  - expression_worker (§4.B Track 3) — reads `level` per hormone every
    evaluate_all tick.
  - any future consumer that needs cross-process access to the NNS-
    hosted hormone-levels without spawning a HormonalSystem instance.

Cold-boot tolerant: returns empty dict when the slot is absent or
unreadable. Caller is expected to handle empty-dict gracefully (in
expression_worker this short-circuits evaluate_all).

G18-compliant — no bus dependency, no work-RPC. Sub-ms read per call.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from titan_hcl.core.state_registry import (
    NNS_HORMONAL_STATE,
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)

logger = logging.getLogger(__name__)

# Canonical hormone order — must match hormonal_worker.HORMONE_NAMES =
# NS_PROGRAMS row order. Frozen here for read-side determinism. If the
# slot schema bumps (HORMONAL_STATE_SCHEMA_VERSION), this list moves in
# lock-step with hormonal_worker.HORMONE_NAMES under the same PATCH bump
# (SPEC §3.1 D05 — per-slot schema versioning).
HORMONE_NAMES: tuple[str, ...] = (
    "REFLEX", "FOCUS", "INTUITION", "IMPULSE", "METABOLISM",      # 5 inner
    "CREATIVITY", "CURIOSITY", "EMPATHY", "REFLECTION",
    "INSPIRATION", "VIGILANCE",                                    # 6 outer
)

# Field axis indexes (within the (11, 4) ndarray) — same as
# hormonal_worker.FIELD_*.
_FIELD_LEVEL = 0
_FIELD_THRESHOLD = 1
_FIELD_REFRACTORY = 2
_FIELD_PEAK_LEVEL = 3


class HormonalShmReader:
    """Sub-ms SHM-direct reader for hormonal_state.bin (11×4 float32 LE).

    Cold-boot tolerant — returns sensible empty/zero defaults if the
    slot hasn't been written yet (hormonal_worker not yet up, or first
    run after restart).
    """

    def __init__(self, titan_id: Optional[str] = None,
                 shm_root: Optional[Path] = None) -> None:
        self._titan_id = titan_id or resolve_titan_id()
        self._shm_root: Path = shm_root or ensure_shm_root(self._titan_id)
        self._reader = StateRegistryReader(NNS_HORMONAL_STATE, self._shm_root)
        self._fallback_count = 0
        logger.debug(
            "[HormonalShmReader] initialized — titan_id=%s shm_root=%s",
            self._titan_id, self._shm_root)

    def _read_array(self) -> Optional[np.ndarray]:
        try:
            arr = self._reader.read()
        except Exception as e:
            self._fallback_count += 1
            if self._fallback_count == 1:
                logger.info(
                    "[HormonalShmReader] FIRST FALLBACK: read raised %s — "
                    "using cold defaults until hormonal_worker first publish",
                    type(e).__name__)
            return None
        if arr is None:
            self._fallback_count += 1
            return None
        if arr.shape != (len(HORMONE_NAMES), 4):
            self._fallback_count += 1
            logger.warning(
                "[HormonalShmReader] shape mismatch: expected (%d, 4), got %s",
                len(HORMONE_NAMES), arr.shape)
            return None
        return arr

    # ── Public surface ────────────────────────────────────────────────

    def get_hormone_levels(self) -> dict[str, float]:
        """Return `{hormone_name: level}` (the dict shape evaluator
        expects). Empty dict if slot absent / unreadable / mis-shaped.
        """
        arr = self._read_array()
        if arr is None:
            return {}
        return {
            name: float(arr[i, _FIELD_LEVEL])
            for i, name in enumerate(HORMONE_NAMES)
        }

    def get_hormone_state(self) -> dict[str, dict[str, float]]:
        """Return full per-hormone state dict
        `{hormone_name: {level, threshold, refractory, peak_level}}`.
        Empty dict on read failure. Mirrors
        `hormonal_worker.decode_hormonal_state(arr)`.
        """
        arr = self._read_array()
        if arr is None:
            return {}
        return {
            name: {
                "level": float(arr[i, _FIELD_LEVEL]),
                "threshold": float(arr[i, _FIELD_THRESHOLD]),
                "refractory": float(arr[i, _FIELD_REFRACTORY]),
                "peak_level": float(arr[i, _FIELD_PEAK_LEVEL]),
            }
            for i, name in enumerate(HORMONE_NAMES)
        }
