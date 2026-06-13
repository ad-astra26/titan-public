"""
consciousness_age_reader — sub-µs SHM reader for the emergent epoch counter.

RFP_synthesis_self_learning_meta_reasoning §7.D-knowledge DK.3 (M0, pinned
2026-06-13). The DK.3 wiki-lint TTL/decay + the M1 volatile-prune gate need
Titan's *emergent* age in epochs — the FAST cognitive self-observation tick
(``consciousness_age.bin :: age_epochs``, ~10k/day → ~417/hr), NOT the slow
``unified_spirit_metadata.epoch_count`` GreatEpoch (~5min cycles) which is the
wrong scale for a ~1hr TTL.

Mirrors ``LifeForceShmReader`` (proxies/life_force_proxy.py) exactly — a
``StateRegistryReader`` over a per-Titan SHM slot, no bus dependency, G18-clean
(state transport is SHM, never DB/RPC). Cold-boot tolerant: returns 0 until the
cognitive_worker producer first publishes the slot, so a missing/uninitialized
slot is the grandfather signal (age_epochs==0 → never TTL'd) rather than a crash.

Producer: cognitive_worker (G21 single-writer) via ``ConsciousnessAgePublisher``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import msgpack

from titan_hcl.core.state_registry import (
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)
from titan_hcl.logic.consciousness_age_state_specs import (
    CONSCIOUSNESS_AGE_SLOT,
    CONSCIOUSNESS_AGE_SPEC,
)

logger = logging.getLogger(__name__)


class ConsciousnessAgeReader:
    """Sub-µs SHM-direct reader for ``consciousness_age.bin::age_epochs``.

    No bus dependency — only a per-Titan SHM root path is needed. Cold-boot
    tolerant: returns 0 if the slot hasn't been written yet (producer not up,
    or first run after restart). A 0 reading is the M0 *grandfather* signal —
    a finding/concept whose age cannot be computed is never TTL-pruned.
    """

    def __init__(self, titan_id: Optional[str] = None,
                 shm_root: Optional[Path] = None) -> None:
        self._titan_id = titan_id or resolve_titan_id()
        self._shm_root: Path = shm_root or ensure_shm_root(self._titan_id)
        self._r_state = StateRegistryReader(
            CONSCIOUSNESS_AGE_SPEC, self._shm_root)
        self._fallback_count = 0
        logger.debug(
            "[ConsciousnessAgeReader] initialized — titan_id=%s shm_root=%s "
            "slot=%s", self._titan_id, self._shm_root, CONSCIOUSNESS_AGE_SLOT)

    def get_age_epochs(self) -> int:
        """Lifetime emergent epoch count (int). 0 on cold-boot / read failure
        (the grandfather signal — never raises)."""
        try:
            raw = self._r_state.read_variable()
        except Exception as e:  # noqa: BLE001 — cold-boot / slot-absent tolerant
            self._fallback_count += 1
            if self._fallback_count == 1:
                logger.info(
                    "[ConsciousnessAgeReader] FIRST FALLBACK: read raised %s — "
                    "returning 0 (grandfather) until producer first publish",
                    type(e).__name__)
            return 0
        if raw is None:
            self._fallback_count += 1
            return 0
        try:
            decoded = msgpack.unpackb(raw, raw=False)
        except Exception as e:  # noqa: BLE001
            self._fallback_count += 1
            logger.warning("[ConsciousnessAgeReader] msgpack decode raised: %s", e)
            return 0
        if not isinstance(decoded, dict):
            return 0
        try:
            return int(decoded.get("age_epochs", 0) or 0)
        except (TypeError, ValueError):
            return 0


__all__ = ("ConsciousnessAgeReader",)
