"""
life_force_inputs_publisher — owns life_force_inputs.bin SHM writer.

Phase C v1.8.3 (D-SPEC-57) per rFP_titan_hcl_l2_separation_strategy §4.G.

Invoked from cognitive_worker at every KERNEL_EPOCH_TICK (the same trigger
that previously called `life_force_engine.evaluate(...)` directly at
`cognitive_worker.py:2474` pre-Track-1-drift-retirement). Single-threaded
(G21 single-writer-per-slot, owned by cognitive_worker process).

The publisher takes a fully-built inputs dict from
`compute_life_force_inputs(state_refs)` (extracted into
`titan_hcl.logic.life_force_inputs_builder.py`), msgpack-encodes it,
and writes to `life_force_inputs.bin`. life_force_worker reads in its
KERNEL_EPOCH_TICK handler.

This mirrors the §4.Q `neuromod_inputs.bin` pattern but with a dedicated
Publisher class (matches §4.J `MetabolismStatePublisher` ergonomics).

Failure modes:
  - encode/oversize/write fails handled per-slot; first WARN with exc_info,
    subsequent throttled to every 60 ticks.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import msgpack

from titan_hcl.core.state_registry import (
    StateRegistryWriter,
    ensure_shm_root,
)
from titan_hcl.logic.life_force_inputs_specs import (
    LIFE_FORCE_INPUTS_SLOT,
    LIFE_FORCE_INPUTS_SPEC,
)

logger = logging.getLogger(__name__)


_WARN_THROTTLE_EVERY = 60
_HEARTBEAT_TICKS = (1, 10, 60, 600, 3600)


class LifeForceInputsPublisher:
    """Owns life_force_inputs.bin SHM writer (G21 single-writer = cognitive_worker)."""

    def __init__(self, titan_id: str):
        self._titan_id = titan_id
        self._shm_root = ensure_shm_root(titan_id)
        self._writer: Optional[StateRegistryWriter] = None
        self._publish_count = 0
        self._publish_success = 0
        self._encode_fails = 0
        self._oversize_fails = 0
        self._write_fails = 0
        logger.info(
            "[LifeForceInputsPublisher] initialized — titan_id=%s shm_root=%s "
            "(slot=%s — SPEC §7.1 v1.8.3 / Preamble G21; mirrors §4.Q "
            "neuromod_inputs.bin pattern)",
            titan_id, self._shm_root, LIFE_FORCE_INPUTS_SLOT)

    def _writer_attach(self) -> StateRegistryWriter:
        if self._writer is not None:
            return self._writer
        self._writer = StateRegistryWriter(LIFE_FORCE_INPUTS_SPEC, self._shm_root)
        logger.info(
            "[LifeForceInputsPublisher] writer attached — slot=%s "
            "max_bytes=%d schema_version=%d path=%s",
            LIFE_FORCE_INPUTS_SLOT, LIFE_FORCE_INPUTS_SPEC.payload_bytes,
            LIFE_FORCE_INPUTS_SPEC.schema_version,
            self._shm_root / f"{LIFE_FORCE_INPUTS_SLOT}.bin")
        return self._writer

    def publish(self, inputs: dict[str, Any]) -> None:
        """Encode + write the 16-input bridge payload.

        `inputs` must come from `compute_life_force_inputs(state_refs)` with
        the schema-v1 keys (pi_heartbeat_ratio, developmental_age,
        sovereignty_index, spirit_coherence, vocabulary_size,
        learning_rate_gain, emotional_coherence, neuromodulator_homeostasis,
        mind_coherence, expression_fire_rate, sol_balance, anchor_freshness,
        hormonal_vitality, body_coherence, topology_grounding,
        infrastructure_health). The publisher attaches schema_version + ts
        and writes.
        """
        self._publish_count += 1
        payload = dict(inputs)
        payload["schema_version"] = LIFE_FORCE_INPUTS_SPEC.schema_version
        payload["ts"] = time.time()
        self._write(payload)

        if self._publish_count in _HEARTBEAT_TICKS:
            logger.info(
                "[LifeForceInputsPublisher] heartbeat — publish_count=%d "
                "success=%d fails={encode=%d oversize=%d write=%d}",
                self._publish_count, self._publish_success,
                self._encode_fails, self._oversize_fails, self._write_fails)

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            encoded = msgpack.packb(payload, use_bin_type=True)
        except (TypeError, ValueError) as e:
            self._encode_fails += 1
            if self._encode_fails == 1 or self._encode_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[LifeForceInputsPublisher] msgpack encode failed (#%d): "
                    "%s — keys=%s",
                    self._encode_fails, e, sorted(payload.keys()),
                    exc_info=True)
            return

        if len(encoded) > LIFE_FORCE_INPUTS_SPEC.payload_bytes:
            self._oversize_fails += 1
            logger.critical(
                "[LifeForceInputsPublisher] payload %dB > MAX %dB (#%d) — "
                "slot retains last-known. Investigate upstream shape drift; "
                "do NOT silently truncate.",
                len(encoded), LIFE_FORCE_INPUTS_SPEC.payload_bytes,
                self._oversize_fails)
            return

        try:
            writer = self._writer_attach()
            writer.write_variable(encoded)
            self._publish_success += 1
            if self._publish_success == 1:
                logger.info(
                    "[LifeForceInputsPublisher] FIRST PUBLISH SUCCESS — "
                    "slot=%s payload_bytes=%d (life_force_worker can now "
                    "read inputs and run evaluate; closes the cross-process "
                    "16-input bridge per rFP §4.G + D-SPEC-57 Maker Q3 lock)",
                    LIFE_FORCE_INPUTS_SLOT, len(encoded))
        except Exception as e:
            self._write_fails += 1
            if self._write_fails == 1 or self._write_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[LifeForceInputsPublisher] shm write failed (#%d): %s",
                    self._write_fails, e, exc_info=True)

    def get_stats(self) -> dict[str, Any]:
        return {
            "publish_count": self._publish_count,
            "publish_success": self._publish_success,
            "encode_fails": self._encode_fails,
            "oversize_fails": self._oversize_fails,
            "write_fails": self._write_fails,
            "writer_attached": self._writer is not None,
        }
