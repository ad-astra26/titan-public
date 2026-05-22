"""
life_force_state_publisher — owns life_force_state.bin SHM writer.

Phase C v1.8.3 (D-SPEC-57) per rFP_titan_hcl_l2_separation_strategy §4.G.

Invoked from life_force_worker after every LifeForceEngine.evaluate (per
KERNEL_EPOCH_TICK) and from a 1Hz periodic publisher thread (republish-on-
unchanged for cold consumers). Single-threaded (G21 single-writer-per-slot).
Hot-path chi_total + metabolic_drain reads (cognitive_worker MSL static_context,
reasoning body_state, hormonal_pressure inputs, ground_up_enricher chi_overlay,
NN modulation cap) bypass the bus entirely via this slot per G18+G20.

Cold-boot safe — if LifeForceEngine instance is None or evaluate hasn't fired
yet, publish a stub payload with default chi (0.5 / HEALTHY / no contemplation)
+ cold-boot ts. Consumers treat defaults as "cold" and proceed (matches the
existing kernel.py:2118 frontend cold-boot ChiLifeForce TrinityBar placeholder
shape).

Failure modes mirror MetabolismStatePublisher precedent:
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
from titan_hcl.logic.life_force_state_specs import (
    LIFE_FORCE_STATE_SLOT,
    LIFE_FORCE_STATE_SPEC,
)

logger = logging.getLogger(__name__)


_WARN_THROTTLE_EVERY = 60
_HEARTBEAT_TICKS = (1, 10, 60, 600, 3600)


def _empty_chi_layer() -> dict[str, Any]:
    """Stub ChiLayer for cold-boot publishes (matches life_force.py shape)."""
    return {
        "raw": 0.5,
        "effective": 0.5,
        "weight": 0.33,
        "thinking": 0.5,
        "feeling": 0.5,
        "willing": 0.5,
        "components": {},
    }


def _empty_contemplation() -> dict[str, Any]:
    return {
        "active": False,
        "phase": 0,
        "conviction": 0,
        "conviction_threshold": 300,  # CONVICTION_THRESHOLD from life_force.py
        "mature_enough": False,
    }


class LifeForceStatePublisher:
    """Owns life_force_state.bin SHM writer (G21 single-writer)."""

    def __init__(self, titan_id: str):
        self._titan_id = titan_id
        self._shm_root = ensure_shm_root(titan_id)
        self._writer: Optional[StateRegistryWriter] = None
        self._publish_count = 0
        self._publish_success = 0
        self._encode_fails = 0
        self._oversize_fails = 0
        self._write_fails = 0
        self._last_published_state: Optional[str] = None
        self._last_payload_hash: Optional[int] = None
        logger.info(
            "[LifeForceStatePublisher] initialized — titan_id=%s shm_root=%s "
            "(slot=%s — SPEC §7.1 v1.8.3 / Preamble G18 + G21)",
            titan_id, self._shm_root, LIFE_FORCE_STATE_SLOT)

    def _writer_attach(self) -> StateRegistryWriter:
        if self._writer is not None:
            return self._writer
        self._writer = StateRegistryWriter(LIFE_FORCE_STATE_SPEC, self._shm_root)
        logger.info(
            "[LifeForceStatePublisher] writer attached — slot=%s "
            "max_bytes=%d schema_version=%d path=%s",
            LIFE_FORCE_STATE_SLOT, LIFE_FORCE_STATE_SPEC.payload_bytes,
            LIFE_FORCE_STATE_SPEC.schema_version,
            self._shm_root / f"{LIFE_FORCE_STATE_SLOT}.bin")
        return self._writer

    def publish(
        self,
        life_force_engine: Any,
        chi_result: Optional[dict[str, Any]] = None,
        is_dreaming: bool = False,
    ) -> None:
        """Compute payload from LifeForceEngine state + write to SHM slot.

        `life_force_engine` is the in-process LifeForceEngine held by
        life_force_worker (NOT a proxy — direct access). `chi_result` is the
        most-recent evaluate() return dict (skip recompute — engine state is
        consumed for drain + contemplation). Cold-boot safe.

        Args:
            life_force_engine: LifeForceEngine instance or None (cold-boot).
            chi_result: Most recent evaluate() output dict, or None to publish
                a stub (cold-boot, before first evaluate).
            is_dreaming: Cached dream state from DREAM_STATE_CHANGED (passed by
                worker; engine has its own _is_dreaming for drain math but the
                slot publishes this view for consumers).
        """
        self._publish_count += 1
        payload = self._compute_payload(life_force_engine, chi_result, is_dreaming)
        self._write(payload)

        if self._publish_count in _HEARTBEAT_TICKS:
            logger.info(
                "[LifeForceStatePublisher] heartbeat — publish_count=%d "
                "success=%d fails={encode=%d oversize=%d write=%d} "
                "last_state=%s",
                self._publish_count, self._publish_success,
                self._encode_fails, self._oversize_fails, self._write_fails,
                self._last_published_state)

    def _compute_payload(
        self,
        life_force_engine: Any,
        chi_result: Optional[dict[str, Any]],
        is_dreaming: bool,
    ) -> dict[str, Any]:
        now_ts = time.time()

        if chi_result is None:
            # Cold-boot stub — preserves /v4/chi frontend shape so the
            # TrinityBar doesn't crash pre-first-evaluate.
            drain = 0.0
            if life_force_engine is not None:
                try:
                    drain = float(
                        getattr(life_force_engine, "_metabolic_drain", 0.0)
                    )
                except (TypeError, ValueError):
                    drain = 0.0
            return {
                "total": 0.5,
                "spirit": _empty_chi_layer(),
                "mind": _empty_chi_layer(),
                "body": _empty_chi_layer(),
                "circulation": 0.0,
                "weights": {"spirit": 0.33, "mind": 0.34, "body": 0.33},
                "state": "BOOTSTRAP",
                "developmental_phase": "BIRTH",
                "contemplation": _empty_contemplation(),
                "metabolic_drain": drain,
                "is_dreaming": bool(is_dreaming),
                "schema_version": LIFE_FORCE_STATE_SPEC.schema_version,
                "ts": now_ts,
            }

        # Healthy path — pack chi_result + engine state.
        drain = 0.0
        if life_force_engine is not None:
            try:
                drain = float(
                    getattr(life_force_engine, "_metabolic_drain", 0.0)
                )
            except (TypeError, ValueError):
                drain = 0.0

        state = str(chi_result.get("state", "HEALTHY"))
        if state != self._last_published_state:
            self._last_published_state = state

        return {
            "total": float(chi_result.get("total", 0.5)),
            "spirit": chi_result.get("spirit", _empty_chi_layer()),
            "mind": chi_result.get("mind", _empty_chi_layer()),
            "body": chi_result.get("body", _empty_chi_layer()),
            "circulation": float(chi_result.get("circulation", 0.0)),
            "weights": chi_result.get(
                "weights", {"spirit": 0.33, "mind": 0.34, "body": 0.33}
            ),
            "state": state,
            "developmental_phase": str(
                chi_result.get("developmental_phase", "BIRTH")
            ),
            "contemplation": chi_result.get(
                "contemplation", _empty_contemplation()
            ),
            "metabolic_drain": drain,
            "is_dreaming": bool(is_dreaming),
            "schema_version": LIFE_FORCE_STATE_SPEC.schema_version,
            "ts": now_ts,
        }

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            encoded = msgpack.packb(payload, use_bin_type=True)
        except (TypeError, ValueError) as e:
            self._encode_fails += 1
            if self._encode_fails == 1 or self._encode_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[LifeForceStatePublisher] msgpack encode failed (#%d): "
                    "%s — keys=%s",
                    self._encode_fails, e, sorted(payload.keys()),
                    exc_info=True)
            return

        if len(encoded) > LIFE_FORCE_STATE_SPEC.payload_bytes:
            self._oversize_fails += 1
            logger.critical(
                "[LifeForceStatePublisher] payload %dB > MAX %dB (#%d) — "
                "slot retains last-known. Investigate upstream shape drift; "
                "do NOT silently truncate.",
                len(encoded), LIFE_FORCE_STATE_SPEC.payload_bytes,
                self._oversize_fails)
            return

        # Content-hash gate (only publish on change to avoid SHM thrash;
        # consumers still re-read via SeqLock — this is producer-side
        # efficiency).
        payload_hash = hash(encoded)
        if payload_hash == self._last_payload_hash:
            return
        self._last_payload_hash = payload_hash

        try:
            writer = self._writer_attach()
            writer.write_variable(encoded)
            self._publish_success += 1
            if self._publish_success == 1:
                logger.info(
                    "[LifeForceStatePublisher] FIRST PUBLISH SUCCESS — "
                    "slot=%s payload_bytes=%d total=%.3f state=%s "
                    "(consumers can now read; closes cognitive_worker chunk "
                    "8M.6 Track 1 drift per rFP §4.G + D-SPEC-57)",
                    LIFE_FORCE_STATE_SLOT, len(encoded),
                    payload.get("total", 0.0), payload.get("state"))
        except Exception as e:
            self._write_fails += 1
            if self._write_fails == 1 or self._write_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[LifeForceStatePublisher] shm write failed (#%d): %s",
                    self._write_fails, e, exc_info=True)

    def get_stats(self) -> dict[str, Any]:
        return {
            "publish_count": self._publish_count,
            "publish_success": self._publish_success,
            "encode_fails": self._encode_fails,
            "oversize_fails": self._oversize_fails,
            "write_fails": self._write_fails,
            "writer_attached": self._writer is not None,
            "last_published_state": self._last_published_state,
        }
