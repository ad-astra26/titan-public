"""
metabolism_state_publisher — owns metabolism_state.bin SHM writer.

Phase C v1.7.2 (D-SPEC-51) per rFP_titan_hcl_l2_separation_strategy §4.J.

Invoked from metabolism_worker's periodic publisher thread @ 1 Hz.
Single-threaded (G21 single-writer-per-slot). Hot-path tier + gates_enforced
reads (Soul NFT mint gate, memo_inscribe, dashboard /v4/metabolism/*, kernel
metabolism.get_metabolic_tier proxy) bypass the bus entirely via this slot
per G18+G20.

Cold-boot safe — if MetabolismController instance is None or a query fails,
publish a stub payload with HEALTHY tier + zeros + cold-boot ts. Consumers
treat defaults as "cold" and proceed.

Failure modes mirror MemoryStatePublisher + SocialGraphStatePublisher
precedents:
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
from titan_hcl.logic.metabolism_state_specs import (
    METABOLISM_STATE_SLOT,
    METABOLISM_STATE_SPEC,
)

logger = logging.getLogger(__name__)


_WARN_THROTTLE_EVERY = 60
_HEARTBEAT_TICKS = (1, 10, 60, 600, 3600)


class MetabolismStatePublisher:
    """Owns metabolism_state.bin SHM writer (G21 single-writer)."""

    def __init__(self, titan_id: str):
        self._titan_id = titan_id
        self._shm_root = ensure_shm_root(titan_id)
        self._writer: Optional[StateRegistryWriter] = None
        self._publish_count = 0
        self._publish_success = 0
        self._encode_fails = 0
        self._oversize_fails = 0
        self._write_fails = 0
        self._last_published_tier: Optional[str] = None
        self._last_tier_change_ts: float = 0.0
        logger.info(
            "[MetabolismStatePublisher] initialized — titan_id=%s shm_root=%s "
            "(slot=%s — SPEC §7.1 v1.7.2 / Preamble G18 + G21)",
            titan_id, self._shm_root, METABOLISM_STATE_SLOT)

    def _writer_attach(self) -> StateRegistryWriter:
        if self._writer is not None:
            return self._writer
        self._writer = StateRegistryWriter(METABOLISM_STATE_SPEC, self._shm_root)
        logger.info(
            "[MetabolismStatePublisher] writer attached — slot=%s "
            "max_bytes=%d schema_version=%d path=%s",
            METABOLISM_STATE_SLOT, METABOLISM_STATE_SPEC.payload_bytes,
            METABOLISM_STATE_SPEC.schema_version,
            self._shm_root / f"{METABOLISM_STATE_SLOT}.bin")
        return self._writer

    def publish(
        self,
        metabolism: Any,
        social_gravity_score: float = 0.0,
        last_gate_decision_reason: str = "",
    ) -> None:
        """Compute payload from MetabolismController + write to SHM slot.

        `metabolism` is the in-process MetabolismController held by
        metabolism_worker (NOT a proxy — direct access). Cold-boot safe.

        Args:
            metabolism: MetabolismController instance or None (cold-boot).
            social_gravity_score: Cached gravity score (0..1) for /status mood ribbon.
                Computed elsewhere (the math lives in get_social_gravity_score)
                because it needs SocialGraphProxy stats — the publisher takes
                the precomputed scalar to avoid double IO.
            last_gate_decision_reason: Human-readable reason of the last
                evaluate_gate decision (passed in by the worker dispatcher).
        """
        self._publish_count += 1
        payload = self._compute_payload(
            metabolism, social_gravity_score, last_gate_decision_reason
        )
        self._write(payload)

        if self._publish_count in _HEARTBEAT_TICKS:
            logger.info(
                "[MetabolismStatePublisher] heartbeat — publish_count=%d "
                "success=%d fails={encode=%d oversize=%d write=%d} "
                "last_tier=%s",
                self._publish_count, self._publish_success,
                self._encode_fails, self._oversize_fails, self._write_fails,
                self._last_published_tier)

    def _compute_payload(
        self,
        metabolism: Any,
        social_gravity_score: float,
        last_gate_decision_reason: str,
    ) -> dict[str, Any]:
        now_ts = time.time()
        if metabolism is None:
            return {
                "tier": "HEALTHY",
                "balance_pct": 1.0,
                "gates_enforced": False,
                "last_gate_decision_reason": "",
                "tier_info": {},
                "last_tier_change_ts": 0.0,
                "social_gravity_score": 0.0,
                "schema_version": METABOLISM_STATE_SPEC.schema_version,
                "ts": now_ts,
            }

        try:
            tier = str(metabolism.get_metabolic_tier())
        except Exception as e:
            logger.warning(
                "[MetabolismStatePublisher] get_metabolic_tier raised: %s",
                e, exc_info=True)
            tier = "HEALTHY"

        if tier != self._last_published_tier:
            self._last_tier_change_ts = now_ts
            self._last_published_tier = tier

        try:
            gates_enforced = bool(metabolism.get_gates_enforced())
        except Exception:
            gates_enforced = False

        try:
            balance_pct = float(metabolism._last_balance_pct)
        except Exception:
            balance_pct = 1.0

        try:
            tier_info = metabolism.get_tier_info()
            if not isinstance(tier_info, dict):
                tier_info = {}
        except Exception as e:
            logger.warning(
                "[MetabolismStatePublisher] get_tier_info raised: %s",
                e, exc_info=True)
            tier_info = {}

        # If caller didn't pass a reason, fall back to controller's getter.
        if not last_gate_decision_reason:
            try:
                last_gate_decision_reason = str(
                    metabolism.get_last_gate_decision_reason() or ""
                )
            except Exception:
                last_gate_decision_reason = ""

        return {
            "tier": tier,
            "balance_pct": balance_pct,
            "gates_enforced": gates_enforced,
            "last_gate_decision_reason": last_gate_decision_reason,
            "tier_info": tier_info,
            "last_tier_change_ts": self._last_tier_change_ts,
            "social_gravity_score": float(social_gravity_score),
            "schema_version": METABOLISM_STATE_SPEC.schema_version,
            "ts": now_ts,
        }

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            encoded = msgpack.packb(payload, use_bin_type=True)
        except (TypeError, ValueError) as e:
            self._encode_fails += 1
            if self._encode_fails == 1 or self._encode_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[MetabolismStatePublisher] msgpack encode failed (#%d): "
                    "%s — keys=%s",
                    self._encode_fails, e, sorted(payload.keys()),
                    exc_info=True)
            return

        if len(encoded) > METABOLISM_STATE_SPEC.payload_bytes:
            self._oversize_fails += 1
            logger.critical(
                "[MetabolismStatePublisher] payload %dB > MAX %dB (#%d) — "
                "slot retains last-known. Investigate upstream shape drift; "
                "do NOT silently truncate.",
                len(encoded), METABOLISM_STATE_SPEC.payload_bytes,
                self._oversize_fails)
            return

        try:
            writer = self._writer_attach()
            writer.write_variable(encoded)
            self._publish_success += 1
            if self._publish_success == 1:
                logger.info(
                    "[MetabolismStatePublisher] FIRST PUBLISH SUCCESS — "
                    "slot=%s payload_bytes=%d tier=%s "
                    "(consumers can now read; closes plugin.py:1612 inline-wire "
                    "per rFP §4.J + D-SPEC-51)",
                    METABOLISM_STATE_SLOT, len(encoded), payload.get("tier"))
        except Exception as e:
            self._write_fails += 1
            if self._write_fails == 1 or self._write_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[MetabolismStatePublisher] shm write failed (#%d): %s",
                    self._write_fails, e, exc_info=True)

    def get_stats(self) -> dict[str, Any]:
        return {
            "publish_count": self._publish_count,
            "publish_success": self._publish_success,
            "encode_fails": self._encode_fails,
            "oversize_fails": self._oversize_fails,
            "write_fails": self._write_fails,
            "writer_attached": self._writer is not None,
            "last_published_tier": self._last_published_tier,
            "last_tier_change_ts": self._last_tier_change_ts,
        }
