"""
interface_advisor_publisher — owns interface_advisor_state.bin SHM writer.

Phase C v1.8.5 (D-SPEC-59) per `rFP_titan_hcl_l2_separation_strategy.md §4.H`.
Maker greenlit 2026-05-15 inline (SHM-rate-oracle pattern).

Invoked from interface_advisor_worker on every IMPULSE_RECEIVED arrival,
rate-throttled to INTERFACE_ADVISOR_RATE_REFRESH_CADENCE_S = 0.1s (10Hz cap)
to avoid SHM thrash under burst. Single-threaded (G21 single-writer).

Hot-path `current_rate` reads from parent `_handle_impulse` bypass the bus
entirely via this slot per G18+G20 (sub-µs SHM-direct via the companion
InterfaceAdvisorStateReader with 100ms cache).

Cold-boot safe — first publish before any IMPULSE_RECEIVED arrives uses
defaults (rates={}, limits=INITIAL_LIMITS snapshot, rate_limit_count=0).
Readers see the cold values and proceed (zero rates → all checks pass).

Failure modes mirror DreamStatePublisher precedent:
  - encode/oversize/write fails handled per-publish; first WARN with
    exc_info, subsequent throttled to every 60 publishes.
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
from titan_hcl.logic.interface_advisor import InterfaceAdvisor
from titan_hcl.logic.interface_advisor_specs import (
    INTERFACE_ADVISOR_STATE_SLOT,
    INTERFACE_ADVISOR_STATE_SPEC,
)

logger = logging.getLogger(__name__)


_WARN_THROTTLE_EVERY = 60
_HEARTBEAT_TICKS = (1, 10, 60, 600, 3600)


class InterfaceAdvisorStatePublisher:
    """Owns interface_advisor_state.bin SHM writer (G21 single-writer).

    Composes an InterfaceAdvisor instance — every IMPULSE_RECEIVED arrival
    routes through `record_and_publish(msg_type, source)`, which:
      1. Calls advisor.check(msg_type, source) — advisor records the
         timestamp in its sliding-window deque + returns RATE_LIMIT
         feedback dict (or None if within limits)
      2. Encodes current rates snapshot + writes SHM
      3. Returns the RATE_LIMIT feedback dict to caller (None if OK)

    The caller (interface_advisor_worker) decides whether to emit a
    RATE_LIMIT bus event back to the source based on the returned
    feedback.
    """

    def __init__(self, titan_id: str,
                 advisor: Optional[InterfaceAdvisor] = None):
        self._titan_id = titan_id
        self._shm_root = ensure_shm_root(titan_id)
        self._writer: Optional[StateRegistryWriter] = None
        self._advisor = advisor or InterfaceAdvisor()
        self._publish_count = 0
        self._publish_success = 0
        self._encode_fails = 0
        self._oversize_fails = 0
        self._write_fails = 0
        logger.info(
            "[InterfaceAdvisorStatePublisher] initialized — titan_id=%s "
            "shm_root=%s (slot=%s — SPEC §7.1 v1.8.5 / Preamble G18 + G21 / "
            "D-SPEC-59)",
            titan_id, self._shm_root, INTERFACE_ADVISOR_STATE_SLOT)

    @property
    def advisor(self) -> InterfaceAdvisor:
        """Expose the underlying InterfaceAdvisor for set_limit / reset /
        get_stats (used by config-reload paths + status endpoints that
        still need the full advisor surface)."""
        return self._advisor

    def _writer_attach(self) -> StateRegistryWriter:
        if self._writer is not None:
            return self._writer
        self._writer = StateRegistryWriter(
            INTERFACE_ADVISOR_STATE_SPEC, self._shm_root)
        logger.info(
            "[InterfaceAdvisorStatePublisher] writer attached — slot=%s "
            "max_bytes=%d schema_version=%d path=%s",
            INTERFACE_ADVISOR_STATE_SLOT,
            INTERFACE_ADVISOR_STATE_SPEC.payload_bytes,
            INTERFACE_ADVISOR_STATE_SPEC.schema_version,
            self._shm_root / f"{INTERFACE_ADVISOR_STATE_SLOT}.bin")
        return self._writer

    def record_and_publish(self, msg_type: str,
                           source: str = "") -> Optional[dict]:
        """Atomic record + publish path called from interface_advisor_worker
        on IMPULSE_RECEIVED.

        Args:
            msg_type: the bus message type being rate-checked (matches
                      InterfaceAdvisor.check signature)
            source:   source module name (for feedback routing)

        Returns:
            None if within limits (caller does nothing extra), or a
            RATE_LIMIT feedback dict if exceeded (caller emits
            RATE_LIMIT bus event with this payload to `source`).
        """
        feedback = self._advisor.check(msg_type, source)
        self.publish()
        return feedback

    def publish(self) -> None:
        """Encode current advisor state + write SHM. Idempotent under
        rate-refresh-cadence throttling (caller decides cadence)."""
        self._publish_count += 1
        payload = self._compute_payload()
        self._write(payload)

        if self._publish_count in _HEARTBEAT_TICKS:
            logger.info(
                "[InterfaceAdvisorStatePublisher] heartbeat — publish_count=%d "
                "success=%d fails={encode=%d oversize=%d write=%d} "
                "active_types=%d rate_limit_count=%d",
                self._publish_count, self._publish_success,
                self._encode_fails, self._oversize_fails, self._write_fails,
                len(payload.get("rates", {})),
                int(payload.get("rate_limit_count", 0)))

    def _compute_payload(self) -> dict[str, Any]:
        """Build msgpack payload from the advisor's live state. Calls
        `get_stats()` which lazily prunes expired window entries — so the
        snapshot is fresh-as-of-call."""
        stats = self._advisor.get_stats()
        return {
            "rates": dict(stats.get("current_rates", {})),
            "limits": dict(stats.get("limits", {})),
            "window_s": float(stats.get("window_seconds", 60.0)),
            "rate_limit_count": int(stats.get("rate_limit_count", 0)),
            "schema_version": INTERFACE_ADVISOR_STATE_SPEC.schema_version,
            "ts": time.time(),
        }

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            encoded = msgpack.packb(payload, use_bin_type=True)
        except (TypeError, ValueError) as e:
            self._encode_fails += 1
            if (self._encode_fails == 1
                    or self._encode_fails % _WARN_THROTTLE_EVERY == 0):
                logger.warning(
                    "[InterfaceAdvisorStatePublisher] msgpack encode failed "
                    "(#%d): %s — keys=%s",
                    self._encode_fails, e, sorted(payload.keys()),
                    exc_info=True)
            return

        if len(encoded) > INTERFACE_ADVISOR_STATE_SPEC.payload_bytes:
            self._oversize_fails += 1
            logger.critical(
                "[InterfaceAdvisorStatePublisher] payload %dB > MAX %dB "
                "(#%d) — slot retains last-known. Investigate upstream "
                "shape drift; do NOT silently truncate.",
                len(encoded), INTERFACE_ADVISOR_STATE_SPEC.payload_bytes,
                self._oversize_fails)
            return

        try:
            writer = self._writer_attach()
            writer.write_variable(encoded)
            self._publish_success += 1
            if self._publish_success == 1:
                logger.info(
                    "[InterfaceAdvisorStatePublisher] FIRST PUBLISH SUCCESS "
                    "— slot=%s payload_bytes=%d active_types=%d "
                    "(consumers can now read; closes pre-Phase-C in-code "
                    "ADR at plugin.py:1970-1984 per D-SPEC-59)",
                    INTERFACE_ADVISOR_STATE_SLOT, len(encoded),
                    len(payload.get("rates", {})))
        except Exception as e:
            self._write_fails += 1
            if (self._write_fails == 1
                    or self._write_fails % _WARN_THROTTLE_EVERY == 0):
                logger.warning(
                    "[InterfaceAdvisorStatePublisher] writer.write_variable "
                    "failed (#%d): %s",
                    self._write_fails, e, exc_info=True)
