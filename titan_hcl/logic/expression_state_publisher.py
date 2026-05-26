"""
expression_state_publisher — Sprint 7 §4.6 closure of
rFP_phase_c_130d_rust_l1_port.

Publishes expression_state.bin once per second (G18 SHM state transport)
so the spirit_worker subprocess can read sovereignty + expression
intensity + posture-authenticity stats without sync bus.request.

Closes the inner_spirit SAT[2]/CHIT[28]/ANANDA[8] + outer_spirit
SAT[1]/ANANDA[11] feed: the SHM slot now carries the exact dict shape
that `_expression_intensity()` (logic/spirit_tensor.py) expects, plus
ExpressionTranslator.get_stats() output, plus the new
posture_authenticity_ratio_30 tracker.

Payload schema (msgpack):
  {
    # ExpressionTranslator.get_stats() output
    "sovereignty_ratio":              float,    # learned/(learned+llm)
    "learned_actions":                int,
    "llm_actions":                    int,
    "total_actions":                  int,
    "top_mappings":                   list[dict],
    "total_learned_pairs":            int,
    "posture_authenticity_ratio_30":  float,    # SAT[1] outer_spirit

    # ExpressionManager.get_stats() output (when manager attached)
    "intensity":                      float,    # mean composite urge/threshold
    "composites": {                              # CHIT[13] inner via _expression_intensity
      "<name>": {"urge": float, "threshold": float, "fire_count": int, ...},
    },

    "ts":                             float,    # publish wall_ns at write
  }

Owner (G21 single-writer): main plugin only. Consumers attach
`StateRegistryReader` against the shared EXPRESSION_STATE_SPEC from
`expression_state_specs`.

Cadence (G20 hot-path safety): 1 Hz. Compute is bounded (single
ExpressionTranslator.get_stats() call + optional ExpressionManager
get_stats() composites copy).
"""
from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Optional

import msgpack

from titan_hcl.core.state_registry import (
    StateRegistryWriter,
    ensure_shm_root,
)
from titan_hcl.logic.expression_state_specs import (
    EXPRESSION_STATE_SLOT,
    EXPRESSION_STATE_SPEC,
)

logger = logging.getLogger(__name__)


_HEARTBEAT_TICKS = (1, 10, 60, 600, 3600)


class ExpressionStatePublisher:
    """
    Owns expression_state.bin SHM writer; called from main plugin's
    periodic loop @ 1 Hz. Single-threaded (G21).
    """

    def __init__(self, titan_id: Optional[str] = None) -> None:
        self._titan_id = titan_id
        self._shm_root = ensure_shm_root(titan_id)
        self._writer: Optional[StateRegistryWriter] = None
        self._publish_count = 0
        self._publish_success = 0
        self._encode_fails = 0
        self._oversize_fails = 0
        self._write_fails = 0
        logger.info(
            "[ExpressionStatePublisher] initialized — titan_id=%s "
            "shm_root=%s (slot=%s — SPEC §7.1 / Preamble G18)",
            titan_id, self._shm_root, EXPRESSION_STATE_SLOT)

    def _writer_attach(self) -> StateRegistryWriter:
        if self._writer is not None:
            return self._writer
        self._writer = StateRegistryWriter(
            EXPRESSION_STATE_SPEC, self._shm_root)
        logger.info(
            "[ExpressionStatePublisher] writer attached — slot=%s "
            "max_bytes=%d schema_version=%d path=%s",
            EXPRESSION_STATE_SLOT,
            EXPRESSION_STATE_SPEC.payload_bytes,
            EXPRESSION_STATE_SPEC.schema_version,
            self._shm_root / f"{EXPRESSION_STATE_SLOT}.bin")
        return self._writer

    def publish(
        self,
        translator: Any,
        manager: Optional[Any] = None,
        translator_stats: Optional[dict] = None,
    ) -> None:
        """
        Compute payload from translator + (optional) manager and write
        to expression_state.bin. Cold-boot safe — if translator is None
        or missing expected methods, publish stub payload.

        L3 housekeeping closure 2026-05-26: ``translator_stats`` is the
        cross-process alternative for callers that don't hold the live
        translator object (i.e. expression_worker under
        ``l0_rust_enabled=true``). When supplied, it overrides the
        translator-call path — the publisher uses the snapshot directly
        instead of invoking translator.get_stats(). Expected shape:
        ``{"sovereignty_ratio", "learned_actions", "llm_actions",
        "total_actions", "top_mappings", "total_learned_pairs",
        "posture_authenticity_ratio_30"}``. Missing keys default to
        their stub values — partial snapshots are safe.
        """
        self._publish_count += 1
        payload = self._compute_payload(translator, manager, translator_stats)
        self._write(payload)

        if self._publish_count in _HEARTBEAT_TICKS:
            logger.info(
                "[ExpressionStatePublisher] heartbeat — publish_count=%d "
                "success=%d fails={encode=%d oversize=%d write=%d}",
                self._publish_count, self._publish_success,
                self._encode_fails, self._oversize_fails, self._write_fails)

    def _compute_payload(
        self,
        translator: Any,
        manager: Optional[Any],
        translator_stats: Optional[dict] = None,
    ) -> dict:
        # L3 housekeeping (2026-05-26): cross-process snapshot path —
        # expression_worker under l0_rust_enabled=true receives stats
        # via EXPRESSION_TRANSLATOR_STATS_UPDATED bus event (emitted
        # by main plugin's translator owner) and passes them here
        # instead of holding the translator object directly.
        if translator_stats is not None:
            tstats = dict(translator_stats)
            par30 = float(tstats.pop(
                "posture_authenticity_ratio_30", 0.0) or 0.0)
        elif translator is not None:
            try:
                tstats = translator.get_stats()
            except Exception:
                logger.warning(
                    "[ExpressionStatePublisher] translator.get_stats() "
                    "raised — publishing stub:\n%s",
                    traceback.format_exc())
                tstats = {}
            # Posture-authenticity tracker (SAT[1] expressive_authenticity).
            try:
                par30 = float(translator.posture_authenticity_ratio_30())
            except (AttributeError, TypeError, ValueError):
                par30 = 0.0
        else:
            tstats = {}
            par30 = 0.0

        # ExpressionManager.get_stats() — composites for inner spirit
        # _expression_intensity (CHIT[13] / CHIT[28] depending on block).
        intensity = 0.0
        composites: dict = {}
        if manager is not None:
            try:
                mstats = manager.get_stats() or {}
                # Some manager implementations return a top-level
                # `intensity` field; others derive from composites.
                intensity = float(mstats.get("intensity", 0.0))
                raw_composites = mstats.get("composites") or {}
                if isinstance(raw_composites, dict):
                    # Normalize to {name: {urge, threshold, fire_count}}
                    # so the consumer code path (Python + Rust) sees a
                    # stable shape regardless of manager-internal extras.
                    for name, c in raw_composites.items():
                        if not isinstance(c, dict):
                            continue
                        composites[str(name)] = {
                            "urge": float(c.get("urge", c.get(
                                "last_urge", 0.0)) or 0.0),
                            "threshold": float(c.get("threshold", 0.0) or 0.0),
                            "fire_count": int(c.get("fire_count", 0) or 0),
                        }
            except Exception:
                logger.warning(
                    "[ExpressionStatePublisher] manager.get_stats() raised "
                    "— composites omitted:\n%s",
                    traceback.format_exc())

        return {
            "sovereignty_ratio": float(tstats.get("sovereignty_ratio", 0.0)),
            "learned_actions": int(tstats.get("learned_actions", 0)),
            "llm_actions": int(tstats.get("llm_actions", 0)),
            "total_actions": int(tstats.get("total_actions", 0)),
            "top_mappings": tstats.get("top_mappings") or [],
            "total_learned_pairs": int(tstats.get("total_learned_pairs", 0)),
            "posture_authenticity_ratio_30": par30,
            "intensity": intensity,
            "composites": composites,
            "ts": time.time(),
        }

    def _write(self, payload: dict) -> None:
        try:
            encoded = msgpack.packb(payload, use_bin_type=True)
        except Exception:
            self._encode_fails += 1
            if self._encode_fails <= 3 or self._encode_fails % 60 == 0:
                logger.warning(
                    "[ExpressionStatePublisher] encode failure "
                    "(count=%d):\n%s",
                    self._encode_fails, traceback.format_exc())
            return

        if len(encoded) > EXPRESSION_STATE_SPEC.payload_bytes:
            self._oversize_fails += 1
            if self._oversize_fails <= 3 or self._oversize_fails % 60 == 0:
                logger.warning(
                    "[ExpressionStatePublisher] payload %dB exceeds cap %dB "
                    "(count=%d) — top_mappings/composites may need trim",
                    len(encoded), EXPRESSION_STATE_SPEC.payload_bytes,
                    self._oversize_fails)
            return

        try:
            writer = self._writer_attach()
            writer.write_variable(encoded)
            self._publish_success += 1
        except Exception:
            self._write_fails += 1
            if self._write_fails <= 3 or self._write_fails % 60 == 0:
                logger.warning(
                    "[ExpressionStatePublisher] write failure "
                    "(count=%d):\n%s",
                    self._write_fails, traceback.format_exc())

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
