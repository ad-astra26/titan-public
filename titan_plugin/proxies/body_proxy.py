"""
Body Module Proxy — SHM-direct bridge to the supervised Body process.

Phase C Session 4 (rFP §4.C.3): both methods read SHM directly per
Preamble G18 (state transport is SHM, never bus). Zero sync bus.request.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import msgpack
import numpy as np

from ..bus import DivineBus
from ..core.state_registry import (
    INNER_BODY_5D,
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)
from ..guardian import Guardian
from ..logic.session4_state_specs import BODY_STATE_SLOT, BODY_STATE_SPEC

logger = logging.getLogger(__name__)


class BodyProxy:
    """SHM-direct proxy for the Body module."""

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian

        self._titan_id = resolve_titan_id()
        self._shm_root: Path = ensure_shm_root(self._titan_id)
        self._r_inner_body = StateRegistryReader(INNER_BODY_5D, self._shm_root)
        self._r_body_state = StateRegistryReader(BODY_STATE_SPEC, self._shm_root)

        self._fallback_counts: dict[str, int] = {}

        logger.info(
            "[BodyProxy] initialized SHM-direct readers — titan_id=%s "
            "shm_root=%s (inner_body_5d.bin + body_state.bin per "
            "Preamble G18)", self._titan_id, self._shm_root)

    def _track_fallback(self, slot_name: str, reason: str) -> None:
        prev = self._fallback_counts.get(slot_name, 0)
        self._fallback_counts[slot_name] = prev + 1
        if prev == 0:
            logger.info(
                "[BodyProxy] FIRST FALLBACK slot=%s reason=%s",
                slot_name, reason)

    def _read_body_state(self) -> Optional[dict]:
        try:
            raw = self._r_body_state.read_variable()
        except Exception as e:
            self._track_fallback("body_state",
                                 f"read_raised:{type(e).__name__}")
            return None
        if raw is None:
            self._track_fallback("body_state", "shm_unavailable")
            return None
        try:
            decoded = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            self._track_fallback("body_state",
                                 f"decode_raised:{type(e).__name__}")
            return None
        return decoded if isinstance(decoded, dict) else None

    def get_body_tensor(self) -> list:
        """Get the 5DT Body state tensor via SHM (inner_body_5d.bin)."""
        try:
            arr = self._r_inner_body.read()
        except Exception as e:
            self._track_fallback("inner_body_5d",
                                 f"read_raised:{type(e).__name__}")
            return [0.5] * 5
        if arr is None:
            self._track_fallback("inner_body_5d", "shm_unavailable")
            return [0.5] * 5
        try:
            tensor = arr.view(np.float32)
            if len(tensor) >= 5:
                return [float(x) for x in tensor[:5]]
        except Exception as e:
            self._track_fallback("inner_body_5d",
                                 f"decode_raised:{type(e).__name__}")
        return [0.5] * 5

    def get_body_details(self) -> dict:
        """Get detailed sensor readings via SHM (body_state.bin).

        Returns the same shape as the legacy bus.request("get_details")
        path: {tensor, details, history_size, severity_multipliers,
        focus_nudges, ...} plus extended outer_context fields populated
        by BodyStatePublisher.
        """
        decoded = self._read_body_state()
        if decoded is None:
            return {}
        # Reconstruct legacy "tensor" + "details" shape from per-dim
        # SHM payload so existing callers don't break.
        return {
            "tensor": [
                decoded.get("interoception", 0.5),
                decoded.get("proprioception", 0.5),
                decoded.get("somatosensation", 0.5),
                decoded.get("entropy", 0.5),
                decoded.get("thermal", 0.5),
            ],
            "details": decoded.get("body_details", {}),
            "body_health": decoded.get("body_health", 0.5),
            "outer_context": {
                "sol_balance": decoded.get("sol_balance", 0.0),
                "sol_norm": decoded.get("sol_norm", 0.0),
                "block_delta_norm": decoded.get("block_delta_norm", 0.0),
                "anchor_fresh": decoded.get("anchor_fresh", 0.0),
            },
            "ts": decoded.get("ts", 0.0),
        }
