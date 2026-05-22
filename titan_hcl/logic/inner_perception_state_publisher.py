"""
inner_perception_state_publisher — InnerPerceptionStatePublisher writes
inner_perception_state.bin SHM slot.

Phase C dissolution (2026-05-22). G21 single-writer: only the main plugin
publishes here (InnerPerception is parent-resident hardware). Replaces the
OUTER_SOURCES_SNAPSHOT.inner_perception_stats bus delivery to mind_worker
(Preamble G18).
"""
from __future__ import annotations

import time
from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.inner_perception_state_specs import (
    INNER_PERCEPTION_STATE_SLOT,
    INNER_PERCEPTION_STATE_SPEC,
)
from titan_hcl._phase_c_constants import INNER_PERCEPTION_STATE_SCHEMA_VERSION


class InnerPerceptionStatePublisher(BaseStatePublisher):
    slot_name = INNER_PERCEPTION_STATE_SLOT
    slot_spec = INNER_PERCEPTION_STATE_SPEC

    def _compute_payload(self, inner_perception: Any) -> dict[str, Any]:
        if inner_perception is None or not hasattr(inner_perception, "get_stats"):
            return self._stub()
        try:
            stats = inner_perception.get_stats() or {}
        except Exception:
            return self._stub()
        return {
            "audio_state": stats.get("audio_state", {}) or {},
            "visual_state": stats.get("visual_state", {}) or {},
            "ambient_change": float(stats.get("ambient_change", 0.0) or 0.0),
            "last_create_ts": float(stats.get("last_create_ts", 0.0) or 0.0),
            "schema_version": INNER_PERCEPTION_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }

    def _stub(self) -> dict[str, Any]:
        return {
            "audio_state": {},
            "visual_state": {},
            "ambient_change": 0.0,
            "last_create_ts": 0.0,
            "schema_version": INNER_PERCEPTION_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }
