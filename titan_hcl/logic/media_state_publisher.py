"""
media_state_publisher — MediaStatePublisher writes media_state.bin SHM slot.

Producer for media_state slot per SPEC §7.1 (D-SPEC-71 v1.10.0). G21
single-writer contract: only studio_worker publishes here.

Closes the media.stats bus-cache state-lookup per Preamble G18.
"""
from __future__ import annotations

import time
from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.media_state_specs import (
    MEDIA_STATE_SLOT,
    MEDIA_STATE_SPEC,
)
from titan_hcl._phase_c_constants import MEDIA_STATE_SCHEMA_VERSION


class MediaStatePublisher(BaseStatePublisher):
    slot_name = MEDIA_STATE_SLOT
    slot_spec = MEDIA_STATE_SPEC

    def _compute_payload(self, studio: Any) -> dict[str, Any]:
        if studio is None:
            return self._stub()
        try:
            stats = studio.get_stats() if hasattr(studio, "get_stats") else {}
        except Exception:
            stats = {}
        return {
            "meditation_render_count": int(
                stats.get("meditation_count", 0) or 0),
            "epoch_render_count": int(stats.get("epoch_count", 0) or 0),
            "eureka_render_count": int(stats.get("eureka_count", 0) or 0),
            "last_render_ts": float(stats.get("last_render_ts", 0.0) or 0.0),
            "last_render_type": str(stats.get("last_render_type", "") or ""),
            "total_disk_mb": float(stats.get("total_disk_mb", 0.0) or 0.0),
            "nft_composite_count": int(
                stats.get("nft_composite_count", 0) or 0),
            "schema_version": MEDIA_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }

    def _stub(self) -> dict[str, Any]:
        return {
            "meditation_render_count": 0,
            "epoch_render_count": 0,
            "eureka_render_count": 0,
            "last_render_ts": 0.0,
            "last_render_type": "",
            "total_disk_mb": 0.0,
            "nft_composite_count": 0,
            "schema_version": MEDIA_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }
