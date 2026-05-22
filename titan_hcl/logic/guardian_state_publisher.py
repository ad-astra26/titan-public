"""
guardian_state_publisher — GuardianStatePublisher writes
guardian_state.bin SHM slot.

Producer for guardian_state slot per SPEC §7.1 (D-SPEC-71 v1.10.0). G21
single-writer contract: only guardian (Python L1 supervisor) publishes here.

Closes the guardian.status bus-cache state-lookup per Preamble G18.
"""
from __future__ import annotations

import time
from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.guardian_state_specs import (
    GUARDIAN_STATE_SLOT,
    GUARDIAN_STATE_SPEC,
)
from titan_hcl._phase_c_constants import GUARDIAN_STATE_SCHEMA_VERSION


class GuardianStatePublisher(BaseStatePublisher):
    slot_name = GUARDIAN_STATE_SLOT
    slot_spec = GUARDIAN_STATE_SPEC

    def _compute_payload(self, guardian: Any) -> dict[str, Any]:
        if guardian is None:
            return self._stub()
        try:
            status = guardian.get_status() if hasattr(
                guardian, "get_status") else {}
        except Exception:
            status = {}
        if not isinstance(status, dict):
            status = {}
        modules = {}
        modules_by_layer: dict[str, list[str]] = {}
        escalation_count = 0
        for name, info in status.items():
            if not isinstance(info, dict):
                continue
            modules[name] = {
                "state": str(info.get("state", "unknown")),
                "pid": int(info.get("pid", 0) or 0),
                "rss_mb": float(info.get("rss_mb", 0.0) or 0.0),
                "uptime": float(info.get("uptime", 0.0) or 0.0),
                "restart_count": int(info.get("restart_count", 0) or 0),
                "restarts_in_window": int(
                    info.get("restarts_in_window", 0) or 0),
                "last_heartbeat_age": float(
                    info.get("last_heartbeat_age", 0.0) or 0.0),
                "layer": str(info.get("layer", "")),
                "start_method": str(info.get("start_method", "")),
                "adopted": bool(info.get("adopted", False)),
                "adopt_ts": float(info.get("adopt_ts", 0.0) or 0.0),
            }
            layer = modules[name]["layer"]
            if layer:
                modules_by_layer.setdefault(layer, []).append(name)
            escalation_count += int(info.get("escalation_count", 0) or 0)
        return {
            "modules": modules,
            "total_modules": len(modules),
            "modules_by_layer": modules_by_layer,
            "escalation_count": escalation_count,
            "schema_version": GUARDIAN_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }

    def _stub(self) -> dict[str, Any]:
        return {
            "modules": {},
            "total_modules": 0,
            "modules_by_layer": {},
            "escalation_count": 0,
            "schema_version": GUARDIAN_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }
