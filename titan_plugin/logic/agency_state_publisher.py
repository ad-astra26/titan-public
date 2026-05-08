"""
agency_state_publisher — Phase C Session 3 §4.B.3.

Publishes agency_state.bin from an ``AgencyModule`` instance
(`titan_plugin/logic/agency/module.py`). Owned by agency_worker
(co-located with the AgencyModule instance).

Source: AgencyModule attributes — `_action_counter` / `_llm_calls_this_hour`
/ `_history` (deque of last 50 actions) / `_retry_history` / `_registry`
(HelperRegistry exposing `get_all_statuses()`) / `_budget_per_hour`.

Per SPEC §7.1 row, payload contains:
  { total_actions, actions_this_hour, success_rate, llm_calls_this_hour,
    helper_statuses, last_action_ts, posture_history_digest, ts }
"""
from __future__ import annotations

from typing import Any

from titan_plugin.logic.base_state_publisher import BaseStatePublisher
from titan_plugin.logic.session3_state_specs import (
    AGENCY_STATE_SLOT,
    AGENCY_STATE_SPEC,
)


class AgencyStatePublisher(BaseStatePublisher):
    slot_name = AGENCY_STATE_SLOT
    slot_spec = AGENCY_STATE_SPEC

    def _compute_payload(self, agency: Any) -> dict[str, Any]:
        import time
        if agency is None:
            return {
                "total_actions": 0,
                "actions_this_hour": 0,
                "success_rate": 0.0,
                "llm_calls_this_hour": 0,
                "budget_per_hour": 0,
                "budget_remaining": 0,
                "helper_statuses": {},
                "registered_helpers": [],
                "last_action_ts": 0.0,
                "posture_history_digest": [],
                "ts": time.time(),
            }

        # Counters (safe attribute reads)
        total_actions = int(getattr(agency, "_action_counter", 0) or 0)
        llm_calls_this_hour = int(
            getattr(agency, "_llm_calls_this_hour", 0) or 0)
        budget_per_hour = int(getattr(agency, "_budget_per_hour", 0) or 0)
        budget_remaining = max(0, budget_per_hour - llm_calls_this_hour)

        # History deque — last action timestamp + posture digest of last 50
        history = list(getattr(agency, "_history", []) or [])
        last_action_ts = 0.0
        posture_digest: list[dict[str, Any]] = []
        success_count = 0
        actions_in_hour = 0
        now = time.time()
        cutoff_hour = now - 3600
        for entry in history:
            if not isinstance(entry, dict):
                continue
            ts = float(entry.get("ts", 0) or 0)
            if ts > last_action_ts:
                last_action_ts = ts
            if ts >= cutoff_hour:
                actions_in_hour += 1
            if entry.get("success"):
                success_count += 1
            posture_digest.append({
                "posture": str(entry.get("posture", "")),
                "helper": str(entry.get("helper", "")),
                "success": bool(entry.get("success", False)),
                "ts": ts,
            })
        success_rate = (
            success_count / len(history) if history else 0.0)

        # Helper registry — names + statuses
        registered_helpers: list[str] = []
        helper_statuses: dict[str, Any] = {}
        try:
            registry = getattr(agency, "_registry", None)
            if registry is not None:
                if hasattr(registry, "list_all_names"):
                    registered_helpers = list(registry.list_all_names())
                if hasattr(registry, "get_all_statuses"):
                    raw = registry.get_all_statuses() or {}
                    helper_statuses = {
                        str(k): v for k, v in raw.items()
                    }
        except Exception:
            # Defensive — registry shape variance is tolerated
            pass

        return {
            "total_actions": total_actions,
            "actions_this_hour": actions_in_hour,
            "success_rate": round(success_rate, 4),
            "llm_calls_this_hour": llm_calls_this_hour,
            "budget_per_hour": budget_per_hour,
            "budget_remaining": budget_remaining,
            "helper_statuses": helper_statuses,
            "registered_helpers": registered_helpers,
            "last_action_ts": last_action_ts,
            "posture_history_digest": posture_digest,
            "ts": time.time(),
        }
