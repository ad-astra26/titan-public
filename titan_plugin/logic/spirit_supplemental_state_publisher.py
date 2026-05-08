"""
spirit_supplemental_state_publisher — Phase C Session 4 §4.C.1 expansion.

Publishes spirit_supplemental_state.bin from spirit_worker's 4
non-trinity state sources:

  - filter_down_status — filter_down_v5.get_stats() output
  - meditation_health  — meditation_tracker + med_watchdog.health_snapshot()
                          (matches the /v4/meditation/health endpoint shape)
  - coordinator        — _COORD_SNAPSHOT_CACHE["data"] (background-built)
  - nervous_system     — _NS_SNAPSHOT_CACHE["data"] (background-built)

Closes the 4 spirit_proxy methods Session 1 retained sync (Maker greenlit
2026-05-07 to land all 4 in Session 4 — full G19 closure for spirit_proxy).

Schema per SPEC §7.1 spirit_supplemental_state.bin:
    {
        filter_down_status: dict,
        meditation_health: dict,
        coordinator: dict,
        nervous_system: dict,
        ts: float,
    }

Each section may be None / error-stub if the underlying source is
unavailable (cold-boot — first ~30s after worker start before background
caches populate). The publisher NEVER raises; consumers MUST tolerate
section=None.
"""
from __future__ import annotations

import time
from typing import Any

from titan_plugin.logic.base_state_publisher import BaseStatePublisher
from titan_plugin.logic.session4_state_specs import (
    SPIRIT_SUPPLEMENTAL_STATE_SLOT,
    SPIRIT_SUPPLEMENTAL_STATE_SPEC,
)


class SpiritSupplementalStatePublisher(BaseStatePublisher):
    slot_name = SPIRIT_SUPPLEMENTAL_STATE_SLOT
    slot_spec = SPIRIT_SUPPLEMENTAL_STATE_SPEC

    def _compute_payload(self, refs: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
          refs — dict with keys (any may be None at cold-boot):
            filter_down_v5             (object with .get_stats())
            config                     (dict — for publish_enabled flag)
            meditation_tracker         (dict)
            med_watchdog               (object with .health_snapshot,
                                        .expected_interval, .min_alert_hours)
            coord_snapshot_cache       (dict — global _COORD_SNAPSHOT_CACHE)
            ns_snapshot_cache          (dict — global _NS_SNAPSHOT_CACHE)
        """
        if not isinstance(refs, dict):
            refs = {}

        # ── filter_down_status ─────────────────────────────────────────
        filter_down_v5 = refs.get("filter_down_v5")
        config = refs.get("config") or {}
        try:
            v5_stats = (filter_down_v5.get_stats()
                        if filter_down_v5 is not None else None)
        except Exception as e:
            v5_stats = {"error": f"v5.get_stats failed: {e}"}
        v5_publishing = bool(
            (config or {}).get("filter_down_v5", {}).get(
                "publish_enabled", False))
        filter_down_status = {
            "v5": v5_stats,
            "v5_publishing": v5_publishing,
            "coexistence_phase": "v5_only",
        }

        # ── meditation_health ──────────────────────────────────────────
        meditation_tracker = refs.get("meditation_tracker")
        med_watchdog = refs.get("med_watchdog")
        meditation_health: dict[str, Any] = {}
        if meditation_tracker:
            meditation_health["tracker"] = {
                "count": int(meditation_tracker.get("count", 0) or 0),
                "count_since_nft": int(
                    meditation_tracker.get("count_since_nft", 0) or 0),
                "last_epoch": int(
                    meditation_tracker.get("last_epoch", 0) or 0),
                "last_ts": float(
                    meditation_tracker.get("last_ts", 0) or 0),
                "in_meditation": bool(
                    meditation_tracker.get("in_meditation", False)),
            }
        else:
            meditation_health["tracker"] = {"error": "tracker not available"}

        if med_watchdog is not None:
            try:
                meditation_health["watchdog"] = med_watchdog.health_snapshot()
            except Exception as e:
                meditation_health["watchdog"] = {
                    "error": f"snapshot error: {e}"}
        else:
            meditation_health["watchdog"] = {
                "error": "watchdog not initialized"}

        # Overdue flag — same logic as the spirit_loop handler
        meditation_health["overdue"] = False
        if (meditation_tracker is not None
                and med_watchdog is not None):
            try:
                last_ts = float(meditation_tracker.get("last_ts", 0) or 0)
                if last_ts > 0:
                    now = time.time()
                    elapsed = now - last_ts
                    expected = med_watchdog.expected_interval()
                    floor = med_watchdog.min_alert_hours * 3600.0
                    threshold = max(floor, expected)
                    if elapsed > threshold:
                        meditation_health["overdue"] = True
                        meditation_health["overdue_since_ts"] = (
                            last_ts + threshold)
                        meditation_health["overdue_elapsed_hours"] = round(
                            elapsed / 3600, 2)
            except Exception:
                pass

        # ── coordinator ────────────────────────────────────────────────
        coord_cache = refs.get("coord_snapshot_cache") or {}
        coordinator = coord_cache.get("data")
        if coordinator is None:
            coordinator = {"error": "coordinator snapshot not yet built"}

        # ── nervous_system ─────────────────────────────────────────────
        ns_cache = refs.get("ns_snapshot_cache") or {}
        nervous_system = ns_cache.get("data")
        if nervous_system is None:
            nervous_system = {"error": "ns snapshot not yet built"}

        return {
            "filter_down_status": filter_down_status,
            "meditation_health": meditation_health,
            "coordinator": coordinator,
            "nervous_system": nervous_system,
            "ts": time.time(),
        }
