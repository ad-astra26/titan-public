"""
Assessment Proxy — bus-routed bridge to agency_worker (L3 §A.8.6).

Drop-in interface match for `SelfAssessment`. When flag
`microkernel.a8_agency_subprocess_enabled=true`, parent's
`_agency_assessment` becomes this proxy; assess() calls translate to a
bus QUERY/RESPONSE round-trip against the agency_worker (which owns the
SelfAssessment instance + LLM scoring fn).

ASYNC-FRIENDLY: assess() runs via `await self._bus.request_async(...)`
— event loop stays unblocked during the assessment LLM call. request_async
routes through the dedicated bus_ipc_pool (RCA 2026-04-29) — isolates
this latency-sensitive RPC from the default 64-worker pool serving
Observatory snapshots.

Cached stats (total, avg_score, recent) are refreshed by parent's
_agency_loop on ASSESSMENT_STATS broadcast (every 60s).

Hard-fail return on timeout: assess() returns a neutral assessment
(score=0.5, threshold_direction="hold", reflection="proxy_unavailable")
so parent's _handle_impulse can still publish ACTION_RESULT without
stalling.

See: titan-docs/rFP_microkernel_phase_a8_l2_l3_residency_completion.md §A.8.6
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from ..bus import DivineBus

logger = logging.getLogger(__name__)


def _neutral(action_result: dict, reason: str) -> dict:
    """Used when bus.request fails / times out — same shape as
    SelfAssessment.assess() return so parent's _handle_impulse can
    blindly destructure assessment["score"], assessment["reflection"],
    etc. without exception."""
    return {
        "action_id": int(action_result.get("action_id", 0) or 0),
        "impulse_id": int(action_result.get("impulse_id", 0) or 0),
        "score": 0.5,
        "reflection": f"proxy_neutral: {reason}",
        "enrichment": {},
        "mood_delta": 0.0,
        "threshold_direction": "hold",
        "ts": time.time(),
    }


class AssessmentProxy:
    """Bus-routed proxy that mirrors SelfAssessment's public surface.

    Methods (mirrors SelfAssessment):
        assess(action_result: dict) → assessment dict   [async]
        get_stats() → cached stats dict                 [sync]
    """

    def __init__(self, bus: DivineBus, request_timeout_s: float = 30.0):
        self._bus = bus
        self._timeout = float(request_timeout_s)
        self._reply_queue = bus.subscribe("assessment_proxy", reply_only=True)
        self._stats_cache: dict = {
            "total": 0,
            "avg_score": 0.0,
            "recent": [],
        }

    async def assess(self, action_result: dict) -> dict:
        """Bus IPC reply via dedicated bus_ipc_pool — isolated from the
        default 64-worker pool (RCA 2026-04-29). Returns assessment dict
        (same shape as SelfAssessment.assess); neutral on timeout / worker
        error."""
        payload = {"action": "assess", "action_result": action_result}
        try:
            reply = await self._bus.request_async(
                "assessment_proxy", "agency_worker", payload, self._timeout,
                self._reply_queue,
            )
        except Exception as e:
            logger.warning("[AssessmentProxy] bus.request raised: %s", e)
            return _neutral(action_result, f"bus_raise: {e}")
        if reply is None:
            logger.warning("[AssessmentProxy] assess timeout — neutral")
            return _neutral(action_result, "timeout")
        body = reply.get("payload") or {}
        if "error" in body:
            logger.warning("[AssessmentProxy] worker error: %s", body["error"])
            return _neutral(action_result, str(body["error"]))
        result = body.get("assessment")
        if not isinstance(result, dict):
            return _neutral(action_result, "malformed_response")
        return result

    def get_stats(self) -> dict:
        return dict(self._stats_cache)

    def refresh_stats(self) -> dict:
        """Force-fetch via synchronous bus.request — diagnostics only."""
        try:
            reply = self._bus.request(
                src="assessment_proxy", dst="agency_worker",
                payload={"action": "assessment_stats"},
                timeout=10.0, reply_queue=self._reply_queue,
            )
        except Exception as e:
            logger.warning("[AssessmentProxy] sync request raised: %s", e)
            return dict(self._stats_cache)
        if reply is None:
            return dict(self._stats_cache)
        body = reply.get("payload") or {}
        if "stats" in body and isinstance(body["stats"], dict):
            self._stats_cache = dict(body["stats"])
        return dict(self._stats_cache)

    def update_cached_stats(self, payload: dict) -> None:
        """Called by parent's _agency_loop when ASSESSMENT_STATS arrives."""
        try:
            if isinstance(payload, dict):
                self._stats_cache = dict(payload)
        except Exception:
            pass
