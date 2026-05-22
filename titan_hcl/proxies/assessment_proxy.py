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
from pathlib import Path
from typing import Optional

import msgpack

from ..bus import DivineBus
from ..core.state_registry import (
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)
from ..logic.session3_state_specs import ASSESSMENT_STATE_SPEC

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
        # Phase C Session 4 (rFP §4.C.5) — SHM-direct reader for
        # assessment_state.bin (Session 3 publisher in agency_worker).
        self._titan_id = resolve_titan_id()
        self._shm_root: Path = ensure_shm_root(self._titan_id)
        self._r_assessment_state = StateRegistryReader(
            ASSESSMENT_STATE_SPEC, self._shm_root)
        self._fallback_counts: dict[str, int] = {}

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
        """Force-fetch via SHM read of assessment_state.bin (Session 3
        publisher). Preamble G18 — state transport is SHM, never bus.
        Replaces prior sync bus.request("assessment_stats") path.
        Diagnostics only."""
        try:
            raw = self._r_assessment_state.read_variable()
        except Exception as e:
            self._track_fallback("assessment_state",
                                 f"read_raised:{type(e).__name__}")
            return dict(self._stats_cache)
        if raw is None:
            self._track_fallback("assessment_state", "shm_unavailable")
            return dict(self._stats_cache)
        try:
            decoded = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            self._track_fallback("assessment_state",
                                 f"decode_raised:{type(e).__name__}")
            return dict(self._stats_cache)
        if isinstance(decoded, dict):
            # SHM payload uses average_score (matches assessment_state schema);
            # cache uses avg_score (legacy ASSESSMENT_STATS broadcast key).
            # Preserve cache-key conventions for downstream callers.
            mapped = dict(decoded)
            if "average_score" in mapped and "avg_score" not in mapped:
                mapped["avg_score"] = mapped["average_score"]
            self._stats_cache = mapped
        return dict(self._stats_cache)

    def _track_fallback(self, slot_name: str, reason: str) -> None:
        prev = self._fallback_counts.get(slot_name, 0)
        self._fallback_counts[slot_name] = prev + 1
        if prev == 0:
            logger.info(
                "[AssessmentProxy] FIRST FALLBACK slot=%s reason=%s",
                slot_name, reason)

    def update_cached_stats(self, payload: dict) -> None:
        """Called by parent's _agency_loop when ASSESSMENT_STATS arrives."""
        try:
            if isinstance(payload, dict):
                self._stats_cache = dict(payload)
        except Exception:
            pass
