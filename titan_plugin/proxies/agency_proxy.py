"""
Agency Proxy — bus-routed bridge to agency_worker (L3 §A.8.6).

Drop-in interface match for `AgencyModule`. When flag
`microkernel.a8_agency_subprocess_enabled=true`, parent's
`_agency` becomes this proxy; calls translate to bus
QUERY/RESPONSE round-trips against the agency_worker.

When flag is false (default), parent retains a local AgencyModule
instance — no proxy is created, no behavior change.

ASYNC-FRIENDLY (vs A.8.3's sync OutputVerifierProxy): handle_intent +
dispatch_from_nervous_signals run via `await self._bus.request_async(...)`
so the parent's event loop is NEVER blocked during the worker's LLM
round-trip (which can take 1-30 seconds). request_async routes through
the dedicated bus_ipc_pool (RCA 2026-04-29) — isolates this LLM-time
RPC wait from the default 64-worker asyncio pool that serves Observatory
snapshots, so saturation bursts can't queue this behind heavy work.
This is the architectural unblock that lets shadow-swap adoption +
general bus traffic flow even while agency is mid-LLM-call.

Cached stats (action_count, llm_calls_this_hour, recent_actions etc.)
are refreshed by parent's _agency_loop on AGENCY_STATS broadcast (every
60s from worker). Dashboard /v3/agency reads return immediately from
the cache without a per-call bus round-trip.

Hard-fail return on timeout: handle_intent → None (Agency-skipped
semantics, parent gracefully advances), dispatch_from_nervous_signals
→ [] (no actions), get_stats → cached snapshot (or empty dict).

See: titan-docs/rFP_microkernel_phase_a8_l2_l3_residency_completion.md §A.8.6
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from ..bus import DivineBus

logger = logging.getLogger(__name__)


class AgencyProxy:
    """Bus-routed proxy that mirrors AgencyModule's public surface.

    Methods (mirrors AgencyModule):
        handle_intent(intent: dict) → action_result dict | None    [async]
        dispatch_from_nervous_signals(outer_signals, trinity_snapshot) → list  [async]
        get_stats() → cached stats dict                            [sync]

    Attributes (cached, updated by AGENCY_STATS broadcast):
        action_count, llm_calls_this_hour, budget_per_hour,
        budget_remaining, registered_helpers, helper_statuses,
        recent_actions

    Compatibility shim:
        The current AgencyModule exposes a private `_registry` attribute
        that core/plugin.py:_handle_impulse reads
        (`self._agency._registry.list_helper_names()`). When proxy mode is
        on, helpers live in the worker — `_registry` is a thin facade that
        returns cached helper names from the most recent AGENCY_READY /
        AGENCY_STATS broadcast.
    """

    def __init__(self, bus: DivineBus, request_timeout_s: float = 60.0):
        self._bus = bus
        self._timeout = float(request_timeout_s)
        # Bus reply queue used by bus.request() for RESPONSE matching.
        # reply_only=True → broadcast messages don't leak in.
        self._reply_queue = bus.subscribe("agency_proxy", reply_only=True)
        # Cached stats — populated from AGENCY_STATS broadcast via
        # plugin._agency_loop.update_cached_stats(...).
        self._stats_cache: dict = {
            "action_count": 0,
            "llm_calls_this_hour": 0,
            "budget_per_hour": 0,
            "budget_remaining": 0,
            "registered_helpers": [],
            "helper_statuses": {},
            "recent_actions": 0,
        }
        # Compat shim — see _registry property below.
        self._registry = _RegistryFacade(self)

    # ── Hot path (async) ───────────────────────────────────────────

    async def handle_intent(self, intent: dict) -> Optional[dict]:
        """Async-friendly bus.request — yields to event loop while worker
        runs the LLM call + helper.execute(). Returns the action_result
        dict (same shape as AgencyModule._build_result) or None when the
        worker reports Agency skipped (no helpers / budget exhausted /
        no_suitable_helper)."""
        payload = {"action": "handle_intent", "intent": intent}
        body = await self._await_response(payload, timeout=self._timeout)
        if body is None:
            logger.warning("[AgencyProxy] handle_intent timeout — proxy_unavailable")
            return None
        if "error" in body:
            logger.warning("[AgencyProxy] handle_intent worker error: %s", body["error"])
            return None
        return body.get("action_result")

    async def dispatch_from_nervous_signals(
        self,
        outer_signals: list[dict],
        trinity_snapshot: Optional[dict] = None,
    ) -> list[dict]:
        """Async-friendly mirror of AgencyModule.dispatch_from_nervous_signals.
        Returns list of action_result dicts (one per executed signal); empty
        list on timeout or worker-side error."""
        payload = {
            "action": "dispatch_from_nervous_signals",
            "outer_signals": list(outer_signals or []),
            "trinity_snapshot": trinity_snapshot or {},
        }
        # Longer timeout — multiple helpers in sequence
        body = await self._await_response(payload, timeout=self._timeout * 1.5)
        if body is None:
            logger.warning("[AgencyProxy] dispatch_from_nervous_signals timeout")
            return []
        if "error" in body:
            logger.warning("[AgencyProxy] dispatch worker error: %s", body["error"])
            return []
        return list(body.get("action_results") or [])

    # ── Cold path (cached, sync) ───────────────────────────────────

    def get_stats(self) -> dict:
        """Cached stats — refreshed by AGENCY_STATS broadcast (60s).
        Dashboard /v3/agency hits this — never a bus round-trip on the
        request path."""
        return dict(self._stats_cache)

    def refresh_stats(self) -> dict:
        """Force-fetch fresh stats via synchronous bus.request. NOT for
        hot-path use — only diagnostics / explicit refresh."""
        body = self._sync_request({"action": "agency_stats"}, timeout=10.0)
        if body and "stats" in body:
            self._stats_cache = dict(body["stats"])
        return dict(self._stats_cache)

    def update_cached_stats(self, payload: dict) -> None:
        """Called by parent's _agency_loop when AGENCY_STATS arrives."""
        try:
            if isinstance(payload, dict):
                self._stats_cache = dict(payload)
        except Exception:
            pass

    # ── Bus plumbing ───────────────────────────────────────────────

    async def _await_response(self, payload: dict, timeout: float) -> Optional[dict]:
        """Bus IPC reply via dedicated bus_ipc_pool — isolated from the
        default 64-worker pool so Observatory snapshot bursts can't queue
        this latency-sensitive RPC behind heavy work (RCA 2026-04-29)."""
        try:
            reply = await self._bus.request_async(
                "agency_proxy", "agency_worker", payload, timeout, self._reply_queue,
            )
        except Exception as e:
            logger.warning("[AgencyProxy] bus.request raised: %s", e)
            return None
        if reply is None:
            return None
        return reply.get("payload") or {}

    def _sync_request(self, payload: dict, timeout: float = 10.0) -> Optional[dict]:
        """Synchronous bus.request for cold-path stats refresh."""
        try:
            reply = self._bus.request(
                src="agency_proxy", dst="agency_worker",
                payload=payload, timeout=timeout, reply_queue=self._reply_queue,
            )
        except Exception as e:
            logger.warning("[AgencyProxy] sync request raised: %s", e)
            return None
        if reply is None:
            return None
        return reply.get("payload") or {}


class _RegistryFacade:
    """Thin compat facade — replicates the subset of HelperRegistry that
    parent code reads on the proxy path.

    Today's only callsite (core/plugin.py:1527-1528):
        available = self._agency._registry.list_helper_names() \\
            if hasattr(self._agency._registry, 'list_helper_names') else []

    We satisfy that single read by returning the cached helper-names list
    populated from AGENCY_READY + AGENCY_STATS broadcasts. Any other
    accessor falls through to an empty result (no AttributeError — proxy
    mode degrades gracefully)."""

    def __init__(self, proxy: "AgencyProxy"):
        self._proxy = proxy

    def list_helper_names(self) -> list[str]:
        return list(self._proxy._stats_cache.get("registered_helpers", []) or [])

    def list_all_names(self) -> list[str]:
        return list(self._proxy._stats_cache.get("registered_helpers", []) or [])

    def get_all_statuses(self) -> dict:
        return dict(self._proxy._stats_cache.get("helper_statuses", {}) or {})

    def get_helper(self, name: str) -> Any:
        # Helpers live in the worker — parent never has a real instance.
        return None
