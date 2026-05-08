"""
TimeChain Proxy — bus bridge to TimeChain v2 Consumer API.

Phase C Session 4 (rFP §4.C.16): the 4 query methods (recall/check/
compare/aggregate) are PARAMETERIZED SQL QUERIES — true work-RPC, not
state lookup. Migrated from sync bus.request to async bus.request_async
per Preamble G19 (≤5s timeout). Allowlist entry in
phase_c_rpc_exemptions.yaml.

The diagnostic refresh_stats() reads timechain_state.bin (Session 3 SHM
slot owned by timechain_worker) — pure state lookup, SHM-direct.

Used by: pre-prompt enrichment, API endpoints, any module needing
to QUERY cognitive memory without direct TimeChain access.
"""
import logging
from pathlib import Path
from typing import Optional

import msgpack

from ..bus import DivineBus
from ..core.state_registry import (
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)
from ..logic.session3_state_specs import TIMECHAIN_STATE_SPEC

logger = logging.getLogger(__name__)


class TimechainProxy:
    """
    Consumer API proxy for TimeChain v2.

    Query methods are async work-RPC (parameterized SQL — runs in
    worker). State lookup (refresh_stats) is SHM-direct.
    """

    def __init__(self, bus: DivineBus, guardian=None):
        self._bus = bus
        self._guardian = guardian
        self._reply_queue = bus.subscribe("timechain_proxy", reply_only=True)

        # Phase C Session 4 (rFP §4.C.16) — SHM-direct reader for
        # timechain_state.bin (Session 3 publisher).
        self._titan_id = resolve_titan_id()
        self._shm_root: Path = ensure_shm_root(self._titan_id)
        self._r_timechain_state = StateRegistryReader(
            TIMECHAIN_STATE_SPEC, self._shm_root)
        self._stats_cache: dict = {}
        self._fallback_counts: dict[str, int] = {}

    # ── Work-RPC methods (async, ≤5s — true work, not state lookup) ─

    async def recall(self, fork: str = "", source: str = "",
                     tag_contains: str = "", since_hours: float = 0,
                     since_epoch: int = 0, significance_min: float = 0.0,
                     significance_max: float = 1.0, limit: int = 10,
                     order: str = "desc", include_content: bool = False,
                     thought_type: str = "") -> list[dict]:
        """Query blocks from TimeChain. Returns list of block metadata."""
        try:
            reply = await self._bus.request_async(
                "timechain_proxy", "timechain",
                {"action": "recall", "fork": fork, "source": source,
                 "tag_contains": tag_contains, "since_hours": since_hours,
                 "since_epoch": since_epoch,
                 "significance_min": significance_min,
                 "significance_max": significance_max,
                 "limit": limit, "order": order,
                 "include_content": include_content,
                 "thought_type": thought_type},
                5.0, self._reply_queue,
            )
        except Exception as e:
            logger.warning("[TimechainProxy] recall raised: %s", e)
            return []
        if reply:
            return reply.get("payload", {}).get("results", [])
        logger.debug("[TimechainProxy] recall timed out")
        return []

    async def check(self, fork: str = "", source: str = "",
                    tag_contains: str = "", since_hours: float = 0,
                    since_epoch: int = 0,
                    significance_min: float = 0.0) -> bool:
        """Quick boolean check — does X exist in memory?"""
        try:
            reply = await self._bus.request_async(
                "timechain_proxy", "timechain",
                {"action": "check", "fork": fork, "source": source,
                 "tag_contains": tag_contains, "since_hours": since_hours,
                 "since_epoch": since_epoch,
                 "significance_min": significance_min},
                3.0, self._reply_queue,
            )
        except Exception as e:
            logger.warning("[TimechainProxy] check raised: %s", e)
            return False
        if reply:
            return reply.get("payload", {}).get("result", False)
        return False

    async def compare(self, field: str, fork: str = "main",
                      window_a_hours: float = 6,
                      window_b_hours: float = 12) -> dict:
        """Compare a state field across two time windows."""
        try:
            reply = await self._bus.request_async(
                "timechain_proxy", "timechain",
                {"action": "compare", "fork": fork, "field": field,
                 "window_a_hours": window_a_hours,
                 "window_b_hours": window_b_hours},
                5.0, self._reply_queue,
            )
        except Exception as e:
            logger.warning("[TimechainProxy] compare raised: %s", e)
            return {"direction": "unknown", "delta": 0,
                    "a_value": None, "b_value": None}
        if reply:
            return reply.get("payload", {}).get("result", {})
        return {"direction": "unknown", "delta": 0,
                "a_value": None, "b_value": None}

    async def aggregate(self, fork: str = "", op: str = "count",
                        field: str = "significance", source: str = "",
                        thought_type: str = "",
                        since_hours: float = 24) -> float:
        """Aggregate over blocks (count, sum, avg, max, min)."""
        try:
            reply = await self._bus.request_async(
                "timechain_proxy", "timechain",
                {"action": "aggregate", "fork": fork, "op": op,
                 "field": field, "source": source,
                 "thought_type": thought_type,
                 "since_hours": since_hours},
                5.0, self._reply_queue,
            )
        except Exception as e:
            logger.warning("[TimechainProxy] aggregate raised: %s", e)
            return 0.0
        if reply:
            return float(reply.get("payload", {}).get("result", 0.0))
        return 0.0

    # ── State lookup (SHM-direct, Preamble G18) ─────────────────────

    def get_stats(self) -> dict:
        """Return cached timechain stats. For force-fresh, use refresh_stats()."""
        return dict(self._stats_cache)

    def refresh_stats(self) -> dict:
        """Force-fetch via SHM read of timechain_state.bin (Session 3
        publisher). Preamble G18 — state transport is SHM, never bus."""
        try:
            raw = self._r_timechain_state.read_variable()
        except Exception as e:
            self._track_fallback("timechain_state",
                                 f"read_raised:{type(e).__name__}")
            return dict(self._stats_cache)
        if raw is None:
            self._track_fallback("timechain_state", "shm_unavailable")
            return dict(self._stats_cache)
        try:
            decoded = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            self._track_fallback("timechain_state",
                                 f"decode_raised:{type(e).__name__}")
            return dict(self._stats_cache)
        if isinstance(decoded, dict):
            self._stats_cache = dict(decoded)
        return dict(self._stats_cache)

    def _track_fallback(self, slot_name: str, reason: str) -> None:
        prev = self._fallback_counts.get(slot_name, 0)
        self._fallback_counts[slot_name] = prev + 1
        if prev == 0:
            logger.info(
                "[TimechainProxy] FIRST FALLBACK slot=%s reason=%s",
                slot_name, reason)
