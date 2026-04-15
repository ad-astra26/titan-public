"""
TimeChain Proxy — bus bridge to TimeChain v2 Consumer API.

Provides recall(), check(), compare(), aggregate() methods that
route through DivineBus to the TimeChain worker (v2 orchestrator).

Used by: pre-prompt enrichment, API endpoints, any module needing
to QUERY cognitive memory without direct TimeChain access.
"""
import logging
from typing import Optional

from ..bus import DivineBus

logger = logging.getLogger(__name__)


class TimechainProxy:
    """
    Consumer API proxy for TimeChain v2.
    Routes queries via DivineBus to the timechain worker.
    """

    def __init__(self, bus: DivineBus, guardian=None):
        self._bus = bus
        self._guardian = guardian
        self._reply_queue = bus.subscribe("timechain_proxy", reply_only=True)

    def recall(self, fork: str = "", source: str = "",
               tag_contains: str = "", since_hours: float = 0,
               since_epoch: int = 0, significance_min: float = 0.0,
               significance_max: float = 1.0, limit: int = 10,
               order: str = "desc", include_content: bool = False,
               thought_type: str = "") -> list[dict]:
        """Query blocks from TimeChain. Returns list of block metadata."""
        reply = self._bus.request(
            "timechain_proxy", "timechain",
            {"action": "recall", "fork": fork, "source": source,
             "tag_contains": tag_contains, "since_hours": since_hours,
             "since_epoch": since_epoch,
             "significance_min": significance_min,
             "significance_max": significance_max,
             "limit": limit, "order": order,
             "include_content": include_content,
             "thought_type": thought_type},
            timeout=5.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("results", [])
        logger.debug("[TimechainProxy] recall timed out")
        return []

    def check(self, fork: str = "", source: str = "",
              tag_contains: str = "", since_hours: float = 0,
              since_epoch: int = 0,
              significance_min: float = 0.0) -> bool:
        """Quick boolean check — does X exist in memory?"""
        reply = self._bus.request(
            "timechain_proxy", "timechain",
            {"action": "check", "fork": fork, "source": source,
             "tag_contains": tag_contains, "since_hours": since_hours,
             "since_epoch": since_epoch,
             "significance_min": significance_min},
            timeout=3.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("result", False)
        return False

    def compare(self, field: str, fork: str = "main",
                window_a_hours: float = 6,
                window_b_hours: float = 12) -> dict:
        """Compare a state field across two time windows."""
        reply = self._bus.request(
            "timechain_proxy", "timechain",
            {"action": "compare", "fork": fork, "field": field,
             "window_a_hours": window_a_hours,
             "window_b_hours": window_b_hours},
            timeout=5.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("result", {})
        return {"direction": "unknown", "delta": 0,
                "a_value": None, "b_value": None}

    def aggregate(self, fork: str = "", op: str = "count",
                  field: str = "significance", source: str = "",
                  thought_type: str = "",
                  since_hours: float = 24) -> float:
        """Aggregate over blocks (count, sum, avg, max, min)."""
        reply = self._bus.request(
            "timechain_proxy", "timechain",
            {"action": "aggregate", "fork": fork, "op": op,
             "field": field, "source": source,
             "thought_type": thought_type,
             "since_hours": since_hours},
            timeout=5.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return float(reply.get("payload", {}).get("result", 0.0))
        return 0.0
