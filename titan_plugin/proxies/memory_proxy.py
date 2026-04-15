"""
Memory Module Proxy — lazy bridge to the supervised Memory process.

Provides the same interface as TieredMemoryGraph but delegates
all calls to the Memory module process via the Divine Bus.
When called for the first time, signals Guardian to start the module.
"""
import logging
from typing import Optional

from ..bus import DivineBus, QUERY, RESPONSE, make_request
from ..guardian import Guardian

logger = logging.getLogger(__name__)


class MemoryProxy:
    """
    Drop-in replacement for TieredMemoryGraph that routes calls
    through the Divine Bus to the supervised Memory module.
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        self._reply_queue = bus.subscribe("memory_proxy", reply_only=True)
        self._started = False

    def _ensure_started(self) -> None:
        """Start the Memory module if not already running. Async-safe —
        see _start_safe.py for rationale (do not block event loop)."""
        from ._start_safe import ensure_started_async_safe
        if ensure_started_async_safe(
            self._guardian, "memory", id(self), proxy_label="MemoryProxy"
        ):
            self._started = True

    async def query(self, text: str, top_k: int = 5) -> list:
        """Query semantic + episodic memory."""
        self._ensure_started()
        reply = self._bus.request(
            "memory_proxy", "memory",
            {"action": "query", "text": text, "top_k": top_k},
            timeout=15.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("results", [])
        logger.warning("[MemoryProxy] query timed out")
        return []

    async def add_memory(self, text: str, **kwargs) -> Optional[str]:
        """Add a memory node."""
        self._ensure_started()
        reply = self._bus.request(
            "memory_proxy", "memory",
            {"action": "add", "text": text, **kwargs},
            timeout=10.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("node_id")
        return None

    def get_persistent_count(self) -> int:
        """Get count of persistent memory nodes."""
        self._ensure_started()
        reply = self._bus.request(
            "memory_proxy", "memory",
            {"action": "count"},
            timeout=5.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("count", 0)
        return 0

    async def fetch_mempool(self) -> list:
        """Retrieve all mempool nodes (with decay applied)."""
        self._ensure_started()
        reply = self._bus.request(
            "memory_proxy", "memory",
            {"action": "fetch_mempool"},
            timeout=10.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("mempool", [])
        logger.warning("[MemoryProxy] fetch_mempool timed out")
        return []

    def get_top_memories(self, n: int = 5) -> list:
        """Get top N persistent memories by weight."""
        self._ensure_started()
        # 2026-04-09: timeout bumped 10s → 30s to support full-store fetches
        # for the dashboard endpoint, which now requests n ≈ persistent_count
        # to filter out internal-injection rows. With ~3.5k persistent memories
        # the decay+sort+serialize path on the worker side can take 2-3s, and
        # bus contention can add another second or two. 30s gives clear headroom
        # without making real timeouts wait excessively.
        reply = self._bus.request(
            "memory_proxy", "memory",
            {"action": "top_memories", "n": n},
            timeout=30.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("memories", [])
        return []

    def get_memory_status(self) -> dict:
        """Get memory subsystem status (cognee_ready, counts)."""
        self._ensure_started()
        reply = self._bus.request(
            "memory_proxy", "memory",
            {"action": "status"},
            timeout=5.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        return {"cognee_ready": False, "persistent_count": 0, "mempool_size": 0}

    async def add_to_mempool(self, user_prompt: str, agent_response: str,
                            user_identifier: str = "Anonymous") -> None:
        """Add conversation to mempool for later promotion."""
        self._ensure_started()
        self._bus.request(
            "memory_proxy", "memory",
            {
                "action": "add_to_mempool",
                "user_prompt": user_prompt,
                "agent_response": agent_response,
                "user_identifier": user_identifier,
            },
            timeout=10.0,
            reply_queue=self._reply_queue,
        )

    def get_growth_metrics(self, node_saturation_24h: int = 30) -> dict:
        """Get growth metrics (learning velocity, directive alignment) computed in worker."""
        self._ensure_started()
        reply = self._bus.request(
            "memory_proxy", "memory",
            {"action": "growth_metrics", "node_saturation_24h": node_saturation_24h},
            timeout=10.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        return {"learning_velocity": 0.5, "directive_alignment": 0.5}

    def run_meditation(self) -> dict:
        """Trigger a meditation cycle in the memory worker."""
        self._ensure_started()
        reply = self._bus.request(
            "memory_proxy", "memory",
            {"action": "run_meditation"},
            timeout=120.0,  # meditation can take a while (LLM scoring)
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        return {"success": False, "error": "timeout"}

    def get_topology(self, topic_keywords: dict) -> dict:
        """Compute cognitive heatmap topology (runs in memory worker process)."""
        self._ensure_started()
        # Convert sets to lists for serialization
        serializable_kws = {k: list(v) for k, v in topic_keywords.items()}
        reply = self._bus.request(
            "memory_proxy", "memory",
            {"action": "topology", "topic_keywords": serializable_kws},
            timeout=10.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        return {"total_persistent": 0, "clusters": {}}

    def get_knowledge_graph(self, limit: int = 200) -> dict:
        """Get Kuzu entity graph data for 3D visualization (runs in memory worker process)."""
        self._ensure_started()
        reply = self._bus.request(
            "memory_proxy", "memory",
            {"action": "knowledge_graph", "limit": limit},
            timeout=15.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        return {"available": False, "error": "timeout"}
