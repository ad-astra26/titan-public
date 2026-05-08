"""
Memory Module Proxy — bridge to the supervised Memory process.

Phase C Session 2 of rFP_phase_c_async_shm_consumer_migration §4.C.13
(LANDED 2026-05-07).

State-lookup methods migrated to non-blocking SHM read via
``StateRegistryReader`` against ``memory_state.bin`` (published by
``memory_worker`` @ 1 Hz). Closes the second-layer deadlock surface
py-spy revealed post-Session-1 — outer-sensor sidecars stuck inside
``memory_proxy.get_growth_metrics → bus.request`` once spirit_proxy
unblocked.

Migrated to SHM-direct (3 methods):
  - ``get_persistent_count``     → memory_state.bin.persistent_count
  - ``get_memory_status``        → memory_state.bin (cognee_ready,
                                    persistent_count, mempool_size)
  - ``get_growth_metrics``       → memory_state.bin (learning_velocity,
                                    directive_alignment, plus raw
                                    counts so non-default
                                    node_saturation_24h still works)

Methods that REMAIN bus-RPC (true work-RPCs, parameterized queries):
``query``, ``add_memory``, ``add_to_mempool``, ``fetch_mempool``,
``fetch_mempool_for_observatory``, ``get_top_memories``,
``get_top_memories_for_observatory``, ``get_topology``,
``get_knowledge_graph``, ``run_meditation``. These are NOT in sidecar /
hot-path code (FastAPI endpoints + dashboard + explicit user actions),
so they don't recreate the sidecar deadlock. Migration to
``bus.request_async`` with bounded timeout is Session 3 §4.C.13bis.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

import msgpack

from ..bus import DivineBus, QUERY, RESPONSE, make_request
from ..core.state_registry import (
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)
from ..guardian import Guardian
from ..logic.memory_state_specs import MEMORY_STATE_SLOT, MEMORY_STATE_SPEC

logger = logging.getLogger(__name__)


# Default node_saturation_24h matches the legacy growth_metrics handler
# default (memory_worker.py:668). Most callers in _gather_outer_sources
# pass this value; the publisher pre-computes learning_velocity +
# directive_alignment at this saturation.
_DEFAULT_NODE_SATURATION_24H = 30


class MemoryProxy:
    """
    Drop-in replacement for TieredMemoryGraph that routes calls
    through SHM (state lookups) and the Divine Bus (work RPCs).
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        self._reply_queue = bus.subscribe("memory_proxy", reply_only=True)
        self._started = False

        # SHM-direct reader for memory_state.bin (Session 2 §4.C.13).
        # Lazy attach on first read.
        self._titan_id = resolve_titan_id()
        self._shm_root: Path = ensure_shm_root(self._titan_id)
        self._r_memory_state = StateRegistryReader(
            MEMORY_STATE_SPEC, self._shm_root)

        # Per-cause fallback counter (cold-boot diagnostics).
        self._fallback_counts: dict[str, int] = {}

        logger.info(
            "[MemoryProxy] initialized — SHM reader for slot=%s "
            "(state-lookup methods G18/G19 compliant; parameterized "
            "queries still bus.request — Session 3 scope)",
            MEMORY_STATE_SLOT)

    # ── SHM read helper ─────────────────────────────────────────────

    def _read_memory_state(self) -> Optional[dict]:
        """Read memory_state.bin → msgpack-decoded dict, or None on
        cold-boot/missing/torn. First fallback per cause logs INFO."""
        try:
            raw = self._r_memory_state.read_variable()
        except Exception as e:
            self._track_fallback(f"read_raised:{type(e).__name__}")
            logger.warning(
                "[MemoryProxy] memory_state SHM read raised: %s",
                e, exc_info=True)
            return None
        if raw is None:
            self._track_fallback("shm_unavailable")
            return None
        try:
            decoded = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            self._track_fallback(f"decode_raised:{type(e).__name__}")
            logger.warning(
                "[MemoryProxy] memory_state msgpack decode failed "
                "(raw_bytes=%d): %s", len(raw), e, exc_info=True)
            return None
        if not isinstance(decoded, dict):
            self._track_fallback(f"decode_wrong_type:{type(decoded).__name__}")
            return None
        return decoded

    async def _work_rpc_async(self, payload: dict, timeout: float) -> Optional[dict]:
        """Single async work-RPC primitive."""
        try:
            return await self._bus.request_async(
                "memory_proxy", "memory", payload, timeout, self._reply_queue,
            )
        except Exception as e:
            logger.warning(
                "[MemoryProxy] %s bus.request_async raised: %s",
                payload.get("action", "?"), e)
            return None

    def _work_rpc_sync(self, payload: dict, timeout: float) -> Optional[dict]:
        """Sync wrapper around the async work-RPC. If we're inside a
        running event loop (FastAPI endpoint), fall back to bounded
        sync bus.request — allowlisted as work-RPC exemption in
        phase_c_rpc_exemptions.yaml (explicit timeout, not state lookup).
        """
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False
        if not in_loop:
            try:
                return asyncio.run(self._work_rpc_async(payload, timeout))
            except Exception as e:
                logger.warning(
                    "[MemoryProxy] %s asyncio.run failed: %s — falling "
                    "back to bounded sync bus.request",
                    payload.get("action", "?"), e)
        return self._bus.request(
            "memory_proxy", "memory", payload,
            timeout=timeout, reply_queue=self._reply_queue,
        )

    def _track_fallback(self, reason: str) -> None:
        prev = self._fallback_counts.get(reason, 0)
        self._fallback_counts[reason] = prev + 1
        if prev == 0:
            logger.info(
                "[MemoryProxy] FIRST FALLBACK reason=%s — using default "
                "(likely cold-boot before publisher first publish; "
                "should clear within ~1s)", reason)

    def _ensure_started(self) -> None:
        """Start the Memory module if not already running. Async-safe —
        see _start_safe.py for rationale (do not block event loop)."""
        from ._start_safe import ensure_started_async_safe
        if ensure_started_async_safe(
            self._guardian, "memory", id(self), proxy_label="MemoryProxy"
        ):
            self._started = True

    async def query(self, text: str, top_k: int = 5) -> list:
        """Query semantic + episodic memory.

        Phase C Session 4 (rFP §4.C.13): true work-RPC (FAISS+Kuzu+DuckDB
        parameterized query — runs in worker). Migrated sync→async per G19.
        """
        self._ensure_started()
        try:
            reply = await self._bus.request_async(
                "memory_proxy", "memory",
                {"action": "query", "text": text, "top_k": top_k},
                15.0, self._reply_queue,
            )
        except Exception as e:
            logger.warning("[MemoryProxy] query bus.request_async raised: %s", e)
            return []
        if reply:
            return reply.get("payload", {}).get("results", [])
        logger.warning("[MemoryProxy] query timed out")
        return []

    async def add_memory(self, text: str, **kwargs) -> Optional[str]:
        """Add a memory node.

        Work-RPC (write/upsert) — async per G19.
        """
        self._ensure_started()
        try:
            reply = await self._bus.request_async(
                "memory_proxy", "memory",
                {"action": "add", "text": text, **kwargs},
                10.0, self._reply_queue,
            )
        except Exception as e:
            logger.warning("[MemoryProxy] add_memory bus.request_async raised: %s", e)
            return None
        if reply:
            return reply.get("payload", {}).get("node_id")
        return None

    def get_persistent_count(self) -> int:
        """Get count of persistent memory nodes. SHM-direct via
        memory_state.bin (Session 2 §4.C.13). NEVER blocks."""
        decoded = self._read_memory_state()
        if decoded is None:
            return 0
        return int(decoded.get("persistent_count", 0))

    async def fetch_mempool(self) -> list:
        """Retrieve all mempool nodes (with decay applied). Work-RPC async."""
        self._ensure_started()
        try:
            reply = await self._bus.request_async(
                "memory_proxy", "memory",
                {"action": "fetch_mempool"},
                10.0, self._reply_queue,
            )
        except Exception as e:
            logger.warning("[MemoryProxy] fetch_mempool bus.request_async raised: %s", e)
            return []
        if reply:
            return reply.get("payload", {}).get("mempool", [])
        logger.warning("[MemoryProxy] fetch_mempool timed out")
        return []

    def get_top_memories(self, n: int = 5) -> list:
        """Get top N persistent memories by weight. Work-RPC (decay+sort+
        serialize on worker — runs in worker process). Migrated to
        sync-or-async via _work_rpc_sync per Preamble G19."""
        self._ensure_started()
        reply = self._work_rpc_sync(
            {"action": "top_memories", "n": n}, 30.0)
        if reply:
            return reply.get("payload", {}).get("memories", [])
        return []

    def get_top_memories_for_observatory(self, n: int = 200) -> list:
        """Get top N persistent memories with embeddings stripped — bus-safe shape.
        Work-RPC (parameterized N + decay/sort) — sync-or-async via
        _work_rpc_sync per Preamble G19.

        rFP_bus_payload_contracts §3.1 (2026-05-01): /status/memory endpoint
        path. Worker handler `top_memories_observatory` returns lightweight
        items. Tight 10s timeout: endpoint UX expects <10s.
        """
        self._ensure_started()
        reply = self._work_rpc_sync(
            {"action": "top_memories_observatory", "n": n}, 10.0)
        if reply:
            payload = reply.get("payload", {})
            mems = payload.get("memories", [])
            logger.info(
                "[MemoryProxy] get_top_memories_for_observatory: reply received "
                "with %d memories (count=%s, error=%s)",
                len(mems) if isinstance(mems, list) else 0,
                payload.get("count"),
                payload.get("error", "none"),
            )
            return mems
        logger.warning(
            "[MemoryProxy] get_top_memories_for_observatory: NO REPLY (timeout 10s)")
        return []

    def fetch_mempool_for_observatory(self) -> list:
        """Get mempool with embeddings stripped — bus-safe shape. Work-RPC
        (decay applied on worker) — sync-or-async via _work_rpc_sync."""
        self._ensure_started()
        reply = self._work_rpc_sync(
            {"action": "fetch_mempool_observatory"}, 3.0)
        if reply:
            return reply.get("payload", {}).get("mempool", [])
        return []

    def get_memory_status(self) -> dict:
        """Get memory subsystem status (cognee_ready, counts).
        SHM-direct via memory_state.bin (Session 2 §4.C.13).
        NEVER blocks — was the deadlock-causing call from
        _gather_outer_sources sidecar path post-Session-1."""
        decoded = self._read_memory_state()
        if decoded is None:
            return {
                "cognee_ready": False,
                "backend_ready": False,
                "persistent_count": 0,
                "mempool_size": 0,
            }
        return {
            "cognee_ready": bool(decoded.get("cognee_ready", False)),
            "backend_ready": bool(decoded.get("cognee_ready", False)),
            "persistent_count": int(decoded.get("persistent_count", 0)),
            "mempool_size": int(decoded.get("mempool_size", 0)),
        }

    async def add_to_mempool(self, user_prompt: str, agent_response: str,
                            user_identifier: str = "Anonymous") -> None:
        """Add conversation to mempool for later promotion. Work-RPC async."""
        self._ensure_started()
        try:
            await self._bus.request_async(
                "memory_proxy", "memory",
                {
                    "action": "add_to_mempool",
                    "user_prompt": user_prompt,
                    "agent_response": agent_response,
                    "user_identifier": user_identifier,
                },
                10.0, self._reply_queue,
            )
        except Exception as e:
            logger.warning("[MemoryProxy] add_to_mempool raised: %s", e)

    def get_growth_metrics(self, node_saturation_24h: int = 30) -> dict:
        """Get growth metrics (learning velocity, directive alignment).
        SHM-direct via memory_state.bin (Session 2 §4.C.13).

        **THE deadlock-causing method** that py-spy revealed wedging T3
        outer-sensor sidecars on 2026-05-07 post-Session-1. Now SHM-direct
        — never touches the bus.

        If caller passes the publisher-default ``node_saturation_24h=30``
        (which all _gather_outer_sources callers do), returns
        pre-computed values from memory_state.bin directly.

        If a caller passes a different saturation, recomputes
        learning_velocity locally from the raw effective_nodes_24h /
        total_persistent_for_growth / high_quality_count fields the
        publisher exposes — same formula as memory_worker action handler
        (single source of truth).
        """
        import math
        decoded = self._read_memory_state()
        if decoded is None:
            return {
                "learning_velocity": 0.5,
                "directive_alignment": 0.5,
                "effective_nodes_24h": 0.0,
                "total_persistent": 0,
                "high_quality_count": 0,
            }

        if node_saturation_24h == _DEFAULT_NODE_SATURATION_24H:
            # Fast path — publisher pre-computed at this saturation
            return {
                "learning_velocity": float(decoded.get("learning_velocity", 0.5)),
                "directive_alignment": float(decoded.get("directive_alignment", 0.5)),
                "effective_nodes_24h": float(decoded.get("effective_nodes_24h", 0.0)),
                "total_persistent": int(decoded.get("total_persistent_for_growth", 0)),
                "high_quality_count": int(decoded.get("high_quality_count", 0)),
            }

        # Slow path — caller passed non-default saturation; recompute
        # learning_velocity from raw counts using the same formula.
        # directive_alignment is saturation-independent (no recompute).
        eff = float(decoded.get("effective_nodes_24h", 0.0))
        learning_vel = min(
            1.0, math.log(eff + 1) / math.log(node_saturation_24h + 1))
        return {
            "learning_velocity": round(learning_vel, 4),
            "directive_alignment": float(decoded.get("directive_alignment", 0.5)),
            "effective_nodes_24h": eff,
            "total_persistent": int(decoded.get("total_persistent_for_growth", 0)),
            "high_quality_count": int(decoded.get("high_quality_count", 0)),
        }

    async def run_meditation_async(self) -> dict:
        """Trigger a meditation cycle in the memory worker (async path).

        rFP_meditation_worker_latency Fix #A (2026-05-07): plugin.
        _meditation_loop is itself an asyncio coroutine. Calling the
        legacy sync `run_meditation()` from there hits `_work_rpc_sync`'s
        sync-fallback (bus.request reply_queue.get) which BLOCKS the
        main asyncio event loop for up to 300s. While blocked, the
        kernel's `_anchor_request_loop` coroutine cannot advance to
        process the worker's ANCHOR_REQUEST — a self-deadlock that
        causes worker's anchor_request_timeout (120s), MEDITATION_LOCAL
        sig, and the cascade that gated art generation + Arweave backup
        for 8-26 days fleet-wide. Live measurement 2026-05-07 17:41:54
        → 17:44:30 captured `bus_dispatch=156263.1ms` between worker
        emit and kernel handler entry — pure asyncio loop deadlock.

        This async variant uses `bus.request_async` which runs the
        blocking reply-wait in `bus_ipc_pool` (dedicated executor) so
        the main asyncio loop stays free to drain the kernel's anchor
        queue concurrently. Cycle completes in ~3-30s typical instead
        of 156-280s timeout cascade.

        Timeout sizing: 300s end-to-end budget covers worker handler
        runtime: fetch_mempool (≤1s) + scoring (≤30s × promoted) +
        anchor RPC (≤120s — kernel can advance now, so this is
        bounded by Solana TX confirm ≈3-10s) + migration (≤30s/node ×
        promoted) + consolidate (now no-op = 0.05s). Realistic
        completion ≤30s; 300s is paranoid headroom.
        """
        self._ensure_started()
        try:
            reply = await self._bus.request_async(
                "memory_proxy", "memory",
                {"action": "run_meditation"},
                300.0, self._reply_queue,
            )
        except Exception as e:
            logger.warning(
                "[MemoryProxy] run_meditation bus.request_async raised: %s", e)
            return {"success": False, "error": str(e)}
        if reply:
            return reply.get("payload", {})
        return {"success": False, "error": "timeout"}

    def run_meditation(self) -> dict:
        """Sync compat shim around `run_meditation_async`.

        Retained for the few non-async callers (legacy_core path + tests)
        that still expect a synchronous return. ALL ASYNC CALLERS MUST
        SWITCH TO `run_meditation_async()` to avoid the asyncio event
        loop deadlock documented at the top of `run_meditation_async`.
        """
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False
        if in_loop:
            # Caller is on an asyncio loop but went through the sync
            # path — log a loud warning since this re-introduces the
            # deadlock surface. Fall back to the legacy `_work_rpc_sync`
            # which routes the blocking wait through bus_ipc_pool when
            # called via async context (it actually does NOT — it falls
            # back to direct sync bus.request — so we add an explicit
            # warning that the caller should switch).
            logger.warning(
                "[MemoryProxy] run_meditation() called from asyncio context — "
                "this re-introduces the 156s event-loop deadlock. Migrate "
                "this caller to `await run_meditation_async()` (rFP_meditation_"
                "worker_latency Fix #A 2026-05-07).")
            reply = self._work_rpc_sync(
                {"action": "run_meditation"}, 300.0)
        else:
            reply = self._work_rpc_sync(
                {"action": "run_meditation"}, 300.0)
        if reply:
            return reply.get("payload", {})
        return {"success": False, "error": "timeout"}

    def get_topology(self, topic_keywords: dict) -> dict:
        """Compute cognitive heatmap topology (runs in memory worker process).
        Work-RPC — sync-or-async via _work_rpc_sync."""
        self._ensure_started()
        serializable_kws = {k: list(v) for k, v in topic_keywords.items()}
        reply = self._work_rpc_sync(
            {"action": "topology", "topic_keywords": serializable_kws}, 10.0)
        if reply:
            return reply.get("payload", {})
        return {"total_persistent": 0, "clusters": {}}

    def get_knowledge_graph(self, limit: int = 200) -> dict:
        """Get Kuzu entity graph data for 3D visualization (runs in memory
        worker process). Work-RPC — sync-or-async via _work_rpc_sync."""
        self._ensure_started()
        reply = self._work_rpc_sync(
            {"action": "knowledge_graph", "limit": limit}, 15.0)
        if reply:
            return reply.get("payload", {})
        return {"available": False, "error": "timeout"}

    # ── Diagnostics ─────────────────────────────────────────────────

    def get_diagnostics(self) -> dict:
        """Per-cause SHM fallback counts + migration metadata. Used by
        health-check endpoints + arch_map verify."""
        return {
            "titan_id": self._titan_id,
            "shm_root": str(self._shm_root),
            "fallback_counts": dict(self._fallback_counts),
            "session2_migrated_methods": [
                "get_persistent_count",
                "get_memory_status",
                "get_growth_metrics",
            ],
            "session3_pending_methods": [
                "query",
                "add_memory",
                "add_to_mempool",
                "fetch_mempool",
                "fetch_mempool_for_observatory",
                "get_top_memories",
                "get_top_memories_for_observatory",
                "get_topology",
                "get_knowledge_graph",
                "run_meditation",
            ],
        }
