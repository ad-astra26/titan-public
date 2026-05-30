"""
memory_state_publisher — Phase C Session 2 of
rFP_phase_c_async_shm_consumer_migration §4.B.8.

Publishes memory_state.bin once per second (SPEC §7.1) so consumers in
titan_HCL (memory_proxy + sidecars + dashboards) can read memory state
non-blocking from SHM instead of synchronous bus.request.

Closes the second-layer deadlock surface uncovered post-Session-1:
py-spy on T3 2026-05-07 caught outer-sensor sidecars stuck inside
``memory_proxy.get_growth_metrics → bus.request`` once spirit_proxy
unblocked. memory_state.bin removes that wait by publishing the same
fields the worker computes.

Payload schema (msgpack):
  {
    "cognee_ready":           bool,    # backend is ready (memory.get_memory_status)
    "persistent_count":       int,     # # persistent MemoryNodes
    "mempool_size":           int,     # # mempool entries
    "effective_nodes_24h":    float,   # weighted-active-recent (raw)
    "high_quality_count":     int,     # nodes with effective_weight >= 1.15
    "total_persistent_for_growth": int,# persistent count counted in growth pass
    "learning_velocity":      float,   # = log(eff+1) / log(node_sat+1) @ default sat=30
    "directive_alignment":    float,   # = high_quality / max(1,total) * 1.2 (clamped 1.0)
    "kg_node_count":          int,     # Kuzu graph stats: sum of all entity tables
    "kg_edge_count":          int,     # Kuzu edge total (best-effort; 0 if unavailable)
    "kg_stats_by_table":      dict,    # per-table counts (Person/Topic/Body/Mind/Spirit/Media)
    "ts":                     float,   # publisher wall_ns at write time
  }

Owner (G21 single-writer): memory_worker only. Consumers attach
``StateRegistryReader`` against the shared MEMORY_STATE_SPEC from
``memory_state_specs``.

Cadence (G20 hot-path safety): 1 Hz. The compute is bounded — iterates
``memory._node_store`` once per tick to fill in growth metrics. Same
formula the legacy ``growth_metrics`` action handler uses (single source
of truth, not duplicated math). For very large stores (>50k nodes) the
loop dominates publisher cost; if that becomes a problem the publisher
can downshift to event-driven (publish on mempool/promotion change)
without changing the consumer contract.

Failure modes (G20 + memory_state_publisher.py logging):
  - memory ref None / not started → publish stub with zeros; consumers
    treat as cold-boot and use defaults
  - encode/oversize/write fails handled per slot like spirit_state
    publisher — first WARN with exc_info, subsequent throttled
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Optional

import msgpack

from titan_hcl.core.state_registry import (
    StateRegistryWriter,
    ensure_shm_root,
)
from titan_hcl.logic.memory_state_specs import (
    MEMORY_STATE_SLOT,
    MEMORY_STATE_SPEC,
)

logger = logging.getLogger(__name__)


_WARN_THROTTLE_EVERY = 60
_HEARTBEAT_TICKS = (1, 10, 60, 600, 3600)

#: Default node_saturation threshold matching what _gather_outer_sources
#: uses (memory_proxy.get_growth_metrics() default arg). Most callers
#: pass this default; those that pass a different value can recompute
#: client-side from raw counts (publisher exposes both).
_DEFAULT_NODE_SATURATION_24H = 30


class MemoryStatePublisher:
    """
    Owns memory_state.bin SHM writer; called from memory_worker's
    periodic loop @ 1 Hz. Single-threaded (G21).
    """

    def __init__(self, titan_id: str):
        self._titan_id = titan_id
        self._shm_root = ensure_shm_root(titan_id)
        self._writer: Optional[StateRegistryWriter] = None
        self._publish_count = 0
        self._publish_success = 0
        self._encode_fails = 0
        self._oversize_fails = 0
        self._write_fails = 0
        logger.info(
            "[MemoryStatePublisher] initialized — titan_id=%s shm_root=%s "
            "(slot=%s — SPEC §7.1 / Preamble G18)",
            titan_id, self._shm_root, MEMORY_STATE_SLOT)

    def _writer_attach(self) -> StateRegistryWriter:
        if self._writer is not None:
            return self._writer
        self._writer = StateRegistryWriter(MEMORY_STATE_SPEC, self._shm_root)
        logger.info(
            "[MemoryStatePublisher] writer attached — slot=%s "
            "max_bytes=%d schema_version=%d path=%s",
            MEMORY_STATE_SLOT, MEMORY_STATE_SPEC.payload_bytes,
            MEMORY_STATE_SPEC.schema_version,
            self._shm_root / f"{MEMORY_STATE_SLOT}.bin")
        return self._writer

    def publish(self, memory: Any) -> None:
        """
        Compute payload from `memory` (TieredMemoryGraph instance) and
        write to memory_state.bin. Cold-boot safe — if memory is None
        or missing expected attrs, publish stub payload.

        ``memory`` is the in-process TieredMemoryGraph reference held by
        memory_worker (NOT a proxy — direct access). It exposes
        ``_node_store``, ``_mempool``, ``_graph``, ``_cognee_ready``,
        ``get_persistent_count()``.
        """
        self._publish_count += 1
        payload = self._compute_payload(memory)
        self._write(payload)

        if self._publish_count in _HEARTBEAT_TICKS:
            logger.info(
                "[MemoryStatePublisher] heartbeat — publish_count=%d "
                "success=%d fails={encode=%d oversize=%d write=%d}",
                self._publish_count, self._publish_success,
                self._encode_fails, self._oversize_fails, self._write_fails)

    def _compute_payload(self, memory: Any) -> dict[str, Any]:
        """Build the msgpack payload from memory state. Defensive against
        None / partial-state at cold boot per G20."""
        # Cold-boot stub
        if memory is None:
            return {
                "cognee_ready": False,
                "persistent_count": 0,
                "mempool_size": 0,
                "effective_nodes_24h": 0.0,
                "high_quality_count": 0,
                "total_persistent_for_growth": 0,
                "learning_velocity": 0.5,
                "directive_alignment": 0.5,
                "kg_node_count": 0,
                "kg_edge_count": 0,
                "kg_stats_by_table": {},
                # Phase A.4 gap-1 closure (rFP §A.1.3 row #9): schema
                # expansion with the 3 fields previously served via
                # memory_proxy work-RPC.
                "top_memories": [],
                "mempool_preview": [],
                "knowledge_graph": {},
                "ts": time.time(),
            }

        # Status fields (cheap)
        try:
            cognee_ready = bool(getattr(memory, "_cognee_ready", False))
        except Exception:
            cognee_ready = False
        try:
            persistent_count = int(memory.get_persistent_count())
        except Exception as e:
            logger.warning(
                "[MemoryStatePublisher] get_persistent_count raised: %s",
                e, exc_info=True)
            persistent_count = 0
        try:
            mempool_size = int(len(getattr(memory, "_mempool", []) or []))
        except Exception:
            mempool_size = 0

        # Growth metrics (single iteration over _node_store — bounded
        # by store size; same formula as memory_worker action=growth_metrics)
        now = time.time()
        cutoff = now - 86400  # 24h
        effective_nodes = 0.0
        total_persistent_growth = 0
        high_quality = 0
        try:
            node_store = getattr(memory, "_node_store", None) or {}
            for v in node_store.values():
                if not isinstance(v, dict):
                    continue
                if v.get("type") != "MemoryNode":
                    continue
                if v.get("status") != "persistent":
                    continue
                total_persistent_growth += 1
                if v.get("effective_weight", 1.0) >= 1.15:
                    high_quality += 1
                created = v.get("created_at", 0) or 0
                accessed = v.get("last_accessed", 0) or 0
                if created >= cutoff or accessed >= cutoff:
                    effective_nodes += float(v.get("effective_weight", 1.0))
        except Exception as e:
            logger.warning(
                "[MemoryStatePublisher] node_store iteration raised: %s",
                e, exc_info=True)

        # Pre-computed metrics at default saturation (matches the
        # legacy growth_metrics handler at memory_worker.py:668-670)
        learning_vel = min(
            1.0,
            math.log(effective_nodes + 1) /
            math.log(_DEFAULT_NODE_SATURATION_24H + 1))
        directive_align = min(
            1.0,
            (high_quality / max(1, total_persistent_growth)) * 1.2)

        # Kuzu graph stats — best-effort, never block
        kg_node_count = 0
        kg_edge_count = 0
        kg_stats: dict[str, int] = {}
        try:
            graph = getattr(memory, "_graph", None)
            if graph is not None:
                _stats = graph.get_stats()
                if isinstance(_stats, dict):
                    kg_stats = {str(k): int(v) for k, v in _stats.items()}
                    kg_node_count = int(sum(kg_stats.values()))
                    # Edge count via fast path if exposed; otherwise leave
                    # at 0 (full edge query is expensive — caller needing
                    # exact edge count uses get_knowledge_graph RPC)
                    kg_edge_count = int(getattr(graph, "_cached_edge_count", 0))
        except Exception as e:
            logger.warning(
                "[MemoryStatePublisher] kg stats fetch raised: %s",
                e, exc_info=True)

        # Phase A.4 gap-1 closure — top memories (bounded ≤20), mempool
        # preview (bounded ≤20), knowledge_graph summary.
        top_memories: list[dict] = []
        mempool_preview: list[dict] = []
        knowledge_graph: dict = {}
        try:
            node_store = getattr(memory, "_node_store", None) or {}
            persistent = [
                v for v in node_store.values()
                if isinstance(v, dict) and v.get("type") == "MemoryNode"
                and v.get("status") == "persistent"
            ]
            persistent.sort(
                key=lambda v: float(v.get("effective_weight", 1.0)),
                reverse=True)
            for v in persistent[:20]:
                top_memories.append({
                    "id": str(v.get("id", "") or "")[:64],
                    "weight": round(float(v.get("effective_weight", 1.0)), 4),
                    "title": str(v.get("title", "") or "")[:120],
                    "created_at": float(v.get("created_at", 0.0) or 0.0),
                })
        except Exception as e:
            logger.warning(
                "[MemoryStatePublisher] top_memories build raised: %s",
                e, exc_info=True)
        try:
            mempool = getattr(memory, "_mempool", None) or []
            if isinstance(mempool, (list, tuple)):
                _iter = list(mempool)[:20]
            elif isinstance(mempool, dict):
                _iter = list(mempool.values())[:20]
            else:
                _iter = []
            for v in _iter:
                if isinstance(v, dict):
                    mempool_preview.append({
                        "id": str(v.get("id", "") or "")[:64],
                        "title": str(v.get("title", "") or "")[:120],
                        "weight": round(
                            float(v.get("effective_weight", 1.0) or 1.0), 4),
                    })
        except Exception as e:
            logger.warning(
                "[MemoryStatePublisher] mempool_preview build raised: %s",
                e, exc_info=True)
        try:
            # Light snapshot of graph: per-table counts (already in kg_stats)
            # + edge_count. Heavy KG queries stay on memory_proxy work-RPC;
            # this slot intentionally carries only the cheap summary so it
            # fits the 8KB MEMORY_STATE_MAX_BYTES cap.
            knowledge_graph = {
                "node_count": kg_node_count,
                "edge_count": kg_edge_count,
                "tables": kg_stats,
            }
        except Exception:
            pass

        return {
            "cognee_ready": cognee_ready,
            "persistent_count": persistent_count,
            "mempool_size": mempool_size,
            "effective_nodes_24h": round(effective_nodes, 4),
            "high_quality_count": high_quality,
            "total_persistent_for_growth": total_persistent_growth,
            "learning_velocity": round(learning_vel, 4),
            "directive_alignment": round(directive_align, 4),
            "kg_node_count": kg_node_count,
            "kg_edge_count": kg_edge_count,
            "kg_stats_by_table": kg_stats,
            # Phase A.4 gap-1 closure (rFP §A.1.3 row #9)
            "top_memories": top_memories,
            "mempool_preview": mempool_preview,
            "knowledge_graph": knowledge_graph,
            "ts": time.time(),
        }

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            encoded = msgpack.packb(payload, use_bin_type=True)
        except (TypeError, ValueError) as e:
            self._encode_fails += 1
            if self._encode_fails == 1 or self._encode_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[MemoryStatePublisher] msgpack encode failed (#%d): "
                    "%s — keys=%s",
                    self._encode_fails, e, sorted(payload.keys()),
                    exc_info=True)
            return

        if len(encoded) > MEMORY_STATE_SPEC.payload_bytes:
            self._oversize_fails += 1
            logger.critical(
                "[MemoryStatePublisher] payload %dB > MAX %dB (#%d) — "
                "slot retains last-known. Investigate upstream shape drift; "
                "do NOT silently truncate.",
                len(encoded), MEMORY_STATE_SPEC.payload_bytes,
                self._oversize_fails)
            return

        try:
            writer = self._writer_attach()
            writer.write_variable(encoded)
            self._publish_success += 1
            if self._publish_success == 1:
                logger.info(
                    "[MemoryStatePublisher] FIRST PUBLISH SUCCESS — "
                    "slot=%s payload_bytes=%d (consumers can now read; "
                    "T3 sidecar memory_proxy.get_growth_metrics deadlock "
                    "surface closed)",
                    MEMORY_STATE_SLOT, len(encoded))
        except Exception as e:
            self._write_fails += 1
            if self._write_fails == 1 or self._write_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[MemoryStatePublisher] shm write failed (#%d): %s",
                    self._write_fails, e, exc_info=True)

    def get_stats(self) -> dict[str, Any]:
        return {
            "publish_count": self._publish_count,
            "publish_success": self._publish_success,
            "encode_fails": self._encode_fails,
            "oversize_fails": self._oversize_fails,
            "write_fails": self._write_fails,
            "writer_attached": self._writer is not None,
        }
