"""Pydantic schemas for bus message payloads — bus contract enforcement.

Per `rFP_bus_payload_contracts.md` (2026-05-01). Every bus message type with a
contract has a Pydantic schema declared here. Validation runs:
  - at SEND time in `_packb_safe` (producer side, fail-loud)
  - at RECEIVE time in `BusSubscriber.handle_message` (consumer side, fail-safe)

Bus carries EVENTS (notifications, light state ≤100 KB), NOT bulk data.
Bulk data goes through SHM (like inner trinity) or RPC-on-demand
(like memory_proxy.get_top_memories_for_observatory).

Add new schemas alphabetically by msg_type.

EVENT_SCHEMAS dict at the bottom is the single source of truth that
`bus_contracts.REGISTRY` references and that `BusSubscriber` looks up.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class _BusEvent(BaseModel):
    """Base class for all bus event schemas. Permissive on extra fields
    during the migration window (set forbid once all producers migrated)."""

    model_config = ConfigDict(extra="ignore")


# ── Memory worker events (rFP §3.1) ────────────────────────────────────


class MemoryStatus(_BusEvent):
    """MEMORY_STATUS_UPDATED — counts + readiness flags. Light by design."""

    persistent_count: int = 0
    mempool_size: int = 0
    cognee_ready: bool = False
    memory_backend_ready: bool = False
    updated_at: float = 0.0


class MemoryTopEvent(_BusEvent):
    """MEMORY_TOP_UPDATED — notification only.

    Bulk data fetched via RPC: memory_proxy.get_top_memories_for_observatory().
    Pre-rFP this carried 250 items × ~8KB embeddings = 2.1 MB on bus, which
    failed msgpack UTF-8 decode at broker boundary on T2/T3.
    """

    updated_at: float
    count: int
    last_id: Optional[str] = None  # cache-bust hint for stale-detection


class MemoryMempoolEvent(_BusEvent):
    """MEMORY_MEMPOOL_UPDATED — notification only.

    Bulk data via memory_proxy.fetch_mempool().
    """

    updated_at: float
    count: int


class MemoryTopologyEvent(_BusEvent):
    """MEMORY_TOPOLOGY_UPDATED — light cluster summary.

    cluster_counts is intentionally inlined (small dict, ~6-7 entries)
    because the topology heatmap renders directly from these counts.
    Bulk per-cluster samples fetched via memory_proxy.get_topology().
    """

    updated_at: float
    total_classified: int
    cluster_counts: dict[str, int] = {}


class MemoryKnowledgeGraphEvent(_BusEvent):
    """MEMORY_KNOWLEDGE_GRAPH_UPDATED — node/edge summary.

    Bulk graph data via memory_proxy.get_knowledge_graph(limit=N).
    """

    updated_at: float
    node_count: int = 0
    edge_count: int = 0
    entity_types: dict[str, int] = {}


# ── Registry — single source of truth ──────────────────────────────────


EVENT_SCHEMAS: dict[str, type[_BusEvent]] = {
    "MEMORY_STATUS_UPDATED": MemoryStatus,
    "MEMORY_TOP_UPDATED": MemoryTopEvent,
    "MEMORY_MEMPOOL_UPDATED": MemoryMempoolEvent,
    "MEMORY_TOPOLOGY_UPDATED": MemoryTopologyEvent,
    "MEMORY_KNOWLEDGE_GRAPH_UPDATED": MemoryKnowledgeGraphEvent,
}
"""Map msg_type → Pydantic schema class. Looked up by bus boundary
validators on both send and receive paths.

Extending this dict: add the msg_type entry + the schema class above.
Run `arch_map bus-contracts --audit` to verify a contract exists in
bus_contracts.REGISTRY for the same msg_type.

NOT a complete map of all bus message types in the system — only those
under contract. Unschematized types pass through validation untouched
(legacy compat during migration window).
"""
