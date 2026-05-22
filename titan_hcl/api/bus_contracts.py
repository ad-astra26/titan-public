"""Bus contract registry — single source of truth for msg type → schema +
size-limit + producer/consumer wiring.

Per `rFP_bus_payload_contracts.md` (2026-05-01).

A bus contract binds:
  - msg_type           — bus.X constant string
  - schema             — Pydantic model from cache_schemas.py
  - max_payload_bytes  — hard cap; oversize → fail at send (default 100 KB)
  - producer_module    — for arch_map static cross-check
  - consumer_modules   — same
  - delivery           — broadcast | targeted | both
  - summary            — human description for arch_map docs

Lookup by msg_type via `get_contract(msg_type)`. Returns None if no
contract registered (legacy / unschematized — validation skipped).

`arch_map bus-contracts --audit` cross-references this REGISTRY against
the AST-derived publish/consume wiring + reports gaps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from titan_hcl.api.cache_schemas import (
    EVENT_SCHEMAS,
    MemoryKnowledgeGraphEvent,
    MemoryMempoolEvent,
    MemoryStatus,
    MemoryTopEvent,
    MemoryTopologyEvent,
    _BusEvent,
)


DEFAULT_MAX_PAYLOAD_BYTES = 100_000
"""Default per-msg-type payload size limit. Anything larger should go
through SHM (like inner trinity tensors) or RPC-on-demand (like
memory_proxy bulk fetches). The bus is for events + light state.

Tighter per-contract limits override this (e.g. MemoryStatus is just
counts + flags — 4 KB is plenty)."""


@dataclass(frozen=True)
class BusContract:
    msg_type: str
    schema: type[_BusEvent]
    producer_module: str
    consumer_modules: tuple[str, ...]
    summary: str
    delivery: str = "broadcast"  # broadcast | targeted | both
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES


REGISTRY: tuple[BusContract, ...] = (
    # ── Memory events (rFP §3.1) ─────────────────────────────────────
    BusContract(
        msg_type="MEMORY_STATUS_UPDATED",
        schema=MemoryStatus,
        producer_module="memory_worker",
        consumer_modules=("api",),
        delivery="broadcast",
        max_payload_bytes=4_000,
        summary="Persistent + mempool counts; api populates `memory.status` cache "
                "for /status/memory endpoint.",
    ),
    BusContract(
        msg_type="MEMORY_TOP_UPDATED",
        schema=MemoryTopEvent,
        producer_module="memory_worker",
        consumer_modules=("api",),
        delivery="broadcast",
        max_payload_bytes=10_000,
        summary="Notification that top-memories list changed. Bulk data "
                "via memory_proxy.get_top_memories_for_observatory(). "
                "Pre-rFP carried 2.1 MB embedding vectors → broker rejected.",
    ),
    BusContract(
        msg_type="MEMORY_MEMPOOL_UPDATED",
        schema=MemoryMempoolEvent,
        producer_module="memory_worker",
        consumer_modules=("api",),
        delivery="broadcast",
        max_payload_bytes=4_000,
        summary="Notification that mempool changed. Bulk data via "
                "memory_proxy.fetch_mempool().",
    ),
    BusContract(
        msg_type="MEMORY_TOPOLOGY_UPDATED",
        schema=MemoryTopologyEvent,
        producer_module="memory_worker",
        consumer_modules=("api",),
        delivery="broadcast",
        max_payload_bytes=8_000,
        summary="Light topic-cluster heatmap (counts only, ~6-7 buckets). "
                "Bulk per-cluster sample texts via memory_proxy.get_topology().",
    ),
    BusContract(
        msg_type="MEMORY_KNOWLEDGE_GRAPH_UPDATED",
        schema=MemoryKnowledgeGraphEvent,
        producer_module="memory_worker",
        consumer_modules=("api",),
        delivery="broadcast",
        max_payload_bytes=8_000,
        summary="Knowledge-graph node/edge counts + entity-type histogram. "
                "Bulk graph data via memory_proxy.get_knowledge_graph(limit).",
    ),
)


# ── Lookup helpers ─────────────────────────────────────────────────────


_BY_MSG_TYPE: dict[str, BusContract] = {c.msg_type: c for c in REGISTRY}


def get_contract(msg_type: str) -> BusContract | None:
    """Return the contract for msg_type, or None if no contract registered.

    Untrusted callers (broker boundary on receive side) MUST handle None
    by passing the message through untouched (legacy compat during
    migration window — full coverage rolls in via follow-up sessions).
    """
    return _BY_MSG_TYPE.get(msg_type)


def all_contracted_msg_types() -> frozenset[str]:
    """For arch_map cross-reference — every msg type with a contract."""
    return frozenset(_BY_MSG_TYPE.keys())


# Sanity: every contract entry has a matching schema in EVENT_SCHEMAS.
for _c in REGISTRY:
    if EVENT_SCHEMAS.get(_c.msg_type) is not _c.schema:
        raise RuntimeError(
            f"BusContract registry inconsistency: {_c.msg_type} schema "
            f"in REGISTRY ({_c.schema.__name__}) does not match "
            f"EVENT_SCHEMAS entry "
            f"({EVENT_SCHEMAS.get(_c.msg_type, type(None)).__name__})"
        )
del _c
