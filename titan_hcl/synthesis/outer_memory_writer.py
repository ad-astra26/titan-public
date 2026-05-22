"""OuterMemoryWriter — the single semantic write path into outer memory.

Synthesis Engine Phase 0 / 0C (arch §16.3, INV-4). Producers stop hand-assembling
the "(pick classification) + (build payload) + (emit TIMECHAIN_COMMIT)" triple and
instead describe *what* they wrote as a typed `OuterMemoryEvent` and call the writer.

Scope (Phase 0): the **anchor** path — build + emit the canonical `TIMECHAIN_COMMIT`
payload (the chain write is already funnelled through `timechain_worker`; the 0A tier
classifier + 0B CAS slimming apply downstream at seal). The **substrate** write
(`TitanMemory.add_to_mempool` for chat-like events) folds in with the chat-producer
migration — deliberately omitted here so this is not a placeholder.

Ownership: a thin library in Phase 0 (the producers import + call it, D-P0-4); the
dedicated `synthesis_worker` adopts it in Phase 1 (INV-11). It does NOT create CGN
groundings — `knowledge_concepts`/CGN stays CGN's (INV-1).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from titan_hcl import bus


@dataclass
class OuterMemoryEvent:
    """A single thought written to outer memory. Producer-provided weighting — the
    writer computes nothing (no new significance/novelty logic)."""

    fork: str                       # conversation|episodic|procedural|declarative|meta
    thought_type: str
    source: str                     # producer id (e.g. "knowledge_research")
    content: dict
    tags: list = field(default_factory=list)
    significance: float = 0.3
    novelty: float = 0.1
    coherence: float = 0.5
    # Optional anchor fields — emitted only when provided, so the payload matches a
    # producer's existing shape exactly (parity for INV-14 migration).
    db_ref: Optional[str] = None
    neuromods: Optional[dict] = None
    chi_available: Optional[float] = None
    attention: Optional[float] = None
    i_confidence: Optional[float] = None
    chi_coherence: Optional[float] = None


class OuterMemoryWriter:
    """Builds + emits the canonical TIMECHAIN_COMMIT for an OuterMemoryEvent."""

    def __init__(self, send_queue, src: str):
        self._send_queue = send_queue
        self._src = src

    def build_payload(self, e: OuterMemoryEvent) -> dict:
        """The TIMECHAIN_COMMIT inner payload. Optional fields included only when set,
        so a migrated producer's payload is structurally identical to its old inline one."""
        payload: dict = {
            "fork": e.fork,
            "thought_type": e.thought_type,
            "source": e.source,
            "content": e.content,
            "significance": e.significance,
            "novelty": e.novelty,
            "coherence": e.coherence,
            "tags": e.tags,
        }
        if e.db_ref is not None:
            payload["db_ref"] = e.db_ref
        if e.neuromods is not None:
            payload["neuromods"] = e.neuromods
        if e.chi_available is not None:
            payload["chi_available"] = e.chi_available
        if e.attention is not None:
            payload["attention"] = e.attention
        if e.i_confidence is not None:
            payload["i_confidence"] = e.i_confidence
        if e.chi_coherence is not None:
            payload["chi_coherence"] = e.chi_coherence
        return payload

    def emit(self, e: OuterMemoryEvent) -> None:
        """Publish the anchor as a TIMECHAIN_COMMIT to the timechain worker."""
        self._send_queue.put({
            "type": bus.TIMECHAIN_COMMIT,
            "src": self._src,
            "dst": "timechain",
            "ts": time.time(),
            "payload": self.build_payload(e),
        })
