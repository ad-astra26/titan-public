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

Phase 4 extension (§P4.D): `write_concept_version()` emits the canonical concept-
version TX per arch §10 (versioned concepts as their own canonical Timechain TXs).
Returns a deterministic content-hash usable as the spine's anchor reference — see
the method docstring for the content-hash vs chain-block-hash distinction.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from titan_hcl import bus


# Fork routing for concept-version TX, per §10 + memory-type semantics.
# meta-class concepts (consolidations, syntheses) ride the meta fork.
_CONCEPT_VERSION_FORK_BY_MEMORY_TYPE = {
    "declarative": "declarative",
    "procedural": "procedural",
    "episodic":   "episodic",
    "meta":       "meta",
}


# §P4.D bound: cap composed_from list size on the TX to keep block sizes
# bounded under tiered anchoring (§16.1). Excess parents are dropped from
# the on-chain TX but persist in the Kuzu spine (the chain is canonical for
# the named parents; the spine carries the complete neighborhood).
_COMPOSED_FROM_CAP_ON_TX = 50


def _canonical_concept_content_hash(content: dict) -> str:
    """Deterministic SHA-256 over the canonical JSON of the concept-version
    content. Matches `Transaction.compute_hash`'s canonicalization style
    (sort_keys, separators) so the hash is stable across producers /
    languages. Returns hex (matches the existing chain `tx_hash` field
    format consumers expect — see `output_verifier.build_timechain_payload`
    `output_hash`)."""
    canonical = json.dumps(
        content, sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


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

    # ── Phase 4 — concept-version write path (§10 / §P4.D) ──────────────

    def write_concept_version(
        self,
        *,
        concept_id: str,
        version: int,
        name: str,
        memory_type: str,
        parent_version_tx: Optional[str],
        composed_from: list[tuple[str, int]],
        derivation_evidence: list[str],
        groundedness: float,
        derivation_merkle_root: Optional[str] = None,
        significance: float = 0.7,
        novelty: float = 0.5,
        coherence: float = 0.8,
    ) -> str:
        """Anchor a canonical concept-version TX per arch §10. Each v(n) bump
        is its own canonical Timechain TX referencing the parent version's
        anchor and the base-concept hashes it was built from.

        **Returns:** a deterministic SHA-256 content-hash (hex) of the
        concept-version payload. This is what the spine's `anchor_tx`
        column stores. It is **content-addressed** (arch §16.2) — the hash
        is stable regardless of which chain block ultimately includes the
        TX, so synthesis_worker can write the Kuzu row immediately without
        waiting for the chain commit round-trip. The chain block hash
        (assigned by timechain_worker on seal) carries its own identity;
        the link between the two is the `concept:<id>:v<n>` tag (queryable
        via FORK_READ).

        **TX shape (per PLAN §P4.D):**

        Tags: ["concept_version", "concept:<id>", "v:<n>", <memory_type>]
        Content:
          {concept_id, version, name, memory_type, parent_version_tx,
           composed_from: [{concept_id, version}, ...],
           derivation_evidence: [tx_hash, ...],
           groundedness, derivation_merkle_root, created_at}

        Fork routing: memory_type → declarative|procedural|episodic|meta.
        Unknown memory_type defaults to meta (preserves §17.3 "no back-door
        writer" — every write lands somewhere canonical).
        """
        if memory_type not in _CONCEPT_VERSION_FORK_BY_MEMORY_TYPE:
            raise ValueError(
                f"write_concept_version: invalid memory_type {memory_type!r} "
                f"(want one of {list(_CONCEPT_VERSION_FORK_BY_MEMORY_TYPE)})"
            )
        if version < 1:
            raise ValueError(
                f"write_concept_version: version must be >= 1, got {version}"
            )
        if version == 1 and parent_version_tx is not None:
            raise ValueError(
                "write_concept_version: v=1 must have parent_version_tx=None "
                "(genesis of a concept; INV-3)"
            )
        if version > 1 and parent_version_tx is None:
            raise ValueError(
                f"write_concept_version: v={version} requires parent_version_tx "
                "(non-genesis bump must link to predecessor)"
            )

        fork = _CONCEPT_VERSION_FORK_BY_MEMORY_TYPE[memory_type]
        created_at = time.time()

        # Cap composed_from on the TX (Kuzu spine carries full neighborhood;
        # chain payload stays bounded — tiered anchoring §16.1).
        composed_from_tx = [
            {"concept_id": p[0], "version": int(p[1])}
            for p in composed_from[:_COMPOSED_FROM_CAP_ON_TX]
        ]

        content = {
            "concept_id": concept_id,
            "version": int(version),
            "name": name,
            "memory_type": memory_type,
            "parent_version_tx": parent_version_tx,
            "composed_from": composed_from_tx,
            "derivation_evidence": list(derivation_evidence),
            "groundedness": float(groundedness),
            "derivation_merkle_root": derivation_merkle_root,
            "created_at": created_at,
        }
        anchor_tx = _canonical_concept_content_hash(content)

        tags = [
            "concept_version",
            f"concept:{concept_id}",
            f"v:{int(version)}",
            memory_type,
        ]

        event = OuterMemoryEvent(
            fork=fork,
            thought_type="concept_version",
            source=self._src,
            content=content,
            tags=tags,
            significance=significance,
            novelty=novelty,
            coherence=coherence,
        )
        self.emit(event)
        return anchor_tx

    # ── Phase 5 — hypothesis-fork graduation + tombstone (§P5.E / §P5.G) ─

    def write_concept_version_with_proof(
        self,
        *,
        concept_id: str,
        version: int,
        name: str,
        memory_type: str,
        parent_version_tx: Optional[str],
        composed_from: list[tuple[str, int]],
        derivation_evidence: list[str],
        groundedness: float,
        derivation_merkle_root: str,
        oracle_verdict: dict,
        significance: float = 0.85,
        novelty: float = 0.7,
        coherence: float = 0.9,
    ) -> tuple[str, str]:
        """Phase 5 graduation-path write — extends `write_concept_version` with
        a mandatory `derivation_merkle_root` (the Merkle root of the
        graduating hypothesis fork's exploration TXs, §P5.E) AND emits a
        companion `oracle_verdict` TX so the verification itself is anchored
        (arch §11.1: "the verdict itself is anchored on the Timechain —
        verification is provenanced, not just the attempt").

        Returns `(concept_anchor_tx, oracle_verdict_tx)`.

        Both TXs ride the same fork as the concept-version (declarative /
        procedural / episodic / meta) so retrieval that walks the concept
        spine surfaces the proof + verdict alongside the concept itself —
        the audit trail is co-located with the claim.

        Significance / novelty / coherence default *higher* than the
        Phase-4 `write_concept_version` because a graduated concept-version
        is *earned* (oracle-verified or used-counted ≥3) — semantically
        more valuable than a consolidation-pass-proposed bump.
        """
        # Step 1: anchor the concept-version TX with the Merkle root.
        anchor_tx = self.write_concept_version(
            concept_id=concept_id, version=version, name=name,
            memory_type=memory_type, parent_version_tx=parent_version_tx,
            composed_from=composed_from,
            derivation_evidence=derivation_evidence,
            groundedness=groundedness,
            derivation_merkle_root=derivation_merkle_root,
            significance=significance, novelty=novelty, coherence=coherence,
        )

        # Step 2: anchor the OracleVerdict TX. Per arch §11.1 every verdict
        # is on-chain so a future Titan can audit "WHY did v(n+1) graduate?".
        fork = _CONCEPT_VERSION_FORK_BY_MEMORY_TYPE[memory_type]
        verdict_content = {
            "concept_id": concept_id,
            "concept_version": int(version),
            "concept_anchor_tx": anchor_tx,
            "derivation_merkle_root": derivation_merkle_root,
            "oracle_id": str(oracle_verdict.get("oracle_id", "")),
            "verdict": str(oracle_verdict.get("verdict", "unknown")),
            "evidence_ref": str(oracle_verdict.get("evidence_ref", "")),
            "cost": float(oracle_verdict.get("cost", 0.0)),
            "latency_ms": int(oracle_verdict.get("latency_ms", 0)),
            "ts": float(oracle_verdict.get("ts", time.time())),
        }
        verdict_hash = _canonical_concept_content_hash(verdict_content)
        verdict_tags = [
            "oracle_verdict",
            f"concept:{concept_id}",
            f"v:{int(version)}",
            f"oracle:{verdict_content['oracle_id']}",
        ]
        verdict_event = OuterMemoryEvent(
            fork=fork,
            thought_type="oracle_verdict",
            source=self._src,
            content=verdict_content,
            tags=verdict_tags,
            significance=0.6,
            novelty=0.3,
            coherence=0.9,
        )
        self.emit(verdict_event)
        return anchor_tx, verdict_hash

    def write_tombstone(
        self,
        *,
        fork_id: str,
        root_anchor: Optional[str],
        intent: str,
        explored_from: float,
        explored_to: float,
        exploration_root: str,
        abandonment_reason: str,
        reference_count_pruned: int,
        significance: float = 0.45,
        novelty: float = 0.25,
        coherence: float = 0.7,
    ) -> str:
        """Phase 5 abandonment — emits a permanent canonical TX on the
        `meta` fork carrying the scar of an abandoned hypothesis exploration.

        Arch §9.3 + INV-Syn-9 (proposed): the tombstone is **canonical** even
        though the fork's exploration TXs are not. It is the verifiable scar
        — a future Titan can ask "what did I once explore here?" and get a
        Merkle proof of the (now-deleted) TX list via `exploration_root`.

        Routed to the `meta` fork because tombstones are meta-cognitive
        events (records *about* Titan's exploration history), not domain
        knowledge in declarative/procedural/episodic. Same metabolic budget
        rule as other meta TXs (chain payload < 1 KB per arch §16.1).

        Returns the canonical content-hash of the tombstone TX (deterministic
        SHA-256 of the canonical JSON of the content payload, same shape as
        write_concept_version's return).

        `intent` is truncated to 256 chars on-chain to keep payload bounded;
        the full text is preserved in the DuckDB hypothesis_forks row pre-GC.
        """
        truncated_intent = intent[:256]
        content = {
            "fork_id": fork_id,
            "root_anchor": root_anchor,
            "intent": truncated_intent,
            "explored_from": float(explored_from),
            "explored_to": float(explored_to),
            "exploration_root": exploration_root,
            "abandonment_reason": abandonment_reason,
            "reference_count_pruned": int(reference_count_pruned),
            "created_at": time.time(),
        }
        tombstone_tx = _canonical_concept_content_hash(content)
        tags = [
            "fork_tombstone",
            f"fork:{fork_id}",
            f"reason:{abandonment_reason}",
        ]
        if root_anchor:
            tags.append(f"root:{root_anchor[:16]}")
        event = OuterMemoryEvent(
            fork="meta",
            thought_type="fork_tombstone",
            source=self._src,
            content=content,
            tags=tags,
            significance=significance,
            novelty=novelty,
            coherence=coherence,
        )
        self.emit(event)
        return tombstone_tx
