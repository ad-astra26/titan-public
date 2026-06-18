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

    # ── Phase 6 — standalone OracleVerdict + companion batch (§P6.F / INV-Syn-12) ──

    def write_oracle_verdict_standalone(
        self,
        *,
        verdict,                       # OracleVerdict from titan_hcl/synthesis/plugs.py
        claim_domain: str,
        fork: str,
        significance: float = 0.5,
        novelty: float = 0.3,
        coherence: float = 0.7,
    ) -> str:
        """Anchor a single standalone OracleVerdict on the canonical fork
        determined by INV-Syn-12 (claim domain → fork; computed by the
        ``OracleRouter`` caller). Returns a deterministic SHA-256
        content-hash of the verdict payload (same shape as the other
        write_* methods).

        Distinct from `write_concept_version_with_proof`'s companion
        oracle_verdict TX (which is fork-graduation-tied per Phase 5).
        Standalone verdicts come from Phase 6's OracleRouter when no
        ``parent_tool_call_tx`` is supplied.

        Tags: ["oracle_verdict_standalone", "oracle:<id>",
               "domain:<claim_domain>", "verdict:<true|false|unknown>"]
        Content: full verdict payload + claim_domain + ts.
        """
        content = {
            "oracle_id": str(getattr(verdict, "oracle_id", "")),
            "verdict": str(getattr(verdict, "verdict", "unknown")),
            "evidence_ref": str(getattr(verdict, "evidence_ref", "")),
            "cost": float(getattr(verdict, "cost", 0.0)),
            "latency_ms": int(getattr(verdict, "latency_ms", 0)),
            "claim_domain": claim_domain,
            "ts": float(getattr(verdict, "ts", time.time())),
        }
        anchor_tx = _canonical_concept_content_hash(content)
        tags = [
            "oracle_verdict_standalone",
            f"oracle:{content['oracle_id']}",
            f"domain:{claim_domain}",
            f"verdict:{content['verdict']}",
        ]
        event = OuterMemoryEvent(
            fork=fork,
            thought_type="oracle_verdict",
            source=self._src,
            content=content,
            tags=tags,
            significance=significance,
            novelty=novelty,
            coherence=coherence,
        )
        self.emit(event)
        return anchor_tx

    def write_tool_call(
        self,
        *,
        tool_id: str,
        args: dict,
        success: bool,
        result_summary: str,
        result_full_hash: Optional[str] = None,
        latency_ms: int = 0,
        scored_by: Optional[str] = None,         # INV-Syn-15: "oracle"|"llm"|None
        parent_chat_tx: Optional[str] = None,    # links to the chat turn that triggered the tool call
        parent_goal: Optional[str] = None,
        parent_skill_id: Optional[str] = None,
        significance: float = 0.35,
        novelty: float = 0.25,
        coherence: float = 0.7,
    ) -> str:
        """Phase 6 / §P6.I — anchor a tool invocation as a procedural-fork TX.

        Every tool call routed through a `ToolPlug` lands here. The
        ``scored_by`` field is the §A.6 coverage instrumentation
        (INV-Syn-15):

          * ``"oracle"`` — a TruthOraclePlug.verify() returned a
            non-`unknown` verdict for this call's outcome (oracle-scored).
          * ``"llm"`` — Phase-8 dream-time skill miner's LLM-judge
            fallback supplied the success signal.
          * ``None`` — neither (unscored at write time; Phase 8 miner may
            score later via a follow-up TX).

        Phase 6 ships scored_by=None at call time + a companion oracle
        verdict path (P6.F OracleRouter) updates it via the companion
        verdict batch. Phase 8 wires the LLM-judge fallback.

        Returns the SHA-256 content-hash of the tool-call payload.

        Tags: ["tool_call", "tool:<tool_id>", "scored_by:<v>" (if set),
               "scored_by:none" (if unset)]
        Content: full invocation record + scored_by field.
        """
        content = {
            "tool_id": tool_id,
            "args": args,
            "success": bool(success),
            "result_summary": result_summary[:512] if result_summary else "",
            "latency_ms": int(latency_ms),
            "ts": time.time(),
            "scored_by": scored_by,
        }
        if result_full_hash:
            content["result_full_hash"] = result_full_hash
        if parent_chat_tx:
            content["parent_chat_tx"] = parent_chat_tx
        if parent_goal:
            content["parent_goal"] = parent_goal
        if parent_skill_id:
            content["parent_skill_id"] = parent_skill_id

        anchor_tx = _canonical_concept_content_hash(content)
        tags = [
            "tool_call",
            f"tool:{tool_id}",
            f"scored_by:{scored_by if scored_by else 'none'}",
        ]
        if parent_chat_tx:
            tags.append(f"chat:{parent_chat_tx[:16]}")
        event = OuterMemoryEvent(
            fork="procedural",
            thought_type="tool_call",
            source=self._src,
            content=content,
            tags=tags,
            significance=significance,
            novelty=novelty,
            coherence=coherence,
        )
        self.emit(event)
        return anchor_tx

    def write_presence_interaction(
        self,
        *,
        person_id: str,
        person_ref: str,
        evidence_strength: str,
        channel: str,
        age_epochs: int,
        ts: Optional[float] = None,
        significance: float = 0.45,
        novelty: float = 0.2,
        coherence: float = 0.8,
    ) -> str:
        """RFP_verifiable_autobiographical_presence_memory §7.A (CAPTURE) — anchor
        ONE verified/asserted person-presence interaction as an EPISODIC-fork TX:
        the autobiographical atom. **Content-addressed** (like write_concept_version):
        returns the deterministic SHA-256 hex content-hash (the `tx_hash`)
        immediately, so synthesis writes the `person_interactions` row without
        waiting for the chain seal — the same hash the timechain TX will carry.

        Titan-time: `age_epochs` is the time KEY; `ts` is human-display metadata
        only (INV-PAM-TITAN-TIME). `evidence_strength` ∈ {crypto_verified_maker,
        crypto_verified_device, asserted_identity} (INV-PAM-HONEST-GRADIENT) — carried
        end-to-end so recall never overstates certainty.

        Tags: ["presence_interaction", "person:<id>", "evidence:<strength>",
               "channel:<channel>"]
        """
        content = {
            "person_id": person_id,
            "person_ref": person_ref or "",
            "evidence_strength": evidence_strength,
            "channel": channel or "",
            "age_epochs": int(age_epochs),
            "ts": float(ts) if ts is not None else time.time(),
        }
        anchor_tx = _canonical_concept_content_hash(content)
        tags = [
            "presence_interaction",
            f"person:{person_id}",
            f"evidence:{evidence_strength}",
            f"channel:{channel or '?'}",
        ]
        event = OuterMemoryEvent(
            fork="episodic",
            thought_type="presence_interaction",
            source=self._src,
            content=content,
            tags=tags,
            significance=significance,
            novelty=novelty,
            coherence=coherence,
        )
        self.emit(event)
        return anchor_tx

    def write_presence_seal(
        self,
        *,
        cycle_id: int,
        age_epoch_range: list,
        merkle_root: str,
        person_rollups: list,
        interaction_count: int,
        ts_utc_range: Optional[list] = None,
        significance: float = 0.85,
        novelty: float = 0.5,
        coherence: float = 0.8,
    ) -> str:
        """RFP_verifiable_autobiographical_presence_memory §7.C (SEAL) — anchor ONE
        circadian-cycle presence seal as a FORK_MAIN (autobiographical spine) TX: the
        Merkle root over the closed cycle's interaction tx_hashes + the per-person
        rollups, keyed by Titan-time (``cycle_id`` + ``age_epoch_range``). A LOCAL
        immutable timechain block (Option A — no per-cycle Solana memo). An empty
        cycle still seals (``merkle_root`` = SHA-256(b"") via merkle_root_hex([]),
        ``person_rollups``=[]) — INV-PAM-NO-GAPS. ``significance`` 0.85 clears the
        main-fork PoT (0.20); it stays BELOW the 0.9 immediate-seal threshold so the
        seal rides the normal seal cadence (it does not coordinate block timing).

        Returns the deterministic SHA-256 content-hash (the seal's content-addressed
        reference; NOT the chain block_hash — Phase D anchors CHAINED on the
        TIMECHAIN_SEALED block, see autobiography_seal.py).

        Titan-time: ``age_epoch_range`` is the KEY; ``ts_utc_range`` is human-display
        metadata only (INV-PAM-TITAN-TIME).

        Tags: ["presence_seal", "cycle:<id>", "autobiography"]
        """
        content = {
            "cycle_id": int(cycle_id),
            "age_epoch_range": [int(age_epoch_range[0]), int(age_epoch_range[1])],
            "merkle_root": merkle_root,
            "person_rollups": list(person_rollups),
            "interaction_count": int(interaction_count),
        }
        if ts_utc_range is not None:
            content["ts_utc_range"] = [float(ts_utc_range[0]), float(ts_utc_range[1])]
        anchor_tx = _canonical_concept_content_hash(content)
        tags = ["presence_seal", f"cycle:{int(cycle_id)}", "autobiography"]
        event = OuterMemoryEvent(
            fork="main",
            thought_type="presence_seal",
            source=self._src,
            content=content,
            tags=tags,
            significance=significance,
            novelty=novelty,
            coherence=coherence,
        )
        self.emit(event)
        return anchor_tx

    def write_reasoning_composite(
        self,
        *,
        reasoning_id: str,
        goal_class: str,
        action: str,
        use_count: int = 1,
        composed_from: Optional[list] = None,
        significance: float = 0.6,
        novelty: float = 0.4,
        coherence: float = 0.85,
    ) -> str:
        """§7.D D.2 (FC-3 / INV-OML-6) — anchor a VERIFIED Reasoning composite
        (`Reasoning(kind='macro_strategy')`) as a procedural-fork TX at the Idea
        tier. Only verified macro-strategies are individually anchored; per-use
        `tool_use` leaves stay Merkle-snapshot-covered. Returns the SHA-256
        content-hash (= the `anchor_tx` pointer stored on the Reasoning record)."""
        content = {
            "reasoning_id": reasoning_id,
            "goal_class": goal_class,
            "action": action,
            "use_count": int(use_count),
            "idea_type": "procedural",         # FC-8 §6.2.8 — composite IS procedural Idea
            "composed_from": list(composed_from or []),
            "ts": time.time(),
        }
        anchor_tx = _canonical_concept_content_hash(content)
        tags = [
            "reasoning_composite",
            "macro_strategy",
            f"goal_class:{goal_class}",
            f"action:{action}",
            "idea_type:procedural",
        ]
        event = OuterMemoryEvent(
            fork="procedural",
            thought_type="reasoning_composite",
            source=self._src,
            content=content,
            tags=tags,
            significance=significance,
            novelty=novelty,
            coherence=coherence,
        )
        self.emit(event)
        return anchor_tx

    # ── Phase 8 writers (D-SPEC-PHASE8) ─────────────────────────────

    def write_llm_judge_batch(
        self,
        *,
        merkle_root: str,
        entries: list[dict],
        significance: float = 0.35,
        novelty: float = 0.2,
        coherence: float = 0.75,
    ) -> str:
        """Anchor ONE LLMJudgeBatch TX on the meta fork per INV-Syn-21
        (one batch per dream pass; Merkle-batched per INV-Syn-12 micro-event
        tier). Each entry: `{parent_tool_call_tx, verdict, rationale, version_tag}`.
        Returns the SHA-256 content-hash of the batch payload."""
        content = {
            "merkle_root": merkle_root,
            "n_entries": int(len(entries)),
            "entries": list(entries),
            "ts": time.time(),
        }
        anchor_tx = _canonical_concept_content_hash(content)
        tags = ["llm_judge_batch", f"merkle:{merkle_root[:16]}", "dream_boundary"]
        event = OuterMemoryEvent(
            fork="meta",
            thought_type="llm_judge_batch",
            source=self._src,
            content=content,
            tags=tags,
            significance=significance,
            novelty=novelty,
            coherence=coherence,
        )
        self.emit(event)
        return anchor_tx

    def write_scored_by_patch(
        self,
        *,
        entries: list[dict],
        significance: float = 0.25,
        novelty: float = 0.1,
        coherence: float = 0.7,
    ) -> str:
        """Anchor ONE ScoredByPatch TX on the meta fork — INV-Syn-21
        scored_by lifecycle TX. Each entry: `{parent_tool_call_tx, scored_by}`.
        OracleCoverage reads the meta fork for these patches to update the
        A.6 numerator on the next instrumentation tick."""
        content = {
            "n_entries": int(len(entries)),
            "entries": list(entries),
            "ts": time.time(),
        }
        anchor_tx = _canonical_concept_content_hash(content)
        tags = ["scored_by_patch", "llm_judge", "dream_boundary"]
        event = OuterMemoryEvent(
            fork="meta",
            thought_type="scored_by_patch",
            source=self._src,
            content=content,
            tags=tags,
            significance=significance,
            novelty=novelty,
            coherence=coherence,
        )
        self.emit(event)
        return anchor_tx

    def write_tool_call_score(
        self,
        *,
        parent_tool_call_tx: str,
        scored_by: str,
        significance: float = 0.2,
        novelty: float = 0.05,
        coherence: float = 0.7,
    ) -> str:
        """Anchor ONE tool_call_score TX on the PROCEDURAL fork (G1, AUDIT §5.3).

        The LLM judge's scored_by also lands in a meta-fork scored_by_patch TX
        (kept as audit), but that TX's per-entry `content` does NOT survive v2
        batch-sealing (the slim tx_summaries drop content), so the procedural
        tool_call_reader cannot overlay it. This per-call score TX rides the
        SAME procedural fork the reader already walks and carries the verdict in
        its TAGS — which DO survive v2 sealing: `scored_by:<v>` + the FULL
        64-hex `parent:<parent_tool_call_tx>` (NOT a truncated prefix — exact
        join key, no collisions). The reader collects these and overlays
        scored_by onto the matching tool_call record so the miner + coverage
        see llm-scored calls (INV-Syn-15 / INV-Syn-21)."""
        content = {
            "parent_tool_call_tx": parent_tool_call_tx,
            "scored_by": scored_by,
            "ts": time.time(),
        }
        anchor_tx = _canonical_concept_content_hash(content)
        tags = [
            "tool_call_score",
            f"scored_by:{scored_by}",
            f"parent:{parent_tool_call_tx}",
        ]
        event = OuterMemoryEvent(
            fork="procedural",
            thought_type="tool_call_score",
            source=self._src,
            content=content,
            tags=tags,
            significance=significance,
            novelty=novelty,
            coherence=coherence,
        )
        self.emit(event)
        return anchor_tx

    def write_skill_mining_pass(self, summary: dict) -> str:
        """Anchor ONE skill_mining_pass TX on the meta fork per dream window.
        `summary` carries the ProceduralMiner pass output (txs_scanned,
        clusters_recurrent, positive_skills_compiled, negative_skills_compiled,
        llm_calls, llm_failures, compiled_ids)."""
        content = dict(summary or {})
        content["ts"] = float(summary.get("ts") or time.time())
        anchor_tx = _canonical_concept_content_hash(content)
        tags = ["skill_mining_pass", "synthesis_worker", "dream_boundary"]
        event = OuterMemoryEvent(
            fork="meta",
            thought_type="skill_mining_pass",
            source=self._src,
            content=content,
            tags=tags,
            significance=0.35,
            novelty=0.2,
            coherence=0.75,
        )
        self.emit(event)
        return anchor_tx

    def write_skill_lifecycle_tx(
        self,
        *,
        skill_id: str,
        event_kind: str,  # "verified" | "rejected" | "soft_retired"
        reason: str = "",
        compiled_from: Optional[list] = None,
    ) -> str:
        """Anchor ONE META_SKILL_VERIFIED / REJECTED / SOFT_RETIRED TX on the
        meta fork per INV-Syn-20 / Q4. Lightweight by design — the canonical
        skill row + utility_score lives in `procedural_skills`."""
        content = {
            "skill_id": skill_id,
            "event_kind": event_kind,
            "reason": reason,
            "compiled_from": list(compiled_from or []),
            "ts": time.time(),
        }
        anchor_tx = _canonical_concept_content_hash(content)
        tags = [f"meta_skill_{event_kind}", f"skill:{skill_id[:24]}"]
        event = OuterMemoryEvent(
            fork="meta",
            thought_type=f"meta_skill_{event_kind}",
            source=self._src,
            content=content,
            tags=tags,
            significance=0.3,
            novelty=0.15,
            coherence=0.7,
        )
        self.emit(event)
        return anchor_tx

    def write_oracle_verdict_batch(
        self,
        *,
        fork: str,
        merkle_root: str,
        entries: list[dict],
        significance: float = 0.4,
        novelty: float = 0.2,
        coherence: float = 0.8,
    ) -> str:
        """Anchor an ``OracleVerdictBatch`` TX rolling up tool-call
        companion verdicts on the parent tool-call's fork (per
        INV-Syn-12 micro-event tier). One batch TX per fork per dream
        window, called from ``OracleRouter.flush_companion_batches()``.

        Returns the SHA-256 content-hash of the batch payload.

        Tags: ["oracle_verdict_batch", "merkle:<root[:16]>"]
        Content: ``{merkle_root, n_entries, entries, ts}`` — full
        per-entry list inlined (entries are small; ~100B each) so a
        future Titan can FORK_READ the batch and walk its leaves
        without separate CAS dereferencing.
        """
        content = {
            "merkle_root": merkle_root,
            "n_entries": int(len(entries)),
            "entries": list(entries),
            "ts": time.time(),
        }
        anchor_tx = _canonical_concept_content_hash(content)
        tags = [
            "oracle_verdict_batch",
            f"merkle:{merkle_root[:16]}",
        ]
        event = OuterMemoryEvent(
            fork=fork,
            thought_type="oracle_verdict_batch",
            source=self._src,
            content=content,
            tags=tags,
            significance=significance,
            novelty=novelty,
            coherence=coherence,
        )
        self.emit(event)
        return anchor_tx
