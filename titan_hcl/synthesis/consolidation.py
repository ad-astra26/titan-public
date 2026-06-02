"""Dream-boundary consolidation pass (Phase 4 / §P4.G).

Per `ARCHITECTURE_synthesis_engine.md` §10.4 + INV-11:

  > "Consolidation runs in dream consolidation; each act is itself a
  > thought anchored on the chain ('on May 5 I synthesized X, Y, Z into
  > W, referencing these hashes'). Even Titan's self-reorganization is
  > verifiable."

This module owns the orchestration logic of one consolidation pass:

  1. Mine recent canonical TXs (window: last N hours; all forks except
     `meta` and `conversation` since those are noise sources for spine
     concepts — meta is for thoughts about thoughts, conversation is
     chat turns).
  2. Cluster by semantic similarity (cosine ≥ threshold) — the primary gate
     per PLAN_synthesis_engine_operator_closure §B4/W4 ("cluster by cosine, not
     tags-only"). Tag co-occurrence (Jaccard ≥ threshold, §10.4) is an additional
     anti-spurious filter applied ONLY when both TXs carry tags, so unrelated
     tagged TXs don't merge on a shared tag while untagged-but-semantically-near
     chain TXs still form concepts on cosine alone.
  3. For each cluster of size ≥ `min_cluster_size`: ask the LLM
     `propose(cluster)` → one of:
       - `new_concept`: materialize a brand-new spine concept (P4.B
         create_concept).
       - `version_bump`: bump an existing spine concept (P4.B bump_version).
       - `reject`: no coherent concept; skip the cluster.
  4. Apply each accepted proposal:
       * Register the concept_id with CGN (P4.C).
       * Create / bump via ConceptStore (P4.B) — which anchors the
         canonical concept-version TX via OuterMemoryWriter (P4.D) AND
         maintains COMPOSED_FROM / COMPOSED_INTO edges to the cluster's
         contributing concept_ids.
       * Recompute groundedness for the new version using the cluster
         stats (episodic_encounters = cluster size, etc.).
  5. Anchor ONE `consolidation_pass` canonical TX summarizing the pass —
     auditable per §10.4. Tags: ["consolidation_pass", "synthesis_worker",
     "dream_boundary"].

C.5 invariant: consolidation NEVER deletes a Concept row. "Merging two
concepts" = adding a new unified `Concept(v=1)` with COMPOSED_FROM edges
to both parents; parents stay queryable + on-chain.

INV-11 / G21: runs INSIDE synthesis_worker (sole writer of synthesis
substrate). The pass is triggered by a DREAM_STATE_CHANGED bus event
(synthesis_worker subscribes; this module is the pure orchestration
logic so it's testable without bus + Guardian).

The required `mine_recent_txs_fn` and `llm_propose_fn` are injected at
construction — production synthesis_worker wires the real implementations
(FORK_READ + Ollama-cloud); tests inject fakes.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from titan_hcl.synthesis.concept_store import (
    ConceptStore,
    ParentVersionMissing,
    WriterFailure,
)
from titan_hcl.synthesis.cgn_bridge import CGNRegistrationBridge
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryEvent, OuterMemoryWriter

logger = logging.getLogger(__name__)


# ── DTOs ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TxCandidate:
    """One canonical TX considered for consolidation."""

    tx_hash: str
    fork: str
    tags: tuple[str, ...]
    embedding: Optional[tuple[float, ...]]  # None → tag-only clustering
    content_summary: str = ""


@dataclass
class Cluster:
    """A semantically + tag-coherent group of TXs."""

    members: list[TxCandidate]
    centroid_tags: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class LLMProposal:
    """The LLM's verdict on what a cluster represents."""

    action: Literal["new_concept", "version_bump", "reject"]
    concept_id: Optional[str] = None
    proposed_name: Optional[str] = None
    memory_type: Optional[str] = None  # declarative|procedural|episodic|meta
    base_concept_refs: tuple[tuple[str, int], ...] = ()
    reason: str = ""


@dataclass
class ConsolidationResult:
    """Result of one ConsolidationPass.run() call — anchored as the
    summary TX + returned for observability."""

    pass_id: str
    started_at: float
    finished_at: float
    concepts_created: list[tuple[str, int]] = field(default_factory=list)
    concepts_bumped: list[tuple[str, int]] = field(default_factory=list)
    rejected_clusters: int = 0
    llm_calls: int = 0
    clusters_considered: int = 0
    txs_mined: int = 0
    duration_ms: float = 0.0
    pass_tx_hash: Optional[str] = None
    skipped: bool = False
    skip_reason: str = ""


# ── ConsolidationPass ───────────────────────────────────────────────


class ConsolidationPass:
    """Pure orchestration of one dream-boundary consolidation pass.

    Required injected callables:
      mine_recent_txs_fn(since_ts: float, exclude_forks: set[str])
          → list[TxCandidate]
      llm_propose_fn(cluster: Cluster) → LLMProposal

    Optional injected callable (defaults to numpy-free implementation):
      cosine_fn(a: tuple[float, ...], b: tuple[float, ...]) → float
    """

    def __init__(
        self,
        concept_store: ConceptStore,
        cgn_bridge: CGNRegistrationBridge,
        outer_memory_writer: OuterMemoryWriter,
        mine_recent_txs_fn: Callable[..., list[TxCandidate]],
        llm_propose_fn: Callable[[Cluster], LLMProposal],
        *,
        cosine_fn: Optional[Callable[
            [tuple[float, ...], tuple[float, ...]], float,
        ]] = None,
        clock: Callable[[], float] = time.time,
        # Tunables (mirror titan_params.toml defaults; can be overridden
        # in tests or via a future [synthesis.consolidation] subtable).
        window_hours: float = 24.0,
        excluded_forks: tuple[str, ...] = ("meta", "conversation"),
        cluster_cosine_threshold: float = 0.85,
        cluster_jaccard_threshold: float = 0.4,
        min_cluster_size: int = 3,
        max_concepts_per_pass: int = 10,
        llm_calls_max: int = 20,
        source: str = "synthesis_worker",
    ):
        self._store = concept_store
        self._bridge = cgn_bridge
        self._writer = outer_memory_writer
        self._mine = mine_recent_txs_fn
        self._propose = llm_propose_fn
        self._cosine = cosine_fn or _default_cosine
        self._clock = clock

        self._window_hours = window_hours
        self._excluded_forks = set(excluded_forks)
        self._cos_thresh = cluster_cosine_threshold
        self._jac_thresh = cluster_jaccard_threshold
        self._min_cluster = min_cluster_size
        self._cap_concepts = max_concepts_per_pass
        self._cap_llm = llm_calls_max
        self._source = source

    # ── Public API ──────────────────────────────────────────

    def run(self) -> ConsolidationResult:
        """Execute one consolidation pass + anchor the summary TX.

        Returns a ConsolidationResult. Never raises (per arch §10.4
        "auditable thought" — every pass produces a TX that records what
        happened, including a no-op pass). Internal errors are logged
        + reflected in the result (`rejected_clusters` increments for
        LLM errors; the pass continues with remaining clusters).
        """
        started_at = self._clock()
        pass_id = f"consolidation_{int(started_at * 1000)}"
        result = ConsolidationResult(
            pass_id=pass_id, started_at=started_at, finished_at=started_at,
        )

        # Step 1 — mine.
        since_ts = started_at - self._window_hours * 3600.0
        try:
            txs = list(self._mine(
                since_ts=since_ts, exclude_forks=self._excluded_forks,
            ))
        except Exception as e:
            logger.warning(
                "[ConsolidationPass] mine_recent_txs raised: %s — pass aborted",
                e,
            )
            result.skipped = True
            result.skip_reason = f"mine_failed: {e}"
            result.finished_at = self._clock()
            result.duration_ms = (result.finished_at - started_at) * 1000.0
            return result

        result.txs_mined = len(txs)
        if len(txs) < self._min_cluster:
            result.skipped = True
            result.skip_reason = (
                f"insufficient_txs (mined={len(txs)} < "
                f"min_cluster={self._min_cluster})"
            )
            result.finished_at = self._clock()
            result.duration_ms = (result.finished_at - started_at) * 1000.0
            self._anchor_pass_tx(result)
            return result

        # Step 2 — cluster.
        clusters = self._cluster_txs(txs)
        result.clusters_considered = len(clusters)

        # Step 3+4 — propose + apply.
        for cluster in clusters:
            if (len(result.concepts_created) + len(result.concepts_bumped)
                    >= self._cap_concepts):
                logger.info(
                    "[ConsolidationPass] %s cap_concepts=%d reached — "
                    "skipping remaining %d cluster(s)",
                    pass_id, self._cap_concepts,
                    result.clusters_considered - (
                        len(result.concepts_created)
                        + len(result.concepts_bumped)
                        + result.rejected_clusters
                    ),
                )
                break
            if result.llm_calls >= self._cap_llm:
                logger.info(
                    "[ConsolidationPass] %s llm_calls_max=%d reached — "
                    "skipping remaining clusters",
                    pass_id, self._cap_llm,
                )
                break

            try:
                proposal = self._propose(cluster)
                result.llm_calls += 1
            except Exception as e:
                logger.warning(
                    "[ConsolidationPass] llm_propose raised on cluster of %d: %s",
                    len(cluster.members), e,
                )
                result.rejected_clusters += 1
                continue

            if proposal.action == "reject":
                result.rejected_clusters += 1
                continue

            applied = self._apply_proposal(proposal, cluster, result)
            if not applied:
                result.rejected_clusters += 1

        # Step 5 — anchor the pass-summary TX.
        result.finished_at = self._clock()
        result.duration_ms = (result.finished_at - started_at) * 1000.0
        self._anchor_pass_tx(result)
        return result

    # ── Internal — clustering ──────────────────────────────

    def _cluster_txs(self, txs: list[TxCandidate]) -> list[Cluster]:
        """Greedy single-pass clustering. COSINE is the primary semantic gate
        (PLAN_synthesis_engine_operator_closure §B4/W4: "clusters by cosine
        (0.85) NOT tags-only → real concepts form" — now that real tx_hash-spine
        embeddings are filled in). Tag co-occurrence (Jaccard, §10.4 "both gates")
        is applied as an ADDITIONAL anti-spurious filter ONLY when BOTH TXs carry
        tags — the §10.4 intent ("unrelated TXs sharing a tag don't cluster on tag
        alone") is preserved for tagged TXs, while untagged-but-semantically-near
        chain TXs (most of them) still cluster on cosine alone. When an embedding
        is missing, fall back to tag-only at a high Jaccard (legacy safety).

        Only clusters of size ≥ min_cluster_size are returned. Output is
        size-DESC so the densest clusters get LLM budget first.
        """
        clusters: list[Cluster] = []
        for tx in txs:
            best: Optional[Cluster] = None
            for cluster in clusters:
                centroid = cluster.members[0]
                tx_tag_set = set(tx.tags)
                centroid_tag_set = set(centroid.tags)
                if tx.embedding is not None and centroid.embedding is not None:
                    # Primary: cosine semantic similarity (PLAN B4).
                    if self._cosine(tx.embedding, centroid.embedding) < self._cos_thresh:
                        continue
                    # Anti-spurious co-gate (§10.4) — only when BOTH are tagged.
                    if tx_tag_set and centroid_tag_set:
                        if _jaccard(tx_tag_set, centroid_tag_set) < self._jac_thresh:
                            continue
                    best = cluster
                    break
                # No embedding on one side → tag-only fallback at high Jaccard
                # so we never merge weakly-related TXs without semantic evidence.
                jac = _jaccard(tx_tag_set, centroid_tag_set)
                if jac >= max(0.8, self._jac_thresh):
                    best = cluster
                    break

            if best is None:
                clusters.append(Cluster(members=[tx], centroid_tags=set(tx.tags)))
            else:
                best.members.append(tx)
                best.centroid_tags.update(tx.tags)

        viable = [c for c in clusters if len(c.members) >= self._min_cluster]
        viable.sort(key=lambda c: -len(c.members))
        return viable

    # ── Internal — apply proposal ──────────────────────────

    def _apply_proposal(
        self,
        proposal: LLMProposal,
        cluster: Cluster,
        result: ConsolidationResult,
    ) -> bool:
        if proposal.action == "new_concept":
            return self._apply_new_concept(proposal, cluster, result)
        if proposal.action == "version_bump":
            return self._apply_version_bump(proposal, cluster, result)
        # Unknown action → treat as reject for safety.
        logger.warning(
            "[ConsolidationPass] unknown proposal action %r — rejecting",
            proposal.action,
        )
        return False

    def _apply_new_concept(
        self,
        proposal: LLMProposal,
        cluster: Cluster,
        result: ConsolidationResult,
    ) -> bool:
        if not proposal.concept_id or not proposal.proposed_name:
            logger.warning(
                "[ConsolidationPass] new_concept proposal missing concept_id "
                "or proposed_name — rejecting",
            )
            return False
        memory_type = proposal.memory_type or "meta"

        # 1. Register with CGN (idempotent; soft-fails on persistence).
        self._bridge.register_spine_concept(
            proposal.concept_id, proposal.proposed_name,
            seed_consumer="synthesis_engine",
        )

        # 2. Create via ConceptStore (anchors the TX via OuterMemoryWriter +
        #    inserts the Kuzu row + maintains composition edges).
        evidence = [m.tx_hash for m in cluster.members]
        try:
            cv = self._store.create_concept(
                concept_id=proposal.concept_id,
                name=proposal.proposed_name,
                memory_type=memory_type,
                composed_from=list(proposal.base_concept_refs),
                derivation_evidence=evidence,
            )
        except (WriterFailure, ValueError) as e:
            logger.warning(
                "[ConsolidationPass] create_concept(%s) failed: %s",
                proposal.concept_id, e,
            )
            return False

        # 3. Recompute groundedness from cluster stats.
        self._store.recompute_groundedness(
            cv.concept_id, cv.version,
            episodic_encounters=len(cluster.members),
            distinct_contexts=len(cluster.centroid_tags),
            procedural_links=len(proposal.base_concept_refs),
            felt_coverage=0.0,
        )

        result.concepts_created.append((cv.concept_id, cv.version))
        return True

    def _apply_version_bump(
        self,
        proposal: LLMProposal,
        cluster: Cluster,
        result: ConsolidationResult,
    ) -> bool:
        if not proposal.concept_id:
            logger.warning(
                "[ConsolidationPass] version_bump proposal missing "
                "concept_id — rejecting",
            )
            return False

        evidence = [m.tx_hash for m in cluster.members]
        try:
            cv = self._store.bump_version(
                concept_id=proposal.concept_id,
                composed_from=list(proposal.base_concept_refs),
                derivation_evidence=evidence,
            )
        except (ParentVersionMissing, WriterFailure) as e:
            logger.warning(
                "[ConsolidationPass] bump_version(%s) failed: %s",
                proposal.concept_id, e,
            )
            return False

        self._store.recompute_groundedness(
            cv.concept_id, cv.version,
            episodic_encounters=len(cluster.members),
            distinct_contexts=len(cluster.centroid_tags),
            procedural_links=len(proposal.base_concept_refs),
            felt_coverage=0.0,
        )

        result.concepts_bumped.append((cv.concept_id, cv.version))
        return True

    # ── Internal — anchor pass TX ──────────────────────────

    def _anchor_pass_tx(self, result: ConsolidationResult) -> None:
        """Anchor one canonical TX summarizing the pass (§10.4 auditable
        thought). Tags: ['consolidation_pass', 'synthesis_worker',
        'dream_boundary']. Even a fully-skipped pass anchors its summary —
        the absence of activity is also auditable history."""
        content = {
            "pass_id": result.pass_id,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "duration_ms": result.duration_ms,
            "txs_mined": result.txs_mined,
            "clusters_considered": result.clusters_considered,
            "concepts_created": [
                {"concept_id": c[0], "version": c[1]}
                for c in result.concepts_created
            ],
            "concepts_bumped": [
                {"concept_id": c[0], "version": c[1]}
                for c in result.concepts_bumped
            ],
            "rejected_clusters": result.rejected_clusters,
            "llm_calls": result.llm_calls,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
        }
        # Use OuterMemoryEvent directly (this isn't a concept-version TX,
        # it's a summary thought on the meta fork). emit() is the canonical
        # entry per INV-4.
        event = OuterMemoryEvent(
            fork="meta",
            thought_type="consolidation_pass",
            source=self._source,
            content=content,
            tags=["consolidation_pass", "synthesis_worker", "dream_boundary"],
            significance=0.5,
            novelty=0.3,
            coherence=0.9,
        )
        try:
            self._writer.emit(event)
            # Use the OuterMemoryWriter's content-hash machinery indirectly
            # — for visibility we record a hash of the content here so
            # observability + tests can correlate.
            import hashlib
            import json
            canonical = json.dumps(
                content, sort_keys=True, separators=(",", ":"),
            ).encode()
            result.pass_tx_hash = hashlib.sha256(canonical).hexdigest()
        except Exception as e:
            logger.warning(
                "[ConsolidationPass] pass-summary TX emit failed: %s", e,
            )
            # The pass results are still in `result`; just no chain anchor.


# ── Helpers ─────────────────────────────────────────────────────────


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def _default_cosine(
    a: tuple[float, ...], b: tuple[float, ...],
) -> float:
    """Numpy-free cosine; safe for any-length embeddings. Returns 0.0
    on zero-norm vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


__all__ = (
    "ConsolidationPass",
    "ConsolidationResult",
    "TxCandidate",
    "Cluster",
    "LLMProposal",
)
