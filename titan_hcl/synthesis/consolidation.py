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
       * Create / bump via EngramStore (P4.B) — which anchors the
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

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from titan_hcl.synthesis.engram_store import (
    EngramStore,
    ParentVersionMissing,
    WriterFailure,
)
from titan_hcl.synthesis.cgn_bridge import CGNRegistrationBridge
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryEvent, OuterMemoryWriter
# CGN-felt RFP — the shared, torch-free neuromod helpers (one source with cgn.py).
from titan_hcl.logic.cgn_types import FELT_FRAME_DIVERGENCE, felt_distance

logger = logging.getLogger(__name__)


# ── Felt-at-lived-time coverage (RFP_synthesis_engram_grounding §7.C / §6.2 Q2) ──

# The neuromod homeostatic centre — `Neuromodulator.initial_setpoint`
# (titan_hcl/logic/neuromodulator.py:115; bounded [0.3,0.7]). The live per-modulator
# dynamic setpoint is not available cross-process at consolidation, so the felt
# intensity is measured as deviation from this canonical centre (not a magic
# number — the documented setpoint default). Sanity-checked vs emotional_intensity.
_NEUROMOD_SETPOINT_CENTRE = 0.5
# felt-dict keys that are metadata, NOT neuromod levels (see cognitive_worker:2563).
_FELT_META_KEYS = frozenset({"emotion", "emotion_confidence", "dream_cycle", "ts"})

# Inner↔Outer Felt-Teaching Bridge §1.3 — a fresh felt candidate's confidence `c`
# starts low (probationary; CGN's value_net refines it over exposure, BRAIN-INV-4).
PROBATION_C = 0.05


def _parse_felt(felt: Any) -> dict:
    """Coerce a felt value (sidecar JSON string | dict | None) → dict. Soft-fail → {}."""
    if not felt:
        return {}
    if isinstance(felt, dict):
        return felt
    if isinstance(felt, str):
        try:
            d = json.loads(felt)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def _felt_magnitude(felt: dict) -> float:
    """Normalized felt intensity ∈ [0,1] = ‖levels − setpoint_centre‖ / max_dev over
    the numeric neuromod levels (metadata excluded). Key-agnostic (robust to
    5HT/5-HT naming). Empty / level-less felt → 0.0."""
    levels = [
        float(v) for k, v in felt.items()
        if k not in _FELT_META_KEYS and isinstance(v, (int, float))
    ]
    if not levels:
        return 0.0
    dev = math.sqrt(sum((x - _NEUROMOD_SETPOINT_CENTRE) ** 2 for x in levels))
    max_dev = math.sqrt(len(levels)) * _NEUROMOD_SETPOINT_CENTRE  # each |x-0.5| ≤ 0.5
    if max_dev <= 0.0:
        return 0.0
    return min(1.0, dev / max_dev)


def felt_coverage_from_members(members: list) -> float:
    """felt_coverage = (members_with_felt / total) × mean_magnitude  (§6.2 Q2:
    consistency × intensity). ∈ [0,1]. Members carry `.felt` (sidecar JSON)."""
    if not members:
        return 0.0
    mags = []
    for m in members:
        felt = _parse_felt(getattr(m, "felt", None))
        if felt:
            mag = _felt_magnitude(felt)
            if mag > 0.0:
                mags.append(mag)
    if not mags:
        return 0.0
    coverage = len(mags) / len(members)
    mean_magnitude = sum(mags) / len(mags)
    return coverage * mean_magnitude


def agg_felt(members: list) -> dict:
    """Representative lived felt-state for a cluster — the mean of each numeric
    neuromod level across the members that carried felt (metadata keys excluded).
    This is the felt-state the Engram's thoughts were lived under, seeded into the
    felt candidates (RFP_inner_outer_felt_teaching_bridge §1.2). Felt lives on the
    members (`TxCandidate.felt`), NOT on the Engram node. Felt-less cluster → {}."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for m in (members or []):
        felt = _parse_felt(getattr(m, "felt", None))
        for k, v in felt.items():
            if k in _FELT_META_KEYS or not isinstance(v, (int, float)):
                continue
            sums[k] = sums.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}


def _decompose_sample_from_members(
    members: list, max_samples: int = 5, max_chars: int = 240,
) -> str:
    """A bounded sample of the cluster's REAL thought content — the signal the
    decompose LLM names the Engram's constituent Objects from (mirrors
    consolidation_defaults._build_cluster_prompt's content sample)."""
    samples = []
    for m in (members or [])[:max_samples]:
        text = (getattr(m, "content_summary", "") or "").strip().replace("\n", " ")
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        samples.append(f"- {text}")
    return "\n".join(samples)


# ── DTOs ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TxCandidate:
    """One canonical TX considered for consolidation."""

    tx_hash: str
    fork: str
    tags: tuple[str, ...]
    embedding: Optional[tuple[float, ...]]  # None → tag-only clustering
    content_summary: str = ""
    felt: Optional[str] = None  # felt-at-lived-time JSON (neuromod_context); §7.C


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
    domain_hint: str = ""  # §7.F — advisory free-text domain (mutable; never gates)


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
        engram_store: EngramStore,
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
        recall_attribution: Optional[Any] = None,
        decompose_fn: Optional[Callable[[str, str], list[str]]] = None,
        felt_bridge: Optional[Any] = None,
        emit_candidate_fn: Optional[Callable[[dict], None]] = None,
        frame_divergence: float = FELT_FRAME_DIVERGENCE,
    ):
        self._store = engram_store
        self._bridge = cgn_bridge
        # §7.E.0 — per-Engram citation attribution (membership write + the live
        # `fluent` axis feed at the dream recompute). Optional/soft — None disables.
        self._attribution = recall_attribution
        # Inner↔Outer Felt-Teaching Bridge §7.1 — Engram(Idea)→Object decompose +
        # cache (off the hot path). Optional/soft — either None disables the bridge
        # (synthesis otherwise unaffected). `decompose_fn(name, sample) -> list[str]`.
        self._decompose = decompose_fn
        self._felt_bridge = felt_bridge
        # Bridge §7.3 — the bus handoff to felt_teaching_worker. `emit_candidate_fn(
        # payload: dict)` (synthesis_worker binds it to _send(...ENGRAM_FELT_CANDIDATE
        # ...)). None → the durable engram_felt_candidates row is still written (audit/
        # retry), just no live event.
        self._emit_candidate = emit_candidate_fn
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
        # CGN-felt RFP §1.2 UPGRADE — RMS felt_distance above this between a cluster's
        # lived felt and a grounded Object's centroid = a new felt frame (config-tunable).
        self._frame_divergence = frame_divergence

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

        # G6 (AUDIT §5.3): if NO candidate carried an embedding, the tx_hash
        # FAISS spine is unavailable → clustering degrades to tag-only, but
        # spine TXs have no tags → ~0 clusters → 0 concepts SILENTLY. Surface
        # the degraded pass as a health WARNING (not a silent debug) so the
        # operator can see + fix the FAISS wiring.
        if txs and not any(getattr(t, "embedding", None) is not None for t in txs):
            logger.warning(
                "[ConsolidationPass] %s — 0/%d candidate TXs carry an embedding "
                "(tx_hash FAISS spine unavailable?); clustering degraded to "
                "tag-only → expect ~0 concepts this pass. Check synth_vector_store.",
                pass_id, len(txs))

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

        # Step 4.5 — §7.D dream-boundary population recompute: percentile-blend
        # the grounding scalar across ALL Engrams so groundedness DISCRIMINATES
        # (replaces the old saturate-at-50 per-Engram scalar). Only when the
        # population changed this pass.
        if result.concepts_created or result.concepts_bumped:
            try:
                # §7.E.0 — feed the LIVE recall-citation rate into the `fluent` axis
                # + cache the fresh axes for the recall-event snapshots. Soft: a
                # missing/failed attribution falls back to pure §7.D behaviour.
                fluent_lookup = None
                axes_sink = None
                if self._attribution is not None:
                    try:
                        _fmap = self._attribution.fluent_map()
                        fluent_lookup = (
                            lambda cid, ver, _m=_fmap: _m.get((str(cid), int(ver))))
                        axes_sink = self._attribution.update_axes_cache
                    except Exception as _fa_err:
                        logger.debug(
                            "[ConsolidationPass] fluent feed unavailable: %s", _fa_err)
                self._store.recompute_population_groundedness(
                    fluent_lookup=fluent_lookup, axes_sink=axes_sink)
            except Exception as e:
                logger.warning(
                    "[ConsolidationPass] population groundedness recompute "
                    "failed: %s", e)

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

        # 2. Create via EngramStore (anchors the TX via OuterMemoryWriter +
        #    inserts the Kuzu row + maintains composition edges).
        evidence = [m.tx_hash for m in cluster.members]
        try:
            cv = self._store.create_concept(
                concept_id=proposal.concept_id,
                name=proposal.proposed_name,
                memory_type=memory_type,
                composed_from=list(proposal.base_concept_refs),
                derivation_evidence=evidence,
                domain_hint=proposal.domain_hint,  # §7.F advisory
            )
        except (WriterFailure, ValueError) as e:
            logger.warning(
                "[ConsolidationPass] create_concept(%s) failed: %s",
                proposal.concept_id, e,
            )
            return False

        # 3. Recompute groundedness + store the felt axis (§7.C). felt_coverage
        #    feeds the legacy scalar (w_f=0 → no scalar change yet) AND is stored
        #    to Engram.axis_felt so it's independently visible (Phase D consumes it).
        felt_coverage = felt_coverage_from_members(cluster.members)
        self._store.recompute_groundedness(
            cv.concept_id, cv.version,
            episodic_encounters=len(cluster.members),
            distinct_contexts=len(cluster.centroid_tags),
            procedural_links=len(proposal.base_concept_refs),
            felt_coverage=felt_coverage,
        )
        if felt_coverage > 0.0:
            logger.info(
                "[ConsolidationPass] Engram %s v%d axis_felt=%.4f (%d/%d members "
                "felt-laden)", cv.concept_id, cv.version, felt_coverage,
                sum(1 for m in cluster.members
                    if _parse_felt(getattr(m, "felt", None))),
                len(cluster.members))

        # §7.E.0 — persist the member tx_hash → Engram reverse-index (the
        # citation-attribution resolver source). Soft; never blocks the pass.
        if self._attribution is not None:
            self._attribution.record_membership(cv.concept_id, cv.version, evidence)

        # Bridge §7.1 — decompose Engram(Idea)→Objects + cache (off the hot path).
        # Phase 1 populates the cache; Phase 3 reads it for gap detection.
        self._maybe_decompose(cv, proposal, cluster)

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
                domain_hint=proposal.domain_hint,  # §7.F advisory
            )
        except (ParentVersionMissing, WriterFailure) as e:
            logger.warning(
                "[ConsolidationPass] bump_version(%s) failed: %s",
                proposal.concept_id, e,
            )
            return False

        felt_coverage = felt_coverage_from_members(cluster.members)
        self._store.recompute_groundedness(
            cv.concept_id, cv.version,
            episodic_encounters=len(cluster.members),
            distinct_contexts=len(cluster.centroid_tags),
            procedural_links=len(proposal.base_concept_refs),
            felt_coverage=felt_coverage,
        )
        if felt_coverage > 0.0:
            logger.info(
                "[ConsolidationPass] Engram %s v%d axis_felt=%.4f (bump; %d/%d "
                "members felt-laden)", cv.concept_id, cv.version, felt_coverage,
                sum(1 for m in cluster.members
                    if _parse_felt(getattr(m, "felt", None))),
                len(cluster.members))

        # §7.E.0 — persist the member tx_hash → Engram reverse-index for the new
        # version (latest-version-only credit resolves to it). Soft.
        if self._attribution is not None:
            self._attribution.record_membership(cv.concept_id, cv.version, evidence)

        # Bridge §7.1 — decompose Engram(Idea)→Objects + cache (off the hot path).
        # Phase 1 populates the cache; Phase 3 reads it for gap detection.
        self._maybe_decompose(cv, proposal, cluster)

        result.concepts_bumped.append((cv.concept_id, cv.version))
        return True

    # ── Internal — felt-teaching bridge (§7.1 decompose) ───

    def _maybe_decompose(
        self, cv: Any, proposal: LLMProposal, cluster: Cluster,
    ) -> Optional[list[str]]:
        """Bridge §7.1 + §7.3 (OFF the hot path) — decompose the Engram(Idea) into its
        constituent Object labels (cached by (concept_id, version) so a re-touched
        Engram never re-calls the LLM), then queue a propose-only felt candidate for
        every Object with NO CGN grounding (a gap). Returns the labels; None when the
        bridge is disabled or decompose failed. Soft — never blocks the pass."""
        if self._felt_bridge is None or self._decompose is None:
            return None
        try:
            objects = self._felt_bridge.get_cached_objects(
                cv.concept_id, cv.version)
            if objects is None:
                name = (proposal.proposed_name or proposal.concept_id
                        or cv.concept_id)
                sample = _decompose_sample_from_members(cluster.members)
                objects = self._decompose(name, sample) or []
                if objects:
                    self._felt_bridge.cache_objects(
                        cv.concept_id, cv.version, objects)
            # Phase 3 — gap detection → propose-only candidate queue + bus handoff.
            if objects:
                self._queue_felt_gaps(cv, proposal, cluster, objects)
            return objects
        except Exception as e:  # noqa: BLE001 — soft per INV-Syn-17
            logger.debug("[ConsolidationPass] decompose soft-fail: %s", e)
            return None

    def _queue_felt_gaps(
        self, cv: Any, proposal: LLMProposal, cluster: Cluster,
        objects: list[str],
    ) -> None:
        """For each decomposed Object with NO CGN felt-grounding (a gap), queue a
        propose-only §3.4 candidate seeded with the cluster's lived felt + emit the
        ENGRAM_FELT_CANDIDATE handoff (RFP §7.3). PROPOSE-ONLY — writes synthesis
        only, ZERO CGN writes (INV-Syn-ENG-4). The bus event fires ONCE per candidate
        (only on a fresh queue), not every dream.

        frame_dependent (CGN-felt RFP §1.2 UPGRADE): a grounded Object is no longer
        blindly skipped. The grounded-set now carries CGN's per-concept felt centroid
        (event-sourced on CGN_CONCEPT_GROUNDED — G18, no RPC), so we compare the
        cluster's lived felt against it: divergent (> frame_divergence) → queue a
        `frame_dependent` candidate (a NEW felt frame for an existing concept, F⊗C
        scoped to domain_hint); aligned — or no centroid yet (pre-felt grounding) →
        skip (genuinely redundant). Still propose-only (INV-Syn-ENG-4 — zero CGN writes)."""
        lived_felt = agg_felt(cluster.members)
        felt_json = json.dumps(lived_felt, sort_keys=True) if lived_felt else "{}"
        domain_hint = getattr(proposal, "domain_hint", "") or ""
        for obj in objects:
            status = "candidate"
            provenance = "engram_felt_gap"
            if self._felt_bridge.is_object_grounded(obj):
                gc = self._felt_bridge.grounded_felt(obj)
                if gc and felt_distance(lived_felt, gc) > self._frame_divergence:
                    status = "frame_dependent"        # divergent felt → a new frame
                    provenance = "engram_felt_frame"
                else:
                    continue  # grounded + aligned / no centroid yet → redundant, skip
            newly = self._felt_bridge.queue_candidate(
                object_label=obj, felt_state_json=felt_json, c=PROBATION_C,
                source_engram=cv.concept_id, source_version=cv.version,
                provenance=provenance, domain_hint=domain_hint,
                status=status)
            if newly and self._emit_candidate is not None:
                self._emit_candidate({
                    "object_label": obj,
                    "felt_state": lived_felt,
                    "source_engram": cv.concept_id,
                    "source_version": cv.version,
                    "domain_hint": domain_hint,
                    "status": status,
                })

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
            # G5 (AUDIT §5.3): correlate pass_tx_hash to the REAL chain anchor
            # by using the SAME canonicalization every OuterMemoryWriter
            # anchor_tx uses (was an inline duplicate that could silently drift
            # from the chain tx_hash convention). _canonical_concept_content_hash
            # == Transaction.compute_hash style == the tx_hash consumers expect.
            from titan_hcl.synthesis.outer_memory_writer import (
                _canonical_concept_content_hash,
            )
            result.pass_tx_hash = _canonical_concept_content_hash(content)
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
