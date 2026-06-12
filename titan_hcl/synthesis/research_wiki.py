"""research_wiki — DK.1: the sovereign LLM-Wiki research→declarative-concept seed.

RFP_synthesis_self_learning_meta_reasoning §7.D-knowledge (Phase D, Maker-locked
2026-06-12). Closes the *compounding* loop: a confirmed `acquired:research`
finding becomes a durable, anchored, **declarative `Engram` concept** — Titan's
own verifiable "wiki page" — so the next time the topic is asked he answers
`direct` from his own knowledge (DK.4) instead of re-researching it (no SOL).

WHY a dedicated continuous seed and NOT the dream `ConsolidationPass` clustering:
the dream pass only materializes a concept when ≥`min_cluster_size` cosine-near
siblings cluster (`consolidation._cluster_txs`), so a **solo** research finding
stays stranded as episodic, recalled weakly, re-researched later (the compounding
leak). DK.1 forms the concept **continuously at the EEL-A confirm event**
(INV-OML-12 — learning is continuous, not dream-gated), even solo.

THE SOVEREIGNTY GATE (GD10 / INV-OML-1 / INV-OML-10): the LLM is **librarian,
never author**. It only *names* the concept over the *already-verified* finding
(EEL-A confirmed research, the confirm-gate upstream in agno) — it never decides
whether the fact enters (the confirm did) and never supplies the fact's content.
`memory_type` is FORCED `declarative` (a research fact); `derivation_evidence` is
the finding's own anchored `tx_hash`, so a later wiki hit derefs to a real chain
record (INV-OML-10/11). A naming failure must NEVER drop verified knowledge — it
falls back to a deterministic name, the fact still persists.

This module is the PURE seed logic (no bus, no threads). The synthesis_worker
RESEARCH_CONCEPT_SEED handler drives it from a small daemon (the LLM name call
must run OFF the recv-loop AND OFF the writer thread — a blocking provider call
on the writer thread would freeze the heartbeat → the synthesis crash-loop). The
EngramStore writes are `@on_writer` and auto-route to the writer thread.

All spine writes route through `EngramStore`→`OuterMemoryWriter` (INV-Syn-7/19/28,
the single canonical write path); no parallel store, no markdown KB (the storage
ontology resolved substrate-native, RFP §7.D STORAGE ONTOLOGY block).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from titan_hcl.synthesis.consolidation import Cluster, LLMProposal, TxCandidate

logger = logging.getLogger(__name__)

# A research concept's name/id come from the LLM librarian over the verified
# finding. The proposer prompt + parser are reused from consolidation_defaults
# (`make_default_llm_propose` → `LLMProposal`) so naming is consistent with the
# dream-pass concepts. The fork label marks the 1-member cluster as research.
_RESEARCH_TAG = "acquired:research"

# Deterministic fallback id/name when the LLM declines/fails to name the finding
# (the fact is verified — never drop it over a naming hiccup). Slug = the first
# salient tokens of the finding, lowercased + underscored, capped.
_STOPWORDS = frozenset((
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "what", "which", "who", "does", "do", "did", "how", "why",
    "with", "that", "this", "it", "its", "as", "by", "at", "be", "can", "you",
))


def _slug_tokens(text: str, *, max_tokens: int = 4) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    out = [t for t in toks if t not in _STOPWORDS and len(t) >= 3]
    return out[:max_tokens] or toks[:max_tokens]


def fallback_concept_name(content: str) -> tuple[str, str]:
    """Deterministic (concept_id, name) from the finding content — the librarian
    fallback (LLM declined / failed). concept_id = `research_<tok_tok>` (lowercase
    + underscores; the chain tag space treats concept_ids as identifiers); name =
    a Title-Cased phrase. Empty content → a stable generic id (still seeds, never
    raises)."""
    toks = _slug_tokens(content)
    if not toks:
        return ("research_finding", "Research Finding")
    cid = "research_" + "_".join(toks)
    name = " ".join(t.capitalize() for t in toks)
    return (cid[:64], name[:120])


def make_research_name_fn(propose_fn: Optional[Callable[[Cluster], LLMProposal]]
                          ) -> Callable[[str], tuple[str, str, str]]:
    """Bind the consolidation LLM proposer into a `name_fn(content) ->
    (concept_id, name, domain_hint)` for DK.1. The proposer NAMES a 1-member
    research cluster; `memory_type` from the LLM is ignored (DK.1 forces
    `declarative`). A reject / empty id / missing proposer → the deterministic
    fallback. The blocking provider call runs wherever the caller invokes
    name_fn — the synthesis daemon, OFF the writer + recv-loop threads."""

    def _name_fn(content: str) -> tuple[str, str, str]:
        if propose_fn is not None:
            try:
                cluster = Cluster(
                    members=[TxCandidate(
                        tx_hash="", fork="declarative",
                        tags=(_RESEARCH_TAG,), embedding=None,
                        content_summary=content)],
                    centroid_tags={_RESEARCH_TAG},
                )
                proposal = propose_fn(cluster)
                cid = (proposal.concept_id or "").strip()
                if proposal.action != "reject" and cid:
                    name = (proposal.proposed_name or cid).strip()
                    return (cid, name, (proposal.domain_hint or "").strip())
            except Exception as e:  # noqa: BLE001
                logger.debug("[research_wiki] LLM name proposal failed: %s", e)
        cid, name = fallback_concept_name(content)
        return (cid, name, "")

    return _name_fn


def seed_research_concept(
    *,
    engram_store: Any,
    cgn_bridge: Any,
    tx_hash: str,
    content: str,
    name_fn: Callable[[str], tuple[str, str, str]],
    domain_hint: str = "",
    felt_coverage: float = 0.0,
) -> Optional[Any]:
    """DK.1 — seed (or refine) the declarative `Engram` concept for one confirmed
    research finding. Returns the created/bumped `Engram` (v=n) or ``None`` on a
    soft failure (never raises — a librarian hiccup must not break the chat/
    promotion path that triggered it).

    Sequence (mirrors `ConsolidationPass._apply_new_concept`, single finding):
      1. `name_fn(content)` — the LLM librarian proposes (concept_id, name,
         domain_hint) over the *verified* finding; deterministic fallback on fail.
      2. `cgn_bridge.register_spine_concept(concept_id, name)` (idempotent).
      3. dedup: `engram_store.latest_concept(concept_id)` exists → `bump_version`
         (refinement, INV-OML-5 mutate-not-update); else `create_concept` v=1.
         Either way `memory_type='declarative'`, `derivation_evidence=[tx_hash]`
         (the anchored finding → deref-able, INV-OML-10/11).
      4. `recompute_groundedness` so the fresh concept ranks in recall (DK.4).

    `tx_hash` MUST be the finding's anchored per-node hash (from memory_worker's
    `_anchor_promoted_node`); an empty tx_hash means no verifiable evidence — we
    refuse to seed (sovereignty: no concept without a deref target, INV-OML-10)."""
    if not tx_hash:
        logger.warning(
            "[research_wiki] DK.1 seed refused: empty tx_hash (no deref "
            "target → would break INV-OML-10); content=%.50r", content)
        return None
    if not (content or "").strip():
        return None

    concept_id, name, llm_domain = name_fn(content)
    domain_hint = (domain_hint or llm_domain or "").strip().lower()

    try:
        cgn_bridge.register_spine_concept(
            concept_id, name, seed_consumer="research_wiki")
    except Exception as e:  # noqa: BLE001
        logger.debug("[research_wiki] register_spine_concept(%s) soft-fail: %s",
                     concept_id, e)

    try:
        existing = engram_store.latest_concept(concept_id)
    except Exception as e:  # noqa: BLE001
        logger.debug("[research_wiki] latest_concept(%s) probe failed: %s — "
                     "treating as new", concept_id, e)
        existing = None

    try:
        if existing is None:
            cv = engram_store.create_concept(
                concept_id=concept_id,
                name=name,
                memory_type="declarative",
                derivation_evidence=[tx_hash],
                domain_hint=domain_hint,
            )
            _action = "create"
        else:
            cv = engram_store.bump_version(
                concept_id=concept_id,
                derivation_evidence=[tx_hash],
                domain_hint=domain_hint,
            )
            _action = "bump(v%d)" % cv.version
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[research_wiki] DK.1 %s for %s failed: %s",
            "bump" if existing is not None else "create", concept_id, e)
        return None

    # Groundedness: one verified research encounter, one context. Gives the
    # fresh concept a non-zero rank base so DK.4 concept-recall can surface it
    # (the dream-boundary population recompute refines it later). felt rides if
    # the finding carried a felt-at-lived-time snapshot (§7.C). axis_verified is
    # structurally 0 on thought-Engrams (§6.2.3 spine-partition); the concept's
    # verifiability is the derivation_evidence deref, not a verified axis.
    try:
        engram_store.recompute_groundedness(
            cv.concept_id, cv.version,
            episodic_encounters=1,
            distinct_contexts=1,
            procedural_links=0,
            felt_coverage=float(felt_coverage or 0.0),
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("[research_wiki] recompute_groundedness(%s) soft-fail: %s",
                     concept_id, e)

    logger.info(
        "[research_wiki] DK.1 %s declarative concept %s v%d "
        "(evidence=%s…, domain=%s)",
        _action, cv.concept_id, cv.version, str(tx_hash)[:12],
        domain_hint or "-")
    return cv


_SUMMARY_PREFIX = "summary::"


def compose_concept_summaries(
    *,
    engram_store: Any,
    cgn_bridge: Any,
    min_children: int = 2,
    max_summaries_per_pass: int = 3,
    name_fn: Optional[Callable[[str], tuple[str, str, str]]] = None,
) -> int:
    """DK.2 (§7.D-knowledge) — concept-of-concepts (the "wiki index"). Survey BASE
    declarative `Engram` concepts, group by `domain_hint`, and compose a SUMMARY
    `Engram` whose `composed_from` = the domain's child concepts — a higher-order
    page that cites pages. Idempotent via the `summary::{domain}` concept_id
    convention: an existing summary is **bumped** (refreshed with the current
    children, INV-OML-5 mutate-not-update), else **created**. Summary concepts are
    themselves declarative but are EXCLUDED from the base set (the `summary::`
    prefix), so there is no infinite re-summarization. The LLM is librarian — it
    may NAME the summary over the *already-verified* child concepts; it never
    authors a fact (GD10). Returns # summaries created/bumped. Soft — never raises.

    Rides a periodic survey (the research-wiki daemon throttle): curation over the
    existing population, not primary fact capture, so a population-scan cadence
    (not every tick) is correct."""
    made = 0
    try:
        concepts = engram_store.list_declarative_concepts(limit=400)
    except Exception as e:  # noqa: BLE001
        logger.debug("[research_wiki] DK.2 list failed: %s", e)
        return 0

    by_domain: dict[str, list] = {}
    for c in concepts or []:
        cid = str(c.get("concept_id") or "")
        if not cid or cid.startswith(_SUMMARY_PREFIX):
            continue  # skip existing summaries → no infinite re-summarization
        dom = (c.get("domain_hint") or "").strip().lower()
        if not dom:
            continue
        by_domain.setdefault(dom, []).append(c)

    for dom, children in sorted(by_domain.items(), key=lambda kv: -len(kv[1])):
        if made >= int(max_summaries_per_pass):
            break
        if len(children) < int(min_children):
            continue
        summary_id = _SUMMARY_PREFIX + re.sub(r"[^a-z0-9_]+", "_", dom)[:48]
        child_refs = [(str(c["concept_id"]), int(c["version"])) for c in children]
        evidence = [str(c.get("anchor_tx") or "")
                    for c in children if c.get("anchor_tx")]
        name = f"{dom.replace('_', ' ').title()} — Overview"
        if name_fn is not None:
            try:
                _cid, _nm, _dh = name_fn(
                    "Overview of: " + "; ".join(
                        str(c.get("name") or "") for c in children[:8]))
                if _nm:
                    name = _nm
            except Exception:  # noqa: BLE001
                pass
        try:
            cgn_bridge.register_spine_concept(
                summary_id, name, seed_consumer="research_wiki_index")
        except Exception:  # noqa: BLE001
            pass
        try:
            existing = engram_store.latest_concept(summary_id)
            if existing is None:
                engram_store.create_concept(
                    concept_id=summary_id, name=name, memory_type="declarative",
                    composed_from=child_refs, derivation_evidence=evidence,
                    domain_hint=dom)
            else:
                engram_store.bump_version(
                    concept_id=summary_id, composed_from=child_refs,
                    derivation_evidence=evidence, domain_hint=dom)
            made += 1
            logger.info(
                "[research_wiki] DK.2 concept-of-concepts %s %s "
                "composed_from=%d (domain=%s)",
                "created" if existing is None else "bumped",
                summary_id, len(child_refs), dom)
        except Exception as e:  # noqa: BLE001
            logger.debug("[research_wiki] DK.2 compose %s soft-fail: %s",
                         summary_id, e)
    return made


__all__ = (
    "seed_research_concept",
    "make_research_name_fn",
    "fallback_concept_name",
    "compose_concept_summaries",
)
