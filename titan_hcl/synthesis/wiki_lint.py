"""wiki_lint — DK.3 librarian rhythm: TTL re-verify · contradiction · orphan/decay.

RFP_synthesis_self_learning_meta_reasoning §7.D-knowledge DK.3 (pinned mechanic
M2–M4, 2026-06-13). A periodic health-check over Titan's declarative `Engram`
concepts (the sovereign LLM-Wiki) that rides the existing Contemplate/idle pass
(FC-6 metabolic-gated — the synthesis research-wiki daemon throttles it, NOT the
chat hot path). Three disciplines, all DURABLE (they move `groundedness`/
`axis_used`, the rank base the dream population-recompute re-blends from):

  • **M2 TTL re-verify** — a `volatile`-classified declarative concept (legacy
    over-anchored "current X" facts predate the Axis-1 discern gate) whose
    EMERGENT age ≥ `lifetime_epochs` is demoted below `recall_known_floor` via
    `recompute_groundedness` with zero inputs → DK.4 stops routing `direct` →
    the next ask re-researches (implicit re-verify; no explicit event in v1).
    Evergreen (durable) concepts are untouched.

  • **M3 contradiction** — same-domain declarative concepts paired by NAME-TOKEN
    JACCARD (concepts carry NO cosine embeddings — DK.4-proven; recall is
    groundedness×name-match), pairs over `contradiction_name_overlap` are sent to
    the LLM-librarian which ADJUDICATES YES/NO (it never rewrites the fact —
    GD10). On YES the LOWER-grounded concept is proportionally decayed. Capped.

  • **M4 orphan/decay** — a concept past `orphan_window_epochs` with NO recall
    citation (RecallAttribution `cited_count==0`) AND NO incoming `COMPOSED_FROM`
    consumer (nothing composes from it) is proportionally decayed. Capped.

PURE orchestration — no bus, no threads, no provider construction. All
collaborators (`engram_store`, `judge_fn`, recall counts, epoch) are injected, so
the pass is unit-testable in isolation (test_wiki_lint.py) and the synthesis
daemon owns the cadence + metabolic gate. Soft throughout: one bad concept never
aborts the pass. The LLM `judge_fn` is invoked HERE (daemon thread, off the
writer) — every EngramStore write it triggers auto-routes @on_writer.
"""
from __future__ import annotations

import logging
import re
from itertools import combinations
from typing import Any, Callable, Optional

from titan_hcl.synthesis.research_volatility import (
    DEFAULT_VOLATILE_LIFETIME_EPOCHS,
    age_epochs,
    classify_volatility,
    is_stale,
)

logger = logging.getLogger(__name__)

_SUMMARY_PREFIX = "summary::"

# Name-token stopwords for the M3 Jaccard pairing (mirrors research_wiki's slug
# stopwords — generic glue tokens must not inflate name overlap).
_NAME_STOPWORDS = frozenset((
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "with", "that", "this", "it", "its", "as", "by", "at", "be", "overview",
    "current", "latest", "research", "finding", "titan",
))


def _name_tokens(name: str) -> frozenset[str]:
    """Lowercased content tokens of a concept name (len ≥ 3, minus stopwords)."""
    toks = re.findall(r"[a-z0-9]+", (name or "").lower())
    return frozenset(t for t in toks if len(t) >= 3 and t not in _NAME_STOPWORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def run_wiki_lint(
    *,
    engram_store: Any,
    now_epochs: int,
    lifetime_epochs: float = DEFAULT_VOLATILE_LIFETIME_EPOCHS,
    recall_floor: float = 0.65,
    judge_fn: Optional[Callable[[str, str], bool]] = None,
    recall_counts: Optional[dict] = None,
    orphan_window_epochs: Optional[float] = None,
    contradiction_overlap: float = 0.5,
    decay_factor: float = 0.5,
    max_concepts: int = 200,
    max_contradiction_pairs: int = 8,
    max_orphans: int = 16,
) -> dict:
    """Run one DK.3 wiki-lint pass. Returns a stats dict:
    ``{scanned, stale, contradiction, contradiction_pairs_judged, orphan}``.

    `now_epochs` = the live emergent epoch (consciousness_age.bin::age_epochs); 0
    (cold-boot / no slot) → the TTL/orphan epoch gates are inert this pass (M0
    grandfather), only the cheap survey runs. `recall_counts` =
    `RecallAttribution.recall_counts_map()` ({(cid,ver):(surfaced,cited,ts)});
    None → the orphan pass treats every concept as never-cited but the
    `concept_has_consumers` + age gates still protect live/young concepts.
    `judge_fn(name_a,name_b)->bool` = the M3 LLM adjudicator; None → M3 skipped."""
    stats = {"scanned": 0, "stale": 0, "contradiction": 0,
             "contradiction_pairs_judged": 0, "orphan": 0}
    lifetime = float(lifetime_epochs or DEFAULT_VOLATILE_LIFETIME_EPOCHS)
    if orphan_window_epochs is None:
        orphan_window_epochs = 10.0 * lifetime
    recall_counts = recall_counts or {}

    try:
        concepts = engram_store.list_declarative_concepts(limit=int(max_concepts))
    except Exception as e:  # noqa: BLE001
        logger.debug("[wiki_lint] list_declarative_concepts failed: %s", e)
        return stats
    concepts = concepts or []
    stats["scanned"] = len(concepts)

    # ── M2 — TTL re-verify (volatile + stale + currently routable → demote) ──
    for c in concepts:
        try:
            cid = str(c.get("concept_id") or "")
            ver = int(c.get("version") or 0)
            name = str(c.get("name") or "")
            dom = str(c.get("domain_hint") or "")
            grounded = float(c.get("groundedness") or 0.0)
            created_epoch = float(c.get("created_epoch") or 0.0)
            if not cid or ver <= 0:
                continue
            if classify_volatility(name, dom) != "volatile":
                continue  # evergreen → never TTL'd
            if not is_stale(created_epoch, now_epochs, lifetime):
                continue  # not past half-life (or grandfathered)
            if grounded < float(recall_floor):
                continue  # already below the floor → DK.4 won't route direct
            # Demote below the floor (zero inputs → groundedness→0): DK.4 stops
            # routing `direct`; the next ask re-researches (implicit re-verify).
            engram_store.recompute_groundedness(
                cid, ver, episodic_encounters=0, distinct_contexts=0,
                procedural_links=0, felt_coverage=0.0)
            stats["stale"] += 1
            logger.info("[wiki_lint] M2 stale volatile concept demoted: %s v%d "
                        "(age≈%d epochs ≥ %d, was groundedness=%.3f)",
                        cid, ver, int(age_epochs(created_epoch, now_epochs)),
                        int(lifetime), grounded)
        except Exception as e:  # noqa: BLE001
            logger.debug("[wiki_lint] M2 concept soft-fail: %s", e)

    # ── M3 — contradiction (same-domain, name-Jaccard, LLM adjudication) ──
    if judge_fn is not None:
        by_domain: dict[str, list] = {}
        for c in concepts:
            cid = str(c.get("concept_id") or "")
            if not cid or cid.startswith(_SUMMARY_PREFIX):
                continue  # summaries are indexes, not factual assertions
            dom = str(c.get("domain_hint") or "").strip().lower()
            if not dom:
                continue  # need a shared domain to be a candidate pair
            by_domain.setdefault(dom, []).append(c)

        candidate_pairs: list[tuple[float, dict, dict]] = []
        for dom, members in by_domain.items():
            if len(members) < 2:
                continue
            for a, b in combinations(members, 2):
                jac = _jaccard(_name_tokens(str(a.get("name") or "")),
                               _name_tokens(str(b.get("name") or "")))
                if jac >= float(contradiction_overlap):
                    candidate_pairs.append((jac, a, b))
        # Highest-overlap pairs first (most likely about the same subject).
        candidate_pairs.sort(key=lambda t: t[0], reverse=True)

        for jac, a, b in candidate_pairs[: int(max_contradiction_pairs)]:
            try:
                stats["contradiction_pairs_judged"] += 1
                if not judge_fn(str(a.get("name") or ""), str(b.get("name") or "")):
                    continue
                # Decay the LOWER-grounded of the contradictory pair.
                weaker = a if (float(a.get("groundedness") or 0.0)
                               <= float(b.get("groundedness") or 0.0)) else b
                engram_store.decay_groundedness(
                    str(weaker["concept_id"]), int(weaker["version"]),
                    factor=float(decay_factor))
                stats["contradiction"] += 1
                logger.info("[wiki_lint] M3 contradiction (jaccard=%.2f): "
                            "%s ⟂ %s → decayed weaker %s",
                            jac, a.get("name"), b.get("name"),
                            weaker.get("concept_id"))
            except Exception as e:  # noqa: BLE001
                logger.debug("[wiki_lint] M3 pair soft-fail: %s", e)

    # ── M4 — orphan/decay (old + never cited + no consumers) ──
    n_orphaned = 0
    for c in concepts:
        if n_orphaned >= int(max_orphans):
            break
        try:
            cid = str(c.get("concept_id") or "")
            ver = int(c.get("version") or 0)
            if not cid or ver <= 0 or cid.startswith(_SUMMARY_PREFIX):
                continue  # never orphan the wiki-index summaries
            created_epoch = float(c.get("created_epoch") or 0.0)
            if age_epochs(created_epoch, now_epochs) < float(orphan_window_epochs):
                continue  # too young (or grandfathered) → protected
            cited = int((recall_counts.get((cid, ver)) or (0, 0, 0.0))[1])
            if cited > 0:
                continue  # recalled at least once → not an orphan
            if engram_store.concept_has_consumers(cid):
                continue  # something composes from it → kept
            engram_store.decay_groundedness(cid, ver, factor=float(decay_factor))
            stats["orphan"] += 1
            n_orphaned += 1
            logger.info("[wiki_lint] M4 orphan decayed: %s v%d "
                        "(age≈%d epochs, never cited, no consumers)",
                        cid, ver, int(age_epochs(created_epoch, now_epochs)))
        except Exception as e:  # noqa: BLE001
            logger.debug("[wiki_lint] M4 concept soft-fail: %s", e)

    if stats["stale"] or stats["contradiction"] or stats["orphan"]:
        logger.info("[wiki_lint] DK.3 pass: scanned=%d stale=%d "
                    "contradiction=%d/%d orphan=%d",
                    stats["scanned"], stats["stale"], stats["contradiction"],
                    stats["contradiction_pairs_judged"], stats["orphan"])
    return stats


__all__ = ("run_wiki_lint",)
