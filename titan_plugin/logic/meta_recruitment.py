"""
Meta-Recruitment Layer (F-phase, rFP §5 / Upgrade B).

Each of the 9 meta-reasoning primitives becomes a recruiter that dispatches
to already-stable lower-level machinery (NS programs, reasoning engine
primitives, memory systems, prediction engine, pattern primitives, TimeChain
primitives). When a primitive has multiple candidate recruiters, a β-posterior
selector biases toward higher expected value from recruitment-outcome history.

Key principle (per Maker decision):
    "Include everything; log visibility; don't blacklist."
    Health-log-not-blacklist policy — if a recruiter is broken, the bus +
    exception path surfaces it; we don't silently omit capability.

Session 1 scope (this commit):
    - Full RECRUITMENT_CATALOG per rFP §5.1
    - Boot-time _catalog_health_check() audits resolver coverage
    - β-posterior selector + update_outcome / get_stats
    - Resolver registry scaffold (Session 2 fills in real callable bindings)
    - recruitment_trace builder (consumed by meta_service response building)

Session 2+ will wire actual reasoning.DECOMPOSE / CREATIVITY etc. callables
into _resolvers so dispatch becomes functional. Session 1 keeps resolvers
empty so dry-run responses are honest about "not_yet_implemented".
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── The catalog (rFP §5.1) ──────────────────────────────────────────
# Keys: "PRIMITIVE.sub_mode"
# Values: list of recruiter IDs (dotted "category.name" or bare NS program)
#
# Special syntax: "<self:meta.xxx>" — recursive meta-chain, depth-gated via
# titan_params [meta_service_interface] delegate_recursion_max_depth
# (default 1 = no recursion; opt-in 2-3 per rFP §14.2).

RECRUITMENT_CATALOG: Dict[str, List[str]] = {
    # FORMULATE family ──────────────────────────────────────────────
    "FORMULATE.define": [
        "reasoning.DECOMPOSE",
        "language_reasoner.formulate_query",
        "pattern_primitives.extract_structure",
    ],
    "FORMULATE.refine": [
        "reasoning.COMPARE",
        "reasoning.CONTRAST",
    ],
    "FORMULATE.load_wisdom": [
        "meta_wisdom.query_by_embedding",
        "timechain.similar",
    ],
    "FORMULATE.compose_intersection": [
        "reasoning.DECOMPOSE",
        "pattern_primitives.merge",
    ],
    "FORMULATE.compose_union": [
        "pattern_primitives.merge",
    ],
    "FORMULATE.compose_difference": [
        "reasoning.CONTRAST",
        "reasoning.COMPARE",
    ],
    "FORMULATE.narrow_to_subset": [
        "reasoning.DECOMPOSE",
    ],
    "FORMULATE.generalize_from_instance": [
        "reasoning.GENERALIZE",
        "pattern_primitives.abstract",
    ],

    # RECALL family ─────────────────────────────────────────────────
    "RECALL.chain_archive": [
        "chain_archive.query",
        "timechain.recall",
    ],
    "RECALL.experience": [
        "episodic_memory.search",
        "INTUITION",
    ],
    "RECALL.entity": [
        "semantic_graph.neighbors",
    ],
    "RECALL.wisdom": [
        "meta_wisdom.query_by_embedding",
        "timechain.similar",
    ],
    "RECALL.episodic_specific": [
        "episodic_memory.search",
    ],
    "RECALL.semantic_neighbors": [
        "semantic_graph.neighbors",
    ],
    "RECALL.procedural_matching": [
        "chain_archive.query",
        "pattern_primitives.match",
    ],
    "RECALL.autobiographical_relevant": [
        "episodic_memory.search",
        "timechain.recall",
    ],

    # HYPOTHESIZE family ────────────────────────────────────────────
    "HYPOTHESIZE.generate": [
        "reasoning.IF_THEN",
        "CREATIVITY",
    ],
    "HYPOTHESIZE.refine": [
        "reasoning.COMPARE",
    ],
    "HYPOTHESIZE.compare": [
        "reasoning.COMPARE",
        "reasoning.CONTRAST",
    ],
    "HYPOTHESIZE.analogize_from": [
        "reasoning.ANALOGIZE",
        "CREATIVITY",
    ],
    "HYPOTHESIZE.contrast_with": [
        "reasoning.CONTRAST",
    ],
    "HYPOTHESIZE.propose_by_inversion": [
        "reasoning.CONTRAST",
        "INTUITION",
    ],
    "HYPOTHESIZE.extend_pattern": [
        "pattern_primitives.extrapolate",
        "prediction_engine",
    ],

    # EVALUATE family ───────────────────────────────────────────────
    "EVALUATE.check_progress": [
        "self_reasoning.predict",
        "prediction_engine",
        "timechain.compare",
    ],
    "EVALUATE.check_strategy": [
        "reasoning.consistency_check",
        "REFLECTION",
    ],
    "EVALUATE.check_resources": [
        "VIGILANCE",
        "METABOLISM",
    ],

    # SYNTHESIZE family ─────────────────────────────────────────────
    "SYNTHESIZE.combine": [
        "CREATIVITY",
        "pattern_primitives.merge",
        "INSPIRATION",
    ],
    "SYNTHESIZE.abstract": [
        "reasoning.GENERALIZE",
        "pattern_primitives.abstract",
    ],
    "SYNTHESIZE.rank": [
        "reasoning.COMPARE",
    ],
    "SYNTHESIZE.distill_save": [
        "meta_wisdom.store_wisdom",
    ],

    # BREAK family ──────────────────────────────────────────────────
    "BREAK.rewind_last": [
        "IMPULSE",
    ],
    "BREAK.rewind_to_checkpoint": [
        "self_reasoning.meta_audit",
        "REFLEX",
    ],
    "BREAK.restart_fresh": [
        "IMPULSE",
        "self_reasoning.meta_audit",
    ],

    # SPIRIT_SELF family — direct NS nudges ─────────────────────────
    "SPIRIT_SELF.boost_curiosity": ["CURIOSITY"],
    "SPIRIT_SELF.boost_focus":     ["FOCUS"],
    "SPIRIT_SELF.boost_calm":      ["GABA_nudge"],
    "SPIRIT_SELF.boost_energy":    ["DA_nudge", "NE_nudge"],
    "SPIRIT_SELF.release_tension": ["REFLEX"],

    # INTROSPECT family ─────────────────────────────────────────────
    "INTROSPECT.state_audit": [
        "self_reasoning.predict",
        "REFLECTION",
    ],
    "INTROSPECT.prediction": [
        "prediction_engine",
    ],
    "INTROSPECT.coherence_check": [
        "VIGILANCE",
        "self_reasoning.meta_audit",
    ],
    "INTROSPECT.vocabulary_probe": [
        "language_reasoner",
    ],
    "INTROSPECT.architecture_query": [
        "self_reasoning.meta_audit",
        "timechain.recall",
    ],
    "INTROSPECT.maker_alignment": [
        "self_reasoning.meta_audit",
        "EMPATHY",
    ],

    # DELEGATE — recursive meta-chain (depth-gated, rFP §14.2) ──────
    "DELEGATE.full_chain":   ["<self:meta.run_chain>"],
    "DELEGATE.quick_chain":  ["<self:meta.run_chain_short>"],
    "DELEGATE.biased_chain": ["<self:meta.run_chain_biased>"],
}


# Known resolver categories. Session 2 registers real resolvers; Session 1
# uses this list to audit catalog coverage and surface mismatches early.
KNOWN_RESOLVER_CATEGORIES = frozenset({
    "reasoning",
    "pattern_primitives",
    "language_reasoner",
    "meta_wisdom",
    "chain_archive",
    "episodic_memory",
    "semantic_graph",
    "prediction_engine",
    "self_reasoning",
    "timechain",
})

# NS programs are recruiters by bare-uppercase name (no dot).
KNOWN_NS_PROGRAMS = frozenset({
    "METABOLISM",
    "FOCUS",
    "IMPULSE",
    "INTUITION",
    "REFLEX",
    "CURIOSITY",
    "EMPATHY",
    "INSPIRATION",
    "CREATIVITY",
    "REFLECTION",
    "VIGILANCE",
})

# Neuromod nudges — special direct-to-NeuromodulatorSystem calls.
KNOWN_NEUROMOD_NUDGES = frozenset({
    "GABA_nudge",
    "DA_nudge",
    "NE_nudge",
    "5HT_nudge",
})

# Self-recursive meta-chain operators (DELEGATE paths).
SELF_PREFIX = "<self:"


# ── β-posterior selector ─────────────────────────────────────────────

class MetaRecruitment:
    """Catalog + β-posterior selector + outcome-driven learning.

    One instance per spirit_worker process.
    Thread-safe via an internal lock (posterior updates come from any
    request/response path).
    """

    def __init__(self, catalog: Optional[Dict[str, List[str]]] = None,
                 prior_alpha: float = 1.0, prior_beta: float = 1.0,
                 reward_midpoint: float = 0.0):
        self._catalog = dict(catalog or RECRUITMENT_CATALOG)
        self._prior_alpha = float(prior_alpha)
        self._prior_beta = float(prior_beta)
        # reward_midpoint — signed rewards are in [-1, +1]; map to [0, 1]
        # for Beta updates by adding (reward - midpoint + 1) / 2
        self._reward_midpoint = float(reward_midpoint)

        # Resolvers registered by Session 2+. Session 1 = empty → dry-run.
        # "category" → callable(name, ctx) → Optional[result_dict]
        self._resolvers: Dict[str, Callable[[str, dict], Optional[dict]]] = {}

        # Posterior state per (primitive, sub_mode, recruiter)
        self._alpha: Dict[Tuple[str, str, str], float] = {}
        self._beta: Dict[Tuple[str, str, str], float] = {}
        self._fire_count: Dict[Tuple[str, str, str], int] = {}
        self._last_reward: Dict[Tuple[str, str, str], float] = {}

        # Boot-time diagnostics
        self._stale_recruiters: List[str] = []   # populated by health check
        self._orphan_keys: List[str] = []        # primitives with no recruiters
        self._recruiter_categories: Dict[str, str] = {}  # id → category

        self._lock = threading.Lock()
        self._t_boot = time.time()

        self._classify_catalog()

    # ── Classification + health check ───────────────────────────────

    def _classify_catalog(self) -> None:
        """Bucket each catalog entry into categories for health auditing."""
        for key, recruiters in self._catalog.items():
            if not recruiters:
                self._orphan_keys.append(key)
                continue
            for rid in recruiters:
                self._recruiter_categories[rid] = self._classify_recruiter(rid)

    @staticmethod
    def _classify_recruiter(rid: str) -> str:
        """Return category tag: ns|neuromod|self|resolver:<cat>|unknown.

        Accepts two resolver forms per rFP §5.1:
            "category.name" → resolver:category with explicit name
            "category"      → resolver:category with default name
        Both map to the same resolver; name defaults to the category itself
        when no dot is present (resolver dispatcher handles the fallback).
        """
        if rid.startswith(SELF_PREFIX):
            return "self"
        if rid in KNOWN_NEUROMOD_NUDGES:
            return "neuromod"
        if rid in KNOWN_NS_PROGRAMS:
            return "ns_program"
        if "." in rid:
            cat = rid.split(".", 1)[0]
            if cat in KNOWN_RESOLVER_CATEGORIES:
                return f"resolver:{cat}"
            return f"resolver:unknown({cat})"
        # Bare resolver category name (e.g. "prediction_engine")
        if rid in KNOWN_RESOLVER_CATEGORIES:
            return f"resolver:{rid}"
        return "unknown"

    def catalog_health_check(self) -> dict:
        """Audit recruiter coverage. Call once at boot (or on /v4 request).

        Returns a report dict. Also logs STALE entries at WARNING so they
        surface in the brain log.
        """
        categories: Dict[str, int] = {}
        unknown: List[str] = []
        for rid, cat in self._recruiter_categories.items():
            categories[cat] = categories.get(cat, 0) + 1
            if cat.startswith("resolver:unknown"):
                unknown.append(rid)

        # Resolver coverage (Session 2 populates self._resolvers)
        covered_cats = set(self._resolvers.keys())
        missing_cats = sorted(KNOWN_RESOLVER_CATEGORIES - covered_cats)

        # Stale = resolver-backed but category has no resolver registered
        stale = [rid for rid, cat in self._recruiter_categories.items()
                 if cat.startswith("resolver:") and not cat.startswith("resolver:unknown")
                 and cat.split(":", 1)[1] not in covered_cats]
        self._stale_recruiters = stale

        if stale:
            logger.info(
                "[MetaRecruit] %d recruiter(s) currently STALE (no resolver "
                "registered for their category — expected in Session 1; "
                "Session 2 fills these in): %s",
                len(stale),
                sorted(set(stale))[:10])
        if unknown:
            logger.warning(
                "[MetaRecruit] %d recruiter(s) have unrecognized category — "
                "probable catalog typo: %s",
                len(unknown), unknown[:10])
        if self._orphan_keys:
            logger.warning(
                "[MetaRecruit] %d catalog key(s) have no recruiters: %s",
                len(self._orphan_keys), self._orphan_keys)

        return {
            "catalog_keys": len(self._catalog),
            "recruiters_total": sum(len(v) for v in self._catalog.values()),
            "categories": dict(categories),
            "resolver_categories_covered": sorted(covered_cats),
            "resolver_categories_missing": missing_cats,
            "stale_recruiter_count": len(stale),
            "stale_sample": sorted(set(stale))[:10],
            "unknown_recruiter_count": len(unknown),
            "unknown_sample": unknown[:10],
            "orphan_keys": list(self._orphan_keys),
            "boot_age_seconds": round(time.time() - self._t_boot, 1),
        }

    # ── Resolver registration (Session 2+) ──────────────────────────

    def register_resolver(self, category: str,
                          resolver_fn: Callable[[str, dict], Optional[dict]]
                          ) -> None:
        """Bind a category (e.g. "reasoning") to a resolver callable.

        resolver_fn is called as resolver_fn(name, ctx) where name is the
        post-dot portion (e.g. "DECOMPOSE" for "reasoning.DECOMPOSE") and
        ctx is the per-call context dict. Must return Optional[dict] with
        at minimum {"success": bool, "output": any}.

        Session 1 never calls this; Session 2 wires reasoning, pattern,
        prediction, self_reasoning etc.
        """
        if not callable(resolver_fn):
            raise ValueError(
                f"MetaRecruitment: resolver_fn for category={category!r} "
                f"must be callable")
        self._resolvers[category] = resolver_fn
        logger.info(
            "[MetaRecruit] resolver registered for category='%s'", category)

    # ── β-posterior selector ────────────────────────────────────────

    def select_recruiter(self, primitive: str, sub_mode: str,
                         rng: Optional[random.Random] = None
                         ) -> Optional[str]:
        """Pick a recruiter for (primitive, sub_mode) via Thompson sampling.

        Returns None if no recruiters are listed for this key.

        When multiple recruiters listed, each one's Beta(α, β) posterior is
        sampled; highest-sampled recruiter wins. Uniform prior Beta(1,1) on
        first call → explicit exploration; biases toward high-outcome
        recruiters as evidence accumulates.
        """
        key = f"{primitive}.{sub_mode}"
        recruiters = self._catalog.get(key, [])
        if not recruiters:
            return None
        if len(recruiters) == 1:
            return recruiters[0]

        r = rng or random
        best = None
        best_sample = -1.0
        with self._lock:
            for rid in recruiters:
                pkey = (primitive, sub_mode, rid)
                a = self._alpha.get(pkey, self._prior_alpha)
                b = self._beta.get(pkey, self._prior_beta)
                try:
                    sample = r.betavariate(a, b)
                except (ValueError, OverflowError):
                    # Degenerate params → fall back to uniform
                    sample = r.random()
                if sample > best_sample:
                    best_sample = sample
                    best = rid
        return best or recruiters[0]

    def update_outcome(self, primitive: str, sub_mode: str, recruiter: str,
                       outcome_reward: float) -> None:
        """Bump the (primitive, sub_mode, recruiter) Beta posterior.

        outcome_reward is SIGNED in [-1, +1] per rFP §4.6. Mapped to a
        fractional success signal for Beta updates:
            (reward + 1) / 2 ∈ [0, 1]
        Split between α (success weight) and β (failure weight) per call.
        """
        try:
            r = max(-1.0, min(1.0, float(outcome_reward)))
        except (TypeError, ValueError):
            return
        frac = (r + 1.0) / 2.0   # map signed → [0, 1]
        key = (primitive, sub_mode, recruiter)
        with self._lock:
            self._alpha[key] = self._alpha.get(key, self._prior_alpha) + frac
            self._beta[key] = self._beta.get(key, self._prior_beta) + (1.0 - frac)
            self._fire_count[key] = self._fire_count.get(key, 0) + 1
            self._last_reward[key] = r

    def get_stats(self, top_n: int = 10) -> dict:
        """Snapshot for /v4/meta-service/recruitment."""
        with self._lock:
            n_tracked = len(self._alpha)
            # Top-N fired recruiters (recently-evolved posteriors)
            fired = sorted(
                self._fire_count.items(),
                key=lambda kv: -kv[1])[:top_n]
            top = []
            for k, n in fired:
                prim, sub, rid = k
                a = self._alpha.get(k, self._prior_alpha)
                b = self._beta.get(k, self._prior_beta)
                mean = a / max(1e-9, (a + b))
                top.append({
                    "primitive": prim,
                    "sub_mode": sub,
                    "recruiter": rid,
                    "n_fired": n,
                    "alpha": round(a, 3),
                    "beta": round(b, 3),
                    "posterior_mean": round(mean, 4),
                    "last_reward": round(self._last_reward.get(k, 0.0), 3),
                })
        return {
            "posterior_tuples_tracked": n_tracked,
            "resolvers_registered": sorted(self._resolvers.keys()),
            "stale_recruiter_count": len(self._stale_recruiters),
            "top_fired": top,
        }

    # ── Catalog reads (read-only) ───────────────────────────────────

    def get_recruiters(self, primitive: str, sub_mode: str) -> List[str]:
        """Return the raw recruiter list for (primitive, sub_mode). Empty
        list if the key isn't in the catalog — consumers should treat that
        as "primitive not recruitment-wired"."""
        return list(self._catalog.get(f"{primitive}.{sub_mode}", []))

    def classify_recruiter(self, rid: str) -> str:
        """Public classifier — used by observability."""
        return self._recruiter_categories.get(rid, self._classify_recruiter(rid))

    def categories_present(self) -> Dict[str, int]:
        """Count of recruiters per category — for /v4/meta-service/recruitment."""
        out: Dict[str, int] = {}
        for cat in self._recruiter_categories.values():
            out[cat] = out.get(cat, 0) + 1
        return out
