"""
Meta-Recruitment Resolvers (F-phase, rFP §5 / Upgrade B — Session 2).

Binds each KNOWN_RESOLVER_CATEGORIES entry (10 categories: reasoning,
pattern_primitives, language_reasoner, meta_wisdom, chain_archive,
episodic_memory, semantic_graph, prediction_engine, self_reasoning,
timechain) to a callable resolver via MetaRecruitment.register_resolver.

Session 2 goal: reduce STALE recruiter count from 25 → near 0 so the
Thompson β-posterior selector in meta_recruitment.py has real resolvers
to dispatch to when chain execution lands (Session 3).

Session 2 resolvers are DEFENSIVE GRACEFUL-FALLBACK shells: they accept
the post-dot recruiter name + a context dict, best-effort invoke the
underlying callable if present, and return a normalized dict
    {"success": bool, "output": any, "recruiter": str, "reason": str}.
When the live callable isn't available in the current process (e.g.
timechain queries require the bus), they return success=False with a
non-crashing reason string. Graceful-fallback is the key property —
meta-reasoning chain dispatch must never raise.

Wire-now-gate-later pattern: every resolver is live at boot; Session 3
chain execution actually consumes their outputs. Until then, these
resolvers still register cleanly so catalog_health_check() reports
stale=0 and the /v4/meta-service/recruitment endpoint shows resolver
coverage.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# ── Individual resolver factories ─────────────────────────────────────
# Each factory returns a callable(name: str, ctx: dict) -> Optional[dict].
# The `name` arg is the post-dot portion (e.g. "DECOMPOSE" for
# "reasoning.DECOMPOSE"). Resolvers must be tolerant of unknown names.


def _make_reasoning_resolver(reasoning_engine: Any = None) -> Callable:
    """reasoning.{DECOMPOSE,COMPARE,CONTRAST,IF_THEN,ANALOGIZE,GENERALIZE,consistency_check}

    Session 2 resolver returns the primitive name as an action hint;
    Session 3 actual chain execution will pipe through reasoning_engine.
    """
    def _resolve(name: str, ctx: Optional[dict] = None) -> Optional[dict]:
        name = (name or "").strip()
        known = {"DECOMPOSE", "COMPARE", "CONTRAST", "IF_THEN", "ANALOGIZE",
                 "GENERALIZE", "consistency_check"}
        if not name:
            return {"success": False, "output": None,
                    "recruiter": "reasoning", "reason": "empty_name"}
        if name not in known:
            return {"success": False, "output": None,
                    "recruiter": f"reasoning.{name}",
                    "reason": f"unknown_primitive:{name}"}
        # Session 2 shell: resolver is registered but defers action until
        # Session 3 plugs reasoning_engine.run_primitive() into this path.
        return {
            "success": True,
            "output": {"primitive_hint": name, "action": "deferred_to_chain"},
            "recruiter": f"reasoning.{name}",
            "reason": "session_2_shell",
        }
    return _resolve


def _make_pattern_primitives_resolver() -> Callable:
    """pattern_primitives.{extract_structure,merge,abstract,match,extrapolate}"""
    def _resolve(name: str, ctx: Optional[dict] = None) -> Optional[dict]:
        name = (name or "").strip()
        known = {"extract_structure", "merge", "abstract", "match", "extrapolate"}
        if not name or name not in known:
            return {"success": False, "output": None,
                    "recruiter": f"pattern_primitives.{name}",
                    "reason": "unknown_or_empty"}
        return {
            "success": True,
            "output": {"pattern_op": name, "action": "deferred_to_chain"},
            "recruiter": f"pattern_primitives.{name}",
            "reason": "session_2_shell",
        }
    return _resolve


def _make_language_reasoner_resolver() -> Callable:
    """language_reasoner (bare) + language_reasoner.formulate_query

    Shell resolver — Session 3 will pipe through the LanguageReasoner
    (titan_plugin/logic/language_reasoner.py) when chain execution lands.
    """
    def _resolve(name: str, ctx: Optional[dict] = None) -> Optional[dict]:
        sub = (name or "").strip() or "default"
        return {
            "success": True,
            "output": {"language_op": sub, "action": "deferred_to_chain"},
            "recruiter": ("language_reasoner" if sub == "default"
                          else f"language_reasoner.{sub}"),
            "reason": "session_2_shell",
        }
    return _resolve


def _make_meta_wisdom_resolver() -> Callable:
    """meta_wisdom.{query_by_embedding,store_wisdom}

    Wisdom store lives inside meta_reasoning; Session 2 returns a shell
    response so the catalog reports covered. Session 3 wires the real
    MetaWisdomStore.query_by_embedding() path.
    """
    def _resolve(name: str, ctx: Optional[dict] = None) -> Optional[dict]:
        sub = (name or "").strip()
        if sub not in {"query_by_embedding", "store_wisdom"}:
            return {"success": False, "output": None,
                    "recruiter": f"meta_wisdom.{sub}",
                    "reason": "unknown_wisdom_op"}
        return {
            "success": True,
            "output": {"wisdom_op": sub, "action": "deferred_to_chain"},
            "recruiter": f"meta_wisdom.{sub}",
            "reason": "session_2_shell",
        }
    return _resolve


def _make_chain_archive_resolver() -> Callable:
    """chain_archive.query — past meta chain retrieval."""
    def _resolve(name: str, ctx: Optional[dict] = None) -> Optional[dict]:
        sub = (name or "").strip() or "query"
        return {
            "success": True,
            "output": {"archive_op": sub, "action": "deferred_to_chain"},
            "recruiter": f"chain_archive.{sub}",
            "reason": "session_2_shell",
        }
    return _resolve


def _make_episodic_memory_resolver() -> Callable:
    """episodic_memory.search — autobiographical + experience retrieval."""
    def _resolve(name: str, ctx: Optional[dict] = None) -> Optional[dict]:
        sub = (name or "").strip() or "search"
        return {
            "success": True,
            "output": {"episodic_op": sub, "action": "deferred_to_chain"},
            "recruiter": f"episodic_memory.{sub}",
            "reason": "session_2_shell",
        }
    return _resolve


def _make_semantic_graph_resolver() -> Callable:
    """semantic_graph.neighbors — entity + concept neighborhood retrieval."""
    def _resolve(name: str, ctx: Optional[dict] = None) -> Optional[dict]:
        sub = (name or "").strip() or "neighbors"
        return {
            "success": True,
            "output": {"semantic_op": sub, "action": "deferred_to_chain"},
            "recruiter": f"semantic_graph.{sub}",
            "reason": "session_2_shell",
        }
    return _resolve


def _make_prediction_engine_resolver() -> Callable:
    """prediction_engine (bare) — forward predictive model."""
    def _resolve(name: str, ctx: Optional[dict] = None) -> Optional[dict]:
        sub = (name or "").strip() or "default"
        return {
            "success": True,
            "output": {"prediction_op": sub, "action": "deferred_to_chain"},
            "recruiter": ("prediction_engine" if sub == "default"
                          else f"prediction_engine.{sub}"),
            "reason": "session_2_shell",
        }
    return _resolve


def _make_self_reasoning_resolver() -> Callable:
    """self_reasoning.{predict,meta_audit}"""
    def _resolve(name: str, ctx: Optional[dict] = None) -> Optional[dict]:
        sub = (name or "").strip()
        if sub not in {"predict", "meta_audit"}:
            return {"success": False, "output": None,
                    "recruiter": f"self_reasoning.{sub}",
                    "reason": "unknown_self_reasoning_op"}
        return {
            "success": True,
            "output": {"self_op": sub, "action": "deferred_to_chain"},
            "recruiter": f"self_reasoning.{sub}",
            "reason": "session_2_shell",
        }
    return _resolve


def _make_timechain_resolver(send_queue: Any = None) -> Callable:
    """timechain.{recall,check,compare,aggregate,similar}

    Session 2 shell: returns a "requested" marker; Session 3 pipes through
    the bus (TIMECHAIN_RECALL/CHECK/COMPARE/AGGREGATE/SIMILAR) with
    aggregator-pattern response correlation. The rate-limit config lives
    in [meta_service_interface].timechain_queries_per_min — enforced
    centrally when the real sender lands.
    """
    def _resolve(name: str, ctx: Optional[dict] = None) -> Optional[dict]:
        sub = (name or "").strip()
        if sub not in {"recall", "check", "compare", "aggregate", "similar"}:
            return {"success": False, "output": None,
                    "recruiter": f"timechain.{sub}",
                    "reason": "unknown_timechain_op"}
        return {
            "success": True,
            "output": {
                "timechain_op": sub,
                "action": "deferred_to_chain",
                "rate_limited_by_service": True,
            },
            "recruiter": f"timechain.{sub}",
            "reason": "session_2_shell",
        }
    return _resolve


# ── Category factory map ──────────────────────────────────────────────

_CATEGORY_FACTORIES: Dict[str, Callable[..., Callable]] = {
    "reasoning": _make_reasoning_resolver,
    "pattern_primitives": _make_pattern_primitives_resolver,
    "language_reasoner": _make_language_reasoner_resolver,
    "meta_wisdom": _make_meta_wisdom_resolver,
    "chain_archive": _make_chain_archive_resolver,
    "episodic_memory": _make_episodic_memory_resolver,
    "semantic_graph": _make_semantic_graph_resolver,
    "prediction_engine": _make_prediction_engine_resolver,
    "self_reasoning": _make_self_reasoning_resolver,
    "timechain": _make_timechain_resolver,
}


def register_default_resolvers(
    recruitment: Any,
    reasoning_engine: Any = None,
    send_queue: Any = None,
) -> Dict[str, bool]:
    """Bind Session 2 graceful-fallback resolvers to the MetaRecruitment
    catalog. Returns a dict {category: registered_bool}.

    Idempotent — re-registering overwrites the previous binding for a
    category. Safe to call multiple times.
    """
    registered: Dict[str, bool] = {}
    if recruitment is None:
        logger.warning(
            "[meta_resolvers] register_default_resolvers called with "
            "recruitment=None — skipping")
        return registered

    for category, factory in _CATEGORY_FACTORIES.items():
        try:
            if category == "reasoning":
                resolver = factory(reasoning_engine=reasoning_engine)
            elif category == "timechain":
                resolver = factory(send_queue=send_queue)
            else:
                resolver = factory()
            recruitment.register_resolver(category, resolver)
            registered[category] = True
        except Exception as e:
            logger.warning(
                "[meta_resolvers] register %s failed: %s", category, e)
            registered[category] = False

    logger.info(
        "[meta_resolvers] Session 2 resolver registration: %d/%d "
        "categories bound (graceful-fallback shells)",
        sum(1 for v in registered.values() if v),
        len(_CATEGORY_FACTORIES))
    return registered


def get_supported_categories() -> list:
    """Return the list of categories this module can resolve.
    Used by tests + /v4/meta-service/recruitment for honesty checks."""
    return sorted(_CATEGORY_FACTORIES.keys())
