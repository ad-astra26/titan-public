"""
web_search helper — thin adapter over knowledge_router + knowledge_dispatcher.

Post-KP-3.5: the agno-tool that the agency layer calls no longer has its
own SearXNG/Firecrawl client. It goes through the same pipeline as
knowledge_worker: classify → cache → direct REST backends (Wiktionary,
Free Dictionary, Wikipedia) for structured queries, fetch_searxng_raw
(no Sage distillation) for conceptual / technical / news queries. Same
persistent cache, same internal-name rejection, same error taxonomy.

`raw_results=True` on the dispatch means WebSearchHelper skips Sage's
LLM distillation — the agency tool wants raw hits for its own caller to
interpret (typically the agno reasoning layer). knowledge_worker paths
that want distilled output continue to invoke Sage via dispatch().

Enriches: Mind Vision[0] (research freshness boost).
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class WebSearchHelper:
    """Web-search agno-tool. Agency asks, we dispatch.

    Compatibility: constructor signature preserved (searxng_url +
    firecrawl_api_key) so agency wiring doesn't change. firecrawl_api_key
    is accepted but no longer honoured — SearXNG/direct-REST via the
    router cover the same query space without the premium dependency. If
    a future rFP re-adds Firecrawl it would plug in as another backend.

    `budgets` (bytes/day per backend) must be forwarded by the caller so
    this process's HealthTracker has the same per-backend defaults as
    knowledge_worker. The two processes share
    data/knowledge_pipeline_health.json via atomic writes; if WebSearch's
    HealthTracker lacks defaults it can clobber knowledge_worker's correct
    budget values on save (last-writer-wins). Fix for BUG-KP-WEBSEARCH-
    HEALTH-DEFAULTS (2026-04-21).
    """

    def __init__(self,
                 searxng_url: str = "http://localhost:8080/search",
                 firecrawl_api_key: str = "",
                 cache_db_path: str = "data/search_cache.db",
                 cache_size_cap: int = 10_000,
                 health_path: str = "data/knowledge_pipeline_health.json",
                 decision_log_path: str = "data/logs/knowledge_router_decisions.jsonl",
                 budgets: Optional[Dict[str, int]] = None):
        self._searxng_url = searxng_url
        self._firecrawl_key = firecrawl_api_key  # preserved for introspection
        self._cache_db_path = cache_db_path
        self._cache_size_cap = cache_size_cap
        self._health_path = health_path
        self._decision_log_path = decision_log_path
        self._budgets: Dict[str, int] = dict(budgets) if budgets else {}
        self._cache = None
        self._health = None
        self._dispatch_inflight: dict = {}

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web and scrape pages for research"

    @property
    def capabilities(self) -> list[str]:
        return ["search", "scrape", "summarize"]

    @property
    def resource_cost(self) -> str:
        return "medium"

    @property
    def latency(self) -> str:
        return "medium"

    @property
    def enriches(self) -> list[str]:
        return ["mind"]

    @property
    def requires_sandbox(self) -> bool:
        return False

    # ── Lazy cache construction ─────────────────────────────────────

    def _ensure_cache(self):
        """Build KnowledgeCache on first use.

        Lazy construction avoids opening the SQLite handle at import
        time (matters because agno-tools are discovered eagerly by the
        agency layer at titan_main boot).
        """
        if self._cache is not None:
            return self._cache
        try:
            from titan_plugin.logic.knowledge_cache import KnowledgeCache
            self._cache = KnowledgeCache(
                db_path=self._cache_db_path,
                size_cap=self._cache_size_cap)
            logger.info("[WebSearch] KnowledgeCache ready (%s)",
                        self._cache_db_path)
        except Exception as e:
            logger.warning("[WebSearch] KnowledgeCache init failed: %s — "
                           "proceeding without cache", e)
        return self._cache

    def _ensure_health(self):
        """Build HealthTracker lazily (same shared health.json file).

        Passes `budgets=self._budgets` so this instance's HealthTracker
        has the same per-backend defaults as knowledge_worker's instance
        — prevents cross-process budget clobber in the shared health.json
        file (BUG-KP-WEBSEARCH-HEALTH-DEFAULTS).
        """
        if self._health is not None:
            return self._health
        try:
            from titan_plugin.logic.knowledge_health import HealthTracker
            self._health = HealthTracker(
                health_path=self._health_path,
                decision_log_path=self._decision_log_path,
                budgets=self._budgets)
            logger.info("[WebSearch] HealthTracker ready (%s, %d budgets)",
                        self._health_path, len(self._budgets))
        except Exception as e:
            logger.warning("[WebSearch] HealthTracker init failed: %s — "
                           "proceeding without circuit breaker", e)
        return self._health

    # ── Execute ─────────────────────────────────────────────────────

    async def execute(self, params: dict) -> dict:
        """Run an agency web search via the knowledge pipeline.

        Params:
            query: Search query string (required)
            max_results: Max results to surface (default 3; applied at
                         result formatting time, not a dispatcher
                         concern)

        Returns:
            {"success": bool, "result": str, "enrichment_data": dict,
             "error": Optional[str]}
        """
        query = (params.get("query") or "").strip()

        if not query:
            return {"success": False, "result": "",
                    "enrichment_data": {},
                    "error": "No query provided"}

        from titan_plugin.logic.knowledge_dispatcher import dispatch

        cache = self._ensure_cache()
        health = self._ensure_health()
        try:
            out = await dispatch(
                topic=query,
                cache=cache,
                sage=None,  # raw mode — agency tool doesn't need distillation
                raw_results=True,
                searxng_url=self._searxng_url,
                inflight=self._dispatch_inflight,
                timeout_per_backend=15.0,
                health=health,
                requestor="agency.web_search",
            )
        except Exception as e:
            logger.warning("[WebSearch] dispatch error: %s", e)
            return {"success": False, "result": "",
                    "enrichment_data": {},
                    "error": f"dispatch: {e}"}

        if out.rejected:
            return {"success": False, "result": "",
                    "enrichment_data": {},
                    "error": "Query rejected (Titan-internal name)"}

        if not out.success or out.result is None:
            return {"success": False, "result": "",
                    "enrichment_data": {},
                    "error": (f"no backend succeeded "
                              f"(attempts={out.attempts})")}

        # Cap body to agency's response budget
        body = out.result.raw_text or ""
        return {
            "success": True,
            "result": body[:600],
            "enrichment_data": {
                "mind": [0],
                "boost": 0.06 if out.cache_hit else 0.05,
                "backend": out.backend_used,
                "cache_hit": out.cache_hit,
                "query_type": out.query_type.value,
            },
            "error": None,
        }

    def status(self) -> str:
        """Report availability based on config presence (no network probe).

        Matches post-2026-04-14 contract — /health + agency stats call
        this synchronously, so we must NOT block. SearXNG URL presence
        is the pragmatic proxy for "pipeline available"; real connectivity
        is verified lazily per dispatch.
        """
        if self._searxng_url:
            return "available"
        return "unavailable"
