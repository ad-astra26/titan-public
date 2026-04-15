"""
web_search helper — Wraps SearXNG + Firecrawl for autonomous web research.

Primary: SearXNG (local, free, fast)
Fallback: Firecrawl API (premium, best quality for JS-heavy pages)

Enriches: Mind Vision[0] (research freshness boost)
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class WebSearchHelper:
    """Web search helper using SearXNG with Firecrawl fallback."""

    def __init__(self, searxng_url: str = "http://localhost:8080/search",
                 firecrawl_api_key: str = ""):
        self._searxng_url = searxng_url
        self._firecrawl_key = firecrawl_api_key

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

    async def execute(self, params: dict) -> dict:
        """
        Execute a web search.

        Params:
            query: Search query string
            max_results: Max results to return (default 3)
        """
        query = params.get("query", "")
        max_results = params.get("max_results", 3)

        if not query:
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": "No query provided"}

        # Try SearXNG first (local, free)
        result = await self._search_searxng(query, max_results)
        if result["success"]:
            return result

        # Fallback to Firecrawl if available
        if self._firecrawl_key:
            logger.info("[WebSearch] SearXNG failed, falling back to Firecrawl")
            return await self._search_firecrawl(query, max_results)

        return result  # Return SearXNG error

    async def _search_searxng(self, query: str, max_results: int) -> dict:
        """Search via local SearXNG instance."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                _search_url = self._searxng_url.rstrip("/") + "/search"
                resp = await client.get(_search_url, params={
                    "q": query, "format": "json", "categories": "general",
                })
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])[:max_results]
            summaries = []
            for r in results:
                title = r.get("title", "")
                content = r.get("content", "")[:200]
                url = r.get("url", "")
                summaries.append(f"- {title}: {content} ({url})")

            result_text = f"Found {len(results)} results for '{query}':\n" + "\n".join(summaries)

            return {
                "success": True,
                "result": result_text[:500],
                "enrichment_data": {"mind": [0], "boost": 0.05},
                "error": None,
            }
        except Exception as e:
            logger.warning("[WebSearch] SearXNG search failed: %s", e)
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": f"SearXNG: {e}"}

    async def _search_firecrawl(self, query: str, max_results: int) -> dict:
        """Search via Firecrawl API (premium fallback)."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.firecrawl.dev/v1/search",
                    json={
                        "query": query,
                        "limit": max_results,
                        "lang": "en",
                        "scrapeOptions": {"formats": ["markdown"]},
                    },
                    headers={
                        "Authorization": f"Bearer {self._firecrawl_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("data", [])[:max_results]
            summaries = []
            for r in results:
                title = r.get("metadata", {}).get("title", r.get("url", ""))
                content = r.get("markdown", "")[:200]
                url = r.get("url", "")
                summaries.append(f"- {title}: {content} ({url})")

            result_text = f"[Firecrawl] Found {len(results)} results for '{query}':\n" + "\n".join(summaries)

            return {
                "success": True,
                "result": result_text[:600],
                "enrichment_data": {"mind": [0], "boost": 0.06},
                "error": None,
            }
        except Exception as e:
            logger.warning("[WebSearch] Firecrawl search failed: %s", e)
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": f"Firecrawl: {e}"}

    def status(self) -> str:
        """Report availability based on config presence (no network probe).

        2026-04-14 fix: previous version did synchronous httpx.get to
        SearXNG with 5s timeout. Called from /health and embedded in
        many endpoints via _agency.get_stats() → blocked the asyncio
        event loop for up to 5s every cache miss (60s TTL). Made the
        entire Observatory API unresponsive while bus was healthy.
        Status now checks config like social_post + Firecrawl branch
        already did. Real connectivity is verified lazily on each search.
        """
        if self._searxng_url:
            return "available"
        if self._firecrawl_key:
            return "available"
        return "unavailable"
