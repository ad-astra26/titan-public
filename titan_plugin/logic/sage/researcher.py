"""
logic/sage/researcher.py

The Stealth-Sage Retrieval Pipeline Orchestrator for Titan V2.0.

Implements the full autonomous research loop:
  Step A: Discovery via self-hosted SearXNG (or Firecrawl search)
  Step B: Web scraping — httpx + html2text (fast, no browser), with optional
          Crawl4AI fallback for JS-heavy pages, or Firecrawl API (premium)
  Step C: X/Twitter pulse via TwitterAPI.io (conditional trigger)
  Step D: Distillation via Ollama Cloud (or Venice/OpenRouter)
  Step E: Formatted findings returned as [SAGE_RESEARCH_FINDINGS] block

All external services degrade gracefully if unconfigured or unreachable.
A configurable hard timeout prevents research from blocking responses.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from .document_processor import DocumentProcessor
from .x_researcher import XResearcher

log = logging.getLogger(__name__)

# X-Search is triggered only when the query explicitly signals social/sentiment context
_X_TRIGGER_KEYWORDS = {
    "people saying", "trending", "sentiment", "twitter", "community",
    "reaction", "think about", "opinion", "reddit", "social media", "vibe",
}

# Stealth browser headers to mask bot signals (Accept-Language + Referer are critical)
_STEALTH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
}

# Distillation prompt for multi-source research context
_DISTILL_PROMPT_TEMPLATE = (
    "You are the Titan's research assistant. "
    "Your task is to distill the following raw research data into a concise, "
    "factual summary of 3 to 5 sentences strictly relevant to this question: "
    '"{knowledge_gap}"\n\n'
    "Be precise. Focus only on measurable facts, figures, or confirmed events. "
    "Do not speculate. Return only the summary text, no preamble.\n\n"
    "Research Data:\n{research_data}"
)

# Research audit log path (relative to project root)
_RESEARCH_LOG_PATH = Path("data/logs/sage_research.log")


class StealthSageResearcher:
    """
    Orchestrates the full Titan Stealth-Sage research pipeline.

    On each call to research(), the Titan:
    1. Queries SearXNG (or Firecrawl) for relevant URLs.
    2. Scrapes each URL via httpx + html2text (fast, no browser overhead).
       Optional: Crawl4AI fallback for JS-heavy sites, or Firecrawl API (premium).
    3. Routes document links (.pdf, .docx, .pptx, .xlsx) to DocumentProcessor.
    4. Optionally searches X/Twitter via XResearcher for live sentiment.
    5. Distills all raw content via Ollama Cloud into a concise summary.
    6. Logs the research event to sage_research.log with a transition_id link.
    7. Returns the formatted [SAGE_RESEARCH_FINDINGS] block for context injection.
    """

    def __init__(self, config: dict) -> None:
        self._searxng_host = config.get("searxng_host", "http://localhost:8080").rstrip("/")
        self._top_num_urls = int(config.get("searxng_top_num_urls", 3))
        # Ollama Cloud client — wired by TitanPlugin.__init__ if configured
        self._ollama_cloud = None
        self._timeout = float(config.get("research_timeout_seconds", 30))

        proxy_url: Optional[str] = config.get("webshare_rotating_url", "").strip() or None

        self._x_researcher = XResearcher(
            api_key=config.get("twitterapi_io_key", ""),
            proxy_url=proxy_url,
            search_depth=int(config.get("twitterapi_search_depth", 20)),
        )

        self._doc_processor = DocumentProcessor(
            safe_room=config.get("doc_safe_room", "/tmp/titan_sage_docs"),
            max_load_avg=float(config.get("max_load_avg", 2.0)),
            proxy_url=proxy_url,
        )

        self._proxy_url = proxy_url

        # Firecrawl premium provider (optional — set firecrawl_api_key to enable)
        self._firecrawl_api_key = config.get("firecrawl_api_key", "").strip()

        # Scrape strategy: "fast" (httpx+html2text), "crawl4ai" (Playwright), "firecrawl"
        # Default to "fast" unless Firecrawl key is set
        if self._firecrawl_api_key:
            self._scrape_strategy = config.get("scrape_strategy", "firecrawl")
        else:
            self._scrape_strategy = config.get("scrape_strategy", "fast")

        # Cloud inference for distillation (Venice preferred, OpenRouter fallback)
        inference_cfg = config.get("_inference", {})
        provider = inference_cfg.get("inference_provider", "venice")
        if provider == "venice":
            self._cloud_api_key = inference_cfg.get("venice_api_key", "")
            self._cloud_base_url = "https://api.venice.ai/api/v1/chat/completions"
            self._cloud_model = "llama-3.3-70b"
        elif provider == "openrouter":
            self._cloud_api_key = inference_cfg.get("openrouter_api_key", "")
            self._cloud_base_url = "https://openrouter.ai/api/v1/chat/completions"
            self._cloud_model = "meta-llama/llama-3.3-70b-instruct:free"
        else:
            self._cloud_api_key = ""
            self._cloud_base_url = ""
            self._cloud_model = ""

        # Ensure research audit log directory exists
        _RESEARCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        log.info(
            "[StealthSage] Initialized — scrape_strategy=%s, firecrawl=%s",
            self._scrape_strategy, bool(self._firecrawl_api_key),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def research(
        self,
        knowledge_gap: str,
        transition_id: int = -1,
    ) -> str:
        """
        Executes the full Stealth-Sage research pipeline for the given knowledge gap.

        Wrapped in asyncio.wait_for with a hard timeout to prevent blocking the
        OpenClaw response loop. Returns "" on timeout or total failure.

        Args:
            knowledge_gap (str): The question or topic the Titan needs to research.
            transition_id (int): The current SageRecorder buffer index at call time.
                                 Used to link this research event to the RL transition log.

        Returns:
            str: A formatted "[SAGE_RESEARCH_FINDINGS]: ..." block ready for
                 injection into the LLM system prompt, or "" if nothing was found.
        """
        try:
            return await asyncio.wait_for(
                self._research_pipeline(knowledge_gap, transition_id),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            log.warning(
                f"[StealthSage] Research pipeline timed out after {self._timeout}s "
                f"for query='{knowledge_gap}'. Proceeding without findings."
            )
            return ""
        except Exception as e:
            log.error(f"[StealthSage] Unexpected error in research pipeline: {e}")
            return ""

    # ------------------------------------------------------------------
    # Internal Pipeline Steps
    # ------------------------------------------------------------------

    async def _research_pipeline(self, knowledge_gap: str, transition_id: int) -> str:
        """Internal pipeline execution (called within the timeout wrapper)."""
        raw_chunks: list[str] = []
        sources_used: list[str] = []
        doc_summaries: list[str] = []

        # ---- Step A: Discovery ----
        # Firecrawl search if available, otherwise SearXNG
        if self._firecrawl_api_key and self._scrape_strategy == "firecrawl":
            urls = await self._firecrawl_search(knowledge_gap)
        else:
            urls = await self._searxng_discover(knowledge_gap)

        # ---- Step B: Stealth Scrape + Document Deep-Dive ----
        if urls:
            scrape_results, doc_results = await self._scrape_urls(urls)
            raw_chunks.extend(scrape_results)
            doc_summaries.extend(doc_results)

            if scrape_results:
                sources_used.append("Web")
            if doc_results:
                sources_used.append("Document")

        # ---- Step C: X-Search (conditional on query semantics) ----
        if self._x_researcher.is_enabled and self._should_trigger_x_search(knowledge_gap):
            x_results = await self._x_researcher.search(knowledge_gap)
            if x_results:
                raw_chunks.append(x_results)
                sources_used.append("X")

        # Combine document summaries and raw web content
        all_content_parts = doc_summaries + raw_chunks

        if not all_content_parts:
            log.info(f"[StealthSage] No research content gathered for '{knowledge_gap}'.")
            return ""

        combined_raw = "\n\n---\n\n".join(all_content_parts)

        # ---- Step D: Local Distillation (Ollama phi3:mini) ----
        distilled = await self._distill_with_ollama(knowledge_gap, combined_raw)

        if not distilled:
            return ""

        # ---- Step E: Memory Ingestion + Audit Log ----
        self._write_research_log(
            knowledge_gap=knowledge_gap,
            sources_used=sources_used,
            urls_scraped=urls,
            distilled_summary=distilled,
            transition_id=transition_id,
        )

        findings = f"[SAGE_RESEARCH_FINDINGS]: {distilled}"
        log.info(
            f"[StealthSage] Research complete for '{knowledge_gap}'. "
            f"Sources: {sources_used}. Length: {len(distilled)} chars."
        )
        return findings

    async def _searxng_discover(self, query: str) -> list[str]:
        """
        Queries the self-hosted SearXNG instance and returns the top N result URLs.

        Args:
            query (str): The search query.

        Returns:
            list[str]: Up to searxng_top_num_urls URL strings, or [] on failure.
        """
        try:
            params = {"q": query, "format": "json", "language": "en"}
            # SearXNG is local — never proxy localhost requests.
            # 2026-04-09: timeout bumped 10s → 20s. SearXNG aggregates from
            # multiple upstream engines (bing, wikipedia, brave, wikidata, ...)
            # and waits for ALL of them up to its own per-engine deadline (~10s).
            # Real-world latency is consistently 9-12s when slow engines like
            # wikidata are in the rotation, so a 10s client timeout was racing
            # SearXNG's own deadline and producing empty results. 20s gives
            # safe headroom; the overall research pipeline still has 90s budget.
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(
                    f"{self._searxng_host}/search",
                    params=params,
                )
                response.raise_for_status()

            data = response.json()
            results = data.get("results", [])
            urls = [r["url"] for r in results if r.get("url")]
            selected = urls[: self._top_num_urls]
            log.info(f"[StealthSage] SearXNG returned {len(results)} results; selected {len(selected)} URLs.")
            return selected

        except httpx.ConnectError:
            log.warning(
                "[StealthSage] SearXNG unreachable at "
                f"{self._searxng_host}. Is the Docker container running?"
            )
            return []
        except Exception as e:
            log.warning(f"[StealthSage] SearXNG query failed: {e}")
            return []

    async def _firecrawl_search(self, query: str) -> list[str]:
        """
        Search via Firecrawl API (premium). Returns top N URLs.
        Falls back to SearXNG if Firecrawl search fails.
        """
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://api.firecrawl.dev/v1/search",
                    json={"query": query, "limit": self._top_num_urls},
                    headers={
                        "Authorization": f"Bearer {self._firecrawl_api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()

            data = response.json()
            results = data.get("data", [])
            urls = [r.get("url", "") for r in results if r.get("url")]
            log.info("[StealthSage] Firecrawl search returned %d URLs.", len(urls))
            return urls[:self._top_num_urls]

        except Exception as e:
            log.warning("[StealthSage] Firecrawl search failed: %s. Falling back to SearXNG.", e)
            return await self._searxng_discover(query)

    async def _scrape_urls(self, urls: list[str]) -> tuple[list[str], list[str]]:
        """
        Routes each URL to the appropriate scraper based on configured strategy.

        Strategies:
          - "fast" (default): httpx + html2text — no browser, sub-second per page
          - "crawl4ai": Playwright stealth browser (slow but handles JS-rendered pages)
          - "firecrawl": Firecrawl API (premium, best quality)

        Documents (.pdf, .docx, .pptx, .xlsx) always route to DocumentProcessor.
        """
        web_chunks: list[str] = []
        doc_chunks: list[str] = []

        # Scrape web pages concurrently (huge speedup over sequential)
        web_urls = []
        for url in urls:
            if DocumentProcessor.is_document_url(url):
                try:
                    result = await self._doc_processor.process(url)
                    if result and result.get("summary"):
                        doc_chunks.append(result["summary"])
                except Exception as e:
                    log.warning("[StealthSage] Doc processing failed for %s: %s", url, e)
            else:
                web_urls.append(url)

        if web_urls:
            tasks = [self._scrape_single(url) for url in web_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for url, result in zip(web_urls, results):
                if isinstance(result, Exception):
                    log.warning("[StealthSage] Scrape failed for %s: %s", url, result)
                elif result:
                    web_chunks.append(f"[Source: {url}]\n{result}")

        return web_chunks, doc_chunks

    async def _scrape_single(self, url: str) -> str:
        """Scrape a single URL using the configured strategy."""
        if self._scrape_strategy == "firecrawl":
            return await self._firecrawl_scrape(url)
        elif self._scrape_strategy == "crawl4ai":
            return await self._crawl4ai_scrape(url)
        else:
            # "fast" — httpx + html2text (default)
            return await self._fast_scrape(url)

    async def _fast_scrape(self, url: str) -> str:
        """
        Fast scraping via httpx + html2text. No browser overhead.
        Handles 90%+ of web pages (articles, blogs, docs) in <1 second.
        Falls back to Crawl4AI for JS-heavy pages if available.
        """
        try:
            import html2text

            client_kwargs: dict = {
                "headers": _STEALTH_HEADERS,
                "timeout": 15.0,
                "follow_redirects": True,
            }
            if self._proxy_url:
                client_kwargs["proxy"] = self._proxy_url

            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.get(url)
                response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                log.info("[StealthSage] Non-HTML content at %s (%s), skipping.", url, content_type)
                return ""

            html = response.text
            if not html or len(html.strip()) < 100:
                return ""

            # Convert HTML to clean markdown
            converter = html2text.HTML2Text()
            converter.ignore_links = False
            converter.ignore_images = True
            converter.ignore_emphasis = False
            converter.body_width = 0  # No line wrapping
            converter.skip_internal_links = True
            converter.inline_links = True

            markdown = converter.handle(html)

            # Strip boilerplate: nav menus, footers, cookie banners tend to be
            # short lines at the start/end. Keep the meaty middle.
            lines = markdown.strip().split("\n")
            # Remove very short lines (likely nav/menu items) from start
            start = 0
            for i, line in enumerate(lines):
                if len(line.strip()) > 60:
                    start = i
                    break
            cleaned = "\n".join(lines[start:])

            content = cleaned[:5000].strip()
            if len(content) < 50:
                # Page might be JS-rendered — try Crawl4AI fallback
                log.info("[StealthSage] Sparse content from %s, trying Crawl4AI fallback.", url)
                return await self._crawl4ai_scrape(url)

            log.info("[StealthSage] Fast-scraped %d chars from %s.", len(content), url)
            return content

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                # Bot-blocked — try Crawl4AI with stealth mode
                log.info("[StealthSage] 403 from %s, trying Crawl4AI stealth fallback.", url)
                return await self._crawl4ai_scrape(url)
            log.warning("[StealthSage] HTTP %d from %s", e.response.status_code, url)
            return ""
        except Exception as e:
            log.warning("[StealthSage] Fast scrape failed for %s: %s", url, e)
            return ""

    async def _crawl4ai_scrape(self, url: str) -> str:
        """
        Crawl4AI stealth scraper (Playwright-based). Slow but handles JS-rendered
        pages and bot-protected sites. Used as fallback when fast scrape fails.
        """
        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

            browser_kwargs: dict = {
                "headless": True,
                "enable_stealth": True,
                "headers": _STEALTH_HEADERS,
            }
            if self._proxy_url:
                try:
                    from crawl4ai import ProxyConfig
                    from urllib.parse import urlparse
                    parsed = urlparse(self._proxy_url)
                    proxy_server = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
                    browser_kwargs["proxy_config"] = ProxyConfig(
                        server=proxy_server,
                        username=parsed.username or None,
                        password=parsed.password or None,
                    )
                except ImportError:
                    browser_kwargs["proxy"] = self._proxy_url
            browser_cfg = BrowserConfig(**browser_kwargs)

            run_cfg = CrawlerRunConfig(
                word_count_threshold=10,
                exclude_external_links=True,
                remove_overlay_elements=True,
            )

            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                result = await crawler.arun(url=url, config=run_cfg)

            if result.success and result.markdown:
                content = result.markdown[:5000].strip()
                log.info("[StealthSage] Crawl4AI scraped %d chars from %s.", len(content), url)
                return content
            else:
                log.warning("[StealthSage] Crawl4AI returned no content for %s.", url)
                return ""

        except ImportError:
            log.warning("[StealthSage] crawl4ai not installed, no JS fallback available.")
            return ""
        except Exception as e:
            log.warning("[StealthSage] Crawl4AI error for %s: %s", url, e)
            return ""

    async def _firecrawl_scrape(self, url: str) -> str:
        """
        Premium scraping via Firecrawl API. Returns clean markdown.
        Requires firecrawl_api_key in [stealth_sage] config.
        """
        if not self._firecrawl_api_key:
            return await self._fast_scrape(url)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.firecrawl.dev/v1/scrape",
                    json={"url": url, "formats": ["markdown"]},
                    headers={
                        "Authorization": f"Bearer {self._firecrawl_api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()

            data = response.json()
            if data.get("success") and data.get("data", {}).get("markdown"):
                content = data["data"]["markdown"][:5000].strip()
                log.info("[StealthSage] Firecrawl scraped %d chars from %s.", len(content), url)
                return content
            log.warning("[StealthSage] Firecrawl returned no content for %s.", url)
            return ""
        except Exception as e:
            log.warning("[StealthSage] Firecrawl error for %s: %s. Falling back to fast scrape.", url, e)
            return await self._fast_scrape(url)

    async def _distill_with_ollama(self, knowledge_gap: str, research_data: str) -> str:
        """
        Distills research data into a concise summary.

        Tries cloud inference first (Venice/OpenRouter — fast), then falls back to
        local Ollama (CPU — slow but free). Returns "" if both fail.

        Args:
            knowledge_gap (str): The original research question.
            research_data (str): All concatenated raw research content.

        Returns:
            str: The distilled summary text, or "" on failure.
        """
        prompt = _DISTILL_PROMPT_TEMPLATE.format(
            knowledge_gap=knowledge_gap,
            research_data=research_data[:12_000],
        )

        # Try cloud inference first (Venice/OpenRouter — fast)
        if self._cloud_api_key:
            result = await self._distill_cloud(prompt)
            if result:
                return result

        # Fall back to Ollama Cloud
        if self._ollama_cloud:
            return await self._distill_ollama_cloud(prompt)

        log.warning("[StealthSage] No inference backend available for distillation.")
        return ""

    async def _distill_cloud(self, prompt: str) -> str:
        """Distill via cloud LLM (Venice or OpenRouter)."""
        payload = {
            "model": self._cloud_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.3,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self._cloud_base_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._cloud_api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()

            data = response.json()
            distilled = data["choices"][0]["message"]["content"].strip()
            if distilled:
                log.info(f"[StealthSage] Cloud distilled {len(distilled)} chars of wisdom.")
                return distilled
            return ""
        except Exception as e:
            log.warning(f"[StealthSage] Cloud distillation failed: {e}")
            return ""

    async def _distill_ollama_cloud(self, prompt: str) -> str:
        """Distill via Ollama Cloud API."""
        try:
            from titan_plugin.utils.ollama_cloud import get_model_for_task
            model = get_model_for_task("research_distill")
            distilled = await self._ollama_cloud.complete(
                prompt=prompt,
                model=model,
                temperature=0.3,
                max_tokens=500,
                timeout=60.0,
            )
            if distilled:
                log.info(f"[StealthSage] Ollama Cloud distilled {len(distilled)} chars of wisdom.")
                return distilled
            log.warning("[StealthSage] Ollama Cloud returned empty distillation response.")
            return ""
        except Exception as e:
            log.warning(f"[StealthSage] Ollama Cloud distillation failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_trigger_x_search(query: str) -> bool:
        """
        Returns True if the query contains keywords indicating that social/sentiment
        data from X/Twitter would add meaningful context to the research.

        Args:
            query (str): The raw knowledge gap query string.

        Returns:
            bool: True if X-Search should be triggered for this query.
        """
        lower = query.lower()
        return any(kw in lower for kw in _X_TRIGGER_KEYWORDS)

    def _write_research_log(
        self,
        knowledge_gap: str,
        sources_used: list[str],
        urls_scraped: list[str],
        distilled_summary: str,
        transition_id: int,
    ) -> None:
        """
        Appends a structured JSON line to the Sage Research Chronicle log.

        Each entry links to the SageRecorder transition_id so RL audit trails
        can cross-reference research events with their downstream reward outcomes.

        Log format (one JSON object per line):
            {
              "timestamp": "2026-03-09T08:45:00Z",
              "knowledge_gap": "<query>",
              "sources_used": ["Web", "Document", "X"],
              "urls_scraped": ["https://..."],
              "distilled_summary": "<3-5 sentence summary>",
              "transition_id": 42
            }

        Args:
            knowledge_gap (str): The original research query.
            sources_used (list[str]): Sources that contributed data (Web, Document, X).
            urls_scraped (list[str]): URLs passed to the scraping pipeline.
            distilled_summary (str): The Ollama-distilled summary.
            transition_id (int): SageRecorder buffer index at research call time.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "knowledge_gap": knowledge_gap,
            "sources_used": sources_used,
            "urls_scraped": urls_scraped,
            "distilled_summary": distilled_summary,
            "transition_id": transition_id,
        }
        try:
            with open(_RESEARCH_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning(f"[StealthSage] Failed to write to research audit log: {e}")
