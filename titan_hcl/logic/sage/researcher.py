"""
logic/sage/researcher.py

The Stealth-Sage Retrieval Pipeline Orchestrator for Titan V2.0.

Implements the full autonomous research loop (§24.11 tiered, 2026-06-12):
  Step A: Discovery via self-hosted SearXNG (or Firecrawl search) — returns
          URLs AND per-result content snippets in one call
  Tier 1: SNIPPET-FIRST — distill the answer from the SearXNG snippets directly
          (no scrape, ~3s); fall through only if thin / flagged insufficient
  Tier 2: Web scraping — httpx + trafilatura main-content extraction (strips
          nav/ads/boilerplate; html2text fallback), or Firecrawl API (premium)
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
import time
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
        "Chrome/203.0.113.10 Safari/537.36"
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

# Tier-1 sufficiency sentinel (§24.11): when distilling from search SNIPPETS, the
# LLM returns EXACTLY this token if the snippets don't actually answer the
# question → the pipeline falls through to the scrape tier (Tier 2).
_INSUFFICIENT_SENTINEL = "INSUFFICIENT_EVIDENCE"
_DISTILL_SUFFICIENCY_SUFFIX = (
    "\n\nIMPORTANT: If the research data above does NOT contain enough "
    "information to actually answer the question, reply with EXACTLY this "
    f"token and nothing else: {_INSUFFICIENT_SENTINEL}")

# Research audit log path (relative to project root)
_RESEARCH_LOG_PATH = Path("data/logs/sage_research.log")


class StealthSageResearcher:
    """
    Orchestrates the full Titan Stealth-Sage research pipeline.

    On each call to research(), the Titan:
    1. Queries SearXNG (or Firecrawl) for relevant URLs + content snippets.
    1b. TIER 1 (snippet-first): distills the answer from the search snippets
        directly when they suffice — no scrape (~3s).
    2. TIER 2 (fallback): scrapes each URL via httpx + trafilatura main-content
       extraction (html2text fallback; no browser), or Firecrawl API (premium).
    3. Routes document links (.pdf, .docx, .pptx, .xlsx) to DocumentProcessor.
    4. Optionally searches X/Twitter via XResearcher for live sentiment.
    5. Distills all raw content via Ollama Cloud into a concise summary.
    6. Logs the research event to sage_research.log with a transition_id link.
    7. Returns the formatted [SAGE_RESEARCH_FINDINGS] block for context injection.
    """

    def __init__(self, config: dict) -> None:
        self._searxng_host = config.get("searxng_host", "http://localhost:8080").rstrip("/")
        self._top_num_urls = int(config.get("searxng_top_num_urls", 3))
        # Ollama Cloud client — wired by TitanHCL.__init__ if configured
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

        # Scrape strategy: "fast" (httpx + trafilatura main-content extraction —
        # strips nav/ads/boilerplate, mean F1 0.883 SIGIR'23; html2text fallback)
        # or "firecrawl" (premium API). The heavy Playwright "crawl4ai" strategy
        # was RETIRED 2026-06-12 (too heavy for the box; trafilatura cascades to
        # readability/jusText internally, covering most JS-light pages).
        # Default to "fast" unless a Firecrawl key is set.
        if self._firecrawl_api_key:
            self._scrape_strategy = config.get("scrape_strategy", "firecrawl")
        else:
            self._scrape_strategy = config.get("scrape_strategy", "fast")

        # Snippet-first (research Tier 1, §24.11): SearXNG returns a per-result
        # `content` snippet in the SAME call as the URLs — distill the answer from
        # those directly (~3s, no scrape) and only fall through to scrape (Tier 2)
        # when the snippets are too thin or the distiller flags them insufficient.
        # The pre-2026-06-12 pipeline discarded these snippets, then scraped +
        # distilled to re-derive what was already in hand. (2026-06-12.)
        self._snippet_first = bool(config.get("snippet_first_enabled", True))
        self._snippet_min_chars = int(config.get("snippet_min_chars", 400))
        self._snippet_min_results = int(config.get("snippet_min_results", 2))

        # Phase 3 Chunk ψ (D-SPEC-88, 2026-05-18) — Venice/OpenRouter direct
        # httpx REPLACED by /v4/llm-distill. Provider abstraction +
        # failover now centralized in llm_worker.
        api_cfg = config.get("api", {}) or {}
        self._llm_api_base = (
            f"http://127.0.0.1:{int(api_cfg.get('port', 7777))}")
        self._llm_internal_key = api_cfg.get("internal_key", "") or ""
        # Kept for any external test that constructs the researcher and
        # inspects these — empty values short-circuit any legacy reads.
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
    ) -> str:
        """
        Executes the full Stealth-Sage research pipeline for the given knowledge gap.

        Wrapped in asyncio.wait_for with a hard timeout to prevent blocking the
        OpenClaw response loop. Returns "" on timeout or total failure.

        Args:
            knowledge_gap (str): The question or topic the Titan needs to research.

        Returns:
            str: A formatted "[SAGE_RESEARCH_FINDINGS]: ..." block ready for
                 injection into the LLM system prompt, or "" if nothing was found.

        (The `transition_id` arg — which linked the research event to the
        SageRecorder RL buffer — was RETIRED with the offline-RL subsystem,
        RFP_synthesis_decision_authority P1.)
        """
        try:
            return await asyncio.wait_for(
                self._research_pipeline(knowledge_gap),
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

    async def _research_pipeline(self, knowledge_gap: str) -> str:
        """Internal pipeline execution (called within the timeout wrapper)."""
        raw_chunks: list[str] = []
        sources_used: list[str] = []
        doc_summaries: list[str] = []

        # [PERF-INSTRUMENT 2026-06-15] Per-phase wall-clock timing — logging only,
        # zero behavior change. Diagnoses the ~80s research turn (research-speed
        # MUST-DO). Remove or demote to debug once the round/phase fix lands.
        _t_start = time.monotonic()
        _ms = lambda t0: round((time.monotonic() - t0) * 1000)

        # ---- Step A: Discovery (URLs + per-result snippets in ONE call) ----
        _t_search = time.monotonic()
        if self._firecrawl_api_key and self._scrape_strategy == "firecrawl":
            urls = await self._firecrawl_search(knowledge_gap)
            results = [{"url": u, "title": "", "content": ""} for u in urls]
        else:
            results = await self._searxng_search(knowledge_gap)
            urls = [r["url"] for r in results]
        log.info("[PERF] search phase: %d ms (%d urls)",
                 _ms(_t_search), len(urls))

        # Assemble the snippet evidence the search already handed us (free).
        snippet_parts = [
            f"{r['title']}: {r['content']}".strip(": ").strip()
            for r in results if r.get("content")]
        snippet_evidence = "\n\n".join(snippet_parts)

        # ---- Tier 1: SNIPPET-FIRST (§24.11) — distill from the snippets the
        # search already returned (no scrape, ~3s). Fall through to the scrape
        # tier only when the snippets are too thin OR the distiller flags them
        # insufficient (the INSUFFICIENT sentinel). ----
        _tier1_eligible = (self._snippet_first
                and len(snippet_parts) >= self._snippet_min_results
                and len(snippet_evidence) >= self._snippet_min_chars)
        log.info("[PERF] tier-1 eligible=%s (snippets=%d/%d min, chars=%d/%d min)",
                 _tier1_eligible, len(snippet_parts), self._snippet_min_results,
                 len(snippet_evidence), self._snippet_min_chars)
        if _tier1_eligible:
            _t_t1 = time.monotonic()
            snippet_distilled = await self._distill_with_ollama(
                knowledge_gap, snippet_evidence, sufficiency_check=True)
            _t1_sufficient = bool(
                snippet_distilled and snippet_distilled.strip() != _INSUFFICIENT_SENTINEL)
            log.info("[PERF] tier-1 distill: %d ms, sufficient=%s (result_len=%d)",
                     _ms(_t_t1), _t1_sufficient,
                     len(snippet_distilled or ""))
            if _t1_sufficient:
                self._write_research_log(
                    knowledge_gap=knowledge_gap, sources_used=["WebSnippet"],
                    urls_scraped=urls, distilled_summary=snippet_distilled)
                log.info("[StealthSage] Tier-1 snippet-first answered '%s' "
                         "(%d chars, no scrape). [PERF] TOTAL %d ms",
                         knowledge_gap, len(snippet_distilled), _ms(_t_start))
                return f"[SAGE_RESEARCH_FINDINGS]: {snippet_distilled}"
            log.info("[StealthSage] Tier-1 snippets insufficient for '%s' → "
                     "scrape tier.", knowledge_gap)

        # ---- Tier 2: Scrape (trafilatura main-content) + Document Deep-Dive ----
        if urls:
            _t_scrape = time.monotonic()
            scrape_results, doc_results = await self._scrape_urls(urls)
            log.info("[PERF] tier-2 scrape: %d ms (%d web chunks, %d docs from %d urls)",
                     _ms(_t_scrape), len(scrape_results), len(doc_results), len(urls))
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

        # The snippets remain useful evidence in Tier 2 — lead with them.
        all_content_parts = (
            ([snippet_evidence] if snippet_evidence else [])
            + doc_summaries + raw_chunks)

        if not all_content_parts:
            log.info(f"[StealthSage] No research content gathered for '{knowledge_gap}'.")
            return ""
        if snippet_evidence and "WebSnippet" not in sources_used:
            sources_used.insert(0, "WebSnippet")

        combined_raw = "\n\n---\n\n".join(all_content_parts)

        # ---- Step D: Local Distillation (/v4/llm-distill) ----
        _t_t2 = time.monotonic()
        distilled = await self._distill_with_ollama(knowledge_gap, combined_raw)
        log.info("[PERF] tier-2 distill: %d ms (%d raw chars in, %d out)",
                 _ms(_t_t2), len(combined_raw), len(distilled or ""))

        if not distilled:
            log.info("[PERF] TOTAL (tier-2 empty): %d ms", _ms(_t_start))
            return ""

        # ---- Step E: Memory Ingestion + Audit Log ----
        self._write_research_log(
            knowledge_gap=knowledge_gap,
            sources_used=sources_used,
            urls_scraped=urls,
            distilled_summary=distilled,
        )

        findings = f"[SAGE_RESEARCH_FINDINGS]: {distilled}"
        log.info(
            f"[StealthSage] Research complete for '{knowledge_gap}'. "
            f"Sources: {sources_used}. Length: {len(distilled)} chars. "
            f"[PERF] TOTAL %d ms" % _ms(_t_start)
        )
        return findings

    async def _searxng_search(self, query: str) -> list[dict]:
        """
        Queries the self-hosted SearXNG instance and returns the top N results as
        `{url, title, content}` dicts. `content` is SearXNG's per-result snippet —
        the answer is frequently in it directly, so Tier-1 (snippet-first) distills
        from these without scraping. The pre-2026-06-12 path kept only the URLs.

        Returns up to searxng_top_num_urls result dicts, or [] on failure.
        """
        try:
            params = {"q": query, "format": "json", "language": "en"}
            # SearXNG is local — never proxy localhost requests.
            # 2026-04-09: timeout 10s → 20s (SearXNG waits on slow upstream
            # engines like wikidata, ~9-12s real latency; 90s pipeline budget).
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(
                    f"{self._searxng_host}/search", params=params)
                response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            out: list[dict] = []
            for r in results:
                if r.get("url"):
                    out.append({
                        "url": r["url"],
                        "title": str(r.get("title", "") or ""),
                        "content": str(r.get("content", "") or ""),
                    })
                if len(out) >= self._top_num_urls:
                    break
            log.info("[StealthSage] SearXNG returned %d results; took top %d "
                     "(snippets=%d chars).", len(results), len(out),
                     sum(len(r["content"]) for r in out))
            return out
        except httpx.ConnectError:
            log.warning("[StealthSage] SearXNG unreachable at %s. Is the Docker "
                        "container running?", self._searxng_host)
            return []
        except Exception as e:
            log.warning("[StealthSage] SearXNG query failed: %s", e)
            return []

    async def _searxng_discover(self, query: str) -> list[str]:
        """URLs-only wrapper over `_searxng_search` (used by the Firecrawl-search
        fallback path, which only needs URLs)."""
        return [r["url"] for r in await self._searxng_search(query)]


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
          - "fast" (default): httpx + trafilatura main-content extraction
            (html2text fallback) — no browser, sub-second per page, strips boilerplate
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
        # "fast" — httpx + trafilatura main-content extraction (default).
        # ("crawl4ai" Playwright strategy retired 2026-06-12; trafilatura's
        # internal readability/jusText cascade covers most JS-light pages.)
        return await self._fast_scrape(url)

    async def _fast_scrape(self, url: str) -> str:
        """
        Fast scraping via httpx + trafilatura main-content extraction. No browser.
        trafilatura strips nav/ads/boilerplate (mean F1 0.883, SIGIR'23) and
        cascades to readability/jusText internally; html2text is the last-resort
        fallback when trafilatura returns nothing.
        """
        try:
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

            content = self._extract_main_content(html)[:5000].strip()
            if len(content) < 50:
                # Sparse after extraction (likely a JS-rendered shell) — skip.
                # (The retired crawl4ai Playwright fallback used to run here.)
                log.info("[StealthSage] Sparse content from %s after extraction.", url)
                return ""

            log.info("[StealthSage] Fast-scraped %d chars from %s.", len(content), url)
            return content

        except httpx.HTTPStatusError as e:
            log.warning("[StealthSage] HTTP %d from %s", e.response.status_code, url)
            return ""
        except Exception as e:
            log.warning("[StealthSage] Fast scrape failed for %s: %s", url, e)
            return ""

    @staticmethod
    def _extract_main_content(html: str) -> str:
        """Main-content extraction: trafilatura (strips nav/ads/boilerplate) →
        html2text whole-page fallback. Both pure-python, no browser."""
        # Primary: trafilatura — the SIGIR'23 benchmark winner for clean text.
        try:
            import trafilatura
            extracted = trafilatura.extract(
                html, include_comments=False, include_tables=True,
                no_fallback=False)  # cascades to readability/jusText internally
            if extracted and len(extracted.strip()) >= 50:
                return extracted.strip()
        except Exception as e:  # noqa: BLE001
            log.debug("[StealthSage] trafilatura extract failed: %s", e)
        # Fallback: html2text whole-page → crude lead-boilerplate trim.
        try:
            import html2text
            converter = html2text.HTML2Text()
            converter.ignore_links = False
            converter.ignore_images = True
            converter.body_width = 0
            converter.skip_internal_links = True
            converter.inline_links = True
            markdown = converter.handle(html)
            lines = markdown.strip().split("\n")
            start = 0
            for i, line in enumerate(lines):
                if len(line.strip()) > 60:
                    start = i
                    break
            return "\n".join(lines[start:]).strip()
        except Exception as e:  # noqa: BLE001
            log.warning("[StealthSage] html2text fallback failed: %s", e)
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

    async def _distill_with_ollama(self, knowledge_gap: str, research_data: str,
                                   *, sufficiency_check: bool = False) -> str:
        """
        Distills research data into a concise summary via /v4/llm-distill.

        Phase 3 Chunk ψ (D-SPEC-88, 2026-05-18): replaces the 2-tier
        Venice/OpenRouter direct httpx + Ollama-Cloud-direct fallback.
        Provider abstraction + failover now lives in llm_worker. All
        distillation traffic appears in llm_state.bin.

        Args:
            knowledge_gap (str): The original research question.
            research_data (str): All concatenated raw research content.

        Returns:
            str: The distilled summary text, or "" on failure.
        """
        if not self._llm_internal_key:
            log.warning(
                "[StealthSage] No internal_key for /v4/llm-distill — "
                "research distillation disabled.")
            return ""

        prompt = _DISTILL_PROMPT_TEMPLATE.format(
            knowledge_gap=knowledge_gap,
            research_data=research_data[:12_000],
        )
        if sufficiency_check:
            prompt += _DISTILL_SUFFICIENCY_SUFFIX

        try:
            from titan_hcl.inference import get_model_for_task
            from titan_hcl.logic.llm_distill_client import (
                distill_via_http_async)
            model = get_model_for_task("research_distill")
            distilled = await distill_via_http_async(
                text=prompt,
                instruction="",
                api_base=self._llm_api_base,
                internal_key=self._llm_internal_key,
                model=model,
                max_tokens=500,
                temperature=0.3,
                consumer="sage_researcher_distill",
                timeout_s=60.0,
            )
            if distilled:
                log.info(
                    f"[StealthSage] Distilled {len(distilled)} chars of wisdom.")
                return distilled
            log.warning(
                "[StealthSage] /v4/llm-distill returned empty for research distillation.")
            return ""
        except Exception as e:
            log.warning(f"[StealthSage] /v4/llm-distill failed: {e}")
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
    ) -> None:
        """
        Appends a structured JSON line to the Sage Research Chronicle log.

        Log format (one JSON object per line):
            {
              "timestamp": "2026-03-09T08:45:00Z",
              "knowledge_gap": "<query>",
              "sources_used": ["Web", "Document", "X"],
              "urls_scraped": ["https://..."],
              "distilled_summary": "<3-5 sentence summary>"
            }

        Args:
            knowledge_gap (str): The original research query.
            sources_used (list[str]): Sources that contributed data (Web, Document, X).
            urls_scraped (list[str]): URLs passed to the scraping pipeline.
            distilled_summary (str): The Ollama-distilled summary.

        (The `transition_id` field — which linked the research event to the
        SageRecorder RL transition log — was RETIRED with the offline-RL
        subsystem, RFP_synthesis_decision_authority P1.)
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "knowledge_gap": knowledge_gap,
            "sources_used": sources_used,
            "urls_scraped": urls_scraped,
            "distilled_summary": distilled_summary,
        }
        try:
            with open(_RESEARCH_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning(f"[StealthSage] Failed to write to research audit log: {e}")
