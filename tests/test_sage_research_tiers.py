"""§24.11 — tiered Stealth-Sage research: snippet-first (Tier 1) + trafilatura
main-content extraction (Tier 2), with the heavy Playwright crawl4ai path retired.

The pre-2026-06-12 pipeline discarded SearXNG's per-result snippets and always
scraped whole pages via html2text (nav/TOC boilerplate → the distiller choked) —
~11s for often-empty answers. These tests pin: Tier-1 answers from snippets without
scraping when they suffice; falls through to the scrape tier on the INSUFFICIENT
sentinel or thin snippets; trafilatura extracts main content (html2text fallback);
and crawl4ai is gone."""
import asyncio

from titan_hcl.logic.sage.researcher import (
    StealthSageResearcher, _INSUFFICIENT_SENTINEL)


def _researcher(**over):
    cfg = {
        "searxng_host": "http://localhost:8080",
        "research_timeout_seconds": 30,
        "snippet_first_enabled": True,
        "snippet_min_chars": 60,
        "snippet_min_results": 2,
        "api": {"internal_key": "k", "port": 7777},
    }
    cfg.update(over)
    r = StealthSageResearcher(cfg)
    r._write_research_log = lambda **kw: None  # no file writes in tests
    return r


# ── trafilatura main-content extraction (Tier 2 engine) ─────────────────────
def test_extract_main_content_strips_boilerplate():
    body = "George Orwell wrote the dystopian novel Nineteen Eighty-Four in 1949. " * 8
    html = (f"<html><body><nav>Home About Contact Login</nav>"
            f"<article><h1>1984</h1><p>{body}</p></article>"
            f"<footer>Copyright 2026 · Privacy · Cookies</footer></body></html>")
    out = StealthSageResearcher._extract_main_content(html)
    assert "George Orwell" in out and "Nineteen Eighty-Four" in out
    assert len(out) >= 50


def test_crawl4ai_retired():
    # the heavy Playwright path is gone; no strategy branch references it
    assert not hasattr(StealthSageResearcher, "_crawl4ai_scrape")


# ── Tier 1: snippet-first answers without scraping ──────────────────────────
def test_tier1_snippet_first_no_scrape(monkeypatch):
    r = _researcher()
    results = [
        {"url": "u1", "title": "BBC", "content": "Prague weather today is 18°C and sunny."},
        {"url": "u2", "title": "AccuWeather", "content": "Current Prague temperature ~18 degrees, clear."},
    ]
    scraped = {"called": False}

    async def fake_search(q):
        return results

    async def fake_distill(gap, data, *, sufficiency_check=False):
        assert sufficiency_check is True  # Tier-1 always asks for the sentinel
        return "Prague is about 18°C and sunny today."

    async def fake_scrape(urls):
        scraped["called"] = True
        return ([], [])

    monkeypatch.setattr(r, "_searxng_search", fake_search)
    monkeypatch.setattr(r, "_distill_with_ollama", fake_distill)
    monkeypatch.setattr(r, "_scrape_urls", fake_scrape)

    out = asyncio.run(r._research_pipeline("weather in prague"))
    assert "18°C" in out
    assert scraped["called"] is False  # the win: no scrape


# ── Tier 1 → Tier 2 fallthrough on INSUFFICIENT ─────────────────────────────
def test_tier1_insufficient_falls_through_to_scrape(monkeypatch):
    r = _researcher()
    results = [
        {"url": "u1", "title": "Nav", "content": "menu home about contact login cart help"},
        {"url": "u2", "title": "Foot", "content": "cookie banner privacy terms subscribe newsletter"},
    ]
    scraped = {"called": False}

    async def fake_search(q):
        return results

    async def fake_distill(gap, data, *, sufficiency_check=False):
        if sufficiency_check:
            return _INSUFFICIENT_SENTINEL          # snippets don't answer
        return "Answer assembled from the scraped page."

    async def fake_scrape(urls):
        scraped["called"] = True
        return (["[Source: u1]\nthe real scraped article body"], [])

    monkeypatch.setattr(r, "_searxng_search", fake_search)
    monkeypatch.setattr(r, "_distill_with_ollama", fake_distill)
    monkeypatch.setattr(r, "_scrape_urls", fake_scrape)

    out = asyncio.run(r._research_pipeline("a deep question"))
    assert scraped["called"] is True            # fell through to the scrape tier
    assert "scraped page" in out


# ── thin snippets skip Tier 1 entirely ──────────────────────────────────────
def test_thin_snippets_skip_tier1(monkeypatch):
    r = _researcher(snippet_min_chars=10_000)    # force "too thin"
    results = [{"url": "u1", "title": "T", "content": "short"}]
    sufficiency_calls = {"n": 0}
    scraped = {"called": False}

    async def fake_search(q):
        return results

    async def fake_distill(gap, data, *, sufficiency_check=False):
        if sufficiency_check:
            sufficiency_calls["n"] += 1
        return "answer"

    async def fake_scrape(urls):
        scraped["called"] = True
        return (["[Source: u1]\ncontent"], [])

    monkeypatch.setattr(r, "_searxng_search", fake_search)
    monkeypatch.setattr(r, "_distill_with_ollama", fake_distill)
    monkeypatch.setattr(r, "_scrape_urls", fake_scrape)

    asyncio.run(r._research_pipeline("q"))
    assert sufficiency_calls["n"] == 0           # Tier-1 distill never attempted
    assert scraped["called"] is True
