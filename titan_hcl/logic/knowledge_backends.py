"""
knowledge_backends — direct-REST backend adapters called by knowledge_router.

Three backends that bypass SearXNG for queries where a structured public API
gives cleaner, faster, cheaper results:

  * Wiktionary          — en.wiktionary.org/api/rest_v1/page/definition/{word}
  * Free Dictionary     — api.dictionaryapi.dev/api/v2/entries/en/{word}
  * Wikipedia summary   — en.wikipedia.org/api/rest_v1/page/summary/{title}

All are async httpx calls with dumb contract: receive a string, attempt
one HTTP request, return a BackendResult. No retry, no cache, no budget
tracking — those wrap at the router level in KP-2..KP-7. Query
transformation (e.g. "own meaning" → lookup "own") lives in KP-3.

See: titan-docs/rFP_knowledge_pipeline_v2.md §3.1 + KP-1.
"""

from __future__ import annotations

import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# ── Result shape ─────────────────────────────────────────────────────

@dataclass
class BackendResult:
    """Outcome of a single backend fetch attempt.

    `error_type` is a stable string (mapped to typed exceptions in KP-4):
      - ""              : success
      - "timeout"       : httpx.TimeoutException
      - "rate_limit"    : HTTP 429
      - "http_4xx"      : HTTP 4xx other than 429
      - "http_5xx"      : HTTP 5xx
      - "empty"         : 200 OK but no useful content
      - "parse_error"   : JSON malformed or unexpected shape
      - "network"       : connection errors, DNS, etc.
    """
    backend: str
    query: str
    success: bool = False
    raw_text: str = ""
    structured: Optional[dict] = None
    error_type: str = ""
    error_msg: str = ""
    bytes_consumed: int = 0
    latency_ms: float = 0.0
    status_code: int = 0


# ── Shared constants ─────────────────────────────────────────────────

_UA = ("Mozilla/5.0 (compatible; TitanKnowledgeBot/1.0; "
       "+https://iamtitan.tech)")

# Cap raw_text stored per result so cache entries don't balloon on verbose
# Wiktionary responses (common words have 50+ definitions → 100KB+ JSON).
_RAW_TEXT_CAP = 5000


def _truncate(text: str) -> str:
    if len(text) > _RAW_TEXT_CAP:
        return text[:_RAW_TEXT_CAP] + f"… [truncated at {_RAW_TEXT_CAP} chars]"
    return text


def _encode_title(topic: str) -> str:
    """URL-encode a topic for REST API paths."""
    return urllib.parse.quote(topic.strip(), safe="")


# ── Wiktionary backend ───────────────────────────────────────────────

_WIKTIONARY_URL = "https://en.wiktionary.org/api/rest_v1/page/definition/{w}"


async def fetch_wiktionary(word: str, timeout: float = 10.0) -> BackendResult:
    """Fetch definitions from the Wiktionary REST API.

    Returns all English (``en``) definitions flattened into a single
    raw_text block plus the original structured JSON. Non-English entries
    are ignored. Word is URL-encoded; caller should pass a single lookup
    token (not a phrase).
    """
    result = BackendResult(backend="wiktionary", query=word)
    url = _WIKTIONARY_URL.format(w=_encode_title(word))
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(
                timeout=timeout, headers={"User-Agent": _UA},
                follow_redirects=True) as client:
            resp = await client.get(url)
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.status_code = resp.status_code
        result.bytes_consumed = len(resp.content)

        if resp.status_code == 429:
            result.error_type = "rate_limit"
            return result
        if 500 <= resp.status_code < 600:
            result.error_type = "http_5xx"
            return result
        if resp.status_code == 404:
            result.error_type = "empty"  # word not in dictionary
            return result
        if resp.status_code >= 400:
            result.error_type = "http_4xx"
            result.error_msg = f"HTTP {resp.status_code}"
            return result

        try:
            payload = resp.json()
        except Exception as e:
            result.error_type = "parse_error"
            result.error_msg = str(e)[:200]
            return result

        en_entries = payload.get("en") or []
        if not en_entries:
            result.error_type = "empty"
            return result

        parts = []
        for entry in en_entries:
            pos = entry.get("partOfSpeech", "")
            for defn in entry.get("definitions", []):
                text = defn.get("definition", "").strip()
                if text:
                    parts.append(f"[{pos}] {text}" if pos else text)
        if not parts:
            result.error_type = "empty"
            return result

        result.structured = payload
        result.raw_text = _truncate("\n".join(parts))
        result.success = True
        return result

    except httpx.TimeoutException as e:
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.error_type = "timeout"
        result.error_msg = str(e)[:200]
        return result
    except (httpx.ConnectError, httpx.NetworkError) as e:
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.error_type = "network"
        result.error_msg = str(e)[:200]
        return result
    except Exception as e:
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.error_type = "parse_error"
        result.error_msg = str(e)[:200]
        return result


# ── Free Dictionary backend ──────────────────────────────────────────

_FREE_DICT_URL = "https://api.dictionaryapi.dev/api/v2/entries/en/{w}"


async def fetch_free_dictionary(word: str, timeout: float = 10.0) -> BackendResult:
    """Fetch definitions from the free dictionary API.

    Returns meanings flattened into a raw_text block plus the original
    structured JSON. Lenient but rate-limited; used as Wiktionary fallback.
    """
    result = BackendResult(backend="free_dictionary", query=word)
    url = _FREE_DICT_URL.format(w=_encode_title(word))
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(
                timeout=timeout, headers={"User-Agent": _UA},
                follow_redirects=True) as client:
            resp = await client.get(url)
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.status_code = resp.status_code
        result.bytes_consumed = len(resp.content)

        if resp.status_code == 429:
            result.error_type = "rate_limit"
            return result
        if 500 <= resp.status_code < 600:
            result.error_type = "http_5xx"
            return result
        if resp.status_code == 404:
            result.error_type = "empty"
            return result
        if resp.status_code >= 400:
            result.error_type = "http_4xx"
            result.error_msg = f"HTTP {resp.status_code}"
            return result

        try:
            payload = resp.json()
        except Exception as e:
            result.error_type = "parse_error"
            result.error_msg = str(e)[:200]
            return result

        if not isinstance(payload, list) or not payload:
            result.error_type = "empty"
            return result

        parts = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            for meaning in entry.get("meanings", []) or []:
                pos = meaning.get("partOfSpeech", "")
                for defn in meaning.get("definitions", []) or []:
                    text = (defn.get("definition") or "").strip()
                    if text:
                        parts.append(f"[{pos}] {text}" if pos else text)
        if not parts:
            result.error_type = "empty"
            return result

        result.structured = {"entries": payload}
        result.raw_text = _truncate("\n".join(parts))
        result.success = True
        return result

    except httpx.TimeoutException as e:
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.error_type = "timeout"
        result.error_msg = str(e)[:200]
        return result
    except (httpx.ConnectError, httpx.NetworkError) as e:
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.error_type = "network"
        result.error_msg = str(e)[:200]
        return result
    except Exception as e:
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.error_type = "parse_error"
        result.error_msg = str(e)[:200]
        return result


# ── Wikipedia summary backend ────────────────────────────────────────

_WIKIPEDIA_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{t}"


async def fetch_wikipedia_summary(topic: str, timeout: float = 10.0) -> BackendResult:
    """Fetch an article summary from the Wikipedia REST API.

    Returns the `extract` plain-text summary + title + description in
    structured form. Follows redirects (auto-handles "French revolution"
    → "French Revolution"). Wikipedia tolerates VPS IPs without auth.
    """
    result = BackendResult(backend="wikipedia_direct", query=topic)
    url = _WIKIPEDIA_URL.format(t=_encode_title(topic))
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(
                timeout=timeout, headers={"User-Agent": _UA},
                follow_redirects=True) as client:
            resp = await client.get(url)
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.status_code = resp.status_code
        result.bytes_consumed = len(resp.content)

        if resp.status_code == 429:
            result.error_type = "rate_limit"
            return result
        if 500 <= resp.status_code < 600:
            result.error_type = "http_5xx"
            return result
        if resp.status_code == 404:
            result.error_type = "empty"
            return result
        if resp.status_code >= 400:
            result.error_type = "http_4xx"
            result.error_msg = f"HTTP {resp.status_code}"
            return result

        try:
            payload = resp.json()
        except Exception as e:
            result.error_type = "parse_error"
            result.error_msg = str(e)[:200]
            return result

        extract = (payload.get("extract") or "").strip()
        if not extract:
            result.error_type = "empty"
            return result

        result.structured = {
            "title": payload.get("title", ""),
            "description": payload.get("description", ""),
            "extract": extract,
            "content_urls": payload.get("content_urls", {}),
            "type": payload.get("type", ""),
        }
        result.raw_text = _truncate(extract)
        result.success = True
        return result

    except httpx.TimeoutException as e:
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.error_type = "timeout"
        result.error_msg = str(e)[:200]
        return result
    except (httpx.ConnectError, httpx.NetworkError) as e:
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.error_type = "network"
        result.error_msg = str(e)[:200]
        return result
    except Exception as e:
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.error_type = "parse_error"
        result.error_msg = str(e)[:200]
        return result


# ── SearXNG raw backend ──────────────────────────────────────────────

# Used by WebSearchHelper (agno-tool path) when raw_results=True in the
# dispatcher. Returns concatenated title+snippet+url lines, skipping the
# Sage scrape+distill pipeline entirely. Cheaper — no LLM inference — but
# lower-signal than Sage-distilled output. knowledge_worker still uses
# Sage via dispatcher delegation; this is the agency-tool fast path.


async def fetch_searxng_raw(topic: str,
                             searxng_url: str = "http://localhost:8080/search",
                             max_results: int = 5,
                             timeout: float = 10.0) -> BackendResult:
    """Fetch SearXNG results directly (no Sage, no distillation).

    Returns a concatenated text block of "title — snippet (url)" lines
    plus the parsed JSON payload in structured. Caller (WebSearchHelper)
    uses raw_text; decision log + cache see the same telemetry as direct
    REST backends.
    """
    result = BackendResult(backend="searxng_raw", query=topic)
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(
                timeout=timeout, headers={"User-Agent": _UA},
                follow_redirects=True) as client:
            resp = await client.get(
                searxng_url,
                params={"q": topic, "format": "json", "categories": "general"})
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.status_code = resp.status_code
        result.bytes_consumed = len(resp.content)

        if resp.status_code == 429:
            result.error_type = "rate_limit"
            return result
        if 500 <= resp.status_code < 600:
            result.error_type = "http_5xx"
            return result
        if resp.status_code >= 400:
            result.error_type = "http_4xx"
            result.error_msg = f"HTTP {resp.status_code}"
            return result

        try:
            payload = resp.json()
        except Exception as e:
            result.error_type = "parse_error"
            result.error_msg = str(e)[:200]
            return result

        items = (payload.get("results") or [])[:max_results]
        if not items:
            result.error_type = "empty"
            return result

        lines = []
        for it in items:
            title = (it.get("title") or "").strip()
            content = (it.get("content") or "").strip()[:200]
            url = (it.get("url") or "").strip()
            if title or content:
                lines.append(f"- {title}: {content} ({url})")
        if not lines:
            result.error_type = "empty"
            return result

        result.structured = {"results": items}
        result.raw_text = _truncate(
            f"Found {len(items)} results for '{topic}':\n"
            + "\n".join(lines))
        result.success = True
        return result

    except httpx.TimeoutException as e:
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.error_type = "timeout"
        result.error_msg = str(e)[:200]
        return result
    except (httpx.ConnectError, httpx.NetworkError) as e:
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.error_type = "network"
        result.error_msg = str(e)[:200]
        return result
    except Exception as e:
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        result.error_type = "parse_error"
        result.error_msg = str(e)[:200]
        return result


# ── Backend registry (name → callable) ───────────────────────────────

# The three direct-REST adapters plus fetch_searxng_raw. The dispatcher
# looks up backends here first; searxng_* chain entries without an
# explicit registry match fall back to Sage delegation (or fetch_searxng_raw
# when the caller sets raw_results=True).

BACKEND_REGISTRY = {
    "wiktionary": fetch_wiktionary,
    "free_dictionary": fetch_free_dictionary,
    "wikipedia_direct": fetch_wikipedia_summary,
}


__all__ = [
    "BackendResult",
    "fetch_wiktionary",
    "fetch_free_dictionary",
    "fetch_wikipedia_summary",
    "fetch_searxng_raw",
    "BACKEND_REGISTRY",
]
