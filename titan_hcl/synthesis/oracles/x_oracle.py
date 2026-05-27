"""Phase 6 — `x_oracle` TruthOraclePlug (§P6.E; SPEC §25.3 + §25.5).

Wraps the existing `SocialXGateway`
(`titan_hcl/logic/social_x_gateway.py`) read methods as a
`TruthOraclePlug` for X (Twitter) social-truth claims.

Per memory rule `feedback_social_x_gateway_post_is_sole_sanctioned_x_path`:
**all X traffic flows through SocialXGateway** — this oracle is a
read-only consumer; it never posts. SocialXGateway's write surface is
reserved for the matching `x_research` ToolPlug (P6.I).

Per SPEC §25.3 day-one set + arch §11.1: metered oracle (cost class
``"metered"``; oracle_id ``x_api``). The INV-Syn-13 gate enforces
``daily_sol_budget["x_api"]`` from
``titan_params.toml [synthesis.oracle.daily_sol_budget]``.

Claim domains served (SPEC §25.3 + arch §11.1):

- **topic_trending** — "is this topic actually trending on X right now?"
  Payload: ``{"topic": "<phrase>"}``. The plug calls
  ``search_tweets(query=topic, queryType="Latest", count=20)`` and
  counts distinct authors in the last 24h. Verdict:
    - ``"true"`` if distinct-author count ≥ ``min_distinct_authors`` (default 5).
    - ``"false"`` if < that threshold.
    - ``"unknown"`` on gateway error / missing api_key / circuit-open.

- **account_exists** — "is X handle Y a real, live account?"
  Payload: ``{"handle": "<username without @>"}``. The plug calls
  ``fetch_recent_tweets(handle, count=1)``. Verdict:
    - ``"true"`` if response carries ≥ 1 tweet OR a recognizable user
      object.
    - ``"false"`` if the gateway returns the upstream "user not found"
      marker.
    - ``"unknown"`` on gateway error / malformed response.

- **post_real** — "does post id P exist and resolve to non-deleted state?"
  Payload: ``{"post_id": "<tweet_id>"}``. The plug calls
  ``search_tweets(query=post_id, queryType="Latest", count=5)`` and
  looks for the id in returned results. Verdict:
    - ``"true"`` on direct id match in results.
    - ``"false"`` if response is empty or no result matches.
    - ``"unknown"`` on gateway error.

- **x_event_real** — "is this event/topic real per a live-conversation
  signal threshold?" Payload: ``{"event_query": "<query>"}``. Same
  search call as `topic_trending` but with a stricter recency lens
  (≥ 3 posts from distinct accounts in last 24h). Verdict mapping
  parallels `topic_trending`.

The gateway's responses are dicts shaped by twitterapi.io — keys
vary (``tweets`` / ``data`` / ``items``); the plug parses defensively
so a schema drift surfaces as ``"unknown"`` (not a hard crash).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Optional, Protocol

from titan_hcl.synthesis.plugs import OracleClaim, OracleVerdict

logger = logging.getLogger(__name__)


SUPPORTED_DOMAINS = frozenset(
    {"topic_trending", "account_exists", "post_real", "x_event_real"}
)

DEFAULT_PER_CALL_COST_SOL: float = 0.0003   # twitterapi.io quota-amortized
DEFAULT_MIN_DISTINCT_AUTHORS_TRENDING: int = 5
DEFAULT_MIN_DISTINCT_AUTHORS_EVENT: int = 3
DEFAULT_RECENCY_WINDOW_S: float = 24 * 3600.0  # 24h


class GatewayLike(Protocol):
    """Minimum surface the oracle needs from SocialXGateway."""

    def search_tweets(
        self, query: str, query_type: str = "Latest", count: int = 20, *, api_key: str = ""
    ) -> dict: ...
    def fetch_recent_tweets(
        self, user_name: str, count: int = 10, *, api_key: str = ""
    ) -> dict: ...


def _extract_tweets(resp: dict) -> list[dict]:
    """Defensive parser — twitterapi.io schemas drift across endpoints."""
    if not isinstance(resp, dict):
        return []
    for key in ("tweets", "data", "items", "results"):
        v = resp.get(key)
        if isinstance(v, list):
            return [t for t in v if isinstance(t, dict)]
    # Nested shape: {"data": {"tweets": [...]}}
    data = resp.get("data")
    if isinstance(data, dict):
        for key in ("tweets", "items", "results"):
            v = data.get(key)
            if isinstance(v, list):
                return [t for t in v if isinstance(t, dict)]
    return []


def _author_of(tweet: dict) -> Optional[str]:
    """Pull the author handle/id out of a tweet dict (schema-tolerant)."""
    for k in ("author_name", "user_name", "author", "username", "screen_name"):
        v = tweet.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    author = tweet.get("user")
    if isinstance(author, dict):
        for k in ("screen_name", "user_name", "username"):
            v = author.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
    author = tweet.get("author")
    if isinstance(author, dict):
        for k in ("screen_name", "user_name", "username"):
            v = author.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
    return None


def _tweet_id(tweet: dict) -> Optional[str]:
    for k in ("id", "tweet_id", "id_str"):
        v = tweet.get(k)
        if v is not None:
            return str(v)
    return None


def _tweet_ts(tweet: dict) -> Optional[float]:
    """Best-effort timestamp parse (seconds since epoch); None if absent."""
    for k in ("timestamp", "created_at_ts", "ts"):
        v = tweet.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    # ISO8601-ish "created_at" — parse leniently.
    raw = tweet.get("created_at")
    if isinstance(raw, str) and raw.strip():
        try:
            from datetime import datetime
            # Try fromisoformat first (handles "2026-05-27T14:00:00+00:00").
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None
    return None


def _is_user_not_found(resp: dict) -> bool:
    """Heuristic — twitterapi.io returns various "not found" markers."""
    if not isinstance(resp, dict):
        return False
    status = str(resp.get("status", "")).lower()
    message = str(resp.get("message", "")).lower()
    return (
        status in ("error", "not_found")
        and ("not found" in message or "no user" in message or "does not exist" in message)
    )


def _gateway_error(resp: dict) -> bool:
    """True if the response signals an upstream error (vs valid-but-empty)."""
    if not isinstance(resp, dict):
        return True
    status = str(resp.get("status", "")).lower()
    if status in ("error", "failed"):
        return True
    if "error" in resp and resp["error"]:
        return True
    return False


class XOracle:
    """TruthOraclePlug wrapping SocialXGateway read methods.

    Constructed with a `gateway` instance (real `SocialXGateway` in
    production; mock in unit tests) and the api_key the gateway needs
    for outbound calls. The api_key resolution belongs to the
    synthesis_worker boot path (P6.F); the oracle treats it opaquely.
    """

    oracle_id: str = "x_api"           # matches daily_sol_budget key
    cost_class: str = "metered"        # INV-Syn-13 gated

    def __init__(
        self,
        gateway: GatewayLike,
        *,
        api_key: str = "",
        per_call_cost_sol: float = DEFAULT_PER_CALL_COST_SOL,
        min_distinct_authors_trending: int = DEFAULT_MIN_DISTINCT_AUTHORS_TRENDING,
        min_distinct_authors_event: int = DEFAULT_MIN_DISTINCT_AUTHORS_EVENT,
        recency_window_s: float = DEFAULT_RECENCY_WINDOW_S,
    ):
        self._gateway = gateway
        self._api_key = api_key
        self._per_call_cost_sol = float(per_call_cost_sol)
        self._min_trending = int(min_distinct_authors_trending)
        self._min_event = int(min_distinct_authors_event)
        self._recency_window_s = float(recency_window_s)

    def can_handle(self, domain: str) -> bool:
        return domain in SUPPORTED_DOMAINS

    def verify(self, claim: OracleClaim) -> OracleVerdict:
        t0 = time.perf_counter()
        ts_now = time.time()

        if not self.can_handle(claim.domain):
            return self._verdict(t0, ts_now, "unknown", "domain_unsupported")

        payload = claim.payload or {}

        try:
            if claim.domain == "topic_trending":
                return self._verify_topic_trending(payload, t0, ts_now)
            if claim.domain == "account_exists":
                return self._verify_account_exists(payload, t0, ts_now)
            if claim.domain == "post_real":
                return self._verify_post_real(payload, t0, ts_now)
            if claim.domain == "x_event_real":
                return self._verify_x_event_real(payload, t0, ts_now)
        except Exception:
            logger.exception("[x_oracle] verify() raised")
            return self._verdict(t0, ts_now, "unknown", "gateway_exception", cost=self._per_call_cost_sol)

        return self._verdict(  # pragma: no cover — unreachable by can_handle filter
            t0, ts_now, "unknown", "domain_unsupported"
        )

    # ── domain-specific verifiers ───────────────────────────────────────

    def _verify_topic_trending(
        self, payload: dict[str, Any], t0: float, ts_now: float
    ) -> OracleVerdict:
        topic = str(payload.get("topic", "")).strip()
        if not topic:
            return self._verdict(t0, ts_now, "unknown", "missing_topic")
        resp = self._gateway.search_tweets(query=topic, query_type="Latest", count=20, api_key=self._api_key)
        if _gateway_error(resp):
            return self._verdict(t0, ts_now, "unknown", "gateway_error", cost=self._per_call_cost_sol)
        return self._distinct_authors_verdict(
            resp, ts_now, t0, evidence=topic,
            min_authors=self._min_trending,
        )

    def _verify_x_event_real(
        self, payload: dict[str, Any], t0: float, ts_now: float
    ) -> OracleVerdict:
        query = str(payload.get("event_query", "")).strip()
        if not query:
            return self._verdict(t0, ts_now, "unknown", "missing_event_query")
        resp = self._gateway.search_tweets(query=query, query_type="Latest", count=20, api_key=self._api_key)
        if _gateway_error(resp):
            return self._verdict(t0, ts_now, "unknown", "gateway_error", cost=self._per_call_cost_sol)
        return self._distinct_authors_verdict(
            resp, ts_now, t0, evidence=query,
            min_authors=self._min_event,
        )

    def _verify_account_exists(
        self, payload: dict[str, Any], t0: float, ts_now: float
    ) -> OracleVerdict:
        handle = str(payload.get("handle", "")).strip().lstrip("@")
        if not handle:
            return self._verdict(t0, ts_now, "unknown", "missing_handle")
        resp = self._gateway.fetch_recent_tweets(user_name=handle, count=1, api_key=self._api_key)
        if _is_user_not_found(resp):
            return self._verdict(t0, ts_now, "false", handle, cost=self._per_call_cost_sol)
        if _gateway_error(resp):
            return self._verdict(t0, ts_now, "unknown", "gateway_error", cost=self._per_call_cost_sol)
        tweets = _extract_tweets(resp)
        if tweets:
            return self._verdict(t0, ts_now, "true", handle, cost=self._per_call_cost_sol)
        # Some twitterapi.io responses carry user metadata even when
        # tweets list is empty (silent account with 0 posts). Look for
        # a user object anywhere in the response.
        if any(k in resp for k in ("user", "user_info", "profile")):
            return self._verdict(t0, ts_now, "true", handle, cost=self._per_call_cost_sol)
        # No tweets + no user marker — treat as not-found (NOT a transient).
        return self._verdict(t0, ts_now, "false", handle, cost=self._per_call_cost_sol)

    def _verify_post_real(
        self, payload: dict[str, Any], t0: float, ts_now: float
    ) -> OracleVerdict:
        post_id = str(payload.get("post_id", "")).strip()
        if not post_id:
            return self._verdict(t0, ts_now, "unknown", "missing_post_id")
        resp = self._gateway.search_tweets(query=post_id, query_type="Latest", count=5, api_key=self._api_key)
        if _gateway_error(resp):
            return self._verdict(t0, ts_now, "unknown", "gateway_error", cost=self._per_call_cost_sol)
        tweets = _extract_tweets(resp)
        for t in tweets:
            tid = _tweet_id(t)
            if tid == post_id:
                return self._verdict(t0, ts_now, "true", post_id, cost=self._per_call_cost_sol)
        return self._verdict(t0, ts_now, "false", post_id, cost=self._per_call_cost_sol)

    # ── shared helper ───────────────────────────────────────────────────

    def _distinct_authors_verdict(
        self,
        resp: dict,
        ts_now: float,
        t0: float,
        *,
        evidence: str,
        min_authors: int,
    ) -> OracleVerdict:
        tweets = _extract_tweets(resp)
        if not tweets:
            return self._verdict(t0, ts_now, "false", evidence, cost=self._per_call_cost_sol)
        cutoff = ts_now - self._recency_window_s
        authors: set[str] = set()
        for t in tweets:
            ts = _tweet_ts(t)
            if ts is not None and ts < cutoff:
                continue   # outside recency window
            author = _author_of(t)
            if author:
                authors.add(author)
        verdict = "true" if len(authors) >= min_authors else "false"
        evidence_ref = (
            evidence
            if evidence
            else hashlib.sha256(json.dumps(sorted(authors)).encode()).hexdigest()
        )
        return self._verdict(t0, ts_now, verdict, evidence_ref, cost=self._per_call_cost_sol)

    # ── plumbing ────────────────────────────────────────────────────────

    def _verdict(
        self,
        t0: float,
        ts_now: float,
        verdict: str,
        evidence_ref: str,
        *,
        cost: float = 0.0,
    ) -> OracleVerdict:
        return OracleVerdict(
            oracle_id=self.oracle_id,
            verdict=verdict,  # type: ignore[arg-type]
            evidence_ref=evidence_ref,
            cost=cost,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            ts=ts_now,
        )


__all__ = (
    "XOracle",
    "SUPPORTED_DOMAINS",
    "DEFAULT_PER_CALL_COST_SOL",
    "DEFAULT_MIN_DISTINCT_AUTHORS_TRENDING",
    "DEFAULT_MIN_DISTINCT_AUTHORS_EVENT",
    "GatewayLike",
)
