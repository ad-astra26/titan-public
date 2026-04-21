"""
knowledge_errors — structured exception taxonomy for the knowledge pipeline.

Maps the stable string `error_type` field carried on BackendResult to typed
exceptions callers can catch selectively. Every backend fetch records a
string tag (rFP §3.3); this module lifts that tag to a real exception
hierarchy so downstream code (retry policy, Maker alerts, bus events,
arch_map panels) can pattern-match without string comparisons.

Per rFP_knowledge_pipeline_v2.md §3.3.
"""

from __future__ import annotations

from typing import Optional


class KnowledgePipelineError(Exception):
    """Base for every typed pipeline error.

    Exposes backend name + original string tag so catchers can log
    structured context without knowing each subclass individually.
    """

    # Stable mapping from BackendResult.error_type → subclass. Populated
    # at module load by subclass declarations below.
    error_type: str = ""

    def __init__(self, message: str = "",
                 *, backend: str = "", status_code: int = 0,
                 query: str = ""):
        super().__init__(message or self.error_type)
        self.backend = backend
        self.status_code = status_code
        self.query = query

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(backend={self.backend!r}, "
                f"status={self.status_code}, query={self.query[:40]!r})")


class ProxyQuotaExhausted(KnowledgePipelineError):
    """Upstream proxy ran out of quota (Webshare 402-style).

    Distinct from BackendRateLimit because the remedy is different
    (renew plan / rotate proxy vs wait-and-retry).
    """
    error_type = "proxy_quota"


class BackendRateLimit(KnowledgePipelineError):
    """Backend returned HTTP 429 or equivalent."""
    error_type = "rate_limit"


class BackendDown(KnowledgePipelineError):
    """Backend returned 5xx or is unreachable (persistent)."""
    error_type = "http_5xx"


class BackendTimeout(KnowledgePipelineError):
    """Backend exceeded per-request timeout.

    Not cached (Q-KP3) — next call retries from scratch.
    """
    error_type = "timeout"


class BackendNetworkError(KnowledgePipelineError):
    """Connection refused / DNS failure / transport failure.

    Not cached (Q-KP3) — next call retries.
    """
    error_type = "network"


class QueryMalformed(KnowledgePipelineError):
    """Query was invalid (empty, unparseable, too long, etc.)."""
    error_type = "parse_error"


class QueryRejectedInternal(KnowledgePipelineError):
    """Titan-internal name rejected at router entry.

    Raised only when callers explicitly opt-in to exception-mode
    (knowledge_dispatcher defaults to returning DispatchResult.rejected
    instead). This exception exists for KP-5 API error paths that want
    HTTP 400 semantics.
    """
    error_type = "internal_rejected"


class BackendEmpty(KnowledgePipelineError):
    """Backend returned 200 OK with zero usable content."""
    error_type = "empty"


class Backend4xx(KnowledgePipelineError):
    """Backend returned 4xx other than 429 (bad request, 404, etc.)."""
    error_type = "http_4xx"


class BandwidthBudgetExceeded(KnowledgePipelineError):
    """Per-backend daily bandwidth budget reached — request blocked.

    Circuit breaker keeps this state until daily reset. Surface via
    SEARCH_PIPELINE_BUDGET_EXCEEDED bus event + Maker Telegram alert
    (KP-7 wiring).
    """
    error_type = "budget_exceeded"


class CircuitBreakerOpen(KnowledgePipelineError):
    """Backend circuit breaker is open — request skipped without trying.

    Set by HealthTracker after N consecutive failures within a window.
    Recovery: half-open probe after M seconds → closed on first success.
    """
    error_type = "circuit_open"


# ── Lookup table ─────────────────────────────────────────────────────

_ERROR_TYPE_TO_CLASS = {
    cls.error_type: cls for cls in [
        ProxyQuotaExhausted, BackendRateLimit, BackendDown, BackendTimeout,
        BackendNetworkError, QueryMalformed, QueryRejectedInternal,
        BackendEmpty, Backend4xx, BandwidthBudgetExceeded,
        CircuitBreakerOpen,
    ]
}


def error_type_to_exception(error_type: str, message: str = "",
                             *, backend: str = "", status_code: int = 0,
                             query: str = "") -> Optional[KnowledgePipelineError]:
    """Instantiate the matching exception class for a stable error_type.

    Returns None if error_type is empty string or unknown (caller should
    treat as generic KnowledgePipelineError or skip).
    """
    if not error_type:
        return None
    cls = _ERROR_TYPE_TO_CLASS.get(error_type, KnowledgePipelineError)
    return cls(message, backend=backend, status_code=status_code, query=query)


__all__ = [
    "KnowledgePipelineError",
    "ProxyQuotaExhausted",
    "BackendRateLimit",
    "BackendDown",
    "BackendTimeout",
    "BackendNetworkError",
    "QueryMalformed",
    "QueryRejectedInternal",
    "BackendEmpty",
    "Backend4xx",
    "BandwidthBudgetExceeded",
    "CircuitBreakerOpen",
    "error_type_to_exception",
]
