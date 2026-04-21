"""Unit tests for titan_plugin.logic.knowledge_errors (KP-4)."""

import pytest

from titan_plugin.logic.knowledge_errors import (
    Backend4xx,
    BackendDown,
    BackendEmpty,
    BackendNetworkError,
    BackendRateLimit,
    BackendTimeout,
    BandwidthBudgetExceeded,
    CircuitBreakerOpen,
    KnowledgePipelineError,
    ProxyQuotaExhausted,
    QueryMalformed,
    QueryRejectedInternal,
    error_type_to_exception,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        subclasses = [
            ProxyQuotaExhausted, BackendRateLimit, BackendDown, BackendTimeout,
            BackendNetworkError, QueryMalformed, QueryRejectedInternal,
            BackendEmpty, Backend4xx, BandwidthBudgetExceeded,
            CircuitBreakerOpen,
        ]
        for cls in subclasses:
            assert issubclass(cls, KnowledgePipelineError)
            assert issubclass(cls, Exception)

    def test_context_fields(self):
        e = BackendRateLimit("slow down", backend="wiktionary",
                              status_code=429, query="python")
        assert e.backend == "wiktionary"
        assert e.status_code == 429
        assert e.query == "python"
        assert "slow down" in str(e)

    def test_error_type_classvar_stable(self):
        # These strings are in persisted health.json + decision log —
        # renaming breaks on-disk state. Lock them.
        assert ProxyQuotaExhausted.error_type == "proxy_quota"
        assert BackendRateLimit.error_type == "rate_limit"
        assert BackendDown.error_type == "http_5xx"
        assert BackendTimeout.error_type == "timeout"
        assert BackendNetworkError.error_type == "network"
        assert QueryMalformed.error_type == "parse_error"
        assert QueryRejectedInternal.error_type == "internal_rejected"
        assert BackendEmpty.error_type == "empty"
        assert Backend4xx.error_type == "http_4xx"
        assert BandwidthBudgetExceeded.error_type == "budget_exceeded"
        assert CircuitBreakerOpen.error_type == "circuit_open"


class TestErrorTypeLookup:
    @pytest.mark.parametrize("tag,cls", [
        ("timeout", BackendTimeout),
        ("rate_limit", BackendRateLimit),
        ("http_5xx", BackendDown),
        ("http_4xx", Backend4xx),
        ("empty", BackendEmpty),
        ("parse_error", QueryMalformed),
        ("network", BackendNetworkError),
        ("budget_exceeded", BandwidthBudgetExceeded),
        ("circuit_open", CircuitBreakerOpen),
        ("internal_rejected", QueryRejectedInternal),
        ("proxy_quota", ProxyQuotaExhausted),
    ])
    def test_tag_maps_to_class(self, tag, cls):
        e = error_type_to_exception(tag, "msg", backend="b", query="q")
        assert isinstance(e, cls)
        assert e.backend == "b"

    def test_empty_tag_returns_none(self):
        assert error_type_to_exception("") is None

    def test_unknown_tag_returns_base(self):
        e = error_type_to_exception("banana", backend="x")
        assert isinstance(e, KnowledgePipelineError)
        assert e.backend == "x"

    def test_repr_includes_context(self):
        e = BackendTimeout("", backend="wiktionary", status_code=0,
                            query="python async")
        r = repr(e)
        assert "BackendTimeout" in r
        assert "wiktionary" in r
