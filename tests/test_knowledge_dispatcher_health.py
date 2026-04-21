"""Integration tests for dispatch() + HealthTracker (KP-4).

Verifies that health-wired dispatch() respects the circuit breaker,
honors budgets, writes decision-log entries, and records telemetry.
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from titan_plugin.logic.knowledge_backends import BackendResult
from titan_plugin.logic.knowledge_cache import KnowledgeCache
from titan_plugin.logic.knowledge_dispatcher import dispatch
from titan_plugin.logic.knowledge_health import CIRCUIT_OPEN, HealthTracker
from titan_plugin.logic.knowledge_router import QueryType


@pytest.fixture
def ctx():
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = os.path.join(tmp, "cache.db")
        health_path = os.path.join(tmp, "health.json")
        decision_path = os.path.join(tmp, "logs", "decisions.jsonl")
        cache = KnowledgeCache(db_path=cache_path, size_cap=100)
        health = HealthTracker(
            health_path=health_path,
            decision_log_path=decision_path,
            cb_fail_threshold=2,
            cb_cooldown_seconds=60,
        )
        yield {
            "cache": cache,
            "health": health,
            "decision_path": decision_path,
        }


class TestDispatchWithHealth:
    @pytest.mark.asyncio
    async def test_circuit_breaker_skips_backend(self, ctx, monkeypatch):
        from titan_plugin.logic import knowledge_dispatcher as kd
        wikt_calls = 0

        async def wikt_failing(topic, timeout=10.0):
            nonlocal wikt_calls
            wikt_calls += 1
            return BackendResult(
                backend="wiktionary", query=topic, success=False,
                error_type="http_5xx", bytes_consumed=50)

        async def other_failing(topic, timeout=10.0):
            return BackendResult(
                backend="x", query=topic, success=False,
                error_type="empty", bytes_consumed=10)

        monkeypatch.setitem(kd.BACKEND_REGISTRY, "wiktionary", wikt_failing)
        monkeypatch.setitem(kd.BACKEND_REGISTRY, "free_dictionary", other_failing)
        monkeypatch.setitem(kd.BACKEND_REGISTRY, "wikipedia_direct", other_failing)

        # Fire 2 failures on wiktionary (chain always probes it first)
        await dispatch("chi", cache=ctx["cache"], health=ctx["health"])
        await dispatch("ontology", cache=ctx["cache"], health=ctx["health"])
        assert ctx["health"]._backends["wiktionary"].circuit_state == CIRCUIT_OPEN

        # 3rd dispatch — circuit is open, wiktionary should be SKIPPED
        wikt_calls_before = wikt_calls
        out = await dispatch("philosophy", cache=ctx["cache"],
                              health=ctx["health"])
        assert wikt_calls == wikt_calls_before  # wiktionary not called
        # The circuit_open marker should appear in attempts
        assert any(a[0] == "wiktionary" and a[1] == "circuit_open"
                   for a in out.attempts)

    @pytest.mark.asyncio
    async def test_successful_dispatch_writes_decision_log(self, ctx, monkeypatch):
        from titan_plugin.logic import knowledge_dispatcher as kd

        async def ok(topic, timeout=10.0):
            return BackendResult(
                backend="wiktionary", query=topic, success=True,
                raw_text="def", bytes_consumed=50)

        monkeypatch.setitem(kd.BACKEND_REGISTRY, "wiktionary", ok)
        await dispatch("hypothesis", cache=ctx["cache"],
                       health=ctx["health"], requestor="test")

        assert os.path.exists(ctx["decision_path"])
        with open(ctx["decision_path"]) as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["query"] == "hypothesis"
        assert entry["query_type"] == "dictionary"
        assert entry["backend_used"] == "wiktionary"
        assert entry["success"] is True
        assert entry["requestor"] == "test"
        assert entry["rejected"] is False
        assert entry["bytes"] == 50

    @pytest.mark.asyncio
    async def test_rejected_query_writes_decision_log(self, ctx):
        await dispatch("inner_spirit", cache=ctx["cache"],
                       health=ctx["health"], requestor="self_reasoning")
        with open(ctx["decision_path"]) as f:
            entry = json.loads(f.readline())
        assert entry["rejected"] is True
        assert entry["query_type"] == "internal_rejected"
        assert entry["requestor"] == "self_reasoning"

    @pytest.mark.asyncio
    async def test_health_records_every_attempt(self, ctx, monkeypatch):
        from titan_plugin.logic import knowledge_dispatcher as kd

        async def ok(topic, timeout=10.0):
            return BackendResult(
                backend="wiktionary", query=topic, success=True,
                raw_text="def", bytes_consumed=100, latency_ms=50)

        monkeypatch.setitem(kd.BACKEND_REGISTRY, "wiktionary", ok)
        await dispatch("hypothesis", cache=ctx["cache"], health=ctx["health"])
        snap = ctx["health"].snapshot()
        assert snap["backends"]["wiktionary"]["requests_today"] == 1
        assert snap["backends"]["wiktionary"]["bytes_consumed_today"] == 100
        assert snap["backends"]["wiktionary"]["last_success_ts"] > 0

    @pytest.mark.asyncio
    async def test_budget_exhausted_skips_backend(self, ctx, monkeypatch):
        from titan_plugin.logic import knowledge_dispatcher as kd

        async def ok(topic, timeout=10.0):
            return BackendResult(
                backend="wiktionary", query=topic, success=True,
                raw_text="def", bytes_consumed=600)

        monkeypatch.setitem(kd.BACKEND_REGISTRY, "wiktionary", ok)
        ctx["health"].set_budget("wiktionary", 1000)  # 1000-byte budget

        await dispatch("hypothesis", cache=ctx["cache"], health=ctx["health"])
        # 600 bytes consumed, 400 remaining — budget check still OK
        r2 = await dispatch("ontology", cache=ctx["cache"], health=ctx["health"])
        # 1200 consumed — over budget. Next call blocked at budget gate
        r3 = await dispatch("philosophy", cache=ctx["cache"], health=ctx["health"])
        # r3 should show budget_exceeded attempts for wiktionary
        attempts_r3 = r3.attempts
        assert any(a[0] == "wiktionary" and a[1] == "budget_exceeded"
                   for a in attempts_r3)

    @pytest.mark.asyncio
    async def test_near_dup_detection_logs(self, ctx, monkeypatch, caplog):
        import logging
        import os as _os
        import tempfile as _tmp

        # Create a fresh tracker with lower threshold — default 0.8 is
        # strict and our test tokens Jaccard at 0.75.
        with _tmp.TemporaryDirectory() as tmp:
            low_health = HealthTracker(
                health_path=_os.path.join(tmp, "h.json"),
                decision_log_path=_os.path.join(tmp, "d.jsonl"),
                near_dup_jaccard=0.5)

            caplog.set_level(logging.INFO,
                             logger="titan_plugin.logic.knowledge_health")

            # Seed two conceptual queries with >50% token overlap
            low_health.note_query_for_near_dup("hypothesis generation critical")
            match = low_health.note_query_for_near_dup(
                "hypothesis generation thinking critical")
            assert match == "hypothesis generation critical"

            near_dup_logs = [r for r in caplog.records if "near-dup" in r.message]
            assert len(near_dup_logs) >= 1
