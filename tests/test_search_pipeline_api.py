"""Unit tests for /v4/search-pipeline/* endpoints (KP-5).

Uses FastAPI TestClient with a mounted dashboard router and a stub
titan_plugin providing .bus.publish(). Filesystem paths point to a tmp
directory so tests don't touch live data.
"""

import json
import os
import tempfile
import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from titan_plugin.api.dashboard import router as dashboard_router
from titan_plugin.logic.knowledge_cache import KnowledgeCache
from titan_plugin.logic.knowledge_health import HealthTracker
from titan_plugin.logic.knowledge_router import QueryType


@pytest.fixture
def app_client(monkeypatch):
    """Build a tiny FastAPI app with just the dashboard router + stub plugin.

    Points the endpoints at a tmp directory by chdir'ing into it. Restores
    the original cwd on teardown so later fixtures (and tests) don't see
    a stale cwd referencing a deleted tmpdir.
    """
    _original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        os.makedirs("data/logs", exist_ok=True)

        # Seed health.json with some realistic backend state
        health = HealthTracker(
            health_path="data/knowledge_pipeline_health.json",
            decision_log_path="data/logs/knowledge_router_decisions.jsonl",
            budgets={"wiktionary": 50 * 1024 * 1024},
        )
        health.record_attempt("wiktionary", success=True,
                               bytes_consumed=1024, latency_ms=180)
        health.record_attempt("wiktionary", success=True,
                               bytes_consumed=2048, latency_ms=200)
        health.record_attempt("wikipedia_direct", success=False,
                               error_type="http_5xx", bytes_consumed=0)

        # Seed search_cache.db
        cache = KnowledgeCache(db_path="data/search_cache.db")
        cache.put(query_hash="h1", query_text="chi",
                  query_type=QueryType.DICTIONARY,
                  backend="wiktionary",
                  result_payload={"raw_text": "Greek letter"},
                  success=True, bytes_consumed=512)
        cache.put(query_hash="h2", query_text="mitochondria",
                  query_type=QueryType.WIKIPEDIA_LIKE,
                  backend="wikipedia_direct",
                  result_payload={"raw_text": "Cell organelle"},
                  success=True, bytes_consumed=1024)

        # Build app
        app = FastAPI()
        app.include_router(dashboard_router)
        # Stub plugin for budget-reset endpoint
        plugin = MagicMock()
        plugin.bus = MagicMock()
        plugin.bus.publish = MagicMock()
        app.state.titan_plugin = plugin

        with TestClient(app) as client:
            try:
                yield client, plugin
            finally:
                os.chdir(_original_cwd)


# ── GET /v4/search-pipeline/health ───────────────────────────────────

class TestPipelineHealth:
    def test_returns_backends_and_cache(self, app_client):
        client, _ = app_client
        resp = client.get("/v4/search-pipeline/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        data = body["data"]
        assert "backends" in data
        assert "cache" in data
        assert "wiktionary" in data["backends"]
        assert data["backends"]["wiktionary"]["requests_today"] == 2
        assert data["backends"]["wiktionary"]["bytes_consumed_today"] == 3072
        assert data["cache"]["entries"] == 2
        assert data["health_file_age_sec"] is not None

    def test_missing_health_file_ok_with_empty_backends(self, tmp_path,
                                                         monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = FastAPI()
        app.include_router(dashboard_router)
        app.state.titan_plugin = MagicMock()
        with TestClient(app) as client:
            resp = client.get("/v4/search-pipeline/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["backends"] == {}


# ── GET /v4/search-pipeline/backend/{name} ───────────────────────────

class TestPipelineBackend:
    def test_returns_backend_slice(self, app_client):
        client, _ = app_client
        resp = client.get("/v4/search-pipeline/backend/wiktionary")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["requests_today"] == 2
        assert data["circuit_state"] == "closed"
        assert data["cache_entries"] == 1  # only "chi" cached under wiktionary

    def test_unknown_backend_returns_error(self, app_client):
        client, _ = app_client
        resp = client.get("/v4/search-pipeline/backend/doesnotexist")
        assert resp.status_code == 500
        body = resp.json()
        assert "unknown backend" in body["detail"]
        assert "wiktionary" in body["detail"]   # surfaces known list

    def test_errored_backend_still_returned(self, app_client):
        client, _ = app_client
        resp = client.get("/v4/search-pipeline/backend/wikipedia_direct")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["last_error_type"] == "http_5xx"
        assert data["errors_today"] == 1


# ── POST /v4/search-pipeline/budget-reset ────────────────────────────

class TestBudgetReset:
    def test_reset_all_backends(self, app_client):
        client, plugin = app_client
        resp = client.post("/v4/search-pipeline/budget-reset", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["queued"] is True
        assert body["data"]["backend"] == "ALL"
        # One bus publish should have fired
        assert plugin.bus.publish.called
        args, kwargs = plugin.bus.publish.call_args
        # make_msg() returns a dict-like — just verify the type key
        msg = args[0]
        assert msg["type"] == "SEARCH_PIPELINE_BUDGET_RESET"
        assert msg["dst"] == "knowledge"
        assert msg["payload"]["backend"] == ""

    def test_reset_single_backend(self, app_client):
        client, plugin = app_client
        resp = client.post("/v4/search-pipeline/budget-reset",
                            json={"backend": "wiktionary"})
        assert resp.status_code == 200
        assert resp.json()["data"]["backend"] == "wiktionary"
        msg = plugin.bus.publish.call_args[0][0]
        assert msg["payload"]["backend"] == "wiktionary"

    def test_empty_body_equivalent_to_all(self, app_client):
        client, plugin = app_client
        resp = client.post("/v4/search-pipeline/budget-reset")
        assert resp.status_code == 200
        assert resp.json()["data"]["backend"] == "ALL"
