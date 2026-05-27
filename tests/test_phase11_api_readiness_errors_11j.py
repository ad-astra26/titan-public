"""
Phase 11 §11.I.5 / D-SPEC-141 — /v6/readiness + /v6/errors (Chunk 11J).

Covers:
  1. Both routes register on the v6 APIRouter.
  2. /v6/readiness returns fleet_ready/fleet_optional_ready snapshot from
     titan_hcl_state.bin plus per-module entries from module_<name>_state.bin.
  3. /v6/errors surfaces every module whose SHM slot carries a non-null
     last_error envelope, sorted newest first.
  4. Both routes are registered in the v6_manifest so /v6/manifest's
     drift check stays clean.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from titan_hcl.core.module_state import (
    BootPriority,
    ModuleStateWriter,
)
from titan_hcl.core.titan_hcl_state import TitanHclStateWriter
from titan_hcl.errors import ModuleError, Severity


def _app_with_v6_router(modules: list[str]) -> FastAPI:
    """Construct a minimal FastAPI app that mounts the v6 router with a
    fake `app.state.titan_hcl.guardian._modules` mapping so the routes'
    orchestrator-driven module-name discovery has something to consume."""
    # Force-reload v6 so the module-level _wire() picks up any test-time
    # router/manifest reset; FastAPI caches handler references via the
    # module's `router` object so a fresh import is needed per test app.
    import importlib

    from titan_hcl.api import v6 as _v6_mod
    importlib.reload(_v6_mod)

    app = FastAPI()
    app.include_router(_v6_mod.router)
    # Mock the orchestrator surface the route's module-name discovery
    # consults: `app.state.titan_hcl.guardian._modules`.
    fake_orch = SimpleNamespace(_modules={name: SimpleNamespace() for name in modules})
    fake_titan_hcl = SimpleNamespace(guardian=fake_orch)
    app.state.titan_hcl = fake_titan_hcl
    return app


# ── 1. Routes register ───────────────────────────────────────────────


def test_v6_readiness_and_errors_routes_registered():
    from titan_hcl.api.v6 import router
    paths = {r.path for r in router.routes}
    assert "/v6/readiness" in paths
    assert "/v6/errors" in paths


# ── 2. /v6/readiness ─────────────────────────────────────────────────


def test_v6_readiness_returns_fleet_and_modules(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    monkeypatch.setenv("TITAN_ID", "test_j")

    # Seed a titan_hcl_state.bin with fleet_ready=True.
    fleet_writer = TitanHclStateWriter(titan_id="test_j")
    fleet_writer.update(
        fleet_ready=True,
        fleet_optional_ready=False,
        boot_phase="phase_a_done",
        mandatory_total=2, mandatory_ready=2,
        post_boot_total=1, post_boot_ready=0,
    )
    # Seed two module slots.
    w_a = ModuleStateWriter(
        module_name="alpha", layer="L2",
        boot_priority=BootPriority.MANDATORY, titan_id="test_j", pid=111)
    w_a.write_state("running")
    w_b = ModuleStateWriter(
        module_name="beta", layer="L2",
        boot_priority=BootPriority.OPTIONAL_POST_BOOT, titan_id="test_j", pid=222)
    w_b.write_state("starting")

    app = _app_with_v6_router(["alpha", "beta", "gamma"])  # gamma has no SHM yet
    try:
        with TestClient(app) as client:
            resp = client.get("/v6/readiness")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["fleet"]["fleet_ready"] is True
        assert body["fleet"]["boot_phase"] == "phase_a_done"
        # Modules: alpha + beta have slots; gamma falls back to "unknown".
        names = {m["name"]: m for m in body["modules"]}
        assert "alpha" in names and names["alpha"]["state"] == "running"
        assert "beta" in names and names["beta"]["state"] == "starting"
        assert "gamma" in names and names["gamma"]["state"] == "unknown"
        assert body["module_count"] == 3
        assert body["module_running_count"] == 1
    finally:
        fleet_writer.close()
        w_a.close()
        w_b.close()


def test_v6_readiness_handles_missing_fleet_slot(tmp_path, monkeypatch):
    """When titan_hcl_state.bin doesn't exist yet (boot ≤1Hz cold start),
    /v6/readiness returns fleet=None without 5xx-ing."""
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    monkeypatch.setenv("TITAN_ID", "test_jb")

    app = _app_with_v6_router(["alpha"])
    with TestClient(app) as client:
        resp = client.get("/v6/readiness")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    # fleet is None — no slot exists.
    assert body["fleet"] is None
    # alpha has no SHM either → state="unknown"
    assert body["modules"][0]["state"] == "unknown"


# ── 3. /v6/errors ────────────────────────────────────────────────────


def test_v6_errors_returns_per_module_last_errors_sorted(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    monkeypatch.setenv("TITAN_ID", "test_je")

    # Write two unhealthy modules with different timestamps + one healthy.
    w_a = ModuleStateWriter(
        module_name="alpha", layer="L2",
        boot_priority=BootPriority.MANDATORY, titan_id="test_je", pid=11)
    err_a = ModuleError(
        module_name="alpha", subsystem="ovg.warmup",
        error_code="OVG_WARMUP_TIMEOUT", severity=Severity.FATAL,
        message="warmup timed out", detail="see brain.log",
        ts=time.time(),
    )
    w_a.write_state("unhealthy", last_error=err_a)
    time.sleep(0.02)
    w_b = ModuleStateWriter(
        module_name="beta", layer="L2",
        boot_priority=BootPriority.MANDATORY, titan_id="test_je", pid=22)
    err_b = ModuleError(
        module_name="beta", subsystem="cache",
        error_code="CACHE_MISS_STORM", severity=Severity.ERROR,
        message="cache miss storm", detail="-",
        ts=time.time(),
    )
    w_b.write_state("unhealthy", last_error=err_b)
    w_c = ModuleStateWriter(
        module_name="gamma", layer="L2",
        boot_priority=BootPriority.OPTIONAL_POST_BOOT, titan_id="test_je", pid=33)
    w_c.write_state("running")

    app = _app_with_v6_router(["alpha", "beta", "gamma"])
    try:
        with TestClient(app) as client:
            resp = client.get("/v6/errors")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["error_count"] == 2
        names = [e["module_name"] for e in body["errors"]]
        assert "alpha" in names
        assert "beta" in names
        assert "gamma" not in names
        # Newest first by ts.
        assert (body["errors"][0]["ts"] >= body["errors"][1]["ts"])
        # Each error envelope carries the state + restart_count fields
        # the route appends from the SHM entry.
        for e in body["errors"]:
            assert e["state"] == "unhealthy"
            assert "restart_count" in e
            assert "error_count_24h" in e
        # Forward-compat history key always present.
        assert body["history"] == []
    finally:
        w_a.close()
        w_b.close()
        w_c.close()


def test_v6_errors_empty_when_no_unhealthy_modules(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    monkeypatch.setenv("TITAN_ID", "test_je_clean")

    w = ModuleStateWriter(
        module_name="alpha", layer="L2",
        boot_priority=BootPriority.MANDATORY,
        titan_id="test_je_clean", pid=11)
    w.write_state("running")

    app = _app_with_v6_router(["alpha"])
    try:
        with TestClient(app) as client:
            resp = client.get("/v6/errors")
        assert resp.status_code == 200
        body = resp.json()
        assert body["error_count"] == 0
        assert body["errors"] == []
    finally:
        w.close()


# ── 4. Manifest registration ────────────────────────────────────────


def test_routes_registered_in_v6_manifest():
    from titan_hcl.api import v6_manifest as _m
    rows = _m.as_rows()
    paths = {row["route"] for row in rows}
    assert "/v6/readiness" in paths
    assert "/v6/errors" in paths
    # The manifest rows have the right group + shm_slots lineage.
    for row in rows:
        if row["route"] in ("/v6/readiness", "/v6/errors"):
            assert row["group"] == "phase11"
            assert row["kind"] == "readout"
            assert "module_<name>_state.bin" in row["shm_slots"]


def test_manifest_drift_check_passes_with_new_routes():
    """/v6/manifest's `drift.in_sync` flag should still be True after
    11J because the two new routes are registered in both the live
    router AND the manifest."""
    from titan_hcl.api import v6_manifest as _m

    live_paths = {"/v6/readiness", "/v6/errors"}
    manifest_paths = {row["route"] for row in _m.as_rows()}
    missing_in_manifest = live_paths - manifest_paths
    assert missing_in_manifest == set(), (
        f"Phase 11 11J routes missing from manifest: {missing_in_manifest}")
