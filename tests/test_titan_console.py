"""TC² Console Agent — backend unit tests (W8, stdlib-only agent).

All side effects (subprocess, HTTP to api_hcl) are injected via Context, so
nothing here touches the real shell or network. host.py reads real /proc on
the Linux test box (structure-asserted only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
# config_api imports setup_titan.config_model — make it importable (append, so
# the real titan_hcl package isn't shadowed by scripts/titan_hcl.py).
sys.path.append(str(_REPO / "scripts"))

from titan_console import config_api, host, ops, proxy  # noqa: E402
from titan_console.agent import dispatch  # noqa: E402
from titan_console.context import Context, resolve_titan_id  # noqa: E402
from titan_console.titan_status import titan_status  # noqa: E402


# ── fakes ────────────────────────────────────────────────────────────────


def _ctx(tmp_path, *, run=None, http=None, token=None, titan_id="T1") -> Context:
    return Context(
        install_root=tmp_path, titan_id=titan_id,
        run=run or (lambda argv, **k: (0, "", "")),
        http=http or (lambda *a, **k: (0, b"")),
        token=token,
    )


# ── host ─────────────────────────────────────────────────────────────────


def test_host_resources_structure():
    r = host.read_host_resources(sample_cpu=False)
    assert set(r) >= {"cpu", "memory", "swap", "disk", "uptime_s"}
    assert r["cpu"]["count"] >= 1
    assert r["memory"]["total"] > 0
    assert r["disk"]["total"] is None or r["disk"]["total"] > 0


# ── titan_status ───────────────────────────────────────────────────────────


def test_titan_status_up(tmp_path):
    def run(argv, **k):
        if "show" in argv:
            return 0, "ActiveState=active\nSubState=running\n", ""
        return 0, "", ""

    def http(method, url, **k):
        return (200, json.dumps({"status": "healthy", "modules": "32/32"}).encode())

    s = titan_status(_ctx(tmp_path, run=run, http=http))
    assert s["up"] is True
    assert s["health"]["status"] == "healthy"
    assert s["why_down"] is None


def test_titan_status_down_reports_why_and_journal(tmp_path):
    def run(argv, **k):
        if "show" in argv:
            return 0, "ActiveState=failed\nSubState=dead\n", ""
        if argv[0] == "journalctl":
            return 0, "2026-05-30 line1\n2026-05-30 OOM-killed\n", ""
        return 0, "", ""

    s = titan_status(_ctx(tmp_path, run=run))
    assert s["up"] is False
    assert "failed/dead" in s["why_down"]
    assert any("OOM" in ln for ln in s["journal_tail"])


def test_titan_status_half_up_when_service_active_but_api_dead(tmp_path):
    def run(argv, **k):
        if "show" in argv:
            return 0, "ActiveState=active\nSubState=running\n", ""
        return 0, "boot log\n", ""

    def http(method, url, **k):
        return (0, b"")  # api unreachable

    s = titan_status(_ctx(tmp_path, run=run, http=http))
    assert s["up"] is False
    assert "api_hcl" in s["why_down"]


# ── ops ────────────────────────────────────────────────────────────────────


def test_restart_builds_dreaming_aware_command(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "t1_manage.sh").write_text("#!/bin/bash\n")
    seen = {}

    def run(argv, **k):
        seen["argv"] = argv
        return 0, "restarted", ""

    r = ops.restart_titan(_ctx(tmp_path, run=run), force=False)
    assert r["ok"] and r["dreaming_aware"] is True
    assert seen["argv"][-1] == "restart" and "--force" not in seen["argv"]

    ops.restart_titan(_ctx(tmp_path, run=run), force=True)
    assert "--force" in seen["argv"]


def test_restart_missing_script(tmp_path):
    r = ops.restart_titan(_ctx(tmp_path))
    assert r["ok"] is False and "not found" in r["error"]


def test_clean_hdd_dry_run_then_confirm(tmp_path):
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    junk = tmpdir / "titan_resurrection.tar.gz"
    junk.write_bytes(b"x" * 1000)
    log = tmp_path / "agent.log"
    log.write_bytes(b"y" * 500)

    ctx = _ctx(tmp_path)
    dry = ops.clean_hdd(ctx, confirm=False, tmp_dir=str(tmpdir))
    assert dry["reclaimable_bytes"] == 1500
    assert all(not t["removed"] for t in dry["targets"])
    assert junk.exists() and log.exists()

    done = ops.clean_hdd(ctx, confirm=True, tmp_dir=str(tmpdir))
    assert done["removed_bytes"] == 1500
    assert not junk.exists() and not log.exists()


def test_list_backups_reads_records_and_manifest(tmp_path):
    data = tmp_path / "data"
    (data / "backup_records").mkdir(parents=True)
    (data / "backup_records" / "personality_123.json").write_text(
        json.dumps({"backup_type": "personality", "timestamp": 123,
                    "arweave_tx": "ar_x", "size_bytes": 99}))
    (data / "backup_unified_manifest_T1.json").write_text(json.dumps({
        "titan_id": "T1", "current_baseline_event_id": "b1",
        "events": [{"event_id": "b1", "type": "baseline"},
                   {"event_id": "e2", "type": "incremental"}]}))
    out = ops.list_backups(_ctx(tmp_path))
    assert out["records"][0]["arweave_tx"] == "ar_x"
    assert out["manifest"]["events"] == 2 and out["manifest"]["latest_type"] == "incremental"


# ── proxy ──────────────────────────────────────────────────────────────────


def test_proxy_allow_list():
    assert proxy.is_allowed("/v6/trinity")
    assert proxy.is_allowed("/v6/cognition/reasoning")
    assert not proxy.is_allowed("/v6/admin/heap-dump")
    assert not proxy.is_allowed("/etc/passwd")


def test_proxy_readout_titan_down(tmp_path):
    ctx = _ctx(tmp_path, http=lambda *a, **k: (0, b""))
    status, payload = proxy.proxy_readout(ctx, "/v6/trinity")
    assert status == 503 and payload["titan_down"] is True


def test_proxy_readout_ok(tmp_path):
    ctx = _ctx(tmp_path, http=lambda *a, **k: (200, json.dumps({"mood": "calm"}).encode()))
    status, payload = proxy.proxy_readout(ctx, "/v6/trinity")
    assert status == 200 and payload["mood"] == "calm"


# ── dispatch (the router) ────────────────────────────────────────────────


def test_dispatch_health(tmp_path):
    status, p = dispatch(_ctx(tmp_path), "GET", "/console/health", {}, b"", {})
    assert status == 200 and p["ok"] is True and p["agent"] == "titan-console"


def test_dispatch_host(tmp_path):
    status, p = dispatch(_ctx(tmp_path), "GET", "/console/host", {}, b"", {})
    assert status == 200 and "memory" in p


def test_dispatch_proxy_api(tmp_path):
    ctx = _ctx(tmp_path, http=lambda *a, **k: (200, json.dumps({"ok": 1}).encode()))
    status, p = dispatch(ctx, "GET", "/console/api/v6/trinity", {}, b"", {})
    assert status == 200 and p["ok"] == 1


def test_dispatch_mutation_token_gate(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "t1_manage.sh").write_text("#!/bin/bash\n")
    ctx = _ctx(tmp_path, run=lambda argv, **k: (0, "ok", ""), token="secret")
    # no token header → 401
    status, p = dispatch(ctx, "POST", "/console/restart", {}, b"{}", {})
    assert status == 401
    # correct token → 200
    status, p = dispatch(ctx, "POST", "/console/restart", {}, b"{}",
                         {"x-console-token": "secret"})
    assert status == 200 and p["ok"] is True


def test_dispatch_chat_proxy(tmp_path):
    ctx = _ctx(tmp_path, http=lambda *a, **k: (200, json.dumps({"reply": "hi"}).encode()))
    status, p = dispatch(ctx, "POST", "/console/chat", {}, json.dumps({"message": "yo"}).encode(), {})
    assert status == 200 and p["reply"] == "hi"


def test_dispatch_unknown_console_route_404(tmp_path):
    status, p = dispatch(_ctx(tmp_path), "GET", "/console/nope", {}, b"", {})
    assert status == 404


def test_dispatch_static_fallback(tmp_path):
    status, p = dispatch(_ctx(tmp_path), "GET", "/", {}, b"", {})
    assert status == 200 and isinstance(p, (bytes, bytearray))


# ── config_api (uses the real repo's config + tmp for writes) ──────────────


def test_config_list_real_repo():
    out = config_api.list_config(_REPO)
    assert out["entries"], "expected config entries from titan_hcl/config.toml"
    assert any(e["help"] for e in out["entries"]), "expected inline-comment help"


def test_config_set_on_tmp(tmp_path):
    cfg = tmp_path / "titan_hcl"
    cfg.mkdir()
    (cfg / "config.toml").write_text(
        "[network]\n# the RPC the Titan talks to\napi_port = 7777  # listen port\n")
    got = config_api.get_config(tmp_path, "network.api_port")
    assert got["found"] and got["value"] == "7777"
    res = config_api.set_config(tmp_path, "network.api_port", "8888")
    assert res["ok"] is True
    assert config_api.get_config(tmp_path, "network.api_port")["value"] == "8888"


def test_resolve_titan_id(tmp_path):
    assert resolve_titan_id(tmp_path) == "T1"  # fallback
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "titan_identity.json").write_text(json.dumps({"titan_id": "T2"}))
    assert resolve_titan_id(tmp_path) == "T2"
