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

from titan_console import alerts, backup_config, config_api, host, ops, proxy  # noqa: E402
from titan_console.agent import dispatch  # noqa: E402
from titan_console.context import Context, resolve_titan_id  # noqa: E402
from titan_console.titan_status import titan_status  # noqa: E402


# ── fakes ────────────────────────────────────────────────────────────────


def _ctx(tmp_path, *, run=None, http=None, token=None, titan_id="T1",
         secrets_path=None, internal_key=None) -> Context:
    return Context(
        install_root=tmp_path, titan_id=titan_id,
        run=run or (lambda argv, **k: (0, "", "")),
        http=http or (lambda *a, **k: (0, b"")),
        token=token, secrets_path=secrets_path, internal_key=internal_key,
    )


class _FakeCron:
    """Captures the crontab a `crontab <file>` install would apply."""

    def __init__(self, existing=""):
        self.existing = existing
        self.installed = None

    def run(self, argv, **k):
        if argv[:2] == ["crontab", "-l"]:
            return (0 if self.existing else 1), self.existing, ""
        if argv[0] == "crontab" and len(argv) == 2:
            self.installed = Path(argv[1]).read_text()  # read before unlink
            return 0, "", ""
        return 0, "", ""


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


def test_restart_uses_systemctl_on_resolved_unit(tmp_path):
    seen = {}

    def run(argv, **k):
        seen["argv"] = argv
        return 0, "restarted", ""

    # http returns awake (not dreaming) → restart proceeds
    def http(method, url, **k):
        return 200, json.dumps({"data": {"is_dreaming": False}}).encode()

    r = ops.restart_titan(_ctx(tmp_path, run=run, http=http), force=False)
    assert r["ok"] and r["dreaming_aware"] is True
    # systemctl restart on the resolved unit (no fleet manage script)
    assert seen["argv"][:3] == ["sudo", "systemctl", "restart"]
    assert seen["argv"][3] == r["service"]


def test_restart_refused_while_dreaming(tmp_path):
    ran = {"called": False}

    def run(argv, **k):
        ran["called"] = True
        return 0, "", ""

    def http(method, url, **k):
        return 200, json.dumps({"data": {"is_dreaming": True}}).encode()

    r = ops.restart_titan(_ctx(tmp_path, run=run, http=http), force=False)
    assert r["ok"] is False and r["dreaming"] is True and not ran["called"]
    # force overrides the dreaming guard
    r2 = ops.restart_titan(_ctx(tmp_path, run=run, http=http), force=True)
    assert ran["called"] and r2["dreaming_aware"] is False


def test_restart_no_internal_key_still_restarts(tmp_path):
    # restart does not need the chat internal_key; dreaming probe just no-ops on
    # an unreadable state and the restart proceeds.
    seen = {}

    def run(argv, **k):
        seen["argv"] = argv
        return 0, "", ""

    r = ops.restart_titan(_ctx(tmp_path, run=run), force=True)
    assert r["ok"] and seen["argv"][:3] == ["sudo", "systemctl", "restart"]


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
    seen = {}

    def http(method, url, *, body=None, headers=None, timeout=None):
        seen["url"] = url
        seen["payload"] = json.loads(body.decode())
        seen["headers"] = headers or {}
        return 200, json.dumps({"response": "hi", "thread_id": "x"}).encode()

    ctx = _ctx(tmp_path, http=http, internal_key="OWNERKEY")
    status, p = dispatch(ctx, "POST", "/console/chat", {},
                         json.dumps({"message": "yo"}).encode(), {})
    assert status == 200 and p["response"] == "hi"
    # owner auth header + PitchChatRequest-shaped payload to /v6/pitch/chat
    assert seen["url"].endswith("/v6/pitch/chat")
    assert seen["headers"]["X-Titan-Internal-Key"] == "OWNERKEY"
    assert seen["payload"]["titan"] == "T1"
    assert seen["payload"]["message"] == "yo"
    assert len(seen["payload"]["thread_id"]) >= 8


def test_dispatch_chat_no_internal_key_503(tmp_path):
    # no owner key configured → clear 503, not a confusing auth error downstream
    ctx = _ctx(tmp_path)
    status, p = dispatch(ctx, "POST", "/console/chat", {},
                         json.dumps({"message": "yo"}).encode(), {})
    assert status == 503 and "internal_key" in p["error"]


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


# ── backup_config (off-site convenience copy — decision #15) ────────────────


def test_backup_config_set_local_writes_secrets_and_cron(tmp_path):
    secrets = tmp_path / "secrets.toml"
    cron = _FakeCron()
    ctx = _ctx(tmp_path, run=cron.run, secrets_path=secrets)

    res = backup_config.set_backup_config(ctx, {
        "enabled": True, "backend": "local",
        "local_dir": "/mnt/backups/titan", "schedule_cron": "0 5 * * *"})
    assert res["ok"] and res["backend"] == "local"

    # secrets.toml got a [backup_offsite] block with the values
    text = secrets.read_text()
    assert "[backup_offsite]" in text
    assert 'local_dir = "/mnt/backups/titan"' in text
    assert "enabled = true" in text
    # cron line installed, points at the executor + carries the marker
    assert cron.installed and "offsite_backup.py" in cron.installed
    assert "TITAN_OFFSITE_BACKUP" in cron.installed
    assert cron.installed.startswith("0 5 * * *")


def test_backup_config_get_redacts_secret(tmp_path):
    secrets = tmp_path / "secrets.toml"
    secrets.write_text(
        '[backup_offsite]\nenabled = true\nbackend = "s3"\n'
        's3_bucket = "my-bkt"\naws_secret_access_key = "SHHH"\n')
    out = backup_config.get_backup_config(_ctx(tmp_path, secrets_path=secrets))
    assert out["configured"] and out["enabled"]
    assert out["offsite"]["aws_secret_access_key"] == "***set***"
    assert out["offsite"]["s3_bucket"] == "my-bkt"  # non-secret kept


def test_backup_config_validation_errors(tmp_path):
    secrets = tmp_path / "secrets.toml"
    ctx = _ctx(tmp_path, run=_FakeCron().run, secrets_path=secrets)
    assert backup_config.set_backup_config(ctx, {"bogus": 1})["ok"] is False
    r = backup_config.set_backup_config(ctx, {"enabled": True, "backend": "s3"})
    assert r["ok"] is False and "s3_bucket" in r["error"]
    r = backup_config.set_backup_config(ctx, {"enabled": True, "backend": "local"})
    assert r["ok"] is False and "local_dir" in r["error"]


def test_backup_config_disable_removes_cron(tmp_path):
    secrets = tmp_path / "secrets.toml"
    existing = "0 5 * * * cd /x && python scripts/offsite_backup.py # TITAN_OFFSITE_BACKUP\n"
    cron = _FakeCron(existing=existing + "# keep me\n")
    ctx = _ctx(tmp_path, run=cron.run, secrets_path=secrets)
    res = backup_config.set_backup_config(ctx, {"enabled": False})
    assert res["ok"] and res["enabled"] is False
    assert "TITAN_OFFSITE_BACKUP" not in cron.installed
    assert "# keep me" in cron.installed  # other crontab lines preserved


def test_dispatch_backup_config_get_and_set(tmp_path):
    secrets = tmp_path / "secrets.toml"
    cron = _FakeCron()
    ctx = _ctx(tmp_path, run=cron.run, secrets_path=secrets)
    status, p = dispatch(ctx, "GET", "/console/backup/config", {}, b"", {})
    assert status == 200 and p["configured"] is False
    body = json.dumps({"enabled": True, "backend": "local",
                       "local_dir": "/mnt/x"}).encode()
    status, p = dispatch(ctx, "POST", "/console/backup/config", {}, body, {})
    assert status == 200 and p["ok"] is True


# ── offsite_backup executor (real local sync) ───────────────────────────────


def _load_offsite():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "offsite_backup", _REPO / "scripts" / "offsite_backup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_offsite_disabled_returns_2(tmp_path):
    ob = _load_offsite()
    secrets = tmp_path / "secrets.toml"  # absent
    rc, msg = ob.run(tmp_path, secrets_path=secrets)
    assert rc == 2 and "disabled" in msg


def test_offsite_local_sync_copies_snapshots(tmp_path):
    ob = _load_offsite()
    src = tmp_path / "data" / "backups"
    src.mkdir(parents=True)
    (src / "snap_001.tar.gz").write_bytes(b"x" * 64)
    dest = tmp_path / "offsite"
    secrets = tmp_path / "secrets.toml"
    secrets.write_text(
        f'[backup_offsite]\nenabled = true\nbackend = "local"\n'
        f'local_dir = "{dest}"\n')
    rc, msg = ob.run(tmp_path, secrets_path=secrets)
    assert rc == 0, msg
    assert (dest / "snap_001.tar.gz").exists()


def test_offsite_misconfigured_backend(tmp_path):
    ob = _load_offsite()
    (tmp_path / "data" / "backups").mkdir(parents=True)
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('[backup_offsite]\nenabled = true\nbackend = "ftp"\n')
    rc, msg = ob.run(tmp_path, secrets_path=secrets)
    assert rc == 3 and "unknown backend" in msg


# ── alerts (decoupled degraded-health push) ─────────────────────────────────


def _alert_ctx(tmp_path, *, up=True, secrets=None):
    """Context whose run/http simulate a healthy or failed Titan."""
    def run(argv, **k):
        if "show" in argv:
            state = "active\nSubState=running" if up else "failed\nSubState=dead"
            return 0, f"ActiveState={state}\n", ""
        if argv and argv[0] == "journalctl":
            return 0, "boot ok\nOOM-killed worker\n", ""
        return 0, "", ""

    def http(method, url, **k):
        return (200, json.dumps({"status": "healthy"}).encode()) if up else (0, b"")

    return _ctx(tmp_path, run=run, http=http, secrets_path=secrets)


def test_resolve_telegram_creds_precedence(tmp_path):
    (tmp_path / "titan_hcl").mkdir()
    (tmp_path / "titan_hcl" / "config.toml").write_text(
        '[channels]\ntelegram_bot_token = "CFG_TOK"\n'
        '[maker_relationship]\nmaker_telegram_id = "111"\n')
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('[console]\nalert_chat_id = "999"\n')  # console override wins for chat
    tok, chat = alerts.resolve_telegram_creds(_ctx(tmp_path, secrets_path=secrets))
    assert tok == "CFG_TOK"        # from config.toml [channels]
    assert chat == "999"           # [console] override beats maker_relationship


def test_send_telegram_ok_and_fail():
    class _Resp:
        def __init__(self, body): self._b = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._b

    sent = alerts.send_telegram("t", "c", "hi",
                                opener=lambda r, timeout: _Resp(b'{"ok":true}'))
    assert sent is True
    bad = alerts.send_telegram("t", "c", "hi",
                               opener=lambda r, timeout: _Resp(b'{"ok":false}'))
    assert bad is False
    assert alerts.send_telegram("", "c", "hi") is False  # missing token


def test_health_monitor_edge_triggers_on_down_then_recovery(tmp_path):
    (tmp_path / "titan_hcl").mkdir()
    (tmp_path / "titan_hcl" / "config.toml").write_text(
        '[channels]\ntelegram_bot_token = "TOK"\n'
        '[maker_relationship]\nmaker_telegram_id = "42"\n')
    state = {"up": True}
    pushes = []

    def run(argv, **k):
        if "show" in argv:
            s = "active\nSubState=running" if state["up"] else "failed\nSubState=dead"
            return 0, f"ActiveState={s}\n", ""
        if argv and argv[0] == "journalctl":
            return 0, "line\nOOM-killed\n", ""
        return 0, "", ""

    def http(method, url, **k):
        return (200, b'{"status":"healthy"}') if state["up"] else (0, b"")

    ctx = _ctx(tmp_path, run=run, http=http)
    mon = alerts.HealthMonitor(ctx, sender=lambda t, c, txt: pushes.append(txt) or True)
    mon._last_up = True  # prime baseline (start() would do this)

    assert mon.check_once()["transition"] is None  # steady up → no alert
    state["up"] = False
    r = mon.check_once()
    assert r["transition"] == "down" and r["alert_sent"] is True
    assert "DOWN" in pushes[-1] and "OOM" in pushes[-1]
    state["up"] = True
    r = mon.check_once()
    assert r["transition"] == "up" and "recovered" in pushes[-1]


def test_health_monitor_no_creds_no_crash(tmp_path):
    ctx = _alert_ctx(tmp_path, up=True)  # no config.toml → no creds
    mon = alerts.HealthMonitor(ctx)
    mon._last_up = True
    # flip down: should report transition but creds False, no exception
    ctx2 = _alert_ctx(tmp_path, up=False)
    mon.ctx = ctx2
    r = mon.check_once()
    assert r["transition"] == "down" and r["creds"] is False
