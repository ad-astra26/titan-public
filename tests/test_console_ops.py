"""Tests for the advanced layered ops console — RFP_titan_mobile_app §7.2b.

Covers the kernel-admin proxy ops (L2 module reload/restart/enable, L3 reload-api),
the host VPS reboot (primary-device + typed-phrase gate, decision-a), the zombie/stale
reap (allowlist + fail-closed, dry-run scan vs confirmed kill, decision-b), the
arweave_devnet prune (keep-newest-N), console self-status, and the dispatch auth gates.
"""
import base64
import json
import os
import time

import pytest

from titan_console import _ed25519, ops, pairing
from titan_console.agent import dispatch
from titan_console.context import Context

_INTERNAL_KEY = "TEST-INTERNAL-KEY"


class _FakeKernel:
    """Stands in for api_hcl:7777 — records calls; answers nervous-system + admin POSTs."""

    def __init__(self, modules=("cognitive_worker", "synthesis"),
                 admin_status=200, admin_body=b'{"ok": true}', health=200):
        self.calls = []
        self.modules = list(modules)
        self.admin_status = admin_status
        self.admin_body = admin_body
        self.health = health

    def __call__(self, method, url, *, body=None, headers=None, timeout=5.0):
        self.calls.append({"method": method, "url": url, "body": body,
                           "headers": headers or {}})
        if url.endswith("/v6/nervous-system"):
            payload = {"modules": [{"name": m} for m in self.modules]}
            return 200, json.dumps(payload).encode()
        if url.endswith("/health"):
            return self.health, b'{"ok": true}'
        return self.admin_status, self.admin_body


class _FakeRunner:
    """Records argv; never touches the real shell. Returns (0, "ok", "")."""

    def __init__(self, rc=0):
        self.calls = []
        self.rc = rc

    def __call__(self, argv, *, timeout=30.0):
        self.calls.append(list(argv))
        return self.rc, "ok", ""


def _b64(b):
    return base64.b64encode(b).decode()


def _ctx(tmp_path, *, http=None, run=None, token=None):
    (tmp_path / "secrets.toml").write_text(f'[api]\ninternal_key = "{_INTERNAL_KEY}"\n')
    return Context(install_root=tmp_path, titan_id="T1",
                   secrets_path=tmp_path / "secrets.toml",
                   internal_key=_INTERNAL_KEY, token=token,
                   http=http or (lambda *a, **k: (0, b"")),
                   run=run or _FakeRunner())


@pytest.fixture(autouse=True)
def _clear():
    pairing._clear_caches()
    yield
    pairing._clear_caches()


def _register_device(ctx, device_id="dev-1", t0=1000.0):
    _, payload = pairing.mint_pairing(ctx, now=t0)
    token = payload["pairing_token"]
    seed, pub = _ed25519.keygen()
    pairing.submit_device(ctx, {"pairing_token": token, "device_pubkey": _b64(pub),
                                "device_id": device_id, "label": "phone"}, now=t0 + 1)
    code = pairing.code6(base64.b64decode(token), pub)
    pairing.confirm_device(ctx, token, code, now=t0 + 2)
    return seed


def _signed(seed, device_id, method, path, body=b""):
    ts = str(int(time.time()))
    msg = pairing.canonical_request(method, path, ts,
                                    pairing.body_sha256_hex(body)).encode()
    return {"x-device-id": device_id, "x-timestamp": ts,
            "x-signature": _b64(_ed25519.sign(msg, seed))}


def _mk_proc(tmp_path, entries):
    """Build a fake /proc. entries: (pid, comm, state, ppid, cmdline)."""
    root = tmp_path / "proc"
    root.mkdir()
    for pid, comm, state, ppid, cmd in entries:
        d = root / str(pid)
        d.mkdir()
        (d / "stat").write_text(f"{pid} ({comm}) {state} {ppid} 0 0 0 0 0 0 0 0 0 0 0 0")
        (d / "cmdline").write_bytes(cmd.encode())
        (d / "statm").write_text("1000 250 100 0 0 0 0")
    return str(root)


# ── (a) primary flag ─────────────────────────────────────────────────────────
def test_first_device_is_primary_second_is_not(tmp_path):
    ctx = _ctx(tmp_path)
    _register_device(ctx, "dev-1")
    _register_device(ctx, "dev-2", t0=2000.0)
    assert pairing.is_primary_device(ctx, "dev-1") is True
    assert pairing.is_primary_device(ctx, "dev-2") is False
    assert pairing.is_primary_device(ctx, "ghost") is False


def test_repair_preserves_primary(tmp_path):
    ctx = _ctx(tmp_path)
    _register_device(ctx, "dev-1")
    _register_device(ctx, "dev-1", t0=3000.0)  # re-pair same id
    assert pairing.is_primary_device(ctx, "dev-1") is True


# ── L2/L3 kernel-admin proxy ops ─────────────────────────────────────────────
def test_module_restart_proxies_with_spawn_and_internal_key(tmp_path):
    fake = _FakeKernel()
    ctx = _ctx(tmp_path, http=fake)
    seed = _register_device(ctx)
    path = "/console/ops/module/restart/cognitive_worker"
    status, payload = dispatch(ctx, "POST", path, {}, b"",
                               _signed(seed, "dev-1", "POST", path))
    assert status == 200
    admin = [c for c in fake.calls if "restart-module" in c["url"]][0]
    assert admin["url"].endswith("/v6/admin/restart-module/cognitive_worker?spawn=true")
    assert admin["headers"].get("X-Titan-Internal-Key") == _INTERNAL_KEY


def test_module_reload_and_enable_route_to_right_kernel_paths(tmp_path):
    fake = _FakeKernel()
    ctx = _ctx(tmp_path, http=fake)
    seed = _register_device(ctx)
    for action, frag in (("reload", "/v6/admin/reload-module/synthesis"),
                         ("enable", "/v6/system/guardian/enable/synthesis")):
        path = f"/console/ops/module/{action}/synthesis"
        status, _ = dispatch(ctx, "POST", path, {}, b"",
                             _signed(seed, "dev-1", "POST", path))
        assert status == 200
        assert any(c["url"].endswith(frag) for c in fake.calls), action


def test_unknown_module_rejected_before_proxy(tmp_path):
    fake = _FakeKernel(modules=("cognitive_worker",))
    ctx = _ctx(tmp_path, http=fake)
    seed = _register_device(ctx)
    path = "/console/ops/module/restart/no_such_worker"
    status, payload = dispatch(ctx, "POST", path, {}, b"",
                               _signed(seed, "dev-1", "POST", path))
    assert status == 404 and "roster" in payload["error"]
    assert not any("restart-module" in c["url"] for c in fake.calls)  # never proxied


def test_invalid_module_name_rejected(tmp_path):
    ctx = _ctx(tmp_path, http=_FakeKernel())
    seed = _register_device(ctx)
    path = "/console/ops/module/restart/bad.name"
    status, payload = dispatch(ctx, "POST", path, {}, b"",
                               _signed(seed, "dev-1", "POST", path))
    assert status == 400 and "invalid module name" in payload["error"]


def test_reload_api_proxies_v4(tmp_path):
    fake = _FakeKernel()
    ctx = _ctx(tmp_path, http=fake)
    seed = _register_device(ctx)
    path = "/console/ops/reload-api"
    status, _ = dispatch(ctx, "POST", path, {}, b"",
                         _signed(seed, "dev-1", "POST", path))
    assert status == 200
    assert any(c["url"].endswith("/v4/reload-api") for c in fake.calls)


def test_admin_proxy_503_without_internal_key(tmp_path):
    fake = _FakeKernel()
    ctx = _ctx(tmp_path, http=fake)
    ctx.internal_key = None
    seed = _register_device(ctx)
    path = "/console/ops/reload-api"
    status, payload = dispatch(ctx, "POST", path, {}, b"",
                               _signed(seed, "dev-1", "POST", path))
    assert status == 503 and "internal_key" in payload["error"]


# ── (a) reboot gate ──────────────────────────────────────────────────────────
def test_reboot_primary_device_correct_phrase(tmp_path):
    runner = _FakeRunner()
    ctx = _ctx(tmp_path, run=runner)
    seed = _register_device(ctx)  # dev-1 → primary
    body = json.dumps({"confirm_phrase": "REBOOT"}).encode()
    path = "/console/ops/reboot"
    status, payload = dispatch(ctx, "POST", path, {}, body,
                               _signed(seed, "dev-1", "POST", path, body))
    assert status == 200 and payload["ok"] is True and payload["rebooting"] is True
    assert ["sudo", "systemctl", "reboot"] in runner.calls


def test_reboot_wrong_phrase_does_not_run(tmp_path):
    runner = _FakeRunner()
    ctx = _ctx(tmp_path, run=runner)
    seed = _register_device(ctx)
    body = json.dumps({"confirm_phrase": "reboot"}).encode()  # wrong case
    path = "/console/ops/reboot"
    status, payload = dispatch(ctx, "POST", path, {}, body,
                               _signed(seed, "dev-1", "POST", path, body))
    assert status == 200 and payload["ok"] is False and "mismatch" in payload["error"]
    assert runner.calls == []  # never touched systemctl


def test_reboot_refused_for_non_primary_device(tmp_path):
    runner = _FakeRunner()
    ctx = _ctx(tmp_path, run=runner)
    _register_device(ctx, "dev-1")                      # primary
    seed2 = _register_device(ctx, "dev-2", t0=2000.0)   # NOT primary
    body = json.dumps({"confirm_phrase": "REBOOT"}).encode()
    path = "/console/ops/reboot"
    status, payload = dispatch(ctx, "POST", path, {}, body,
                               _signed(seed2, "dev-2", "POST", path, body))
    assert status == 403 and "primary" in payload["error"]
    assert runner.calls == []


# ── (b) reap: scan dry-run + allowlist + fail-closed ─────────────────────────
def test_scan_classifies_orphan_helper_zombie_protected(tmp_path):
    ctx = _ctx(tmp_path)
    proc = _mk_proc(tmp_path, [
        (90001, "chromium", "S", 1, "/usr/bin/chromium --headless"),   # orphan helper
        (90002, "chromium", "S", 4242, "/usr/bin/chromium"),           # parented → other
        (90003, "python3", "S", 1, "python -m titan_hcl cognitive"),   # titan → protected
        (90004, "chromium", "Z", 1, ""),                               # zombie
        (90005, "vim", "S", 1, "vim notes"),                           # not allow-listed
    ])
    res = ops.scan_processes(ctx, proc_root=proc)
    assert res["dry_run"] is True
    assert res["reapable"] == [90001]
    assert res["zombies"] == [90004]
    by_pid = {p["pid"]: p for p in res["processes"]}
    assert by_pid[90002]["classification"] == "other"
    assert by_pid[90003]["classification"] == "protected"
    assert by_pid[90005]["classification"] == "other"


def test_reap_kills_only_allowlisted_orphans(tmp_path):
    runner = _FakeRunner()
    ctx = _ctx(tmp_path, run=runner)
    proc = _mk_proc(tmp_path, [
        (90001, "chromium", "S", 1, "/usr/bin/chromium --headless"),   # reapable
        (90003, "python3", "S", 1, "python -m titan_hcl cognitive"),   # protected
        (90004, "chromium", "Z", 1, ""),                               # zombie
    ])
    res = ops.reap_processes(ctx, pids=[90001, 90003, 90004, 99999], proc_root=proc)
    assert res["killed"] == 1
    assert runner.calls == [["kill", "90001"]]            # ONLY the orphan helper
    verdict = {r["pid"]: r for r in res["results"]}
    assert verdict[90001]["killed"] is True
    assert "not reapable" in verdict[90003]["skipped"]
    assert "not reapable" in verdict[90004]["skipped"]
    assert "gone" in verdict[99999]["skipped"]


def test_reap_never_kills_self_or_pid1(tmp_path):
    # Defence-in-depth against the live /proc: our own pid + init are never reapable.
    runner = _FakeRunner()
    ctx = _ctx(tmp_path, run=runner)
    res = ops.reap_processes(ctx, pids=[os.getpid(), 1])
    assert res["killed"] == 0 and runner.calls == []


def test_reap_route_requires_pids_list(tmp_path):
    ctx = _ctx(tmp_path)
    seed = _register_device(ctx)
    path = "/console/ops/processes/reap"
    body = json.dumps({}).encode()
    status, payload = dispatch(ctx, "POST", path, {}, body,
                               _signed(seed, "dev-1", "POST", path, body))
    assert status == 400 and "pids" in payload["error"]


# ── arweave_devnet prune ─────────────────────────────────────────────────────
def test_prune_arweave_devnet_keep_newest(tmp_path):
    ctx = _ctx(tmp_path)
    cache = tmp_path / "data" / "arweave_devnet"
    cache.mkdir(parents=True)
    for i in range(7):
        f = cache / f"chunk_{i}.dat"
        f.write_bytes(b"x" * 100)
        os.utime(f, (1000 + i, 1000 + i))  # ascending mtime → chunk_6 newest
    dry = ops.prune_arweave_devnet(ctx, keep=5, confirm=False)
    assert dry["kept"] == 5 and len(dry["candidates"]) == 2
    assert dry["reclaimable_bytes"] == 200 and dry["removed_bytes"] == 0
    assert all(c["removed"] is False for c in dry["candidates"])
    assert {os.path.basename(c["path"]) for c in dry["candidates"]} == {"chunk_0.dat", "chunk_1.dat"}
    # confirm actually removes the 2 oldest, keeps 5
    done = ops.prune_arweave_devnet(ctx, keep=5, confirm=True)
    assert done["removed_bytes"] == 200
    assert sorted(p.name for p in cache.iterdir()) == \
        ["chunk_2.dat", "chunk_3.dat", "chunk_4.dat", "chunk_5.dat", "chunk_6.dat"]


def test_prune_missing_dir_is_safe(tmp_path):
    ctx = _ctx(tmp_path)
    res = ops.prune_arweave_devnet(ctx, keep=5)
    assert res["exists"] is False and res["kept"] == 0


# ── console self-status ──────────────────────────────────────────────────────
def test_agent_status_reports_reachable(tmp_path):
    ctx = _ctx(tmp_path, http=_FakeKernel(health=200))
    seed = _register_device(ctx)
    path = "/console/agent-status"
    status, payload = dispatch(ctx, "GET", path, {}, b"",
                               _signed(seed, "dev-1", "GET", path))
    assert status == 200 and payload["titan_reachable"] is True
    assert payload["agent"] == "titan-console" and "uptime_seconds" in payload


# ── auth gates ───────────────────────────────────────────────────────────────
def test_scan_route_requires_device_signature(tmp_path):
    ctx = _ctx(tmp_path)
    status, payload = dispatch(ctx, "GET", "/console/ops/processes", {}, b"", {})
    assert status == 401 and "signature" in payload["error"]


def test_ops_post_blocked_remotely_without_credentials(tmp_path):
    # AD-5: beyond localhost every route needs a device sig or a strict operator token.
    ctx = _ctx(tmp_path, http=_FakeKernel(), token="optok")
    status, payload = dispatch(ctx, "POST", "/console/ops/reload-api", {}, b"", {},
                               is_local=False)
    assert status == 401
