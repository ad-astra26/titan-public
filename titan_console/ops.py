"""Owner ops: restart (dreaming-aware), safe HDD cleanup, backup listing.

Destructive actions are conservative: clean_hdd defaults to a dry run and only
ever targets clearly-disposable artifacts (temp archives, rotated logs,
pre-restore snapshots) — NEVER live sovereign data/ files
(directive_memory_preservation).
"""
from __future__ import annotations

import glob
import hmac
import json
import os
import shutil
import time
from pathlib import Path

from .context import Context

# Console process start (best-effort uptime for agent_status).
_STARTED_AT = time.time()

# ── RFP_titan_mobile_app §7.2b — advanced ops constants ──────────────────────
# (a) Reboot is gated by a fixed, server-known confirm phrase (deliberate-intent
# gate, not a CSRF token) AND a primary-device check in the route.
_REBOOT_PHRASE = "REBOOT"

# (b) Zombie/stale reap is ALLOWLIST + fail-closed. The hard-exclude set is checked
# FIRST and overrides everything — a process whose comm/cmdline contains any of these
# is NEVER reapable (never self-immolate the Titan, kernel, console, or init system).
_HARD_EXCLUDE_SUBSTR = ("titan", "kernel", "console", "systemd", "init", "sshd",
                        "dbus", "journald", "logind", "agetty")
# A process is reapable ONLY if its comm matches this explicit allowlist of headless
# automation helpers (the realistic orphans of the research lane) AND it is orphaned
# (reparented to init) AND it is not a zombie (a Z is already dead — killing its PID
# is a no-op; only its parent/init reaps it, so we report but never "reap" it).
_REAP_COMM_ALLOWLIST = ("chrome", "chromium", "chromedriver", "headless_shell",
                        "geckodriver", "playwright", "ffmpeg")


def restart_titan(ctx: Context, *, force: bool = False) -> dict:
    """Restart the Titan via systemd — dreaming-aware (never pkill).

    The fleet's `tN_manage.sh restart` is NOT shipped to a user install, so we
    reproduce its core guard inline: refuse to wake a dreaming Titan unless
    ``force`` (mid-dream restart loses the in-flight consolidation), then
    ``systemctl restart`` the RESOLVED unit (titan.service or titan-<id>.service).
    """
    if not force:
        status, body = ctx.http("GET", f"{ctx.api_base}/v6/dreaming", timeout=5.0)
        if status == 200:
            try:
                if json.loads(body.decode()).get("data", {}).get("is_dreaming"):
                    return {"ok": False, "dreaming": True,
                            "error": "Titan is dreaming — restart refused. "
                                     "Retry with force to override."}
            except (ValueError, AttributeError, TypeError):
                pass  # unreadable dreaming state → don't block the restart
    unit = ctx.service_unit
    rc, out, err = ctx.run(["sudo", "systemctl", "restart", unit])
    return {"ok": rc == 0, "returncode": rc, "stdout": out, "stderr": err,
            "service": unit, "dreaming_aware": not force}


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def _path_size(p: Path) -> int:
    try:
        if p.is_dir():
            return _dir_size(p)
        return p.stat().st_size
    except OSError:
        return 0


def _cleanup_targets(install_root: Path, tmp_dir: str = "/tmp") -> list[Path]:
    """Clearly-disposable artifacts only. Globs resolved at call time."""
    t = tmp_dir.rstrip("/")
    patterns = [
        f"{t}/titan_*.tar.gz", f"{t}/titan_*.tar", f"{t}/titan_resurrection*",
        f"{t}/titan_personality_*", f"{t}/titan_soul_*",
        str(install_root / "*.log"),
        str(install_root / "data.pre_restore*"),
        str(install_root / "data" / "*.OLD.*"),
        str(install_root / "data" / "*.bak.prev"),
    ]
    seen: list[Path] = []
    for pat in patterns:
        for hit in glob.glob(pat):
            p = Path(hit)
            if p not in seen:
                seen.append(p)
    return seen


def clean_hdd(ctx: Context, *, confirm: bool = False, tmp_dir: str = "/tmp") -> dict:
    """List (and, with confirm=True, delete) disposable artifacts.

    Returns per-target reclaimable bytes. Live data/ files are never targeted.
    """
    targets = _cleanup_targets(ctx.install_root, tmp_dir)
    items = []
    total = 0
    for p in targets:
        size = _path_size(p)
        total += size
        item = {"path": str(p), "bytes": size, "is_dir": p.is_dir(), "removed": False}
        if confirm:
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                item["removed"] = True
            except OSError as e:
                item["error"] = str(e)
        items.append(item)
    return {"confirm": confirm, "reclaimable_bytes": total,
            "removed_bytes": sum(i["bytes"] for i in items if i.get("removed")),
            "targets": items}


def list_backups(ctx: Context) -> dict:
    """Summarize local backup records + the unified manifest chain."""
    data = ctx.install_root / "data"
    out = {"records": [], "manifest": None}

    rec_dir = data / "backup_records"
    if rec_dir.is_dir():
        for f in sorted(rec_dir.glob("*.json"), reverse=True)[:50]:
            try:
                rec = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            out["records"].append({
                "file": f.name,
                "type": rec.get("backup_type") or rec.get("type"),
                "ts": rec.get("timestamp") or rec.get("ts"),
                "arweave_tx": rec.get("arweave_tx") or rec.get("tx_id"),
                "size_bytes": rec.get("size_bytes"),
            })

    manifest = data / f"backup_unified_manifest_{ctx.titan_id}.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text())
            evs = m.get("events", [])
            out["manifest"] = {
                "events": len(evs),
                "current_baseline": m.get("current_baseline_event_id"),
                "latest_event": evs[-1].get("event_id") if evs else None,
                "latest_type": evs[-1].get("type") if evs else None,
            }
        except (OSError, ValueError):
            pass
    return out


# ── RFP_titan_mobile_app §7.2b — advanced layered ops ────────────────────────
def reboot(ctx: Context, *, confirm_phrase: str) -> dict:
    """Reboot the whole VPS (host) — the single most destructive op.

    Gate (§7.2b decision-a): the route enforces device-auth + a *primary*-device
    check; THIS function enforces the typed confirm phrase. Both must hold. The
    phrase is a fixed server-known constant compared in constant time.
    """
    if not hmac.compare_digest(str(confirm_phrase or ""), _REBOOT_PHRASE):
        return {"ok": False, "error": "reboot confirm phrase mismatch"}
    rc, out, err = ctx.run(["sudo", "systemctl", "reboot"])
    return {"ok": rc == 0, "returncode": rc, "stdout": out, "stderr": err,
            "rebooting": rc == 0}


def _read_proc_entry(proc_root: str, pid: str) -> dict | None:
    """Parse /proc/<pid>/{stat,cmdline,statm} → {pid,ppid,state,comm,cmdline,rss_kb}.

    Robust to a comm containing spaces/parens (parse after the LAST ')'). Returns
    None if the process vanished mid-scan (races are expected)."""
    base = os.path.join(proc_root, pid)
    try:
        with open(os.path.join(base, "stat")) as f:
            raw = f.read()
    except (OSError, ValueError):
        return None
    rparen = raw.rfind(")")
    lparen = raw.find("(")
    if lparen < 0 or rparen < 0 or rparen < lparen:
        return None
    comm = raw[lparen + 1:rparen]
    rest = raw[rparen + 2:].split()
    if len(rest) < 2:
        return None
    state = rest[0]
    try:
        ppid = int(rest[1])
    except ValueError:
        ppid = -1
    cmdline = ""
    try:
        with open(os.path.join(base, "cmdline"), "rb") as f:
            cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except OSError:
        pass
    rss_kb = 0
    try:
        with open(os.path.join(base, "statm")) as f:
            resident_pages = int(f.read().split()[1])
            rss_kb = resident_pages * (os.sysconf("SC_PAGE_SIZE") // 1024)
    except (OSError, ValueError, IndexError):
        pass
    return {"pid": int(pid), "ppid": ppid, "state": state, "comm": comm,
            "cmdline": cmdline, "rss_kb": rss_kb}


def _is_hard_excluded(entry: dict) -> bool:
    hay = f"{entry.get('comm', '')} {entry.get('cmdline', '')}".lower()
    return any(sub in hay for sub in _HARD_EXCLUDE_SUBSTR)


def _classify(entry: dict, *, self_tree: set) -> dict:
    """Tag a /proc entry with {classification, reapable} per the §7.2b allowlist."""
    if entry["pid"] in self_tree or _is_hard_excluded(entry):
        return {**entry, "classification": "protected", "reapable": False}
    if entry["state"] == "Z":
        # Defunct: already dead. Killing its PID is a no-op — only the parent (or
        # init, once reparented) reaps it. Report it as a signal, never "reap" it.
        return {**entry, "classification": "zombie", "reapable": False,
                "note": "defunct — reaped by its parent/init, not killable"}
    comm = entry["comm"].lower()
    matches_helper = any(comm == c or comm.startswith(c) for c in _REAP_COMM_ALLOWLIST)
    if matches_helper and entry["ppid"] == 1:
        return {**entry, "classification": "orphan_helper", "reapable": True}
    return {**entry, "classification": "other", "reapable": False}


def _self_tree(proc_root: str) -> set:
    """The console's own pid + its ancestor chain — never reapable (defence in depth
    on top of the 'console'/'systemd' hard-exclude substrings)."""
    tree, pid = set(), os.getpid()
    for _ in range(64):  # bounded walk; guards a malformed/cyclic ppid chain
        if pid <= 0 or pid in tree:
            break
        tree.add(pid)
        entry = _read_proc_entry(proc_root, str(pid))
        if not entry:
            break
        pid = entry["ppid"]
    return tree


def scan_processes(ctx: Context, *, proc_root: str = "/proc") -> dict:
    """ALWAYS a dry run — classify the live process table, never kill (§7.2b decision-b).

    Returns every process tagged {protected, zombie, orphan_helper, other} + a
    `reapable` flag. Only `orphan_helper` entries are reapable; the app shows the
    reapable set and the operator confirms specific PIDs to `reap_processes`."""
    self_tree = _self_tree(proc_root)
    procs = []
    try:
        pids = [d for d in os.listdir(proc_root) if d.isdigit()]
    except OSError as e:
        return {"error": f"cannot read {proc_root}: {e}", "processes": [], "reapable": []}
    for pid in pids:
        entry = _read_proc_entry(proc_root, pid)
        if entry is None:
            continue
        procs.append(_classify(entry, self_tree=self_tree))
    procs.sort(key=lambda p: p["pid"])
    reapable = [p["pid"] for p in procs if p["reapable"]]
    return {"dry_run": True, "count": len(procs), "reapable": reapable,
            "zombies": [p["pid"] for p in procs if p["classification"] == "zombie"],
            "processes": procs}


def reap_processes(ctx: Context, *, pids: list, proc_root: str = "/proc") -> dict:
    """Kill ONLY the given PIDs, and ONLY if a FRESH scan still classifies each as a
    reapable `orphan_helper` (§7.2b decision-b: allowlist + fail-closed). Every PID is
    re-checked against the hard-exclude + allowlist at kill time — a stale/forged PID
    from the app can never bypass the gate."""
    if not isinstance(pids, list) or not pids:
        return {"error": "pids must be a non-empty list", "results": []}
    self_tree = _self_tree(proc_root)
    results = []
    killed = 0
    for raw_pid in pids:
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            results.append({"pid": raw_pid, "killed": False, "skipped": "not an integer"})
            continue
        entry = _read_proc_entry(proc_root, str(pid))
        if entry is None:
            results.append({"pid": pid, "killed": False, "skipped": "gone / not found"})
            continue
        verdict = _classify(entry, self_tree=self_tree)
        if not verdict["reapable"]:
            results.append({"pid": pid, "killed": False,
                            "skipped": f"not reapable ({verdict['classification']})",
                            "comm": entry["comm"]})
            continue
        rc, out, err = ctx.run(["kill", str(pid)])  # SIGTERM, via the injected runner
        ok = rc == 0
        killed += int(ok)
        results.append({"pid": pid, "killed": ok, "comm": entry["comm"],
                        "returncode": rc, **({"stderr": err} if err else {})})
    return {"requested": len(pids), "killed": killed, "results": results}


def prune_arweave_devnet(ctx: Context, *, keep: int = 5, confirm: bool = False) -> dict:
    """Keep-newest-N of data/arweave_devnet/ (the devnet file:// Arweave cache, ~1.2G
    per backup, never pruned by the L5 retention cascade). Dry-run unless confirm=True.

    Never touches data/backups/ tarballs or any sovereign data — only this cache dir."""
    keep = max(0, int(keep))
    cache = ctx.install_root / "data" / "arweave_devnet"
    if not cache.is_dir():
        return {"confirm": confirm, "dir": str(cache), "exists": False,
                "kept": 0, "candidates": [], "reclaimable_bytes": 0}
    entries = [p for p in cache.iterdir()]
    entries.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    keep_set = entries[:keep]
    prune = entries[keep:]
    items, total = [], 0
    for p in prune:
        size = _path_size(p)
        total += size
        item = {"path": str(p), "bytes": size, "is_dir": p.is_dir(), "removed": False}
        if confirm:
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                item["removed"] = True
            except OSError as e:
                item["error"] = str(e)
        items.append(item)
    return {"confirm": confirm, "dir": str(cache), "exists": True, "keep": keep,
            "kept": len(keep_set), "reclaimable_bytes": total,
            "removed_bytes": sum(i["bytes"] for i in items if i.get("removed")),
            "candidates": items}


def agent_status(ctx: Context) -> dict:
    """Console self-status — uptime/version/bind/Titan-reachable. Used by the app to
    poll for the console coming back after a VPS reboot (§7.2b worked example)."""
    from . import __version__
    status, _ = ctx.http("GET", f"{ctx.api_base}/health", timeout=3.0)
    return {"ok": True, "agent": "titan-console", "version": __version__,
            "titan_id": ctx.titan_id, "uptime_seconds": round(time.time() - _STARTED_AT, 1),
            "bind_port": ctx.console_port, "titan_reachable": status == 200}
