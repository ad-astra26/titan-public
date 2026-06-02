"""Owner ops: restart (dreaming-aware), safe HDD cleanup, backup listing.

Destructive actions are conservative: clean_hdd defaults to a dry run and only
ever targets clearly-disposable artifacts (temp archives, rotated logs,
pre-restore snapshots) — NEVER live sovereign data/ files
(directive_memory_preservation).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import time
from pathlib import Path

from .context import Context


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
