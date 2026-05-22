"""Off-host backup mirror (rFP_backup_worker §5.2 / §9 Phase 9).

T1 pulls T2/T3 local backup snapshots over SSH via rsync. This gives every
Titan a second-VPS copy of its state (T1 ships its own state to Arweave, T2/T3
rely on local_only + this mirror). When a T2/T3 VPS is lost, the latest
snapshot survives on T1's disk.

Operational model:
  - Runs ONLY on T1. T2/T3 should disable this section in their configs.
  - Invoked AFTER BackupWorker's personality cascade succeeds on T1 — that's
    our natural cadence (once per day, or on manual trigger).
  - rsync --append-verify semantics: resumes partial pulls, checksums finals.
    Incremental: a single rsync call handles all new files in the remote dir.
  - Local mirror target: data/backups/mirror/<titan_id>/  (flat layout — rsync
    preserves filenames; remote Titan's naming is already self-describing).
  - Retention: purge files older than retention_days (default 7) — we want
    recent copies, not a full history. Latest ≥2 entries preserved even if
    older than retention (ensures we always have SOMETHING).

Security note: Phase 9 ships with optional Phase 7 encryption — if T2/T3 have
their encryption_enabled=true, the mirrored tarballs are already ciphertext,
so T1 "holds but cannot read" per rFP §5.8 privacy posture. Without encryption,
T1 holds plaintext T2/T3 state; that's an explicit trust choice the Maker
makes by leaving encryption off.
"""

import asyncio
import glob
import logging
import os
import shutil
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)


class OffhostMirror:
    """Owns the rsync-over-SSH pull + retention cleanup for T2/T3 → T1."""

    def __init__(self, config: dict):
        mirror_cfg = (config or {}).get("backup", {}).get("mirror", {}) or {}
        self.enabled: bool = bool(mirror_cfg.get("enabled", False))
        self.ssh_user: str = mirror_cfg.get("ssh_user", "antigravity")
        self.retention_days: int = int(mirror_cfg.get("retention_days", 7))
        self.local_base: str = mirror_cfg.get("local_base", "data/backups/mirror")
        # Hosts: list of (titan_id, host, remote_path). Defaults align with
        # the standard T1-T2 / T1-T3 VPC layout (T2 at :/projects/titan,
        # T3 at :/projects/titan3, both on 10.135.0.6 different users/paths).
        self.hosts = []
        t2_host = mirror_cfg.get("t2_host", "")
        t2_path = mirror_cfg.get("t2_backup_path", "")
        if t2_host and t2_path:
            self.hosts.append(("T2", t2_host, t2_path))
        t3_host = mirror_cfg.get("t3_host", t2_host)
        t3_path = mirror_cfg.get("t3_backup_path", "")
        if t3_host and t3_path:
            self.hosts.append(("T3", t3_host, t3_path))
        self.ssh_opts = mirror_cfg.get(
            "ssh_opts",
            ["-o", "StrictHostKeyChecking=accept-new",
             "-o", "ConnectTimeout=15"],
        )
        self.rsync_timeout = int(mirror_cfg.get("rsync_timeout_sec", 900))  # 15 min
        self.max_file_age_hours = int(mirror_cfg.get("pull_recent_hours", 48))

    # ── rsync primitive ─────────────────────────────────────────────────

    def _local_dir_for(self, titan_id: str) -> str:
        return os.path.join(self.local_base, titan_id)

    def _build_rsync_cmd(self, host: str, remote_path: str, local_dir: str) -> list:
        ssh_cmd = "ssh " + " ".join(self.ssh_opts)
        # -avP: archive + verbose + progress; --append-verify: resume + checksum
        # --include: only mirror our tarballs + encrypted variants
        # --exclude: everything else (foreign cron tarballs, legacy backups)
        return [
            "rsync", "-avP", "--append-verify",
            "-e", ssh_cmd,
            "--include=personality_*.tar.gz",
            "--include=personality_*.tar.gz.enc",
            "--include=soul_*.tar.gz",
            "--include=soul_*.tar.gz.enc",
            "--include=timechain_*.tar.zst",
            "--include=timechain_*.tar.zst.enc",
            "--include=timechain_*.tar.gz",
            "--include=timechain_*.tar.gz.enc",
            "--exclude=*",
            f"{self.ssh_user}@{host}:{remote_path.rstrip('/')}/",
            f"{local_dir.rstrip('/')}/",
        ]

    def pull_one(self, titan_id: str, host: str, remote_path: str) -> dict:
        """Synchronous rsync pull. Returns {ok, titan_id, host, duration_s, files,
        bytes, error?}. Caller wraps in asyncio.to_thread for non-blocking use."""
        local_dir = self._local_dir_for(titan_id)
        os.makedirs(local_dir, exist_ok=True)
        cmd = self._build_rsync_cmd(host, remote_path, local_dir)
        started = time.time()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.rsync_timeout,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "titan_id": titan_id, "host": host,
                    "duration_s": round(time.time() - started, 1),
                    "error": f"rsync timeout after {self.rsync_timeout}s"}
        except Exception as e:
            return {"ok": False, "titan_id": titan_id, "host": host,
                    "duration_s": round(time.time() - started, 1),
                    "error": f"rsync invocation failed: {e}"}

        duration = round(time.time() - started, 1)
        if proc.returncode != 0:
            return {"ok": False, "titan_id": titan_id, "host": host,
                    "duration_s": duration, "returncode": proc.returncode,
                    "error": (proc.stderr or proc.stdout or "").strip()[:500]}

        # Parse rsync verbose output (last 2 lines for totals)
        stats = self._parse_rsync_stats(proc.stdout)
        return {"ok": True, "titan_id": titan_id, "host": host,
                "duration_s": duration, **stats}

    @staticmethod
    def _parse_rsync_stats(output: str) -> dict:
        """Extract 'sent/received/total size' from rsync -av trailing lines."""
        files = 0
        bytes_total = 0
        for line in reversed(output.splitlines()):
            ls = line.strip()
            if ls.startswith("total size is "):
                try:
                    # "total size is 12,345 speedup is ..."
                    part = ls.split("total size is ", 1)[1].split(" ", 1)[0]
                    bytes_total = int(part.replace(",", ""))
                except Exception:
                    pass
            if ls.startswith("Number of regular files transferred:"):
                try:
                    files = int(ls.split(":", 1)[1].strip().replace(",", ""))
                except Exception:
                    pass
        return {"files_transferred": files, "bytes_total": bytes_total}

    async def pull_all(self) -> dict:
        """Concurrent pulls from all configured hosts (ThreadPoolExecutor)."""
        if not self.enabled:
            return {"ok": True, "mode": "disabled", "results": []}
        if not self.hosts:
            return {"ok": True, "mode": "no_hosts", "results": []}
        results = await asyncio.gather(
            *[asyncio.to_thread(self.pull_one, tid, host, rpath)
              for tid, host, rpath in self.hosts],
            return_exceptions=True,
        )
        normalized = []
        any_fail = False
        for r in results:
            if isinstance(r, Exception):
                normalized.append({"ok": False, "error": str(r)})
                any_fail = True
            else:
                normalized.append(r)
                if not r.get("ok"):
                    any_fail = True
        return {"ok": not any_fail, "results": normalized,
                "completed_at": int(time.time())}

    # ── Retention cleanup ──────────────────────────────────────────────

    def cleanup(self, titan_id: str) -> int:
        """Remove mirror files older than retention_days. Always retains the
        two newest files per pattern (safety floor)."""
        local_dir = self._local_dir_for(titan_id)
        if not os.path.isdir(local_dir):
            return 0
        patterns = ["personality_*.tar.*", "soul_*.tar.*", "timechain_*.tar.*"]
        cutoff = time.time() - (self.retention_days * 86400)
        deleted = 0
        for pat in patterns:
            paths = sorted(glob.glob(os.path.join(local_dir, pat)),
                           key=lambda p: os.path.getmtime(p), reverse=True)
            # Skip the two newest regardless of age
            for path in paths[2:]:
                try:
                    if os.path.getmtime(path) < cutoff:
                        os.remove(path)
                        deleted += 1
                except OSError as e:
                    logger.warning("[OffhostMirror] cleanup: %s: %s", path, e)
        if deleted:
            logger.info("[OffhostMirror] Cleaned %d old %s mirror files",
                        deleted, titan_id)
        return deleted

    def cleanup_all(self) -> dict:
        """Retention cleanup across all configured hosts."""
        removed = {}
        for tid, _h, _p in self.hosts:
            removed[tid] = self.cleanup(tid)
        return removed

    # ── Status ─────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Summary dict for /v4/backup/status integration."""
        out = {"enabled": self.enabled, "hosts": []}
        for tid, host, _rpath in self.hosts:
            local_dir = self._local_dir_for(tid)
            if not os.path.isdir(local_dir):
                out["hosts"].append({"titan_id": tid, "host": host,
                                      "files": 0, "size_mb": 0.0, "newest_age_h": None})
                continue
            files = [p for p in glob.glob(os.path.join(local_dir, "*"))
                     if os.path.isfile(p)]
            total_bytes = sum(os.path.getsize(p) for p in files)
            newest_age = None
            if files:
                newest = max(os.path.getmtime(p) for p in files)
                newest_age = round((time.time() - newest) / 3600, 2)
            out["hosts"].append({
                "titan_id": tid, "host": host,
                "files": len(files),
                "size_mb": round(total_bytes / 1024 / 1024, 2),
                "newest_age_h": newest_age,
            })
        return out
