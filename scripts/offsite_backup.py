#!/usr/bin/env python3
"""offsite_backup.py — convenience off-site COPY of local backup snapshots.

W8 / RFP_Titan_setup_release decision #15. This is the executor the TC²
"backup config" cron invokes. It is a *convenience* off-site copy and is
**NOT** the sovereign restore path:

  - The SOVEREIGN restore path is the Arweave + ZK-Vault chain (mainnet,
    automatic, never hand-triggered). It is the only thing required to
    resurrect a Titan and is governed entirely by the backup_worker.
  - This script copies the already-produced *local* snapshot tarballs in
    ``data/backups/`` to a second location (a mounted disk / NAS path, or an
    S3 bucket) purely so an owner has a redundant off-box copy. It NEVER reads
    the wallet, NEVER touches Arweave, NEVER mutates sovereign state. Losing
    these copies costs nothing — the chain remains the source of truth.

Backends (set in ``~/.titan/secrets.toml [backup_offsite]``):
  - ``local``  — rsync ``data/backups/`` → ``local_dir`` (a path/mount).
  - ``s3``     — sync to ``s3://<s3_bucket>/<s3_prefix>`` via the ``aws`` CLI
                 (preferred) or boto3 (fallback). Credentials come from the
                 same config section.

Config is written by the TC² System tab (``titan_console.backup_config``) or
by hand. Run standalone:

    python scripts/offsite_backup.py [--install-root .] [--dry-run]

Exit codes: 0 ok / 2 disabled / 3 misconfigured / 4 executor missing / 5 sync failed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

SECRETS_PATH = Path(os.path.expanduser("~/.titan/secrets.toml"))
SECTION = "backup_offsite"
SOURCE_REL = "data/backups"


def load_offsite_config(secrets_path: Path = SECRETS_PATH) -> dict:
    """Read the [backup_offsite] section (empty dict if absent/unparseable)."""
    if not secrets_path.exists():
        return {}
    try:
        with open(secrets_path, "rb") as f:
            return tomllib.load(f).get(SECTION, {}) or {}
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _sync_local(source: Path, dest: str, *, dry_run: bool) -> tuple[int, str]:
    dst = Path(os.path.expanduser(dest))
    if shutil.which("rsync"):
        argv = ["rsync", "-a", "--delete"]
        if dry_run:
            argv.append("--dry-run")
        # trailing slash on source → copy contents into dst
        argv += [f"{source}/", f"{dst}/"]
        p = subprocess.run(argv, capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip()
    # rsync absent → stdlib copytree (no --delete semantics; best effort)
    if dry_run:
        return 0, f"[dry-run] would copytree {source} → {dst}"
    try:
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, dst, dirs_exist_ok=True)
        return 0, f"copied {source} → {dst} (stdlib, rsync not installed)"
    except OSError as e:
        return 5, str(e)


def _sync_s3(source: Path, cfg: dict, *, dry_run: bool) -> tuple[int, str]:
    bucket = cfg.get("s3_bucket")
    if not bucket:
        return 3, "s3 backend selected but s3_bucket is unset"
    prefix = str(cfg.get("s3_prefix", "")).strip("/")
    uri = f"s3://{bucket}" + (f"/{prefix}" if prefix else "")

    env = dict(os.environ)
    if cfg.get("aws_access_key_id"):
        env["AWS_ACCESS_KEY_ID"] = str(cfg["aws_access_key_id"])
    if cfg.get("aws_secret_access_key"):
        env["AWS_SECRET_ACCESS_KEY"] = str(cfg["aws_secret_access_key"])
    if cfg.get("aws_region"):
        env["AWS_DEFAULT_REGION"] = str(cfg["aws_region"])

    if shutil.which("aws"):
        argv = ["aws", "s3", "sync", f"{source}/", uri, "--delete"]
        if dry_run:
            argv.append("--dryrun")
        p = subprocess.run(argv, capture_output=True, text=True, env=env)
        return (0 if p.returncode == 0 else 5), (p.stdout + p.stderr).strip()

    # boto3 fallback
    try:
        import boto3  # type: ignore
    except ImportError:
        return 4, ("no S3 executor available: install the AWS CLI "
                   "(`apt install awscli`) or `pip install boto3`")
    if dry_run:
        return 0, f"[dry-run] would upload {source}/* → {uri} via boto3"
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=cfg.get("aws_access_key_id") or None,
            aws_secret_access_key=cfg.get("aws_secret_access_key") or None,
            region_name=cfg.get("aws_region") or None,
        )
        uploaded = 0
        for root, _dirs, files in os.walk(source):
            for fn in files:
                local = Path(root) / fn
                key = "/".join(p for p in (prefix, str(local.relative_to(source))) if p)
                s3.upload_file(str(local), bucket, key)
                uploaded += 1
        return 0, f"uploaded {uploaded} files → {uri} via boto3"
    except Exception as e:  # boto3 raises many client errors
        return 5, f"boto3 upload failed: {e}"


def run(install_root: Path, *, dry_run: bool = False,
        secrets_path: Path = SECRETS_PATH) -> tuple[int, str]:
    """Perform the off-site copy. Returns (exit_code, message)."""
    cfg = load_offsite_config(secrets_path)
    if not cfg or not _truthy(cfg.get("enabled", False)):
        return 2, "off-site backup disabled (no [backup_offsite] enabled=true)"

    source = (install_root / SOURCE_REL).resolve()
    if not source.is_dir():
        return 3, f"source snapshot dir not found: {source}"

    backend = str(cfg.get("backend", "")).lower()
    if backend == "local":
        dest = cfg.get("local_dir")
        if not dest:
            return 3, "local backend selected but local_dir is unset"
        rc, msg = _sync_local(source, str(dest), dry_run=dry_run)
    elif backend == "s3":
        rc, msg = _sync_s3(source, cfg, dry_run=dry_run)
    else:
        return 3, f"unknown backend: {backend!r} (expected 'local' or 's3')"

    tag = "[dry-run] " if dry_run else ""
    return rc, f"{tag}offsite[{backend}]: {msg}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Off-site convenience backup copy.")
    ap.add_argument("--install-root", default=".",
                    help="Titan install root (default: cwd)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be synced without copying")
    args = ap.parse_args(argv)
    rc, msg = run(Path(args.install_root).resolve(), dry_run=args.dry_run)
    print(msg, file=sys.stderr if rc not in (0, 2) else sys.stdout)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
