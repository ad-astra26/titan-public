#!/usr/bin/env python3
"""
Daily Arweave backup — uploads TimeChain snapshot (Zstd-compressed) to permanent storage.

Usage:
  python scripts/arweave_backup.py [--titan-id T1] [--dry-run]

Cron (T1 only — TRANSITIONAL until backup_worker lands):
  7 4 * * * cd /home/antigravity/projects/titan && . test_env/bin/activate \\
            && NODE_PATH=$(npm root -g) python scripts/arweave_backup.py \\
            >> /tmp/titan_arweave_backup.log 2>&1

NOTE: must use `. ` (POSIX) not `source` (bash builtin) — cron defaults to /bin/sh
(dash) which has no `source`. The previous form silently failed every night.
The cd-into-project-dir is also required because cron starts in $HOME, not the
project directory, so the relative test_env/bin/activate path needs cwd setup.
"""
import asyncio
import argparse
import json
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ArweaveBackup")


async def run_backup(titan_id: str, dry_run: bool = False):
    from titan_plugin.logic.timechain_backup import TimeChainBackup
    from titan_plugin.utils.arweave_store import ArweaveStore

    data_dir = "data/timechain"
    keypair_path = "data/titan_identity_keypair.json"

    if not os.path.exists(keypair_path):
        logger.error("Keypair not found: %s", keypair_path)
        return False

    # Create snapshot
    logger.info("Creating Zstd snapshot for %s...", titan_id)
    arweave = ArweaveStore(keypair_path=keypair_path, network="mainnet")
    backup = TimeChainBackup(data_dir=data_dir, titan_id=titan_id, arweave_store=arweave)

    tarball, metadata = backup.create_snapshot_tarball()
    if not tarball:
        logger.error("Snapshot creation failed (no genesis?)")
        return False

    logger.info("Snapshot: %.1f MB, %d blocks, %s",
                len(tarball) / 1024 / 1024, metadata["total_blocks"],
                metadata.get("compression", "?"))

    if dry_run:
        logger.info("DRY RUN — skipping upload")
        return True

    # Upload through rFP §5.3 10-step failsafe cascade
    # (validate + local-always + balance check + verify + cleanup)
    logger.info("Uploading to Arweave via Irys (cascade applied)...")
    try:
        from titan_plugin.config_loader import load_titan_config
        _full_cfg = load_titan_config()
    except Exception:
        _full_cfg = {}
    tx_id = await backup.snapshot_to_arweave(full_config=_full_cfg)

    if tx_id:
        logger.info("SUCCESS — Arweave TX: %s", tx_id)
        logger.info("URL: https://arweave.net/%s", tx_id)

        # Also check Irys balance remaining
        status = backup.get_backup_status()
        logger.info("Total backups: %d, latest blocks: %s",
                     status["total_snapshots"], status["latest_blocks"])
        return True
    else:
        logger.error("UPLOAD FAILED")
        return False


def main():
    parser = argparse.ArgumentParser(description="Daily Arweave backup")
    parser.add_argument("--titan-id", default="T1", help="Titan ID (T1/T2/T3)")
    parser.add_argument("--dry-run", action="store_true", help="Create snapshot but don't upload")
    args = parser.parse_args()

    success = asyncio.run(run_backup(args.titan_id, args.dry_run))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
