#!/usr/bin/env python3
"""Salvage an orphaned unified_v2 backup event — uploads SUCCEEDED but the
ZK commit FAILED, so SOL was spent but the event was never anchored in the
manifest / on-chain (the recurring failure mode: 05-29 personality+timechain,
and 06-01 soul-first).

This does NOT re-upload (that would waste the SOL again). It re-fetches the
already-uploaded tarballs from Arweave, recomputes their sha256 (= the manifest
merkle_root, which is the sha256 of the on-disk .tar.zst per
backup_event_tarball.pack_event_tarball), commits the SPEC §24.7 v=3 chain
(commit_event_v3_chain), and appends the event to the manifest — exactly as
run_unified_event would have, but with the EXISTING tx_ids.

Usage:
    python scripts/backup_salvage_orphaned_event.py --titan-id T1 --dry-run \
        --event-type incremental \
        --pt 0evqY14OwV67NpAj --tc q5lHPuYI6jH-NOwj --sl 0Heg6y0cxr_G-7Eh \
        --refresh-soul-baseline-dir

    # then --commit to anchor (writes chain + manifest)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import subprocess
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("backup_salvage")


def _resolve_titan_id(arg: Optional[str]) -> str:
    return arg or os.environ.get("TITAN_ID", "T1")


def _fetch_and_sha256(tx_id: str, label: str) -> tuple[str, int]:
    """Download the Arweave tx bytes to a temp file via the Irys gateway,
    return (sha256_hex, size_bytes). Uses curl -L (gateway 302-redirects)."""
    out_path = f"/tmp/salvage_{label}_{tx_id[:12]}.bin"
    # arweave.net serves freshly-bundled Irys tx before the Irys gateway
    # propagates (verified 2026-06-01); fall back to the Irys gateway.
    endpoints = [f"https://gateway.irys.xyz/{tx_id}",
                 f"https://arweave.net/{tx_id}"]
    last_err = ""
    for url in endpoints:
        logger.info("[salvage] fetching %s tier from %s ...", label, url)
        rc = subprocess.run(
            ["curl", "-sfL", "--max-time", "600", "-o", out_path, url],
            capture_output=True, text=True)
        if rc.returncode == 0 and os.path.exists(out_path):
            break
        last_err = f"rc={rc.returncode} {rc.stderr[:160]}"
    else:
        raise RuntimeError(f"fetch {label} tx={tx_id} failed: {last_err}")
    h = hashlib.sha256()
    size = 0
    with open(out_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    sha = h.hexdigest()
    logger.info("[salvage] %s: %d bytes, sha256=%s", label, size, sha)
    try:
        os.unlink(out_path)
    except OSError:
        pass
    return sha, size


async def _do_salvage(titan_id: str, dry_run: bool, event_type: str,
                      pt_tx: str, tc_tx: str, sl_tx: Optional[str],
                      refresh_soul: bool) -> int:
    os.environ["TITAN_ID"] = titan_id
    from titan_hcl.config_loader import load_titan_config
    from titan_hcl.logic.backup import RebirthBackup
    from titan_hcl.logic.backup_unified_manifest import (
        UnifiedManifest, make_event,
    )
    from titan_hcl.logic.backup_zk_commit import compute_event_merkle_root
    from titan_hcl.utils.arweave_store import ArweaveStore

    cfg = load_titan_config()
    net_cfg = cfg.get("network", {}) or {}
    net = net_cfg.get("solana_network", "devnet")
    if net == "mainnet-beta":
        net = "mainnet"
    keypair_path = net_cfg.get("wallet_keypair_path", "")
    if not keypair_path or not os.path.exists(keypair_path):
        logger.error("[salvage] wallet_keypair_path missing: %s", keypair_path)
        return 6

    # ── Re-fetch each orphaned tarball + recompute sha256 (= merkle_root) ──
    pt_sha, pt_size = _fetch_and_sha256(pt_tx, "personality")
    tc_sha, tc_size = _fetch_and_sha256(tc_tx, "timechain")
    sl_sha = sl_size = None
    if sl_tx:
        sl_sha, sl_size = _fetch_and_sha256(sl_tx, "soul")

    event_root = compute_event_merkle_root(
        personality_merkle_root=pt_sha,
        timechain_merkle_root=tc_sha,
        soul_merkle_root=sl_sha,
    )

    manifest = UnifiedManifest.load(titan_id=titan_id, base_dir="data")
    prev_event = manifest.get_latest_event()
    prev_event_id = prev_event["event_id"] if prev_event else None
    prev_sig = prev_event.get("zk_commit_tx") if prev_event else None

    components = [
        {"tier": "PT", "tx_id": pt_tx, "arc": pt_sha},
        {"tier": "TC", "tx_id": tc_tx, "arc": tc_sha},
    ]
    if sl_tx and sl_sha:
        components.append({"tier": "SL", "tx_id": sl_tx, "arc": sl_sha})

    event_id = f"salvage{int(time.time())}".ljust(32, "0")[:32]
    ts = int(time.time())

    print()
    print(f"=== SALVAGE PLAN — Titan {titan_id} ({event_type}) ===")
    print(f"  new event_id:       {event_id}")
    print(f"  prev_event_id:      {prev_event_id}")
    print(f"  prev_sig:           {(prev_sig or 'genesis')[:24]}")
    print(f"  personality:        {pt_tx}  ({pt_size} B)  sha={pt_sha[:16]}")
    print(f"  timechain:          {tc_tx}  ({tc_size} B)  sha={tc_sha[:16]}")
    if sl_tx:
        print(f"  soul (FULL):        {sl_tx}  ({sl_size} B)  sha={sl_sha[:16]}")
    print(f"  event_merkle_root:  {event_root}")
    print()

    if dry_run:
        logger.info("[salvage] DRY-RUN — no chain write, no manifest mutation. "
                    "Re-run with --commit to anchor.")
        return 0

    # ── Wire the in-process ZK network client + RebirthBackup ──
    arweave_store = ArweaveStore(keypair_path=keypair_path, network=net)
    backup_network = None
    if net == "mainnet":
        from titan_hcl.core.network import HybridNetworkClient
        backup_network = HybridNetworkClient(config=net_cfg)
    backup = RebirthBackup(
        network_client=backup_network,
        config=cfg.get("memory_and_storage", {}),
        titan_id=titan_id, arweave_store=arweave_store, full_config=cfg,
    )

    ans = input("Type EXACTLY 'salvage commit' to anchor on-chain: ").strip()
    if ans != "salvage commit":
        logger.info("[salvage] not confirmed — aborting.")
        return 7

    # ── v=3 chain commit over the EXISTING tx_ids ──
    logger.info("[salvage] committing v=3 chain ...")
    commit_result = await backup.commit_event_v3_chain(
        event_id=event_id, ts=ts, event_type=event_type,
        event_merkle_root=event_root, components=components, prev_sig=prev_sig,
    )
    head_sig = (commit_result or {}).get("head_sig") if commit_result else None
    if not head_sig:
        logger.error("[salvage] v=3 chain commit returned no head_sig — "
                     "NOT writing manifest (chain not anchored). result=%s",
                     commit_result)
        return 8
    logger.info("[salvage] v=3 chain committed: head_sig=%s", head_sig)

    # ── Append manifest event (mirrors run_unified_event finalize) ──
    personality_sub = {
        "tx_id": pt_tx, "merkle_root": pt_sha, "size_bytes": pt_size,
        "diff_mode": event_type, "skipped_files": [],
    }
    timechain_sub = {
        "tx_id": tc_tx, "merkle_root": tc_sha, "size_bytes": tc_size,
        "diff_mode": event_type, "block_ranges": {},
    }
    soul_sub = None
    if sl_tx and sl_sha:
        soul_sub = {
            "tx_id": sl_tx, "merkle_root": sl_sha, "size_bytes": sl_size,
            "diff_mode": "full",  # first soul baseline
        }
    event = make_event(
        event_id=event_id, event_type=event_type, prev_event_id=prev_event_id,
        baseline_trigger=None, personality=personality_sub,
        timechain=timechain_sub, soul=soul_sub, zk_commit_tx=head_sig,
        zk_memo_prev_short=(prev_sig[:16] if prev_sig else "genesis"),
        ts_unix=float(ts),
    )
    manifest.append_event(event)
    manifest.save()
    logger.info("[salvage] manifest event appended + saved: %s", event_id)

    # ── Refresh soul files into baseline working dir (so next Sunday diffs) ──
    if refresh_soul and soul_sub is not None:
        import shutil
        base_dir = backup._baseline_working_dir()
        refreshed = 0
        for entry in backup.WEEKLY_EXTRA_PATHS:
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                continue
            src, arc = entry[0], entry[1]
            if not os.path.exists(src):
                continue
            dst = os.path.join(base_dir, arc)
            try:
                parent = os.path.dirname(dst)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
                refreshed += 1
            except Exception as e:
                logger.warning("[salvage] soul refresh %s → %s FAILED: %s",
                               src, dst, e)
        logger.info("[salvage] refreshed %d soul path(s) into %s — next Sunday "
                    "soul package diffs (xdelta3)", refreshed, base_dir)

    print()
    print("=== SALVAGE COMPLETE ===")
    print(f"  event_id:     {event_id}")
    print(f"  zk_commit_tx: {head_sig}")
    print(f"  soul tx:      {sl_tx} (diff_mode=full — first soul baseline)")
    print()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Salvage an orphaned unified_v2 event.")
    p.add_argument("--titan-id", default=None)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--commit", action="store_true")
    p.add_argument("--event-type", default="incremental",
                   choices=["incremental", "baseline"])
    p.add_argument("--pt", required=True, help="personality Arweave tx_id")
    p.add_argument("--tc", required=True, help="timechain Arweave tx_id")
    p.add_argument("--sl", default=None, help="soul Arweave tx_id (optional)")
    p.add_argument("--refresh-soul-baseline-dir", action="store_true",
                   help="copy soul source files into the baseline working dir "
                        "so the next Sunday soul package diffs (xdelta3).")
    args = p.parse_args()
    titan_id = _resolve_titan_id(args.titan_id)
    return asyncio.run(_do_salvage(
        titan_id=titan_id, dry_run=args.dry_run, event_type=args.event_type,
        pt_tx=args.pt, tc_tx=args.tc, sl_tx=args.sl,
        refresh_soul=args.refresh_soul_baseline_dir,
    ))


if __name__ == "__main__":
    sys.exit(main())
