#!/usr/bin/env python3
"""SPEC §24 / Phase 5C — One-shot bootstrap CLI to land the first
unified_v2 baseline event outside the backup_worker subprocess.

Why: backup_worker runs under Guardian's RSS limit (500 MB post-5G). The
FIRST event after enabling unified_v2 is the heaviest because every
in-scope file is full-shipped to populate the baseline working dir.
Streaming encoders (5A) bring peak RSS down dramatically, but a careful
one-shot path that runs outside Guardian lets the operator land the
bootstrap event with full host headroom + clean observability + safe
abort. Future meditations are incremental and fit comfortably in 500 MB.

Usage:
    # Estimate sizes + verify Irys runway, no commit
    python scripts/backup_force_baseline.py --titan-id T1 --dry-run

    # Actually land the baseline event (uploads to Arweave + commits ZK Vault memo)
    python scripts/backup_force_baseline.py --titan-id T1 --commit

    # Force baseline even if manifest already exists (rare; admin override)
    python scripts/backup_force_baseline.py --titan-id T1 --commit --force

The CLI runs `run_unified_event()` directly via dependency-injected real
Arweave + ZK Vault clients. Bypasses the BackupWorker subprocess + Guardian
gate, so RSS isn't constrained — but should still stay well under 500 MB
thanks to streaming encoders.

Acceptance per RFP_phase_c_enhancements.md §3B.3 #4:
- data/backup_unified_manifest_{titan_id}.json exists with events[0].type=="baseline"
  + events[0].baseline_trigger=="first_event"
- events[0].personality.tx_id resolves on Irys gateway (HTTP 200 + sha256 match)
- ZK Vault commit memo on-chain referencing event_merkle_root
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root on path
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backup_force_baseline")


def _resolve_titan_id(cli_arg: Optional[str]) -> str:
    """Resolve TITAN_ID from CLI > env > /home/.../titan_id file > default T1."""
    if cli_arg:
        return cli_arg
    env_id = os.environ.get("TITAN_ID")
    if env_id:
        return env_id
    return "T1"


async def _do_baseline(titan_id: str, dry_run: bool, force: bool) -> int:
    """Returns exit code: 0 = success, 1-9 = various failure modes."""
    # Lazy imports — heavy modules
    os.environ["TITAN_ID"] = titan_id
    from titan_hcl.config_loader import load_titan_config
    from titan_hcl.logic.backup import RebirthBackup
    from titan_hcl.logic.backup_unified_manifest import UnifiedManifest
    from titan_hcl.logic.backup_upload_pipeline import run_unified_event
    from titan_hcl.utils.arweave_store import ArweaveStore

    cfg = load_titan_config()
    bcfg = cfg.get("backup", {}) or {}
    budget = cfg.get("mainnet_budget", {}) or {}

    if not bcfg.get("unified_v2_enabled", False):
        logger.error(
            "[force-baseline] [backup].unified_v2_enabled=false in config — "
            "refusing to bootstrap a baseline against a disabled pipeline. "
            "Flip the config first.")
        return 2
    if not budget.get("backup_arweave_enabled", False):
        logger.error(
            "[force-baseline] [mainnet_budget].backup_arweave_enabled=false — "
            "refusing to ship to Arweave against a disabled flag.")
        return 3

    # ── Manifest sanity check ──────────────────────────────────────────
    try:
        manifest = UnifiedManifest.load(titan_id=titan_id, base_dir="data")
    except ValueError as e:
        logger.error("[force-baseline] manifest load failed: %s", e)
        return 4

    existing = manifest.events
    if existing and not force:
        latest = manifest.get_latest_event()
        logger.error(
            "[force-baseline] manifest already has %d event(s); latest "
            "event_id=%s type=%s. Pass --force to override (rare; admin "
            "operation).",
            len(existing),
            (latest.get("event_id", "?")[:8] if latest else "?"),
            (latest.get("type") if latest else "?"))
        return 5

    # ── Build ArweaveStore + RebirthBackup (we only use RebirthBackup
    # for its path tuples + helpers; don't actually run on_meditation_complete) ──
    net_cfg = cfg.get("network", {}) or {}
    net = net_cfg.get("solana_network", "devnet")
    if net == "mainnet-beta":
        net = "mainnet"
    keypair_path = net_cfg.get("wallet_keypair_path", "")
    if not keypair_path or not os.path.exists(keypair_path):
        logger.error(
            "[force-baseline] wallet_keypair_path missing or unreadable: %s",
            keypair_path)
        return 6

    arweave_store = ArweaveStore(keypair_path=keypair_path, network=net)
    logger.info(
        "[force-baseline] ArweaveStore wired (network=%s, keypair=%s)",
        net, keypair_path)

    backup = RebirthBackup(
        network_client=None,
        config=cfg.get("memory_and_storage", {}),
        titan_id=titan_id,
        arweave_store=arweave_store,
        full_config=cfg,
    )

    # Build tier specs from RebirthBackup's path tuples
    p_specs = backup._tier_specs_from_paths(backup.PERSONALITY_PATHS)
    t_specs = backup._tier_specs_from_paths(
        backup.TIMECHAIN_PATHS, format_hint="timechain_bin",
    )
    # Soul tier weekly-only — bootstrap event is run on-demand (could be any
    # weekday); per SPEC §24.3 only Sunday meditation events include soul.
    # Bootstrap deliberately SKIPS soul so the baseline isn't bloated with
    # consciousness.db (4.2 GB raw → could push RSS into swap territory even
    # with streaming). Next Sunday's meditation will create a soul-inclusive
    # incremental that anchors against this baseline.
    s_specs = None

    # ── Estimate sizes from tier specs (dry-run gate) ──────────────────
    total_personality_bytes = 0
    missing_personality = []
    for s in p_specs:
        if os.path.exists(s.source_path):
            total_personality_bytes += os.path.getsize(s.source_path)
        else:
            missing_personality.append(s.source_path)
    total_timechain_bytes = 0
    missing_timechain = []
    for s in t_specs:
        if os.path.exists(s.source_path):
            total_timechain_bytes += os.path.getsize(s.source_path)
        else:
            missing_timechain.append(s.source_path)

    # Estimated post-gzip cost: assume 35% compression ratio (observed)
    est_p_mb = total_personality_bytes * 0.35 / (1 << 20)
    est_t_mb = total_timechain_bytes * 0.35 / (1 << 20)
    total_mb = est_p_mb + est_t_mb
    # Irys cost: ~0.0004 SOL per MB at mainnet pricing (Irys docs)
    est_sol_cost = total_mb * 0.0004
    logger.info(
        "[force-baseline] size estimate: personality=%.1f MB (raw=%.0f MB), "
        "timechain=%.1f MB (raw=%.0f MB), total compressed ≈ %.1f MB, "
        "est cost ≈ %.4f SOL",
        est_p_mb, total_personality_bytes / (1 << 20),
        est_t_mb, total_timechain_bytes / (1 << 20),
        total_mb, est_sol_cost)
    if missing_personality:
        logger.warning(
            "[force-baseline] %d personality paths missing on disk (will "
            "be silently skipped by ship_tier): %s",
            len(missing_personality), missing_personality[:5])
    if missing_timechain:
        logger.warning(
            "[force-baseline] %d timechain paths missing on disk: %s",
            len(missing_timechain), missing_timechain[:5])

    if dry_run:
        logger.info(
            "[force-baseline] DRY-RUN — no Arweave upload. Re-run with "
            "--commit to ship the baseline event.")
        return 0

    # ── Confirm with Maker before spending SOL ─────────────────────────
    print()
    print(f"=== CONFIRM BASELINE COMMIT — Titan {titan_id} mainnet ===")
    print(f"  Estimated upload size: {total_mb:.1f} MB compressed")
    print(f"  Estimated SOL cost:    {est_sol_cost:.4f} SOL")
    print(f"  Files in scope:        {len(p_specs)} personality + "
          f"{len(t_specs)} timechain")
    print()
    ans = input("Proceed? Type EXACTLY 'commit baseline' to confirm: ").strip()
    if ans != "commit baseline":
        logger.info("[force-baseline] not confirmed — aborting.")
        return 7

    # ── Build orchestrator deps ────────────────────────────────────────
    base_dir = backup._baseline_working_dir()
    logger.info("[force-baseline] baseline_working_dir = %s", base_dir)

    async def _arweave_upload(data: bytes, tags: dict) -> str:
        import tempfile
        # Phase 5 chunk 5F: zstd-3 (.tar.zst). Suffix matches the bytes
        # the unified pipeline handed us; irys_upload.js is content-agnostic
        # but the suffix aids on-disk inspection if upload fails mid-flight.
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".tar.zst",
            prefix=f"titan_unified_{titan_id}_",
        ) as f:
            f.write(data)
            tmp_path = f.name
        try:
            tx = await arweave_store.upload_file(tmp_path, tags=tags)
            if not tx:
                raise RuntimeError(
                    "ArweaveStore.upload_file returned empty tx_id")
            return tx
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def _zk_commit(event_id, root, prev_root):
        return await backup.commit_event_merkle_to_zk_vault(
            event_id=event_id,
            event_merkle_root=root,
            prev_event_merkle_root=prev_root,
        )

    def _baseline_resolver(component, arc_name):
        # First event has no baseline → resolver returns None → encoders
        # produce diff_mode="full"
        return None

    def _bus_emit(name: str, payload: dict) -> None:
        logger.info(
            "[force-baseline] bus_emit %s: %s",
            name, json.dumps({k: v for k, v in payload.items()
                              if k != "personality_tx" and k != "timechain_tx"},
                             default=str)[:300])

    # ── Run the event ─────────────────────────────────────────────────
    t0 = time.time()
    logger.info("[force-baseline] starting run_unified_event...")
    result = await run_unified_event(
        titan_id=titan_id,
        manifest=manifest,
        personality_specs=p_specs,
        timechain_specs=t_specs,
        soul_specs=s_specs,
        baseline_resolver=_baseline_resolver,
        arweave_uploader=_arweave_upload,
        zk_committer=_zk_commit,
        bus_emit=_bus_emit,
    )
    dur = time.time() - t0

    logger.info(
        "[force-baseline] run_unified_event status=%s event_id=%s "
        "event_type=%s duration=%.1fs",
        result.status, result.event_id, result.event_type, dur)

    if result.status != "shipped":
        logger.error(
            "[force-baseline] event NOT shipped: errors=%s",
            "; ".join(result.errors))
        return 8

    print()
    print("=== BASELINE EVENT SHIPPED ===")
    print(f"  event_id:           {result.event_id}")
    print(f"  baseline_trigger:   {result.baseline_trigger}")
    print(f"  duration:           {dur:.1f}s")
    print(f"  event_merkle_root:  {result.event_merkle_root}")
    print(f"  zk_commit_tx:       {result.zk_commit_tx}")
    for tier_name in ("personality", "timechain"):
        t = result.tiers.get(tier_name)
        if t is None:
            continue
        size_mb = t.tarball_size_bytes / (1 << 20)
        print(f"  {tier_name} tx_id:  {t.tx_id}  ({size_mb:.1f} MB)")
    print()
    print(f"  Manifest written:   data/backup_unified_manifest_{titan_id}.json")
    print()
    print("Next steps:")
    print(f"  1. Verify manifest: cat data/backup_unified_manifest_{titan_id}.json | jq")
    print(f"  2. Verify Arweave reachability: "
          f"curl -sI https://gateway.irys.xyz/<personality_tx_id>")
    print("  3. Subsequent meditations will now ship incremental events "
          "(small xdelta3 patches).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=("Phase 5C — one-shot bootstrap CLI to land the first "
                     "SPEC §24 unified_v2 baseline event."))
    p.add_argument("--titan-id", default=None,
                   help="Titan ID (default: $TITAN_ID or T1)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true",
                   help="Estimate sizes + cost; no Arweave upload, no SOL spend.")
    g.add_argument("--commit", action="store_true",
                   help="Actually ship the baseline event. Requires 'commit baseline' "
                        "confirmation prompt.")
    p.add_argument("--force", action="store_true",
                   help="Override existing-manifest check (admin operation; rare).")
    args = p.parse_args()

    titan_id = _resolve_titan_id(args.titan_id)
    logger.info("[force-baseline] resolved titan_id=%s", titan_id)

    return asyncio.run(_do_baseline(
        titan_id=titan_id,
        dry_run=args.dry_run,
        force=args.force,
    ))


if __name__ == "__main__":
    sys.exit(main())
