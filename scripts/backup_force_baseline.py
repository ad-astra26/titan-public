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


async def _do_baseline(titan_id: str, dry_run: bool, force: bool,
                       soul_first: bool = False) -> int:
    """Returns exit code: 0 = success, 1-9 = various failure modes.

    soul_first=True: the SPEC §24.4.C soul-tier FIRST baseline. The 05-29
    bootstrap deliberately skipped soul (lines below), so every Sunday soul
    package has since full-shipped consciousness.db (no baseline to diff
    against) → 530 MB → 402. This mode ships the soul tier FULL as its first
    baseline while keeping personality/timechain as cheap diffs against the
    EXISTING baseline (forced incremental — today is a month-boundary day but
    we deliberately do NOT re-baseline personality, just 3 days after 05-29).
    After ship it refreshes the soul files into the baseline working dir so
    the next Sunday soul package diffs (xdelta3) instead of full-shipping.
    """
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
    if soul_first:
        # soul-first APPENDS to an existing chain (it needs the personality
        # baseline to diff against) — existing events are REQUIRED, not a
        # block condition.
        if not existing:
            logger.error(
                "[force-baseline] --soul-first-baseline needs an existing "
                "personality baseline to diff against; manifest is empty. "
                "Run a normal baseline first.")
            return 5
    elif existing and not force:
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

    # §24.7.a — in-process ZK-Vault network client (identity keypair is already
    # in this process; signs the v=3 chain commit). Without it
    # commit_event_v3_chain returns None ("no network client") → event fails
    # AFTER the Arweave uploads already spent SOL (orphaned). Mirrors
    # backup_worker.py's HybridNetworkClient wiring.
    backup_network = None
    if net == "mainnet":
        try:
            from titan_hcl.core.network import HybridNetworkClient
            backup_network = HybridNetworkClient(config=net_cfg)
            logger.info("[force-baseline] in-process ZK network client wired "
                        "(§24.7.a) — v=3 chain commit can sign")
        except Exception as e:
            logger.error(
                "[force-baseline] HybridNetworkClient wiring FAILED: %s — "
                "v=3 chain commit will fail; aborting before any SOL spend", e)
            return 6

    backup = RebirthBackup(
        network_client=backup_network,
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
    # Soul tier. Normal bootstrap SKIPS soul (the original first-baseline was
    # personality+timechain only — that's exactly why soul never got an
    # Arweave baseline). --soul-first-baseline includes it FULL.
    if soul_first:
        s_specs = backup._tier_specs_from_paths(backup.WEEKLY_EXTRA_PATHS)
    else:
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

    # Soul tier raw size (full-shipped in soul_first mode)
    total_soul_bytes = 0
    missing_soul = []
    if soul_first:
        for s in s_specs:
            if os.path.exists(s.source_path):
                total_soul_bytes += os.path.getsize(s.source_path)
            else:
                missing_soul.append(s.source_path)

    # Estimated post-zstd cost: assume 35% compression ratio (observed).
    # NOTE: in soul_first mode personality/timechain ship as DIFFS (xdelta3
    # vs the existing baseline), so their FULL-size estimate here is an
    # UPPER bound — the diff payloads are far smaller. Soul is full-ship.
    est_p_mb = total_personality_bytes * 0.35 / (1 << 20)
    est_t_mb = total_timechain_bytes * 0.35 / (1 << 20)
    est_s_mb = total_soul_bytes * 0.35 / (1 << 20)
    total_mb = est_p_mb + est_t_mb + est_s_mb
    # Irys cost: ~0.0004 SOL per MB at mainnet pricing (Irys docs)
    est_sol_cost = total_mb * 0.0004
    if soul_first:
        logger.info(
            "[force-baseline] SOUL-FIRST estimate: soul=%.1f MB FULL "
            "(raw=%.0f MB) ≈ %.4f SOL; personality+timechain ship as DIFFS "
            "(upper bound full=%.1f MB — real diff payload is far smaller). "
            "Forced event_type=incremental (no personality re-baseline).",
            est_s_mb, total_soul_bytes / (1 << 20), est_s_mb * 0.0004,
            est_p_mb + est_t_mb)
    else:
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
    if soul_first:
        print(f"=== CONFIRM SOUL-FIRST-BASELINE COMMIT — Titan {titan_id} mainnet ===")
        print(f"  Soul tier:             {est_s_mb:.1f} MB FULL ≈ {est_s_mb*0.0004:.4f} SOL")
        print(f"  Personality/timechain: DIFFS vs existing baseline (cheap)")
        print(f"  Event type:            incremental (NO personality re-baseline)")
        print(f"  Soul paths in scope:   {len(s_specs)}")
    else:
        print(f"=== CONFIRM BASELINE COMMIT — Titan {titan_id} mainnet ===")
        print(f"  Estimated upload size: {total_mb:.1f} MB compressed")
        print(f"  Estimated SOL cost:    {est_sol_cost:.4f} SOL")
        print(f"  Files in scope:        {len(p_specs)} personality + "
              f"{len(t_specs)} timechain")
    print()
    _confirm = "commit soul baseline" if soul_first else "commit baseline"
    ans = input(f"Proceed? Type EXACTLY '{_confirm}' to confirm: ").strip()
    if ans != _confirm:
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

    async def _zk_commit(event_id, ts, event_type, event_root, components,
                         prev_sig):
        # v=3 sovereign chain commit (chunk 5J-2): one memo per component +
        # commit_state(event_root) co-bundled with the head. Matches the
        # pipeline's 6-arg ZkCommitter contract (was a stale 3-arg call that
        # failed AFTER the uploads → orphaned SOL).
        return await backup.commit_event_v3_chain(
            event_id=event_id, ts=ts, event_type=event_type,
            event_merkle_root=event_root, components=components,
            prev_sig=prev_sig,
        )

    def _baseline_resolver(component, arc_name):
        if soul_first:
            # Personality/timechain diff against the EXISTING baseline working
            # dir; soul files are absent there → resolver returns None → soul
            # full-ships (its first baseline). This is exactly the per-tier
            # behavior we want from one forced-incremental event.
            candidate = os.path.join(base_dir, arc_name)
            return candidate if os.path.exists(candidate) else None
        # Normal first-baseline: no baseline → full-ship everything.
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
        # soul-first stays incremental even on a month-boundary day so it
        # does NOT redundantly re-baseline personality (just done 05-29).
        force_event_type="incremental" if soul_first else None,
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

    # ── soul-first: refresh soul files into the baseline working dir so the
    # next Sunday soul package diffs (xdelta3) instead of full-shipping.
    # CRITICAL for restore integrity: the baseline-dir bytes MUST equal the
    # bytes we just shipped full to Arweave, so future soul diffs reconstruct
    # correctly. Copy from source now (the shipped tarball was packed from
    # these same source paths moments ago). ──
    if soul_first and result.tiers.get("soul") and result.tiers["soul"].tx_id:
        import shutil
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
                logger.warning(
                    "[force-baseline] soul baseline-dir refresh %s → %s "
                    "FAILED: %s (next Sunday will full-ship this file)",
                    src, dst, e)
        logger.info(
            "[force-baseline] refreshed %d soul path(s) into baseline "
            "working dir %s — next Sunday soul package will diff (xdelta3)",
            refreshed, base_dir)

    print()
    print("=== %s EVENT SHIPPED ===" % (
        "SOUL-FIRST-BASELINE" if soul_first else "BASELINE"))
    print(f"  event_id:           {result.event_id}")
    print(f"  event_type:         {result.event_type}")
    print(f"  baseline_trigger:   {result.baseline_trigger}")
    print(f"  duration:           {dur:.1f}s")
    print(f"  event_merkle_root:  {result.event_merkle_root}")
    print(f"  zk_commit_tx:       {result.zk_commit_tx}")
    for tier_name in ("personality", "timechain", "soul"):
        t = result.tiers.get(tier_name)
        if t is None:
            continue
        size_mb = t.tarball_size_bytes / (1 << 20)
        print(f"  {tier_name} tx_id:  {t.tx_id}  ({size_mb:.1f} MB, "
              f"packed={t.files_packed} skipped={t.files_skipped})")
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
    p.add_argument("--soul-first-baseline", action="store_true",
                   help="Ship the SPEC §24.4.C soul-tier FIRST baseline (soul "
                        "FULL + personality/timechain DIFFS, forced incremental). "
                        "Establishes the soul baseline the 05-29 bootstrap "
                        "skipped, so weekly Sunday soul packages diff (xdelta3) "
                        "instead of full-shipping consciousness.db every week.")
    args = p.parse_args()

    titan_id = _resolve_titan_id(args.titan_id)
    logger.info("[force-baseline] resolved titan_id=%s", titan_id)

    return asyncio.run(_do_baseline(
        titan_id=titan_id,
        dry_run=args.dry_run,
        force=args.force,
        soul_first=args.soul_first_baseline,
    ))


if __name__ == "__main__":
    sys.exit(main())
