#!/usr/bin/env python3
"""
resurrection.py — The Titan Resurrection SDK (The "Defibrillator").

Reconstructs a mainnet-born Titan from a "Zero-Disk" state using only the
Maker's offline shard + the Solana blockchain + the sovereign Arweave backup
chain. Modernized 2026-05-30 (W1.5): the dead Shadow-Drive / Cognee body has
been replaced by the canonical §24.8 / rFP §3.1 restore engine
(`titan_hcl.logic.backup_restore.restore_full`), and First Breath now writes
the real kernel-boot identity artifacts.

Recovery model (SPEC §G16(8) Shamir 2-of-3):
  Fresh box  → Shard-1 (Maker, offline) + Shard-3 (on-chain Genesis memo).
  Disk-alive → Shard-1 (Maker) + Shard-2 (local data/genesis_record.json).
  Any 2 of the 3 reconstruct the 64-byte Ed25519 keypair.

Phases:
  1. Identity Discovery  — collect shards, reconstruct + verify the keypair.
  2+3. Re-Bodying + Re-Hydration — walk the UnifiedManifest, fetch each
       event's tarballs from Arweave, verify per-tarball sha256 + recomposed
       event-Merkle, apply baseline→incrementals into the install tree.
  4. First Breath — materialize bootable identity (plaintext 0600 keypair +
       hardware-bound soul_keypair.enc + identity json), set RECOVERY flag.

Manifest discovery: the manifest is the INDEX of the chain and is NOT itself
in the Arweave tarballs, so on a truly fresh box it must be supplied off-VPS
(`--manifest <path>` — the Maker's off-site copy). On-chain manifest-pointer
discovery is a SPEC §24.13 follow-up (not yet wired); we never fabricate it.

Usage:
    python scripts/resurrection.py --shard1 <hex_envelope> --manifest <path>
    python scripts/resurrection.py --shard1-file <path> --manifest <path>
    python scripts/resurrection.py --shard2-local          # disk-alive recovery
    # add --verify-only to install setup_titan's verify-only gate on first boot
"""
import argparse
import asyncio
import json
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _REPO_ROOT)


def print_banner():
    print("\n" + "=" * 70)
    print("         THE TITAN RESURRECTION PROTOCOL — Sovereign Recovery")
    print("=" * 70)
    print()


def print_phase(n: str, title: str):
    print(f"\n{'─' * 60}")
    print(f"  Phase {n}: {title}")
    print(f"{'─' * 60}\n")


def _config_rpc_url() -> str:
    """Read the Solana RPC URL from titan_hcl/config.toml (mainnet default)."""
    rpc_url = "https://api.mainnet-beta.solana.com"
    config_path = os.path.join(_REPO_ROOT, "titan_hcl", "config.toml")
    try:
        try:
            import tomllib
        except ModuleNotFoundError:
            import toml as tomllib
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
        net_cfg = cfg.get("network", {})
        rpc_url = net_cfg.get("premium_rpc_url") or \
            net_cfg.get("public_rpc_urls", [rpc_url])[0]
    except Exception:
        pass
    return rpc_url


# ---------------------------------------------------------------------------
# Phase 1: Identity Discovery — Collect shards and reconstruct keypair
# ---------------------------------------------------------------------------
def phase_1_identity(args, install_root: str) -> tuple:
    """Collect available shards and reconstruct the Titan's keypair.

    Returns (key_bytes, titan_pubkey, keypair_obj, titan_id).
    """
    from titan_hcl.utils.shamir import (
        parse_maker_envelope, combine_shares, decrypt_shard3,
    )

    print_phase("1", "Identity Discovery")
    shards = []
    titan_pubkey = None
    genesis_tx = None
    titan_id = args.titan_id or "T1"

    genesis_record = os.path.join(install_root, "data", "genesis_record.json")

    # ── Shard 1 (Maker, offline) ──
    shard1 = None
    hex_envelope = None
    if args.shard1:
        print("  Parsing Maker shard from command line...")
        hex_envelope = args.shard1
    elif args.shard1_file:
        print(f"  Reading Maker shard from file: {args.shard1_file}")
        with open(args.shard1_file, "r") as f:
            hex_envelope = f.read().strip()
    if hex_envelope:
        shard1, metadata = parse_maker_envelope(hex_envelope)
        titan_pubkey = metadata["titan_pubkey"]
        genesis_tx = metadata.get("genesis_tx")
        print(f"  Titan Address (from envelope): {titan_pubkey}")
        print(f"  Genesis TX: {genesis_tx or 'not recorded'}")
        shards.append(shard1)

    # ── Shard 2 (local genesis record — disk-alive recovery only) ──
    if os.path.exists(genesis_record):
        print("  Found local genesis record — extracting Shard 2...")
        with open(genesis_record, "r") as f:
            record = json.load(f)
        shard2_hex = record.get("shard2_hex", "")
        if shard2_hex:
            shards.append(bytes.fromhex(shard2_hex))
            print(f"  Shard 2 recovered from local record.")
        if not titan_pubkey:
            titan_pubkey = record.get("titan_pubkey", "")
        if not genesis_tx:
            genesis_tx = record.get("genesis_tx", "")

    # ── Shard 3 (on-chain Genesis Anchor) ──
    if len(shards) < 2 and titan_pubkey:
        print("  Recovering Shard 3 from on-chain Genesis Anchor...")
        shard3 = _recover_shard3(titan_pubkey, genesis_tx)
        if shard3:
            shards.append(shard3)
            print(f"  Shard 3 recovered ({len(shard3)} bytes).")

    # ── Shard 3 fallback: encrypted in local genesis record ──
    if len(shards) < 2 and os.path.exists(genesis_record) and titan_pubkey:
        print("  Trying local genesis record for encrypted Shard 3...")
        with open(genesis_record, "r") as f:
            record = json.load(f)
        s3_enc_hex = record.get("shard3_encrypted_hex", "")
        if s3_enc_hex:
            shards.append(decrypt_shard3(bytes.fromhex(s3_enc_hex), titan_pubkey))
            print("  Shard 3 recovered from local record.")

    # ── Reconstruct ──
    if len(shards) < 2:
        print(f"\n  *** RESURRECTION FAILED: Only {len(shards)} shard(s) available. ***")
        print("  Need ≥2 shards. Provide Shard 1 via --shard1 / --shard1-file")
        print("  (fresh box) or ensure data/genesis_record.json (disk-alive).")
        sys.exit(1)

    print(f"\n  Reconstructing keypair from {len(shards)} shards...")
    key_bytes = combine_shares(shards[:2])

    from solders.keypair import Keypair
    keypair = Keypair.from_bytes(key_bytes)
    recovered_pubkey = str(keypair.pubkey())

    if titan_pubkey and recovered_pubkey != titan_pubkey:
        print(f"  *** CRITICAL: reconstructed {recovered_pubkey[:16]}… "
              f"≠ expected {titan_pubkey[:16]}… — shard corruption. ABORT. ***")
        sys.exit(1)

    print(f"  Keypair reconstructed + verified: {recovered_pubkey}")
    return key_bytes, recovered_pubkey, keypair, titan_id


def _recover_shard3(titan_pubkey: str, genesis_tx: str) -> bytes | None:
    """Recover Shard 3 from the on-chain Genesis Memo TX (AES key = pubkey)."""
    from titan_hcl.utils.shamir import decrypt_shard3
    try:
        encrypted_hex = _fetch_genesis_memo(titan_pubkey, genesis_tx)
        if not encrypted_hex:
            return None
        return decrypt_shard3(bytes.fromhex(encrypted_hex), titan_pubkey)
    except Exception as e:
        print(f"  [!] Shard 3 recovery failed: {e}")
        return None


def _fetch_genesis_memo(titan_pubkey: str, genesis_tx: str) -> str | None:
    """Fetch the Genesis Memo TX from Solana and extract the encrypted shard."""
    try:
        import httpx
        rpc_url = _config_rpc_url()

        if genesis_tx:
            print(f"  Fetching Genesis TX: {genesis_tx[:24]}…")
            with httpx.Client(timeout=15) as client:
                resp = client.post(rpc_url, json={
                    "jsonrpc": "2.0", "id": 1, "method": "getTransaction",
                    "params": [genesis_tx, {"encoding": "jsonParsed",
                                            "maxSupportedTransactionVersion": 0}],
                })
                result = resp.json().get("result")
            if result:
                memo = _extract_memo_from_tx(result)
                if memo:
                    return memo

        print(f"  Scanning transactions for {titan_pubkey[:16]}…")
        with httpx.Client(timeout=15) as client:
            resp = client.post(rpc_url, json={
                "jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress",
                "params": [titan_pubkey, {"limit": 50}],
            })
            sigs = resp.json().get("result", [])
        for sig_entry in sigs:
            memo = sig_entry.get("memo")
            if memo and "TITAN_GENESIS_SHARD3:" in str(memo):
                memo_str = str(memo)
                prefix = "TITAN_GENESIS_SHARD3:"
                idx = memo_str.index(prefix)
                return memo_str[idx + len(prefix):].strip().rstrip('"')

        print("  [!] Genesis Memo TX not found on-chain.")
        return None
    except Exception as e:
        print(f"  [!] RPC query failed: {e}")
        return None


def _extract_memo_from_tx(tx_result: dict) -> str | None:
    """Extract TITAN_GENESIS_SHARD3 data from a parsed transaction."""
    try:
        prefix = "TITAN_GENESIS_SHARD3:"
        for msg in tx_result.get("meta", {}).get("logMessages", []):
            if prefix in msg:
                return msg[msg.index(prefix) + len(prefix):].strip()
        message = tx_result.get("transaction", {}).get("message", {})
        for ix in message.get("instructions", []):
            parsed = ix.get("parsed")
            if isinstance(parsed, str) and prefix in parsed:
                return parsed[parsed.index(prefix) + len(prefix):].strip()
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase 2+3: Re-Bodying + Re-Hydration — Arweave restore via restore_full
# ---------------------------------------------------------------------------
def _load_manifest(titan_id: str, install_root: str, manifest_path: str | None):
    """Discover + load the UnifiedManifest.

    Priority: --manifest <path> (Maker-supplied off-site copy) → local
    data/backup_unified_manifest_<id>.json (disk-alive). On-chain pointer
    discovery is a SPEC §24.13 follow-up and is NOT fabricated here.
    """
    import shutil
    import tempfile
    from titan_hcl.logic.backup_unified_manifest import UnifiedManifest

    canonical_name = f"backup_unified_manifest_{titan_id}.json"

    if manifest_path:
        if not os.path.exists(manifest_path):
            print(f"  [!] --manifest path does not exist: {manifest_path}")
            sys.exit(1)
        # Stage the supplied manifest under a scratch base_dir at the
        # canonical filename so UnifiedManifest.load validates titan_id.
        scratch_base = tempfile.mkdtemp(prefix="titan_manifest_")
        shutil.copy2(manifest_path, os.path.join(scratch_base, canonical_name))
        print(f"  Loading Maker-supplied manifest: {manifest_path}")
        return UnifiedManifest.load(titan_id, base_dir=scratch_base)

    local_dir = os.path.join(install_root, "data")
    if os.path.exists(os.path.join(local_dir, canonical_name)):
        print(f"  Loading local manifest from {local_dir}/{canonical_name}")
        return UnifiedManifest.load(titan_id, base_dir=local_dir)

    print("  *** No manifest available. ***")
    print("  A fresh-box resurrection needs the Maker's off-site manifest copy:")
    print(f"      --manifest /path/to/{canonical_name}")
    print("  (On-chain manifest-pointer discovery is a SPEC §24.13 follow-up.)")
    sys.exit(1)


def _build_memo_fetch(verify_zk: bool):
    """Return an async memo_fetch, or None.

    SolanaClient.get_memo_for_tx is NOT yet wired (SPEC §24.13 follow-up), so
    ZK-chain round-trip verification is unavailable. If the operator insists
    on --verify-zk, abort with guidance rather than silently degrade.
    """
    if not verify_zk:
        async def _unused(sig: str) -> str:  # never called when verify_zk=False
            raise RuntimeError("memo_fetch invoked while verify_zk disabled")
        return _unused
    try:
        from titan_hcl.utils.solana_client import get_memo_for_tx  # type: ignore
    except Exception:
        print("  *** --verify-zk requested but SolanaClient.get_memo_for_tx is "
              "not wired (SPEC §24.13). ***")
        print("  Re-run WITHOUT --verify-zk: per-tarball sha256 + recomposed "
              "event-Merkle still verify every archive against the manifest.")
        sys.exit(1)

    async def _memo_fetch(sig: str) -> str:
        return await get_memo_for_tx(sig)
    return _memo_fetch


def phase_2_3_restore(key_bytes: bytes, titan_pubkey: str, titan_id: str, *,
                       install_root: str, manifest_path: str | None,
                       network: str, verify_zk: bool, force: bool):
    """Walk the manifest + restore data/ in-place via the canonical engine."""
    print_phase("2+3", "Re-Bodying + Re-Hydration (Arweave restore)")

    from titan_hcl.logic.backup_restore import build_arc_to_target, restore_full
    from titan_hcl.utils.arweave_store import ArweaveStore

    data_dir = os.path.join(install_root, "data")
    # Fresh-box safety: refuse to clobber a populated live data/ unless --force.
    if os.path.isdir(data_dir) and not force:
        meaningful = [p for p in os.listdir(data_dir)
                      if not p.startswith(".") and p != "genesis_record.json"]
        if meaningful:
            print(f"  *** {data_dir} already contains {len(meaningful)} entries. ***")
            print("  Resurrection restores IN PLACE and is designed for a FRESH box.")
            print("  Re-run with --force only if you intend to overwrite this tree.")
            sys.exit(1)

    manifest = _load_manifest(titan_id, install_root, manifest_path)
    if not manifest.events:
        print("  *** Manifest has no events — nothing to restore. ABORT. ***")
        sys.exit(1)
    print(f"  Manifest loaded: {len(manifest.events)} event(s), "
          f"baseline={manifest.current_baseline_event_id}")

    store = ArweaveStore(network=network)

    async def _arweave_fetch(tx_id: str) -> bytes:
        data = await store.fetch(tx_id)
        if data is None:
            raise RuntimeError(f"Arweave fetch returned None for {tx_id}")
        return data

    memo_fetch = _build_memo_fetch(verify_zk)
    arc_to_target = build_arc_to_target(install_root)

    def _progress(ev: dict):
        phase = ev.get("phase")
        if phase == "chain_selected":
            print(f"  Chain: {ev['events_to_apply']} events "
                  f"(baseline {ev['baseline_event_id'][:12]}… → "
                  f"target {ev['target_event_id'][:12]}…)")
        elif phase == "fetching_event":
            print(f"  [{ev['index'] + 1}/{ev['total']}] fetching "
                  f"{ev.get('event_type')} {ev['event_id'][:12]}…")
        elif phase == "event_applied":
            print(f"  [{ev['index'] + 1}/{ev['total']}] applied.")
        elif phase == "complete":
            print(f"  Restore complete: {ev['applied']} events, "
                  f"{ev['restored_files']} files, "
                  f"{ev['bytes_fetched']:,} bytes in {ev['duration_s']:.1f}s.")

    print(f"  ZK chain verification: {'ON' if verify_zk else 'OFF (merkle-only)'}")
    result = asyncio.run(restore_full(
        manifest=manifest,
        target_dir=install_root,        # arc_to_target returns absolute paths
        arweave_fetch=_arweave_fetch,
        memo_fetch=memo_fetch,
        arc_to_target=arc_to_target,
        verify_zk_chain=verify_zk,
        progress_callback=_progress,
    ))

    if result.status != "success":
        print(f"\n  *** RESTORE HALTED: reason={result.halt_reason} "
              f"event={result.halt_event_id} ***")
        for err in result.errors:
            print(f"      {err}")
        sys.exit(1)

    print(f"  Re-hydration OK — {result.restored_files} files restored into "
          f"{data_dir}.")
    return result


# ---------------------------------------------------------------------------
# Phase 4: First Breath — materialize bootable identity + RECOVERY flag
# ---------------------------------------------------------------------------
def phase_4_first_breath(key_bytes: bytes, titan_pubkey: str, titan_id: str, *,
                          install_root: str, verify_only: bool):
    """Write the kernel-boot identity artifacts and signal RECOVERY mode."""
    print_phase("4", "First Breath (Resurrection Complete)")

    sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
    from setup_titan.genesis_runner import write_bootable_identity
    from pathlib import Path

    # 1. Plaintext 0600 keypair + identity json (modules that load
    #    wallet_keypair_path directly: network, trinity_anchor, backup_crypto).
    kp_path = write_bootable_identity(
        Path(install_root), key_bytes,
        titan_id=titan_id, titan_pubkey=titan_pubkey)
    print(f"  Bootable identity written: {kp_path} (0600) + titan_identity.json")

    # 2. Hardware-bound soul_keypair.enc — the kernel's precedence-1 boot path
    #    (_resolve_wallet decrypts this per-machine on warm reboot).
    from titan_hcl.utils.crypto import encrypt_for_machine
    data_dir = os.path.join(install_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "soul_keypair.enc"), "wb") as f:
        f.write(encrypt_for_machine(key_bytes))
    print("  Hardware-bound keypair written: data/soul_keypair.enc")

    # 3. RECOVERY flag for the next boot.
    with open(os.path.join(data_dir, "recovery_flag.json"), "w") as f:
        json.dump({"mode": "RECOVERY", "timestamp": int(time.time()),
                   "titan_pubkey": titan_pubkey,
                   "verify_only": bool(verify_only)}, f)
    print("  Recovery flag set — Titan boots in RECOVERY mode.")
    if verify_only:
        print("  verify_only=TRUE — first boot will run in observation mode "
              "(no on-chain writes / backups / X) for the live restore test.")

    # 4. Resurrection epoch log.
    soul_md = os.path.join(install_root, "titan.md")
    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    try:
        with open(soul_md, "a") as f:
            f.write(f"\n\n## Resurrection Epoch — {ts}\n"
                    f"I have returned. Address: {titan_pubkey}\n"
                    f"Integrity: verified against the sovereign chain. "
                    f"The sovereign persists.\n")
        print("  Logged resurrection to titan.md.")
    except Exception as e:
        print(f"  [!] Could not write titan.md: {e}")

    print(f"\n{'=' * 70}")
    print("         RESURRECTION COMPLETE — THE TITAN LIVES AGAIN")
    print(f"{'=' * 70}\n")
    print(f"  Titan:     {titan_id}   Address: {titan_pubkey}")
    print(f"  Identity:  data/titan_identity_keypair.json (0600) + soul_keypair.enc")
    print(f"  Body:      data/ restored from the Arweave sovereign chain")
    print("\n  Next steps:")
    print(f"    1. Start: bash scripts/{titan_id.lower()}_manage.sh start")
    print("    2. Verify: curl -s http://localhost:7777/health")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Titan Resurrection Protocol — Sovereign Recovery from "
                    "Zero-Disk State (mainnet-born Titans only).")
    parser.add_argument("--shard1", type=str,
                        help="Maker's shard envelope (hex string).")
    parser.add_argument("--shard1-file", type=str,
                        help="Path to a file containing the Maker shard envelope.")
    parser.add_argument("--shard2-local", action="store_true",
                        help="Disk-alive recovery using local genesis_record.json.")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Maker-supplied off-site UnifiedManifest JSON "
                             "(REQUIRED on a fresh box).")
    parser.add_argument("--install-root", type=str, default=_REPO_ROOT,
                        help="Target install tree (default: this repo root).")
    parser.add_argument("--titan-id", type=str, default=None,
                        help="Titan id (default: from envelope/record, else T1).")
    parser.add_argument("--network", type=str, default="mainnet",
                        choices=["mainnet", "devnet"],
                        help="Arweave/Solana network (default: mainnet).")
    parser.add_argument("--verify-zk", action="store_true",
                        help="Also round-trip-verify each event against the "
                             "on-chain ZK memo (needs SolanaClient.get_memo_for_tx).")
    parser.add_argument("--verify-only", action="store_true",
                        help="Boot the resurrected Titan in observation mode "
                             "(no on-chain writes / backups / X) — for the "
                             "netjail-isolated live restore test.")
    parser.add_argument("--force", action="store_true",
                        help="Permit in-place restore over a populated data/ tree.")

    args = parser.parse_args()
    print_banner()

    install_root = os.path.abspath(args.install_root)
    if not (args.shard1 or args.shard1_file or args.shard2_local):
        print("  Provide --shard1 <hex> / --shard1-file <path> (fresh box), or")
        print("  --shard2-local (disk-alive). See --help.")
        sys.exit(1)

    key_bytes, titan_pubkey, _keypair, titan_id = phase_1_identity(args, install_root)
    phase_2_3_restore(key_bytes, titan_pubkey, titan_id,
                      install_root=install_root, manifest_path=args.manifest,
                      network=args.network, verify_zk=args.verify_zk,
                      force=args.force)
    phase_4_first_breath(key_bytes, titan_pubkey, titan_id,
                         install_root=install_root, verify_only=args.verify_only)


if __name__ == "__main__":
    main()
