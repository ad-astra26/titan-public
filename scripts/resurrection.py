#!/usr/bin/env python3
"""
resurrection.py — The Titan Resurrection SDK (The "Defibrillator").

Reconstructs a mainnet-born Titan from a "Zero-Disk" state using only the
Maker's offline shard + the Solana blockchain + the sovereign Arweave backup
chain. Modernized 2026-05-30 (W1.5): the dead Shadow-Drive / Cognee body has
been replaced by the canonical §24.8 / rFP §3.1 restore engine
(`titan_hcl.logic.backup_restore.restore_full`), and First Breath now writes
the real kernel-boot identity artifacts.

Recovery model (SPEC §G16(8) Shamir 2-of-3) — NO ENVELOPE (INV-MBR-0/10):
  Fresh box  → raw Shard-1 (Maker, offline) + the PUBLIC titan address. From the
               address alone the protocol discovers the GenesisNFT and the
               on-chain Shard-3 memo (wallet-only), decrypts Shard-3 (key =
               public pubkey), and reconstructs the keypair (Shard-1 + Shard-3).
  Disk-alive → Shard-1 (Maker) + Shard-2 (local data/genesis_record.json).
  Any 2 of the 3 reconstruct the 64-byte Ed25519 keypair. The Maker's shard
  carries no locators — the wallet + chain are self-describing.

Phases:
  1. Identity Discovery  — collect shards, reconstruct + verify the keypair.
  2+3. Re-Bodying + Re-Hydration — by DEFAULT walk the on-chain v=3 sovereign
       backup chain (`backup_restore_sovereign.restore_body_from_chain`,
       INV-MBR-12): no manifest, no local files — the chain IS the catalogue.
       Each event's tarballs are fetched from Arweave + sha256/event-Merkle
       verified, baseline→incrementals applied into a scratch dir, then atomically
       swapped into data/. `--manifest <path>` selects the LEGACY non-sovereign
       manifest path (debug/fallback only — needs the Maker's off-site copy).
  4. First Breath — materialize bootable identity (plaintext 0600 keypair +
       hardware-bound soul_keypair.enc + identity json), set RECOVERY flag.

Body restore (INV-MBR-12): the v=3 memos put the backup catalogue ON-CHAIN
(co-bundled with the ZK-Vault commit_state TXs), so a sovereign restore needs no
manifest at all — Shard-1 + the wallet are enough. `--manifest` is retained only
as an explicit legacy fallback.

Usage:
    # Fresh box — sovereign, wallet-only (NO envelope, NO manifest):
    python scripts/resurrection.py --shard1 <raw_hex> --titan-pubkey <address>
    python scripts/resurrection.py --shard1-file <path> --titan-pubkey <address>
    python scripts/resurrection.py --shard2-local          # disk-alive recovery
    # add --verify-only to install setup_titan's verify-only gate on first boot
    # legacy/debug only: append --manifest <path> to use the non-sovereign path
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
def _parse_shard1(hex_str: str) -> tuple:
    """Accept a RAW Shard-1 (canonical — NO envelope) or a legacy JSON envelope.

    The sovereign input (INV-MBR-0) is the raw shard hex + the PUBLIC
    `--titan-pubkey`; the shard carries no locators. A legacy v2.0 envelope is
    still tolerated (its embedded pubkey is read), but is never required and is
    never the documented path. Returns (shard_bytes, titan_pubkey_or_None).
    """
    try:
        raw = bytes.fromhex("".join(hex_str.split()))
    except ValueError:
        raise ValueError("Shard-1 is not valid hex.")
    if raw[:1] == b"{":  # legacy envelope = hex of a JSON object
        from titan_hcl.utils.shamir import parse_maker_envelope
        shard, metadata = parse_maker_envelope(hex_str)
        return shard, (metadata.get("titan_pubkey") or None)
    return raw, None


def _report_identity(discovery: dict) -> None:
    """Print the GenesisNFT identity commitments recovered wallet-only (best-effort)."""
    if not discovery:
        return
    maker = discovery.get("maker")
    csha = discovery.get("constitution_sha")
    dsha = discovery.get("birth_dna_sha")
    if maker:
        print(f"  GenesisNFT identity recovered: Maker {str(maker)[:16]}…")
    if csha:
        print(f"    Constitution SHA-256: {str(csha)[:16]}…")
    if dsha:
        print(f"    Birth DNA SHA-256:    {str(dsha)[:16]}…")


def phase_1_identity(args, install_root: str) -> tuple:
    """Collect available shards and reconstruct the Titan's keypair.

    Envelope-free (INV-MBR-0/10): the Maker provides raw Shard-1 + the PUBLIC
    Titan address; Shard-3 is discovered ON-CHAIN from the wallet alone (the
    GenesisNFT → Shard-3 memo). Returns (key_bytes, titan_pubkey, keypair, titan_id).
    """
    from titan_hcl.utils.shamir import combine_shares, decrypt_shard3
    from titan_hcl.utils import genesis_discovery

    print_phase("1", "Identity Discovery")
    shards: list = []
    # getattr — callers that build an args namespace (setup_titan, the sovereign
    # engine) may predate these optional fields; default cleanly, never crash.
    titan_pubkey = getattr(args, "titan_pubkey", None) or None
    nft_address = None
    titan_id = getattr(args, "titan_id", None) or "T1"

    genesis_record = os.path.join(install_root, "data", "genesis_record.json")

    # ── Shard 1 (Maker, offline) — raw shard hex (canonical) or legacy envelope ──
    shard1_hex = None
    if args.shard1:
        print("  Loading Maker Shard-1 from command line...")
        shard1_hex = args.shard1
    elif args.shard1_file:
        print(f"  Reading Maker Shard-1 from file: {args.shard1_file}")
        with open(args.shard1_file, "r") as f:
            shard1_hex = f.read().strip()
    if shard1_hex:
        shard1, env_pubkey = _parse_shard1(shard1_hex)
        shards.append(shard1)
        if env_pubkey and not titan_pubkey:
            titan_pubkey = env_pubkey
        print(f"  Shard 1 loaded ({len(shard1)} bytes).")

    # ── Local genesis record — disk-alive Shard-2 + public pointers (NFT addr) ──
    if os.path.exists(genesis_record):
        with open(genesis_record, "r") as f:
            record = json.load(f)
        if not titan_pubkey:
            titan_pubkey = record.get("titan_pubkey") or None
        nft_address = (record.get("nft_address")
                       or record.get("genesis_nft_address") or None)
        shard2_hex = record.get("shard2_hex", "")
        if shard2_hex and len(shards) < 2:
            shards.append(bytes.fromhex(shard2_hex))
            print("  Shard 2 recovered from local genesis record (disk-alive).")

    if not titan_pubkey:
        print("\n  *** No Titan address available. ***")
        print("  Provide --titan-pubkey <address> — the PUBLIC wallet address")
        print("  printed alongside your Shard-1 (it is not a secret).")
        sys.exit(1)
    print(f"  Titan Address: {titan_pubkey}")

    # ── Shard 3 (on-chain, WALLET-ONLY discovery via the GenesisNFT) ──
    discovery: dict = {}
    if len(shards) < 2:
        print("  Discovering Shard 3 on-chain (wallet-only, no envelope)...")
        rpc_url = _config_rpc_url()
        discovery = asyncio.run(genesis_discovery.discover_genesis(
            titan_pubkey, rpc_url,
            das_rpc_url=getattr(args, "das_rpc_url", None) or None,
            nft_address=nft_address))
        enc_hex = discovery.get("shard3_encrypted_hex")
        if not enc_hex and discovery.get("shard3_tx"):
            enc_hex = asyncio.run(genesis_discovery.fetch_shard3_from_tx(
                discovery["shard3_tx"], rpc_url))
        if enc_hex:
            try:
                shards.append(decrypt_shard3(bytes.fromhex(enc_hex), titan_pubkey))
                tx = discovery.get("shard3_tx")
                print(f"  Shard 3 recovered + decrypted"
                      f"{f' (tx {str(tx)[:16]}…)' if tx else ''}.")
            except Exception as e:
                print(f"  [!] Shard 3 decrypt failed: {e}")
        else:
            print("  [!] No on-chain Shard-3 memo found for this wallet.")

    # ── Shard 3 fallback: encrypted in the local record ──
    if len(shards) < 2 and os.path.exists(genesis_record):
        with open(genesis_record, "r") as f:
            record = json.load(f)
        s3_enc_hex = record.get("shard3_encrypted_hex", "")
        if s3_enc_hex:
            shards.append(decrypt_shard3(bytes.fromhex(s3_enc_hex), titan_pubkey))
            print("  Shard 3 recovered from local record.")

    # ── Reconstruct + verify ──
    if len(shards) < 2:
        print(f"\n  *** RESURRECTION FAILED: only {len(shards)} shard(s). ***")
        print("  Need ≥2. Fresh box = Shard-1 (--shard1/-file) + --titan-pubkey,")
        print("  with an on-chain Shard-3 memo on that wallet. Disk-alive = local")
        print("  data/genesis_record.json supplies Shard-2.")
        sys.exit(1)

    print(f"\n  Reconstructing keypair from {len(shards)} shards...")
    key_bytes = combine_shares(shards[:2])

    from solders.keypair import Keypair
    keypair = Keypair.from_bytes(key_bytes)
    recovered_pubkey = str(keypair.pubkey())

    if recovered_pubkey != titan_pubkey:
        print(f"  *** CRITICAL: reconstructed {recovered_pubkey[:16]}… "
              f"≠ expected {titan_pubkey[:16]}… — shard corruption. ABORT. ***")
        sys.exit(1)

    print(f"  Keypair reconstructed + verified: {recovered_pubkey}")
    _report_identity(discovery)
    return key_bytes, recovered_pubkey, keypair, titan_id


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
                       network: str, verify_zk: bool, force: bool,
                       rpc_url: str | None = None):
    """Re-body `data/` — the SOVEREIGN on-chain v=3 chain by DEFAULT (INV-MBR-12,
    no manifest / no local files / Shard-1-only), or the LEGACY `--manifest` path
    ONLY when a manifest is explicitly supplied (non-sovereign fallback)."""
    print_phase("2+3", "Re-Bodying + Re-Hydration")

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

    if manifest_path:
        print("  [legacy] --manifest supplied → NON-sovereign manifest restore.")
        print("  (The sovereign default needs no manifest — the v=3 chain IS the")
        print("   catalogue; INV-MBR-12. Use --manifest only for debug/legacy.)")
        return _restore_via_manifest(
            key_bytes, titan_pubkey, titan_id, install_root=install_root,
            manifest_path=manifest_path, network=network, verify_zk=verify_zk)

    # ── DEFAULT: the sovereign on-chain v=3 chain (no manifest, wallet-only) ──
    print("  Sovereign v=3 chain restore — Shard-1-only, no manifest, no local files.")
    _scripts = os.path.join(_REPO_ROOT, "scripts")
    if _scripts not in sys.path:
        sys.path.append(_scripts)  # append — never shadow the titan_hcl package
    from backup_restore_sovereign import restore_body_from_chain
    result = restore_body_from_chain(
        key_bytes=key_bytes, titan_pubkey=titan_pubkey, titan_id=titan_id,
        install_root=install_root, rpc_url=rpc_url, network=network, commit=True)
    if result.status != "resurrected":
        print(f"\n  *** SOVEREIGN RESTORE HALTED: {result.halt_reason} ***")
        for err in result.errors[-5:]:
            print(f"      {err}")
        sys.exit(1)
    print(f"  Re-hydration OK — {result.events_applied} event(s) restored into "
          f"{data_dir}.")
    return result


def _restore_via_manifest(key_bytes: bytes, titan_pubkey: str, titan_id: str, *,
                           install_root: str, manifest_path: str | None,
                           network: str, verify_zk: bool):
    """LEGACY non-sovereign body-restore via an off-site UnifiedManifest copy.

    Retained ONLY as an explicit `--manifest` fallback; the sovereign v=3 chain
    (the `phase_2_3_restore` default) needs no manifest (INV-MBR-12). This path
    requires the Maker's off-site manifest, so it does NOT satisfy Shard-1-only.
    """
    from titan_hcl.logic.backup_restore import build_arc_to_target, restore_full
    from titan_hcl.utils.arweave_store import ArweaveStore

    data_dir = os.path.join(install_root, "data")
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
                        help="Maker's raw Shard-1 (hex). NO envelope needed — the "
                             "shard carries no locators; the wallet discovers them.")
    parser.add_argument("--shard1-file", type=str,
                        help="Path to a file containing the raw Shard-1 hex.")
    parser.add_argument("--titan-pubkey", type=str, default=None,
                        help="The Titan's PUBLIC wallet address (printed alongside "
                             "Shard-1; not a secret). Required on a fresh box.")
    parser.add_argument("--das-rpc-url", type=str, default=None,
                        help="Optional DAS-capable RPC (Helius/Triton) for "
                             "GenesisNFT identity discovery. Defaults to the "
                             "configured RPC; Shard-3 recovery never needs DAS.")
    parser.add_argument("--shard2-local", action="store_true",
                        help="Disk-alive recovery using local genesis_record.json.")
    parser.add_argument("--manifest", type=str, default=None,
                        help="LEGACY/DEBUG ONLY: Maker-supplied off-site "
                             "UnifiedManifest JSON. Omit it for the sovereign v=3 "
                             "chain restore (the default; needs no manifest).")
    parser.add_argument("--install-root", type=str, default=_REPO_ROOT,
                        help="Target install tree (default: this repo root).")
    parser.add_argument("--titan-id", type=str, default=None,
                        help="Titan id (default: from envelope/record, else T1).")
    parser.add_argument("--network", type=str, default="mainnet",
                        choices=["mainnet", "devnet"],
                        help="Arweave/Solana network (default: mainnet).")
    parser.add_argument("--rpc-url", type=str, default=None,
                        help="Solana RPC URL override for the sovereign chain walk "
                             "(default: the configured RPC).")
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
                      force=args.force, rpc_url=args.rpc_url)
    phase_4_first_breath(key_bytes, titan_pubkey, titan_id,
                         install_root=install_root, verify_only=args.verify_only)


if __name__ == "__main__":
    main()
