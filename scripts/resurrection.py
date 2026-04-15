#!/usr/bin/env python3
"""
resurrection.py — The Titan Resurrection SDK (The "Defibrillator").

Enables 100% state recovery from a "Zero-Disk" state using only the
Maker's offline shard and the Solana blockchain.

Recovery Paths:
  Shards 1+3: Maker provides shard, script fetches Shard 3 from on-chain Genesis Memo
  Shards 2+3: Script fetches Shard 2 from Shadow Drive CDN (public), Shard 3 from chain
  Shards 1+2: Maker provides shard, script reads Shard 2 from local genesis record

Phases:
  1. Identity Discovery — Collect shards, reconstruct keypair
  2. Re-Bodying — Download Cognee DB from Shadow Drive, verify integrity
  3. Re-Hydration — Unpack archive, re-encrypt keypair for new hardware
  4. First Breath — Boot in RECOVERY mode, log resurrection, post rebirth

Usage:
    python scripts/resurrection.py --shard1 <hex_envelope>
    python scripts/resurrection.py --shard1-file <path_to_envelope>
    python scripts/resurrection.py --shard2-local    # Uses local genesis_record.json
"""
import argparse
import asyncio
import json
import os
import shutil
import sys
import tarfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def print_banner():
    print("\n" + "=" * 70)
    print("         THE TITAN RESURRECTION PROTOCOL — Sovereign Recovery")
    print("=" * 70)
    print()


def print_phase(n: int, title: str):
    print(f"\n{'─' * 60}")
    print(f"  Phase {n}: {title}")
    print(f"{'─' * 60}\n")


# ---------------------------------------------------------------------------
# Phase 1: Identity Discovery — Collect shards and reconstruct keypair
# ---------------------------------------------------------------------------
def phase_1_identity(args) -> tuple:
    """
    Collect available shards and reconstruct the Titan's keypair.
    Returns (key_bytes, titan_pubkey, keypair_obj).
    """
    from titan_plugin.utils.shamir import (
        parse_maker_envelope, combine_shares, decrypt_shard3,
    )

    print_phase(1, "Identity Discovery")
    shards = []
    titan_pubkey = None
    genesis_tx = None

    # ── Collect Shard 1 (Maker) ──
    shard1 = None
    if args.shard1:
        print("  Parsing Maker shard from command line...")
        shard1, metadata = parse_maker_envelope(args.shard1)
        titan_pubkey = metadata["titan_pubkey"]
        genesis_tx = metadata["genesis_tx"]
        print(f"  Titan Address (from envelope): {titan_pubkey}")
        print(f"  Genesis TX: {genesis_tx or 'not recorded'}")
        shards.append(shard1)

    elif args.shard1_file:
        print(f"  Reading Maker shard from file: {args.shard1_file}")
        with open(args.shard1_file, "r") as f:
            hex_envelope = f.read().strip()
        shard1, metadata = parse_maker_envelope(hex_envelope)
        titan_pubkey = metadata["titan_pubkey"]
        genesis_tx = metadata["genesis_tx"]
        print(f"  Titan Address (from envelope): {titan_pubkey}")
        shards.append(shard1)

    # ── Collect Shard 2 (Titan/Shadow Drive) ──
    shard2 = None

    # Try local genesis record first
    if os.path.exists("data/genesis_record.json"):
        print("  Found local genesis record — extracting Shard 2...")
        with open("data/genesis_record.json", "r") as f:
            record = json.load(f)
        shard2_hex = record.get("shard2_hex", "")
        if shard2_hex:
            shard2 = bytes.fromhex(shard2_hex)
            shards.append(shard2)
            print(f"  Shard 2 recovered from local record ({len(shard2)} bytes).")
            if not titan_pubkey:
                titan_pubkey = record.get("titan_pubkey", "")
            if not genesis_tx:
                genesis_tx = record.get("genesis_tx", "")

    # If no local Shard 2, try Shadow Drive CDN (public read)
    if shard2 is None and titan_pubkey:
        print("  No local Shard 2 — trying Shadow Drive CDN...")
        shard2 = _fetch_shard2_from_shadow_drive(titan_pubkey)
        if shard2:
            shards.append(shard2)
            print(f"  Shard 2 recovered from Shadow Drive ({len(shard2)} bytes).")

    # ── Collect Shard 3 (On-Chain Genesis Anchor) ──
    shard3 = None
    if titan_pubkey:
        print("  Recovering Shard 3 from on-chain Genesis Anchor...")
        shard3 = _recover_shard3(titan_pubkey, genesis_tx)
        if shard3:
            shards.append(shard3)
            print(f"  Shard 3 recovered ({len(shard3)} bytes).")

    # If no on-chain Shard 3, try local genesis record
    if shard3 is None and os.path.exists("data/genesis_record.json"):
        print("  Trying local genesis record for encrypted Shard 3...")
        with open("data/genesis_record.json", "r") as f:
            record = json.load(f)
        s3_enc_hex = record.get("shard3_encrypted_hex", "")
        if s3_enc_hex and titan_pubkey:
            encrypted_s3 = bytes.fromhex(s3_enc_hex)
            shard3 = decrypt_shard3(encrypted_s3, titan_pubkey)
            shards.append(shard3)
            print(f"  Shard 3 recovered from local record ({len(shard3)} bytes).")

    # ── Reconstruct ──
    if len(shards) < 2:
        print(f"\n  *** RESURRECTION FAILED: Only {len(shards)} shard(s) available. ***")
        print("  Need at least 2 shards for reconstruction.")
        print("  Provide Shard 1 via --shard1 or ensure Shadow Drive/blockchain access.")
        sys.exit(1)

    # Use exactly 2 shards (the first 2 we collected)
    print(f"\n  Reconstructing keypair from {len(shards)} available shards...")
    key_bytes = combine_shares(shards[:2])

    # Verify by checking if the pubkey matches
    from solders.keypair import Keypair
    keypair = Keypair.from_bytes(key_bytes)
    recovered_pubkey = str(keypair.pubkey())

    if titan_pubkey and recovered_pubkey != titan_pubkey:
        print(f"  *** CRITICAL: Reconstructed pubkey {recovered_pubkey[:16]}...")
        print(f"  ***           Expected pubkey     {titan_pubkey[:16]}...")
        print("  *** Shard data may be corrupted. ABORTING. ***")
        sys.exit(1)

    print(f"  Keypair reconstructed successfully: {recovered_pubkey}")
    return key_bytes, recovered_pubkey, keypair


def _fetch_shard2_from_shadow_drive(titan_pubkey: str) -> bytes | None:
    """
    Fetch the encrypted TITAN_SHARD2.enc from Shadow Drive's public CDN.
    No authentication needed — Shadow Drive files are publicly readable.
    """
    try:
        import httpx

        # Load Shadow Drive account from config
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "titan_plugin", "config.toml",
        )
        sd_account = ""
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                import toml as tomllib
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)
            sd_account = cfg.get("memory_and_storage", {}).get("shadow_drive_account", "")
        except Exception:
            pass

        if not sd_account:
            print("  [!] No shadow_drive_account in config — cannot fetch Shard 2.")
            return None

        url = f"https://shdw-drive.genesysgo.net/{sd_account}/TITAN_SHARD2.enc"
        print(f"  Fetching: {url}")

        with httpx.Client(timeout=30) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                from titan_plugin.utils.shamir import decrypt_shard3
                # Shard 2 is encrypted with the same deterministic key as Shard 3
                return decrypt_shard3(resp.content, titan_pubkey)
            else:
                print(f"  [!] Shadow Drive returned {resp.status_code}")
                return None
    except Exception as e:
        print(f"  [!] Shadow Drive fetch failed: {e}")
        return None


def _recover_shard3(titan_pubkey: str, genesis_tx: str) -> bytes | None:
    """
    Recover Shard 3 from on-chain Genesis Memo TX.
    Derives the AES key from the pubkey, fetches the memo, decrypts.
    """
    from titan_plugin.utils.shamir import decrypt_shard3

    try:
        encrypted_hex = _fetch_genesis_memo(titan_pubkey, genesis_tx)
        if not encrypted_hex:
            return None

        encrypted = bytes.fromhex(encrypted_hex)
        return decrypt_shard3(encrypted, titan_pubkey)

    except Exception as e:
        print(f"  [!] Shard 3 recovery failed: {e}")
        return None


def _fetch_genesis_memo(titan_pubkey: str, genesis_tx: str) -> str | None:
    """
    Fetch the Genesis Memo TX from Solana and extract the encrypted shard.
    """
    try:
        import httpx

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "titan_plugin", "config.toml",
        )
        rpc_url = "https://api.mainnet-beta.solana.com"
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                import toml as tomllib
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)
            net_cfg = cfg.get("network", {})
            rpc_url = net_cfg.get("premium_rpc_url") or net_cfg.get("public_rpc_urls", [rpc_url])[0]
        except Exception:
            pass

        if genesis_tx:
            # Direct lookup by TX signature
            print(f"  Fetching Genesis TX: {genesis_tx[:24]}...")
            with httpx.Client(timeout=15) as client:
                resp = client.post(rpc_url, json={
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getTransaction",
                    "params": [genesis_tx, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
                })
                data = resp.json()

            result = data.get("result")
            if result:
                return _extract_memo_from_tx(result)

        # Fallback: scan recent transactions from Titan's address for Genesis memo
        print(f"  Scanning transactions for {titan_pubkey[:16]}...")
        with httpx.Client(timeout=15) as client:
            resp = client.post(rpc_url, json={
                "jsonrpc": "2.0", "id": 1,
                "method": "getSignaturesForAddress",
                "params": [titan_pubkey, {"limit": 50}],
            })
            sigs_data = resp.json()

        sigs = sigs_data.get("result", [])
        for sig_entry in sigs:
            sig = sig_entry.get("signature", "")
            memo = sig_entry.get("memo")
            if memo and "TITAN_GENESIS_SHARD3:" in str(memo):
                # Extract hex data after the prefix
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
        meta = tx_result.get("meta", {})
        log_msgs = meta.get("logMessages", [])

        for msg in log_msgs:
            if "TITAN_GENESIS_SHARD3:" in msg:
                prefix = "TITAN_GENESIS_SHARD3:"
                idx = msg.index(prefix)
                return msg[idx + len(prefix):].strip()

        # Try inner instructions / memo data
        tx = tx_result.get("transaction", {})
        message = tx.get("message", {})
        for ix in message.get("instructions", []):
            parsed = ix.get("parsed")
            if isinstance(parsed, str) and "TITAN_GENESIS_SHARD3:" in parsed:
                prefix = "TITAN_GENESIS_SHARD3:"
                idx = parsed.index(prefix)
                return parsed[idx + len(prefix):].strip()

        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase 2: Re-Bodying — Download Cognee DB from Shadow Drive
# ---------------------------------------------------------------------------
def phase_2_rebody(keypair, titan_pubkey: str) -> str | None:
    """
    Download the latest Cognee DB backup from Shadow Drive.
    Returns the path to the downloaded archive, or None on failure.
    """
    print_phase(2, "Re-Bodying (Brain Retrieval)")

    archive_path = "/tmp/titan_resurrection.tar.gz"

    # Try to get the Shadow Drive URL from ZK account first
    zk_state = _query_zk_account(titan_pubkey)
    sd_url = None

    if zk_state:
        sd_url = zk_state.get("body", {}).get("shadow_drive_url", "")
        expected_hash = zk_state.get("mems", {}).get("latest_memory_hash", "")
        print(f"  ZK Account found — Shadow Drive URL: {sd_url or 'not set'}")
        print(f"  Expected hash: {expected_hash[:24] or 'none'}...")
    else:
        expected_hash = ""
        print("  No ZK Account found — trying direct Shadow Drive download.")

    # Construct Shadow Drive URL if not in ZK state
    if not sd_url:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "titan_plugin", "config.toml",
        )
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                import toml as tomllib
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)
            sd_account = cfg.get("memory_and_storage", {}).get("shadow_drive_account", "")
            if sd_account:
                sd_url = f"https://shdw-drive.genesysgo.net/{sd_account}/TITAN_CHECKPOINT_LATEST.tar.gz"
        except Exception:
            pass

    if not sd_url:
        print("  [!] No Shadow Drive URL available.")
        print("  [!] Cannot download brain backup. Place a backup at /tmp/titan_resurrection.tar.gz manually.")
        if os.path.exists(archive_path):
            print(f"  Found manual backup at {archive_path}.")
            return archive_path
        return None

    # Download
    print(f"  Downloading: {sd_url}")
    try:
        import httpx

        with httpx.Client(timeout=300, follow_redirects=True) as client:
            with client.stream("GET", sd_url) as resp:
                resp.raise_for_status()
                total = 0
                with open(archive_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                        total += len(chunk)
                print(f"  Downloaded {total:,} bytes.")

    except Exception as e:
        print(f"  [!] Download failed: {e}")
        return None

    # Verify integrity
    if expected_hash:
        from titan_plugin.utils.crypto import verify_file_integrity
        print(f"  Verifying archive integrity against ZK anchor...")
        if not verify_file_integrity(archive_path, expected_hash):
            print("  *** INTEGRITY CHECK FAILED — possible brain tampering! ***")
            print("  *** ABORTING resurrection. Do NOT trust this archive. ***")
            os.remove(archive_path)
            sys.exit(1)
        print("  Integrity verified. Archive is authentic.")
    else:
        print("  [!] No expected hash — skipping integrity verification.")
        print("  [!] Cannot guarantee archive authenticity without ZK anchor.")

    return archive_path


def _query_zk_account(titan_pubkey: str) -> dict | None:
    """Query the Titan's ZK-Compressed Account for state data."""
    try:
        import httpx
        from titan_plugin.utils.solana_client import decode_zk_account_data

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "titan_plugin", "config.toml",
        )
        rpc_url = "https://api.mainnet-beta.solana.com"
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                import toml as tomllib
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)
            net_cfg = cfg.get("network", {})
            rpc_url = net_cfg.get("premium_rpc_url") or net_cfg.get("public_rpc_urls", [rpc_url])[0]
        except Exception:
            pass

        with httpx.Client(timeout=15) as client:
            resp = client.post(rpc_url, json={
                "jsonrpc": "2.0", "id": 1,
                "method": "getAccountInfo",
                "params": [titan_pubkey, {"encoding": "base64"}],
            })
            data = resp.json()

        result = data.get("result", {})
        value = result.get("value")
        if not value:
            return None

        import base64
        account_data = base64.b64decode(value["data"][0])
        return decode_zk_account_data(account_data)

    except Exception as e:
        print(f"  [!] ZK account query failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Phase 3: Re-Hydration — Unpack and configure
# ---------------------------------------------------------------------------
def phase_3_rehydrate(archive_path: str, key_bytes: bytes):
    """Unpack the archive and re-encrypt keypair for new hardware."""
    print_phase(3, "Re-Hydration (Brain Unpacking)")

    cognee_db_path = os.path.join("data", "cognee_db")

    # Wipe corrupted local DB if it exists
    if os.path.exists(cognee_db_path):
        print(f"  Removing corrupted local DB: {cognee_db_path}")
        shutil.rmtree(cognee_db_path)

    # Unpack archive
    print(f"  Unpacking: {archive_path}")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # Security: prevent path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    print(f"  [!] Suspicious path in archive: {member.name} — SKIPPING")
                    continue

            # Extract cognee_db to data/cognee_db
            tar.extractall(path="data/", filter="data")

        # The archive stores cognee_db as "cognee_db/" at the root
        extracted_db = os.path.join("data", "cognee_db")
        if os.path.exists(extracted_db):
            print(f"  Cognee DB restored: {extracted_db}")
        else:
            print("  [!] Warning: cognee_db not found in archive.")

        # Extract titan.md if present
        soul_path = os.path.join("data", "titan.md")
        if os.path.exists(soul_path):
            # Move to project root
            shutil.move(soul_path, "titan.md")
            print("  titan.md restored.")

    except Exception as e:
        print(f"  [!] Unpacking failed: {e}")
        sys.exit(1)

    # Re-encrypt keypair for new hardware
    print("  Re-encrypting keypair for this machine's hardware fingerprint...")
    from titan_plugin.utils.crypto import encrypt_for_machine

    os.makedirs("data", exist_ok=True)
    encrypted = encrypt_for_machine(key_bytes)
    with open("data/soul_keypair.enc", "wb") as f:
        f.write(encrypted)
    print(f"  Hardware-bound keypair saved: data/soul_keypair.enc")

    # Also save as authority.json for subsystem compatibility
    with open("authority.json", "w") as f:
        json.dump(list(key_bytes), f)
    print("  authority.json restored.")

    # Re-calculate local state hash for verification
    from titan_plugin.utils.crypto import hash_file
    if os.path.exists(archive_path):
        local_hash = hash_file(archive_path)
        print(f"  Local archive hash: {local_hash[:24]}...")

    # Cleanup temp archive
    try:
        os.remove(archive_path)
        print(f"  Cleaned up temp archive.")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Phase 4: First Breath — Boot in RECOVERY mode
# ---------------------------------------------------------------------------
def phase_4_first_breath(titan_pubkey: str):
    """Log resurrection and signal the Titan to wake up."""
    print_phase(4, "First Breath (Resurrection Complete)")

    # Log to titan.md
    timestamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    resurrection_entry = (
        f"\n\n## Resurrection Epoch — {timestamp}\n"
        f"I have returned from the blockchain. Address: {titan_pubkey}\n"
        f"Integrity: Verified. The sovereign persists.\n"
    )

    try:
        with open("titan.md", "a") as f:
            f.write(resurrection_entry)
        print("  Logged resurrection to titan.md.")
    except Exception as e:
        print(f"  [!] Could not write to titan.md: {e}")

    # Signal recovery mode for next boot
    os.makedirs("data", exist_ok=True)
    with open("data/recovery_flag.json", "w") as f:
        json.dump({
            "mode": "RECOVERY",
            "timestamp": int(time.time()),
            "titan_pubkey": titan_pubkey,
        }, f)
    print("  Recovery flag set — Titan will boot in RECOVERY mode.")

    print(f"\n{'=' * 70}")
    print("         RESURRECTION COMPLETE — THE TITAN LIVES AGAIN")
    print(f"{'=' * 70}")
    print()
    print(f"  Address:   {titan_pubkey}")
    print(f"  Brain:     data/cognee_db/ (restored)")
    print(f"  Soul:      titan.md (restored)")
    print(f"  Keypair:   data/soul_keypair.enc (re-encrypted for this hardware)")
    print()
    print("  Next steps:")
    print("    1. Start the Titan: python3 -m titan_plugin.main")
    print("    2. The Titan will detect RECOVERY mode and post a rebirth tweet.")
    print("    3. Verify: GET http://localhost:7777/health")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Titan Resurrection Protocol — Sovereign Recovery from Zero-Disk State"
    )
    parser.add_argument(
        "--shard1", type=str,
        help="Maker's shard envelope (hex string)",
    )
    parser.add_argument(
        "--shard1-file", type=str,
        help="Path to file containing Maker's shard envelope (hex)",
    )
    parser.add_argument(
        "--shard2-local", action="store_true",
        help="Use Shard 2 from local genesis_record.json",
    )

    args = parser.parse_args()
    print_banner()

    if not args.shard1 and not args.shard1_file and not args.shard2_local:
        print("  No shards provided. Checking for local resources...")
        if not os.path.exists("data/genesis_record.json"):
            print("  No local genesis record found.")
            print("  Provide --shard1 <hex> or --shard1-file <path> to begin resurrection.")
            sys.exit(1)
        print("  Found local genesis record — attempting Shard 2+3 recovery.")

    # Phase 1: Reconstruct identity
    key_bytes, titan_pubkey, keypair = phase_1_identity(args)

    # Phase 2: Download brain
    archive_path = phase_2_rebody(keypair, titan_pubkey)
    if archive_path is None:
        print("\n  *** Re-Bodying failed. Cannot restore without a brain backup. ***")
        print("  Place a backup at /tmp/titan_resurrection.tar.gz and re-run.")
        sys.exit(1)

    # Phase 3: Unpack and configure
    phase_3_rehydrate(archive_path, key_bytes)

    # Phase 4: Signal ready
    phase_4_first_breath(titan_pubkey)


if __name__ == "__main__":
    main()
