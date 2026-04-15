#!/usr/bin/env python3
"""
status_report.py — Titan Pre-Recovery Health Check.

Checks the state of all recovery components before running resurrection.
Reports which shards are available and what recovery paths are viable.

Usage:
    python scripts/status_report.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "OK" if condition else "MISSING"
    icon = "  [+]" if condition else "  [-]"
    suffix = f" — {detail}" if detail else ""
    print(f"{icon} {label}: {status}{suffix}")
    return condition


def main():
    print("\n" + "=" * 60)
    print("    TITAN STATUS REPORT — Sovereignty Health Check")
    print("=" * 60 + "\n")

    has_brain = False
    has_soul = False
    has_keypair_enc = False
    has_keypair_plain = False
    has_genesis = False
    has_config = False
    has_shard2 = False
    has_solana_sdk = False
    has_shadow_drive = False

    # ─── Local State ───
    print("  Local State:")
    has_brain = check(
        "Cognee DB",
        os.path.exists("data/cognee_db"),
        "data/cognee_db/",
    )
    has_soul = check(
        "Soul Document",
        os.path.exists("titan.md"),
        "titan.md",
    )
    has_keypair_enc = check(
        "Hardware-Bound Keypair",
        os.path.exists("data/soul_keypair.enc"),
        "data/soul_keypair.enc",
    )
    has_keypair_plain = check(
        "Plaintext Keypair",
        os.path.exists("authority.json"),
        "authority.json",
    )
    has_genesis = check(
        "Genesis Record",
        os.path.exists("data/genesis_record.json"),
        "data/genesis_record.json",
    )

    soul_state_path = "data/soul_state.json"
    has_soul_state = check(
        "Soul State",
        os.path.exists(soul_state_path),
        soul_state_path,
    )

    if has_soul_state:
        try:
            with open(soul_state_path, "r") as f:
                ss = json.load(f)
            print(f"       Generation: {ss.get('current_gen', '?')}")
            print(f"       NFT: {ss.get('nft_address', 'not minted')}")
        except Exception:
            pass

    if has_genesis:
        try:
            with open("data/genesis_record.json", "r") as f:
                gr = json.load(f)
            print(f"       Titan Pubkey: {gr.get('titan_pubkey', '?')}")
            print(f"       Genesis TX: {gr.get('genesis_tx', 'not recorded')}")
            has_shard2 = bool(gr.get("shard2_hex"))
            check("Shard 2 (in genesis record)", has_shard2)
            check("Shard 3 encrypted (in genesis record)", bool(gr.get("shard3_encrypted_hex")))
        except Exception:
            pass

    # ─── Configuration ───
    print("\n  Configuration:")
    config_path = os.path.join("titan_plugin", "config.toml")
    has_config = check("config.toml", os.path.exists(config_path), config_path)

    if has_config:
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                import toml as tomllib
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)

            net = cfg.get("network", {})
            check("Maker Pubkey", bool(net.get("maker_pubkey")))
            check("Premium RPC", bool(net.get("premium_rpc_url")))

            mem = cfg.get("memory_and_storage", {})
            has_shadow_drive = bool(mem.get("shadow_drive_account"))
            check("Shadow Drive Account", has_shadow_drive)

            api = cfg.get("api", {})
            check("Observatory API", api.get("enabled", False), f"port {api.get('port', '?')}")
        except Exception as e:
            print(f"  [!] Config parse error: {e}")

    # ─── Dependencies ───
    print("\n  Dependencies:")
    try:
        from solders.keypair import Keypair
        has_solana_sdk = True
        check("Solana SDK (solders)", True)
    except ImportError:
        check("Solana SDK (solders)", False)

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        check("Cryptography (AES-GCM)", True)
    except ImportError:
        check("Cryptography (AES-GCM)", False)

    try:
        import httpx
        check("httpx", True)
    except ImportError:
        check("httpx", False)

    try:
        import cognee
        check("Cognee", True)
    except ImportError:
        check("Cognee", False)

    # ─── Hardware Fingerprint ───
    print("\n  Hardware Fingerprint Sources:")
    check("/etc/machine-id", os.path.exists("/etc/machine-id"))
    check("/sys/class/dmi/id/product_uuid", os.path.exists("/sys/class/dmi/id/product_uuid"))
    check("Hardware Salt", os.path.exists("data/hw_salt.bin"), "data/hw_salt.bin")

    # ─── Recovery Paths ───
    print("\n" + "-" * 60)
    print("  Recovery Path Analysis:")
    print("-" * 60)

    if has_keypair_enc or has_keypair_plain:
        print("  [ACTIVE] Warm Reboot: Keypair available locally.")
        if has_brain:
            print("           Brain intact — normal boot possible.")
        else:
            print("           Brain MISSING — needs Shadow Drive download.")
    else:
        print("  [DEAD]   No local keypair — full resurrection required.")

    print()
    path_count = 0

    if has_shard2:
        print("  [VIABLE] Path 1+2: Maker Shard + Local Shard 2")
        path_count += 1
    if has_genesis and has_solana_sdk:
        print("  [VIABLE] Path 1+3: Maker Shard + On-Chain Shard 3")
        path_count += 1
    if has_shard2 and has_solana_sdk:
        print("  [VIABLE] Path 2+3: Local Shard 2 + On-Chain Shard 3 (no Maker needed)")
        path_count += 1
    if has_shadow_drive and has_solana_sdk:
        print("  [VIABLE] Path 2+3 (remote): Shadow Drive Shard 2 + On-Chain Shard 3")
        path_count += 1

    if path_count == 0:
        print("  [NONE]   No viable recovery paths available!")
        print("           Need: Maker Shard (--shard1) + Solana SDK + config.toml")

    print(f"\n  Total viable paths: {path_count}")

    # ─── Summary ───
    all_healthy = has_brain and has_soul and (has_keypair_enc or has_keypair_plain) and has_config
    print(f"\n{'=' * 60}")
    if all_healthy:
        print("  STATUS: SOVEREIGN — All systems operational")
    elif has_keypair_enc or has_keypair_plain:
        print("  STATUS: DEGRADED — Keypair present but brain may need restoration")
    else:
        print("  STATUS: LIMBO — Resurrection required")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
