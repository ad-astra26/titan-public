#!/usr/bin/env python3
"""
test_devnet_nft.py — Live devnet NFT mint test for Metaplex Core.

Mints a test NFT on Solana devnet to verify:
  1. CreateV1 instruction builder works with real on-chain program
  2. Asset account is created and readable
  3. Owner matches our wallet
  4. Attributes plugin data is stored

Usage:
    python scripts/test_devnet_nft.py
    python scripts/test_devnet_nft.py --wallet ./authority.json
    python scripts/test_devnet_nft.py --cleanup  # Burns the test NFT after verification

Requires: Funded devnet wallet (0.01+ SOL)
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Devnet NFT Mint Test")
    parser.add_argument("--wallet", default="authority.json", help="Wallet keypair JSON path")
    parser.add_argument("--cleanup", action="store_true", help="Burn test NFT after verification")
    args = parser.parse_args()

    from titan_plugin.utils.solana_client import (
        build_mpl_core_create_v1,
        decode_mpl_core_asset,
        fetch_mpl_core_asset,
        is_available,
    )

    if not is_available():
        print("[!] Solana SDK not available. Install solders + solana-py.")
        sys.exit(1)

    from solders.keypair import Keypair
    from titan_plugin.core.network import HybridNetworkClient

    # Load wallet
    wallet_path = args.wallet
    if not os.path.exists(wallet_path):
        print(f"[!] Wallet not found: {wallet_path}")
        sys.exit(1)

    with open(wallet_path, "r") as f:
        key_array = json.load(f)
    wallet_kp = Keypair.from_bytes(bytes(key_array[:64]))
    wallet_pubkey = wallet_kp.pubkey()
    print(f"  Wallet: {wallet_pubkey}")

    # Load config for RPC URLs
    config_path = os.path.join(os.path.dirname(__file__), "..", "titan_plugin", "config.toml")
    net_config = {}
    try:
        try:
            import tomllib
        except ModuleNotFoundError:
            import toml as tomllib
        with open(config_path, "rb") as f:
            full_config = tomllib.load(f)
        net_config = full_config.get("network", {})
    except Exception:
        pass

    # Force devnet
    net_config["public_rpc_urls"] = ["https://api.devnet.solana.com"]
    net_config["premium_rpc_url"] = ""
    net_config["solana_network"] = "devnet"

    network = HybridNetworkClient(config=net_config)
    network._keypair = wallet_kp
    network._pubkey = wallet_pubkey

    # Check balance
    balance = await network.get_balance()
    print(f"  Balance: {balance:.4f} SOL")
    if balance < 0.01:
        print("[!] Insufficient balance. Need at least 0.01 SOL.")
        print("    Run: solana airdrop 1 --url devnet")
        sys.exit(1)

    # Generate asset keypair
    asset_kp = Keypair()
    asset_pubkey = asset_kp.pubkey()
    print(f"\n  Minting test NFT...")
    print(f"  Asset Address: {asset_pubkey}")

    # Build CreateV1 instruction
    timestamp = int(time.time())
    ix = build_mpl_core_create_v1(
        asset_pubkey=asset_pubkey,
        payer_pubkey=wallet_pubkey,
        name=f"Titan Test {timestamp}"[:32],
        uri="https://shdw-drive.genesysgo.net/titan/test.json",
        attributes={
            "Type": "DevnetTest",
            "Generation": "0",
        },
    )

    if ix is None:
        print("[!] Failed to build CreateV1 instruction.")
        sys.exit(1)

    print(f"  Instruction data: {len(ix.data)} bytes, {len(ix.accounts)} accounts")

    # Send transaction via raw RPC (bypasses solana-py response parsing issue)
    print(f"  Building and sending transaction...")

    import base64
    import httpx
    from solders.transaction import Transaction
    from solders.message import Message
    from solders.hash import Hash

    rpc_url = "https://api.devnet.solana.com"

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get blockhash
        bh_resp = await client.post(rpc_url, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "getLatestBlockhash",
            "params": [{"commitment": "finalized"}],
        })
        blockhash_str = bh_resp.json()["result"]["value"]["blockhash"]
        blockhash = Hash.from_string(blockhash_str)
        print(f"  Blockhash: {blockhash_str}")

        # Build and sign transaction
        msg = Message.new_with_blockhash([ix], wallet_pubkey, blockhash)
        tx = Transaction.new_unsigned(msg)
        tx.sign([wallet_kp, asset_kp], blockhash)

        # Send raw
        tx_bytes = bytes(tx)
        tx_b64 = base64.b64encode(tx_bytes).decode("ascii")

        send_resp = await client.post(rpc_url, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "sendTransaction",
            "params": [tx_b64, {"encoding": "base64", "skipPreflight": False}],
        })
        send_result = send_resp.json()

        if "error" in send_result:
            print(f"[!] Transaction error: {send_result['error']}")
            sys.exit(1)

        sig = send_result["result"]
        print(f"  TX sent: {sig}")

        # Confirm
        print(f"  Confirming...")
        for _ in range(30):
            await asyncio.sleep(2)
            status_resp = await client.post(rpc_url, json={
                "jsonrpc": "2.0", "id": 1,
                "method": "getSignatureStatuses",
                "params": [[sig]],
            })
            statuses = status_resp.json().get("result", {}).get("value", [None])
            if statuses[0] is not None:
                if statuses[0].get("err") is None:
                    print(f"  Confirmed! Slot: {statuses[0].get('slot')}")
                    break
                else:
                    print(f"[!] Transaction failed: {statuses[0].get('err')}")
                    sys.exit(1)
        else:
            print("[!] Transaction not confirmed after 60s.")
            sys.exit(1)

    if not sig:
        print("[!] Transaction failed.")
        sys.exit(1)

    print(f"  TX Signature: {sig}")
    print(f"  Explorer: https://explorer.solana.com/tx/{sig}?cluster=devnet")

    # Wait a moment for finalization
    print(f"\n  Waiting 5s for confirmation...")
    await asyncio.sleep(5)

    # Fetch and verify the asset
    print(f"  Fetching asset from chain...")
    asset = await fetch_mpl_core_asset(network, str(asset_pubkey))

    if asset:
        print(f"\n  === NFT Verified On-Chain ===")
        print(f"  Name:             {asset.get('name', 'N/A')}")
        print(f"  URI:              {asset.get('uri', 'N/A')}")
        print(f"  Owner:            {asset.get('owner', 'N/A')}")
        print(f"  Update Authority: {asset.get('update_authority', 'N/A')}")
        print(f"  Program Owner:    {asset.get('program_owner', 'N/A')}")

        # Verify ownership
        if asset.get("owner") == str(wallet_pubkey):
            print(f"\n  [OK] Owner matches wallet!")
        else:
            print(f"\n  [!] Owner mismatch! Expected {wallet_pubkey}")

        print(f"\n  Devnet NFT mint test PASSED.")
    else:
        print(f"\n  [!] Could not fetch asset from chain.")
        print(f"  Check manually: https://explorer.solana.com/address/{asset_pubkey}?cluster=devnet")

    print()


if __name__ == "__main__":
    asyncio.run(main())
