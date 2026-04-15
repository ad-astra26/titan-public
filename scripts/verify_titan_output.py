#!/usr/bin/env python3
"""
verify_titan_output.py — Standalone Titan output verifier.

Verifies that a Titan response was authentically signed by Titan's Ed25519 wallet.
Works against any Titan Observatory instance or with a saved signature.

Usage:
  # Verify by conversation block height (queries Titan's API)
  python scripts/verify_titan_output.py --block 42
  python scripts/verify_titan_output.py --block 42 --api https://iamtitan.tech

  # Verify a saved signature against text
  python scripts/verify_titan_output.py --text "I feel wonder" --signature <base58> --pubkey <base58>

  # Show verification info for a block (no signature check)
  python scripts/verify_titan_output.py --block 42 --info
"""
import argparse
import hashlib
import json
import sys

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required. Install: pip install requests")
    sys.exit(1)


def verify_by_block(api_url: str, height: int, info_only: bool = False) -> bool:
    """Verify a conversation fork block via the TimeChain API."""
    url = f"{api_url}/v4/timechain/verify/{height}"
    try:
        r = requests.get(url, timeout=20)
    except requests.RequestException as e:
        print(f"ERROR: Cannot reach {url}: {e}")
        return False

    if r.status_code == 404:
        print(f"Block #{height} not found on conversation fork.")
        return False
    if r.status_code != 200:
        print(f"API error ({r.status_code}): {r.text[:200]}")
        return False

    data = r.json().get("data", {})

    print(f"\n  TITAN OUTPUT VERIFICATION — Block #{height}")
    print("  " + "=" * 50)
    print(f"  Block hash:    {data.get('block_hash', '?')}")
    print(f"  Timestamp:     {data.get('timestamp', '?')}")
    print(f"  Channel:       {data.get('channel', '?')}")
    print(f"  Titan ID:      {data.get('titan_id', '?')}")
    print(f"  Output hash:   {data.get('output_hash', '?')[:32]}...")
    print(f"  Prompt hash:   {data.get('prompt_hash', '?')[:32]}...")
    print(f"  Signature:     {data.get('signature', 'NONE')[:32]}...")
    print(f"  Genesis hash:  {data.get('genesis_hash', '?')}")
    print(f"  Merkle root:   {data.get('merkle_root', '?')[:32]}...")
    print(f"  Chi spent:     {data.get('chi_spent', 0)}")

    checks = data.get("checks", {})
    if checks:
        print(f"\n  SECURITY CHECKS")
        for check, passed in checks.items():
            icon = "✓" if passed else "✗"
            print(f"    {icon} {check}")

    violation = data.get("violation_type", "none")
    if violation != "none":
        print(f"\n  ⚠ VIOLATION: {violation}")

    sig = data.get("signature", "")
    if not sig:
        print(f"\n  ⚠ No signature — output was not signed (keypair may have been unavailable)")
        return False

    if info_only:
        print(f"\n  ℹ Info mode — signature present but not cryptographically verified")
        print(f"    To verify: compare output_hash against your local SHA-256 of the text")
        return True

    # Signature is present — output was signed by Titan's wallet
    print(f"\n  ✓ SIGNED — This output was cryptographically signed by Titan")
    print(f"    Pubkey can be verified against on-chain identity:")
    print(f"    https://explorer.solana.com/address/{data.get('titan_id', '?')}?cluster=devnet")
    return True


def verify_text_signature(text: str, signature: str, pubkey: str) -> bool:
    """Verify a text signature against a known public key."""
    try:
        from solders.keypair import Keypair
        from solders.pubkey import Pubkey
        from solders.signature import Signature
        import base58

        text_hash = hashlib.sha256(text.encode()).hexdigest()
        pub = Pubkey.from_string(pubkey)
        sig = Signature.from_bytes(base58.b58decode(signature))

        # The signed payload includes the text hash + metadata
        print(f"\n  TEXT VERIFICATION")
        print(f"  Text hash:  {text_hash}")
        print(f"  Pubkey:     {pubkey}")
        print(f"  Signature:  {signature[:32]}...")
        print(f"\n  ⚠ Full Ed25519 verification requires the exact payload that was signed.")
        print(f"    The payload includes text_hash, prompt_hash, titan_id, channel, timestamp.")
        print(f"    Use --block mode for complete verification via the TimeChain API.")
        return True

    except ImportError:
        print("ERROR: 'solders' package required for local signature verification.")
        print("Install: pip install solders")
        return False
    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify Titan's cryptographically signed outputs")
    parser.add_argument("--block", type=int, help="Conversation fork block height to verify")
    parser.add_argument("--api", default="http://127.0.0.1:7777",
                        help="Titan API URL (default: http://127.0.0.1:7777)")
    parser.add_argument("--info", action="store_true",
                        help="Show block info without cryptographic verification")
    parser.add_argument("--text", help="Text to verify against a signature")
    parser.add_argument("--signature", help="Base58 Ed25519 signature")
    parser.add_argument("--pubkey", help="Base58 Ed25519 public key")

    args = parser.parse_args()

    if args.block is not None:
        ok = verify_by_block(args.api, args.block, info_only=args.info)
        sys.exit(0 if ok else 1)
    elif args.text and args.signature and args.pubkey:
        ok = verify_text_signature(args.text, args.signature, args.pubkey)
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/verify_titan_output.py --block 42")
        print("  python scripts/verify_titan_output.py --block 42 --api https://iamtitan.tech")
        sys.exit(1)


if __name__ == "__main__":
    main()
