#!/usr/bin/env python3
"""
maker_cli.py — Maker Console CLI for the Sovereign Observatory.

Signs requests with the Maker's Ed25519 keypair and sends them to
the Titan's authenticated API endpoints.

Usage:
    python scripts/maker_cli.py directive "New prime directive text"
    python scripts/maker_cli.py inject-memory "The project deadline moved to March 15th"
    python scripts/maker_cli.py inject-memory "Critical: new wallet address is ABC..." --weight 8.0
    python scripts/maker_cli.py divine-inspiration
    python scripts/maker_cli.py audit
    python scripts/maker_cli.py status

Requires:
    - Solana keypair JSON file (same key configured as maker_pubkey in config.toml)
    - pip install httpx solders
"""
import argparse
import json
import sys
import time

try:
    import httpx
    from solders.keypair import Keypair  # type: ignore
except ImportError:
    print("Missing dependencies. Install: pip install httpx solders")
    sys.exit(1)


def load_keypair(path: str) -> Keypair:
    """Load a Solana keypair from a JSON file."""
    with open(path) as f:
        secret = json.load(f)
    return Keypair.from_bytes(bytes(secret[:64]))


def sign_request(keypair: Keypair, timestamp: str, body: str) -> str:
    """Sign '{timestamp}:{body}' with Ed25519 and return base58 signature."""
    from solders.signature import Signature  # type: ignore

    message = f"{timestamp}:{body}".encode("utf-8")
    sig = keypair.sign_message(message)
    return str(sig)


def make_request(
    base_url: str,
    method: str,
    path: str,
    keypair: Keypair | None,
    body: dict | None = None,
) -> dict:
    """Send an authenticated or public request to the Observatory."""
    url = f"{base_url}{path}"
    body_str = json.dumps(body) if body else ""
    headers = {"Content-Type": "application/json"}

    # Add auth headers for Maker endpoints
    if keypair is not None:
        ts = str(time.time())
        sig = sign_request(keypair, ts, body_str)
        headers["X-Titan-Signature"] = sig
        headers["X-Titan-Timestamp"] = ts

    with httpx.Client(timeout=15) as client:
        if method == "GET":
            resp = client.get(url, headers=headers)
        else:
            resp = client.post(url, headers=headers, content=body_str)

    return resp.json()


def cmd_directive(args, keypair: Keypair, base_url: str):
    """Submit a new Prime Directive."""
    memo_data = args.text
    # Also sign the directive text itself for the soul
    ts = str(time.time())
    memo_sig = sign_request(keypair, ts, memo_data)

    result = make_request(base_url, "POST", "/maker/directive", keypair, {
        "memo_data": memo_data,
        "memo_signature": memo_sig,
    })
    print(json.dumps(result, indent=2))


def cmd_divine_inspiration(args, keypair: Keypair, base_url: str):
    """Trigger Divine Inspiration."""
    result = make_request(base_url, "POST", "/maker/divine-inspiration", keypair, {})
    print(json.dumps(result, indent=2))


def cmd_audit(args, keypair: Keypair, base_url: str):
    """Fetch full sovereignty audit."""
    result = make_request(base_url, "GET", "/maker/audit", keypair)
    print(json.dumps(result, indent=2))


def cmd_inject_memory(args, keypair: Keypair, base_url: str):
    """Inject a high-weight memory directly into the Titan's persistent graph."""
    result = make_request(base_url, "POST", "/maker/inject-memory", keypair, {
        "text": args.text,
        "weight": args.weight,
    })
    print(json.dumps(result, indent=2))


def cmd_status(args, keypair: Keypair | None, base_url: str):
    """Fetch public status (no auth needed)."""
    result = make_request(base_url, "GET", "/status", None)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Maker Console CLI for Titan Sovereign Observatory"
    )
    parser.add_argument(
        "--keypair", "-k",
        default="./authority.json",
        help="Path to Solana keypair JSON (default: ./authority.json)",
    )
    parser.add_argument(
        "--url", "-u",
        default="http://localhost:7777",
        help="Observatory base URL (default: http://localhost:7777)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_dir = sub.add_parser("directive", help="Submit a new Prime Directive")
    p_dir.add_argument("text", help="Directive text")

    p_mem = sub.add_parser("inject-memory", help="Inject a memory into the Titan's persistent graph")
    p_mem.add_argument("text", help="Memory text to inject")
    p_mem.add_argument("--weight", "-w", type=float, default=5.0, help="Memory weight 1.0-10.0 (default: 5.0)")

    sub.add_parser("divine-inspiration", help="Trigger Divine Inspiration")
    sub.add_parser("audit", help="Fetch full sovereignty audit")
    sub.add_parser("status", help="Fetch public status (no auth)")

    args = parser.parse_args()

    # Load keypair for authenticated commands
    keypair = None
    if args.command in ("directive", "divine-inspiration", "audit", "inject-memory"):
        try:
            keypair = load_keypair(args.keypair)
            print(f"Loaded keypair: {keypair.pubkey()}")
        except Exception as e:
            print(f"Failed to load keypair from {args.keypair}: {e}")
            sys.exit(1)

    commands = {
        "directive": cmd_directive,
        "inject-memory": cmd_inject_memory,
        "divine-inspiration": cmd_divine_inspiration,
        "audit": cmd_audit,
        "status": cmd_status,
    }

    commands[args.command](args, keypair, args.url)


if __name__ == "__main__":
    main()
