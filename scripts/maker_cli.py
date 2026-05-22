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
    python scripts/maker_cli.py force-post --titan T1 "wonder at my own architecture"

Requires:
    - For Ed25519 endpoints (directive / inject-memory / divine-inspiration /
      audit): Solana keypair JSON file (same key configured as maker_pubkey
      in config.toml).
    - For internal-key endpoints (force-post): no keypair needed — reads
      the per-Titan `api.internal_key` from the host's titan_hcl/config.toml
      (T1) or ssh-fetches it from /home/antigravity/projects/titan{,3}/titan_hcl/config.toml
      on 10.135.0.6 for T2/T3 (auth.py:146 internal-key bypass).
    - pip install httpx solders
"""
import argparse
import json
import os
import subprocess
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
    internal_key: str | None = None,
) -> dict:
    """Send an authenticated or public request to the Observatory.

    Auth precedence (matches auth.py:146 + auth.py:156):
      1. If `internal_key` is provided, send via X-Titan-Internal-Key header.
      2. Else if `keypair` is provided, sign as Ed25519 (X-Titan-Signature /
         X-Titan-Timestamp).
      3. Else no auth headers (public endpoints only).
    """
    url = f"{base_url}{path}"
    body_str = json.dumps(body) if body else ""
    headers = {"Content-Type": "application/json"}

    if internal_key:
        headers["X-Titan-Internal-Key"] = internal_key
    elif keypair is not None:
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


# ── Per-Titan resolution helpers (force-post + future per-Titan commands) ──

_TITAN_TARGETS = {
    "T1": {
        "url": "http://localhost:7777",
        "config_path": "titan_hcl/config.toml",   # local
        "ssh_host": None,
    },
    "T2": {
        "url": "http://10.135.0.6:7777",
        "config_path": "/home/antigravity/projects/titan/titan_hcl/config.toml",
        "ssh_host": "root@10.135.0.6",
    },
    "T3": {
        "url": "http://10.135.0.6:7778",
        "config_path": "/home/antigravity/projects/titan3/titan_hcl/config.toml",
        "ssh_host": "root@10.135.0.6",
    },
}


def resolve_titan_internal_key(titan_id: str) -> str:
    """Read api.internal_key from the target Titan's config.toml.

    T1 reads local file. T2/T3 ssh-fetch from 10.135.0.6 (matches
    `feedback_t2t3_deployment_via_git_pull` deploy model — config.toml
    is the canonical per-Titan config carrier).

    Raises ValueError on missing key / unreachable target.
    """
    if titan_id not in _TITAN_TARGETS:
        raise ValueError(
            f"Unknown titan_id={titan_id!r} — expected one of "
            f"{sorted(_TITAN_TARGETS)}")
    target = _TITAN_TARGETS[titan_id]
    config_path = target["config_path"]
    ssh_host = target["ssh_host"]

    if ssh_host is None:
        # Local read for T1.
        try:
            with open(config_path) as f:
                contents = f.read()
        except FileNotFoundError as exc:
            raise ValueError(
                f"{titan_id}: config.toml not found at {config_path}"
            ) from exc
    else:
        # Remote read via ssh for T2/T3.
        try:
            result = subprocess.run(
                ["ssh", ssh_host, f"cat {config_path}"],
                capture_output=True, text=True, timeout=10, check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"{titan_id}: ssh cat failed ({ssh_host}:{config_path}): "
                f"{exc.stderr.strip()}") from exc
        contents = result.stdout

    # Extract internal_key from [api] section. Tolerant of leading whitespace,
    # quoting style, comments. We do this manually rather than importing
    # tomllib because tomllib needs bytes and we don't want a roundtrip.
    in_api_section = False
    for raw_line in contents.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_api_section = line == "[api]"
            continue
        if in_api_section and line.startswith("internal_key"):
            # internal_key = "value"
            _, _, value_part = line.partition("=")
            value = value_part.strip().strip('"').strip("'")
            if value:
                return value

    raise ValueError(
        f"{titan_id}: api.internal_key not found in {config_path}")


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


def cmd_force_post(args, keypair: Keypair | None, base_url: str):
    """Force an X post via /maker/x-force-post using per-Titan internal_key.

    The endpoint at titan_hcl/api/maker.py:147 queues a high-significance
    catalyst into social_worker's PostDispatchOrchestrator. Auth via the
    X-Titan-Internal-Key bypass (auth.py:146) — no Ed25519 keypair needed
    since the per-Titan config.toml [api].internal_key is the sanctioned
    operator-tool credential.
    """
    try:
        internal_key = resolve_titan_internal_key(args.titan)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    target_url = _TITAN_TARGETS[args.titan]["url"]
    body = {
        "topic": args.topic,
        "text_hint": args.text_hint or "",
        "catalyst_type": args.catalyst_type,
    }
    print(f"[{args.titan}] POST {target_url}/maker/x-force-post  "
          f"topic={args.topic!r} catalyst_type={args.catalyst_type}")
    result = make_request(
        target_url, "POST", "/maker/x-force-post",
        keypair=None, body=body, internal_key=internal_key)
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

    p_fp = sub.add_parser(
        "force-post",
        help="Force an X post via /maker/x-force-post (per-Titan internal_key auth)")
    p_fp.add_argument(
        "topic",
        help="Subject of the post, e.g. \"wonder at my own architecture\"")
    p_fp.add_argument(
        "--titan", "-t", choices=sorted(_TITAN_TARGETS), required=True,
        help="Which Titan to post from (T1=local, T2/T3=10.135.0.6)")
    p_fp.add_argument(
        "--text-hint", default="",
        help="Optional seed content. Empty = Titan composes freely.")
    p_fp.add_argument(
        "--catalyst-type", default="maker_force",
        help="Catalyst label for telemetry / post_type selection (default: maker_force)")

    args = parser.parse_args()

    # Load keypair for Ed25519-authenticated commands. force-post uses the
    # X-Titan-Internal-Key bypass instead — no keypair needed.
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
        "force-post": cmd_force_post,
    }

    commands[args.command](args, keypair, args.url)


if __name__ == "__main__":
    main()
