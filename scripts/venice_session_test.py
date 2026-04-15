#!/usr/bin/env python3
"""
Quick test for Venice session token authentication with auto-refresh.

Usage:
    # Test with both cookies (auto-refresh):
    python scripts/venice_session_test.py --session "eyJ..." --client "eyJ..."

    # Test with session token only (no refresh):
    python scripts/venice_session_test.py "eyJhbGciOi..."

    # Test from config.toml:
    python scripts/venice_session_test.py

    # Decode a JWT to inspect claims:
    python scripts/venice_session_test.py --decode "eyJ..."
"""
import asyncio
import sys
import pathlib
import argparse

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def decode_and_print(token: str) -> None:
    """Decode and print JWT claims for inspection."""
    from titan_plugin.inference.venice_session import _decode_jwt_payload
    import time

    claims = _decode_jwt_payload(token)
    if not claims:
        print("Failed to decode JWT. Make sure it's a valid token.")
        return

    print("JWT Claims:")
    for k, v in sorted(claims.items()):
        label = k
        if k == "exp":
            ttl = v - time.time()
            label = f"exp (expires in {ttl:.0f}s)"
        elif k == "iat":
            age = time.time() - v
            label = f"iat (issued {age:.0f}s ago)"
        print(f"  {label}: {v}")


async def main():
    parser = argparse.ArgumentParser(description="Test Venice session auth")
    parser.add_argument("token", nargs="?", help="Session token (__session cookie)")
    parser.add_argument("--session", "-s", help="Session token (__session cookie)")
    parser.add_argument("--client", "-c", help="Client cookie (__client cookie)")
    parser.add_argument("--decode", "-d", help="Just decode a JWT and show claims")
    args = parser.parse_args()

    if args.decode:
        decode_and_print(args.decode)
        return

    session_token = args.session or args.token or ""
    client_cookie = args.client or ""

    # Fallback: read from config.toml
    if not session_token:
        import re
        config_path = PROJECT_ROOT / "titan_plugin" / "config.toml"
        content = config_path.read_text()

        match = re.search(r'^venice_session_token\s*=\s*"([^"]*)"', content, re.MULTILINE)
        if match:
            session_token = match.group(1)

        if not client_cookie:
            match = re.search(r'^venice_client_cookie\s*=\s*"([^"]*)"', content, re.MULTILINE)
            if match:
                client_cookie = match.group(1)

    if not session_token:
        print("No session token provided.")
        print()
        print("How to get your Venice Pro session cookies:")
        print("  1. Log into venice.ai in your browser")
        print("  2. Open DevTools (F12) → Application tab → Cookies → venice.ai")
        print("  3. Copy the '__session' cookie value")
        print("  4. Copy the '__client' cookie value (for auto-refresh)")
        print()
        print("Usage:")
        print('  python scripts/venice_session_test.py --session "eyJ..." --client "eyJ..."')
        print()
        print("Or paste into config.toml:")
        print('  venice_session_token = "..."')
        print('  venice_client_cookie = "..."')
        sys.exit(1)

    print(f"Session token: {len(session_token)} chars, prefix: {session_token[:20]}...")
    if client_cookie:
        print(f"Client cookie: {len(client_cookie)} chars (auto-refresh enabled)")
    else:
        print("Client cookie: NOT SET (no auto-refresh)")
    print()

    # Decode and show claims
    print("--- Token Claims ---")
    decode_and_print(session_token)
    print()

    # Test connection
    from titan_plugin.inference.venice_session import VeniceSessionClient

    client = VeniceSessionClient(
        session_token=session_token,
        client_cookie=client_cookie,
    )

    print("--- Connection Test ---")
    print(f"Auto-refresh: {client.can_auto_refresh}")
    success, message = await client.test_connection()

    if success:
        print(f"SUCCESS: {message}")
        print()
        print(f"Stats: {client.stats}")
        print()
        if client_cookie:
            print("Both cookies work! Add to config.toml:")
            print(f'  venice_session_token = "{session_token}"')
            print(f'  venice_client_cookie = "{client_cookie}"')
            print('  inference_provider = "venice_session"')
        else:
            print("Session token works! But without __client cookie, it will expire in ~60s.")
            print("Add the __client cookie for auto-refresh.")
    else:
        print(f"FAILED: {message}")
        print()
        if "401" in message or "expired" in message.lower():
            if client_cookie:
                print("Token expired but auto-refresh was attempted. Check logs above.")
                print("The __client cookie may also be expired. Re-copy both from browser.")
            else:
                print("Token expired. The __session cookie only lasts ~60 seconds.")
                print("Add the __client cookie for auto-refresh, or re-copy __session.")
        elif "402" in message:
            print("402 — Venice didn't recognize this as session auth.")
            print("Make sure you copied __session cookie, not an API key.")


if __name__ == "__main__":
    asyncio.run(main())
