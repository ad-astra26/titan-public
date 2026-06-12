"""`python -m titan_console` — run the Console Agent.

Independent of the Titan process. Reads an optional mutation token from
~/.titan/console_token (so restart/clean/config-set require auth when exposed
beyond localhost); readouts stay open on the bind address.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .agent import make_server
from .alerts import HealthMonitor, resolve_internal_key, resolve_telegram_creds
from .context import Context, resolve_titan_id


def _read_token() -> str | None:
    p = Path(os.path.expanduser("~/.titan/console_token"))
    try:
        if p.exists():
            tok = p.read_text().strip()
            return tok or None
    except OSError:
        pass
    return None


def _pair_cli(argv: list) -> int:
    """`python -m titan_console pair` — headless operator pairing (no browser).

    Mints a QR pairing session, prints the payload to paste into the phone's
    "Paste pairing code" dialog, waits for the phone to submit, then confirms the
    mutual code-match. Pure-stdlib; calls the pairing functions directly (no HTTP).
    """
    import json as _json
    import time as _time

    from . import pairing
    from .context import Context, resolve_titan_id

    ap = argparse.ArgumentParser(prog="titan_console pair",
                                 description="Pair a phone with this Titan (headless).")
    ap.add_argument("--install-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--titan-id", default=None)
    ap.add_argument("--public-url", default=None,
                    help="Endpoint URL to embed in the QR (e.g. the Tailscale URL). "
                         "Omit for a localhost/emulator test.")
    ap.add_argument("--ttl", type=int, default=300, help="Seconds the code stays valid.")
    args = ap.parse_args(argv)

    install_root = Path(args.install_root).resolve()
    titan_id = args.titan_id or resolve_titan_id(install_root)
    ctx = Context(install_root=install_root, titan_id=titan_id)

    status, payload = pairing.mint_pairing(ctx, ttl=args.ttl, public_url=args.public_url)
    if status != 200:
        print(f"[pair] mint failed: {payload}", file=sys.stderr)
        return 1
    token = payload["pairing_token"]
    print("\n── Pair your phone ─────────────────────────────────────────")
    print("On the phone: Titan → “Paste pairing code”, then paste EXACTLY:\n")
    print(_json.dumps(payload))
    try:
        import qrcode  # optional: a scannable QR if the operator has it installed
        qr = qrcode.QRCode(border=1)
        qr.add_data(_json.dumps(payload))
        qr.print_ascii(invert=True)
    except Exception:
        print("\n(install `qrcode` for a scannable QR; paste works without it)")

    print(f"\nWaiting up to {args.ttl}s for the phone to submit…", flush=True)
    deadline = _time.time() + args.ttl
    while _time.time() < deadline:
        _, st = pairing.pair_status(ctx, token)
        if st.get("state") == "submitted":
            phone_code = st.get("code6", "")
            print(f"\nPhone submitted. Console computed code: {phone_code}")
            entered = input("Type the 6-digit code shown ON YOUR PHONE to confirm: ").strip()
            cstatus, cres = pairing.confirm_device(ctx, token, entered)
            if cstatus == 200:
                print(f"✓ Paired “{cres.get('label', 'phone')}”. The phone can now talk to your Titan.")
                return 0
            print(f"✗ Confirm failed: {cres.get('error', cstatus)}", file=sys.stderr)
            return 1
        if st.get("state") == "expired":
            print("✗ Pairing expired before the phone submitted.", file=sys.stderr)
            return 1
        _time.sleep(2)
    print("✗ Timed out waiting for the phone.", file=sys.stderr)
    return 1


def main(argv: list | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    if raw and raw[0] == "pair":
        return _pair_cli(raw[1:])

    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(prog="titan_console",
                                 description="Titan Command Center — Console Agent")
    ap.add_argument("--version", action="version", version=f"titan-console {__version__}")
    ap.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1).")
    ap.add_argument("--port", type=int, default=7799, help="Bind port (default 7799).")
    ap.add_argument("--install-root", default=str(repo_root), help="Titan install tree.")
    ap.add_argument("--titan-id", default=None, help="Titan id (default: resolved from data/).")
    ap.add_argument("--api-base", default="http://127.0.0.1:7777", help="api_hcl base URL.")
    ap.add_argument("--dist-dir", default=None, help="Built SPA dir (titan-console/dist).")
    ap.add_argument("--alert-interval", type=float, default=60.0,
                    help="Seconds between health polls for Telegram alerts (default 60).")
    ap.add_argument("--no-alerts", action="store_true",
                    help="Disable the degraded-health Telegram pusher.")
    ap.add_argument("--dev", action="store_true",
                    help="Enable dev-only endpoints (/dev/latest.apk, /console/dev/log).")
    args = ap.parse_args(argv)

    install_root = Path(args.install_root).resolve()
    titan_id = args.titan_id or resolve_titan_id(install_root)
    dist = Path(args.dist_dir).resolve() if args.dist_dir else None

    ctx = Context(install_root=install_root, titan_id=titan_id,
                  api_base=args.api_base, token=_read_token(), dist_dir=dist,
                  dev_enabled=args.dev)
    # Owner chat auth (X-Titan-Internal-Key → pitch_chat owner bypass): load the
    # api internal_key so the owner can chat with their Titan without Privy/wallet.
    ctx.internal_key = resolve_internal_key(ctx)
    httpd = make_server(ctx, host=args.host, port=args.port)

    monitor = None
    if not args.no_alerts:
        token, chat_id = resolve_telegram_creds(ctx)
        if token and chat_id:
            monitor = HealthMonitor(ctx, interval_s=args.alert_interval)
            monitor.start()

    print(f"[titan-console] {__version__} on http://{args.host}:{args.port} "
          f"(titan={titan_id}, api={args.api_base}, "
          f"auth={'on' if ctx.token else 'off'}, "
          f"alerts={'on' if monitor else 'off'})", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if monitor:
            monitor.stop()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
