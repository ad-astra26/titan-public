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
from .alerts import HealthMonitor, resolve_telegram_creds
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


def main(argv: list | None = None) -> int:
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
    args = ap.parse_args(argv)

    install_root = Path(args.install_root).resolve()
    titan_id = args.titan_id or resolve_titan_id(install_root)
    dist = Path(args.dist_dir).resolve() if args.dist_dir else None

    ctx = Context(install_root=install_root, titan_id=titan_id,
                  api_base=args.api_base, token=_read_token(), dist_dir=dist)
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
