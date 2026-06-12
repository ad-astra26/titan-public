"""Decoupled degraded-health alert push (W8, decision #13/#14).

The Console Agent's killer feature: because it runs in its own crash domain
(stdlib-only, not Guardian-supervised), it can tell the owner *why their Titan
is down* — over Telegram — precisely when the Titan (and its own command bot)
can't. This module is the outbound push half:

  - resolve_telegram_creds() reads the bot token + chat id straight from
    config.toml/secrets.toml with stdlib tomllib (NO titan_hcl import — the
    whole point is to survive a broken Titan).
  - HealthMonitor polls titan_status() on a timer and pushes a Telegram alert
    on a state TRANSITION (up→down with the why + journal tail; down→up
    recovery). Edge-triggered + deduped so a flapping Titan doesn't spam.

The inbound `/status` command stays with the in-Titan CommandRegistry (works
when the Titan is up); the agent does not poll getUpdates (that would steal
updates from the Titan's own bot on the shared token). Push-only here.
"""
from __future__ import annotations

import json
import threading
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from .context import Context
from .titan_status import titan_status


def _read_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def resolve_telegram_creds(ctx: Context) -> tuple[Optional[str], Optional[str]]:
    """(bot_token, chat_id) from config.toml ⊕ secrets.toml ⊕ [console] override.

    Precedence (high→low): secrets.toml > config.toml; within each, a [console]
    section override beats the shared [channels]/[maker_relationship] keys so an
    owner can route console alerts to a different chat without touching the
    Titan's own bot config.
    """
    config = _read_toml(ctx.install_root / "titan_hcl" / "config.toml")
    secrets = _read_toml(_secrets_path(ctx))

    def pick(section: str, *keys: str) -> Optional[str]:
        for src in (secrets, config):           # secrets wins
            sec = src.get(section)
            if isinstance(sec, dict):
                for k in keys:
                    v = sec.get(k)
                    if v:
                        return str(v)
        return None

    token = (pick("console", "telegram_bot_token", "bot_token")
             or pick("channels", "telegram_bot_token", "bot_token"))
    chat_id = (pick("console", "alert_chat_id", "maker_telegram_id")
               or pick("maker_relationship", "maker_telegram_id"))
    return token, chat_id


def resolve_internal_key(ctx: Context) -> Optional[str]:
    """api.internal_key from secrets.toml ⊕ config.toml (secrets wins).

    This is the owner auth the console sends as X-Titan-Internal-Key to chat with
    its own Titan (pitch_chat owner bypass). setup_titan generates it into
    ~/.titan/secrets.toml [api] at install.
    """
    config = _read_toml(ctx.install_root / "titan_hcl" / "config.toml")
    secrets = _read_toml(_secrets_path(ctx))
    for src in (secrets, config):                 # secrets wins
        api = src.get("api")
        if isinstance(api, dict):
            v = api.get("internal_key")
            if v:
                return str(v)
    return None


def _secrets_path(ctx: Context) -> Path:
    override = getattr(ctx, "secrets_path", None)
    import os
    return Path(override) if override else Path(os.path.expanduser("~/.titan/secrets.toml"))


def send_telegram(token: str, chat_id: str, text: str, *, opener=None) -> bool:
    """Outbound-only Telegram sendMessage (urllib, stdlib). True on ok=true."""
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id, "text": text,
        "parse_mode": "Markdown", "disable_web_page_preview": "true",
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    _open = opener or (lambda r, timeout: urllib.request.urlopen(r, timeout=timeout))
    try:
        with _open(req, timeout=15) as resp:
            return bool(json.loads(resp.read().decode()).get("ok"))
    except Exception:
        return False


def format_alert(ctx: Context, status: dict) -> str:
    """Render a down/recovery alert from a titan_status() dict."""
    tid = ctx.titan_id
    if status.get("up"):
        return f"✅ *{tid} recovered* — service active and /health responding."
    why = status.get("why_down") or "unknown"
    lines = [f"🔴 *{tid} is DOWN*", f"why: `{why}`"]
    tail = status.get("journal_tail") or []
    if tail:
        snippet = "\n".join(tail[-3:])
        lines.append(f"last log:\n```\n{snippet}\n```")
    lines.append("TC² console is still up — open it for System/Settings.")
    return "\n".join(lines)


class HealthMonitor:
    """Edge-triggered Telegram pusher. Poll check_once() on a timer (or let
    start() run its own daemon thread)."""

    def __init__(self, ctx: Context, *, interval_s: float = 60.0,
                 sender=send_telegram):
        self.ctx = ctx
        self.interval_s = interval_s
        self._sender = sender
        self._last_up: Optional[bool] = None     # None = unknown (no alert on boot)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def check_once(self) -> dict:
        """One poll. Pushes an alert iff up/down state CHANGED. Returns a dict
        describing what happened (for tests/logs)."""
        status = titan_status(self.ctx)
        up = bool(status.get("up"))
        prev = self._last_up
        self._last_up = up
        if prev is None or prev == up:
            return {"transition": None, "up": up}

        # App event channel: enqueue a health event to every paired phone (AG-EVT-3).
        # Independent of Telegram + best-effort — the monitor must never crash the agent.
        try:
            from . import events, pairing
            for did in pairing.registered_device_ids(self.ctx):
                events.enqueue(
                    self.ctx, did, type="health",
                    payload={"up": up, "why": status.get("why_down"),
                             "text": format_alert(self.ctx, status)},
                    urgency=("high" if not up else "normal"))
        except Exception:
            pass

        token, chat_id = resolve_telegram_creds(self.ctx)
        sent = False
        if token and chat_id:
            sent = self._sender(token, chat_id, format_alert(self.ctx, status))
        return {"transition": "down" if not up else "up", "up": up,
                "alert_sent": sent, "creds": bool(token and chat_id)}

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.check_once()
            except Exception:
                pass  # the monitor must never crash the agent

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        # prime baseline without alerting, then start polling
        try:
            self._last_up = bool(titan_status(self.ctx).get("up"))
        except Exception:
            self._last_up = None
        self._thread = threading.Thread(target=self._loop, name="tc2-health-monitor",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
