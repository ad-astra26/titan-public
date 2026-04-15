"""
titan_plugin/utils/maker_alert.py — Unified Telegram-to-Maker alerting.

Fire-and-forget alerts to Maker's Telegram channel, rate-limited per alert_key
(typically "failure_type" or "module") to avoid alert-storms. Used by:

  • logic/backup.py (backup status) — existing inline impl predates this module
  • modules/spirit_worker.py (meditation Tier-3 alerts — new 2026-04-15)
  • Future: any module that needs Maker attention for rare, critical events

Design:
  • In-memory rate-limit map {alert_key: last_ts} — process-local, no persistence
  • Daemon-thread delivery — never blocks caller event loop
  • Config-controlled via [alerts] section in titan_params.toml (or env overrides)
  • Silent failure (Telegram outage shouldn't cascade into another issue)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Defaults mirror titan_plugin/logic/backup.py values (2026-04-15) — future
# refactor could move these to config. Env-overridable for test/staging.
_DEFAULT_TOKEN = "8531091229:AAElGsqbsLDvDfxaCwMwG1qGdBzyVlksI0c"
_DEFAULT_CHAT_ID = "6345894322"
_DEFAULT_RATE_LIMIT_SECONDS = 3600.0  # 1 alert per (key) per hour

_last_alert_ts: dict[str, float] = {}
_lock = threading.Lock()


def send_maker_alert(
    text: str,
    alert_key: str,
    rate_limit_seconds: float = _DEFAULT_RATE_LIMIT_SECONDS,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> bool:
    """Send a Telegram message to Maker. Rate-limited per alert_key.

    Args:
        text: markdown-formatted message. Telegram has 4096-char limit.
        alert_key: dedup key. Typically "{module}.{failure_type}" e.g.
            "meditation.F1_F2_OVERDUE".
        rate_limit_seconds: min interval between alerts sharing the same key.
        token: override Telegram bot token (else env TITAN_TELEGRAM_BOT_TOKEN
            or module default).
        chat_id: override Telegram chat id (else env TITAN_TELEGRAM_CHAT_ID
            or module default).

    Returns True if the alert was queued for delivery, False if rate-limited
    or misconfigured.
    """
    now = time.time()
    with _lock:
        last = _last_alert_ts.get(alert_key, 0.0)
        if now - last < rate_limit_seconds:
            logger.debug(
                "[MakerAlert] Rate-limited %s (last %.0fs ago, window %.0fs)",
                alert_key, now - last, rate_limit_seconds,
            )
            return False
        _last_alert_ts[alert_key] = now

    bot_token = token or os.environ.get("TITAN_TELEGRAM_BOT_TOKEN") or _DEFAULT_TOKEN
    bot_chat = chat_id or os.environ.get("TITAN_TELEGRAM_CHAT_ID") or _DEFAULT_CHAT_ID
    if not bot_token or not bot_chat:
        logger.warning("[MakerAlert] No Telegram credentials — dropping alert %s", alert_key)
        return False

    # Telegram hard limit: truncate defensively, leave room for ellipsis.
    if len(text) > 4000:
        text = text[:3997] + "..."

    def _post():
        try:
            import httpx
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            r = httpx.post(
                url,
                json={
                    "chat_id": bot_chat,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=10.0,
            )
            if r.status_code != 200:
                logger.warning(
                    "[MakerAlert] Telegram POST non-200: %d %s",
                    r.status_code, r.text[:200],
                )
        except Exception as e:
            logger.debug("[MakerAlert] Delivery error (non-critical) for %s: %s",
                         alert_key, e)

    threading.Thread(target=_post, daemon=True, name=f"maker-alert-{alert_key[:40]}").start()
    return True


def reset_rate_limit(alert_key: Optional[str] = None) -> None:
    """Clear rate-limit state. alert_key=None clears all (test use)."""
    with _lock:
        if alert_key is None:
            _last_alert_ts.clear()
        else:
            _last_alert_ts.pop(alert_key, None)
