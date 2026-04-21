"""
Maker alert helper — backup-worker → Maker via Telegram.

Lightweight outbound-only notifier for rFP_backup_worker Phase 6. Uses the
Telegram Bot API HTTP endpoint directly (no python-telegram-bot dependency)
to avoid coupling with the command-bot lifecycle.

Anti-spam: at-most-daily per (class, titan_id) via dedup state file.

Usage:
    from titan_plugin.utils.maker_notify import notify_maker
    notify_maker("runway_red", "T1", "Irys ⚠ red tier — 0.003 SOL")
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEDUP_STATE_PATH = "data/backup_telegram_dedup.json"
_DEFAULT_COOLDOWN_S = 86400  # 1 day


def _get_telegram_creds() -> tuple[Optional[str], Optional[str]]:
    """Resolve (bot_token, maker_chat_id) from config."""
    try:
        from titan_plugin.config_loader import load_titan_config
        cfg = load_titan_config()
        ch = cfg.get("channels", {}).get("telegram", {}) or {}
        mr = cfg.get("maker_relationship", {}) or {}
        # Try both common keys
        token = (ch.get("telegram_bot_token")
                 or ch.get("bot_token")
                 or cfg.get("telegram_bot_token"))
        chat_id = mr.get("maker_telegram_id")
        return token, chat_id
    except Exception as e:
        logger.debug("[MakerNotify] Creds lookup failed: %s", e)
        return None, None


def _load_dedup() -> dict:
    try:
        if os.path.exists(_DEDUP_STATE_PATH):
            with open(_DEDUP_STATE_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_dedup(state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_DEDUP_STATE_PATH) or ".", exist_ok=True)
        tmp = _DEDUP_STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, _DEDUP_STATE_PATH)
    except Exception as e:
        logger.debug("[MakerNotify] Dedup save failed: %s", e)


def notify_maker(alert_class: str, titan_id: str, text: str,
                 cooldown_s: int = _DEFAULT_COOLDOWN_S,
                 force: bool = False) -> bool:
    """Send a Telegram message to Maker if not in cooldown for (class, titan_id).

    Args:
        alert_class: dedup key like "backup_success_daily" / "runway_red" /
                      "backup_failure" — same class fires at most once per
                      cooldown_s window per titan_id.
        titan_id: "T1"/"T2"/"T3".
        text: the message body (markdown allowed).
        cooldown_s: dedup window (default 1 day).
        force: bypass dedup (used for critical failures).

    Returns True if message was sent (reached Telegram API ok=true), else False.
    """
    token, chat_id = _get_telegram_creds()
    if not token or not chat_id:
        logger.debug("[MakerNotify] No Telegram creds — skipping")
        return False

    now = time.time()
    key = f"{alert_class}:{titan_id}"
    dedup = _load_dedup()

    if not force:
        last = float(dedup.get(key, 0))
        if now - last < cooldown_s:
            logger.debug(
                "[MakerNotify] Dedup hit: %s fired %.1fh ago (cooldown=%.1fh)",
                key, (now - last) / 3600.0, cooldown_s / 3600.0)
            return False

    # Send via Telegram Bot API
    try:
        import urllib.request as _ur
        import urllib.parse as _up
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        # Include titan_id prefix in body for Maker clarity
        body = f"[{titan_id}] {text}"
        payload = _up.urlencode({
            "chat_id": chat_id,
            "text": body,
            "parse_mode": "Markdown",
            "disable_web_page_preview": "true",
        }).encode("utf-8")
        req = _ur.Request(
            url, data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with _ur.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        if data.get("ok"):
            dedup[key] = now
            _save_dedup(dedup)
            logger.info("[MakerNotify] Sent: class=%s titan=%s (%d chars)",
                        alert_class, titan_id, len(body))
            return True
        logger.warning("[MakerNotify] Telegram API !ok: %s",
                       data.get("description", "?"))
        return False
    except Exception as e:
        logger.warning("[MakerNotify] Send failed (class=%s): %s",
                       alert_class, e)
        return False


def format_backup_success(backup_type: str, size_mb: float,
                            arweave_tx: Optional[str] = None,
                            duration_s: Optional[float] = None) -> str:
    parts = [f"✅ *{backup_type.title()} backup OK* — {size_mb:.1f} MB"]
    if arweave_tx and not arweave_tx.startswith("devnet"):
        short = arweave_tx[:20]
        parts.append(f"tx: `{short}...`")
        parts.append(f"https://arweave.net/{arweave_tx}")
    elif arweave_tx:
        parts.append("(devnet)")
    if duration_s:
        parts.append(f"duration {duration_s:.0f}s")
    return " · ".join(parts)


def format_backup_failure(backup_type: str, error: str,
                           step: Optional[str] = None) -> str:
    parts = [f"❌ *{backup_type.title()} backup FAILED*"]
    if step:
        parts.append(f"step: `{step}`")
    # Truncate long errors
    parts.append(f"err: `{error[:200]}`")
    return "\n".join(parts)


def format_runway_alert(tier: str, irys_sol: float, days: float) -> str:
    emoji = {"green": "✅", "yellow": "ℹ️", "orange": "⚠️", "red": "🔴"}.get(tier, "ℹ️")
    header = f"{emoji} *Irys runway — tier {tier.upper()}*"
    body = f"{irys_sol:.4f} SOL deposit ≈ {days:.1f} days"
    cta = ""
    if tier == "red":
        cta = "\n\n*TOP UP NOW* — `node scripts/irys_upload.js fund <lamports> data/titan_identity_keypair.json https://api.mainnet-beta.solana.com`"
    elif tier == "orange":
        cta = "\n\nTop up today if possible."
    elif tier == "yellow":
        cta = "\n\nTop up within 30 days."
    return f"{header}\n{body}{cta}"
