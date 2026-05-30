"""Phase 5 — comm channel selection.

Per locked W2 substrate inventory:
- Telegram is the **guaranteed** comm channel — always prompted, always required.
- X (Twitter) is **opt-in** — needs a twitterapi.io key + a static Webshare proxy URL.

The owner UI (TC² Console Agent) is NOT a comm channel and NOT opt-in: it installs
unconditionally in its own `phase_console`. The heavy Observatory web UI no longer
ships to users (2026-05-30) — TC² replaces it.

All credentials land in `~/.titan/secrets.toml` via `inference.upsert_secret`
(flat-TOML upsert, no library dep). Format is validated at prompt time.
"""
from __future__ import annotations

import re

from .inference import upsert_secret
from .preflight import Result
from .prompts import Prompter, StdinPrompter
from .ui import cprint

# BotFather emits tokens shaped like `8531091229:AAElGsqbsLDvDfxaCwMwG1qGdBzyVlksI0c`
# (numeric chat id : alphanumeric secret). Strict enough to catch fat-fingered keys.
TELEGRAM_TOKEN_RE = re.compile(r"^\d{8,}:[A-Za-z0-9_-]{30,}$")

# twitterapi.io API keys are UUID-style: 8-4-4-4-12 hex chars.
TWITTERAPI_KEY_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# Webshare static-proxy URLs: http://user:pass@host:port/  — host is IP or domain.
WEBSHARE_URL_RE = re.compile(r"^https?://[^:]+:[^@]+@[\w.-]+:\d+/?$")


def _matcher(pattern: re.Pattern[str]):
    """Adapt a compiled regex to a Prompter.until validate callable."""
    return lambda s: bool(pattern.match(s))


def run_comms_phase(*, default: bool, state: dict | None = None,
                    prompter: Prompter | None = None) -> list[Result]:
    """Phase 5 body — prompt for Telegram (required) and X (opt-in).

    `--default`: still asks. Telegram cannot be auto-defaulted (no key to detect
    locally); X in --default skips unless the user explicitly opts in.

    All input goes through ``prompter`` (CLI stdin by default; the TUI seeds a
    ScriptedPrompter). Prompt keys: ``telegram_bot_token`` (until), ``enable_x``
    (confirm), ``twitterapi_key`` / ``webshare_url`` (until).
    """
    state = state if state is not None else {}
    prompter = prompter or StdinPrompter()
    results: list[Result] = []

    # ── Telegram (required) ────────────────────────────────────────────────
    cprint("  Telegram is the guaranteed comm channel — Titan will always be reachable via /chat there.",
           role="text_strong")
    cprint("  Need a bot token from @BotFather (https://t.me/BotFather → /newbot).",
           role="text_muted")
    token = prompter.until(
        "telegram_bot_token", "Telegram bot token",
        validate=_matcher(TELEGRAM_TOKEN_RE),
        hint="expected format: '<numeric_id>:<alphanumeric>' (≥30 char secret)",
        secret=True)
    upsert_secret("channels", "telegram_bot_token", token)
    results.append(Result("telegram", "ok", "bot token → secrets.toml [channels]"))

    # ── X (opt-in) ─────────────────────────────────────────────────────────
    if default or not prompter.confirm(
            "enable_x", "Enable X (Twitter) posting? Needs twitterapi.io + Webshare.",
            default_yes=False):
        results.append(Result("x_social", "ok", "skipped (opt-in)"))
    else:
        cprint("  twitterapi.io key (from https://twitterapi.io dashboard).", role="text_muted")
        key = prompter.until(
            "twitterapi_key", "twitterapi.io API key",
            validate=_matcher(TWITTERAPI_KEY_RE),
            hint="expected UUID format: 8-4-4-4-12 hex chars", secret=True)
        upsert_secret("stealth_sage", "twitterapi_io_key", key)
        cprint("  Webshare static-proxy URL (from https://www.webshare.io/proxy → static IPs).",
               role="text_muted")
        url = prompter.until(
            "webshare_url", "Webshare static URL",
            validate=_matcher(WEBSHARE_URL_RE),
            hint="expected format: http://user:pass@host:port/", secret=True)
        upsert_secret("twitter_social", "webshare_static_url", url)
        results.append(Result("x_social", "ok",
                              "twitterapi.io → [stealth_sage], Webshare → [twitter_social]"))

    return results
