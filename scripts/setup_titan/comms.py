"""Phase 5 — comm channel selection.

Per locked W2 substrate inventory:
- Telegram is the **guaranteed** comm channel — always prompted, always required.
- X (Twitter) is **opt-in** — needs a twitterapi.io key + a static Webshare proxy URL.
- Observatory is **opt-in** — heavier (nginx + TLS + domain). v0.0.1 prints config
  instructions rather than auto-editing `titan_hcl/config.toml` (out of scope here;
  config.toml mutation lands in W1.f `setup_titan config`).

All credentials land in `~/.titan/secrets.toml` via `inference.upsert_secret`
(flat-TOML upsert, no library dep). Format is validated at prompt time.
"""
from __future__ import annotations

import re

from .inference import upsert_secret
from .preflight import Result
from .ui import cprint

# BotFather emits tokens shaped like `8531091229:AAElGsqbsLDvDfxaCwMwG1qGdBzyVlksI0c`
# (numeric chat id : alphanumeric secret). Strict enough to catch fat-fingered keys.
TELEGRAM_TOKEN_RE = re.compile(r"^\d{8,}:[A-Za-z0-9_-]{30,}$")

# twitterapi.io API keys are UUID-style: 8-4-4-4-12 hex chars.
TWITTERAPI_KEY_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# Webshare static-proxy URLs: http://user:pass@host:port/  — host is IP or domain.
WEBSHARE_URL_RE = re.compile(r"^https?://[^:]+:[^@]+@[\w.-]+:\d+/?$")


def _prompt_yes_no(question: str, *, default_yes: bool) -> bool:
    suffix = " [Y/n]" if default_yes else " [y/N]"
    try:
        ans = input(f"  {question}{suffix}: ").strip().lower()
    except EOFError:
        raise SystemExit(f"setup_titan: stdin closed during prompt: {question!r}")
    if not ans:
        return default_yes
    return ans in ("y", "yes")


def _prompt_until(question: str, pattern: re.Pattern[str], hint: str) -> str:
    while True:
        try:
            ans = input(f"  {question}: ").strip()
        except EOFError:
            raise SystemExit(f"setup_titan: stdin closed during prompt: {question!r}")
        if pattern.match(ans):
            return ans
        cprint(f"    {hint}", role="warning")


def run_comms_phase(*, default: bool) -> list[Result]:
    """Phase 5 body — prompt for Telegram (required), X (opt-in), Observatory (opt-in).

    `--default`: still asks. Telegram cannot be auto-defaulted (no key to detect
    locally); X / Observatory in --default skip unless the user explicitly opts in.
    """
    results: list[Result] = []

    # ── Telegram (required) ────────────────────────────────────────────────
    cprint("  Telegram is the guaranteed comm channel — Titan will always be reachable via /chat there.",
           role="text_strong")
    cprint("  Need a bot token from @BotFather (https://t.me/BotFather → /newbot).",
           role="text_muted")
    token = _prompt_until(
        "Telegram bot token",
        TELEGRAM_TOKEN_RE,
        "expected format: '<numeric_id>:<alphanumeric>' (≥30 char secret)",
    )
    upsert_secret("channels", "telegram_bot_token", token)
    results.append(Result("telegram", "ok", "bot token → secrets.toml [channels]"))

    # ── X (opt-in) ─────────────────────────────────────────────────────────
    if default or not _prompt_yes_no("Enable X (Twitter) posting? Needs twitterapi.io + Webshare.",
                                      default_yes=False):
        results.append(Result("x_social", "ok", "skipped (opt-in)"))
    else:
        cprint("  twitterapi.io key (from https://twitterapi.io dashboard).", role="text_muted")
        key = _prompt_until(
            "twitterapi.io API key",
            TWITTERAPI_KEY_RE,
            "expected UUID format: 8-4-4-4-12 hex chars",
        )
        upsert_secret("stealth_sage", "twitterapi_io_key", key)
        cprint("  Webshare static-proxy URL (from https://www.webshare.io/proxy → static IPs).",
               role="text_muted")
        url = _prompt_until(
            "Webshare static URL",
            WEBSHARE_URL_RE,
            "expected format: http://user:pass@host:port/",
        )
        upsert_secret("twitter_social", "webshare_static_url", url)
        results.append(Result("x_social", "ok",
                              "twitterapi.io → [stealth_sage], Webshare → [twitter_social]"))

    # ── Observatory (opt-in, info-only in v0.0.1) ─────────────────────────
    if default or not _prompt_yes_no("Enable Observatory web UI? (Heavier: nginx + TLS + domain.)",
                                      default_yes=False):
        results.append(Result("observatory", "ok", "skipped (opt-in)"))
    else:
        cprint("  Observatory enablement is config-driven. After install, edit:",
               role="text_strong")
        cprint("    titan_hcl/config.toml → [observatory] enabled = true", role="text_muted")
        cprint("  …then arrange your reverse-proxy / TLS / domain. The Observatory listens on :3000.",
               role="text_muted")
        cprint("  (Automated Observatory wiring lands in a future setup_titan release — W1.f.)",
               role="warning")
        results.append(Result("observatory", "warn",
                              "opt-in noted; manual config.toml edit + reverse-proxy still required"))

    return results
