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
    """Phase 5 body — prompt for Telegram (required), X (opt-in), Observatory (opt-in).

    `--default`: still asks. Telegram cannot be auto-defaulted (no key to detect
    locally); X / Observatory in --default skip unless the user explicitly opts in.
    Opting into Observatory sets ``state['observatory_enabled']`` so the later
    Observatory phase fetches + runs the prebuilt bundle.

    All input goes through ``prompter`` (CLI stdin by default; the TUI seeds a
    ScriptedPrompter). Prompt keys: ``telegram_bot_token`` (until), ``enable_x``
    (confirm), ``twitterapi_key`` / ``webshare_url`` (until), ``enable_observatory``
    (confirm).
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

    # ── Observatory (opt-in — installs the prebuilt bundle after boot) ────
    if default or not prompter.confirm(
            "enable_observatory",
            "Enable the Observatory web UI? (prebuilt; runs on localhost:3000)",
            default_yes=False):
        state["observatory_enabled"] = False
        results.append(Result("observatory", "ok", "skipped (opt-in)"))
    else:
        state["observatory_enabled"] = True
        cprint("  Observatory will be fetched as a prebuilt bundle from the release and started",
               role="text_strong")
        cprint("  on http://127.0.0.1:3000 after the Titan boots — it reads your local Titan via",
               role="text_muted")
        cprint("  /v6. For remote access, put your own reverse-proxy / TLS in front of :3000.",
               role="text_muted")
        results.append(Result("observatory", "ok", "enabled — prebuilt bundle will install on :3000"))

    return results
