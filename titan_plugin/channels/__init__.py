"""
channels/__init__.py — Channel adapter registry and shared utilities.

Provides the common HTTP bridge (forward_to_titan), response formatting,
config loading, error translation, and message-splitting helpers used by
all channel adapters.
"""
import asyncio
import logging
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------
_DEFAULT_TITAN_URL = "http://127.0.0.1:7777"
_FORWARD_TIMEOUT = 120.0  # seconds — Titan pipeline can be slow (research, RL)
_MAX_RETRIES = 2           # Retry transient errors (429, 502, 503)
_RETRY_BACKOFF = 3.0       # Seconds between retries


# ---------------------------------------------------------------------------
# Internal key loader (cached)
# ---------------------------------------------------------------------------
_cached_internal_key: Optional[str] = None


def _load_internal_key() -> str:
    global _cached_internal_key
    if _cached_internal_key is not None:
        return _cached_internal_key
    try:
        import pathlib
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]
        config_path = pathlib.Path(__file__).resolve().parent.parent / "config.toml"
        if config_path.exists():
            with open(config_path, "rb") as fh:
                cfg = tomllib.load(fh)
            _cached_internal_key = cfg.get("api", {}).get("internal_key", "")
            return _cached_internal_key
    except Exception:
        pass
    _cached_internal_key = ""
    return ""


# ---------------------------------------------------------------------------
# Core bridge (with retry for transient errors)
# ---------------------------------------------------------------------------
async def forward_to_titan(
    message: str,
    session_id: str,
    user_id: str,
    base_url: str = _DEFAULT_TITAN_URL,
) -> dict:
    """POST /chat and return the parsed JSON response.

    Returns a dict with at least ``response`` on success, or ``error`` on
    failure.  The caller never sees raw exceptions — all errors are
    normalised into the dict.

    Retries up to _MAX_RETRIES times on transient errors (429, 502, 503).
    """
    payload = {
        "message": message,
        "session_id": session_id,
        "user_id": user_id,
    }
    headers: dict[str, str] = {}
    internal_key = _load_internal_key()
    if internal_key:
        headers["X-Titan-Internal-Key"] = internal_key

    last_error = ""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=_FORWARD_TIMEOUT) as client:
                resp = await client.post(f"{base_url}/chat", json=payload, headers=headers)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 403:
                body = resp.json()
                return {
                    "error": body.get("error", "Blocked by Guardian."),
                    "blocked": True,
                    "mode": "Guardian",
                }

            # Transient errors — retry
            if resp.status_code in (429, 502, 503) and attempt < _MAX_RETRIES:
                logger.info("[Channel] HTTP %d, retrying in %.0fs (attempt %d/%d)",
                            resp.status_code, _RETRY_BACKOFF, attempt + 1, _MAX_RETRIES)
                await asyncio.sleep(_RETRY_BACKOFF)
                continue

            if resp.status_code == 503:
                body = resp.json()
                return {
                    "error": body.get("error", "Titan unavailable."),
                    "limbo": True,
                }

            if resp.status_code == 429:
                return {"error": "Titan is busy processing other requests. Try again in a moment."}

            # Unexpected status
            return {"error": _user_friendly_error(f"HTTP {resp.status_code}")}

        except httpx.TimeoutException:
            if attempt < _MAX_RETRIES:
                logger.info("[Channel] Timeout, retrying (attempt %d/%d)", attempt + 1, _MAX_RETRIES)
                await asyncio.sleep(_RETRY_BACKOFF)
                continue
            logger.warning("[Channel] Titan request timed out after %.0fs", _FORWARD_TIMEOUT)
            return {"error": "Titan is taking longer than expected. Try again in a moment."}

        except httpx.ConnectError:
            logger.warning("[Channel] Cannot reach Titan at %s", base_url)
            return {"error": "Titan is currently offline. The maker has been notified."}

        except Exception as exc:  # pragma: no cover — safety net
            logger.error("[Channel] Unexpected error forwarding to Titan: %s", exc)
            last_error = str(exc)
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BACKOFF)
                continue
            return {"error": _user_friendly_error(last_error)}

    return {"error": _user_friendly_error(last_error)}


# ---------------------------------------------------------------------------
# User-friendly error translation
# ---------------------------------------------------------------------------
def _user_friendly_error(raw: str) -> str:
    """Translate raw error messages into user-friendly language."""
    lower = raw.lower()
    if "429" in raw or "rate limit" in lower or "too many" in lower:
        return "Titan is busy. Try again in a moment."
    if "timeout" in lower:
        return "Titan is taking longer than expected. Try again shortly."
    if "connect" in lower or "refused" in lower:
        return "Titan is currently offline. The maker has been notified."
    if "500" in raw or "internal" in lower:
        return "Titan encountered an internal issue. Try again or contact the maker."
    if "memory" in lower or "oom" in lower:
        return "Titan ran out of resources. The maker has been notified."
    # Generic fallback — never show raw tracebacks
    return "Something went wrong. Try again or contact the maker."


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------
async def fetch_banner(base_url: str = _DEFAULT_TITAN_URL) -> str:
    """Fetch the info banner from /status and render it."""
    try:
        from titan_plugin.utils.banner import build_banner
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/status")
        if resp.status_code != 200:
            return ""
        body = resp.json()
        d = body.get("data", body)
        mood_raw = d.get("mood", "Unknown")
        mood = mood_raw.get("label", "Unknown") if isinstance(mood_raw, dict) else mood_raw
        sol = d.get("sol_balance", -1)
        life_pct = min(sol / 0.5 * 100, 100) if sol >= 0 else -1  # 0.5 SOL = 100%
        sov_pct = d.get("sovereignty_pct", 0) * 100 if d.get("sovereignty_pct") else 0
        mem_nodes = d.get("persistent_nodes", d.get("memory_nodes", 0)) or 0
        mem_pct = min(mem_nodes / 1000 * 100, 100)  # 1000 nodes = 100%
        epoch = d.get("epoch", 0) or 0
        return build_banner(life_pct, sov_pct, mem_pct, mood, epoch, style="minimal")
    except Exception:
        return ""


def format_response(result: dict, banner: str = "") -> str:
    """Format a Titan response dict for display in a chat channel.

    On success:  ``banner\\n[Sovereign | Stable] Hello! I'm Titan...``
    On error:    ``[Guardian] Sovereignty Violation: ...``  or plain error.
    """
    if "error" in result:
        mode = result.get("mode", "Error")
        error_msg = result["error"]
        # Don't double-translate already-friendly messages
        if not any(w in error_msg.lower() for w in ("try again", "maker", "guardian")):
            error_msg = _user_friendly_error(error_msg)
        return f"[{mode}] {error_msg}"

    mode = result.get("mode", "Unknown")
    mood = result.get("mood", "Unknown")
    text = result.get("response", "")

    # Strip leaked <function=...> syntax (LLMs output raw tool calls as text)
    if '<function=' in text:
        text = re.sub(
            r'<function=\w+[^>]*>(?:\s*</function>)?',
            '', text, flags=re.DOTALL,
        ).strip()
        text = re.sub(r'\n{3,}', '\n\n', text)

    header = f"[{mode} | {mood}]"
    if banner:
        return f"{banner}\n\n{header} {text}"
    return f"{header} {text}"


# ---------------------------------------------------------------------------
# Message splitting
# ---------------------------------------------------------------------------
def split_message(text: str, limit: int) -> list[str]:
    """Split *text* into chunks that each fit within *limit* characters.

    Splits on paragraph boundaries (``\\n\\n``) first, then on single
    newlines, then on spaces.  Never breaks mid-word unless a single word
    exceeds *limit* (unlikely in practice).
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        # Find the best split point within the limit
        cut = _find_split_point(remaining, limit)
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip("\n")

    return [c for c in chunks if c]


def _find_split_point(text: str, limit: int) -> int:
    """Return the best character index to split *text* at, within *limit*."""
    window = text[:limit]

    # Prefer paragraph break
    idx = window.rfind("\n\n")
    if idx > 0:
        return idx + 1  # include one newline

    # Prefer line break
    idx = window.rfind("\n")
    if idx > 0:
        return idx + 1

    # Prefer space
    idx = window.rfind(" ")
    if idx > 0:
        return idx + 1

    # Hard cut (single very long token — shouldn't happen)
    return limit


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def get_channel_config(channel: str) -> dict:
    """Load channel-specific config from ``titan_plugin/config.toml``.

    Returns all keys prefixed with ``{channel}_`` (without the prefix),
    from the ``[channels]`` section.  Falls back to an empty dict if the
    section or keys are missing.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            logger.error("[Channel] Neither tomllib nor tomli available for config loading.")
            return {}

    import pathlib

    config_path = pathlib.Path(__file__).resolve().parent.parent / "config.toml"
    if not config_path.exists():
        logger.warning("[Channel] config.toml not found at %s", config_path)
        return {}

    with open(config_path, "rb") as fh:
        full = tomllib.load(fh)

    channels_section = full.get("channels", {})
    prefix = f"{channel}_"
    return {
        k[len(prefix):]: v
        for k, v in channels_section.items()
        if k.startswith(prefix)
    }
