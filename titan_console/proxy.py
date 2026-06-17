"""Proxy live cognition + chat to api_hcl:7777.

The agent holds no Titan state of its own — Stats/Chat panels are proxied to
the real API. When the Titan is down these return a structured
``{"titan_down": true}`` envelope (HTTP 503 from the agent) so the SPA can show
"Titan down" inline instead of a dead panel.
"""
from __future__ import annotations

import json

from .context import Context

# Allow-list of v6 readout prefixes the SPA may proxy (no admin/mutation).
_ALLOWED_PREFIXES = (
    "/health", "/v6/trinity", "/v6/nervous-system", "/v6/metabolism",
    "/v6/cognition", "/v6/language", "/v6/dreaming", "/v6/expression",
    "/v6/social", "/v6/timechain", "/v6/backup", "/v6/reflexes", "/v6/manifest",
)


def is_allowed(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") or path.startswith(p + "?")
               for p in _ALLOWED_PREFIXES)


def proxy_readout(ctx: Context, path: str) -> tuple[int, dict]:
    """GET a whitelisted v6 readout from api_hcl. Returns (status, json-dict)."""
    if not is_allowed(path):
        return 403, {"error": f"path not proxyable: {path}"}
    status, body = ctx.http("GET", f"{ctx.api_base}{path}", timeout=6.0)
    if status == 0:
        return 503, {"titan_down": True,
                     "detail": "api_hcl:7777 unreachable", "path": path}
    try:
        return status, json.loads(body.decode())
    except (ValueError, UnicodeDecodeError):
        return status, {"raw": body.decode("utf-8", "replace")}


def proxy_chat(ctx: Context, message: str, *, session: str | None = None,
               maker_verified: bool = False) -> tuple[int, dict]:
    """POST an owner chat turn to /v6/pitch/chat using the api internal_key.

    The owner bypass on that endpoint authenticates the local console via the
    X-Titan-Internal-Key header (no Privy/wallet, no pitch token). Payload matches
    PitchChatRequest: {titan, thread_id (≥8 chars), message (≤500)}.

    `maker_verified` — RFP_affective_grounding_loop §7.D (D.2): the originating
    request was `device_authed` (the phone's pairing-Ed25519 signature verified
    against the maker_pubkey-bound binding) → relay X-Titan-Maker-Verified so the
    Titan fires the app-channel maker_bond. NEVER set for a non-device-authed
    (local web-UI) owner chat — the bond is cryptographic-only (INV-AFF-HONEST).
    """
    if not message or not message.strip():
        return 400, {"error": "empty message"}
    if not ctx.internal_key:
        return 503, {"error": "owner chat unavailable — no internal_key configured "
                     "(set api.internal_key in ~/.titan/secrets.toml)."}
    thread_id = (session or "console-owner").strip() or "console-owner"
    if len(thread_id) < 8:
        thread_id = (thread_id + "-console-owner")[:64]
    payload = {"titan": ctx.titan_id, "thread_id": thread_id[:64],
               "message": message.strip()[:500]}
    headers = {"Content-Type": "application/json",
               "X-Titan-Internal-Key": ctx.internal_key}
    if maker_verified:
        headers["X-Titan-Channel"] = "app"
        headers["X-Titan-Maker-Verified"] = "1"
    status, body = ctx.http("POST", f"{ctx.api_base}/v6/pitch/chat",
                            body=json.dumps(payload).encode(),
                            headers=headers, timeout=60.0)
    if status == 0:
        return 503, {"titan_down": True,
                     "detail": "api_hcl:7777 unreachable — Titan cannot chat while down"}
    try:
        return status, json.loads(body.decode())
    except (ValueError, UnicodeDecodeError):
        return status, {"raw": body.decode("utf-8", "replace")}
