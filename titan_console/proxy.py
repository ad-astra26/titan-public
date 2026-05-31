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


def proxy_chat(ctx: Context, message: str, *, session: str | None = None) -> tuple[int, dict]:
    """POST a chat turn to api_hcl with the internal key (CLI-style auth)."""
    if not message or not message.strip():
        return 400, {"error": "empty message"}
    payload = {"message": message}
    if session:
        payload["session_id"] = session
    headers = {"Content-Type": "application/json"}
    if ctx.internal_key:
        headers["X-Internal-Key"] = ctx.internal_key
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
