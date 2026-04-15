"""
api/auth.py
Authentication middleware for the Sovereign Observatory.

Two tiers:
  1. Maker auth (Ed25519) — for admin /maker/* endpoints
  2. Privy JWT auth — for user-facing /chat endpoints (wallet, OAuth, email login)

Anti-replay: Maker requests expire after 30 seconds.
Privy JWTs are verified via JWKS with key caching.
"""
import logging
import time
from typing import Optional

import jwt as pyjwt
import httpx

from fastapi import Request, HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Privy JWT verification (user auth for /chat)
# ---------------------------------------------------------------------------

_jwks_cache: dict = {}
_jwks_cache_ts: float = 0.0
_JWKS_CACHE_TTL = 3600  # 1 hour

async def _get_privy_jwks(app_id: str) -> dict:
    """Fetch and cache Privy JWKS for token verification."""
    global _jwks_cache, _jwks_cache_ts
    now = time.time()
    if _jwks_cache and (now - _jwks_cache_ts) < _JWKS_CACHE_TTL:
        return _jwks_cache
    url = f"https://auth.privy.io/api/v1/apps/{app_id}/jwks.json"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        _jwks_cache = resp.json()
        _jwks_cache_ts = now
    return _jwks_cache


def _decode_privy_token(token: str, jwks: dict, app_id: str) -> dict:
    """Decode and verify a Privy access token against JWKS."""
    header = pyjwt.get_unverified_header(token)
    kid = header.get("kid")
    key_data = None
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            key_data = k
            break
    if not key_data:
        raise pyjwt.InvalidTokenError("No matching key found in JWKS")
    public_key = pyjwt.algorithms.ECAlgorithm.from_jwk(key_data)
    return pyjwt.decode(
        token,
        public_key,
        algorithms=["ES256"],
        issuer="privy.io",
        audience=app_id,
    )


async def verify_privy_token(request: Request) -> Optional[dict]:
    """
    FastAPI dependency that verifies a Privy Bearer token.

    Also accepts X-Titan-Internal-Key for internal scripts (endurance tests, MCP bridge).
    Returns the decoded JWT claims dict on success, or a synthetic claims dict for internal keys.
    Raises HTTPException 401 on invalid/missing credentials.
    """
    # Internal API key bypass (for endurance tests, scripts, MCP bridge)
    internal_key = request.headers.get("X-Titan-Internal-Key", "")
    if internal_key:
        plugin = getattr(request.app.state, "titan_plugin", None)
        if plugin:
            expected_key = getattr(plugin, "_full_config", {}).get("api", {}).get("internal_key", "")
            if expected_key and internal_key == expected_key:
                user_id = request.headers.get("X-Titan-User-Id", "internal")
                logger.debug("[Auth] Internal key verified for user: %s", user_id)
                return {"sub": user_id, "iss": "titan-internal"}
        raise HTTPException(status_code=401, detail="Invalid internal key.")

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    token = auth_header[7:]

    plugin = getattr(request.app.state, "titan_plugin", None)
    if plugin is None:
        raise HTTPException(status_code=503, detail="Titan plugin not initialized.")

    # Get Privy app ID from config
    frontend_cfg = getattr(plugin, "_full_config", {}).get("frontend", {})
    app_id = frontend_cfg.get("privy_app_id", "")
    if not app_id:
        raise HTTPException(status_code=503, detail="Privy not configured.")

    try:
        jwks = await _get_privy_jwks(app_id)
        claims = _decode_privy_token(token, jwks, app_id)
        logger.debug("[Auth] Privy token verified for user: %s", claims.get("sub", "unknown"))
        return claims
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except pyjwt.InvalidTokenError as e:
        logger.warning("[Auth] Privy token verification failed: %s", e)
        raise HTTPException(status_code=401, detail="Invalid token.")
    except httpx.HTTPError as e:
        logger.error("[Auth] Failed to fetch Privy JWKS: %s", e)
        raise HTTPException(status_code=503, detail="Auth service unavailable.")


# ---------------------------------------------------------------------------
# Maker Ed25519 authentication (admin endpoints)
# ---------------------------------------------------------------------------

# Maximum age of a signed request (seconds)
# 60s allows buffer for mainnet finality (~400ms slots but network variance)
_REQUEST_TTL = 60


async def verify_maker_auth(request: Request):
    """
    FastAPI dependency that enforces Maker Ed25519 authentication.

    Required headers:
        X-Titan-Signature: Base58-encoded Ed25519 signature of "{timestamp}:{body}"
        X-Titan-Timestamp: Unix timestamp (float) of when the request was signed

    Raises:
        HTTPException 401 on missing/invalid/expired signatures.
    """
    sig = request.headers.get("X-Titan-Signature")
    ts_str = request.headers.get("X-Titan-Timestamp")

    if not sig or not ts_str:
        raise HTTPException(
            status_code=401,
            detail="Missing X-Titan-Signature or X-Titan-Timestamp headers.",
        )

    # Anti-replay guard
    try:
        req_time = float(ts_str)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid timestamp format.")

    if abs(time.time() - req_time) > _REQUEST_TTL:
        raise HTTPException(
            status_code=401,
            detail=f"Request expired. Signatures are valid for {_REQUEST_TTL}s.",
        )

    # Reconstruct the signed message: "{timestamp}:{body}"
    body = await request.body()
    message = f"{ts_str}:{body.decode('utf-8')}"

    # Get maker pubkey from the plugin instance stored in app.state
    plugin = getattr(request.app.state, "titan_plugin", None)
    if plugin is None:
        raise HTTPException(status_code=503, detail="Titan plugin not initialized.")

    maker_pubkey_str = ""
    if hasattr(plugin, "soul") and plugin.soul:
        mk = getattr(plugin.soul, "_maker_pubkey", None)
        if mk:
            maker_pubkey_str = str(mk)

    if not maker_pubkey_str:
        raise HTTPException(
            status_code=503,
            detail="No maker_pubkey configured. Maker Console unavailable.",
        )

    # Verify signature
    from titan_plugin.utils.crypto import verify_maker_signature

    if not verify_maker_signature(message, sig, maker_pubkey_str):
        logger.warning("[Auth] Maker signature verification failed.")
        raise HTTPException(status_code=401, detail="Invalid signature.")

    logger.info("[Auth] Maker authenticated successfully.")
