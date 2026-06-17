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
        plugin = getattr(request.app.state, "titan_hcl", None)
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

    plugin = getattr(request.app.state, "titan_hcl", None)
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


def resolve_maker_pubkey(request: Request) -> str:
    """Resolve the Maker's Base58 pubkey (the sovereign trust root) from the api
    process — SoulAccessor SHM-direct first (G18), then the legacy
    plugin.soul._maker_pubkey fallback. Returns "" if unresolvable.

    Shared by verify_maker_auth (admin Ed25519) and the §7.D (D.0) verified-Maker
    presence endpoint, so both verify against the SAME maker_pubkey Titan holds."""
    maker_pubkey_str = ""
    titan_state = getattr(request.app.state, "titan_state", None)
    if titan_state is not None:
        soul_accessor = getattr(titan_state, "soul", None)
        if soul_accessor is not None:
            try:
                # SoulAccessor.maker_pubkey is a @property (reads soul_state.bin
                # SHM slot) — property access, NOT a call.
                mk = soul_accessor.maker_pubkey
                if mk:
                    maker_pubkey_str = str(mk)
            except Exception:
                logger.debug("[Auth] soul_accessor.maker_pubkey lookup failed",
                             exc_info=True)
    if not maker_pubkey_str:
        plugin = getattr(request.app.state, "titan_hcl", None)
        if plugin is not None and hasattr(plugin, "soul") and plugin.soul:
            # Legacy fallback — in api_subprocess `plugin` is an RPC proxy and
            # bare attribute access returns an `_RPCRemoteRef`; resolve by call.
            try:
                ref = plugin.soul._maker_pubkey
                mk = ref() if callable(ref) else ref
                if mk and not str(mk).startswith("<_RPCRemoteRef"):
                    maker_pubkey_str = str(mk)
            except Exception:
                logger.debug("[Auth] plugin.soul._maker_pubkey fallback failed",
                             exc_info=True)
    return maker_pubkey_str


async def verify_maker_auth(request: Request):
    """
    FastAPI dependency that enforces Maker Ed25519 authentication.

    Required headers (one of):
        - Ed25519 path:
            X-Titan-Signature: Base58-encoded Ed25519 signature of "{timestamp}:{body}"
            X-Titan-Timestamp: Unix timestamp (float) of when the request was signed
        - Internal-key path (CLI / scripts / MCP bridge):
            X-Titan-Internal-Key: matches config api.internal_key

    Internal-key bypass mirrors verify_privy_token's pattern — used by
    scripts/shadow_swap.py and other operator tools that don't have
    direct access to the maker keypair.

    Raises:
        HTTPException 401 on missing/invalid/expired credentials.
    """
    # Internal-key bypass (CLI scripts, MCP bridge, shadow_swap.py)
    internal_key = request.headers.get("X-Titan-Internal-Key", "")
    if internal_key:
        plugin = getattr(request.app.state, "titan_hcl", None)
        if plugin is not None:
            expected_key = getattr(plugin, "_full_config", {}).get("api", {}).get("internal_key", "")
            if expected_key and internal_key == expected_key:
                logger.debug("[Auth] Maker internal-key verified")
                return
        raise HTTPException(status_code=401, detail="Invalid internal key.")

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

    # Get maker pubkey — try the SoulAccessor first (G18 SHM-direct), then
    # fall back to plugin.soul._maker_pubkey for the legacy path. The original
    # code referenced an undefined `titan_state` variable (pre-existing typo
    # / incomplete refactor), which made every Ed25519-signed admin call
    # crash with 500 NameError. MakerPanel had been silently using the
    # internal-key bypass instead; the wallet-sig path was untested for
    # months until /admin/pitch-sessions exercised it (2026-05-27).
    plugin = getattr(request.app.state, "titan_hcl", None)
    if plugin is None:
        raise HTTPException(status_code=503, detail="Titan plugin not initialized.")

    # The original code referenced an undefined `titan_state` variable
    # (pre-existing typo / incomplete refactor), which made every Ed25519-signed
    # admin call crash with 500 NameError. MakerPanel had been silently using the
    # internal-key bypass instead; the wallet-sig path was untested for months
    # until /admin/pitch-sessions exercised it (2026-05-27). Pubkey resolution is
    # now the shared resolve_maker_pubkey helper (also used by §7.D D.0).
    maker_pubkey_str = resolve_maker_pubkey(request)

    if not maker_pubkey_str:
        raise HTTPException(
            status_code=503,
            detail="No maker_pubkey configured. Maker Console unavailable.",
        )

    # Verify signature
    from titan_hcl.utils.crypto import verify_maker_signature

    if not verify_maker_signature(message, sig, maker_pubkey_str):
        logger.warning(
            "[Auth] Maker signature verification failed. "
            "expected_pubkey=%s message_len=%d message_preview=%r sig_preview=%s ts=%s body_len=%d",
            maker_pubkey_str,
            len(message),
            message[:80],
            sig[:24] if sig else "",
            ts_str,
            len(body),
        )
        raise HTTPException(status_code=401, detail="Invalid signature.")

    logger.info("[Auth] Maker authenticated successfully.")
    # RFP_affective_grounding_loop §7.D (D.2) — a cryptographically-verified Maker
    # interaction on the TCC / Maker-Console channel. Fire the cross-platform
    # maker_bond tap (fire-and-forget → synthesis_worker). ONLY the Ed25519 path
    # reaches here; the internal-key bypass returned early above and does NOT fire
    # (a bearer secret is not a cryptographic Maker-presence proof — INV-AFF-HONEST).
    try:
        from titan_hcl.api.maker_presence_session import emit_maker_presence
        emit_maker_presence(request, "tcc")
    except Exception:  # noqa: BLE001
        logger.debug("[Auth] maker_presence tcc emit failed", exc_info=True)
