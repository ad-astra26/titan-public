"""
Venice Session Adapter — Use Venice Pro plan via web session auth.

Instead of paying for API credits, this adapter authenticates using the
Clerk session cookie from Venice's web interface. Pro plan users get
rate-limited inference included in their subscription.

Setup:
    1. Log into venice.ai in your browser
    2. Open DevTools → Application → Cookies → venice.ai
    3. Copy BOTH cookies:
       - `__client` → paste into config.toml: venice_client_cookie = "..."
       - `__session` → paste into config.toml: venice_session_token = "..."
    4. The adapter auto-refreshes __session tokens using the __client cookie.

How it works:
    - __session is a Clerk JWT with ~60s TTL (too short to use statically)
    - __client contains a rotating_token that authenticates refresh requests
    - Before each API call, we check token expiry and refresh if needed
    - Refresh: POST https://clerk.venice.ai/v1/client/sessions/{sid}/tokens
      with __client cookie → returns fresh __session JWT
"""
import asyncio
import base64
import json
import logging
import time

import httpx

logger = logging.getLogger(__name__)

# Venice web chat uses the same API structure but authenticates via Clerk JWT
VENICE_WEB_BASE = "https://api.venice.ai/api/v1"

# Clerk Frontend API for Venice (token refresh)
CLERK_FAPI_BASE = "https://clerk.venice.ai/v1"

# Refresh buffer — refresh token this many seconds before expiry
REFRESH_BUFFER_SECONDS = 15

# Models available on Venice Pro plan
VENICE_PRO_MODELS = [
    "llama-3.3-70b",
    "llama-3.1-405b",
    "deepseek-r1-671b",
    "qwen-2.5-vl-72b",
]


def _decode_jwt_payload(token: str) -> dict:
    """Decode JWT payload without verification (we just need exp/sid claims)."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        # Add padding for base64url
        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload_bytes)
    except Exception:
        return {}


class VeniceSessionClient:
    """
    Async client for Venice AI using Pro plan session authentication.

    Uses Clerk __session JWT as Bearer token. Auto-refreshes the short-lived
    JWT (~60s TTL) using the __client cookie's rotating_token via Clerk's
    Frontend API.
    """

    def __init__(
        self,
        session_token: str,
        client_cookie: str = "",
        model: str = "llama-3.3-70b",
        base_url: str = VENICE_WEB_BASE,
    ):
        self._session_token = session_token.strip()
        self._client_cookie = client_cookie.strip()
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._request_count = 0
        self._refresh_count = 0
        self._last_error: str | None = None
        self._refresh_lock = asyncio.Lock()

        # Parse initial token for session ID and expiry
        self._session_id = ""
        self._token_expires_at = 0.0
        if self._session_token:
            self._parse_token(self._session_token)

    def _parse_token(self, token: str) -> None:
        """Extract session ID and expiry from JWT claims."""
        claims = _decode_jwt_payload(token)
        if claims:
            self._session_id = claims.get("sid", "")
            self._token_expires_at = float(claims.get("exp", 0))
            logger.info(
                "[VeniceSession] Token parsed: sid=%s, expires_in=%.0fs",
                self._session_id[:20] if self._session_id else "?",
                self._token_expires_at - time.time(),
            )

    @property
    def is_configured(self) -> bool:
        return bool(self._session_token)

    @property
    def can_auto_refresh(self) -> bool:
        return bool(self._client_cookie and self._session_id)

    @property
    def stats(self) -> dict:
        return {
            "requests": self._request_count,
            "refreshes": self._refresh_count,
            "model": self._model,
            "last_error": self._last_error,
            "configured": self.is_configured,
            "auto_refresh": self.can_auto_refresh,
            "session_id": self._session_id[:20] + "..." if self._session_id else "",
            "token_expires_in": max(0, self._token_expires_at - time.time()),
        }

    def _is_token_expired(self) -> bool:
        """Check if token is expired or will expire within the buffer window."""
        if not self._token_expires_at:
            return True  # Unknown expiry = treat as expired
        return time.time() >= (self._token_expires_at - REFRESH_BUFFER_SECONDS)

    async def _refresh_token(self) -> bool:
        """
        Refresh the __session JWT via Clerk's Frontend API.

        Clerk flow:
        - POST /v1/client/sessions/{sid}/tokens
        - Cookie: __client=<client_cookie_value>
        - Query: _clerk_session_id=<sid>
        - Response: JSON with { jwt: "new_session_token" } or
                    client response with lastActiveSessionId pointing to new token
        """
        async with self._refresh_lock:
            # Double-check after acquiring lock (another coroutine may have refreshed)
            if not self._is_token_expired():
                return True

            if not self._client_cookie or not self._session_id:
                logger.warning(
                    "[VeniceSession] Cannot refresh — missing client_cookie or session_id"
                )
                return False

            url = f"{CLERK_FAPI_BASE}/client/sessions/{self._session_id}/tokens"
            params = {"_clerk_session_id": self._session_id}

            headers = {
                "Cookie": f"__client={self._client_cookie}",
                "Origin": "https://venice.ai",
                "Referer": "https://venice.ai/",
                "Content-Type": "application/x-www-form-urlencoded",
            }

            try:
                async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                    resp = await client.post(url, params=params, headers=headers)

                    if resp.status_code == 401:
                        logger.error(
                            "[VeniceSession] Refresh 401 — client cookie expired. "
                            "Re-copy __client cookie from browser."
                        )
                        self._last_error = "Client cookie expired — re-copy from browser"
                        return False

                    if resp.status_code != 200:
                        body = resp.text[:300]
                        logger.error(
                            "[VeniceSession] Refresh failed: %d — %s", resp.status_code, body
                        )
                        self._last_error = f"Refresh failed: HTTP {resp.status_code}"
                        return False

                    data = resp.json()

                    # Clerk response format: { jwt: "..." } or nested in response
                    new_jwt = None

                    # Direct JWT response
                    if "jwt" in data:
                        new_jwt = data["jwt"]

                    # Nested in response.client or response object
                    if not new_jwt and isinstance(data, dict):
                        # Try response → sessions → find active → last_active_token
                        response = data.get("response", data)
                        if isinstance(response, dict):
                            # Check for direct jwt field
                            if "jwt" in response:
                                new_jwt = response["jwt"]
                            # Check client.sessions[].last_active_token.jwt
                            client_data = response.get("client", response)
                            if isinstance(client_data, dict):
                                sessions = client_data.get("sessions", [])
                                for s in sessions:
                                    if isinstance(s, dict) and s.get("id") == self._session_id:
                                        token_obj = s.get("last_active_token", {})
                                        if isinstance(token_obj, dict) and "jwt" in token_obj:
                                            new_jwt = token_obj["jwt"]
                                            break

                    if not new_jwt:
                        # Last resort: check Set-Cookie header for __session
                        set_cookie = resp.headers.get("set-cookie", "")
                        if "__session=" in set_cookie:
                            for part in set_cookie.split(";"):
                                part = part.strip()
                                if part.startswith("__session="):
                                    new_jwt = part[len("__session="):]
                                    break

                    if not new_jwt:
                        logger.error(
                            "[VeniceSession] Refresh response missing JWT. Keys: %s",
                            list(data.keys()) if isinstance(data, dict) else type(data).__name__,
                        )
                        logger.debug("[VeniceSession] Response body: %s", json.dumps(data)[:500])
                        self._last_error = "Refresh response missing JWT"
                        return False

                    # Update token
                    old_expires = self._token_expires_at
                    self._session_token = new_jwt
                    self._parse_token(new_jwt)
                    self._refresh_count += 1
                    self._last_error = None

                    logger.info(
                        "[VeniceSession] Token refreshed (#%d). New TTL: %.0fs",
                        self._refresh_count,
                        self._token_expires_at - time.time(),
                    )

                    # Update rotating_token from response if present (Clerk rotates it)
                    if isinstance(data, dict):
                        response = data.get("response", data)
                        client_data = (response if isinstance(response, dict) else {}).get("client", {})
                        if isinstance(client_data, dict):
                            new_rotating = client_data.get("rotating_token", "")
                            if new_rotating:
                                # Rebuild __client cookie with new rotating_token
                                # The __client cookie is URL-encoded JSON
                                self._update_client_cookie_rotating_token(new_rotating)

                    return True

            except Exception as e:
                logger.error("[VeniceSession] Refresh error: %s", e)
                self._last_error = f"Refresh error: {e}"
                return False

    def _update_client_cookie_rotating_token(self, new_rotating_token: str) -> None:
        """Update the rotating_token in the __client cookie value."""
        try:
            import urllib.parse
            decoded = urllib.parse.unquote(self._client_cookie)
            client_data = json.loads(decoded)
            if isinstance(client_data, dict) and "rotating_token" in client_data:
                client_data["rotating_token"] = new_rotating_token
                self._client_cookie = urllib.parse.quote(json.dumps(client_data))
                logger.debug("[VeniceSession] Updated rotating_token in client cookie")
        except Exception:
            pass  # Cookie format may vary, ignore if can't parse

    async def _ensure_fresh_token(self) -> bool:
        """Ensure we have a valid, non-expired session token."""
        if not self._is_token_expired():
            return True

        if self.can_auto_refresh:
            logger.info("[VeniceSession] Token expired/expiring, refreshing...")
            return await self._refresh_token()

        logger.warning(
            "[VeniceSession] Token expired and no auto-refresh configured. "
            "Set venice_client_cookie in config.toml for auto-refresh."
        )
        return False  # Will try anyway with stale token

    async def chat_completion(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> dict:
        """
        Send a chat completion request using session auth.
        Auto-refreshes token before request if expired.
        """
        if not self._session_token:
            return {"error": "No Venice session token configured"}

        # Auto-refresh if needed
        await self._ensure_fresh_token()

        use_model = model or self._model
        payload = {
            "model": use_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self._session_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True
            ) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )

                self._request_count += 1

                # If 401 and we can refresh, try once more
                if resp.status_code == 401 and self.can_auto_refresh:
                    logger.info("[VeniceSession] 401 — attempting token refresh and retry...")
                    # Force expiry so refresh runs
                    self._token_expires_at = 0
                    refreshed = await self._refresh_token()
                    if refreshed:
                        # Retry with new token
                        headers["Authorization"] = f"Bearer {self._session_token}"
                        resp = await client.post(
                            f"{self._base_url}/chat/completions",
                            headers=headers,
                            json=payload,
                        )
                        self._request_count += 1

                if resp.status_code == 401:
                    self._last_error = "Session expired — re-copy cookies from browser"
                    logger.warning("[VeniceSession] 401 — session token expired or invalid")
                    return {"error": self._last_error}

                if resp.status_code == 402:
                    self._last_error = "402 — session token not recognized as Pro plan auth"
                    logger.warning("[VeniceSession] 402 — token may be an API key, not session")
                    return {"error": self._last_error}

                if resp.status_code == 429:
                    self._last_error = "Rate limited — Pro plan quota reached"
                    logger.warning("[VeniceSession] 429 — rate limited")
                    return {"error": self._last_error}

                resp.raise_for_status()
                self._last_error = None
                return resp.json()

        except httpx.TimeoutException:
            self._last_error = f"Timeout after {timeout}s"
            logger.error("[VeniceSession] Request timed out")
            return {"error": self._last_error}
        except Exception as e:
            self._last_error = str(e)
            logger.error("[VeniceSession] Request failed: %s", e)
            return {"error": self._last_error}

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Simple single-turn completion. Returns text or empty string."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        result = await self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if "error" in result:
            return ""

        try:
            return result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            return ""

    async def test_connection(self) -> tuple[bool, str]:
        """Test if the session token works (with auto-refresh)."""
        result = await self.chat_completion(
            messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
            max_tokens=5,
            timeout=15.0,
        )

        if "error" in result:
            return False, result["error"]

        try:
            text = result["choices"][0]["message"]["content"].strip()
            return True, f"Connected — model responded: {text}"
        except (KeyError, IndexError):
            return False, f"Unexpected response format: {json.dumps(result)[:200]}"
