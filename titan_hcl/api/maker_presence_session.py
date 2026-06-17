"""api/maker_presence_session.py — RFP_affective_grounding_loop §7.D (D.0).

Closes the `/chat` plaintext Maker gap. Today `is_maker = (user_id == "maker")`
(`api/chat.py`) trusts a bare Privy DID / internal-key claim with NO
wallet→`maker_pubkey` check — anyone who can set `user_id="maker"` is "the Maker".
This store mints a short-lived **verified-Maker session marker** only AFTER the
caller proves control of `maker_pubkey` by Ed25519-signing a one-time server
nonce with the same Solana wallet the MakerPanel already uses for proposals
(reusing `utils.crypto.verify_maker_signature`). `/chat` then reads the marker
for `is_maker=verified` + the cross-platform `maker_bond` tap (D.2).

Sovereign by construction: Titan is the SOLE verifier — he checks the signature
against the `maker_pubkey` he himself holds (`core/soul.py`), with no external
identity service (the reason option B beat Privy `getUser` in §7.D).

NON-BREAKING (additive):
  • The MakerPanel proposal sign-flow / `verify_maker_auth` path are untouched.
  • The internal-key operator bypass is untouched and does NOT mint a marker — a
    bearer secret is not a cryptographic Maker-presence proof (INV-AFF-HONEST).

Lifetime: the marker store is per-api-process in-memory. An api zero-downtime
reload resets it → the Maker simply re-signs once (cheap, one wallet popup). No
durable state to corrupt; no SHM slot (this is request-edge auth, not Titan state).
"""
from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# A signing challenge is short-lived + single-use (replay-resistant): a captured
# signature is useless once its nonce is consumed or aged out.
NONCE_TTL_S: float = 120.0
# A minted verified-Maker marker is valid for one working session; the Maker
# re-signs after it lapses. Mirrors verify_maker_auth's "fresh proof" spirit at
# a session (not per-request) granularity so chat UX stays a single popup.
MARKER_TTL_S: float = 3600.0

# Hard caps so a flood of nonce/marker requests can't grow memory unbounded.
_MAX_NONCES = 256
_MAX_MARKERS = 256

# The nonce is prefixed so the wallet-signed payload is self-describing in the
# wallet UI ("Titan: verify Maker presence …") and can never collide with the
# "{ts}:{body}" proposal-sign payload verify_maker_auth consumes.
NONCE_PREFIX = "titan-maker-presence:"


@dataclass
class _Marker:
    expires_at: float


class VerifiedMakerSessions:
    """Per-process store of issued nonces + minted verified-Maker markers.

    Thread-safe (FastAPI runs handlers on a threadpool / the event loop). All
    time math is wall-clock `time.time()` so expiry survives across awaits."""

    def __init__(self, *, nonce_ttl_s: float = NONCE_TTL_S,
                 marker_ttl_s: float = MARKER_TTL_S) -> None:
        self._nonce_ttl = float(nonce_ttl_s)
        self._marker_ttl = float(marker_ttl_s)
        self._lock = threading.Lock()
        # nonce -> (session_key, expires_at). Single-use: consumed on verify.
        self._nonces: dict[str, tuple[str, float]] = {}
        # session_key -> _Marker
        self._markers: dict[str, _Marker] = {}

    # ── housekeeping ────────────────────────────────────────────────────────
    def _gc(self, now: float) -> None:
        """Drop expired nonces/markers. Caller holds the lock."""
        if self._nonces:
            dead = [n for n, (_, exp) in self._nonces.items() if exp <= now]
            for n in dead:
                self._nonces.pop(n, None)
        if self._markers:
            dead_m = [k for k, m in self._markers.items() if m.expires_at <= now]
            for k in dead_m:
                self._markers.pop(k, None)
        # Bound memory even under a churn of distinct sessions/nonces.
        if len(self._nonces) > _MAX_NONCES:
            for n in sorted(self._nonces, key=lambda x: self._nonces[x][1])[
                    :len(self._nonces) - _MAX_NONCES]:
                self._nonces.pop(n, None)
        if len(self._markers) > _MAX_MARKERS:
            for k in sorted(self._markers,
                            key=lambda x: self._markers[x].expires_at)[
                    :len(self._markers) - _MAX_MARKERS]:
                self._markers.pop(k, None)

    # ── nonce issue ───────────────────────────────────────────────────────────
    def issue_nonce(self, session_key: str) -> str:
        """Mint a fresh single-use challenge bound to this Privy session. The
        caller signs the RETURNED string verbatim with the Maker wallet."""
        now = time.time()
        nonce = NONCE_PREFIX + secrets.token_urlsafe(24)
        with self._lock:
            self._gc(now)
            self._nonces[nonce] = (str(session_key), now + self._nonce_ttl)
        return nonce

    # ── verify + mint ─────────────────────────────────────────────────────────
    def verify_and_mint(self, session_key: str, nonce: str, signature: str,
                        maker_pubkey: str) -> bool:
        """Consume the nonce (single-use), verify the Ed25519 signature against
        `maker_pubkey`, and on success mint a verified-Maker marker for this
        session. Returns True iff a marker was minted. Honest-failure on every
        bad-input path (unknown/expired/mismatched nonce, bad signature) → no
        marker, no bond (INV-AFF-HONEST)."""
        if not nonce or not signature or not maker_pubkey or not session_key:
            return False
        now = time.time()
        with self._lock:
            self._gc(now)
            entry = self._nonces.pop(nonce, None)   # single-use: pop on attempt
        if entry is None:
            return False
        bound_key, expires_at = entry
        if expires_at <= now or bound_key != str(session_key):
            return False
        try:
            from titan_hcl.utils.crypto import verify_maker_signature
            ok = verify_maker_signature(nonce, signature, str(maker_pubkey))
        except Exception as e:  # noqa: BLE001
            logger.warning("[MakerPresence] signature verify raised: %s", e)
            return False
        if not ok:
            logger.info("[MakerPresence] verify-presence signature REJECTED "
                        "(session=%s)", _redact(session_key))
            return False
        with self._lock:
            self._markers[str(session_key)] = _Marker(
                expires_at=now + self._marker_ttl)
        logger.info("[MakerPresence] verified-Maker marker minted "
                    "(session=%s ttl=%.0fs)", _redact(session_key),
                    self._marker_ttl)
        return True

    # ── read ──────────────────────────────────────────────────────────────────
    def is_verified(self, session_key: str) -> bool:
        """True iff this session holds an unexpired verified-Maker marker."""
        if not session_key:
            return False
        now = time.time()
        with self._lock:
            m = self._markers.get(str(session_key))
            if m is None:
                return False
            if m.expires_at <= now:
                self._markers.pop(str(session_key), None)
                return False
            return True


def emit_maker_presence(request, channel: str) -> None:
    """Fire MAKER_PRESENCE_VERIFIED{channel} (RFP §7.D D.2/D.4).

    Fans out to BOTH consumers of a verified Maker-presence:
      • synthesis  — the cross-platform maker_bond → DA affective nudge (D.2).
      • emot_cgn   — the inner grounding of the MAKER_PRESENCE primitive (D.4),
                     "in parallel the bridge feeds emot-cgn evidence".
    Shared by every api-edge tap (web `/chat`, app `/v6/pitch/chat`, TCC
    `verify_maker_auth`). CALLERS gate on cryptographic verification before
    calling this — this helper only transports (INV-AFF-HONEST is upstream).
    Soft: a missing bridge / send failure never breaks the triggering request."""
    bridge = getattr(getattr(request, "app", None), "state", None)
    bridge = getattr(bridge, "chat_bridge_bus", None) if bridge is not None else None
    if bridge is None or not hasattr(bridge, "emit"):
        return
    try:
        from titan_hcl import bus
        payload = {"channel": str(channel), "ts": time.time()}
        bridge.emit("synthesis", bus.MAKER_PRESENCE_VERIFIED, dict(payload))
        bridge.emit("emot_cgn", bus.MAKER_PRESENCE_VERIFIED, dict(payload))
    except Exception:  # noqa: BLE001
        logger.debug("[MakerPresence] emit failed", exc_info=True)


def session_key_from_claims(claims: Optional[dict]) -> str:
    """Derive a stable per-session key from Privy claims. Binds the marker to the
    Privy session (`sid` when present) under the user DID (`sub`) so a marker
    cannot leak across users/sessions. Empty when unauthenticated."""
    if not claims:
        return ""
    sub = str(claims.get("sub", "") or "")
    sid = str(claims.get("sid", "") or "")
    if not sub:
        return ""
    return f"{sub}|{sid}" if sid else sub


def _redact(session_key: str) -> str:
    s = str(session_key)
    return (s[:10] + "…") if len(s) > 12 else s
