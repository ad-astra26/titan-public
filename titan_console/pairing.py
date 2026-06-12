"""QR mutual-pairing + Ed25519 request signing for the mobile app (SPEC AG3/AG4/AG5, §3).

Stdlib-only (vendored `_ed25519`), so the Console Agent keeps its crash-decoupling
(AG2). State files — sole writer = this module (AG7):
  ~/.titan/console_server_key.json   server Ed25519 keypair (TOFU pin in the QR)
  ~/.titan/pairing_pending.json      the in-flight pairing session (single-use, TTL)
  ~/.titan/devices.json              registered devices [{device_id,pubkey,label,...}]

Crypto contract (MUST match the Kotlin shared/ side, vectors pinned in tests):
  code6  = uint32_be( sha256(pairing_token_raw ‖ device_pubkey_raw)[0:4] ) % 1_000_000
  signed = ed25519( "<method>\\n<path>\\n<timestamp>\\n<sha256hex(body)>" )
Both `pairing_token` and `device_pubkey` are compared/hashed as RAW bytes
(base64-decoded first) — never the base64 text.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path

from . import _ed25519
from .context import Context

_TTL_DEFAULT = 90          # seconds a QR pairing token is valid
_SIG_SKEW = 300            # seconds a signed request timestamp may deviate
_seen_sigs: dict[str, float] = {}   # replay guard: signature(b64) → expiry


# ── paths (overridable in tests via ctx.secrets_path's dir) ──────────────────
def _titan_dir(ctx: Context) -> Path:
    if ctx.secrets_path:
        return Path(ctx.secrets_path).parent
    return Path(os.path.expanduser("~/.titan"))


def _server_key_path(ctx: Context) -> Path:
    return _titan_dir(ctx) / "console_server_key.json"


def _pending_path(ctx: Context) -> Path:
    return _titan_dir(ctx) / "pairing_pending.json"


def _devices_path(ctx: Context) -> Path:
    return _titan_dir(ctx) / "devices.json"


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode())


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(p)  # atomic single-writer (AG7)


def _read_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


# ── crypto contract (mirrors shared/ Kotlin) ─────────────────────────────────
def code6(pairing_token: bytes, device_pubkey: bytes) -> str:
    h = hashlib.sha256(pairing_token + device_pubkey).digest()
    n = int.from_bytes(h[:4], "big")
    return f"{n % 1_000_000:06d}"


def body_sha256_hex(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def canonical_request(method: str, path: str, timestamp: str, body_sha_hex: str) -> str:
    return f"{method}\n{path}\n{timestamp}\n{body_sha_hex}"


# ── server identity (TOFU pin) ───────────────────────────────────────────────
def load_or_create_server_key(ctx: Context) -> tuple[bytes, bytes]:
    """Return (seed, pubkey); generate + persist on first use."""
    p = _server_key_path(ctx)
    data = _read_json(p, None)
    if data and "seed" in data and "pub" in data:
        return _unb64(data["seed"]), _unb64(data["pub"])
    seed, pub = _ed25519.keygen()
    _write_json(p, {"seed": _b64(seed), "pub": _b64(pub)})
    return seed, pub


# ── pairing lifecycle ────────────────────────────────────────────────────────
_VALID_MODES = {"local", "remote", "install"}


def _detect_lan_ip() -> str | None:
    """Default-route interface IP via a connect-less UDP socket (no packets sent).
    Defensive: returns None on any failure so the caller can omit the endpoint."""
    import socket
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("203.0.113.10", 80))  # picks the iface for the default route; sends nothing
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        if s is not None:
            s.close()


def _https_endpoint(host: str, port: int) -> str:
    """Normalize an operator-supplied/detected host into an https endpoint (AG-TLS:
    every mode is pinned-TLS, so the endpoint is always https)."""
    host = host.strip()
    for scheme in ("https://", "http://"):
        if host.startswith(scheme):
            host = host[len(scheme):]
            break
    host = host.rstrip("/")
    return f"https://{host}" if (":" in host) else f"https://{host}:{port}"


def mint_pairing(ctx: Context, *, ttl: int = _TTL_DEFAULT, public_url: str | None = None,
                 mode: str | None = None, now: float | None = None) -> tuple[int, dict]:
    """Operator-triggered: mint a single-use QR pairing token. (token-gated route)

    `mode` ∈ {local, remote, install} resolves the endpoint (AG-MODE/AD-8):
      local  → https://<LAN-ip>:<console_port>   (auto-detected)
      remote → public_url (Maker-typed, normalized to https)
      install→ no endpoint (the CLI drives both ends on the box)
    The QR additionally carries the TLS pin (AG-TLS) when the agent serves TLS.
    """
    now = now if now is not None else time.time()
    if mode is not None and mode not in _VALID_MODES:
        return 400, {"error": f"invalid mode {mode!r}; one of {sorted(_VALID_MODES)}"}
    token_raw = os.urandom(32)
    token_b64 = _b64(token_raw)
    _, server_pub = load_or_create_server_key(ctx)
    _write_json(_pending_path(ctx), {
        "token": token_b64, "created_at": now, "ttl": ttl, "state": "minted",
    })
    payload = {
        "pairing_token": token_b64,
        "server_pubkey": _b64(server_pub),
        "titan_id": ctx.titan_id,
        "ttl": ttl,
    }
    if mode:
        payload["mode"] = mode
    # endpoint resolution: explicit public_url wins; else local-mode auto-detects the LAN IP.
    endpoint = public_url
    if not endpoint and mode == "local":
        ip = _detect_lan_ip()
        if ip:
            endpoint = f"{ip}:{ctx.console_port}"
    if endpoint:
        payload["endpoint_url"] = _https_endpoint(endpoint, ctx.console_port)
    if ctx.tls_pin:
        payload["server_tls_pin"] = ctx.tls_pin
    return 200, payload


def _load_valid_pending(ctx: Context, pairing_token: str, now: float):
    """Return the pending dict if it matches the token + is unexpired, else None."""
    pend = _read_json(_pending_path(ctx), None)
    if not pend or pend.get("token") != pairing_token:
        return None
    if now - float(pend.get("created_at", 0)) > float(pend.get("ttl", _TTL_DEFAULT)):
        return None
    return pend


def submit_device(ctx: Context, payload: dict, *, now: float | None = None) -> tuple[int, dict]:
    """App-side: submit device pubkey under a valid pairing token. Pairing-token-gated."""
    now = now if now is not None else time.time()
    token = payload.get("pairing_token", "")
    pub_b64 = payload.get("device_pubkey", "")
    device_id = payload.get("device_id", "")
    if not token or not pub_b64 or not device_id:
        return 400, {"error": "pairing_token, device_pubkey, device_id required"}
    pend = _load_valid_pending(ctx, token, now)
    if not pend:
        return 401, {"error": "invalid or expired pairing token"}
    if pend.get("state") == "confirmed":
        return 409, {"error": "pairing already completed"}
    try:
        token_raw = _unb64(token)
        pub_raw = _unb64(pub_b64)
    except (ValueError, base64.binascii.Error):
        return 400, {"error": "device_pubkey/pairing_token not valid base64"}
    if len(pub_raw) != 32:
        return 400, {"error": "device_pubkey must be 32 bytes"}
    pend.update({
        "state": "submitted", "device_pubkey": pub_b64, "device_id": device_id,
        "fingerprint": payload.get("fingerprint", ""), "label": payload.get("label", "phone"),
        "code6": code6(token_raw, pub_raw), "submitted_at": now,
    })
    _write_json(_pending_path(ctx), pend)
    return 200, {"ok": True, "awaiting_confirm": True}


def pair_status(ctx: Context, pairing_token: str, *, now: float | None = None) -> tuple[int, dict]:
    """Operator polls this; shows the code6 to match against the phone. Token-gated."""
    now = now if now is not None else time.time()
    pend = _read_json(_pending_path(ctx), None)
    if not pend or pend.get("token") != pairing_token:
        return 404, {"state": "none"}
    if now - float(pend.get("created_at", 0)) > float(pend.get("ttl", _TTL_DEFAULT)) \
            and pend.get("state") != "confirmed":
        return 200, {"state": "expired"}
    out = {"state": pend.get("state", "minted")}
    if pend.get("state") in ("submitted", "confirmed"):
        out["code6"] = pend.get("code6")
        out["label"] = pend.get("label")
        # The device pubkey (public, not secret) so the Maker panel can wallet-sign it
        # for the additive on-chain Maker-binding (AG-MAKER-BIND).
        out["device_pubkey"] = pend.get("device_pubkey")
    return 200, out


def _maker_binding(device_pubkey_b64: str, maker_pubkey: str | None,
                   maker_sig: str | None) -> dict:
    """Additive Maker-binding (AG-MAKER-BIND/AD-10). If the Observatory path supplied a
    wallet signature over the device pubkey, verify it (vendored Ed25519, AG2) and record
    {method:'wallet', maker_pubkey}; otherwise the operator-token gate that admitted this
    confirm IS the Maker proof → {method:'operator_token'}. A SUPPLIED-but-INVALID
    signature is a hard error (caller maps to 403) — we never downgrade silently."""
    if not (maker_pubkey and maker_sig):
        return {"method": "operator_token"}
    try:
        pub_raw = _unb64(maker_pubkey)   # base64 of the raw 32-byte Solana pubkey
        sig_raw = _unb64(maker_sig)      # base64 of the raw 64-byte Ed25519 signature
    except (ValueError, base64.binascii.Error):
        raise ValueError("maker_pubkey/maker_sig not valid base64")
    if len(pub_raw) != 32 or len(sig_raw) != 64:
        raise ValueError("maker_pubkey must be 32 bytes, maker_sig 64 bytes")
    if not _ed25519.verify(sig_raw, device_pubkey_b64.encode(), pub_raw):
        raise ValueError("maker wallet signature does not verify over the device pubkey")
    return {"method": "wallet", "maker_pubkey": maker_pubkey}


def confirm_device(ctx: Context, pairing_token: str, code: str,
                   *, maker_pubkey: str | None = None, maker_sig: str | None = None,
                   now: float | None = None) -> tuple[int, dict]:
    """Operator-confirmed mutual code match → register the device. Token-gated.

    Optionally carries an additive Maker wallet-binding (AG-MAKER-BIND): the Maker's
    Solana/Privy wallet signs the device pubkey on the Observatory path; console/CLI
    paths omit it and bind by operator-token possession.
    """
    now = now if now is not None else time.time()
    pend = _load_valid_pending(ctx, pairing_token, now)
    if not pend:
        return 401, {"error": "invalid or expired pairing token"}
    if pend.get("state") != "submitted":
        return 409, {"error": f"cannot confirm in state '{pend.get('state')}'"}
    if str(code).strip() != str(pend.get("code6")):
        return 403, {"error": "code mismatch"}
    try:
        authorized_by = _maker_binding(pend["device_pubkey"], maker_pubkey, maker_sig)
    except ValueError as e:
        return 403, {"error": str(e)}
    devices = _read_json(_devices_path(ctx), [])
    devices = [d for d in devices if d.get("device_id") != pend["device_id"]]  # re-pair replaces
    devices.append({
        "device_id": pend["device_id"], "pubkey": pend["device_pubkey"],
        "label": pend.get("label", "phone"), "fingerprint": pend.get("fingerprint", ""),
        "paired_at": now, "authorized_by": authorized_by,
    })
    _write_json(_devices_path(ctx), devices)
    # clear the pending session (single-use)
    try:
        _pending_path(ctx).unlink()
    except OSError:
        pass
    return 200, {"ok": True, "device_id": pend["device_id"], "label": pend.get("label")}


# ── per-request signature verification (the auth gate) ───────────────────────
def _find_device(ctx: Context, device_id: str):
    for d in _read_json(_devices_path(ctx), []):
        if d.get("device_id") == device_id:
            return d
    return None


def registered_device_ids(ctx: Context) -> list[str]:
    """All paired device ids (for fan-out — e.g. HealthMonitor enqueues to every phone)."""
    return [d["device_id"] for d in _read_json(_devices_path(ctx), [])
            if d.get("device_id")]


def device_record(ctx: Context, device_id: str) -> dict | None:
    """Public registration view for the signed /console/device/me self-check.

    The app polls this (signed) after submit; a 200 means the operator confirmed
    the code-match and this device is now in devices.json. None ⇒ not registered.
    """
    d = _find_device(ctx, device_id)
    if not d:
        return None
    return {"device_id": d.get("device_id"), "label": d.get("label", "phone"),
            "paired_at": d.get("paired_at")}


def _prune_seen(now: float) -> None:
    for sig, exp in list(_seen_sigs.items()):
        if exp < now:
            del _seen_sigs[sig]


def verify_request_signature(ctx: Context, *, device_id: str, timestamp: str,
                             signature_b64: str, method: str, path: str, body: bytes,
                             now: float | None = None, skew: int = _SIG_SKEW) -> bool:
    """True iff a registered device's key signed this exact request, fresh + non-replayed."""
    now = now if now is not None else time.time()
    if not device_id or not timestamp or not signature_b64:
        return False
    try:
        ts = float(timestamp)
    except (TypeError, ValueError):
        return False
    if abs(now - ts) > skew:
        return False
    dev = _find_device(ctx, device_id)
    if not dev:
        return False
    _prune_seen(now)
    replay_key = f"{device_id}:{signature_b64}"
    if replay_key in _seen_sigs:
        return False
    try:
        pubkey = _unb64(dev["pubkey"])
        sig = _unb64(signature_b64)
    except (ValueError, KeyError, base64.binascii.Error):
        return False
    msg = canonical_request(method, path, str(timestamp), body_sha256_hex(body)).encode()
    if not _ed25519.verify(sig, msg, pubkey):
        return False
    _seen_sigs[replay_key] = now + skew
    return True


def _clear_caches() -> None:
    """Test hygiene — reset the in-memory replay cache."""
    _seen_sigs.clear()
