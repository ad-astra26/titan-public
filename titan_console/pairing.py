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
def mint_pairing(ctx: Context, *, ttl: int = _TTL_DEFAULT,
                 public_url: str | None = None, now: float | None = None) -> tuple[int, dict]:
    """Operator-triggered: mint a single-use QR pairing token. (token-gated route)"""
    now = now if now is not None else time.time()
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
    if public_url:
        payload["endpoint_url"] = public_url
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
    return 200, out


def confirm_device(ctx: Context, pairing_token: str, code: str,
                   *, now: float | None = None) -> tuple[int, dict]:
    """Operator-confirmed mutual code match → register the device. Token-gated."""
    now = now if now is not None else time.time()
    pend = _load_valid_pending(ctx, pairing_token, now)
    if not pend:
        return 401, {"error": "invalid or expired pairing token"}
    if pend.get("state") != "submitted":
        return 409, {"error": f"cannot confirm in state '{pend.get('state')}'"}
    if str(code).strip() != str(pend.get("code6")):
        return 403, {"error": "code mismatch"}
    devices = _read_json(_devices_path(ctx), [])
    devices = [d for d in devices if d.get("device_id") != pend["device_id"]]  # re-pair replaces
    devices.append({
        "device_id": pend["device_id"], "pubkey": pend["device_pubkey"],
        "label": pend.get("label", "phone"), "fingerprint": pend.get("fingerprint", ""),
        "paired_at": now,
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
