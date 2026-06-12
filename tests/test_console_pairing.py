"""Tests for QR mutual-pairing + Ed25519 request signing (RFP titan-app §7.1).

Proves: (1) the vendored Ed25519 interops with STANDARD Ed25519 (what Android
Keystore emits); (2) code6 matches the Kotlin vector; (3) the full pairing
lifecycle + every rejection path (wrong code / expired / bad sig / unknown
device / replay); (4) the dispatch auth gate (operator token OR device sig).
"""
import base64

import pytest

from titan_console import _ed25519, pairing
from titan_console.agent import dispatch
from titan_console.context import Context


def _ctx(tmp_path):
    # _titan_dir → secrets_path.parent, so all pairing state lands in tmp_path
    return Context(install_root=tmp_path, titan_id="T1",
                   secrets_path=tmp_path / "secrets.toml")


def _b64(b):
    return base64.b64encode(b).decode()


@pytest.fixture(autouse=True)
def _clear():
    pairing._clear_caches()
    yield
    pairing._clear_caches()


# ── vendored Ed25519 correctness ─────────────────────────────────────────────
def test_ed25519_roundtrip_and_tamper():
    seed, pub = _ed25519.keygen()
    msg = b"are you there?"
    sig = _ed25519.sign(msg, seed)
    assert _ed25519.verify(sig, msg, pub) is True
    assert _ed25519.verify(sig, b"different", pub) is False
    assert _ed25519.verify(bytes([sig[0] ^ 1]) + sig[1:], msg, pub) is False


def test_ed25519_interop_with_standard():
    """Vendored impl ↔ the `cryptography` library (== what Android Keystore emits)."""
    ed = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    from cryptography.hazmat.primitives.serialization import (
        Encoding, NoEncryption, PrivateFormat, PublicFormat,
    )
    msg = b"hello titan"
    # standard signs → ours verifies (the real Android → Console Agent path)
    sk = ed.Ed25519PrivateKey.generate()
    seed = sk.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    pub = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    assert _ed25519.verify(sk.sign(msg), msg, pub) is True
    # our pubkey derivation matches standard
    assert _ed25519.publickey(seed) == pub
    # ours signs → standard verifies
    my_seed, my_pub = _ed25519.keygen()
    ed.Ed25519PublicKey.from_public_bytes(my_pub).verify(_ed25519.sign(msg, my_seed), msg)


# ── code6 cross-language contract ────────────────────────────────────────────
def test_code6_matches_kotlin_vector():
    assert pairing.code6(b"TESTTOKEN", b"PUBKEY") == "028241"
    assert pairing.code6(b"TESTTOKEN", b"PUBKEY2") == "635194"


# ── pairing lifecycle ────────────────────────────────────────────────────────
def _mint_and_submit(ctx, t0=1000.0):
    _, payload = pairing.mint_pairing(ctx, now=t0)
    token = payload["pairing_token"]
    seed, pub = _ed25519.keygen()
    s, r = pairing.submit_device(
        ctx, {"pairing_token": token, "device_pubkey": _b64(pub),
              "device_id": "dev-1", "label": "Maker phone"}, now=t0 + 1)
    return token, seed, pub, s, r


def test_pairing_happy_path(tmp_path):
    ctx = _ctx(tmp_path)
    token, seed, pub, s, r = _mint_and_submit(ctx)
    assert s == 200 and r["awaiting_confirm"]
    expected = pairing.code6(base64.b64decode(token), pub)
    st, status = pairing.pair_status(ctx, token, now=1002)
    assert status["state"] == "submitted" and status["code6"] == expected
    sc, conf = pairing.confirm_device(ctx, token, expected, now=1003)
    assert sc == 200 and conf["device_id"] == "dev-1"
    # device persisted; pending cleared (single-use)
    assert pairing._find_device(ctx, "dev-1")["pubkey"] == _b64(pub)
    assert not pairing._pending_path(ctx).exists()


def test_confirm_wrong_code_rejected(tmp_path):
    ctx = _ctx(tmp_path)
    token, *_ = _mint_and_submit(ctx)
    s, _ = pairing.confirm_device(ctx, token, "000000", now=1003)
    assert s == 403
    assert pairing._find_device(ctx, "dev-1") is None


def test_expired_token_rejected(tmp_path):
    ctx = _ctx(tmp_path)
    _, payload = pairing.mint_pairing(ctx, ttl=90, now=1000)
    seed, pub = _ed25519.keygen()
    s, _ = pairing.submit_device(
        ctx, {"pairing_token": payload["pairing_token"], "device_pubkey": _b64(pub),
              "device_id": "dev-1"}, now=1000 + 91)  # past TTL
    assert s == 401


def test_cannot_confirm_unsubmitted(tmp_path):
    ctx = _ctx(tmp_path)
    _, payload = pairing.mint_pairing(ctx, now=1000)
    s, _ = pairing.confirm_device(ctx, payload["pairing_token"], "000000", now=1001)
    assert s == 409  # minted but not submitted


# ── request signature verification ───────────────────────────────────────────
def _register_signed_device(ctx, t0=1000.0):
    token, seed, pub, *_ = _mint_and_submit(ctx, t0)
    code = pairing.code6(base64.b64decode(token), pub)
    pairing.confirm_device(ctx, token, code, now=t0 + 2)
    return seed, pub


def _sign_request(seed, method, path, ts, body=b""):
    msg = pairing.canonical_request(method, path, str(ts),
                                    pairing.body_sha256_hex(body)).encode()
    return _b64(_ed25519.sign(msg, seed))


def test_valid_device_signature_accepted(tmp_path):
    ctx = _ctx(tmp_path)
    seed, _ = _register_signed_device(ctx)
    sig = _sign_request(seed, "POST", "/console/chat", 2000, b'{"message":"hi"}')
    assert pairing.verify_request_signature(
        ctx, device_id="dev-1", timestamp="2000", signature_b64=sig,
        method="POST", path="/console/chat", body=b'{"message":"hi"}', now=2000) is True


def test_unknown_device_rejected(tmp_path):
    ctx = _ctx(tmp_path)
    seed, _ = _register_signed_device(ctx)
    sig = _sign_request(seed, "POST", "/console/chat", 2000)
    assert pairing.verify_request_signature(
        ctx, device_id="ghost", timestamp="2000", signature_b64=sig,
        method="POST", path="/console/chat", body=b"", now=2000) is False


def test_bad_signature_rejected(tmp_path):
    ctx = _ctx(tmp_path)
    _register_signed_device(ctx)
    bad = _b64(b"\x00" * 64)
    assert pairing.verify_request_signature(
        ctx, device_id="dev-1", timestamp="2000", signature_b64=bad,
        method="POST", path="/console/chat", body=b"", now=2000) is False


def test_stale_timestamp_rejected(tmp_path):
    ctx = _ctx(tmp_path)
    seed, _ = _register_signed_device(ctx)
    sig = _sign_request(seed, "POST", "/console/chat", 2000)
    assert pairing.verify_request_signature(
        ctx, device_id="dev-1", timestamp="2000", signature_b64=sig,
        method="POST", path="/console/chat", body=b"", now=2000 + 9999) is False


def test_replay_rejected(tmp_path):
    ctx = _ctx(tmp_path)
    seed, _ = _register_signed_device(ctx)
    sig = _sign_request(seed, "POST", "/console/chat", 2000)
    kw = dict(device_id="dev-1", timestamp="2000", signature_b64=sig,
              method="POST", path="/console/chat", body=b"", now=2000)
    assert pairing.verify_request_signature(ctx, **kw) is True   # first ok
    assert pairing.verify_request_signature(ctx, **kw) is False  # replay


# ── dispatch auth gate ───────────────────────────────────────────────────────
def test_dispatch_operator_route_requires_token(tmp_path):
    ctx = Context(install_root=tmp_path, titan_id="T1",
                  secrets_path=tmp_path / "secrets.toml", token="OP-SECRET")
    # pair/start without the operator token → 401
    s, _ = dispatch(ctx, "POST", "/console/pair/start", {}, b"{}", {})
    assert s == 401
    # with it → 200 (mints)
    s, payload = dispatch(ctx, "POST", "/console/pair/start", {}, b"{}",
                          {"x-console-token": "OP-SECRET"})
    assert s == 200 and "pairing_token" in payload


# ── /console/device/me self-check + operator pairing page ───────────────────
def test_device_record_present_and_absent(tmp_path):
    ctx = _ctx(tmp_path)
    _register_signed_device(ctx)
    rec = pairing.device_record(ctx, "dev-1")
    assert rec is not None and rec["device_id"] == "dev-1" and rec["label"] == "Maker phone"
    assert pairing.device_record(ctx, "ghost") is None


def test_device_me_requires_valid_device_signature(tmp_path):
    import time as _t
    ctx = _ctx(tmp_path)
    seed, _ = _register_signed_device(ctx)
    ts = str(int(_t.time()))
    sig = _sign_request(seed, "GET", "/console/device/me", ts, b"")
    s, rec = dispatch(ctx, "GET", "/console/device/me", {}, b"",
                      {"x-device-id": "dev-1", "x-timestamp": ts, "x-signature": sig})
    assert s == 200 and rec["device_id"] == "dev-1"
    # unsigned → 401 even on an open (no-token) localhost bind
    s2, _ = dispatch(ctx, "GET", "/console/device/me", {}, b"", {})
    assert s2 == 401


def test_pair_page_is_served(tmp_path):
    s, body = dispatch(_ctx(tmp_path), "GET", "/console/pair", {}, b"", {})
    assert s == 200 and isinstance(body, (bytes, bytearray))
    assert b"Pair your phone" in body


def test_dispatch_device_signed_mutation_bypasses_token(tmp_path):
    import time as _t
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]  # has scripts/setup_titan for config_api
    ctx = Context(install_root=repo_root, titan_id="T1",
                  secrets_path=tmp_path / "secrets.toml", token="OP-SECRET")
    seed, _ = _register_signed_device(ctx)  # device + state under tmp_path (secrets_path)
    body = b'{"key":"nonexistent.test_key","value":"x"}'  # no such key → no file mutation
    ts = str(int(_t.time()))  # fresh ts so the gate's real-clock skew check passes
    sig = _sign_request(seed, "POST", "/console/config/set", ts, body)
    # no operator token supplied — ONLY the device signature
    s, _ = dispatch(ctx, "POST", "/console/config/set", {}, body,
                    {"x-device-id": "dev-1", "x-timestamp": ts, "x-signature": sig})
    assert s != 401  # device signature alone satisfied the gate
