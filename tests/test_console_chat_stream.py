"""app-chat-streaming — /console/chat/stream SSE relay auth gate.

Verifies _stream_authorize() mirrors dispatch()'s AG4/AD-5 gate: localhost open,
remote requires a device signature OR operator token, and device_authed (the
cryptographically-verified Maker) is surfaced so the app-channel maker_bond (D.2)
is relayed only then.

Run: python -m pytest tests/test_console_chat_stream.py -v -p no:anchorpy
"""
from titan_console import pairing
from titan_console.agent import _stream_authorize
from titan_console.context import Context

PATH = "/console/chat/stream"


def _ctx(tmp_path, *, token=None, internal_key="k"):
    return Context(install_root=tmp_path, titan_id="T1",
                   token=token, internal_key=internal_key)


def test_localhost_open_no_device(tmp_path):
    authorized, device = _stream_authorize(_ctx(tmp_path), PATH, b"{}", {}, is_local=True)
    assert authorized is True
    assert device is False          # no signature → no maker_bond


def test_remote_denied_without_auth(tmp_path):
    authorized, device = _stream_authorize(
        _ctx(tmp_path, token="secret"), PATH, b"{}", {}, is_local=False)
    assert authorized is False
    assert device is False


def test_remote_operator_token_authorizes_without_bond(tmp_path):
    hdr = {"x-console-token": "secret"}
    authorized, device = _stream_authorize(
        _ctx(tmp_path, token="secret"), PATH, b"{}", hdr, is_local=False)
    assert authorized is True
    assert device is False          # operator token ≠ cryptographic Maker → NO bond


def test_remote_bad_operator_token_denied(tmp_path):
    hdr = {"x-console-token": "wrong"}
    authorized, _ = _stream_authorize(
        _ctx(tmp_path, token="secret"), PATH, b"{}", hdr, is_local=False)
    assert authorized is False


def test_remote_device_signature_authorizes_and_fires_bond(tmp_path, monkeypatch):
    monkeypatch.setattr(pairing, "verify_request_signature", lambda *a, **k: True)
    hdr = {"x-device-id": "dev1", "x-timestamp": "t", "x-signature": "sig"}
    authorized, device = _stream_authorize(
        _ctx(tmp_path, token="secret"), PATH, b'{"message":"hi"}', hdr, is_local=False)
    assert authorized is True
    assert device is True           # verified Maker → maker_bond relayed


def test_invalid_device_signature_not_authed(tmp_path, monkeypatch):
    monkeypatch.setattr(pairing, "verify_request_signature", lambda *a, **k: False)
    hdr = {"x-device-id": "dev1", "x-timestamp": "t", "x-signature": "bad"}
    authorized, device = _stream_authorize(
        _ctx(tmp_path, token="secret"), PATH, b"{}", hdr, is_local=False)
    assert authorized is False
    assert device is False
