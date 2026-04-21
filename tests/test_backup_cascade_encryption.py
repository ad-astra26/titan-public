"""Tests for rFP_backup_worker Phase 7.2 — encryption wired into BackupCascade.

Covers:
  1. run() — encryption opt-out (no context) behaves exactly as before
  2. run() — encryption opt-in: tarball on disk + local copy are ciphertext,
     archive_hash is the ciphertext hash, `encryption` stanza is returned
  3. run_bytes() — same for TimeChain in-memory flow
  4. Roundtrip: ciphertext pulled from local path decrypts back to original tarball
  5. build_encryption_context_from_config honors encryption_enabled flag
  6. Manifest version bump in _store_backup_record
"""
import asyncio
import base64
import hashlib
import io
import json
import os
import tarfile

import pytest

from titan_plugin.logic.backup_cascade import BackupCascade
from titan_plugin.logic.backup_crypto import (
    build_encryption_context_from_config,
    decrypt_tarball,
    derive_backup_key,
    derive_master_key,
)


TITAN_PUBKEY = "J1cdk4f1qZWTV1j8MSWAkPJ6Nqg63AXBn8d5JbaGLNoG"


def _make_tarball(path: str, payload: bytes = b"sovereign state v1") -> str:
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(name="state.bin")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return path


def _fake_master() -> bytes:
    return derive_master_key(bytes([0x7A] * 32) + bytes([0x85] * 32), TITAN_PUBKEY)


def _fake_encryption_context(tier: str = "private") -> dict:
    return {"master_key": _fake_master(), "tier": tier}


def _cascade(tmp_path, mode="local_only", arweave_store=object()):
    cfg = {
        "backup": {"mode": mode},
        "network": {"wallet_keypair_path": "data/titan_identity_keypair.json"},
    }
    return BackupCascade(full_config=cfg, arweave_store=arweave_store,
                         local_dir=str(tmp_path))


# ────────────────────────────────────────────────────────────────────────────
# 1. Opt-out — backward compatibility
# ────────────────────────────────────────────────────────────────────────────

def test_run_without_encryption_is_plaintext(tmp_path):
    c = _cascade(tmp_path, mode="local_only")
    src = _make_tarball(str(tmp_path / "src.tar.gz"))

    async def _no_upload(_):
        return {"arweave_tx": "devnet-skip"}

    result = asyncio.run(c.run(src, "personality", _no_upload))
    assert result is not None
    assert result["mode"] == "local_only"
    assert "encryption" not in result
    # Local file is the original plaintext tarball
    with open(result["local_path"], "rb") as f:
        assert f.read() == open(src, "rb").read()


def test_run_bytes_without_encryption_is_plaintext(tmp_path):
    c = _cascade(tmp_path, mode="local_only")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="tc.bin")
        payload = b"timechain blocks"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    tarball_bytes = buf.getvalue()
    h = hashlib.sha256(tarball_bytes).hexdigest()

    async def _no_upload(_):
        return {"arweave_tx": "devnet-skip"}

    result = asyncio.run(c.run_bytes(tarball_bytes, h, "timechain", _no_upload,
                                      ext="tar.gz"))
    assert result["mode"] == "local_only"
    assert "encryption" not in result
    assert result["archive_hash"] == h
    with open(result["local_path"], "rb") as f:
        assert f.read() == tarball_bytes


# ────────────────────────────────────────────────────────────────────────────
# 2. Opt-in — encryption applied (path-based)
# ────────────────────────────────────────────────────────────────────────────

def test_run_with_encryption_produces_ciphertext_local(tmp_path):
    c = _cascade(tmp_path, mode="local_only")
    src = _make_tarball(str(tmp_path / "src.tar.gz"))
    plaintext = open(src, "rb").read()

    async def _no_upload(_):
        return {"arweave_tx": "devnet-skip"}

    result = asyncio.run(c.run(src, "personality", _no_upload,
                                encryption=_fake_encryption_context()))
    assert result is not None
    assert "encryption" in result
    enc = result["encryption"]
    assert enc["algorithm"] == "AES-256-GCM"
    assert enc["tier"] == "private"
    assert enc["plaintext_sha256"] == hashlib.sha256(plaintext).hexdigest()

    # Local copy is ciphertext (not plaintext); archive_hash matches ciphertext
    ciphertext_on_disk = open(result["local_path"], "rb").read()
    assert ciphertext_on_disk != plaintext
    assert hashlib.sha256(ciphertext_on_disk).hexdigest() == result["archive_hash"]
    assert result["archive_hash"] == enc["ciphertext_sha256"]


def test_run_encryption_roundtrip_restore(tmp_path):
    """The restore-side invariant: given (local ciphertext, manifest iv, key), decrypt → plaintext."""
    c = _cascade(tmp_path, mode="local_only")
    src = _make_tarball(str(tmp_path / "src.tar.gz"), payload=b"long payload " * 500)
    plaintext = open(src, "rb").read()
    ctx = _fake_encryption_context()

    async def _no_upload(_):
        return {"arweave_tx": "devnet-skip"}

    result = asyncio.run(c.run(src, "personality", _no_upload, encryption=ctx))
    enc = result["encryption"]

    # Reconstruct what the restore path would do
    iv = base64.b64decode(enc["iv_b64"])
    bkey = derive_backup_key(ctx["master_key"], enc["backup_id"], "personality")
    ct_on_disk = open(result["local_path"], "rb").read()
    recovered = decrypt_tarball(ct_on_disk, iv, bkey)
    assert recovered == plaintext


# ────────────────────────────────────────────────────────────────────────────
# 3. Opt-in — encryption applied (bytes-based / TimeChain)
# ────────────────────────────────────────────────────────────────────────────

def test_run_bytes_with_encryption_local_only(tmp_path):
    c = _cascade(tmp_path, mode="local_only")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="tc.bin")
        payload = b"chain payload"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    tarball_bytes = buf.getvalue()
    plaintext_hash = hashlib.sha256(tarball_bytes).hexdigest()
    ctx = _fake_encryption_context()

    async def _no_upload(_):
        return {"arweave_tx": "devnet-skip"}

    result = asyncio.run(c.run_bytes(tarball_bytes, plaintext_hash, "timechain",
                                      _no_upload, ext="tar.gz",
                                      encryption=ctx))
    assert "encryption" in result
    assert result["encryption"]["algorithm"] == "AES-256-GCM"
    assert result["archive_hash"] != plaintext_hash, "ciphertext hash must differ"
    assert result["archive_hash"] == result["encryption"]["ciphertext_sha256"]
    # Local file ends in .enc
    assert result["local_path"].endswith(".tar.gz.enc")
    # And decrypts back to original
    iv = base64.b64decode(result["encryption"]["iv_b64"])
    bkey = derive_backup_key(ctx["master_key"],
                               result["encryption"]["backup_id"], "timechain")
    recovered = decrypt_tarball(open(result["local_path"], "rb").read(), iv, bkey)
    assert recovered == tarball_bytes


# ────────────────────────────────────────────────────────────────────────────
# 4. build_encryption_context_from_config
# ────────────────────────────────────────────────────────────────────────────

def test_context_none_when_disabled():
    cfg = {"backup": {"encryption_enabled": False}, "network": {}}
    assert build_encryption_context_from_config(cfg) is None


def test_context_none_when_missing_section():
    assert build_encryption_context_from_config({}) is None


def test_context_returns_none_when_enabled_but_keypair_absent(tmp_path):
    """Fail-open: encryption enabled but no keypair → None → backup proceeds unencrypted."""
    cfg = {
        "backup": {"encryption_enabled": True, "encryption_tier": "private"},
        "network": {"wallet_keypair_path": str(tmp_path / "does_not_exist.json")},
    }
    assert build_encryption_context_from_config(cfg) is None


def test_context_returns_none_when_keypair_malformed(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([1, 2, 3]))  # not 64 bytes
    cfg = {
        "backup": {"encryption_enabled": True},
        "network": {"wallet_keypair_path": str(bad)},
    }
    assert build_encryption_context_from_config(cfg) is None


def test_context_returns_master_when_enabled_with_valid_keypair(tmp_path):
    """End-to-end: give a well-formed 64-byte keypair file; context should populate."""
    kp = tmp_path / "kp.json"
    kp.write_text(json.dumps(list(bytes(range(64)))))  # 0..63 bytes
    cfg = {
        "backup": {"encryption_enabled": True, "encryption_tier": "private"},
        "network": {"wallet_keypair_path": str(kp)},
    }
    ctx = build_encryption_context_from_config(cfg)
    assert ctx is not None
    assert isinstance(ctx["master_key"], bytes) and len(ctx["master_key"]) == 32
    assert ctx["tier"] == "private"


# ────────────────────────────────────────────────────────────────────────────
# 5. Cascade fail-path still returns encryption stanza (for diagnostics)
# ────────────────────────────────────────────────────────────────────────────

def test_cascade_upload_failure_still_returns_encryption_stanza(tmp_path):
    """If S5 upload returns None, the result should still record the encryption
    metadata so the operator can diagnose without guessing."""
    c = _cascade(tmp_path, mode="mainnet_arweave")
    src = _make_tarball(str(tmp_path / "src.tar.gz"))

    async def _fail_upload(_):
        return None

    # mainnet_arweave mode but no Irys keypair → S4 short-circuits to low-balance
    # fallback path. That path should also carry encryption stanza.
    result = asyncio.run(c.run(src, "personality", _fail_upload,
                                encryption=_fake_encryption_context()))
    assert result is not None
    # Either S4 short-circuit (local_fallback_low_balance) or S5 fail — both carry encryption
    assert "encryption" in result
