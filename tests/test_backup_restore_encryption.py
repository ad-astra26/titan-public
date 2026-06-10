"""Tests for rFP_backup_worker Phase 7.3 — restore-side decryption.

Covers:
  1. decrypt_from_manifest on a real ciphertext roundtrip (incl. plaintext_sha256
     verification invariant).
  2. decrypt_from_manifest passthrough for legacy records (algorithm="none").
  3. decrypt_from_manifest rejects unknown algorithm.
  4. load_keypair_bytes on a well-formed file returns (kp_bytes, pubkey).
  5. End-to-end: encrypt a tarball via backup_crypto (the unified encryption
     SoT), build the manifest stanza a backup record would hold, and decrypt it
     — simulates the live RebirthBackup.restore_personality_from_arweave path.
"""
import base64
import hashlib
import io
import json
import os
import tarfile

import pytest

from titan_hcl.logic.backup_crypto import (
    ALGORITHM_ID,
    decrypt_from_manifest,
    derive_backup_key,
    derive_master_key,
    encrypt_tarball,
    key_id,
    load_keypair_bytes,
)


TITAN_PUBKEY = "YOUR_TITAN_PUBKEY"


def _fake_kp(seed: int = 0x5E) -> bytes:
    return bytes([seed] * 32) + bytes([seed ^ 0xFF] * 32)


def _make_tarball_bytes(payload: bytes = b"restore me") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="restore.bin")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


# ────────────────────────────────────────────────────────────────────────────
# 1. Direct decrypt_from_manifest roundtrip
# ────────────────────────────────────────────────────────────────────────────

def test_decrypt_from_manifest_roundtrip(tmp_path):
    from titan_hcl.logic.backup_crypto import derive_backup_key, encrypt_tarball
    kp = _fake_kp()
    master = derive_master_key(kp, TITAN_PUBKEY)
    plaintext = _make_tarball_bytes(b"restore-payload")
    backup_id = hashlib.sha256(plaintext).hexdigest()[:16]
    bkey = derive_backup_key(master, backup_id, "personality")
    ct, iv, tag = encrypt_tarball(plaintext, bkey)

    manifest = {
        "algorithm": ALGORITHM_ID,
        "iv_b64": base64.b64encode(iv).decode(),
        "auth_tag_b64": base64.b64encode(tag).decode(),
        "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
        "backup_id": backup_id,
    }
    recovered = decrypt_from_manifest(ct, manifest, kp, TITAN_PUBKEY, "personality")
    assert recovered == plaintext
    assert hashlib.sha256(recovered).hexdigest() == manifest["plaintext_sha256"]


def test_decrypt_from_manifest_passthrough_on_legacy():
    """algorithm='none' (legacy pre-encryption record) → bytes returned as-is."""
    plaintext = b"legacy plaintext tarball"
    got = decrypt_from_manifest(
        plaintext, {"algorithm": "none"}, _fake_kp(), TITAN_PUBKEY, "personality")
    assert got == plaintext


def test_decrypt_from_manifest_empty_dict_treated_as_legacy():
    plaintext = b"legacy too"
    got = decrypt_from_manifest(
        plaintext, {}, _fake_kp(), TITAN_PUBKEY, "personality")
    assert got == plaintext


def test_decrypt_from_manifest_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="Unsupported"):
        decrypt_from_manifest(
            b"xx", {"algorithm": "ROT13", "iv_b64": "", "backup_id": "x"},
            _fake_kp(), TITAN_PUBKEY, "personality")


def test_decrypt_from_manifest_missing_backup_id():
    with pytest.raises(ValueError, match="backup_id"):
        decrypt_from_manifest(
            b"xx", {"algorithm": ALGORITHM_ID, "iv_b64": base64.b64encode(b"x"*12).decode()},
            _fake_kp(), TITAN_PUBKEY, "personality")


# ────────────────────────────────────────────────────────────────────────────
# 2. load_keypair_bytes
# ────────────────────────────────────────────────────────────────────────────

def test_load_keypair_bytes_valid(tmp_path):
    kp_path = tmp_path / "kp.json"
    kp_path.write_text(json.dumps(list(bytes(range(64)))))
    kp_bytes, pubkey = load_keypair_bytes(str(kp_path))
    assert len(kp_bytes) == 64
    assert isinstance(pubkey, str) and len(pubkey) > 0


def test_load_keypair_bytes_malformed_raises(tmp_path):
    kp_path = tmp_path / "bad.json"
    kp_path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="malformed"):
        load_keypair_bytes(str(kp_path))


# ────────────────────────────────────────────────────────────────────────────
# 3. End-to-end: cascade-encrypted tarball decrypts via manifest
# ────────────────────────────────────────────────────────────────────────────

def test_unified_encrypted_then_decrypt_via_manifest(tmp_path):
    """End-to-end: a tarball encrypted by the unified pipeline (backup_crypto —
    the live encryption SoT) decrypts via its captured manifest stanza. Mirrors
    the restore path: the record holds the encryption stanza, restore fetches
    the ciphertext bytes and reconstructs plaintext.
    """
    kp = _fake_kp()
    master = derive_master_key(kp, TITAN_PUBKEY)

    payload = b"end-to-end restore payload"
    plaintext = _make_tarball_bytes(payload)
    backup_id = hashlib.sha256(plaintext).hexdigest()[:16]
    bkey = derive_backup_key(master, backup_id, "personality")
    ct, iv, tag = encrypt_tarball(plaintext, bkey)  # ct = ciphertext_with_tag

    # The encryption stanza a backup record would hold (what restore reads).
    stanza = {
        "algorithm": ALGORITHM_ID,
        "key_id": key_id(backup_id, "personality"),
        "tier": "private",
        "iv_b64": base64.b64encode(iv).decode("ascii"),
        "auth_tag_b64": base64.b64encode(tag).decode("ascii"),
        "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
        "ciphertext_sha256": hashlib.sha256(ct).hexdigest(),
        "backup_id": backup_id,
    }

    # Simulate remote fetch: we already have the bytes. Decrypt via the stanza.
    recovered = decrypt_from_manifest(ct, stanza, kp, TITAN_PUBKEY, "personality")
    assert recovered == plaintext


# NOTE: the former `test_resurrection_phase_3_decrypts_then_extracts` was removed
# 2026-05-30. It exercised `resurrection.phase_3_rehydrate`, which W1.5
# (RFP_Titan_setup_release) deleted when Phase 2/3 were modernized to delegate to
# `backup_restore.restore_full` (commit 8d945912). The decrypt-then-extract
# capability it covered is now exercised against the live API by
# `test_cascade_encrypted_then_decrypt_via_manifest` (above) + the five
# `decrypt_from_manifest` tests; restore-side decryption rides through
# restore_full. No coverage lost.
