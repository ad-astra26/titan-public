"""Tests for rFP_backup_worker Phase 7.3 — restore-side decryption.

Covers:
  1. decrypt_from_manifest on a real ciphertext roundtrip (incl. plaintext_sha256
     verification invariant).
  2. decrypt_from_manifest passthrough for legacy records (algorithm="none").
  3. decrypt_from_manifest rejects unknown algorithm.
  4. load_keypair_bytes on a well-formed file returns (kp_bytes, pubkey).
  5. End-to-end: encrypt a tarball via cascade, then look up the record that
     cascade-enriched pipeline would produce and decrypt it — simulates the live
     RebirthBackup.restore_personality_from_arweave path.
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
    ALGORITHM_ID,
    decrypt_from_manifest,
    derive_master_key,
    load_keypair_bytes,
)


TITAN_PUBKEY = "J1cdk4f1qZWTV1j8MSWAkPJ6Nqg63AXBn8d5JbaGLNoG"


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
    from titan_plugin.logic.backup_crypto import derive_backup_key, encrypt_tarball
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

def test_cascade_encrypted_then_decrypt_via_manifest(tmp_path):
    """Simulates RebirthBackup.restore_personality_from_arweave flow:
    record was created by cascade (captures encryption stanza), restore path
    reads the record and reconstructs plaintext from the Arweave bytes.
    """
    cfg = {"backup": {"mode": "local_only"}, "network": {}}
    cascade = BackupCascade(full_config=cfg, arweave_store=object(),
                             local_dir=str(tmp_path))
    kp = _fake_kp()
    master = derive_master_key(kp, TITAN_PUBKEY)
    ctx = {"master_key": master, "tier": "private"}

    src = tmp_path / "src.tar.gz"
    payload = b"end-to-end restore payload"
    with open(src, "wb") as f:
        f.write(_make_tarball_bytes(payload))
    plaintext = src.read_bytes()

    async def _no_upload(_):
        return {"arweave_tx": "devnet-skip"}

    result = asyncio.run(cascade.run(str(src), "personality", _no_upload,
                                      encryption=ctx))
    # Local copy is ciphertext
    assert "encryption" in result
    ciphertext_on_disk = open(result["local_path"], "rb").read()

    # Simulate remote fetch: we already have the bytes. Use the record's
    # encryption stanza (what backup records would hold) to decrypt.
    recovered = decrypt_from_manifest(
        ciphertext_on_disk, result["encryption"], kp, TITAN_PUBKEY, "personality")
    assert recovered == plaintext


# ────────────────────────────────────────────────────────────────────────────
# 4. phase_3_rehydrate decryption integration (resurrection.py)
# ────────────────────────────────────────────────────────────────────────────

def test_resurrection_phase_3_decrypts_then_extracts(tmp_path, monkeypatch):
    """resurrection.py phase_3_rehydrate must decrypt the archive in-place
    before tarfile.open is attempted. Exercise with a fresh encrypted archive."""
    import scripts.resurrection as res
    from titan_plugin.logic.backup_crypto import derive_backup_key, encrypt_tarball

    kp = _fake_kp()
    master = derive_master_key(kp, TITAN_PUBKEY)
    plaintext_tarball = _make_tarball_bytes(b"resurrection payload")
    backup_id = hashlib.sha256(plaintext_tarball).hexdigest()[:16]
    bkey = derive_backup_key(master, backup_id, "personality")
    ct, iv, tag = encrypt_tarball(plaintext_tarball, bkey)

    archive_path = tmp_path / "enc.tar.gz"
    archive_path.write_bytes(ct)

    manifest = {
        "algorithm": ALGORITHM_ID,
        "iv_b64": base64.b64encode(iv).decode(),
        "auth_tag_b64": base64.b64encode(tag).decode(),
        "plaintext_sha256": hashlib.sha256(plaintext_tarball).hexdigest(),
        "backup_id": backup_id,
    }

    # Stub out the filesystem-mutating + re-encrypt sections so the test only
    # covers the decryption + extraction phase.
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    from unittest import mock
    with mock.patch("titan_plugin.utils.crypto.encrypt_for_machine", return_value=b""):
        # tar extract doesn't write anything we care about (payload is "restore.bin" → data/restore.bin)
        res.phase_3_rehydrate(str(archive_path), kp,
                                encryption_manifest=manifest,
                                titan_pubkey=TITAN_PUBKEY,
                                backup_type="personality")
    # After decryption, archive was rewritten with plaintext; phase_3 deletes it at the end.
    assert not archive_path.exists(), "archive should be cleaned up by phase_3"
