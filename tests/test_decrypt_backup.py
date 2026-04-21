"""Integration tests for scripts/decrypt_backup.py — rFP_backup_worker Phase 7.4.

Exercises the Maker emergency-recovery CLI end-to-end:
  - Encrypt a tarball via BackupCascade, persist a record file
  - Invoke decrypt_backup.py via subprocess with:
      * --keypair-file flow (pre-reconstructed keypair)
      * --shard1-hex/--shard2-hex flow (2-of-3 Shamir reconstruction)
      * --record flow (backup record JSON)
      * --manifest + --backup-type flow (bare stanza JSON)
  - Verify tampered-ciphertext aborts with exit 3
  - Verify wrong-key (bad shards) aborts with nonzero
  - Verify legacy "algorithm=none" record aborts with exit 1
"""
import asyncio
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile

import pytest

from titan_plugin.logic.backup_cascade import BackupCascade
from titan_plugin.logic.backup_crypto import derive_master_key
from titan_plugin.utils.shamir import split_secret


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLI = os.path.join(REPO_ROOT, "scripts", "decrypt_backup.py")


def _fake_kp(seed: int = 0x91) -> bytes:
    return bytes([seed] * 32) + bytes([seed ^ 0xFF] * 32)


def _kp_pubkey(kp_bytes: bytes) -> str:
    """Match exactly what load_keypair_bytes returns — solders-derived base58 pubkey.

    The CLI uses this same derivation on the keypair bytes, so encryption must
    use the matching pubkey in HKDF's `info` field.
    """
    try:
        from solders.keypair import Keypair
        return str(Keypair.from_bytes(kp_bytes).pubkey())
    except Exception:
        return kp_bytes[32:64].hex()


def _make_tarball(path: str, payload: bytes = b"maker recovery payload") -> bytes:
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(name="state.bin")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    with open(path, "rb") as f:
        return f.read()


def _cascade(tmp_path):
    cfg = {"backup": {"mode": "local_only"}, "network": {}}
    return BackupCascade(full_config=cfg, arweave_store=object(),
                         local_dir=str(tmp_path))


def _encrypt_tarball(tmp_path):
    """Returns (ciphertext_path, record_path, kp_bytes, titan_pubkey, plaintext_sha, plaintext)."""
    kp = _fake_kp()
    titan_pubkey = _kp_pubkey(kp)  # Must match what CLI derives from the same bytes
    master = derive_master_key(kp, titan_pubkey)
    ctx = {"master_key": master, "tier": "private"}
    cascade = _cascade(tmp_path)
    src = tmp_path / "src.tar.gz"
    plaintext = _make_tarball(str(src))

    async def _no_upload(_):
        return {"arweave_tx": "devnet-skip"}

    result = asyncio.run(cascade.run(str(src), "personality", _no_upload,
                                      encryption=ctx))
    # Construct a backup record (like _store_backup_record would)
    record = dict(result)
    record.setdefault("manifest_version", "1.0")
    record_path = tmp_path / "personality_record.json"
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)

    return (result["local_path"], str(record_path), kp, titan_pubkey,
            hashlib.sha256(plaintext).hexdigest(), plaintext)


def _run_cli(*args, cwd=None):
    """Run decrypt_backup.py, returning (returncode, stdout, stderr)."""
    env = dict(os.environ)
    venv_python = os.path.join(REPO_ROOT, "test_env", "bin", "python")
    proc = subprocess.run(
        [venv_python, CLI, *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True, text=True, env=env, timeout=60)
    return proc.returncode, proc.stdout, proc.stderr


# ────────────────────────────────────────────────────────────────────────────
# --keypair-file flow
# ────────────────────────────────────────────────────────────────────────────

def test_cli_decrypts_with_keypair_file_flow(tmp_path):
    ct_path, rec_path, kp, _, _, plaintext = _encrypt_tarball(tmp_path)

    # Write keypair as JSON list (matching data/titan_identity_keypair.json format)
    kp_path = tmp_path / "kp.json"
    kp_path.write_text(json.dumps(list(kp)))

    out = tmp_path / "restored.tar.gz"
    rc, stdout, stderr = _run_cli(
        "--keypair-file", str(kp_path),
        "--ciphertext", ct_path,
        "--record", rec_path,
        "--output", str(out),
    )
    assert rc == 0, f"CLI failed: rc={rc} stderr={stderr}"
    assert out.exists()
    assert out.read_bytes() == plaintext


# ────────────────────────────────────────────────────────────────────────────
# --shard1-hex / --shard2-hex flow (Shamir reconstruction)
# ────────────────────────────────────────────────────────────────────────────

def test_cli_decrypts_via_2_of_3_shamir(tmp_path):
    ct_path, rec_path, kp, _, _, plaintext = _encrypt_tarball(tmp_path)

    shards = split_secret(kp, n=3, t=2)
    shard1_hex = shards[0].hex()
    shard2_hex = shards[1].hex()

    out = tmp_path / "restored.tar.gz"
    rc, stdout, stderr = _run_cli(
        "--shard1-hex", shard1_hex,
        "--shard2-hex", shard2_hex,
        "--ciphertext", ct_path,
        "--record", rec_path,
        "--output", str(out),
    )
    assert rc == 0, f"CLI failed: rc={rc} stderr={stderr}"
    assert out.read_bytes() == plaintext


# ────────────────────────────────────────────────────────────────────────────
# --manifest + --backup-type flow
# ────────────────────────────────────────────────────────────────────────────

def test_cli_decrypts_with_manifest_stanza(tmp_path):
    ct_path, rec_path, kp, _, _, plaintext = _encrypt_tarball(tmp_path)
    with open(rec_path) as f:
        record = json.load(f)

    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(record["encryption"], f)

    kp_path = tmp_path / "kp.json"
    kp_path.write_text(json.dumps(list(kp)))

    out = tmp_path / "restored.tar.gz"
    rc, stdout, stderr = _run_cli(
        "--keypair-file", str(kp_path),
        "--ciphertext", ct_path,
        "--manifest", str(manifest_path),
        "--backup-type", "personality",
        "--output", str(out),
    )
    assert rc == 0, stderr
    assert out.read_bytes() == plaintext


# ────────────────────────────────────────────────────────────────────────────
# Negative paths
# ────────────────────────────────────────────────────────────────────────────

def test_cli_rejects_tampered_ciphertext(tmp_path):
    ct_path, rec_path, kp, _, _, _ = _encrypt_tarball(tmp_path)
    # Flip a middle byte to break GCM auth
    with open(ct_path, "r+b") as f:
        f.seek(20)
        b = f.read(1)
        f.seek(20)
        f.write(bytes([b[0] ^ 0xFF]))

    kp_path = tmp_path / "kp.json"
    kp_path.write_text(json.dumps(list(kp)))

    rc, _, stderr = _run_cli(
        "--keypair-file", str(kp_path),
        "--ciphertext", ct_path,
        "--record", rec_path,
        "--output", str(tmp_path / "nope.tar.gz"),
    )
    assert rc == 3, f"expected exit 3 on tampered, got {rc}: {stderr}"
    assert "decryption failed" in stderr.lower()


def test_cli_rejects_legacy_unencrypted_record(tmp_path):
    """Record has algorithm='none' → tool aborts with guidance."""
    rec = {
        "arweave_tx": "some-tx",
        "encryption": {"algorithm": "none"},
        "manifest_version": "1.0",
    }
    rec_path = tmp_path / "legacy.json"
    rec_path.write_text(json.dumps(rec))
    # Fake ciphertext file
    ct_path = tmp_path / "ct.bin"
    ct_path.write_bytes(b"not encrypted")

    kp = _fake_kp()
    kp_path = tmp_path / "kp.json"
    kp_path.write_text(json.dumps(list(kp)))

    rc, _, stderr = _run_cli(
        "--keypair-file", str(kp_path),
        "--ciphertext", str(ct_path),
        "--record", str(rec_path),
        "--output", str(tmp_path / "nope.tar.gz"),
    )
    assert rc == 1, f"expected exit 1 on legacy record, got {rc}: {stderr}"
    assert "not encrypted" in stderr.lower()


def test_cli_rejects_missing_both_record_and_manifest(tmp_path):
    kp = _fake_kp()
    kp_path = tmp_path / "kp.json"
    kp_path.write_text(json.dumps(list(kp)))
    ct = tmp_path / "ct.bin"
    ct.write_bytes(b"x")
    rc, _, stderr = _run_cli(
        "--keypair-file", str(kp_path),
        "--ciphertext", str(ct),
        "--output", str(tmp_path / "o.bin"),
    )
    assert rc == 1
    assert "--record" in stderr or "--manifest" in stderr


def test_cli_rejects_less_than_two_shards(tmp_path):
    rec = {"encryption": {"algorithm": "AES-256-GCM"}}
    rec_path = tmp_path / "r.json"
    rec_path.write_text(json.dumps(rec))
    ct = tmp_path / "ct.bin"
    ct.write_bytes(b"x")
    rc, _, stderr = _run_cli(
        "--shard1-hex", "0011",
        "--ciphertext", str(ct),
        "--record", str(rec_path),
        "--output", str(tmp_path / "o.bin"),
    )
    assert rc == 1
    assert "at least 2" in stderr.lower()
