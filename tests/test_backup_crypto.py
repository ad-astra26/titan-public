"""Tests for titan_plugin/logic/backup_crypto.py — rFP_backup_worker Phase 7.1.

Covers:
  1. Roundtrip encrypt/decrypt
  2. HKDF determinism (same inputs → same keys)
  3. Domain separation (different pubkey OR different backup_id/type → different key)
  4. Wrong key / tampered ciphertext / tampered IV → InvalidTag
  5. Shamir-reconstruction equivalence: splitting keypair into shards and recombining
     yields the same master key (the primary Maker-recovery invariant).
  6. Input validation on bad sizes / empty args
"""
import os
import pytest

from cryptography.exceptions import InvalidTag

from titan_plugin.logic.backup_crypto import (
    ALGORITHM_ID,
    IV_LEN,
    KEY_LEN,
    MASTER_KEY_SALT,
    PER_BACKUP_KEY_INFO_PREFIX,
    SEED_LEN,
    TAG_LEN,
    decrypt_tarball,
    derive_backup_key,
    derive_master_key,
    encrypt_tarball,
    key_id,
)
from titan_plugin.utils.shamir import combine_shares, split_secret


TITAN_PUBKEY = "J1cdk4f1qZWTV1j8MSWAkPJ6Nqg63AXBn8d5JbaGLNoG"


def _fake_keypair(seed: int = 0x42) -> bytes:
    """Deterministic 64-byte keypair (seed || pub placeholder) for stable tests."""
    return bytes([seed] * SEED_LEN) + bytes([seed ^ 0xFF] * SEED_LEN)


# ────────────────────────────────────────────────────────────────────────────
# 1. Roundtrip
# ────────────────────────────────────────────────────────────────────────────

def test_roundtrip_encrypt_decrypt_preserves_plaintext():
    master = derive_master_key(_fake_keypair(), TITAN_PUBKEY)
    bkey = derive_backup_key(master, 42, "personality")
    plaintext = b"TITAN BACKUP PAYLOAD " + os.urandom(1024)

    ct, iv, tag = encrypt_tarball(plaintext, bkey)
    assert len(iv) == IV_LEN
    assert len(tag) == TAG_LEN
    assert ct.endswith(tag), "AES-GCM API appends tag to ciphertext"
    assert len(ct) == len(plaintext) + TAG_LEN

    recovered = decrypt_tarball(ct, iv, bkey)
    assert recovered == plaintext


def test_roundtrip_large_payload():
    """8 MB roundtrip — closer to real backup sizes."""
    master = derive_master_key(_fake_keypair(), TITAN_PUBKEY)
    bkey = derive_backup_key(master, 7, "timechain")
    plaintext = os.urandom(8 * 1024 * 1024)

    ct, iv, _ = encrypt_tarball(plaintext, bkey)
    assert decrypt_tarball(ct, iv, bkey) == plaintext


# ────────────────────────────────────────────────────────────────────────────
# 2. HKDF determinism
# ────────────────────────────────────────────────────────────────────────────

def test_master_key_deterministic():
    kp = _fake_keypair()
    k1 = derive_master_key(kp, TITAN_PUBKEY)
    k2 = derive_master_key(kp, TITAN_PUBKEY)
    assert k1 == k2
    assert len(k1) == KEY_LEN


def test_backup_key_deterministic():
    master = derive_master_key(_fake_keypair(), TITAN_PUBKEY)
    k1 = derive_backup_key(master, 99, "soul")
    k2 = derive_backup_key(master, 99, "soul")
    assert k1 == k2
    assert len(k1) == KEY_LEN


# ────────────────────────────────────────────────────────────────────────────
# 3. Domain separation
# ────────────────────────────────────────────────────────────────────────────

def test_different_pubkey_different_master():
    kp = _fake_keypair()
    k1 = derive_master_key(kp, "T1_PUB_A")
    k2 = derive_master_key(kp, "T2_PUB_B")
    assert k1 != k2


def test_different_backup_id_different_key():
    master = derive_master_key(_fake_keypair(), TITAN_PUBKEY)
    assert derive_backup_key(master, 1, "personality") != derive_backup_key(master, 2, "personality")


def test_different_backup_type_different_key():
    master = derive_master_key(_fake_keypair(), TITAN_PUBKEY)
    assert derive_backup_key(master, 1, "personality") != derive_backup_key(master, 1, "soul")


def test_different_keypair_different_master():
    k1 = derive_master_key(_fake_keypair(0x11), TITAN_PUBKEY)
    k2 = derive_master_key(_fake_keypair(0x22), TITAN_PUBKEY)
    assert k1 != k2


# ────────────────────────────────────────────────────────────────────────────
# 4. Auth-failure paths
# ────────────────────────────────────────────────────────────────────────────

def test_wrong_key_raises_invalid_tag():
    master = derive_master_key(_fake_keypair(), TITAN_PUBKEY)
    bkey_a = derive_backup_key(master, 1, "personality")
    bkey_b = derive_backup_key(master, 2, "personality")
    ct, iv, _ = encrypt_tarball(b"secret", bkey_a)
    with pytest.raises(InvalidTag):
        decrypt_tarball(ct, iv, bkey_b)


def test_tampered_ciphertext_raises():
    master = derive_master_key(_fake_keypair(), TITAN_PUBKEY)
    bkey = derive_backup_key(master, 1, "personality")
    ct, iv, _ = encrypt_tarball(b"payload" * 100, bkey)
    tampered = bytearray(ct)
    tampered[10] ^= 0xFF
    with pytest.raises(InvalidTag):
        decrypt_tarball(bytes(tampered), iv, bkey)


def test_tampered_iv_raises():
    master = derive_master_key(_fake_keypair(), TITAN_PUBKEY)
    bkey = derive_backup_key(master, 1, "personality")
    ct, iv, _ = encrypt_tarball(b"payload", bkey)
    bad_iv = bytes([b ^ 0xFF for b in iv])
    with pytest.raises(InvalidTag):
        decrypt_tarball(ct, bad_iv, bkey)


def test_tampered_tag_raises():
    master = derive_master_key(_fake_keypair(), TITAN_PUBKEY)
    bkey = derive_backup_key(master, 1, "personality")
    ct, iv, _ = encrypt_tarball(b"payload" * 50, bkey)
    # Tag is the last TAG_LEN bytes — flip one
    tampered = bytearray(ct)
    tampered[-1] ^= 0xFF
    with pytest.raises(InvalidTag):
        decrypt_tarball(bytes(tampered), iv, bkey)


# ────────────────────────────────────────────────────────────────────────────
# 5. Shamir-reconstruction equivalence (the Maker recovery invariant)
# ────────────────────────────────────────────────────────────────────────────

def test_shamir_reconstructed_keypair_yields_same_master():
    """Primary Maker recovery invariant:
    If Maker reconstructs the 64-byte keypair via 2-of-3 Shamir, they derive the
    same master key → can decrypt any backup. Without this property, encryption
    breaks the existing resurrection SDK.
    """
    kp = _fake_keypair(0xA5)
    shards = split_secret(kp, n=3, t=2)
    # Any 2-of-3 subset reconstructs
    for subset in ([shards[0], shards[1]], [shards[0], shards[2]], [shards[1], shards[2]]):
        reconstructed = combine_shares(subset)
        assert reconstructed == kp
        assert derive_master_key(reconstructed, TITAN_PUBKEY) == derive_master_key(kp, TITAN_PUBKEY)


def test_shamir_roundtrip_decrypts_real_backup():
    """End-to-end: encrypt with live key, decrypt with Shamir-reconstructed key."""
    kp = _fake_keypair(0xC3)
    master_live = derive_master_key(kp, TITAN_PUBKEY)
    bkey_live = derive_backup_key(master_live, 17, "personality")
    plaintext = b"sovereign state " + os.urandom(512)
    ct, iv, _ = encrypt_tarball(plaintext, bkey_live)

    # Maker reconstructs via Shamir (different subset than the one that encrypted)
    shards = split_secret(kp, n=3, t=2)
    kp_recovered = combine_shares([shards[1], shards[2]])
    master_maker = derive_master_key(kp_recovered, TITAN_PUBKEY)
    bkey_maker = derive_backup_key(master_maker, 17, "personality")

    assert decrypt_tarball(ct, iv, bkey_maker) == plaintext


# ────────────────────────────────────────────────────────────────────────────
# 6. Input validation
# ────────────────────────────────────────────────────────────────────────────

def test_derive_master_key_short_input_raises():
    with pytest.raises(ValueError):
        derive_master_key(b"short", TITAN_PUBKEY)


def test_derive_master_key_empty_pubkey_raises():
    with pytest.raises(ValueError):
        derive_master_key(_fake_keypair(), "")


def test_derive_backup_key_bad_master_raises():
    with pytest.raises(ValueError):
        derive_backup_key(b"too-short", 1, "personality")


def test_derive_backup_key_empty_id_raises():
    master = derive_master_key(_fake_keypair(), TITAN_PUBKEY)
    with pytest.raises(ValueError):
        derive_backup_key(master, "", "personality")


def test_derive_backup_key_empty_type_raises():
    master = derive_master_key(_fake_keypair(), TITAN_PUBKEY)
    with pytest.raises(ValueError):
        derive_backup_key(master, 1, "")


def test_encrypt_bad_key_len_raises():
    with pytest.raises(ValueError):
        encrypt_tarball(b"x", b"short")


def test_decrypt_bad_iv_len_raises():
    master = derive_master_key(_fake_keypair(), TITAN_PUBKEY)
    bkey = derive_backup_key(master, 1, "personality")
    with pytest.raises(ValueError):
        decrypt_tarball(b"x" * 32, b"short", bkey)


# ────────────────────────────────────────────────────────────────────────────
# Constants + key_id format
# ────────────────────────────────────────────────────────────────────────────

def test_algorithm_id_is_aes_256_gcm():
    assert ALGORITHM_ID == "AES-256-GCM"


def test_key_id_format_stable():
    assert key_id(42, "personality") == "hkdf:master:backup_42:personality"
    assert key_id(99, "soul") == "hkdf:master:backup_99:soul"


def test_salt_and_info_prefix_version_pinned():
    """Catch accidental salt/prefix changes — would invalidate all existing backups."""
    assert MASTER_KEY_SALT == b"titan-backup-master-v1"
    assert PER_BACKUP_KEY_INFO_PREFIX == b"backup/"
