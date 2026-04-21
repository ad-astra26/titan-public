"""
Backup encryption primitives — rFP_backup_worker Phase 7.

Design (Q2 locked 2026-04-20):
  - Master key derived on demand from Titan identity keypair (first 32 bytes = Ed25519 seed)
    via HKDF-SHA256. No new key-at-rest file; Maker recovery via 2-of-3 Shamir reconstructs
    the same keypair → derives the same master → decrypts any backup.
  - Per-backup key via HKDF-SHA256(master, info=f"backup/{backup_id}/{backup_type}").
    Compromise of one per-backup key does NOT leak others.
  - AES-256-GCM (96-bit IV, 128-bit auth tag) for confidentiality + integrity.

Threat model: protects the Arweave-public-permanent disclosure surface. VPS compromise
is explicitly out of scope (attacker with VPS already has the keypair + plaintext live
state), per rFP §5.8.

Domain-separation constants are version-pinned via the salt so future key-derivation
changes cannot collide with existing backups.
"""

import json
import logging
import os
from typing import Optional, Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger(__name__)


MASTER_KEY_SALT = b"titan-backup-master-v1"
PER_BACKUP_KEY_INFO_PREFIX = b"backup/"

SEED_LEN = 32
KEY_LEN = 32
IV_LEN = 12
TAG_LEN = 16

ALGORITHM_ID = "AES-256-GCM"
KEY_ID_FORMAT = "hkdf:master:backup_{backup_id}:{backup_type}"


def derive_master_key(keypair_bytes: bytes, titan_pubkey: str) -> bytes:
    """HKDF-SHA256 from the Ed25519 seed (first 32B of the 64B keypair) bound to titan_pubkey.

    Any Titan instance that reconstructs the same 64-byte keypair (via live load or
    2-of-3 Shamir recovery) derives the same 32-byte master key.
    """
    if not isinstance(keypair_bytes, (bytes, bytearray)) or len(keypair_bytes) < SEED_LEN:
        raise ValueError(
            f"keypair_bytes must be ≥{SEED_LEN} bytes (got {len(keypair_bytes)})"
        )
    if not titan_pubkey:
        raise ValueError("titan_pubkey required for domain separation")

    seed = bytes(keypair_bytes[:SEED_LEN])
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_LEN,
        salt=MASTER_KEY_SALT,
        info=titan_pubkey.encode("utf-8"),
    )
    return hkdf.derive(seed)


def derive_backup_key(master_key: bytes, backup_id, backup_type: str) -> bytes:
    """Per-backup 32-byte key from master; unique per (backup_id, backup_type) pair."""
    if not isinstance(master_key, (bytes, bytearray)) or len(master_key) != KEY_LEN:
        raise ValueError(f"master_key must be {KEY_LEN} bytes")
    if backup_id is None or str(backup_id) == "":
        raise ValueError("backup_id required")
    if not backup_type:
        raise ValueError("backup_type required")

    info = PER_BACKUP_KEY_INFO_PREFIX + f"{backup_id}/{backup_type}".encode("utf-8")
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_LEN,
        salt=b"",
        info=info,
    )
    return hkdf.derive(bytes(master_key))


def key_id(backup_id, backup_type: str) -> str:
    """Stable string that goes into the manifest. Not a secret."""
    return KEY_ID_FORMAT.format(backup_id=backup_id, backup_type=backup_type)


def encrypt_tarball(plaintext: bytes, backup_key: bytes) -> Tuple[bytes, bytes, bytes]:
    """AES-256-GCM encrypt. Returns (ciphertext_with_tag, iv, tag).

    The `cryptography` AESGCM API returns ciphertext||tag concatenated; we also expose
    the tag separately so the manifest can record it for restoration.
    """
    if not isinstance(backup_key, (bytes, bytearray)) or len(backup_key) != KEY_LEN:
        raise ValueError(f"backup_key must be {KEY_LEN} bytes")
    iv = os.urandom(IV_LEN)
    ct_and_tag = AESGCM(bytes(backup_key)).encrypt(iv, bytes(plaintext), None)
    tag = ct_and_tag[-TAG_LEN:]
    return ct_and_tag, iv, tag


def decrypt_tarball(ciphertext_with_tag: bytes, iv: bytes, backup_key: bytes) -> bytes:
    """Inverse of encrypt_tarball. Raises InvalidTag on auth failure (wrong key / tamper)."""
    if not isinstance(backup_key, (bytes, bytearray)) or len(backup_key) != KEY_LEN:
        raise ValueError(f"backup_key must be {KEY_LEN} bytes")
    if not isinstance(iv, (bytes, bytearray)) or len(iv) != IV_LEN:
        raise ValueError(f"iv must be {IV_LEN} bytes")
    return AESGCM(bytes(backup_key)).decrypt(bytes(iv), bytes(ciphertext_with_tag), None)


def decrypt_from_manifest(ciphertext_with_tag: bytes,
                            encryption_manifest: dict,
                            keypair_bytes: bytes,
                            titan_pubkey: str,
                            backup_type: str) -> bytes:
    """Restore-side one-shot: given raw bytes + manifest stanza + keypair,
    return plaintext. Handles legacy (algorithm="none") as passthrough.

    Caller is responsible for verifying plaintext SHA256 against the manifest's
    `plaintext_sha256` after this returns — that confirms correct-key decryption
    (the AES-GCM auth tag already caught wrong-key attempts).
    """
    algo = (encryption_manifest or {}).get("algorithm", "none")
    if algo == "none" or not encryption_manifest:
        return ciphertext_with_tag  # legacy: tarball was never encrypted

    if algo != ALGORITHM_ID:
        raise ValueError(f"Unsupported encryption algorithm: {algo}")

    import base64 as _b64
    iv = _b64.b64decode(encryption_manifest["iv_b64"])
    backup_id = encryption_manifest.get("backup_id")
    if not backup_id:
        raise ValueError("encryption manifest missing backup_id")

    master = derive_master_key(keypair_bytes, titan_pubkey)
    bkey = derive_backup_key(master, backup_id, backup_type)
    return decrypt_tarball(ciphertext_with_tag, iv, bkey)


def load_keypair_bytes(keypair_path: str) -> Tuple[bytes, str]:
    """Read the Titan Ed25519 keypair JSON. Returns (64-byte keypair, pubkey_b58)."""
    with open(keypair_path) as f:
        raw = json.load(f)
    if not isinstance(raw, list) or len(raw) < 64:
        raise ValueError(f"malformed keypair at {keypair_path}")
    kp_bytes = bytes(raw)
    try:
        from solders.keypair import Keypair as _KP
        pubkey = str(_KP.from_bytes(kp_bytes).pubkey())
    except Exception:
        pubkey = kp_bytes[32:64].hex()
    return kp_bytes, pubkey


def build_encryption_context_from_config(full_config: dict) -> Optional[dict]:
    """Shared helper: resolve [backup].encryption_enabled → {master_key, tier} or None.

    Called by every cascade-caller (RebirthBackup, TimeChainBackup). Fails open
    on missing/malformed keypair — logs a warning and returns None so the backup
    still proceeds unencrypted, rather than silently dropping a backup.
    """
    backup_cfg = (full_config or {}).get("backup", {}) or {}
    if not backup_cfg.get("encryption_enabled", False):
        return None
    try:
        net = (full_config or {}).get("network", {}) or {}
        kp_path = net.get(
            "wallet_keypair_path", "data/titan_identity_keypair.json")
        if not os.path.exists(kp_path):
            logger.warning(
                "[backup-crypto] encryption_enabled=true but keypair missing at %s — "
                "backup proceeds UNENCRYPTED (fail-open)", kp_path)
            return None
        kp_bytes, titan_pubkey = load_keypair_bytes(kp_path)
        return {
            "master_key": derive_master_key(kp_bytes, titan_pubkey),
            "tier": backup_cfg.get("encryption_tier", "private"),
        }
    except Exception as e:
        logger.error(
            "[backup-crypto] build_encryption_context_from_config failed: %s — "
            "UNENCRYPTED", e)
        return None
