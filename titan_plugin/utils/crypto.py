"""
utils/crypto.py
Cryptographic Root of Trust for the Titan V2.0 Sovereign Stack.

Centralizes all hash, signing, and verification primitives used across:
  - Meditation Epoch (state root hashing for ZK-compressed accounts)
  - Rebirth Backup (file hashing, Shadow Drive payload signing)
  - Sovereign Soul (Maker signature verification, directive inscription)
  - Resurrection Protocol (downloaded snapshot integrity verification)

All functions degrade gracefully when `solders` is unavailable (test envs,
minimal installs). Upgrading to post-quantum signatures in V3.0 only
requires changes to this single file.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

# Buffer size for streaming file hashes (8 KB)
_HASH_CHUNK_SIZE = 8192

# Salt file stored alongside the encrypted keypair
_HW_SALT_FILE = "data/hw_salt.bin"


# ---------------------------------------------------------------------------
# State Hashing — Deterministic SHA-256 for on-chain anchoring
# ---------------------------------------------------------------------------
def generate_state_hash(data: Union[dict, bytes, str]) -> str:
    """
    Deterministic SHA-256 hash of the Titan's cognitive state.

    Used to generate the ``latest_memory_hash`` stored in ZK-compressed
    Solana accounts during Meditation Epochs.

    Args:
        data: Cognitive state payload — dict (serialized to sorted JSON),
              bytes (hashed directly), or str (UTF-8 encoded then hashed).

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    if isinstance(data, dict):
        # Sorted keys for determinism across Python versions/runs
        raw = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    elif isinstance(data, str):
        raw = data.encode("utf-8")
    else:
        raw = bytes(data)

    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# File Hashing — Streaming SHA-256 for large archives
# ---------------------------------------------------------------------------
def hash_file(file_path: Union[str, Path]) -> str:
    """
    Stream a file and return its SHA-256 hex digest.

    Used by Rebirth Backup to compute the archive hash that gets
    committed on-chain and verified during Resurrection.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Hex-encoded SHA-256 digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_SIZE), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_file_integrity(file_path: Union[str, Path], expected_hash: str) -> bool:
    """
    Verify a file's SHA-256 hash against an expected value.

    Essential for the Resurrection Protocol: ensures the downloaded
    Cognee DB snapshot from Shadow Drive hasn't been tampered with.

    Args:
        file_path: Path to the file to verify.
        expected_hash: Expected hex-encoded SHA-256 digest.

    Returns:
        True if the computed hash matches the expected hash.
    """
    try:
        actual = hash_file(file_path)
        if actual == expected_hash:
            logger.debug("[Crypto] File integrity verified: %s", file_path)
            return True
        logger.warning(
            "[Crypto] File integrity FAILED for %s — expected %s, got %s",
            file_path, expected_hash[:16], actual[:16],
        )
        return False
    except FileNotFoundError:
        logger.error("[Crypto] File not found for integrity check: %s", file_path)
        return False


# ---------------------------------------------------------------------------
# Solana Message Signing — Unified keypair.sign_message wrapper
# ---------------------------------------------------------------------------
def sign_solana_payload(keypair, payload: str) -> str | None:
    """
    Sign an arbitrary message with a Solana Ed25519 keypair.

    Used for Memo program inscriptions and Shadow Drive authentication.

    Args:
        keypair: A ``solders.keypair.Keypair`` instance.
        payload: The message string to sign.

    Returns:
        Base58-encoded signature string, or None on failure.
    """
    if keypair is None:
        logger.error("[Crypto] Cannot sign — no keypair provided.")
        return None

    try:
        sig = keypair.sign_message(payload.encode("utf-8"))
        return str(sig)
    except Exception as e:
        logger.error("[Crypto] Signing failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Shadow Drive Authentication — Standardized message format
# ---------------------------------------------------------------------------
def sign_shadow_drive_message(
    keypair, storage_account: str, filename: str,
) -> tuple[str, str] | None:
    """
    Construct and sign the exact message format required by GenesysGo
    Shadow Drive for file upload authentication.

    Prevents "Signature Drift" by centralizing the message template
    that both add_file and edit_file operations share.

    Args:
        keypair: A ``solders.keypair.Keypair`` instance.
        storage_account: The Shadow Drive storage account public key.
        filename: The target filename on Shadow Drive.

    Returns:
        Tuple of (signature_string, signer_pubkey_string), or None on failure.
    """
    if keypair is None:
        logger.error("[Crypto] Cannot sign Shadow Drive request — no keypair.")
        return None

    message = (
        f"Shadow Drive Signed Message:\n"
        f"Storage Account: {storage_account}\n"
        f"Upload file: {filename}"
    )

    sig_str = sign_solana_payload(keypair, message)
    if sig_str is None:
        return None

    try:
        signer = str(keypair.pubkey())
        return (sig_str, signer)
    except Exception as e:
        logger.error("[Crypto] Failed to get signer pubkey: %s", e)
        return None


# ---------------------------------------------------------------------------
# Maker Signature Verification — Ed25519 Divine Inspiration guard
# ---------------------------------------------------------------------------
def verify_maker_signature(
    message: str, signature: str, maker_pubkey_str: str,
) -> bool:
    """
    Verify that a Prime Directive update was signed by the Maker's
    Ed25519 key. Protects the Sovereign Soul from identity hijacking.

    Args:
        message: The directive payload to verify.
        signature: Base58-encoded Ed25519 signature from the Maker.
        maker_pubkey_str: Base58-encoded Maker public key string.

    Returns:
        True if the signature is valid and from the specified Maker.
    """
    if not maker_pubkey_str:
        logger.warning("[Crypto] No maker pubkey provided — cannot verify.")
        return False

    try:
        from solders.pubkey import Pubkey
        from solders.signature import Signature

        pubkey = Pubkey.from_string(maker_pubkey_str)
        sig = Signature.from_string(signature)
        verified = sig.verify(pubkey, message.encode("utf-8"))

        if verified:
            logger.info("[Crypto] Maker signature verified.")
        else:
            logger.warning("[Crypto] Maker signature verification FAILED.")

        return verified

    except ImportError:
        logger.error("[Crypto] solders not installed — signature verification unavailable.")
        return False
    except Exception as e:
        logger.error("[Crypto] Signature verification error: %s", e)
        return False


# ---------------------------------------------------------------------------
# Hardware-Bound Encryption — Machine-specific key for "warm reboot" persistence
# ---------------------------------------------------------------------------
def get_hardware_fingerprint(salt_path: str = _HW_SALT_FILE) -> bytes:
    """
    Generate a composite hardware fingerprint unique to this machine.

    Combines multiple sources to prevent VPS cloning attacks:
      1. /etc/machine-id (stable across reboots, unique per install)
      2. hostname (additional entropy)
      3. A local random salt (prevents rainbow table attacks)

    The local salt is generated once and stored alongside the encrypted
    keypair. It is NOT secret — it's a diversifier, not a key.

    Args:
        salt_path: Path to the salt file (relative to project root).

    Returns:
        32-byte hardware fingerprint suitable as AES-256 key material.
    """
    components = []

    # Source 1: /etc/machine-id (Linux standard, stable across reboots)
    try:
        with open("/etc/machine-id", "r") as f:
            components.append(f.read().strip())
    except (FileNotFoundError, PermissionError):
        logger.debug("[Crypto] /etc/machine-id not available.")

    # Source 2: DMI product UUID (requires root on some systems)
    try:
        with open("/sys/class/dmi/id/product_uuid", "r") as f:
            components.append(f.read().strip())
    except (FileNotFoundError, PermissionError):
        logger.debug("[Crypto] DMI product UUID not available.")

    # Source 3: hostname
    import socket
    components.append(socket.gethostname())

    if not components:
        logger.warning(
            "[Crypto] No hardware identifiers found. "
            "Hardware-bound encryption will be weak on this platform."
        )
        components.append("fallback_no_hw_id")

    # Source 4: Local random salt (generated once, stored on disk)
    salt_path = Path(salt_path)
    if salt_path.exists():
        salt = salt_path.read_bytes()
    else:
        salt = os.urandom(32)
        salt_path.parent.mkdir(parents=True, exist_ok=True)
        salt_path.write_bytes(salt)
        logger.info("[Crypto] Generated new hardware salt at %s", salt_path)

    # Combine all sources into a single fingerprint
    combined = "|".join(components).encode("utf-8") + salt
    fingerprint = hashlib.pbkdf2_hmac(
        "sha256",
        combined,
        b"TITAN_HW_BIND_V2",
        iterations=100_000,
        dklen=32,
    )
    return fingerprint


def encrypt_for_machine(
    data: bytes, salt_path: str = _HW_SALT_FILE,
) -> bytes:
    """
    Encrypt data using a hardware-bound AES-256-GCM key.

    The ciphertext is useless on any other machine because the hardware
    fingerprint (and thus the AES key) will be different.

    Output format: 12-byte nonce + ciphertext + 16-byte GCM tag.

    Args:
        data: Plaintext bytes to encrypt (e.g., keypair JSON).
        salt_path: Path to the hardware salt file.

    Returns:
        Encrypted bytes (nonce + ciphertext + tag).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = get_hardware_fingerprint(salt_path)
    nonce = os.urandom(12)

    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, data, b"titan_hw_bound_v2.0")
    return nonce + ciphertext


def decrypt_for_machine(
    encrypted: bytes, salt_path: str = _HW_SALT_FILE,
) -> bytes:
    """
    Decrypt hardware-bound AES-256-GCM ciphertext.

    Will fail with InvalidTag if run on different hardware than the
    encryption machine (because the fingerprint-derived key differs).

    Args:
        encrypted: Encrypted bytes (nonce + ciphertext + tag).
        salt_path: Path to the hardware salt file.

    Returns:
        Decrypted plaintext bytes.

    Raises:
        cryptography.exceptions.InvalidTag: If the hardware fingerprint
            doesn't match (wrong machine or tampered data).
        ValueError: If the encrypted data is too short.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if len(encrypted) < 28:  # 12 nonce + 16 tag minimum
        raise ValueError("Encrypted data too short — corrupted or not hardware-bound.")

    key = get_hardware_fingerprint(salt_path)
    nonce = encrypted[:12]
    ciphertext = encrypted[12:]

    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, b"titan_hw_bound_v2.0")
