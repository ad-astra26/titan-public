"""
utils/shamir.py
Pure Python Shamir's Secret Sharing over GF(2^8) for the Titan Resurrection SDK.

Implements 2-of-3 threshold secret splitting with zero external dependencies,
ensuring the Titan can be resurrected on minimal infrastructure.

The Galois Field GF(2^8) uses the AES irreducible polynomial x^8 + x^4 + x^3 + x + 1
(0x11B), providing 255 non-zero elements for polynomial evaluation.

Security properties:
  - Information-theoretic security: any single shard reveals zero bits of the secret
  - Threshold: any 2 of 3 shards can reconstruct the full secret
  - Each shard is the same length as the secret (64 bytes for an Ed25519 keypair)
"""
import hashlib
import json
import logging
import os
import struct
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GF(2^8) Arithmetic — AES polynomial x^8 + x^4 + x^3 + x + 1
# ---------------------------------------------------------------------------
# Precompute exp and log tables for fast GF(2^8) multiplication
_GF_EXP = [0] * 512  # Anti-log table (doubled for wraparound)
_GF_LOG = [0] * 256   # Log table

def _init_gf_tables():
    """Initialize GF(2^8) exp/log lookup tables using generator 3."""
    x = 1
    for i in range(255):
        _GF_EXP[i] = x
        _GF_LOG[x] = i
        # Multiply by generator 3 in GF(2^8): x*3 = x*2 XOR x
        x = ((x << 1) ^ x)
        if x & 0x100:
            x ^= 0x11B  # AES irreducible polynomial
    # Extend exp table for easy modular wraparound
    for i in range(255, 512):
        _GF_EXP[i] = _GF_EXP[i - 255]

_init_gf_tables()


def _gf_mul(a: int, b: int) -> int:
    """Multiply two GF(2^8) elements using log/exp tables."""
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _gf_inv(a: int) -> int:
    """Multiplicative inverse in GF(2^8)."""
    if a == 0:
        raise ZeroDivisionError("No inverse for zero in GF(2^8)")
    return _GF_EXP[255 - _GF_LOG[a]]


def _gf_div(a: int, b: int) -> int:
    """Division in GF(2^8): a / b."""
    if b == 0:
        raise ZeroDivisionError("Division by zero in GF(2^8)")
    if a == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] - _GF_LOG[b]) % 255]


# ---------------------------------------------------------------------------
# Polynomial Evaluation & Interpolation
# ---------------------------------------------------------------------------
def _eval_poly(coeffs: List[int], x: int) -> int:
    """
    Evaluate polynomial with given coefficients at point x in GF(2^8).
    coeffs[0] is the constant term (the secret byte).
    """
    result = 0
    for coeff in reversed(coeffs):
        result = _gf_mul(result, x) ^ coeff
    return result


def _lagrange_interpolate(points: List[Tuple[int, int]], at_x: int = 0) -> int:
    """
    Lagrange interpolation at a given x in GF(2^8).
    Points are (x_i, y_i) tuples. Default evaluates at x=0 (recovers secret).
    """
    result = 0
    for i, (xi, yi) in enumerate(points):
        # Compute the Lagrange basis polynomial l_i(at_x)
        num = 1
        den = 1
        for j, (xj, _) in enumerate(points):
            if i != j:
                num = _gf_mul(num, at_x ^ xj)
                den = _gf_mul(den, xi ^ xj)
        # l_i(at_x) = num / den
        basis = _gf_div(num, den)
        result ^= _gf_mul(yi, basis)
    return result


# ---------------------------------------------------------------------------
# Public API — Split & Combine
# ---------------------------------------------------------------------------
def split_secret(secret: bytes, n: int = 3, t: int = 2) -> List[bytes]:
    """
    Split a secret into n shares with threshold t using Shamir's Secret Sharing.

    Args:
        secret: The secret bytes to split (e.g., 64-byte Ed25519 keypair).
        n: Total number of shares to generate (default 3).
        t: Minimum shares required for reconstruction (default 2).

    Returns:
        List of n shares. Each share is prefixed with a 1-byte x-coordinate
        (evaluation point), followed by the share bytes.

    Raises:
        ValueError: If parameters are invalid.
    """
    if t > n:
        raise ValueError(f"Threshold {t} cannot exceed total shares {n}")
    if n > 254:
        raise ValueError(f"Maximum 254 shares (GF(2^8) has 255 non-zero elements)")
    if t < 2:
        raise ValueError("Threshold must be at least 2")

    # Use x-coordinates 1..n (never 0, since f(0) = secret)
    x_coords = list(range(1, n + 1))

    shares = [bytearray() for _ in range(n)]

    for byte_val in secret:
        # Generate t-1 random coefficients; coeffs[0] = secret byte
        coeffs = [byte_val] + [int.from_bytes(os.urandom(1), "big") for _ in range(t - 1)]

        for i, x in enumerate(x_coords):
            y = _eval_poly(coeffs, x)
            shares[i].append(y)

    # Prepend x-coordinate to each share
    result = []
    for i, share in enumerate(shares):
        result.append(bytes([x_coords[i]]) + bytes(share))

    logger.info("[Shamir] Split %d-byte secret into %d shares (threshold %d).", len(secret), n, t)
    return result


def combine_shares(shares: List[bytes]) -> bytes:
    """
    Reconstruct a secret from threshold-or-more shares.

    Args:
        shares: List of share bytes (each prefixed with 1-byte x-coordinate).

    Returns:
        The reconstructed secret bytes.

    Raises:
        ValueError: If fewer than 2 shares are provided or shares are malformed.
    """
    if len(shares) < 2:
        raise ValueError("Need at least 2 shares for reconstruction")

    # Parse x-coordinates and share data
    parsed = []
    share_len = None
    for share in shares:
        if len(share) < 2:
            raise ValueError("Share too short — must have x-coordinate + at least 1 byte")
        x = share[0]
        data = share[1:]
        if share_len is None:
            share_len = len(data)
        elif len(data) != share_len:
            raise ValueError(f"Share length mismatch: expected {share_len}, got {len(data)}")
        parsed.append((x, data))

    # Check for duplicate x-coordinates
    x_coords = [x for x, _ in parsed]
    if len(set(x_coords)) != len(x_coords):
        raise ValueError("Duplicate x-coordinates detected — shares may be corrupted")

    # Reconstruct byte-by-byte
    secret = bytearray()
    for byte_idx in range(share_len):
        points = [(x, data[byte_idx]) for x, data in parsed]
        recovered = _lagrange_interpolate(points, at_x=0)
        secret.append(recovered)

    logger.info("[Shamir] Reconstructed %d-byte secret from %d shares.", len(secret), len(shares))
    return bytes(secret)


# ---------------------------------------------------------------------------
# Shard Envelope — Structured shard packaging with metadata
# ---------------------------------------------------------------------------
def create_maker_envelope(
    shard: bytes,
    titan_pubkey: str,
    genesis_tx: str = "",
) -> str:
    """
    Package Shard 1 (Maker's shard) into a self-describing JSON envelope,
    then encode as a hex string for display/storage.

    The envelope contains the Titan's public address and the Genesis TX
    signature, enabling the resurrection script to identify the Titan
    and locate Shard 3 on-chain without scanning.

    Args:
        shard: Raw shard bytes (x-coordinate + share data).
        titan_pubkey: Base58 Titan public key.
        genesis_tx: Solana TX signature of the Genesis Memo (Shard 3 anchor).

    Returns:
        Hex-encoded envelope string.
    """
    import time

    envelope = {
        "version": "2.0",
        "type": "maker_shard",
        "titan_pubkey": titan_pubkey,
        "shard_data": shard.hex(),
        "genesis_tx": genesis_tx,
        "created_at": int(time.time()),
    }
    envelope_json = json.dumps(envelope, sort_keys=True)
    return envelope_json.encode("utf-8").hex()


def parse_maker_envelope(hex_envelope: str) -> Tuple[bytes, dict]:
    """
    Decode a Maker's shard envelope from hex back to raw shard + metadata.

    Args:
        hex_envelope: Hex-encoded envelope string.

    Returns:
        Tuple of (raw_shard_bytes, metadata_dict).

    Raises:
        ValueError: If the envelope is malformed or version is unsupported.
    """
    try:
        envelope_json = bytes.fromhex(hex_envelope).decode("utf-8")
        envelope = json.loads(envelope_json)
    except Exception as e:
        raise ValueError(f"Invalid envelope format: {e}")

    if envelope.get("version") != "2.0":
        raise ValueError(f"Unsupported envelope version: {envelope.get('version')}")
    if envelope.get("type") != "maker_shard":
        raise ValueError(f"Wrong envelope type: {envelope.get('type')}")

    shard_hex = envelope.get("shard_data", "")
    if not shard_hex:
        raise ValueError("Envelope contains no shard data")

    shard = bytes.fromhex(shard_hex)
    metadata = {
        "titan_pubkey": envelope.get("titan_pubkey", ""),
        "genesis_tx": envelope.get("genesis_tx", ""),
        "created_at": envelope.get("created_at", 0),
    }
    return shard, metadata


# ---------------------------------------------------------------------------
# Shard 3 Encryption — Deterministic AES key from Titan pubkey
# ---------------------------------------------------------------------------
def derive_shard3_key(titan_pubkey: str, salt: str = "TITAN_GENESIS_ANCHOR") -> bytes:
    """
    Derive a deterministic AES-256 key from the Titan's public address.

    This key encrypts Shard 3 before on-chain storage. Anyone with the
    Titan's pubkey (publicly visible on Solana) and this code can derive
    the key — making Shard 3 "deterministically recoverable."

    Uses PBKDF2-HMAC-SHA512 with 600,000 iterations for key stretching.

    Args:
        titan_pubkey: Base58-encoded Titan public key.
        salt: Static salt string (default "TITAN_GENESIS_ANCHOR").

    Returns:
        32-byte AES-256 key.
    """
    return hashlib.pbkdf2_hmac(
        "sha512",
        titan_pubkey.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=600_000,
        dklen=32,
    )


def encrypt_shard3(shard: bytes, titan_pubkey: str) -> bytes:
    """
    Encrypt Shard 3 with a deterministic AES-256-GCM key derived from
    the Titan's public address.

    The output format: 12-byte nonce + ciphertext + 16-byte GCM tag.

    Args:
        shard: Raw Shard 3 bytes.
        titan_pubkey: Base58-encoded Titan public key.

    Returns:
        Encrypted shard bytes (nonce + ciphertext + tag).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = derive_shard3_key(titan_pubkey)
    # Deterministic nonce from pubkey hash (safe because key is unique per Titan)
    nonce = hashlib.sha256(f"SHARD3_NONCE:{titan_pubkey}".encode()).digest()[:12]

    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, shard, b"titan_shard3_v2.0")
    return nonce + ciphertext


def decrypt_shard3(encrypted: bytes, titan_pubkey: str) -> bytes:
    """
    Decrypt Shard 3 using the deterministic AES-256-GCM key.

    Args:
        encrypted: Encrypted shard bytes (nonce + ciphertext + tag).
        titan_pubkey: Base58-encoded Titan public key.

    Returns:
        Decrypted Shard 3 bytes.

    Raises:
        cryptography.exceptions.InvalidTag: If decryption fails (tampered data).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = derive_shard3_key(titan_pubkey)
    nonce = encrypted[:12]
    ciphertext = encrypted[12:]

    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, b"titan_shard3_v2.0")


# ---------------------------------------------------------------------------
# Verification Ceremony — Exhaustive reconstruction test
# ---------------------------------------------------------------------------
def verify_all_combinations(
    secret: bytes, shares: List[bytes], t: int = 2,
) -> bool:
    """
    Genesis Verification Ceremony: test ALL valid combinations of t shares
    to confirm they reconstruct the original secret.

    This MUST pass before the plaintext keypair is destroyed.

    Args:
        secret: The original secret bytes.
        shares: All n shares from split_secret().
        t: Reconstruction threshold.

    Returns:
        True if every combination reconstructs correctly.
    """
    from itertools import combinations

    all_passed = True
    tested = 0

    for combo in combinations(shares, t):
        try:
            recovered = combine_shares(list(combo))
            if recovered != secret:
                x_coords = [s[0] for s in combo]
                logger.error(
                    "[Shamir] VERIFICATION FAILED for combination x=%s",
                    x_coords,
                )
                all_passed = False
            tested += 1
        except Exception as e:
            x_coords = [s[0] for s in combo]
            logger.error(
                "[Shamir] VERIFICATION ERROR for combination x=%s: %s",
                x_coords, e,
            )
            all_passed = False

    if all_passed:
        logger.info("[Shamir] Genesis Verification Ceremony PASSED (%d combinations tested).", tested)
    else:
        logger.error("[Shamir] Genesis Verification Ceremony FAILED. Aborting genesis.")

    return all_passed
