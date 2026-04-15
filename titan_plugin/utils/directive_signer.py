"""
titan_plugin/utils/directive_signer.py — Prime Directive Signing & Verification.

Cryptographic integrity for Titan's soul constitution. The constitution is signed
at boot with SHA256 + Ed25519. Every LLM call verifies the hash (microseconds).
Tampering triggers CRITICAL alert and Arweave restoration (M2).

This is Titan's constitutional immune system — the most important 50 lines in the codebase.
"""
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default paths
CONSTITUTION_PATH = "titan_constitution.md"
SIGNATURE_FILE = "data/titan_directives.sig"


class DirectiveTamperingError(Exception):
    """Raised when prime directives are compromised and unrecoverable."""
    pass


def compute_constitution_hash(constitution_path: str = CONSTITUTION_PATH) -> str:
    """Compute SHA256 hash of the constitution file. Microsecond operation."""
    text = Path(constitution_path).read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sign_directives(
    constitution_path: str = CONSTITUTION_PATH,
    signature_file: str = SIGNATURE_FILE,
    keypair_path: str = None,
) -> dict:
    """Sign the constitution and store the signature file.

    Args:
        constitution_path: Path to titan_constitution.md
        signature_file: Where to store the .sig file
        keypair_path: Solana keypair for Ed25519 signing (optional, for on-chain)

    Returns:
        Dict with hash, signature (if keypair), timestamp
    """
    directive_hash = compute_constitution_hash(constitution_path)

    sig_data = {
        "hash": directive_hash,
        "constitution_path": constitution_path,
        "signed_at": time.time(),
        "signed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Ed25519 signature if keypair available
    if keypair_path and os.path.exists(keypair_path):
        try:
            from titan_plugin.utils.crypto import sign_solana_payload
            signature = sign_solana_payload(directive_hash.encode(), keypair_path)
            sig_data["ed25519_signature"] = signature
            logger.info("[DirectiveSigner] Constitution signed with Ed25519")
        except Exception as e:
            logger.warning("[DirectiveSigner] Ed25519 signing failed: %s", e)
            sig_data["ed25519_signature"] = None
    else:
        sig_data["ed25519_signature"] = None

    # Write signature file atomically
    os.makedirs(os.path.dirname(signature_file) or ".", exist_ok=True)
    tmp_path = signature_file + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(sig_data, f, indent=2)
    os.replace(tmp_path, signature_file)

    logger.info("[DirectiveSigner] Constitution hash: %s...%s",
                directive_hash[:8], directive_hash[-8:])
    return sig_data


def verify_directives(
    constitution_path: str = CONSTITUTION_PATH,
    signature_file: str = SIGNATURE_FILE,
) -> bool:
    """Verify constitution hash matches stored signature. Microsecond operation.

    Returns True if constitution is untampered, False if compromised.
    """
    try:
        if not os.path.exists(constitution_path):
            logger.error("[DirectiveSigner] Constitution file missing: %s", constitution_path)
            return False

        if not os.path.exists(signature_file):
            logger.warning("[DirectiveSigner] No signature file — constitution unsigned")
            return False

        current_hash = compute_constitution_hash(constitution_path)

        with open(signature_file) as f:
            sig_data = json.load(f)

        stored_hash = sig_data.get("hash", "")
        return current_hash == stored_hash

    except Exception as e:
        logger.error("[DirectiveSigner] Verification error: %s", e)
        return False


def get_stored_hash(signature_file: str = SIGNATURE_FILE) -> Optional[str]:
    """Get the stored constitution hash without full verification."""
    try:
        if os.path.exists(signature_file):
            with open(signature_file) as f:
                return json.load(f).get("hash")
    except Exception:
        pass
    return None


def restore_from_arweave(genesis_nft_address: str = None, rpc_url: str = None) -> bool:
    """Restore constitution from Arweave permanent storage.

    This is a stub for M2 (GenesisNFT) — once the constitution is uploaded to
    Arweave and the TX ID stored in the GenesisNFT, this function will:
    1. Read GenesisNFT metadata → extract Arweave TX ID
    2. Fetch constitution from Arweave
    3. Verify against on-chain directive hash
    4. Replace local copy
    5. Re-sign

    For now, logs the attempt and returns False.
    """
    logger.critical(
        "[DirectiveSigner] Arweave restore requested (genesis=%s) — "
        "NOT YET IMPLEMENTED (requires M2 GenesisNFT)",
        genesis_nft_address)
    return False


def ensure_signed(
    constitution_path: str = CONSTITUTION_PATH,
    signature_file: str = SIGNATURE_FILE,
    keypair_path: str = None,
) -> dict:
    """Ensure constitution is signed. Sign if not yet signed or if hash changed.

    Called at boot. Returns the signature data.
    """
    if verify_directives(constitution_path, signature_file):
        with open(signature_file) as f:
            sig_data = json.load(f)
        logger.info("[DirectiveSigner] Constitution verified (signed %s)",
                    sig_data.get("signed_at_iso", "?"))
        return sig_data

    # Not signed or hash changed — (re)sign
    logger.info("[DirectiveSigner] Signing constitution...")
    return sign_directives(constitution_path, signature_file, keypair_path)
