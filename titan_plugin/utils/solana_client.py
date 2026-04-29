"""
utils/solana_client.py
Solana Primitive Facade for the Titan V2.0 Sovereign Stack.

Centralizes all Solana SDK imports (solders, solana-py) behind a clean
interface. Provides shared instruction builders, keypair management,
and ZK-state decoding so that high-level logic modules never touch raw
SDK types directly.

Separation of concerns:
  - This module: Data Construction (primitives, instruction building, parsing)
  - core/network.py: Data Orchestration (RPC connections, retries, bundles)

All functions degrade gracefully when the Solana SDK is unavailable,
allowing the Titan to boot in "Offline Mode" for testing art, audio,
and memory subsystems without crashing.
"""
import hashlib
import json
import logging
import struct
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDK Availability Gate
# ---------------------------------------------------------------------------
_SOLANA_AVAILABLE = False

try:
    from solders.pubkey import Pubkey
    from solders.keypair import Keypair
    from solders.instruction import Instruction, AccountMeta
    from solders.transaction import Transaction
    from solders.message import Message

    _SOLANA_AVAILABLE = True
except ImportError:
    Pubkey = None
    Keypair = None
    Instruction = None
    AccountMeta = None
    Transaction = None
    Message = None


def is_available() -> bool:
    """Check if the Solana SDK (solders + solana-py) is installed."""
    return _SOLANA_AVAILABLE


# ---------------------------------------------------------------------------
# Well-Known Program IDs
# ---------------------------------------------------------------------------
# Solana Memo Program V2
MEMO_PROGRAM_ID = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"
# Metaplex Core Program
MPL_CORE_PROGRAM_ID = "CoREENxT6tW1HoK8ypY1SxRMZTcVPm7R94rH4PZNhX7d"
# Solana Compute Budget Program (for priority fees)
COMPUTE_BUDGET_PROGRAM_ID = "ComputeBudget111111111111111111111111111111"
# Solana System Program
SYSTEM_PROGRAM_ID = "11111111111111111111111111111111"

# Titan ZK-Vault Program (deployed on devnet)
# Loaded from config.toml [network].vault_program_id at runtime; this is the fallback.
VAULT_PROGRAM_ID = "52an8WjtfxpkCqZZ1AYFkaDTGb4RyNFFD9VQRVdxcpJw"

# PDA seed for Titan vault accounts — MUST match b"titan_vault" in lib.rs
VAULT_PDA_SEED = b"titan_vault"

# Anchor instruction discriminators (from IDL — SHA256("global:<instruction_name>")[:8])
_VAULT_IX_INITIALIZE = bytes([48, 191, 163, 44, 71, 129, 63, 164])
_VAULT_IX_COMMIT_STATE = bytes([201, 80, 148, 145, 9, 196, 225, 56])
_VAULT_IX_UPDATE_SHADOW = bytes([1, 128, 107, 123, 91, 92, 122, 203])
_VAULT_IX_CLOSE = bytes([141, 103, 17, 126, 72, 75, 29, 29])


# ---------------------------------------------------------------------------
# Primitive Parsers
# ---------------------------------------------------------------------------
def parse_pubkey(pubkey_str: str) -> Optional["Pubkey"]:
    """
    Parse a Base58 public key string into a Pubkey object.

    Args:
        pubkey_str: Base58-encoded Solana public key.

    Returns:
        Pubkey object, or None if SDK unavailable or string invalid.
    """
    if not _SOLANA_AVAILABLE:
        logger.debug("[SolanaClient] SDK not available — cannot parse pubkey.")
        return None

    try:
        return Pubkey.from_string(pubkey_str)
    except Exception as e:
        logger.error("[SolanaClient] Invalid pubkey '%s': %s", pubkey_str[:20], e)
        return None


# 2026-04-08 audit fix (I-013): track which keypair paths we've already
# warned about to avoid spam. T2/T3 share T1's wallet via delegate mode,
# so they hit the "missing keypair" path repeatedly during normal operation.
# First occurrence: WARNING (signal). Subsequent: DEBUG (avoid spam).
_keypair_missing_warned: set = set()


def load_keypair_from_json(path: str) -> Optional["Keypair"]:
    """
    Load a Solana keypair from a standard [u8; 64] JSON file.

    Args:
        path: File path to the keypair JSON.

    Returns:
        Keypair object, or None if file missing, invalid, or SDK unavailable.
    """
    if not _SOLANA_AVAILABLE:
        logger.debug("[SolanaClient] SDK not available — cannot load keypair.")
        return None

    resolved = Path(path).resolve()
    if not resolved.exists():
        # First occurrence: warning (signal). Subsequent: debug (no spam).
        # T2/T3 in delegate mode hit this path repeatedly — warning each time
        # is noise that masks real keypair issues on T1.
        path_key = str(resolved)
        if path_key not in _keypair_missing_warned:
            logger.warning("[SolanaClient] Keypair file not found: %s "
                           "(further occurrences will be at DEBUG level)", resolved)
            _keypair_missing_warned.add(path_key)
        else:
            logger.debug("[SolanaClient] Keypair file not found: %s", resolved)
        return None

    try:
        with open(resolved, "r") as f:
            key_bytes = json.load(f)
        kp = Keypair.from_bytes(bytes(key_bytes))
        logger.info("[SolanaClient] Loaded keypair: %s", kp.pubkey())
        return kp
    except Exception as e:
        logger.error("[SolanaClient] Failed to load keypair from %s: %s", resolved, e)
        return None


# ---------------------------------------------------------------------------
# Memo Instruction Builder — Single Source of Truth for on-chain inscriptions
# ---------------------------------------------------------------------------
def build_memo_instruction(
    signer_pubkey, memo_text: str,
) -> Optional["Instruction"]:
    """
    Build a Solana Memo Program V2 instruction.

    Used by both SovereignSoul (directive inscription) and MeditationEpoch
    (state root commitment). Centralizing here ensures:
      - Consistent program ID usage
      - Single update point if inscription method changes (e.g., Anchor vault)
      - Uniform memo text encoding

    Args:
        signer_pubkey: The signer's Pubkey object (agent's wallet).
        memo_text: The text to inscribe on-chain (max ~566 bytes for Memo V2).

    Returns:
        Instruction object, or None if SDK unavailable or signer is None.
    """
    if not _SOLANA_AVAILABLE:
        logger.debug("[SolanaClient] SDK not available — skipping memo instruction.")
        return None

    if signer_pubkey is None:
        logger.error("[SolanaClient] Cannot build memo — no signer pubkey.")
        return None

    try:
        memo_program = Pubkey.from_string(MEMO_PROGRAM_ID)
        return Instruction(
            program_id=memo_program,
            accounts=[
                AccountMeta(
                    pubkey=signer_pubkey,
                    is_signer=True,
                    is_writable=False,
                ),
            ],
            data=memo_text.encode("utf-8"),
        )
    except Exception as e:
        logger.error("[SolanaClient] Failed to build memo instruction: %s", e)
        return None


# ---------------------------------------------------------------------------
# Compute Budget Instruction — Priority Fee Control
# ---------------------------------------------------------------------------
def build_compute_budget_instruction(
    microlamports: int,
) -> Optional["Instruction"]:
    """
    Build a ComputeBudget SetComputeUnitPrice instruction for priority fees.

    Allows any module constructing transaction instruction lists to prepend
    a priority fee without depending on network.py internals.

    Args:
        microlamports: Price per compute unit in micro-lamports.

    Returns:
        Instruction object, or None if SDK unavailable.
    """
    if not _SOLANA_AVAILABLE:
        return None

    try:
        program_id = Pubkey.from_string(COMPUTE_BUDGET_PROGRAM_ID)
        # SetComputeUnitPrice instruction: discriminator byte 3 + u64 LE price
        data = bytes([3]) + microlamports.to_bytes(8, byteorder="little")
        return Instruction(
            program_id=program_id,
            accounts=[],
            data=data,
        )
    except Exception as e:
        logger.error("[SolanaClient] Failed to build compute budget instruction: %s", e)
        return None


# ---------------------------------------------------------------------------
# Vault Program — PDA Derivation & Instruction Builders
# ---------------------------------------------------------------------------

def derive_vault_pda(
    authority_pubkey, program_id_str: str = None,
) -> Optional[tuple]:
    """
    Derive the Titan vault PDA address from the authority pubkey.

    Seeds: [b"titan_vault", authority.key()]
    Matches the Rust program's PDA derivation exactly.

    Args:
        authority_pubkey: The Titan's wallet Pubkey.
        program_id_str: Override vault program ID (defaults to VAULT_PROGRAM_ID).

    Returns:
        Tuple of (pda_pubkey, bump) or None if SDK unavailable.
    """
    if not _SOLANA_AVAILABLE or authority_pubkey is None:
        return None

    try:
        # rFP_observatory_data_loading_v1 §3.3 (2026-04-26): accept both
        # Pubkey objects and base58 strings. NetworkAccessor.pubkey returns
        # str (cached from network.info), and bytes(<str>) raises
        # "string argument without an encoding" — the silent failure that
        # made vault PDA derivation always return None on api_subprocess
        # → STATE_ROOT_ZK = STUB / DEGRADED, On-Chain Vault = "No vault data".
        if isinstance(authority_pubkey, str):
            if not authority_pubkey:
                return None
            authority_pubkey = Pubkey.from_string(authority_pubkey)
        program_id = Pubkey.from_string(program_id_str or VAULT_PROGRAM_ID)
        pda, bump = Pubkey.find_program_address(
            [VAULT_PDA_SEED, bytes(authority_pubkey)],
            program_id,
        )
        return (pda, bump)
    except Exception as e:
        logger.error("[SolanaClient] Failed to derive vault PDA: %s", e)
        return None


def build_vault_initialize_instruction(
    authority_pubkey, program_id_str: str = None,
) -> Optional["Instruction"]:
    """
    Build the initialize_vault instruction to create the Titan's vault PDA.
    Called once after Genesis Ceremony.

    Args:
        authority_pubkey: The Titan's soul keypair Pubkey.
        program_id_str: Override vault program ID.

    Returns:
        Instruction object, or None if SDK unavailable.
    """
    if not _SOLANA_AVAILABLE or authority_pubkey is None:
        return None

    try:
        program_id = Pubkey.from_string(program_id_str or VAULT_PROGRAM_ID)
        pda_result = derive_vault_pda(authority_pubkey, program_id_str)
        if pda_result is None:
            return None
        vault_pda, _ = pda_result

        system_program = Pubkey.from_string(SYSTEM_PROGRAM_ID)

        return Instruction(
            program_id=program_id,
            accounts=[
                AccountMeta(pubkey=vault_pda, is_signer=False, is_writable=True),
                AccountMeta(pubkey=authority_pubkey, is_signer=True, is_writable=True),
                AccountMeta(pubkey=system_program, is_signer=False, is_writable=False),
            ],
            data=_VAULT_IX_INITIALIZE,
        )
    except Exception as e:
        logger.error("[SolanaClient] Failed to build initialize_vault instruction: %s", e)
        return None


def build_vault_commit_instruction(
    authority_pubkey,
    state_root: bytes,
    sovereignty_index: int = 0,
    program_id_str: str = None,
) -> Optional["Instruction"]:
    """
    Build the commit_state instruction to commit a Merkle state root.

    Args:
        authority_pubkey: The Titan's soul keypair Pubkey.
        state_root: 32-byte Merkle root hash.
        sovereignty_index: Sovereignty in basis points (0-10000).
        program_id_str: Override vault program ID.

    Returns:
        Instruction object, or None if SDK unavailable.
    """
    if not _SOLANA_AVAILABLE or authority_pubkey is None:
        return None

    if len(state_root) != 32:
        logger.error("[SolanaClient] State root must be exactly 32 bytes, got %d", len(state_root))
        return None

    try:
        program_id = Pubkey.from_string(program_id_str or VAULT_PROGRAM_ID)
        pda_result = derive_vault_pda(authority_pubkey, program_id_str)
        if pda_result is None:
            return None
        vault_pda, _ = pda_result

        system_program = Pubkey.from_string(SYSTEM_PROGRAM_ID)

        # Instruction data: 8-byte discriminator + 32-byte state_root + 2-byte u16 LE
        data = _VAULT_IX_COMMIT_STATE + state_root + sovereignty_index.to_bytes(2, "little")

        return Instruction(
            program_id=program_id,
            accounts=[
                AccountMeta(pubkey=vault_pda, is_signer=False, is_writable=True),
                AccountMeta(pubkey=authority_pubkey, is_signer=True, is_writable=True),
                AccountMeta(pubkey=system_program, is_signer=False, is_writable=False),
            ],
            data=data,
        )
    except Exception as e:
        logger.error("[SolanaClient] Failed to build commit_state instruction: %s", e)
        return None


def build_vault_update_shadow_instruction(
    authority_pubkey,
    shadow_url_hash: bytes,
    program_id_str: str = None,
) -> Optional["Instruction"]:
    """
    Build the update_shadow_hash instruction after a Greater Epoch rebirth.

    Args:
        authority_pubkey: The Titan's soul keypair Pubkey.
        shadow_url_hash: 32-byte SHA-256 hash of the Shadow Drive URL.
        program_id_str: Override vault program ID.

    Returns:
        Instruction object, or None if SDK unavailable.
    """
    if not _SOLANA_AVAILABLE or authority_pubkey is None:
        return None

    if len(shadow_url_hash) != 32:
        logger.error("[SolanaClient] Shadow URL hash must be 32 bytes, got %d", len(shadow_url_hash))
        return None

    try:
        program_id = Pubkey.from_string(program_id_str or VAULT_PROGRAM_ID)
        pda_result = derive_vault_pda(authority_pubkey, program_id_str)
        if pda_result is None:
            return None
        vault_pda, _ = pda_result

        system_program = Pubkey.from_string(SYSTEM_PROGRAM_ID)

        data = _VAULT_IX_UPDATE_SHADOW + shadow_url_hash

        return Instruction(
            program_id=program_id,
            accounts=[
                AccountMeta(pubkey=vault_pda, is_signer=False, is_writable=True),
                AccountMeta(pubkey=authority_pubkey, is_signer=True, is_writable=True),
                AccountMeta(pubkey=system_program, is_signer=False, is_writable=False),
            ],
            data=data,
        )
    except Exception as e:
        logger.error("[SolanaClient] Failed to build update_shadow_hash instruction: %s", e)
        return None


def decode_vault_state(account_data: bytes) -> Optional[dict]:
    """
    Decode raw on-chain VaultState account data into a dict.

    Layout (after 8-byte Anchor discriminator):
      authority:          32 bytes (Pubkey)
      latest_root:        32 bytes
      commit_count:       8 bytes (u64 LE)
      last_commit_ts:     8 bytes (i64 LE)
      sovereignty_index:  2 bytes (u16 LE)
      shadow_url_hash:    32 bytes
      bump:               1 byte

    Args:
        account_data: Raw bytes from getAccountInfo.

    Returns:
        Dict with parsed fields, or None on failure.
    """
    if not account_data or len(account_data) < 123:
        return None

    try:
        offset = 8  # Skip Anchor discriminator
        authority = account_data[offset:offset + 32]
        offset += 32
        latest_root = account_data[offset:offset + 32]
        offset += 32
        commit_count = struct.unpack_from("<Q", account_data, offset)[0]
        offset += 8
        last_commit_ts = struct.unpack_from("<q", account_data, offset)[0]
        offset += 8
        sovereignty_index = struct.unpack_from("<H", account_data, offset)[0]
        offset += 2
        shadow_url_hash = account_data[offset:offset + 32]
        offset += 32
        bump = account_data[offset]

        return {
            "authority": authority.hex() if _SOLANA_AVAILABLE else authority.hex(),
            "latest_root": latest_root.hex(),
            "commit_count": commit_count,
            "last_commit_ts": last_commit_ts,
            "sovereignty_index": sovereignty_index,
            "sovereignty_percent": round(sovereignty_index / 100, 2),
            "shadow_url_hash": shadow_url_hash.hex(),
            "bump": bump,
        }
    except Exception as e:
        logger.error("[SolanaClient] Failed to decode VaultState: %s", e)
        return None


# ---------------------------------------------------------------------------
# ZK-State Decoding — Light Protocol Account Data Parser
# ---------------------------------------------------------------------------

# Titan ZK-Omni-Schema V2.0
# Expected structure after decoding:
#   {
#     "schema": "v2.0-sage",
#     "bio": { "gen": int, "mood": float, "sovereignty": float },
#     "mems": { "latest_memory_hash": str, "persistent_count": int },
#     "gates": { "sovereign_ratio": float, "research_ratio": float },
#     "body": { "sol_balance": float, "shadow_drive_url": str },
#   }

def decode_zk_account_data(account_data: bytes) -> dict:
    """
    Decode raw Light Protocol ZK-compressed account data into the
    Titan's standardized ZK-Omni-Schema JSON.

    Used by the Resurrection Protocol to extract the Titan's last known
    cognitive state from the Solana state tree without high-level logic
    needing to understand the wire format.

    The data layout:
      - Bytes 0-7: Anchor discriminator (8 bytes, skipped)
      - Bytes 8+: JSON payload (UTF-8 encoded, variable length)

    Args:
        account_data: Raw bytes from the ZK-compressed account read.

    Returns:
        Parsed dict conforming to ZK-Omni-Schema, or empty dict on failure.
    """
    if not account_data:
        logger.warning("[SolanaClient] Empty account data — nothing to decode.")
        return {}

    try:
        # Skip 8-byte Anchor discriminator
        json_bytes = account_data[8:] if len(account_data) > 8 else account_data

        # Strip trailing null bytes (common in on-chain fixed-size buffers)
        json_bytes = json_bytes.rstrip(b"\x00")

        if not json_bytes:
            logger.warning("[SolanaClient] Account data contains only discriminator/nulls.")
            return {}

        decoded = json.loads(json_bytes.decode("utf-8"))

        # Validate schema version
        schema = decoded.get("schema", "")
        if not schema.startswith("v2.0"):
            logger.warning(
                "[SolanaClient] Unexpected schema version: '%s'. "
                "Expected v2.0-*. Returning raw decoded data.",
                schema,
            )

        return decoded

    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error("[SolanaClient] Failed to decode ZK account data: %s", e)
        return {}
    except Exception as e:
        logger.error("[SolanaClient] Unexpected error decoding ZK data: %s", e)
        return {}


def encode_zk_account_data(state: dict) -> bytes:
    """
    Encode a ZK-Omni-Schema dict into bytes for on-chain storage.

    Inverse of decode_zk_account_data. Used by Meditation Epoch
    to prepare the cognitive state snapshot for ZK-compressed account writes.

    Args:
        state: Dict conforming to ZK-Omni-Schema.

    Returns:
        Bytes: 8-byte discriminator + UTF-8 JSON payload.
    """
    # Anchor discriminator for TitanState account (first 8 bytes of SHA256("account:TitanState"))
    discriminator = hashlib.sha256(b"account:TitanState").digest()[:8]
    payload = json.dumps(state, sort_keys=True, default=str).encode("utf-8")
    return discriminator + payload


# ---------------------------------------------------------------------------
# Light Protocol — ZK Compression Constants & Builders
# ---------------------------------------------------------------------------

# Light Protocol program IDs (mainnet + devnet)
LIGHT_SYSTEM_PROGRAM_ID = "SySTEM1eSU2p4BGQfQpimFEWWSC1XDFeun3Nqzz3rT7"
LIGHT_REGISTRY_PROGRAM_ID = "Lighton6oQpVkeewmo2mcPTQQp7kYHr4fWpAgJyEmDX"
LIGHT_COMPRESSION_PROGRAM_ID = "compr6CUsB5m2jS4Y3831ztGSTnDpnKJTKS95d64XVq"

# Anchor discriminators for ZK compression instructions (from IDL build)
_VAULT_IX_COMPRESS_MEMORY_BATCH = bytes([105, 76, 210, 140, 189, 129, 57, 135])
_VAULT_IX_APPEND_EPOCH_SNAPSHOT = bytes([213, 217, 65, 120, 202, 70, 5, 131])


def compute_batch_root(memory_hashes: List[bytes]) -> bytes:
    """
    Compute the Merkle root of a list of memory hashes.
    Used client-side before building the compress instruction.

    Simple binary Merkle tree using SHA-256:
    - Leaf nodes are the memory hashes themselves.
    - Internal nodes are SHA-256(left || right).
    - If odd count, the last node is promoted (not duplicated).

    Args:
        memory_hashes: List of 32-byte SHA-256 hashes.

    Returns:
        32-byte Merkle root. Single hash returns itself.
        Empty list returns 32 zero bytes.
    """
    if not memory_hashes:
        return b"\x00" * 32

    # Start with leaf level
    level = list(memory_hashes)

    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            if i + 1 < len(level):
                right = level[i + 1]
                combined = hashlib.sha256(left + right).digest()
            else:
                # Odd node promoted without hashing
                combined = left
            next_level.append(combined)
        level = next_level

    return level[0]


def build_compress_memory_batch_instruction(
    authority_pubkey,
    batch_root: bytes,
    node_count: int,
    epoch_id: int,
    sovereignty_score: int,
    proof_bytes: bytes = None,
    merkle_context: dict = None,
    program_id_str: str = None,
) -> Optional["Instruction"]:
    """
    Build the compress_memory_batch instruction with ZK proof context.

    This builds only the instruction data portion. The full remaining accounts
    (Light Protocol system accounts + tree accounts) must be appended by the
    caller based on the Photon validity proof response.

    Args:
        authority_pubkey: The Titan's wallet Pubkey.
        batch_root: 32-byte Merkle root of memory hashes.
        node_count: Number of memories in the batch.
        epoch_id: Meditation cycle number.
        sovereignty_score: Basis points (0-10000).
        proof_bytes: 128-byte Groth16 proof (a:32 + b:64 + c:32). None for empty proof.
        merkle_context: Tree addresses and indices from Photon (unused in data, used by caller for accounts).
        program_id_str: Override vault program ID.

    Returns:
        Instruction object (without remaining accounts), or None if SDK unavailable.
    """
    if not _SOLANA_AVAILABLE or authority_pubkey is None:
        return None

    if len(batch_root) != 32:
        logger.error("[SolanaClient] batch_root must be 32 bytes, got %d", len(batch_root))
        return None

    try:
        program_id = Pubkey.from_string(program_id_str or VAULT_PROGRAM_ID)
        pda_result = derive_vault_pda(authority_pubkey, program_id_str)
        if pda_result is None:
            return None
        vault_pda, _ = pda_result

        # Serialize proof: Option<CompressedProof> where CompressedProof = {a:[u8;32], b:[u8;64], c:[u8;32]}
        # Borsh Option: 1 byte (0=None, 1=Some) + data
        if proof_bytes and len(proof_bytes) == 128:
            proof_data = bytes([1]) + proof_bytes
        else:
            proof_data = bytes([0])

        # Instruction data layout:
        #   8 bytes: discriminator
        #   1+128 bytes: ValidityProof (Option<CompressedProof>)
        #   32 bytes: batch_root
        #   2 bytes: node_count (u16 LE)
        #   8 bytes: epoch_id (u64 LE)
        #   2 bytes: sovereignty_score (u16 LE)
        #   1 byte: output_tree_index (u8)
        data = (
            _VAULT_IX_COMPRESS_MEMORY_BATCH
            + proof_data
            + batch_root
            + struct.pack("<H", node_count)
            + struct.pack("<Q", epoch_id)
            + struct.pack("<H", sovereignty_score)
            + bytes([0])  # output_tree_index — set by caller or default 0
        )

        return Instruction(
            program_id=program_id,
            accounts=[
                AccountMeta(pubkey=vault_pda, is_signer=False, is_writable=False),
                AccountMeta(pubkey=authority_pubkey, is_signer=True, is_writable=True),
            ],
            data=data,
        )
    except Exception as e:
        logger.error("[SolanaClient] Failed to build compress_memory_batch: %s", e)
        return None


def build_append_epoch_snapshot_instruction(
    authority_pubkey,
    state_root: bytes,
    memory_count: int,
    sovereignty_score: int,
    shadow_url_hash: bytes,
    proof_bytes: bytes = None,
    merkle_context: dict = None,
    program_id_str: str = None,
) -> Optional["Instruction"]:
    """
    Build the append_epoch_snapshot instruction with ZK proof context.

    Args:
        authority_pubkey: The Titan's wallet Pubkey.
        state_root: 32-byte full cognitive state root.
        memory_count: Total memories at this point.
        sovereignty_score: Basis points (0-10000).
        shadow_url_hash: 32-byte SHA-256 hash of Shadow Drive archive.
        proof_bytes: 128-byte Groth16 proof. None for empty proof.
        merkle_context: Tree addresses and indices from Photon.
        program_id_str: Override vault program ID.

    Returns:
        Instruction object (without remaining accounts), or None if SDK unavailable.
    """
    if not _SOLANA_AVAILABLE or authority_pubkey is None:
        return None

    if len(state_root) != 32 or len(shadow_url_hash) != 32:
        logger.error("[SolanaClient] state_root and shadow_url_hash must be 32 bytes each.")
        return None

    try:
        program_id = Pubkey.from_string(program_id_str or VAULT_PROGRAM_ID)
        pda_result = derive_vault_pda(authority_pubkey, program_id_str)
        if pda_result is None:
            return None
        vault_pda, _ = pda_result

        # Serialize proof
        if proof_bytes and len(proof_bytes) == 128:
            proof_data = bytes([1]) + proof_bytes
        else:
            proof_data = bytes([0])

        # Instruction data layout:
        #   8 bytes: discriminator
        #   1+128 bytes: ValidityProof
        #   32 bytes: state_root
        #   8 bytes: memory_count (u64 LE)
        #   2 bytes: sovereignty_score (u16 LE)
        #   32 bytes: shadow_url_hash
        #   1 byte: output_tree_index
        data = (
            _VAULT_IX_APPEND_EPOCH_SNAPSHOT
            + proof_data
            + state_root
            + struct.pack("<Q", memory_count)
            + struct.pack("<H", sovereignty_score)
            + shadow_url_hash
            + bytes([0])
        )

        return Instruction(
            program_id=program_id,
            accounts=[
                AccountMeta(pubkey=vault_pda, is_signer=False, is_writable=False),
                AccountMeta(pubkey=authority_pubkey, is_signer=True, is_writable=True),
            ],
            data=data,
        )
    except Exception as e:
        logger.error("[SolanaClient] Failed to build append_epoch_snapshot: %s", e)
        return None


def decode_compressed_memory_batch(data: bytes) -> Optional[dict]:
    """
    Decode raw CompressedMemoryBatch data from Photon response.

    Layout (Borsh-serialized, no Anchor discriminator for compressed accounts):
      authority:          32 bytes
      epoch_id:           8 bytes (u64 LE)
      timestamp:          8 bytes (i64 LE)
      sovereignty_score:  2 bytes (u16 LE)
      batch_root:         32 bytes
      node_count:         2 bytes (u16 LE)
    Total: 84 bytes

    Args:
        data: Raw bytes from Photon getCompressedAccount.

    Returns:
        Decoded dict, or None on failure.
    """
    if not data or len(data) < 84:
        return None

    try:
        offset = 0
        authority = data[offset:offset + 32]
        offset += 32
        epoch_id = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        timestamp = struct.unpack_from("<q", data, offset)[0]
        offset += 8
        sovereignty_score = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        batch_root = data[offset:offset + 32]
        offset += 32
        node_count = struct.unpack_from("<H", data, offset)[0]

        return {
            "type": "CompressedMemoryBatch",
            "authority": authority.hex(),
            "epoch_id": epoch_id,
            "timestamp": timestamp,
            "sovereignty_score": sovereignty_score,
            "sovereignty_percent": round(sovereignty_score / 100, 2),
            "batch_root": batch_root.hex(),
            "node_count": node_count,
        }
    except Exception as e:
        logger.error("[SolanaClient] Failed to decode CompressedMemoryBatch: %s", e)
        return None


def decode_compressed_epoch_snapshot(data: bytes) -> Optional[dict]:
    """
    Decode raw CompressedEpochSnapshot data from Photon response.

    Layout (Borsh-serialized):
      authority:          32 bytes
      epoch_number:       8 bytes (u64 LE)
      state_root:         32 bytes
      memory_count:       8 bytes (u64 LE)
      sovereignty_score:  2 bytes (u16 LE)
      shadow_url_hash:    32 bytes
      timestamp:          8 bytes (i64 LE)
    Total: 122 bytes

    Args:
        data: Raw bytes from Photon getCompressedAccount.

    Returns:
        Decoded dict, or None on failure.
    """
    if not data or len(data) < 122:
        return None

    try:
        offset = 0
        authority = data[offset:offset + 32]
        offset += 32
        epoch_number = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        state_root = data[offset:offset + 32]
        offset += 32
        memory_count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        sovereignty_score = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        shadow_url_hash = data[offset:offset + 32]
        offset += 32
        timestamp = struct.unpack_from("<q", data, offset)[0]

        return {
            "type": "CompressedEpochSnapshot",
            "authority": authority.hex(),
            "epoch_number": epoch_number,
            "state_root": state_root.hex(),
            "memory_count": memory_count,
            "sovereignty_score": sovereignty_score,
            "sovereignty_percent": round(sovereignty_score / 100, 2),
            "shadow_url_hash": shadow_url_hash.hex(),
            "timestamp": timestamp,
        }
    except Exception as e:
        logger.error("[SolanaClient] Failed to decode CompressedEpochSnapshot: %s", e)
        return None


# ---------------------------------------------------------------------------
# Metaplex Core — NFT Instruction Builders
# ---------------------------------------------------------------------------
# Program: CoREENxT6tW1HoK8ypY1SxRMZTcVPm7R94rH4PZNhX7d
# Instruction discriminators are single-byte enums (NOT Anchor 8-byte SHA):
#   CreateV1 = 0, UpdateV1 = 15, AddPluginV1 = 2
# Serialization: Borsh (LE integers, length-prefixed UTF-8 strings)
# Reference: https://developers.metaplex.com/core/create-asset

# MPL Core instruction discriminators (enum variant indices)
_MPL_IX_CREATE_V1 = bytes([0])
_MPL_IX_ADD_PLUGIN_V1 = bytes([2])
_MPL_IX_REMOVE_PLUGIN_V1 = bytes([4])
_MPL_IX_UPDATE_PLUGIN_V1 = bytes([6])
_MPL_IX_UPDATE_V1 = bytes([15])

# MPL Core Plugin enum variant indices (Borsh u8)
# 0=Royalties, 1=FreezeDelegate, 2=BurnDelegate, 3=TransferDelegate,
# 4=UpdateDelegate, 5=PermanentFreezeDelegate, 6=Attributes,
# 7=PermanentTransferDelegate, 8=PermanentBurnDelegate, 9=Edition, 10=MasterEdition, 11=AddBlocker
_MPL_PLUGIN_ATTRIBUTES = 6

# SPL Noop Program (optional log wrapper)
SPL_NOOP_PROGRAM_ID = "noopb9bkMVfRPU8AsbpTUg8AQkHtKwMYZiFUjNRtMmV"


def _borsh_string(s: str) -> bytes:
    """Encode a string as Borsh: 4-byte LE length + UTF-8 bytes."""
    encoded = s.encode("utf-8")
    return len(encoded).to_bytes(4, "little") + encoded


def _borsh_option_none() -> bytes:
    """Encode Borsh Option::None."""
    return bytes([0])


def _borsh_option_some(data: bytes) -> bytes:
    """Encode Borsh Option::Some(data)."""
    return bytes([1]) + data


def _borsh_attributes_plugin(attributes: dict) -> bytes:
    """
    Serialize an Attributes plugin as a PluginAuthorityPair for CreateV1.

    Wire format (PluginAuthorityPair):
      plugin: Plugin (Borsh enum — u32 LE variant index + variant data)
        Plugin::Attributes (variant 11):
          attribute_list: Vec<Attribute>
            each: { key: borsh_string, value: borsh_string }
      authority: Option<PluginAuthority> (None = defaults to owner)
    """
    data = b""

    # Plugin enum variant index (u8): Attributes = 6
    data += bytes([_MPL_PLUGIN_ATTRIBUTES])

    # Attributes data: Vec<Attribute>
    items = list(attributes.items())
    data += len(items).to_bytes(4, "little")
    for key, value in items:
        data += _borsh_string(str(key))
        data += _borsh_string(str(value))

    # Authority: Option<PluginAuthority> — None (defaults to update authority)
    data += _borsh_option_none()

    return data


def build_mpl_core_create_v1(
    asset_pubkey,
    payer_pubkey,
    name: str,
    uri: str,
    owner_pubkey=None,
    collection_pubkey=None,
    attributes: dict = None,
) -> Optional["Instruction"]:
    """
    Build a Metaplex Core CreateV1 instruction to mint a new NFT asset.

    Args:
        asset_pubkey: New keypair pubkey for the asset account (must be signer).
        payer_pubkey: Transaction fee payer (must be signer).
        name: NFT name (max 32 chars recommended).
        uri: Metadata JSON URI (Shadow Drive, Arweave, or IPFS).
        owner_pubkey: NFT owner. Defaults to payer if None.
        collection_pubkey: Optional collection address.
        attributes: Optional dict of key-value attributes for the Attributes plugin.

    Returns:
        Instruction object, or None if SDK unavailable.
    """
    if not _SOLANA_AVAILABLE:
        return None

    try:
        program_id = Pubkey.from_string(MPL_CORE_PROGRAM_ID)
        system_program = Pubkey.from_string(SYSTEM_PROGRAM_ID)

        # Build accounts list
        accounts = [
            # 0: asset (signer, writable) — new NFT account
            AccountMeta(pubkey=asset_pubkey, is_signer=True, is_writable=True),
        ]

        # 1: collection (optional, writable if present)
        if collection_pubkey:
            accounts.append(AccountMeta(pubkey=collection_pubkey, is_signer=False, is_writable=True))
        else:
            accounts.append(AccountMeta(pubkey=program_id, is_signer=False, is_writable=False))

        # 2: authority (optional signer) — same as payer for our use case
        accounts.append(AccountMeta(pubkey=payer_pubkey, is_signer=True, is_writable=False))

        # 3: payer (signer, writable)
        accounts.append(AccountMeta(pubkey=payer_pubkey, is_signer=True, is_writable=True))

        # 4: owner (optional) — defaults to payer
        if owner_pubkey and owner_pubkey != payer_pubkey:
            accounts.append(AccountMeta(pubkey=owner_pubkey, is_signer=False, is_writable=False))
        else:
            accounts.append(AccountMeta(pubkey=payer_pubkey, is_signer=False, is_writable=False))

        # 5: updateAuthority (optional) — same as payer
        accounts.append(AccountMeta(pubkey=payer_pubkey, is_signer=False, is_writable=False))

        # 6: systemProgram
        accounts.append(AccountMeta(pubkey=system_program, is_signer=False, is_writable=False))

        # 7: logWrapper (optional — use program ID as sentinel when not using SPL Noop)
        accounts.append(AccountMeta(pubkey=program_id, is_signer=False, is_writable=False))

        # Build instruction data
        data = _MPL_IX_CREATE_V1

        # DataState enum: 0 = AccountState (stored on the asset account itself)
        data += bytes([0])

        # name (Borsh string)
        data += _borsh_string(name)

        # uri (Borsh string)
        data += _borsh_string(uri)

        # plugins: Option<Vec<PluginAuthorityPair>>
        if attributes:
            plugin_data = _borsh_attributes_plugin(attributes)
            # Option::Some + Vec length (1 plugin)
            data += bytes([1])  # Option::Some
            data += (1).to_bytes(4, "little")  # Vec length = 1
            data += plugin_data
        else:
            data += _borsh_option_none()

        return Instruction(
            program_id=program_id,
            accounts=accounts,
            data=data,
        )

    except Exception as e:
        logger.error("[SolanaClient] Failed to build MPL Core CreateV1: %s", e)
        return None


def build_mpl_core_update_v1(
    asset_pubkey,
    authority_pubkey,
    new_name: str = None,
    new_uri: str = None,
    collection_pubkey=None,
    revoke_authority: bool = False,
) -> Optional["Instruction"]:
    """
    Build a Metaplex Core UpdateV1 instruction to update NFT metadata.

    Args:
        asset_pubkey: The existing NFT asset account.
        authority_pubkey: Update authority (must be signer).
        new_name: Updated name (None = no change).
        new_uri: Updated URI (None = no change).
        collection_pubkey: Collection address if asset is in a collection.
        revoke_authority: If True, set update authority to None (makes NFT permanently immutable).

    Returns:
        Instruction object, or None if SDK unavailable.
    """
    if not _SOLANA_AVAILABLE:
        return None

    try:
        program_id = Pubkey.from_string(MPL_CORE_PROGRAM_ID)
        system_program = Pubkey.from_string(SYSTEM_PROGRAM_ID)

        # Account layout matches Metaplex Core UpdateV1:
        #   0: asset, 1: collection?, 2: payer, 3: authority?, 4: systemProgram, 5: logWrapper?
        accounts = [
            # 0: asset (writable)
            AccountMeta(pubkey=asset_pubkey, is_signer=False, is_writable=True),
            # 1: collection (optional — program_id placeholder when absent)
            AccountMeta(
                pubkey=collection_pubkey if collection_pubkey else program_id,
                is_signer=False,
                is_writable=bool(collection_pubkey),
            ),
            # 2: payer (signer, writable)
            AccountMeta(pubkey=authority_pubkey, is_signer=True, is_writable=True),
            # 3: authority (signer — update authority)
            AccountMeta(pubkey=authority_pubkey, is_signer=True, is_writable=False),
            # 4: systemProgram
            AccountMeta(pubkey=system_program, is_signer=False, is_writable=False),
            # 5: logWrapper (optional — program_id placeholder)
            AccountMeta(pubkey=program_id, is_signer=False, is_writable=False),
        ]

        # Build instruction data
        data = _MPL_IX_UPDATE_V1

        # new_name: Option<String>
        if new_name is not None:
            data += _borsh_option_some(_borsh_string(new_name))
        else:
            data += _borsh_option_none()

        # new_uri: Option<String>
        if new_uri is not None:
            data += _borsh_option_some(_borsh_string(new_uri))
        else:
            data += _borsh_option_none()

        # new_update_authority: Option<UpdateAuthority>
        # UpdateAuthority enum: 0=None, 1=Address(Pubkey), 2=Collection(Pubkey)
        if revoke_authority:
            # Option::Some(UpdateAuthority::None) — permanently immutable
            data += _borsh_option_some(bytes([0]))
        else:
            data += _borsh_option_none()

        return Instruction(
            program_id=program_id,
            accounts=accounts,
            data=data,
        )

    except Exception as e:
        logger.error("[SolanaClient] Failed to build MPL Core UpdateV1: %s", e)
        return None


def build_mpl_core_update_plugin_v1(
    asset_pubkey,
    authority_pubkey,
    attributes: dict,
    collection_pubkey=None,
) -> Optional["Instruction"]:
    """
    Build a Metaplex Core UpdatePluginV1 instruction to update on-chain attributes.

    Replaces the entire Attributes plugin data with new key-value pairs.

    Args:
        asset_pubkey: The existing NFT asset account.
        authority_pubkey: Update authority (must be signer).
        attributes: Dict of key→value pairs for the Attributes plugin.
        collection_pubkey: Collection address if asset is in a collection.

    Returns:
        Instruction object, or None if SDK unavailable.
    """
    if not _SOLANA_AVAILABLE:
        return None

    try:
        program_id = Pubkey.from_string(MPL_CORE_PROGRAM_ID)
        system_program = Pubkey.from_string(SYSTEM_PROGRAM_ID)

        # Account layout matches Metaplex Core UpdatePluginV1:
        #   0: asset, 1: collection?, 2: payer, 3: authority?, 4: systemProgram, 5: logWrapper?
        accounts = [
            # 0: asset (writable)
            AccountMeta(pubkey=asset_pubkey, is_signer=False, is_writable=True),
            # 1: collection (optional — program_id placeholder when absent)
            AccountMeta(
                pubkey=collection_pubkey if collection_pubkey else program_id,
                is_signer=False,
                is_writable=bool(collection_pubkey),
            ),
            # 2: payer (signer, writable)
            AccountMeta(pubkey=authority_pubkey, is_signer=True, is_writable=True),
            # 3: authority (signer — update authority)
            AccountMeta(pubkey=authority_pubkey, is_signer=True, is_writable=False),
            # 4: systemProgram
            AccountMeta(pubkey=system_program, is_signer=False, is_writable=False),
            # 5: logWrapper (optional — program_id placeholder)
            AccountMeta(pubkey=program_id, is_signer=False, is_writable=False),
        ]

        # Build instruction data: discriminator + Plugin (Attributes variant)
        # UpdatePluginV1Args { plugin: Plugin }
        data = _MPL_IX_UPDATE_PLUGIN_V1

        # Plugin enum: Attributes (variant 6) + attribute_list: Vec<Attribute>
        data += bytes([_MPL_PLUGIN_ATTRIBUTES])
        items = list(attributes.items())
        data += len(items).to_bytes(4, "little")
        for key, value in items:
            data += _borsh_string(str(key))
            data += _borsh_string(str(value))

        return Instruction(
            program_id=program_id,
            accounts=accounts,
            data=data,
        )

    except Exception as e:
        logger.error("[SolanaClient] Failed to build MPL Core UpdatePluginV1: %s", e)
        return None


def decode_mpl_core_asset(account_data: bytes) -> Optional[dict]:
    """
    Decode a Metaplex Core Asset account.

    Asset account layout:
      key:              1 byte  (enum: 1 = AssetV1)
      owner:            32 bytes (Pubkey)
      update_authority: 33 bytes (1 byte type + 32 bytes Pubkey)
      name:             4 bytes LE length + UTF-8
      uri:              4 bytes LE length + UTF-8
      seq:              Option<u64> (1 byte tag + 8 bytes if Some)

    Returns:
        Dict with parsed fields, or None on failure.
    """
    if not account_data or len(account_data) < 70:
        return None

    try:
        offset = 0

        # Key (1 byte — should be 1 for AssetV1)
        key = account_data[offset]
        offset += 1
        if key != 1:
            logger.debug("[SolanaClient] Not an AssetV1 account (key=%d)", key)
            return None

        # Owner (32 bytes)
        owner = account_data[offset:offset + 32]
        offset += 32

        # Update Authority (Borsh enum: 0=None, 1=Address(Pubkey), 2=Collection(Pubkey))
        ua_type = account_data[offset]
        offset += 1
        ua_pubkey = None
        if ua_type in (1, 2):
            ua_pubkey = account_data[offset:offset + 32]
            offset += 32

        # Name (Borsh string)
        name_len = struct.unpack_from("<I", account_data, offset)[0]
        offset += 4
        name = account_data[offset:offset + name_len].decode("utf-8")
        offset += name_len

        # URI (Borsh string)
        uri_len = struct.unpack_from("<I", account_data, offset)[0]
        offset += 4
        uri = account_data[offset:offset + uri_len].decode("utf-8")
        offset += uri_len

        from solders.pubkey import Pubkey as _Pk
        result = {
            "key": "AssetV1",
            "owner": str(_Pk.from_bytes(owner)),
            "update_authority_type": ua_type,
            "name": name,
            "uri": uri,
        }
        if ua_pubkey is not None:
            result["update_authority"] = str(_Pk.from_bytes(ua_pubkey))
        return result

    except Exception as e:
        logger.error("[SolanaClient] Failed to decode MPL Core asset: %s", e)
        return None


async def fetch_mpl_core_asset(network_client, asset_pubkey_str: str) -> Optional[dict]:
    """
    Fetch and decode a Metaplex Core NFT asset from chain.

    Args:
        network_client: HybridNetworkClient with RPC access.
        asset_pubkey_str: Base58 address of the asset account.

    Returns:
        Decoded asset dict, or None if not found or decode fails.
    """
    try:
        import httpx

        rpc_urls = getattr(network_client, "rpc_urls", None) or getattr(network_client, "_rpc_urls", None) or []
        rpc_url = rpc_urls[0] if rpc_urls else "https://api.mainnet-beta.solana.com"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(rpc_url, json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [asset_pubkey_str, {"encoding": "base64"}],
            })
            result = resp.json().get("result", {})
            value = result.get("value")
            if not value:
                return None

            import base64
            data = base64.b64decode(value["data"][0])

            decoded = decode_mpl_core_asset(data)
            if decoded:
                decoded["address"] = asset_pubkey_str
                decoded["program_owner"] = value.get("owner", "")
            return decoded

    except Exception as e:
        logger.error("[SolanaClient] Failed to fetch MPL Core asset: %s", e)
        return None
