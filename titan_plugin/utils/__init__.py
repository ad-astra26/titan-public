"""
titan_plugin.utils — Shared utility modules for the Sovereign Stack.

Exposes key primitives for clean imports:
    from titan_plugin.utils import generate_state_hash, build_memo_instruction
"""
# Crypto Root of Trust
from .crypto import (
    generate_state_hash,
    hash_file,
    verify_file_integrity,
    sign_solana_payload,
    sign_shadow_drive_message,
    verify_maker_signature,
    get_hardware_fingerprint,
    encrypt_for_machine,
    decrypt_for_machine,
)

# Shamir Secret Sharing
from .shamir import (
    split_secret,
    combine_shares,
    verify_all_combinations,
    create_maker_envelope,
    parse_maker_envelope,
    encrypt_shard3,
    decrypt_shard3,
)

# Solana Primitive Facade
from .solana_client import (
    is_available as solana_available,
    parse_pubkey,
    load_keypair_from_json,
    build_memo_instruction,
    build_compute_budget_instruction,
    decode_zk_account_data,
    encode_zk_account_data,
)

__all__ = [
    # crypto
    "generate_state_hash",
    "hash_file",
    "verify_file_integrity",
    "sign_solana_payload",
    "sign_shadow_drive_message",
    "verify_maker_signature",
    "get_hardware_fingerprint",
    "encrypt_for_machine",
    "decrypt_for_machine",
    # shamir
    "split_secret",
    "combine_shares",
    "verify_all_combinations",
    "create_maker_envelope",
    "parse_maker_envelope",
    "encrypt_shard3",
    "decrypt_shard3",
    # solana_client
    "solana_available",
    "parse_pubkey",
    "load_keypair_from_json",
    "build_memo_instruction",
    "build_compute_budget_instruction",
    "decode_zk_account_data",
    "encode_zk_account_data",
]
