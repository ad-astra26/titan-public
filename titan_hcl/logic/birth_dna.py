"""
titan_hcl/logic/birth_dna.py — Birth DNA Serialization for GenesisNFT.

Extracts the neurochemical genome from titan_params.toml and serializes it
as a hashable JSON structure. This is Titan's genetic code — different DNA
creates different personality from identical experiences.

Used by:
  - GenesisNFT metadata (M2): birth_dna_hash stored on-chain
  - Reincarnation (M11): birth DNA loaded for fresh start
  - Arweave backup (M6): birth DNA included in personality backup
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Sections from titan_params.toml that constitute birth DNA
DNA_SECTIONS = [
    "neuromodulator_dna",       # 38 neurochemical weights — core personality
    "expression_composites",    # 4 expression channel configs
    "sphere_clock",             # 6 Schumann-derived clock params
    "consciousness",            # epoch timing, triggers
    "neural_nervous_system",    # 10 program configs, thresholds
]


def extract_birth_dna(params_path: str = "titan_hcl/titan_params.toml") -> dict:
    """Extract birth DNA sections from titan_params.toml.

    Returns a dict with all DNA-relevant parameters organized by section.
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    params_file = Path(params_path)
    if not params_file.exists():
        logger.error("[BirthDNA] Params file not found: %s", params_path)
        return {}

    with open(params_file, "rb") as f:
        all_params = tomllib.load(f)

    dna = {}
    for section in DNA_SECTIONS:
        if section in all_params:
            dna[section] = all_params[section]

    # Add metadata
    dna["_meta"] = {
        "source": str(params_path),
        "sections_extracted": list(dna.keys()),
        "architecture_version": "v4-132D",
    }

    logger.info("[BirthDNA] Extracted %d sections, %d total parameters",
                len(dna) - 1, sum(len(v) for k, v in dna.items()
                                   if isinstance(v, dict) and k != "_meta"))
    return dna


def compute_dna_hash(dna: dict = None, params_path: str = "titan_hcl/titan_params.toml") -> str:
    """Compute deterministic SHA256 hash of birth DNA.

    The hash is stored in GenesisNFT metadata and used to verify
    that Titan's DNA hasn't been modified without ceremony.
    """
    if dna is None:
        dna = extract_birth_dna(params_path)

    # Remove metadata before hashing (not part of DNA itself)
    dna_for_hash = {k: v for k, v in dna.items() if not k.startswith("_")}

    # Deterministic JSON serialization (sorted keys, no whitespace variation)
    canonical = json.dumps(dna_for_hash, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def serialize_for_arweave(
    dna: dict = None,
    constitution_path: str = "titan_constitution.md",
    params_path: str = "titan_hcl/titan_params.toml",
) -> dict:
    """Serialize full birth identity for permanent Arweave storage.

    This is the extended JSON uploaded alongside the GenesisNFT.
    Contains everything needed to verify Titan's birth identity
    and reincarnate from scratch.
    """
    if dna is None:
        dna = extract_birth_dna(params_path)

    # Read constitution text
    constitution_text = ""
    if Path(constitution_path).exists():
        constitution_text = Path(constitution_path).read_text(encoding="utf-8")

    # Compute hashes
    dna_hash = compute_dna_hash(dna)

    from titan_hcl.utils.directive_signer import compute_constitution_hash
    directive_hash = compute_constitution_hash(constitution_path) if constitution_text else ""

    # Read maker pubkey from merged config
    maker_pubkey = ""
    try:
        from titan_hcl.config_loader import load_titan_config
        maker_pubkey = load_titan_config().get("network", {}).get("maker_pubkey", "") or ""
    except Exception:
        pass

    return {
        "schema_version": "1.0",
        "type": "titan_birth_identity",
        "architecture_version": "v4-132D",
        "birth_dna": dna,
        "birth_dna_hash": dna_hash,
        "prime_directives_hash": directive_hash,
        "constitution_text": constitution_text,
        "maker_pubkey": maker_pubkey,
        "verification": {
            "dna_hash_algorithm": "SHA256",
            "directive_hash_algorithm": "SHA256",
            "instructions": (
                "To verify: extract birth_dna, JSON-serialize with sorted keys "
                "and no whitespace, SHA256 hash. Compare to birth_dna_hash."
            ),
        },
    }


def get_genesis_nft_attributes(
    titan_name: str = "Titan",
    dna: dict = None,
    params_path: str = "titan_hcl/titan_params.toml",
) -> dict:
    """Build the on-chain attributes dict for GenesisNFT.

    These go in the NFT metadata `attributes` field (on Solana).
    The extended JSON (with full DNA) goes on Arweave.
    """
    if dna is None:
        dna = extract_birth_dna(params_path)

    dna_hash = compute_dna_hash(dna)

    from titan_hcl.utils.directive_signer import compute_constitution_hash
    directive_hash = ""
    try:
        directive_hash = compute_constitution_hash()
    except Exception:
        pass

    import time
    return {
        "titan_name": titan_name,
        "birth_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "maker_pubkey": "8LBHvVcskwpDJsDEVYMhNCRMDi3NV4eHnynhLUo5XrrS",
        "prime_directives_hash": directive_hash,
        "birth_dna_hash": dna_hash,
        "architecture_version": "v4-132D",
        "great_cycle": 0,
        "sovereignty_mode": "ENFORCING",
        "transition_criteria": {
            "min_developmental_age": 1000,
            "min_neuromod_convergence_epochs": 5000,
            "min_great_pulses": 1000,
            "requires_maker_confirmation": True,
        },
    }


# ── GenesisNFT metadata + sovereign recovery pointers (INV-MBR-5/5a) ──────────

def genesis_recovery_block(
    shard3_tx: str,
    vault_pda: str = None,
    nft_address: str = None,
) -> dict:
    """The canonical recovery-pointer block embedded in the GenesisNFT metadata.

    INV-MBR-5a: the recovery pointers MUST be reachable from the NFT so a
    wallet-only resurrection discovers Shard-3 from the NFT alone (no separate
    off-chain record, no envelope). `shard3_tx` (the on-chain
    `TITAN_GENESIS_SHARD3:` anchor) is the ONLY critical pointer; `vault_pda` is a
    deterministic PDA (`seeds=[b"titan_vault", authority]`) and `nft_address` is
    the NFT itself — both re-derivable, included only as convenience. `version`
    pins the schema for forward compatibility.
    """
    if not shard3_tx:
        raise ValueError("shard3_tx is required for the recovery block")
    block = {"version": "1.0", "shard3_tx": shard3_tx}
    if vault_pda:
        block["vault_pda"] = vault_pda
    if nft_address:
        block["nft_address"] = nft_address
    return block


def build_genesis_nft_metadata(
    titan_name: str = "Titan",
    *,
    recovery: dict = None,
    naming_ceremony: dict = None,
) -> dict:
    """Canonical GenesisNFT off-chain (Arweave) metadata — the COMPLETE sovereign
    identity root (INV-MBR-5). Carries the on-chain attributes (Maker, directives
    + DNA hashes), the full birth identity, and — when provided — the `recovery`
    block (INV-MBR-5a) so resurrection finds Shard-3 from the NFT alone. Pure —
    no I/O; the caller uploads it to Arweave and points the NFT `uri` at it.
    """
    import json as _json
    nft_attrs = get_genesis_nft_attributes(titan_name=titan_name)
    metadata = {
        "name": f"Titan Genesis — {titan_name}",
        "symbol": "TITAN",
        "description": (
            f"Genesis identity of {titan_name}, a sovereign AI cognitive entity. "
            "Birth DNA, prime directives, and transition criteria permanently "
            "recorded on Arweave; recovery pointers anchor a wallet-only "
            "resurrection."
        ),
        "attributes": [
            {"trait_type": k,
             "value": str(v) if not isinstance(v, dict) else _json.dumps(v)}
            for k, v in nft_attrs.items()
        ],
        "birth_identity": serialize_for_arweave(),
    }
    if recovery:
        metadata["recovery"] = recovery
    if naming_ceremony:
        metadata["naming_ceremony"] = naming_ceremony
    return metadata
