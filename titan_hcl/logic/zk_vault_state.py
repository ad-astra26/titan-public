"""ZK-Vault on-chain state — the dedicated owner of the vault shadow-hash +
the per-Titan ZK-Vault snapshot history (RFP_backup_redesign_spine Phase E /
Q-BRS-4).

Evicted from the `RebirthBackup` god-class (INV-BRS-10: the vault shadow-hash +
snapshot persistence are NOT a backup concern — they are vault-state writes that
happen to be triggered around an anchor). The on-chain write goes through the
in-process `HybridNetworkClient` (the SAME client the backup module already
holds); this module owns NO keypair of its own.

These were dormant on `backup.py` (their caller — the legacy `on_meditation_complete`
ZK-epoch tail — was deleted in Phase B-1); they live here so the capability has a
proper home when re-wired (e.g. a vault-state worker / the anchor step).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


async def update_vault_shadow_hash(network, archive_hash: str) -> Optional[str]:
    """Store the backup verification hash on-chain in the vault PDA
    (`shadow_url_hash`), making it queryable via Photon alongside the TimeChain
    merkle root. `network` is the in-process HybridNetworkClient. Returns the tx
    signature on success, else None. Non-critical — never raises."""
    if network is None or getattr(network, "pubkey", None) is None:
        return None
    try:
        from titan_hcl.utils.solana_client import (
            build_vault_update_shadow_instruction, is_available,
        )
        if not is_available():
            return None

        vault_program_id = getattr(network, "_vault_program_id", None)
        if not vault_program_id:
            cfg = getattr(network, "_config", {})
            vault_program_id = cfg.get("network", {}).get("vault_program_id", "")
        if not vault_program_id:
            return None

        hash_bytes = hashlib.sha256(archive_hash.encode("utf-8")).digest()
        ix = build_vault_update_shadow_instruction(
            network.pubkey, hash_bytes, vault_program_id,
        )
        if ix:
            sig = await network.send_sovereign_transaction([ix], priority="LOW")
            if sig:
                logger.info("[ZKVaultState] Vault shadow hash updated: %s (tx=%s)",
                            archive_hash[:12], sig[:16] if len(sig) > 16 else sig)
                return sig
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[ZKVaultState] Vault shadow hash update failed (non-critical): %s", e)
    return None


def vault_snapshots_path(titan_id: str) -> str:
    """Per-Titan ZK Vault snapshot history file (rFP_x_voice_enrichment §4.3.1)."""
    return f"data/zk_vault_snapshots_{titan_id}.json"


def persist_vault_snapshot(titan_id: str, *, tx_sig: str, archive_hash: str,
                           memory_count: int, sovereignty_bp: int,
                           arweave_url: str = "") -> None:
    """Append a successful ZK Vault snapshot to the per-Titan history file.

    Atomic write via tmp+rename. Bounded to the last 200 entries (~6 months at
    one meditation per ~daily). Failures are non-critical — the on-chain TX
    already succeeded; this file is purely a queryable mirror (PROOF_DAY reads
    the most-recent vault commit to render the 'Seal' URL)."""
    try:
        p = vault_snapshots_path(titan_id)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        existing: list = []
        if os.path.exists(p):
            try:
                with open(p) as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing = loaded.get("snapshots", []) or []
                elif isinstance(loaded, list):
                    existing = loaded
            except Exception:  # noqa: BLE001
                existing = []
        existing.append({
            "tx_sig": tx_sig,
            "archive_hash": archive_hash,
            "memory_count": int(memory_count),
            "sovereignty_bp": int(sovereignty_bp),
            "arweave_url": arweave_url,
            "ts": int(time.time()),
        })
        existing = existing[-200:]
        payload = {"version": 1, "titan_id": titan_id, "snapshots": existing}
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, p)
    except Exception as e:  # noqa: BLE001
        logger.warning("[ZKVaultState] vault snapshot persist failed: %s", e)
