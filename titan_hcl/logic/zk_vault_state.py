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


# ───────────────────────────────────────────────────────────────────────────
# ZK-compressed audit trail — Light Protocol `append_epoch_snapshot`
# (PLAN_zk_vault_proof_completion.md / SPEC §B4 "the append-only audit trail").
# Fired in-process on the daily Arweave backup event (NOT per-meditation).
# ───────────────────────────────────────────────────────────────────────────

def read_timechain_block_count(data_dir: str = "data") -> Optional[int]:
    """Real, lock-free `memory_count` source for the epoch snapshot — the count
    of TimeChain blocks (the Titan's recorded-experience ledger). Read-only over
    the WAL'd `index.db`, so it never blocks the live writer. Returns None on any
    failure (caller then SKIPS the snapshot rather than writing a 0 stub —
    INV-NO-STUBS)."""
    import sqlite3
    db_path = os.path.join(data_dir, "timechain", "index.db")
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            row = conn.execute("SELECT COUNT(*) FROM block_index").fetchone()
            return int(row[0]) if row else None
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        logger.warning("[ZKVaultState] timechain block count read failed: %s", e)
        return None


def zk_audit_state_path(titan_id: str) -> str:
    """Per-Titan ZK-compressed-audit status file — read by the dashboard gate."""
    return f"data/zk_compressed_audit_state_{titan_id}.json"


def write_zk_audit_state(titan_id: str, *, enabled: bool, last_tx: Optional[str] = None,
                         last_state_root: Optional[str] = None,
                         last_verified: Optional[bool] = None,
                         last_memory_count: Optional[int] = None,
                         last_error: Optional[str] = None) -> None:
    """Persist the observable gate state (switch + last append_epoch_snapshot tx
    + parse-back verify). Atomic tmp+rename. Non-critical."""
    try:
        p = zk_audit_state_path(titan_id)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        payload = {
            "titan_id": titan_id,
            "continuous_enabled": bool(enabled),
            "last_tx": last_tx,
            "last_state_root": last_state_root,
            "last_verified": last_verified,
            "last_memory_count": last_memory_count,
            "last_error": last_error,
            "ts": int(time.time()),
        }
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, p)
    except Exception as e:  # noqa: BLE001
        logger.warning("[ZKVaultState] zk audit state write failed: %s", e)


def read_zk_audit_state(titan_id: str) -> Optional[dict]:
    """Read the ZK-compressed-audit status file (dashboard gate). None if absent."""
    try:
        p = zk_audit_state_path(titan_id)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


async def verify_epoch_snapshot(photon, authority_b58: str,
                                expected_state_root_hex: str) -> dict:
    """Parse-back proof (mirrors `backup_zk_commit`'s commit_state verify): fetch
    the Titan's compressed accounts via Photon, decode the most-recent
    CompressedEpochSnapshot, and assert its `state_root` matches what we wrote.
    Robust to a possible 8-byte LightDiscriminator prefix (tries both offsets).
    Best-effort: the on-chain tx is the primary proof; this is the audit check."""
    import base64
    out = {"ok": False, "found": False, "state_root": None}
    if photon is None:
        return out
    try:
        from titan_hcl.utils.solana_client import decode_compressed_epoch_snapshot
        accounts = await photon.fetch_compressed_accounts_by_owner(authority_b58)
        expected = (expected_state_root_hex or "").lower()
        best = None
        for acct in accounts or []:
            raw = acct.get("data", {})
            if isinstance(raw, dict):
                data_bytes = base64.b64decode(raw.get("data", "") or "")
            elif isinstance(raw, str):
                data_bytes = base64.b64decode(raw)
            else:
                continue
            dec = (decode_compressed_epoch_snapshot(data_bytes)
                   or decode_compressed_epoch_snapshot(data_bytes[8:]))
            if dec and (best is None or dec["timestamp"] >= best["timestamp"]):
                best = dec
        if best:
            out["found"] = True
            out["state_root"] = best["state_root"]
            out["ok"] = (best["state_root"].lower() == expected)
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("[ZKVaultState] verify_epoch_snapshot failed: %s", e)
        return out


# ───────────────────────────────────────────────────────────────────────────
# E2 — SovereignState: the running canonical SNARK-per-write account.
# RFP_zk_vault_snark_per_write.md. One addressed compressed account per Titan;
# genesis = create (non-inclusion proof), every subsequent write = update
# (new_mut, inclusion proof). Fires alongside E1's append-trail (option A) under
# the SAME `zk_compressed_audit_enabled` switch. In-process, no new keypair.
# ───────────────────────────────────────────────────────────────────────────

def _b64_or_list_to_bytes(v) -> Optional[bytes]:
    """Decode a Photon proof/byte field that may arrive as base64 str, hex str,
    or list[int]. 🚩 [I]-VERIFY-ON-DEVNET: confirm Photon's actual encoding for
    compressedProof a/b/c against the live indexer before trusting on mainnet."""
    import base64
    try:
        if v is None:
            return None
        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
        if isinstance(v, list):
            return bytes(v)
        if isinstance(v, str):
            try:
                return base64.b64decode(v)
            except Exception:  # noqa: BLE001
                return bytes.fromhex(v[2:] if v.startswith("0x") else v)
    except Exception:  # noqa: BLE001
        return None
    return None


def _photon_proof_128(value: dict) -> Optional[bytes]:
    """Extract the 128-byte Groth16 proof (a:32 ‖ b:64 ‖ c:32) from a Photon
    getValidityProof `value`. 🚩 [I]-VERIFY-ON-DEVNET: field spelling."""
    if not isinstance(value, dict):
        return None
    cp = value.get("compressedProof") or value.get("compressed_proof") or value.get("proof")
    if not isinstance(cp, dict):
        return None
    a = _b64_or_list_to_bytes(cp.get("a"))
    b = _b64_or_list_to_bytes(cp.get("b"))
    c = _b64_or_list_to_bytes(cp.get("c"))
    if not (a and b and c) or len(a) != 32 or len(b) != 64 or len(c) != 32:
        return None
    return a + b + c


def _first(value: dict, *keys, default=None):
    """Return the first present key from a Photon `value` (tolerant of the
    array-vs-scalar / camelCase spelling drift across Photon versions)."""
    for k in keys:
        if isinstance(value, dict) and value.get(k) is not None:
            v = value[k]
            return v[0] if isinstance(v, list) and v else v
    return default


async def verify_sovereign_state(photon, address_bytes: bytes,
                                 expected_state_root_hex: str) -> dict:
    """Parse-back proof for the running account: fetch by its canonical address,
    decode the SovereignState, assert `state_root` == what we wrote (INV-ZKW-4).
    Best-effort — the on-chain tx + its proof are the primary proof."""
    out = {"ok": False, "found": False, "state_root": None}
    if photon is None or not address_bytes:
        return out
    try:
        from based58 import b58encode
        from titan_hcl.utils.solana_client import decode_sovereign_state
        import base64
        address_b58 = b58encode(address_bytes).decode()
        acct = await photon.get_compressed_account(address_b58)
        if not acct:
            return out
        raw = acct.get("data", {})
        if isinstance(raw, dict):
            data_bytes = base64.b64decode(raw.get("data", "") or "")
        elif isinstance(raw, str):
            data_bytes = base64.b64decode(raw)
        else:
            return out
        dec = (decode_sovereign_state(data_bytes)
               or decode_sovereign_state(data_bytes[8:]))
        if dec:
            out["found"] = True
            out["state_root"] = dec["state_root"]
            out["ok"] = (dec["state_root"].lower() == (expected_state_root_hex or "").lower())
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("[ZKVaultState] verify_sovereign_state failed: %s", e)
        return out


def write_sovereign_state_status(titan_id: str, *, enabled: bool,
                                 last_update_tx: Optional[str] = None,
                                 last_proof_verified: Optional[bool] = None,
                                 last_op: Optional[str] = None,
                                 last_state_root: Optional[str] = None,
                                 last_error: Optional[str] = None) -> None:
    """Persist the E2 observable: `snark_per_write` gate + last create/update tx.
    Separate file from E1's zk_audit_state (decode-distinct accounts)."""
    try:
        p = f"data/zk_sovereign_state_status_{titan_id}.json"
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        payload = {
            "titan_id": titan_id,
            "snark_per_write": bool(enabled),
            "last_update_tx": last_update_tx,
            "last_proof_verified": last_proof_verified,
            "last_op": last_op,  # "create" | "update"
            "last_state_root": last_state_root,
            "last_error": last_error,
            "ts": int(time.time()),
        }
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, p)
    except Exception as e:  # noqa: BLE001
        logger.warning("[ZKVaultState] sovereign state status write failed: %s", e)


def read_sovereign_state_status(titan_id: str) -> Optional[dict]:
    """Read the E2 snark_per_write status file (dashboard gate). None if absent."""
    try:
        p = f"data/zk_sovereign_state_status_{titan_id}.json"
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


async def emit_sovereign_state(network, *, state_root_hex: str, sovereignty_bp: int,
                               arweave_url: str, archive_hash: str, titan_id: str,
                               memory_count: int, photon=None,
                               program_id_str: Optional[str] = None) -> dict:
    """Write the running canonical SovereignState (E2 SNARK-per-write) for a
    backup event — in-process via the HybridNetworkClient.

    Selector (RFP §6.5 3a): derive the canonical address → Photon lookup. Absent
    ⇒ GENESIS `create_sovereign_state` (non-inclusion proof). Present ⇒
    `update_sovereign_state` (new_mut, inclusion proof) consuming the prior leaf.

    Returns {tx, op, verified, state_root, error}. Non-critical — never raises;
    the backup chain + E1 append-trail committed independently."""
    result = {"tx": None, "op": None, "verified": None,
              "state_root": state_root_hex, "error": None}
    if photon is None:
        # The SNARK path REQUIRES Photon (proofs are fetched from it).
        result["error"] = "no_photon"
        return result
    if network is None or getattr(network, "pubkey", None) is None:
        result["error"] = "no_network"
        return result
    try:
        from based58 import b58encode
        from titan_hcl.utils.solana_client import (
            is_available, derive_light_v1_address,
            build_create_sovereign_state_instruction,
            build_update_sovereign_state_instruction,
            decode_sovereign_state, LIGHT_V1_ADDRESS_TREE,
        )
        if not is_available():
            result["error"] = "sdk_unavailable"
            return result

        vault_program_id = (
            program_id_str
            or getattr(network, "_vault_program_id", None)
            or (getattr(network, "_config", {}) or {}).get("network", {}).get("vault_program_id", "")
        )
        if not vault_program_id:
            result["error"] = "no_vault_program_id"
            return result

        try:
            state_root = bytes.fromhex(state_root_hex)
        except (ValueError, TypeError):
            result["error"] = "bad_state_root_hex"
            return result
        if len(state_root) != 32:
            result["error"] = f"state_root_not_32B ({len(state_root)})"
            return result

        shadow_src = arweave_url or archive_hash or ""
        shadow_hash = hashlib.sha256(shadow_src.encode("utf-8")).digest()

        # Canonical address (keccak v1 derive_address) — the selector key.
        address = derive_light_v1_address(network.pubkey, program_id_str=vault_program_id)
        if address is None:
            result["error"] = "address_derive_failed"
            return result
        address_b58 = b58encode(address).decode()

        existing = await photon.get_compressed_account(address_b58)

        if not existing:
            # ── GENESIS: addressed create with a non-inclusion proof ──
            result["op"] = "create"
            proof_val = await photon.get_new_address_proof(address_b58, LIGHT_V1_ADDRESS_TREE)
            proof_128 = _photon_proof_128(proof_val) if proof_val else None
            if proof_128 is None:
                result["error"] = "noninclusion_proof_unavailable"
                write_sovereign_state_status(titan_id, enabled=True, last_op="create",
                                             last_error="noninclusion_proof_unavailable")
                return result
            addr_root_index = int(_first(proof_val, "addressRootIndex", "address_root_index",
                                         "rootIndex", "root_index", default=0) or 0)
            ix = build_create_sovereign_state_instruction(
                network.pubkey, proof_bytes=proof_128,
                address_merkle_tree_index=1, address_queue_index=2,
                address_root_index=addr_root_index,
                state_root=state_root,
                memory_count=int(memory_count), sovereignty_score=int(sovereignty_bp),
                shadow_url_hash=shadow_hash, program_id_str=vault_program_id,
            )
        else:
            # ── UPDATE: new_mut consume+recreate with an inclusion proof ──
            result["op"] = "update"
            import base64
            raw = existing.get("data", {})
            data_bytes = (base64.b64decode(raw.get("data", "") or "")
                          if isinstance(raw, dict) else base64.b64decode(raw or ""))
            old_state = (decode_sovereign_state(data_bytes)
                         or decode_sovereign_state(data_bytes[8:]))
            cur_hash = existing.get("hash") or existing.get("hashB58")
            if old_state is None or not cur_hash:
                result["error"] = "current_account_undecodable"
                write_sovereign_state_status(titan_id, enabled=True, last_op="update",
                                             last_error="current_account_undecodable")
                return result
            proof_val = await photon.get_validity_proof([cur_hash])
            proof_128 = _photon_proof_128(proof_val) if proof_val else None
            if proof_128 is None:
                result["error"] = "inclusion_proof_unavailable"
                write_sovereign_state_status(titan_id, enabled=True, last_op="update",
                                             last_error="inclusion_proof_unavailable")
                return result
            root_index = int(_first(proof_val, "rootIndex", "root_index", default=0) or 0)
            leaf_index = int(existing.get("leafIndex", existing.get("leaf_index", 0)) or 0)
            ix = build_update_sovereign_state_instruction(
                network.pubkey, proof_bytes=proof_128, old_state=old_state,
                address=address, leaf_index=leaf_index, root_index=root_index,
                merkle_tree_index=0, queue_index=1, output_state_tree_index=0,
                state_root=state_root,
                memory_count=int(memory_count), sovereignty_score=int(sovereignty_bp),
                shadow_url_hash=shadow_hash, program_id_str=vault_program_id,
            )

        if ix is None:
            result["error"] = f"ix_build_failed ({result['op']})"
            write_sovereign_state_status(titan_id, enabled=True, last_op=result["op"],
                                         last_error=result["error"])
            return result

        sig = await network.send_sovereign_transaction([ix], priority="HIGH")
        if not sig:
            result["error"] = "tx_send_failed"
            write_sovereign_state_status(titan_id, enabled=True, last_op=result["op"],
                                         last_error="tx_send_failed")
            return result
        result["tx"] = sig

        v = await verify_sovereign_state(photon, address, state_root_hex)
        result["verified"] = v.get("ok")

        write_sovereign_state_status(
            titan_id, enabled=True, last_update_tx=sig, last_op=result["op"],
            last_proof_verified=result["verified"], last_state_root=state_root_hex,
        )
        logger.info(
            "[ZKVaultState] SovereignState %s committed: tx=%s root=%s verified=%s",
            result["op"], sig[:16] if len(sig) > 16 else sig,
            state_root_hex[:16], result["verified"],
        )
        return result
    except Exception as e:  # noqa: BLE001
        result["error"] = f"exception: {e}"
        logger.warning("[ZKVaultState] emit_sovereign_state failed (non-critical): %s",
                       e, exc_info=True)
        write_sovereign_state_status(titan_id, enabled=True, last_op=result.get("op"),
                                     last_error=str(e))
        return result


async def emit_epoch_snapshot(network, *, state_root_hex: str, sovereignty_bp: int,
                              archive_hash: str, arweave_url: str, titan_id: str,
                              memory_count: int, photon=None,
                              program_id_str: Optional[str] = None) -> dict:
    """Write the ZK-compressed epoch snapshot (Light Protocol `append_epoch_snapshot`,
    SPEC §B4 append-only audit trail) for a backup event — in-process via the
    HybridNetworkClient (the SAME client the backup module holds; no new keypair).

    Output-only addressless create ⇒ NO validity proof (the chain rejects one).
    `state_root_hex` = the backup event_merkle_root (64-hex). Returns
    {tx, verified, state_root, error}. Non-critical — never raises; the backup
    chain already committed independently."""
    result = {"tx": None, "verified": None, "state_root": state_root_hex, "error": None}
    if network is None or getattr(network, "pubkey", None) is None:
        result["error"] = "no_network"
        return result
    try:
        from titan_hcl.utils.solana_client import (
            build_append_epoch_snapshot_instruction, is_available,
        )
        if not is_available():
            result["error"] = "sdk_unavailable"
            return result

        vault_program_id = (
            program_id_str
            or getattr(network, "_vault_program_id", None)
            or (getattr(network, "_config", {}) or {}).get("network", {}).get("vault_program_id", "")
        )
        if not vault_program_id:
            result["error"] = "no_vault_program_id"
            return result

        try:
            state_root = bytes.fromhex(state_root_hex)
        except (ValueError, TypeError):
            result["error"] = "bad_state_root_hex"
            return result
        if len(state_root) != 32:
            result["error"] = f"state_root_not_32B ({len(state_root)})"
            return result

        # shadow_url_hash = sha256 of the backup's Arweave URL (its data-plane
        # location) — the on-chain pointer to where this snapshot's data lives.
        shadow_src = arweave_url or archive_hash or ""
        shadow_hash = hashlib.sha256(shadow_src.encode("utf-8")).digest()

        ix = build_append_epoch_snapshot_instruction(
            authority_pubkey=network.pubkey,
            state_root=state_root,
            memory_count=int(memory_count),
            sovereignty_score=int(sovereignty_bp),
            shadow_url_hash=shadow_hash,
            program_id_str=vault_program_id,
        )
        if ix is None:
            result["error"] = "ix_build_failed"
            return result

        sig = await network.send_sovereign_transaction([ix], priority="MEDIUM")
        if not sig:
            result["error"] = "tx_send_failed"
            write_zk_audit_state(titan_id, enabled=True, last_error="tx_send_failed")
            return result
        result["tx"] = sig

        # Parse-back verify (best-effort; the tx is the primary proof).
        if photon is not None:
            v = await verify_epoch_snapshot(photon, str(network.pubkey), state_root_hex)
            result["verified"] = v.get("ok")

        persist_vault_snapshot(
            titan_id, tx_sig=sig, archive_hash=archive_hash,
            memory_count=int(memory_count), sovereignty_bp=int(sovereignty_bp),
            arweave_url=arweave_url or "",
        )
        write_zk_audit_state(
            titan_id, enabled=True, last_tx=sig, last_state_root=state_root_hex,
            last_verified=result["verified"], last_memory_count=int(memory_count),
        )
        logger.info(
            "[ZKVaultState] append_epoch_snapshot committed: tx=%s root=%s "
            "memory_count=%d verified=%s",
            sig[:16] if len(sig) > 16 else sig, state_root_hex[:16],
            int(memory_count), result["verified"],
        )
        return result
    except Exception as e:  # noqa: BLE001
        result["error"] = f"exception: {e}"
        logger.warning(
            "[ZKVaultState] emit_epoch_snapshot failed (non-critical): %s",
            e, exc_info=True,
        )
        write_zk_audit_state(titan_id, enabled=True, last_error=str(e))
        return result
