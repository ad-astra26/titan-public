"""
utils/photon_client.py
Lightweight httpx-based client for the Helius-hosted Photon JSON-RPC indexer.

Provides compressed account queries and validity proof fetching
for the Titan's ZK-compressed memory receipts. Zero new dependencies
— uses httpx (already installed) exclusively.

All methods degrade gracefully: network errors return None/empty,
never raise. Callers check return values and fall back to Memo mode.
"""
import base64
import logging
import struct
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

# Compressed account discriminators (from Anchor IDL build)
# SHA256("global:compress_memory_batch")[:8] — actually from LightDiscriminator derive
# These match the LightDiscriminator output for our structs.
_MEMORY_BATCH_DISCRIMINATOR = bytes([105, 76, 210, 140, 189, 129, 57, 135])
_EPOCH_SNAPSHOT_DISCRIMINATOR = bytes([213, 217, 65, 120, 202, 70, 5, 131])


class PhotonClient:
    """
    Raw httpx interface to the Helius-hosted Photon indexer.
    Provides compressed account queries and validity proof fetching
    for the Titan's ZK-compressed memory receipts.
    """

    def __init__(self, rpc_url: str):
        """
        Args:
            rpc_url: The Helius RPC endpoint (same URL serves standard + Photon).
        """
        self.rpc_url = rpc_url.rstrip("/")

    async def _rpc_call(self, method: str, params: dict) -> Optional[dict]:
        """
        Execute a single Photon JSON-RPC 2.0 call.

        Args:
            method: RPC method name (e.g. "getCompressedAccount").
            params: Method parameters dict.

        Returns:
            Parsed JSON response dict, or None on any error.
        """
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    self.rpc_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": method,
                        "params": params,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                if "error" in data:
                    logger.debug(
                        "[PhotonClient] RPC error for %s: %s",
                        method,
                        data["error"],
                    )
                    return None

                return data.get("result")

        except httpx.TimeoutException:
            logger.warning("[PhotonClient] Timeout calling %s", method)
            return None
        except httpx.HTTPStatusError as e:
            logger.warning("[PhotonClient] HTTP %d for %s", e.response.status_code, method)
            return None
        except Exception as e:
            logger.warning("[PhotonClient] Error calling %s: %s", method, e)
            return None

    async def health_check(self) -> bool:
        """
        Call getIndexerHealth — returns True if Photon is responsive.
        The response value is the string "ok" on success.
        """
        result = await self._rpc_call("getIndexerHealth", {})
        return result == "ok"

    async def fetch_compressed_account(self, address_b58: str) -> Optional[dict]:
        """
        Fetch a single compressed account by its base58 address.

        Args:
            address_b58: Base58-encoded compressed account address.

        Returns:
            Dict with {data, hash, lamports, address, tree, leaf_index, slot}
            or None if not found / error.
        """
        result = await self._rpc_call(
            "getCompressedAccount",
            {"address": address_b58},
        )
        if result and "value" in result:
            return result["value"]
        return None

    async def fetch_compressed_accounts_by_owner(self, owner: str) -> List[dict]:
        """
        Fetch ALL compressed accounts owned by the given pubkey.
        Used for timeline reconstruction and resurrection verification.

        Args:
            owner: Base58-encoded owner public key.

        Returns:
            List of compressed account dicts, empty on error.
        """
        result = await self._rpc_call(
            "getCompressedAccountsByOwner",
            {"owner": owner},
        )
        if result and "value" in result:
            items = result["value"].get("items", [])
            return items if isinstance(items, list) else []
        return []

    async def get_validity_proof(self, hashes: List[str]) -> Optional[dict]:
        """
        Fetch Groth16 validity proof for the given compressed account hashes.

        Args:
            hashes: List of base58-encoded compressed account hashes.

        Returns:
            Dict with proof bytes + Merkle context, or None if unavailable.
        """
        result = await self._rpc_call(
            "getValidityProof",
            {"hashes": hashes},
        )
        if result and "value" in result:
            return result["value"]
        return None

    async def prove_memory_existence(self, memory_hash_hex: str) -> Optional[dict]:
        """
        Proof-of-Knowledge primitive.
        Given a memory hash (hex), prove it exists in the ZK state tree.

        This fetches compressed accounts by the Titan's owner pubkey and
        searches for one whose batch_root contains the given memory hash.

        Args:
            memory_hash_hex: Hex-encoded memory hash to prove.

        Returns:
            Dict with {exists: bool, compressed_account: dict, slot: int}
            or None on error.
        """
        # This is a higher-level query: the Photon indexer doesn't directly
        # search by batch_root content. The caller would need to fetch all
        # accounts and search client-side. This method provides the stub
        # for future Photon index extensions.
        logger.debug(
            "[PhotonClient] prove_memory_existence is a client-side search "
            "over compressed accounts — use fetch_compressed_accounts_by_owner."
        )
        return None

    async def fetch_memory_timeline(self, authority: str) -> List[dict]:
        """
        Fetch ALL CompressedMemoryBatch accounts for the Titan,
        ordered by epoch_id. This is the verifiable "mind timeline".

        Args:
            authority: Base58-encoded Titan wallet pubkey.

        Returns:
            List of decoded CompressedMemoryBatch dicts, ordered by epoch_id.
        """
        accounts = await self.fetch_compressed_accounts_by_owner(authority)
        if not accounts:
            return []

        from titan_plugin.utils.solana_client import decode_compressed_memory_batch

        timeline = []
        for acct in accounts:
            raw_data = acct.get("data", {})
            # Photon returns data as base64 in the "data" field
            if isinstance(raw_data, dict):
                data_bytes = base64.b64decode(raw_data.get("data", ""))
            elif isinstance(raw_data, str):
                data_bytes = base64.b64decode(raw_data)
            else:
                continue

            decoded = decode_compressed_memory_batch(data_bytes)
            if decoded:
                timeline.append(decoded)

        # Sort by epoch_id ascending
        timeline.sort(key=lambda x: x.get("epoch_id", 0))
        return timeline
