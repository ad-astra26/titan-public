"""
titan_plugin/utils/arweave_store.py — Permanent storage on Arweave via Irys.

Uploads data permanently to Arweave using Irys Node 2 REST API.
Payment in SOL — no bridging, no AR tokens needed.

Uses httpx for REST calls — no Irys SDK dependency.

Architecture:
  upload_permanent(data, tags) → Arweave TX ID (permanent URL: https://arweave.net/{tx_id})
  fetch_permanent(tx_id) → data
  upload_file(filepath, tags) → Arweave TX ID

Irys Node 2 (Solana):
  - Endpoint: https://node2.irys.xyz
  - Fund with SOL, upload bytes, get Arweave TX ID
  - Data is permanent and immutable once uploaded

Cost: ~$0.005/MB (varies with Arweave network demand)
"""
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Irys Node 2 (Solana-native)
IRYS_NODE_URL = "https://node2.irys.xyz"
IRYS_DEVNET_URL = "https://devnet.irys.xyz"  # For devnet testing

# Arweave gateway for fetching
ARWEAVE_GATEWAY = "https://arweave.net"


class ArweaveStore:
    """Permanent storage on Arweave via Irys REST API."""

    def __init__(self, keypair_path: str = None, network: str = "devnet"):
        """
        Args:
            keypair_path: Path to Solana keypair JSON (for funding/signing)
            network: "devnet" or "mainnet" — determines Irys endpoint
        """
        self._keypair_path = keypair_path
        self._network = network
        self._node_url = IRYS_DEVNET_URL if network == "devnet" else IRYS_NODE_URL
        self._funded = False

        # Cache uploaded TX IDs for verification
        self._upload_cache: dict[str, str] = {}  # content_hash → tx_id

        logger.info("[ArweaveStore] Initialized (network=%s, node=%s)",
                    network, self._node_url)

    async def upload_json(
        self,
        data: dict,
        tags: dict = None,
        content_type: str = "application/json",
    ) -> Optional[str]:
        """Upload JSON data permanently to Arweave.

        Args:
            data: Dict to serialize as JSON
            tags: Optional metadata tags (key→value)
            content_type: MIME type

        Returns:
            Arweave TX ID on success, None on failure.
            Permanent URL: https://arweave.net/{tx_id}
        """
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        return await self._upload_bytes(payload, tags, content_type)

    async def upload_file(
        self,
        filepath: str,
        tags: dict = None,
        content_type: str = "application/octet-stream",
    ) -> Optional[str]:
        """Upload a file permanently to Arweave.

        Args:
            filepath: Local file path
            tags: Optional metadata tags
            content_type: MIME type

        Returns:
            Arweave TX ID on success, None on failure.
        """
        if not os.path.exists(filepath):
            logger.error("[ArweaveStore] File not found: %s", filepath)
            return None

        with open(filepath, "rb") as f:
            payload = f.read()

        file_tags = {"filename": os.path.basename(filepath)}
        if tags:
            file_tags.update(tags)

        return await self._upload_bytes(payload, file_tags, content_type)

    async def upload_file_bytes(
        self,
        payload: bytes,
        tags: dict = None,
        content_type: str = "application/octet-stream",
    ) -> Optional[str]:
        """Upload raw bytes permanently to Arweave (public API for TimeChainBackup)."""
        file_tags = {}
        if tags:
            file_tags.update(tags)
        return await self._upload_bytes(payload, file_tags, content_type)

    async def _upload_bytes(
        self,
        payload: bytes,
        tags: dict = None,
        content_type: str = "application/octet-stream",
    ) -> Optional[str]:
        """Upload raw bytes to Arweave via Irys.

        Irys upload flow:
        1. Sign the data with Solana keypair
        2. POST to /tx/solana with signed data
        3. Get back Arweave TX ID

        For devnet: uses free uploads (no funding needed for small payloads)
        For mainnet: requires funded Irys account (SOL)
        """
        import httpx

        content_hash = hashlib.sha256(payload).hexdigest()

        # Check cache — don't re-upload identical content
        if content_hash in self._upload_cache:
            cached_tx = self._upload_cache[content_hash]
            logger.info("[ArweaveStore] Cache hit for %s... → %s",
                        content_hash[:12], cached_tx[:16])
            return cached_tx

        all_tags = {
            "Content-Type": content_type,
            "App-Name": "Titan-Sovereign-AI",
            "App-Version": "4.0",
            "Unix-Time": str(int(time.time())),
        }
        if tags:
            all_tags.update(tags)

        try:
            # For devnet/testing: use the simple upload endpoint
            # For production: would need Irys SDK signing flow
            # This implementation provides the interface — actual Irys signing
            # requires the @irys/sdk or manual Ed25519 signing of the upload receipt

            if self._network == "devnet":
                # Devnet: store locally as fallback + log what would be uploaded
                return await self._devnet_upload(payload, content_hash, all_tags)
            else:
                return await self._mainnet_upload(payload, content_hash, all_tags)

        except Exception as e:
            logger.error("[ArweaveStore] Upload failed: %s", e)
            return None

    async def _devnet_upload(
        self, payload: bytes, content_hash: str, tags: dict
    ) -> str:
        """Devnet mode: store locally and return a deterministic pseudo-TX ID.

        In devnet, we don't actually upload to Arweave (costs real AR on mainnet Irys).
        Instead, store the payload locally and return a deterministic ID that can be
        used for testing the full flow.

        When switching to mainnet, _mainnet_upload() handles real Irys uploads.
        """
        # Store locally for devnet testing
        store_dir = Path("data/arweave_devnet")
        store_dir.mkdir(parents=True, exist_ok=True)

        # Deterministic TX ID from content hash (simulates Arweave TX)
        pseudo_tx_id = f"devnet_{content_hash[:43]}"

        # Save payload
        (store_dir / f"{pseudo_tx_id}.data").write_bytes(payload)

        # Save tags
        with open(store_dir / f"{pseudo_tx_id}.tags.json", "w") as f:
            json.dump(tags, f, indent=2)

        self._upload_cache[content_hash] = pseudo_tx_id
        logger.info("[ArweaveStore] DEVNET upload: %s (%d bytes) → %s",
                    tags.get("filename", "json"), len(payload), pseudo_tx_id)
        return pseudo_tx_id

    async def _mainnet_upload(
        self, payload: bytes, content_hash: str, tags: dict
    ) -> Optional[str]:
        """Mainnet mode: upload to Arweave via Irys Node 2 using @irys/sdk.

        Uses scripts/irys_upload.js (Node.js) subprocess for proper Ed25519 signing.
        Requires: npm install -g @irys/sdk, funded Irys account.
        """
        import asyncio
        import tempfile

        if not self._keypair_path:
            logger.error("[ArweaveStore] No keypair path — cannot sign Irys uploads")
            return None

        # Write payload to temp file for the Node.js helper
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            tmp.write(payload)
            tmp_path = tmp.name

        try:
            content_type = tags.get("Content-Type", "application/octet-stream")
            # Remove Content-Type from tags to avoid duplication
            extra_tags = {k: v for k, v in tags.items() if k != "Content-Type"}
            tags_json = json.dumps(extra_tags) if extra_tags else ""

            rpc_url = self._get_rpc_url()
            cmd = [
                "node", "scripts/irys_upload.js", "upload",
                tmp_path, self._keypair_path, rpc_url, content_type,
            ]
            if tags_json:
                cmd.append(tags_json)

            env = os.environ.copy()
            env["NODE_PATH"] = "/usr/lib/node_modules"

            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

            result = json.loads(stdout.decode())
            if result.get("status") == "ok":
                tx_id = result["tx_id"]
                self._upload_cache[content_hash] = tx_id
                logger.info("[ArweaveStore] Uploaded to Arweave: %s (%d bytes) → %s",
                            tx_id[:16], len(payload), result["url"])
                return tx_id

            logger.warning("[ArweaveStore] Irys upload failed: %s", result.get("message", "unknown"))
            return None

        except Exception as e:
            logger.error("[ArweaveStore] Mainnet upload error: %s", e)
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def upload_file_mainnet(self, filepath: str, tags: dict = None,
                                   content_type: str = "application/octet-stream") -> Optional[str]:
        """Upload a file directly to Arweave (mainnet only, uses Irys SDK).

        More efficient than upload_file() for large files — streams directly
        without loading into memory.
        """
        import asyncio

        if not self._keypair_path:
            logger.error("[ArweaveStore] No keypair path")
            return None

        extra_tags = tags or {}
        extra_tags["filename"] = os.path.basename(filepath)
        tags_json = json.dumps(extra_tags)
        rpc_url = self._get_rpc_url()

        cmd = [
            "node", "scripts/irys_upload.js", "upload",
            filepath, self._keypair_path, rpc_url, content_type, tags_json,
        ]
        env = os.environ.copy()
        env["NODE_PATH"] = "/usr/lib/node_modules"

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            result = json.loads(stdout.decode())
            if result.get("status") == "ok":
                logger.info("[ArweaveStore] File uploaded: %s → %s", filepath, result["url"])
                return result["tx_id"]
            logger.warning("[ArweaveStore] File upload failed: %s", result.get("message"))
            return None
        except Exception as e:
            logger.error("[ArweaveStore] File upload error: %s", e)
            return None

    def _get_rpc_url(self) -> str:
        """Get the RPC URL from config or default."""
        try:
            import tomllib
            with open("titan_plugin/config.toml", "rb") as f:
                cfg = tomllib.load(f)
            return cfg.get("network", {}).get("premium_rpc_url", "https://api.mainnet-beta.solana.com")
        except Exception:
            return "https://api.mainnet-beta.solana.com"

    async def fetch(self, tx_id: str) -> Optional[bytes]:
        """Fetch data from Arweave by TX ID.

        Works for both devnet (local) and mainnet (Arweave gateway).
        """
        import httpx

        # Devnet: fetch from local store
        if tx_id.startswith("devnet_"):
            local_path = Path("data/arweave_devnet") / f"{tx_id}.data"
            if local_path.exists():
                return local_path.read_bytes()
            return None

        # Mainnet: fetch from Arweave gateway
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(f"{ARWEAVE_GATEWAY}/{tx_id}")
                if resp.status_code == 200:
                    return resp.content
                logger.warning("[ArweaveStore] Fetch %s returned %d", tx_id[:16], resp.status_code)
        except Exception as e:
            logger.error("[ArweaveStore] Fetch error: %s", e)
        return None

    async def fetch_json(self, tx_id: str) -> Optional[dict]:
        """Fetch and parse JSON from Arweave."""
        data = await self.fetch(tx_id)
        if data:
            return json.loads(data.decode("utf-8"))
        return None

    def get_permanent_url(self, tx_id: str) -> str:
        """Get the permanent URL for an Arweave TX ID."""
        if tx_id.startswith("devnet_"):
            return f"file://data/arweave_devnet/{tx_id}.data"
        return f"{ARWEAVE_GATEWAY}/{tx_id}"
