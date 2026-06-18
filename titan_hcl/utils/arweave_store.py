"""
titan_hcl/utils/arweave_store.py — Permanent storage on Arweave via Irys.

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
from titan_hcl.params import get_params

logger = logging.getLogger(__name__)

# Irys Node 2 (Solana-native)
IRYS_NODE_URL = "https://node2.irys.xyz"
IRYS_DEVNET_URL = "https://devnet.irys.xyz"  # For devnet testing

# Arweave gateway for fetching. Primary + fallbacks: a sovereign restore must
# retrieve every component tarball, and individual public gateways are flaky
# (transient 5xx / Cloudflare 52x / rate limits), so `fetch` tries them in order
# with retry-with-backoff before giving up.
ARWEAVE_GATEWAY = "https://arweave.net"
# Gateway order matters: Titans UPLOAD their backup tarballs via Irys
# (scripts/irys_upload.js), so the Irys gateway is the canonical, most-reliable
# retrieval host (it serves the bundle the moment it's posted; arweave.net can be
# minutes-to-hours behind finalizing the bundle to base layer, and intermittently
# 5xx/Cloudflare-52x's — e.g. 572 on a T1 restore 2026-06-02). Arweave is immutable,
# so the DATA is never gone; a fetch "miss" is always a flaky-gateway problem —
# hence Irys first, then base-layer gateways as fallbacks.
_ARWEAVE_GATEWAYS = [
    "https://gateway.irys.xyz",
    ARWEAVE_GATEWAY,
    "https://permagate.io",
]


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

        Phase 5 chunk 5E: prefer the persistent `irys_upload_daemon.js`
        path (single long-lived Node process per keypair, ~2 s SDK
        startup paid once); fall through to one-shot
        `scripts/irys_upload.js` if the daemon is unavailable or fails.
        Requires: npm install -g @irys/sdk, funded Irys account.
        """
        import asyncio
        import tempfile

        if not self._keypair_path:
            logger.error("[ArweaveStore] No keypair path — cannot sign Irys uploads")
            return None

        # Write payload to temp file for both daemon + legacy helper paths
        # (Irys.uploadFile signs the file by path, never by bytes).
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            tmp.write(payload)
            tmp_path = tmp.name

        try:
            content_type = tags.get("Content-Type", "application/octet-stream")
            extra_tags = {k: v for k, v in tags.items() if k != "Content-Type"}
            rpc_url = self._get_rpc_url()

            tx_id = await self._daemon_upload_file(
                tmp_path, content_type, extra_tags, rpc_url,
            )
            if tx_id:
                self._upload_cache[content_hash] = tx_id
                logger.info(
                    "[ArweaveStore] Uploaded to Arweave: %s (%d bytes) "
                    "via persistent daemon",
                    tx_id[:16], len(payload),
                )
                return tx_id

            # Fall-through: one-shot legacy path (matches pre-5E behaviour)
            tags_json = json.dumps(extra_tags) if extra_tags else ""
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

    async def _daemon_upload_file(
        self, file_path: str, content_type: str, tags: dict, rpc_url: str,
    ) -> Optional[str]:
        """Phase 5 chunk 5E — try the persistent Irys daemon for one upload.

        Returns the Arweave tx_id on success; None on any daemon error so
        the caller falls through to the legacy one-shot path. We never let
        a daemon failure surface as an unhandled exception in the cascade.
        """
        try:
            from titan_hcl.utils.irys_persistent_client import (
                IrysDaemonError, get_daemon,
            )
        except ImportError:
            return None
        try:
            daemon = await get_daemon(self._keypair_path, rpc_url)
            result = await daemon.upload_file(
                file_path, content_type=content_type, tags=tags,
            )
            return result.tx_id
        except IrysDaemonError as e:
            logger.warning(
                "[ArweaveStore] persistent Irys daemon unavailable (%s) — "
                "falling back to one-shot helper", e,
            )
            return None
        except Exception as e:  # noqa: BLE001 — daemon errors must never crash cascade
            logger.warning(
                "[ArweaveStore] persistent Irys daemon raised unexpectedly "
                "(%s) — falling back to one-shot helper", e,
            )
            return None

    async def upload_file_mainnet(self, filepath: str, tags: dict = None,
                                   content_type: str = "application/octet-stream") -> Optional[str]:
        """Upload a file directly to Arweave (mainnet only, uses Irys SDK).

        More efficient than upload_file() for large files — streams directly
        without loading into memory. Phase 5 chunk 5E prefers the persistent
        daemon path; falls through to one-shot helper if unavailable.
        """
        import asyncio

        if not self._keypair_path:
            logger.error("[ArweaveStore] No keypair path")
            return None

        extra_tags = tags or {}
        extra_tags["filename"] = os.path.basename(filepath)
        rpc_url = self._get_rpc_url()

        daemon_tx = await self._daemon_upload_file(
            filepath, content_type, extra_tags, rpc_url,
        )
        if daemon_tx:
            logger.info(
                "[ArweaveStore] File uploaded via persistent daemon: %s → "
                "https://arweave.net/%s", filepath, daemon_tx,
            )
            return daemon_tx

        # Fall-through: legacy one-shot path
        tags_json = json.dumps(extra_tags)
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
        """Get the RPC URL from merged config or default."""
        try:
            return get_params("network").get("premium_rpc_url", "https://api.mainnet-beta.solana.com")
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

        # Mainnet: fetch from an Arweave gateway, with MULTI-GATEWAY FALLBACK +
        # retry-with-backoff. A sovereign restore MUST retrieve every component
        # tarball or it halts, but public gateways are individually flaky (transient
        # 5xx / Cloudflare 52x / rate limits — e.g. arweave.net returned 572 on a T1
        # restore 2026-06-02). Try each gateway a few times before giving up.
        # follow_redirects=True is REQUIRED — gateways 302-redirect {tx_id} to a
        # content-addressed subdomain. Large timeout for multi-hundred-MB tarballs.
        import asyncio
        for gw in _ARWEAVE_GATEWAYS:
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
                        resp = await client.get(f"{gw}/{tx_id}")
                    if resp.status_code == 200:
                        return resp.content
                    # transient (5xx / Cloudflare 52x / 429) → backoff + retry; a
                    # plain 4xx (404 etc.) → no retry, fall through to next gateway.
                    if resp.status_code == 429 or resp.status_code >= 500:
                        logger.warning("[ArweaveStore] %s/%s… HTTP %d (attempt %d) — retrying",
                                       gw, tx_id[:12], resp.status_code, attempt + 1)
                        await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
                        continue
                    logger.warning("[ArweaveStore] %s/%s… HTTP %d — next gateway",
                                   gw, tx_id[:12], resp.status_code)
                    break
                except Exception as e:
                    logger.warning("[ArweaveStore] %s fetch error: %s — retry/next", gw, e)
                    await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
        logger.error("[ArweaveStore] Fetch %s FAILED across all gateways %s",
                     tx_id[:16], _ARWEAVE_GATEWAYS)
        return None

    async def fetch_to_file(self, tx_id: str, dest_path: str) -> bool:
        """STREAM a tarball from Arweave to `dest_path` in fixed-size chunks —
        constant memory regardless of tarball size (a sovereign restore fetches
        multi-hundred-MB component tarballs; loading them whole OOMs a small box).
        Same multi-gateway + retry-with-backoff policy as `fetch`. Returns True on
        success (file fully written), False otherwise."""
        import asyncio
        import httpx

        if tx_id.startswith("devnet_"):
            local_path = Path("data/arweave_devnet") / f"{tx_id}.data"
            if local_path.exists():
                import shutil
                shutil.copyfile(local_path, dest_path)
                return True
            return False

        for gw in _ARWEAVE_GATEWAYS:
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
                        async with client.stream("GET", f"{gw}/{tx_id}") as resp:
                            if resp.status_code == 200:
                                with open(dest_path, "wb") as fh:
                                    async for chunk in resp.aiter_bytes(1 << 20):  # 1 MiB
                                        fh.write(chunk)
                                return True
                            if resp.status_code == 429 or resp.status_code >= 500:
                                logger.warning("[ArweaveStore] stream %s/%s… HTTP %d (attempt %d) — retrying",
                                               gw, tx_id[:12], resp.status_code, attempt + 1)
                                await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
                                continue
                            logger.warning("[ArweaveStore] stream %s/%s… HTTP %d — next gateway",
                                           gw, tx_id[:12], resp.status_code)
                            break
                except Exception as e:
                    logger.warning("[ArweaveStore] %s stream error: %s — retry/next", gw, e)
                    await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
        logger.error("[ArweaveStore] Stream-fetch %s FAILED across all gateways %s",
                     tx_id[:16], _ARWEAVE_GATEWAYS)
        return False

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
