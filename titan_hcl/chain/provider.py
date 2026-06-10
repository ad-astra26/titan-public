"""ChainProvider — one lean async interface for all on-chain I/O.

RFP_chain_provider §1.2. This module holds the ABSTRACT interface every
component will depend on, plus the real `ArweaveChainProvider` (Phase A: the
Arweave/Irys data plane). The Solana trust plane (Phase B) and funding plane
(Phase C) are declared on the ABC and raise NotImplementedError until built.

Design rules (the invariants this realizes):
  • INV-CP-2 — transfer is STREAMED by default: `put` of a path streams through
    the Irys daemon; `get_to_file` streams the gateway response to disk in 1 MiB
    chunks (constant RAM). `get_bytes` is for SMALL objects (memos/JSON) only.
  • INV-CP-3 — NO per-call `subprocess node`: Irys I/O goes through the
    persistent daemon (`IrysPersistentClient`). Daemon-down ⇒ HARD-FAIL (loud),
    never a subprocess fallback (Q-CP-1).
  • INV-CP-6 — devnet ≠ mainnet: on devnet `put` writes the local
    `data/arweave_devnet/` cache + a `devnet_*` pseudo-tx; it never claims
    immutability, and `get_*`/`head` read that cache.
"""
from __future__ import annotations

import abc
import hashlib
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal, Optional, Union

logger = logging.getLogger(__name__)

HeadStatus = Literal["present", "missing", "unverified"]

# Gateways for reads/HEAD — Irys first (canonical, most-reliable), then base
# layer (mirrors ArweaveStore + scripts/backup_reconcile.py).
_GATEWAYS = (
    "https://gateway.irys.xyz",
    "https://arweave.net",
)
_DEVNET_CACHE = "data/arweave_devnet"
_CHUNK = 1 << 20  # 1 MiB streaming chunk (INV-CP-2)

PutSource = Union[bytes, bytearray, str]  # bytes/bytearray ⇒ data; str ⇒ a path


class ChainProvider(abc.ABC):
    """The only chain verbs the codebase needs. Inject one per process; tests use
    `FakeChainProvider`. See RFP_chain_provider §1.2."""

    # ── DATA plane (Phase A) — immutable bytes on Arweave/Irys ──────────────
    @abc.abstractmethod
    async def put(self, src: PutSource, *,
                  content_type: str = "application/octet-stream",
                  tags: Optional[dict] = None) -> str:
        """Store `src` (a path → streamed, or bytes → in-memory) and return its
        tx_id. Honors INV-CP-2/3/6."""

    @abc.abstractmethod
    async def get_to_file(self, tx_id: str, dest_path: str) -> bool:
        """Stream the object to `dest_path` in fixed chunks (constant RAM).
        Returns True on success (file fully written), False otherwise."""

    @abc.abstractmethod
    async def get_bytes(self, tx_id: str) -> Optional[bytes]:
        """Fetch a SMALL object (memo/JSON) whole. Do NOT use for tarballs —
        use `get_to_file` (INV-CP-2)."""

    @abc.abstractmethod
    async def head(self, tx_id: str) -> HeadStatus:
        """Reachability without download: `present` (HTTP 200) / `missing`
        (definitive 404/410) / `unverified` (timeout/transient — NOT proven
        gone). Propagation-tolerant."""

    # ── TRUST plane (Phase B — Solana v3 ZK-memo chain) ─────────────────────
    async def commit_memo(self, memo_text: str, *, state_root: Optional[str] = None,
                          sovereignty_bp: Optional[int] = None) -> Optional[str]:
        raise NotImplementedError("ChainProvider.commit_memo — RFP Phase B")

    async def read_memo(self, tx_sig: str) -> Optional[str]:
        raise NotImplementedError("ChainProvider.read_memo — RFP Phase B")

    async def list_memos(self, address: str, *, limit: int) -> list[str]:
        raise NotImplementedError("ChainProvider.list_memos — RFP Phase B")

    # ── FUNDING plane (Phase C — Irys deposit) ──────────────────────────────
    async def balance(self) -> float:
        raise NotImplementedError("ChainProvider.balance — RFP Phase C")

    async def fund(self, lamports_sol: float) -> Optional[str]:
        raise NotImplementedError("ChainProvider.fund — RFP Phase C")


def _is_devnet_tx(tx_id: str) -> bool:
    return isinstance(tx_id, str) and tx_id.startswith("devnet_")


class ArweaveChainProvider(ChainProvider):
    """Real data-plane provider: persistent Irys daemon (mainnet) + local cache
    (devnet). Absorbs `ArweaveStore`'s upload/fetch I/O (RFP §5).
    """

    def __init__(self, *, keypair_path: str, network: str, rpc_url: str = "",
                 network_client=None, vault_program_id: Optional[str] = None):
        self._keypair_path = keypair_path
        self._network = (network or "devnet").strip().lower()
        self._rpc_url = rpc_url
        self._devnet = self._network not in ("mainnet", "mainnet-beta")
        # Solana TRUST plane (Phase B): the signer/sender (a HybridNetworkClient
        # exposing `.pubkey` + `.send_sovereign_transaction`). `commit_memo`
        # REQUIRES it; reads (`read_memo`/`list_memos`) need only `rpc_url`.
        self._network_client = network_client
        self._vault_program_id = vault_program_id

    # ── put ──
    async def put(self, src: PutSource, *,
                  content_type: str = "application/octet-stream",
                  tags: Optional[dict] = None) -> str:
        tags = tags or {}
        if self._devnet:
            return self._devnet_put(src, tags)
        from titan_hcl.utils.irys_persistent_client import get_daemon, IrysDaemonError
        try:
            daemon = await get_daemon(self._keypair_path, self._rpc_url)
        except Exception as e:  # daemon unavailable
            raise IrysDaemonError(
                f"Irys daemon unavailable for put — HARD-FAIL (INV-CP-3, no "
                f"subprocess fallback): {e}") from e
        if isinstance(src, str):
            res = await daemon.upload_file(src, content_type=content_type, tags=tags)
        else:
            res = await daemon.upload_data(bytes(src), content_type=content_type, tags=tags)
        return res.tx_id

    def _devnet_put(self, src: PutSource, tags: dict) -> str:
        data = self._read_src(src)
        store = Path(_DEVNET_CACHE)
        store.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(data).hexdigest()
        tx_id = f"devnet_{h[:43]}"
        (store / f"{tx_id}.data").write_bytes(data)
        try:
            import json
            (store / f"{tx_id}.tags.json").write_text(json.dumps(tags, indent=2))
        except OSError:
            pass
        logger.info("[ChainProvider] devnet put → %s (%d bytes)", tx_id, len(data))
        return tx_id

    @staticmethod
    def _read_src(src: PutSource) -> bytes:
        if isinstance(src, (bytes, bytearray)):
            return bytes(src)
        if isinstance(src, str):
            with open(src, "rb") as f:
                return f.read()
        raise TypeError(f"put src must be bytes or path str, got {type(src).__name__}")

    # ── get_to_file (streamed) ──
    async def get_to_file(self, tx_id: str, dest_path: str) -> bool:
        if _is_devnet_tx(tx_id):
            local = Path(_DEVNET_CACHE) / f"{tx_id}.data"
            if not local.exists():
                return False
            import shutil
            shutil.copyfile(local, dest_path)
            return True
        import asyncio
        import httpx
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        for gw in _GATEWAYS:
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as c:
                        async with c.stream("GET", f"{gw}/{tx_id}") as resp:
                            if resp.status_code == 200:
                                with open(dest_path, "wb") as fh:
                                    async for chunk in resp.aiter_bytes(_CHUNK):
                                        fh.write(chunk)
                                return True
                            if resp.status_code == 429 or resp.status_code >= 500:
                                await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
                                continue
                            break  # 4xx → next gateway
                except Exception as e:
                    logger.debug("[ChainProvider] get_to_file %s/%s… %s",
                                 gw, tx_id[:12], e)
                    await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
        logger.error("[ChainProvider] get_to_file FAILED %s across %s",
                     tx_id[:16], _GATEWAYS)
        return False

    # ── get_bytes (small objects only) ──
    async def get_bytes(self, tx_id: str) -> Optional[bytes]:
        if _is_devnet_tx(tx_id):
            local = Path(_DEVNET_CACHE) / f"{tx_id}.data"
            return local.read_bytes() if local.exists() else None
        import asyncio
        import httpx
        for gw in _GATEWAYS:
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as c:
                        resp = await c.get(f"{gw}/{tx_id}")
                    if resp.status_code == 200:
                        return resp.content
                    if resp.status_code == 429 or resp.status_code >= 500:
                        await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
                        continue
                    break
                except Exception:
                    await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
        return None

    # ── head (reachability, no download) ──
    async def head(self, tx_id: str, *, timeout: int = 30, attempts: int = 2) -> HeadStatus:
        if _is_devnet_tx(tx_id):
            local = Path(_DEVNET_CACHE) / f"{tx_id}.data"
            return "present" if local.exists() else "missing"
        last = ""
        for gw in _GATEWAYS:
            for _ in range(max(1, attempts)):
                try:
                    req = urllib.request.Request(f"{gw}/{tx_id}", method="HEAD")
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        if 200 <= resp.status < 300:
                            return "present"
                        last = f"HTTP {resp.status}"
                except urllib.error.HTTPError as e:
                    if e.code in (404, 410):
                        return "missing"  # definitive
                    last = f"HTTP {e.code}"
                except Exception as e:
                    last = str(e)[:48]
        logger.debug("[ChainProvider] head(%s) unverified: %s", tx_id[:16], last)
        return "unverified"

    # ── TRUST plane (Phase B) — absorbs the I/O tail of commit_event_v3_chain ──
    async def commit_memo(self, memo_text: str, *, state_root: Optional[str] = None,
                          sovereignty_bp: Optional[int] = None) -> Optional[str]:
        """Send ONE memo-bearing Solana tx (RFP §1.2). state_root set ⇒ bundle the
        ZK-Vault commit_state on the same tx (the event HEAD). Mirrors the live
        commit pattern: ixs = [commit_ix, memo_ix] for the head, [memo_ix] for a
        tail; priority="LOW". Memo BUILDING stays in the caller (`build_v3_memo`)."""
        if self._network_client is None:
            raise RuntimeError(
                "ChainProvider.commit_memo needs a network_client (Solana signer) "
                "— inject one for the trust plane")
        from titan_hcl.utils.solana_client import (
            build_memo_instruction, build_vault_commit_instruction)
        memo_ix = build_memo_instruction(self._network_client.pubkey, memo_text)
        if memo_ix is None:
            logger.error("[ChainProvider] commit_memo: memo ix build failed")
            return None
        ixs = [memo_ix]
        if state_root is not None:
            try:
                root_bytes = bytes.fromhex(state_root)
            except ValueError:
                logger.error("[ChainProvider] commit_memo: state_root not valid hex")
                return None
            commit_ix = build_vault_commit_instruction(
                self._network_client.pubkey, root_bytes,
                int(sovereignty_bp or 0), self._vault_program_id)
            if commit_ix is not None:
                ixs = [commit_ix, memo_ix]  # commit FIRST, then memo (live order)
        return await self._network_client.send_sovereign_transaction(ixs, priority="LOW")

    async def read_memo(self, tx_sig: str) -> Optional[str]:
        from titan_hcl.utils.solana_client import get_memo_for_tx
        return await get_memo_for_tx(tx_sig, rpc_url=self._rpc_url or None)

    async def list_memos(self, address: str, *, limit: int) -> list[str]:
        from titan_hcl.utils.solana_client import get_signatures_for_address
        return await get_signatures_for_address(
            address, rpc_url=self._rpc_url or None, limit=limit)
