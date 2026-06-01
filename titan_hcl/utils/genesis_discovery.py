"""
utils/genesis_discovery.py — Wallet-only genesis discovery for resurrection.

INV-MBR-0 / INV-MBR-10 (canonical contract): from ONLY the Titan's public
address (`titan_pubkey`), discover everything a resurrection needs that is NOT
the Maker's offline Shard-1:

  - **shard3_tx** — the on-chain `TITAN_GENESIS_SHARD3:{hex}` memo (the Shard-3
    anchor). Found by walking the wallet's OWN signature history (the memo is a
    transaction on the identity wallet) — works on any RPC, no DAS needed.
  - **the GenesisNFT + its `ar://` identity metadata** — Maker, Constitution
    SHA-256, Birth DNA SHA-256, identity, provenance. Found via the DAS read API
    (`getAssetsByOwner` / `getAsset`); best-effort (needs a DAS-capable RPC).

THERE IS NO ENVELOPE. The Maker holds Shard-1 (secret) and knows the public
address (printed alongside the shard; the address is not a secret). The wallet +
chain + Arweave supply the rest. This module is **pure read** — no writes, no SOL
spend, no signing. Why a blockchain and not an off-site bucket: the recovery is
sovereign and publicly anchored, reconstructable from the wallet + one human shard.

See ARCHITECTURE_mainnet_birth_resurrection.md §R2 (Canonical Shard-3 discovery).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_SHARD3_PREFIX = "TITAN_GENESIS_SHARD3:"
# The GenesisNFT mint name (genesis_ceremony.py mints "Titan Soul Gen 1").
_GENESIS_NFT_NAME_HINTS = ("titan soul", "titan genesis", "sovereign soul")
# A full paginated wallet walk caps at this many pages (1000 sigs/page) so a
# pathological history can't loop forever. 50 pages = 50k sigs ≫ any real wallet.
_MAX_SIG_PAGES = 50
_SIG_PAGE_LIMIT = 1000


async def _rpc(client, rpc_url: str, method: str, params: list) -> dict:
    """Single JSON-RPC POST. Raises on transport error; returns the parsed body."""
    resp = await client.post(rpc_url, json={
        "jsonrpc": "2.0", "id": 1, "method": method, "params": params,
    })
    resp.raise_for_status()
    return resp.json()


def _extract_shard3_hex(memo: object) -> Optional[str]:
    """Pull the encrypted-shard hex out of a `TITAN_GENESIS_SHARD3:{hex}` memo.

    The RPC `memo` field may be prefixed with a byte-count, e.g.
    `"[186] TITAN_GENESIS_SHARD3:ab12…"` — we slice from the prefix onward and
    strip any trailing quote/whitespace.
    """
    if not memo:
        return None
    s = str(memo)
    if _SHARD3_PREFIX not in s:
        return None
    idx = s.index(_SHARD3_PREFIX)
    return s[idx + len(_SHARD3_PREFIX):].strip().rstrip('"').strip()


async def find_shard3_anchor(titan_pubkey: str, rpc_url: str) -> Optional[dict]:
    """Walk the wallet's full signature history for the Shard-3 memo.

    The `TITAN_GENESIS_SHARD3:` memo is a transaction ON the identity wallet, so
    it is always reachable from `titan_pubkey` alone — no NFT, no envelope. We
    paginate (`before=`) to genesis rather than the old `limit=50` blind peek,
    because a months-deep genesis memo is far beyond the most recent 50 sigs.

    Returns ``{"shard3_tx": <sig>, "encrypted_hex": <hex>}`` or ``None``.
    Pure read.
    """
    import httpx

    before: Optional[str] = None
    pages = 0
    async with httpx.AsyncClient(timeout=20) as client:
        while pages < _MAX_SIG_PAGES:
            pages += 1
            params: list = [titan_pubkey, {"limit": _SIG_PAGE_LIMIT}]
            if before:
                params[1]["before"] = before
            body = await _rpc(client, rpc_url, "getSignaturesForAddress", params)
            sigs = body.get("result") or []
            if not sigs:
                break
            for entry in sigs:
                enc = _extract_shard3_hex(entry.get("memo"))
                if enc:
                    sig = entry.get("signature")
                    logger.info("[genesis_discovery] Shard-3 memo found at %s "
                                "(page %d).", str(sig)[:16], pages)
                    return {"shard3_tx": sig, "encrypted_hex": enc}
            before = sigs[-1].get("signature")
            if not before:
                break
        logger.warning("[genesis_discovery] Shard-3 memo not found in %d page(s) "
                        "of wallet history.", pages)
    return None


async def fetch_shard3_from_tx(shard3_tx: str, rpc_url: str) -> Optional[str]:
    """Fetch a known Shard-3 anchor TX and extract the encrypted-shard hex.

    Used when the anchor TX is already known (e.g. from the NFT recovery pointer
    or the local record) — a single `getTransaction` instead of a wallet walk.
    Pure read. Returns the hex or ``None``.
    """
    import httpx

    async with httpx.AsyncClient(timeout=20) as client:
        body = await _rpc(client, rpc_url, "getTransaction", [
            shard3_tx,
            {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0},
        ])
    result = body.get("result")
    if not result:
        return None
    # Memo program data surfaces in logMessages and in parsed instructions.
    for msg in result.get("meta", {}).get("logMessages", []) or []:
        enc = _extract_shard3_hex(msg)
        if enc:
            return enc
    message = result.get("transaction", {}).get("message", {})
    for ix in message.get("instructions", []) or []:
        enc = _extract_shard3_hex(ix.get("parsed"))
        if enc:
            return enc
    return None


def _is_genesis_asset(asset: dict) -> bool:
    """Heuristic match for the GenesisNFT among a wallet's DAS assets."""
    content = asset.get("content", {}) or {}
    meta = content.get("metadata", {}) or {}
    name = str(meta.get("name", "")).lower()
    return any(h in name for h in _GENESIS_NFT_NAME_HINTS)


async def find_genesis_nft(titan_pubkey: str, das_rpc_url: str) -> Optional[dict]:
    """Enumerate the wallet's DAS assets and return the GenesisNFT asset.

    Wallet-only (`getAssetsByOwner`) — requires a DAS-capable RPC (Helius /
    Triton / QuikNode). Returns the raw DAS asset dict (carries `id` = the NFT
    address and `content.json_uri` = the `ar://` metadata) or ``None`` (no DAS
    support, or no matching asset). Pure read.
    """
    import httpx

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            body = await _rpc(client, das_rpc_url, "getAssetsByOwner", [{
                "ownerAddress": titan_pubkey,
                "page": 1, "limit": 1000,
            }])
        except Exception as e:
            logger.info("[genesis_discovery] getAssetsByOwner unavailable "
                        "(non-DAS RPC?): %s", e)
            return None
    if "error" in body:
        logger.info("[genesis_discovery] DAS error (non-DAS RPC?): %s",
                    body.get("error"))
        return None
    items = (body.get("result") or {}).get("items") or []
    for asset in items:
        if _is_genesis_asset(asset):
            return asset
    logger.warning("[genesis_discovery] No GenesisNFT among %d owned asset(s).",
                   len(items))
    return None


async def get_asset(nft_address: str, das_rpc_url: str) -> Optional[dict]:
    """DAS `getAsset(nft_address)` — direct fetch when the NFT address is known.

    Returns the asset dict or ``None``. Pure read.
    """
    import httpx

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            body = await _rpc(client, das_rpc_url, "getAsset", [nft_address])
        except Exception as e:
            logger.info("[genesis_discovery] getAsset unavailable: %s", e)
            return None
    if "error" in body:
        return None
    return body.get("result")


def _asset_json_uri(asset: dict) -> Optional[str]:
    """Extract the off-chain metadata URI from a DAS asset."""
    return ((asset.get("content", {}) or {}).get("json_uri")) or None


async def read_nft_identity(asset: dict) -> dict:
    """Fetch the GenesisNFT's `ar://` metadata and parse the identity it commits.

    Returns a dict with whatever it can read: ``maker``, ``titan_pubkey``,
    ``constitution_sha``, ``birth_dna_sha``, ``architecture`` + the raw
    ``metadata``. The NFT metadata is the COMPLETE identity root (INV-MBR-5/8):
    directives + DNA are committed here as SHA-256 hashes, verifiable against the
    full linked Arweave docs. Pure read.
    """
    import httpx

    out: dict = {}
    uri = _asset_json_uri(asset)
    if not uri:
        return out
    # Normalize ar:// → gateway URL.
    if uri.startswith("ar://"):
        uri = "https://arweave.net/" + uri[len("ar://"):]
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(uri)
            resp.raise_for_status()
            meta = resp.json()
    except Exception as e:
        logger.info("[genesis_discovery] NFT metadata fetch failed (%s): %s",
                    uri, e)
        return out

    out["metadata"] = meta
    # Recovery block (INV-MBR-5a) — the wallet-only Shard-3 pointer embedded in
    # the NFT's own metadata. Present on Titans minted with the recovery block.
    recovery = meta.get("recovery")
    if isinstance(recovery, dict):
        out["recovery"] = recovery
        if recovery.get("shard3_tx"):
            out["shard3_tx"] = recovery["shard3_tx"]
        if recovery.get("vault_pda"):
            out.setdefault("vault_pda", recovery["vault_pda"])
    # Map Metaplex attributes (trait_type/value) onto the identity fields.
    attr_map = {
        "maker": "maker", "creator": "maker",
        "titan pubkey": "titan_pubkey", "titan address": "titan_pubkey",
        "constitution sha-256": "constitution_sha",
        "constitution sha256": "constitution_sha",
        "birth dna sha-256": "birth_dna_sha",
        "birth dna sha256": "birth_dna_sha",
        "architecture": "architecture",
    }
    for attr in meta.get("attributes", []) or []:
        key = str(attr.get("trait_type", "")).strip().lower()
        field = attr_map.get(key)
        if field and field not in out:
            out[field] = attr.get("value")
    return out


async def discover_genesis(
    titan_pubkey: str,
    rpc_url: str,
    *,
    das_rpc_url: Optional[str] = None,
    nft_address: Optional[str] = None,
) -> dict:
    """Wallet-only discovery orchestrator (INV-MBR-10).

    From `titan_pubkey` alone, returns a best-effort dict:
      - ``shard3_tx`` / ``shard3_encrypted_hex`` — the Shard-3 anchor (CRITICAL;
        recovered via the wallet-history walk, works on any RPC).
      - ``nft_address`` / ``nft_metadata`` + ``maker`` / ``constitution_sha`` /
        ``birth_dna_sha`` — the identity (best-effort; needs a DAS RPC).

    `das_rpc_url` defaults to `rpc_url` (works if that RPC is DAS-capable).
    `nft_address`, if supplied (public, e.g. from a local record), is used as a
    direct `getAsset` fast-path — but is NOT required (true wallet-only via
    `getAssetsByOwner`). Pure read; never raises for a missing artifact — the
    caller decides what is fatal (Shard-3 absence is; identity absence degrades).
    """
    das_url = das_rpc_url or rpc_url
    out: dict = {}

    # ── 1. GenesisNFT identity + recovery pointer (canonical, INV-MBR-10) ──
    # The NFT carries identity (Maker, Constitution/DNA hashes) and — when minted
    # with the recovery block (INV-MBR-5a) — the `shard3_tx` pointer, so Shard-3
    # discovery is wallet-only via the NFT. Best-effort (needs a DAS RPC).
    asset = None
    if nft_address:
        asset = await get_asset(nft_address, das_url)
    if asset is None:
        asset = await find_genesis_nft(titan_pubkey, das_url)
    if asset is not None:
        out["nft_address"] = asset.get("id") or nft_address
        ident = await read_nft_identity(asset)
        out.update({k: v for k, v in ident.items() if v is not None})

    # ── 2. Resolve Shard-3 ──
    # Prefer the NFT-embedded `shard3_tx` pointer (INV-MBR-5a). Fall back to the
    # full wallet-history walk (`find_shard3_anchor`) — works on ANY RPC and
    # covers T1 (immutable NFT, no embedded pointer) + non-DAS endpoints.
    if out.get("shard3_tx") and not out.get("shard3_encrypted_hex"):
        enc = await fetch_shard3_from_tx(out["shard3_tx"], rpc_url)
        if enc:
            out["shard3_encrypted_hex"] = enc
    if not out.get("shard3_encrypted_hex"):
        anchor = await find_shard3_anchor(titan_pubkey, rpc_url)
        if anchor:
            out["shard3_tx"] = anchor["shard3_tx"]
            out["shard3_encrypted_hex"] = anchor["encrypted_hex"]

    return out
