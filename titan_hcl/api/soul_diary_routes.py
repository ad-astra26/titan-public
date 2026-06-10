"""titan_hcl/api/soul_diary_routes.py — the public Soul-Diary archive API.

`RFP_titan_authored_soul_diary` §7.P9 / §6.8 / INV-SD-3. A read-only, PUBLIC
surface (the target of the daily X-post link + the example.com archive route)
that serves the **sanitized public projection** of each daily entry from the
hash-chain ledger — NEVER the private chronicle. The ledger only ever holds the
public projection + the verification hashes (the private entry lives in
`titan_chronicles.md` + self-memory, off this surface), so the API is
private-safe by construction; we additionally re-run `sanitize_for_public` at the
edge as the fail-closed backstop (G9). Sequenced BEFORE P10 so the X-post link
resolves.

Isolated `APIRouter` (mirrors `chat.py`/`pitch_chat.py`) — no auth (public read),
no coupling to the big v6 route table.
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from titan_hcl.core import soul_diary_chain
from titan_hcl.utils.privacy import sanitize_for_public

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v6/soul_diary", tags=["soul_diary"])


def _solana_explorer(nft_addr: str | None, *, devnet: bool = False) -> str | None:
    if not nft_addr:
        return None
    suffix = "?cluster=devnet" if devnet else ""
    return f"https://explorer.solana.com/address/{nft_addr}{suffix}"


def _arweave_gateway(arweave_uri: str | None) -> str | None:
    if not arweave_uri:
        return None
    tx = arweave_uri.replace("ar://", "")
    if tx.startswith("devnet_") or tx.startswith("fake_"):
        return None                       # local pseudo-tx — not publicly resolvable
    return f"https://gateway.irys.xyz/{tx}"


def build_public_view(row: dict) -> dict:
    """The PUBLIC projection of a ledger row (INV-SD-3) — sanitized at the edge,
    with resolvable explorer/gateway links. Never exposes the private entry."""
    pub_entry, _ = sanitize_for_public(row.get("public_entry") or "")
    distillation, _ = sanitize_for_public(row.get("distillation") or "")
    nft_addr = row.get("nft_addr")
    arweave_uri = row.get("arweave_uri")
    return {
        "date": row.get("date"),
        "entry": pub_entry,                       # full sanitized entry (archive renders)
        "distillation": distillation,             # the short public share
        "entry_hash": row.get("entry_hash"),
        "cumulative_hash": row.get("cumulative_hash"),
        "art": os.path.basename(row["art_path"]) if row.get("art_path") else None,
        "nft_addr": nft_addr,
        "arweave_uri": arweave_uri,
        "timechain_block": row.get("timechain_block"),
        "links": {
            "solana_explorer": _solana_explorer(nft_addr),
            "arweave": _arweave_gateway(arweave_uri),
        },
    }


def _row_for_date(date: str) -> dict | None:
    for row in soul_diary_chain.load_chain():
        if row.get("date") == date:
            return row
    return None


def get_soul_diary_entry(date: str) -> dict:
    """The public projection for one UTC date (``ok=False`` when absent)."""
    try:
        row = _row_for_date(date)
        if row is not None:
            return {"ok": True, "entry": build_public_view(row)}
        return {"ok": False, "error": "not_found", "date": date}
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary_api] entry read failed for %s: %s", date, e)
        return {"ok": False, "error": "read_error", "date": date}


def get_soul_diary_index(limit: int = 30) -> dict:
    """Newest-first list of dates + their anchor refs (the archive index)."""
    try:
        rows = soul_diary_chain.load_chain()
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary_api] index read failed: %s", e)
        return {"ok": False, "error": "read_error", "entries": [], "total": 0}
    items = [{
        "date": r.get("date"),
        "cumulative_hash": r.get("cumulative_hash"),
        "nft_addr": r.get("nft_addr"),
        "has_art": bool(r.get("art_path")),
    } for r in rows]
    items.reverse()  # newest-first
    capped = max(1, min(int(limit), 365))
    return {"ok": True, "entries": items[:capped], "total": len(rows)}


@router.get("")
def soul_diary_index(limit: int = 30) -> dict:
    return get_soul_diary_index(limit)


@router.get("/{date}")
def soul_diary_entry(date: str) -> dict:
    return get_soul_diary_entry(date)


@router.get("/{date}/art")
def soul_diary_art(date: str):
    """Serve the day's procedural felt-art image (INV-SD-4 — no private data, so
    public-safe). The archive page <img> renders this."""
    try:
        row = _row_for_date(date)
    except Exception:  # noqa: BLE001
        row = None
    art_path = (row or {}).get("art_path") if row else None
    if art_path and os.path.isfile(art_path):
        return FileResponse(art_path, media_type="image/jpeg")
    return JSONResponse({"ok": False, "error": "not_found", "date": date},
                        status_code=404)
