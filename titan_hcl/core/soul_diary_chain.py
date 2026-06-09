"""titan_hcl/core/soul_diary_chain.py — the Soul-Diary hash-chain ledger.

A durable, append-only, tamper-evident chain of daily diary entries
(`RFP_titan_authored_soul_diary` §1.0 ⑤ / INV-SD-10/12). One row per UTC day:

    entry_hash      = sha256(entry_text)
    cumulative_hash = sha256(prev_cumulative ‖ entry_hash)   # chains across days

The same ``{entry_hash, cumulative_hash}`` is later carried by the Solana
DailyNFT (P8) and the main-chain ``dailyDiary`` tx (P2 ⑦) — three independent
roots of trust over one diary (INV-SD-10). Altering any past entry breaks the
chain from that day forward (``verify_chain``).

The ledger lives at ``data/soul_diary_chain.json`` (shadow-aware via
``resolve_data_path``) and is registered in §24.4.B ``PERSONALITY_PATHS`` (P8)
so it rides the daily incremental + Arweave backup (INV-SD-12). The parent
process never writes it; ``soul_diary_worker`` is the single-writer (INV-SD-6).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Optional

from titan_hcl.core.shadow_data_dir import resolve_data_path

logger = logging.getLogger(__name__)

LEDGER_REL = "data/soul_diary_chain.json"


def _ledger_path(path: Optional[str] = None) -> str:
    """The ledger path — explicit override (tests) or the shadow-aware default."""
    return path if path is not None else resolve_data_path(LEDGER_REL)


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_chain(path: Optional[str] = None) -> list[dict]:
    """Return the ledger rows (oldest-first), or ``[]`` if absent/unreadable."""
    p = _ledger_path(path)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError) as e:
        logger.error("[soul_diary_chain] ledger read failed (%s): %s", p, e)
        return []
    rows = data.get("entries") if isinstance(data, dict) else data
    return rows if isinstance(rows, list) else []


def _atomic_write(rows: list[dict], path: str) -> None:
    """tmp + fsync + os.replace — the §11.H.2 atomic-write pattern."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"entries": rows}, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def last_cumulative(path: Optional[str] = None) -> str:
    """The current chain head (``""`` for an empty chain) — the next row's prev."""
    rows = load_chain(path)
    return rows[-1].get("cumulative_hash", "") if rows else ""


def append_entry(date: str, entry_text: str, *, ts: Optional[float] = None,
                 path: Optional[str] = None) -> dict:
    """Append one chained row for ``date``'s ``entry_text``; return the row.

    Idempotent on ``date``: if a row already exists it is returned unchanged
    (the daily latch should prevent a second call; the ledger is defensive).
    """
    p = _ledger_path(path)
    rows = load_chain(p)
    for row in rows:
        if row.get("date") == date:
            logger.warning(
                "[soul_diary_chain] row for %s already exists — not re-appending",
                date)
            return row
    prev = rows[-1].get("cumulative_hash", "") if rows else ""
    entry_hash = _sha256_hex(entry_text)
    cumulative_hash = _sha256_hex(prev + entry_hash)
    row = {
        "date": date,
        "entry_hash": entry_hash,
        "cumulative_hash": cumulative_hash,
        "prev_cumulative": prev,
        "ts": ts,
        # anchor refs filled by later phases (P2 ⑦ / P8 ⑩)
        "nft_addr": None,
        "timechain_block": None,
        "arweave_uri": None,
    }
    rows.append(row)
    _atomic_write(rows, p)
    logger.info("[soul_diary_chain] appended %s: entry=%s cum=%s",
                date, entry_hash[:12], cumulative_hash[:12])
    return row


def update_refs(date: str, *, nft_addr: Optional[str] = None,
                timechain_block: Optional[str] = None,
                arweave_uri: Optional[str] = None,
                path: Optional[str] = None) -> bool:
    """Fill anchor refs on an existing row (P2 ⑦ / P8 ⑩). Returns True if updated."""
    p = _ledger_path(path)
    rows = load_chain(p)
    for row in rows:
        if row.get("date") == date:
            if nft_addr is not None:
                row["nft_addr"] = nft_addr
            if timechain_block is not None:
                row["timechain_block"] = timechain_block
            if arweave_uri is not None:
                row["arweave_uri"] = arweave_uri
            _atomic_write(rows, p)
            return True
    return False


def verify_chain(path: Optional[str] = None) -> bool:
    """Recompute the cumulative chain from stored entry_hashes; True iff intact.

    Verifies ``cumulative_hash[i] == sha256(cumulative_hash[i-1] ‖ entry_hash[i])``
    for every row (the G7 tamper-evidence check). Needs only the ledger — the
    per-row ``entry_hash`` is the commitment to each day's text.
    """
    rows = load_chain(path)
    prev = ""
    for i, row in enumerate(rows):
        entry_hash = row.get("entry_hash", "")
        expected = _sha256_hex(prev + entry_hash)
        if row.get("cumulative_hash") != expected:
            logger.error("[soul_diary_chain] chain broken at row %d (%s)",
                         i, row.get("date"))
            return False
        if row.get("prev_cumulative", prev) != prev:
            logger.error("[soul_diary_chain] prev_cumulative mismatch at row %d (%s)",
                         i, row.get("date"))
            return False
        prev = row.get("cumulative_hash", "")
    return True
