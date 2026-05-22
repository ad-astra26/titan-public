"""Backup anchor chain utilities — rFP_backup_worker Phase 8 (§5.9).

Read + verify the append-only chain file written by RebirthBackup.anchor_backup_hash.
Chain format:

    data/backup_anchor_chain_{titan_id}.json
    {
      "version": 1,
      "titan_id": "T1",
      "anchors": [
        {
          "backup_id": int,            # monotonic index (0-based)
          "archive_hash": str,         # sha256 hex of the (encrypted, if enabled) tarball
          "prev_anchor_hash": str,     # "" for genesis, else prior entry's archive_hash
          "tx": str,                   # Solana memo TX signature
          "ts": int,                   # unix timestamp
          "backup_type": str,          # "personality"|"soul"|"timechain"
          "size_mb": float
        },
        ...
      ]
    }

Verification walks the chain and confirms each `prev_anchor_hash` matches the
immediately-preceding `archive_hash`. Any break reports the exact index so
Maker knows where tamper or loss began.
"""

import json
import os
from typing import Optional


def chain_path(titan_id: str, base_dir: str = "data") -> str:
    return os.path.join(base_dir, f"backup_anchor_chain_{titan_id}.json")


def read_chain(titan_id: str, base_dir: str = "data") -> list:
    p = chain_path(titan_id, base_dir)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("anchors", [])
    return []


def verify_chain(anchors: list) -> dict:
    """Walk the chain. Return {ok, length, break_index, break_reason}.

    ok = True iff every entry's prev_anchor_hash matches the previous entry's
    archive_hash (and the first entry has prev="").
    """
    if not anchors:
        return {"ok": True, "length": 0, "break_index": None, "break_reason": None}

    # Genesis: prev must be empty
    first = anchors[0]
    if first.get("prev_anchor_hash", "") not in ("", None):
        return {
            "ok": False, "length": len(anchors), "break_index": 0,
            "break_reason": "genesis entry has non-empty prev_anchor_hash",
        }

    for i in range(1, len(anchors)):
        expected = anchors[i - 1].get("archive_hash", "")
        got = anchors[i].get("prev_anchor_hash", "")
        if expected != got:
            return {
                "ok": False, "length": len(anchors), "break_index": i,
                "break_reason": (
                    f"entry[{i}].prev_anchor_hash={got[:16]}... "
                    f"does not match entry[{i-1}].archive_hash={expected[:16]}..."
                ),
            }
    return {"ok": True, "length": len(anchors), "break_index": None, "break_reason": None}


def verify_chain_file(titan_id: str, base_dir: str = "data") -> dict:
    """Convenience: read then verify. Returns same shape as verify_chain."""
    return verify_chain(read_chain(titan_id, base_dir))
