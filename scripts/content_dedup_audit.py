#!/usr/bin/env python3
"""Content-dedup audit — where does CAS dedup actually pay? (Phase 0 / 0B)

READ-ONLY. Walks every chain block and, for each large top-level content field,
records (field_key, sha256(value), size). Aggregates per field key:
  occurrences · unique blobs · total bytes · DEDUPABLE bytes (total − unique).

This is the empirical basis for the 0B pivot to repeated-content dedup: it tells us
which fields (chat bodies, tool results, art/audio refs, …) carry the repeated bytes
worth content-addressing — so we don't repeat the tx_summaries lesson (where dedup
turned out to be 0). Writes nothing.

Usage:
    python scripts/content_dedup_audit.py [--data-dir DIR] [--min-field-bytes N]
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import msgpack

from titan_hcl.synthesis.chain_reader import iter_block_contents


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f}{unit}"
        n /= 1024.0
    return f"{n:,.1f}TB"


def audit(data_dir: Path, min_field_bytes: int = 256) -> dict:
    """Per-field content-size + repetition analysis. Returns a result dict."""
    data_dir = Path(data_dir)
    chain_files = sorted(data_dir.glob("chain_*.bin"))
    sidechain_dir = data_dir / "sidechains"
    if sidechain_dir.exists():
        chain_files += sorted(sidechain_dir.glob("chain_*.bin"))

    # field_key -> {"occ": n, "bytes": total, "hashes": {hash: size}}
    fields: dict[str, dict] = defaultdict(lambda: {"occ": 0, "bytes": 0, "hashes": {}})

    for cf in chain_files:
        for _height, _ttype, _source, content in iter_block_contents(cf):
            for key, value in content.items():
                try:
                    blob = msgpack.packb(value, use_bin_type=True)
                except Exception:
                    continue
                if len(blob) < min_field_bytes:
                    continue
                h = hashlib.sha256(blob).hexdigest()
                rec = fields[key]
                rec["occ"] += 1
                rec["bytes"] += len(blob)
                rec["hashes"].setdefault(h, len(blob))

    rows = []
    for key, rec in fields.items():
        unique_bytes = sum(rec["hashes"].values())
        rows.append({
            "field": key,
            "occurrences": rec["occ"],
            "unique": len(rec["hashes"]),
            "total_bytes": rec["bytes"],
            "unique_bytes": unique_bytes,
            "dedupable_bytes": rec["bytes"] - unique_bytes,
        })
    rows.sort(key=lambda r: -r["dedupable_bytes"])
    return {
        "data_dir": str(data_dir),
        "min_field_bytes": min_field_bytes,
        "rows": rows,
        "total_dedupable": sum(r["dedupable_bytes"] for r in rows),
        "total_field_bytes": sum(r["total_bytes"] for r in rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/home/antigravity/projects/titan/data/timechain")
    ap.add_argument("--min-field-bytes", type=int, default=256)
    args = ap.parse_args()

    r = audit(Path(args.data_dir), args.min_field_bytes)
    print("=" * 100)
    print(f"CONTENT-DEDUP AUDIT  ·  {r['data_dir']}  ·  fields ≥ {r['min_field_bytes']}B")
    print("=" * 100)
    print(f"{'field':<26}{'occ':>9}{'unique':>9}{'total':>13}{'unique':>13}{'dedupable':>13}")
    for row in r["rows"][:25]:
        print(
            f"{row['field']:<26}{row['occurrences']:>9,}{row['unique']:>9,}"
            f"{_human(row['total_bytes']):>13}{_human(row['unique_bytes']):>13}"
            f"{_human(row['dedupable_bytes']):>13}"
        )
    print("-" * 100)
    print(f"  total field bytes (≥{r['min_field_bytes']}B fields) : {_human(r['total_field_bytes'])}")
    print(f"  total DEDUPABLE bytes via CAS              : {_human(r['total_dedupable'])}")
    print("=" * 100)
    print("Reading: 'dedupable' = bytes a single content-addressed store would collapse")
    print("(repeated identical field values). High dedupable ⇒ worth CAS-offloading that")
    print("field; ~0 dedupable ⇒ same lesson as tx_summaries (offload bounds growth only).")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
