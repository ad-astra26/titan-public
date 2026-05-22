#!/usr/bin/env python3
"""Outer-memory → CAS migration — DRY-RUN analyzer (Phase 0 / 0B, D-SPEC-102).

Reads existing batch blocks, computes exactly what a backfill into the content-
addressed store WOULD write (which blobs, how many bytes, how much dedup), and
reports it. **This tool writes NOTHING** — it is the dry-run + verification baseline
the real (gated, greenlit) executor will be built on (rFP D-P0-3).

Honesty note printed in the report: backfilling tx_summaries into the CAS is
*additive* — it gives the engine day-one history + a dedup'd content store, but it
does **not** shrink the immutable `chain_*.bin` files (their `payload_hash` must keep
matching the inlined bytes — INV-3 / G16). Reclaiming the existing on-disk bloat is a
separate *chain-compaction* decision, flagged here, never performed by this tool.

READ-ONLY. Tolerant of unparseable / tampered blocks (skips, counts them).

Usage:
    python scripts/migrate_outer_memory_to_cas.py [--data-dir DIR]
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


def plan(data_dir: Path) -> dict:
    """Compute the backfill plan WITHOUT writing anything. Returns a result dict."""
    data_dir = Path(data_dir)
    chain_files = sorted(data_dir.glob("chain_*.bin"))
    sidechain_dir = data_dir / "sidechains"
    if sidechain_dir.exists():
        chain_files += sorted(sidechain_dir.glob("chain_*.bin"))

    per_fork: dict[str, dict] = defaultdict(
        lambda: {"blocks": 0, "inline_blocks": 0, "inline_bytes": 0}
    )
    unique_blobs: dict[str, int] = {}  # hash -> blob size (dedup projection)
    total_inline_bytes = 0
    already_slim = 0

    for cf in chain_files:
        fork = cf.stem.replace("chain_", "")
        for _height, _ttype, _source, content in iter_block_contents(cf):
            per_fork[fork]["blocks"] += 1
            if "content_summaries_hash" in content:
                already_slim += 1
                continue
            summaries = content.get("tx_summaries")
            if not summaries:
                continue
            blob = msgpack.packb(summaries, use_bin_type=True)
            h = hashlib.sha256(blob).hexdigest()
            unique_blobs.setdefault(h, len(blob))
            per_fork[fork]["inline_blocks"] += 1
            per_fork[fork]["inline_bytes"] += len(blob)
            total_inline_bytes += len(blob)

    projected_cas_bytes = sum(unique_blobs.values())
    return {
        "data_dir": str(data_dir),
        "per_fork": dict(per_fork),
        "inline_blocks_total": sum(f["inline_blocks"] for f in per_fork.values()),
        "total_inline_bytes": total_inline_bytes,
        "unique_blobs": len(unique_blobs),
        "projected_cas_bytes": projected_cas_bytes,
        "dedup_saved_bytes": total_inline_bytes - projected_cas_bytes,
        "already_slim_blocks": already_slim,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/home/antigravity/projects/titan/data/timechain")
    args = ap.parse_args()

    r = plan(Path(args.data_dir))
    print("=" * 88)
    print(f"OUTER-MEMORY → CAS MIGRATION — DRY-RUN (writes nothing)  ·  {r['data_dir']}")
    print("=" * 88)
    print(f"{'fork':<16}{'blocks':>12}{'w/inline':>12}{'inline bytes':>16}")
    for fork, s in sorted(r["per_fork"].items(), key=lambda kv: -kv[1]["inline_bytes"]):
        print(f"{fork:<16}{s['blocks']:>12,}{s['inline_blocks']:>12,}{_human(s['inline_bytes']):>16}")
    print("-" * 88)
    print(f"  batch blocks with inline tx_summaries : {r['inline_blocks_total']:,}")
    print(f"  total inline summary bytes            : {_human(r['total_inline_bytes'])}")
    print(f"  unique CAS blobs to write             : {r['unique_blobs']:,}")
    print(f"  projected CAS size (after dedup)      : {_human(r['projected_cas_bytes'])}")
    print(f"  dedup savings                         : {_human(r['dedup_saved_bytes'])}")
    print(f"  blocks already slim (have hash ref)   : {r['already_slim_blocks']:,}")
    print("=" * 88)
    print("NOTE: backfill is ADDITIVE — it populates the CAS, it does NOT shrink the")
    print("      immutable chain_*.bin files (INV-3/G16: payload_hash must keep matching")
    print("      inlined bytes). Reclaiming on-disk chain bloat is a SEPARATE chain-")
    print("      compaction decision, never performed by this dry-run tool.")
    print("      No bytes were written. Real backfill executor is a future gated step.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
