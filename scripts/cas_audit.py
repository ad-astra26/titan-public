#!/usr/bin/env python3
"""CAS audit — read-only consistency check for the content-addressed store.

Synthesis Engine Phase 0 / 0B safety net (D-SPEC-102). Verifies the invariant that
every batch block referencing CAS content resolves to a self-verifying blob, and
reports orphan blobs (present in the store, referenced by no block).

READ-ONLY. Parses `chain_*.bin` directly (never opens a writable TimeChain on live
data) and only reads the CAS. Writes nothing, deletes nothing — orphans are reported
for awareness, never collected (INV-3: canonical blobs are never removed; reference-
counted GC is a later phase).

Exit code: 0 = clean; 1 = missing/corrupt referenced blob (a real integrity fault).

Usage:
    python scripts/cas_audit.py [--data-dir DIR] [--cas-dir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from titan_hcl.synthesis.chain_reader import iter_block_contents
from titan_hcl.synthesis.content_store import (
    BlobNotFound,
    ContentStore,
    CorruptBlob,
)


def audit(data_dir: Path, cas_dir: Path) -> dict:
    """Read-only audit. Returns a result dict; writes nothing, deletes nothing."""
    data_dir = Path(data_dir)
    store = ContentStore(root=Path(cas_dir))

    referenced: set[str] = set()
    ok = missing = corrupt = 0
    faults: list[str] = []

    chain_files = sorted(data_dir.glob("chain_*.bin"))
    sidechain_dir = data_dir / "sidechains"
    if sidechain_dir.exists():
        chain_files += sorted(sidechain_dir.glob("chain_*.bin"))

    for cf in chain_files:
        for height, _ttype, _source, content in iter_block_contents(cf):
            h = content.get("content_summaries_hash")
            if not h:
                continue
            referenced.add(h)
            try:
                store.get(h)  # self-verifying read
                ok += 1
            except CorruptBlob:
                corrupt += 1
                faults.append(f"CORRUPT {h} (referenced by {cf.name}#{height})")
            except BlobNotFound:
                missing += 1
                faults.append(f"MISSING {h} (referenced by {cf.name}#{height})")

    # Orphan scan: blobs present in the store but referenced by no block.
    stored = (
        {
            p.name
            for p in store.root.rglob("*")
            if p.is_file() and not p.name.endswith(".tmp")
        }
        if store.root.exists()
        else set()
    )
    orphans = stored - referenced

    return {
        "data_dir": str(data_dir),
        "cas_dir": str(store.root),
        "referenced": len(referenced),
        "ok": ok,
        "missing": missing,
        "corrupt": corrupt,
        "stored": len(stored),
        "orphans": len(orphans),
        "faults": faults,
        "clean": missing == 0 and corrupt == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/home/antigravity/projects/titan/data/timechain")
    ap.add_argument("--cas-dir", default="/home/antigravity/projects/titan/data/content_blobs")
    args = ap.parse_args()

    r = audit(Path(args.data_dir), Path(args.cas_dir))
    print("=" * 84)
    print(f"CAS AUDIT  ·  chains={r['data_dir']}  ·  cas={r['cas_dir']}")
    print("=" * 84)
    print(f"  referenced blobs : {r['referenced']:,}")
    print(f"    ✓ resolved      : {r['ok']:,}")
    print(f"    ✗ missing       : {r['missing']:,}")
    print(f"    ✗ corrupt       : {r['corrupt']:,}")
    print(f"  stored blobs      : {r['stored']:,}")
    print(f"  orphan blobs      : {r['orphans']:,}  (reported only — never collected in Phase 0)")
    if r["faults"]:
        print("\n  FAULTS:")
        for fault in r["faults"][:50]:
            print(f"    {fault}")
        if len(r["faults"]) > 50:
            print(f"    ... and {len(r['faults']) - 50} more")
    print("=" * 84)
    if not r["clean"]:
        print("RESULT: ✗ FAIL — referenced CAS blobs are missing/corrupt")
        return 1
    print("RESULT: ✓ clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
