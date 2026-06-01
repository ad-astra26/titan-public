#!/usr/bin/env python3
"""Backfill the tx_hash-native synthesis FAISS index from the existing chain.

Synthesis Engine Operator Closure, Phase A4. Populates the per-fork
`data/synthesis_vectors_<fork>.faiss` shards from the historical
conversation / declarative / procedural fork TXs already on the canonical
chain — so SEARCH returns hits over real history from the first soak, not just
forward from boot.

DESIGN DISCIPLINE (PLAN §A4):
  * Dry-run FIRST (the default). Reports how many blocks WOULD be embedded per
    fork, with zero writes, so you eyeball the scope before paying the embed cost.
  * Read-only on the canonical chain — ADDS the new index, never mutates
    chain_*.bin or block_index (INV-3; G16 no integrity-for-perf trade).
  * Idempotent — re-running is safe; a tx already in a shard is skipped, and the
    per-fork height watermark resumes where the last pass stopped.

Run ON each Titan against its local data/ (T3 first, then T1):
    source /home/antigravity/projects/titan/test_env/bin/activate
    python scripts/backfill_synthesis_tx_index.py                 # dry-run report
    python scripts/backfill_synthesis_tx_index.py --execute       # populate
    python scripts/backfill_synthesis_tx_index.py --execute --data-dir data --max 200000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Allow running from the repo root without an install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_embedders(threads: int = 2):
    """Fleet-standard llama.cpp BAAI/bge-small-en-v1.5 — the one engine embedding
    path (§3J.1). Returns (single, batch) — the batch path embeds a whole list in
    one call. Vectors are L2-normalized by the singleton. `threads` / OMP_NUM_THREADS
    (set in main() before import) caps CPU usage so the backfill leaves cores for
    the live Titans on a small/shared box (load discipline — devnet T2+T3 share 4
    cores); the singleton's n_threads is fixed, OMP still bounds the kernels."""
    import numpy as np
    from titan_hcl.utils.text_embedder import get_text_embedder
    model = get_text_embedder()

    def embed(text: str):
        return np.asarray(model.encode(text), dtype=np.float32)

    def embed_many(texts: list):
        vecs = np.asarray(model.encode(list(texts)), dtype=np.float32)
        return [vecs[i] for i in range(vecs.shape[0])]

    return embed, embed_many


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.environ.get("TITAN_DATA_DIR", "data"),
                    help="Titan data directory (holds timechain/index.db). Default: data")
    ap.add_argument("--execute", action="store_true",
                    help="Actually embed + write. Without this flag = dry-run report only.")
    ap.add_argument("--max", type=int, default=500000,
                    help="Max blocks to scan this run (default 500000).")
    ap.add_argument("--threads", type=int, default=2,
                    help="Cap embed (OMP) threads (default 2) so the backfill "
                         "leaves cores for live Titans on a shared box.")
    args = ap.parse_args()

    # Cap embed threads BEFORE the llama.cpp embedder is constructed. Load
    # discipline on a 4-core box shared by two live devnet twins.
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, int(args.threads))))

    data_dir = args.data_dir
    index_db = os.path.join(data_dir, "timechain", "index.db")
    if not os.path.exists(index_db):
        print(f"[backfill] no index.db at {index_db} — nothing to backfill.")
        return 1

    from titan_hcl.synthesis.synthesis_vector_index import INDEXED_FORKS, SynthesisVectorStore
    from titan_hcl.synthesis.tx_index_builder import TxIndexBuilder

    dry = not args.execute
    mode = "DRY-RUN (no writes)" if dry else "EXECUTE (populate index)"
    print(f"[backfill] {mode} — data_dir={data_dir} forks={list(INDEXED_FORKS)} max={args.max}")

    embedder = batch_embedder = None
    if not dry:
        embedder, batch_embedder = _build_embedders(threads=args.threads)
    store = SynthesisVectorStore(
        data_dir=data_dir, embedder=embedder, batch_embedder=batch_embedder)
    builder = TxIndexBuilder(store=store, data_dir=data_dir)

    t0 = time.time()
    # Backfill = from_scratch (ignore watermark) so a re-run re-verifies the full
    # history; has()/dedup keeps it idempotent and cheap on a populated index.
    summary = builder.run(max_blocks=args.max, from_scratch=True, dry_run=dry)
    dt = time.time() - t0
    builder.close()

    print(f"\n[backfill] done in {dt:.1f}s")
    print(f"  scanned    : {summary['scanned']}")
    print(f"  {'would index' if dry else 'indexed'}: {summary['indexed']}")
    print(f"  skipped    : {summary['skipped']} (already present)")
    print(f"  no_content : {summary['no_content']}")
    for fork, st in summary.get("by_fork", {}).items():
        print(f"    {fork:13s}: scanned={st['scanned']} "
              f"{'would_index' if dry else 'indexed'}={st['indexed']} "
              f"skipped={st['skipped']} no_content={st['no_content']}")
    if not dry:
        print(f"\n  shard sizes: {store.stats()}")
    else:
        print("\n  (dry-run — re-run with --execute to populate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
