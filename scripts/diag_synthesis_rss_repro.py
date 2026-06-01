#!/usr/bin/env python3
"""Offline RSS reproduction of the synthesis tx-index tick on the llama.cpp
embedder (embedding-migration P6 verification, 2026-06-01).

Reads T1's REAL chain (read-only) but writes the FAISS shards to a temp dir so it
never touches live data. Samples RSS during the tick + reports tracemalloc top
allocations. The whole point of the migration: with llama.cpp replacing
fastembed/onnxruntime, the PEAK RSS during a multi-thousand-block backfill must
stay FLAT (~singleton 197 MB + spine), NOT climb to the ~3.5 GB the onnxruntime
CPU arena leaked.

Usage:
  TITAN_DATA_DIR=<dir holding .gguf_cache> python scripts/diag_synthesis_rss_repro.py [max_blocks]
"""
import os, sys, time, tracemalloc, sqlite3, tempfile, threading

os.environ.setdefault("TITAN_ID", "T1")
# Real chain (read-only). Override with TITAN_CHAIN_DIR if running off-box.
DATA = os.environ.get("TITAN_CHAIN_DIR", "/home/antigravity/projects/titan/data")
TMP = tempfile.mkdtemp(prefix="synth_rss_repro_")        # shards write here (safe)
MAX_BLOCKS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rss_mb():
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


_peak = {"rss": 0.0}
_stop = threading.Event()


def _sampler():
    while not _stop.is_set():
        _peak["rss"] = max(_peak["rss"], rss_mb())
        _stop.wait(0.25)


threading.Thread(target=_sampler, daemon=True).start()

print(f"baseline RSS: {rss_mb():.0f}MB  (chain={DATA}, shards→{TMP})")
tracemalloc.start(25)

import numpy as np
from titan_hcl.utils.text_embedder import get_text_embedder
_model = get_text_embedder()
print(f"after llama.cpp embedder load: {rss_mb():.0f}MB")


def _emb(text):
    return np.asarray(_model.encode(text), dtype=np.float32)


def _bemb(texts):
    vecs = np.asarray(_model.encode(list(texts)), dtype=np.float32)
    return [vecs[i] for i in range(vecs.shape[0])]


from titan_hcl.synthesis.synthesis_vector_index import SynthesisVectorStore
from titan_hcl.synthesis.tx_index_builder import TxIndexBuilder

store = SynthesisVectorStore(data_dir=TMP, embedder=_emb, batch_embedder=_bemb)
print(f"after store ctor: {rss_mb():.0f}MB")

conn = sqlite3.connect(f"file:{DATA}/timechain/index.db?mode=ro", uri=True, check_same_thread=False)
conn.row_factory = sqlite3.Row
builder = TxIndexBuilder(store=store, data_dir=DATA, index_db=conn)
print(f"after builder ctor: {rss_mb():.0f}MB")

t0 = time.perf_counter()
summary = builder.run(max_blocks=MAX_BLOCKS, from_scratch=True)
dt = time.perf_counter() - t0
_stop.set()
print(f"\n=== tick done in {dt:.1f}s — PEAK RSS during tick: {_peak['rss']:.0f}MB ===")
print(f"summary: scanned={summary['scanned']} indexed={summary['indexed']} "
      f"skipped={summary['skipped']} no_content={summary['no_content']}")
print(f"by_fork: {summary.get('by_fork')}")

cur, peak = tracemalloc.get_traced_memory()
print(f"tracemalloc current={cur/1e6:.0f}MB peak={peak/1e6:.0f}MB")
print("\n=== top 12 allocations by size ===")
for s in tracemalloc.take_snapshot().statistics("lineno")[:12]:
    fr = s.traceback[0]
    print(f"  {s.size/1e6:7.1f}MB  {s.count:>7} blks  {fr.filename.split('/')[-1]}:{fr.lineno}")
