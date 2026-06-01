"""Tests for the llama.cpp text embedder (Phase 13 §3J.1 migration P2).

Covers: dim/norm correctness, single-vs-batch shape semantics, parity against the
saved fastembed bge-small vectors (cosine >= 0.999), the boot self-test, and the
critical concurrency invariant — llama.cpp embed is NOT thread-safe, so the
singleton serialises behind a lock; 8 threads x N concurrent encodes must complete
with 0 errors and 0 segfaults.

Run in its own process: `python -m pytest tests/test_text_embedder_llamacpp.py -v -p no:anchorpy`
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest

# The vendored GGUF lives in the worktree's data/.gguf_cache (runtime artifact).
# Ensure TITAN_DATA_DIR resolves to a dir that holds it before importing the module.
os.environ.setdefault("TITAN_DATA_DIR", "data")

from titan_hcl.utils import text_embedder as te  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "fastembed_parity_bge_small.json"


@pytest.fixture(scope="module")
def encoder():
    gguf = te._gguf_path()
    if not os.path.exists(gguf):
        pytest.skip(f"GGUF not vendored at {gguf} — run P1 seed first")
    return te.get_text_embedder()


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_single_string_shape_and_norm(encoder):
    v = encoder.encode("sovereignty")
    assert v.shape == (te.EMBED_DIM,)
    assert v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-4


def test_batch_shape(encoder):
    vs = encoder.encode(["sovereignty", "freedom", "cold weather"])
    assert vs.shape == (3, te.EMBED_DIM)
    norms = np.linalg.norm(vs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_empty_input(encoder):
    vs = encoder.encode([])
    assert vs.shape == (0, te.EMBED_DIM)


def test_semantic_ordering(encoder):
    a, b, c = encoder.encode(
        ["I value my freedom", "sovereignty", "the weather is cold"])
    assert _cos(a, b) > _cos(a, c)


def test_self_test_passes():
    assert te.self_test() is True


def test_parity_vs_fastembed(encoder):
    """llama.cpp bge-small vectors must match the saved fastembed vectors
    (cosine >= 0.999) — same model, swappable runtime, no re-embed needed."""
    if not FIXTURE.exists():
        pytest.skip("parity fixture missing")
    data = json.loads(FIXTURE.read_text())
    corpus = data["corpus"]
    ref = np.asarray(data["vectors"], dtype=np.float32)
    got = encoder.encode(corpus)
    assert got.shape == ref.shape
    cosines = [_cos(got[i], ref[i]) for i in range(len(corpus))]
    worst = min(cosines)
    assert worst >= 0.999, f"parity below threshold: min cosine {worst:.5f} ({cosines})"


def test_concurrency_no_segfault(encoder):
    """8 threads x 15 encodes each = 120 concurrent encodes. llama.cpp embed is
    not thread-safe; the singleton's lock must serialise them with 0 errors."""
    n_threads, per_thread = 8, 15
    errors: list[Exception] = []
    results: list[int] = []
    lock = threading.Lock()
    probes = ["sovereignty", "freedom", "metabolic energy", "graph memory"]

    def worker(tid: int):
        try:
            for i in range(per_thread):
                v = encoder.encode(probes[(tid + i) % len(probes)])
                assert v.shape == (te.EMBED_DIM,)
                assert float(np.linalg.norm(v)) > 0.0
                with lock:
                    results.append(tid)
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"concurrency errors: {errors}"
    assert len(results) == n_threads * per_thread, "not all encodes completed"
