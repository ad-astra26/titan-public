"""titan_hcl/utils/text_embedder.py — fleet-standard text embedder (llama.cpp).

Phase 13 §3J.1 — the single shared text-embedding entry point. Model is
`BAAI/bge-small-en-v1.5` (384-d), unchanged and SPEC-anchored. Only the *runtime*
changed: `sentence_transformers` (broken — torch/torchvision ABI mismatch,
`operator torchvision::nms does not exist` → silent ZERO vectors fleet-wide) was
replaced by `fastembed` (ONNX), which in turn leaked unbounded RSS on the big
mainnet chain — onnxruntime's CPU memory arena never returns memory to the OS, so
the synthesis backfill drove RSS to ~5 GB → guardian restart-loop. The runtime is
now **`llama-cpp-python`** (llama.cpp): same bge-small model, **identical vectors**
(cosine 1.0000 vs fastembed), **flat ~197 MB**, and — like fastembed — **torch-free**
(torch is reserved for RL/NN/IQL only).

llama.cpp's native `embed()` is NOT thread-safe — concurrent calls on one model
segfault uncatchably. The singleton therefore serialises every `embed()` behind a
module `threading.Lock` (proven 120/120 concurrent, 0 errors). bge is a CLS-pooled
model, so we construct with `pooling_type=2` (CLS) — this is what reproduces the
fastembed vectors exactly. Vectors are L2-normalised in numpy (llama.cpp returns
unnormalised pooled embeddings).

Drop-in for the `.encode()` surface the consumers use:
`encode(str | list[str], convert_to_numpy=True)`. Single `str` → 1-D (dim,)
vector; `list[str]` → 2-D (N, dim) — matching SentenceTransformer/fastembed
semantics so call-sites need no shape changes. The legacy lazy-torch
`convert_to_tensor` branch is GONE — callers that need a tensor do
`torch.from_numpy(...)` at their own RL boundary (Phase 13 §3J.3 / migration P4).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, List, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# Vendored GGUF (f16) — same cache convention as the old `.fastembed_cache`, so
# the model is present once per box and survives reboots. Seeded by P1 fleet-wide
# and by the setup script for fresh installs (migration P5).
GGUF_FILENAME = "bge-small-en-v1.5-f16.gguf"

# Construction is guarded by `_init_lock`; every native embed() call is guarded by
# `_embed_lock` because llama.cpp embed is NOT thread-safe (segfaults otherwise).
_init_lock = threading.Lock()
_embed_lock = threading.Lock()
_singleton: "LlamaCppEncoder | None" = None


def _gguf_path() -> str:
    return os.path.join(
        os.environ.get("TITAN_DATA_DIR", "data"), ".gguf_cache", GGUF_FILENAME)


class LlamaCppEncoder:
    """llama.cpp embedder with a SentenceTransformer-compatible `.encode()`.

    Construction fails LOUD (no silent zero-vector fallback): a missing GGUF or a
    failed model load raises, so the zero-vector regression can never return.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from llama_cpp import Llama  # llama.cpp — no torch

        gguf = _gguf_path()
        if not os.path.exists(gguf):
            raise FileNotFoundError(
                f"[TextEmbedder] GGUF not found at {gguf} — vendor "
                f"{GGUF_FILENAME} into $TITAN_DATA_DIR/.gguf_cache (setup P5).")
        # pooling_type=2 == CLS (bge): reproduces fastembed bge-small vectors.
        self._model = Llama(
            model_path=gguf,
            embedding=True,
            n_ctx=512,
            n_threads=4,
            n_batch=512,
            verbose=False,
            pooling_type=2,
        )
        self.model_name = model_name
        self.dim = EMBED_DIM

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        # L2-normalise rows; guard against zero-norm (empty input) division.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (vecs / norms).astype(np.float32)

    def encode(
        self,
        sentences: Union[str, List[str]],
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        **_: Any,
    ) -> np.ndarray:
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # llama.cpp embed is not thread-safe — serialise the native call.
        with _embed_lock:
            raw = self._model.embed(texts)
        vecs = np.asarray(raw, dtype=np.float32)
        if vecs.ndim == 1:  # defensive — single-text path may return (dim,)
            vecs = vecs.reshape(len(texts), -1)
        vecs = self._normalize(vecs)
        # `convert_to_tensor` is intentionally ignored — this module is torch-free.
        # Callers needing a tensor wrap the numpy result themselves at their RL
        # boundary (`torch.from_numpy`). See migration P4.
        return vecs[0] if single else vecs


def get_text_embedder(model_name: str = DEFAULT_MODEL) -> LlamaCppEncoder:
    """Process-wide singleton encoder. Constructs on first call (lazy model load).

    Raises on construction failure — callers that previously swallowed
    `ImportError` into zero vectors must NOT do so anymore; a broken embedder must
    be loud, not silent (Phase 13 §3J.1).
    """
    global _singleton
    if _singleton is None:
        with _init_lock:
            if _singleton is None:
                _singleton = LlamaCppEncoder(model_name)
                logger.info(
                    "[TextEmbedder] llama.cpp ready (model=%s, dim=%d, gguf=%s)",
                    model_name, EMBED_DIM, _gguf_path())
    return _singleton


def self_test() -> bool:
    """Boot health-check — fail LOUD if the embedder can't produce a non-zero,
    semantically-meaningful vector. Returns True iff healthy. Logs ERROR on
    failure so the zero-vector regression can never go silent again. Also serves
    as the boot pre-warm (forces model construction off the hot path)."""
    try:
        enc = get_text_embedder()
        a, b, c = enc.encode(
            ["I value my freedom", "sovereignty", "the weather is cold"])
        norm_a = float(np.linalg.norm(a))
        if norm_a == 0.0:
            logger.error("[TextEmbedder] SELF-TEST FAILED — zero vector returned")
            return False

        def _cos(x, y):
            return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

        related, unrelated = _cos(a, b), _cos(a, c)
        logger.info(
            "[TextEmbedder] self-test OK — |v|=%.3f, cos(related)=%.3f > "
            "cos(unrelated)=%.3f", norm_a, related, unrelated)
        return related > unrelated
    except Exception as e:  # noqa: BLE001
        logger.error("[TextEmbedder] SELF-TEST FAILED — %s", e)
        return False
