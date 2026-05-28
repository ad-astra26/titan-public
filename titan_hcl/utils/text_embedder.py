"""titan_hcl/utils/text_embedder.py — fleet-standard text embedder (fastembed).

Phase 13 §3J.1 — replaces the broken `sentence_transformers` path. That import
chain (`sentence_transformers → transformers → torchvision`) fails fleet-wide
on a torch/torchvision ABI mismatch (`operator torchvision::nms does not exist`),
so every consumer silently fell back to ZERO vectors — the Sage RL gatekeeper
was deciding on noise, sage/guardian Tier-2 similarity was disabled, and
meta_teacher semantic retrieval was dead.

`fastembed` (ONNX, `BAAI/bge-small-en-v1.5`, 384-d) has NO torch/torchvision/
transformers dependency, imports cleanly, and is already the working embedding
path in `memory_worker`. This module is the single shared entry point so all
consumers use the same model + benefit from the same per-process cache.

Drop-in for the `sentence_transformers` `.encode()` surface the 3 consumers
used: `encode(str | list[str], convert_to_numpy=True, convert_to_tensor=False)`.
Single `str` → 1-D vector; `list[str]` → 2-D (N, dim) — exactly matching
SentenceTransformer semantics so call-sites need no shape changes.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, List, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

_lock = threading.Lock()
_singleton: "FastembedEncoder | None" = None


class FastembedEncoder:
    """Lazy fastembed wrapper with a SentenceTransformer-compatible `.encode()`."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding  # ONNX — no torch
        self._model = TextEmbedding(model_name)
        self.model_name = model_name
        self.dim = EMBED_DIM

    def encode(
        self,
        sentences: Union[str, List[str]],
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        **_: Any,
    ):
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        # fastembed.embed → generator of (dim,) float32 arrays.
        vecs = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        if vecs.ndim == 1:  # defensive — should be (N, dim)
            vecs = vecs.reshape(len(texts), -1)
        if convert_to_tensor:
            # Lazy torch — only hosts that still want a tensor pay the import.
            # (Phase 13 §3J.3 will remove the remaining tensor call-sites.)
            import torch
            t = torch.from_numpy(vecs)
            return t[0] if single else t
        return vecs[0] if single else vecs


def get_text_embedder(model_name: str = DEFAULT_MODEL) -> FastembedEncoder:
    """Process-wide singleton encoder. Constructs on first call (lazy model load).

    Raises on construction failure — callers that previously swallowed
    `ImportError` into zero vectors should NOT do so anymore; a broken embedder
    must be loud, not silent (Phase 13 §3J.1).
    """
    global _singleton
    if _singleton is None:
        with _lock:
            if _singleton is None:
                _singleton = FastembedEncoder(model_name)
                logger.info(
                    "[TextEmbedder] fastembed ready (model=%s, dim=%d)",
                    model_name, EMBED_DIM)
    return _singleton


def self_test() -> bool:
    """Boot health-check — fail LOUD if the embedder can't produce a non-zero,
    semantically-meaningful vector. Returns True iff healthy. Logs ERROR on
    failure so the zero-vector regression can never go silent again."""
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
