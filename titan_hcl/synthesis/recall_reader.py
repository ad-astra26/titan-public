"""build_recall_reader — in-process read-only EngineRecall for the RECALL operator.

Synthesis Engine Phase 9 (INV-Syn-22). meta_reasoning's `RECALL` sub-modes must
resolve through the SC-op-backed `EngineRecall`, NOT the legacy per-worker bus
resolvers — and the resolution must be **in-process** in the host worker
(cognitive_worker), via a read-only `EngineRecall`, with **no sync `bus.request`
for recall data** (extends INV-8 / G18–G22; arch §14.2 "<100ms hot-path").

This factory mirrors the synthesis_worker EngineRecall construction but with
read-only substrate handles and activation served by the watermark-gated
`BridgeRecall` (cross-process read of `activation_snapshot.json` gated by
`synth_status.bin`). On a stale/missing watermark, `activation_lookup` returns
`{}` → composite scoring degrades to cosine-only (no crash). Any construction
failure returns None → the caller (during the parity-soak window only) falls
back to the legacy resolver; that fallback is removed at cascade-close.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = ["build_recall_reader"]


def build_recall_reader(
    *,
    data_dir: str,
    bridge_recall: Any,
    embedder: Optional[Callable[[str], list]] = None,
    faiss_reader: Optional[Any] = None,
    kuzu_reader: Optional[Any] = None,
    procedural_reader: Optional[Any] = None,
):
    """Construct a read-only `EngineRecall` for the cognitive_worker RECALL operator.

    Args:
        data_dir: the `data/` directory (holds `timechain/index.db`).
        bridge_recall: a `BridgeRecall` instance — supplies `activation_lookup`
            (watermark-gated; `{}` on stale → cosine-only fallback).
        embedder: `(text) -> list[float]`; None disables embedding-dependent
            granularities (turn/topic/session/archive) — concept/procedural/
            autobiographical still work.
        faiss_reader: read-only `.knn(fork, vec, k, min_similarity)` adapter over
            `memory_vectors`. None → SEARCH ops return no hits (cosine cold).
        kuzu_reader: read-only Kuzu handle for granularity="concept".
        procedural_reader: ProceduralSkillReader (or BridgeRecall wrapper) for
            granularity="procedural".

    Returns an `EngineRecall`, or None on any construction error (caller falls back).
    """
    try:
        from titan_hcl.synthesis.recall import EngineRecall
        from titan_hcl.logic.timechain_v2 import RuleEvaluator
    except Exception as exc:
        logger.warning(
            "[recall_reader] import failed: %s — RECALL operator unavailable", exc,
        )
        return None

    if bridge_recall is None:
        logger.warning(
            "[recall_reader] bridge_recall is None — cannot build read-only "
            "EngineRecall (no activation source); caller falls back",
        )
        return None

    # Read-only index.db (URI mode=ro — never writes; INV-Syn-22). THREAD-LOCAL:
    # one connection per accessing thread — a single shared connection is NOT safe
    # for concurrent execute() across threads (the agno PreHook runs FORK_READ recall
    # concurrently → sqlite3.InterfaceError). mode=ro → unlimited concurrent readers,
    # contention-free. See titan_hcl/synthesis/ro_sqlite.py.
    index_db_conn = None
    try:
        from titan_hcl.synthesis.ro_sqlite import ThreadLocalRoSqlite
        index_db_path = os.path.join(data_dir, "timechain", "index.db")
        if os.path.exists(index_db_path):
            index_db_conn = ThreadLocalRoSqlite(
                f"file:{index_db_path}?mode=ro", timeout=1.0)
        else:
            logger.info(
                "[recall_reader] index.db not present at %s — FORK_READ/"
                "CROSS_REF ops will be empty until it exists", index_db_path,
            )
    except Exception as exc:
        logger.warning("[recall_reader] index.db open failed: %s", exc)
        index_db_conn = None

    try:
        evaluator = RuleEvaluator(
            orchestrator=None,
            faiss_reader=faiss_reader,
            index_db=index_db_conn,
        )
        reader = EngineRecall(
            rule_evaluator=evaluator,
            activation_lookup=bridge_recall.activation_lookup,
            embedder=embedder,
            kuzu_reader=kuzu_reader,
            procedural_reader=procedural_reader,
        )
    except Exception as exc:
        logger.warning(
            "[recall_reader] EngineRecall construction failed: %s — caller "
            "falls back to legacy resolver", exc, exc_info=True,
        )
        return None

    logger.info(
        "[recall_reader] read-only EngineRecall built (INV-Syn-22) — "
        "index_db=%s faiss=%s kuzu=%s procedural=%s embedder=%s",
        "ro" if index_db_conn is not None else "missing",
        "attached" if faiss_reader is not None else "none",
        "attached" if kuzu_reader is not None else "none",
        "attached" if procedural_reader is not None else "none",
        "attached" if embedder is not None else "none",
    )
    return reader
