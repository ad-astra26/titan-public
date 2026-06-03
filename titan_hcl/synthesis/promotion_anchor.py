"""Promotion-anchor payload + deterministic tx_hash (RFP_synthesis_spine_reads_real_data Phase B).

At meditation promotion each promoted ``memory_node`` is anchored to its ACT-R
``memory_type`` fork. This module builds that canonical ``TIMECHAIN_COMMIT``
payload and computes its deterministic per-TX hash the SAME way
``timechain_worker`` will (``Transaction.from_commit_payload(...).compute_hash()``)
— so the emitted chain TX, the ``timechain_tx_hash`` stamped on the node, and the
content-sidecar key are all byte-identical.

The per-TX hash is ``sha256`` over the canonical ``{t,s,e,g,c,a,ts}``
(``logic/timechain_v2.py:79-88``). Two of those inputs would otherwise be
non-deterministic at the writer: ``epoch_id`` (defaults 0 if absent) and
``timestamp`` (defaults ``time.time()`` AT COMMIT if absent). We therefore set
BOTH explicitly in the payload so the writer can recompute the exact hash the
chain will assign — pass the one ``now`` you also store as the sidecar ``ts``.

Pure + side-effect-free (the only import is the canonical hash function) →
unit-testable in isolation.
"""
from __future__ import annotations

import hashlib
from typing import Optional


def memory_type_for(source_id: Optional[str]) -> str:
    """ACT-R memory-type routing for a promoted node.

    Single source of truth = the canonical backfill rule
    (``scripts/migrate_memory_type_backfill.py``):
      * ``source_id LIKE 'identity_%'`` → ``declarative`` (identity / facts)
      * everything else                → ``episodic``    (lived chat encounter)
    ``procedural`` / ``meta`` are reserved for those sources (none promote via
    this path today; the routing mechanism supports them when they do).
    """
    s = source_id or ""
    return "declarative" if s.startswith("identity_") else "episodic"


def build_promotion_tx(node: dict, *, now: float) -> tuple[dict, str]:
    """Return ``(timechain_commit_payload, tx_hash_hex)`` for a promoted node.

    ``now`` MUST be the timestamp written into the payload (so the hash is
    deterministic + recomputable); the caller passes a single ``time.time()`` and
    reuses it as the sidecar ``ts``.
    """
    node_id = node.get("id")
    prompt = node.get("user_prompt", "") or ""
    response = node.get("agent_response", "") or ""
    mem_type = memory_type_for(node.get("source_id"))
    payload = {
        "fork": mem_type,
        "thought_type": mem_type,
        "source": "memory_promotion",
        "epoch_id": 0,            # explicit → deterministic hash input
        "timestamp": now,         # explicit → deterministic hash input
        "content": {
            "event": "MEMORY_PROMOTED",
            "node_id": node_id,
            "prompt_hash": hashlib.sha256(prompt[:100].encode()).hexdigest()[:16],
            "response_hash": hashlib.sha256(response[:100].encode()).hexdigest()[:16],
        },
        "significance": 0.5,
        "novelty": 0.6,
        "coherence": 0.5,
        "tags": ["memory_node", "promoted", "persistent", mem_type],
        "db_ref": f"memory_nodes:{node_id}",
        "neuromods": {},
        "chi_available": 0.5,
        "attention": 0.5,
        "i_confidence": 0.5,
        "chi_coherence": 0.3,
    }
    # Compute the hash via the SAME path the chain uses → byte-identical key.
    from titan_hcl.logic.timechain_v2 import Transaction
    tx_hash = Transaction.from_commit_payload(payload).compute_hash().hex()
    return payload, tx_hash


__all__ = ["memory_type_for", "build_promotion_tx"]
