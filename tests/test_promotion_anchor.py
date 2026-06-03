"""promotion_anchor (RFP_synthesis_spine_reads_real_data Phase B) — the canonical
promotion-anchor payload + its DETERMINISTIC per-TX hash. The hash MUST equal
what timechain_worker computes from the same payload (so the chain TX, the
node's `timechain_tx_hash`, and the sidecar key are byte-identical)."""
from titan_hcl.synthesis.promotion_anchor import (
    build_promotion_tx,
    memory_type_for,
)
from titan_hcl.logic.timechain_v2 import Transaction


def test_memory_type_routing():
    # Canonical backfill rule: identity_* = declarative, else episodic.
    assert memory_type_for("identity_user123") == "declarative"
    assert memory_type_for("identity_") == "declarative"
    assert memory_type_for("chat_session_7") == "episodic"
    assert memory_type_for("") == "episodic"
    assert memory_type_for(None) == "episodic"


def _node(**kw):
    base = {
        "id": 42,
        "user_prompt": "I race karts on weekends",
        "agent_response": "Nice, karting is great!",
        "source_id": "identity_user123",
    }
    base.update(kw)
    return base


def test_build_promotion_tx_deterministic():
    n = _node()
    _p1, h1 = build_promotion_tx(n, now=1000.0)
    _p2, h2 = build_promotion_tx(n, now=1000.0)
    assert h1 == h2                 # same node + same now → identical hash
    assert len(h1) == 64           # full sha256 hex (not the 16-hex summary)


def test_build_promotion_tx_hash_matches_chain():
    # The whole point of A1: the writer's hash == the chain's hash.
    n = _node()
    payload, h = build_promotion_tx(n, now=1234.5)
    chain_h = Transaction.from_commit_payload(payload).compute_hash().hex()
    assert h == chain_h


def test_build_promotion_tx_now_changes_hash():
    n = _node()
    _, h1 = build_promotion_tx(n, now=1.0)
    _, h2 = build_promotion_tx(n, now=2.0)
    assert h1 != h2                 # timestamp is a hash input → distinct


def test_build_promotion_tx_routes_fork_by_memory_type():
    decl_p, _ = build_promotion_tx(_node(source_id="identity_x"), now=1.0)
    assert decl_p["fork"] == "declarative"
    assert decl_p["thought_type"] == "declarative"
    epi_p, _ = build_promotion_tx(_node(source_id="chat_session_7"), now=1.0)
    assert epi_p["fork"] == "episodic"
    assert epi_p["thought_type"] == "episodic"


def test_build_promotion_tx_carries_the_link():
    payload, _ = build_promotion_tx(_node(id=77), now=1.0)
    assert payload["content"]["node_id"] == 77
    assert payload["db_ref"] == "memory_nodes:77"
    assert payload["source"] == "memory_promotion"
    assert payload["content"]["event"] == "MEMORY_PROMOTED"
    # explicit deterministic hash inputs present
    assert payload["epoch_id"] == 0
    assert payload["timestamp"] == 1.0
