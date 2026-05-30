"""PROFILING.md F11 — get_top_memories: cadence-gated bulk decay + nlargest.

get_top_memories previously applied _apply_decay to EVERY persistent node and
did a full sort on every call (n=max(pcount,1000) every 5 s from the memory
publisher ≈ 1.5-2% of a core). Anchor decay is negligible second-to-second, so
the bulk sweep is now gated to once per _DECAY_ALL_INTERVAL_S and selection uses
heapq.nlargest. Verifies ordering, exclusion, the cadence gate, and n>=len.
"""
import time

from titan_hcl.core.memory import TieredMemoryGraph, _DECAY_ALL_INTERVAL_S


def _mk_memory(n=30):
    m = object.__new__(TieredMemoryGraph)  # skip heavy __init__
    m._last_bulk_decay_ts = 0.0
    now = time.time()
    store = {
        f"k{i}": {"type": "MemoryNode", "status": "persistent", "id": f"id{i}",
                  "base_weight": 1.0 + i * 0.01, "anchor_bonus": 0.0,
                  "reinforcement_count": 0, "last_accessed": now}
        for i in range(n)
    }
    store["np"] = {"type": "MemoryNode", "status": "mempool", "base_weight": 9.0}
    store["x"] = {"type": "Other", "status": "persistent", "base_weight": 9.0}
    m._node_store = store
    return m


def test_ordering_and_excludes_nonpersistent():
    top = _mk_memory(30).get_top_memories(5)
    assert len(top) == 5
    w = [t.get("effective_weight", 0) for t in top]
    assert w == sorted(w, reverse=True)
    assert [t["id"] for t in top] == [f"id{i}" for i in (29, 28, 27, 26, 25)]
    assert all(t.get("status") == "persistent"
               and t.get("type") == "MemoryNode" for t in top)


def test_bulk_decay_is_cadence_gated(monkeypatch):
    m = _mk_memory(30)
    calls = {"n": 0}
    real = TieredMemoryGraph._apply_decay

    def counting(self, node):
        calls["n"] += 1
        return real(self, node)
    monkeypatch.setattr(TieredMemoryGraph, "_apply_decay", counting)

    m.get_top_memories(5)               # first call → full sweep
    assert calls["n"] == 30
    assert m._last_bulk_decay_ts > 0.0

    calls["n"] = 0
    m.get_top_memories(5)               # within interval → NO re-decay
    assert calls["n"] == 0

    calls["n"] = 0
    m._last_bulk_decay_ts -= (_DECAY_ALL_INTERVAL_S + 1)
    m.get_top_memories(5)               # interval elapsed → sweep again
    assert calls["n"] == 30


def test_n_larger_than_store_returns_all_sorted():
    top = _mk_memory(5).get_top_memories(1000)
    assert len(top) == 5
    w = [t.get("effective_weight", 0) for t in top]
    assert w == sorted(w, reverse=True)
