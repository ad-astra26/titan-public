"""Phase D backfill + shared promotion-anchor tests
(RFP_synthesis_spine_reads_real_data §7.D / D4).

Covers `memory_worker._anchor_promoted_node` + `_backfill_thought_sidecar`:
- the shared single anchor mechanic emits a TIMECHAIN_COMMIT, stamps the per-TX
  hash on the node, and writes the real thought to the sidecar
- the per-TX hash is deterministic in `now` (so backfill via `created_at` is
  idempotent and byte-identical to the live path's hash)
- backfill links ONLY persistent nodes with `timechain_tx_hash IS NULL` and real
  content; re-running is a no-op (self-watermarking)
"""
from __future__ import annotations

import tempfile
import threading

from titan_hcl import bus
from titan_hcl.modules import memory_worker
from titan_hcl.synthesis.promotion_anchor import build_promotion_tx
from titan_hcl.synthesis.thought_sidecar import ThoughtSidecarReader


# ── Fakes ───────────────────────────────────────────────────────────


class _FakeQueue:
    def __init__(self):
        self.items = []

    def put_nowait(self, item):
        self.items.append(item)


class _FakeDuckDB:
    def __init__(self, rows):
        self._rows = rows
        self.update_calls = []

    def get_nodes_by_status(self, status):
        return [dict(r) for r in self._rows if r.get("status") == status]

    def update_node(self, node_id, **fields):
        self.update_calls.append((node_id, fields))
        for r in self._rows:
            if r.get("id") == node_id:
                r.update(fields)


class _FakeMemory:
    def __init__(self, rows):
        self._duckdb = _FakeDuckDB(rows)
        self._node_store = {}

    def set_timechain_tx_hash(self, node_id, tx_hash):
        self._duckdb.update_node(node_id, timechain_tx_hash=tx_hash)


class _FakeCtx:
    def __init__(self, memory):
        self.memory = memory
        self.send_queue = _FakeQueue()
        self.name = "memory"
        self.write_lock = threading.RLock()


def _get_sidecar(tmp):
    return memory_worker._get_thought_sidecar(tmp)


# ── _anchor_promoted_node — the shared mechanic ─────────────────────


def test_anchor_emits_stamps_and_sidecars():
    with tempfile.TemporaryDirectory() as tmp:
        node = {
            "id": 42, "user_prompt": "I race go-karts on Saturdays",
            "agent_response": "At the Brno circuit, nice!",
            "source_id": "identity_jirka", "created_at": 1000.0,
        }
        sidecar = _get_sidecar(tmp)
        ctx = _FakeCtx(_FakeMemory([]))
        txh = memory_worker._anchor_promoted_node(
            node, now=1000.0, sidecar=sidecar, ctx=ctx)

        assert txh is not None
        # byte-identical to the canonical builder (no drift)
        _, expected = build_promotion_tx(node, now=1000.0)
        assert txh == expected

        # emitted exactly one TIMECHAIN_COMMIT pointer on the declarative fork
        # (source_id identity_* → declarative, ACT-R routing)
        assert len(ctx.send_queue.items) == 1
        msg = ctx.send_queue.items[0]
        assert msg["type"] == bus.TIMECHAIN_COMMIT
        assert msg["dst"] == "timechain"
        assert msg["payload"]["fork"] == "declarative"

        # stamped the link on the node
        assert ctx.memory._duckdb.update_calls[-1][0] == 42
        assert ctx.memory._duckdb.update_calls[-1][1]["timechain_tx_hash"] == txh

        # wrote the real thought to the sidecar, keyed by that hash
        row = ThoughtSidecarReader(tmp).get(txh)
        assert row is not None
        assert "go-karts" in row["user_prompt"]
        assert row["fork"] == "declarative"


def test_anchor_hash_deterministic_in_now():
    node = {"id": 7, "user_prompt": "hello", "agent_response": "hi",
            "source_id": "chat_user", "created_at": 500.0}
    _, h1 = build_promotion_tx(node, now=500.0)
    _, h2 = build_promotion_tx(node, now=500.0)
    _, h3 = build_promotion_tx(node, now=999.0)
    assert h1 == h2          # same now → same hash (idempotent backfill)
    assert h1 != h3          # different now → different hash
    # non-identity source → episodic fork
    payload, _ = build_promotion_tx(node, now=500.0)
    assert payload["fork"] == "episodic"


# ── _backfill_thought_sidecar — D4 ──────────────────────────────────


def test_backfill_links_only_null_persistent_and_is_idempotent(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("TITAN_DATA_DIR", tmp)
        # reset the memoized sidecar so it opens under tmp
        memory_worker._THOUGHT_SIDECAR.clear()
        rows = [
            # A: persistent, unlinked, has content → SHOULD link
            {"id": 1, "status": "persistent", "user_prompt": "I love karts",
             "agent_response": "", "source_id": "identity_jirka",
             "created_at": 1000.0, "timechain_tx_hash": None},
            # B: persistent, ALREADY linked → skip
            {"id": 2, "status": "persistent", "user_prompt": "old thought",
             "agent_response": "", "source_id": "chat_u",
             "created_at": 1100.0, "timechain_tx_hash": "deadbeef"},
            # C: mempool → not persistent → skip
            {"id": 3, "status": "mempool", "user_prompt": "transient",
             "agent_response": "", "source_id": "chat_u",
             "created_at": 1200.0, "timechain_tx_hash": None},
            # D: persistent, unlinked, NO content → skip (nothing to anchor)
            {"id": 4, "status": "persistent", "user_prompt": "",
             "agent_response": "", "source_id": "chat_u",
             "created_at": 1300.0, "timechain_tx_hash": None},
        ]
        ctx = _FakeCtx(_FakeMemory(rows))

        memory_worker._backfill_thought_sidecar(ctx, settle_s=0.0)

        linked_ids = [c[0] for c in ctx.memory._duckdb.update_calls]
        assert linked_ids == [1]                 # only A
        # A's hash is created_at-deterministic
        _, expected_a = build_promotion_tx(
            {"id": 1, "user_prompt": "I love karts", "agent_response": "",
             "source_id": "identity_jirka"}, now=1000.0)
        assert ctx.memory._duckdb.update_calls[0][1]["timechain_tx_hash"] == expected_a
        # one pointer emitted (for A)
        assert len(ctx.send_queue.items) == 1
        # A is in the sidecar
        assert ThoughtSidecarReader(tmp).get(expected_a) is not None

        # Re-run → A now has a hash → self-watermarked → no new work.
        memory_worker._backfill_thought_sidecar(ctx, settle_s=0.0)
        assert [c[0] for c in ctx.memory._duckdb.update_calls] == [1]   # unchanged
        assert len(ctx.send_queue.items) == 1                            # no new emit
        memory_worker._THOUGHT_SIDECAR.clear()


def test_backfill_empty_when_no_unlinked(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("TITAN_DATA_DIR", tmp)
        memory_worker._THOUGHT_SIDECAR.clear()
        rows = [
            {"id": 1, "status": "persistent", "user_prompt": "x",
             "agent_response": "", "source_id": "chat_u",
             "created_at": 1.0, "timechain_tx_hash": "already"},
        ]
        ctx = _FakeCtx(_FakeMemory(rows))
        memory_worker._backfill_thought_sidecar(ctx, settle_s=0.0)
        assert ctx.memory._duckdb.update_calls == []
        assert ctx.send_queue.items == []
        memory_worker._THOUGHT_SIDECAR.clear()
