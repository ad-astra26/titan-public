"""Post-seal contract hook in BlockBuilder — Phase 2 D-P2-4 unit tests.

Covers the BlockBuilder._emit_maintain_bundle_events helper directly with
mock contracts + ContractStore + RuleEvaluator + send_queue. Validates:
  - Sealed conv-fork TX with user:<hash> tag → MAINTAIN_BUNDLE emitted
    with correctly extracted entity_id.
  - Non-matching TX (no user tag) does NOT emit.
  - Inactive contract is skipped.
  - Non-trigger contract is skipped.
  - Missing tx_sealed in triggers → skipped.
  - entity_id_from micro-DSL: tag_prefix / field / literal.
  - Multiple matches in one block emit multiple events.

PLAN_synthesis_engine_Phase2.md §2B.6.
"""
from __future__ import annotations

import queue
import time
import unittest

from titan_hcl.logic.timechain_v2 import (
    BlockBuilder,
    Contract,
    FORK_IDS,
    RuleEvaluator,
    Transaction,
)
from titan_hcl.bus import MAINTAIN_BUNDLE


# ─────────────────────────────────────────────────────────────────────────
# Test doubles
# ─────────────────────────────────────────────────────────────────────────

class _MockContractStore:
    """Minimal ContractStore stub — only get_all() is consumed by the hook."""

    def __init__(self, contracts: list[Contract]) -> None:
        self._contracts = list(contracts)

    def get_all(self) -> list[Contract]:
        return list(self._contracts)


class _MockBlock:
    """Stand-in for a sealed Block — only `block_hash` + `header.block_height`
    are read by the hook."""

    class _Header:
        block_height = 42

    def __init__(self, block_hash_hex: str = "00" * 32) -> None:
        self.block_hash = bytes.fromhex(block_hash_hex)
        self.header = self._Header()


def _conv_tx(tx_hash: str, tags: list[str], src: str = "chat",
             sig: float = 0.6, epoch: int = 100) -> Transaction:
    """Build a conversation-fork Transaction fixture."""
    tx = Transaction(
        tx_type="episodic", source=src, epoch_id=epoch,
        significance=sig, content={}, neuromod_snapshot={},
        tags=tags, timestamp=time.time(), fork_name="conversation",
    )
    tx.tx_hash = (tx_hash.encode() + b"\x00" * 32)[:32]
    return tx


def _user_bundle_contract() -> Contract:
    """Faithful copy of the actr_user_conversation_bundle.json shape
    (PLAN §2C.d) — except it ships pre-active so we don't need a signer
    in the unit test."""
    return Contract(
        contract_id="actr_user_conversation_bundle",
        version=1,
        contract_type="trigger",
        author="titan",
        description="Phase 2 standing pilot",
        rules=[{
            "op": "AND",
            "clauses": [
                {"op": "IF", "field": "event", "cmp": "EQ",
                 "value": "tx_sealed"},
                {"op": "IF", "field": "fork", "cmp": "EQ",
                 "value": "conversation"},
                {"op": "IF", "field": "tags", "cmp": "STARTSWITH_ANY",
                 "value": "user:"},
            ],
            "then": {
                "action": "maintain_bundle",
                "entity_class": "user",
                "entity_id_from": "tag_prefix:user:",
                "fork": "conversation",
                "ring_size": 50,
            },
        }],
        triggers=["tx_sealed"],
        fork_scope="conversation",
        status="active",
    )


# ─────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────

class TestPostSealHook(unittest.TestCase):

    def _make_builder_with_contract(
        self, contract: Contract,
    ) -> tuple[BlockBuilder, queue.Queue]:
        """Construct a BlockBuilder with only the post-seal hook wired
        (no timechain — we exercise _emit_maintain_bundle_events
        directly, sidestepping the seal/commit path)."""
        builder = BlockBuilder(timechain=None, config={})
        store = _MockContractStore([contract])
        evaluator = RuleEvaluator()
        builder.set_post_seal_hook(store, evaluator)
        sq: queue.Queue = queue.Queue()
        return builder, sq

    def _collect_emits(self, sq: queue.Queue) -> list[dict]:
        out = []
        while not sq.empty():
            out.append(sq.get_nowait())
        return [m for m in out if m.get("type") == MAINTAIN_BUNDLE]

    def test_matching_tx_emits_maintain_bundle(self) -> None:
        builder, sq = self._make_builder_with_contract(_user_bundle_contract())
        tx = _conv_tx("TX1", ["chat", "chat:abc", "user:hash_alpha"])
        block = _MockBlock("ab" * 32)
        builder._emit_maintain_bundle_events(
            [tx], "conversation", block, sq, "timechain")
        emits = self._collect_emits(sq)
        assert len(emits) == 1
        p = emits[0]["payload"]
        assert p["entity_class"] == "user"
        assert p["entity_id"] == "hash_alpha"
        assert p["fork"] == "conversation"
        assert p["contract_id"] == "actr_user_conversation_bundle"

    def test_non_matching_tx_does_not_emit(self) -> None:
        builder, sq = self._make_builder_with_contract(_user_bundle_contract())
        tx_no_user_tag = _conv_tx("TX2", ["chat", "chat:abc"])
        builder._emit_maintain_bundle_events(
            [tx_no_user_tag], "conversation", _MockBlock(), sq, "timechain")
        assert self._collect_emits(sq) == []

    def test_inactive_contract_skipped(self) -> None:
        c = _user_bundle_contract()
        c.status = "disabled"
        builder, sq = self._make_builder_with_contract(c)
        tx = _conv_tx("TX1", ["user:hash_alpha"])
        builder._emit_maintain_bundle_events(
            [tx], "conversation", _MockBlock(), sq, "timechain")
        assert self._collect_emits(sq) == []

    def test_wrong_contract_type_skipped(self) -> None:
        c = _user_bundle_contract()
        c.contract_type = "filter"  # not "trigger"
        builder, sq = self._make_builder_with_contract(c)
        tx = _conv_tx("TX1", ["user:hash_alpha"])
        builder._emit_maintain_bundle_events(
            [tx], "conversation", _MockBlock(), sq, "timechain")
        assert self._collect_emits(sq) == []

    def test_missing_tx_sealed_trigger_skipped(self) -> None:
        c = _user_bundle_contract()
        c.triggers = ["something_else"]
        builder, sq = self._make_builder_with_contract(c)
        tx = _conv_tx("TX1", ["user:hash_alpha"])
        builder._emit_maintain_bundle_events(
            [tx], "conversation", _MockBlock(), sq, "timechain")
        assert self._collect_emits(sq) == []

    def test_multiple_matching_txs_emit_multiple(self) -> None:
        builder, sq = self._make_builder_with_contract(_user_bundle_contract())
        txs = [
            _conv_tx("TX1", ["user:alice"]),
            _conv_tx("TX2", ["user:bob"]),
            _conv_tx("TX3", ["chat"]),     # not matching
            _conv_tx("TX4", ["user:carol"]),
        ]
        builder._emit_maintain_bundle_events(
            txs, "conversation", _MockBlock(), sq, "timechain")
        emits = self._collect_emits(sq)
        assert len(emits) == 3
        ids = sorted(e["payload"]["entity_id"] for e in emits)
        assert ids == ["alice", "bob", "carol"]

    def test_no_hook_attached_is_noop(self) -> None:
        """Pre-2D Orchestrator boot order: BlockBuilder constructed before
        ContractStore. Hook should silently no-op."""
        builder = BlockBuilder(timechain=None, config={})
        sq: queue.Queue = queue.Queue()
        tx = _conv_tx("TX1", ["user:alice"])
        builder._emit_maintain_bundle_events(
            [tx], "conversation", _MockBlock(), sq, "timechain")
        assert self._collect_emits(sq) == []

    def test_stats_counters_advance(self) -> None:
        builder, sq = self._make_builder_with_contract(_user_bundle_contract())
        for h in ("alice", "bob"):
            builder._emit_maintain_bundle_events(
                [_conv_tx(h, [f"user:{h}"])], "conversation",
                _MockBlock(), sq, "timechain")
        # 2 evals (one per TX × one contract), 2 hits, 2 emits.
        assert builder._post_seal_evals == 2
        assert builder._post_seal_hits == 2
        assert builder._maintain_bundle_emits == 2


class TestEntityIdMicroDSL(unittest.TestCase):
    """BlockBuilder._resolve_entity_id micro-DSL parsing."""

    def test_tag_prefix_extracts_suffix(self) -> None:
        ctx = {"tags": ["chat", "user:alice", "topic:foo"]}
        assert BlockBuilder._resolve_entity_id("tag_prefix:user:", ctx) == "alice"

    def test_tag_prefix_no_match_returns_empty(self) -> None:
        ctx = {"tags": ["chat", "topic:foo"]}
        assert BlockBuilder._resolve_entity_id("tag_prefix:user:", ctx) == ""

    def test_field_returns_ctx_field(self) -> None:
        ctx = {"source": "chat", "thought_type": "chat_turn"}
        assert BlockBuilder._resolve_entity_id("field:source", ctx) == "chat"

    def test_literal_returns_literal(self) -> None:
        assert BlockBuilder._resolve_entity_id("literal:globalbundle", {}) \
            == "globalbundle"

    def test_unknown_spec_treated_as_field_name(self) -> None:
        ctx = {"alpha": "BETA"}
        assert BlockBuilder._resolve_entity_id("alpha", ctx) == "BETA"

    def test_empty_spec_returns_empty(self) -> None:
        assert BlockBuilder._resolve_entity_id("", {"tags": []}) == ""


if __name__ == "__main__":
    unittest.main()
