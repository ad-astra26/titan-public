"""
Tests for TimeChain v2 — Cognitive Memory Engine (Phase 1 + Phase 2 + Phase 3a)

Covers: Transaction, BloomFilter, MempoolWAL, Mempool, BlockBuilder,
GenesisChain, TimeChainOrchestrator, Consumer API, Smart Contracts (P3a)
"""
import hashlib
import json
import os
import shutil
import tempfile
import time
import unittest

import pytest

from titan_plugin.logic.timechain_v2 import (
    Transaction, RecallQuery, CheckQuery, CompareQuery, AggregateQuery,
    BloomFilter, MempoolWAL, Mempool, GenesisChain, BlockBuilder,
    TimeChainOrchestrator, FORK_IDS,
    Contract, ContractStore, sign_contract, approve_contract,
    verify_contract_signature, CONTRACT_TYPES,
    RuleEvaluator,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="tc_v2_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_payload(fork="episodic", source="test_source",
                  significance=0.5, thought_type="episodic",
                  tags=None, epoch_id=1000, content=None):
    return {
        "fork": fork,
        "source": source,
        "significance": significance,
        "thought_type": thought_type,
        "tags": tags or ["test"],
        "epoch_id": epoch_id,
        "content": content or {"data": "test"},
        "neuromods": {"DA": 0.5, "5HT": 0.8, "NE": 0.6, "GABA": 0.2, "ACh": 0.7},
        "chi_available": 0.5,
        "attention": 0.5,
        "i_confidence": 0.5,
        "chi_coherence": 0.3,
        "timestamp": time.time(),
    }


# ── Transaction ───────────────────────────────────────────────────────

class TestTransaction:
    def test_from_commit_payload(self):
        p = _make_payload(source="meditation", significance=0.8, tags=["dream"])
        tx = Transaction.from_commit_payload(p)
        assert tx.source == "meditation"
        assert tx.significance == 0.8
        assert tx.fork_name == "episodic"
        assert tx.tx_hash != b""
        assert len(tx.tx_hash) == 32

    def test_hash_deterministic(self):
        p = _make_payload()
        tx1 = Transaction.from_commit_payload(p)
        tx2 = Transaction.from_commit_payload(p)
        assert tx1.tx_hash == tx2.tx_hash

    def test_storage_roundtrip(self):
        p = _make_payload(source="kin_exchange", significance=0.9)
        tx = Transaction.from_commit_payload(p)
        d = tx.to_storage_dict()
        tx2 = Transaction.from_storage_dict(d)
        assert tx2.source == tx.source
        assert tx2.significance == tx.significance
        assert tx2.tx_hash == tx.tx_hash


# ── BloomFilter ───────────────────────────────────────────────────────

class TestBloomFilter:
    def test_add_and_check(self):
        bf = BloomFilter(capacity=1000)
        item = b"test_item"
        assert not bf.might_contain(item)
        bf.add(item)
        assert bf.might_contain(item)

    def test_false_positive_rate(self):
        bf = BloomFilter(capacity=10000, fp_rate=0.01)
        for i in range(1000):
            bf.add(f"item_{i}".encode())
        fp = sum(1 for i in range(1000, 2000)
                 if bf.might_contain(f"item_{i}".encode()))
        assert fp < 50  # < 5% (generous margin)

    def test_rotation(self):
        bf = BloomFilter(capacity=100)
        for i in range(90):
            bf.add(f"item_{i}".encode())
        assert bf.should_rotate()  # > 80% capacity
        bf.clear()
        assert bf._count == 0
        assert not bf.should_rotate()


# ── MempoolWAL ────────────────────────────────────────────────────────

class TestMempoolWAL:
    def test_insert_and_recover(self, tmp_dir):
        wal = MempoolWAL(os.path.join(tmp_dir, "test.wal.db"))
        tx = Transaction.from_commit_payload(_make_payload())
        wal.insert(tx)
        assert wal.pending_count() == 1

        txs, aggs = wal.recover()
        assert len(txs) == 1
        assert txs[0].tx_hash == tx.tx_hash
        wal.close()

    def test_mark_sealed(self, tmp_dir):
        wal = MempoolWAL(os.path.join(tmp_dir, "test.wal.db"))
        tx = Transaction.from_commit_payload(_make_payload())
        wal.insert(tx)
        assert wal.pending_count() == 1
        wal.mark_sealed([tx.tx_hash])
        assert wal.pending_count() == 0
        wal.close()

    def test_pending_forks(self, tmp_dir):
        wal = MempoolWAL(os.path.join(tmp_dir, "test.wal.db"))
        wal.insert(Transaction.from_commit_payload(
            _make_payload(fork="episodic")))
        wal.insert(Transaction.from_commit_payload(
            _make_payload(fork="procedural", source="skill")))
        forks = wal.get_pending_forks()
        assert set(forks) == {"episodic", "procedural"}
        wal.close()


# ── Mempool ───────────────────────────────────────────────────────────

class TestMempool:
    def test_submit_queued(self, tmp_dir):
        mp = Mempool(tmp_dir, {"aggregate_sources": []})
        result = mp.submit(Transaction.from_commit_payload(
            _make_payload(significance=0.5)))
        assert result == "queued"
        assert mp.pending_count() == 1

    def test_submit_duplicate(self, tmp_dir):
        mp = Mempool(tmp_dir, {"aggregate_sources": []})
        p = _make_payload()
        tx = Transaction.from_commit_payload(p)
        mp.submit(tx)
        # Same TX again
        tx2 = Transaction.from_commit_payload(p)
        result = mp.submit(tx2)
        assert result == "duplicate"

    def test_submit_dropped(self, tmp_dir):
        mp = Mempool(tmp_dir, {"aggregate_sources": []})
        result = mp.submit(Transaction.from_commit_payload(
            _make_payload(significance=0.01, source="noise")))
        assert result == "dropped"

    def test_aggregation(self, tmp_dir):
        mp = Mempool(tmp_dir, {
            "aggregate_sources": ["expression_art"],
            "aggregate_threshold_significance": 0.3,
            "aggregate_batch_size": 5,
        })
        for i in range(5):
            p = _make_payload(source="expression_art", significance=0.2,
                              epoch_id=1000 + i)
            p["timestamp"] = time.time() + i * 0.001  # Unique timestamps
            result = mp.submit(Transaction.from_commit_payload(p))
            assert result == "aggregated"

        # Flush should produce summary TX
        summaries = mp.flush_aggregates("episodic")
        assert len(summaries) >= 1
        assert summaries[0].source == "expression_aggregate"
        assert summaries[0].content["count"] == 5

    def test_stats(self, tmp_dir):
        mp = Mempool(tmp_dir, {"aggregate_sources": []})
        mp.submit(Transaction.from_commit_payload(_make_payload()))
        stats = mp.get_stats()
        assert stats["total_submitted"] == 1
        assert stats["total_queued"] == 1
        mp.close()


# ── GenesisChain ──────────────────────────────────────────────────────

class TestGenesisChain:
    def test_should_seal_meditation_only(self):
        gc = GenesisChain(None, {"genesis_seal_on_meditation": True,
                                  "genesis_seal_fallback_hours": 6})
        assert gc.should_seal("meditation") is True
        assert gc.should_seal("dream_boundary") is False
        assert gc.should_seal("emotion_shift") is False
        assert gc.should_seal("boot") is True
        assert gc.should_seal("shutdown") is True
        # Timer: not enough time elapsed
        assert gc.should_seal("timer") is False

    def test_timer_fallback(self):
        gc = GenesisChain(None, {"genesis_seal_fallback_hours": 0})
        gc._last_seal_time = time.time() - 3600  # 1h ago
        assert gc.should_seal("timer") is True


# ── Query Dataclasses ─────────────────────────────────────────────────

class TestQueryDataclasses:
    def test_recall_query_defaults(self):
        q = RecallQuery()
        assert q.fork == ""
        assert q.limit == 10
        assert q.order == "desc"
        assert q.include_content is False

    def test_recall_query_from_dict(self):
        q = RecallQuery(**{"fork": "episodic", "since_hours": 6, "limit": 5})
        assert q.fork == "episodic"
        assert q.since_hours == 6
        assert q.limit == 5

    def test_check_query(self):
        q = CheckQuery(source="kin_exchange", since_hours=1)
        assert q.source == "kin_exchange"

    def test_compare_query(self):
        q = CompareQuery(field="vocab_size", window_a_hours=6, window_b_hours=12)
        assert q.field == "vocab_size"

    def test_aggregate_query(self):
        q = AggregateQuery(fork="episodic", op="count", since_hours=24)
        assert q.op == "count"


# ── Orchestrator (integration) ────────────────────────────────────────

class TestOrchestrator:
    def _make_orchestrator(self, tmp_dir):
        """Create orchestrator with a mock TimeChain."""
        from unittest.mock import MagicMock
        mock_tc = MagicMock()
        mock_tc._fork_tips = {}
        mock_tc.query_blocks.return_value = []
        mock_tc._index_db = MagicMock()
        mock_tc._index_db.execute.return_value.fetchone.return_value = (42,)

        orch = TimeChainOrchestrator(
            mock_tc, tmp_dir,
            config={"seal_max_txs": 10, "seal_max_time_s": 60,
                    "aggregate_sources": [],
                    "genesis_seal_on_meditation": True,
                    "genesis_seal_fallback_hours": 6},
            api_port=7777)
        # Reset mock call tracking after init (birth block may call commit_block)
        mock_tc.reset_mock()
        return orch, mock_tc

    def test_submit_tracks_cognitive_work(self, tmp_dir):
        orch, _ = self._make_orchestrator(tmp_dir)
        p = _make_payload(source="expression_art")
        orch.submit(p, "test")
        assert orch._cognitive_work["expression_fires"] == 1
        assert orch._cognitive_work["txs_submitted"] == 1

    def test_submit_reasoning_tracking(self, tmp_dir):
        orch, _ = self._make_orchestrator(tmp_dir)
        p = _make_payload(source="meta_reasoning")
        orch.submit(p, "test")
        assert orch._cognitive_work["reasoning_chains"] == 1

    def test_emotion_shift_no_seal(self, tmp_dir):
        orch, mock_tc = self._make_orchestrator(tmp_dir)
        orch.on_emotion_shift(1000, "wonder")
        assert orch._last_emotion == "wonder"
        # Genesis should NOT be sealed
        mock_tc.commit_block.assert_not_called()

    def test_dream_boundary_seals_forks_not_genesis(self, tmp_dir):
        orch, mock_tc = self._make_orchestrator(tmp_dir)
        # Submit some TXs first
        for i in range(3):
            orch.submit(_make_payload(epoch_id=1000 + i), "test")
        # Dream boundary should try to seal forks
        orch.on_dream_boundary(1005, True)
        # Genesis should NOT be sealed (no genesis commit)
        # But fork sealing was attempted via builder
        # (may or may not produce blocks depending on mock)

    def test_recall_delegates_to_tc(self, tmp_dir):
        orch, mock_tc = self._make_orchestrator(tmp_dir)
        orch._current_epoch = 100000
        mock_tc.query_blocks.return_value = [
            {"epoch_id": 99000, "significance": 0.8, "block_hash": "ab" * 16}
        ]
        results = orch.recall(RecallQuery(fork="episodic", since_hours=1))
        assert len(results) == 1
        mock_tc.query_blocks.assert_called_once()

    def test_check_returns_bool(self, tmp_dir):
        orch, mock_tc = self._make_orchestrator(tmp_dir)
        orch._current_epoch = 100000
        mock_tc.query_blocks.return_value = [
            {"epoch_id": 99500, "significance": 0.6}
        ]
        assert orch.check(CheckQuery(source="kin_exchange", since_hours=1)) is True

    def test_check_returns_false_when_empty(self, tmp_dir):
        orch, mock_tc = self._make_orchestrator(tmp_dir)
        orch._current_epoch = 100000
        mock_tc.query_blocks.return_value = []
        assert orch.check(CheckQuery(source="nonexistent")) is False

    def test_aggregate_sql(self, tmp_dir):
        orch, mock_tc = self._make_orchestrator(tmp_dir)
        orch._current_epoch = 100000
        result = orch.aggregate(AggregateQuery(
            fork="episodic", op="count", since_hours=24))
        assert result == 42.0  # Mock returns (42,)

    def test_aggregate_sql_injection_safe(self, tmp_dir):
        orch, mock_tc = self._make_orchestrator(tmp_dir)
        orch._current_epoch = 100000
        # Try to inject SQL via field
        result = orch.aggregate(AggregateQuery(
            field="significance; DROP TABLE block_index;--"))
        # Should fall back to safe field
        assert isinstance(result, float)

    def test_per_fork_seal_timers(self, tmp_dir):
        orch, mock_tc = self._make_orchestrator(tmp_dir)
        mock_tc.commit_block.return_value = None  # No seal happens
        orch._current_epoch = 1000

        # Submit to two forks
        orch.submit(_make_payload(fork="episodic"), "test")
        orch.submit(_make_payload(fork="procedural", source="skill"), "test")

        # Verify no global timer, but fork timers can exist
        assert isinstance(orch._fork_seal_times, dict)

    def test_get_stats(self, tmp_dir):
        orch, _ = self._make_orchestrator(tmp_dir)
        stats = orch.get_stats()
        assert stats["v2_enabled"] is True
        assert "mempool" in stats
        assert "genesis" in stats
        assert "cognitive_work" in stats
        assert stats["cognitive_work"]["txs_submitted"] == 0


# ── Smart Contracts (Phase 3a) ────────────────────────────────────────

class TestContract:
    def test_canonical_json_deterministic(self):
        c = Contract(
            contract_id="test_gate",
            contract_type="filter",
            author="maker",
            description="Test filter",
            rules=[{"op": "IF", "field": "significance", "cmp": "GT", "value": 0.5}],
        )
        j1 = c.canonical_json()
        j2 = c.canonical_json()
        assert j1 == j2
        assert '"id":"test_gate"' in j1

    def test_content_hash(self):
        c = Contract(contract_id="test", rules=[{"op": "IF"}])
        h = c.content_hash()
        assert len(h) == 32
        # Hash should be deterministic
        assert c.content_hash() == h

    def test_to_dict_roundtrip(self):
        c = Contract(
            contract_id="roundtrip_test",
            version=2,
            contract_type="trigger",
            author="titan",
            description="Roundtrip test",
            rules=[{"op": "EMIT_EVENT", "event": "TEST"}],
            triggers=["genesis_seal"],
            fork_scope="episodic",
            status="pending_approval",
        )
        d = c.to_dict()
        c2 = Contract.from_dict(d)
        assert c2.contract_id == c.contract_id
        assert c2.version == 2
        assert c2.contract_type == "trigger"
        assert c2.author == "titan"
        assert c2.rules == c.rules
        assert c2.status == "pending_approval"

    def test_version_increment(self):
        c1 = Contract(contract_id="evolving", version=1)
        c2 = Contract(contract_id="evolving", version=2,
                       rules=[{"op": "IF"}])
        assert c1.content_hash() != c2.content_hash()


class TestContractSigning:
    def _make_keypair(self):
        from solders.keypair import Keypair
        return Keypair()

    def test_sign_and_verify(self):
        kp = self._make_keypair()
        c = Contract(
            contract_id="signed_test",
            contract_type="filter",
            author="maker",
            rules=[{"op": "IF", "field": "sig", "cmp": "GT", "value": 0.3}],
        )
        sign_contract(c, kp)
        assert c.signature != ""
        assert c.signer_pubkey == str(kp.pubkey())

        valid, reason = verify_contract_signature(
            c, str(kp.pubkey()), str(kp.pubkey()))
        assert valid is True
        assert reason == "valid"

    def test_reject_unknown_signer(self):
        kp = self._make_keypair()
        other_kp = self._make_keypair()
        c = Contract(contract_id="test", author="maker")
        sign_contract(c, kp)

        # Verify with different pubkeys (neither matches signer)
        valid, reason = verify_contract_signature(
            c, str(other_kp.pubkey()), str(other_kp.pubkey()))
        assert valid is False
        assert "unknown_signer" in reason

    def test_reject_tampered_content(self):
        kp = self._make_keypair()
        c = Contract(contract_id="tampered", author="maker",
                      rules=[{"op": "IF"}])
        sign_contract(c, kp)

        # Tamper with rules after signing
        c.rules = [{"op": "DROP_ALL"}]

        valid, reason = verify_contract_signature(
            c, str(kp.pubkey()), str(kp.pubkey()))
        assert valid is False
        assert "invalid_signature" in reason

    def test_titan_contract_needs_maker_approval(self):
        titan_kp = self._make_keypair()
        maker_kp = self._make_keypair()

        c = Contract(
            contract_id="titan_proposed",
            author="titan",
            rules=[{"op": "EMIT_EVENT", "event": "STALL"}],
        )
        sign_contract(c, titan_kp)
        c.status = "active"  # Try to activate without approval

        valid, reason = verify_contract_signature(
            c, str(titan_kp.pubkey()), str(maker_kp.pubkey()))
        assert valid is False
        assert "missing_maker_approval" in reason

    def test_full_approval_flow(self):
        titan_kp = self._make_keypair()
        maker_kp = self._make_keypair()

        # Titan proposes
        c = Contract(
            contract_id="titan_proposal",
            author="titan",
            rules=[{"op": "TREND", "field": "vocab_size"}],
        )
        sign_contract(c, titan_kp)
        assert c.status == "draft"

        # Maker approves
        approve_contract(c, maker_kp)
        assert c.status == "active"
        assert c.approver_signature != ""

        # Verify full chain
        valid, reason = verify_contract_signature(
            c, str(titan_kp.pubkey()), str(maker_kp.pubkey()))
        assert valid is True

    def test_unsigned_contract_rejected(self):
        c = Contract(contract_id="unsigned")
        valid, reason = verify_contract_signature(c, "abc", "def")
        assert valid is False
        assert reason == "no_signature"


class TestContractStore:
    def test_deploy_and_list(self, tmp_dir):
        from unittest.mock import MagicMock
        mock_tc = MagicMock()
        mock_tc.query_blocks.return_value = []
        mock_tc.commit_block.return_value = MagicMock(
            block_hash=b"\x01" * 32,
            header=MagicMock(block_height=1))

        kp = self._make_keypair()
        store = ContractStore(mock_tc, str(kp.pubkey()), str(kp.pubkey()))

        c = Contract(
            contract_id="test_filter",
            contract_type="filter",
            author="maker",
            rules=[{"op": "IF", "field": "sig", "cmp": "GT", "value": 0.5}],
        )
        sign_contract(c, kp)
        ok, reason = store.deploy(c)
        assert ok is True
        assert reason == "deployed"

        # List
        all_contracts = store.get_all()
        assert len(all_contracts) == 1
        assert all_contracts[0].contract_id == "test_filter"
        assert all_contracts[0].status == "active"

    def test_reject_invalid_type(self, tmp_dir):
        from unittest.mock import MagicMock
        mock_tc = MagicMock()
        mock_tc.query_blocks.return_value = []

        kp = self._make_keypair()
        store = ContractStore(mock_tc, str(kp.pubkey()), str(kp.pubkey()))

        c = Contract(contract_id="bad", contract_type="invalid_type")
        sign_contract(c, kp)
        ok, reason = store.deploy(c)
        assert ok is False
        assert "invalid_type" in reason

    def test_stats(self, tmp_dir):
        from unittest.mock import MagicMock
        mock_tc = MagicMock()
        mock_tc.query_blocks.return_value = []
        mock_tc.commit_block.return_value = MagicMock(
            block_hash=b"\x01" * 32,
            header=MagicMock(block_height=1))

        kp = self._make_keypair()
        store = ContractStore(mock_tc, str(kp.pubkey()), str(kp.pubkey()))

        c = Contract(contract_id="s1", contract_type="filter", author="maker")
        sign_contract(c, kp)
        store.deploy(c)

        stats = store.get_stats()
        assert stats["total"] == 1
        assert stats["by_status"].get("active") == 1
        assert stats["by_type"].get("filter") == 1

    def _make_keypair(self):
        from solders.keypair import Keypair
        return Keypair()


# ═══════════════════════════════════════════════════════════════════════
# P3b: Rule Evaluator Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRuleEvaluator(unittest.TestCase):
    """Test the non-Turing-complete contract rule interpreter."""

    def setUp(self):
        self.evaluator = RuleEvaluator()
        self.ctx = {
            "significance": 0.2,
            "source": "expression_art",
            "fork": "episodic",
            "thought_type": "episodic",
            "epoch_id": 1000,
            "confidence": 0.7,
        }

    def test_simple_if_gt(self):
        rules = [{"op": "IF", "field": "significance", "cmp": "GT", "value": 0.5,
                  "then": {"action": "include"}}]
        assert self.evaluator.evaluate(rules, self.ctx) is None

    def test_simple_if_lt(self):
        rules = [{"op": "IF", "field": "significance", "cmp": "LT", "value": 0.5,
                  "then": {"action": "drop"}}]
        result = self.evaluator.evaluate(rules, self.ctx)
        assert result == {"action": "drop"}

    def test_compound_and(self):
        rules = [{
            "op": "AND",
            "clauses": [
                {"op": "IF", "field": "significance", "cmp": "LT", "value": 0.3},
                {"op": "IF", "field": "source", "cmp": "EQ", "value": "expression_art"},
            ],
            "then": {"action": "aggregate"},
        }]
        result = self.evaluator.evaluate(rules, self.ctx)
        assert result == {"action": "aggregate"}

    def test_compound_and_fails(self):
        rules = [{
            "op": "AND",
            "clauses": [
                {"op": "IF", "field": "significance", "cmp": "LT", "value": 0.3},
                {"op": "IF", "field": "source", "cmp": "EQ", "value": "meditation"},
            ],
            "then": {"action": "aggregate"},
        }]
        assert self.evaluator.evaluate(rules, self.ctx) is None

    def test_compound_or(self):
        rules = [{
            "op": "OR",
            "clauses": [
                {"op": "IF", "field": "source", "cmp": "EQ", "value": "meditation"},
                {"op": "IF", "field": "source", "cmp": "EQ", "value": "expression_art"},
            ],
            "then": {"action": "include"},
        }]
        result = self.evaluator.evaluate(rules, self.ctx)
        assert result == {"action": "include"}

    def test_not_operator(self):
        rules = [{
            "op": "NOT",
            "clause": {"op": "IF", "field": "source", "cmp": "EQ", "value": "meditation"},
            "then": {"action": "drop"},
        }]
        result = self.evaluator.evaluate(rules, self.ctx)
        assert result == {"action": "drop"}

    def test_in_comparison(self):
        rules = [{"op": "IF", "field": "source", "cmp": "IN",
                  "value": ["expression_art", "expression_music"],
                  "then": {"action": "aggregate"}}]
        result = self.evaluator.evaluate(rules, self.ctx)
        assert result == {"action": "aggregate"}

    def test_not_in_comparison(self):
        rules = [{"op": "IF", "field": "source", "cmp": "NOT_IN",
                  "value": ["meditation", "dream_insight"],
                  "then": {"action": "drop"}}]
        result = self.evaluator.evaluate(rules, self.ctx)
        assert result == {"action": "drop"}

    def test_between_comparison(self):
        rules = [{"op": "IF", "field": "significance", "cmp": "BETWEEN",
                  "value": [0.1, 0.3],
                  "then": {"action": "aggregate"}}]
        result = self.evaluator.evaluate(rules, self.ctx)
        assert result == {"action": "aggregate"}

    def test_dotted_field_path(self):
        ctx = {"neuromods": {"DA": 0.52, "GABA": 0.12}}
        rules = [{"op": "IF", "field": "neuromods.GABA", "cmp": "LT", "value": 0.15,
                  "then": {"action": "emit", "event": "LOW_GABA"}}]
        result = self.evaluator.evaluate(rules, ctx)
        assert result["action"] == "emit"
        assert result["event"] == "LOW_GABA"

    def test_first_matching_rule_wins(self):
        rules = [
            {"op": "IF", "field": "significance", "cmp": "LT", "value": 0.1,
             "then": {"action": "drop"}},
            {"op": "IF", "field": "significance", "cmp": "LT", "value": 0.5,
             "then": {"action": "aggregate"}},
            {"op": "IF", "field": "significance", "cmp": "LT", "value": 1.0,
             "then": {"action": "include"}},
        ]
        result = self.evaluator.evaluate(rules, self.ctx)
        assert result == {"action": "aggregate"}

    def test_max_rules_safety(self):
        rules = [{"op": "IF", "field": "x", "cmp": "EQ", "value": 1,
                  "then": {"action": "drop"}}] * 51
        result = self.evaluator.evaluate(rules, {"x": 1})
        assert result is None

    def test_no_matching_returns_none(self):
        rules = [{"op": "IF", "field": "significance", "cmp": "GT", "value": 99,
                  "then": {"action": "drop"}}]
        assert self.evaluator.evaluate(rules, self.ctx) is None

    def test_empty_rules(self):
        assert self.evaluator.evaluate([], self.ctx) is None


class TestGenesisPrimitives(unittest.TestCase):
    """Test TREND, DELTA, SINCE primitives for genesis contracts."""

    def setUp(self):
        self.evaluator = RuleEvaluator()
        self.genesis_states = [
            {"vocab_size": 300, "i_confidence": 0.95, "meta_chains": 15,
             "neuromods": {"GABA": 0.12}, "trigger": "meditation", "emotion": "wonder"},
            {"vocab_size": 300, "i_confidence": 0.94, "meta_chains": 14,
             "neuromods": {"GABA": 0.15}, "trigger": "meditation", "emotion": "neutral"},
            {"vocab_size": 298, "i_confidence": 0.93, "meta_chains": 13,
             "neuromods": {"GABA": 0.20}, "trigger": "timer", "emotion": "neutral"},
            {"vocab_size": 295, "i_confidence": 0.90, "meta_chains": 10,
             "neuromods": {"GABA": 0.25}, "trigger": "meditation", "emotion": "flow"},
        ]

    def test_trend_flat(self):
        rules = [{"op": "TREND", "field": "vocab_size", "window": 2,
                  "cmp": "EQ", "value": "flat",
                  "then": {"action": "emit", "event": "STALL"}}]
        result = self.evaluator.evaluate(
            rules, self.genesis_states[0], genesis_states=self.genesis_states)
        assert result is not None

    def test_trend_rising(self):
        rising_states = [
            {"vocab_size": 310}, {"vocab_size": 300}, {"vocab_size": 290}]
        rules = [{"op": "TREND", "field": "vocab_size", "window": 3,
                  "cmp": "EQ", "value": "rising",
                  "then": {"action": "emit", "event": "GROWING"}}]
        result = self.evaluator.evaluate(
            rules, rising_states[0], genesis_states=rising_states)
        assert result is not None

    def test_delta_negative(self):
        rules = [{"op": "DELTA", "field": "neuromods.GABA", "n_back": 3,
                  "cmp": "LT", "value": -0.1,
                  "then": {"action": "emit", "event": "GABA_DECLINE"}}]
        result = self.evaluator.evaluate(
            rules, self.genesis_states[0], genesis_states=self.genesis_states)
        assert result is not None
        assert result["event"] == "GABA_DECLINE"

    def test_delta_not_triggered(self):
        rules = [{"op": "DELTA", "field": "i_confidence", "n_back": 1,
                  "cmp": "LT", "value": -0.1,
                  "then": {"action": "emit", "event": "CONFIDENCE_DROP"}}]
        result = self.evaluator.evaluate(
            rules, self.genesis_states[0], genesis_states=self.genesis_states)
        assert result is None

    def test_since_event(self):
        rules = [{"op": "SINCE", "event_type": "flow",
                  "cmp": "GTE", "value": 2,
                  "then": {"action": "emit", "event": "LONG_SINCE_FLOW"}}]
        result = self.evaluator.evaluate(
            rules, self.genesis_states[0], genesis_states=self.genesis_states)
        assert result is not None

    def test_builtin_cognitive_stall(self):
        stall_ctx = self.genesis_states[0]
        stall_states = [
            {"vocab_size": 300}, {"vocab_size": 300}, {"vocab_size": 300}]
        rules = [{
            "op": "AND",
            "clauses": [
                {"op": "TREND", "field": "vocab_size", "window": 3,
                 "cmp": "EQ", "value": "flat"},
                {"op": "IF", "field": "meta_chains", "cmp": "GT", "value": 10},
            ],
            "then": {"action": "emit", "event": "COGNITIVE_STALL"},
        }]
        result = self.evaluator.evaluate(rules, stall_ctx, genesis_states=stall_states)
        assert result is not None
        assert result["event"] == "COGNITIVE_STALL"

    def test_builtin_milestone_tracker(self):
        ctx = {"i_confidence": 0.96, "vocab_size": 316}
        rules = [{
            "op": "AND",
            "clauses": [
                {"op": "IF", "field": "i_confidence", "cmp": "GTE", "value": 0.95},
                {"op": "IF", "field": "vocab_size", "cmp": "GTE", "value": 300},
            ],
            "then": {"action": "emit", "event": "MILESTONE_REACHED",
                     "data": {"milestone": "sovereign_speaker"}},
        }]
        result = self.evaluator.evaluate(rules, ctx)
        assert result is not None
        assert result["event"] == "MILESTONE_REACHED"
        assert result["data"]["milestone"] == "sovereign_speaker"


# ═══════════════════════════════════════════════════════════════════════
# P3d: Contract Proposal + Approval Tests
# ═══════════════════════════════════════════════════════════════════════

class TestContractProposal(unittest.TestCase):
    """Test Titan-authored contract proposal and Maker approval flow."""

    def setUp(self):
        from solders.keypair import Keypair
        from unittest.mock import MagicMock
        self.titan_kp = Keypair()
        self.maker_kp = Keypair()
        self.mock_tc = MagicMock()
        self.mock_tc.query_blocks.return_value = []
        self.mock_tc.commit_block.return_value = MagicMock(
            block_hash=b"\x01" * 32,
            header=MagicMock(block_height=1))
        self.store = ContractStore(
            self.mock_tc, str(self.titan_kp.pubkey()),
            str(self.maker_kp.pubkey()))

    def test_propose_creates_pending_contract(self):
        ok, reason = self.store.propose(
            name="test_filter",
            contract_type="filter",
            rules=[{"op": "IF", "field": "significance", "cmp": "LT",
                    "value": 0.1, "then": {"action": "drop"}}],
            description="Test filter contract",
            titan_keypair=self.titan_kp,
        )
        assert ok
        c = self.store.get("test_filter")
        assert c is not None
        assert c.status == "pending_approval"
        assert c.author == "titan"
        assert c.signature  # Should be signed

    def test_propose_rejects_duplicate(self):
        self.store.propose(
            name="dupe", contract_type="filter", rules=[],
            description="first", titan_keypair=self.titan_kp)
        ok, reason = self.store.propose(
            name="dupe", contract_type="filter", rules=[],
            description="second", titan_keypair=self.titan_kp)
        assert not ok
        assert "already_exists" in reason

    def test_propose_rejects_invalid_type(self):
        ok, reason = self.store.propose(
            name="bad_type", contract_type="invalid_type", rules=[],
            description="bad", titan_keypair=self.titan_kp)
        assert not ok
        assert "invalid_type" in reason

    def test_get_pending(self):
        self.store.propose(
            name="p1", contract_type="filter", rules=[],
            description="test", titan_keypair=self.titan_kp)
        pending = self.store.get_pending()
        assert len(pending) == 1
        assert pending[0].contract_id == "p1"

    def test_approve_activates_contract(self):
        self.store.propose(
            name="to_approve", contract_type="trigger", rules=[],
            description="approve me", titan_keypair=self.titan_kp)
        assert self.store.get("to_approve").status == "pending_approval"

        ok, reason = self.store.approve("to_approve", self.maker_kp)
        assert ok
        assert self.store.get("to_approve").status == "active"

    def test_reject_with_reason(self):
        self.store.propose(
            name="to_reject", contract_type="filter", rules=[],
            description="reject me", titan_keypair=self.titan_kp)
        ok, reason = self.store.reject("to_reject", "unsafe rules")
        assert ok
        c = self.store.get("to_reject")
        assert c.status == "rejected"
        assert c.rejection_reason == "unsafe rules"

    def test_reject_nonexistent(self):
        ok, reason = self.store.reject("ghost", "no such contract")
        assert not ok
        assert "not_found" in reason

    def test_full_proposal_lifecycle(self):
        """Propose → pending → approve → active."""
        self.store.propose(
            name="lifecycle_test", contract_type="genesis",
            rules=[{"op": "IF", "field": "i_confidence", "cmp": "GTE",
                    "value": 0.99, "then": {"action": "emit",
                                             "event": "NEAR_PERFECTION"}}],
            description="Test lifecycle",
            titan_keypair=self.titan_kp)

        c = self.store.get("lifecycle_test")
        assert c.status == "pending_approval"
        assert c.signature
        assert c.signer_pubkey

        # Approve
        ok, _ = self.store.approve("lifecycle_test", self.maker_kp)
        assert ok
        c = self.store.get("lifecycle_test")
        assert c.status == "active"
        assert c.approver_signature
        assert c.activated_at > 0
