"""
Tests for TimeChain — Proof of Thought Tripartite Memory Architecture.

Tests cover:
  - Block creation + hash verification
  - Chain append + integrity verification
  - Fork creation + sidechain management
  - PoT validation (pass/reject cases)
  - Cross-reference validation
  - Serialization roundtrip
  - Merkle checkpointing
  - Dream compaction
  - Query interface
  - Index rebuild
"""

import os
import shutil
import tempfile
import time

import pytest

from titan_plugin.logic.timechain import (
    Block,
    BlockHeader,
    BlockPayload,
    CrossRef,
    TimeChain,
    FORK_DECLARATIVE,
    FORK_EPISODIC,
    FORK_MAIN,
    FORK_META,
    FORK_PROCEDURAL,
    FORK_SIDECHAIN_START,
    GENESIS_NEUROMOD_HASH,
    GENESIS_PREV_HASH,
    HEADER_SIZE,
    sha256,
)
from titan_plugin.logic.proof_of_thought import (
    BASE_CHI_COSTS,
    DEFAULT_THRESHOLDS,
    MIN_THRESHOLD,
    PoTValidator,
    ProofOfThought,
)


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    """Temporary directory for chain files."""
    d = tempfile.mkdtemp(prefix="timechain_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def chain(tmp_dir):
    """A TimeChain instance with genesis block created."""
    tc = TimeChain(data_dir=os.path.join(tmp_dir, "timechain"), titan_id="T_TEST")
    tc.create_genesis(
        genesis_content={
            "maker_pubkey": "test_maker_pubkey",
            "soul_hash": "test_soul_hash",
            "prime_directives": ["Sovereign Integrity", "Cognitive Safety"],
            "born": "2026-04-05",
        },
        birth_timestamp=1712345678.0,
    )
    return tc


@pytest.fixture
def validator():
    """A PoTValidator with default thresholds."""
    return PoTValidator()


def _neuromods(da=0.5, ach=0.5, ne=0.5, sht=0.5, gaba=0.2, endo=0.3):
    """Helper to create neuromod state dict."""
    return {
        "DA": da, "ACh": ach, "NE": ne,
        "5HT": sht, "GABA": gaba, "endorphin": endo,
    }


def _make_payload(thought_type="declarative", source="teacher",
                  content=None, significance=0.5, tags=None):
    """Helper to create a BlockPayload."""
    return BlockPayload(
        thought_type=thought_type,
        source=source,
        content=content or {"word": "test", "confidence": 0.8},
        significance=significance,
        confidence=0.7,
        tags=tags or ["test"],
        db_ref="vocabulary:42",
    )


# ══════════════════════════════════════════════════════════════════════
# Block Structure Tests
# ══════════════════════════════════════════════════════════════════════

class TestBlockStructure:
    """Test block creation, serialization, and hash verification."""

    def test_header_size_is_128_bytes(self):
        header = BlockHeader(
            version=1, block_height=0, timestamp=time.time(),
            epoch_id=100, prev_hash=GENESIS_PREV_HASH,
            payload_hash=b"\x01" * 32, fork_id=0, fork_parent=0,
            pot_nonce=1, chi_spent=0.005, neuromod_hash=GENESIS_NEUROMOD_HASH,
            cross_ref_count=0,
        )
        assert len(header.to_bytes()) == HEADER_SIZE

    def test_header_roundtrip(self):
        header = BlockHeader(
            version=1, block_height=42, timestamp=1712345678.123,
            epoch_id=99999, prev_hash=sha256(b"prev"),
            payload_hash=sha256(b"payload"), fork_id=2,
            fork_parent=10, pot_nonce=777, chi_spent=0.0123,
            neuromod_hash=sha256(b"neuro")[:16], cross_ref_count=2,
        )
        raw = header.to_bytes()
        restored = BlockHeader.from_bytes(raw)
        assert restored.version == 1
        assert restored.block_height == 42
        assert abs(restored.timestamp - 1712345678.123) < 0.001
        assert restored.epoch_id == 99999
        assert restored.prev_hash == header.prev_hash
        assert restored.payload_hash == header.payload_hash
        assert restored.fork_id == 2
        assert restored.fork_parent == 10
        assert restored.pot_nonce == 777
        assert abs(restored.chi_spent - 0.0123) < 0.001
        assert restored.neuromod_hash == header.neuromod_hash
        assert restored.cross_ref_count == 2

    def test_payload_roundtrip(self):
        payload = BlockPayload(
            thought_type="declarative",
            source="teacher",
            content={"word": "bright", "confidence": 0.85, "associations": ["light"]},
            felt_tensor=b"\x01\x02\x03" * 10,
            significance=0.72,
            confidence=0.85,
            tags=["bright", "language"],
            db_ref="vocabulary:123",
        )
        raw = payload.to_bytes()
        restored = BlockPayload.from_bytes(raw)
        assert restored.thought_type == "declarative"
        assert restored.source == "teacher"
        assert restored.content["word"] == "bright"
        assert restored.felt_tensor == b"\x01\x02\x03" * 10
        assert abs(restored.significance - 0.72) < 0.001
        assert restored.tags == ["bright", "language"]
        assert restored.db_ref == "vocabulary:123"

    def test_cross_ref_roundtrip(self):
        ref = CrossRef(fork_id=3, block_height=12345)
        raw = ref.to_bytes()
        assert len(raw) == 10
        restored = CrossRef.from_bytes(raw)
        assert restored.fork_id == 3
        assert restored.block_height == 12345

    def test_full_block_roundtrip(self):
        payload = _make_payload()
        header = BlockHeader(
            version=1, block_height=5, timestamp=time.time(),
            epoch_id=100, prev_hash=sha256(b"prev"),
            payload_hash=sha256(payload.to_bytes()), fork_id=1,
            fork_parent=0, pot_nonce=1, chi_spent=0.005,
            neuromod_hash=GENESIS_NEUROMOD_HASH, cross_ref_count=2,
        )
        refs = [CrossRef(1, 3), CrossRef(3, 7)]
        block = Block(header=header, cross_refs=refs, payload=payload)

        raw = block.to_bytes()
        restored = Block.from_bytes(raw)

        assert restored.header.block_height == 5
        assert restored.header.fork_id == 1
        assert len(restored.cross_refs) == 2
        assert restored.cross_refs[0].fork_id == 1
        assert restored.cross_refs[1].block_height == 7
        assert restored.payload.thought_type == "declarative"
        assert restored.payload.content["word"] == "test"

    def test_block_hash_deterministic(self):
        payload = _make_payload()
        header = BlockHeader(
            version=1, block_height=0, timestamp=1712345678.0,
            epoch_id=0, prev_hash=GENESIS_PREV_HASH,
            payload_hash=sha256(payload.to_bytes()), fork_id=0,
            fork_parent=0, pot_nonce=0, chi_spent=0.0,
            neuromod_hash=GENESIS_NEUROMOD_HASH, cross_ref_count=0,
        )
        block = Block(header=header, payload=payload)
        hash1 = block.block_hash
        hash2 = block.block_hash
        assert hash1 == hash2
        assert len(hash1) == 32

    def test_different_payloads_different_hashes(self):
        p1 = _make_payload(content={"a": 1})
        p2 = _make_payload(content={"b": 2})
        assert sha256(p1.to_bytes()) != sha256(p2.to_bytes())


# ══════════════════════════════════════════════════════════════════════
# Genesis Tests
# ══════════════════════════════════════════════════════════════════════

class TestGenesis:
    """Test genesis block creation and properties."""

    def test_genesis_creates_successfully(self, chain):
        assert chain.has_genesis
        assert chain.total_blocks == 1
        genesis = chain.get_block(FORK_MAIN, 0)
        assert genesis is not None
        assert genesis.header.block_height == 0
        assert genesis.header.prev_hash == GENESIS_PREV_HASH
        assert genesis.payload.thought_type == "genesis"
        assert genesis.payload.source == "maker"

    def test_genesis_content(self, chain):
        genesis = chain.get_block(FORK_MAIN, 0)
        assert genesis.payload.content["maker_pubkey"] == "test_maker_pubkey"
        assert "Sovereign Integrity" in genesis.payload.content["prime_directives"]

    def test_genesis_hash_is_deterministic(self, chain):
        h1 = chain.genesis_hash
        h2 = chain.genesis_hash
        assert h1 == h2
        assert len(h1) == 32

    def test_double_genesis_is_noop(self, chain):
        chain.create_genesis({"duplicate": True})
        assert chain.total_blocks == 1  # still 1

    def test_primary_forks_registered(self, chain):
        stats = chain.get_fork_stats()
        assert 0 in stats  # main
        assert 1 in stats  # declarative
        assert 2 in stats  # procedural
        assert 3 in stats  # episodic
        assert 4 in stats  # meta


# ══════════════════════════════════════════════════════════════════════
# Chain Append + Integrity Tests
# ══════════════════════════════════════════════════════════════════════

class TestChainAppend:
    """Test block commitment and hash chain integrity."""

    def test_commit_single_block(self, chain):
        block = chain.commit_block(
            fork_id=FORK_DECLARATIVE, epoch_id=1000,
            payload=_make_payload(), pot_nonce=1, chi_spent=0.005,
            neuromod_state=_neuromods(),
        )
        assert block is not None
        assert block.header.block_height == 0
        assert block.header.fork_id == FORK_DECLARATIVE
        assert chain.total_blocks == 2  # genesis + 1

    def test_commit_chain_of_blocks(self, chain):
        for i in range(10):
            block = chain.commit_block(
                fork_id=FORK_DECLARATIVE, epoch_id=1000 + i,
                payload=_make_payload(
                    content={"word": f"word_{i}", "idx": i},
                    tags=[f"word_{i}"],
                ),
                pot_nonce=1, chi_spent=0.005,
                neuromod_state=_neuromods(),
            )
            assert block.header.block_height == i
        assert chain.total_blocks == 11  # genesis + 10

    def test_hash_chain_integrity(self, chain):
        # Add 5 blocks
        for i in range(5):
            chain.commit_block(
                fork_id=FORK_DECLARATIVE, epoch_id=100 + i,
                payload=_make_payload(content={"i": i}),
                pot_nonce=1, chi_spent=0.005,
                neuromod_state=_neuromods(),
            )
        # Verify chain
        valid, msg = chain.verify_fork(FORK_DECLARATIVE)
        assert valid, f"Chain invalid: {msg}"

    def test_main_chain_integrity(self, chain):
        valid, msg = chain.verify_fork(FORK_MAIN)
        assert valid, f"Main chain invalid: {msg}"

    def test_verify_all_forks(self, chain):
        # Add blocks to multiple forks
        for fork in [FORK_DECLARATIVE, FORK_PROCEDURAL, FORK_EPISODIC]:
            for i in range(3):
                chain.commit_block(
                    fork_id=fork, epoch_id=100 + i,
                    payload=_make_payload(content={"f": fork, "i": i}),
                    pot_nonce=1, chi_spent=0.005,
                    neuromod_state=_neuromods(),
                )
        valid, results = chain.verify_all()
        assert valid, f"Verification failed: {results}"

    def test_prev_hash_chains_correctly(self, chain):
        b1 = chain.commit_block(
            fork_id=FORK_EPISODIC, epoch_id=100,
            payload=_make_payload(thought_type="episodic"),
            pot_nonce=1, chi_spent=0.003,
            neuromod_state=_neuromods(),
        )
        b2 = chain.commit_block(
            fork_id=FORK_EPISODIC, epoch_id=101,
            payload=_make_payload(thought_type="episodic"),
            pot_nonce=1, chi_spent=0.003,
            neuromod_state=_neuromods(),
        )
        # b2's prev_hash should be b1's header hash
        assert b2.header.prev_hash == b1.header.compute_hash()

    def test_commit_to_nonexistent_fork_returns_none(self, chain):
        result = chain.commit_block(
            fork_id=999, epoch_id=100,
            payload=_make_payload(), pot_nonce=1, chi_spent=0.005,
            neuromod_state=_neuromods(),
        )
        assert result is None

    def test_cross_references(self, chain):
        # Add a declarative block first
        b1 = chain.commit_block(
            fork_id=FORK_DECLARATIVE, epoch_id=100,
            payload=_make_payload(content={"word": "bright"}),
            pot_nonce=1, chi_spent=0.005,
            neuromod_state=_neuromods(),
        )
        # Add episodic block referencing declarative
        b2 = chain.commit_block(
            fork_id=FORK_EPISODIC, epoch_id=100,
            payload=_make_payload(thought_type="episodic",
                                  content={"event": "learned bright"}),
            pot_nonce=1, chi_spent=0.003,
            neuromod_state=_neuromods(),
            cross_refs=[CrossRef(FORK_DECLARATIVE, 0)],
        )
        assert len(b2.cross_refs) == 1
        assert b2.cross_refs[0].fork_id == FORK_DECLARATIVE
        assert b2.cross_refs[0].block_height == 0

    def test_max_cross_refs_capped_at_4(self, chain):
        refs = [CrossRef(i, 0) for i in range(10)]
        block = chain.commit_block(
            fork_id=FORK_EPISODIC, epoch_id=100,
            payload=_make_payload(thought_type="episodic"),
            pot_nonce=1, chi_spent=0.003,
            neuromod_state=_neuromods(),
            cross_refs=refs,
        )
        assert len(block.cross_refs) <= 4


# ══════════════════════════════════════════════════════════════════════
# Block Retrieval Tests
# ══════════════════════════════════════════════════════════════════════

class TestRetrieval:
    """Test block retrieval by height, hash, and queries."""

    def test_get_block_by_height(self, chain):
        chain.commit_block(
            fork_id=FORK_DECLARATIVE, epoch_id=100,
            payload=_make_payload(content={"word": "hello"}),
            pot_nonce=1, chi_spent=0.005,
            neuromod_state=_neuromods(),
        )
        block = chain.get_block(FORK_DECLARATIVE, 0)
        assert block is not None
        assert block.payload.content["word"] == "hello"

    def test_get_block_by_hash(self, chain):
        b = chain.commit_block(
            fork_id=FORK_DECLARATIVE, epoch_id=100,
            payload=_make_payload(), pot_nonce=1, chi_spent=0.005,
            neuromod_state=_neuromods(),
        )
        retrieved = chain.get_block_by_hash(b.block_hash)
        assert retrieved is not None
        assert retrieved.header.block_height == b.header.block_height

    def test_get_recent_blocks(self, chain):
        for i in range(5):
            chain.commit_block(
                fork_id=FORK_DECLARATIVE, epoch_id=100 + i,
                payload=_make_payload(content={"i": i}),
                pot_nonce=1, chi_spent=0.005,
                neuromod_state=_neuromods(),
            )
        recent = chain.get_recent_blocks(FORK_DECLARATIVE, n=3)
        assert len(recent) == 3
        # Should be most recent first
        assert recent[0]["height"] == 4
        assert recent[1]["height"] == 3

    def test_get_nonexistent_block_returns_none(self, chain):
        assert chain.get_block(FORK_DECLARATIVE, 999) is None

    def test_get_fork_tip(self, chain):
        chain.commit_block(
            fork_id=FORK_DECLARATIVE, epoch_id=100,
            payload=_make_payload(), pot_nonce=1, chi_spent=0.005,
            neuromod_state=_neuromods(),
        )
        height, tip_hash = chain.get_fork_tip(FORK_DECLARATIVE)
        assert height == 0
        assert len(tip_hash) == 32


# ══════════════════════════════════════════════════════════════════════
# Sidechain Tests
# ══════════════════════════════════════════════════════════════════════

class TestSidechains:
    """Test automatic topic sidechain creation."""

    def test_sidechain_auto_created_at_threshold(self, chain):
        # Add 3 blocks with same tag "warmth" to trigger auto-sidechain
        for i in range(3):
            chain.commit_block(
                fork_id=FORK_DECLARATIVE, epoch_id=100 + i,
                payload=_make_payload(content={"word": "warm"},
                                      tags=["warmth"]),
                pot_nonce=1, chi_spent=0.005,
                neuromod_state=_neuromods(),
            )
        # Sidechain should exist now
        sc_id = chain.get_sidechain_for_topic("warmth")
        assert sc_id is not None
        assert sc_id >= FORK_SIDECHAIN_START

    def test_sidechain_not_created_below_threshold(self, chain):
        for i in range(2):
            chain.commit_block(
                fork_id=FORK_DECLARATIVE, epoch_id=100 + i,
                payload=_make_payload(tags=["rare_topic"]),
                pot_nonce=1, chi_spent=0.005,
                neuromod_state=_neuromods(),
            )
        assert chain.get_sidechain_for_topic("rare_topic") is None

    def test_sidechain_can_receive_blocks(self, chain):
        # Create sidechain manually
        for i in range(3):
            chain.commit_block(
                fork_id=FORK_DECLARATIVE, epoch_id=100 + i,
                payload=_make_payload(tags=["symmetry"]),
                pot_nonce=1, chi_spent=0.005,
                neuromod_state=_neuromods(),
            )
        sc_id = chain.get_sidechain_for_topic("symmetry")
        assert sc_id is not None

        # Commit block to sidechain
        block = chain.commit_block(
            fork_id=sc_id, epoch_id=200,
            payload=_make_payload(tags=["symmetry"], content={"deep": True}),
            pot_nonce=1, chi_spent=0.004,
            neuromod_state=_neuromods(),
        )
        assert block is not None
        assert block.header.fork_id == sc_id


# ══════════════════════════════════════════════════════════════════════
# Proof of Thought Tests
# ══════════════════════════════════════════════════════════════════════

class TestProofOfThought:
    """Test PoT validation — the cognitive admission gate."""

    def test_healthy_thought_passes(self, validator):
        pot = validator.create_pot(
            chi_available=0.5, metabolic_drain=0.2,
            attention=0.6, i_confidence=0.5, chi_coherence=0.4,
            neuromods=_neuromods(da=0.6, ach=0.5, ne=0.4),
            novelty=0.7, significance=0.6, coherence=0.5,
            source="teacher", thought_type="declarative",
            fork_name="declarative",
        )
        assert pot.valid
        assert pot.pot_score > 0
        assert pot.nonce == 1

    def test_no_chi_fails(self, validator):
        pot = validator.create_pot(
            chi_available=0.001, metabolic_drain=0.0,
            attention=0.6, i_confidence=0.5, chi_coherence=0.4,
            neuromods=_neuromods(),
            novelty=0.7, significance=0.6, coherence=0.5,
            source="teacher", thought_type="declarative",
            fork_name="declarative",
        )
        assert not pot.valid
        assert pot.rejection_reason == "insufficient_chi"

    def test_low_attention_fails(self, validator):
        pot = validator.create_pot(
            chi_available=0.5, metabolic_drain=0.0,
            attention=0.01, i_confidence=0.01, chi_coherence=0.01,
            neuromods=_neuromods(),
            novelty=0.7, significance=0.6, coherence=0.5,
            source="teacher", thought_type="declarative",
            fork_name="declarative",
        )
        assert not pot.valid
        assert pot.rejection_reason == "low_cognitive_readiness"

    def test_episodic_lower_threshold_than_procedural(self, validator):
        """Episodic memories form more easily than procedural skills."""
        kwargs = dict(
            chi_available=0.5, metabolic_drain=0.3,
            attention=0.4, i_confidence=0.3, chi_coherence=0.3,
            neuromods=_neuromods(da=0.3, ach=0.3, ne=0.3),
            novelty=0.4, significance=0.3, coherence=0.3,
            source="social",
        )
        ep = validator.create_pot(**kwargs, thought_type="episodic", fork_name="episodic")
        pr = validator.create_pot(**kwargs, thought_type="procedural", fork_name="procedural")
        # Same score, but episodic has lower threshold
        assert ep.threshold < pr.threshold

    def test_meta_highest_threshold(self, validator):
        assert DEFAULT_THRESHOLDS["meta"] >= DEFAULT_THRESHOLDS["declarative"]
        assert DEFAULT_THRESHOLDS["meta"] > DEFAULT_THRESHOLDS["episodic"]
        assert DEFAULT_THRESHOLDS["meta"] >= DEFAULT_THRESHOLDS["procedural"]

    def test_high_curvature_lowers_threshold(self, validator):
        base = validator.create_pot(
            chi_available=0.5, metabolic_drain=0.0,
            attention=0.6, i_confidence=0.5, chi_coherence=0.4,
            neuromods=_neuromods(), novelty=0.5, significance=0.5,
            coherence=0.5, source="teacher",
            thought_type="declarative", fork_name="declarative",
            pi_curvature=1.0,  # normal
        )
        surprising = validator.create_pot(
            chi_available=0.5, metabolic_drain=0.0,
            attention=0.6, i_confidence=0.5, chi_coherence=0.4,
            neuromods=_neuromods(), novelty=0.5, significance=0.5,
            coherence=0.5, source="teacher",
            thought_type="declarative", fork_name="declarative",
            pi_curvature=3.0,  # very surprising
        )
        assert surprising.threshold < base.threshold

    def test_threshold_never_below_minimum(self, validator):
        pot = validator.create_pot(
            chi_available=0.5, metabolic_drain=0.0,
            attention=0.6, i_confidence=0.5, chi_coherence=0.4,
            neuromods=_neuromods(), novelty=0.5, significance=0.5,
            coherence=0.5, source="teacher",
            thought_type="episodic", fork_name="episodic",
            pi_curvature=10.0,  # extreme
        )
        assert pot.threshold >= MIN_THRESHOLD

    def test_chi_cost_scales_with_significance(self, validator):
        low = validator.compute_chi_cost("declarative", 0.1)
        high = validator.compute_chi_cost("declarative", 0.9)
        assert high > low

    def test_procedural_costs_more_than_episodic(self, validator):
        proc = validator.compute_chi_cost("procedural", 0.5)
        epis = validator.compute_chi_cost("episodic", 0.5)
        assert proc > epis

    def test_acceptance_rate_tracking(self, validator):
        # Submit some thoughts
        for _ in range(3):
            validator.create_pot(
                chi_available=0.5, metabolic_drain=0.0,
                attention=0.6, i_confidence=0.5, chi_coherence=0.4,
                neuromods=_neuromods(), novelty=0.7, significance=0.6,
                coherence=0.5, source="teacher",
                thought_type="declarative", fork_name="declarative",
            )
        stats = validator.stats
        assert stats["submitted"] == 3
        assert stats["admitted"] >= 0
        assert 0 <= stats["acceptance_rate"] <= 1.0

    def test_high_drain_reduces_score(self, validator):
        normal = validator.create_pot(
            chi_available=0.5, metabolic_drain=0.1,
            attention=0.6, i_confidence=0.5, chi_coherence=0.4,
            neuromods=_neuromods(), novelty=0.6, significance=0.5,
            coherence=0.5, source="teacher",
            thought_type="declarative", fork_name="declarative",
        )
        tired = validator.create_pot(
            chi_available=0.5, metabolic_drain=0.9,
            attention=0.6, i_confidence=0.5, chi_coherence=0.4,
            neuromods=_neuromods(), novelty=0.6, significance=0.5,
            coherence=0.5, source="teacher",
            thought_type="declarative", fork_name="declarative",
        )
        assert tired.pot_score < normal.pot_score

    def test_neuromod_dict_extraction(self, validator):
        flat = {"DA": 0.5, "ACh": 0.4, "NE": 0.3, "5HT": 0.6, "GABA": 0.2, "endorphin": 0.1}
        result = validator.get_neuromod_dict(flat)
        assert result["DA"] == 0.5
        assert result["endorphin"] == 0.1

    def test_neuromod_dict_nested(self, validator):
        nested = {"levels": {"DA": 0.5, "ACh": 0.4}}
        result = validator.get_neuromod_dict(nested)
        assert result["DA"] == 0.5


# ══════════════════════════════════════════════════════════════════════
# Merkle Checkpoint Tests
# ══════════════════════════════════════════════════════════════════════

class TestMerkle:
    """Test Merkle root computation and checkpointing."""

    def test_merkle_root_deterministic(self, chain):
        r1 = chain.compute_merkle_root()
        r2 = chain.compute_merkle_root()
        assert r1 == r2

    def test_merkle_root_changes_after_commit(self, chain):
        r1 = chain.compute_merkle_root()
        chain.commit_block(
            fork_id=FORK_DECLARATIVE, epoch_id=100,
            payload=_make_payload(), pot_nonce=1, chi_spent=0.005,
            neuromod_state=_neuromods(),
        )
        r2 = chain.compute_merkle_root()
        assert r1 != r2

    def test_create_checkpoint(self, chain):
        chain.commit_block(
            fork_id=FORK_DECLARATIVE, epoch_id=100,
            payload=_make_payload(), pot_nonce=1, chi_spent=0.005,
            neuromod_state=_neuromods(),
        )
        cp = chain.create_checkpoint(epoch_id=100)
        assert "merkle_root" in cp
        assert cp["total_blocks"] == 2
        assert cp["epoch_id"] == 100


# ══════════════════════════════════════════════════════════════════════
# Query Tests
# ══════════════════════════════════════════════════════════════════════

class TestQuery:
    """Test the query interface."""

    def test_query_by_thought_type(self, chain):
        chain.commit_block(
            fork_id=FORK_DECLARATIVE, epoch_id=100,
            payload=_make_payload(thought_type="declarative"),
            pot_nonce=1, chi_spent=0.005, neuromod_state=_neuromods(),
        )
        chain.commit_block(
            fork_id=FORK_EPISODIC, epoch_id=101,
            payload=_make_payload(thought_type="episodic", source="social"),
            pot_nonce=1, chi_spent=0.003, neuromod_state=_neuromods(),
        )
        results = chain.query_blocks(thought_type="declarative")
        assert len(results) == 1
        assert results[0]["thought_type"] == "declarative"

    def test_query_by_source(self, chain):
        chain.commit_block(
            fork_id=FORK_DECLARATIVE, epoch_id=100,
            payload=_make_payload(source="teacher"),
            pot_nonce=1, chi_spent=0.005, neuromod_state=_neuromods(),
        )
        results = chain.query_blocks(source="teacher")
        assert len(results) >= 1

    def test_query_by_fork(self, chain):
        for fork in [FORK_DECLARATIVE, FORK_PROCEDURAL]:
            chain.commit_block(
                fork_id=fork, epoch_id=100,
                payload=_make_payload(), pot_nonce=1, chi_spent=0.005,
                neuromod_state=_neuromods(),
            )
        results = chain.query_blocks(fork_id=FORK_PROCEDURAL)
        assert len(results) == 1

    def test_query_by_epoch_range(self, chain):
        for i in range(5):
            chain.commit_block(
                fork_id=FORK_DECLARATIVE, epoch_id=100 + i,
                payload=_make_payload(), pot_nonce=1, chi_spent=0.005,
                neuromod_state=_neuromods(),
            )
        results = chain.query_blocks(epoch_range=(102, 104))
        assert len(results) == 3

    def test_query_limit(self, chain):
        for i in range(10):
            chain.commit_block(
                fork_id=FORK_DECLARATIVE, epoch_id=100 + i,
                payload=_make_payload(), pot_nonce=1, chi_spent=0.005,
                neuromod_state=_neuromods(),
            )
        results = chain.query_blocks(fork_id=FORK_DECLARATIVE, limit=3)
        assert len(results) == 3


# ══════════════════════════════════════════════════════════════════════
# Stats + Status Tests
# ══════════════════════════════════════════════════════════════════════

class TestStats:
    """Test chain status and fork statistics."""

    def test_get_chain_status(self, chain):
        status = chain.get_chain_status()
        assert status["titan_id"] == "T_TEST"
        assert status["genesis_exists"]
        assert status["total_blocks"] == 1
        assert status["total_forks"] == 5

    def test_fork_stats(self, chain):
        chain.commit_block(
            fork_id=FORK_DECLARATIVE, epoch_id=100,
            payload=_make_payload(), pot_nonce=1, chi_spent=0.005,
            neuromod_state=_neuromods(),
        )
        stats = chain.get_fork_stats()
        assert stats[FORK_DECLARATIVE]["block_count"] == 1
        assert stats[FORK_DECLARATIVE]["total_chi_spent"] == 0.005

    def test_total_blocks_tracks_correctly(self, chain):
        assert chain.total_blocks == 1  # genesis
        chain.commit_block(
            fork_id=FORK_DECLARATIVE, epoch_id=100,
            payload=_make_payload(), pot_nonce=1, chi_spent=0.005,
            neuromod_state=_neuromods(),
        )
        chain.commit_block(
            fork_id=FORK_EPISODIC, epoch_id=101,
            payload=_make_payload(thought_type="episodic"),
            pot_nonce=1, chi_spent=0.003,
            neuromod_state=_neuromods(),
        )
        assert chain.total_blocks == 3


# ══════════════════════════════════════════════════════════════════════
# Dream Compaction Tests
# ══════════════════════════════════════════════════════════════════════

class TestDreamCompaction:
    """Test sidechain compaction during dream cycles."""

    def test_compactable_sidechains_found(self, chain):
        # Create sidechain with 5+ blocks
        for i in range(3):
            chain.commit_block(
                fork_id=FORK_DECLARATIVE, epoch_id=100 + i,
                payload=_make_payload(tags=["music"]),
                pot_nonce=1, chi_spent=0.005,
                neuromod_state=_neuromods(),
            )
        sc_id = chain.get_sidechain_for_topic("music")
        assert sc_id is not None

        # Add 5 blocks to sidechain
        for i in range(5):
            chain.commit_block(
                fork_id=sc_id, epoch_id=200 + i,
                payload=_make_payload(tags=["music"], content={"note": i}),
                pot_nonce=1, chi_spent=0.004,
                neuromod_state=_neuromods(),
            )

        compactable = chain.get_compactable_sidechains(min_blocks=5)
        assert len(compactable) == 1
        assert compactable[0]["topic"] == "music"
        assert compactable[0]["uncompacted_blocks"] == 5


# ══════════════════════════════════════════════════════════════════════
# Index Rebuild Tests
# ══════════════════════════════════════════════════════════════════════

class TestIndexRebuild:
    """Test that index can be fully rebuilt from chain files."""

    def test_rebuild_preserves_block_count(self, chain):
        for i in range(5):
            chain.commit_block(
                fork_id=FORK_DECLARATIVE, epoch_id=100 + i,
                payload=_make_payload(content={"i": i}),
                pot_nonce=1, chi_spent=0.005,
                neuromod_state=_neuromods(),
            )
        original_count = chain.total_blocks
        chain.rebuild_index()
        assert chain.total_blocks == original_count

    def test_rebuild_preserves_fork_tips(self, chain):
        chain.commit_block(
            fork_id=FORK_DECLARATIVE, epoch_id=100,
            payload=_make_payload(), pot_nonce=1, chi_spent=0.005,
            neuromod_state=_neuromods(),
        )
        original_tip = chain.get_fork_tip(FORK_DECLARATIVE)
        chain.rebuild_index()
        rebuilt_tip = chain.get_fork_tip(FORK_DECLARATIVE)
        assert original_tip[0] == rebuilt_tip[0]  # same height


# ══════════════════════════════════════════════════════════════════════
# Neuromod Coloring Tests
# ══════════════════════════════════════════════════════════════════════

class TestNeuromodColoring:
    """Test neuromodulator-colored blocks."""

    def test_neuromod_hash_varies_with_state(self, chain):
        from titan_plugin.logic.timechain import TimeChain
        # Different neuromod states should produce different hashes
        h1 = chain._compute_neuromod_hash({"DA": 0.9, "ACh": 0.1})
        h2 = chain._compute_neuromod_hash({"DA": 0.1, "ACh": 0.9})
        assert h1 != h2
        assert len(h1) == 16
        assert len(h2) == 16

    def test_neuromod_stored_in_index(self, chain):
        chain.commit_block(
            fork_id=FORK_DECLARATIVE, epoch_id=100,
            payload=_make_payload(), pot_nonce=1, chi_spent=0.005,
            neuromod_state={"DA": 0.8, "ACh": 0.6, "NE": 0.4},
        )
        results = chain.query_blocks(
            fork_id=FORK_DECLARATIVE,
            neuromod_filter={"DA": (">", 0.7)},
        )
        assert len(results) == 1
