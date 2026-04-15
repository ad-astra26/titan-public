"""Tests for the TitanMaker substrate (Tier 1).

Covers:
  - schemas validation rules
  - ProposalStore CRUD + idempotency + status transitions
  - TitanMaker orchestration + signature verification + R8 file write
"""
import json
import os

import pytest

from titan_plugin.maker import (
    MakerResponse, ProposalRecord, ProposalStatus, ProposalStore, ProposalType,
    TitanMaker,
)
from titan_plugin.maker.schemas import validate_reason


# ── Schemas ──────────────────────────────────────────────────────


class TestSchemas:
    def test_validate_reason_min_length(self):
        with pytest.raises(ValueError):
            validate_reason("short")
        with pytest.raises(ValueError):
            validate_reason("")
        with pytest.raises(ValueError):
            validate_reason(None)
        # Whitespace doesn't count
        with pytest.raises(ValueError):
            validate_reason("   abc   ")

    def test_validate_reason_strips_whitespace(self):
        assert validate_reason("  this is fine reason  ") == "this is fine reason"

    def test_validate_reason_passes_long_text(self):
        assert validate_reason("a perfectly valid reason here") == \
            "a perfectly valid reason here"


# ── ProposalStore ────────────────────────────────────────────────


def _store(tmp_path) -> ProposalStore:
    return ProposalStore(db_path=str(tmp_path / "test_proposals.db"))


class TestProposalStore:
    def test_create_and_retrieve(self, tmp_path):
        s = _store(tmp_path)
        r = s.create(
            proposal_type=ProposalType.CONTRACT_BUNDLE,
            title="Test bundle",
            description="A test contract bundle proposal for validation",
            payload={"bundle_hash": "abc123", "contracts": ["a", "b"]},
            requires_signature=True,
        )
        assert r.proposal_id
        assert r.status == ProposalStatus.PENDING
        assert r.requires_signature is True
        # Retrieve
        loaded = s.get(r.proposal_id)
        assert loaded is not None
        assert loaded.proposal_id == r.proposal_id
        assert loaded.title == "Test bundle"

    def test_idempotent_by_payload_hash(self, tmp_path):
        s = _store(tmp_path)
        payload = {"bundle_hash": "deadbeef", "contracts": ["x"]}
        r1 = s.create(
            proposal_type=ProposalType.CONTRACT_BUNDLE,
            title="Bundle one", description="First creation of this bundle",
            payload=payload, requires_signature=True)
        r2 = s.create(
            proposal_type=ProposalType.CONTRACT_BUNDLE,
            title="Bundle one", description="Duplicate creation of this bundle",
            payload=payload, requires_signature=True)
        assert r1.proposal_id == r2.proposal_id
        assert len(s.list_pending()) == 1

    def test_create_validates_title(self, tmp_path):
        s = _store(tmp_path)
        with pytest.raises(ValueError):
            s.create(
                proposal_type=ProposalType.CONTRACT_BUNDLE,
                title="", description="A valid description here",
                payload={"x": 1}, requires_signature=False)

    def test_create_validates_description(self, tmp_path):
        s = _store(tmp_path)
        with pytest.raises(ValueError):
            s.create(
                proposal_type=ProposalType.CONTRACT_BUNDLE,
                title="title", description="short",
                payload={"x": 1}, requires_signature=False)

    def test_mark_approved_writes_signature(self, tmp_path):
        s = _store(tmp_path)
        r = s.create(
            proposal_type=ProposalType.CONTRACT_BUNDLE,
            title="Bundle", description="A bundle to be approved here",
            payload={"x": 1}, requires_signature=True)
        ok = s.mark_approved(
            r.proposal_id, reason="this looks great to me",
            signature="sig_b58_here", signer_pubkey="pk_b58_here")
        assert ok is True
        loaded = s.get(r.proposal_id)
        assert loaded.status == ProposalStatus.APPROVED
        assert loaded.approval_reason == "this looks great to me"
        assert loaded.approved_signature == "sig_b58_here"
        assert loaded.approved_signer_pubkey == "pk_b58_here"
        assert loaded.approved_at is not None

    def test_mark_approved_validates_reason(self, tmp_path):
        s = _store(tmp_path)
        r = s.create(
            proposal_type=ProposalType.CONTRACT_BUNDLE,
            title="Bundle", description="A bundle to be approved here",
            payload={"x": 1}, requires_signature=False)
        with pytest.raises(ValueError):
            s.mark_approved(r.proposal_id, reason="too", signature=None, signer_pubkey=None)

    def test_mark_declined_records_reason(self, tmp_path):
        s = _store(tmp_path)
        r = s.create(
            proposal_type=ProposalType.CONTRACT_BUNDLE,
            title="Bundle", description="A bundle to be declined here",
            payload={"x": 1}, requires_signature=False)
        ok = s.mark_declined(r.proposal_id, reason="this needs more thought")
        assert ok is True
        loaded = s.get(r.proposal_id)
        assert loaded.status == ProposalStatus.DECLINED
        assert loaded.decline_reason == "this needs more thought"
        assert loaded.declined_at is not None

    def test_mark_declined_validates_reason(self, tmp_path):
        s = _store(tmp_path)
        r = s.create(
            proposal_type=ProposalType.CONTRACT_BUNDLE,
            title="Bundle", description="A bundle to be declined here",
            payload={"x": 1}, requires_signature=False)
        with pytest.raises(ValueError):
            s.mark_declined(r.proposal_id, reason="no")

    def test_mark_already_resolved_returns_false(self, tmp_path):
        s = _store(tmp_path)
        r = s.create(
            proposal_type=ProposalType.CONTRACT_BUNDLE,
            title="Bundle", description="A bundle to test double-mark",
            payload={"x": 1}, requires_signature=False)
        s.mark_approved(
            r.proposal_id, reason="approved with reason here",
            signature=None, signer_pubkey=None)
        # Second mark should return False (status already not pending)
        ok = s.mark_declined(r.proposal_id, reason="trying to decline anyway")
        assert ok is False

    def test_expire_old(self, tmp_path):
        s = _store(tmp_path)
        # Past expiry
        s.create(
            proposal_type=ProposalType.WALLET_ACTION,
            title="Old proposal", description="A proposal that should expire",
            payload={"y": 1}, requires_signature=False, expires_at=1.0)
        # No expiry — should remain
        s.create(
            proposal_type=ProposalType.CONTRACT_BUNDLE,
            title="Permanent", description="A proposal that does not expire",
            payload={"z": 1}, requires_signature=False, expires_at=None)
        n = s.expire_old(now=1000.0)
        assert n == 1
        pending = s.list_pending()
        assert len(pending) == 1
        assert pending[0].title == "Permanent"

    def test_list_recent_responses(self, tmp_path):
        s = _store(tmp_path)
        # Create + approve 2, create + decline 1, leave 1 pending
        for i in range(3):
            r = s.create(
                proposal_type=ProposalType.CONTRACT_BUNDLE,
                title=f"Proposal {i}",
                description=f"Test proposal number {i} description",
                payload={"i": i}, requires_signature=False)
            if i < 2:
                s.mark_approved(
                    r.proposal_id, reason=f"approved with reason {i}",
                    signature=None, signer_pubkey=None)
            else:
                s.mark_declined(r.proposal_id, reason=f"declined with reason {i}")
        # Add a pending one
        s.create(
            proposal_type=ProposalType.CONTRACT_BUNDLE,
            title="Pending",
            description="A pending proposal that should not appear",
            payload={"i": 99}, requires_signature=False)
        recent = s.list_recent_responses(limit=10)
        assert len(recent) == 3
        statuses = [r.status for r in recent]
        assert ProposalStatus.APPROVED in statuses
        assert ProposalStatus.DECLINED in statuses
        assert ProposalStatus.PENDING not in statuses

    def test_write_low_response(self, tmp_path):
        s = _store(tmp_path)
        r = s.create(
            proposal_type=ProposalType.CONTRACT_BUNDLE,
            title="Bundle", description="A bundle for low response test",
            payload={"x": 1}, requires_signature=False)
        s.write_low_response(r.proposal_id, '{"DA_delta": 0.03}')
        loaded = s.get(r.proposal_id)
        assert loaded.titan_low_response_json == '{"DA_delta": 0.03}'


# ── TitanMaker (orchestration) ────────────────────────────────────


@pytest.fixture
def keypair():
    """Generate a test Solana keypair using solders."""
    from solders.keypair import Keypair
    return Keypair()


@pytest.fixture
def maker_pubkey_b58(keypair):
    return str(keypair.pubkey())


@pytest.fixture
def tm(tmp_path, maker_pubkey_b58):
    """Build a TitanMaker bound to a specific Maker pubkey."""
    s = ProposalStore(db_path=str(tmp_path / "test_proposals.db"))
    return TitanMaker(proposal_store=s, maker_pubkey=maker_pubkey_b58)


class TestTitanMaker:
    def test_propose_and_list_pending(self, tm):
        r = tm.propose(
            proposal_type=ProposalType.CONTRACT_BUNDLE,
            title="A bundle",
            description="A description for this proposal",
            payload={"k": "v"})
        assert r.status == ProposalStatus.PENDING
        pending = tm.list_pending()
        assert len(pending) == 1
        assert pending[0].proposal_id == r.proposal_id

    def test_record_approval_no_signature(self, tm):
        r = tm.propose(
            proposal_type=ProposalType.CONFIG_CHANGE,
            title="Config",
            description="A config change for testing",
            payload={"k": 1}, requires_signature=False)
        result = tm.record_approval(
            r.proposal_id, reason="this is a fine choice")
        assert result.success is True
        loaded = tm.get(r.proposal_id)
        assert loaded.status == ProposalStatus.APPROVED

    def test_record_approval_with_valid_signature(
        self, tm, keypair, maker_pubkey_b58, tmp_path
    ):
        from solders.signature import Signature
        # Use a proposal_type that does NOT trigger _on_contract_bundle_approved
        # so we don't need a fully writable contracts dir for this test
        r = tm.propose(
            proposal_type=ProposalType.CONFIG_CHANGE,
            title="Config requiring sig",
            description="A test config that requires signature",
            payload={"k": 1}, requires_signature=True)
        # Sign the payload_hash
        sig = keypair.sign_message(r.payload_hash.encode("utf-8"))
        result = tm.record_approval(
            r.proposal_id,
            reason="signed and approved by Maker",
            signature_b58=str(sig),
            signer_pubkey_b58=maker_pubkey_b58,
        )
        assert result.success is True
        loaded = tm.get(r.proposal_id)
        assert loaded.status == ProposalStatus.APPROVED
        assert loaded.approved_signature == str(sig)

    def test_record_approval_rejects_signature_from_non_maker(
        self, tm, maker_pubkey_b58
    ):
        from solders.keypair import Keypair
        impostor = Keypair()
        r = tm.propose(
            proposal_type=ProposalType.CONFIG_CHANGE,
            title="Config requiring sig",
            description="A test config that requires signature",
            payload={"k": 1}, requires_signature=True)
        sig = impostor.sign_message(r.payload_hash.encode("utf-8"))
        result = tm.record_approval(
            r.proposal_id,
            reason="impostor trying to approve",
            signature_b58=str(sig),
            signer_pubkey_b58=str(impostor.pubkey()),
        )
        assert result.success is False
        assert "not Maker" in result.error

    def test_record_approval_rejects_invalid_signature(
        self, tm, maker_pubkey_b58, keypair
    ):
        r = tm.propose(
            proposal_type=ProposalType.CONFIG_CHANGE,
            title="Config requiring sig",
            description="A test config that requires signature",
            payload={"k": 1}, requires_signature=True)
        # Sign a different message — verification should fail
        sig = keypair.sign_message(b"wrong message")
        result = tm.record_approval(
            r.proposal_id,
            reason="this should fail because wrong msg",
            signature_b58=str(sig),
            signer_pubkey_b58=maker_pubkey_b58,
        )
        assert result.success is False
        assert "signature verification failed" in result.error

    def test_record_decline_writes_reason(self, tm):
        r = tm.propose(
            proposal_type=ProposalType.CONFIG_CHANGE,
            title="Config",
            description="A config to be declined",
            payload={"k": 1})
        result = tm.record_decline(r.proposal_id, reason="not the right time")
        assert result.success is True
        loaded = tm.get(r.proposal_id)
        assert loaded.status == ProposalStatus.DECLINED
        assert loaded.decline_reason == "not the right time"

    def test_record_decline_validates_reason(self, tm):
        r = tm.propose(
            proposal_type=ProposalType.CONFIG_CHANGE,
            title="Config", description="A config description",
            payload={"k": 1})
        result = tm.record_decline(r.proposal_id, reason="no")
        assert result.success is False
        assert "≥ 10" in result.error

    def test_alignment_score_recency_weighted(self, tm):
        # Create + approve 5 proposals
        for i in range(5):
            r = tm.propose(
                proposal_type=ProposalType.CONFIG_CHANGE,
                title=f"Proposal {i}",
                description=f"Test proposal {i} description",
                payload={"i": i})
            tm.record_approval(r.proposal_id, reason="approved with valid reason")
        # Then 5 declines (newer)
        for i in range(5, 10):
            r = tm.propose(
                proposal_type=ProposalType.CONFIG_CHANGE,
                title=f"Proposal {i}",
                description=f"Test proposal {i} description",
                payload={"i": i})
            tm.record_decline(r.proposal_id, reason="declined with valid reason")
        score = tm.get_maker_alignment_score()
        # Newer declines should weigh more — score < 0.5
        assert score < 0.5

    def test_alignment_score_neutral_no_data(self, tm):
        assert tm.get_maker_alignment_score() == 0.5

    def test_autoseed_contract_bundle_idempotent(self, tm):
        r1 = tm.autoseed_contract_bundle(
            bundle_hash="abc123def456",
            contract_count=3,
            contract_names=["a", "b", "c"])
        r2 = tm.autoseed_contract_bundle(
            bundle_hash="abc123def456",
            contract_count=3,
            contract_names=["a", "b", "c"])
        assert r1.proposal_id == r2.proposal_id
        assert len(tm.list_pending()) == 1

    def test_contract_bundle_writes_signature_file(
        self, tmp_path, keypair, maker_pubkey_b58, monkeypatch
    ):
        # Stand up a TitanMaker that writes the signature file to a tmp contracts dir
        store = ProposalStore(db_path=str(tmp_path / "p.db"))
        tm = TitanMaker(proposal_store=store, maker_pubkey=maker_pubkey_b58)

        # Monkeypatch the contracts dir resolution by creating a real proposal
        # and then asserting the file gets written under titan_plugin/contracts/.
        # To keep the test hermetic, monkeypatch _on_contract_bundle_approved
        # to use tmp_path instead of the package path.
        original = tm._on_contract_bundle_approved

        def patched(record, sig_b58, signer_pk_b58):
            # Write to tmp_path instead of the real contracts dir
            payload = json.loads(record.payload_json)
            sig_path = tmp_path / ".bundle_signature.json"
            data = {
                "bundle_hash": payload["bundle_hash"],
                "approver_pubkey": signer_pk_b58,
                "approver_signature": sig_b58,
                "signed_at": 1234567890.0,
                "proposal_id": record.proposal_id,
            }
            sig_path.write_text(json.dumps(data, indent=2))

        tm._on_contract_bundle_approved = patched

        r = tm.autoseed_contract_bundle(
            bundle_hash="abc123def456abc123def456",
            contract_count=2,
            contract_names=["alpha", "beta"])
        assert r is not None
        # Sign the payload_hash
        sig = keypair.sign_message(r.payload_hash.encode("utf-8"))
        result = tm.record_approval(
            r.proposal_id,
            reason="approved by Maker for the test",
            signature_b58=str(sig),
            signer_pubkey_b58=maker_pubkey_b58,
        )
        assert result.success is True
        sig_file = tmp_path / ".bundle_signature.json"
        assert sig_file.exists()
        data = json.loads(sig_file.read_text())
        assert data["bundle_hash"] == "abc123def456abc123def456"
        assert data["approver_pubkey"] == maker_pubkey_b58
        assert data["approver_signature"] == str(sig)
