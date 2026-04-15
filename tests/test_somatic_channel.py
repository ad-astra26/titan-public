"""Tests for SomaticChannel — Tier 2 dispatch layer for Maker responses.

Verifies that:
  - emit_proposal_created publishes MAKER_PROPOSAL_CREATED via the bus
  - emit_response_received publishes MAKER_RESPONSE_RECEIVED for both
    approve and decline
  - Bus failures are caught and logged (no crash)
"""
from titan_plugin.maker.schemas import (
    ProposalRecord, ProposalStatus, ProposalType,
)
from titan_plugin.maker.somatic_channel import SomaticChannel


class FakeBus:
    """Captures published messages instead of routing them."""
    def __init__(self):
        self.published: list[dict] = []

    def publish(self, msg: dict) -> int:
        self.published.append(msg)
        return 1


def _record() -> ProposalRecord:
    return ProposalRecord(
        proposal_id="proposal_id_test_hex",
        proposal_type=ProposalType.CONTRACT_BUNDLE,
        title="Test bundle",
        description="A test bundle for somatic channel verification",
        payload_json='{"x": 1}',
        payload_hash="aabbccdd" * 8,
        created_at=1.0,
        created_epoch=42,
        requires_signature=True,
        status=ProposalStatus.PENDING,
    )


def test_somatic_channel_emits_proposal_created():
    bus = FakeBus()
    sc = SomaticChannel(bus=bus, src_module="titan_maker")
    sc.emit_proposal_created(_record())
    assert len(bus.published) == 1
    msg = bus.published[0]
    assert msg["type"] == "MAKER_PROPOSAL_CREATED"
    assert msg["src"] == "titan_maker"
    assert msg["dst"] == "all"
    assert msg["payload"]["proposal_id"] == "proposal_id_test_hex"
    assert msg["payload"]["proposal_type"] == "contract_bundle"
    assert msg["payload"]["requires_signature"] is True


def test_somatic_channel_emits_response_received_approve():
    bus = FakeBus()
    sc = SomaticChannel(bus=bus)
    sc.emit_response_received(
        proposal_id="abc123",
        proposal_type=ProposalType.CONTRACT_BUNDLE,
        response="approve",
        reason="this looks great to me",
    )
    assert len(bus.published) == 1
    msg = bus.published[0]
    assert msg["type"] == "MAKER_RESPONSE_RECEIVED"
    assert msg["payload"]["response"] == "approve"
    assert msg["payload"]["reason"] == "this looks great to me"
    assert msg["payload"]["proposal_type"] == "contract_bundle"


def test_somatic_channel_emits_response_received_decline():
    bus = FakeBus()
    sc = SomaticChannel(bus=bus)
    sc.emit_response_received(
        proposal_id="def456",
        proposal_type=ProposalType.WALLET_ACTION,
        response="decline",
        reason="not the right time for this",
    )
    assert len(bus.published) == 1
    msg = bus.published[0]
    assert msg["type"] == "MAKER_RESPONSE_RECEIVED"
    assert msg["payload"]["response"] == "decline"
    assert msg["payload"]["proposal_type"] == "wallet_action"
    assert msg["payload"]["reason"] == "not the right time for this"


def test_somatic_channel_swallows_bus_errors():
    """A failing bus must not crash the substrate — just log."""
    class BrokenBus:
        def publish(self, msg):
            raise RuntimeError("bus down")
    sc = SomaticChannel(bus=BrokenBus())
    # Should NOT raise
    sc.emit_proposal_created(_record())
    sc.emit_response_received(
        proposal_id="x", proposal_type=ProposalType.CONTRACT_BUNDLE,
        response="approve", reason="this is a test reason here",
    )
