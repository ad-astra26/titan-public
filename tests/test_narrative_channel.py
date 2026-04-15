"""Tests for NarrativeChannel + MakerProfile — Tier 3 bond layer.

Verifies:
  - NarrativeChannel emits MAKER_NARRATION_REQUEST via bus
  - NarrativeChannel swallows bus errors
  - MakerProfile creates dialogue table and stores entries
  - MakerProfile computes bond health metrics
  - MakerProfile formats dialogue for INTROSPECT
"""
import os
import tempfile
import time

from titan_plugin.maker.narrative_channel import NarrativeChannel
from titan_plugin.maker.maker_profile import MakerProfile
from titan_plugin.maker.schemas import ProposalType


class FakeBus:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)
        return 1


class BrokenBus:
    def publish(self, msg):
        raise RuntimeError("bus down")


# ── NarrativeChannel tests ──────────────────────────────────────


def test_narrative_channel_emits_narration_request():
    bus = FakeBus()
    nc = NarrativeChannel(bus=bus, src_module="test_maker")
    nc.queue_narration(
        proposal_id="abc123",
        proposal_type=ProposalType.CONTRACT_BUNDLE,
        title="Test bundle",
        response="approve",
        reason="this looks great to me",
    )
    assert len(bus.published) == 1
    msg = bus.published[0]
    assert msg["type"] == "MAKER_NARRATION_REQUEST"
    assert msg["payload"]["proposal_id"] == "abc123"
    assert msg["payload"]["response"] == "approve"
    assert msg["payload"]["proposal_type"] == "contract_bundle"
    assert msg["payload"]["reason"] == "this looks great to me"


def test_narrative_channel_swallows_bus_errors():
    nc = NarrativeChannel(bus=BrokenBus())
    # Should NOT raise
    nc.queue_narration(
        proposal_id="x", proposal_type=ProposalType.CONTRACT_BUNDLE,
        title="t", response="decline", reason="not the right time for this")


# ── MakerProfile tests ──────────────────────────────────────────


def _make_profile(tmp_path) -> MakerProfile:
    db_path = os.path.join(str(tmp_path), "test_maker.db")
    return MakerProfile(db_path=db_path)


def test_maker_profile_add_and_retrieve(tmp_path):
    mp = _make_profile(tmp_path)
    did = mp.add_dialogue_entry(
        proposal_id="p1", proposal_type="contract_bundle",
        response="approve", maker_reason="great work here",
        titan_narration="I feel validated by this approval.",
        grounded_words=["great", "work"],
    )
    assert len(did) == 16  # sha256[:16]
    recent = mp.get_recent_dialogue(n=5)
    assert len(recent) == 1
    assert recent[0]["proposal_id"] == "p1"
    assert recent[0]["response"] == "approve"
    assert recent[0]["titan_narration"] == "I feel validated by this approval."
    assert recent[0]["grounded_words"] == ["great", "work"]


def test_maker_profile_bond_health_empty(tmp_path):
    mp = _make_profile(tmp_path)
    health = mp.get_bond_health()
    assert health["interaction_count"] == 0
    assert health["approves"] == 0
    assert health["declines"] == 0


def test_maker_profile_bond_health_with_entries(tmp_path):
    mp = _make_profile(tmp_path)
    for i in range(5):
        mp.add_dialogue_entry(
            proposal_id=f"p{i}", proposal_type="contract_bundle",
            response="approve" if i % 2 == 0 else "decline",
            maker_reason=f"reason number {i} is detailed enough",
        )
    health = mp.get_bond_health()
    assert health["interaction_count"] == 5
    assert health["approves"] == 3
    assert health["declines"] == 2
    assert health["topic_diversity"] == 1  # all contract_bundle
    assert health["avg_reason_depth"] > 10
    assert health["recent_approval_rate"] == 0.6  # 3/5


def test_maker_profile_dialogue_for_introspect(tmp_path):
    mp = _make_profile(tmp_path)
    # Empty case
    assert "No Maker dialogue history" in mp.get_dialogue_for_introspect()
    # With entries
    mp.add_dialogue_entry(
        proposal_id="p1", proposal_type="contract_bundle",
        response="approve", maker_reason="solid architecture",
        titan_narration="My Maker trusts the foundation I built.",
    )
    text = mp.get_dialogue_for_introspect(n=5)
    assert "approved" in text
    assert "solid architecture" in text
    assert "My Maker trusts" in text


def test_maker_profile_bond_health_trajectory(tmp_path):
    """Agreement trajectory should show improvement when recent approvals increase."""
    mp = _make_profile(tmp_path)
    # First 10: all decline
    for i in range(10):
        mp.add_dialogue_entry(
            proposal_id=f"old{i}", proposal_type="wallet_action",
            response="decline", maker_reason="not ready for this yet",
        )
        time.sleep(0.01)  # ensure ordering
    # Next 10: all approve
    for i in range(10):
        mp.add_dialogue_entry(
            proposal_id=f"new{i}", proposal_type="wallet_action",
            response="approve", maker_reason="this is exactly right now",
        )
        time.sleep(0.01)
    health = mp.get_bond_health()
    assert health["agreement_trajectory"] > 0  # improving
    assert health["recent_approval_rate"] == 1.0  # last 10 all approve
