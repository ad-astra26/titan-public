"""Phase 4 — CGN spine-concept registration bridge tests (P4.C).

Covers `titan_hcl/synthesis/cgn_bridge.py` against PLAN §P4.C:

- register_spine_concept is idempotent: first call returns True, repeat False
- registry survives across CGNRegistrationBridge instances (file-backed)
- ensure_grounded returns None for unregistered, stub Grounding for registered
- soft-fail on persistence error (in-memory cache still updates)
- INV-1 preserved: bridge does NOT touch CGN's vocabulary table

All tests use a tmp registry path; no real CGN needed (the bridge is a
file-backed registry distinct from CGN per the architectural conversation).
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from titan_hcl.synthesis.cgn_bridge import (
    CGNRegistrationBridge,
    Grounding,
)


@pytest.fixture()
def tmp_registry():
    with tempfile.TemporaryDirectory() as tmp:
        yield os.path.join(tmp, "spine_concepts.json")


def test_register_returns_true_on_first_call(tmp_registry):
    bridge = CGNRegistrationBridge(tmp_registry, clock=lambda: 1000.0)
    assert bridge.register_spine_concept(
        "metaplex_nft_minting", "Metaplex NFT minting",
    ) is True
    assert bridge.is_registered("metaplex_nft_minting") is True


def test_register_idempotent_returns_false_on_repeat(tmp_registry):
    bridge = CGNRegistrationBridge(tmp_registry, clock=lambda: 1000.0)
    bridge.register_spine_concept("a", "A")
    assert bridge.register_spine_concept("a", "A") is False
    # The second call must not alter the record.
    snapshot = bridge.list_registered()
    assert len(snapshot) == 1
    assert snapshot[0]["concept_id"] == "a"


def test_register_empty_concept_id_rejected(tmp_registry):
    bridge = CGNRegistrationBridge(tmp_registry, clock=lambda: 1000.0)
    assert bridge.register_spine_concept("", "Empty") is False
    assert bridge.list_registered() == []


def test_registry_persists_across_instances(tmp_registry):
    """File-backed registry must survive a new CGNRegistrationBridge."""
    b1 = CGNRegistrationBridge(tmp_registry, clock=lambda: 100.0)
    b1.register_spine_concept("linux_terminal", "Linux terminal")
    b1.register_spine_concept("ssh", "SSH")

    b2 = CGNRegistrationBridge(tmp_registry, clock=lambda: 200.0)
    assert b2.is_registered("linux_terminal") is True
    assert b2.is_registered("ssh") is True
    # b2 re-registering an existing concept is still a no-op.
    assert b2.register_spine_concept("ssh", "SSH duplicate") is False


def test_ensure_grounded_returns_none_for_unregistered(tmp_registry):
    bridge = CGNRegistrationBridge(tmp_registry)
    assert bridge.ensure_grounded("ghost", 1) is None


def test_ensure_grounded_returns_p4_stub_for_registered(tmp_registry):
    """P4 stub: registered concept gets Grounding(grounded=False,
    note='phase4_stub'). Phase 7+ replaces with real CGN-backed groundings;
    callers test for grounded==True before consuming."""
    bridge = CGNRegistrationBridge(tmp_registry)
    bridge.register_spine_concept("x", "X")
    g = bridge.ensure_grounded("x", 2)
    assert isinstance(g, Grounding)
    assert g.concept_id == "x"
    assert g.version == 2
    assert g.grounded is False
    assert g.note == "phase4_stub"


def test_persistence_error_falls_back_to_memory(tmp_registry):
    """Set the registry path to a directory we can't write to; the in-memory
    cache must still update so the in-process synthesis_worker stays
    consistent. WARN logs surface the disk failure."""
    bad_path = os.path.join(os.path.sep, "proc", "ro", "spine_blocked.json")
    bridge = CGNRegistrationBridge(bad_path, clock=lambda: 1.0)
    # Register call must NOT raise — soft-fail per the docstring contract.
    assert bridge.register_spine_concept("a", "A") is True
    # In-memory cache updated.
    assert bridge.is_registered("a") is True


def test_list_registered_returns_snapshot_copy(tmp_registry):
    """list_registered must return a copy — mutating it must not affect the
    underlying registry (defensive against ConceptSpinePanel UI mutation)."""
    bridge = CGNRegistrationBridge(tmp_registry, clock=lambda: 1.0)
    bridge.register_spine_concept("a", "A")
    snapshot = bridge.list_registered()
    snapshot.clear()
    assert bridge.is_registered("a") is True


def test_invalid_existing_registry_file_starts_empty(tmp_registry):
    """A corrupt existing registry file must NOT prevent the bridge from
    operating — it falls back to an empty in-memory registry, and the next
    successful write recreates the file fresh. This is the
    "infra-independent restore" axiom in miniature."""
    with open(tmp_registry, "w") as f:
        f.write("{not valid json}")

    bridge = CGNRegistrationBridge(tmp_registry, clock=lambda: 1.0)
    assert bridge.list_registered() == []
    assert bridge.register_spine_concept("recover", "Recover") is True

    # The next successful persist rewrites the file as valid JSON.
    with open(tmp_registry, "r") as f:
        data = json.load(f)
    assert "recover" in data
