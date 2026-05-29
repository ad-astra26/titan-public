"""Regression guard for nn_iql_rl audit C9 finding.

Agent 2's audit of Component 9 (CGN Sigma) reported `_consumer_freq`
was not persisted. That finding was wrong — the dict IS saved at
cgn.py:1501 and restored at cgn.py:1582. This test is a
belt-and-braces round-trip check that the save/load actually works,
so if someone later refactors cgn._save_state / _load_state, they
don't silently reintroduce the flaw the audit worksheet warned about.

See: titan-docs/audits/COMPONENT_9_CGN_Sigma.md §Severity (corrected).
"""

from __future__ import annotations

import tempfile

import pytest

from titan_hcl.logic.cgn import ConceptGroundingNetwork, CGNConsumerConfig


@pytest.fixture
def cgn_in_tmpdir():
    with tempfile.TemporaryDirectory() as td:
        cgn = ConceptGroundingNetwork(state_dir=td)
        cgn.register_consumer(CGNConsumerConfig(
            name="test_reader",
            feature_dims=30,
            action_dims=8,
            action_names=[f"a{i}" for i in range(8)],
        ))
        cgn.register_consumer(CGNConsumerConfig(
            name="test_writer",
            feature_dims=30,
            action_dims=8,
            action_names=[f"b{i}" for i in range(8)],
        ))
        yield cgn, td


def test_consumer_freq_roundtrips_through_save_load(cgn_in_tmpdir):
    """_consumer_freq survives save + reload — prevents frequency-scaling
    factor from resetting on every Titan restart."""
    cgn, state_dir = cgn_in_tmpdir
    # Simulate a skewed runtime: test_reader has fired many more times
    cgn._consumer_freq["test_reader"] = 487
    cgn._consumer_freq["test_writer"] = 23

    cgn._save_state()

    # Fresh instance pointing to same state_dir — mimics restart
    fresh = ConceptGroundingNetwork(state_dir=state_dir)
    fresh._load_state()

    assert fresh._consumer_freq.get("test_reader") == 487, (
        "_consumer_freq['test_reader'] did not survive reload — "
        "frequency-scaling factors will reset on restart")
    assert fresh._consumer_freq.get("test_writer") == 23


def test_consumer_freq_empty_is_safe(cgn_in_tmpdir):
    """Fresh CGN with no frequency data saves + loads a clean empty dict
    (no attribute error when the Sigma update path reads it)."""
    cgn, state_dir = cgn_in_tmpdir
    # Don't touch _consumer_freq; it should still be persistable as {}
    assert cgn._consumer_freq == {}
    cgn._save_state()

    fresh = ConceptGroundingNetwork(state_dir=state_dir)
    fresh._load_state()
    assert fresh._consumer_freq == {}


def test_haov_sidecar_includes_consumer_freq(cgn_in_tmpdir):
    """The torch-free JSON sidecar (read by the api/v6 cgn-haov-stats
    handler) must carry consumer_freq under the reserved "_consumer_freq"
    key — otherwise /v6/cognition/cgn-haov-stats reports {} for it even
    though the in-memory counter is healthy
    (BUG-CGN-CONSUMER-FREQ-INVISIBLE-VIA-API-SIDECAR-20260526)."""
    import json
    import os

    cgn, state_dir = cgn_in_tmpdir
    cgn._consumer_freq["test_reader"] = 312
    cgn._consumer_freq["test_writer"] = 9
    cgn._save_state()

    sidecar = os.path.join(state_dir, "haov_stats.json")
    assert os.path.exists(sidecar), "sidecar should be written when consumer_freq is populated"
    with open(sidecar) as fh:
        payload = json.load(fh)

    assert "_consumer_freq" in payload, "sidecar dropped consumer_freq"
    assert payload["_consumer_freq"].get("test_reader") == 312
    assert payload["_consumer_freq"].get("test_writer") == 9

    # The reserved key must not collide with a real consumer name so the
    # API reader's pop()-before-iterate stays correct.
    assert not payload["_consumer_freq"].get("stats"), (
        "_consumer_freq must be a flat consumer→count map, not a HAOV entry")
