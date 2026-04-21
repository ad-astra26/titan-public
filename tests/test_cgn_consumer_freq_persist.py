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

from titan_plugin.logic.cgn import ConceptGroundingNetwork, CGNConsumerConfig


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
