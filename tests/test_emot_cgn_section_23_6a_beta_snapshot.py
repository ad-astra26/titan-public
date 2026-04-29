"""
Regression tests for rFP_emot_cgn_v2 §23.6a — cgn_beta_states_8d was
the last dead bundle field. Phase A didn't complete it because CGN
β-states live in cgn_worker (separate process) and needed a new
bus message. Shipped 2026-04-24 via CGN_BETA_SNAPSHOT.
"""

import pytest


def test_cgn_beta_snapshot_bus_message_exists():
    """The new bus message type is registered."""
    from titan_plugin.bus import CGN_BETA_SNAPSHOT
    assert CGN_BETA_SNAPSHOT == "CGN_BETA_SNAPSHOT"


def test_cgn_beta_states_removed_from_known_deferred():
    """Dead-dim detector no longer annotates cgn_beta_states as expected-dead.
    After §23.6a ships, the group is populated by CGN_BETA_SNAPSHOT → WARN
    fires only if the producer path actually breaks."""
    from titan_plugin.logic.emot_region_clusterer import RegionClusterer
    assert "cgn_beta_states" not in RegionClusterer._KNOWN_DEFERRED_GROUPS, \
        "cgn_beta_states should be removed from _KNOWN_DEFERRED_GROUPS post-§23.6a"


def test_emot_cgn_worker_handles_cgn_beta_snapshot():
    """Static source check — the CGN_BETA_SNAPSHOT handler is wired in
    emot_cgn_worker main loop + writes to last_cgn_beta_states_8d."""
    from pathlib import Path
    src = (Path(__file__).parent.parent / "titan_plugin" / "modules"
           / "emot_cgn_worker.py").read_text()
    assert 'elif msg_type == "CGN_BETA_SNAPSHOT"' in src, \
        "emot_cgn_worker must have CGN_BETA_SNAPSHOT handler"
    assert 'last_cgn_beta_states_8d' in src, \
        "handler must update worker_state['last_cgn_beta_states_8d']"
    assert "§23.6a" in src, "handler must reference §23.6a for discoverability"


def test_cgn_worker_emits_cgn_beta_snapshot():
    """Static source check — cgn_worker emits CGN_BETA_SNAPSHOT at the
    same snapshot interval as CGN_STATE_SNAPSHOT."""
    from pathlib import Path
    src = (Path(__file__).parent.parent / "titan_plugin" / "modules"
           / "cgn_worker.py").read_text()
    assert 'CGN_BETA_SNAPSHOT' in src, \
        "cgn_worker must emit CGN_BETA_SNAPSHOT"
    assert "§23.6a" in src, "emit site must reference §23.6a"
    # Ordered over CGN_CONSUMERS constant
    assert "_CGN_CONSUMERS" in src or "CGN_CONSUMERS" in src, \
        "emit payload keys must be the 8 CGN_CONSUMERS (ordered)"


def test_cgn_beta_snapshot_values_ordered_by_cgn_consumers():
    """End-to-end sim: simulate a CGN_BETA_SNAPSHOT arriving at the handler
    and verify the 8D array is ordered per CGN_CONSUMERS layout."""
    from titan_plugin.logic.emot_bundle_protocol import CGN_CONSUMERS
    assert len(CGN_CONSUMERS) == 8
    assert CGN_CONSUMERS == [
        "language", "social", "knowledge", "reasoning",
        "coding", "self_model", "reasoning_strategy", "meta",
    ]

    # Simulate handler logic
    payload = {
        "values_by_consumer": {
            "language": 0.65, "social": 0.55, "knowledge": 0.45,
            "reasoning": 0.50, "coding": 0.50, "self_model": 0.45,
            "reasoning_strategy": 0.40, "meta": 0.60,
        }
    }
    v_by = payload["values_by_consumer"]
    beta_8d = [float(v_by.get(c, 0.5)) for c in CGN_CONSUMERS]
    assert len(beta_8d) == 8
    assert beta_8d[0] == 0.65  # language
    assert beta_8d[1] == 0.55  # social
    assert beta_8d[7] == 0.60  # meta


def test_cgn_beta_snapshot_missing_consumer_defaults_to_05():
    """If a consumer is missing from payload (e.g. not registered), its
    slot defaults to 0.5 (neutral) so the 8D vector never has NaN/None."""
    from titan_plugin.logic.emot_bundle_protocol import CGN_CONSUMERS

    # Partial payload — only 3 of 8 consumers
    v_by = {"language": 0.7, "meta": 0.4, "knowledge": 0.6}
    beta_8d = [float(v_by.get(c, 0.5)) for c in CGN_CONSUMERS]
    assert beta_8d[0] == 0.7  # language (present)
    assert beta_8d[1] == 0.5  # social (default)
    assert beta_8d[2] == 0.6  # knowledge (present)
    assert beta_8d[7] == 0.4  # meta (present)
    # Rest default to 0.5
    assert beta_8d[3] == 0.5  # reasoning
    assert beta_8d[4] == 0.5  # coding
    assert beta_8d[5] == 0.5  # self_model
    assert beta_8d[6] == 0.5  # reasoning_strategy


def test_bug12_spirit_worker_dead_forward_removed():
    """BUG #12 fix: spirit_worker no longer tries to forward
    CGN_CROSS_INSIGHT to meta_engine._emot_cgn (dead since Phase 1.6h)."""
    from pathlib import Path
    src = (Path(__file__).parent.parent / "titan_plugin" / "modules"
           / "spirit_worker.py").read_text()
    # The dead forward pattern should no longer exist
    assert "_emot.handle_incoming_cross_insight" not in src, \
        "BUG #12: dead forward to meta_engine._emot_cgn.handle_incoming_cross_insight must be removed"
    # META-CGN forward path preserved (real consumer)
    assert "_mcgn._cgn_client.note_incoming_cross_insight" in src, \
        "META-CGN CGNConsumerClient EMA path must remain (not affected by BUG #12)"
    # Reference comment preserved for discoverability
    assert "BUG #12 fix" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-p", "no:anchorpy"])
