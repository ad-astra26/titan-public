"""RFP_cgn_loop_closure §7.A / G10 (INV-PERSIST) — DynamicRewardAccumulator
persistence + full-chain multi-step credit.

The emergent-reward accumulator was in-memory-only: every restart reset
`_total_outcomes` → 0 (α-ramp back to cold-start) and dropped every learned
(consumer, primitive, sub_mode) rolling mean. These tests pin:
  1. save_all → load roundtrip preserves total_outcomes + rolling means +
     per-tuple counts + per-consumer pos/neg + time-escape anchors.
  2. A reloaded accumulator resumes the α-ramp WARM (total_outcomes survives,
     so α is not forced back to the warm-up tier) — the restart-survival check.
  3. ingest_outcome_record with a primitive_sequence applies the outcome to
     EVERY (consumer, primitive) tuple (record_outcome multi-step credit), the
     ARC-4 path; without one it falls back to single-step (back-compat).
  4. Persistence is a no-op (returns False, never raises) when no path is set.

Run: python -m pytest tests/test_meta_dynamic_rewards_persist.py -v -p no:anchorpy
"""
from __future__ import annotations

import os
import tempfile

from titan_hcl.logic.meta_dynamic_rewards import DynamicRewardAccumulator


def _populate(acc):
    """Feed a known set of outcomes so there is real state to persist."""
    # Multi-step chain (knowledge): HYPOTHESIZE→RECALL→SYNTHESIZE, reward 0.4
    acc.record_outcome("knowledge", ["HYPOTHESIZE", "RECALL", "SYNTHESIZE"],
                       None, 0.4)
    # Single step (language): SYNTHESIZE, reward 0.2
    acc.record_single_step("language", "SYNTHESIZE", "_all", 0.2)
    # A negative one (social) to exercise neg_count
    acc.record_single_step("social", "FORMULATE", "_all", -0.5)


def test_meta_dynamic_rewards_persist_roundtrip():
    """save_all → fresh load reproduces total_outcomes, rolling means, counts,
    and per-consumer pos/neg counters exactly."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "reasoning", "meta_dynamic_rewards.json")
        acc = DynamicRewardAccumulator(alpha_ramp_enabled=True, save_path=path)
        _populate(acc)
        assert acc.save_all() is True
        assert os.path.exists(path)

        stats_before = acc.get_stats()

        # Fresh accumulator, same path → load.
        acc2 = DynamicRewardAccumulator(alpha_ramp_enabled=True, save_path=path)
        assert acc2._total_outcomes == 0            # truly fresh pre-load
        assert acc2.load() is True

        stats_after = acc2.get_stats()
        assert stats_after["total_outcomes"] == stats_before["total_outcomes"]
        assert stats_after["tuples_tracked"] == stats_before["tuples_tracked"]
        # A specific learned mean survives.
        assert acc2._rolling_mean[("knowledge", "HYPOTHESIZE", "_all")] == \
            acc._rolling_mean[("knowledge", "HYPOTHESIZE", "_all")]
        assert acc2._count[("knowledge", "RECALL", "_all")] == 1
        assert acc2._neg_count["social"] == 1
        assert acc2._pos_count["knowledge"] == 1


def test_load_resumes_alpha_ramp_warm():
    """The restart-survival check: total_outcomes survives a reload so the
    α-ramp does NOT reset to cold-start (the INV-PERSIST bug)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "meta_dynamic_rewards.json")
        acc = DynamicRewardAccumulator(alpha_ramp_enabled=True, save_path=path)
        # Push past the warm-up tier boundary (default 500) into phase_0.
        acc._total_outcomes = 600
        assert acc.current_phase() == "phase_0"
        assert acc.save_all() is True

        acc2 = DynamicRewardAccumulator(alpha_ramp_enabled=True, save_path=path)
        # Without load, a fresh accumulator would be cold (warm_up / α=0.10).
        assert acc2.current_phase() == "warm_up"
        acc2.load()
        # After load it resumes where it was — NOT reset to cold.
        assert acc2._total_outcomes == 600
        assert acc2.current_phase() == "phase_0"


def test_ingest_outcome_record_multistep_credit():
    """ARC-4 path: a record carrying primitive_sequence credits EVERY
    (consumer, primitive) tuple (record_outcome), not just one."""
    acc = DynamicRewardAccumulator(alpha_ramp_enabled=True)
    n = acc.ingest_outcome_record({
        "consumer_id": "knowledge",
        "outcome_reward": 0.5,
        "primitive_sequence": ["HYPOTHESIZE", "RECALL", "DELEGATE"],
    })
    assert n == 1
    # All three primitives got the credit (multi-step), each n=1, mean=0.5.
    for p in ("HYPOTHESIZE", "RECALL", "DELEGATE"):
        assert acc._count[("knowledge", p, "_all")] == 1
        assert acc._rolling_mean[("knowledge", p, "_all")] == 0.5
    # Exactly one outcome counted toward the α-ramp (chain = one outcome).
    assert acc._total_outcomes == 1


def test_ingest_outcome_record_single_step_backcompat():
    """Without a primitive_sequence, ingest falls back to single-step on
    actual_primitive_used (back-compat with the pre-ARC-4 record shape)."""
    acc = DynamicRewardAccumulator(alpha_ramp_enabled=True)
    n = acc.ingest_outcome_record({
        "consumer_id": "social",
        "outcome_reward": -0.3,
        "actual_primitive_used": "FORMULATE",
    })
    assert n == 1
    assert acc._count[("social", "FORMULATE", "_all")] == 1
    assert acc._neg_count["social"] == 1


def test_persistence_disabled_without_path_is_safe():
    """No save_path → save_all/load are no-ops returning False, never raising."""
    acc = DynamicRewardAccumulator(alpha_ramp_enabled=True)  # no save_path
    assert acc.save_all() is False
    assert acc.load() is False


def test_load_missing_file_is_safe():
    """load on a path that doesn't exist yet leaves a fresh accumulator
    intact and returns False (first boot, before any save)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "nope.json")
        acc = DynamicRewardAccumulator(alpha_ramp_enabled=True, save_path=path)
        assert acc.load() is False
        assert acc._total_outcomes == 0
