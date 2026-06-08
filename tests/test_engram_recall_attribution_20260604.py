"""RFP §7.E.0 — per-Engram citation attribution (E's prerequisite + live `fluent`).

Covers `titan_hcl/synthesis/recall_attribution.py` + the fluent-axis path:
  • membership reverse-index persist + idempotency
  • resolve = latest-version-only credit (a tx in v1 & v2 → v2 only)
  • record_recall: per-turn counters (each Engram surfaced once / cited once,
    NOT per member tx) + an event row only for cited Engrams
  • fluent_map = cited/(surfaced+k)
  • `reduce_population_to_scalars` admits `fluent` into the blend once it varies
  • `compute_axes` carries the `fluent` input

These exercise the new module directly (the chat path runs through MemoryProxy +
the bus — offline we drive core, per the §7.E.0 data-flow).
"""
from __future__ import annotations

import duckdb
import pytest

from titan_hcl.synthesis.recall_attribution import RecallAttribution
from titan_hcl.synthesis.engram_store import (
    compute_axes,
    reduce_population_to_scalars,
)


class _DirectWriter:
    """Test double for `SynthesisWriter` — runs each unit synchronously on the
    caller (production serializes the SAME closures on the writer thread, so
    correctness is identical; only the thread differs)."""

    def submit(self, fn):
        return fn()

    def submit_sync(self, fn):
        return fn()


@pytest.fixture()
def attribution():
    conn = duckdb.connect(":memory:")
    attr = RecallAttribution(conn, _DirectWriter())
    assert attr.ensure_schema() is True
    return attr, conn


# ── Membership ───────────────────────────────────────────────────────────
def test_membership_persist_and_idempotent(attribution):
    attr, conn = attribution
    attr.record_membership("glacier", 2, ["tx_a", "tx_b"])
    attr.record_membership("glacier", 2, ["tx_a", "tx_b"])  # ON CONFLICT DO NOTHING
    rows = conn.execute(
        "SELECT engram_id, version, member_tx_hash FROM engram_members "
        "ORDER BY member_tx_hash").fetchall()
    assert rows == [("glacier", 2, "tx_a"), ("glacier", 2, "tx_b")]


def test_resolve_latest_version_only(attribution):
    attr, _ = attribution
    # tx_a composes glacier v1 AND v2 → credit v2 only (Maker: latest-version-only).
    attr.record_membership("glacier", 1, ["tx_a"])
    attr.record_membership("glacier", 2, ["tx_a"])
    attr.record_membership("gokart", 1, ["tx_c"])
    got = attr._resolve_set(["tx_a", "tx_c", "tx_unknown"])
    assert got == {("glacier", 2), ("gokart", 1)}


# ── Record recall ────────────────────────────────────────────────────────
def test_record_recall_counters_and_events(attribution):
    attr, conn = attribution
    attr.record_membership("glacier", 2, ["tx_a", "tx_b"])
    attr.record_membership("gokart", 1, ["tx_c"])
    # One turn surfaces tx_a,tx_b(→glacier) + tx_c(→gokart); the LLM cites tx_a.
    attr.record_recall(["tx_a", "tx_b", "tx_c"], ["tx_a"], ts=100.0)
    stats = {(r[0], r[1]): (r[2], r[3]) for r in conn.execute(
        "SELECT engram_id, version, surfaced_count, cited_count "
        "FROM engram_recall_stats").fetchall()}
    # glacier counts ONCE surfaced (not 2× for its 2 members) + once cited;
    # gokart surfaced, not cited.
    assert stats[("glacier", 2)] == (1, 1)
    assert stats[("gokart", 1)] == (1, 0)
    # EVERY surfaced Engram logs a training event — glacier cited=True, gokart
    # cited=False (the negative class §7.E's combiner needs to discriminate).
    events = sorted(conn.execute(
        "SELECT engram_id, version, cited FROM engram_recall_events").fetchall())
    assert events == [("glacier", 2, True), ("gokart", 1, False)]


def test_negative_events_accumulate_as_training_tuples(attribution):
    # §7.E.0 fix: surfaced-but-not-cited recalls must accrue cited=False events
    # so §7.E's combiner has BOTH classes (the prior bug logged only positives →
    # a constant label → the learner self-gated off forever).
    attr, conn = attribution
    attr.record_membership("glacier", 2, ["tx_a"])
    attr.record_membership("gokart", 1, ["tx_c"])
    # 2 turns cite glacier; 3 turns surface gokart without citing it.
    attr.record_recall(["tx_a"], ["tx_a"], ts=1.0)
    attr.record_recall(["tx_a"], ["tx_a"], ts=2.0)
    attr.record_recall(["tx_c"], [], ts=3.0)
    attr.record_recall(["tx_c"], [], ts=4.0)
    attr.record_recall(["tx_c"], [], ts=5.0)
    counts = dict(conn.execute(
        "SELECT cited, COUNT(*) FROM engram_recall_events GROUP BY cited").fetchall())
    assert counts.get(True) == 2     # glacier positives
    assert counts.get(False) == 3    # gokart negatives — the decision-boundary signal


def test_record_recall_accumulates_across_turns(attribution):
    attr, conn = attribution
    attr.record_membership("glacier", 2, ["tx_a"])
    attr.record_recall(["tx_a"], ["tx_a"], ts=1.0)   # surfaced+cited
    attr.record_recall(["tx_a"], ["tx_a"], ts=2.0)   # surfaced+cited
    attr.record_recall(["tx_a"], [], ts=3.0)         # surfaced only
    row = conn.execute(
        "SELECT surfaced_count, cited_count FROM engram_recall_stats "
        "WHERE engram_id='glacier' AND version=2").fetchone()
    assert row == (3, 2)


def test_record_recall_empty_is_noop(attribution):
    attr, conn = attribution
    attr.record_recall([], [], ts=1.0)
    assert conn.execute("SELECT count(*) FROM engram_recall_stats").fetchone()[0] == 0


# ── Fluent feed ──────────────────────────────────────────────────────────
def test_fluent_map_rate(attribution):
    attr, _ = attribution
    attr.record_membership("glacier", 2, ["tx_a"])
    attr.record_membership("gokart", 1, ["tx_c"])
    attr.record_recall(["tx_a"], ["tx_a"], ts=1.0)
    attr.record_recall(["tx_a"], ["tx_a"], ts=2.0)
    attr.record_recall(["tx_a"], [], ts=3.0)          # glacier: surfaced 3, cited 2
    attr.record_recall(["tx_c"], [], ts=4.0)
    attr.record_recall(["tx_c"], [], ts=5.0)          # gokart: surfaced 2, cited 0
    fmap = attr.fluent_map(k=1.0)
    assert fmap[("glacier", 2)] == pytest.approx(2 / 4)   # cited/(surfaced+k)
    assert fmap[("gokart", 1)] == pytest.approx(0 / 3)


def test_fluent_enters_blend_once_it_varies():
    # `used` flat across the population (variance-gated OUT); only `fluent` varies →
    # it alone drives the discriminating scalar (the whole point of §7.E.0).
    rows = [
        {"key": "glacier", "used": 0.5, "verified": 0.0, "felt": 0.0, "fluent": 0.5},
        {"key": "gokart", "used": 0.5, "verified": 0.0, "felt": 0.0, "fluent": 0.0},
    ]
    scalars = reduce_population_to_scalars(rows)
    assert scalars["glacier"] > scalars["gokart"]


def test_fluent_flat_population_does_not_discriminate():
    # All axes flat → nothing to discriminate on (no spurious ordering).
    rows = [
        {"key": "a", "used": 0.5, "verified": 0.0, "felt": 0.0, "fluent": 0.0},
        {"key": "b", "used": 0.5, "verified": 0.0, "felt": 0.0, "fluent": 0.0},
    ]
    scalars = reduce_population_to_scalars(rows)
    # Empty (caller KEEPS the prior scalar) or equal — never a fabricated split.
    assert scalars == {} or scalars["a"] == scalars["b"]


def test_compute_axes_carries_fluent():
    axes = compute_axes(provisional=0.4, oracle_evidence=0, felt_coverage=0.0,
                        fluent=0.75)
    assert axes["fluent"] == pytest.approx(0.75)
    assert axes["used"] == pytest.approx(0.4)


# ── Axes cache (event snapshot source) ───────────────────────────────────
def test_update_axes_cache_feeds_event_snapshot(attribution):
    attr, conn = attribution
    attr.record_membership("glacier", 2, ["tx_a"])
    attr.update_axes_cache([
        {"concept_id": "glacier", "version": 2,
         "used": 0.8, "verified": 0.3, "felt": 0.1, "fluent": 0.5}])
    attr.record_recall(["tx_a"], ["tx_a"], ts=10.0)
    ev = conn.execute(
        "SELECT axis_used, axis_verified, axis_felt, axis_fluent, cited "
        "FROM engram_recall_events").fetchone()
    assert ev == (pytest.approx(0.8), pytest.approx(0.3),
                  pytest.approx(0.1), pytest.approx(0.5), True)
