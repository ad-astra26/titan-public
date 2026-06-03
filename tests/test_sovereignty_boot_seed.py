"""G9 (INV-Syn-25 / Phase F): SovereigntyRatioMeter durable reseed from
``synthesis.duckdb::sovereignty_marks``.

The meter's rolling-window marks are in-memory; a ``synthesis_worker`` respawn
zeros them (crash-loop audit §5.3 "respawn zeros windows"). Phase F persists
every mark to a durable ``sovereignty_marks(ts, mark_type, kind)`` table — via
the injected ``on_record`` callback, serialized on the SynthesisWriter
(INV-Syn-28) — and replays in-window rows on boot via ``boot_seed_from_marks``.
This replaces the v0.22.0 conv-TX boot-seed (the conversation fork's v2 batch
envelopes drop per-TX content at seal → that read found 0).

These pin the mechanic against an in-memory marks store: the duckdb/writer
plumbing is the worker's; here ``query_fn`` + ``on_record`` are plain callables,
keeping the test pure. The ``kind`` is preserved end-to-end so BRAIN Phase 12
can extend the numerator (``brain_grounded``) with zero migration.
"""
from titan_hcl.synthesis.cited_use import (
    CitedUseDetector,
    SurfacedItem,
    knowledge_moment_signal,
)
from titan_hcl.synthesis.sovereignty_meter import (
    SovereigntyRatioMeter,
    boot_seed_from_marks,
)


def _query_fn(rows):
    """rows: list of (ts, mark_type, kind). Returns a query_fn mirroring the
    worker's `SELECT ... WHERE ts > ? ORDER BY ts ASC LIMIT ?`."""
    def q(since_ts, limit):
        sel = [r for r in rows if r[0] > since_ts]
        sel.sort(key=lambda r: r[0])
        return sel[:limit]
    return q


# ── on_record persistence (every live mark writes a durable row) ─────────────

def test_on_record_fires_on_live_marks_with_kind():
    captured = []
    meter = SovereigntyRatioMeter(
        windows=["all"], clock=lambda: 100.0,
        on_record=lambda mt, ts, kind: captured.append((mt, ts, kind)))
    meter.record_knowledge_moment(10.0)
    meter.record_recall_satisfied(kind="cited_recall", ts=10.0)
    meter.record_recall_satisfied(kind="skill_delegation", ts=11.0)
    # knowledge marks carry kind=None; satisfied marks preserve their kind
    # (open-ended — BRAIN later adds 'brain_grounded' through the same path).
    assert captured == [
        ("knowledge", 10.0, None),
        ("satisfied", 10.0, "cited_recall"),
        ("satisfied", 11.0, "skill_delegation"),
    ]


def test_on_record_not_fired_on_replay():
    # Boot-seed replay must NOT re-persist — else it double-writes the very rows
    # it just read back.
    captured = []
    meter = SovereigntyRatioMeter(
        windows=["all"], clock=lambda: 100.0,
        on_record=lambda mt, ts, kind: captured.append((mt, ts, kind)))
    meter.record_knowledge_moment(5.0, _persist=False)
    meter.record_recall_satisfied(kind="cited_recall", ts=5.0, _persist=False)
    assert captured == []


# ── boot_seed_from_marks ─────────────────────────────────────────────────────

def test_boot_seed_replays_knowledge_and_satisfied_with_kind():
    rows = [
        (1000.0, "knowledge", None),
        (1000.0, "satisfied", "cited_recall"),
        (1100.0, "knowledge", None),
        (1100.0, "satisfied", "skill_delegation"),
        (1200.0, "knowledge", None),
    ]
    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 1300.0)
    summary = boot_seed_from_marks(meter, _query_fn(rows), since_ts=0.0)

    assert summary["scanned"] == 5
    assert summary["knowledge_moments"] == 3
    assert summary["recall_satisfied"] == 2
    assert summary["capped"] is False

    stats = meter.compute(1300.0)["all"]
    assert stats["knowledge_moments"] == 3          # windows >0 after restart (G4)
    assert stats["recall_satisfied"] == 2
    assert stats["cited_recalls"] == 1              # per-kind breakdown survives
    assert stats["skill_delegations"] == 1
    assert stats["ratio"] == round(2 / 3, 4)


def test_boot_seed_respects_since_ts():
    rows = [
        (100.0, "knowledge", None),
        (9000.0, "knowledge", None),
    ]
    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 9100.0)
    summary = boot_seed_from_marks(meter, _query_fn(rows), since_ts=5000.0)
    assert summary["scanned"] == 1                  # only the row with ts > 5000
    assert summary["knowledge_moments"] == 1


def test_boot_seed_cap_flag():
    rows = [(1000.0 + i, "knowledge", None) for i in range(5)]
    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 2000.0)
    summary = boot_seed_from_marks(meter, _query_fn(rows), since_ts=0.0, cap=3)
    assert summary["capped"] is True                # surfaced, not silent
    assert summary["scanned"] == 3
    assert summary["knowledge_moments"] == 3


def test_boot_seed_query_error_returns_empty():
    def boom(since_ts, limit):
        raise RuntimeError("db down")
    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 1.0)
    summary = boot_seed_from_marks(meter, boom, since_ts=0.0)
    assert summary["scanned"] == 0
    assert summary["knowledge_moments"] == 0
    assert summary["recall_satisfied"] == 0


def test_boot_seed_unknown_mark_type_skipped():
    rows = [(1000.0, "knowledge", None), (1001.0, "bogus", None)]
    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 1100.0)
    summary = boot_seed_from_marks(meter, _query_fn(rows), since_ts=0.0)
    assert summary["scanned"] == 2
    assert summary["knowledge_moments"] == 1
    assert summary["recall_satisfied"] == 0


def test_round_trip_persist_then_reseed_matches_live():
    # Full loop: live records → durable rows → fresh meter reseeds → identical
    # windows (the restart invariant, G4).
    rows = []
    live = SovereigntyRatioMeter(
        windows=["all"], clock=lambda: 1300.0,
        on_record=lambda mt, ts, kind: rows.append((ts, mt, kind)))
    live.record_knowledge_moment(1000.0)
    live.record_recall_satisfied(kind="cited_recall", ts=1000.0)
    live.record_knowledge_moment(1100.0)
    live_stats = live.compute(1300.0)["all"]

    reborn = SovereigntyRatioMeter(windows=["all"], clock=lambda: 1300.0)
    boot_seed_from_marks(reborn, _query_fn(rows), since_ts=0.0)
    assert reborn.compute(1300.0)["all"] == live_stats


# ── knowledge_moment_signal (single source: live emit AND durable mark) ──────

def test_knowledge_moment_signal_needed_and_satisfied():
    det = CitedUseDetector()
    items = [SurfacedItem(item_id="tx:1", title="solana minting guide",
                          content_snippet="how to mint a solana nft")]
    needed, satisfied, cited = knowledge_moment_signal(
        det, items, "Here is how solana minting works.")
    assert needed is True
    assert satisfied is True
    assert cited == ["tx:1"]


def test_knowledge_moment_signal_needed_not_satisfied():
    det = CitedUseDetector()
    items = [SurfacedItem(item_id="tx:1", title="quantum chromodynamics",
                          content_snippet="gluon color confinement")]
    needed, satisfied, cited = knowledge_moment_signal(
        det, items, "I like cats and dogs.")
    assert needed is True
    assert satisfied is False
    assert cited == []


def test_knowledge_moment_signal_not_needed_when_empty():
    det = CitedUseDetector()
    needed, satisfied, cited = knowledge_moment_signal(det, [], "anything")
    assert needed is False
    assert satisfied is False
    assert cited == []
