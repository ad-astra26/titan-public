"""G9 (INV-Syn-25): SovereigntyRatioMeter durable reseed from
``synthesis.duckdb::sovereignty_marks(ts, s)`` (RFP_synthesis_decision_authority P3).

The meter's rolling-window marks are in-memory; a ``synthesis_worker`` respawn
zeros them (crash-loop audit §5.3 "respawn zeros windows"). Every per-reply S
mark is persisted to a durable ``sovereignty_marks(ts, s)`` table — via the
injected ``on_record`` callback, serialized on the SynthesisWriter (INV-Syn-28)
— and replayed in-window on boot via ``boot_seed_from_marks`` so the rolling
mean survives restart (G4).

These pin the mechanic against an in-memory marks store: the duckdb/writer
plumbing is the worker's; here ``query_fn`` + ``on_record`` are plain callables,
keeping the test pure.
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
    """rows: list of (ts, s). Returns a query_fn mirroring the worker's
    `SELECT ts, s FROM sovereignty_marks WHERE ts > ? ORDER BY ts ASC LIMIT ?`."""
    def q(since_ts, limit):
        sel = [r for r in rows if r[0] > since_ts]
        sel.sort(key=lambda r: r[0])
        return sel[:limit]
    return q


# ── on_record persistence (every live mark writes a durable row) ─────────────

def test_on_record_fires_on_live_marks():
    captured = []
    meter = SovereigntyRatioMeter(
        windows=["all"], clock=lambda: 100.0,
        on_record=lambda ts, s, e, v: captured.append((ts, s, e, v)))
    meter.record_reply(0.6, 10.0, e=0.7, v=0.2)
    meter.record_reply(0.3, 11.0)
    # The durable callback carries the full (ts, s, e, v) mark.
    assert captured == [(10.0, 0.6, 0.7, 0.2), (11.0, 0.3, 0.0, 0.0)]


def test_on_record_not_fired_on_replay():
    # Boot-seed replay must NOT re-persist — else it double-writes the rows it
    # just read back.
    captured = []
    meter = SovereigntyRatioMeter(
        windows=["all"], clock=lambda: 100.0,
        on_record=lambda ts, s, e, v: captured.append((ts, s, e, v)))
    meter.record_reply(0.5, 5.0, e=0.4, v=0.1, _persist=False)
    assert captured == []


# ── boot_seed_from_marks ─────────────────────────────────────────────────────

def test_boot_seed_replays_per_reply_s():
    rows = [(1000.0, 0.2), (1100.0, 0.4), (1200.0, 0.9)]
    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 1300.0)
    summary = boot_seed_from_marks(meter, _query_fn(rows), since_ts=0.0)

    assert summary["scanned"] == 3
    assert summary["replies"] == 3
    assert summary["capped"] is False

    stats = meter.compute(1300.0)["all"]
    assert stats["replies"] == 3                       # windows >0 after restart (G4)
    assert stats["sovereignty"] == round((0.2 + 0.4 + 0.9) / 3, 4)


def test_boot_seed_respects_since_ts():
    rows = [(100.0, 0.5), (9000.0, 0.7)]
    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 9100.0)
    summary = boot_seed_from_marks(meter, _query_fn(rows), since_ts=5000.0)
    assert summary["scanned"] == 1                     # only the row with ts > 5000
    assert summary["replies"] == 1
    assert meter.compute(9100.0)["all"]["sovereignty"] == 0.7


def test_boot_seed_cap_flag():
    rows = [(1000.0 + i, 0.5) for i in range(5)]
    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 2000.0)
    summary = boot_seed_from_marks(meter, _query_fn(rows), since_ts=0.0, cap=3)
    assert summary["capped"] is True                   # surfaced, not silent
    assert summary["scanned"] == 3
    assert summary["replies"] == 3


def test_boot_seed_query_error_returns_empty():
    def boom(since_ts, limit):
        raise RuntimeError("db down")
    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 1.0)
    summary = boot_seed_from_marks(meter, boom, since_ts=0.0)
    assert summary["scanned"] == 0
    assert summary["replies"] == 0


def test_boot_seed_bad_row_skipped():
    # Bad S value (numeric ts so the query filter is fine) → that row is skipped,
    # the good row still seeds.
    rows = [(1000.0, 0.5), (1001.0, "notafloat")]
    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 1100.0)
    summary = boot_seed_from_marks(meter, _query_fn(rows), since_ts=0.0)
    assert summary["replies"] == 1
    assert meter.compute(1100.0)["all"]["sovereignty"] == 0.5


def test_round_trip_persist_then_reseed_matches_live():
    # Full loop: live records → durable rows → fresh meter reseeds → identical
    # windows (the restart invariant, G4). The (ts, s, e, v) mark round-trips
    # intact — rolling E/V survive a respawn alongside S.
    rows = []
    live = SovereigntyRatioMeter(
        windows=["all"], clock=lambda: 1300.0,
        on_record=lambda ts, s, e, v: rows.append((ts, s, e, v)))
    live.record_reply(0.7, 1000.0, e=0.8, v=0.1)
    live.record_reply(0.3, 1100.0, e=0.2, v=0.5)
    live_stats = live.compute(1300.0)["all"]
    assert live_stats["e"] == 0.5 and live_stats["v"] == 0.3  # rolled

    reborn = SovereigntyRatioMeter(windows=["all"], clock=lambda: 1300.0)
    boot_seed_from_marks(reborn, _query_fn(rows), since_ts=0.0)
    assert reborn.compute(1300.0)["all"] == live_stats


def test_boot_seed_tolerates_legacy_two_col_rows():
    # A pre-P3 marks table is (ts, s) only — those rows must still replay, with
    # e/v defaulting to 0.0 (no migration crash, no NaN).
    rows = [(1000.0, 0.6), (1100.0, 0.4)]   # 2-tuples, no e/v
    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 1300.0)
    summary = boot_seed_from_marks(meter, _query_fn(rows), since_ts=0.0)
    assert summary["replies"] == 2
    stats = meter.compute(1300.0)["all"]
    assert stats["sovereignty"] == 0.5
    assert stats["e"] == 0.0 and stats["v"] == 0.0


# ── knowledge_moment_signal (UNCHANGED — per-item INV-Syn-23 reinforcement +
#     supplies the cited subset that feeds S) ─────────────────────────────────

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
