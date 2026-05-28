"""Phase 10 — SovereigntyRatioMeter (headline metric)."""

from titan_hcl.synthesis.sovereignty_meter import SovereigntyRatioMeter


def _meter(now=10_000.0):
    clk = {"t": now}
    m = SovereigntyRatioMeter(windows=["24h", "7d", "all"], clock=lambda: clk["t"])
    return m, clk


def test_ratio_basic():
    m, _ = _meter()
    for _ in range(10):
        m.record_knowledge_moment(ts=10_000.0)
    for _ in range(4):
        m.record_recall_satisfied(kind="cited_recall", ts=10_000.0)
    out = m.compute(now_ts=10_000.0)
    assert out["all"]["knowledge_moments"] == 10
    assert out["all"]["recall_satisfied"] == 4
    assert out["all"]["ratio"] == 0.4
    assert out["all"]["cited_recalls"] == 4


def test_zero_denominator_no_nan():
    m, _ = _meter()
    out = m.compute(now_ts=10_000.0)
    assert out["all"]["ratio"] == 0.0
    assert out["24h"]["ratio"] == 0.0


def test_window_bucketing():
    m, _ = _meter()
    now = 1_000_000.0
    # one moment now, one 2 days ago
    m.record_knowledge_moment(ts=now)
    m.record_knowledge_moment(ts=now - 2 * 24 * 3600)
    m.record_recall_satisfied(ts=now)
    out = m.compute(now_ts=now)
    assert out["24h"]["knowledge_moments"] == 1   # only the recent one
    assert out["7d"]["knowledge_moments"] == 2
    assert out["all"]["knowledge_moments"] == 2


def test_skill_vs_cited_split():
    m, _ = _meter()
    m.record_knowledge_moment(ts=10_000.0)
    m.record_knowledge_moment(ts=10_000.0)
    m.record_recall_satisfied(kind="skill_delegation", ts=10_000.0)
    m.record_recall_satisfied(kind="cited_recall", ts=10_000.0)
    out = m.compute(now_ts=10_000.0)["all"]
    assert out["skill_delegations"] == 1
    assert out["cited_recalls"] == 1
    assert out["recall_satisfied"] == 2


def test_trend_sign():
    m, _ = _meter()
    now = 1_000_000.0
    span = 24 * 3600
    # prior window: 2 moments, 0 satisfied → ratio 0
    m.record_knowledge_moment(ts=now - span - 10)
    m.record_knowledge_moment(ts=now - span - 20)
    # current window: 2 moments, 2 satisfied → ratio 1
    m.record_knowledge_moment(ts=now - 10)
    m.record_knowledge_moment(ts=now - 20)
    m.record_recall_satisfied(ts=now - 10)
    m.record_recall_satisfied(ts=now - 20)
    out = m.compute(now_ts=now)["24h"]
    assert out["ratio"] == 1.0
    assert out["trend"] == 1.0  # 1.0 - 0.0


def test_ratio_capped_at_one():
    m, _ = _meter()
    m.record_knowledge_moment(ts=10_000.0)
    m.record_recall_satisfied(ts=10_000.0)
    m.record_recall_satisfied(ts=10_000.0)  # more satisfied than moments
    out = m.compute(now_ts=10_000.0)["all"]
    assert out["ratio"] == 1.0
