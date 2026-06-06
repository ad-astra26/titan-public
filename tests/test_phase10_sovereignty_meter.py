"""P3 — SovereigntyRatioMeter as the rolling-mean of per-reply S (RFP_synthesis_decision_authority).

The meter now records one `S_reply ∈ [0,1]` per completed turn and reports the
rolling-mean sovereignty per window (replacing the Phase-10 count ratio).
"""

from titan_hcl.synthesis.sovereignty_meter import SovereigntyRatioMeter


def _meter(now=10_000.0):
    clk = {"t": now}
    m = SovereigntyRatioMeter(windows=["24h", "7d", "all"], clock=lambda: clk["t"])
    return m, clk


def test_mean_basic():
    m, _ = _meter()
    for s in (0.2, 0.4, 0.6, 0.8):
        m.record_reply(s, ts=10_000.0)
    out = m.compute(now_ts=10_000.0)["all"]
    assert out["replies"] == 4
    assert out["sovereignty"] == 0.5  # mean of 0.2,0.4,0.6,0.8


def test_zero_replies_no_nan():
    m, _ = _meter()
    out = m.compute(now_ts=10_000.0)
    assert out["all"]["sovereignty"] == 0.0
    assert out["all"]["replies"] == 0
    assert out["24h"]["sovereignty"] == 0.0


def test_record_clamps_out_of_range():
    m, _ = _meter()
    m.record_reply(1.5, ts=10_000.0)   # clamps to 1.0
    m.record_reply(-0.3, ts=10_000.0)  # clamps to 0.0
    out = m.compute(now_ts=10_000.0)["all"]
    assert out["replies"] == 2
    assert out["sovereignty"] == 0.5


def test_window_bucketing():
    m, _ = _meter()
    now = 1_000_000.0
    m.record_reply(0.8, ts=now)
    m.record_reply(0.2, ts=now - 2 * 24 * 3600)  # 2 days ago
    out = m.compute(now_ts=now)
    assert out["24h"]["replies"] == 1
    assert out["24h"]["sovereignty"] == 0.8       # only the recent reply
    assert out["7d"]["replies"] == 2
    assert out["7d"]["sovereignty"] == 0.5        # mean of both
    assert out["all"]["replies"] == 2


def test_trend_sign():
    m, _ = _meter()
    now = 1_000_000.0
    span = 24 * 3600
    # prior window mean 0.2; current window mean 0.8 → trend +0.6
    m.record_reply(0.2, ts=now - span - 10)
    m.record_reply(0.2, ts=now - span - 20)
    m.record_reply(0.8, ts=now - 10)
    m.record_reply(0.8, ts=now - 20)
    out = m.compute(now_ts=now)["24h"]
    assert out["sovereignty"] == 0.8
    assert out["trend"] == 0.6  # 0.8 - 0.2


def test_all_window_has_no_trend():
    m, _ = _meter()
    m.record_reply(0.5, ts=10_000.0)
    assert m.compute(now_ts=10_000.0)["all"]["trend"] is None
