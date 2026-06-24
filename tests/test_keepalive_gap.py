"""Phase A keepalive warm-gap logic (RFP_load_adaptive_inference_routing §7.A).

Run:  python -m pytest tests/test_keepalive_gap.py -v -p no:anchorpy
"""
from titan_hcl.modules.agno_worker import _keepalive_gap


def test_scaled_fills_gap_to_recent_peak():
    # keep ~recent_peak warm; live chats already warm `in_flight`, fill only the gap
    assert _keepalive_gap(in_flight=0, recent_peak=5, warm_cap=8, scale=True) == 5
    assert _keepalive_gap(in_flight=3, recent_peak=5, warm_cap=8, scale=True) == 2
    assert _keepalive_gap(in_flight=8, recent_peak=5, warm_cap=8, scale=True) == 0
    assert _keepalive_gap(in_flight=6, recent_peak=5, warm_cap=8, scale=True) == 0


def test_warm_cap_bounds_width():
    # recent_peak above the cap is clamped (never warm more than warm_cap units)
    assert _keepalive_gap(in_flight=0, recent_peak=12, warm_cap=8, scale=True) == 8
    assert _keepalive_gap(in_flight=2, recent_peak=12, warm_cap=8, scale=True) == 6


def test_unscaled_keeps_one_unit():
    # scale=False → keep exactly 1 unit warm (idle → 1 ping; busy → 0)
    assert _keepalive_gap(in_flight=0, recent_peak=9, warm_cap=8, scale=False) == 1
    assert _keepalive_gap(in_flight=1, recent_peak=9, warm_cap=8, scale=False) == 0
    assert _keepalive_gap(in_flight=4, recent_peak=9, warm_cap=8, scale=False) == 0


def test_never_negative():
    assert _keepalive_gap(in_flight=20, recent_peak=3, warm_cap=8, scale=True) == 0
    assert _keepalive_gap(in_flight=-1, recent_peak=2, warm_cap=8, scale=True) == 2
