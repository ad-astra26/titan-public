"""RFP_supervision_lifecycle §7.B — predictive boot back-pressure (boot_throttle).

Proves the SMART throttle decision: read box pressure (MemAvailable/swap/load) vs
a module's rss_limit, predict whether booting it would starve the box, and if so
throttle (cgroup memory.high → page to swap) rather than let it OOM-flap → DISABLE.
cgroup-unavailable falls back to predictive-defer. Everything FAIL-OPEN.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from titan_hcl.orchestrator import boot_throttle as bt
from titan_hcl.orchestrator.boot_throttle import BoxPressure


def _box(mem_avail_mb, swap_used=0.0, swap_total=16000.0, load1=1.0, ncpu=4):
    return BoxPressure(mem_avail_mb, swap_used, swap_total, load1, ncpu)


# ── predict_needs_throttle (the pure predictor) ───────────────────────────

def test_predict_throttle_when_growth_would_starve_box():
    # 200MB free, module wants to grow to 700MB → projected -500MB < 256 floor.
    assert bt.predict_needs_throttle(_box(200), rss_limit_mb=700, current_rss_mb=0) is True


def test_no_throttle_when_ample_headroom():
    # 3GB free, module needs +300MB → projected 2700MB > floor.
    assert bt.predict_needs_throttle(_box(3000), rss_limit_mb=700, current_rss_mb=400) is False


def test_no_throttle_on_bad_meminfo_read_fail_open():
    # mem_available 0 (bad read) → must NOT throttle (fail-open, no information).
    assert bt.predict_needs_throttle(_box(0.0), rss_limit_mb=700, current_rss_mb=0) is False


def test_throttle_respects_custom_floor():
    box = _box(900)  # +700 growth → projected 200
    assert bt.predict_needs_throttle(box, 700, 0, mem_floor_mb=256) is True   # 200 < 256
    assert bt.predict_needs_throttle(box, 700, 0, mem_floor_mb=100) is False  # 200 > 100


# ── read_box_pressure (parse /proc) ───────────────────────────────────────

def test_read_box_pressure_smoke():
    box = bt.read_box_pressure()
    assert isinstance(box, BoxPressure)
    assert box.mem_available_mb >= 0.0 and box.ncpu >= 1


# ── cgroup helpers (mocked fs) ────────────────────────────────────────────

def test_cgroup_unavailable_when_no_own_dir(monkeypatch):
    monkeypatch.setattr(bt, "_own_cgroup_dir", lambda: None)
    assert bt.cgroup_throttle_available() is False


def test_cgroup_available_requires_memory_in_subtree(monkeypatch, tmp_path):
    (tmp_path / "cgroup.subtree_control").write_text("cpu io memory pids\n")
    monkeypatch.setattr(bt, "_own_cgroup_dir", lambda: tmp_path)
    monkeypatch.setattr(bt.os, "access", lambda *a, **k: True)
    assert bt.cgroup_throttle_available() is True
    # memory controller absent → unavailable
    (tmp_path / "cgroup.subtree_control").write_text("cpu io pids\n")
    assert bt.cgroup_throttle_available() is False


def test_apply_memory_high_writes_cgroup_files(monkeypatch, tmp_path):
    monkeypatch.setattr(bt, "_own_cgroup_dir", lambda: tmp_path)
    ok = bt.apply_memory_high(pid=4242, module_name="synthesis", high_mb=770)
    assert ok is True
    child = tmp_path / "throttle_synthesis"
    assert child.is_dir()
    assert (child / "memory.high").read_text() == str(770 * 1024 * 1024)
    assert (child / "cgroup.procs").read_text() == "4242"


def test_apply_memory_high_fail_open_on_error(monkeypatch):
    # No cgroup dir → returns False, never raises.
    monkeypatch.setattr(bt, "_own_cgroup_dir", lambda: None)
    assert bt.apply_memory_high(1, "x", 100) is False


# ── evaluate_and_throttle (the full decision) ─────────────────────────────

def test_evaluate_no_pressure_returns_none(monkeypatch):
    monkeypatch.setattr(bt, "read_box_pressure", lambda: _box(4000))
    d = bt.evaluate_and_throttle(123, "synthesis", rss_limit_mb=700, current_rss_mb=400)
    assert d.throttled is False and d.mechanism == "none"


def test_evaluate_pressure_uses_cgroup_when_available(monkeypatch):
    monkeypatch.setattr(bt, "read_box_pressure", lambda: _box(150))
    monkeypatch.setattr(bt, "cgroup_throttle_available", lambda: True)
    monkeypatch.setattr(bt, "apply_memory_high", lambda pid, n, h: True)
    d = bt.evaluate_and_throttle(123, "synthesis", rss_limit_mb=700, current_rss_mb=100)
    assert d.throttled is True and d.mechanism == "cgroup"
    assert d.high_mb == pytest.approx(700 * bt.MEMORY_HIGH_MULTIPLE)


def test_evaluate_pressure_defers_when_cgroup_unavailable(monkeypatch):
    monkeypatch.setattr(bt, "read_box_pressure", lambda: _box(150))
    monkeypatch.setattr(bt, "cgroup_throttle_available", lambda: False)
    d = bt.evaluate_and_throttle(123, "synthesis", rss_limit_mb=700, current_rss_mb=100)
    assert d.throttled is True and d.mechanism == "defer"


def test_evaluate_pressure_defers_when_cgroup_disabled(monkeypatch):
    monkeypatch.setattr(bt, "read_box_pressure", lambda: _box(150))
    monkeypatch.setattr(bt, "cgroup_throttle_available", lambda: True)
    d = bt.evaluate_and_throttle(
        123, "synthesis", rss_limit_mb=700, current_rss_mb=100, cgroup_enabled=False)
    assert d.throttled is True and d.mechanism == "defer"


# ── guardian start-hook integration (_maybe_boot_throttle) ────────────────

import logging  # noqa: E402

from titan_hcl.bus import DivineBus  # noqa: E402
from titan_hcl.guardian_hcl import Guardian, ModuleSpec  # noqa: E402


def _heavy_guardian(name, rss_limit_mb):
    g = Guardian(DivineBus())
    g.register(ModuleSpec(
        name=name, layer="L2", entry_fn=lambda *a, **kw: None,
        autostart=False, restart_on_crash=True, rss_limit_mb=rss_limit_mb))
    return g, g._modules[name]


def test_start_hook_throttles_heavy_module_under_pressure(monkeypatch, caplog):
    """A heavy module under pressure → MODULE_BOOT_THROTTLED journal line."""
    g, info = _heavy_guardian("synthesis", rss_limit_mb=700)
    monkeypatch.setattr(bt, "read_box_pressure", lambda: _box(150))
    monkeypatch.setattr(bt, "cgroup_throttle_available", lambda: True)
    monkeypatch.setattr(bt, "apply_memory_high", lambda *a: True)
    with caplog.at_level(logging.WARNING):
        g._maybe_boot_throttle("synthesis", info, pid=4242)
    assert any("MODULE_BOOT_THROTTLED" in r.getMessage() and "mechanism=cgroup" in r.getMessage()
               for r in caplog.records), \
        f"expected throttle log; got {[r.getMessage() for r in caplog.records]}"


def test_start_hook_skips_light_module(monkeypatch, caplog):
    """A light module (rss_limit below the heavy threshold) is never throttled —
    evaluate_and_throttle is not even consulted."""
    g, info = _heavy_guardian("info_banner", rss_limit_mb=100)
    called = []
    monkeypatch.setattr(bt, "evaluate_and_throttle", lambda *a, **k: called.append(1))
    with caplog.at_level(logging.WARNING):
        g._maybe_boot_throttle("info_banner", info, pid=4242)
    assert not called, "light module must not be evaluated for throttle"
    assert not any("MODULE_BOOT_THROTTLED" in r.getMessage() for r in caplog.records)


def test_start_hook_disabled_flag_skips(monkeypatch):
    """boot_throttle_enabled=False → no evaluation at all."""
    g, info = _heavy_guardian("synthesis", rss_limit_mb=700)
    g._boot_throttle_enabled = False
    called = []
    monkeypatch.setattr(bt, "evaluate_and_throttle", lambda *a, **k: called.append(1))
    g._maybe_boot_throttle("synthesis", info, pid=4242)
    assert not called
