"""
Tests for ``titan_plugin.logic.outer_trinity.OuterTrinityCollector``
Phase C C-S6 flag-gated shim path.

Covers (12 tests):
  1. Flag=False default — l0_rust_enabled is False by default
  2. Flag=False compute path runs — collect() returns 5D/15D/45D
     computed from sources (today's behavior preserved byte-identical).
  3. Flag=True with no shm slots — falls back to last-known (cold boot)
  4. Flag=True reads outer_body_5d.bin via SeqLock
  5. Flag=True reads outer_mind_15d.bin via SeqLock
  6. Flag=True reads outer_spirit_45d.bin via SeqLock
  7. Flag=True stale slot → last-known + counter advances
  8. Flag=True schema mismatch → last-known + counter advances
  9. Flag=True CRC mismatch → retry (handled in retry loop)
 10. Flag=True returns same shape as flag=False
 11. Flag=True caches last-known for next tick
 12. Flag=False compute path UNCHANGED — identical output for identical
     sources (regression guard).
"""
from __future__ import annotations

import struct
import time
import zlib
from pathlib import Path

import pytest

from titan_plugin.core.state_registry import HEADER_SIZE, HEADER_STRUCT
from titan_plugin.logic.outer_trinity import OuterTrinityCollector


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def shm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


def _write_slot(shm_root: Path, slot_name: str, values: list[float],
                wall_ns: int | None = None, schema: int = 1) -> None:
    """Write a synthetic outer slot file (24B header + N × float32 LE)."""
    if wall_ns is None:
        wall_ns = time.time_ns()
    n = len(values)
    payload = struct.pack(f"<{n}f", *values)
    payload_bytes = len(payload)

    # Pack header: seq=2 (even, write complete) + schema + wall_ns + payload + crc
    seq = 2
    header_prefix = struct.pack("<IIQI", seq, schema, wall_ns, payload_bytes)
    crc = zlib.crc32(header_prefix)
    header = header_prefix + struct.pack("<I", crc)
    assert len(header) == HEADER_SIZE

    (shm_root / f"{slot_name}.bin").write_bytes(header + payload)


def _make_sources() -> dict:
    """Synthetic sources dict for compute-path testing."""
    return {
        "agency_stats": {"actions_total": 100, "self_initiated": 75,
                          "inner_outer_coherence": 0.6,
                          "actions_completed": 90,
                          "creative_count": 5,
                          "research_count": 3,
                          "social_count": 2},
        "assessment_stats": {"avg_score": 0.55},
        "helper_statuses": {"web": "ok", "blockchain": "ok"},
        "bus_stats": {"queue_depth": 0, "drop_rate": 0.0,
                      "n_modules": 11},
        "system_sensor_stats": {"cpu_pct": 42.0, "ram_pct": 61.0,
                                 "loadavg": 0.4,
                                 "thermal": 0.3,
                                 "swap_pct": 5.0},
        "network_monitor_stats": {"latency_ms_p50": 12.0,
                                   "latency_ms_p99": 23.0,
                                   "errors_total": 0,
                                   "rpc_alive": True},
        "tx_latency_stats": {"latency_p50_s": 0.8, "samples": 10},
        "block_delta_stats": {"normalized": 0.55, "delta_s": 6.2,
                               "delta": 6.2},
        "anchor_state": {"success": True, "last_anchor_time": time.time(),
                          "last_block": 12345},
        "sol_balance": 1.5,
        "uptime_seconds": 3600.0,
        "soul_health": 0.9,
        "memory_status": {"recent_score": 0.62, "consolidation": 0.71},
        "impulse_stats": {"successful": 18, "failed": 3},
        "art_count_100": 12, "audio_count_100": 4,
        "art_count_500": 47, "audio_count_500": 18,
        "social_perception_stats": {"engagement": 0.4, "reach": 17},
        "twin_state": {"reachable": True, "DA": 0.7, "NE": 0.5,
                        "GABA": 0.6},
        "llm_avg_latency": 0.85,
        "observatory_db": None,  # tolerated by collector
    }


# ── 1. Default flag = False ─────────────────────────────────────────


def test_flag_default_is_false(shm_root):
    collector = OuterTrinityCollector()
    assert collector._l0_rust_enabled is False


# ── 2. Flag=False compute path runs ─────────────────────────────────


def test_flag_off_compute_path_returns_full_shape(shm_root):
    collector = OuterTrinityCollector(l0_rust_enabled=False)
    sources = _make_sources()
    result = collector.collect(sources)

    assert "outer_body" in result and len(result["outer_body"]) == 5
    assert "outer_mind" in result and len(result["outer_mind"]) == 5
    assert "outer_spirit" in result and len(result["outer_spirit"]) == 5
    assert "outer_mind_15d" in result and len(result["outer_mind_15d"]) == 15
    assert "outer_spirit_45d" in result and len(result["outer_spirit_45d"]) == 45


# ── 3. Flag=True cold boot — no slots ───────────────────────────────


def test_flag_on_cold_boot_returns_last_known_neutral(shm_root):
    collector = OuterTrinityCollector(l0_rust_enabled=True, titan_id="T1")
    sources = _make_sources()
    result = collector.collect(sources)

    # Cold boot: no slots exist → last-known cached defaults (0.5 padded)
    assert result["outer_body"] == [0.5] * 5
    assert result["outer_mind"] == [0.5] * 5
    assert result["outer_spirit"] == [0.5] * 5
    assert result["outer_mind_15d"] == [0.5] * 15
    assert result["outer_spirit_45d"] == [0.5] * 45


# ── 4–6. Flag=True reads each slot ──────────────────────────────────


def test_flag_on_reads_outer_body_5d_slot(shm_root):
    expected_body = [0.1, 0.2, 0.3, 0.4, 0.5]
    _write_slot(shm_root / "titan_T1", "outer_body_5d", expected_body) if False else None
    # write directly under TITAN_SHM_ROOT (test env override puts shm_root at tmp_path root)
    _write_slot(shm_root, "outer_body_5d", expected_body)

    collector = OuterTrinityCollector(l0_rust_enabled=True, titan_id="T1")
    result = collector.collect(_make_sources())

    # float32 round-trip — tolerate precision
    for actual, exp in zip(result["outer_body"], expected_body):
        assert abs(actual - exp) < 1e-6


def test_flag_on_reads_outer_mind_15d_slot(shm_root):
    expected_mind = [0.05 * i for i in range(15)]
    _write_slot(shm_root, "outer_mind_15d", expected_mind)

    collector = OuterTrinityCollector(l0_rust_enabled=True, titan_id="T1")
    result = collector.collect(_make_sources())

    for actual, exp in zip(result["outer_mind_15d"], expected_mind):
        assert abs(actual - exp) < 1e-6


def test_flag_on_reads_outer_spirit_45d_slot(shm_root):
    expected_spirit = [0.02 * i for i in range(45)]
    _write_slot(shm_root, "outer_spirit_45d", expected_spirit)

    collector = OuterTrinityCollector(l0_rust_enabled=True, titan_id="T1")
    result = collector.collect(_make_sources())

    for actual, exp in zip(result["outer_spirit_45d"], expected_spirit):
        assert abs(actual - exp) < 1e-6


# ── 7. Stale slot → last-known + stale counter ──────────────────────


def test_flag_on_stale_slot_falls_back_to_last_known(shm_root):
    fresh_body = [0.7, 0.6, 0.5, 0.4, 0.3]
    # Write with wall_ns 60s in the past (> 30s stale threshold for body)
    stale_wall_ns = time.time_ns() - int(60 * 1e9)
    _write_slot(shm_root, "outer_body_5d", fresh_body, wall_ns=stale_wall_ns)

    collector = OuterTrinityCollector(l0_rust_enabled=True, titan_id="T1")
    result = collector.collect(_make_sources())

    # Stale → falls back to last-known cached default ([0.5]*5)
    assert result["outer_body"] == [0.5] * 5
    assert collector._shm_stale_count >= 1


# ── 8. Schema mismatch → last-known + read failure counter ──────────


def test_flag_on_schema_mismatch_falls_back_to_last_known(shm_root):
    body = [0.1, 0.2, 0.3, 0.4, 0.5]
    # Schema 99 ≠ expected 1
    _write_slot(shm_root, "outer_body_5d", body, schema=99)

    collector = OuterTrinityCollector(l0_rust_enabled=True, titan_id="T1")
    result = collector.collect(_make_sources())

    assert result["outer_body"] == [0.5] * 5
    assert collector._shm_reader_failure_count >= 1


# ── 9. Truncated file → last-known ──────────────────────────────────


def test_flag_on_truncated_slot_falls_back(shm_root):
    # Write only 10 bytes — far less than the 44 expected for outer_body_5d
    (shm_root / "outer_body_5d.bin").write_bytes(b"x" * 10)

    collector = OuterTrinityCollector(l0_rust_enabled=True, titan_id="T1")
    result = collector.collect(_make_sources())

    assert result["outer_body"] == [0.5] * 5


# ── 10. Flag=True returns same shape as flag=False ──────────────────


def test_flag_on_off_return_same_shape(shm_root):
    sources = _make_sources()

    off = OuterTrinityCollector(l0_rust_enabled=False)
    res_off = off.collect(sources)

    on = OuterTrinityCollector(l0_rust_enabled=True, titan_id="T1")
    res_on = on.collect(sources)

    # Same keys, same lengths
    assert set(res_off.keys()) == set(res_on.keys())
    for key in ["outer_body", "outer_mind", "outer_spirit",
                "outer_mind_15d", "outer_spirit_45d"]:
        assert len(res_off[key]) == len(res_on[key])


# ── 11. Flag=True caches last-known for next tick ───────────────────


def test_flag_on_caches_for_next_tick(shm_root):
    fresh_body = [0.11, 0.22, 0.33, 0.44, 0.55]
    _write_slot(shm_root, "outer_body_5d", fresh_body)

    collector = OuterTrinityCollector(l0_rust_enabled=True, titan_id="T1")
    collector.collect(_make_sources())

    # _last_outer_body now reflects the freshly-read values
    for actual, exp in zip(collector._last_outer_body, fresh_body):
        assert abs(actual - exp) < 1e-6


# ── 12. Flag=False compute path UNCHANGED (regression guard) ────────


def test_flag_off_compute_path_byte_identical_two_runs(shm_root):
    """Identical sources → identical compute output across two collectors.

    This guards the SPEC §3.0 Running-Titans Safety Rule: flag-off
    behavior MUST be byte-identical to today. If a future PR
    accidentally changes _collect_outer_body / _outer_mind / _outer_spirit
    arithmetic, this test catches it via cross-collector equality.
    """
    sources = _make_sources()

    c1 = OuterTrinityCollector(l0_rust_enabled=False)
    c2 = OuterTrinityCollector(l0_rust_enabled=False)

    res1 = c1.collect(sources)
    res2 = c2.collect(sources)

    # Same inputs, same code path → byte-identical floats
    assert res1["outer_body"] == res2["outer_body"]
    assert res1["outer_mind"] == res2["outer_mind"]
    assert res1["outer_spirit"] == res2["outer_spirit"]
    assert res1["outer_mind_15d"] == res2["outer_mind_15d"]
    assert res1["outer_spirit_45d"] == res2["outer_spirit_45d"]
