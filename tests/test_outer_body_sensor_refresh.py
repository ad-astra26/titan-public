"""
Tests for ``titan_plugin.logic.outer_body_sensor_refresh``.

Covers:
  1. Slot bytes round-trip — write a synthetic source-dict, read shm,
     assert msgpack-decoded payload matches.
  2. Source-dict normalization — missing keys → None; extra keys dropped.
  3. Cadence — sidecar advances tick_count at REFRESH_PERIOD_S.
  4. sources_provider raise — caught, counter advances, no crash.
  5. Oversize payload — slot retains last-known; counter advances.
  6. Graceful stop — stop() unblocks run() within tick.
  7. Wall_ns advancement — every successful write bumps wall_ns
     (consumer freshness preserved per SPEC §18.1).
  8. Restart-on-crash — uncaught exception in inner loop triggers
     CRITICAL log + backoff + restart (in-process exception handler
     per SPEC §11.B line 1236).
"""
from __future__ import annotations

import asyncio
import os
import struct
import time
from pathlib import Path

import msgpack
import numpy as np
import pytest

from titan_plugin._phase_c_constants import (
    OUTER_BODY_TICK_BASE_S,
    OUTER_SENSOR_CACHE_BODY_MAX_BYTES,
)
from titan_plugin.core.state_registry import HEADER_SIZE, HEADER_STRUCT
from titan_plugin.logic.outer_body_sensor_refresh import (
    MAX_PAYLOAD_BYTES,
    REFRESH_PERIOD_S,
    SLOT_NAME,
    SOURCE_KEYS,
    OuterBodySensorRefresh,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def shm_root(tmp_path, monkeypatch):
    """Override TITAN_SHM_ROOT to a tmp dir so test writes don't pollute /dev/shm."""
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


def _read_slot_payload(shm_path: Path) -> tuple[int, int, int, bytes]:
    """Read header + payload from a sensor_cache_*.bin file. Returns (seq, schema, wall_ns, payload_bytes)."""
    raw = shm_path.read_bytes()
    seq, schema, wall_ns, payload_bytes, _crc = struct.unpack(
        HEADER_STRUCT, raw[:HEADER_SIZE]
    )
    payload = bytes(raw[HEADER_SIZE:HEADER_SIZE + payload_bytes])
    return seq, schema, wall_ns, payload


def _make_sources(sol_balance: float = 1.5) -> dict:
    return {
        "agency_stats": {"actions_total": 100, "uptime_s": 3600.0},
        "helper_statuses": {"web": "ok", "blockchain": "ok"},
        "bus_stats": {"queue_depth": 0, "drop_rate": 0.0},
        "system_sensor_stats": {"cpu": 0.42, "ram": 0.61},
        "network_monitor_stats": {"latency_p99_ms": 12.3},
        "tx_latency_stats": {"latency_p50_s": 0.8},
        "block_delta_stats": {"normalized": 0.55, "delta_s": 6.2},
        "anchor_state": {"success": True, "last_anchor_time": 1714400000.0},
        "sol_balance": sol_balance,
    }


# ── 1. Slot bytes round-trip ────────────────────────────────────────


def test_slot_bytes_roundtrip(shm_root):
    """Synthetic source-dict written to shm decodes byte-identically via msgpack."""
    expected = _make_sources(sol_balance=2.7)
    sidecar = OuterBodySensorRefresh(sources_provider=lambda: expected)
    sidecar._writer = sidecar._build_writer()
    sidecar._refresh_and_write()

    seq, schema, wall_ns, payload = _read_slot_payload(
        shm_root / f"{SLOT_NAME}.bin"
    )
    assert seq == 2  # one successful write = seq goes 0 → 1 (odd) → 2 (even)
    assert schema == 1
    assert wall_ns > 0
    assert payload != b""

    decoded = msgpack.unpackb(payload, raw=False)
    # Canonical key set + values preserved
    assert set(decoded.keys()) == set(SOURCE_KEYS)
    assert decoded["sol_balance"] == 2.7
    assert decoded["agency_stats"]["actions_total"] == 100
    assert decoded["anchor_state"]["success"] is True

    sidecar._writer.close()


# ── 2. Source-dict normalization ────────────────────────────────────


def test_normalize_sources_missing_keys_become_none(shm_root):
    """Sources missing canonical keys → None; extra keys dropped."""
    raw = {"agency_stats": {"x": 1}, "extraneous_key": "should_drop"}
    normalized = OuterBodySensorRefresh._normalize_sources(raw)

    assert set(normalized.keys()) == set(SOURCE_KEYS)
    assert normalized["agency_stats"] == {"x": 1}
    assert normalized["sol_balance"] is None
    assert normalized["bus_stats"] is None
    assert "extraneous_key" not in normalized


# ── 3. Cadence — tick_count advances ────────────────────────────────


@pytest.mark.asyncio
async def test_cadence_tick_count_advances(shm_root):
    """Sidecar at fast period_s ticks N times within window."""
    sidecar = OuterBodySensorRefresh(
        sources_provider=lambda: _make_sources(),
        refresh_period_s=0.05,  # fast for test
    )
    task = asyncio.create_task(sidecar.run())
    await asyncio.sleep(0.30)  # ~6 ticks at 50ms
    await sidecar.stop()
    await task

    assert sidecar.tick_count >= 4  # tolerate scheduler jitter
    assert sidecar.last_payload_bytes > 0


# ── 4. sources_provider raise ────────────────────────────────────────


@pytest.mark.asyncio
async def test_provider_raise_caught_counter_advances(shm_root):
    """sources_provider raising → caught, counter increments, sidecar continues."""
    call_count = {"n": 0}

    def flaky_provider():
        call_count["n"] += 1
        raise RuntimeError(f"synthetic failure #{call_count['n']}")

    sidecar = OuterBodySensorRefresh(
        sources_provider=flaky_provider,
        refresh_period_s=0.02,
    )
    task = asyncio.create_task(sidecar.run())
    await asyncio.sleep(0.10)
    await sidecar.stop()
    await task

    assert sidecar.provider_failure_count >= 3
    assert sidecar.last_payload_bytes == 0  # no successful write
    # Sidecar didn't crash
    assert task.done() and task.exception() is None


# ── 5. Oversize payload → slot retains last-known ───────────────────


def test_oversize_payload_skipped(shm_root):
    """Payload > MAX_PAYLOAD_BYTES → CRITICAL log + skip; counter advances."""
    big_blob = "x" * (MAX_PAYLOAD_BYTES * 2)
    sources = _make_sources()
    sources["agency_stats"] = {"oversized_blob": big_blob}

    sidecar = OuterBodySensorRefresh(sources_provider=lambda: sources)
    sidecar._writer = sidecar._build_writer()
    sidecar._refresh_and_write()

    assert sidecar.oversize_failure_count == 1
    assert sidecar.last_payload_bytes == 0  # no write happened

    # Slot should still exist (created by writer init) but seq=0 (no write)
    seq, _, _, _ = _read_slot_payload(shm_root / f"{SLOT_NAME}.bin")
    assert seq == 0

    sidecar._writer.close()


# ── 6. Graceful stop ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_graceful_stop_returns_within_tick(shm_root):
    """stop() unblocks run() within ~refresh_period_s."""
    sidecar = OuterBodySensorRefresh(
        sources_provider=lambda: _make_sources(),
        refresh_period_s=0.10,
    )
    task = asyncio.create_task(sidecar.run())
    await asyncio.sleep(0.05)  # let it tick once
    t0 = time.monotonic()
    await sidecar.stop()
    await task
    elapsed = time.monotonic() - t0

    # stop() should return within ~refresh_period_s (it interrupts the
    # asyncio.wait_for sleep)
    assert elapsed < 0.30, f"stop took {elapsed:.3f}s — should be < 0.3s"


# ── 7. Wall_ns advances on every successful write ───────────────────


def test_wall_ns_advances_per_tick(shm_root):
    """Every successful write produces a strictly larger wall_ns (per SPEC §18.1)."""
    sidecar = OuterBodySensorRefresh(sources_provider=lambda: _make_sources())
    sidecar._writer = sidecar._build_writer()

    sidecar._refresh_and_write()
    _, _, wall_ns_1, _ = _read_slot_payload(shm_root / f"{SLOT_NAME}.bin")

    time.sleep(0.001)  # ensure wall clock advances at ns granularity
    sidecar._refresh_and_write()
    _, _, wall_ns_2, _ = _read_slot_payload(shm_root / f"{SLOT_NAME}.bin")

    assert wall_ns_2 > wall_ns_1, (
        f"wall_ns must advance every tick: {wall_ns_1} → {wall_ns_2}"
    )

    sidecar._writer.close()


# ── 8. Restart-on-crash via in-process exception handler ────────────


@pytest.mark.asyncio
async def test_restart_on_inner_crash(shm_root, monkeypatch):
    """If _refresh_loop crashes mid-flight, run() catches + restarts after backoff."""
    # Force _refresh_loop to crash on first call, succeed on second
    crash_count = {"n": 0}

    original_loop = OuterBodySensorRefresh._refresh_loop

    async def flaky_loop(self):
        crash_count["n"] += 1
        if crash_count["n"] == 1:
            raise RuntimeError("synthetic inner-loop crash")
        # Second call: real loop, runs until stop_event
        await original_loop(self)

    monkeypatch.setattr(OuterBodySensorRefresh, "_refresh_loop", flaky_loop)
    # Shrink restart backoff so test runs fast
    monkeypatch.setattr(
        "titan_plugin.logic.outer_body_sensor_refresh._RESTART_BACKOFF_S", 0.05
    )

    sidecar = OuterBodySensorRefresh(
        sources_provider=lambda: _make_sources(),
        refresh_period_s=0.02,
    )
    task = asyncio.create_task(sidecar.run())
    await asyncio.sleep(0.30)  # crash → backoff (50ms) → real loop ticks
    await sidecar.stop()
    await task

    # Verified: ran twice (crash then real), and real loop ticked
    assert crash_count["n"] >= 2
    assert sidecar.tick_count >= 2
    assert task.done() and task.exception() is None
