"""
Tests for ``titan_hcl.logic.outer_mind_sensor_refresh``.

Structural twin to ``test_outer_body_sensor_refresh.py`` — covers the
same 8 scenarios with outer_mind-specific source-keys + cadence:
  1. Slot bytes round-trip
  2. Source-dict normalization
  3. Cadence — tick_count advances
  4. sources_provider raise
  5. Oversize payload skipped
  6. Graceful stop
  7. Wall_ns advances per successful tick
  8. Restart-on-crash via in-process exception handler
"""
from __future__ import annotations

import asyncio
import struct
import time
from pathlib import Path

import msgpack
import pytest

from titan_hcl._phase_c_constants import (
    OUTER_MIND_TICK_BASE_S,
    OUTER_SENSOR_CACHE_MIND_MAX_BYTES,
)
from titan_hcl.core.state_registry import (
    BUFFER_META_SIZE,
    HEADER_SIZE,
    HEADER_STRUCT,
    _unpack_header_seq,
)
from titan_hcl.logic.outer_mind_sensor_refresh import (
    MAX_PAYLOAD_BYTES,
    REFRESH_PERIOD_S,
    SLOT_NAME,
    SOURCE_KEYS,
    OuterMindSensorRefresh,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def shm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


def _read_slot_payload(shm_path: Path) -> tuple[int, int, int, bytes]:
    """Read header + payload from v1.0.0 triple-buffer slot.

    Returns (version, schema, wall_ns, payload) — `version` is the
    monotonically-incrementing publish counter extracted from header_seq.
    """
    raw = shm_path.read_bytes()
    header_seq, schema, payload_cap = struct.unpack(
        HEADER_STRUCT, raw[:HEADER_SIZE]
    )
    version, ready_idx = _unpack_header_seq(header_seq)
    buf_block_sz = BUFFER_META_SIZE + payload_cap
    off = HEADER_SIZE + ready_idx * buf_block_sz
    wall_ns, payload_bytes, _crc = struct.unpack(
        "<QII", raw[off:off + BUFFER_META_SIZE]
    )
    payload = bytes(
        raw[off + BUFFER_META_SIZE : off + BUFFER_META_SIZE + payload_bytes]
    )
    return version, schema, wall_ns, payload


def _make_sources() -> dict:
    """Synthetic outer_mind sources dict matching SOURCE_KEYS canonical shape."""
    return {
        "uptime_seconds": 7200.0,
        "art_count_100": 12,
        "audio_count_100": 4,
        "art_count_500": 47,
        "audio_count_500": 18,
        "memory_status": {"recent_score": 0.62, "consolidation": 0.71},
        "assessment_stats": {"avg_score": 0.55, "n_assessments": 23},
        "impulse_stats": {"successful": 18, "failed": 3},
        "soul_health": 0.9,
        "agency_stats": {"actions_total": 120, "research_count": 5,
                          "creative_count": 8, "social_count": 3},
        "social_perception_stats": {"engagement": 0.4, "reach": 17},
        "twin_state": {"reachable": True, "DA": 0.7, "NE": 0.5, "GABA": 0.6},
        "anchor_state": {"success": True, "last_anchor_time": 1714400000.0},
        "bus_stats": {"queue_depth": 0, "drop_rate": 0.0},
        "helper_statuses": {"web": "ok", "blockchain": "ok"},
        "llm_avg_latency": 0.85,
    }


# ── 1. Slot bytes round-trip ────────────────────────────────────────


def test_slot_bytes_roundtrip(shm_root):
    expected = _make_sources()
    sidecar = OuterMindSensorRefresh(sources_provider=lambda: expected)
    sidecar._writer = sidecar._build_writer()
    sidecar._refresh_and_write()

    seq, schema, wall_ns, payload = _read_slot_payload(
        shm_root / f"{SLOT_NAME}.bin"
    )
    assert seq == 2
    assert schema == 1
    assert wall_ns > 0

    decoded = msgpack.unpackb(payload, raw=False)
    assert set(decoded.keys()) == set(SOURCE_KEYS)
    assert decoded["soul_health"] == 0.9
    assert decoded["art_count_100"] == 12
    assert decoded["twin_state"]["reachable"] is True
    assert decoded["agency_stats"]["actions_total"] == 120

    sidecar._writer.close()


# ── 2. Source-dict normalization ────────────────────────────────────


def test_normalize_sources_missing_keys_become_none(shm_root):
    raw = {"art_count_100": 5, "extra_unknown_key": "x"}
    normalized = OuterMindSensorRefresh._normalize_sources(raw)

    assert set(normalized.keys()) == set(SOURCE_KEYS)
    assert normalized["art_count_100"] == 5
    assert normalized["soul_health"] is None
    assert normalized["twin_state"] is None
    assert "extra_unknown_key" not in normalized


# ── 3. Cadence — tick_count advances ────────────────────────────────


@pytest.mark.asyncio
async def test_cadence_tick_count_advances(shm_root):
    sidecar = OuterMindSensorRefresh(
        sources_provider=lambda: _make_sources(),
        refresh_period_s=0.05,
    )
    task = asyncio.create_task(sidecar.run())
    await asyncio.sleep(0.30)
    await sidecar.stop()
    await task

    assert sidecar.tick_count >= 4
    assert sidecar.last_payload_bytes > 0


# ── 4. sources_provider raise ────────────────────────────────────────


@pytest.mark.asyncio
async def test_provider_raise_caught_counter_advances(shm_root):
    call_count = {"n": 0}

    def flaky_provider():
        call_count["n"] += 1
        raise RuntimeError(f"synthetic failure #{call_count['n']}")

    sidecar = OuterMindSensorRefresh(
        sources_provider=flaky_provider,
        refresh_period_s=0.02,
    )
    task = asyncio.create_task(sidecar.run())
    await asyncio.sleep(0.10)
    await sidecar.stop()
    await task

    assert sidecar.provider_failure_count >= 3
    assert sidecar.last_payload_bytes == 0
    assert task.done() and task.exception() is None


# ── 5. Oversize payload → slot retains last-known ───────────────────


def test_oversize_payload_skipped(shm_root):
    big_blob = "x" * (MAX_PAYLOAD_BYTES * 2)
    sources = _make_sources()
    sources["agency_stats"] = {"oversized_blob": big_blob}

    sidecar = OuterMindSensorRefresh(sources_provider=lambda: sources)
    sidecar._writer = sidecar._build_writer()
    sidecar._refresh_and_write()

    assert sidecar.oversize_failure_count == 1
    assert sidecar.last_payload_bytes == 0

    seq, _, _, _ = _read_slot_payload(shm_root / f"{SLOT_NAME}.bin")
    # v1.0.0 triple-buffer init publishes a zero-payload snapshot at seq=1;
    # oversize skip means no further publish — seq retained at 1.
    assert seq == 1

    sidecar._writer.close()


# ── 6. Graceful stop ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_graceful_stop_returns_within_tick(shm_root):
    sidecar = OuterMindSensorRefresh(
        sources_provider=lambda: _make_sources(),
        refresh_period_s=0.10,
    )
    task = asyncio.create_task(sidecar.run())
    await asyncio.sleep(0.05)
    t0 = time.monotonic()
    await sidecar.stop()
    await task
    elapsed = time.monotonic() - t0

    assert elapsed < 0.30, f"stop took {elapsed:.3f}s — should be < 0.3s"


# ── 7. Wall_ns advances on every successful write ───────────────────


def test_wall_ns_advances_per_tick(shm_root):
    sidecar = OuterMindSensorRefresh(sources_provider=lambda: _make_sources())
    sidecar._writer = sidecar._build_writer()

    sidecar._refresh_and_write()
    _, _, wall_ns_1, _ = _read_slot_payload(shm_root / f"{SLOT_NAME}.bin")

    time.sleep(0.001)
    sidecar._refresh_and_write()
    _, _, wall_ns_2, _ = _read_slot_payload(shm_root / f"{SLOT_NAME}.bin")

    assert wall_ns_2 > wall_ns_1

    sidecar._writer.close()


# ── 8. Restart-on-crash via in-process exception handler ────────────


@pytest.mark.asyncio
async def test_restart_on_inner_crash(shm_root, monkeypatch):
    crash_count = {"n": 0}
    original_loop = OuterMindSensorRefresh._refresh_loop

    async def flaky_loop(self):
        crash_count["n"] += 1
        if crash_count["n"] == 1:
            raise RuntimeError("synthetic inner-loop crash")
        await original_loop(self)

    monkeypatch.setattr(OuterMindSensorRefresh, "_refresh_loop", flaky_loop)
    monkeypatch.setattr(
        "titan_hcl.logic.outer_mind_sensor_refresh._RESTART_BACKOFF_S", 0.05
    )

    sidecar = OuterMindSensorRefresh(
        sources_provider=lambda: _make_sources(),
        refresh_period_s=0.02,
    )
    task = asyncio.create_task(sidecar.run())
    await asyncio.sleep(0.30)
    await sidecar.stop()
    await task

    assert crash_count["n"] >= 2
    assert sidecar.tick_count >= 2
    assert task.done() and task.exception() is None


# ── 9. SPEC binding sanity (period from generated TOML constant) ────


def test_default_period_matches_spec_toml():
    """REFRESH_PERIOD_S binds to OUTER_MIND_TICK_BASE_S from generated TOML."""
    assert REFRESH_PERIOD_S == OUTER_MIND_TICK_BASE_S
    assert REFRESH_PERIOD_S == 15.0  # SPEC §18.1 + D-SPEC-100 (G13 1:3:9 mid)


def test_max_payload_matches_spec_toml():
    assert MAX_PAYLOAD_BYTES == OUTER_SENSOR_CACHE_MIND_MAX_BYTES
    # Bumped 8KB→64KB 2026-05-10 (commit dd7e1d91) after Step 3 §4.2 P2
    # SOURCE_KEYS extension produced ~33KB payloads on T3.
    assert MAX_PAYLOAD_BYTES == 65536
