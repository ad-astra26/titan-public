"""
Tests for ``titan_plugin.logic.outer_spirit_sensor_refresh``.

Structural twin to outer_body / outer_mind sidecar tests with
outer_spirit-specific source-keys + 30s cadence.

10 tests covering:
  1-8. Round-trip / normalization / cadence / provider-raise / oversize
       / graceful-stop / wall_ns / restart-on-crash
  9-10. SPEC binding sanity (period + max_payload from generated TOML)
"""
from __future__ import annotations

import asyncio
import struct
import time
from pathlib import Path

import msgpack
import pytest

from titan_plugin._phase_c_constants import (
    OUTER_SENSOR_CACHE_SPIRIT_MAX_BYTES,
    OUTER_SPIRIT_TICK_BASE_S,
)
from titan_plugin.core.state_registry import HEADER_SIZE, HEADER_STRUCT
from titan_plugin.logic.outer_spirit_sensor_refresh import (
    MAX_PAYLOAD_BYTES,
    REFRESH_PERIOD_S,
    SLOT_NAME,
    SOURCE_KEYS,
    OuterSpiritSensorRefresh,
)


@pytest.fixture()
def shm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


def _read_slot_payload(shm_path: Path) -> tuple[int, int, int, bytes]:
    raw = shm_path.read_bytes()
    seq, schema, wall_ns, payload_bytes, _crc = struct.unpack(
        HEADER_STRUCT, raw[:HEADER_SIZE]
    )
    payload = bytes(raw[HEADER_SIZE:HEADER_SIZE + payload_bytes])
    return seq, schema, wall_ns, payload


def _make_sources() -> dict:
    """Synthetic outer_spirit sources matching SOURCE_KEYS canonical shape."""
    return {
        "action_stats": {"actions_total": 240, "self_initiated": 180,
                          "inner_outer_coherence": 0.72},
        "creative_stats": {"art_count": 47, "audio_count": 18,
                            "creative_rate_per_day": 5.2},
        "guardian_stats": {"threats_detected": 0, "interventions": 0,
                            "uptime_pct": 0.998},
        "sovereignty_ratio": 0.75,
        "uptime_ratio": 0.95,
        "recovery_stats": {"crashes": 0, "recovery_time_s": 0.0,
                            "successful_restarts": 0},
        "social_stats": {"engagement": 0.4, "reach": 47, "interactions": 23},
        "memory_stats": {"recall_score": 0.62, "consolidation": 0.71,
                          "depth": 0.55},
        "hormone_levels": {"DA": 0.65, "5HT": 0.55, "NE": 0.45,
                            "ACh": 0.5, "Endorphin": 0.4, "GABA": 0.6,
                            "cortisol": 0.3, "oxytocin": 0.55},
        "solana_stats": {"balance": 1.5, "tx_success_rate": 0.99,
                          "anchor_age_s": 120.0, "identity_verified": 1.0},
        "assessment_stats": {"avg_score": 0.55, "n_assessments": 23},
        "history": {"recent_45d": [[0.5] * 45 for _ in range(3)],
                     "trend": "stable"},
    }


def test_slot_bytes_roundtrip(shm_root):
    expected = _make_sources()
    sidecar = OuterSpiritSensorRefresh(sources_provider=lambda: expected)
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
    assert decoded["sovereignty_ratio"] == 0.75
    assert decoded["hormone_levels"]["DA"] == 0.65
    assert decoded["solana_stats"]["balance"] == 1.5
    assert len(decoded["history"]["recent_45d"]) == 3
    assert len(decoded["history"]["recent_45d"][0]) == 45

    sidecar._writer.close()


def test_normalize_sources_missing_keys_become_none(shm_root):
    raw = {"sovereignty_ratio": 0.5, "extraneous": 1}
    normalized = OuterSpiritSensorRefresh._normalize_sources(raw)

    assert set(normalized.keys()) == set(SOURCE_KEYS)
    assert normalized["sovereignty_ratio"] == 0.5
    assert normalized["hormone_levels"] is None
    assert normalized["history"] is None
    assert "extraneous" not in normalized


@pytest.mark.asyncio
async def test_cadence_tick_count_advances(shm_root):
    sidecar = OuterSpiritSensorRefresh(
        sources_provider=lambda: _make_sources(),
        refresh_period_s=0.05,
    )
    task = asyncio.create_task(sidecar.run())
    await asyncio.sleep(0.30)
    await sidecar.stop()
    await task

    assert sidecar.tick_count >= 4
    assert sidecar.last_payload_bytes > 0


@pytest.mark.asyncio
async def test_provider_raise_caught_counter_advances(shm_root):
    call_count = {"n": 0}

    def flaky_provider():
        call_count["n"] += 1
        raise RuntimeError(f"synthetic failure #{call_count['n']}")

    sidecar = OuterSpiritSensorRefresh(
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


def test_oversize_payload_skipped(shm_root):
    big_blob = "x" * (MAX_PAYLOAD_BYTES * 2)
    sources = _make_sources()
    sources["history"] = {"oversized_blob": big_blob}

    sidecar = OuterSpiritSensorRefresh(sources_provider=lambda: sources)
    sidecar._writer = sidecar._build_writer()
    sidecar._refresh_and_write()

    assert sidecar.oversize_failure_count == 1
    assert sidecar.last_payload_bytes == 0

    seq, _, _, _ = _read_slot_payload(shm_root / f"{SLOT_NAME}.bin")
    assert seq == 0

    sidecar._writer.close()


@pytest.mark.asyncio
async def test_graceful_stop_returns_within_tick(shm_root):
    sidecar = OuterSpiritSensorRefresh(
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


def test_wall_ns_advances_per_tick(shm_root):
    sidecar = OuterSpiritSensorRefresh(sources_provider=lambda: _make_sources())
    sidecar._writer = sidecar._build_writer()

    sidecar._refresh_and_write()
    _, _, wall_ns_1, _ = _read_slot_payload(shm_root / f"{SLOT_NAME}.bin")

    time.sleep(0.001)
    sidecar._refresh_and_write()
    _, _, wall_ns_2, _ = _read_slot_payload(shm_root / f"{SLOT_NAME}.bin")

    assert wall_ns_2 > wall_ns_1

    sidecar._writer.close()


@pytest.mark.asyncio
async def test_restart_on_inner_crash(shm_root, monkeypatch):
    crash_count = {"n": 0}
    original_loop = OuterSpiritSensorRefresh._refresh_loop

    async def flaky_loop(self):
        crash_count["n"] += 1
        if crash_count["n"] == 1:
            raise RuntimeError("synthetic inner-loop crash")
        await original_loop(self)

    monkeypatch.setattr(OuterSpiritSensorRefresh, "_refresh_loop", flaky_loop)
    monkeypatch.setattr(
        "titan_plugin.logic.outer_spirit_sensor_refresh._RESTART_BACKOFF_S", 0.05
    )

    sidecar = OuterSpiritSensorRefresh(
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


def test_default_period_matches_spec_toml():
    assert REFRESH_PERIOD_S == OUTER_SPIRIT_TICK_BASE_S
    assert REFRESH_PERIOD_S == 30.0


def test_max_payload_matches_spec_toml():
    assert MAX_PAYLOAD_BYTES == OUTER_SENSOR_CACHE_SPIRIT_MAX_BYTES
    assert MAX_PAYLOAD_BYTES == 8192
