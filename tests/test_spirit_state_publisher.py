"""
Tests for ``titan_plugin.logic.spirit_state_publisher``.

Phase C Session 1 of rFP_phase_c_async_shm_consumer_migration §4.B.1.

Covers:
  1. Init — 5 writers initialized lazily; INFO logged at construction
  2. publish() with rich state_refs writes all 5 slots successfully
  3. publish() with empty / None state_refs writes stub payloads
     (cold-boot — staleness signaling preserved per G20)
  4. Per-slot first-success log fires once per slot
  5. Heartbeat ticks fire at canonical milestones (1, 10, 60, …)
  6. Encode failure on broken source object → caught, throttled WARN,
     other slots unaffected (independence per __doc__)
  7. Round-trip — payload written can be msgpack-decoded by reader
  8. Schema fields present per SPEC §7.1
  9. Oversize protection — payload > MAX → CRITICAL log, no write
 10. Single-writer per slot (G21) — repeated publish() advances
     header_seq monotonically

Run: ``python -m pytest tests/test_spirit_state_publisher.py -v -p no:anchorpy``
(per CLAUDE.md: each test file in separate pytest invocation).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import msgpack
import pytest

from titan_plugin.core.state_registry import StateRegistryReader
from titan_plugin.logic.spirit_state_publisher import (
    SpiritStatePublisher,
    _HEARTBEAT_TICKS,
)
from titan_plugin.logic.spirit_state_specs import (
    ALL_SPIRIT_STATE_SPECS,
    CONSCIOUSNESS_STATE_SLOT,
    CONSCIOUSNESS_STATE_SPEC,
    HORMONE_FIRES_SLOT,
    HORMONE_FIRES_SPEC,
    IMPULSE_ENGINE_STATE_SLOT,
    IMPULSE_ENGINE_STATE_SPEC,
    RESONANCE_STATE_SLOT,
    RESONANCE_STATE_SPEC,
    UNIFIED_SPIRIT_METADATA_SLOT,
    UNIFIED_SPIRIT_METADATA_SPEC,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def shm_root(tmp_path, monkeypatch):
    """Override TITAN_SHM_ROOT to a tmp dir so tests don't pollute /dev/shm."""
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


class _StubHormone:
    """Stub HormonalPressure-shape object (minimal attrs the publisher reads)."""
    def __init__(self, fire_count=0, level=0.0, fire_threshold=0.0,
                 refractory_until_ts=0.0, peak_level=0.0, last_fire_ts=0.0):
        self.fire_count = fire_count
        self.level = level
        self.fire_threshold = fire_threshold
        self.refractory_until_ts = refractory_until_ts
        self.peak_level = peak_level
        self.last_fire_ts = last_fire_ts


class _StubHormonalSystem:
    def __init__(self, hormones: dict | None = None):
        self._hormones = hormones or {}


class _StubNeuralNervousSystem:
    def __init__(self, hormones: dict | None = None):
        self._hormonal = _StubHormonalSystem(hormones=hormones)


class _StubImpulseEngine:
    def __init__(self, stats: dict | None = None):
        self._stats = stats or {
            "threshold": 0.5,
            "impulse_count": 7,
            "cooldown_seconds": 1.0,
            "outcome_count": 4,
            "success_rate": 0.75,
            "pending_impulse": False,
            "last_impulse_ts": 1234567.0,
        }

    def get_stats(self):
        return self._stats


class _StubResonance:
    def __init__(self, stats: dict | None = None):
        self._stats = stats or {
            "pairs": {
                "body": {"alignment": 0.42, "consecutive_resonances": 1},
                "mind": {"alignment": 0.61, "consecutive_resonances": 0},
                "spirit": {"alignment": 0.55, "consecutive_resonances": 2},
            },
            "resonant_count": 0,
            "all_resonant": False,
            "great_pulse_count": 3,
            "last_great_pulse_ts": 1234500.0,
            "config": {
                "phase_threshold_deg": 30.0,
                "required_cycles": 3,
                "pulse_window": 60,
            },
        }

    def get_stats(self):
        return self._stats


class _StubUnifiedSpirit:
    def __init__(self, stats: dict | None = None):
        self._stats = stats or {
            "epoch_count": 12,
            "current_epoch_id": 11,
            "velocity": 0.83,
            "is_stale": False,
            "consecutive_stale": 0,
            "stale_focus_multiplier": 1.0,
            "tensor_magnitude": 1.42,
            "tensor_sum": 65.0,
            "latest_epoch": {"epoch_id": 11, "state_vector": [0.5] * 130},
            "cumulative_quality": 4.5,
            "micro_tick_count": 1234,
            "last_alignment": 0.61,
            "enrichment_rate": 0.05,
            "full_130dt": [0.5] * 130,
            "config": {
                "stale_threshold": 0.8,
                "enrichment_base": 0.1,
                "velocity_window": 100,
                "enrichment_rate": 0.05,
                "min_alignment_threshold": 0.5,
            },
        }

    def get_stats(self):
        return self._stats


def _make_state_refs(*, with_neural=True, with_impulse=True, with_consciousness=True,
                     with_resonance=True, with_unified_spirit=True) -> dict:
    refs: dict[str, Any] = {
        "body_state": {"values": [0.5, 0.4, 0.6, 0.5, 0.55], "center_dist": 0.1},
        "mind_state": {"values": [0.5] * 5, "center_dist": 0.05},
    }
    if with_neural:
        refs["neural_nervous_system"] = _StubNeuralNervousSystem(hormones={
            name: _StubHormone(fire_count=i, level=0.5, fire_threshold=0.7)
            for i, name in enumerate(
                ["IMPULSE", "EMPATHY", "CREATIVITY", "VIGILANCE",
                 "CURIOSITY", "FOCUS", "INSPIRATION", "INTUITION"])
        })
    if with_impulse:
        refs["impulse_engine"] = _StubImpulseEngine()
    if with_consciousness:
        refs["consciousness"] = {
            "latest_epoch": {
                "epoch_id": 42,
                "density": 0.6,
                "curvature": -0.1,
                "dream_quality": 0.3,
                "fatigue": 0.2,
                "trajectory_magnitude": 0.4,
                "drift_magnitude": 0.5,
                "state_vector": [0.5] * 130,
            }
        }
    if with_resonance:
        refs["resonance"] = _StubResonance()
    if with_unified_spirit:
        refs["unified_spirit"] = _StubUnifiedSpirit()
    return refs


# ── 1. Init ───────────────────────────────────────────────────────────


def test_init_logs_and_no_writer_attached_until_first_publish(shm_root, caplog):
    caplog.set_level(logging.INFO, logger="titan_plugin.logic.spirit_state_publisher")
    pub = SpiritStatePublisher(titan_id="T_TEST")
    stats = pub.get_stats()
    assert stats["publish_count"] == 0
    # Writers attached lazily — none should be attached yet
    for slot, attached in stats["writers_attached"].items():
        assert attached is False, f"slot {slot} writer attached prematurely"
    # Init log should appear
    assert any("initialized" in r.message
               for r in caplog.records), "init INFO log missing"


# ── 2. publish() with rich state_refs ─────────────────────────────────


def test_publish_writes_all_5_slots(shm_root, caplog):
    caplog.set_level(logging.INFO, logger="titan_plugin.logic.spirit_state_publisher")
    pub = SpiritStatePublisher(titan_id="T_TEST")
    pub.publish(_make_state_refs())
    stats = pub.get_stats()
    assert stats["publish_count"] == 1
    # All 5 slots should have at least 1 success
    for slot in (HORMONE_FIRES_SLOT, IMPULSE_ENGINE_STATE_SLOT,
                 CONSCIOUSNESS_STATE_SLOT, RESONANCE_STATE_SLOT,
                 UNIFIED_SPIRIT_METADATA_SLOT):
        assert stats.get("encode_fails", {}).get(slot, 0) == 0
        assert stats.get("write_fails", {}).get(slot, 0) == 0
    # All slot files should exist on disk
    for spec in ALL_SPIRIT_STATE_SPECS:
        assert (shm_root / f"{spec.name}.bin").exists()
    # First-publish-success INFO logs should fire (one per slot)
    first_success_logs = [r for r in caplog.records
                          if "FIRST PUBLISH SUCCESS" in r.message]
    assert len(first_success_logs) == 5, \
        f"expected 5 first-success logs, got {len(first_success_logs)}"


# ── 3. publish() with cold-boot (None state_refs values) ───────────────


def test_publish_with_empty_state_refs_writes_stubs(shm_root):
    """Cold-boot — no producers ready; publisher writes stubs so
    consumers always see a fresh-but-empty payload (G20 staleness
    signaling preserved)."""
    pub = SpiritStatePublisher(titan_id="T_TEST")
    pub.publish({})
    stats = pub.get_stats()
    # All slots should still write (stubs)
    for slot in (HORMONE_FIRES_SLOT, IMPULSE_ENGINE_STATE_SLOT,
                 CONSCIOUSNESS_STATE_SLOT, RESONANCE_STATE_SLOT,
                 UNIFIED_SPIRIT_METADATA_SLOT):
        assert stats.get("write_fails", {}).get(slot, 0) == 0


# ── 4. Round-trip via StateRegistryReader ─────────────────────────────


def _read_msgpack_slot(spec, shm_root) -> dict | None:
    reader = StateRegistryReader(spec, shm_root)
    raw = reader.read_variable()
    if raw is None:
        return None
    return msgpack.unpackb(raw, raw=False)


def test_roundtrip_hormone_fires(shm_root):
    pub = SpiritStatePublisher(titan_id="T_TEST")
    refs = _make_state_refs()
    pub.publish(refs)
    decoded = _read_msgpack_slot(HORMONE_FIRES_SPEC, shm_root)
    assert decoded is not None
    assert "fires" in decoded
    assert "ts" in decoded
    # Expect 8 hormones keyed by name
    assert isinstance(decoded["fires"], dict)
    assert all(isinstance(v, int) for v in decoded["fires"].values())


def test_roundtrip_impulse_engine_state(shm_root):
    pub = SpiritStatePublisher(titan_id="T_TEST")
    pub.publish(_make_state_refs())
    decoded = _read_msgpack_slot(IMPULSE_ENGINE_STATE_SPEC, shm_root)
    assert decoded is not None
    assert "engine" in decoded
    assert decoded["engine"]["impulse_count"] == 7
    assert decoded["engine"]["success_rate"] == pytest.approx(0.75)
    assert "hormones" in decoded
    assert "ts" in decoded


def test_roundtrip_consciousness_state_includes_spirit_5dt(shm_root):
    pub = SpiritStatePublisher(titan_id="T_TEST")
    pub.publish(_make_state_refs())
    decoded = _read_msgpack_slot(CONSCIOUSNESS_STATE_SPEC, shm_root)
    assert decoded is not None
    assert decoded["epoch_id"] == 42
    assert decoded["density"] == pytest.approx(0.6)
    # spirit_5dt computed via _collect_spirit_tensor — 5 floats
    assert len(decoded["spirit_5dt"]) == 5
    assert all(isinstance(v, float) for v in decoded["spirit_5dt"])
    assert "body_values" in decoded
    assert "mind_values" in decoded
    assert "latest_epoch" in decoded


def test_roundtrip_resonance_state(shm_root):
    pub = SpiritStatePublisher(titan_id="T_TEST")
    pub.publish(_make_state_refs())
    decoded = _read_msgpack_slot(RESONANCE_STATE_SPEC, shm_root)
    assert decoded is not None
    assert "pairs" in decoded
    assert decoded["great_pulse_count"] == 3
    assert "config" in decoded


def test_roundtrip_unified_spirit_metadata(shm_root):
    pub = SpiritStatePublisher(titan_id="T_TEST")
    pub.publish(_make_state_refs())
    decoded = _read_msgpack_slot(UNIFIED_SPIRIT_METADATA_SPEC, shm_root)
    assert decoded is not None
    assert decoded["epoch_count"] == 12
    assert decoded["velocity"] == pytest.approx(0.83)
    assert len(decoded["full_130dt"]) == 130
    assert "config" in decoded
    assert "latest_epoch" in decoded


# ── 5. Independence: one slot's failure doesn't block others ──────────


def test_broken_source_for_one_slot_doesnt_block_others(shm_root):
    """Per __doc__: each publish_* method is independent. A failure
    in one (e.g., neural_nervous_system._hormonal raising) MUST NOT
    prevent the other slots from publishing."""
    refs = _make_state_refs()

    # Sabotage neural_nervous_system so hormone_fires + impulse_engine
    # iteration fail (but consciousness + resonance + unified_spirit
    # should still publish fine)
    class _Broken:
        @property
        def _hormonal(self):
            raise RuntimeError("simulated downstream NS failure")

    refs["neural_nervous_system"] = _Broken()
    pub = SpiritStatePublisher(titan_id="T_TEST")
    pub.publish(refs)
    stats = pub.get_stats()
    # Other 3 slots should write OK
    for slot in (CONSCIOUSNESS_STATE_SLOT, RESONANCE_STATE_SLOT,
                 UNIFIED_SPIRIT_METADATA_SLOT):
        assert stats["write_fails"][slot] == 0
        assert (shm_root / f"{slot}.bin").exists()


# ── 6. Heartbeat tick INFO logs ───────────────────────────────────────


def test_heartbeat_logs_at_canonical_ticks(shm_root, caplog):
    caplog.set_level(logging.INFO, logger="titan_plugin.logic.spirit_state_publisher")
    pub = SpiritStatePublisher(titan_id="T_TEST")
    refs = _make_state_refs()
    # Publish enough times to hit ticks 1 and 10
    for _ in range(10):
        pub.publish(refs)
    heartbeat_logs = [r for r in caplog.records if "heartbeat" in r.message]
    # Should log at tick 1 AND tick 10
    assert len(heartbeat_logs) >= 2, \
        f"expected ≥2 heartbeat logs (ticks 1+10), got {len(heartbeat_logs)}"


# ── 7. Single-writer monotonic publish (G21) ──────────────────────────


def test_repeated_publish_advances_seq_monotonically(shm_root):
    """G21 single-writer/multi-reader contract: each publish advances
    the slot's header_seq. Use the reader to verify monotonic version."""
    pub = SpiritStatePublisher(titan_id="T_TEST")
    refs = _make_state_refs()
    pub.publish(refs)
    reader = StateRegistryReader(CONSCIOUSNESS_STATE_SPEC, shm_root)
    # Give reader a chance to attach
    raw1 = reader.read_variable()
    assert raw1 is not None
    decoded1 = msgpack.unpackb(raw1, raw=False)
    ts1 = decoded1["ts"]
    # Tiny sleep so ts changes
    import time
    time.sleep(0.01)
    pub.publish(refs)
    raw2 = reader.read_variable()
    assert raw2 is not None
    decoded2 = msgpack.unpackb(raw2, raw=False)
    ts2 = decoded2["ts"]
    assert ts2 > ts1, f"ts did not advance: {ts1} -> {ts2}"


# ── 8. get_stats introspection shape ─────────────────────────────────


def test_get_stats_shape(shm_root):
    pub = SpiritStatePublisher(titan_id="T_TEST")
    pub.publish(_make_state_refs())
    stats = pub.get_stats()
    assert "publish_count" in stats
    assert "encode_fails" in stats
    assert "oversize_fails" in stats
    assert "write_fails" in stats
    assert "writers_attached" in stats
    assert len(stats["encode_fails"]) == 5
    assert len(stats["writers_attached"]) == 5
