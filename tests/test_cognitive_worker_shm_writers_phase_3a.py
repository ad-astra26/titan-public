"""
Tests for the 2 SHM writers added to cognitive_worker for Phase 3.A
D-SPEC-86: hormone_fires.bin + consciousness_state.bin.

Pre-fix audit (2026-05-18) found 19 of 45 inner_spirit dims DEAD fleet-wide
on T1+T2+T3 because the inner_spirit_sidecar reads `hormone_fires.bin` +
`consciousness_state.bin` SHM slots that had NO writer. Inner-spirit-rs
producer then received empty hormone_fires + consciousness dicts → 13 dims
defaulted to 0.

This test exercises the round-trip: cognitive_worker's writer → SHM slot →
inner_spirit_sidecar's reader → msgpack-decoded dict shape matches the
fields the Rust producer reads.

Per CLAUDE.md: run as its own pytest invocation
``python -m pytest tests/test_cognitive_worker_shm_writers_phase_3a.py -v -p no:anchorpy``
"""
from __future__ import annotations

import time

import msgpack
import pytest

from titan_hcl.core.state_registry import (
    StateRegistryReader,
    StateRegistryWriter,
)
from titan_hcl.logic.spirit_state_specs import (
    CONSCIOUSNESS_STATE_SPEC,
    HORMONE_FIRES_SPEC,
)


@pytest.fixture()
def shm_root(tmp_path, monkeypatch):
    """Override TITAN_SHM_ROOT to a tmp dir so tests don't pollute /dev/shm."""
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


# ── HORMONE_FIRES_SPEC writer ──────────────────────────────────────────


def test_hormone_fires_writer_round_trip(shm_root):
    """Write the exact payload shape the cognitive_worker block emits;
    decode via the same path inner_spirit_sidecar uses."""
    writer = StateRegistryWriter(HORMONE_FIRES_SPEC, shm_root)
    fires = {
        "REFLEX": 0,
        "FOCUS": 12,
        "INTUITION": 7,
        "IMPULSE": 3,
        "METABOLISM": 5,
        "CREATIVITY": 9,
        "CURIOSITY": 14,
        "EMPATHY": 4,
        "REFLECTION": 6,
        "INSPIRATION": 2,
        "VIGILANCE": 8,
    }
    payload = msgpack.packb({"fires": fires, "ts": time.time()},
                             use_bin_type=True)
    writer.write_variable(payload)

    reader = StateRegistryReader(HORMONE_FIRES_SPEC, shm_root)
    blob = reader.read_variable()
    decoded = msgpack.unpackb(blob, raw=False)

    assert isinstance(decoded, dict)
    assert "fires" in decoded
    assert "ts" in decoded
    assert decoded["fires"] == fires
    # Inner_spirit_sidecar:_read_hormone_fires unwraps `fires` inner dict.
    inner = decoded["fires"]
    assert inner["INTUITION"] == 7
    assert inner["EMPATHY"] == 4
    # Hormone count matches SPIRIT_PROXY_LEGACY_HORMONE_NAMES = 11.
    assert len(inner) == 11


def test_hormone_fires_handles_empty_hormones_dict(shm_root):
    """If neural_nervous_system._hormones is empty, payload writes {} fires."""
    writer = StateRegistryWriter(HORMONE_FIRES_SPEC, shm_root)
    payload = msgpack.packb({"fires": {}, "ts": time.time()},
                             use_bin_type=True)
    writer.write_variable(payload)
    reader = StateRegistryReader(HORMONE_FIRES_SPEC, shm_root)
    decoded = msgpack.unpackb(reader.read_variable(), raw=False)
    assert decoded["fires"] == {}


def test_hormone_fires_payload_fits_max_bytes(shm_root):
    """11 hormones × int count + ts must fit in HORMONE_FIRES_MAX_BYTES (1024)."""
    fires = {n: 99999 for n in (
        "REFLEX", "FOCUS", "INTUITION", "IMPULSE", "METABOLISM",
        "CREATIVITY", "CURIOSITY", "EMPATHY", "REFLECTION",
        "INSPIRATION", "VIGILANCE",
    )}
    payload = msgpack.packb({"fires": fires, "ts": time.time()},
                             use_bin_type=True)
    assert len(payload) < HORMONE_FIRES_SPEC.payload_bytes, (
        f"payload {len(payload)}B exceeds slot cap "
        f"{HORMONE_FIRES_SPEC.payload_bytes}B")


# ── CONSCIOUSNESS_STATE_SPEC writer ────────────────────────────────────


def test_consciousness_state_writer_round_trip(shm_root):
    """Write the exact payload shape the cognitive_worker block emits."""
    writer = StateRegistryWriter(CONSCIOUSNESS_STATE_SPEC, shm_root)
    payload_dict = {
        "epoch_count": 1611,
        "epoch_id": 1611,
        "density": 0.42,
        "curvature": -0.17,
        "dream_quality": 0.0,
        "fatigue": 0.35,
        "trajectory": 1.05,
        "trajectory_magnitude": 1.05,
        "ts": time.time(),
    }
    writer.write_variable(msgpack.packb(payload_dict, use_bin_type=True))

    reader = StateRegistryReader(CONSCIOUSNESS_STATE_SPEC, shm_root)
    decoded = msgpack.unpackb(reader.read_variable(), raw=False)

    # Inner-spirit-rs reads via field_or_default — check the names it expects.
    assert decoded["epoch_count"] == 1611  # SAT[4] temporal_continuity input
    assert decoded["epoch_id"] == 1611      # fallback name
    assert decoded["density"] == pytest.approx(0.42)  # CHIT[21] wisdom + ANANDA[31] meaning
    assert decoded["curvature"] == pytest.approx(-0.17)  # SAT[10] resilience input
    assert decoded["dream_quality"] == 0.0  # CHIT[25] dream_awareness input
    assert decoded["fatigue"] == pytest.approx(0.35)  # ANANDA[40] rest_fulfillment input
    assert decoded["trajectory"] == pytest.approx(1.05)  # CHIT[29] meta_cognition primary
    assert decoded["trajectory_magnitude"] == pytest.approx(1.05)  # fallback


def test_consciousness_state_cold_boot_zeros(shm_root):
    """Empty latest_epoch on cold boot → all defaults zero/0."""
    writer = StateRegistryWriter(CONSCIOUSNESS_STATE_SPEC, shm_root)
    payload_dict = {
        "epoch_count": 0,
        "epoch_id": 0,
        "density": 0.0,
        "curvature": 0.0,
        "dream_quality": 0.0,
        "fatigue": 0.0,
        "trajectory": 0.0,
        "trajectory_magnitude": 0.0,
        "ts": time.time(),
    }
    writer.write_variable(msgpack.packb(payload_dict, use_bin_type=True))
    reader = StateRegistryReader(CONSCIOUSNESS_STATE_SPEC, shm_root)
    decoded = msgpack.unpackb(reader.read_variable(), raw=False)
    assert decoded["epoch_count"] == 0
    assert decoded["density"] == 0.0


def test_consciousness_state_payload_fits_max_bytes(shm_root):
    """Plain consciousness payload must fit in CONSCIOUSNESS_STATE_MAX_BYTES (4096)."""
    payload = msgpack.packb({
        "epoch_count": 99_999_999,
        "epoch_id": 99_999_999,
        "density": 1.0,
        "curvature": 3.14,
        "dream_quality": 1.0,
        "fatigue": 1.0,
        "trajectory": 99.999,
        "trajectory_magnitude": 99.999,
        "ts": time.time(),
    }, use_bin_type=True)
    assert len(payload) < CONSCIOUSNESS_STATE_SPEC.payload_bytes, (
        f"payload {len(payload)}B exceeds slot cap "
        f"{CONSCIOUSNESS_STATE_SPEC.payload_bytes}B")


# ── Integration: round-trip via inner_spirit_sidecar's reader helper ────


def test_inner_spirit_sidecar_can_read_hormone_fires(shm_root):
    """Mirror the sidecar's _read_hormone_fires logic to confirm field shape."""
    writer = StateRegistryWriter(HORMONE_FIRES_SPEC, shm_root)
    fires = {"INTUITION": 50, "CREATIVITY": 30, "EMPATHY": 20}
    writer.write_variable(msgpack.packb(
        {"fires": fires, "ts": time.time()}, use_bin_type=True))

    # Reproduce inner_spirit_sidecar:129-138 logic.
    reader = StateRegistryReader(HORMONE_FIRES_SPEC, shm_root)
    blob = reader.read_variable()
    outer = msgpack.unpackb(blob, raw=False) if blob else None
    assert isinstance(outer, dict)
    inner = outer.get("fires")
    assert isinstance(inner, dict)
    assert inner == fires
    # Rust producer accesses INTUITION fires for SAT[20] pattern_recognition.
    assert inner["INTUITION"] == 50


def test_inner_spirit_sidecar_can_read_consciousness_state(shm_root):
    """Mirror the sidecar's _read_msgpack_slot(_consciousness_reader)."""
    writer = StateRegistryWriter(CONSCIOUSNESS_STATE_SPEC, shm_root)
    expected = {
        "epoch_count": 500,
        "epoch_id": 500,
        "density": 0.7,
        "curvature": 0.1,
        "dream_quality": 0.0,
        "fatigue": 0.4,
        "trajectory": 0.85,
        "trajectory_magnitude": 0.85,
        "ts": time.time(),
    }
    writer.write_variable(msgpack.packb(expected, use_bin_type=True))

    reader = StateRegistryReader(CONSCIOUSNESS_STATE_SPEC, shm_root)
    blob = reader.read_variable()
    decoded = msgpack.unpackb(blob, raw=False) if blob else None
    assert isinstance(decoded, dict)
    # All fields the Rust producer reads are present.
    for key in ("epoch_count", "epoch_id", "density", "curvature",
                "dream_quality", "fatigue", "trajectory",
                "trajectory_magnitude"):
        assert key in decoded, f"missing field: {key}"
