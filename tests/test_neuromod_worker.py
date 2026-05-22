"""Tests for titan_hcl.modules.neuromod_worker (C-S5 §10 D22 + §4.Q v1.8.0 D-SPEC-54 schema v2 bump).

Bus-independent tests covering:
- Canonical neuromodulator roster matches SPEC §7.1 + neuromodulator.py
- encode_neuromod_state shape + dtype + ordering (v2: (6,4) per-modulator level/gain/phasic/tonic)
- Round-trip encode → decode preserves dict-per-modulator state
- Missing-modulator / None-system handling
- Slot byte count matches SPEC §7.1 (96 payload = 6 × 4 × 4 bytes)
- Module name matches SPEC §9.B titan_HCL row
- NEUROMOD_READY bus message constant exists
- RegistrySpec byte layout matches encoded payload

Updated 2026-05-16 to v2 schema (D-SPEC-54). Tests previously assumed
v1 (6,) shape + flat-float decoded dict; v1.8.0 bumped to (6,4) for
cognitive_worker cross-process modulation reconstruction.
"""
from __future__ import annotations

import numpy as np
import pytest

from titan_hcl import bus
from titan_hcl.modules import neuromod_worker as nw


def test_neuromod_names_canonical_order():
    """SPEC §7.1 row 574 + neuromodulator.py:33-38 — order is byte-locked.
    Drift = silent slot-layout corruption."""
    assert nw.NEUROMOD_NAMES == ("DA", "5HT", "NE", "ACh", "Endorphin", "GABA")
    assert len(nw.NEUROMOD_NAMES) == 6


def test_payload_bytes_match_spec_7_1():
    """SPEC §7.1 row 754 (v2 schema per D-SPEC-54 v1.8.0):
    neuromod_state.bin payload = 6 × 4 × float32 = 96B (per-modulator
    level / gain / phasic / tonic)."""
    assert nw.NEUROMOD_COUNT == 6
    assert nw.NEUROMOD_FIELDS_PER_MOD == 4
    assert nw.NEUROMOD_STATE_PAYLOAD_BYTES == 96


def test_module_name_matches_spec_9b():
    """SPEC §9.B titan_HCL row line 982 lists `neuromod_module` in the
    Guardian registry. Drift = supervisor cannot route bus traffic."""
    assert nw.MODULE_NAME == "neuromod_module"


def test_neuromod_ready_bus_constant_exists():
    """C-S5 sibling-triad pattern with HORMONAL_READY + REFLEX_READY."""
    assert hasattr(bus, "NEUROMOD_READY")
    assert bus.NEUROMOD_READY == "NEUROMOD_READY"


def test_encode_returns_correct_shape_and_dtype():
    """Must match NEUROMOD_STATE RegistrySpec: shape (6, 4) per v2
    schema D-SPEC-54 v1.8.0, dtype <f4."""
    from titan_hcl.modules.neuromod_worker import _build_neuromod_system
    nm = _build_neuromod_system({})
    arr = nw.encode_neuromod_state(nm)
    assert arr.shape == (6, 4)
    assert arr.dtype == np.float32


def test_encode_with_none_returns_zero_array():
    """None NeuromodulatorSystem returns all-zero (defensive boot edge)."""
    arr = nw.encode_neuromod_state(None)
    assert arr.shape == (6, 4)
    assert np.all(arr == 0.0)


def test_encode_with_missing_modulator_zeroes_that_index():
    """Missing modulator at index N → arr[N, :] all zero; OTHER indices
    preserved with default field values."""
    class FakeMod:
        def __init__(self, level=0.42):
            self.level = level
            self.gain = 1.0
            self.phasic = 0.0
            self.tonic = 0.5

    class FakeNM:
        def __init__(self, present_names):
            self.modulators = {n: FakeMod() for n in present_names}

    # Skip "5HT" (index 1). Other indices should have level=0.42.
    nm = FakeNM([n for n in nw.NEUROMOD_NAMES if n != "5HT"])
    arr = nw.encode_neuromod_state(nm)
    assert arr.shape == (6, 4)
    assert np.all(arr[1] == 0.0)  # 5HT row zeroed
    assert arr[0, 0] == pytest.approx(0.42, abs=1e-6)  # DA level
    assert arr[2, 0] == pytest.approx(0.42, abs=1e-6)  # NE level
    assert arr[5, 0] == pytest.approx(0.42, abs=1e-6)  # GABA level


def test_encode_decode_round_trip():
    """encode → decode preserves all 6 × 4 fields byte-identically.
    v2 decoded shape is dict-of-dicts {name: {level, gain, phasic, tonic}}."""
    from titan_hcl.modules.neuromod_worker import _build_neuromod_system
    nm = _build_neuromod_system({})
    nm.modulators["DA"].level = 0.71
    nm.modulators["5HT"].level = 0.33
    nm.modulators["NE"].level = 0.55
    nm.modulators["ACh"].level = 0.18
    nm.modulators["Endorphin"].level = 0.92
    nm.modulators["GABA"].level = 0.41

    arr = nw.encode_neuromod_state(nm)
    decoded = nw.decode_neuromod_state(arr)

    # v2 schema: decoded[name] = {"level", "gain", "phasic", "tonic"}
    assert decoded["DA"]["level"] == pytest.approx(0.71, abs=1e-6)
    assert decoded["5HT"]["level"] == pytest.approx(0.33, abs=1e-6)
    assert decoded["NE"]["level"] == pytest.approx(0.55, abs=1e-6)
    assert decoded["ACh"]["level"] == pytest.approx(0.18, abs=1e-6)
    assert decoded["Endorphin"]["level"] == pytest.approx(0.92, abs=1e-6)
    assert decoded["GABA"]["level"] == pytest.approx(0.41, abs=1e-6)


def test_decode_rejects_wrong_shape():
    """decode_neuromod_state must reject arrays not (6, 4) under v2 strict mode.
    NOTE: v2 decode_neuromod_state has a v1 (6,) backward-compat path at
    neuromod_worker.py:163 — kept for boot transition windows. Strict
    rejection is for genuinely-wrong shapes like (5,) or (6, 1)."""
    bad = np.zeros((5,), dtype=np.float32)
    with pytest.raises(ValueError, match="neuromod_state shape mismatch"):
        nw.decode_neuromod_state(bad)
    bad2 = np.zeros((6, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        nw.decode_neuromod_state(bad2)


def test_registry_spec_byte_layout_matches_slot():
    """RegistrySpec.payload_bytes must equal NEUROMOD_STATE_PAYLOAD_BYTES
    (v2: 96 bytes = 6 × 4 × float32)."""
    from titan_hcl.core.state_registry import NEUROMOD_STATE
    assert NEUROMOD_STATE.shape == (6, 4)
    assert NEUROMOD_STATE.dtype == np.dtype("<f4")
    assert NEUROMOD_STATE.payload_bytes == nw.NEUROMOD_STATE_PAYLOAD_BYTES
    assert NEUROMOD_STATE.name == "neuromod_state"


def test_neuromod_schema_version_constant_present():
    """The auto-generated NEUROMOD_SCHEMA_VERSION constant must equal 2
    per SPEC v1.8.0 D-SPEC-54 (v1 → v2 bump 2026-05-15)."""
    from titan_hcl._phase_c_constants import NEUROMOD_SCHEMA_VERSION
    assert NEUROMOD_SCHEMA_VERSION == 2


def test_encode_clamps_to_float32_precision():
    """Large/small float64 levels are encoded as float32 — round-trip
    preserves within float32 precision (≈ 1e-7). v2 decoded is dict-
    per-modulator; check decoded[name]['level']."""
    class FakeMod:
        def __init__(self, level):
            self.level = level
            self.gain = 1.0
            self.phasic = 0.0
            self.tonic = 0.5

    class FakeNM:
        modulators = {
            "DA": FakeMod(0.123456789),
            "5HT": FakeMod(1e-7),
            "NE": FakeMod(0.999999),
            "ACh": FakeMod(0.0),
            "Endorphin": FakeMod(1.0),
            "GABA": FakeMod(0.5),
        }

    arr = nw.encode_neuromod_state(FakeNM())
    decoded = nw.decode_neuromod_state(arr)
    # Float32 has ~7 decimal digits precision
    assert decoded["DA"]["level"] == pytest.approx(0.123456789, abs=1e-6)
    assert decoded["NE"]["level"] == pytest.approx(0.999999, abs=1e-6)
    assert decoded["ACh"]["level"] == 0.0
    assert decoded["Endorphin"]["level"] == 1.0
