"""Tests for titan_plugin.modules.neuromod_worker (C-S5 §10 D22).

Bus-independent tests covering:
- Canonical neuromodulator roster matches SPEC §7.1 + neuromodulator.py
- encode_neuromod_state shape + dtype + ordering
- Round-trip encode → decode preserves levels
- Missing-modulator / None-system handling
- Slot byte count matches SPEC §7.1 (24 payload + 24 header = 48 total)
- Module name matches SPEC §9.B titan_HCL row
- NEUROMOD_READY bus message constant exists
- RegistrySpec byte layout matches encoded payload
"""
from __future__ import annotations

import numpy as np
import pytest

from titan_plugin import bus
from titan_plugin.modules import neuromod_worker as nw


def test_neuromod_names_canonical_order():
    """SPEC §7.1 row 574 + neuromodulator.py:33-38 — order is byte-locked.
    Drift = silent slot-layout corruption."""
    assert nw.NEUROMOD_NAMES == ("DA", "5HT", "NE", "ACh", "Endorphin", "GABA")
    assert len(nw.NEUROMOD_NAMES) == 6


def test_payload_bytes_match_spec_7_1():
    """SPEC §7.1 row 574: neuromod_state.bin payload = 6 × float32 = 24B,
    24 + 24 = 48B total."""
    assert nw.NEUROMOD_COUNT == 6
    assert nw.NEUROMOD_STATE_PAYLOAD_BYTES == 24
    assert (24 + nw.NEUROMOD_STATE_PAYLOAD_BYTES) == 48


def test_module_name_matches_spec_9b():
    """SPEC §9.B titan_HCL row line 982 lists `neuromod_module` in the
    Guardian registry. Drift = supervisor cannot route bus traffic."""
    assert nw.MODULE_NAME == "neuromod_module"


def test_neuromod_ready_bus_constant_exists():
    """C-S5 sibling-triad pattern with HORMONAL_READY + REFLEX_READY."""
    assert hasattr(bus, "NEUROMOD_READY")
    assert bus.NEUROMOD_READY == "NEUROMOD_READY"


def test_encode_returns_correct_shape_and_dtype():
    """Must match NEUROMOD_STATE RegistrySpec: shape (6,), dtype <f4."""
    from titan_plugin.modules.neuromod_worker import _build_neuromod_system
    nm = _build_neuromod_system({})
    arr = nw.encode_neuromod_state(nm)
    assert arr.shape == (6,)
    assert arr.dtype == np.float32


def test_encode_with_none_returns_zero_array():
    """None NeuromodulatorSystem returns all-zero (defensive boot edge)."""
    arr = nw.encode_neuromod_state(None)
    assert arr.shape == (6,)
    assert np.all(arr == 0.0)


def test_encode_with_missing_modulator_zeroes_that_index():
    """Missing modulator at index N → arr[N] = 0; OTHER indices preserved."""
    class FakeNM:
        def __init__(self, present_names):
            self.modulators = {
                n: type("MockMod", (), {"level": 0.42})() for n in present_names
            }

    # Skip "5HT" (index 1). Other indices should have level=0.42.
    nm = FakeNM([n for n in nw.NEUROMOD_NAMES if n != "5HT"])
    arr = nw.encode_neuromod_state(nm)
    assert arr.shape == (6,)
    assert arr[1] == 0.0  # 5HT slot zeroed
    assert arr[0] == pytest.approx(0.42, abs=1e-6)  # DA preserved
    assert arr[2] == pytest.approx(0.42, abs=1e-6)  # NE preserved
    assert arr[5] == pytest.approx(0.42, abs=1e-6)  # GABA preserved


def test_encode_decode_round_trip():
    """encode → decode preserves all 6 levels byte-identically."""
    from titan_plugin.modules.neuromod_worker import _build_neuromod_system
    nm = _build_neuromod_system({})
    nm.modulators["DA"].level = 0.71
    nm.modulators["5HT"].level = 0.33
    nm.modulators["NE"].level = 0.55
    nm.modulators["ACh"].level = 0.18
    nm.modulators["Endorphin"].level = 0.92
    nm.modulators["GABA"].level = 0.41

    arr = nw.encode_neuromod_state(nm)
    decoded = nw.decode_neuromod_state(arr)

    assert decoded["DA"] == pytest.approx(0.71, abs=1e-6)
    assert decoded["5HT"] == pytest.approx(0.33, abs=1e-6)
    assert decoded["NE"] == pytest.approx(0.55, abs=1e-6)
    assert decoded["ACh"] == pytest.approx(0.18, abs=1e-6)
    assert decoded["Endorphin"] == pytest.approx(0.92, abs=1e-6)
    assert decoded["GABA"] == pytest.approx(0.41, abs=1e-6)


def test_decode_rejects_wrong_shape():
    """decode_neuromod_state must reject arrays not (6,)."""
    bad = np.zeros((5,), dtype=np.float32)
    with pytest.raises(ValueError, match="neuromod_state shape mismatch"):
        nw.decode_neuromod_state(bad)
    bad2 = np.zeros((6, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        nw.decode_neuromod_state(bad2)


def test_registry_spec_byte_layout_matches_slot():
    """RegistrySpec.payload_bytes must equal NEUROMOD_STATE_PAYLOAD_BYTES."""
    from titan_plugin.core.state_registry import NEUROMOD_STATE
    assert NEUROMOD_STATE.shape == (6,)
    assert NEUROMOD_STATE.dtype == np.dtype("<f4")
    assert NEUROMOD_STATE.payload_bytes == nw.NEUROMOD_STATE_PAYLOAD_BYTES
    assert NEUROMOD_STATE.name == "neuromod_state"


def test_neuromod_schema_version_constant_present():
    """The auto-generated NEUROMOD_SCHEMA_VERSION constant must equal 1
    (per SPEC §3.1 D05)."""
    from titan_plugin._phase_c_constants import NEUROMOD_SCHEMA_VERSION
    assert NEUROMOD_SCHEMA_VERSION == 1


def test_encode_clamps_to_float32_precision():
    """Large/small float64 levels are encoded as float32 — round-trip
    preserves within float32 precision (≈ 1e-7)."""
    class FakeNM:
        modulators = {
            "DA": type("M", (), {"level": 0.123456789})(),
            "5HT": type("M", (), {"level": 1e-7})(),
            "NE": type("M", (), {"level": 0.999999})(),
            "ACh": type("M", (), {"level": 0.0})(),
            "Endorphin": type("M", (), {"level": 1.0})(),
            "GABA": type("M", (), {"level": 0.5})(),
        }

    arr = nw.encode_neuromod_state(FakeNM())
    decoded = nw.decode_neuromod_state(arr)
    # Float32 has ~7 decimal digits precision
    assert decoded["DA"] == pytest.approx(0.123456789, abs=1e-6)
    assert decoded["NE"] == pytest.approx(0.999999, abs=1e-6)
    assert decoded["ACh"] == 0.0
    assert decoded["Endorphin"] == 1.0
