"""Tests for titan_hcl.modules.hormonal_worker (C-S5 §10 D22).

Bus-independent tests covering:
- Canonical hormone roster matches NS_PROGRAMS
- encode_hormonal_state shape + dtype + field ordering
- Round-trip encode → decode preserves values
- Missing-hormone / None-system handling
- Slot byte count matches SPEC §7.1 v0.1.4 (176 payload + 24 header)
- Module name matches SPEC §9.B titan_HCL row
- HORMONAL_READY bus message constant exists
- HORMONAL_STATE_SCHEMA_VERSION generated constant matches SPEC

Note: this file does NOT spawn the worker subprocess (that requires the
full Guardian + bus stack). Subprocess integration is covered by the
broader Guardian test suite at session-close gate per PLAN §6 step 4.
"""
from __future__ import annotations

import numpy as np
import pytest

from titan_hcl import bus
from titan_hcl.logic.emot_bundle_protocol import NS_PROGRAMS
from titan_hcl.modules import hormonal_worker as hw


def test_hormone_names_match_ns_programs_canonical_order():
    """SPEC v0.1.4 + emot_bundle_protocol.py:164-168 — hormone roster
    is byte-locked to NS_PROGRAMS canonical order. Drift = silent slot-
    layout corruption."""
    assert hw.HORMONE_NAMES == tuple(NS_PROGRAMS)
    assert len(hw.HORMONE_NAMES) == 11
    # 5 inner + 6 outer per emot_bundle_protocol.py:165-167 comment
    assert hw.HORMONE_NAMES[:5] == (
        "REFLEX", "FOCUS", "INTUITION", "IMPULSE", "METABOLISM",
    )
    assert hw.HORMONE_NAMES[5:] == (
        "CREATIVITY", "CURIOSITY", "EMPATHY", "REFLECTION",
        "INSPIRATION", "VIGILANCE",
    )


def test_payload_bytes_match_spec_v0_1_4():
    """SPEC §7.1 v0.1.4 row: hormonal_state.bin payload = 11 × 4 × float32 = 176B."""
    assert hw.HORMONE_COUNT == 11
    assert hw.HORMONE_FIELD_COUNT == 4
    assert hw.HORMONAL_STATE_PAYLOAD_BYTES == 176
    # 200B total = 24 header + 176 payload (matches sibling
    # titanvm_registers.bin per SPEC §7.1).
    assert (24 + hw.HORMONAL_STATE_PAYLOAD_BYTES) == 200


def test_module_name_matches_spec_9b():
    """SPEC §9.B titan_HCL row line 982 lists `hormonal_module` in the
    Guardian registry. Drift = supervisor cannot route bus traffic to
    this worker."""
    assert hw.MODULE_NAME == "hormonal_module"


def test_hormonal_ready_bus_constant_exists():
    """C-S5 sibling-triad pattern with REFLEX_READY: peers know the
    hormonal slot writer is live."""
    assert hasattr(bus, "HORMONAL_READY")
    assert bus.HORMONAL_READY == "HORMONAL_READY"


def test_field_indices_are_stable():
    """Field-axis indices must NEVER move — they're part of the slot
    byte layout per SPEC v0.1.4."""
    assert hw.FIELD_LEVEL == 0
    assert hw.FIELD_THRESHOLD == 1
    assert hw.FIELD_REFRACTORY == 2
    assert hw.FIELD_PEAK_LEVEL == 3


def test_encode_returns_correct_shape_and_dtype():
    """encode_hormonal_state must return (11, 4) float32 — matches
    HORMONAL_STATE RegistrySpec in state_registry.py + slot byte
    layout in SPEC §7.1."""
    from titan_hcl.modules.hormonal_worker import _build_hormonal_system
    hs = _build_hormonal_system({})
    arr = hw.encode_hormonal_state(hs)
    assert arr.shape == (11, 4)
    assert arr.dtype == np.float32


def test_encode_with_none_returns_zero_array():
    """Defensive: None HormonalSystem returns all-zero array (no crash).
    Should not happen in production but matters for boot-order
    edge cases."""
    arr = hw.encode_hormonal_state(None)
    assert arr.shape == (11, 4)
    assert np.all(arr == 0.0)


def test_encode_with_missing_hormone_zeroes_that_row():
    """Defensive: if HormonalSystem is missing a hormone (shouldn't
    happen with correct boot wiring), that row encodes as zeros — does
    NOT shift other rows down."""
    class FakeHS:
        def __init__(self, present_names):
            from titan_hcl.logic.hormonal_pressure import HormonalPressure
            self._hormones = {n: HormonalPressure(name=n) for n in present_names}

        def get_hormone(self, name):
            return self._hormones.get(name)

    # Skip "FOCUS" (index 1) — its row should remain zeros.
    hs = FakeHS([n for n in NS_PROGRAMS if n != "FOCUS"])
    arr = hw.encode_hormonal_state(hs)
    assert arr.shape == (11, 4)
    assert np.all(arr[1, :] == 0.0)  # FOCUS row zero
    # Other rows: HormonalPressure default-constructed has level=0,
    # threshold=0.5, refractory=0, peak_level=0
    assert np.isclose(arr[0, hw.FIELD_THRESHOLD], 0.5)  # REFLEX threshold
    assert np.isclose(arr[2, hw.FIELD_THRESHOLD], 0.5)  # INTUITION threshold


def test_encode_decode_round_trip():
    """encode → decode preserves all 4 fields per hormone byte-identically
    (within float32 precision)."""
    from titan_hcl.modules.hormonal_worker import _build_hormonal_system
    hs = _build_hormonal_system({})
    # Mutate a few hormones to non-default values so we can verify round-trip
    hs._hormones["REFLEX"].level = 0.42
    hs._hormones["REFLEX"].threshold = 0.61
    hs._hormones["REFLEX"].refractory = 0.13
    hs._hormones["REFLEX"].peak_level = 0.88
    hs._hormones["VIGILANCE"].level = 0.95
    hs._hormones["VIGILANCE"].peak_level = 1.0

    arr = hw.encode_hormonal_state(hs)
    decoded = hw.decode_hormonal_state(arr)

    # Roundtrip with float32 precision tolerance
    assert decoded["REFLEX"]["level"] == pytest.approx(0.42, abs=1e-6)
    assert decoded["REFLEX"]["threshold"] == pytest.approx(0.61, abs=1e-6)
    assert decoded["REFLEX"]["refractory"] == pytest.approx(0.13, abs=1e-6)
    assert decoded["REFLEX"]["peak_level"] == pytest.approx(0.88, abs=1e-6)
    assert decoded["VIGILANCE"]["level"] == pytest.approx(0.95, abs=1e-6)
    assert decoded["VIGILANCE"]["peak_level"] == pytest.approx(1.0, abs=1e-6)


def test_decode_rejects_wrong_shape():
    """decode_hormonal_state must reject arrays that aren't (11, 4) —
    catches accidental schema drift."""
    bad = np.zeros((10, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="hormonal_state shape mismatch"):
        hw.decode_hormonal_state(bad)
    bad2 = np.zeros((11, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        hw.decode_hormonal_state(bad2)


def test_registry_spec_byte_layout_matches_slot():
    """RegistrySpec.payload_bytes must equal HORMONAL_STATE_PAYLOAD_BYTES.
    Drift here = Python writer / Rust reader byte mismatch."""
    from titan_hcl.core.state_registry import HORMONAL_STATE
    assert HORMONAL_STATE.shape == (11, 4)
    assert HORMONAL_STATE.dtype == np.dtype("<f4")
    assert HORMONAL_STATE.payload_bytes == hw.HORMONAL_STATE_PAYLOAD_BYTES
    assert HORMONAL_STATE.name == "hormonal_state"


def test_generated_schema_version_constant_matches_spec():
    """The auto-generated HORMONAL_STATE_SCHEMA_VERSION constant in
    titan_hcl/_phase_c_constants.py must equal 1 at SPEC v0.1.4 ship
    (per SPEC §3.1 D05). Drift = SPEC TOML out of sync with code."""
    from titan_hcl._phase_c_constants import HORMONAL_STATE_SCHEMA_VERSION
    assert HORMONAL_STATE_SCHEMA_VERSION == 1
