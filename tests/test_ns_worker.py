"""Tests for titan_plugin.modules.ns_worker (C-S5 §10 D22).

Bus-independent tests covering:
- Canonical NS program roster matches NS_PROGRAMS
- encode_ns_state shape + dtype + field ordering
- Round-trip encode → decode preserves values
- Missing-program / None handling
- Slot byte count matches SPEC §7.1 (176 payload + 24 header = 200 total)
- Module name matches SPEC §9.B titan_HCL row
- NS_READY bus message constant exists
- urgency parameter encoded into FIELD_URGENCY column independently of
  NeuralReflexNet attributes
"""
from __future__ import annotations

import numpy as np
import pytest

from titan_plugin import bus
from titan_plugin.logic.emot_bundle_protocol import NS_PROGRAMS
from titan_plugin.modules import ns_worker as nw


def test_program_names_match_ns_programs_canonical_order():
    """SPEC §7.1 row 578 + emot_bundle_protocol.py:164-168 — program
    roster is byte-locked. Drift = silent slot-layout corruption."""
    assert nw.NS_PROGRAM_NAMES == tuple(NS_PROGRAMS)
    assert len(nw.NS_PROGRAM_NAMES) == 11
    # 5 inner + 6 outer per emot_bundle_protocol.py:165-167
    assert nw.NS_PROGRAM_NAMES[:5] == (
        "REFLEX", "FOCUS", "INTUITION", "IMPULSE", "METABOLISM",
    )
    assert nw.NS_PROGRAM_NAMES[5:] == (
        "CREATIVITY", "CURIOSITY", "EMPATHY", "REFLECTION",
        "INSPIRATION", "VIGILANCE",
    )


def test_payload_bytes_match_spec_7_1():
    """SPEC §7.1 row 578: titanvm_registers.bin payload = 11 × 4 × float32
    = 176B; total = 200B. Same byte count as hormonal_state.bin (sibling
    symmetry)."""
    assert nw.NS_PROGRAM_COUNT == 11
    assert nw.NS_FIELD_COUNT == 4
    assert nw.TITANVM_REGISTERS_PAYLOAD_BYTES == 176
    assert (24 + nw.TITANVM_REGISTERS_PAYLOAD_BYTES) == 200


def test_module_name_matches_spec_9b():
    """SPEC §9.B titan_HCL row line 982 lists `ns_module`. Drift =
    supervisor cannot route bus traffic."""
    assert nw.MODULE_NAME == "ns_module"


def test_ns_ready_bus_constant_exists():
    """C-S5 sibling-triad pattern with HORMONAL_READY + NEUROMOD_READY +
    REFLEX_READY."""
    assert hasattr(bus, "NS_READY")
    assert bus.NS_READY == "NS_READY"


def test_field_indices_are_stable():
    """Field-axis indices must NEVER move — they're part of the slot
    byte layout per SPEC §7.1."""
    assert nw.FIELD_URGENCY == 0
    assert nw.FIELD_FIRE_COUNT == 1
    assert nw.FIELD_TOTAL_UPDATES == 2
    assert nw.FIELD_LAST_LOSS == 3


def test_encode_returns_correct_shape_and_dtype():
    """encode_ns_state must return (11, 4) float32 — matches
    titanvm_registers slot byte layout in SPEC §7.1."""
    class FakeProg:
        fire_count = 0
        total_updates = 0
        last_loss = 0.0

    programs = {n: FakeProg() for n in NS_PROGRAMS}
    arr = nw.encode_ns_state(programs)
    assert arr.shape == (11, 4)
    assert arr.dtype == np.float32


def test_encode_with_none_returns_zero_array():
    """None / empty programs returns all-zero array (no crash)."""
    arr = nw.encode_ns_state(None)
    assert arr.shape == (11, 4)
    assert np.all(arr == 0.0)
    arr2 = nw.encode_ns_state({})
    assert np.all(arr2 == 0.0)


def test_encode_with_missing_program_zeroes_that_row():
    """Defensive: missing program at index N → arr[N, :] = 0; OTHER rows
    preserved."""
    class FakeProg:
        fire_count = 7
        total_updates = 42
        last_loss = 0.13

    # Skip "INTUITION" (index 2) — its row should remain zeros.
    programs = {n: FakeProg() for n in NS_PROGRAMS if n != "INTUITION"}
    arr = nw.encode_ns_state(programs)
    assert arr.shape == (11, 4)
    assert np.all(arr[2, :] == 0.0)  # INTUITION row zero
    # Other rows have the FakeProg values (urgency=0 since not provided)
    assert arr[0, nw.FIELD_FIRE_COUNT] == 7.0  # REFLEX
    assert arr[0, nw.FIELD_TOTAL_UPDATES] == 42.0
    assert arr[0, nw.FIELD_LAST_LOSS] == pytest.approx(0.13, abs=1e-6)
    assert arr[3, nw.FIELD_FIRE_COUNT] == 7.0  # IMPULSE


def test_encode_decode_round_trip():
    """encode → decode preserves all 4 fields per program byte-identically."""
    class FakeProg:
        def __init__(self, fc, tu, ll):
            self.fire_count = fc
            self.total_updates = tu
            self.last_loss = ll

    programs = {
        "REFLEX": FakeProg(1, 100, 0.05),
        "VIGILANCE": FakeProg(50, 5000, 0.001),
    }
    # Provide urgencies for two programs, defaults for others.
    urgencies = {"REFLEX": 0.42, "VIGILANCE": 0.95}

    arr = nw.encode_ns_state(programs, urgencies=urgencies)
    decoded = nw.decode_ns_state(arr)

    assert decoded["REFLEX"]["urgency"] == pytest.approx(0.42, abs=1e-6)
    assert decoded["REFLEX"]["fire_count"] == 1.0
    assert decoded["REFLEX"]["total_updates"] == 100.0
    assert decoded["REFLEX"]["last_loss"] == pytest.approx(0.05, abs=1e-6)
    assert decoded["VIGILANCE"]["urgency"] == pytest.approx(0.95, abs=1e-6)
    assert decoded["VIGILANCE"]["fire_count"] == 50.0
    assert decoded["VIGILANCE"]["total_updates"] == 5000.0
    # Unspecified programs (e.g. FOCUS) are zero
    assert decoded["FOCUS"]["fire_count"] == 0.0


def test_decode_rejects_wrong_shape():
    """decode_ns_state must reject arrays not (11, 4)."""
    bad = np.zeros((10, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="titanvm_registers shape mismatch"):
        nw.decode_ns_state(bad)
    bad2 = np.zeros((11, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        nw.decode_ns_state(bad2)


def test_urgency_independent_of_fire_count():
    """urgency is supplied SEPARATELY from program attrs — encoder MUST
    NOT confuse them. Provides programs with fire_count=99 and urgency=0.1
    → urgency col != fire_count col."""
    class P:
        fire_count = 99
        total_updates = 0
        last_loss = 0.0

    programs = {n: P() for n in NS_PROGRAMS}
    urgencies = {n: 0.1 for n in NS_PROGRAMS}
    arr = nw.encode_ns_state(programs, urgencies=urgencies)
    # urgency col = 0.1, fire_count col = 99 — distinct values
    assert np.allclose(arr[:, nw.FIELD_URGENCY], 0.1, atol=1e-6)
    assert np.allclose(arr[:, nw.FIELD_FIRE_COUNT], 99.0)


def test_titanvm_registers_schema_version_present():
    """Auto-generated TITANVM_REGISTERS_SCHEMA_VERSION = 1 at SPEC v0.1.0."""
    from titan_plugin._phase_c_constants import TITANVM_REGISTERS_SCHEMA_VERSION
    assert TITANVM_REGISTERS_SCHEMA_VERSION == 1


def test_count_fields_cast_to_float32_safely():
    """fire_count + total_updates are int in NeuralReflexNet but encoded
    as float32. Verify large counts still round-trip below 2^24 ≈ 16M.
    Above that, precision loss is documented + acceptable for read-mostly
    observability."""
    class P:
        fire_count = 1_000_000  # 1M — exact in float32
        total_updates = 16_000_000  # 16M — near float32 precision limit
        last_loss = 0.001

    arr = nw.encode_ns_state({"REFLEX": P()})
    decoded = nw.decode_ns_state(arr)
    assert decoded["REFLEX"]["fire_count"] == 1_000_000.0
    # 16M may lose a small amount of precision but should round to nearby float32
    assert abs(decoded["REFLEX"]["total_updates"] - 16_000_000) < 256
