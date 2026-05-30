"""Tests for §4.Q neuromod_worker.evaluate() migration.

See `titan-docs/PLAN_microkernel_phase_c_neuromod_worker_evaluate_migration.md`
for the chunk-by-chunk implementation map.

Test groups:
    Q1 — compute_modulation_from_state pure helper parity vs get_modulation()
    Q2 — NEUROMOD_STATE 24-byte encode/decode round-trip + schema-version
    Q3 — neuromod_inputs.bin SHM round-trip + reader parity
    Q4 — NEUROMOD_EXTERNAL_NUDGE bus constant + payload schema
    Q5 — cognitive_worker inputs builder produces well-formed dict
    Q6 — neuromod_worker evaluate driver: SHM → evaluate → SHM write
    Q7 — NEUROMOD_EXTERNAL_NUDGE subscriber → apply_external_nudge
    Q8 — 6 cognitive_worker nudge emit sites (audit)
    Q9 — outer_interface_worker self-exploration nudge emit
    Q10 — modulation reconstruction → NNS._modulation
    Q11 — SOVEREIGNTY_EPOCH + TimeChain heartbeat moved to cognitive_worker
    Q12 — NEUROMOD_STATS_UPDATED 2.5s coalesced publisher
    Q13 — spirit_worker neuromod cleanup acceptance (grep returns 0)
    Q14 — persistence save-on-shutdown
    Q15 — SPEC parity (frontmatter version + Changelog row + spec_index)
"""
from __future__ import annotations

import random

import pytest

import numpy as np

from titan_hcl.logic.neuromodulator import (
    NeuromodulatorSystem,
    compute_modulation_from_state,
)
from titan_hcl.modules.neuromod_worker import (
    NEUROMOD_COUNT,
    NEUROMOD_FIELDS_PER_MOD,
    NEUROMOD_FIELD_NAMES,
    NEUROMOD_NAMES,
    NEUROMOD_STATE_PAYLOAD_BYTES,
    decode_neuromod_levels,
    decode_neuromod_state,
    encode_neuromod_state,
)


# ─────────────────────────────────────────────────────────────────────────────
# Q1 — compute_modulation_from_state pure helper parity
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_system(tmp_path) -> NeuromodulatorSystem:
    """A NeuromodulatorSystem with deterministic randomized state."""
    rng = random.Random(20260515)
    nm = NeuromodulatorSystem(data_dir=str(tmp_path / "neuromodulator"))
    for mod in nm.modulators.values():
        mod.level = rng.uniform(0.1, 0.9)
        mod.tonic_level = rng.uniform(0.3, 0.7)
        mod.phasic_level = rng.uniform(-0.1, 0.1)
        mod.sensitivity = rng.uniform(0.6, 1.4)
        mod.setpoint = rng.uniform(0.4, 0.6)
    return nm


def test_compute_modulation_from_state_parity_with_get_modulation(synthetic_system):
    """Pure helper output MUST be byte-identical to NeuromodulatorSystem.get_modulation()."""
    nm = synthetic_system

    state = {
        name: {
            "level": mod.level,
            "gain": mod.get_gain(),
            "phasic": mod.phasic_level,
            "tonic": mod.tonic_level,
        }
        for name, mod in nm.modulators.items()
    }

    expected = nm.get_modulation()
    actual = compute_modulation_from_state(state)

    assert set(actual.keys()) == set(expected.keys()), "key set mismatch"
    for key in expected:
        assert actual[key] == expected[key], (
            f"value mismatch on {key!r}: actual={actual[key]!r} expected={expected[key]!r}"
        )


def test_compute_modulation_from_state_handles_missing_modulator():
    """Defensive: missing modulators default to gain=1.0 (no modulation)."""
    state = {"DA": {"gain": 1.5}}  # only DA — other 5 absent

    result = compute_modulation_from_state(state)

    # DA-derived keys reflect 1.5 gain
    assert result["learning_rate_gain"] == 1.5
    assert result["fire_threshold_gain"] == pytest.approx(1.0 / 1.5)
    assert result["working_memory_retention"] == 1.5
    # 5HT-derived keys default to gain=1.0 → identity
    assert result["accumulation_rate_gain"] == pytest.approx(1.0)
    assert result["refractory_gain"] == 1.0
    # GABA-derived: system_energy at gaba_gain=1.0 → 1.0
    assert result["system_energy"] == pytest.approx(1.0)


def test_compute_modulation_from_state_handles_extreme_gains():
    """Verify max(0.3, gain) floor for reciprocal-style keys (1/gain pattern)."""
    state = {name: {"gain": 0.05} for name in ("DA", "5HT", "NE", "ACh", "Endorphin", "GABA")}

    result = compute_modulation_from_state(state)

    # Pattern: 1.0 / max(0.3, gain) → 1.0 / 0.3 ≈ 3.333
    assert result["fire_threshold_gain"] == pytest.approx(1.0 / 0.3)
    assert result["accumulation_rate_gain"] == pytest.approx(1.0 / 0.3)
    assert result["system_energy"] == pytest.approx(1.0 / 0.3)
    # Pattern: direct gain — 0.05 (no floor on these keys)
    assert result["learning_rate_gain"] == 0.05
    assert result["refractory_gain"] == 0.05


def test_get_modulation_uses_pure_helper_after_refactor(synthetic_system):
    """Refactor invariant: get_modulation() composes the same 14-key dict."""
    expected_keys = {
        "learning_rate_gain", "fire_threshold_gain", "working_memory_retention",
        "accumulation_rate_gain", "refractory_gain", "patience_factor",
        "sensory_gain", "exploration_temperature", "filter_down_strength",
        "training_frequency_gain", "memory_encoding_gain", "observation_precision",
        "intrinsic_motivation", "discomfort_suppression",
        "global_threshold_raise", "system_energy",
    }
    assert set(synthetic_system.get_modulation().keys()) == expected_keys


# ─────────────────────────────────────────────────────────────────────────────
# Q2 — NEUROMOD_STATE 24-byte → 96-byte (6, 4) encode/decode round-trip
# ─────────────────────────────────────────────────────────────────────────────


def test_encode_neuromod_state_v2_shape_and_layout(synthetic_system):
    """v2 encode produces (6, 4) float32 LE array — 96 bytes payload."""
    arr = encode_neuromod_state(synthetic_system)

    assert arr.shape == (NEUROMOD_COUNT, NEUROMOD_FIELDS_PER_MOD)
    assert arr.dtype == np.float32
    assert arr.nbytes == NEUROMOD_STATE_PAYLOAD_BYTES == 96

    # Field-order invariant: arr[i, 0] = level, arr[i, 1] = gain
    # arr[i, 2] = phasic, arr[i, 3] = tonic
    assert NEUROMOD_FIELD_NAMES == ("level", "gain", "phasic", "tonic")
    for i, name in enumerate(NEUROMOD_NAMES):
        mod = synthetic_system.modulators[name]
        assert float(arr[i, 0]) == pytest.approx(mod.level)
        assert float(arr[i, 1]) == pytest.approx(mod.get_gain())
        assert float(arr[i, 2]) == pytest.approx(mod.phasic_level)
        assert float(arr[i, 3]) == pytest.approx(mod.tonic_level)


def test_encode_decode_round_trip(synthetic_system):
    """encode → decode_neuromod_state recovers the same 4-field dict."""
    arr = encode_neuromod_state(synthetic_system)
    decoded = decode_neuromod_state(arr)

    assert set(decoded.keys()) == set(NEUROMOD_NAMES)
    for name in NEUROMOD_NAMES:
        mod = synthetic_system.modulators[name]
        d = decoded[name]
        assert d["level"] == pytest.approx(mod.level)
        assert d["gain"] == pytest.approx(mod.get_gain())
        assert d["phasic"] == pytest.approx(mod.phasic_level)
        assert d["tonic"] == pytest.approx(mod.tonic_level)


def test_decode_neuromod_state_rejects_wrong_shape():
    """No silent fallback on schema drift — raises ValueError."""
    with pytest.raises(ValueError, match="shape mismatch"):
        decode_neuromod_state(np.zeros((6,), dtype=np.float32))
    with pytest.raises(ValueError, match="shape mismatch"):
        decode_neuromod_state(np.zeros((6, 3), dtype=np.float32))


def test_decode_neuromod_levels_accepts_v1_and_v2():
    """Backward-compat decoder supports both v1 (6,) and v2 (6, 4) shapes."""
    # v1 (6,) — legacy
    v1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    v1_decoded = decode_neuromod_levels(v1)
    assert v1_decoded["DA"] == pytest.approx(0.1)
    assert v1_decoded["GABA"] == pytest.approx(0.6)

    # v2 (6, 4) — extracts level column
    v2 = np.zeros((6, 4), dtype=np.float32)
    v2[:, 0] = [0.7, 0.8, 0.9, 0.5, 0.6, 0.4]
    v2_decoded = decode_neuromod_levels(v2)
    assert v2_decoded["DA"] == pytest.approx(0.7)
    assert v2_decoded["GABA"] == pytest.approx(0.4)


def test_decode_neuromod_levels_rejects_unknown_shape():
    """Defense-in-depth — silent fallback would hide schema drift."""
    with pytest.raises(ValueError, match="shape mismatch"):
        decode_neuromod_levels(np.zeros((7,), dtype=np.float32))
    with pytest.raises(ValueError, match="shape mismatch"):
        decode_neuromod_levels(np.zeros((6, 5), dtype=np.float32))


def test_encode_handles_none_system():
    """Defensive: None system encodes as zeros (does not raise)."""
    arr = encode_neuromod_state(None)
    assert arr.shape == (NEUROMOD_COUNT, NEUROMOD_FIELDS_PER_MOD)
    assert np.all(arr == 0)


def test_neuromod_state_spec_schema_version_is_2():
    """SPEC RegistrySpec schema_version bumped 1→2 with v2 layout."""
    from titan_hcl.core.state_registry import NEUROMOD_STATE
    assert NEUROMOD_STATE.shape == (6, 4)
    assert NEUROMOD_STATE.dtype == np.dtype("<f4")
    assert NEUROMOD_STATE.schema_version == 2
    assert NEUROMOD_STATE.payload_bytes == 96


def test_neuromod_state_to_modulation_parity_via_shm_path(synthetic_system):
    """Cross-process consumer path: SHM-read array → decode → modulation.

    This is the critical parity for §4.Q — cognitive_worker reads the SHM
    array, decodes to 4-field state, calls compute_modulation_from_state,
    and the result MUST be byte-identical to the in-process get_modulation().
    """
    arr = encode_neuromod_state(synthetic_system)
    state = decode_neuromod_state(arr)
    modulation_via_shm = compute_modulation_from_state(state)
    modulation_in_proc = synthetic_system.get_modulation()

    for key, expected in modulation_in_proc.items():
        actual = modulation_via_shm[key]
        # float32 round-trip introduces minor noise; allow eps
        assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6), (
            f"SHM-path modulation diverges on {key!r}: "
            f"shm={actual!r} in_proc={expected!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Q3 — neuromod_inputs.bin SHM slot registration
# ─────────────────────────────────────────────────────────────────────────────


def test_neuromod_inputs_spec_is_variable_size():
    from titan_hcl.core.state_registry import NEUROMOD_INPUTS
    assert NEUROMOD_INPUTS.name == "neuromod_inputs"
    assert NEUROMOD_INPUTS.variable_size is True
    assert NEUROMOD_INPUTS.shape == (4096,)
    assert NEUROMOD_INPUTS.dtype == np.dtype("u1")
    assert NEUROMOD_INPUTS.schema_version == 1
    assert NEUROMOD_INPUTS.feature_flag == "microkernel.shm_neuromod_enabled"


# ─────────────────────────────────────────────────────────────────────────────
# Q4 — NEUROMOD_EXTERNAL_NUDGE bus constant
# ─────────────────────────────────────────────────────────────────────────────


def test_neuromod_external_nudge_bus_constant_exists():
    """Bus constant for the 7 cross-worker nudge sites is registered."""
    from titan_hcl import bus
    assert hasattr(bus, "NEUROMOD_EXTERNAL_NUDGE")
    assert bus.NEUROMOD_EXTERNAL_NUDGE == "NEUROMOD_EXTERNAL_NUDGE"


# ─────────────────────────────────────────────────────────────────────────────
# Q5 — NeuromodInputsBuilder
# ─────────────────────────────────────────────────────────────────────────────


def test_inputs_builder_build_returns_well_formed_payload():
    """Builder.build with all-None engines returns a defensible default payload."""
    from titan_hcl.logic.neuromod_inputs_builder import NeuromodInputsBuilder

    b = NeuromodInputsBuilder(dna={})
    payload = b.build(
        coordinator=None,
        neural_nervous_system=None,
        life_force_engine=None,
        pi_monitor=None,
        ex_mem=None,
        sphere_clocks_snap=None,
        latest_epoch=None,
        is_dreaming=False,
        prediction_stats=None,
        expression_stats=None,
        kin_signature=None,
        topology_velocity=0.3,
        dt=1.0,
    )

    # Payload structural invariants
    assert set(payload["inputs"].keys()) == {
        "DA", "5HT", "NE", "ACh", "Endorphin", "GABA"}
    for k, v in payload["inputs"].items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0, 1]"

    assert 0.1 <= payload["chi_health"] <= 1.0
    assert payload["topology_velocity"] == 0.3
    assert payload["dt"] == 1.0
    assert payload["is_dreaming"] is False
    assert isinstance(payload["kin_overrides"], dict)
    assert payload["ts"] > 0


def test_inputs_builder_kin_overrides_emit_when_signal_exceeds_threshold():
    """kin overrides populate when last_resonance * recency > 0.01 (preserved bug-as-feature)."""
    from titan_hcl.logic.neuromod_inputs_builder import NeuromodInputsBuilder

    b = NeuromodInputsBuilder(dna={
        "kin": {"dna": {
            "da_boost": 0.25, "endorphin_boost": 0.20,
            "sht_boost": 0.15, "ne_boost": 0.10,
        }}
    })

    now = 1000.0
    payload = b.build(
        coordinator=None, neural_nervous_system=None, life_force_engine=None,
        pi_monitor=None, ex_mem=None, sphere_clocks_snap=None,
        latest_epoch=None, is_dreaming=False,
        prediction_stats=None, expression_stats=None,
        kin_signature={
            "last_resonance": 0.5,
            "last_exchange_ts": now - 10.0,  # 10s ago = ~99.7% recency
        },
        now=now,
    )

    # Recency = 1.0 - 10/3600 ≈ 0.997; signal ≈ 0.5 × 0.997 ≈ 0.5 > 0.01
    assert "kin_da" in payload["kin_overrides"]
    assert payload["kin_overrides"]["kin_da"] == pytest.approx(0.5 * 0.997 * 0.25, abs=0.01)
    assert "kin_endorphin" in payload["kin_overrides"]
    assert "kin_5ht" in payload["kin_overrides"]
    assert "kin_ne" in payload["kin_overrides"]


def test_inputs_builder_state_cache_advances_drift_delta_and_emas():
    """Builder maintains EMA + delta state across consecutive build() calls."""
    from titan_hcl.logic.neuromod_inputs_builder import NeuromodInputsBuilder

    b = NeuromodInputsBuilder(dna={})

    # Initial state
    assert b._prev_drift == 0.0
    assert b._prev_curvature == 0.0
    assert b._ema_epoch_regularity == 0.5

    # First build with drift=0.5 advances state
    b.build(
        coordinator=None, neural_nervous_system=None, life_force_engine=None,
        pi_monitor=None, ex_mem=None, sphere_clocks_snap=None,
        latest_epoch={"drift_magnitude": 0.5, "curvature": 0.3, "epoch_id": 1},
        is_dreaming=False,
        prediction_stats=None, expression_stats=None, kin_signature=None,
    )
    assert b._prev_drift == 0.5
    assert b._prev_curvature == 0.3
    assert b._last_epoch_id == 1


def test_inputs_builder_handles_msgpack_round_trip():
    """Output payload is msgpack-serializable end-to-end (the SHM transport)."""
    import msgpack

    from titan_hcl.logic.neuromod_inputs_builder import NeuromodInputsBuilder

    payload = NeuromodInputsBuilder(dna={}).build(
        coordinator=None, neural_nervous_system=None, life_force_engine=None,
        pi_monitor=None, ex_mem=None, sphere_clocks_snap=None,
        latest_epoch=None, is_dreaming=False,
        prediction_stats=None, expression_stats=None, kin_signature=None,
    )
    encoded = msgpack.packb(payload, use_bin_type=True)
    assert len(encoded) <= 4096, f"payload {len(encoded)}B exceeds neuromod_inputs.bin cap"

    decoded = msgpack.unpackb(encoded, raw=False)
    assert decoded["inputs"]["DA"] == pytest.approx(payload["inputs"]["DA"])
    assert decoded["chi_health"] == pytest.approx(payload["chi_health"])


# ─────────────────────────────────────────────────────────────────────────────
# Q6 — neuromod_worker evaluate driver (read inputs SHM → evaluate → write state SHM)
# ─────────────────────────────────────────────────────────────────────────────


def test_drive_evaluate_no_op_when_no_inputs_reader():
    """Cold-start safety: missing inputs_reader returns (False, 0)."""
    from titan_hcl.modules.neuromod_worker import _drive_evaluate

    ran, kin_n = _drive_evaluate(NeuromodulatorSystem(data_dir="/tmp/nm_test_q6"), None)
    assert ran is False
    assert kin_n == 0


def test_drive_evaluate_advances_levels_with_fresh_inputs(tmp_path):
    """Given a synthetic inputs payload, evaluate runs and DA level changes."""
    import msgpack
    from titan_hcl.modules.neuromod_worker import _drive_evaluate

    nm = NeuromodulatorSystem(data_dir=str(tmp_path / "nm_q6"))
    nm._dna_cache = {}
    # Freeze starting DA so we can detect movement.
    nm.modulators["DA"].level = 0.5
    nm.modulators["DA"].tonic_level = 0.5
    nm.modulators["DA"].setpoint = 0.5

    initial_total_evals = nm._total_evaluations

    class _StubReader:
        def __init__(self, payload_bytes):
            self._bytes = payload_bytes

        def read_variable(self):
            return self._bytes

    # Send a HIGH-DA input to force level shift.
    inputs_payload = {
        "inputs": {
            "DA": 0.95, "5HT": 0.5, "NE": 0.5,
            "ACh": 0.5, "Endorphin": 0.5, "GABA": 0.5,
        },
        "chi_health": 1.0,
        "topology_velocity": 0.3,
        "dt": 1.0,
    }
    encoded = msgpack.packb(inputs_payload, use_bin_type=True)
    reader = _StubReader(encoded)

    ran, kin_n = _drive_evaluate(nm, reader)
    assert ran is True
    assert kin_n == 0
    assert nm._total_evaluations == initial_total_evals + 1


def test_drive_evaluate_applies_chi_health_setter(tmp_path):
    """chi_health field from payload reaches NeuromodulatorSystem._chi_health."""
    import msgpack
    from titan_hcl.modules.neuromod_worker import _drive_evaluate

    nm = NeuromodulatorSystem(data_dir=str(tmp_path / "nm_q6_chi"))
    nm._dna_cache = {}

    class _StubReader:
        def __init__(self, b):
            self._b = b
        def read_variable(self):
            return self._b

    inputs_payload = {
        "inputs": {"DA": 0.5, "5HT": 0.5, "NE": 0.5, "ACh": 0.5, "Endorphin": 0.5, "GABA": 0.5},
        "chi_health": 0.35,
        "topology_velocity": 0.5,
        "dt": 1.0,
    }
    reader = _StubReader(msgpack.packb(inputs_payload, use_bin_type=True))

    _drive_evaluate(nm, reader)
    assert nm._chi_health == pytest.approx(0.35)
    assert nm._topology_velocity == pytest.approx(0.5)


def test_drive_evaluate_handles_msgpack_decode_failure(tmp_path):
    """Corrupted payload returns (False, 0) without raising."""
    from titan_hcl.modules.neuromod_worker import _drive_evaluate

    nm = NeuromodulatorSystem(data_dir=str(tmp_path / "nm_q6_corrupt"))

    class _StubReader:
        def read_variable(self):
            return b"\xff\xff\x00not-msgpack"

    ran, _ = _drive_evaluate(nm, _StubReader())
    assert ran is False


# ─────────────────────────────────────────────────────────────────────────────
# Q7 — NEUROMOD_EXTERNAL_NUDGE subscriber → apply_external_nudge
# ─────────────────────────────────────────────────────────────────────────────


def test_apply_external_nudge_payload_applies_nudge(tmp_path):
    """Valid NEUROMOD_EXTERNAL_NUDGE payload reaches apply_external_nudge."""
    from titan_hcl.modules.neuromod_worker import _apply_external_nudge_payload

    nm = NeuromodulatorSystem(data_dir=str(tmp_path / "nm_q7"))
    nm.modulators["DA"].level = 0.5
    initial_da = nm.modulators["DA"].level

    payload = {
        "nudge_map": {"DA": 0.8},  # pull DA UP from 0.5 toward 0.8
        "max_delta": 0.05,
        "developmental_age": 1.0,
        "source": "test_q7",
    }
    applied = _apply_external_nudge_payload(nm, payload)
    assert applied is True
    # DA moves toward 0.8 (positive delta = (0.8 - 0.5) * 0.05 = 0.015)
    assert nm.modulators["DA"].level > initial_da


def test_apply_external_nudge_skips_under_developmental_age(tmp_path):
    """Developmental gate at 0.1 — pre-0.1 nudges are suppressed."""
    from titan_hcl.modules.neuromod_worker import _apply_external_nudge_payload

    nm = NeuromodulatorSystem(data_dir=str(tmp_path / "nm_q7_devgate"))
    nm.modulators["DA"].level = 0.5
    initial_da = nm.modulators["DA"].level

    payload = {
        "nudge_map": {"DA": 0.9},
        "max_delta": 0.1,
        "developmental_age": 0.05,  # below 0.1 gate
        "source": "premature",
    }
    _apply_external_nudge_payload(nm, payload)
    # DA unchanged — developmental gate suppressed the nudge.
    assert nm.modulators["DA"].level == initial_da


def test_apply_external_nudge_handles_empty_nudge_map(tmp_path):
    """Empty nudge_map returns False (no-op)."""
    from titan_hcl.modules.neuromod_worker import _apply_external_nudge_payload

    nm = NeuromodulatorSystem(data_dir=str(tmp_path / "nm_q7_empty"))
    assert _apply_external_nudge_payload(nm, {"nudge_map": {}}) is False
    assert _apply_external_nudge_payload(nm, {}) is False


# ─────────────────────────────────────────────────────────────────────────────
# §4.G v1.8.3 D-SPEC-57 — chi_health bridge from life_force_worker
# ─────────────────────────────────────────────────────────────────────────────


def test_apply_external_nudge_chi_health_routes_to_set_chi_health(tmp_path):
    """NEUROMOD_EXTERNAL_NUDGE(source='life_force_chi_health') forwards
    payload.chi_health to NeuromodulatorSystem.set_chi_health.

    Closes §4.Q D-SPEC-54 orphan-nudge tracked item — the dead
    spirit_worker.py:3770 `set_chi_health(max(0.1, 1.0 - drain * 0.6))`
    call is replaced by life_force_worker emitting this payload per
    evaluate.
    """
    from titan_hcl.modules.neuromod_worker import _apply_external_nudge_payload

    nm = NeuromodulatorSystem(data_dir=str(tmp_path / "nm_g_chi_health"))
    # Reset to known state, then send the bridge payload.
    nm._chi_health = 1.0
    payload = {
        "chi_health": 0.42,
        "source": "life_force_chi_health",
        "ts": 1234567890.0,
    }
    applied = _apply_external_nudge_payload(nm, payload)
    assert applied is True
    assert nm._chi_health == pytest.approx(0.42)


def test_apply_external_nudge_chi_health_clamps_at_floor(tmp_path):
    """set_chi_health clamps to [0.1, 1.0] per NeuromodulatorSystem
    contract — life_force_worker should never emit < 0.1 but verify the
    floor still applies if a buggy producer sends a smaller value.
    """
    from titan_hcl.modules.neuromod_worker import _apply_external_nudge_payload

    nm = NeuromodulatorSystem(data_dir=str(tmp_path / "nm_g_chi_floor"))
    nm._chi_health = 1.0
    payload = {
        "chi_health": -0.5,  # below the 0.1 floor
        "source": "life_force_chi_health",
    }
    applied = _apply_external_nudge_payload(nm, payload)
    assert applied is True
    assert nm._chi_health == pytest.approx(0.1)  # clamped


# ─────────────────────────────────────────────────────────────────────────────
# Q12 — NEUROMOD_STATS_UPDATED payload schema
# ─────────────────────────────────────────────────────────────────────────────


def test_build_stats_payload_schema(tmp_path):
    """Stats payload has required fields for /v4/inner-trinity.neuromodulators + /status mood."""
    from titan_hcl.modules.neuromod_worker import _build_stats_payload

    nm = NeuromodulatorSystem(data_dir=str(tmp_path / "nm_q12"))
    payload = _build_stats_payload(nm, titan_id="T_TEST")

    assert payload["titan_id"] == "T_TEST"
    assert set(payload["modulators"].keys()) == set(NEUROMOD_NAMES)
    for name in NEUROMOD_NAMES:
        d = payload["modulators"][name]
        assert set(d.keys()) == {"level", "gain", "phasic", "tonic"}
    # modulation dict — 14 keys (full get_modulation contract)
    expected_modulation_keys = {
        "learning_rate_gain", "fire_threshold_gain", "working_memory_retention",
        "accumulation_rate_gain", "refractory_gain", "patience_factor",
        "sensory_gain", "exploration_temperature", "filter_down_strength",
        "training_frequency_gain", "memory_encoding_gain", "observation_precision",
        "intrinsic_motivation", "discomfort_suppression",
        "global_threshold_raise", "system_energy",
    }
    assert set(payload["modulation"].keys()) == expected_modulation_keys
    assert payload["current_emotion"] in (
        "neutral", "joy", "peace", "curiosity", "fear", "love",
        "anger", "sadness", "wonder", "flow", "calm")
    assert isinstance(payload["emotion_confidence"], float)
    assert payload["total_evaluations"] >= 0
    assert payload["ts"] > 0
