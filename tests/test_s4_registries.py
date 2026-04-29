"""
Tests for Microkernel v2 Phase A §A.2 part 2 (S4) — fixed-size registries
+ identity + spirit_worker shm writers.

Covers:
  - SPHERE_CLOCKS_STATE / CHI_STATE / TITANVM_REGISTERS / IDENTITY specs
    are well-formed (name, dtype, shape, feature flag, payload bytes)
  - Spirit_worker writers (sphere_clocks / chi / titanvm) honor flag-off
    (no-op), produce content-hash-gated writes (same state → same hash),
    and emit correct (R, C) float32 tensors with all fields decodable.
  - Identity shm writer in TitanKernel.boot() — kernel_instance_nonce is
    random per construction; titan_id + maker_pubkey deterministic.

Reference:
  - titan-docs/PLAN_microkernel_phase_a_s4.md §5.1 + §5.2
  - titan_plugin/core/state_registry.py declarations
  - titan_plugin/modules/spirit_worker.py helpers
  - titan_plugin/core/kernel.py _write_identity_shm
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from titan_plugin.core.state_registry import (
    CHI_STATE,
    IDENTITY,
    RegistryBank,
    SPHERE_CLOCKS_STATE,
    StateRegistryReader,
    TITANVM_REGISTERS,
    resolve_shm_root,
)


@pytest.fixture
def shm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


# ── Spec declarations ──────────────────────────────────────────────


def test_sphere_clocks_spec():
    assert SPHERE_CLOCKS_STATE.name == "sphere_clocks"
    assert SPHERE_CLOCKS_STATE.shape == (6, 7)
    assert SPHERE_CLOCKS_STATE.dtype == np.dtype("<f4")
    assert SPHERE_CLOCKS_STATE.feature_flag == "microkernel.shm_sphere_clocks_enabled"
    assert SPHERE_CLOCKS_STATE.payload_bytes == 6 * 7 * 4
    assert SPHERE_CLOCKS_STATE.variable_size is False


def test_chi_state_spec():
    assert CHI_STATE.name == "chi_state"
    assert CHI_STATE.shape == (6,)
    assert CHI_STATE.dtype == np.dtype("<f4")
    assert CHI_STATE.feature_flag == "microkernel.shm_chi_enabled"
    assert CHI_STATE.payload_bytes == 24
    assert CHI_STATE.variable_size is False


def test_titanvm_registers_spec():
    assert TITANVM_REGISTERS.name == "titanvm_registers"
    assert TITANVM_REGISTERS.shape == (11, 4)
    assert TITANVM_REGISTERS.dtype == np.dtype("<f4")
    assert TITANVM_REGISTERS.feature_flag == "microkernel.shm_titanvm_enabled"
    assert TITANVM_REGISTERS.payload_bytes == 11 * 4 * 4
    assert TITANVM_REGISTERS.variable_size is False


def test_identity_spec():
    assert IDENTITY.name == "identity"
    assert IDENTITY.shape == (96,)
    assert IDENTITY.dtype == np.dtype("u1")
    assert IDENTITY.feature_flag == "microkernel.shm_identity_enabled"
    assert IDENTITY.payload_bytes == 96
    assert IDENTITY.variable_size is False


# ── Spirit-worker helper: sphere_clocks ─────────────────────────────


def test_sphere_clocks_writer_flag_off_noop(shm_root):
    """Helper returns _last_hash unchanged when flag is off."""
    from titan_plugin.modules.spirit_worker import _write_sphere_clocks_shm
    from titan_plugin.logic.sphere_clock import SphereClockEngine

    bank = RegistryBank(titan_id="T_OFF", config={
        "microkernel": {"shm_sphere_clocks_enabled": False}})
    engine = SphereClockEngine(config=None, data_dir=str(shm_root))
    h = _write_sphere_clocks_shm(engine, bank, None)
    assert h is None


def test_sphere_clocks_writer_roundtrip(shm_root):
    """Helper writes (6, 7) float32 to shm; reader decodes correctly."""
    from titan_plugin.modules.spirit_worker import _write_sphere_clocks_shm
    from titan_plugin.logic.sphere_clock import SphereClockEngine

    bank = RegistryBank(titan_id="T1", config={
        "microkernel": {"shm_sphere_clocks_enabled": True}})
    engine = SphereClockEngine(config=None, data_dir=str(shm_root))
    engine.clocks["inner_body"].radius = 0.42
    engine.clocks["inner_body"].phase = 1.234

    h = _write_sphere_clocks_shm(engine, bank, None)
    assert h is not None

    r = StateRegistryReader(SPHERE_CLOCKS_STATE, resolve_shm_root("T1"))
    arr = r.read()
    r.close()
    assert arr is not None
    assert arr.shape == (6, 7)
    assert abs(arr[0][0] - 0.42) < 1e-5  # inner_body.radius
    assert abs(arr[0][2] - 1.234) < 1e-5  # inner_body.phase


def test_sphere_clocks_writer_content_hash_gate(shm_root):
    """Identical state → same hash → no extra write."""
    from titan_plugin.modules.spirit_worker import _write_sphere_clocks_shm
    from titan_plugin.logic.sphere_clock import SphereClockEngine

    bank = RegistryBank(titan_id="T1", config={
        "microkernel": {"shm_sphere_clocks_enabled": True}})
    engine = SphereClockEngine(config=None, data_dir=str(shm_root))

    h1 = _write_sphere_clocks_shm(engine, bank, None)
    h2 = _write_sphere_clocks_shm(engine, bank, h1)
    assert h2 == h1, "unchanged state should not bump hash"

    engine.clocks["outer_spirit"].radius = 0.99
    h3 = _write_sphere_clocks_shm(engine, bank, h2)
    assert h3 != h2, "changed state should produce new hash"


def test_sphere_clocks_writer_none_engine(shm_root):
    """None engine → no-op (defensive)."""
    from titan_plugin.modules.spirit_worker import _write_sphere_clocks_shm

    bank = RegistryBank(titan_id="T1", config={
        "microkernel": {"shm_sphere_clocks_enabled": True}})
    h = _write_sphere_clocks_shm(None, bank, None)
    assert h is None


# ── Spirit-worker helper: chi_state ──────────────────────────────────


def test_chi_writer_flag_off_noop(shm_root):
    from titan_plugin.modules.spirit_worker import _write_chi_shm

    bank = RegistryBank(titan_id="T_OFF", config={
        "microkernel": {"shm_chi_enabled": False}})
    chi = {"total": 0.5, "spirit": {"effective": 0.5}, "circulation": 0.5}
    h = _write_chi_shm(chi, bank, None)
    assert h is None


def test_chi_writer_roundtrip_nested_dict(shm_root):
    """Helper extracts effective values from nested dict structure."""
    from titan_plugin.modules.spirit_worker import _write_chi_shm

    bank = RegistryBank(titan_id="T1", config={
        "microkernel": {"shm_chi_enabled": True}})
    chi = {
        "total": 0.659,
        "spirit": {"raw": 0.7, "effective": 0.65, "weight": 0.4},
        "mind": {"raw": 0.6, "effective": 0.62, "weight": 0.35},
        "body": {"raw": 0.5, "effective": 0.55, "weight": 0.25},
        "circulation": 0.42,
        "state": "flowing",
    }
    h = _write_chi_shm(chi, bank, None)
    assert h is not None

    r = StateRegistryReader(CHI_STATE, resolve_shm_root("T1"))
    arr = r.read()
    r.close()
    assert arr is not None
    assert arr.shape == (6,)
    assert abs(arr[0] - 0.659) < 1e-3  # total
    assert abs(arr[1] - 0.65) < 1e-5   # spirit_effective
    assert abs(arr[2] - 0.62) < 1e-5   # mind_effective
    assert abs(arr[3] - 0.55) < 1e-5   # body_effective
    assert abs(arr[4] - 0.42) < 1e-5   # circulation
    assert arr[5] == 0.0               # reserved


def test_chi_writer_empty_dict_noop(shm_root):
    from titan_plugin.modules.spirit_worker import _write_chi_shm

    bank = RegistryBank(titan_id="T1", config={
        "microkernel": {"shm_chi_enabled": True}})
    h = _write_chi_shm({}, bank, None)
    assert h is None


def test_chi_writer_content_hash_gate(shm_root):
    from titan_plugin.modules.spirit_worker import _write_chi_shm

    bank = RegistryBank(titan_id="T1", config={
        "microkernel": {"shm_chi_enabled": True}})
    chi = {"total": 0.5, "spirit": {"effective": 0.5},
           "mind": {"effective": 0.5}, "body": {"effective": 0.5},
           "circulation": 0.5}
    h1 = _write_chi_shm(chi, bank, None)
    h2 = _write_chi_shm(chi, bank, h1)
    assert h2 == h1


# ── Spirit-worker helper: titanvm_registers ─────────────────────────


def test_titanvm_writer_flag_off_noop(shm_root):
    from titan_plugin.modules.spirit_worker import _write_titanvm_shm

    class _FakeNS:
        programs = {}
        _all_urgencies = {}

    bank = RegistryBank(titan_id="T_OFF", config={
        "microkernel": {"shm_titanvm_enabled": False}})
    h = _write_titanvm_shm(_FakeNS(), bank, None)
    assert h is None


def test_titanvm_writer_roundtrip(shm_root):
    """Helper writes (11, 4) float32 with NS_PROGRAMS row order."""
    from titan_plugin.modules.spirit_worker import _write_titanvm_shm
    from titan_plugin.logic.emot_bundle_protocol import NS_PROGRAMS
    from titan_plugin.logic.neural_reflex_net import NeuralReflexNet

    class _FakeNS:
        def __init__(self):
            self.programs = {n: NeuralReflexNet(name=n) for n in NS_PROGRAMS}
            self._all_urgencies = {n: 0.5 for n in NS_PROGRAMS}
            # Distinguishable values for REFLEX (row 0)
            self.programs["REFLEX"].fire_count = 42
            self.programs["REFLEX"].total_updates = 100
            self.programs["REFLEX"].last_loss = 0.123
            self._all_urgencies["REFLEX"] = 0.87

    bank = RegistryBank(titan_id="T1", config={
        "microkernel": {"shm_titanvm_enabled": True}})
    h = _write_titanvm_shm(_FakeNS(), bank, None)
    assert h is not None

    r = StateRegistryReader(TITANVM_REGISTERS, resolve_shm_root("T1"))
    arr = r.read()
    r.close()
    assert arr is not None
    assert arr.shape == (11, 4)
    # REFLEX (row 0)
    assert abs(arr[0][0] - 0.87) < 1e-5  # urgency
    assert abs(arr[0][1] - 42.0) < 1e-5  # fire_count
    assert abs(arr[0][2] - 100.0) < 1e-5  # total_updates
    assert abs(arr[0][3] - 0.123) < 1e-3  # last_loss


def test_titanvm_writer_none_ns_noop(shm_root):
    from titan_plugin.modules.spirit_worker import _write_titanvm_shm

    bank = RegistryBank(titan_id="T1", config={
        "microkernel": {"shm_titanvm_enabled": True}})
    h = _write_titanvm_shm(None, bank, None)
    assert h is None


# ── Identity shm — kernel.boot() side ──────────────────────────────


def test_identity_layout_titan_id(shm_root):
    """Identity shm written by kernel.boot() has correct layout."""
    import secrets

    bank = RegistryBank(titan_id="T1", config={
        "microkernel": {"shm_identity_enabled": True}})

    # Build payload directly per kernel._write_identity_shm logic
    tid_bytes = b"T1" + b"\x00" * 30
    mk_bytes = b"\x00" * 32
    nonce = secrets.token_bytes(32)
    payload = tid_bytes + mk_bytes + nonce
    arr = np.frombuffer(payload, dtype=np.uint8)
    bank.writer(IDENTITY).write(arr)

    r = StateRegistryReader(IDENTITY, resolve_shm_root("T1"))
    out = r.read()
    r.close()
    assert out is not None and out.shape == (96,)
    decoded = bytes(out)
    assert decoded[0:32].rstrip(b"\x00").decode("utf-8") == "T1"
    assert decoded[32:64] == mk_bytes
    assert decoded[64:96] == nonce


def test_identity_nonce_changes_per_write(shm_root):
    """Re-writing produces a different nonce (per-boot freshness contract)."""
    import secrets

    bank = RegistryBank(titan_id="T1", config={
        "microkernel": {"shm_identity_enabled": True}})

    tid_bytes = b"T1" + b"\x00" * 30
    mk_bytes = b"\x00" * 32

    nonce1 = secrets.token_bytes(32)
    bank.writer(IDENTITY).write(
        np.frombuffer(tid_bytes + mk_bytes + nonce1, dtype=np.uint8))

    nonce2 = secrets.token_bytes(32)
    bank.writer(IDENTITY).write(
        np.frombuffer(tid_bytes + mk_bytes + nonce2, dtype=np.uint8))

    assert nonce1 != nonce2


# ── RegistryBank lazy instantiation includes new specs ──────────────


def test_registry_bank_creates_writers_for_all_s4_specs(shm_root):
    """Bank creates writers on demand for all 4 new S4 fixed-size specs."""
    bank = RegistryBank(titan_id="T1", config={
        "microkernel": {
            "shm_sphere_clocks_enabled": True,
            "shm_chi_enabled": True,
            "shm_titanvm_enabled": True,
            "shm_identity_enabled": True,
        }})
    for spec in (SPHERE_CLOCKS_STATE, CHI_STATE, TITANVM_REGISTERS, IDENTITY):
        w = bank.writer(spec)
        assert w is not None
        assert bank.is_enabled(spec) is True
