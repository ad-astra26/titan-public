"""
Tests for Microkernel v2 Phase A §A.2 part 2 (S4) — fixed-size registries
+ identity shm writer.

Covers:
  - SPHERE_CLOCKS_STATE / CHI_STATE / TITANVM_REGISTERS / IDENTITY specs
    are well-formed (name, dtype, shape, feature flag, payload bytes)
  - Identity shm writer in TitanKernel.boot() — kernel_instance_nonce is
    random per construction; titan_id + maker_pubkey deterministic.
  - RegistryBank lazily creates writers for all 4 fixed-size S4 specs.

The Phase-A spirit_worker shm-writer helpers (_write_sphere_clocks_shm /
_write_chi_shm / _write_titanvm_shm) and their tests were RETIRED with the
legacy_core monolith / spirit_worker dead-helper purge (2026-05-21,
D-SPEC-106). The WRITE capability was not dropped — it was re-homed:
  - titanvm_registers.bin → ns_worker (Python L2); encode/decode + (11,4)
    tensor layout covered by tests/test_ns_worker.py (encode_ns_state
    round-trip, shape-mismatch guard, payload bytes, schema version).
  - chi_state.bin + sphere_clocks → Rust L1 trinity daemons
    (titan-rust/crates/titan-trinity-rs/src/chi_state.rs +
    titan-unified-spirit-rs/src/slot_handles.rs), per
    feedback_phase_c_trinity_lives_in_rust_l0_l1; covered by the Rust test
    harness (chi_parity_dump, test_kernel_boot). No Python writer remains.

Reference:
  - titan-docs/PLAN_microkernel_phase_a_s4.md §5.1 + §5.2
  - titan_hcl/core/state_registry.py declarations
  - titan_hcl/core/kernel.py _write_identity_shm
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from titan_hcl.core.state_registry import (
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


# ── Spirit-worker shm-writer helper tests RETIRED 2026-05-21 (D-SPEC-106) ──
# The _write_sphere_clocks_shm / _write_chi_shm / _write_titanvm_shm helpers
# were deleted with the spirit_worker dead-helper purge. Coverage of the
# re-homed writers lives elsewhere (see module docstring):
#   - titanvm_registers → tests/test_ns_worker.py (encode_ns_state round-trip)
#   - chi_state + sphere_clocks → Rust titan-trinity-rs / titan-unified-spirit-rs


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
