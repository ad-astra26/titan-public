"""
Tests for Microkernel v2 Phase A §A.2 part 2 (S4) D11 — variable-size
RegistrySpec capability.

Covers:
  - RegistrySpec(variable_size=True) constructs cleanly with arbitrary
    max shape; payload_bytes returns max bytes.
  - StateRegistryWriter.write_variable: empty payload, mid-size, max
    payload, overflow rejection, fixed-spec rejection.
  - StateRegistryReader.read_variable: empty, mid, max roundtrip; fixed
    spec rejects read_variable.
  - SeqLock semantics preserved under variable-size writes.
  - Header CRC validates correctly for variable-size payloads.
  - Backward compat: fixed specs continue to work via write/read.

Reference:
  - titan-docs/PLAN_microkernel_phase_a_s4.md §3 D11 + §4.0 + §5.4
  - titan_plugin/core/state_registry.py write_variable / read_variable
"""
from __future__ import annotations

import numpy as np
import pytest

from titan_plugin.core.state_registry import (
    HEADER_SIZE,
    RegistrySpec,
    StateRegistryReader,
    StateRegistryWriter,
)


@pytest.fixture
def shm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


def _make_var_spec(name: str = "test_var", max_bytes: int = 1024) -> RegistrySpec:
    return RegistrySpec(
        name=name,
        dtype=np.dtype("u1"),
        shape=(max_bytes,),
        variable_size=True,
    )


# ── Spec construction ──────────────────────────────────────────────


def test_variable_size_spec_construction():
    spec = _make_var_spec("foo", 256)
    assert spec.variable_size is True
    assert spec.shape == (256,)
    assert spec.payload_bytes == 256
    assert spec.total_bytes == HEADER_SIZE + 256


def test_default_spec_variable_size_false():
    """Existing fixed specs default variable_size=False (backward compat)."""
    spec = RegistrySpec(name="foo", dtype=np.dtype("<f4"), shape=(10,))
    assert spec.variable_size is False


# ── Writer: write_variable ─────────────────────────────────────────


def test_write_variable_empty_payload(shm_root):
    spec = _make_var_spec("v_empty", 1024)
    w = StateRegistryWriter(spec, shm_root)
    seq = w.write_variable(b"")
    assert seq == 2  # 0 → 1 (odd) → 2 (even)


def test_write_variable_small_payload(shm_root):
    spec = _make_var_spec("v_small", 1024)
    w = StateRegistryWriter(spec, shm_root)
    seq = w.write_variable(b"hello world")
    assert seq == 2


def test_write_variable_max_payload(shm_root):
    spec = _make_var_spec("v_max", 256)
    w = StateRegistryWriter(spec, shm_root)
    payload = b"x" * 256
    seq = w.write_variable(payload)
    assert seq == 2


def test_write_variable_overflow_rejected(shm_root):
    spec = _make_var_spec("v_over", 256)
    w = StateRegistryWriter(spec, shm_root)
    with pytest.raises(ValueError, match="exceeds preallocated max"):
        w.write_variable(b"x" * 257)


def test_write_variable_rejects_fixed_spec(shm_root):
    spec = RegistrySpec(name="fixed", dtype=np.dtype("<f4"), shape=(10,))
    w = StateRegistryWriter(spec, shm_root)
    with pytest.raises(ValueError, match="variable_size=True"):
        w.write_variable(b"abc")


# ── Reader: read_variable ──────────────────────────────────────────


def test_read_variable_empty_roundtrip(shm_root):
    spec = _make_var_spec("rd_empty", 1024)
    w = StateRegistryWriter(spec, shm_root)
    w.write_variable(b"")
    r = StateRegistryReader(spec, shm_root)
    out = r.read_variable()
    r.close()
    assert out == b""


def test_read_variable_small_roundtrip(shm_root):
    spec = _make_var_spec("rd_small", 1024)
    w = StateRegistryWriter(spec, shm_root)
    w.write_variable(b"the quick brown fox")
    r = StateRegistryReader(spec, shm_root)
    out = r.read_variable()
    r.close()
    assert out == b"the quick brown fox"


def test_read_variable_max_roundtrip(shm_root):
    spec = _make_var_spec("rd_max", 256)
    w = StateRegistryWriter(spec, shm_root)
    payload = bytes(range(256))
    w.write_variable(payload)
    r = StateRegistryReader(spec, shm_root)
    out = r.read_variable()
    r.close()
    assert out == payload


def test_read_variable_rejects_fixed_spec(shm_root):
    spec = RegistrySpec(name="fixed_rd", dtype=np.dtype("<f4"), shape=(10,))
    w = StateRegistryWriter(spec, shm_root)
    arr = np.zeros((10,), dtype=np.float32)
    w.write(arr)  # populate via fixed write
    r = StateRegistryReader(spec, shm_root)
    out = r.read_variable()
    r.close()
    # Fixed spec → read_variable returns None (fallback)
    assert out is None


# ── SeqLock + header CRC under variable-size writes ────────────────


def test_seqlock_advances_per_variable_write(shm_root):
    spec = _make_var_spec("seq_test", 1024)
    w = StateRegistryWriter(spec, shm_root)
    seq1 = w.write_variable(b"a")
    seq2 = w.write_variable(b"bb")
    seq3 = w.write_variable(b"ccc")
    assert seq1 == 2 and seq2 == 4 and seq3 == 6
    assert seq1 < seq2 < seq3


def test_meta_payload_bytes_reflects_actual(shm_root):
    """Header's payload_bytes records ACTUAL size, not max."""
    spec = _make_var_spec("meta_test", 1024)
    w = StateRegistryWriter(spec, shm_root)
    w.write_variable(b"x" * 42)

    r = StateRegistryReader(spec, shm_root)
    meta = r.read_meta()
    r.close()
    assert meta is not None
    assert meta["payload_bytes"] == 42
    assert meta["seq"] == 2
    assert meta["write_in_progress"] is False


def test_variable_size_changing_payload_size_works(shm_root):
    """Successive writes can have different sizes."""
    spec = _make_var_spec("changing", 1024)
    w = StateRegistryWriter(spec, shm_root)
    r = StateRegistryReader(spec, shm_root)

    w.write_variable(b"big" * 100)
    assert r.read_variable() == b"big" * 100

    w.write_variable(b"small")
    out = r.read_variable()
    assert out == b"small"


# ── Backward compat: fixed-size specs unchanged ────────────────────


def test_fixed_size_write_read_unchanged(shm_root):
    spec = RegistrySpec(name="back_compat", dtype=np.dtype("<f4"),
                        shape=(5,))
    w = StateRegistryWriter(spec, shm_root)
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    w.write(arr)

    r = StateRegistryReader(spec, shm_root)
    out = r.read()
    r.close()
    assert out is not None
    assert np.array_equal(out, arr)
