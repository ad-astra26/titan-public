"""
Tests for Microkernel v2 Phase A §A.7 / S3b — INNER_SPIRIT_45D shm registry.

Covers:
  - RegistrySpec INNER_SPIRIT_45D is well-formed (name, dtype, shape, feature flag)
  - Writer roundtrip: write a 45D float32 tensor, read it back byte-equivalent
  - Feature flag gating: is_enabled respects microkernel.shm_spirit_fast_enabled
  - Seq counter advances monotonically under repeated writes
  - Shape mismatch is surfaced (writer raises when shape != (45,))
  - Content-hash gate is implicit (each write bumps seq regardless — content
    hashing is the CALLER'S responsibility, matching spirit_worker hook design)
  - Payload size: 45 × 4 bytes = 180 B + 24 B header = 204 B total

Uses pytest tmp_path + TITAN_SHM_ROOT env override. No /dev/shm pollution.

Reference:
  - titan-docs/PLAN_microkernel_phase_a_s3.md §6.3
  - titan_plugin/core/state_registry.py INNER_SPIRIT_45D declaration
  - titan_plugin/modules/spirit_worker.py:2036+ writer hook
"""
from __future__ import annotations

import numpy as np
import pytest

from titan_plugin.core.state_registry import (
    HEADER_SIZE,
    INNER_SPIRIT_45D,
    RegistryBank,
    StateRegistryReader,
    StateRegistryWriter,
)


@pytest.fixture
def shm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


def test_inner_spirit_45d_spec_shape():
    assert INNER_SPIRIT_45D.name == "inner_spirit_45d"
    assert INNER_SPIRIT_45D.shape == (45,)
    assert INNER_SPIRIT_45D.dtype == np.dtype("<f4")
    assert INNER_SPIRIT_45D.feature_flag == "microkernel.shm_spirit_fast_enabled"


def test_inner_spirit_45d_payload_size():
    """45D × 4 bytes = 180 B payload; total = 180 + 24 B header = 204 B."""
    assert INNER_SPIRIT_45D.payload_bytes == 180
    assert INNER_SPIRIT_45D.total_bytes == 180 + HEADER_SIZE


def test_writer_reader_roundtrip_byte_equivalent(shm_root):
    """Writing a 45D tensor and reading it back yields byte-equivalent bytes."""
    writer = StateRegistryWriter(INNER_SPIRIT_45D, shm_root)
    reader = StateRegistryReader(INNER_SPIRIT_45D, shm_root)
    try:
        # Deterministic test vector: sat(15) + chit(15) + ananda(15)
        original = np.linspace(0.0, 1.0, 45, dtype=np.float32)
        writer.write(original)

        read_back = reader.read()
        assert read_back is not None
        assert read_back.shape == (45,)
        assert read_back.dtype == np.float32
        np.testing.assert_array_equal(original, read_back)
    finally:
        writer.close()
        reader.close()


def test_seq_counter_advances_on_write(shm_root):
    writer = StateRegistryWriter(INNER_SPIRIT_45D, shm_root)
    reader = StateRegistryReader(INNER_SPIRIT_45D, shm_root)
    try:
        arr = np.zeros(45, dtype=np.float32)
        writer.write(arr)
        meta1 = reader.read_meta()
        seq1 = meta1.get("seq", 0)

        arr2 = np.ones(45, dtype=np.float32)
        writer.write(arr2)
        meta2 = reader.read_meta()
        seq2 = meta2.get("seq", 0)

        assert seq2 > seq1, f"seq did not advance: {seq1} → {seq2}"
    finally:
        writer.close()
        reader.close()


def test_feature_flag_disabled_by_default(shm_root):
    """Without any flag flipped, is_enabled returns False."""
    bank = RegistryBank(titan_id=None, config={})
    assert bank.is_enabled(INNER_SPIRIT_45D) is False


def test_feature_flag_respected_when_true(shm_root):
    """Flipping the flag in config makes is_enabled return True."""
    bank = RegistryBank(
        titan_id=None,
        config={"microkernel": {"shm_spirit_fast_enabled": True}},
    )
    assert bank.is_enabled(INNER_SPIRIT_45D) is True


def test_shape_mismatch_rejected(shm_root):
    """Writer raises on wrong shape (contract check)."""
    writer = StateRegistryWriter(INNER_SPIRIT_45D, shm_root)
    try:
        # Wrong shape — writer should reject
        wrong = np.zeros(5, dtype=np.float32)
        with pytest.raises((ValueError, AssertionError)):
            writer.write(wrong)
    finally:
        writer.close()


def test_multiple_writes_same_content_still_advance_seq(shm_root):
    """Writer seq advances even on identical content — content-hash
    gating is the CALLER'S responsibility (done in spirit_worker hook
    per PLAN §5.2). This matches the Trinity writer pattern where the
    content-hash lives in _start_trinity_shm_writer, not the Writer."""
    writer = StateRegistryWriter(INNER_SPIRIT_45D, shm_root)
    reader = StateRegistryReader(INNER_SPIRIT_45D, shm_root)
    try:
        arr = np.full(45, 0.5, dtype=np.float32)
        writer.write(arr)
        seq1 = reader.read_meta()["seq"]
        writer.write(arr)
        seq2 = reader.read_meta()["seq"]
        assert seq2 > seq1


    finally:
        writer.close()
        reader.close()


def test_bank_writer_reader_are_cached(shm_root):
    """RegistryBank returns the same writer/reader instance per spec."""
    bank = RegistryBank(
        titan_id=None,
        config={"microkernel": {"shm_spirit_fast_enabled": True}},
    )
    try:
        w1 = bank.writer(INNER_SPIRIT_45D)
        w2 = bank.writer(INNER_SPIRIT_45D)
        assert w1 is w2

        r1 = bank.reader(INNER_SPIRIT_45D)
        r2 = bank.reader(INNER_SPIRIT_45D)
        assert r1 is r2
    finally:
        bank.close_all()


def test_end_to_end_via_registry_bank(shm_root):
    """Full bank-level roundtrip matching spirit_worker's actual code path."""
    bank = RegistryBank(
        titan_id=None,
        config={"microkernel": {"shm_spirit_fast_enabled": True}},
    )
    try:
        assert bank.is_enabled(INNER_SPIRIT_45D) is True
        tensor = np.random.default_rng(42).standard_normal(45).astype(np.float32)
        bank.writer(INNER_SPIRIT_45D).write(tensor)
        read_back = bank.reader(INNER_SPIRIT_45D).read()
        assert read_back is not None
        np.testing.assert_array_equal(tensor, read_back)
    finally:
        bank.close_all()
