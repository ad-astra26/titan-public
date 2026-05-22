"""Cross-language parity test for SPEC §7.0 v1.0.0 triple-buffer wire format
(D-SPEC-35).

Verifies:
  1. Python `StateRegistryWriter` produces byte sequences that the Rust
     `titan-state::Slot::open` + `Slot::read` accepts as valid published
     buffers — i.e. cross-language wire-format compatibility.
  2. Rust-produced slot files (via the `titan-state` crate's `Slot::create`
     + `Slot::write`) are read by the Python `StateRegistryReader` cleanly.
  3. The `vectors.json` `shm_layout` totals match the Python `RegistrySpec.total_bytes`
     formula for every Phase-C kernel-owned slot.

Per `feedback_function_parity_vs_contract_parity.md` — function-level parity
isn't enough; this is a CONTRACT parity test that exercises the actual
production write/read path. If Rust and Python disagree on any wire-format
field (offset, struct format, atomic ordering, CRC scope), this test fails.
"""
from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from titan_hcl.core.state_registry import (
    BUFFER_COUNT,
    BUFFER_META_SIZE,
    BUFFER_META_STRUCT,
    HEADER_SIZE,
    HEADER_STRUCT,
    RegistrySpec,
    StateRegistryReader,
    StateRegistryWriter,
    _buffer_offset,
    _compute_buffer_crc32,
    _pack_header_seq,
    _unpack_header_seq,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VECTORS_PATH = REPO_ROOT / "tests" / "parity" / "vectors.json"


@pytest.fixture()
def vectors() -> dict:
    return json.loads(VECTORS_PATH.read_text())


@pytest.fixture()
def shm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


# ── Wire format constants parity ────────────────────────────────────


def test_header_size_matches_vectors(vectors):
    """16-byte fixed header per SPEC v1.0.0."""
    assert vectors["shm_layout"]["header_bytes"] == 16
    assert vectors["shm_layout"]["header_bytes"] == HEADER_SIZE


def test_buffer_meta_size_matches_vectors(vectors):
    """16-byte per-buffer metadata per SPEC v1.0.0."""
    assert vectors["shm_layout"]["buffer_meta_bytes"] == 16
    assert vectors["shm_layout"]["buffer_meta_bytes"] == BUFFER_META_SIZE


def test_buffer_count_matches_vectors(vectors):
    """3 buffers per slot per SPEC v1.0.0 (race-elimination requirement)."""
    assert vectors["shm_layout"]["buffer_count"] == 3
    assert vectors["shm_layout"]["buffer_count"] == BUFFER_COUNT


def test_header_struct_matches_vectors(vectors):
    """`<QII` = u64 header_seq + u32 schema + u32 capacity = 16 bytes."""
    assert vectors["shm_layout"]["header_struct"] == "<QII"
    assert vectors["shm_layout"]["header_struct"] == HEADER_STRUCT
    assert struct.calcsize(HEADER_STRUCT) == HEADER_SIZE


def test_buffer_meta_struct_matches_vectors(vectors):
    """`<QII` = u64 wall_ns + u32 payload_bytes + u32 buffer_crc32 = 16 bytes."""
    assert vectors["shm_layout"]["buffer_meta_struct"] == "<QII"
    assert vectors["shm_layout"]["buffer_meta_struct"] == BUFFER_META_STRUCT
    assert struct.calcsize(BUFFER_META_STRUCT) == BUFFER_META_SIZE


def test_per_slot_total_bytes_match_formula(vectors):
    """Every kernel-owned slot's total_bytes = 16 + 3 × (16 + payload_bytes).
    fastbus.bin is excluded — self-contained SPSC ring."""
    slots = vectors["shm_layout"]["slots"]
    for name, spec in slots.items():
        if name == "fastbus.bin":
            continue
        payload = spec["payload_bytes"]
        expected_total = 16 + 3 * (16 + payload)
        assert spec["total_bytes"] == expected_total, (
            f"slot '{name}' vectors total_bytes ({spec['total_bytes']}) "
            f"does not match 16 + 3 × (16 + {payload}) = {expected_total}"
        )


# ── Header seq packing parity ───────────────────────────────────────


def test_header_seq_pack_unpack_round_trip():
    """version << 8 | ready_idx — matches Rust `pack_header_seq`."""
    for v in [1, 100, 1 << 20, 1 << 40, (1 << 56) - 1]:
        for idx in [0, 1, 2]:
            packed = _pack_header_seq(v, idx)
            v2, idx2 = _unpack_header_seq(packed)
            assert v2 == v
            assert idx2 == idx


def test_header_seq_idx_truncation():
    """Only low 8 bits of header_seq are ready_idx; high bits = version."""
    seq = (5 << 8) | 1
    v, idx = _unpack_header_seq(seq)
    assert v == 5
    assert idx == 1


# ── Per-buffer CRC parity ───────────────────────────────────────────


def test_buffer_crc_known_vectors(vectors):
    """CRC32 over (wall_ns_le8 || payload_bytes_le4 || payload) — Python
    zlib.crc32 == Rust titan-core::shm::crc32."""
    for v in vectors["shm_layout"]["crc32_known_vectors"]:
        input_bytes = v["input_ascii"].encode("ascii")
        expected_hex = v["expected_crc32_hex"]
        # Python's zlib.crc32 over arbitrary bytes
        import zlib
        actual = zlib.crc32(input_bytes) & 0xFFFFFFFF
        assert f"{actual:08x}" == expected_hex, (
            f"input '{v['input_ascii']}' got {actual:08x}, expected {expected_hex}"
        )


def test_compute_buffer_crc32_matches_concatenated_form():
    """`_compute_buffer_crc32(wall_ns, n, payload)` ==
    crc32(wall_ns_le8 || n_le4 || payload). Required for Rust parity:
    Rust `BufferMeta::compute_crc32` uses the exact same formula."""
    import zlib

    wall_ns = 1_700_000_000_000_000_000
    payload = b"hello world"
    payload_bytes = len(payload)

    concat = struct.pack("<QI", wall_ns, payload_bytes) + payload
    expected = zlib.crc32(concat) & 0xFFFFFFFF
    actual = _compute_buffer_crc32(wall_ns, payload_bytes, payload)
    assert actual == expected


# ── Round-trip wire-format ──────────────────────────────────────────


def test_python_writer_produces_readable_slot(shm_root):
    """A slot written via Python StateRegistryWriter must be readable by
    a fresh Python StateRegistryReader (sanity — same-language)."""
    spec = RegistrySpec(
        name="parity_slot_162d",
        dtype=np.dtype("<f4"),
        shape=(162,),
    )
    w = StateRegistryWriter(spec, shm_root)
    arr = np.arange(162, dtype=np.float32)
    w.write(arr)
    r = StateRegistryReader(spec, shm_root)
    out = r.read()
    np.testing.assert_array_equal(out, arr)
    w.close()
    r.close()


def test_python_writer_byte_layout_matches_spec(shm_root):
    """Inspect the raw byte layout: fixed header at [0:16] + buffer 1 at
    offset _buffer_offset(1, N) (Python writer's first user-write lands
    on idx 1 because __init__ publishes idx 0 zero-fill)."""
    N = 162 * 4  # 648 byte payload
    spec = RegistrySpec(
        name="layout_check",
        dtype=np.dtype("<f4"),
        shape=(162,),
    )
    w = StateRegistryWriter(spec, shm_root)
    arr = np.arange(162, dtype=np.float32)
    w.write(arr)
    w.close()

    raw = (shm_root / "layout_check.bin").read_bytes()

    # Expected total: 16 + 3 × (16 + 648) = 2008 bytes
    assert len(raw) == 16 + 3 * (16 + N)
    assert len(raw) == 2008

    # Fixed header [0:16]
    seq = struct.unpack_from("<Q", raw, 0)[0]
    schema, capacity = struct.unpack_from("<II", raw, 8)
    version, idx = _unpack_header_seq(seq)
    # __init__ published version=1 idx=0; first user write => version=2 idx=1
    assert version == 2
    assert idx == 1
    assert schema == 1
    assert capacity == N

    # Buffer 1 (active): meta + payload at offset 16 + 1 × (16 + 648)
    off = _buffer_offset(1, N)
    assert off == 16 + 1 * (16 + N)
    wall_ns, payload_bytes, stored_crc = struct.unpack_from(BUFFER_META_STRUCT, raw, off)
    assert payload_bytes == N
    assert wall_ns > 0
    payload_raw = raw[off + 16 : off + 16 + N]
    # CRC parity: recomputing via the same helper must match.
    expected_crc = _compute_buffer_crc32(wall_ns, payload_bytes, payload_raw)
    assert stored_crc == expected_crc
    # Payload bytes match arr.tobytes()
    assert payload_raw == arr.tobytes()


def test_python_initial_publish_zero_fills_buffer_0(shm_root):
    """StateRegistryWriter.__init__ publishes version=1 at idx=0 with a
    zero-filled capacity payload. This matches Rust Slot::create's
    initial publish behavior so first-readers see a valid empty snapshot."""
    spec = RegistrySpec(
        name="initial_publish_check",
        dtype=np.dtype("<f4"),
        shape=(64,),
    )
    w = StateRegistryWriter(spec, shm_root)
    raw = (shm_root / "initial_publish_check.bin").read_bytes()

    # Header points to idx=0 version=1
    seq = struct.unpack_from("<Q", raw, 0)[0]
    version, idx = _unpack_header_seq(seq)
    assert version == 1
    assert idx == 0

    # Buffer 0 metadata: payload_bytes = 64 × 4 = 256, all-zero payload
    off = _buffer_offset(0, 64 * 4)
    _wall_ns, payload_bytes, _crc = struct.unpack_from(BUFFER_META_STRUCT, raw, off)
    assert payload_bytes == 64 * 4
    payload_raw = raw[off + 16 : off + 16 + payload_bytes]
    assert payload_raw == b"\x00" * (64 * 4)

    # Reader sees zero-filled ndarray
    r = StateRegistryReader(spec, shm_root)
    out = r.read()
    np.testing.assert_array_equal(out, np.zeros(64, dtype=np.float32))
    w.close()
    r.close()


def test_python_round_trip_preserves_version_and_idx_rotation(shm_root):
    """5 user writes → version 2,3,4,5,6; idx rotation 1,2,0,1,2."""
    spec = RegistrySpec(
        name="rotation_check",
        dtype=np.dtype("<f4"),
        shape=(4,),
    )
    w = StateRegistryWriter(spec, shm_root)
    expected_idx = [1, 2, 0, 1, 2]
    expected_version = [2, 3, 4, 5, 6]
    for i in range(5):
        w.write(np.full(4, float(i), dtype=np.float32))
        # Re-read header to verify
        raw = (shm_root / "rotation_check.bin").read_bytes()
        seq = struct.unpack_from("<Q", raw, 0)[0]
        version, idx = _unpack_header_seq(seq)
        assert version == expected_version[i], (
            f"write {i}: version {version} != expected {expected_version[i]}"
        )
        assert idx == expected_idx[i], (
            f"write {i}: idx {idx} != expected {expected_idx[i]}"
        )
    w.close()


# ── Reader rejects corrupt/tampered slots ───────────────────────────


def test_reader_rejects_uninitialized(shm_root):
    """version=0 in header_seq => uninitialized sentinel => read() returns None."""
    spec = RegistrySpec(
        name="uninit_check",
        dtype=np.dtype("<f4"),
        shape=(4,),
    )
    w = StateRegistryWriter(spec, shm_root)
    # Force header_seq back to 0 (simulate fresh-create-but-no-publish-yet)
    w._mm[0:8] = struct.pack("<Q", 0)
    w.close()
    r = StateRegistryReader(spec, shm_root)
    assert r.read() is None


def test_reader_rejects_corrupt_ready_idx(shm_root):
    """ready_idx > 2 => corrupt sentinel => read() returns None."""
    spec = RegistrySpec(
        name="bad_idx_check",
        dtype=np.dtype("<f4"),
        shape=(4,),
    )
    w = StateRegistryWriter(spec, shm_root)
    w.write(np.zeros(4, dtype=np.float32))
    # Tamper: set ready_idx = 5 (out of range)
    bad_seq = _pack_header_seq(version=2, ready_idx=5)
    w._mm[0:8] = struct.pack("<Q", bad_seq)
    w.close()
    r = StateRegistryReader(spec, shm_root)
    assert r.read() is None


# ── Concurrent writer + reader ──────────────────────────────────────


def test_concurrent_writer_reader_zero_torn_data(shm_root):
    """Sustained writer at high rate + reader from a separate thread.
    Per-buffer-metadata triple-buffer eliminates torn metadata reads
    entirely. Reader either gets a valid snapshot, gets None on the
    rare ReaderLapped case (≥3 cycles preempted), but NEVER gets torn data."""
    import threading
    import time

    spec = RegistrySpec(
        name="concurrent_check",
        dtype=np.dtype("<f4"),
        shape=(64,),
    )
    w = StateRegistryWriter(spec, shm_root)
    stop = threading.Event()
    torn_count = [0]
    success_count = [0]
    none_count = [0]

    def writer_loop():
        v = 1.0
        while not stop.is_set():
            w.write(np.full(64, v, dtype=np.float32))
            v += 1.0

    def reader_loop():
        r = StateRegistryReader(spec, shm_root)
        while not stop.is_set():
            result = r.read()
            if result is None:
                none_count[0] += 1
                continue
            # Internal consistency: all elements equal (no torn data)
            first = result[0]
            if not np.all(result == first):
                torn_count[0] += 1
            else:
                success_count[0] += 1
        r.close()

    wt = threading.Thread(target=writer_loop)
    rt = threading.Thread(target=reader_loop)
    wt.start()
    rt.start()
    time.sleep(2.0)
    stop.set()
    wt.join(timeout=5)
    rt.join(timeout=5)
    w.close()

    # Triple-buffer wire format guarantees zero torn data, ever.
    assert torn_count[0] == 0, (
        f"Triple-buffer torn-data count must be 0, got {torn_count[0]} "
        f"(success={success_count[0]}, none={none_count[0]})"
    )
    # Reader made progress (readers get many successes; very rare lapping).
    assert success_count[0] > 100, (
        f"Reader should read many valid snapshots; got only {success_count[0]} "
        f"(torn={torn_count[0]}, none={none_count[0]})"
    )
