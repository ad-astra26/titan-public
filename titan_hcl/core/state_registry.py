"""
StateRegistry — L0 shared-memory state registries (§7.0 v1.0.0 triple-buffer).

Microkernel v2 Phase A §A.2 (2026-04-24); upgraded to triple-buffer wire
format per D-SPEC-35 (2026-05-05) — closes
``titan-docs/rFP_rust_seqlock_retry_exhaustion.md``.

──────────────────────────────────────────────────────────────────────
Wire format (SPEC §7.0 v1.0.0)

Slot file layout:

    [0  : 16]                    fixed header  (atomic publish word + constants)
    [16 : 16 + (16+N)]           buffer 0      (16-byte meta + N-byte payload)
    [16+(16+N) : 16+2(16+N)]     buffer 1
    [16+2(16+N) : 16+3(16+N)]    buffer 2

Total slot size = 16 + 3·(16+N) = 64 + 3N bytes.

Fixed header (16 bytes, ``<QII``):

    [0:8]    uint64 LE  header_seq        atomic publish word — bits[63:8]=version (monotonic), bits[7:0]=ready_idx ∈ {0,1,2}
    [8:12]   uint32 LE  schema_version    set at create, never updated
    [12:16]  uint32 LE  payload_capacity  per-buffer max payload bytes (= N); set at create

Per-buffer block (16 metadata bytes + N payload bytes, ``<QII`` + payload):

    [0:8]    uint64 LE  wall_ns          time.time_ns() at this buffer's publish
    [8:12]   uint32 LE  payload_bytes    actual payload size in this buffer
    [12:16]  uint32 LE  buffer_crc32     CRC32 over [0:12] || payload[0:payload_bytes]
    [16:16+N]            payload bytes

──────────────────────────────────────────────────────────────────────
Writer protocol (single writer per slot)

Writer holds local ``last_published_idx`` (init 2 ⇒ first publish lands on
idx 0) and ``version`` (init 0 ⇒ first publish bumps to 1):

    1. next_idx ← (last_published_idx + 1) mod 3
    2. off ← 16 + next_idx · (16+N)
    3. write meta prefix: mmap[off:off+8]   ← wall_ns
                          mmap[off+8:off+12] ← len(payload)
    4. memcpy mmap[off+16 : off+16+len(payload)] ← payload
    5. crc ← CRC32(mmap[off:off+12] || payload)
       write mmap[off+12:off+16] ← crc
    6. ATOMIC STORE mmap[0:8] ← (version+1) << 8 | next_idx
       (Python: aligned 8-byte struct.pack_into is single-instruction store
        on x86_64 + aarch64, matching Rust AtomicU64::store(Release))
    7. local: last_published_idx ← next_idx; version ← version + 1

Reader protocol — zero retries, zero spinning:

    1. s1 ← ATOMIC LOAD mmap[0:8] (Acquire on x86_64/aarch64 via aligned read)
    2. version1 ← s1 >> 8;  idx ← s1 & 0xFF
    3. if version1 == 0:                      return None  (uninitialized)
    4. if idx > 2:                            return None  (corrupt sentinel)
    5. off ← 16 + idx · (16+N)
    6. read wall_ns, payload_bytes, stored_crc from mmap[off:off+16]
    7. memcpy payload from mmap[off+16:off+16+payload_bytes]
    8. compute_crc ← CRC32(mmap[off:off+12] || payload)
    9. if compute_crc ≠ stored_crc:           return None  (extraordinarily rare)
    10. s2 ← ATOMIC LOAD mmap[0:8]
    11. if (s2 >> 8) - version1 ≤ 2:          return Ok(payload)
    12. else:                                 return None  (lapped — reader preempted ≥3 cycles)

Per-titan shm root: ``/dev/shm/titan_{titan_id}/``. Matches the live
EMOT-CGN convention. See PLAN §5.1 D1 for the full rationale
(T2+T3 shared-VPS isolation requirement).
"""
from __future__ import annotations

import logging
import mmap
import os
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from titan_hcl._phase_c_constants import (
    SHM_BUFFER_COUNT,
    SHM_BUFFER_META_BYTES,
    SHM_BUFFER_META_STRUCT,
    SHM_HEADER_BYTES,
    SHM_HEADER_STRUCT,
)

logger = logging.getLogger(__name__)

# ── Constants (re-exported from auto-generated SPEC-driven module) ────

HEADER_SIZE = SHM_HEADER_BYTES  # 16
HEADER_STRUCT = SHM_HEADER_STRUCT  # "<QII"
BUFFER_META_SIZE = SHM_BUFFER_META_BYTES  # 16
BUFFER_META_STRUCT = SHM_BUFFER_META_STRUCT  # "<QII"
BUFFER_COUNT = SHM_BUFFER_COUNT  # 3
SCHEMA_VERSION = 1  # default per-spec; overridden per-slot

# Bit positions for header_seq atomic word
_VERSION_SHIFT = 8
_IDX_MASK = 0xFF
_IDX_MAX = BUFFER_COUNT - 1  # 2


# ── Declarative spec ────────────────────────────────────────────────


@dataclass(frozen=True)
class RegistrySpec:
    """
    Immutable spec for a state registry.

    name:           basename under the shm root (e.g. "trinity_state" →
                    /dev/shm/titan_T1/trinity_state.bin)
    dtype:          numpy dtype of the payload array
    shape:          numpy shape of the payload array (= MAX shape if variable_size)
    feature_flag:   dotted path into config (e.g.
                    "microkernel.shm_trinity_enabled"); returns True if
                    writes should be published to shm. Empty = always enabled.
    schema_version: bump to invalidate old readers when layout changes.
    variable_size:  if True (S4 D11), payload is variable up to ``shape``
                    max bytes; actual size encoded in per-buffer metadata
                    on each write. Use ``write_variable(bytes)`` and
                    ``read_variable() -> bytes`` instead of write/read.
    """

    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    feature_flag: str = ""
    schema_version: int = SCHEMA_VERSION
    variable_size: bool = False

    @property
    def payload_bytes(self) -> int:
        """Max payload bytes (= actual when variable_size=False)."""
        return int(np.prod(self.shape)) * self.dtype.itemsize

    @property
    def total_bytes(self) -> int:
        """Total bytes preallocated: header + 3 × (buffer_meta + max payload)."""
        return HEADER_SIZE + BUFFER_COUNT * (BUFFER_META_SIZE + self.payload_bytes)


# ── Path resolution ─────────────────────────────────────────────────


def resolve_titan_id(explicit: str | None = None) -> str:
    """
    Resolve the Titan ID using the canonical precedence chain. Matches
    `titan_hcl.logic.emot_shm_protocol._resolve_titan_id` so all
    per-Titan shm paths share a single source of truth.
    """
    if explicit:
        return str(explicit)
    try:
        import json
        proj_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        identity_path = os.path.join(proj_root, "data", "titan_identity.json")
        if os.path.exists(identity_path):
            with open(identity_path) as f:
                tid = json.load(f).get("titan_id")
            if tid:
                return str(tid)
    except Exception:
        pass
    env_id = os.environ.get("TITAN_ID")
    if env_id:
        return str(env_id)
    return "T1"


def resolve_shm_root(titan_id: str | None = None) -> Path:
    """Return the shm root directory."""
    env = os.environ.get("TITAN_SHM_ROOT")
    if env:
        return Path(env)
    resolved = resolve_titan_id(titan_id)
    return Path(f"/dev/shm/titan_{resolved}")


def ensure_shm_root(titan_id: str | None = None) -> Path:
    """Create the shm root dir if missing; return the Path."""
    root = resolve_shm_root(titan_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _preallocate_file(path: Path, total_bytes: int) -> None:
    """
    Ensure the file exists at exactly total_bytes, zero-filled.
    Safe to call multiple times. Never truncates a correctly-sized file.
    """
    if path.exists() and path.stat().st_size == total_bytes:
        return
    with open(path, "wb") as f:
        f.write(b"\x00" * total_bytes)


# ── Header pack/unpack helpers ──────────────────────────────────────


def _pack_header_seq(version: int, ready_idx: int) -> int:
    """Compose header_seq u64 from version + ready_idx."""
    return ((version & ((1 << 56) - 1)) << _VERSION_SHIFT) | (ready_idx & _IDX_MASK)


def _unpack_header_seq(seq: int) -> tuple[int, int]:
    """Extract (version, ready_idx) from header_seq u64."""
    return (seq >> _VERSION_SHIFT, seq & _IDX_MASK)


def _buffer_offset(idx: int, payload_capacity: int) -> int:
    """Byte offset of buffer ``idx`` within the slot mmap."""
    return HEADER_SIZE + idx * (BUFFER_META_SIZE + payload_capacity)


def _compute_buffer_crc32(wall_ns: int, payload_bytes: int, payload: bytes) -> int:
    """CRC32 over the buffer's [0:12] meta prefix (wall_ns + payload_bytes) + payload."""
    prefix = struct.pack("<QI", wall_ns, payload_bytes)
    return zlib.crc32(prefix + payload) & 0xFFFFFFFF


# ── Writer ──────────────────────────────────────────────────────────


class StateRegistryWriter:
    """
    Single-writer triple-buffer registry writer.

    Concurrency: one writer per registry (Phase A design). Multi-writer
    requires external lock.

    Hot path: 1 memcpy into shared memory (per-buffer payload) + 16-byte
    metadata prefix + CRC32 + atomic 8-byte store. Zero syscalls.
    """

    def __init__(self, spec: RegistrySpec, shm_root: Path):
        self.spec = spec
        self.path = shm_root / f"{spec.name}.bin"
        # Writer state: version starts at 0, last_idx at IDX_MAX so
        # first publish lands on idx 0.
        self._version = 0
        self._last_idx = _IDX_MAX

        _preallocate_file(self.path, spec.total_bytes)
        self._fd = os.open(self.path, os.O_RDWR)
        self._mm = mmap.mmap(
            self._fd,
            spec.total_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        # Write fixed header (constants); header_seq starts 0 = uninitialized.
        struct.pack_into(
            HEADER_STRUCT, self._mm, 0,
            0,                       # header_seq
            spec.schema_version,
            spec.payload_bytes,      # payload_capacity (= N)
        )
        # Initial publish: zero-fill capacity payload at version=1 so the
        # slot is immediately readable at the declared byte size. Matches
        # the Rust Slot::create initialization + the legacy SeqLock
        # semantics where seq=0 (stable) was a readable zero-payload
        # snapshot of declared size.
        self._publish(b"\x00" * spec.payload_bytes)

    # -- internal --------------------------------------------------------

    def _publish(self, payload: bytes) -> int:
        """Triple-buffer publish — used by write/write_variable + create init."""
        next_idx = (self._last_idx + 1) % BUFFER_COUNT
        off = _buffer_offset(next_idx, self.spec.payload_bytes)

        wall_ns = time.time_ns()
        payload_len = len(payload)

        # Step 1: write metadata prefix (wall_ns + payload_bytes) at [off:off+12].
        struct.pack_into("<QI", self._mm, off, wall_ns, payload_len)
        # Step 2: write payload at [off+16 : off+16+payload_len].
        self._mm[off + BUFFER_META_SIZE : off + BUFFER_META_SIZE + payload_len] = payload
        # Step 3: compute CRC and write at [off+12:off+16].
        crc = _compute_buffer_crc32(wall_ns, payload_len, payload)
        struct.pack_into("<I", self._mm, off + 12, crc)
        # Step 4: atomic publish — single 8-byte aligned store at offset 0
        # (synchronizes-with reader Acquire-load on x86_64 via TSO; on
        # aarch64 Python's struct.pack_into compiles to a single STR which
        # is also atomic at aligned 8-byte offset).
        new_version = self._version + 1
        new_seq = _pack_header_seq(new_version, next_idx)
        struct.pack_into("<Q", self._mm, 0, new_seq)
        # Step 5: update local state.
        self._last_idx = next_idx
        self._version = new_version
        return new_version

    # -- hot path ----------------------------------------------------

    def write(self, array: np.ndarray) -> int:
        """
        Triple-buffer publish of a fixed-size ndarray. Returns the new version.
        Raises ValueError on shape/dtype mismatch.
        """
        if array.dtype != self.spec.dtype or array.shape != self.spec.shape:
            raise ValueError(
                f"Registry {self.spec.name}: expected "
                f"{self.spec.dtype}{self.spec.shape}, got "
                f"{array.dtype}{array.shape}"
            )
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        return self._publish(array.tobytes())

    # -- variable-size hot path (D11) --------------------------------

    def write_variable(self, payload: bytes) -> int:
        """
        Triple-buffer publish of a variable-size payload (S4 D11).
        Total payload must fit within ``spec.payload_bytes`` (the per-buffer
        capacity). Returns the new version.
        """
        if not self.spec.variable_size:
            raise ValueError(
                f"Registry {self.spec.name}: write_variable() requires "
                f"variable_size=True; use write(ndarray) for fixed specs"
            )
        if len(payload) > self.spec.payload_bytes:
            raise ValueError(
                f"Registry {self.spec.name}: payload {len(payload)}B "
                f"exceeds preallocated max {self.spec.payload_bytes}B"
            )
        return self._publish(payload)

    # -- introspection ----------------------------------------------

    @property
    def version(self) -> int:
        return self._version

    @property
    def seq(self) -> int:
        """Backward-compat alias (legacy callers expected SeqLock counter)."""
        return self._version

    def close(self) -> None:
        """Called at process shutdown only; not on hot path."""
        try:
            self._mm.close()
        finally:
            os.close(self._fd)


# ── Reader ──────────────────────────────────────────────────────────


class StateRegistryReader:
    """
    Persistent-mmap triple-buffer registry reader. Stateless aside from
    the open mmap handle. Safe to use from any layer/process.

    Hot path: zero syscalls, zero allocations except the defensive
    ``.copy()`` of the returned ndarray. Zero retries. Zero spinning.

    Fallback: ``read()`` returns ``None`` on any error — caller MUST fall
    back to legacy path. Reasons logged at INFO once per reader lifetime.
    """

    def __init__(self, spec: RegistrySpec, shm_root: Path):
        self.spec = spec
        self.path = shm_root / f"{spec.name}.bin"
        self._mm: mmap.mmap | None = None
        self._fd: int | None = None
        self._elem_count = int(np.prod(spec.shape))
        self._fallback_logged = False

    # -- internal --------------------------------------------------

    def _attach(self) -> bool:
        if self._mm is not None:
            return True
        try:
            if not self.path.exists():
                return False
            if self.path.stat().st_size != self.spec.total_bytes:
                return False
            self._fd = os.open(self.path, os.O_RDONLY)
            self._mm = mmap.mmap(
                self._fd,
                self.spec.total_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ,
            )
            return True
        except Exception as e:
            logger.debug(
                "[StateRegistryReader] attach failed for %s: %s",
                self.spec.name, e,
            )
            return False

    def _fallback(self, reason: str) -> None:
        if not self._fallback_logged:
            logger.info(
                "[StateRegistryReader] %s fallback (reason=%s) — legacy path",
                self.spec.name, reason,
            )
            self._fallback_logged = True
        return None

    # -- hot path ---------------------------------------------------

    def read(self) -> Optional[np.ndarray]:
        """
        Triple-buffer read. Returns a fresh ndarray (owned by caller) or
        ``None`` on uninitialized/corrupt/lapped. Zero retries.
        """
        if self._mm is None and not self._attach():
            return self._fallback("not_attached")

        mm = self._mm
        N = self.spec.payload_bytes  # per-buffer capacity

        # Step 1: atomic Acquire-load of header_seq.
        s1 = struct.unpack_from("<Q", mm, 0)[0]
        version1, idx = _unpack_header_seq(s1)

        if version1 == 0:
            return self._fallback("uninitialized")
        if idx > _IDX_MAX:
            return self._fallback(f"ready_idx_out_of_range({idx})")

        # Step 2: schema check (constant — never updated by writer).
        schema_ver = struct.unpack_from("<I", mm, 8)[0]
        if schema_ver != self.spec.schema_version:
            return self._fallback(f"schema_mismatch({schema_ver})")

        # Step 3: read per-buffer metadata.
        off = _buffer_offset(idx, N)
        wall_ns, payload_bytes, stored_crc = struct.unpack_from(BUFFER_META_STRUCT, mm, off)

        # Fixed-size sanity: actual must match expected.
        if payload_bytes != N:
            return self._fallback(f"size_mismatch({payload_bytes}!={N})")

        # Step 4: copy payload via numpy zero-copy view + defensive .copy()
        view = np.frombuffer(
            mm,
            dtype=self.spec.dtype,
            count=self._elem_count,
            offset=off + BUFFER_META_SIZE,
        ).reshape(self.spec.shape)
        snapshot = view.copy()

        # Step 5: verify CRC32 over (meta prefix + payload).
        computed = _compute_buffer_crc32(wall_ns, payload_bytes, snapshot.tobytes())
        if computed != stored_crc:
            return self._fallback("buffer_crc_mismatch")

        # Step 6: atomic Acquire-load of header_seq again — version-delta check.
        s2 = struct.unpack_from("<Q", mm, 0)[0]
        version2 = s2 >> _VERSION_SHIFT
        delta = (version2 - version1) & ((1 << 56) - 1)
        if delta > _IDX_MAX:
            return self._fallback(f"reader_lapped(delta={delta})")

        return snapshot

    def read_meta(self) -> Optional[dict]:
        """Cheap header-only read — returns publish state + age. Returns None if unattached or uninitialized."""
        if self._mm is None and not self._attach():
            return None
        mm = self._mm
        s = struct.unpack_from("<Q", mm, 0)[0]
        version, idx = _unpack_header_seq(s)
        if version == 0:
            return None
        if idx > _IDX_MAX:
            return None
        schema_version, payload_capacity = struct.unpack_from("<II", mm, 8)
        off = _buffer_offset(idx, payload_capacity)
        wall_ns, payload_bytes, crc = struct.unpack_from(BUFFER_META_STRUCT, mm, off)
        return {
            "version": version,
            "ready_idx": idx,
            "seq": version,  # backward-compat alias
            "schema_version": schema_version,
            "payload_capacity": payload_capacity,
            "payload_bytes": payload_bytes,
            "wall_ns": wall_ns,
            "buffer_crc32": crc,
            "age_seconds": (time.time_ns() - wall_ns) / 1e9,
            "write_in_progress": False,  # always False post-D-SPEC-35 — atomic publish
        }

    # -- variable-size hot path (D11) --------------------------------

    def read_variable(self) -> Optional[bytes]:
        """
        Triple-buffer variable-size read (S4 D11). Returns actual payload
        bytes (size = per-buffer payload_bytes from metadata; may be less
        than capacity). Zero retries.
        """
        if not self.spec.variable_size:
            return self._fallback("not_variable_spec")

        if self._mm is None and not self._attach_variable():
            return self._fallback("not_attached")

        mm = self._mm
        N = self.spec.payload_bytes  # per-buffer capacity

        s1 = struct.unpack_from("<Q", mm, 0)[0]
        version1, idx = _unpack_header_seq(s1)

        if version1 == 0:
            return self._fallback("uninitialized")
        if idx > _IDX_MAX:
            return self._fallback(f"ready_idx_out_of_range({idx})")

        schema_ver = struct.unpack_from("<I", mm, 8)[0]
        if schema_ver != self.spec.schema_version:
            return self._fallback(f"schema_mismatch({schema_ver})")

        off = _buffer_offset(idx, N)
        wall_ns, payload_bytes, stored_crc = struct.unpack_from(BUFFER_META_STRUCT, mm, off)

        if payload_bytes > N:
            return self._fallback(f"size_overflow({payload_bytes}>{N})")

        snapshot = bytes(mm[off + BUFFER_META_SIZE : off + BUFFER_META_SIZE + payload_bytes])

        computed = _compute_buffer_crc32(wall_ns, payload_bytes, snapshot)
        if computed != stored_crc:
            return self._fallback("buffer_crc_mismatch")

        s2 = struct.unpack_from("<Q", mm, 0)[0]
        version2 = s2 >> _VERSION_SHIFT
        delta = (version2 - version1) & ((1 << 56) - 1)
        if delta > _IDX_MAX:
            return self._fallback(f"reader_lapped(delta={delta})")

        return snapshot

    def _attach_variable(self) -> bool:
        """Variant of _attach for variable-size specs — file size still matches total_bytes (preallocated to capacity)."""
        if self._mm is not None:
            return True
        try:
            if not self.path.exists():
                return False
            if self.path.stat().st_size != self.spec.total_bytes:
                return False
            self._fd = os.open(self.path, os.O_RDONLY)
            self._mm = mmap.mmap(
                self._fd,
                self.spec.total_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ,
            )
            return True
        except Exception as e:
            logger.debug(
                "[StateRegistryReader] attach_variable failed for %s: %s",
                self.spec.name, e,
            )
            return False

    def close(self) -> None:
        try:
            if self._mm is not None:
                self._mm.close()
                self._mm = None
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
        except Exception:
            pass


# ── Bank — factory + feature-flag lookup ───────────────────────────


class RegistryBank:
    """
    Lazy-instantiated writers + readers keyed by RegistrySpec.name.
    """

    def __init__(self, titan_id: str | None = None, config: dict | None = None):
        self.titan_id = titan_id
        self.config = config or {}
        self.shm_root = ensure_shm_root(titan_id)
        self._writers: dict[str, StateRegistryWriter] = {}
        self._readers: dict[str, StateRegistryReader] = {}

    def writer(self, spec: RegistrySpec) -> StateRegistryWriter:
        if spec.name not in self._writers:
            self._writers[spec.name] = StateRegistryWriter(spec, self.shm_root)
        return self._writers[spec.name]

    def reader(self, spec: RegistrySpec) -> StateRegistryReader:
        if spec.name not in self._readers:
            self._readers[spec.name] = StateRegistryReader(spec, self.shm_root)
        return self._readers[spec.name]

    def is_enabled(self, spec: RegistrySpec) -> bool:
        """Check the registry's feature flag in config."""
        key = spec.feature_flag
        if not key:
            return True
        parts = key.split(".")
        cur: object = self.config
        for p in parts:
            if not isinstance(cur, dict) or p not in cur:
                return False
            cur = cur[p]
        return bool(cur)

    def close_all(self) -> None:
        for w in self._writers.values():
            w.close()
        for r in self._readers.values():
            r.close()
        self._writers.clear()
        self._readers.clear()


# ── Canonical registry declarations ────────────────────────────────

# 162D Trinity state — L1 owned, written by spirit_loop's snapshot-builder
# thread. Layout matches rFP #2 TITAN_SELF composition.
TRINITY_STATE = RegistrySpec(
    name="trinity_state",
    dtype=np.dtype("<f4"),
    shape=(162,),
    feature_flag="microkernel.shm_trinity_enabled",
)

# 6-neuromod state — Python L2 owned (neuromod_worker, NEW v2 layout 2026-05-15
# with §4.Q neuromod_worker.evaluate migration). Per-modulator 4 fields:
# (level, gain, phasic, tonic) so cognitive_worker can reconstruct the
# modulation dict via compute_modulation_from_state() without access to the
# live NeuromodulatorSystem instance. Canonical modulator order matches
# NEUROMOD_NAMES in titan_hcl/modules/neuromod_worker.py:
# (DA, 5HT, NE, ACh, Endorphin, GABA).
NEUROMOD_STATE = RegistrySpec(
    name="neuromod_state",
    dtype=np.dtype("<f4"),
    shape=(6, 4),
    feature_flag="microkernel.shm_neuromod_enabled",
    schema_version=2,
)

# neuromod_inputs — NEW SHM slot owned by cognitive_worker (G21 single writer).
# Carries the 11 emergent inputs aggregated from cognitive_worker's in-process
# state (coordinator, pi_monitor, sphere_clock, life_force_engine,
# neural_nervous_system, cached PREDICTION_STATS_UPDATED, cached
# EXPRESSION_COMPOSITES_UPDATED, cached KIN_SIGNATURE_UPDATED) + chi_health +
# topology_velocity + DNA weights from titan_params.toml [neuromodulator_dna].
# neuromod_worker reads this slot in its evaluate driver per KERNEL_EPOCH_TICK
# and calls compute_emergent_inputs() + evaluate() in its own process.
# Variable-size msgpack payload up to 4096 bytes (typical ≈400-700 bytes).
# Mirrors the §4.B nns_hormonal_state.bin cross-process bridge pattern.
# Added 2026-05-15 (§4.Q).
NEUROMOD_INPUTS = RegistrySpec(
    name="neuromod_inputs",
    dtype=np.dtype("u1"),
    shape=(4096,),
    feature_flag="microkernel.shm_neuromod_enabled",
    schema_version=1,
    variable_size=True,
)

# 11-hormone state — Python L2 owned (NEW in C-S5).
HORMONAL_STATE = RegistrySpec(
    name="hormonal_state",
    dtype=np.dtype("<f4"),
    shape=(11, 4),
    feature_flag="microkernel.shm_hormonal_enabled",
)

# NNS-hosted 11-hormone state — cognitive_worker owned (NEW v1.7.4 D-SPEC-53).
# `neural_nervous_system._hormonal` is an in-process HormonalSystem instance
# inside cognitive_worker that accumulates env_stimuli via
# `NeuralNervousSystem._evaluate -> _hormonal.accumulate_all(env_stimuli, dt)`
# every NS evaluate tick. This is a DIFFERENT instance from the hormonal_module-
# owned HormonalSystem (which writes `hormonal_state.bin` and accumulates only
# the cross-worker HORMONE_STIMULUS bus stream — currently only ns_worker's
# IMPULSE). Pre-§4.B the in-proc HormonalSystem was the authoritative source
# for ExpressionManager.evaluate_all (composites fire from these levels).
# After §4.B Track 3 extracted ExpressionManager into expression_worker,
# expression_worker needs cross-process read access to THIS instance — not
# hormonal_state.bin — to preserve pre-session firing dynamics. cognitive_worker
# writes this slot after each NS evaluate. G21 single-writer = cognitive_worker.
# Same layout as `hormonal_state.bin` (11 × 4 float32 LE; same per-hormone
# field order: level, threshold, refractory, peak_level) so the existing
# encode/decode helpers in hormonal_worker can be reused.
NNS_HORMONAL_STATE = RegistrySpec(
    name="nns_hormonal_state",
    dtype=np.dtype("<f4"),
    shape=(11, 4),
    feature_flag="microkernel.shm_nns_hormonal_enabled",
)

# Consciousness epoch counter — L1 owned, monotonic uint64.
EPOCH_COUNTER = RegistrySpec(
    name="epoch_counter",
    dtype=np.dtype("<u8"),
    shape=(1,),
    feature_flag="microkernel.shm_epoch_enabled",
)

# π-heartbeat state — L0 owned by titan-kernel-rs (12 bytes: f32 phase +
# uint64 pulse_count). Per SPEC §7.1 + titan-rust/crates/titan-state/src/spec.rs
# line 194-201. Phase A.3 reader gap (pi_heartbeat overlay TODO in
# SpiritAccessor.get_coordinator) closed 2026-05-18.
PI_HEARTBEAT_STATE = RegistrySpec(
    name="pi_heartbeat",
    dtype=np.dtype([("phase", "<f4"), ("pulse_count", "<u8")]),
    shape=(1,),
)

# 45D inner spirit tensor (SAT-15 + CHIT-15 + ANANDA-15) — L1 owned.
INNER_SPIRIT_45D = RegistrySpec(
    name="inner_spirit_45d",
    dtype=np.dtype("<f4"),
    shape=(45,),
    feature_flag="microkernel.shm_spirit_fast_enabled",
)

# Sphere clocks state — L1 owned.
SPHERE_CLOCKS_STATE = RegistrySpec(
    name="sphere_clocks",
    dtype=np.dtype("<f4"),
    shape=(6, 7),
    feature_flag="microkernel.shm_sphere_clocks_enabled",
)

# Chi circulation state — L1 owned.
CHI_STATE = RegistrySpec(
    name="chi_state",
    dtype=np.dtype("<f4"),
    shape=(6,),
    feature_flag="microkernel.shm_chi_enabled",
)

# TitanVM NS program registers — L1 owned.
TITANVM_REGISTERS = RegistrySpec(
    name="titanvm_registers",
    dtype=np.dtype("<f4"),
    shape=(11, 4),
    feature_flag="microkernel.shm_titanvm_enabled",
)

# NS-program urgencies INPUT — cross-process bridge from cognitive_worker
# (canonical NS evaluator via inner_coordinator._last_nervous_signals)
# to ns_worker (titanvm_registers.bin SHM writer + NS_URGENCIES_UPDATE
# emitter). G21 single-writer = cognitive_worker. SPEC §7.1 + D-SPEC-68
# v1.13.0 (2026-05-17) — closes the load-bearing wire-up gap from the
# ns_worker L2 carve-out (Phase C migration). Pattern mirrors
# NEUROMOD_INPUTS (§4.Q) + LIFE_FORCE_INPUTS (§4.G). Shape (11,) f32
# in NS_PROGRAMS order (REFLEX..VIGILANCE — same row order as
# titanvm_registers.bin).
NS_PROGRAM_URGENCIES_INPUT = RegistrySpec(
    name="ns_program_urgencies_input",
    dtype=np.dtype("<f4"),
    shape=(11,),
    feature_flag="microkernel.shm_titanvm_enabled",
    schema_version=1,
)

# Trajectory state — cognitive_worker writes curvature + density (the 2
# global meta-scalars from state_132d[130:132] per consciousness.py:46)
# each consciousness epoch. Replaces the TRAJECTORY_UPDATE bus event
# (PART B §8 D-SPEC-65) per G18 + D-SPEC-69 v1.14.0 (2026-05-17). Reader =
# emot_cgn_worker. G21 single-writer = cognitive_worker.
TRAJECTORY_STATE = RegistrySpec(
    name="trajectory_state",
    dtype=np.dtype("<f4"),
    shape=(2,),
    feature_flag="microkernel.shm_titanvm_enabled",
    schema_version=1,
)

# CGN beta state — cgn_worker writes per-consumer reward_ema for the
# 8 consumers in CGN_CONSUMERS order (language, social, knowledge,
# reasoning, coding, self_model, reasoning_strategy, meta — from
# emot_bundle_protocol.py:172-175). Replaces CGN_BETA_SNAPSHOT bus event
# per G18 + D-SPEC-69 v1.14.0. Reader = emot_cgn_worker.
# G21 single-writer = cgn_worker.
CGN_BETA_STATE = RegistrySpec(
    name="cgn_beta_state",
    dtype=np.dtype("<f4"),
    shape=(8,),
    feature_flag="microkernel.shm_cgn_format_alignment_enabled",
    schema_version=1,
)

# Inner self-insight — self_reflection_worker writes the latest output of
# SelfReasoningEngine.introspect() per META_INTROSPECT_REQUEST handler
# completion. cognitive_worker reads pre-warmed cache in _prim_introspect()
# per G20. Variable msgpack payload ≤1024 bytes (typical 250-400 bytes).
# Cross-process bridge replacing the legacy in-process
# meta_reasoning.set_self_reasoning(engine) attachment that broke fleet-wide
# during Phase C cutover (D8-3 spirit_worker retirement). G21 single-writer =
# self_reflection_worker. SPEC §7.1 + D-SPEC-70 v1.15.0 (2026-05-17) —
# closes F-8 from rFP_social_x_improvements.md §B.3. Pattern mirrors
# NEUROMOD_INPUTS (§4.Q) — Python L2 cross-worker bridge.
INNER_SELF_INSIGHT = RegistrySpec(
    name="inner_self_insight",
    dtype=np.dtype("u1"),
    shape=(1024,),
    schema_version=1,
    variable_size=True,
)

# Self-reflection diagnostics surface (2026-06-22) — completes the Track-2
# read-side migration that left /v6/cognition/self-reflection reading an empty
# coordinator under l0_rust_enabled=true (the SelfReasoningEngine itself was
# always alive). G21 single-writer = self_reflection_worker (writes
# SelfReasoningEngine.get_stats() each 2.5s publish cycle). Reader = api
# StateAccessor.spirit._shm → dashboard get_v4_self_reflection. Variable
# msgpack ≤4096B (typical ≈400-600B). Mirrors meta_teacher_state.bin precedent.
SELF_REFLECTION_STATE = RegistrySpec(
    name="self_reflection_state",
    dtype=np.dtype("u1"),
    shape=(4096,),
    schema_version=1,
    variable_size=True,
)

# CodingExplorer diagnostics surface (2026-06-22) — same Track-2 read-side
# completion for /v6/cognition/coding-explorer. G21 single-writer =
# self_reflection_worker (CodingExplorer.get_stats() each 5s publish cycle).
CODING_EXPLORER_STATE = RegistrySpec(
    name="coding_explorer_state",
    dtype=np.dtype("u1"),
    shape=(4096,),
    schema_version=1,
    variable_size=True,
)

# Identity — L0 owned.
IDENTITY = RegistrySpec(
    name="identity",
    dtype=np.dtype("u1"),
    shape=(96,),
    feature_flag="microkernel.shm_identity_enabled",
)

# CGN live weights — L2 owned (variable size).
CGN_LIVE_WEIGHTS = RegistrySpec(
    name="cgn_live_weights",
    dtype=np.dtype("u1"),
    shape=(262144,),
    feature_flag="microkernel.shm_cgn_format_alignment_enabled",
    variable_size=True,
)

# 5D inner body tensor — L1 owned.
INNER_BODY_5D = RegistrySpec(
    name="inner_body_5d",
    dtype=np.dtype("<f4"),
    shape=(5,),
    feature_flag="microkernel.shm_body_fast_enabled",
)

# 15D inner mind tensor — L1 owned.
INNER_MIND_15D = RegistrySpec(
    name="inner_mind_15d",
    dtype=np.dtype("<f4"),
    shape=(15,),
    feature_flag="microkernel.shm_mind_fast_enabled",
)


# ── Phase A.S8 outer trinity slots (2026-04-30) ──────────────────────

OUTER_BODY_5D = RegistrySpec(
    name="outer_body_5d",
    dtype=np.dtype("<f4"),
    shape=(5,),
)

OUTER_MIND_15D = RegistrySpec(
    name="outer_mind_15d",
    dtype=np.dtype("<f4"),
    shape=(15,),
)

OUTER_SPIRIT_45D = RegistrySpec(
    name="outer_spirit_45d",
    dtype=np.dtype("<f4"),
    shape=(45,),
)

# 30D topology slot — kernel-side writer.
TOPOLOGY_30D = RegistrySpec(
    name="topology_30d",
    dtype=np.dtype("<f4"),
    shape=(30,),
)
