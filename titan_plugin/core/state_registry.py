"""
StateRegistry — L0 shared-memory state registries.

Microkernel v2 Phase A §A.2 (2026-04-24). Persistent mmap + SeqLock —
zero-copy, no-syscall hot path. Matches Maker's 2026-04-17 directive
("fast and probably memory-based; file reads would be too slow") and
the rFP: "zero-copy, non-blocking, no bus message needed".

Extends the proven pattern from ``titan_plugin.logic.cgn_shm_protocol``
(live since 2026-04 on all 3 Titans for emot_grounding + emot_state +
emot_latent_bundle + cgn_live_weights).

Protocol reference: ``titan-docs/PLAN_microkernel_phase_a.md`` §5.

──────────────────────────────────────────────────────────────────────
Header format (24 bytes, little-endian):
    [0:4]    uint32  seq              SeqLock counter (odd = write in progress)
    [4:8]    uint32  schema_version   Registry format revision
    [8:16]   uint64  wall_ns          time.time_ns() of last write
    [16:20]  uint32  payload_bytes    Size of payload that follows
    [20:24]  uint32  header_crc       CRC32 of bytes [0:20]

Payload: contiguous bytes of the registered ndarray (row-major).

Writer protocol (SeqLock):
    1. seq := seq + 1  → odd (write in progress)
    2. write payload in-place
    3. seq := seq + 1  → even (write complete)

Reader protocol (SeqLock):
    loop:
        seq1 := read seq
        if seq1 odd → retry (writer active)
        copy payload
        seq2 := read seq
        if seq1 == seq2 → return copy
        else → retry

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

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────

HEADER_SIZE = 24
# [seq:4] [schema:4] [wall_ns:8] [payload_bytes:4] [header_crc:4]
HEADER_STRUCT = "<IIQII"
SCHEMA_VERSION = 1
MAX_READ_RETRIES = 3


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
                    max bytes; actual size encoded in header's payload_bytes
                    field on each write. Use ``write_variable(bytes)`` and
                    ``read_variable() -> bytes`` instead of write/read.
                    Enables CGN + future variable-payload consumers.
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
        """Total bytes preallocated (header + max payload)."""
        return HEADER_SIZE + self.payload_bytes


# ── Path resolution ─────────────────────────────────────────────────


def resolve_titan_id(explicit: str | None = None) -> str:
    """
    Resolve the Titan ID using the canonical precedence chain. Matches
    `titan_plugin.logic.emot_shm_protocol._resolve_titan_id` so all
    per-Titan shm paths share a single source of truth.

    Precedence:
      1. explicit `titan_id` arg (kwarg from caller)
      2. data/titan_identity.json (canonical — same source as emot,
         language_config, meta_outer; each Titan's repo has its own file)
      3. TITAN_ID env var (rare fallback)
      4. "T1" hardcoded fallback

    Critical for T2+T3 which share /dev/shm on the same VPS — without
    the canonical resolver, both would default to "T1" and stomp each
    other's trinity_state.bin.
    """
    if explicit:
        return str(explicit)
    try:
        import json
        # Project root: titan_plugin/core/<file> → ../../
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
    """
    Return the shm root directory. Honors TITAN_SHM_ROOT env override
    (used by tests). Per-titan convention: /dev/shm/titan_{titan_id}/.

    When titan_id is None, uses resolve_titan_id() to follow the
    canonical precedence chain (data/titan_identity.json → TITAN_ID
    env → "T1"). Explicit titan_id always wins.
    """
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
    # Create or resize.
    with open(path, "wb") as f:
        f.write(b"\x00" * total_bytes)


def _pack_header(seq: int, schema: int, wall_ns: int, payload_bytes: int) -> bytes:
    """Pack the 20-byte header prefix and append its CRC32 → 24 bytes."""
    prefix = struct.pack("<IIQI", seq, schema, wall_ns, payload_bytes)
    crc = zlib.crc32(prefix)
    return prefix + struct.pack("<I", crc)


# ── Writer ──────────────────────────────────────────────────────────


class StateRegistryWriter:
    """
    Single-writer persistent-mmap registry writer.

    Concurrency: one writer per registry (Phase A design). If multiple
    writers were ever needed, an external lock would be required.

    Hot path: 1 memcpy into shared memory + 2 struct.pack_into for
    SeqLock bookends. Zero syscalls.
    """

    def __init__(self, spec: RegistrySpec, shm_root: Path):
        self.spec = spec
        self.path = shm_root / f"{spec.name}.bin"
        self._seq = 0

        _preallocate_file(self.path, spec.total_bytes)
        self._fd = os.open(self.path, os.O_RDWR)
        self._mm = mmap.mmap(
            self._fd,
            spec.total_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        # Initialize header with seq=0 (even, no write in progress).
        header = _pack_header(
            seq=0,
            schema=spec.schema_version,
            wall_ns=time.time_ns(),
            payload_bytes=spec.payload_bytes,
        )
        self._mm[:HEADER_SIZE] = header

    # -- hot path ----------------------------------------------------

    def write(self, array: np.ndarray) -> int:
        """
        SeqLock-protected in-place write.

        Returns the new even seq number. Raises ValueError on
        shape/dtype mismatch.
        """
        if array.dtype != self.spec.dtype or array.shape != self.spec.shape:
            raise ValueError(
                f"Registry {self.spec.name}: expected "
                f"{self.spec.dtype}{self.spec.shape}, got "
                f"{array.dtype}{array.shape}"
            )
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)

        # Stage 1: bump to odd (writer active).
        self._seq += 1
        wall_ns = time.time_ns()
        self._mm[:HEADER_SIZE] = _pack_header(
            seq=self._seq,
            schema=self.spec.schema_version,
            wall_ns=wall_ns,
            payload_bytes=self.spec.payload_bytes,
        )

        # Stage 2: write payload in-place via numpy zero-copy view.
        view = np.frombuffer(
            self._mm,
            dtype=self.spec.dtype,
            count=int(np.prod(self.spec.shape)),
            offset=HEADER_SIZE,
        ).reshape(self.spec.shape)
        view[...] = array

        # Stage 3: bump to even (write complete).
        self._seq += 1
        self._mm[:HEADER_SIZE] = _pack_header(
            seq=self._seq,
            schema=self.spec.schema_version,
            wall_ns=wall_ns,
            payload_bytes=self.spec.payload_bytes,
        )
        return self._seq

    # -- variable-size hot path (D11) --------------------------------

    def write_variable(self, payload: bytes) -> int:
        """
        SeqLock-protected variable-size in-place write (S4 D11).

        Writes ``payload`` directly as raw bytes. Header's ``payload_bytes``
        field records the actual size; readers use ``read_variable()`` to
        get only the actual slice. Total payload must fit within the
        spec's preallocated max (``spec.payload_bytes``).

        Returns the new even seq number. Raises ValueError if payload
        exceeds max or spec is not declared as variable_size.
        """
        if not self.spec.variable_size:
            raise ValueError(
                f"Registry {self.spec.name}: write_variable() requires "
                f"variable_size=True; use write(ndarray) for fixed specs"
            )
        actual_bytes = len(payload)
        if actual_bytes > self.spec.payload_bytes:
            raise ValueError(
                f"Registry {self.spec.name}: payload {actual_bytes}B "
                f"exceeds preallocated max {self.spec.payload_bytes}B"
            )

        # Stage 1: bump to odd (writer active).
        self._seq += 1
        wall_ns = time.time_ns()
        self._mm[:HEADER_SIZE] = _pack_header(
            seq=self._seq,
            schema=self.spec.schema_version,
            wall_ns=wall_ns,
            payload_bytes=actual_bytes,  # ACTUAL, not max
        )

        # Stage 2: write payload bytes directly.
        self._mm[HEADER_SIZE:HEADER_SIZE + actual_bytes] = payload

        # Stage 3: bump to even (write complete).
        self._seq += 1
        self._mm[:HEADER_SIZE] = _pack_header(
            seq=self._seq,
            schema=self.spec.schema_version,
            wall_ns=wall_ns,
            payload_bytes=actual_bytes,
        )
        return self._seq

    # -- introspection ----------------------------------------------

    @property
    def seq(self) -> int:
        return self._seq

    def close(self) -> None:
        """Called at process shutdown only; not on hot path."""
        try:
            self._mm.close()
        finally:
            os.close(self._fd)


# ── Reader ──────────────────────────────────────────────────────────


class StateRegistryReader:
    """
    Persistent-mmap registry reader. Stateless aside from the open mmap
    handle. Safe to use from any layer/process.

    Hot path: zero syscalls, zero allocations except the defensive
    ``.copy()`` of the returned ndarray.

    Fallback behavior: ``read()`` returns ``None`` on any error — caller
    MUST fall back to the legacy path. Fallback-reason is logged at
    INFO level ONCE per reader lifetime to avoid log spam.
    """

    def __init__(self, spec: RegistrySpec, shm_root: Path):
        self.spec = spec
        self.path = shm_root / f"{spec.name}.bin"
        self._mm: mmap.mmap | None = None
        self._fd: int | None = None
        self._elem_count = int(np.prod(spec.shape))
        self._fallback_logged = False
        # Defer attach — the file may not exist yet at construction time
        # (writer boots later). attach() is called lazily on first read.

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
        except Exception as e:  # pragma: no cover — defensive
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
        SeqLock-protected zero-copy read + defensive copy.

        Returns a fresh ndarray (owned by caller) or ``None`` on any
        error/mismatch/torn-read. Caller must handle None by falling
        back to the legacy state path.
        """
        if self._mm is None and not self._attach():
            return self._fallback("not_attached")

        mm = self._mm  # local alias

        for _ in range(MAX_READ_RETRIES):
            seq1 = struct.unpack_from("<I", mm, 0)[0]
            if seq1 & 1:
                continue  # writer active — retry

            # Cheap header validation before paying for payload copy.
            schema_ver = struct.unpack_from("<I", mm, 4)[0]
            if schema_ver != self.spec.schema_version:
                return self._fallback(f"schema_mismatch({schema_ver})")
            payload_bytes = struct.unpack_from("<I", mm, 16)[0]
            if payload_bytes != self.spec.payload_bytes:
                return self._fallback(f"size_mismatch({payload_bytes})")
            crc = struct.unpack_from("<I", mm, 20)[0]
            expected = zlib.crc32(bytes(mm[0:20]))
            if crc != expected:
                return self._fallback("header_crc_mismatch")

            # Zero-copy view, then defensive copy.
            view = np.frombuffer(
                mm,
                dtype=self.spec.dtype,
                count=self._elem_count,
                offset=HEADER_SIZE,
            ).reshape(self.spec.shape)
            snapshot = view.copy()

            seq2 = struct.unpack_from("<I", mm, 0)[0]
            if seq1 == seq2:
                return snapshot
            # else: torn read — retry
        return self._fallback("torn_read_after_retries")

    def read_meta(self) -> Optional[dict]:
        """Cheap header-only read. No payload copy. Returns None if unattached."""
        if self._mm is None and not self._attach():
            return None
        seq, schema, wall_ns, pbytes, crc = struct.unpack_from(
            HEADER_STRUCT, self._mm, 0,
        )
        return {
            "seq": seq,
            "schema_version": schema,
            "wall_ns": wall_ns,
            "payload_bytes": pbytes,
            "header_crc": crc,
            "age_seconds": (time.time_ns() - wall_ns) / 1e9,
            "write_in_progress": bool(seq & 1),
        }

    # -- variable-size hot path (D11) --------------------------------

    def read_variable(self) -> Optional[bytes]:
        """
        SeqLock-protected variable-size read (S4 D11).

        Returns the actual payload as raw bytes (size = header's
        payload_bytes field, may be less than the preallocated max).
        Returns None on torn-read/error/spec-mismatch (caller falls back
        to legacy path).

        Requires the spec to have variable_size=True.
        """
        if not self.spec.variable_size:
            return self._fallback("not_variable_spec")

        if self._mm is None and not self._attach_variable():
            return self._fallback("not_attached")

        mm = self._mm  # local alias

        for _ in range(MAX_READ_RETRIES):
            seq1 = struct.unpack_from("<I", mm, 0)[0]
            if seq1 & 1:
                continue  # writer active — retry

            schema_ver = struct.unpack_from("<I", mm, 4)[0]
            if schema_ver != self.spec.schema_version:
                return self._fallback(f"schema_mismatch({schema_ver})")
            payload_bytes = struct.unpack_from("<I", mm, 16)[0]
            if payload_bytes > self.spec.payload_bytes:
                return self._fallback(
                    f"size_overflow({payload_bytes}>{self.spec.payload_bytes})")
            crc = struct.unpack_from("<I", mm, 20)[0]
            expected = zlib.crc32(bytes(mm[0:20]))
            if crc != expected:
                return self._fallback("header_crc_mismatch")

            # Defensive copy of actual payload slice.
            snapshot = bytes(mm[HEADER_SIZE:HEADER_SIZE + payload_bytes])

            seq2 = struct.unpack_from("<I", mm, 0)[0]
            if seq1 == seq2:
                return snapshot
            # else: torn read — retry
        return self._fallback("torn_read_after_retries")

    def _attach_variable(self) -> bool:
        """Variant of _attach that allows file size == total_bytes (max)
        regardless of header's actual payload_bytes (which is variable)."""
        if self._mm is not None:
            return True
        try:
            if not self.path.exists():
                return False
            # For variable-size specs the file is preallocated at total_bytes;
            # we don't validate exact match because header carries actual size.
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
        except Exception as e:  # pragma: no cover — defensive
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
        except Exception:  # pragma: no cover
            pass


# ── Bank — factory + feature-flag lookup ───────────────────────────


class RegistryBank:
    """
    Lazy-instantiated writers + readers keyed by RegistrySpec.name.

    Same bank can be used by both writer-side and reader-side code
    within the same process; writers and readers are separate objects
    but share the spec + shm_root. A writer-side bank instance creates
    writers; a reader-side bank instance creates readers. In practice
    the spirit-worker process has a writer bank and the dashboard
    process has a reader bank, each resolving to the same shm files.
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


# ── Canonical registry declarations (S2 — three Phase-A initial) ───

# 162D Trinity state — L1 owned, written by spirit_loop's snapshot-builder
# thread. Layout matches rFP #2 TITAN_SELF composition already used by
# V5 FilterDown and the "I AM" kin-protocol:
#   [0:130]   full_130dt         (rFP #1 Canonical Atomic Signals)
#   [130:160] full_30d_topology  (rFP #1)
#   [160:162] journey (curvature, density)  per [titan_self] weight_journey=0.5
# Natural cadence: body/mind STATE bus events ≈ 1.15s (Schumann/9).
TRINITY_STATE = RegistrySpec(
    name="trinity_state",
    dtype=np.dtype("<f4"),
    shape=(162,),
    feature_flag="microkernel.shm_trinity_enabled",
)

# 6-neuromod state — L1 owned, written by spirit_worker post-epoch
# neuromod update. Field order: DA, 5HT, NE, ACh, Endorphin, GABA
# (matches neuromodulator.py field order).
# Natural cadence: consciousness epoch (dynamic 10-31s, Schumann-derived).
NEUROMOD_STATE = RegistrySpec(
    name="neuromod_state",
    dtype=np.dtype("<f4"),
    shape=(6,),
    feature_flag="microkernel.shm_neuromod_enabled",
)

# Consciousness epoch counter — L1 owned, monotonic uint64.
# Written by spirit_worker per epoch; readers use this as a cheap
# "has anything changed" signal.
EPOCH_COUNTER = RegistrySpec(
    name="epoch_counter",
    dtype=np.dtype("<u8"),
    shape=(1,),
    feature_flag="microkernel.shm_epoch_enabled",
)

# 45D inner spirit tensor (SAT-15 + CHIT-15 + ANANDA-15) — L1 owned,
# written by spirit_worker at Schumann × 9 = 70.47 Hz (14.2 ms period).
# Validates §L1 Trinity Daemon Internal Design pattern (rFP
# rFP_microkernel_v2_shadow_core.md §L1 + §A.7): spirit tick has zero
# I/O on hot path (benchmarked 46μs/call 2026-04-24 — see
# SESSION_20260424_microkernel_phase_a_s1_s2_shipped.md post-soak
# follow-up §3), so Schumann-rate shm write is trivial at ~0.3% of
# 1 core.
#
# Content-hash gated by the same pattern as Trinity/Neuromod/Epoch:
# writer only bumps seq counter when the 45D vector actually changes.
# Readers check age_seconds for freshness; writer never blocks.
#
# Layout (matches logic.spirit_tensor.collect_spirit_45d output):
#   [0:15]   SAT     (existence / presence — 15D)
#   [15:30]  CHIT    (consciousness / knowing — 15D)
#   [30:45]  ANANDA  (bliss / resonance — 15D)
#
# Natural cadence: spirit_worker tick @ 70.47 Hz (Schumann × 9).
INNER_SPIRIT_45D = RegistrySpec(
    name="inner_spirit_45d",
    dtype=np.dtype("<f4"),
    shape=(45,),
    feature_flag="microkernel.shm_spirit_fast_enabled",
)


# ── S4 registries (Phase A §A.2 part 2 — 2026-04-25) ───────────────

# Sphere clocks state — L1 owned, written by spirit_worker subprocess
# (SphereClockEngine lives there at spirit_worker.py:11297). 6 clocks
# (inner+outer Body/Mind/Spirit) × 7 fields per clock.
#
# Row order (canonical, matches SphereClockEngine._clock_names):
#   [0] inner_body, [1] inner_mind, [2] inner_spirit,
#   [3] outer_body, [4] outer_mind, [5] outer_spirit
#
# Per-clock field order (4 floats per clock):
#   [0] radius                — current contraction radius
#   [1] scalar_position       — phase position scalar
#   [2] phase                 — clock phase (radians)
#   [3] contraction_velocity  — current contraction speed
#   [4] pulse_count           — cumulative pulses (cast float32)
#   [5] consecutive_balanced  — consecutive balanced ticks (cast float32)
#   [6] last_pulse_age_s      — derived: time.time() - last_pulse_ts
#
# Natural cadence: tick rate of SphereClockEngine (Schumann-derived,
# ~1.15s body / 3.45s spirit). Content-hash gated.
SPHERE_CLOCKS_STATE = RegistrySpec(
    name="sphere_clocks",
    dtype=np.dtype("<f4"),
    shape=(6, 7),
    feature_flag="microkernel.shm_sphere_clocks_enabled",
)

# Chi circulation state — L1 owned, written by spirit_worker subprocess
# (_cached_chi_state dict lives there at spirit_worker.py:1668, updated
# at line 5362 after life_force_engine tick).
#
# Field order: [total, spirit, mind, body, coherence, urgency]
#
# Natural cadence: every spirit tick (~Schumann/9 = 1.15s).
# Content-hash gated.
CHI_STATE = RegistrySpec(
    name="chi_state",
    dtype=np.dtype("<f4"),
    shape=(6,),
    feature_flag="microkernel.shm_chi_enabled",
)

# TitanVM NS program registers — L1 owned, written by spirit_worker
# subprocess (NervousSystem.programs lives there via
# nervous_system.py:891). 11 NS programs × 4 fields per program.
#
# Spec verified by `arch_map titanvm-schema` (commit 19023d7d).
#
# Row order (canonical, matches emot_bundle_protocol.NS_PROGRAMS):
#   [0] REFLEX, [1] FOCUS, [2] INTUITION, [3] IMPULSE, [4] METABOLISM,
#   [5] CREATIVITY, [6] CURIOSITY, [7] EMPATHY, [8] REFLECTION,
#   [9] INSPIRATION, [10] VIGILANCE
#
# Per-program field order (4 floats per program):
#   [0] urgency        — current urgency from neural_nervous_system
#                        ._all_urgencies dict (per-tick computed)
#   [1] fire_count     — cumulative fires (cast float32)
#   [2] total_updates  — training iterations (cast float32)
#   [3] last_loss      — most recent training loss
#
# Natural cadence: NS tick rate (~Schumann/9). Content-hash gated.
TITANVM_REGISTERS = RegistrySpec(
    name="titanvm_registers",
    dtype=np.dtype("<f4"),
    shape=(11, 4),
    feature_flag="microkernel.shm_titanvm_enabled",
)

# Identity — L0 owned, written once by TitanKernel.boot() after Soul
# is initialized. titan_id + maker_pubkey are stable across reboots;
# kernel_instance_nonce is random per boot — enables worker reattach
# detection for Phase B shadow-core swap and external kernel-instance
# monitoring (PLAN §2.4 D11+rationale).
#
# Byte layout (96 bytes, uint8 — mixed content):
#   [0:32]   titan_id              (UTF-8, NUL-padded)
#   [32:64]  maker_pubkey          (raw 32-byte Ed25519, zero if absent)
#   [64:96]  kernel_instance_nonce (secrets.token_bytes(32) per boot)
IDENTITY = RegistrySpec(
    name="identity",
    dtype=np.dtype("u1"),
    shape=(96,),
    feature_flag="microkernel.shm_identity_enabled",
)

# CGN live weights — L2 owned (cgn_worker remains sole writer per
# project_cgn_as_higher_state_registry.md invariant 2026-04-21).
# First variable-size StateRegistry consumer.
#
# Per the 2026-04-17 Maker invariant ("Stateregistries(trinity + CGN)"
# at L0) and the 2026-04-21 codification: CGN is the higher-cognitive-
# level state registry and MUST share the same shm+version protocol as
# Trinity. S4 closes the keystone gap — same 24B header, same
# per-titan path convention, same SeqLock semantics.
#
# Payload wraps the existing CGN domain serialization unchanged
# (V-weights + consumer state_dicts + extras). cgn_shm_protocol.py
# becomes a thin format-aware wrapper that delegates to
# StateRegistry when this flag is on.
#
# 256 KB max payload (matches legacy MAX_SHM_SIZE ceiling). Typical
# size 40-50 KB.
CGN_LIVE_WEIGHTS = RegistrySpec(
    name="cgn_live_weights",
    dtype=np.dtype("u1"),
    shape=(262144,),
    feature_flag="microkernel.shm_cgn_format_alignment_enabled",
    variable_size=True,
)


# ── S7 registries (Phase A §A.7 / §L1 — 2026-04-26) ────────────────

# 5D inner body tensor — L1 owned, written by body_worker at Schumann
# fundamental rate (7.83 Hz, 127.7 ms period). Validates §L1 Trinity
# Daemon Internal Design end-to-end for body (rFP §L1 + §A.7):
# body_worker tick reads from per-sense cache substrate (populated by
# background refresh threads at native cadences) — zero I/O on hot path.
# Pre-S7 body_worker did all 5 sensor calls inline at 524ms/call worst
# case (psutil.cpu_percent interval=0.5 + 4 socket connects @ 2s
# timeout); S7 brings that to ~100μs.
#
# Content-hash gated by RegistryWriter (same pattern as
# Trinity / Neuromod / Epoch / Inner-Spirit). Readers check
# age_seconds for freshness; writer never blocks.
#
# Layout (matches body_worker.BODY_STATE bus payload `values` exactly,
# post-FILTER_DOWN-multiplier + post-FOCUS-nudge):
#   [0] interoception   (SOL balance + anchor freshness)
#   [1] proprioception  (body topology self-sensing)
#   [2] somatosensation (CPU/RAM/swap/disk)
#   [3] entropy         (log errors + multi-endpoint network)
#   [4] thermal         (CPU load × topology activity × circadian)
#
# Natural cadence: body_worker shm-writer thread at 7.83 Hz.
INNER_BODY_5D = RegistrySpec(
    name="inner_body_5d",
    dtype=np.dtype("<f4"),
    shape=(5,),
    feature_flag="microkernel.shm_body_fast_enabled",
)

# 15D inner mind tensor — L1 owned, written by mind_worker at Schumann
# × 3 = 23.49 Hz (42.6 ms period). Validates §L1 Trinity Daemon
# Internal Design end-to-end for mind (rFP §L1 + §A.7).
#
# Beautiful Schumann symmetry across Trinity tensors:
#   - Body:   5D × 7.83 Hz  (1× Schumann)
#   - Mind:  15D × 23.49 Hz (3× Schumann)
#   - Spirit: 45D × 70.47 Hz (9× Schumann)
# Each tensor's dimensionality matches its frequency multiplier —
# higher harmonics = richer state space.
#
# mind_worker tick reads sub_a sensor values from cache substrate
# (populated by background refresh threads at native cadences) +
# sub_b media-digest values from media_state (push-driven by
# SENSE_VISUAL/SENSE_AUDIO bus messages) and computes the 15D
# extended tensor (Thinking 5D + Feeling 5D + Willing 5D) via
# `titan_plugin.logic.mind_tensor.collect_mind_15d`. Zero I/O on
# hot path. Pre-S7 baseline was 73ms/call (sqlite3 + 2× http);
# S7 brings tick to ~100μs.
#
# Content-hash gated by RegistryWriter.
#
# Layout (matches mind_worker.MIND_STATE bus payload `values_15d`):
#   [0:5]   Thinking 5D — vision/hearing/taste/smell/touch (cognition layer)
#   [5:10]  Feeling 5D  — emotional valence breakdown
#   [10:15] Willing 5D  — volitional/intentional layer
#
# Natural cadence: mind_worker shm-writer thread at 23.49 Hz.
INNER_MIND_15D = RegistrySpec(
    name="inner_mind_15d",
    dtype=np.dtype("<f4"),
    shape=(15,),
    feature_flag="microkernel.shm_mind_fast_enabled",
)
