"""
EMOT-CGN Shared Memory State Protocol.

Writer: emot_cgn_worker (single writer). Readers: narrator, social, dream,
META-CGN ctx builder, /v4/emot-cgn dashboard endpoint. Zero-copy mmap reads
for downstream STATE queries per rFP_microkernel_v2_shadow_core §State
Registries principle ("Bus is for EVENTS, shm is for STATE").

Mirrors `titan_plugin/logic/cgn_shm_protocol.py` structure but with
fixed-layout binary format (EMOT-CGN state is small + fully-structured,
no variable-length fields needed).

Two shm files:
  /dev/shm/titan/emot_state.bin      — 64 bytes, per-update dominant state
  /dev/shm/titan/emot_grounding.bin  — 176 bytes, per-primitive β posterior

Version-counter pattern (head + trailer) prevents torn reads: reader checks
header-version == trailer-version; mismatch → reread. Same technique as
CGN shm protocol + used throughout microkernel v2 design.
"""
import logging
import mmap
import os
import struct
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Per-Titan path resolution. T2+T3 share /dev/shm on the same VPS, so a
# single shared path causes state-clobber. Resolve titan_id from:
#   1. explicit `titan_id` arg passed to writer/reader (worker path)
#   2. data/titan_identity.json (canonical — same source language_config
#      uses; per-Titan repo has its own file)
#   3. TITAN_ID env var (rare fallback)
#   4. "T1" hardcoded fallback
def _resolve_titan_id(explicit: Optional[str] = None) -> str:
    if explicit:
        return str(explicit)
    # Project root from this module's location: titan_plugin/logic/<file>
    try:
        import json
        proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
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


def _default_state_path(titan_id: Optional[str] = None) -> str:
    return f"/dev/shm/titan_{_resolve_titan_id(titan_id)}/emot_state.bin"


def _default_grounding_path(titan_id: Optional[str] = None) -> str:
    return f"/dev/shm/titan_{_resolve_titan_id(titan_id)}/emot_grounding.bin"


# Module-level constants kept for back-compat (existing tests + tools).
# Computed once at import using best-available titan_id.
DEFAULT_STATE_PATH = _default_state_path()
DEFAULT_GROUNDING_PATH = _default_grounding_path()

# state.bin layout (64 bytes)
# version(u32) ts_ms(u64) idx(u8) active(u8) registered(u8) pad(u8)
# V_beta(f32) V_blended(f32) cluster_conf(f32) cluster_dist(f32)
# total_updates(u64) ci_sent(u32) ci_recv(u32) pad(u64) version_trailer(u32) pad(u32)
STATE_STRUCT = struct.Struct("<IQBBBBffffQIIQII")
STATE_SIZE = STATE_STRUCT.size  # 64 bytes

# grounding.bin layout: header (16 bytes) + 8 × per-primitive (20 bytes) = 176
# header: version(u32) ts_ms(u64) num_prim(u32)
# per-primitive: alpha(f32) beta(f32) V(f32) conf(f32) n_samples(u32)
GROUNDING_HEADER = struct.Struct("<IQI")
GROUNDING_ENTRY = struct.Struct("<ffffI")
GROUNDING_SIZE = GROUNDING_HEADER.size + 8 * GROUNDING_ENTRY.size  # 176 bytes


def _ensure_shm_dir(path: str) -> None:
    """Create /dev/shm/titan/ if absent — matches microkernel v2
    state registry location."""
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass


class ShmEmotWriter:
    """Write EMOT-CGN state + per-primitive grounding to /dev/shm/titan/.

    Single writer — only instantiated in emot_cgn_worker. Writes two files:
    state (64B, hot-path) + grounding (176B, 8 primitives β snapshot).
    """

    def __init__(self, state_path: str = DEFAULT_STATE_PATH,
                 grounding_path: str = DEFAULT_GROUNDING_PATH):
        self._state_path = state_path
        self._grounding_path = grounding_path
        self._state_version = 0
        self._grounding_version = 0
        _ensure_shm_dir(state_path)
        _ensure_shm_dir(grounding_path)
        # Pre-create files at fixed size — mmap'd by readers once
        for p, sz in ((state_path, STATE_SIZE),
                      (grounding_path, GROUNDING_SIZE)):
            if not os.path.exists(p) or os.path.getsize(p) != sz:
                with open(p, "wb") as f:
                    f.write(b"\x00" * sz)

    def write_state(self, *, dominant_idx: int, is_active: bool,
                    cgn_registered: bool, V_beta: float, V_blended: float,
                    cluster_confidence: float, cluster_distance: float,
                    total_updates: int, cross_insights_sent: int,
                    cross_insights_received: int) -> int:
        """Write emot_state.bin with version-counter torn-read protection.

        Returns the new version number. Called from emot_cgn_worker on
        every state-changing event (new chain evidence, new felt cluster).
        """
        try:
            self._state_version += 1
            ts_ms = int(time.time() * 1000)
            packed = STATE_STRUCT.pack(
                self._state_version, ts_ms,
                max(0, min(7, int(dominant_idx))),
                1 if is_active else 0,
                1 if cgn_registered else 0,
                0,  # pad
                float(V_beta), float(V_blended),
                float(cluster_confidence), float(cluster_distance),
                int(max(0, total_updates)),
                int(max(0, cross_insights_sent)),
                int(max(0, cross_insights_received)),
                0,  # reserved u64
                self._state_version,  # trailer
                0,  # reserved u32
            )
            with open(self._state_path, "r+b") as f:
                f.write(packed)
            return self._state_version
        except Exception as e:
            logger.warning("[ShmEmotWriter] write_state failed: %s", e)
            return self._state_version

    def write_grounding(self, primitives_snapshot: list) -> int:
        """Write emot_grounding.bin. primitives_snapshot is a list of 8
        dicts (ordered per EMOT_PRIMITIVES) each with alpha/beta/V/
        confidence/n_samples. Torn-read protection via header-only version;
        writers are infrequent (per save_cadence_chains) so simple header
        + repack is sufficient.
        """
        try:
            self._grounding_version += 1
            ts_ms = int(time.time() * 1000)
            buf = bytearray(GROUNDING_SIZE)
            GROUNDING_HEADER.pack_into(
                buf, 0, self._grounding_version, ts_ms, 8)
            off = GROUNDING_HEADER.size
            for i in range(8):
                p = primitives_snapshot[i] if i < len(primitives_snapshot) else {}
                GROUNDING_ENTRY.pack_into(
                    buf, off,
                    float(p.get("alpha", 1.0)),
                    float(p.get("beta", 1.0)),
                    float(p.get("V", 0.5)),
                    float(p.get("confidence", 0.0)),
                    int(max(0, p.get("n_samples", 0))),
                )
                off += GROUNDING_ENTRY.size
            with open(self._grounding_path, "r+b") as f:
                f.write(bytes(buf))
            return self._grounding_version
        except Exception as e:
            logger.warning("[ShmEmotWriter] write_grounding failed: %s", e)
            return self._grounding_version


class ShmEmotReader:
    """Zero-copy mmap reader for EMOT-CGN state. Multi-reader safe.

    Instantiated by narrator/social/dream/META-CGN/dashboard. Reads return
    None when shm unavailable (e.g. emot_cgn_worker not yet booted or
    crashed — failsafe to caller's in-process fallback).
    """

    def __init__(self, state_path: str = DEFAULT_STATE_PATH,
                 grounding_path: str = DEFAULT_GROUNDING_PATH):
        self._state_path = state_path
        self._grounding_path = grounding_path
        self._last_state_version = 0
        self._last_grounding_version = 0

    def _read_bytes(self, path: str, size: int):
        """Read fixed-size bytes via mmap. Returns None if unavailable."""
        try:
            if not os.path.exists(path) or os.path.getsize(path) != size:
                return None
            with open(path, "rb") as f:
                with mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ) as mm:
                    return bytes(mm)
        except Exception:
            return None

    def read_state(self):
        """Return dict of current emot_state fields, or None on failure.
        Automatic torn-read protection via header+trailer version match."""
        data = self._read_bytes(self._state_path, STATE_SIZE)
        if data is None:
            return None
        try:
            fields = STATE_STRUCT.unpack(data)
            ver_head, ts_ms, idx, active, registered, _pad, vb, vbl = fields[:8]
            cc, cd, total_u, ci_s, ci_r, _res1, ver_tail, _res2 = fields[8:]
            if ver_head != ver_tail:
                return None  # torn read
            self._last_state_version = ver_head
            return {
                "version": ver_head,
                "last_update_ts_ms": ts_ms,
                "dominant_idx": idx,
                "is_active": bool(active),
                "cgn_registered": bool(registered),
                "V_beta": vb,
                "V_blended": vbl,
                "cluster_confidence": cc,
                "cluster_distance": cd,
                "total_updates": total_u,
                "cross_insights_sent": ci_s,
                "cross_insights_received": ci_r,
            }
        except Exception:
            return None

    def read_grounding(self):
        """Return list of 8 per-primitive dicts (ordered per EMOT_PRIMITIVES)
        or None on failure."""
        data = self._read_bytes(self._grounding_path, GROUNDING_SIZE)
        if data is None:
            return None
        try:
            ver, ts_ms, n = GROUNDING_HEADER.unpack(data[:GROUNDING_HEADER.size])
            if n != 8:
                return None
            self._last_grounding_version = ver
            entries = []
            off = GROUNDING_HEADER.size
            for _ in range(8):
                a, b, V, conf, ns = GROUNDING_ENTRY.unpack(
                    data[off:off + GROUNDING_ENTRY.size])
                entries.append({
                    "alpha": a, "beta": b, "V": V,
                    "confidence": conf, "n_samples": ns,
                })
                off += GROUNDING_ENTRY.size
            return entries
        except Exception:
            return None

    def has_new_state(self) -> bool:
        """Check whether state has been updated since last read. Cheap
        version-counter read without full unpack — useful for hot-path
        consumers that only need to poll for changes."""
        try:
            if not os.path.exists(self._state_path):
                return False
            with open(self._state_path, "rb") as f:
                ver_bytes = f.read(4)
            if len(ver_bytes) < 4:
                return False
            ver = struct.unpack("<I", ver_bytes)[0]
            return ver > self._last_state_version
        except Exception:
            return False
