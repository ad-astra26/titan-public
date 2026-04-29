"""
CGN Shared Memory Weight Propagation Protocol.

Writes/reads V(s) and Q(s,a) weights via /dev/shm for near-real-time
propagation (<1ms). Used by CGN Worker (writer) and CGNConsumerClient (reader).

Two operating modes (selected by feature flag at runtime):

LEGACY MODE (default — flag off, byte-identical pre-S4 behavior):
  Single global file ``/dev/shm/cgn_live_weights.bin``.
  Atomic write via tmp-file + rename.
  Header: 16B [version:4][num_consumers:4][vnet_bytes:4][total_payload:4]

STATEREGISTRY MODE (flag on — Microkernel v2 S4 alignment):
  Per-titan path ``/dev/shm/titan_{id}/cgn_live_weights.bin``.
  StateRegistry framework (mmap + SeqLock + 24B header).
  Inner payload preserves the legacy 16B-header-prefixed format
  verbatim — readers parse identically once they have the bytes.

Per the 2026-04-17 Maker invariant + 2026-04-21 codification
(memory/project_cgn_as_higher_state_registry.md), CGN is the higher-
cognitive-level state registry that MUST share the same shm+version
protocol as Trinity. STATEREGISTRY MODE closes the keystone gap.

Authority preserved: cgn_worker remains sole writer in both modes.

No external dependencies in legacy mode (struct + mmap only). Adds
StateRegistry import in stateregistry mode (titan_plugin.core.state_registry).
"""

import logging
import mmap
import os
import struct
import tempfile
import time

import numpy as np
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)

HEADER_SIZE = 16  # legacy header: version(4) + num_consumers(4) + vnet_bytes(4) + total(4)
DEFAULT_SHM_PATH = "/dev/shm/cgn_live_weights.bin"
MAX_SHM_SIZE = 256 * 1024  # 256KB — generous ceiling for 10 consumers


# ── S4: flag-aware mode resolution ─────────────────────────────────

def _resolve_cgn_mode(
    shm_path: str | None,
    titan_id: str | None,
    config: dict | None,
) -> tuple[str, bool]:
    """Resolve (path, use_stateregistry_format) for CGN shm I/O.

    Mode selection (PLAN §2.5 D10):
      1. Explicit non-default shm_path literal: legacy mode at that path
         (escape hatch for tests + custom deployments).
      2. Flag ``microkernel.shm_cgn_format_alignment_enabled = true``:
         STATEREGISTRY mode at per-titan path.
      3. Otherwise: LEGACY mode at DEFAULT_SHM_PATH (current behavior).

    Returns (resolved_path, use_stateregistry_format).
    """
    # Explicit literal escape hatch
    if shm_path and shm_path != DEFAULT_SHM_PATH:
        return (shm_path, False)

    cfg = config or {}
    enabled = cfg.get("microkernel", {}).get(
        "shm_cgn_format_alignment_enabled", False)

    if enabled:
        try:
            from titan_plugin.core.state_registry import resolve_shm_root
            root = resolve_shm_root(titan_id)
            root.mkdir(parents=True, exist_ok=True)
            return (str(root / "cgn_live_weights.bin"), True)
        except Exception as e:
            logger.warning(
                "[cgn_shm_protocol] StateRegistry mode resolution failed "
                "(falling back to legacy): %s", e)
            return (DEFAULT_SHM_PATH, False)

    return (DEFAULT_SHM_PATH, False)


class ShmWeightWriter:
    """Write CGN weights to /dev/shm. Used by CGN Worker only.

    Dual-mode (S4): legacy 16B-header + global path OR 24B StateRegistry
    header + per-titan path. Mode selected at construction via flag.
    """

    def __init__(self, shm_path: str | None = None,
                 titan_id: str | None = None,
                 config: dict | None = None):
        self._path, self._use_stateregistry = _resolve_cgn_mode(
            shm_path, titan_id, config)
        self._version = 0
        self._sr_writer = None  # lazy: only init in stateregistry mode
        if self._use_stateregistry:
            try:
                from titan_plugin.core.state_registry import (
                    CGN_LIVE_WEIGHTS, StateRegistryWriter, resolve_shm_root,
                )
                self._sr_writer = StateRegistryWriter(
                    CGN_LIVE_WEIGHTS, resolve_shm_root(titan_id))
                # Initialize with empty payload — preserves legacy semantics
                # for check_version() (returns -1 when no real write yet).
                # Without this, the preallocated mmap region would expose
                # MAX-sized zero-filled payload as version=0, causing readers
                # to treat init state as a stale write.
                self._sr_writer.write_variable(b"")
                logger.info(
                    "[ShmWriter] StateRegistry mode active (path=%s)",
                    self._path)
            except Exception as e:
                logger.warning(
                    "[ShmWriter] StateRegistry init failed (falling back to "
                    "legacy): %s", e)
                self._use_stateregistry = False
                self._path = DEFAULT_SHM_PATH

    def write_full(self, value_net_state: dict,
                   consumer_nets: dict[str, dict],
                   extra: bytes = b"") -> int:
        """Write complete weight snapshot to /dev/shm.

        Args:
            value_net_state: V(s) state_dict (from net.state_dict())
            consumer_nets: {name: state_dict} for each consumer
            extra: optional additional data (HAOV rules, surprise buffer)

        Returns:
            New version counter.
        """
        self._version += 1

        # Serialize V(s) weights to contiguous float32
        v_bytes = _state_dict_to_bytes(value_net_state)

        # Serialize consumer nets
        consumer_entries = []
        for name, state_dict in sorted(consumer_nets.items()):
            name_bytes = name.encode("utf-8")
            weights_bytes = _state_dict_to_bytes(state_dict)
            # Format: [4B name_len][name][4B weights_len][weights]
            entry = (struct.pack("<I", len(name_bytes)) + name_bytes +
                     struct.pack("<I", len(weights_bytes)) + weights_bytes)
            consumer_entries.append(entry)

        consumers_blob = b"".join(consumer_entries)

        # Build full payload (16B inner header + V-bytes + consumers + extra).
        # This payload format is preserved verbatim in BOTH modes — readers
        # parse it identically once they have the bytes (the 24B SeqLock
        # header in stateregistry mode wraps but does not alter it).
        total_payload = len(v_bytes) + len(consumers_blob) + len(extra)
        header = struct.pack("<IIII",
                             self._version,
                             len(consumer_nets),
                             len(v_bytes),
                             total_payload)
        payload = header + v_bytes + consumers_blob + extra

        if self._use_stateregistry:
            # S4 STATEREGISTRY MODE: write through StateRegistry framework.
            # 24B outer header (SeqLock + per-titan path); inner format
            # unchanged (16B legacy header + V-bytes + consumers + extra).
            try:
                self._sr_writer.write_variable(payload)
            except Exception as e:
                logger.warning(
                    "[ShmWriter] StateRegistry write_variable failed: %s", e)
            return self._version

        # LEGACY MODE: atomic tmp-file + rename (byte-identical pre-S4).
        tmp_path = self._path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except Exception as e:
            logger.warning("[ShmWriter] Write failed: %s", e)
            try:
                os.remove(tmp_path)
            except OSError as _swallow_exc:
                swallow_warn('[logic.cgn_shm_protocol] ShmWeightWriter.write_full: os.remove(tmp_path)', _swallow_exc,
                             key='logic.cgn_shm_protocol.ShmWeightWriter.write_full.line191', throttle=100)

        return self._version

    def get_version(self) -> int:
        return self._version


class ShmWeightReader:
    """Read CGN weights from /dev/shm. Used by CGNConsumerClient.

    Dual-mode (S4): legacy 16B-header file read OR 24B StateRegistry
    SeqLock read. Mode selected at construction via flag.
    """

    def __init__(self, shm_path: str | None = None,
                 titan_id: str | None = None,
                 config: dict | None = None):
        self._path, self._use_stateregistry = _resolve_cgn_mode(
            shm_path, titan_id, config)
        self._last_version = -1
        self._cache = None  # Cached parsed result
        self._sr_reader = None  # lazy: only init in stateregistry mode
        if self._use_stateregistry:
            try:
                from titan_plugin.core.state_registry import (
                    CGN_LIVE_WEIGHTS, StateRegistryReader, resolve_shm_root,
                )
                self._sr_reader = StateRegistryReader(
                    CGN_LIVE_WEIGHTS, resolve_shm_root(titan_id))
                logger.info(
                    "[ShmReader] StateRegistry mode active (path=%s)",
                    self._path)
            except Exception as e:
                logger.warning(
                    "[ShmReader] StateRegistry init failed (falling back to "
                    "legacy): %s", e)
                self._use_stateregistry = False
                self._path = DEFAULT_SHM_PATH

    def _read_payload(self) -> bytes | None:
        """Mode-aware payload fetch — returns the inner [16B header +
        V-bytes + consumers + extra] payload regardless of mode, so the
        downstream parser is identical."""
        if self._use_stateregistry and self._sr_reader is not None:
            try:
                return self._sr_reader.read_variable()
            except Exception as e:
                logger.warning(
                    "[ShmReader] StateRegistry read_variable failed: %s", e)
                return None
        # Legacy mode
        try:
            with open(self._path, "rb") as f:
                return f.read()
        except (FileNotFoundError, OSError):
            return None

    def check_version(self) -> int:
        """Read version counter only (first 4 bytes of inner header).
        Very cheap in legacy mode (4-byte file read); in StateRegistry
        mode requires a full mmap read since SeqLock validates header
        before exposing payload."""
        if self._use_stateregistry:
            payload = self._read_payload()
            if payload is None or len(payload) < 4:
                return -1
            return struct.unpack("<I", payload[:4])[0]
        try:
            with open(self._path, "rb") as f:
                raw = f.read(4)
                if len(raw) == 4:
                    return struct.unpack("<I", raw)[0]
        except (FileNotFoundError, OSError) as _swallow_exc:
            swallow_warn("[logic.cgn_shm_protocol] ShmWeightReader.check_version: with open(self._path, 'rb') as f: raw = f.read(4) if len(...", _swallow_exc,
                         key='logic.cgn_shm_protocol.ShmWeightReader.check_version.line265', throttle=100)
        return -1

    def has_new_version(self) -> bool:
        """Check if weights have been updated since last read."""
        return self.check_version() > self._last_version

    def read(self, consumer_name: str = None) -> dict | None:
        """Read weights if version changed. Returns None if no change.

        Returns:
            {
                "version": int,
                "value_net": OrderedDict (state_dict),
                "consumer_net": OrderedDict (state_dict) or None,
                "extra": bytes,
            }
        """
        current_version = self.check_version()
        if current_version <= self._last_version:
            return None  # No change

        data = self._read_payload()
        if data is None or len(data) < HEADER_SIZE:
            return None

        version, num_consumers, v_bytes_len, total_payload = struct.unpack(
            "<IIII", data[:HEADER_SIZE])

        # Parse V(s) weights
        v_start = HEADER_SIZE
        v_end = v_start + v_bytes_len
        v_state = _bytes_to_state_dict(data[v_start:v_end])

        # Parse consumer nets
        offset = v_end
        consumer_state = None
        for _ in range(num_consumers):
            if offset + 4 > len(data):
                break
            name_len = struct.unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            name = data[offset:offset + name_len].decode("utf-8")
            offset += name_len
            if offset + 4 > len(data):
                break
            weights_len = struct.unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            if consumer_name and name == consumer_name:
                consumer_state = _bytes_to_state_dict(
                    data[offset:offset + weights_len])
            offset += weights_len

        extra = data[offset:] if offset < len(data) else b""

        self._last_version = version
        result = {
            "version": version,
            "value_net": v_state,
            "consumer_net": consumer_state,
            "extra": extra,
        }
        self._cache = result
        return result

    def read_numpy(self, consumer_name: str = None) -> dict | None:
        """Read weights as numpy arrays (NO torch needed).

        Same as read() but returns numpy arrays instead of torch tensors.
        Used by CGNConsumerClient for torch-free inference.
        """
        current_version = self.check_version()
        if current_version <= self._last_version:
            return None

        data = self._read_payload()
        if data is None or len(data) < HEADER_SIZE:
            return None

        version, num_consumers, v_bytes_len, total_payload = struct.unpack(
            "<IIII", data[:HEADER_SIZE])

        v_start = HEADER_SIZE
        v_end = v_start + v_bytes_len
        v_state = _bytes_to_numpy_dict(data[v_start:v_end])

        offset = v_end
        consumer_state = None
        for _ in range(num_consumers):
            if offset + 4 > len(data):
                break
            name_len = struct.unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            name = data[offset:offset + name_len].decode("utf-8")
            offset += name_len
            if offset + 4 > len(data):
                break
            weights_len = struct.unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            if consumer_name and name == consumer_name:
                consumer_state = _bytes_to_numpy_dict(
                    data[offset:offset + weights_len])
            offset += weights_len

        extra = data[offset:] if offset < len(data) else b""

        self._last_version = version
        result = {
            "version": version,
            "value_net": v_state,
            "consumer_net": consumer_state,
            "extra": extra,
        }
        self._cache = result
        return result

    def get_cached(self) -> dict | None:
        """Return last read result without re-reading."""
        return self._cache


# ── Serialization helpers ─────────────────────────────────────────────

def _state_dict_to_bytes(state_dict: dict) -> bytes:
    """Serialize a PyTorch state_dict to bytes.

    Format: [4B num_params][for each: 4B name_len, name, 4B shape_len,
             shape_bytes, flat float32 data]
    """
    import torch
    parts = []
    params = list(state_dict.items())
    parts.append(struct.pack("<I", len(params)))

    for name, tensor in params:
        name_bytes = name.encode("utf-8")
        shape = list(tensor.shape)
        shape_bytes = struct.pack(f"<{len(shape)}i", *shape)
        flat = tensor.detach().cpu().float().numpy().tobytes()

        parts.append(struct.pack("<I", len(name_bytes)))
        parts.append(name_bytes)
        parts.append(struct.pack("<I", len(shape)))
        parts.append(shape_bytes)
        parts.append(struct.pack("<I", len(flat)))
        parts.append(flat)

    return b"".join(parts)


def _bytes_to_state_dict(data: bytes) -> dict:
    """Deserialize bytes back to a state_dict (OrderedDict of tensors)."""
    import torch
    from collections import OrderedDict

    if not data or len(data) < 4:
        return OrderedDict()

    result = OrderedDict()
    offset = 0

    num_params = struct.unpack("<I", data[offset:offset + 4])[0]
    offset += 4

    for _ in range(num_params):
        if offset + 4 > len(data):
            break
        name_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        name = data[offset:offset + name_len].decode("utf-8")
        offset += name_len

        ndims = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        shape = list(struct.unpack(f"<{ndims}i", data[offset:offset + ndims * 4]))
        offset += ndims * 4

        flat_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        flat = np.frombuffer(data[offset:offset + flat_len], dtype=np.float32)
        offset += flat_len

        tensor = torch.from_numpy(flat.copy().reshape(shape))
        result[name] = tensor

    return result


def _bytes_to_numpy_dict(data: bytes) -> dict:
    """Deserialize bytes to OrderedDict of numpy arrays (NO torch needed).

    Identical binary format to _bytes_to_state_dict but returns numpy arrays
    instead of torch tensors. Used by CGNConsumerClient for torch-free inference.
    """
    from collections import OrderedDict

    if not data or len(data) < 4:
        return OrderedDict()

    result = OrderedDict()
    offset = 0

    num_params = struct.unpack("<I", data[offset:offset + 4])[0]
    offset += 4

    for _ in range(num_params):
        if offset + 4 > len(data):
            break
        name_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        name = data[offset:offset + name_len].decode("utf-8")
        offset += name_len

        ndims = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        shape = list(struct.unpack(f"<{ndims}i", data[offset:offset + ndims * 4]))
        offset += ndims * 4

        flat_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        flat = np.frombuffer(data[offset:offset + flat_len], dtype=np.float32)
        offset += flat_len

        result[name] = flat.copy().reshape(shape)

    return result
