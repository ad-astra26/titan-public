"""
CGN Shared Memory Weight Propagation Protocol.

Writes/reads V(s) and Q(s,a) weights via /dev/shm for near-real-time
propagation (<1ms). Used by CGN Worker (writer) and CGNConsumerClient (reader).

Binary format:
  [0:4]    uint32 version_counter
  [4:8]    uint32 num_consumers
  [8:12]   uint32 value_net_bytes
  [12:16]  uint32 total_payload_bytes
  [16:V]   V(s) weights (contiguous float32)
  [V:V+C]  Consumer entries: [4B name_len][name][4B weights_len][weights]

No external dependencies (struct + mmap only).
"""

import logging
import mmap
import os
import struct
import tempfile
import time

import numpy as np

logger = logging.getLogger(__name__)

HEADER_SIZE = 16  # version(4) + num_consumers(4) + vnet_bytes(4) + total(4)
DEFAULT_SHM_PATH = "/dev/shm/cgn_live_weights.bin"
MAX_SHM_SIZE = 256 * 1024  # 256KB — generous ceiling for 10 consumers


class ShmWeightWriter:
    """Write CGN weights to /dev/shm. Used by CGN Worker only."""

    def __init__(self, shm_path: str = DEFAULT_SHM_PATH):
        self._path = shm_path
        self._version = 0

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

        # Build full payload
        total_payload = len(v_bytes) + len(consumers_blob) + len(extra)
        header = struct.pack("<IIII",
                             self._version,
                             len(consumer_nets),
                             len(v_bytes),
                             total_payload)
        payload = header + v_bytes + consumers_blob + extra

        # Atomic write: write to tmp, rename
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
            except OSError:
                pass

        return self._version

    def get_version(self) -> int:
        return self._version


class ShmWeightReader:
    """Read CGN weights from /dev/shm. Used by CGNConsumerClient."""

    def __init__(self, shm_path: str = DEFAULT_SHM_PATH):
        self._path = shm_path
        self._last_version = -1
        self._cache = None  # Cached parsed result

    def check_version(self) -> int:
        """Read version counter only (4 bytes). Very cheap."""
        try:
            with open(self._path, "rb") as f:
                raw = f.read(4)
                if len(raw) == 4:
                    return struct.unpack("<I", raw)[0]
        except (FileNotFoundError, OSError):
            pass
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

        try:
            with open(self._path, "rb") as f:
                data = f.read()
        except (FileNotFoundError, OSError):
            return None

        if len(data) < HEADER_SIZE:
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

        try:
            with open(self._path, "rb") as f:
                data = f.read()
        except (FileNotFoundError, OSError):
            return None

        if len(data) < HEADER_SIZE:
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
