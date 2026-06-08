"""titan_hcl/core/reload_status.py — SHM-native hot-reload result slot.

Per `RFP_shm_native_hot_reload.md` (Phase A; closes D-SPEC-151 G3):

  - The Orchestrator (titan_hcl) owns a single SHM slot
    `/dev/shm/titan_<id>/reload_status.bin` (G21 single-writer = the
    canonical orchestrator process — the only one that runs
    `_reload_module_sync`).
  - On every reload status transition (``spawning → adopted →
    ready|failed|rolled_back``) the orchestrator writes a `ReloadStatusEntry`
    here; the initiator (`GuardianHCLClient.reload_module`) polls it back by
    `swap_id` instead of awaiting a bus `MODULE_RELOAD_ACK` — the deleted
    `dst="all"` broadcast (SPEC §8.2 / Preamble G18 violation: *state*
    readout over the bus).
  - Layout: per-module RING of the last `RELOAD_STATUS_RING_N` swap_ids,
    deduped by swap_id (a later status for the same swap_id updates that ring
    entry in place; a new swap_id pushes and evicts the oldest). Gives the
    Observatory live reload progress; the initiator matches only its own
    swap_id reaching a terminal status.

NO bus broadcast accompanies the write (G18). The transition IS the SHM write.

Wire encoding: msgpack-serialized ``{module_name: [ReloadStatusEntry.as_wire_dict(), ...]}``
into a variable-size `StateRegistryWriter` slot — same `variable_size=True`
pattern as `titan_hcl_state.bin` / `module_<name>_state.bin`.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import msgpack
import numpy as np

from .state_registry import (
    RegistrySpec,
    StateRegistryReader,
    StateRegistryWriter,
    ensure_shm_root,
    resolve_shm_root,
)

# Terminal statuses — the initiator resolves only on these (intermediate
# ``spawning``/``adopted`` are stamped for live observability but ignored by
# the initiator's poll).
RELOAD_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"ready", "failed", "rolled_back"})

# Per-module ring depth (last N swap_ids kept).
RELOAD_STATUS_RING_N: int = 8

# Slot capacity: worst case ~44 modules × RING_N × ~200 B msgpack ≈ 70 KB.
# 128 KB leaves headroom (in practice only reloaded modules appear in the map).
RELOAD_STATUS_PAYLOAD_BYTES: int = 128 * 1024

# Container byte-layout version (state_registry slot header). Fresh slot → 1.
RELOAD_STATUS_SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class ReloadStatusEntry:
    """One reload status transition for one module (per SPEC §8.3 shape +
    new_pid/old_pid for swap forensics)."""
    module_name: str
    swap_id: str
    status: str                       # spawning|adopted|ready|failed|rolled_back
    reason: Optional[str] = None
    new_pid: int = 0
    old_pid: int = 0
    total_elapsed_ms: int = 0
    written_at: float = field(default_factory=time.time)

    @property
    def is_terminal(self) -> bool:
        return self.status in RELOAD_TERMINAL_STATUSES

    def as_wire_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "swap_id": self.swap_id,
            "status": self.status,
            "reason": self.reason,
            "new_pid": int(self.new_pid),
            "old_pid": int(self.old_pid),
            "total_elapsed_ms": int(self.total_elapsed_ms),
            "written_at": float(self.written_at),
        }

    def as_result_dict(self) -> dict[str, Any]:
        """The dict `GuardianHCLClient.reload_module` returns — mirrors the
        legacy `_reload_result` / MODULE_RELOAD_ACK payload (SPEC §8.3), plus
        `new_pid` for the caller's pid-swap confirmation."""
        return {
            "swap_id": self.swap_id,
            "module_name": self.module_name,
            "status": self.status,
            "reason": self.reason,
            "new_pid": int(self.new_pid),
            "total_elapsed_ms": int(self.total_elapsed_ms),
            "ts": float(self.written_at),
        }

    @classmethod
    def from_wire_dict(cls, d: dict[str, Any]) -> "ReloadStatusEntry":
        return cls(
            module_name=str(d["module_name"]),
            swap_id=str(d["swap_id"]),
            status=str(d["status"]),
            reason=(None if d.get("reason") is None else str(d.get("reason"))),
            new_pid=int(d.get("new_pid", 0)),
            old_pid=int(d.get("old_pid", 0)),
            total_elapsed_ms=int(d.get("total_elapsed_ms", 0)),
            written_at=float(d.get("written_at", time.time())),
        )


_RELOAD_STATUS_SLOT_NAME = "reload_status"


def make_reload_status_registry_spec() -> RegistrySpec:
    return RegistrySpec(
        name=_RELOAD_STATUS_SLOT_NAME,
        dtype=np.dtype(np.uint8),
        shape=(RELOAD_STATUS_PAYLOAD_BYTES,),
        feature_flag="",
        schema_version=RELOAD_STATUS_SCHEMA_VERSION,
        variable_size=True,
    )


class ReloadStatusWriter:
    """Orchestrator-side single-writer for `reload_status.bin`.

    Holds the per-module rings in-process; each `write()` updates one module's
    ring (dedup by swap_id) and republishes the whole `{name: [entries]}` map.
    The lock serializes the read-modify-write across orchestrator threads (the
    `_restart_executor` reload thread + the lifecycle-subscriber thread that
    dispatches malformed/dispatch-error statuses).
    """

    def __init__(self, *, titan_id: Optional[str] = None,
                 ring_n: int = RELOAD_STATUS_RING_N) -> None:
        shm_root: Path = ensure_shm_root(titan_id)
        self._spec = make_reload_status_registry_spec()
        self._writer = StateRegistryWriter(self._spec, shm_root)
        self._ring_n = max(1, int(ring_n))
        self._rings: dict[str, list[ReloadStatusEntry]] = {}
        self._lock = threading.Lock()
        self._publish()

    def write(self, module_name: str, entry: ReloadStatusEntry) -> None:
        """Update the module's ring (dedup by swap_id) + republish to SHM."""
        with self._lock:
            ring = self._rings.setdefault(module_name, [])
            for i, existing in enumerate(ring):
                if existing.swap_id == entry.swap_id:
                    ring[i] = entry  # later status for same swap_id → in place
                    break
            else:
                ring.append(entry)
                if len(ring) > self._ring_n:  # evict oldest
                    del ring[0:len(ring) - self._ring_n]
            self._publish()

    def _publish(self) -> int:
        wire = {
            name: [e.as_wire_dict() for e in ring]
            for name, ring in self._rings.items()
        }
        payload = msgpack.packb(wire, use_bin_type=True)
        return self._writer.write_variable(payload)

    def close(self) -> None:
        try:
            self._writer.close()
        except Exception:
            pass


class ReloadStatusReader:
    """Read-only view of `reload_status.bin`. Used by the reload initiator
    (`GuardianHCLClient`) + the Observatory."""

    def __init__(self, *, titan_id: Optional[str] = None) -> None:
        shm_root: Path = resolve_shm_root(titan_id)
        self._spec = make_reload_status_registry_spec()
        self._reader = StateRegistryReader(self._spec, shm_root)

    def _read_map(self) -> dict[str, Any]:
        raw: Optional[bytes] = self._reader.read_variable()
        if raw is None or len(raw) == 0:
            return {}
        try:
            d = msgpack.unpackb(raw, raw=False)
        except Exception:
            return {}
        return d if isinstance(d, dict) else {}

    def read(self, module_name: str) -> list[ReloadStatusEntry]:
        """The module's ring (oldest→newest), or [] if none/uninitialized."""
        rows = self._read_map().get(module_name) or []
        out: list[ReloadStatusEntry] = []
        for r in rows:
            if isinstance(r, dict):
                try:
                    out.append(ReloadStatusEntry.from_wire_dict(r))
                except Exception:
                    continue
        return out

    def read_swap(self, module_name: str,
                  swap_id: str) -> Optional[ReloadStatusEntry]:
        """The ring entry matching `swap_id`, or None."""
        for e in self.read(module_name):
            if e.swap_id == swap_id:
                return e
        return None

    def close(self) -> None:
        try:
            self._reader.close()
        except Exception:
            pass
