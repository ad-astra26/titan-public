"""titan_hcl/core/titan_hcl_state.py — Phase 11 §11.I.7 orchestrator state slot.

Per SPEC §11.I.1 + §11.I.7 (D-SPEC-141 / v1.65.0):

  - The Orchestrator (titan_hcl) owns a single SHM slot
    `/dev/shm/titan_<id>/titan_hcl_state.bin` (G21 single-writer = the
    orchestrator process).
  - After Phase A (MANDATORY) completes — every MANDATORY module's
    `module_<name>_state.bin` shows state=RUNNING — titan_hcl writes
    `fleet_ready=true` to this slot.
  - After Phase B (OPTIONAL_POST_BOOT) completes, titan_hcl writes
    `fleet_optional_ready=true` (informational; kernel-rs only blocks on
    fleet_ready).
  - kernel-rs + guardian_hcl + observatory + api READ this slot via
    `TitanHclStateReader` (1Hz polling, no broker mediation).

NO bus broadcast accompanies the write (locked D1). The transition is the
SHM write itself.

Wire encoding: msgpack-serialized `TitanHclStateEntry.as_wire_dict()` into
a variable-size StateRegistryWriter slot. Capacity is small (counter dict
is tiny) but uses the same `variable_size=True` pattern as
`module_<name>_state.bin` for consistency.
"""
from __future__ import annotations

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

# 4 KB is plenty for a fixed-shape dict of ~10 fields + a few counters.
TITAN_HCL_STATE_PAYLOAD_BYTES: int = 4 * 1024
TITAN_HCL_STATE_SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class TitanHclStateEntry:
    """Per §11.I.7 — the orchestrator's published fleet-readiness snapshot.

    `boot_phase` transitions: ``"booting_a" → "phase_a_done" → "booting_b"
    → "phase_b_done" → "quiescent"``. Workers + observatory ignore the
    string and gate purely on the booleans; the string is operator-facing.
    """
    fleet_ready: bool = False                  # all MANDATORY modules state=RUNNING
    fleet_optional_ready: bool = False         # all OPTIONAL_POST_BOOT modules state=RUNNING
    boot_phase: str = "booting_a"
    mandatory_total: int = 0
    mandatory_ready: int = 0
    post_boot_total: int = 0
    post_boot_ready: int = 0
    lazy_total: int = 0
    boot_started_at: float = 0.0
    fleet_ready_at: float = 0.0
    fleet_optional_ready_at: float = 0.0
    schema_version: int = TITAN_HCL_STATE_SCHEMA_VERSION
    written_at: float = field(default_factory=time.time)

    def as_wire_dict(self) -> dict[str, Any]:
        return {
            "fleet_ready": bool(self.fleet_ready),
            "fleet_optional_ready": bool(self.fleet_optional_ready),
            "boot_phase": str(self.boot_phase),
            "mandatory_total": int(self.mandatory_total),
            "mandatory_ready": int(self.mandatory_ready),
            "post_boot_total": int(self.post_boot_total),
            "post_boot_ready": int(self.post_boot_ready),
            "lazy_total": int(self.lazy_total),
            "boot_started_at": float(self.boot_started_at),
            "fleet_ready_at": float(self.fleet_ready_at),
            "fleet_optional_ready_at": float(self.fleet_optional_ready_at),
            "schema_version": int(self.schema_version),
            "written_at": float(self.written_at),
        }

    @classmethod
    def from_wire_dict(cls, d: dict[str, Any]) -> "TitanHclStateEntry":
        return cls(
            fleet_ready=bool(d.get("fleet_ready", False)),
            fleet_optional_ready=bool(d.get("fleet_optional_ready", False)),
            boot_phase=str(d.get("boot_phase", "booting_a")),
            mandatory_total=int(d.get("mandatory_total", 0)),
            mandatory_ready=int(d.get("mandatory_ready", 0)),
            post_boot_total=int(d.get("post_boot_total", 0)),
            post_boot_ready=int(d.get("post_boot_ready", 0)),
            lazy_total=int(d.get("lazy_total", 0)),
            boot_started_at=float(d.get("boot_started_at", 0.0)),
            fleet_ready_at=float(d.get("fleet_ready_at", 0.0)),
            fleet_optional_ready_at=float(d.get("fleet_optional_ready_at", 0.0)),
            schema_version=int(d.get("schema_version", TITAN_HCL_STATE_SCHEMA_VERSION)),
            written_at=float(d.get("written_at", time.time())),
        )


_TITAN_HCL_STATE_SLOT_NAME = "titan_hcl_state"


def make_titan_hcl_state_registry_spec() -> RegistrySpec:
    return RegistrySpec(
        name=_TITAN_HCL_STATE_SLOT_NAME,
        dtype=np.dtype(np.uint8),
        shape=(TITAN_HCL_STATE_PAYLOAD_BYTES,),
        feature_flag="",
        schema_version=TITAN_HCL_STATE_SCHEMA_VERSION,
        variable_size=True,
    )


class TitanHclStateWriter:
    """Orchestrator-side single-writer for `titan_hcl_state.bin`.

    Holds the latest entry in-process so each phase-boundary write only
    needs to flip the relevant booleans + bump counters + republish.
    """

    def __init__(self, *, titan_id: Optional[str] = None) -> None:
        shm_root: Path = ensure_shm_root(titan_id)
        self._spec = make_titan_hcl_state_registry_spec()
        self._writer = StateRegistryWriter(self._spec, shm_root)
        self._entry: TitanHclStateEntry = TitanHclStateEntry(
            boot_started_at=time.time())
        self._publish()

    @property
    def entry(self) -> TitanHclStateEntry:
        return self._entry

    def update(
        self,
        *,
        fleet_ready: Optional[bool] = None,
        fleet_optional_ready: Optional[bool] = None,
        boot_phase: Optional[str] = None,
        mandatory_total: Optional[int] = None,
        mandatory_ready: Optional[int] = None,
        post_boot_total: Optional[int] = None,
        post_boot_ready: Optional[int] = None,
        lazy_total: Optional[int] = None,
    ) -> TitanHclStateEntry:
        """Mutate the in-memory entry then publish to SHM.

        Only the supplied keyword args change; the rest carry forward.
        First-true transitions of fleet_ready / fleet_optional_ready also
        latch their wall-clock timestamps.
        """
        now = time.time()
        new = TitanHclStateEntry(
            fleet_ready=(self._entry.fleet_ready if fleet_ready is None
                         else bool(fleet_ready)),
            fleet_optional_ready=(self._entry.fleet_optional_ready
                                  if fleet_optional_ready is None
                                  else bool(fleet_optional_ready)),
            boot_phase=(self._entry.boot_phase if boot_phase is None
                        else str(boot_phase)),
            mandatory_total=(self._entry.mandatory_total
                             if mandatory_total is None
                             else int(mandatory_total)),
            mandatory_ready=(self._entry.mandatory_ready
                             if mandatory_ready is None
                             else int(mandatory_ready)),
            post_boot_total=(self._entry.post_boot_total
                             if post_boot_total is None
                             else int(post_boot_total)),
            post_boot_ready=(self._entry.post_boot_ready
                             if post_boot_ready is None
                             else int(post_boot_ready)),
            lazy_total=(self._entry.lazy_total if lazy_total is None
                        else int(lazy_total)),
            boot_started_at=self._entry.boot_started_at,
            fleet_ready_at=(now if (fleet_ready and not self._entry.fleet_ready)
                            else self._entry.fleet_ready_at),
            fleet_optional_ready_at=(
                now if (fleet_optional_ready
                        and not self._entry.fleet_optional_ready)
                else self._entry.fleet_optional_ready_at),
        )
        self._entry = new
        self._publish()
        return new

    def _publish(self) -> int:
        payload = msgpack.packb(self._entry.as_wire_dict(), use_bin_type=True)
        return self._writer.write_variable(payload)

    def close(self) -> None:
        try:
            self._writer.close()
        except Exception:
            pass


class TitanHclStateReader:
    """Read-only view of `titan_hcl_state.bin`. Used by kernel-rs +
    guardian_hcl + observatory + api."""

    def __init__(self, *, titan_id: Optional[str] = None) -> None:
        shm_root: Path = resolve_shm_root(titan_id)
        self._spec = make_titan_hcl_state_registry_spec()
        self._reader = StateRegistryReader(self._spec, shm_root)

    def read(self) -> Optional[TitanHclStateEntry]:
        raw: Optional[bytes] = self._reader.read_variable()
        if raw is None or len(raw) == 0:
            return None
        try:
            d = msgpack.unpackb(raw, raw=False)
        except Exception:
            return None
        if not isinstance(d, dict):
            return None
        try:
            return TitanHclStateEntry.from_wire_dict(d)
        except Exception:
            return None

    def close(self) -> None:
        try:
            self._reader.close()
        except Exception:
            pass
