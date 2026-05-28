"""titan_hcl/core/module_state.py — Phase 11 Chunk 11D module-state contract.

Per SPEC §11.I.2 / §11.I.5 / §11.I.8 (D-SPEC-141 / v1.65.0):

  - Each worker owns ONE `/dev/shm/titan_<id>/module_<name>_state.bin` SHM
    slot (G21 single-writer = the worker process itself).
  - State transitions are WRITTEN to the SHM slot only — NO bus broadcasts
    (per locked D1/D2). Readers (titan_hcl, guardian_hcl, observatory, api)
    poll at 1Hz via this module's `ModuleStateReader`.
  - `BootPriority` partitions cold-boot into mandatory + post-boot + lazy
    waves per §11.I.7 / §11.I.8.
  - `ProbeResult` is the bus-RPC reply payload for the
    MODULE_PROBE_REQUEST/RESPONSE pair (§11.I.3).

Wire encoding: msgpack-serialized `ModuleStateEntry.as_wire_dict()` into a
variable-size StateRegistryWriter slot (capacity 64 KB per slot — well over
the typical envelope size of <1 KB; budget headroom for `last_error`
detail + traceback frames).
"""
from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
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
from ..errors import ModuleError

logger = logging.getLogger("titan.module_state")

# Centralized heartbeat-daemon cadence. SPEC §11.I.5: worker ~30s heartbeat,
# guardian_hcl staleness timeout 60-180s. A dedicated daemon ticking every
# 15s gives ≥4× margin under the tightest (60s) timeout even if the daemon
# thread itself is briefly CPU-starved — the whole point is that liveness no
# longer depends on the worker's main loop being unblocked (§11.B.2).
_HEARTBEAT_DAEMON_INTERVAL_S = 15.0


# ── Constants ─────────────────────────────────────────────────────────────────

# Per-slot SHM payload capacity. ModuleStateEntry msgpack-serializes well
# under 1 KB on the happy path; the 64 KB ceiling accommodates worst-case
# `last_error` traceback frames + context payloads with comfortable margin.
MODULE_STATE_PAYLOAD_BYTES: int = 64 * 1024

# Schema version for ModuleStateEntry wire format. Bump on layout changes.
MODULE_STATE_SCHEMA_VERSION: int = 1


# ── BootPriority enum (§11.I.8) ───────────────────────────────────────────────

class BootPriority(str, enum.Enum):
    """Per-module boot scheduling per SPEC §11.I.8.

    - MANDATORY:          part of Phase A; gates `fleet_ready=true` SHM publication.
    - OPTIONAL_POST_BOOT: scheduled in Phase B background after fleet ready.
    - LAZY:               never auto-started; pre-activated on-demand via
                          §11.G.2.5 ENSURE_RUNNING (memory_worker today).
    """
    MANDATORY = "mandatory"
    OPTIONAL_POST_BOOT = "post_boot"
    LAZY = "lazy"


# ── ProbeResult (§11.I.3) ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProbeResult:
    """Worker → titan_hcl probe reply payload (`MODULE_PROBE_RESPONSE`).

    Per SPEC §11.I.3: `ProbeResult = {ok: bool, latency_ms: float,
    error_envelope: Optional[ModuleError]}`.
    """
    ok: bool
    latency_ms: float = 0.0
    error_envelope: Optional[ModuleError] = None

    @classmethod
    def ok_(cls, latency_ms: float = 0.0) -> "ProbeResult":
        return cls(ok=True, latency_ms=latency_ms, error_envelope=None)

    @classmethod
    def fail(
        cls,
        error: ModuleError,
        latency_ms: float = 0.0,
    ) -> "ProbeResult":
        return cls(ok=False, latency_ms=latency_ms, error_envelope=error)

    def as_wire_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ok": bool(self.ok),
            "latency_ms": float(self.latency_ms),
            "error_envelope": (
                self.error_envelope.as_wire_dict() if self.error_envelope else None
            ),
        }
        return d

    @classmethod
    def from_wire_dict(cls, d: dict[str, Any]) -> "ProbeResult":
        env = d.get("error_envelope")
        return cls(
            ok=bool(d["ok"]),
            latency_ms=float(d.get("latency_ms", 0.0)),
            error_envelope=ModuleError.from_wire_dict(env) if env else None,
        )


# ── ModuleStateEntry (§11.I.5) ────────────────────────────────────────────────

@dataclass(frozen=True)
class ModuleStateEntry:
    """Per-module SHM slot payload per SPEC §11.I.5.

    Worker writes its OWN entry on every state transition (G21 single-writer).
    Readers poll at 1Hz via `ModuleStateReader`.

    `state` is a plain string (lowercase ModuleState enum value) for wire
    portability — the orchestrator-side enum lives in
    `titan_hcl.guardian_hcl.module_registry.ModuleState`.
    """
    name: str
    layer: str                                  # L1/L2/L3
    boot_priority: BootPriority                 # MANDATORY/OPTIONAL_POST_BOOT/LAZY
    state: str                                  # ModuleState.value (lowercase)
    pid: int = 0
    started_at: float = 0.0                     # spawn time (wall clock)
    booted_at: float = 0.0                      # in-process scaffolding done
    running_at: float = 0.0                     # probe passed
    last_heartbeat: float = 0.0                 # worker-updated periodically
    last_probe_result: Optional[ProbeResult] = None
    last_error: Optional[ModuleError] = None    # FULL envelope per D7
    restart_count: int = 0
    error_count_24h: int = 0
    schema_version: int = MODULE_STATE_SCHEMA_VERSION
    written_at: float = field(default_factory=time.time)

    def as_wire_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "layer": self.layer,
            "boot_priority": self.boot_priority.value,
            "state": self.state,
            "pid": int(self.pid),
            "started_at": float(self.started_at),
            "booted_at": float(self.booted_at),
            "running_at": float(self.running_at),
            "last_heartbeat": float(self.last_heartbeat),
            "last_probe_result": (
                self.last_probe_result.as_wire_dict()
                if self.last_probe_result else None
            ),
            "last_error": (
                self.last_error.as_wire_dict() if self.last_error else None
            ),
            "restart_count": int(self.restart_count),
            "error_count_24h": int(self.error_count_24h),
            "schema_version": int(self.schema_version),
            "written_at": float(self.written_at),
        }
        return d

    @classmethod
    def from_wire_dict(cls, d: dict[str, Any]) -> "ModuleStateEntry":
        probe = d.get("last_probe_result")
        err = d.get("last_error")
        return cls(
            name=d["name"],
            layer=d["layer"],
            boot_priority=BootPriority(d["boot_priority"]),
            state=str(d["state"]),
            pid=int(d.get("pid", 0)),
            started_at=float(d.get("started_at", 0.0)),
            booted_at=float(d.get("booted_at", 0.0)),
            running_at=float(d.get("running_at", 0.0)),
            last_heartbeat=float(d.get("last_heartbeat", 0.0)),
            last_probe_result=ProbeResult.from_wire_dict(probe) if probe else None,
            last_error=ModuleError.from_wire_dict(err) if err else None,
            restart_count=int(d.get("restart_count", 0)),
            error_count_24h=int(d.get("error_count_24h", 0)),
            schema_version=int(d.get("schema_version", MODULE_STATE_SCHEMA_VERSION)),
            written_at=float(d.get("written_at", time.time())),
        )


# ── Spec factory ──────────────────────────────────────────────────────────────

def _slot_name_for(module_name: str) -> str:
    """Canonical SHM-file basename (without extension) for a module slot.

    Resolves to `/dev/shm/titan_<id>/module_<module_name>_state.bin`.
    """
    return f"module_{module_name}_state"


def make_module_state_registry_spec(module_name: str) -> RegistrySpec:
    """Build the RegistrySpec for a per-module SHM state slot.

    Uses variable_size=True so msgpack-serialized envelopes of varying size
    (depending on whether `last_error` carries a deep traceback) all fit.
    """
    return RegistrySpec(
        name=_slot_name_for(module_name),
        dtype=np.dtype(np.uint8),
        shape=(MODULE_STATE_PAYLOAD_BYTES,),
        feature_flag="",                          # always-enabled (Phase 11 contract)
        schema_version=MODULE_STATE_SCHEMA_VERSION,
        variable_size=True,
    )


# ── Worker-side writer (G21 single-writer = the worker process) ───────────────

class ModuleStateWriter:
    """Worker-side single-writer for `/dev/shm/titan_<id>/module_<name>_state.bin`.

    Per locked D1/D2 / SPEC §11.I.2: each worker owns its own slot and writes
    state transitions there. NO bus broadcast accompanies the write.

    Typical worker lifecycle:

        from titan_hcl.core.module_state import (
            BootPriority, ModuleStateWriter, ProbeResult,
        )

        # On entry:
        w = ModuleStateWriter(module_name="agno_worker", layer="L2",
                              boot_priority=BootPriority.MANDATORY)
        w.write_state("starting")
        ...  # in-process init
        w.write_state("booted")            # gates titan_hcl probe dispatch
        ...  # receive MODULE_PROBE_REQUEST → run probe_fn:
        w.write_state("probing")
        result = run_my_probe(bus)
        if result.ok:
            w.write_state("running", last_probe_result=result)
        else:
            w.write_state("unhealthy", last_probe_result=result,
                          last_error=result.error_envelope)
        # In recv loop, periodically (every ~30s):
        w.heartbeat()

    Threading: the writer is bound to ONE process (G21 single-writer).
    Calls from multiple threads of the SAME process are safe (the underlying
    StateRegistryWriter holds the mmap; concurrent writes from threads land
    in the next-buffer slot atomically). For G21 compliance, the writer
    must NOT be shared across processes.
    """

    def __init__(
        self,
        *,
        module_name: str,
        layer: str = "L2",
        boot_priority: BootPriority = BootPriority.MANDATORY,
        titan_id: Optional[str] = None,
        pid: Optional[int] = None,
    ) -> None:
        self._module_name = module_name
        self._layer = layer
        self._boot_priority = boot_priority
        import os
        self._pid = int(pid if pid is not None else os.getpid())
        shm_root: Path = ensure_shm_root(titan_id)
        self._spec = make_module_state_registry_spec(module_name)
        self._writer = StateRegistryWriter(self._spec, shm_root)
        self._started_at: float = time.time()
        self._booted_at: float = 0.0
        self._running_at: float = 0.0
        self._last_heartbeat: float = 0.0
        self._restart_count: int = 0
        self._error_count_24h: int = 0
        # Current state + last probe/error, tracked so the centralized
        # heartbeat daemon can re-publish WITHOUT mutating the state machine
        # (heartbeat must preserve state, not force "running").
        self._current_state: Optional[str] = None
        self._last_probe_result: Optional[ProbeResult] = None
        self._last_error: Optional[ModuleError] = None
        # Single intra-process write lock (G21: one process is the writer;
        # main thread + heartbeat daemon both publish, so serialize them).
        self._write_lock = threading.Lock()
        # Daemon heartbeat thread — lazily started on first write_state() so
        # every worker that owns a slot gets robust, main-loop-independent
        # liveness (SPEC §11.B.2 / §11.I.5) without per-worker boilerplate.
        self._hb_stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_interval: float = _HEARTBEAT_DAEMON_INTERVAL_S

    @property
    def module_name(self) -> str:
        return self._module_name

    def write_state(
        self,
        state: str,
        *,
        last_probe_result: Optional[ProbeResult] = None,
        last_error: Optional[ModuleError] = None,
        restart_count: Optional[int] = None,
        error_count_24h: Optional[int] = None,
    ) -> int:
        """Publish a new state transition to the SHM slot.

        Args:
            state: lowercase ModuleState value ("starting"/"booted"/"probing"/
                   "running"/"unhealthy"/"crashed"/"disabled"). Validation is
                   the caller's responsibility; this writer accepts any string
                   so it can carry custom worker-defined sub-states if needed.
            last_probe_result: optional ProbeResult to publish atomically with
                   the state change (typical: published when state→running or
                   state→unhealthy).
            last_error: optional ModuleError envelope to publish atomically
                   (typical: when state→unhealthy or state→crashed).
            restart_count, error_count_24h: optional counter overrides.

        Returns:
            new version (monotonic per slot, used by readers to detect updates).
        """
        state = str(state)
        with self._write_lock:
            now = time.time()
            if state == "booted" and self._booted_at == 0.0:
                self._booted_at = now
            if state == "running" and self._running_at == 0.0:
                self._running_at = now
            if restart_count is not None:
                self._restart_count = int(restart_count)
            if error_count_24h is not None:
                self._error_count_24h = int(error_count_24h)
            # Sticky: last_probe_result / last_error persist across plain
            # heartbeats and state writes that don't carry a fresh one.
            if last_probe_result is not None:
                self._last_probe_result = last_probe_result
            if last_error is not None:
                self._last_error = last_error
            self._current_state = state
            ver = self._publish_locked(state)
        # Bring up the heartbeat daemon once the worker has begun its lifecycle
        # (first write_state, typically "starting"). Started outside the lock so
        # we never hold it across thread creation.
        self._ensure_heartbeat_daemon()
        return ver

    def _publish_locked(self, state: str) -> int:
        """Serialize + write the slot entry. Caller MUST hold self._write_lock."""
        entry = ModuleStateEntry(
            name=self._module_name,
            layer=self._layer,
            boot_priority=self._boot_priority,
            state=str(state),
            pid=self._pid,
            started_at=self._started_at,
            booted_at=self._booted_at,
            running_at=self._running_at,
            last_heartbeat=self._last_heartbeat,
            last_probe_result=self._last_probe_result,
            last_error=self._last_error,
            restart_count=self._restart_count,
            error_count_24h=self._error_count_24h,
        )
        payload = msgpack.packb(entry.as_wire_dict(), use_bin_type=True)
        return self._writer.write_variable(payload)

    def _ensure_heartbeat_daemon(self) -> None:
        """Start the daemon heartbeat thread once (idempotent). The daemon keeps
        `last_heartbeat` fresh independent of the worker's main loop so a busy
        or briefly-starved main loop never trips a false heartbeat_timeout
        restart (the cascade root cause). SPEC §11.B.2 / §11.I.5."""
        if self._hb_thread is not None or self._hb_stop.is_set():
            return
        t = threading.Thread(
            target=self._heartbeat_loop,
            name=f"hb-{self._module_name}",
            daemon=True,
        )
        self._hb_thread = t
        t.start()

    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.wait(self._hb_interval):
            try:
                self.heartbeat()
            except Exception as exc:  # daemon must never die silently
                logger.debug(
                    "[module_state] heartbeat daemon '%s' failed: %s",
                    self._module_name, exc)

    def heartbeat(self) -> int:
        """Bump `last_heartbeat`, PRESERVING the current state.

        Per SPEC §11.I.5: guardian_hcl's staleness check reads this field via
        1Hz SHM poll. Critically this re-publishes with the worker's ACTUAL
        current state — it must never force "running" (a heartbeat during
        boot/probe/unhealthy must not falsely advance the state machine). If
        no state has been written yet, this is a no-op.
        """
        with self._write_lock:
            if self._current_state is None:
                return -1
            self._last_heartbeat = time.time()
            return self._publish_locked(self._current_state)

    def close(self) -> None:
        """Stop the heartbeat daemon + release the mmap. Idempotent."""
        self._hb_stop.set()
        t = self._hb_thread
        if (t is not None and t.is_alive()
                and t is not threading.current_thread()):
            t.join(timeout=2.0)
        try:
            self._writer.close()
        except Exception:
            pass


# ── Cross-process reader (titan_hcl / guardian_hcl / observatory / api) ───────

class ModuleStateReader:
    """Read-only view of a per-module SHM state slot.

    Holds an mmap to the writer-owned slot. `read()` returns the latest
    ModuleStateEntry, or None if the slot is empty / corrupt / never-written.

    A single ModuleStateReader instance is safe to share across threads of
    the SAME process (the underlying StateRegistryReader handles SeqLock
    retry). Cross-process: each process opens its own reader.
    """

    def __init__(
        self,
        *,
        module_name: str,
        titan_id: Optional[str] = None,
    ) -> None:
        self._module_name = module_name
        shm_root: Path = resolve_shm_root(titan_id)
        self._spec = make_module_state_registry_spec(module_name)
        self._reader = StateRegistryReader(self._spec, shm_root)

    @property
    def module_name(self) -> str:
        return self._module_name

    def read(self) -> Optional[ModuleStateEntry]:
        """Return latest ModuleStateEntry, or None if the slot is uninitialized.

        Idempotent + safe to call at high frequency. Per locked D1: typical
        consumer cadence is 1 Hz.
        """
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
            return ModuleStateEntry.from_wire_dict(d)
        except Exception:
            return None

    def close(self) -> None:
        try:
            self._reader.close()
        except Exception:
            pass


# ── Fleet-wide reader bank (titan_hcl orchestrator + guardian_hcl supervisor) ─

class ModuleStateReaderBank:
    """One `ModuleStateReader` per registered module name.

    Maintained by titan_hcl + guardian_hcl + observatory. Lazy-creates a
    reader on first request for a module name.
    """

    def __init__(self, *, titan_id: Optional[str] = None) -> None:
        self._titan_id = titan_id
        self._readers: dict[str, ModuleStateReader] = {}

    def reader_for(self, module_name: str) -> ModuleStateReader:
        if module_name not in self._readers:
            self._readers[module_name] = ModuleStateReader(
                module_name=module_name, titan_id=self._titan_id,
            )
        return self._readers[module_name]

    def read(self, module_name: str) -> Optional[ModuleStateEntry]:
        return self.reader_for(module_name).read()

    def read_all(self, module_names: list[str]) -> dict[str, Optional[ModuleStateEntry]]:
        """Snapshot every registered module's state in one pass.

        Used by titan_hcl's 1Hz SHM poll loop + observatory's /v6/readiness.
        """
        return {name: self.read(name) for name in module_names}

    def close(self) -> None:
        for r in self._readers.values():
            r.close()
        self._readers.clear()
