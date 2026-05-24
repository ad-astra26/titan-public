"""SHM-based module-ready state for cross-process liveness checks.

D-SPEC-123 follow-up (2026-05-23, Option B per Maker decision). Replaces
the tactical guardian=None tolerance in proxies/_start_safe.py with a
proper Phase-C-canonical liveness mechanism: Guardian publishes the
full `{module_name: state}` snapshot to `module_ready.bin` SHM every
1s; cross-process readers (MemoryProxy / SocialGraphProxy / RLProxy
constructed without a Guardian reference in agno_worker_plugin) consult
this slot via `ModuleReadyShmReader.is_running(name)` instead of
crashing on `guardian.is_running` when guardian is None.

Mirrors the BridgeRecall + synth_status.bin watermark pattern (G18 /
INV-Syn-4 — no new sync `bus.request` patterns, watermark-guarded SHM
read). Stale read tolerated: up to 1s old; acceptable for proxy
liveness checks (chat-rate, not Schumann-rate).

Payload format (variable-size, msgpack):
    {module_name: state_str, ...}
where state_str is one of {"stopped", "starting", "running",
"unhealthy", "crashed", "disabled"} — mirrors ModuleState enum values.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import msgpack
import numpy as np

from titan_hcl.core.state_registry import (
    RegistrySpec,
    StateRegistryReader,
    StateRegistryWriter,
    ensure_shm_root,
    resolve_shm_root,
    resolve_titan_id,
)

logger = logging.getLogger(__name__)

MODULE_READY_SLOT_NAME = "module_ready"
# Cap at 16 KB — current Titan has ~42 modules, each entry ~30-40 bytes
# of msgpack overhead. 16 KB leaves >300 modules of headroom.
MODULE_READY_MAX_BYTES = 16 * 1024
MODULE_READY_SCHEMA_VERSION = 1

# States that mean "module is alive enough to receive bus requests."
# UNHEALTHY is included because Guardian still routes to unhealthy modules
# (they're failing heartbeats but the process is alive); only STOPPED /
# CRASHED / DISABLED mean "do not route requests here."
_ALIVE_STATES = frozenset({"running", "starting", "unhealthy"})


def _build_spec() -> RegistrySpec:
    return RegistrySpec(
        name=MODULE_READY_SLOT_NAME,
        dtype=np.dtype(np.uint8),
        shape=(MODULE_READY_MAX_BYTES,),
        feature_flag="",
        schema_version=MODULE_READY_SCHEMA_VERSION,
        variable_size=True,
    )


class ModuleReadyShmWriter:
    """Guardian-side writer for module_ready.bin. One per Titan process
    (lives in titan_HCL/guardian_HCL); writes the snapshot every 1s.

    Snapshot source: any callable that returns a dict {name: state_str}.
    Guardian passes its own ModuleInfo registry getter so the writer has
    no direct coupling to Guardian internals beyond the dict shape.
    """

    def __init__(self, titan_id: Optional[str] = None) -> None:
        self._titan_id = resolve_titan_id(titan_id)
        self._spec = _build_spec()
        shm_root = ensure_shm_root(self._titan_id)
        self._writer = StateRegistryWriter(self._spec, shm_root)
        self._lock = threading.Lock()

    def publish(self, snapshot: dict[str, str]) -> None:
        """Atomic SHM publish of the current module-state snapshot.
        snapshot = {module_name: state_str}. Caller responsible for
        keeping under MODULE_READY_MAX_BYTES of msgpack output (room
        for hundreds of modules in the default cap).
        """
        payload = msgpack.packb(snapshot, use_bin_type=True)
        if len(payload) > MODULE_READY_MAX_BYTES:
            # Hard cap defensive — drop entries until it fits. Should
            # never trigger in practice (16 KB >> typical fleet size).
            logger.warning(
                "[ModuleReadyShmWriter] snapshot %d bytes exceeds cap "
                "%d — truncating", len(payload), MODULE_READY_MAX_BYTES)
            keep = {}
            for k, v in snapshot.items():
                trial = msgpack.packb({**keep, k: v}, use_bin_type=True)
                if len(trial) > MODULE_READY_MAX_BYTES:
                    break
                keep[k] = v
            payload = msgpack.packb(keep, use_bin_type=True)
        with self._lock:
            self._writer.write_variable(payload)

    def close(self) -> None:
        try:
            self._writer.close()
        except Exception:
            pass


class ModuleReadyShmReader:
    """Cross-process reader. Lives in any subprocess that needs to check
    whether a module is alive without holding a Guardian reference (e.g.
    MemoryProxy in agno_worker_plugin which is constructed with
    guardian=None — D-SPEC-78 context).

    Soft-fail: returns False when the slot is missing / uninitialized /
    unreadable. Callers should treat False as "don't know, assume not
    ready" — which for `_ensure_started` means "kick a start" or "skip
    the request." Either is bounded behavior.
    """

    def __init__(self, titan_id: Optional[str] = None) -> None:
        self._titan_id = resolve_titan_id(titan_id)
        self._spec = _build_spec()
        self._reader: Optional[StateRegistryReader] = None
        self._lock = threading.Lock()
        # mtime-style cache: re-decode msgpack only when the reader sees
        # a fresh version (the SHM seq counter changes on every publish).
        self._cache: dict[str, str] = {}
        self._last_seen_seq: int = -1

    def _attach(self) -> None:
        if self._reader is None:
            shm_root = resolve_shm_root(self._titan_id)
            self._reader = StateRegistryReader(self._spec, shm_root)

    def read_snapshot(self) -> dict[str, str]:
        """Return the current {module_name: state_str} snapshot, or {}
        on any failure."""
        with self._lock:
            self._attach()
            payload = self._reader.read_variable()
        if payload is None:
            return {}
        try:
            decoded = msgpack.unpackb(payload, raw=False)
            if isinstance(decoded, dict):
                return decoded
            return {}
        except Exception as exc:
            logger.debug(
                "[ModuleReadyShmReader] decode failed: %s", exc)
            return {}

    def get_state(self, module_name: str) -> Optional[str]:
        """Return module's current state string, or None if absent."""
        snap = self.read_snapshot()
        return snap.get(module_name)

    def is_running(self, module_name: str) -> bool:
        """Return True iff the module's state is one of the alive states
        (running / starting / unhealthy). Matches Guardian.is_running's
        intent: 'are bus requests routable to this module?'"""
        state = self.get_state(module_name)
        return state in _ALIVE_STATES

    def is_started(self, module_name: str) -> bool:
        """Alias for is_running — kept for parity with the getattr
        lookup at proxies/_start_safe.py:80 which prefers `is_started`
        when present."""
        return self.is_running(module_name)

    def close(self) -> None:
        with self._lock:
            if self._reader is not None:
                try:
                    self._reader.close()
                except Exception:
                    pass
                self._reader = None


# ── Process-local singleton accessor ────────────────────────────────────

_reader_singleton: Optional[ModuleReadyShmReader] = None
_reader_lock = threading.Lock()


def get_module_ready_reader(
    titan_id: Optional[str] = None,
) -> ModuleReadyShmReader:
    """Process-local singleton reader. Construct once per consumer
    process (cheap — no I/O at construction; SHM attaches lazily on
    first read)."""
    global _reader_singleton
    with _reader_lock:
        if _reader_singleton is None:
            _reader_singleton = ModuleReadyShmReader(titan_id=titan_id)
    return _reader_singleton


__all__ = [
    "MODULE_READY_SLOT_NAME",
    "MODULE_READY_MAX_BYTES",
    "MODULE_READY_SCHEMA_VERSION",
    "ModuleReadyShmWriter",
    "ModuleReadyShmReader",
    "get_module_ready_reader",
]
