"""
titan_hcl/guardian_hcl_client.py — Bus-message client replacing the in-process
Guardian attribute on TitanKernel + TitanHCL.

Phase 6 / SPEC §11.B.4 / D-SPEC-135 / v1.62.0.

INV-PROC-2 (SPEC §11.B.4):
  - guardian_hcl is a SEPARATE PROCESS spawned by titan-kernel-rs BEFORE
    titan_hcl
  - guardian_hcl is the SOLE owner of the module catalog + supervision loop
  - titan_hcl reaches Guardian's public API via this thin client

Forwards lifecycle mutations (start / stop / restart_module / reload_module)
via bus publish with dst="guardian". Status reads (is_running, get_status,
modules list) are served from a local cache populated by MODULE_READY,
MODULE_CRASHED, MODULE_SHUTDOWN, SUPERVISION_CHILD_RESTARTED events.

This is NOT a shim of the old in-process Guardian — it is the canonical
client for the new bus-message contract. Method names mirror Guardian's
public API only to minimize call-site churn at plugin.py / kernel.py.

Reload protocol (D-SPEC-50 §11.B.3):
  - reload_module() publishes MODULE_RELOAD_REQUEST with a fresh
    correlation_id and awaits MODULE_RELOAD_ACK for that id, with a
    bounded timeout (default = MODULE_RELOAD_DEFAULT_TIMEOUT_S).
  - On timeout we return the ack-not-received result; caller policy
    decides whether to retry. We do NOT call back into Guardian
    internals.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Any, Optional

from titan_hcl.bus import (
    MODULE_CRASHED,
    MODULE_READY,
    MODULE_RELOAD_ACK,
    MODULE_RELOAD_REQUEST,
    MODULE_RESTART_REQUEST,
    MODULE_SHUTDOWN,
    MODULE_START_REQUEST,
    MODULE_STOP_REQUEST,
    SUPERVISION_CHILD_DOWN,
    SUPERVISION_CHILD_RESTARTED,
    make_msg,
)
from titan_hcl._phase_c_constants import MODULE_RELOAD_DEFAULT_TIMEOUT_S

logger = logging.getLogger(__name__)


class GuardianHCLClient:
    """Thin bus-message client mirroring Guardian's public API.

    Acts as `kernel.guardian` in the titan_hcl process under Phase 6. The
    real Guardian instance lives in the guardian_hcl process (scripts/
    guardian_hcl.py).
    """

    def __init__(self, bus):
        self._bus = bus
        # State cache populated by bus events. Each value is a dict with
        # keys: state ("RUNNING" / "CRASHED" / "STOPPED"), restart_count,
        # last_event_ts.
        self._modules_cache: dict[str, dict] = {}
        # Pending reload futures keyed by correlation_id. Each value is a
        # threading.Event + a result dict populated when the ack arrives.
        self._pending_reloads: dict[str, tuple[threading.Event, dict]] = {}
        self._pending_lock = threading.Lock()
        # Subscribe to status-population events. The bus.subscribe API
        # returns a queue; for our event-driven cache we want a callback
        # pattern instead. We attach a small dispatcher thread.
        self._stop_event = threading.Event()
        self._cache_queue = self._bus.subscribe(
            "guardian_hcl_client_cache",
            types=[
                MODULE_READY, MODULE_CRASHED, MODULE_SHUTDOWN,
                SUPERVISION_CHILD_RESTARTED, SUPERVISION_CHILD_DOWN,
                MODULE_RELOAD_ACK,
            ],
            reply_only=True,
        )
        self._dispatcher_thread = threading.Thread(
            target=self._dispatch_loop,
            name="guardian-hcl-client-dispatcher",
            daemon=True,
        )
        self._dispatcher_thread.start()

    # ─────────────── Cache event handling ───────────────

    def _dispatch_loop(self) -> None:
        from queue import Empty
        while not self._stop_event.is_set():
            try:
                msg = self._cache_queue.get(timeout=0.5)
            except Empty:
                continue
            except Exception:
                continue
            try:
                self._handle_event(msg)
            except Exception as e:  # noqa: BLE001
                logger.debug("[GuardianHCLClient] event handler error: %s", e)

    def _handle_event(self, msg: dict) -> None:
        mtype = msg.get("type")
        payload = msg.get("payload", {}) or {}
        name = payload.get("name") or payload.get("module")

        if mtype == MODULE_RELOAD_ACK:
            # Ack-routing: payload includes correlation_id (echo from
            # original MODULE_RELOAD_REQUEST). Wake the waiting future.
            corr_id = payload.get("correlation_id") or msg.get("correlation_id")
            if corr_id:
                with self._pending_lock:
                    entry = self._pending_reloads.pop(corr_id, None)
                if entry is not None:
                    event, result_holder = entry
                    result_holder.update(payload)
                    event.set()
            return

        if not name:
            return

        if mtype == MODULE_READY:
            slot = self._modules_cache.setdefault(name, {})
            slot["state"] = "RUNNING"
            slot["last_event_ts"] = time.time()
        elif mtype == MODULE_CRASHED:
            slot = self._modules_cache.setdefault(name, {})
            slot["state"] = "CRASHED"
            slot["last_event_ts"] = time.time()
        elif mtype == MODULE_SHUTDOWN:
            slot = self._modules_cache.setdefault(name, {})
            slot["state"] = "STOPPED"
            slot["last_event_ts"] = time.time()
        elif mtype == SUPERVISION_CHILD_RESTARTED:
            slot = self._modules_cache.setdefault(name, {})
            slot["restart_count"] = slot.get("restart_count", 0) + 1
            slot["last_event_ts"] = time.time()
        elif mtype == SUPERVISION_CHILD_DOWN:
            slot = self._modules_cache.setdefault(name, {})
            slot["state"] = "CRASHED"
            slot["last_event_ts"] = time.time()

    # ─────────────── Guardian API surface (thin) ───────────────

    def register(self, spec) -> None:
        """Catalog ownership moved to guardian_hcl process — registers are no-ops
        from the plugin side. Raises so any straggler call is found loudly.

        Phase 6 cutover: the canonical catalog lives in
        titan_hcl/module_catalog.py and is invoked exclusively by
        scripts/guardian_hcl.py.
        """
        raise RuntimeError(
            "Guardian.register() is not callable from the titan_hcl process "
            "under Phase 6. The module catalog is owned by guardian_hcl "
            "(see titan_hcl/module_catalog.py:build_catalog). "
            f"Attempted spec name={getattr(spec, 'name', '?')!r}.")

    def start(self, name: str) -> bool:
        self._bus.publish(make_msg(
            MODULE_START_REQUEST, src="titan_hcl", dst="guardian_hcl_lifecycle",
            payload={"name": name},
        ))
        return True

    def stop(self, name: str, reason: str = "requested") -> bool:
        self._bus.publish(make_msg(
            MODULE_STOP_REQUEST, src="titan_hcl", dst="guardian_hcl_lifecycle",
            payload={"name": name, "reason": reason},
        ))
        return True

    def restart_module(self, name: str, reason: str = "requested", **kwargs) -> bool:
        payload = {"name": name, "reason": reason}
        # msgpack-safe kwargs only (primitives + nested dicts/lists/strings)
        for k, v in kwargs.items():
            if isinstance(v, (str, int, float, bool, type(None), dict, list, tuple)):
                payload[k] = v
        self._bus.publish(make_msg(
            MODULE_RESTART_REQUEST, src="titan_hcl", dst="guardian_hcl_lifecycle",
            payload=payload,
        ))
        return True

    async def reload_module(self, name: str,
                            timeout: float = MODULE_RELOAD_DEFAULT_TIMEOUT_S,
                            **kwargs) -> dict:
        """Publish MODULE_RELOAD_REQUEST + await MODULE_RELOAD_ACK ≤ timeout.

        Returns ack payload dict on success; on timeout returns
        {"status": "timeout", "name": name, "elapsed_s": <real_elapsed>}.
        Caller policy (typically dashboard.reload_api or arch_map reload)
        decides retry — we do NOT call back into Guardian internals.
        """
        corr_id = uuid.uuid4().hex
        event = threading.Event()
        result_holder: dict = {}
        with self._pending_lock:
            self._pending_reloads[corr_id] = (event, result_holder)

        payload = {"name": name, "correlation_id": corr_id}
        for k, v in kwargs.items():
            if isinstance(v, (str, int, float, bool, type(None), dict, list, tuple)):
                payload[k] = v

        msg = make_msg(
            MODULE_RELOAD_REQUEST, src="titan_hcl", dst="guardian",
            payload=payload,
        )
        # Some make_msg variants don't propagate correlation_id at the top
        # level — set it explicitly so the broker / consumers can see it
        # either way.
        msg["correlation_id"] = corr_id
        self._bus.publish(msg)

        t0 = time.time()
        ok = await asyncio.get_event_loop().run_in_executor(
            None, lambda: event.wait(timeout=timeout))
        elapsed = time.time() - t0
        if ok:
            return result_holder
        # Timeout — clean up the pending entry.
        with self._pending_lock:
            self._pending_reloads.pop(corr_id, None)
        return {"status": "timeout", "name": name, "elapsed_s": elapsed}

    def stop_all(self, reason: str = "shutdown") -> None:
        """Broadcast a stop request to guardian_hcl.

        Used by the kernel + plugin shutdown paths. guardian_hcl receives
        and walks its module dict; this client does not need to know the
        catalog content.
        """
        self._bus.publish(make_msg(
            MODULE_STOP_REQUEST, src="titan_hcl", dst="guardian_hcl_lifecycle",
            payload={"name": "__all__", "reason": reason},
        ))

    def start_all(self) -> None:
        """No-op from plugin side — guardian_hcl boots autostart modules at
        its own start (see scripts/guardian_hcl.py:run). Provided so legacy
        callers continue to work without raising."""
        return

    # ─────────────── Read paths (cache-served) ───────────────

    def is_running(self, name: str) -> bool:
        return self._modules_cache.get(name, {}).get("state") == "RUNNING"

    def get_status(self) -> dict:
        # Mirror the relevant subset of Guardian.get_status() — guardian_hcl's
        # full status (including restart-window stats) is published into the
        # guardian_state.bin SHM slot and api_subprocess reads that directly.
        return {
            "modules": {
                name: {
                    "state": info.get("state", "UNKNOWN"),
                    "restart_count": info.get("restart_count", 0),
                }
                for name, info in self._modules_cache.items()
            },
            "from_cache": True,
            "source": "GuardianHCLClient (Phase 6)",
        }

    def get_modules_by_layer(self) -> dict[str, list[str]]:
        # Layers are static per-spec metadata; the catalog lives in
        # guardian_hcl. The client cache only tracks runtime state, not
        # ModuleSpec.layer. Callers that need this should read
        # guardian_state.bin SHM (G18). Return empty dict so legacy callers
        # don't crash; api_subprocess uses the SHM path.
        return {}

    def layer_stats(self) -> dict:
        return {}

    def enable(self, name: str) -> bool:
        """Enable a previously-disabled module. Forwards to guardian_hcl
        via a START_REQUEST (Guardian.enable() reinitializes the slot then
        calls start()). guardian_hcl process handles the enable+start
        sequence in its existing Guardian.start() implementation."""
        return self.start(name)

    @property
    def _modules(self) -> dict:
        """Snapshot of the module-state cache. Used by plugin.py for
        status logging; mutation is not supported (event-driven)."""
        return dict(self._modules_cache)

    def drain_send_queues(self) -> int:
        """No-op. Under Phase 6, worker send queues live in guardian_hcl
        process; the kernel-rs broker fans messages by name. The bus
        _poll_fn historically called this; we keep it as a no-op so any
        residual wiring stays safe."""
        return 0

    def monitor_tick(self) -> None:
        """No-op. Supervision loop runs in guardian_hcl process."""
        return

    # ─────────────── Cleanup ───────────────

    def shutdown(self) -> None:
        self._stop_event.set()
