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

Reload protocol (D-SPEC-50 §11.B.3 + RFP_shm_native_hot_reload.md Phase A):
  - reload_module() publishes MODULE_RELOAD_REQUEST carrying a fresh
    swap_id, then polls the orchestrator-owned `reload_status.bin` SHM slot
    for that swap_id reaching a terminal status (ready/failed/rolled_back),
    bounded by MODULE_RELOAD_DEFAULT_TIMEOUT_S.
  - The terminal status is read from SHM (Preamble G18), NOT a bus
    MODULE_RELOAD_ACK reply — the old `dst="all"` broadcast was a SPEC §8.2 /
    G18 violation and is deleted. On timeout we return status="timeout";
    caller policy decides whether to retry. We do NOT call back into Guardian
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
    MODULE_HEARTBEAT,
    MODULE_READY,
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
        # Phase 6 / D-SPEC-135: PRIMARY source of module state is the
        # guardian_state.bin SHM slot written by guardian_hcl process at
        # 1 Hz (G21 single-writer per SPEC §7.1 + D-SPEC-70). Any reader
        # in any process gets fresh data via mmap. The event cache below
        # is a SECONDARY local-bus subscriber that mirrors targeted
        # MODULE_READY etc. when they fan to titan_HCL's "guardian"
        # alias — but the broker may route dst="guardian" exclusively to
        # guardian_hcl's PRIMARY subscriber (name-match priority), leaving
        # the alias starved. SHM is the canonical G18 read path; the
        # event cache is a defense-in-depth fallback for the cold-boot
        # window before the publisher writes guardian_state.bin.
        self._shm_reader = None
        try:
            from titan_hcl.api.shm_reader_bank import ShmReaderBank
            from titan_hcl.core.state_registry import resolve_titan_id
            self._shm_reader = ShmReaderBank(titan_id=resolve_titan_id())
        except Exception as _shm_init_err:  # noqa: BLE001
            logger.debug(
                "[GuardianHCLClient] SHM reader init deferred (test "
                "harness or pre-boot): %s — falling back to event cache",
                _shm_init_err)
        # State cache populated by bus events. Each value is a dict with
        # keys: state, restart_count, last_event_ts.
        self._modules_cache: dict[str, dict] = {}
        # SHM-native reload (RFP_shm_native_hot_reload.md Phase A): reload
        # results are READ from the orchestrator-owned `reload_status.bin` SHM
        # slot (G18), not awaited as a bus MODULE_RELOAD_ACK. Lazily built on
        # first reload (None pre-boot / in test fixtures without /dev/shm).
        self._reload_status_reader = None
        # Subscribe to status-population events. The bus.subscribe API
        # returns a queue; for our event-driven cache we want a callback
        # pattern instead. We attach a small dispatcher thread.
        self._stop_event = threading.Event()
        # Subscribe under name "guardian" so dst="guardian" targeted messages
        # (MODULE_HEARTBEAT, MODULE_READY — workers always publish these
        # dst="guardian") route to this subscriber via name match. (Reload
        # status is NO LONGER a bus event — it's read from reload_status.bin
        # SHM per RFP_shm_native_hot_reload.md Phase A.) Pre-Phase-6 the
        # in-process Guardian held this name;
        # under Phase 6 Guardian lives in a separate process — this client
        # is the canonical "guardian" subscriber inside the titan_hcl
        # process. reply_only=False so broadcast dst="all" events like
        # SUPERVISION_CHILD_RESTARTED also reach the cache.
        self._cache_queue = self._bus.subscribe(
            "guardian",
            types=[
                MODULE_READY, MODULE_HEARTBEAT, MODULE_CRASHED, MODULE_SHUTDOWN,
                SUPERVISION_CHILD_RESTARTED, SUPERVISION_CHILD_DOWN,
            ],
            reply_only=False,
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

        if not name:
            return

        # State values match legacy Guardian.get_status() — ModuleState enum
        # .value emits lowercase ("running" / "stopped" / "starting" /
        # "unhealthy"). dashboard.py:2182 compares against "running" so the
        # case must match for /health to count modules ACTIVE.
        if mtype == MODULE_READY:
            slot = self._modules_cache.setdefault(name, {})
            slot["state"] = "running"
            slot["last_event_ts"] = time.time()
        elif mtype == MODULE_HEARTBEAT:
            # Any heartbeat proves the worker is RUNNING regardless of
            # whether we caught the (one-shot) MODULE_READY event during
            # boot. Closes the cache-cold-boot race where modules booted
            # before the cache subscriber attached.
            slot = self._modules_cache.setdefault(name, {})
            slot["state"] = "running"
            slot["last_event_ts"] = time.time()
        elif mtype == MODULE_CRASHED:
            slot = self._modules_cache.setdefault(name, {})
            slot["state"] = "crashed"
            slot["last_event_ts"] = time.time()
        elif mtype == MODULE_SHUTDOWN:
            slot = self._modules_cache.setdefault(name, {})
            slot["state"] = "stopped"
            slot["last_event_ts"] = time.time()
        elif mtype == SUPERVISION_CHILD_RESTARTED:
            slot = self._modules_cache.setdefault(name, {})
            slot["restart_count"] = slot.get("restart_count", 0) + 1
            slot["last_event_ts"] = time.time()
        elif mtype == SUPERVISION_CHILD_DOWN:
            slot = self._modules_cache.setdefault(name, {})
            slot["state"] = "crashed"
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
        """Publish MODULE_RELOAD_REQUEST + poll the reload_status SHM slot.

        SHM-native (RFP_shm_native_hot_reload.md Phase A): we send a fresh
        `swap_id` on the request, then poll the orchestrator-owned
        `reload_status.bin` slot for that swap_id reaching a TERMINAL status
        (ready/failed/rolled_back) — the result is read from SHM (G18), NOT a
        bus reply (the old `dst="all"` MODULE_RELOAD_ACK was a §8.2/G18
        violation, deleted). Returns the entry's result dict on a terminal
        status; on timeout returns
        {"status": "timeout", "name": name, "elapsed_s": <real_elapsed>}.
        Caller policy (dashboard.reload_api / arch_map reload) decides retry.
        """
        swap_id = uuid.uuid4().hex

        # D-SPEC-151: the reload orchestrator (reload.py:_dispatch_reload_request)
        # reads payload["module_name"] + honors payload["swap_id"] (reload.py:605),
        # then stamps that SAME swap_id into every reload_status.bin entry — which
        # is how this poll correlates the result.
        payload = {"module_name": name, "swap_id": swap_id}
        for k, v in kwargs.items():
            if isinstance(v, (str, int, float, bool, type(None), dict, list, tuple)):
                payload[k] = v

        self._bus.publish(make_msg(
            # reload EXECUTES in titan_hcl (real Orchestrator) via the lifecycle
            # subscriber — NOT guardian_hcl's metadata-only orch.
            MODULE_RELOAD_REQUEST, src="titan_hcl", dst="guardian_hcl_lifecycle",
            payload=payload,
        ))

        reader = self._ensure_reload_status_reader()
        t0 = time.time()
        deadline = t0 + timeout

        def _poll_terminal() -> Optional[dict]:
            # Poll the SHM ring for OUR swap_id reaching a terminal status.
            while time.time() < deadline:
                if reader is not None:
                    try:
                        entry = reader.read_swap(name, swap_id)
                    except Exception:  # noqa: BLE001
                        entry = None
                    if entry is not None and entry.is_terminal:
                        return entry.as_result_dict()
                time.sleep(0.1)
            return None

        result = await asyncio.get_event_loop().run_in_executor(
            None, _poll_terminal)
        if result is not None:
            return result
        return {"status": "timeout", "name": name,
                "elapsed_s": time.time() - t0}

    def _ensure_reload_status_reader(self):
        """Lazy-construct the reload_status.bin reader (RFP Phase A).

        Returns None if SHM can't be resolved yet (pre-boot / test fixture
        without /dev/shm provisioned) — reload_module then bounded-polls to a
        `timeout` result. Re-attempts on the next call so it recovers once the
        orchestrator has provisioned the slot.
        """
        existing = getattr(self, "_reload_status_reader", None)
        if existing is not None:
            return existing
        try:
            from titan_hcl.core.reload_status import ReloadStatusReader
            from titan_hcl.core.state_registry import resolve_titan_id
            self._reload_status_reader = ReloadStatusReader(
                titan_id=resolve_titan_id())
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "[GuardianHCLClient] reload_status reader init deferred "
                "(pre-boot / test harness): %s", e)
            self._reload_status_reader = None
        return self._reload_status_reader

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
        # Match legacy Guardian.is_running which reads ModuleState.RUNNING.value
        # = "running" (lowercase). dashboard.py:2182 also tests against
        # "running" — keep the case consistent fleet-wide.
        return self._modules_cache.get(name, {}).get("state") == "running"

    def get_status(self) -> dict:
        """Return `{module_name: {state, pid, rss_mb, uptime, layer, ...}}`
        matching the legacy in-process Guardian.get_status() shape.

        Phase 6 / D-SPEC-135 / G18: PRIMARY source is the guardian_state.bin
        SHM slot written by guardian_hcl process at 1 Hz. This is the
        canonical cross-process state read path — same payload shape as
        legacy Guardian.get_status (including pid, rss_mb, uptime, layer,
        restart_count, etc., all populated from the real Guardian._modules
        snapshot). FALLBACK is the local event cache (which mirrors a
        subset of fields if dst="guardian" broadcasts reach titan_HCL's
        alias subscriber). The fallback is only meaningful during the
        cold-boot window before guardian_hcl's first publish or in
        unit-test harnesses that don't initialize SHM.

        dashboard.py:2180-2184 iterates this dict and calls .get("state")
        on each value — the shape MUST be a flat `{name: info}` map (no
        wrapping under "modules" key, no sibling bools).
        """
        # PRIMARY — SHM (guardian_state.bin) G18 path
        if self._shm_reader is not None:
            try:
                payload = self._shm_reader.read_guardian_state() or {}
                modules = payload.get("modules") or {}
                if modules:
                    return dict(modules)
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "[GuardianHCLClient] SHM read failed, falling back to "
                    "event cache: %s", e)
        # FALLBACK — event-driven local cache. Only populated if the
        # local bus delivers MODULE_READY/HEARTBEAT events from broker
        # fan-out (rare under name-exclusive routing — the SHM path above
        # is the canonical one).
        return {
            name: {
                "state": info.get("state", "stopped"),
                "restart_count": info.get("restart_count", 0),
                # Legacy fields — not tracked in cache; consumers .get() them.
                "pid": info.get("pid", 0),
                "rss_mb": info.get("rss_mb", 0.0),
                "uptime": info.get("uptime", 0),
                "restarts_in_window": info.get("restarts_in_window", 0),
                "last_heartbeat_age": info.get("last_heartbeat_age", -1),
                "layer": info.get("layer", "L?"),
                "start_method": info.get("start_method", "fork"),
                "adopted": info.get("adopted", False),
                "adopt_ts": info.get("adopt_ts", 0.0),
            }
            for name, info in self._modules_cache.items()
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
