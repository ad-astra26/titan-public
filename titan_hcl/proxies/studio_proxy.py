"""
StudioProxy — G18+G19+G20-compliant client for studio_worker (v1.8.3 / D-SPEC-57).

Per `rFP_titan_hcl_l2_separation_strategy.md §4.K` + Maker greenlight 2026-05-15
(Q1-Q4). Replaces direct `StudioCoordinator` access in `titan_HCL._wire_studio`.

Three call patterns:

  1. Fire-and-forget request_* methods (returns request_id only):
       request_meditation_art(...) → str
       request_epoch_bundle(...)   → str
       request_eureka(...)         → str
     Caller publishes STUDIO_RENDER_REQUEST and ignores the eventual
     STUDIO_RENDER_COMPLETED. Use when the result is not needed (currently
     no production callsite — all 5 await the path).

  2. Awaiting `*_with_completion` methods (D-SPEC-46 Future-registry,
     race-free per the memory_proxy precedent):
       generate_meditation_art_with_completion(timeout=15.0)
       generate_epoch_bundle_with_completion(timeout=60.0)
       express_eureka_with_completion(timeout=30.0)
     Each registers a Future in `_RenderCompletionRegistry` BEFORE publishing
     STUDIO_RENDER_REQUEST, then awaits the matching STUDIO_RENDER_COMPLETED
     broadcast. Returns the completion payload dict
       {request_id, type, success, paths, haiku_text, error, ts}.
     On timeout: cancels Future + returns {success: False, error: "timeout", paths: {}}.

  3. Parameterized read + SHM-direct (one each):
       async get_gallery_async(category, limit) — work-RPC ≤2s (G19 strict)
       get_stats()                              — SHM-direct (G18, sub-µs)

ALL work-RPC paths in this proxy are ≤5s per G19 strict; renders use the
D-SPEC-46 event-driven pattern instead so they NEVER exceed the cap. ZERO
new `phase_c_rpc_exemptions.yaml` entries for renders. Gallery exemption
is documented in `phase_c_rpc_exemptions.yaml::work_rpc_sites:` under
`studio_proxy:` (≤2s, parameterized read).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
import uuid
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Optional

import msgpack

from titan_hcl.bus import (
    STUDIO_RENDER_COMPLETED,
    STUDIO_RENDER_REQUEST,
    DivineBus,
    make_msg,
)
from titan_hcl.core.state_registry import (
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)
from titan_hcl.guardian import Guardian
from titan_hcl.logic.studio_state_specs import (
    STUDIO_STATE_SLOT,
    STUDIO_STATE_SPEC,
)

logger = logging.getLogger(__name__)


# ── D-SPEC-46 Future-registry for STUDIO_RENDER_COMPLETED ──────────────────


class _RenderCompletionRegistry:
    """Routes STUDIO_RENDER_COMPLETED broadcasts to per-request_id Futures.

    Race-free contract: callers MUST call `register(request_id)` BEFORE
    publishing STUDIO_RENDER_REQUEST. Otherwise the COMPLETED broadcast can
    arrive before the Future is in the dict and be silently dropped.

    Mirrors memory_proxy._IngestCompletionRegistry (Phase B / D-SPEC-46)
    line-for-line. One subscription + one daemon dispatcher thread per
    proxy instance, spun up lazily on first register call.
    """

    SUBSCRIBE_NAME = "studio_render_proxy"

    def __init__(self, bus: DivineBus) -> None:
        self._bus = bus
        self._lock = threading.Lock()
        self._futures: dict[str, Future] = {}
        self._queue = None  # bus subscriber queue, allocated lazily
        self._dispatcher: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _ensure_started(self) -> None:
        with self._lock:
            if self._queue is not None:
                return
            self._queue = self._bus.subscribe(
                self.SUBSCRIBE_NAME, types=(STUDIO_RENDER_COMPLETED,))
            self._dispatcher = threading.Thread(
                target=self._dispatch_loop,
                name="studio-render-completion",
                daemon=True,
            )
            self._dispatcher.start()
            logger.info(
                "[StudioProxy] render-completion dispatcher started "
                "(subscribe='%s')", self.SUBSCRIBE_NAME)

    def register(self, request_id: str) -> Future:
        """Allocate a Future for this request_id BEFORE publishing REQUEST."""
        if not request_id:
            raise ValueError("register requires non-empty request_id")
        self._ensure_started()
        fut: Future = Future()
        with self._lock:
            if request_id in self._futures:
                raise RuntimeError(
                    f"_RenderCompletionRegistry: request_id already in-flight: "
                    f"{request_id}")
            self._futures[request_id] = fut
        return fut

    def cancel(self, request_id: str) -> None:
        """Drop the Future for a request_id (caller timed out / no longer waiting)."""
        with self._lock:
            self._futures.pop(request_id, None)

    def in_flight_count(self) -> int:
        with self._lock:
            return len(self._futures)

    def _dispatch_loop(self) -> None:
        """Daemon: read from broadcast queue, resolve matching Future."""
        while not self._stop.is_set():
            try:
                msg = self._queue.get(timeout=1.0)
            except Exception:
                continue
            if msg is None:
                break
            payload = msg.get("payload", {}) or {}
            request_id = payload.get("request_id")
            if not request_id:
                continue
            with self._lock:
                fut = self._futures.pop(request_id, None)
            if fut is None:
                # No waiter (caller fire-and-forgot, or already timed out).
                continue
            try:
                fut.set_result(payload)
            except concurrent.futures.InvalidStateError:
                pass


# ── SHM-direct reader for studio_state.bin ─────────────────────────────────


class StudioStateShmReader:
    """Sub-µs G18 SHM-direct reader for studio_state.bin (G21 sole writer is
    studio_worker). Used by `StudioProxy.get_stats()` and by Observatory
    `/v4/studio/stats` route.

    Returns the decoded msgpack dict, or a cold-boot stub if the slot is
    unavailable / undecodable (first fallback logs INFO; subsequent silent).
    """

    def __init__(self, titan_id: Optional[str] = None):
        tid = resolve_titan_id(titan_id)
        shm_root: Path = ensure_shm_root(tid)
        self._reader = StateRegistryReader(STUDIO_STATE_SPEC, shm_root)
        self._fallback_count: int = 0
        self._titan_id = tid

    def read(self) -> dict[str, Any]:
        try:
            raw = self._reader.read_variable()
        except Exception as e:
            self._track_fallback(f"read_raised:{type(e).__name__}")
            return self._cold_defaults()
        if raw is None:
            self._track_fallback("shm_unavailable")
            return self._cold_defaults()
        try:
            decoded = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            self._track_fallback(f"decode_raised:{type(e).__name__}")
            return self._cold_defaults()
        if not isinstance(decoded, dict):
            self._track_fallback(f"decode_wrong_type:{type(decoded).__name__}")
            return self._cold_defaults()
        return decoded

    def _track_fallback(self, reason: str) -> None:
        self._fallback_count += 1
        if self._fallback_count == 1:
            logger.info(
                "[StudioStateShmReader] FIRST FALLBACK slot=%s reason=%s — "
                "studio_worker may not have published yet (cold-boot)",
                STUDIO_STATE_SLOT, reason)

    @staticmethod
    def _cold_defaults() -> dict[str, Any]:
        return {
            "schema_version": STUDIO_STATE_SPEC.schema_version,
            "meditation_count": 0,
            "epoch_count": 0,
            "eureka_count": 0,
            "last_render_ts": 0.0,
            "last_render_type": "none",
            "output_root": "",
            "default_resolution": 0,
            "highres_resolution": 0,
            "nft_composite_enabled": False,
            "ts": 0.0,
        }


# ── StudioProxy ────────────────────────────────────────────────────────────


# Render-type timeouts per Q1 greenlight (D-SPEC-46 event model; these are
# upper-bound waits on STUDIO_RENDER_COMPLETED, NOT work-RPC timeouts — the
# bus.request_async G19 ≤5s rule does NOT apply to one-way + broadcast).
_TIMEOUT_MEDITATION_S = 15.0
_TIMEOUT_EPOCH_S = 60.0
_TIMEOUT_EUREKA_S = 30.0


class StudioProxy:
    """G18+G19+G20-compliant proxy for studio_worker.

    Replaces direct `StudioCoordinator` access. Same public surface for
    callers (path-returning renders) via `_with_completion` variants;
    callers that don't need the result use fire-and-forget `request_*`.
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        self._reply_queue = bus.subscribe("studio_proxy", reply_only=True)
        self._started = False

        # D-SPEC-46 Future-registry for STUDIO_RENDER_COMPLETED.
        self._completion = _RenderCompletionRegistry(bus)

        # SHM-direct reader for studio_state.bin (G18, sub-µs).
        self._shm_reader = StudioStateShmReader()

        logger.info(
            "[StudioProxy] initialized — D-SPEC-57 / SPEC §9.B v1.8.3. "
            "renders via D-SPEC-46 (STUDIO_RENDER_REQUEST → "
            "STUDIO_RENDER_COMPLETED Future-registry); get_gallery work-RPC "
            "≤2s (G19); get_stats SHM-direct (G18).")

    def _ensure_started(self) -> None:
        from ._start_safe import ensure_started_async_safe
        ready = ensure_started_async_safe(
            self._guardian, "studio", id(self),
            proxy_label="StudioProxy",
        )
        if ready:
            self._started = True

    # ── Fire-and-forget render requests ──────────────────────────────────

    def request_meditation_art(
        self, state_root: str, age_nodes: int, avg_intensity: int,
    ) -> str:
        """Publish STUDIO_RENDER_REQUEST and return request_id immediately."""
        self._ensure_started()
        request_id = uuid.uuid4().hex
        self._publish_request(request_id, "meditation", {
            "state_root": str(state_root),
            "age_nodes": int(age_nodes),
            "avg_intensity": int(avg_intensity),
        })
        return request_id

    def request_epoch_bundle(
        self, tx_signature: str, total_nodes: int,
        beliefs_strength: int, sol_balance: float,
    ) -> str:
        self._ensure_started()
        request_id = uuid.uuid4().hex
        self._publish_request(request_id, "epoch", {
            "tx_signature": str(tx_signature),
            "total_nodes": int(total_nodes),
            "beliefs_strength": int(beliefs_strength),
            "sol_balance": float(sol_balance),
        })
        return request_id

    def request_eureka(
        self, discovery_text: str, query: str, sources: list,
        state_root: str, age_nodes: int,
    ) -> str:
        self._ensure_started()
        request_id = uuid.uuid4().hex
        self._publish_request(request_id, "eureka", {
            "discovery_text": str(discovery_text),
            "query": str(query),
            "sources": list(sources or []),
            "state_root": str(state_root),
            "age_nodes": int(age_nodes),
        })
        return request_id

    def _publish_request(self, request_id: str, render_type: str,
                         args: dict) -> None:
        msg = make_msg(
            STUDIO_RENDER_REQUEST, "studio_proxy", "studio",
            {
                "request_id": request_id,
                "type": render_type,
                "args": args,
                "ts": time.time(),
            },
        )
        try:
            self._bus.publish(msg)
        except Exception as e:
            logger.warning(
                "[StudioProxy] STUDIO_RENDER_REQUEST publish raised "
                "(request_id=%s type=%s): %s", request_id, render_type, e)

    # ── Awaiting variants (D-SPEC-46 Future-registry) ────────────────────

    async def generate_meditation_art_with_completion(
        self, state_root: str, age_nodes: int, avg_intensity: int,
        timeout: float = _TIMEOUT_MEDITATION_S,
    ) -> dict:
        """Publish STUDIO_RENDER_REQUEST and await matching COMPLETED.

        Returns {request_id, type, success, paths: {art_path: str|None},
        haiku_text: None, error: str|None, ts}.
        """
        return await self._render_with_completion(
            "meditation",
            {
                "state_root": str(state_root),
                "age_nodes": int(age_nodes),
                "avg_intensity": int(avg_intensity),
            },
            timeout=timeout,
        )

    async def generate_epoch_bundle_with_completion(
        self, tx_signature: str, total_nodes: int,
        beliefs_strength: int, sol_balance: float,
        timeout: float = _TIMEOUT_EPOCH_S,
    ) -> dict:
        """Returns {..., paths: {tree_path, audio_path, composite_path}, ...}."""
        return await self._render_with_completion(
            "epoch",
            {
                "tx_signature": str(tx_signature),
                "total_nodes": int(total_nodes),
                "beliefs_strength": int(beliefs_strength),
                "sol_balance": float(sol_balance),
            },
            timeout=timeout,
        )

    async def express_eureka_with_completion(
        self, discovery_text: str, query: str, sources: list,
        state_root: str, age_nodes: int,
        timeout: float = _TIMEOUT_EUREKA_S,
    ) -> dict:
        """Returns {..., paths: {neural_map_path, pulse_path}, haiku_text: str|None, ...}."""
        return await self._render_with_completion(
            "eureka",
            {
                "discovery_text": str(discovery_text),
                "query": str(query),
                "sources": list(sources or []),
                "state_root": str(state_root),
                "age_nodes": int(age_nodes),
            },
            timeout=timeout,
        )

    async def _render_with_completion(
        self, render_type: str, args: dict, timeout: float,
    ) -> dict:
        """Race-free: register Future BEFORE publishing REQUEST so a
        COMPLETED arriving before the registration can never orphan.
        """
        self._ensure_started()
        request_id = uuid.uuid4().hex
        fut = self._completion.register(request_id)
        msg = make_msg(
            STUDIO_RENDER_REQUEST, "studio_proxy", "studio",
            {
                "request_id": request_id,
                "type": render_type,
                "args": args,
                "ts": time.time(),
            },
        )
        try:
            self._bus.publish(msg)
        except Exception as e:
            self._completion.cancel(request_id)
            logger.warning(
                "[StudioProxy] _render_with_completion publish raised "
                "(request_id=%s type=%s): %s", request_id, render_type, e)
            return {
                "request_id": request_id,
                "type": render_type,
                "success": False,
                "paths": {},
                "haiku_text": None,
                "error": str(e),
                "ts": time.time(),
            }
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None, lambda: fut.result(timeout=timeout))
        except concurrent.futures.TimeoutError:
            self._completion.cancel(request_id)
            logger.warning(
                "[StudioProxy] _render_with_completion timed out "
                "(request_id=%s type=%s timeout=%.1fs)",
                request_id, render_type, timeout)
            return {
                "request_id": request_id,
                "type": render_type,
                "success": False,
                "paths": {},
                "haiku_text": None,
                "error": f"timeout after {timeout}s",
                "ts": time.time(),
            }

    # ── Parameterized read (work-RPC ≤2s, G19 strict) ─────────────────────

    async def get_gallery_async(
        self, category: str = "all", limit: int = 20,
    ) -> list[dict]:
        """Fetch recent artifacts via bus.QUERY work-RPC. G19-compliant (≤2s).
        Allowlisted in `titan-docs/specs/phase_c_rpc_exemptions.yaml` under
        `studio_proxy:`. Returns [] on timeout / error.
        """
        self._ensure_started()
        try:
            reply = await self._bus.request_async(
                "studio_proxy", "studio",
                {"action": "get_gallery",
                 "category": str(category), "limit": int(limit)},
                2.0, self._reply_queue,
            )
        except Exception as e:
            logger.warning(
                "[StudioProxy] get_gallery_async raised: %s", e)
            return []
        if reply:
            return reply.get("payload", {}).get("items", []) or []
        logger.warning("[StudioProxy] get_gallery_async timed out (2s G19 cap)")
        return []

    # ── SHM-direct sync (G18, sub-µs) ─────────────────────────────────────

    def get_stats(self) -> dict:
        """Return studio statistics via sub-µs SHM-direct read.

        Replaces today's `StudioCoordinator.get_stats()` (in-parent dir scan).
        Same payload shape with the addition of `last_render_ts` +
        `last_render_type` + `ts` for freshness checks.
        """
        return self._shm_reader.read()

    # ── Gallery sync wrapper (backwards-compat for non-async callers) ─────

    def get_gallery(self, category: str = "all", limit: int = 20) -> list[dict]:
        """Sync wrapper around `get_gallery_async`. Used by FastAPI sync
        handlers + tests. Prefer `get_gallery_async` from coroutines.

        Implementation: bounded run via asyncio.run if no loop, else raises
        a helpful error directing the caller to the async variant.
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "StudioProxy.get_gallery() called from async context — "
                "use `await proxy.get_gallery_async(...)` instead. "
                "(sync variant exists for FastAPI sync handlers + tests.)")
        except RuntimeError as e:
            if "no running event loop" not in str(e):
                raise
        return asyncio.run(self.get_gallery_async(category=category, limit=limit))
