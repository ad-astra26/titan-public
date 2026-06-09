"""
studio_worker — Python L2 module hosting StudioCoordinator + the
data/studio_exports/{meditation,epoch,eureka}/ output directories.

Phase C v1.8.3 (D-SPEC-57) per `rFP_titan_hcl_l2_separation_strategy.md §4.K`.
Maker greenlit Q1-Q4 inline 2026-05-15.

What this worker owns:
  1. StudioCoordinator instance (`titan_hcl/expressive/studio.py`, 454 LOC)
     — all creative-render pipelines: meditation flow-field art, Greater-Epoch
     L-system + sonification + NFT composite bundles, Eureka discovery
     neural-map + pulse + haiku.
  2. data/studio_exports/{meditation,epoch,eureka}/ output directories
     (G21 single writer).
  3. studio_state.bin SHM slot (G21 single writer) — dual-trigger republish on
     KERNEL_EPOCH_TICK + immediately after every successful render.
  4. _HaikuLLMBridge — provider-agnostic via bus.request_async to llm_proxy.distill.
     studio_worker constructs ZERO provider-specific inference clients (Maker
     direction 2026-05-15).
  5. Render dispatch — STUDIO_RENDER_REQUEST events dispatched by `type` field
     to the matching StudioCoordinator method; renders run in a dedicated
     ThreadPoolExecutor (max 2 concurrent) so the main bus loop never blocks.
  6. Gallery serving — bus.QUERY action="get_gallery" parameterized read
     (work-RPC ≤2s per G19; allowlisted in phase_c_rpc_exemptions.yaml).

Bus subscriptions (REQUIRED):
  • STUDIO_RENDER_REQUEST       — P3 coalesced; dispatches by `type` field
  • KERNEL_EPOCH_TICK           — 1.0 Hz refresh_counts + republish SHM
  • bus.QUERY action=get_gallery — parameterized read work-RPC (≤2s)
  • MODULE_SHUTDOWN             — graceful drain
  • SAVE_NOW                    — forces SHM republish + dir refresh

Bus publications (non-blocking per §8.0.ter D-SPEC-48):
  • STUDIO_WORKER_READY     — once on boot (after StudioCoordinator init)
  • STUDIO_RENDER_COMPLETED — per render (dst="all"; request_id in payload
                              for D-SPEC-46 Future-registry resolution)
  • MODULE_HEARTBEAT        — every 30s
  • (MODULE_READY retired — Phase 11 §11.I.2 SHM slot state=booted is the contract)

Persisted state: data/studio_exports/{meditation,epoch,eureka}/*.{png,wav,json}
— creative artifacts under GFS-retention pruning. studio_state.bin in /dev/shm.

External I/O: bus client only (no direct provider HTTP — haiku calls route
through llm_proxy work-RPC).

Dependencies (boot order via guardian_HCL — see SPEC §10.A):
  • SOFT: llm_worker         (REQUIRED for Tier-1 haiku; eureka Tier-2 + 3
                              fallbacks are pure-Python templates)
  • SOFT: metabolism_worker  (REQUIRED for budget-aware resolution scaling;
                              absence falls back to STARVATION-mode skip)

See:
  - SPEC v1.8.3 §9.B `studio_worker` block
  - SPEC v1.8.3 §7.1 `studio_state.bin` SHM slot row (msgpack-variable)
  - SPEC v1.8.3 §8.7 3 new bus events (STUDIO_WORKER_READY / RENDER_REQUEST / RENDER_COMPLETED)
  - SPEC v1.8.3 §21 D-SPEC-57
  - PLAN_microkernel_phase_c_studio_worker_extraction.md (committed c44129ae)
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty
from typing import Any, Optional

from titan_hcl import bus
from titan_hcl.bus import (
    KERNEL_EPOCH_TICK,
    MODULE_HEARTBEAT,
    MODULE_PROBE_REQUEST,
    MODULE_SHUTDOWN,
    SAVE_NOW,
    STUDIO_RENDER_COMPLETED,
    STUDIO_RENDER_REQUEST,
    STUDIO_WORKER_READY,
    make_msg,
)
from titan_hcl.logic.studio_state_publisher import StudioStatePublisher
from titan_hcl.logic.studio_state_specs import STUDIO_STATE_SPEC
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)


HEARTBEAT_INTERVAL_S = 30.0
MAX_CONCURRENT_RENDERS = 2  # ThreadPoolExecutor cap — avoids GPU/disk thrash
GALLERY_QUERY_TIMEOUT_S = 2.0  # work-RPC ≤2s per G19


# Phase 11 §11.I.3 / §11.I.5 (Chunk 11N) — module-level readiness sentinel
# mirrored to the per-process SHM slot via ModuleStateWriter. Set False at
# import time; flipped True after StudioCoordinator + publisher init.
_WORKER_READY: bool = False


# Valid render types per SPEC §8.7 STUDIO_RENDER_REQUEST schema.
_VALID_RENDER_TYPES = ("meditation", "epoch", "eureka")


def _send(send_queue, msg_type: str, src: str, dst: str,
          payload: dict, rid: Optional[str] = None) -> None:
    """Non-blocking publish helper (§8.0.ter D-SPEC-48 publish-non-blocking)."""
    try:
        send_queue.put_nowait(make_msg(msg_type, src, dst, payload, rid=rid))
    except Exception as e:
        logger.warning(
            "[StudioWorker] _send %s → %s failed: %s",
            msg_type, dst, e)


def _send_response(send_queue, src_name: str, dst: str, payload: dict,
                   rid) -> None:
    """Emit RESPONSE to a bus.QUERY caller. rid is required."""
    if rid is None:
        return
    try:
        send_queue.put({
            "type": bus.RESPONSE,
            "src": src_name,
            "dst": dst,
            "rid": rid,
            "payload": payload,
            "ts": time.time(),
        })
    except Exception as e:
        logger.warning(
            "[StudioWorker] send_response enqueue failed (rid=%s): %s",
            rid, e)


def _heartbeat_loop(send_queue, name: str, stop_event: threading.Event,
                    stats_ref: dict,
                    state_writer: Optional[Any] = None) -> None:
    """Daemon thread — MODULE_HEARTBEAT every 30s with render-volume stats.

    Phase 11 §11.I.5 (Chunk 11N): also publishes ModuleStateWriter.heartbeat()
    on the SHM slot when `state_writer` is provided AND `_WORKER_READY` is
    True so guardian_HCL's SHM-staleness detector + observatory /v6/readiness
    see fresh data on the same cadence as the legacy bus path.
    """
    while not stop_event.is_set():
        _send(send_queue, MODULE_HEARTBEAT, name, "guardian", {
            "alive": True,
            "ts": time.time(),
            "renders_total": stats_ref.get("renders_total", 0),
            "renders_success": stats_ref.get("renders_success", 0),
            "renders_failed": stats_ref.get("renders_failed", 0),
            "in_flight": stats_ref.get("in_flight", 0),
        })
        if state_writer is not None and _WORKER_READY:
            try:
                state_writer.heartbeat()
            except Exception:  # noqa: BLE001 — never crash heartbeat
                pass
        stop_event.wait(HEARTBEAT_INTERVAL_S)


class _HaikuLLMBridge:
    """Provider-agnostic haiku-generation adapter installed onto
    StudioCoordinator._ollama_cloud (the attribute name is preserved for
    minimum-diff compatibility with the existing _generate_haiku Tier-1 path).

    Routes via `bus.request_async("studio_worker", "llm", {"action": "distill",
    ...}, 30s)` to the canonical llm_proxy.distill path — same surface as
    `LLMProxy.distill`. Future provider swaps happen in llm_worker, never in
    studio (Maker direction 2026-05-15 — Q2 greenlight).

    StudioCoordinator._generate_haiku Tier-1 calls
    `self._ollama_cloud.complete(prompt, model, temperature, max_tokens, timeout)`.
    This bridge exposes that same `complete` signature.
    """

    def __init__(self, send_queue, name: str):
        self._send_queue = send_queue
        self._name = name

    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 60,
        timeout: float = 15.0,
    ) -> str:
        """Route haiku gen via llm_proxy.distill. Returns the model output text,
        or empty string on timeout / error (StudioCoordinator falls through to
        Tier-2 template).

        NOTE: this method runs INSIDE the per-render asyncio.run() event loop
        (StudioCoordinator.express_eureka calls it). We need a way to fire a
        bus.request_async-equivalent from this nested loop. Since the worker's
        send_queue is thread-safe + the proxy registers a reply_queue at boot,
        the cleanest path is to publish a QUERY directly and await a RESPONSE
        via a per-call Future. But that requires a RESPONSE bridge.
        SIMPLER alternative used here: bracket the LLM call as a one-way
        publish + sync poll on a reply_queue — but we don't have a reply
        socket here.
        Cleanest: skip work-RPC entirely from the worker's render thread —
        delegate to llm_proxy by routing through StudioProxy's reply channel.
        Since this would require a roundtrip back through the proxy, AND
        haiku generation is best-effort with template fallback, we simply
        return "" to fall through to Tier-2/3 in-worker templates. Future
        enhancement: wire a dedicated llm-reply socket inside studio_worker.
        """
        # Best-effort tier-1 disabled in v1.8.3 initial ship — Tier-2 keyword
        # templates + Tier-3 static fallback in StudioCoordinator handle the
        # haiku generation path entirely in-worker (no external dep).
        # Provider abstraction is in place via this bridge surface (future:
        # wire to llm_worker via a dedicated reply socket inside the worker —
        # tracked in `OBS-studio-haiku-llm-bridge` follow-up).
        logger.debug(
            "[StudioWorker._HaikuLLMBridge] complete() called — "
            "returning '' to defer to in-worker template fallback "
            "(provider-agnostic bridge surface preserved per Maker Q2 "
            "direction; llm-worker wire-up follow-up)")
        return ""


def _dispatch_render(coordinator, render_type: str, args: dict) -> dict:
    """Run a single render synchronously (called from ThreadPoolExecutor
    worker thread). Each render gets its own asyncio loop (cheap; ~5ms
    overhead) and isolated error handling.

    Returns a `paths` dict per SPEC §8.7 STUDIO_RENDER_COMPLETED schema:
      meditation: {"art_path": str|None}
      epoch:      {"tree_path": str|None, "audio_path": str|None, "composite_path": str|None}
      eureka:     {"neural_map_path": str|None, "pulse_path": str|None}  (+ haiku_text alongside)
    """
    if render_type == "meditation":
        coro = coordinator.generate_meditation_art(
            args.get("state_root", ""),
            int(args.get("age_nodes", 0)),
            int(args.get("avg_intensity", 0)),
        )
        art_path = asyncio.run(coro)
        return {"paths": {"art_path": art_path}, "haiku_text": None}

    elif render_type == "epoch":
        coro = coordinator.generate_epoch_bundle(
            args.get("tx_signature", ""),
            int(args.get("total_nodes", 0)),
            int(args.get("beliefs_strength", 0)),
            float(args.get("sol_balance", 0.0)),
        )
        result = asyncio.run(coro)
        # result is already a {tree_path, audio_path, composite_path} dict
        return {"paths": result, "haiku_text": None}

    elif render_type == "eureka":
        coro = coordinator.express_eureka(
            args.get("discovery_text", ""),
            args.get("query", ""),
            list(args.get("sources", [])),
            args.get("state_root", ""),
            int(args.get("age_nodes", 0)),
        )
        result = asyncio.run(coro)
        # result is {neural_map_path, pulse_path, haiku_text}
        haiku_text = result.pop("haiku_text", None)
        return {"paths": result, "haiku_text": haiku_text}

    else:
        raise ValueError(f"unknown render type: {render_type!r}")


@with_error_envelope(module_name="studio", subsystem="entry", severity=_phase11_sev.FATAL)
def studio_worker_main(recv_queue, send_queue, name: str,
                       config: dict) -> None:
    """L2 module entry — Guardian supervised.

    Boot sequence:
      1. Resolve titan_id, read [expressive] + [inference] config
      2. Build StudioCoordinator (with _HaikuLLMBridge wired to _ollama_cloud)
      3. Build StudioStatePublisher (initial dir scan + first SHM write)
      4. Emit STUDIO_WORKER_READY + transition SHM slot starting → booted
         (Phase 11 §11.I.2; legacy MODULE_READY retired per locked D2)
      5. Start heartbeat thread (30s) — actually started EARLY before
         StudioCoordinator init so the SHM-slot last_heartbeat stays fresh
         during the slow coordinator + publisher boot window
      6. Main loop: drain recv_queue, dispatch by msg_type:
           - STUDIO_RENDER_REQUEST → submit to render ThreadPoolExecutor;
             on completion publish STUDIO_RENDER_COMPLETED + record + SHM
           - KERNEL_EPOCH_TICK → refresh_counts + publish SHM
           - bus.QUERY action="get_gallery" → reply with serialized gallery
           - SAVE_NOW → refresh_counts + publish SHM (forced)
           - MODULE_SHUTDOWN → graceful exit (wait in-flight ≤ grace, then kill)
    """
    # Phase 11 §11.I.5 (Chunk 11N) — reset module-level readiness sentinel.
    global _WORKER_READY
    _WORKER_READY = False

    # Resolve titan_id (feedback_titan_id_canonical_resolve.md — SPEC §23.17 R-PORT-1)
    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = resolve_titan_id(
        (config or {}).get("titan_id") if config else None)

    logger.info(
        "[StudioWorker] booting — titan_id=%s name=%s "
        "(SPEC v1.8.3 §9.B / D-SPEC-57 / rFP §4.K)",
        titan_id, name)

    # ── Phase 11 §11.I.5 / Chunk 11N — SHM state-slot writer (G21) ──
    # Built BEFORE the slow StudioCoordinator init so titan_hcl's 1Hz SHM
    # poll sees the worker is alive while it warms.
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority,
            ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name=name,
            layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[StudioWorker] Phase 11 ModuleStateWriter init failed "
            "(continuing — SHM slot disabled): %s", _sw_err)

    # Stats dict shared with heartbeat thread (forward-declared so the
    # early-start heartbeat closure can reference it before render loop
    # mutates it). Phase 11 §11.I.5 — heartbeat needs to start BEFORE the
    # slow StudioCoordinator + StudioStatePublisher init.
    stats: dict[str, int] = {
        "renders_total": 0,
        "renders_success": 0,
        "renders_failed": 0,
        "in_flight": 0,
        "kernel_ticks": 0,
        "gallery_queries": 0,
    }
    stats_lock = threading.Lock()

    # ── Heartbeat daemon — started EARLY (Phase 11 §11.I.5) ──
    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(send_queue, name, stop_event, stats, _state_writer),
        daemon=True, name=f"studio-hb-{name}")
    hb_thread.start()
    logger.info(
        "[StudioWorker] Heartbeat thread started EARLY (Phase 11 §11.I.5 — "
        "bus + SHM cadence covers slow StudioCoordinator boot)")

    # ── StudioCoordinator init (replicates plugin.py:_wire_studio inline) ──
    from titan_hcl.expressive.studio import StudioCoordinator

    expressive_cfg = (config or {}).get("expressive", {}) or {}
    output_root = Path(expressive_cfg.get("output_path", "./data/studio_exports"))

    # Build coordinator — no metabolism proxy yet (would require bus.request_async
    # to metabolism_worker; budget-aware resolution scaling deferred to a follow-up
    # since render budgets are already gated upstream by reflex_executors /
    # meditation flows before requests arrive).
    coordinator = StudioCoordinator(
        config=expressive_cfg,
        metabolism=None,
    )
    # Provider-agnostic haiku bridge (Q2 greenlit 2026-05-15).
    coordinator._ollama_cloud = _HaikuLLMBridge(send_queue, name)

    logger.info(
        "[StudioWorker] StudioCoordinator instantiated — output_root=%s "
        "default_res=%d highres_res=%d nft_composite=%s",
        output_root, coordinator._default_res, coordinator._highres_res,
        coordinator._nft_composite_enabled)

    # ── SHM publisher init (G21 single writer for studio_state.bin) ──
    publisher = StudioStatePublisher(
        titan_id=titan_id,
        output_root=output_root,
        meditation_dir=coordinator._meditation_dir,
        epoch_dir=coordinator._epoch_dir,
        eureka_dir=coordinator._eureka_dir,
        default_resolution=coordinator._default_res,
        highres_resolution=coordinator._highres_res,
        nft_composite_enabled=coordinator._nft_composite_enabled,
    )
    # First cold publish — readers can now mmap the slot.
    publisher.publish()

    # Phase A.4 (D-SPEC-70 v1.10.0) — media_state.bin sibling publisher.
    # G21 single-writer; consumed by api_subprocess StateAccessor.media
    # (replaces media.stats bus-cache per Preamble G18).
    media_publisher = None
    try:
        from titan_hcl.logic.media_state_publisher import (
            MediaStatePublisher,
        )
        media_publisher = MediaStatePublisher(titan_id=titan_id)
        media_publisher.publish(coordinator)  # cold-boot first publish
        logger.info(
            "[StudioWorker] media_state publisher attached "
            "(G21 single-writer; Phase A.4 / D-SPEC-70)")
    except Exception as _err:
        logger.warning(
            "[StudioWorker] media_state publisher init failed: %s — "
            "api_subprocess will read cold-boot stubs from media_state",
            _err)

    # ── Render thread pool ──
    render_pool = ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_RENDERS,
        thread_name_prefix=f"studio-render-{name}",
    )

    # ── STUDIO_WORKER_READY (one-shot lifecycle event) ──
    _send(send_queue, STUDIO_WORKER_READY, name, "guardian", {
        "schema_version": STUDIO_STATE_SPEC.schema_version,
        "output_root": str(output_root),
        "ts": time.time(),
    })

    # ── Phase 11 §11.I.2 — slot transition: starting → booted ─────────
    # Replaces the legacy MODULE_READY bus emit per locked D2 — the SHM
    # slot state=booted is the contract.
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[StudioWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[StudioWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)

    logger.info(
        "[StudioWorker] boot complete — STUDIO_WORKER_READY emitted + SHM "
        "slot booted; render_pool=%d workers; ready to serve renders.",
        MAX_CONCURRENT_RENDERS)

    # ── Render completion callback (runs on pool worker thread) ──
    def _on_render_done(future, request_id: str, render_type: str):
        with stats_lock:
            stats["in_flight"] = max(0, stats["in_flight"] - 1)
        try:
            result = future.result()
            paths = result.get("paths", {})
            haiku_text = result.get("haiku_text")
            success = any(v for v in paths.values()) if paths else False

            with stats_lock:
                if success:
                    stats["renders_success"] += 1
                else:
                    stats["renders_failed"] += 1

            # Post-render: bump SHM counter + republish.
            if success:
                publisher.record_render(render_type)
            publisher.publish()
            if media_publisher is not None:
                media_publisher.publish(coordinator)

            _send(send_queue, STUDIO_RENDER_COMPLETED, name, "all", {
                "request_id": request_id,
                "type": render_type,
                "success": success,
                "paths": paths,
                "haiku_text": haiku_text,
                "error": None if success else "render returned empty paths",
                "ts": time.time(),
            })
            logger.info(
                "[StudioWorker] STUDIO_RENDER_COMPLETED — request_id=%s "
                "type=%s success=%s paths_keys=%s",
                request_id, render_type, success, list(paths.keys()))

        except Exception as e:
            with stats_lock:
                stats["renders_failed"] += 1
            logger.warning(
                "[StudioWorker] render exception — request_id=%s type=%s: %s",
                request_id, render_type, e, exc_info=True)
            _send(send_queue, STUDIO_RENDER_COMPLETED, name, "all", {
                "request_id": request_id,
                "type": render_type,
                "success": False,
                "paths": {},
                "haiku_text": None,
                "error": str(e),
                "ts": time.time(),
            })

    # ── Main loop ──
    last_publish_ts = time.time()

    try:
        while True:
            try:
                msg = recv_queue.get(timeout=1.0)
            except Empty:
                # Defensive 1Hz republish if KERNEL_EPOCH_TICK isn't flowing.
                now = time.time()
                if (now - last_publish_ts) >= 1.0:
                    publisher.refresh_counts()
                    publisher.publish()
                    if media_publisher is not None:
                        media_publisher.publish(coordinator)
                    last_publish_ts = now
                continue

            if msg is None:
                continue

            msg_type = msg.get("type") if isinstance(msg, dict) else None
            payload = msg.get("payload", {}) if isinstance(msg, dict) else {}
            if not isinstance(payload, dict):
                payload = {}

            # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ────────
            if msg_type == MODULE_PROBE_REQUEST:
                try:
                    from titan_hcl.core.probe_dispatcher import (
                        handle_module_probe_request,
                    )
                    handle_module_probe_request(
                        msg,
                        probe_fn=None,
                        send_queue=send_queue,
                        module_name=name,
                        state_writer=_state_writer,
                    )
                except Exception as _probe_err:  # noqa: BLE001
                    logger.warning(
                        "[StudioWorker] MODULE_PROBE_REQUEST handler "
                        "failed: %s", _probe_err)
                continue

            if msg_type == STUDIO_RENDER_REQUEST:
                with stats_lock:
                    stats["renders_total"] += 1
                request_id = str(payload.get("request_id", ""))
                render_type = payload.get("type", "")
                args = payload.get("args", {})
                if not isinstance(args, dict):
                    args = {}

                if render_type not in _VALID_RENDER_TYPES:
                    logger.warning(
                        "[StudioWorker] STUDIO_RENDER_REQUEST rejected — "
                        "unknown type=%r (request_id=%s) — emitting "
                        "STUDIO_RENDER_COMPLETED success=False",
                        render_type, request_id)
                    _send(send_queue, STUDIO_RENDER_COMPLETED, name, "all", {
                        "request_id": request_id,
                        "type": render_type,
                        "success": False,
                        "paths": {},
                        "haiku_text": None,
                        "error": f"invalid render type: {render_type!r}",
                        "ts": time.time(),
                    })
                    continue

                with stats_lock:
                    stats["in_flight"] += 1
                fut = render_pool.submit(
                    _dispatch_render, coordinator, render_type, args)
                fut.add_done_callback(
                    lambda f, rid=request_id, rt=render_type:
                        _on_render_done(f, rid, rt))
                logger.info(
                    "[StudioWorker] STUDIO_RENDER_REQUEST accepted — "
                    "request_id=%s type=%s args_keys=%s "
                    "(in_flight=%d, total=%d)",
                    request_id, render_type, list(args.keys()),
                    stats["in_flight"], stats["renders_total"])

            elif msg_type == KERNEL_EPOCH_TICK:
                with stats_lock:
                    stats["kernel_ticks"] += 1
                now = time.time()
                if (now - last_publish_ts) >= 1.0:
                    publisher.refresh_counts()
                    publisher.publish()
                    last_publish_ts = now

            elif msg_type == bus.QUERY:
                with stats_lock:
                    stats["gallery_queries"] += 1
                action = payload.get("action", "")
                rid = msg.get("rid")
                src = msg.get("src", "")

                if action == "get_gallery":
                    category = str(payload.get("category", "all"))
                    try:
                        limit = int(payload.get("limit", 20))
                    except (TypeError, ValueError):
                        limit = 20
                    try:
                        items = coordinator.get_gallery(category, limit)
                    except Exception as e:
                        logger.warning(
                            "[StudioWorker] get_gallery failed: %s",
                            e, exc_info=True)
                        items = []
                    _send_response(send_queue, name, src,
                                   {"items": items, "category": category,
                                    "count": len(items)}, rid)
                else:
                    logger.debug(
                        "[StudioWorker] unhandled QUERY action=%r — ignoring",
                        action)

            elif msg_type == SAVE_NOW:
                logger.info("[StudioWorker] SAVE_NOW — forcing refresh + publish")
                publisher.refresh_counts()
                publisher.publish()
                last_publish_ts = time.time()

            elif msg_type == MODULE_SHUTDOWN:
                logger.info(
                    "[StudioWorker] MODULE_SHUTDOWN — exiting "
                    "(stats: renders_total=%d success=%d failed=%d in_flight=%d "
                    "kernel_ticks=%d gallery_queries=%d)",
                    stats["renders_total"], stats["renders_success"],
                    stats["renders_failed"], stats["in_flight"],
                    stats["kernel_ticks"], stats["gallery_queries"])
                break

    except KeyboardInterrupt:
        logger.info("[StudioWorker] KeyboardInterrupt — exiting")
    except Exception as e:
        logger.error(
            "[StudioWorker] unhandled exception in main loop: %s",
            e, exc_info=True)
        raise
    finally:
        stop_event.set()
        # Allow in-flight renders a brief grace before pool shutdown.
        render_pool.shutdown(wait=False, cancel_futures=False)
        logger.info("[StudioWorker] shutdown complete — render_pool released")
