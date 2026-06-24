"""
LLM/Inference Module Worker — runs the Ollama Cloud client and inference
in its own supervised process.

Isolates LLM session state (~500MB) and Agno's SQLite sessions.db.

Entry point: llm_worker_main(recv_queue, send_queue, name, config)
"""
import logging
import os
import sys
import threading
import time
from titan_hcl import bus
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)


# Phase 11 §11.I.3/§11.I.5 — module-level readiness sentinel mirrored to
# `module_llm_state.bin` via ModuleStateWriter so titan_hcl's 1Hz SHM
# poll + the orchestrator's MODULE_PROBE_REQUEST dispatcher see real
# liveness. Flipped True only after the inference provider construct
# completes.
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace


@with_error_envelope(module_name="llm", subsystem="entry", severity=_phase11_sev.FATAL)
def llm_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """
    Main loop for the LLM module process.

    Args:
        recv_queue: receives messages from DivineBus (bus→worker)
        send_queue: sends messages back to DivineBus (worker→bus)
        name: module name ("llm")
        config: dict from [inference] config section
    """
    from queue import Empty

    # Phase 11 §11.I.5 — reset module-level readiness flag.
    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[LLMWorker] Initializing inference subsystem...")
    init_start = time.time()

    # ── Phase 11 §11.I.5 — SHM state-slot writer (G21 single-writer) ──
    # Constructed BEFORE inference-provider init so the slot publishes
    # state="starting" immediately. The heartbeat thread below publishes
    # state_writer.heartbeat() once _WORKER_READY flips True.
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority, ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name=name, layer="L3",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[LLMWorker] Phase 11 ModuleStateWriter init failed "
            "(continuing — SHM slot will be absent): %s", _sw_err)

    # D-SPEC-72 (SPEC v1.17.0 §9.F.1) — construct via canonical
    # provider abstraction. `titan_hcl/utils/ollama_cloud.py` was
    # DELETED in the same rFP commit set (NOT shimmed) per
    # `feedback_no_shim_old_path_must_be_deleted.md` — all 20+ legacy
    # callsites migrated to `titan_hcl.inference.get_provider`.
    ollama_cloud = None
    try:
        ollama_key = config.get("ollama_cloud_api_key", "")
        if ollama_key:
            from titan_hcl.inference import (
                get_provider, resolve_internal_provider_name)
            ollama_cloud = get_provider(
                resolve_internal_provider_name(config), config)
            logger.info(
                "[LLMWorker] Ollama Cloud provider initialized via inference module"
            )
    except Exception as e:
        logger.warning("[LLMWorker] Ollama Cloud init failed: %s", e)

    init_ms = (time.time() - init_start) * 1000
    logger.info("[LLMWorker] Inference subsystem ready in %.0fms", init_ms)

    # Phase A.4 (D-SPEC-70 v1.10.0) — llm_state.bin publisher.
    # G21 single-writer; consumed by api_subprocess StateAccessor.llm
    # (replaces llm.stats bus-cache per Preamble G18).
    llm_state_publisher = None
    llm_stats = {
        "provider": "ollama_cloud" if ollama_cloud else "",
        "model": config.get("ollama_cloud_model", "") or "",
        "total_completions": 0,
        "completions_this_hour": 0,
        "avg_latency_ms": 0.0,
        "p99_latency_ms": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "last_completion_ts": 0.0,
        "last_error": "",
        "error_rate": 0.0,
    }
    try:
        from titan_hcl.logic.llm_state_publisher import LLMStatePublisher
        from titan_hcl.core.state_registry import resolve_titan_id as _resolve_tid_llm
        llm_state_publisher = LLMStatePublisher(titan_id=_resolve_tid_llm())
        llm_state_publisher.publish(llm_stats)
        logger.info(
            "[LLMWorker] llm_state publisher attached "
            "(G21 single-writer; Phase A.4 / D-SPEC-70)")
    except Exception as _err:
        logger.warning(
            "[LLMWorker] llm_state publisher init failed: %s — "
            "api_subprocess will read cold-boot stubs from llm_state",
            _err)

    # ── Phase 11 §11.I.2 — SHM slot transition starting → booted ─────
    # MODULE_READY bus emit DELETED per D-SPEC-141 / v1.65.0 locked D2.
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[LLMWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[LLMWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)

    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Background heartbeat thread — sends heartbeats every 30s regardless of
    # handler blocking. Eliminates Guardian timeout deaths from slow Ollama calls.
    _hb_stop = threading.Event()

    def _heartbeat_loop():
        """Phase 11 §11.I.5: bus MODULE_HEARTBEAT + SHM state_writer
        heartbeat run on the same 30s cadence so guardian_hcl's SHM-
        staleness detector + observatory /v6/readiness see fresh data
        while the inference provider's per-request latency may exceed
        the heartbeat interval."""
        while not _hb_stop.is_set():
            _send_heartbeat(send_queue, name)
            # Phase A.4 — refresh llm_state.bin every heartbeat (30s) so
            # readers see fresh ts even if no recent completions.
            if llm_state_publisher is not None:
                try:
                    llm_state_publisher.publish(dict(llm_stats))
                except Exception:
                    pass
            # Phase 11 §11.I.5 — SHM slot heartbeat sidecar.
            if _state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
                try:
                    _state_writer.heartbeat()
                except Exception:  # noqa: BLE001 — never crash heartbeat
                    pass
            _hb_stop.wait(30.0)

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True, name="llm-heartbeat")
    hb_thread.start()
    logger.info("[LLMWorker] Background heartbeat thread started (30s interval)")

    # ── Microkernel v2 Phase B.1 §6 — readiness/hibernate reporter ──
    from titan_hcl.core.readiness_reporter import trivial_reporter
    def _b1_save_state():
        return []
    _b1_reporter = trivial_reporter(
        worker_name=name, layer="L3", send_queue=send_queue,
        save_state_cb=_b1_save_state,
    )

    while True:
        try:
            msg = recv_queue.get(timeout=5.0)
        except Empty:
            continue  # Background heartbeat thread handles Guardian keepalive
        except (KeyboardInterrupt, SystemExit):
            break

        msg_type = msg.get("type", "")

        # ── Microkernel v2 Phase B.1 §6 — shadow swap dispatch ────
        if _b1_reporter.handles(msg_type):
            _b1_reporter.handle(msg)
            if _b1_reporter.should_exit():
                break
            continue

        # ── Microkernel v2 Phase B.2.1 — supervision-transfer dispatch ──
        from titan_hcl.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ─────────────
        if msg_type == "MODULE_PROBE_REQUEST":
            try:
                from titan_hcl.core.probe_dispatcher import (
                    handle_module_probe_request,
                )
                handle_module_probe_request(
                    msg, send_queue=send_queue, module_name=name,
                    state_writer=_state_writer, probe_fn=None,
                )
            except Exception as _probe_err:  # noqa: BLE001
                logger.warning(
                    "[LLMWorker] MODULE_PROBE_REQUEST handler failed: %s",
                    _probe_err)
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[LLMWorker] Shutdown requested")
            break

        # D-SPEC-77 (SPEC v1.18.0) — explicit topic dispatch. The legacy
        # `bus.QUERY action="chat"` / `action="distill"` paths were dead code
        # post-D-SPEC-72 (agno_worker imports inference module directly; no
        # caller routes via llm_worker for chat LLM calls). New explicit
        # topics expose distill/score with optional pre_hook/post_hook flags
        # so non-agno callers can "talk through the truth-gate" via
        # llm_worker (X replies, autonomous language pipeline, future
        # avatars). See bus.py LLM_DISTILL_REQUEST + LLM_SCORE_REQUEST.
        if msg_type == bus.LLM_TEACHER_REQUEST:
            _send_heartbeat(send_queue, name)
            last_heartbeat = time.time()
            _handle_teacher(msg, ollama_cloud, send_queue, name, loop)

        elif msg_type == bus.LLM_DISTILL_REQUEST:
            _send_heartbeat(send_queue, name)
            last_heartbeat = time.time()
            _handle_distill(msg, ollama_cloud, send_queue, name, loop)

        elif msg_type == bus.LLM_SCORE_REQUEST:
            _send_heartbeat(send_queue, name)
            last_heartbeat = time.time()
            _handle_score(msg, ollama_cloud, send_queue, name, loop)

    logger.info("[LLMWorker] Exiting")
    _hb_stop.set()
    loop.close()


def _handle_distill(msg: dict, ollama_cloud, send_queue, name: str,
                    loop) -> None:
    """Handle LLM_DISTILL_REQUEST (D-SPEC-77, SPEC v1.18.0).

    Replaces the dead `bus.QUERY action="distill"` path. Supports optional
    `pre_hook` / `post_hook` flags — when set, the worker runs
    llm_pipeline.compose_pre / verify_post_async around the provider call,
    making llm_worker the canonical "talk through the truth-gate" surface
    for non-agno callers (X replies, autonomous language pipeline, future
    avatars). Hooks default to FALSE so existing callers see byte-identical
    behavior.

    Request payload:
        text: str             — text to distill
        instruction: str?     — instruction for the LLM
        model: str?           — provider model override
        pre_hook: bool?       — run compose_pre (default False)
        post_hook: bool?      — run verify_post_async (default False)
        channel: str?         — OVG channel when post_hook=True (default "agent")
    Response:
        {result: str, model: str, elapsed_ms: float, ovg: dict?}
    """
    payload = msg.get("payload", {})
    rid = msg.get("rid")
    src = msg.get("src", "")
    text = payload.get("text", "")
    instruction = payload.get("instruction", "Summarize concisely")
    model_override = payload.get("model")
    # Phase 3 Chunk ω (D-SPEC-88, 2026-05-18) — accept max_tokens + temperature
    # overrides so callers needing non-default sampling (events_teacher uses
    # max_tokens=800, temperature=0.3) can migrate without losing parity.
    max_tokens_override = payload.get("max_tokens")
    temperature_override = payload.get("temperature")
    pre_hook = bool(payload.get("pre_hook", False))
    post_hook = bool(payload.get("post_hook", False))
    channel = payload.get("channel", "agent")
    t0 = time.time()
    ovg_dict = None

    try:
        if not ollama_cloud:
            _send_msg(send_queue, bus.LLM_DISTILL_RESPONSE, name, src, {
                "result": text[:200],
                "model": "",
                "elapsed_ms": 0.0,
                "ovg": None,
                "error": "no_provider",
            }, rid=rid)
            return

        if pre_hook:
            # llm_pipeline.compose_pre wraps the prompt with felt_state +
            # primedirectives. Done in-process — no extra bus hop.
            try:
                from titan_hcl import llm_pipeline
                composed = loop.run_until_complete(
                    llm_pipeline.compose_pre(
                        message=text, user_id="distill", channel=channel,
                    )
                )
                if composed and getattr(composed, "augmented_prompt", None):
                    instruction = (
                        composed.augmented_prompt + "\n\n" + instruction
                    )
            except Exception as e:
                logger.warning(
                    "[LLMWorker] pre_hook compose_pre failed: %s — "
                    "continuing without it", e)

        kwargs = {"text": text, "instruction": instruction}
        if model_override:
            kwargs["model"] = model_override
        if max_tokens_override is not None:
            kwargs["max_tokens"] = int(max_tokens_override)
        # Note: OllamaCloudProvider.distill() doesn't accept temperature
        # as a kwarg (hardcoded to 0.3 internally — already matches the
        # 0.3 default events_teacher uses). For callers needing a different
        # temperature, use chat() directly via a future complete() override.
        # Skipping temperature passthrough for now keeps the proxy/worker
        # surface minimal until a real caller needs non-0.3.
        if temperature_override is not None:
            # Reserved for future provider API support; currently no-op.
            pass
        result = loop.run_until_complete(ollama_cloud.distill(**kwargs))

        if post_hook:
            # llm_pipeline.verify_post_async gates the output via OVG.
            # Returns VerifiedResult — replace result text with verified
            # text (which may include guard_message on soft-alert paths).
            try:
                from titan_hcl.llm_pipeline.verifier import verify_post_async
                _verified = loop.run_until_complete(verify_post_async(
                    result, channel=channel, prompt=text,
                    output_verifier=None,   # llm_worker has no OVG proxy
                                            # locally; caller wires it via
                                            # publish_timechain=False path
                    bus=None, publish_timechain=False,
                    concurrent_sign=False, append_guard_on_pass=False,
                ))
                if _verified is not None:
                    result = _verified.text
                    ovg_dict = _verified.ovg_data
            except Exception as e:
                logger.warning(
                    "[LLMWorker] post_hook verify_post failed: %s", e)

        _send_msg(send_queue, bus.LLM_DISTILL_RESPONSE, name, src, {
            "result": result,
            "model": model_override or "",
            "elapsed_ms": (time.time() - t0) * 1000,
            "ovg": ovg_dict,
        }, rid=rid)
    except Exception as e:
        logger.error("[LLMWorker] distill error: %s", e, exc_info=True)
        _send_msg(send_queue, bus.LLM_DISTILL_RESPONSE, name, src, {
            "result": "",
            "model": model_override or "",
            "elapsed_ms": (time.time() - t0) * 1000,
            "ovg": None,
            "error": str(e),
        }, rid=rid)


def _handle_score(msg: dict, ollama_cloud, send_queue, name: str,
                  loop) -> None:
    """Handle LLM_SCORE_REQUEST (D-SPEC-77, SPEC v1.18.0).

    For meta_teacher_worker gating + future meta-cognitive scoring paths.
    Calls InferenceProvider.score(prompt, ...) and emits LLM_SCORE_RESPONSE
    with the float score.

    Request payload:
        prompt: str           — text to score
        model: str?           — provider model override
        timeout: float?       — per-call timeout (default 30s)
    Response:
        {score: float, model: str, elapsed_ms: float}
    """
    payload = msg.get("payload", {})
    rid = msg.get("rid")
    src = msg.get("src", "")
    prompt = payload.get("prompt", "")
    model_override = payload.get("model")
    request_timeout = float(payload.get("timeout", 30.0))
    t0 = time.time()

    try:
        if not ollama_cloud:
            _send_msg(send_queue, bus.LLM_SCORE_RESPONSE, name, src, {
                "score": 0.0,
                "model": "",
                "elapsed_ms": 0.0,
                "error": "no_provider",
            }, rid=rid)
            return

        score_kwargs = {"prompt": prompt, "timeout": request_timeout}
        if model_override:
            score_kwargs["model"] = model_override
        # OllamaCloudProvider.score is positional (prompt, model, timeout)
        # per the no-shim retirement direction.
        score_value = loop.run_until_complete(
            ollama_cloud.score(**score_kwargs))
        _send_msg(send_queue, bus.LLM_SCORE_RESPONSE, name, src, {
            "score": float(score_value),
            "model": model_override or "",
            "elapsed_ms": (time.time() - t0) * 1000,
        }, rid=rid)
    except Exception as e:
        logger.error("[LLMWorker] score error: %s", e, exc_info=True)
        _send_msg(send_queue, bus.LLM_SCORE_RESPONSE, name, src, {
            "score": 0.0,
            "model": model_override or "",
            "elapsed_ms": (time.time() - t0) * 1000,
            "error": str(e),
        }, rid=rid)


def _handle_teacher(msg: dict, ollama_cloud, send_queue, name: str, loop) -> None:
    """Handle language teacher inference request (fire-and-forget from spirit_worker)."""
    payload = msg.get("payload", {})
    src = msg.get("src", "")
    prompt = payload.get("prompt", "")
    system = payload.get("system", "")
    max_tokens = payload.get("max_tokens", 100)
    temperature = payload.get("temperature", 0.4)
    mode = payload.get("mode", "unknown")

    try:
        if ollama_cloud:
            # D-SPEC-72: get_model_for_task lives at inference module per §9.F.1
            # (utils/ollama_cloud.py DELETED — canonical path only).
            from titan_hcl.inference import get_model_for_task
            model = get_model_for_task("language_teacher")
            response = loop.run_until_complete(
                ollama_cloud.complete(
                    prompt=prompt,
                    model=model,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30.0,
                )
            )
        else:
            response = ""
            logger.warning("[LLMWorker] No inference provider for teacher request")

        _send_msg(send_queue, bus.LLM_TEACHER_RESPONSE, name, src, {
            "response": response,
            "mode": mode,
            "original": payload.get("original", ""),
            "sentences": payload.get("sentences", []),
            "neuromod_gate": payload.get("neuromod_gate", ""),
        })
        logger.info("[LLMWorker] Teacher %s completed (%d chars)", mode, len(response))

    except Exception as e:
        logger.error("[LLMWorker] Teacher error: %s", e, exc_info=True)
        _send_msg(send_queue, bus.LLM_TEACHER_RESPONSE, name, src, {
            "response": "",
            "mode": mode,
            "error": str(e),
        })


def _send_msg(send_queue, msg_type: str, src: str, dst: str, payload: dict, rid: str = None) -> None:
    try:
        send_queue.put_nowait({
            "type": msg_type, "src": src, "dst": dst,
            "ts": time.time(), "rid": rid, "payload": payload,
        })
    except Exception:
        from titan_hcl.bus import record_send_drop
        record_send_drop(src, dst, msg_type)


def _send_response(send_queue, src: str, dst: str, payload: dict, rid: str) -> None:
    _send_msg(send_queue, bus.RESPONSE, src, dst, payload, rid)


# Heartbeat throttle (Phase E Fix 2): 3s min interval per process.
_last_hb_ts: float = 0.0


def _send_heartbeat(send_queue, name: str) -> None:
    global _last_hb_ts
    now = time.time()
    if now - _last_hb_ts < 3.0:
        return
    _last_hb_ts = now
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        rss_mb = 0
    _send_msg(send_queue, bus.MODULE_HEARTBEAT, name, "guardian", {"rss_mb": round(rss_mb, 1)})
