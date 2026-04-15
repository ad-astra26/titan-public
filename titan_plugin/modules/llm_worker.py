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

logger = logging.getLogger(__name__)


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

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[LLMWorker] Initializing inference subsystem...")
    init_start = time.time()

    ollama_cloud = None
    try:
        ollama_key = config.get("ollama_cloud_api_key", "")
        ollama_url = config.get("ollama_cloud_base_url", "https://api.ollama.com/v1")
        if ollama_key:
            from titan_plugin.utils.ollama_cloud import OllamaCloudClient
            ollama_cloud = OllamaCloudClient(api_key=ollama_key, base_url=ollama_url)
            logger.info("[LLMWorker] Ollama Cloud client initialized")
    except Exception as e:
        logger.warning("[LLMWorker] Ollama Cloud init failed: %s", e)

    init_ms = (time.time() - init_start) * 1000
    logger.info("[LLMWorker] Inference subsystem ready in %.0fms", init_ms)

    # Signal ready
    _send_msg(send_queue, "MODULE_READY", name, "guardian", {})

    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Background heartbeat thread — sends heartbeats every 30s regardless of
    # handler blocking. Eliminates Guardian timeout deaths from slow Ollama calls.
    _hb_stop = threading.Event()

    def _heartbeat_loop():
        while not _hb_stop.is_set():
            _send_heartbeat(send_queue, name)
            _hb_stop.wait(30.0)

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True, name="llm-heartbeat")
    hb_thread.start()
    logger.info("[LLMWorker] Background heartbeat thread started (30s interval)")

    while True:
        try:
            msg = recv_queue.get(timeout=5.0)
        except Empty:
            continue  # Background heartbeat thread handles Guardian keepalive
        except (KeyboardInterrupt, SystemExit):
            break

        msg_type = msg.get("type", "")

        if msg_type == "MODULE_SHUTDOWN":
            logger.info("[LLMWorker] Shutdown requested")
            break

        if msg_type == "QUERY":
            _send_heartbeat(send_queue, name)  # Prevent starvation during long LLM calls
            last_heartbeat = time.time()
            _handle_query(msg, ollama_cloud, send_queue, name, loop)

        elif msg_type == "LLM_TEACHER_REQUEST":
            _send_heartbeat(send_queue, name)  # Prevent starvation during long LLM calls
            last_heartbeat = time.time()
            _handle_teacher(msg, ollama_cloud, send_queue, name, loop)

    logger.info("[LLMWorker] Exiting")
    _hb_stop.set()
    loop.close()


def _handle_query(msg: dict, ollama_cloud, send_queue, name: str, loop) -> None:
    """Handle LLM-related queries."""
    payload = msg.get("payload", {})
    action = payload.get("action", "")
    rid = msg.get("rid")
    src = msg.get("src", "")

    try:
        if action == "chat":
            prompt = payload.get("prompt", "")
            context = payload.get("context", {})
            if ollama_cloud:
                response = loop.run_until_complete(
                    ollama_cloud.chat(prompt, system_prompt=context.get("system_prompt", ""))
                )
            else:
                response = "[LLM module: no inference provider configured]"
            _send_response(send_queue, name, src, {"response": response}, rid)

        elif action == "distill":
            text = payload.get("text", "")
            instruction = payload.get("instruction", "Summarize concisely")
            if ollama_cloud:
                response = loop.run_until_complete(
                    ollama_cloud.chat(f"{instruction}:\n\n{text}")
                )
            else:
                response = text[:200]
            _send_response(send_queue, name, src, {"result": response}, rid)

        else:
            logger.warning("[LLMWorker] Unknown action: %s", action)

    except Exception as e:
        logger.error("[LLMWorker] Error handling %s: %s", action, e, exc_info=True)
        _send_response(send_queue, name, src, {"error": str(e)}, rid)


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
            from titan_plugin.utils.ollama_cloud import get_model_for_task
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

        _send_msg(send_queue, "LLM_TEACHER_RESPONSE", name, src, {
            "response": response,
            "mode": mode,
            "original": payload.get("original", ""),
            "sentences": payload.get("sentences", []),
            "neuromod_gate": payload.get("neuromod_gate", ""),
        })
        logger.info("[LLMWorker] Teacher %s completed (%d chars)", mode, len(response))

    except Exception as e:
        logger.error("[LLMWorker] Teacher error: %s", e, exc_info=True)
        _send_msg(send_queue, "LLM_TEACHER_RESPONSE", name, src, {
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
        from titan_plugin.bus import record_send_drop
        record_send_drop(src, dst, msg_type)


def _send_response(send_queue, src: str, dst: str, payload: dict, rid: str) -> None:
    _send_msg(send_queue, "RESPONSE", src, dst, payload, rid)


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
    _send_msg(send_queue, "MODULE_HEARTBEAT", name, "guardian", {"rss_mb": round(rss_mb, 1)})
