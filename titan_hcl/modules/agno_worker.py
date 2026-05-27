"""
agno_worker — Python L2 module hosting the Agno Agent + chat pipeline.

Phase C v1.17.0 (D-SPEC-72) per `rFP_agno_worker_and_llm_libraries_extraction.md`.

What this worker owns:
  1. Agno Agent instance (constructed via `modules.agno_agent_factory.create_agent`)
  2. WorkerPlugin shim (`modules.agno_worker_plugin.WorkerPlugin`) — exposes
     the same plugin.X surface that hooks/tools/guardrails expect, backed by
     bus-callable proxies + worker-local state caches
  3. PreHook + PostHook (`titan_hcl/modules/agno_hooks.py`, 1907 LOC)
  4. Tools + GuardianGuardrail (`modules/agno_tools.py` + `modules/agno_guardrails.py`)
  5. Inference provider (via `titan_hcl.inference.get_provider(...)`)
  6. AsyncSqliteDb session store → data/agno_sessions.db (G21 single-writer)
  7. agno_state.bin SHM slot writer (G21 single-writer)

Bus subscriptions (REQUIRED):
  • CHAT_REQUEST            — api_worker → agno_worker (D-SPEC-72 dst flip)
  • CHAT_STREAM_REQUEST     — api_worker → agno_worker SSE path (NEW)
  • KERNEL_EPOCH_TICK       — 1.0 Hz dual-trigger SHM republish cadence
  • MODULE_SHUTDOWN         — graceful drain
  • SAVE_NOW                — forces agno_state.bin republish + session DB checkpoint

Bus publications (non-blocking per §8.0.ter D-SPEC-48):
  • AGNO_WORKER_READY       — once on boot (NEW)
  • CHAT_RESPONSE           — per chat (correlation_id matches REQUEST.rid)
  • CHAT_STREAM_CHUNK       — per token chunk during SSE streaming (NEW)
  • MODULE_HEARTBEAT        — every 30s
  • MODULE_READY            — on first agno_state.bin SHM write completion

See:
  - SPEC v1.17.0 §9.B `agno_worker` block (Chunk D drafts)
  - SPEC v1.17.0 §7.1 `agno_state.bin` SHM slot
  - SPEC v1.17.0 §8.7 bus events (5 affected — 3 new + 2 dst flips)
  - SPEC v1.17.0 §21 D-SPEC-72
  - PLAN_microkernel_phase_c_agno_worker_and_llm_libraries_extraction.md
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import OrderedDict as _OrderedDict
from contextlib import asynccontextmanager
from queue import Empty
from typing import Any, Optional

from titan_hcl import bus
from titan_hcl.bus import (
    AGNO_WORKER_READY,
    CHAT_REQUEST,
    CHAT_RESPONSE,
    CHAT_STREAM_CHUNK,
    CHAT_STREAM_REQUEST,
    DREAM_INBOX_REPLAY,
    KERNEL_EPOCH_TICK,
    MODULE_HEARTBEAT,
    MODULE_READY,
    MODULE_SHUTDOWN,
    SAVE_NOW,
    make_msg,
)
from titan_hcl.logic.agno_state_specs import AGNO_STATE_SPEC

logger = logging.getLogger(__name__)


HEARTBEAT_INTERVAL_S = 30.0
SHM_REPUBLISH_INTERVAL_S = 1.0  # dual-trigger: on tick + on completion

# ── ζ.5 per-tier model routing (D-SPEC-79, 2026-05-18) ─────────────
# Agno's agent.arun() does NOT accept a per-call `model=` override — the
# model identity is read from `agent.model.id` at the moment the upstream
# OpenAILike client issues `client.chat.completions.create(model=self.id, …)`.
# To route per-tier we mutate `agent.model.id` immediately before arun()
# and restore it after. The lock serialises concurrent chats so a fast-
# tier request can't override a heavy-tier request mid-flight.
#
# Single Titan / single user serial chats are the common case; the lock
# is held only across the wrap+arun region (the heavy LLM call itself
# is inside) — effectively the same as serialising the worker's chat
# pipeline, which is already de-facto serial today.
_chat_route_lock = asyncio.Lock()


@asynccontextmanager
async def _route_model_for_tier(agent, worker_plugin, prompt_text: str):
    """Per-tier model swap context manager.

    Classifies `prompt_text` via the worker plugin's ChatTierClassifier
    (lazy-init on first use), resolves the tier's abstract `model_class`
    ("fast"/"light"/"heavy") to a concrete provider model ID via
    `provider.resolve_model_class()`, swaps `agent.model.id` for the
    duration of the `async with` block, and restores on exit.

    No-ops cleanly when the classifier/provider isn't available
    (test paths, fallback modes) — yields without mutating anything.

    Logs the routing decision so cascade probes can confirm wiring.
    """
    classifier = getattr(worker_plugin, "_tier_classifier_cache", None)
    if classifier is None:
        try:
            from titan_hcl.modules.chat_tier_config import ChatTierClassifier
            classifier = ChatTierClassifier.from_config(
                getattr(worker_plugin, "_full_config", {}) or {})
            worker_plugin._tier_classifier_cache = classifier
        except Exception as _cls_err:
            logger.debug("[AgnoWorker] tier classifier init failed: %s", _cls_err)
            yield
            return

    provider = getattr(worker_plugin, "_inference_provider", None)
    if provider is None or not hasattr(agent, "model") or agent.model is None:
        yield
        return

    try:
        result = classifier.classify(prompt_text)
    except Exception as _cls_err:
        logger.debug("[AgnoWorker] classify failed: %s", _cls_err)
        yield
        return

    model_class = result.tier.model_class
    target_id = provider.resolve_model_class(model_class)
    original_id = getattr(agent.model, "id", None)
    # ζ.6 — also swap max_tokens when the tier caps it. None = leave default.
    target_max_tokens = result.tier.max_tokens
    original_max_tokens = getattr(agent.model, "max_tokens", None)
    needs_tokens_swap = (
        target_max_tokens is not None
        and target_max_tokens != original_max_tokens
    )
    needs_model_swap = target_id != original_id

    if not needs_model_swap and not needs_tokens_swap:
        # No swap needed. Log once for observability + return without
        # taking the lock — fast path for heavy tiers that match the
        # constructed agent.
        logger.info(
            "[AgnoWorker] tier=%s model_class=%s model=%s max_tokens=%s (no swap)",
            result.tier.name, model_class, target_id, original_max_tokens,
        )
        yield
        return

    async with _chat_route_lock:
        try:
            if needs_model_swap:
                agent.model.id = target_id
            if needs_tokens_swap:
                agent.model.max_tokens = target_max_tokens
            logger.info(
                "[AgnoWorker] tier=%s model_class=%s model_swap %s→%s max_tokens %s→%s",
                result.tier.name, model_class,
                original_id, target_id,
                original_max_tokens, target_max_tokens,
            )
            yield
        finally:
            try:
                if needs_model_swap:
                    agent.model.id = original_id
                if needs_tokens_swap:
                    agent.model.max_tokens = original_max_tokens
            except Exception:
                pass

# D-SPEC-76 (SPEC v1.18.0) — default session LRU capacity. Canonical value
# lives in `_phase_c_constants` (mirrored from SPEC constants TOML).
from titan_hcl._phase_c_constants import (
    AGNO_SESSION_CACHE_DEFAULT_CAPACITY as DEFAULT_AGNO_SESSION_CACHE_CAPACITY,
)


def _send(send_queue, msg_type: str, src: str, dst: str,
          payload: dict, rid: Optional[str] = None) -> None:
    """Non-blocking publish helper (§8.0.ter D-SPEC-48 publish-non-blocking)."""
    try:
        send_queue.put_nowait(make_msg(msg_type, src, dst, payload, rid=rid))
    except Exception as e:
        logger.warning(
            "[AgnoWorker] _send %s → %s failed: %s",
            msg_type, dst, e,
        )


def _heartbeat_loop(send_queue, name: str,
                    stop_event: threading.Event,
                    stats_ref: dict) -> None:
    """Daemon thread — MODULE_HEARTBEAT every 30s with chat-volume stats."""
    while not stop_event.is_set():
        _send(send_queue, MODULE_HEARTBEAT, name, "guardian", {
            "alive": True,
            "ts": time.time(),
            "session_count": stats_ref.get("session_count", 0),
            "total_chats_24h": stats_ref.get("total_chats_24h", 0),
            "last_chat_ts": stats_ref.get("last_chat_ts", 0.0),
            "in_flight": stats_ref.get("in_flight", 0),
        })
        stop_event.wait(HEARTBEAT_INTERVAL_S)


class AgnoStatePublisher:
    """G21 single-writer for `agno_state.bin` SHM slot.

    Mirrors the StudioStatePublisher pattern (D-SPEC-57). Variable-size
    msgpack payload; triple-buffered SeqLock write via StateRegistryWriter.
    Stub for Chunk C1 — populated state values are real (session_count etc.)
    but provider_stats is empty until Chunk C2 wires the Agno Agent.
    """

    def __init__(self, name: str = "agno_worker"):
        from pathlib import Path

        from titan_hcl.core.state_registry import (
            StateRegistryWriter,
            ensure_shm_root,
            resolve_titan_id,
        )

        self._name = name
        self._titan_id = resolve_titan_id()
        self._shm_root: Path = ensure_shm_root(self._titan_id)
        self._writer = StateRegistryWriter(AGNO_STATE_SPEC, self._shm_root)
        self._last_publish_ts: float = 0.0

    def publish(self, stats: dict[str, Any]) -> bool:
        """Write a fresh agno_state.bin payload. Returns True on success."""
        import msgpack

        from titan_hcl._phase_c_constants import AGNO_STATE_SCHEMA_VERSION

        payload = {
            "schema_version": AGNO_STATE_SCHEMA_VERSION,
            "session_count": int(stats.get("session_count", 0)),
            "last_chat_ts": float(stats.get("last_chat_ts", 0.0)),
            "total_chats_24h": int(stats.get("total_chats_24h", 0)),
            "provider_stats": dict(stats.get("provider_stats", {})),
            "dream_inbox_size": int(stats.get("dream_inbox_size", 0)),
            # D-SPEC-76 (SPEC v1.18.0) — session pre-warm LRU observability.
            # Hit ratio = hits / (hits + misses). Observatory + health monitor
            # surface this via /v4/agno-state.
            "session_cache_size": int(stats.get("session_cache_size", 0)),
            "session_hits": int(stats.get("session_hits", 0)),
            "session_misses": int(stats.get("session_misses", 0)),
            "ts": time.time(),
        }
        try:
            blob = msgpack.packb(payload, use_bin_type=True)
            self._writer.write_variable(blob)
            self._last_publish_ts = payload["ts"]
            return True
        except Exception as e:
            logger.warning("[AgnoWorker] SHM publish failed: %s", e)
            return False


async def _handle_chat_request(msg: dict, agent, worker_plugin, send_queue,
                               name: str, stats_ref: dict) -> None:
    """Dispatch a CHAT_REQUEST through the Agno Agent.

    Updates worker_plugin's per-request state caches (user_id, session) before
    calling agent.arun() so pre/post hooks see the right context, then
    assembles CHAT_RESPONSE from the returned RunOutput + worker_plugin's
    post-hook cached state.

    Dream-state gate (D-SPEC-56): SHM-direct read of dream_state.bin via
    DreamStateReader. If Titan is dreaming, emit DREAM_INBOX_ENQUEUE for
    dream_state_worker to buffer the message + (maker only) emit
    DREAM_WAKE_REQUEST, then reply with a dream-mode CHAT_RESPONSE.
    """
    payload = msg.get("payload", {}) or {}
    src = msg.get("src", "")
    rid = msg.get("rid")
    request_id = payload.get("request_id", rid or "")
    session_id = payload.get("session_id", "default")
    user_id = payload.get("user_id", "anonymous")
    message_text = payload.get("message", "")
    channel = payload.get("channel", "web")
    is_maker = bool(payload.get("is_maker", False))

    stats_ref["in_flight"] = stats_ref.get("in_flight", 0) + 1

    # D-SPEC-76 (SPEC v1.18.0) — session pre-warm LRU.
    # Track (user_id, session_id) recency to surface hit/miss observability
    # for Observatory + health monitor. Actual session DB caching is handled
    # by Agno's AsyncSqliteDb internally; this layer is for visibility +
    # eviction bookkeeping when the cap is reached. Capacity is configurable
    # via [agno_worker].session_cache_capacity (default 32).
    _session_cache = stats_ref.setdefault("_session_cache", _OrderedDict())
    _cache_cap = stats_ref.get("_session_cache_capacity",
                               DEFAULT_AGNO_SESSION_CACHE_CAPACITY)
    _session_key = (user_id, session_id)
    if _session_key in _session_cache:
        stats_ref["session_hits"] = stats_ref.get("session_hits", 0) + 1
        _session_cache.move_to_end(_session_key)
    else:
        stats_ref["session_misses"] = stats_ref.get("session_misses", 0) + 1
        _session_cache[_session_key] = time.time()
        # LRU evict on capacity exceeded
        while len(_session_cache) > _cache_cap:
            _session_cache.popitem(last=False)
    stats_ref["session_cache_size"] = len(_session_cache)

    response_text = ""
    error_str: Optional[str] = None
    try:
        # ── Dream-state gate (D-SPEC-56) ──────────────────────────────
        # Replaces the parent-side gate that lived in chat_pipeline.run_chat
        # (deleted in Chunk H per Q5 LOCKED). agno_worker is the canonical
        # owner per SPEC §9.B agno_worker block "Shm reads: dream_state.bin".
        try:
            _dream_reader = getattr(worker_plugin, '_dream_reader', None)
            if _dream_reader is None:
                from titan_hcl.logic.dream_state_reader import DreamStateReader
                _dream_reader = DreamStateReader()
                worker_plugin._dream_reader = _dream_reader
            _dream_snapshot = _dream_reader.read()
            # Defensive isinstance(dict) — real reader returns dict; MagicMock
            # returns Mock (truthy on .get()). Production behavior unchanged.
            if (isinstance(_dream_snapshot, dict)
                    and _dream_snapshot.get("is_dreaming", False)):
                # Buffer the message via dream_state_worker
                from titan_hcl.bus import (
                    DREAM_INBOX_ENQUEUE,
                    DREAM_WAKE_REQUEST,
                )
                if worker_plugin.bus is not None:
                    worker_plugin.bus.publish(make_msg(
                        DREAM_INBOX_ENQUEUE, name, "dream_state",
                        {
                            "message": message_text[:500],
                            "user_id": user_id,
                            "session_id": session_id,
                            "channel": channel,
                            "priority": 0 if is_maker else 1,
                            "client_ts": time.time(),
                        },
                    ))
                    if is_maker:
                        # Maker messages trigger gentle wake.
                        worker_plugin.bus.publish(make_msg(
                            DREAM_WAKE_REQUEST, name, "dream_state",
                            {"reason": "maker_message", "user_id": user_id},
                        ))
                # Compose + send dream-mode CHAT_RESPONSE
                _recovery = float(_dream_snapshot.get("recovery_pct", 0.0))
                _wake_transition = bool(_dream_snapshot.get("wake_transition", False))
                _remaining = int(_dream_snapshot.get("remaining_epochs", 0))
                _eta_min = round(_remaining * 12.5 / 60, 1)
                _dream_response = {
                    "request_id": request_id,
                    "response": (
                        f"Titan is currently "
                        f"{'waking gently' if _wake_transition else 'dreaming'} "
                        f"(recovery: {_recovery:.0f}%). "
                        f"Your message has been queued. "
                        f"Estimated wake: ~{_eta_min:.0f} minutes."
                    ),
                    "session_id": session_id,
                    "mode": "dreaming",
                    "mood": "sleeping",
                    "state_narration": None,
                    "state_snapshot": {
                        "is_dreaming": True,
                        "recovery_pct": _recovery,
                        "eta_minutes": _eta_min,
                        "wake_transition": _wake_transition,
                    },
                    "ovg_data": None,
                    "error": None,
                    "ts": time.time(),
                }
                _send(send_queue, CHAT_RESPONSE, name, src,
                      _dream_response, rid=rid)
                stats_ref["total_chats_24h"] = stats_ref.get("total_chats_24h", 0) + 1
                stats_ref["last_chat_ts"] = time.time()
                return
        except Exception as _dream_err:
            # Best-effort gate — on any error proceed to normal chat path.
            logger.debug(
                "[AgnoWorker] Dream-state gate error (proceeding to chat): %s",
                _dream_err,
            )

        # Update worker_plugin per-request state (replaces the inline
        # api/chat.py assignments at L155-160 in the parent path)
        worker_plugin._current_user_id = user_id
        worker_plugin._pre_chat_user_id = user_id

        # ── Run Agno agent (ζ.5 per-tier model routing) ──
        # _route_model_for_tier classifies the prompt, swaps agent.model.id
        # to the tier's concrete model (resolved via provider.resolve_model_class),
        # serialises concurrent chats so requests don't trample each other's
        # model id, and restores the original id on exit.
        async with _route_model_for_tier(agent, worker_plugin, message_text):
            run_output = await agent.arun(
                message_text,
                session_id=session_id,
                user_id=user_id,
            )

        if hasattr(run_output, "content"):
            response_text = str(run_output.content)
        elif isinstance(run_output, str):
            response_text = run_output
        else:
            response_text = str(run_output)

    except Exception as e:
        logger.exception("[AgnoWorker] agent.arun failed: %s", e)
        error_str = str(e)

    finally:
        stats_ref["in_flight"] = max(0, stats_ref.get("in_flight", 0) - 1)

    # ── Assemble CHAT_RESPONSE payload ──
    # D-SPEC-74 (SPEC v1.18.0): _last_ovg_result is now a VerifiedResult
    # which exposes `ovg_data` dict directly (per llm_pipeline.verifier).
    # Fall back to OVGResult-shaped attribute access for back-compat
    # with paths that still emit the legacy result type.
    ovg_result = getattr(worker_plugin, "_last_ovg_result", None)
    ovg_data = None
    if ovg_result is not None:
        _od = getattr(ovg_result, "ovg_data", None)
        if isinstance(_od, dict) and _od:
            ovg_data = _od
        else:
            ovg_data = {
                "verified": bool(getattr(ovg_result, "passed", False)),
                "guard_alert": getattr(ovg_result, "guard_alert", None),
                "guard_message": getattr(ovg_result, "guard_message", "") or "",
                "block_height": int(getattr(ovg_result, "block_height", 0) or 0),
                "merkle_root": getattr(ovg_result, "merkle_root", "") or "",
                "signature": getattr(ovg_result, "signature", None),
            }

    response_payload = {
        "request_id": request_id,
        "response": response_text,
        "session_id": session_id,
        "mode": getattr(worker_plugin, "_last_execution_mode", "") or "",
        "mood": "Unknown",  # populated post-Chunk-K when mood SHM read wires in
        "state_narration": None,
        "state_snapshot": None,
        "ovg_data": ovg_data,
        "error": error_str,
        "ts": time.time(),
    }
    _send(send_queue, CHAT_RESPONSE, name, src, response_payload, rid=rid)

    if error_str is None:
        stats_ref["total_chats_24h"] = stats_ref.get("total_chats_24h", 0) + 1
        stats_ref["last_chat_ts"] = time.time()


async def _handle_dream_inbox_replay(msg: dict, agent, worker_plugin,
                                     send_queue, name: str,
                                     stats_ref: dict) -> None:
    """Re-process chat messages buffered while the Titan was dreaming.

    dream_state_worker drains its inbox on dream_end and broadcasts
    DREAM_INBOX_REPLAY (dst="all"). agno_worker — the chat-action owner —
    consumes it here (moved from the retired parent _v4_event_bridge_loop per
    RFP_phase_c_titan_hcl_cleanup Phase A): each buffered message is dispatched
    through the normal chat path so the Titan "answers" the missed message into
    its memory + persona log. No live HTTP client awaits the reply (rid=None →
    the CHAT_RESPONSE is dropped at the broker), preserving the original intent.
    """
    payload = msg.get("payload", {}) or {}
    replay_msgs = payload.get("messages", []) or []
    if not replay_msgs:
        return
    processed = 0
    for rmsg in replay_msgs:
        if not isinstance(rmsg, dict):
            continue
        synthetic = {
            "type": CHAT_REQUEST,
            "src": "dream_replay",
            "dst": name,
            "rid": None,  # no live client — reply drops at broker
            "payload": {
                "request_id": f"replay-{int(time.time() * 1e6)}",
                "message": rmsg.get("message", ""),
                "user_id": rmsg.get("user_id", "anonymous"),
                "session_id": rmsg.get("session_id", "default"),
                "channel": rmsg.get("channel", "web"),
                "is_maker": False,
                "claims_sub": "",
                "replay": True,
                "prefer_streaming": False,
                "ts": time.time(),
            },
        }
        try:
            await _handle_chat_request(
                synthetic, agent, worker_plugin, send_queue, name, stats_ref)
            processed += 1
        except Exception as e:
            logger.warning(
                "[AgnoWorker] DREAM_INBOX_REPLAY re-process error for "
                "user=%s: %s", rmsg.get("user_id"), e)
    logger.info(
        "[AgnoWorker] DREAM_INBOX_REPLAY processed — re-answered %d/%d "
        "buffered messages (dream_duration=%.1fs)",
        processed, len(replay_msgs), payload.get("dream_duration_s", 0.0))


async def _handle_chat_stream_request(msg: dict, agent, worker_plugin,
                                      send_queue, name: str,
                                      stats_ref: dict) -> None:
    """Stream a chat as CHAT_STREAM_CHUNK events AFTER safety verification.

    Phase 2 Chunk δ (D-SPEC-78, 2026-05-18) — POST-SAFETY-PASS streaming.
    Per Maker correction 2026-05-17, the OVG truth-gate is non-negotiable:
    no bytes leave agno_worker until `verify_safety()` returns
    passed=True. Pre-this-change the stream path used `agent.arun(stream=True)`
    which streamed LLM tokens BEFORE OVG ran (PostHook only fires AFTER
    the stream completes) — a truth-gate violation that's been latent
    since D-SPEC-72.

    New flow:
      1. `agent.arun()` to full response (no stream=True; PostHook fires
         normally including OVG verify_safety + concurrent signing).
      2. Once PostHook returns (response = post-OVG verified text, with
         optional guard_message footer), split into ~200-char logical
         segments at sentence boundaries.
      3. Emit each segment as CHAT_STREAM_CHUNK with `done=False`.
      4. Final chunk has `done=True` + `ovg_headers` populated from
         worker_plugin._last_ovg_result (the VerifiedResult VerifiedResult
         already carries .ovg_data with signature/block_height when the
         concurrent sign task has resolved by emit-time).

    Concurrent signing latency win:
      sign_and_commit runs as asyncio.Task spawned inside PostHook.
      Drain-the-stream takes ~50-200ms (network-bound). On a healthy
      host the sign task usually completes BEFORE we emit the final
      chunk; if not, ovg_headers may be empty (signing still committing
      to TimeChain in background; user already has the verified content).

    Future (deferred): incremental sentence-boundary verify_safety would
    let us stream during LLM generation. Real R&D — not this chunk.
    """
    payload = msg.get("payload", {}) or {}
    src = msg.get("src", "")
    rid = msg.get("rid")
    request_id = payload.get("request_id", rid or "")
    session_id = payload.get("session_id", "default")
    user_id = payload.get("user_id", "anonymous")
    message_text = payload.get("message", "")

    stats_ref["in_flight"] = stats_ref.get("in_flight", 0) + 1
    try:
        worker_plugin._current_user_id = user_id
        worker_plugin._pre_chat_user_id = user_id

        # ── 1. Run agent to completion (Pre+LLM+Post-with-OVG) ──
        # ζ.5 per-tier model routing (D-SPEC-79) — see _route_model_for_tier
        async with _route_model_for_tier(agent, worker_plugin, message_text):
            run_output = await agent.arun(
                message_text,
                session_id=session_id,
                user_id=user_id,
            )
        if hasattr(run_output, "content"):
            response_text = str(run_output.content)
        elif isinstance(run_output, str):
            response_text = run_output
        else:
            response_text = str(run_output)

        # ── 2. Pull OVG verdict from PostHook side-channel ──
        # agno_hooks PostHook writes worker_plugin._last_ovg_result =
        # VerifiedResult after verify_post_async. Its `.ovg_data` carries
        # signature/block_height once the concurrent sign task resolves.
        ovg_result = getattr(worker_plugin, "_last_ovg_result", None)
        ovg_data = None
        if ovg_result is not None:
            _od = getattr(ovg_result, "ovg_data", None)
            if isinstance(_od, dict) and _od:
                ovg_data = _od

        # ── 3. Segment the verified response for progressive UX ──
        segments = _segment_for_stream(response_text)

        for idx, seg in enumerate(segments):
            is_last = (idx == len(segments) - 1)
            _send(send_queue, CHAT_STREAM_CHUNK, name, src, {
                "request_id": request_id,
                "chunk": seg,
                "done": False,
                "ts": time.time(),
            }, rid=rid)

        # ── 4. Final done frame with ovg_headers ──
        _send(send_queue, CHAT_STREAM_CHUNK, name, src, {
            "request_id": request_id,
            "chunk": "",
            "done": True,
            "ovg_headers": ovg_data or {},
            "ts": time.time(),
        }, rid=rid)

        stats_ref["total_chats_24h"] = stats_ref.get("total_chats_24h", 0) + 1
        stats_ref["last_chat_ts"] = time.time()
    except Exception as e:
        logger.exception("[AgnoWorker] chat_stream failed: %s", e)
        _send(send_queue, CHAT_STREAM_CHUNK, name, src, {
            "request_id": request_id,
            "chunk": "",
            "done": True,
            "error": str(e),
            "ts": time.time(),
        }, rid=rid)
    finally:
        stats_ref["in_flight"] = max(0, stats_ref.get("in_flight", 0) - 1)


def _segment_for_stream(text: str, target_len: int = 200) -> list[str]:
    """Split verified response into ~target_len-char chunks at sentence
    or whitespace boundaries.

    Used by _handle_chat_stream_request to chunk the FULL-AND-VERIFIED
    response for SSE delivery. Boundaries chosen so the user sees
    coherent phrases appearing rather than mid-word splits.
    """
    if not text:
        return [""]
    if len(text) <= target_len:
        return [text]

    segments: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= target_len:
            segments.append(remaining)
            break
        # Prefer sentence break first
        cut = -1
        for delim in (". ", "! ", "? ", "\n\n", "\n"):
            idx = remaining.rfind(delim, 0, target_len + 50)
            if idx > target_len // 2:
                cut = idx + len(delim)
                break
        if cut <= 0:
            # Fallback: word boundary
            idx = remaining.rfind(" ", target_len // 2, target_len + 50)
            cut = (idx + 1) if idx > 0 else target_len
        segments.append(remaining[:cut])
        remaining = remaining[cut:]
    return segments


def _init_worker_plugin_and_agent(bus_client, config: dict[str, Any]):
    """Construct WorkerPlugin + Agno Agent for this worker subprocess.

    Returns (worker_plugin, agent) tuple. Agent.arun() is the entry point
    for CHAT_REQUEST dispatch.

    Construction order matters: hooks/tools/guardrails close over plugin
    references at create_agent time, so worker_plugin MUST be fully
    initialised before create_agent runs.
    """
    from titan_hcl.modules.agno_agent_factory import create_agent
    from titan_hcl.modules.agno_worker_plugin import WorkerPlugin

    worker_plugin = WorkerPlugin(bus_client=bus_client, config=config)
    agent = create_agent(worker_plugin, agent_config=config.get("agent"))
    return worker_plugin, agent


def _install_phase9_baseline_hook() -> None:
    """Install file-flag → JSON-dump daemon thread for Phase 9 Chunk 9A.

    Pure instrumentation. Starts tracemalloc(1 frame) so per-allocation
    overhead stays minimal (<1% RSS / <2% CPU). A daemon thread polls for
    /tmp/agno_baseline_request_<pid> every 0.5s; when the flag file
    appears, captures a snapshot, writes
    /tmp/agno_baseline_<pid>_<unix_ts_ms>.json, deletes the flag.

    Rationale for thread over SIGUSR1: when the main thread is blocked in
    a C extension (LLM client socket read, bus publish), Python defers
    SIGUSR1 delivery indefinitely — confirmed live on T3 2026-05-27 where
    dumps arrived 3+ minutes after the signal. The daemon thread runs
    free of main-thread state and responds within 0.5s even if main is
    fully hung. Never raises into the worker: any error logs to /tmp.
    Idempotent: tracemalloc.start is a no-op if already tracing;
    duplicate thread launch is guarded by module-global flag.
    """
    import json
    import os
    import sys
    import threading
    import tracemalloc

    if not tracemalloc.is_tracing():
        tracemalloc.start(1)

    # Guard against double-launch across hot-reload
    if getattr(_install_phase9_baseline_hook, "_running", False):
        return
    _install_phase9_baseline_hook._running = True  # type: ignore[attr-defined]

    pid = os.getpid()
    request_path = f"/tmp/agno_baseline_request_{pid}"

    def _write_dump() -> None:
        try:
            ts = time.time()
            snap = tracemalloc.take_snapshot()
            top = snap.statistics("lineno")[:20]
            rss_kb = 0
            try:
                with open(f"/proc/{pid}/status") as fh:
                    for line in fh:
                        if line.startswith("VmRSS:"):
                            rss_kb = int(line.split()[1])
                            break
            except Exception:
                pass
            cur, peak = tracemalloc.get_traced_memory()
            out = {
                "pid": pid,
                "ts_unix": ts,
                "rss_kb": rss_kb,
                "tracemalloc_current_bytes": cur,
                "tracemalloc_peak_bytes": peak,
                "sys_modules_count": len(sys.modules),
                "sys_modules": sorted(sys.modules.keys()),
                "tracemalloc_top20": [
                    {
                        "file": str(s.traceback[0].filename),
                        "line": s.traceback[0].lineno,
                        "size_bytes": int(s.size),
                        "count": int(s.count),
                    }
                    for s in top
                ],
            }
            out_path = f"/tmp/agno_baseline_{pid}_{int(ts * 1000)}.json"
            tmp = out_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(out, fh)
            os.replace(tmp, out_path)
        except Exception as exc:
            try:
                with open(
                    f"/tmp/agno_baseline_error_{pid}_{int(time.time())}.log", "w",
                ) as fh:
                    import traceback as _tb
                    fh.write(f"{exc}\n{_tb.format_exc()}")
            except Exception:
                pass

    def _watcher_loop() -> None:
        while True:
            try:
                if os.path.exists(request_path):
                    try:
                        os.unlink(request_path)
                    except OSError:
                        pass
                    _write_dump()
            except Exception:
                pass
            time.sleep(0.5)

    t = threading.Thread(
        target=_watcher_loop, daemon=True, name="phase9-baseline-watcher",
    )
    t.start()


def agno_worker_main(recv_queue, send_queue, name: str,
                     config: dict[str, Any]) -> None:
    """Entry function for the agno_worker L2 process.

    Boots WorkerPlugin + Agno Agent, then dispatches CHAT_REQUEST /
    CHAT_STREAM_REQUEST through `await agent.arun(...)`. Hooks read/write
    worker-local state caches on WorkerPlugin between pre and post phases.

    Args:
        recv_queue: receives bus messages (bus → worker).
        send_queue: sends bus messages (worker → bus).
        name:       module name ('agno_worker').
        config:     dict — typically merged [agent] + [inference] config
                    from config.toml.
    """
    logger.info("[AgnoWorker] Boot")
    boot_start = time.time()

    # ── Phase 9 Chunk 9A instrumentation — pure baseline capture ──
    # Pre-Phase-9 RSS root-cause baseline per RFP §3F + per-discipline
    # `feedback_eager_init_needs_rss_root_cause_first`. tracemalloc starts
    # FIRST so it observes every subsequent allocation in worker boot. The
    # SIGUSR1 handler is triggered by `scripts/agno_baseline.py` at the 5
    # RFP-defined checkpoints {boot_complete, chat_1_in, chat_1_out,
    # chat_5_out, chat_10_out} and writes a JSON snapshot to /tmp. Zero
    # behavior change to chat handling. To be removed when Phase 9 closes
    # (gated by RFP §3F.5 LOCK; see §3F.2 chunk 9A).
    _install_phase9_baseline_hook()

    # ── asyncio loop owned by this worker (Agno is async-first) ──
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # ── Worker bus client (kernel-owned send/recv queue wrapper) ──
    # The hooks need a bus reference that supports .publish / .subscribe /
    # .request_async. We construct a lightweight WorkerBusClient over
    # send_queue + recv_queue. For C2 with stub-mode acceptance, hooks
    # tolerate worker_plugin.bus being None; production wiring of the
    # full bus client into worker subprocesses is a Chunk-K hardening pass.
    bus_client = _build_worker_bus_client(send_queue, recv_queue, name)

    # ── Stats reference shared with heartbeat thread ──
    # D-SPEC-76 (SPEC v1.18.0) — `_session_cache_capacity` is read by
    # _handle_chat_request for LRU eviction; configurable via
    # [agno_worker].session_cache_capacity in config.toml.
    _agno_cfg = (config or {}).get("agno_worker", {}) or {}
    stats: dict[str, Any] = {
        "session_count": 0,
        "last_chat_ts": 0.0,
        "total_chats_24h": 0,
        "provider_stats": {},
        "dream_inbox_size": 0,
        "in_flight": 0,
        "_session_cache_capacity": int(
            _agno_cfg.get("session_cache_capacity",
                          DEFAULT_AGNO_SESSION_CACHE_CAPACITY)),
        "session_cache_size": 0,
        "session_hits": 0,
        "session_misses": 0,
    }

    # ── WorkerPlugin + Agent construction ──
    worker_plugin: Optional[Any] = None
    agent: Optional[Any] = None
    try:
        worker_plugin, agent = _init_worker_plugin_and_agent(bus_client, config)
        logger.info("[AgnoWorker] Agent constructed in %.0fms",
                    (time.time() - boot_start) * 1000)
    except Exception as e:
        logger.exception(
            "[AgnoWorker] Agent construction failed — chat handler will "
            "return error responses: %s", e,
        )

    # ── D-SPEC-138 (v1.63.1, 2026-05-26) — Eager OVG warmup ──
    # OutputVerifier construction (Solana keypair load + TimeChain.open) is
    # ~30s cold-start on T1 (50 MB mainnet chain). Pre-D-SPEC-138 the OVG
    # was lazy-instantiated via `worker_plugin._output_verifier` property on
    # the FIRST chat's PostHook before_ovg → after_ovg stage. That moved
    # the cold-start latency into the request critical path, blowing past
    # the 90s AgnoBridge CHAT_REQUEST timeout for the very first chat on
    # a freshly-spawned worker. With Guardian RSS-limit restarts hitting
    # agno_worker frequently (separate root-cause investigation), every
    # subsequent chat hit the same cold start → permanent T1 timeout
    # cascade. Eager-init at boot moves the latency where it belongs (out
    # of the request path) and matches the same anti-pattern-correction
    # discipline as D-SPEC-134 (`lazy init in critical path = anti-pattern`).
    if worker_plugin is not None:
        try:
            ovg_start = time.time()
            _ = worker_plugin._output_verifier  # triggers lazy init now, at boot
            logger.info(
                "[AgnoWorker] OVG warmed in %.0fms (eager init at boot — "
                "D-SPEC-138; closes first-chat cold-start cascade)",
                (time.time() - ovg_start) * 1000,
            )
        except Exception as _ovg_err:
            # Defense-in-depth: if eager init fails, the lazy-init path
            # still works on first chat. The warning surfaces the regression
            # without blocking boot.
            logger.warning(
                "[AgnoWorker] OVG eager-init failed (lazy retry will run "
                "on first chat — first-chat latency will spike): %s",
                _ovg_err,
            )

    # ── SHM publisher (G21 single-writer for agno_state.bin) ──
    try:
        publisher = AgnoStatePublisher(name=name)
        publisher.publish(stats)
        _send(send_queue, MODULE_READY, name, "guardian", {})
    except Exception as e:
        publisher = None
        logger.warning(
            "[AgnoWorker] SHM publisher init failed (continuing anyway): %s", e
        )

    # ── B.1 Phase B.1 §6 readiness/hibernate reporter ──
    from titan_hcl.core.readiness_reporter import trivial_reporter

    def _b1_save_state():
        return []  # agno_worker has no shadow-swap state to checkpoint in C1
    b1_reporter = trivial_reporter(
        worker_name=name, layer="L2", send_queue=send_queue,
        save_state_cb=_b1_save_state,
    )

    # ── Heartbeat daemon thread ──
    hb_stop = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(send_queue, name, hb_stop, stats),
        daemon=True, name="agno-heartbeat",
    )
    hb_thread.start()
    logger.info(
        "[AgnoWorker] Heartbeat thread started (30s interval)"
    )

    # ── AGNO_WORKER_READY broadcast (boot signal for guardian + observers) ──
    _send(send_queue, AGNO_WORKER_READY, name, "all", {
        "ts": time.time(),
        "boot_ms": int((time.time() - boot_start) * 1000),
        "scaffold": False,  # C2: Agent integrated
        "agent_ready": agent is not None,
    })

    logger.info(
        "[AgnoWorker] Ready in %.0fms — awaiting CHAT_REQUEST",
        (time.time() - boot_start) * 1000,
    )

    # ── Main dispatch loop ──
    # D-SPEC-128 (BUG-AGNO-SILENT-HANG fix): read from the bus client's
    # `consumer_queue`, NOT the raw `recv_queue`. The bus_client's
    # dispatcher thread owns raw recv_queue exclusively; any other
    # consumer of raw recv_queue would race with it and steal messages.
    # The dispatcher routes rid-bearing replies to request_async waiters
    # via Futures; everything else lands in consumer_queue for this
    # loop.
    consumer_queue = bus_client.consumer_queue
    last_shm_publish = 0.0
    try:
        while True:
            try:
                msg = consumer_queue.get(timeout=5.0)
            except Empty:
                # No message; opportunity to republish SHM if cadence elapsed
                now = time.time()
                if publisher and (now - last_shm_publish) >= SHM_REPUBLISH_INTERVAL_S:
                    publisher.publish(stats)
                    last_shm_publish = now
                continue
            except (KeyboardInterrupt, SystemExit):
                break

            msg_type = msg.get("type", "")

            # ── Phase B.1 §6 shadow-swap dispatch ──
            if b1_reporter.handles(msg_type):
                b1_reporter.handle(msg)
                if b1_reporter.should_exit():
                    break
                continue

            # ── Phase B.2.1 supervision-transfer dispatch ──
            from titan_hcl.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue

            # ── Lifecycle ──
            if msg_type == MODULE_SHUTDOWN:
                logger.info("[AgnoWorker] Shutdown requested")
                break

            if msg_type == SAVE_NOW:
                if publisher:
                    publisher.publish(stats)
                    last_shm_publish = time.time()
                continue

            if msg_type == KERNEL_EPOCH_TICK:
                if publisher:
                    publisher.publish(stats)
                    last_shm_publish = time.time()
                continue

            # ── Chat dispatch via Agno Agent (C2) ──
            if msg_type == CHAT_REQUEST:
                if agent is None or worker_plugin is None:
                    # Agent construction failed at boot — surface error to caller
                    _send(send_queue, CHAT_RESPONSE, name,
                          msg.get("src", ""), {
                              "request_id": (msg.get("payload") or {})
                                  .get("request_id", msg.get("rid") or ""),
                              "response": "",
                              "session_id": (msg.get("payload") or {})
                                  .get("session_id", "default"),
                              "mode": "",
                              "mood": "",
                              "state_narration": None,
                              "state_snapshot": None,
                              "ovg_data": None,
                              "error": "agno_worker_agent_unavailable",
                              "ts": time.time(),
                          }, rid=msg.get("rid"))
                else:
                    loop.run_until_complete(
                        _handle_chat_request(
                            msg, agent, worker_plugin, send_queue, name, stats,
                        )
                    )
                if publisher:
                    publisher.publish(stats)
                    last_shm_publish = time.time()
                continue

            if msg_type == CHAT_STREAM_REQUEST:
                if agent is None or worker_plugin is None:
                    _send(send_queue, CHAT_STREAM_CHUNK, name,
                          msg.get("src", ""), {
                              "request_id": (msg.get("payload") or {})
                                  .get("request_id", msg.get("rid") or ""),
                              "chunk": "",
                              "done": True,
                              "error": "agno_worker_agent_unavailable",
                              "ts": time.time(),
                          }, rid=msg.get("rid"))
                else:
                    loop.run_until_complete(
                        _handle_chat_stream_request(
                            msg, agent, worker_plugin, send_queue, name, stats,
                        )
                    )
                if publisher:
                    publisher.publish(stats)
                    last_shm_publish = time.time()
                continue

            # ── Dream-inbox replay (RFP Phase A — moved from parent bridge) ──
            if msg_type == DREAM_INBOX_REPLAY:
                if agent is not None and worker_plugin is not None:
                    loop.run_until_complete(
                        _handle_dream_inbox_replay(
                            msg, agent, worker_plugin, send_queue, name, stats,
                        )
                    )
                    if publisher:
                        publisher.publish(stats)
                        last_shm_publish = time.time()
                continue

            # Unknown — log at DEBUG (per directive_error_visibility: known
            # set of accepted types; anything else is a wiring drift)
            logger.debug("[AgnoWorker] Ignoring unhandled msg_type=%s", msg_type)

    finally:
        logger.info("[AgnoWorker] Exiting")
        hb_stop.set()
        # D-SPEC-128: stop the bus dispatcher cleanly. Daemon thread so
        # the process will exit either way, but explicit stop avoids
        # spurious "raw_recv.get raised" warnings during teardown.
        try:
            bus_client.stop()
        except Exception:
            pass
        try:
            loop.close()
        except Exception:
            pass


def _build_worker_bus_client(send_queue, recv_queue, worker_name: str):
    """Build the worker's in-process bus client.

    **Architecture (BUG-AGNO-SILENT-HANG proper fix — D-SPEC-128, 2026-05-25):**

    The kernel delivers ALL messages destined for this worker (chat
    requests, lifecycle events, broadcasts, work-RPC replies) into a
    single `recv_queue`. Multiple consumers want to read from it:

      - The **worker's main loop** wants CHAT_REQUEST, CHAT_STREAM_REQUEST,
        MODULE_SHUTDOWN, KERNEL_EPOCH_TICK, broadcasts.
      - Each in-flight `request_async()` wants ITS specific
        `QUERY_RESPONSE` (matched by `rid`).

    Pre-fix (D-SPEC-74 era), all consumers raced on `recv_queue.get()`:
    main loop in its own thread, `request_async` via executor poll. Any
    message popped by the wrong consumer was dropped — silently stealing
    CHAT_REQUESTs during OVG verify_safety_async calls → fleet-wide
    silent agno_worker degradation. See `tests/
    test_agno_worker_request_async_redelivery.py` for the
    reproduction.

    **This implementation (the spec-correct fix):** a single
    **dispatcher thread** owns the raw `recv_queue` exclusively. It
    reads serially and routes each message:

      1. If the message carries an `rid` AND a `request_async()` waiter
         is registered for that rid → resolve the waiter's
         `asyncio.Future` directly (via `loop.call_soon_threadsafe` so
         the resolve runs on the caller's loop).
      2. Otherwise → push the message onto a per-worker
         `consumer_queue` that the main loop reads.

    Consequences:

      - **No race possible.** Dispatcher is the SOLE reader of
        `recv_queue`. Main loop reads only from `consumer_queue`;
        request_async awaits a future. No two consumers ever pop the
        same queue.
      - **No message loss.** Every message lands in exactly one
        destination: either a registered waiter, or the consumer queue.
      - **No CPU burn.** Future-based wait (no polling); dispatcher
        sleeps in a single blocking `recv_queue.get()` between messages.
      - **Back-compat for proxies.** `subscribe()` returns
        `consumer_queue` so any external code that polls it still gets
        non-request_async messages. `request_async`'s `reply_queue`
        parameter is now ignored (kept in signature for back-compat
        with proxies that pass it positionally).
      - **Single fix covers all 6 affected proxies** (output_verifier,
        life_force, assessment, metabolism, rl, media) — they all
        funnel through `_WorkerBusClient.request_async`.

    Main loop wiring requirement: callers of `_build_worker_bus_client`
    MUST read messages from `client.consumer_queue` instead of the
    `recv_queue` they originally passed in. The dispatcher OWNS the raw
    recv_queue; reading from it directly would race with the
    dispatcher's `.get()` call (mp.Queue handles concurrent gets safely
    but each message goes to ONE consumer — so messages would be lost
    to whichever consumer got there first). See `agno_worker_main` for
    the canonical usage.

    Refs:
      - BUG-AGNO-SILENT-HANG (2026-05-25)
      - D-SPEC-128 (this commit)
      - directive_error_visibility.md (silent-worker anti-pattern this closes)
      - feedback_no_quick_patches_only_spec_correct_solutions.md (why this
        replaces the prior minimal re-queue fix)
    """
    import asyncio as _aio
    import threading as _threading
    import queue as _queue

    class _WorkerBusClient:
        def __init__(self, send_q, recv_q, name):
            self._send = send_q
            self._raw_recv = recv_q
            self._name = name

            # Per-rid waiter registry — request_async registers a Future,
            # dispatcher resolves it. Lock guards concurrent register/pop.
            self._rid_lock = _threading.Lock()
            self._rid_waiters: dict[str, _aio.Future] = {}

            # Main-loop-facing queue — receives every message that ISN'T
            # routed to a request_async waiter.
            self.consumer_queue: _queue.Queue = _queue.Queue()

            # Dispatcher thread lifecycle.
            self._stop = _threading.Event()
            self._dispatcher = _threading.Thread(
                target=self._dispatcher_loop,
                daemon=True,
                name=f"{name}-bus-dispatcher",
            )
            self._dispatcher.start()

        # ── Dispatcher (sole reader of raw_recv) ──────────────────────

        def _dispatcher_loop(self) -> None:
            """Read raw_recv serially; route by rid → future OR consumer."""
            while not self._stop.is_set():
                try:
                    # 0.2s poll — keeps dispatcher responsive to stop_event
                    # without busy-spinning. mp.Queue.get with timeout
                    # raises Empty (queue module's Empty); both queue and
                    # mp.Queue use the same exception class.
                    msg = self._raw_recv.get(timeout=0.2)
                except _queue.Empty:
                    continue
                except Exception as exc:
                    # Any other queue exception (e.g. closed at shutdown):
                    # log + retry. Don't kill the dispatcher silently.
                    logger.warning(
                        "[WorkerBusClient:dispatcher] raw_recv.get raised "
                        "%s — retrying", exc)
                    continue
                if not msg:
                    continue

                rid = msg.get("rid") if isinstance(msg, dict) else None

                # Try rid-routed delivery first.
                routed = False
                if rid:
                    with self._rid_lock:
                        fut = self._rid_waiters.pop(rid, None)
                    if fut is not None and not fut.done():
                        try:
                            loop = fut.get_loop()
                            loop.call_soon_threadsafe(
                                _safe_set_future_result, fut, msg)
                            routed = True
                        except Exception as exc:
                            # Loop closed / future already cancelled —
                            # fall through to consumer_queue so the
                            # message isn't lost.
                            logger.warning(
                                "[WorkerBusClient:dispatcher] rid=%s "
                                "future-resolve failed (%s) — routing to "
                                "consumer queue", rid, exc)

                if routed:
                    continue

                # Default route: main consumer queue.
                try:
                    self.consumer_queue.put_nowait(msg)
                except _queue.Full as exc:
                    # consumer_queue is unbounded by default → Full
                    # shouldn't happen. If it ever does, log loudly and
                    # block-with-timeout to apply backpressure.
                    logger.error(
                        "[WorkerBusClient:dispatcher] consumer_queue full "
                        "(unbounded by default) — applying backpressure: %s",
                        exc)
                    try:
                        self.consumer_queue.put(msg, timeout=1.0)
                    except Exception as exc2:
                        logger.error(
                            "[WorkerBusClient:dispatcher] consumer_queue "
                            "blocked-put failed — message LOST "
                            "(type=%s rid=%s): %s",
                            msg.get("type"), rid, exc2)

        def stop(self) -> None:
            """Signal the dispatcher to exit. Safe to call from main loop
            during shutdown. Does NOT join — daemon thread exits with
            process; join would deadlock if main loop holds locks."""
            self._stop.set()

        # ── Public API ────────────────────────────────────────────────

        def publish(self, msg) -> None:
            """Non-blocking send to kernel via send_queue."""
            try:
                self._send.put_nowait(msg)
            except Exception as e:
                logger.warning(
                    "[WorkerBusClient] publish failed: %s", e
                )

        def subscribe(self, name: str, reply_only: bool = False):
            """Return the consumer_queue.

            Per the dispatcher architecture, ALL non-rid-routed messages
            arrive in consumer_queue regardless of `name`. The `name`
            param is preserved in the signature for back-compat with
            proxy code that calls `bus.subscribe("foo_proxy",
            reply_only=True)`; since `request_async` now uses futures
            (not the returned queue), callers using subscribe() purely
            to pass to request_async get a working — if unused — queue
            reference.
            """
            return self.consumer_queue

        async def request_async(self, src, dst, payload, timeout,
                                reply_queue=None,
                                msg_type=None) -> Optional[dict]:
            """Async work-RPC. Publishes QUERY/msg_type, awaits the
            matching RESPONSE via the dispatcher's rid-routing.

            D-SPEC-74 / D-SPEC-128: `src` is overridden with the
            worker's registered name so kernel-side reply routing
            delivers the response back to THIS worker's raw_recv
            (the dispatcher then routes by rid to this caller's future).

            `reply_queue` is IGNORED (back-compat parameter). The
            dispatcher owns the raw queue; routing is via the
            per-rid Future registry. See class docstring.

            Returns the matching reply message, or None on:
              - publish failure (kernel send_queue full)
              - timeout (no matching reply within `timeout` seconds)
              - future cancellation (caller aborted)
            """
            request_id = f"{self._name}_{int(time.time() * 1e6)}"
            type_to_send = msg_type or bus.QUERY

            # Register the future BEFORE sending — eliminates any race
            # where the reply arrives faster than the registration.
            loop = _aio.get_running_loop()
            fut = loop.create_future()
            with self._rid_lock:
                self._rid_waiters[request_id] = fut

            try:
                try:
                    self._send.put_nowait({
                        "type": type_to_send,
                        "src": self._name, "dst": dst,
                        "rid": request_id,
                        "payload": payload,
                        "ts": time.time(),
                    })
                except Exception as e:
                    logger.warning(
                        "[WorkerBusClient] request_async publish failed: %s", e
                    )
                    return None

                try:
                    return await _aio.wait_for(fut, timeout=float(timeout))
                except _aio.TimeoutError:
                    return None
                except _aio.CancelledError:
                    raise
            finally:
                # Always clean up the registration — on timeout/cancel
                # the dispatcher would otherwise hold a stale entry and
                # silently drop a late-arriving reply on consumer_queue
                # (correct — late replies are uninteresting; explicit
                # cleanup just frees the dict slot).
                with self._rid_lock:
                    self._rid_waiters.pop(request_id, None)

    return _WorkerBusClient(send_queue, recv_queue, worker_name)


def _safe_set_future_result(fut, value) -> None:
    """Helper for `loop.call_soon_threadsafe` — guard the actual
    `set_result` call so a cancelled/done future doesn't raise inside
    the asyncio loop machinery."""
    try:
        if not fut.done():
            fut.set_result(value)
    except Exception:
        pass
