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
  • SAVE_NOW                — forces agno_state.bin (SHM stats) republish. NOTE:
    session history is durable per-write via AsyncSqliteDb (no separate DB
    checkpoint needed/done — the prior "+ session DB checkpoint" claim was
    incorrect, AUDIT §C / rFP §P2). The only non-durable state is in-memory
    observability stats (session_cache/hits/misses/total_chats_24h), which are
    SHM-published, not persisted — acceptable loss on respawn.

Bus publications (non-blocking per §8.0.ter D-SPEC-48):
  • AGNO_WORKER_READY       — once on boot (NEW)
  • CHAT_RESPONSE           — per chat (correlation_id matches REQUEST.rid)
  • CHAT_STREAM_CHUNK       — per token chunk during SSE streaming (NEW)
  • MODULE_HEARTBEAT        — every 30s
  • (MODULE_READY retired — Phase 11 §11.I.2 SHM slot state=booted is the contract)

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
import os
import threading
import time
from collections import OrderedDict as _OrderedDict
from contextlib import asynccontextmanager
from queue import Empty
from typing import Any, Optional

from titan_hcl import bus
from titan_hcl.bus import (
    AGNO_WORKER_READY,
    CGN_LEXICON_UPDATED,
    CHAT_REQUEST,
    CHAT_RESPONSE,
    CHAT_STREAM_CHUNK,
    CHAT_STREAM_REQUEST,
    DREAM_INBOX_REPLAY,
    KERNEL_EPOCH_TICK,
    KNOWLEDGE_MOMENT,
    MEMORY_RETRIEVAL_USED,
    RETRIEVAL_SAMPLE,
    MODULE_HEARTBEAT,
    MODULE_SHUTDOWN,
    SAVE_NOW,
    make_msg,
)
from titan_hcl.logic.agno_state_specs import AGNO_STATE_SPEC

logger = logging.getLogger(__name__)


HEARTBEAT_INTERVAL_S = 30.0
SHM_REPUBLISH_INTERVAL_S = 1.0  # dual-trigger: on tick + on completion


# Phase 11 §11.I.3 / §11.I.5 (Chunk 11N) — module-level readiness sentinel
# consumed by `titan_hcl.probes.agno.agno_worker_probe`. Set False at module
# import time; flipped True after BOTH Agent construction AND eager OVG
# warmup complete. Mirrored to the per-process SHM slot
# (`module_agno_worker_state.bin`) via ModuleStateWriter so titan_hcl's
# 1Hz SHM poll + the orchestrator's MODULE_PROBE_REQUEST dispatcher see
# real liveness rather than the boot-time "subscribed-but-not-warm" lie
# that broke /chat on the T3 cascade 2026-05-27.
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)

_AGENT_READY: bool = False
_OVG_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace

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
    # ζ.7 — inject the tier's reply-length GUIDANCE into the agent instructions so
    # the model plans a complete reply within budget (finishes its thought) rather
    # than being hard-cut at max_tokens. Only when instructions is a plain list (the
    # factory shape); restored in finally. max_tokens remains the safety ceiling.
    target_guidance = result.tier.reply_guidance
    original_instructions = getattr(agent, "instructions", None)
    needs_guidance = bool(target_guidance) and isinstance(original_instructions, list)

    if not needs_model_swap and not needs_tokens_swap and not needs_guidance:
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
            if needs_guidance:
                agent.instructions = original_instructions + [
                    f"RESPONSE LENGTH — {target_guidance} Write a complete reply that "
                    f"finishes its thought within that length; never stop mid-sentence."
                ]
            logger.info(
                "[AgnoWorker] tier=%s model_class=%s model_swap %s→%s max_tokens %s→%s guidance=%s",
                result.tier.name, model_class,
                original_id, target_id,
                original_max_tokens, target_max_tokens, bool(needs_guidance),
            )
            yield
        finally:
            try:
                if needs_model_swap:
                    agent.model.id = original_id
                if needs_tokens_swap:
                    agent.model.max_tokens = original_max_tokens
                if needs_guidance:
                    agent.instructions = original_instructions
            except Exception:
                pass


# ── Concurrent multi-user chat (RFP_concurrent_multiuser_chat) ─────────────
# Default ON: each chat mints its OWN agent over the shared chat context with
# the tier baked in (no shared-agent mutation → no _chat_route_lock → concurrent
# chats overlap). flag-OFF = the legacy shared-agent + _route_model_for_tier
# lock path above (INV-CC-7 / kill-switch).
def _concurrent_chat_enabled(worker_plugin) -> bool:
    """`[chat] concurrent_chat_enabled` — default True (Maker flag rule)."""
    try:
        chat_cfg = (getattr(worker_plugin, "_full_config", {}) or {}).get("chat", {}) or {}
        return bool(chat_cfg.get("concurrent_chat_enabled", True))
    except Exception:
        return True


def _get_tier_classifier(worker_plugin):
    """Cached ChatTierClassifier (lazy-init), or None if unavailable.

    Shares `worker_plugin._tier_classifier_cache` with the legacy
    `_route_model_for_tier` so both paths classify identically.
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
            return None
    return classifier


def _classify_message_tier(worker_plugin, message_text):
    """Classify `message_text` → TierConfig (or None if unavailable).

    The concurrent path bakes this tier into a per-call agent instead of
    mutating a shared one — same classification the legacy router does inline.
    """
    if getattr(worker_plugin, "_inference_provider", None) is None:
        return None
    classifier = _get_tier_classifier(worker_plugin)
    if classifier is None:
        return None
    try:
        return classifier.classify(message_text).tier
    except Exception as _cls_err:
        logger.debug("[AgnoWorker] classify failed: %s", _cls_err)
        return None


def _make_chat_agent(worker_plugin, message_text, shared_agent):
    """Mint a fresh per-call agent for this chat (INV-CC-2: own instance per
    concurrent call). Falls back to `shared_agent` if the chat context or
    per-call build is unavailable (keeps chat working in degraded/test paths).
    """
    try:
        from titan_hcl.modules.agno_agent_factory import make_agent
        tier = _classify_message_tier(worker_plugin, message_text)
        return make_agent(worker_plugin._chat_ctx, tier)
    except Exception as _mk_err:  # noqa: BLE001
        logger.warning(
            "[AgnoWorker] per-call agent build failed (%s) — using shared agent",
            _mk_err,
        )
        return shared_agent


# D-SPEC-76 (SPEC v1.18.0) — default session LRU capacity. Canonical value
# lives in `_phase_c_constants` (mirrored from SPEC constants TOML).
from titan_hcl._phase_c_constants import (
    AGNO_SESSION_CACHE_DEFAULT_CAPACITY as DEFAULT_AGNO_SESSION_CACHE_CAPACITY,
)
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev
from titan_hcl.params import get_params


def _ground_for_goal_hook(plugin: Any, text: str) -> list[str]:
    """Lightweight CGN lexicon-cache word-grounding for INV-Syn-17.

    Reads `plugin.cgn_lexicon` if present (a dict[str, str] mapping
    lowercase tokens to concept_ids; populated by an out-of-band loader
    that reads `data/cgn_lexicon_snapshot.json` — Phase 7+ optional).
    No cross-process call; if the cache is missing or any token isn't
    in it, the corresponding concept_id is simply omitted from the
    return list. Returns [] on any error.

    Output capped at 20 to keep the goal-buffer payload small.
    """
    try:
        if not text:
            return []
        lex = getattr(plugin, "cgn_lexicon", None)
        if not isinstance(lex, dict) or not lex:
            return []
        tokens = [t.lower().strip(".,!?:;()[]\"'") for t in text.split() if len(t) > 2]
        out: list[str] = []
        seen: set[str] = set()
        for t in tokens:
            cid = lex.get(t)
            if cid and cid not in seen:
                out.append(cid)
                seen.add(cid)
                if len(out) >= 20:
                    break
        return out
    except Exception:
        return []


def _load_cgn_lexicon(plugin: Any) -> int:
    """P8.Y fold-in: load `data/cgn_lexicon_snapshot.json` into
    `plugin.cgn_lexicon` so `_ground_for_goal_hook` returns real
    concept_ids instead of `[]`.

    Soft-fail: missing snapshot → leaves plugin.cgn_lexicon empty
    (current production behavior, no regression). Called at boot
    + on CGN_LEXICON_UPDATED bus event."""
    try:
        from titan_hcl.cgn.lexicon_exporter import load_lexicon_snapshot
        lex = load_lexicon_snapshot()
        if not isinstance(lex, dict):
            return 0
        plugin.cgn_lexicon = lex
        return len(lex)
    except Exception as e:
        logger.debug("[AgnoWorker] _load_cgn_lexicon failed: %s", e)
        return 0


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
                    stats_ref: dict,
                    state_writer: Optional[Any] = None) -> None:
    """Daemon thread — MODULE_HEARTBEAT every 30s with chat-volume stats.

    Phase 11 §11.I.5 (Chunk 11N): also publishes ModuleStateWriter.heartbeat()
    on the SHM slot when `state_writer` is provided so guardian_hcl's
    SHM-staleness detector + observatory `/v6/readiness` see fresh data
    on the same cadence as the legacy bus path. SHM writes are best-effort —
    failures degrade gracefully (legacy bus heartbeat still load-bears
    Guardian's existing staleness check).
    """
    while not stop_event.is_set():
        _send(send_queue, MODULE_HEARTBEAT, name, "guardian", {
            "alive": True,
            "ts": time.time(),
            "session_count": stats_ref.get("session_count", 0),
            "total_chats_24h": stats_ref.get("total_chats_24h", 0),
            "last_chat_ts": stats_ref.get("last_chat_ts", 0.0),
            "in_flight": stats_ref.get("in_flight", 0),
        })
        if state_writer is not None and shm_heartbeat_allowed(_AGENT_READY and _OVG_READY, _BOOT_DEADLINE):
            try:
                # heartbeat() republishes with state="running" — only
                # valid once both sentinels are True. During the boot
                # window the slot retains state="starting" or "booted"
                # (set explicitly via write_state at the relevant points).
                state_writer.heartbeat()
            except Exception:  # noqa: BLE001 — never crash the heartbeat
                pass
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
        # Phase 7 (D-SPEC-PHASE7): expose session_id so agno_tools'
        # _resolve_chat_id can construct f"{user_id}:{session_id}" for the
        # BufferCache call. Mirrors _current_user_id pattern above.
        worker_plugin._current_session_id = session_id
        # RFP_verifiable_autobiographical_presence_memory §7.A — expose the turn's
        # channel so the PostHook can label the PERSON_TURN_PRESENCE atom
        # (web/app/tcc). Mirrors the _current_user_id stash above.
        worker_plugin._current_channel = channel
        # RFP_missions §7.1 — the turn's verified-Maker flag, computed at the api
        # edge (presence-based is_maker, chat.py) and plumbed in the inbound
        # payload. The PostHook's TURN_REASONING_RECORD gate for the Maker-fact
        # extractor reads THIS — never `plugin.maker_engine`, which is permanently
        # None in the agno process (no MakerEngine is instantiated here), so the
        # old `_me_tr.is_maker(...)` path made is_maker always False → the gate
        # never opened (BUG-MAKER-FACT-EXTRACTOR-GATE-DEAD, 2026-06-22).
        worker_plugin._current_is_maker = is_maker
        # §7.F (F.2) — hashed identity helping-signals for cross-handle merge
        # (internal-only, NEVER rendered). did_hash from the already-plumbed Privy
        # DID (claims_sub); ip_hash hashed at the api edge (raw IP never plumbed).
        # Same per-Titan salt (api.internal_key) in both processes → consistent.
        try:
            from titan_hcl.params import get_params as _gp
            from titan_hcl.utils.identity_hash import (
                derive_salt as _ds, identity_hash as _ih,
            )
            _salt = _ds((_gp("api") or {}).get("internal_key", "") or "")
            worker_plugin._current_did_hash = _ih(
                payload.get("claims_sub", "") or "", _salt)
        except Exception:
            worker_plugin._current_did_hash = ""
        worker_plugin._current_ip_hash = payload.get("ip_hash", "") or ""

        # ── Pre-LLM goal hook (Phase 7 / INV-Syn-17) ────────────────────
        # Write {text, concept_ids, ts} into the `goal` buffer and the
        # latest user message into the `perception` buffer BEFORE
        # agent.arun. Non-blocking + soft-fail per INV-Syn-17: any
        # exception logs at DEBUG and chat continues normally.
        try:
            _bc = getattr(worker_plugin, "synthesis_buffer_cache", None)
            if _bc is not None and message_text:
                _chat_id = f"{user_id}:{session_id}"
                _concept_ids = _ground_for_goal_hook(
                    worker_plugin, message_text,
                )
                _bc.set(
                    _chat_id, "goal",
                    content=message_text[:8192],
                    concept_ids=_concept_ids,
                )
                # Perception buffer = latest raw user input (arch §14).
                _bc.set(
                    _chat_id, "perception",
                    content=message_text[:8192],
                    concept_ids=_concept_ids,
                )
        except Exception as _hook_err:
            logger.debug(
                "[AgnoWorker] pre-LLM goal hook error (proceeding to "
                "chat path normally): %s", _hook_err,
            )

        # ── Run Agno agent ──
        # Concurrent multi-user chat (RFP_concurrent_multiuser_chat): flag-ON
        # (default) mints a per-call agent over the shared chat context with the
        # tier baked in (model.id/max_tokens/guidance) and arun()s it with NO
        # lock, so concurrent chats overlap. flag-OFF = legacy shared agent +
        # _route_model_for_tier lock (which classifies + swaps model.id and
        # serialises concurrent chats so they don't trample each other).
        if _concurrent_chat_enabled(worker_plugin) and \
                getattr(worker_plugin, "_chat_ctx", None) is not None:
            run_agent = _make_chat_agent(worker_plugin, message_text, agent)
            run_output = await run_agent.arun(
                message_text,
                session_id=session_id,
                user_id=user_id,
            )
        else:
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
        # Concurrent multi-user chat: run_agent (per-call Agent) + run_output
        # (RunResponse holding the enriched-prompt messages) are DEAD after the
        # content is extracted — release them now so under N concurrent chats
        # they don't linger through the ~200 lines of post-processing below
        # (shrinks the simultaneous-in-flight peak). Chat continues on
        # response_text. (No-op on the legacy branch where run_agent is unset.)
        run_agent = run_output = None

    except Exception as e:
        logger.exception("[AgnoWorker] agent.arun failed: %s", e)
        error_str = str(e)

    finally:
        stats_ref["in_flight"] = max(0, stats_ref.get("in_flight", 0) - 1)

    # ── Phase 9 strict cited gate (INV-Syn-23) ──
    # Post-LLM, decide which surfaced retrieval items the response actually
    # cited (heuristic, no extra LLM call). Emit MEMORY_RETRIEVAL_USED per
    # surfaced item — used_by_llm=True for cited (→ synthesis_worker
    # record_access reinforcement), False for surfaced-not-cited (telemetry).
    # Non-blocking soft-fail per INV-Syn-17: chat NEVER fails on a gate error.
    try:
        _reg = getattr(worker_plugin, "_last_surfaced_items", None)
        _chat_id = f"{user_id}:{session_id}"
        _surfaced = _reg.pop(_chat_id, []) if isinstance(_reg, dict) else []
        if _surfaced and response_text:
            from titan_hcl.synthesis.cited_use import (
                CitedUseDetector, SurfacedItem,
            )
            _detector = getattr(worker_plugin, "_cited_use_detector", None)
            if _detector is None:
                _detector = CitedUseDetector()
                worker_plugin._cited_use_detector = _detector
            _items = [
                SurfacedItem(
                    item_id=s.get("item_id", ""),
                    title=s.get("title", ""),
                    content_snippet=s.get("content_snippet", ""),
                    concept_ids=s.get("concept_ids", []) or [],
                    source=s.get("source", ""),
                )
                for s in _surfaced if s.get("item_id")
            ]
            # G9: prefer the cited set already computed ONCE at the OVG
            # boundary (agno PostHook, `_last_cited_use`) — avoids a SECOND
            # detect() on the chat hot path. Fall back to computing here when
            # the stash is absent (non-PostHook paths / nothing surfaced there)
            # so behaviour is never worse than pre-G9.
            _cu_stash = getattr(worker_plugin, "_last_cited_use", None)
            _cu_hit = (_cu_stash.pop(_chat_id, None)
                       if isinstance(_cu_stash, dict) else None)
            if _cu_hit is not None:
                _cited = set(_cu_hit.get("cited", []))
                _s_reply = _cu_hit.get("s_reply")
                _e_reply = _cu_hit.get("e_reply")
                _v_reply = _cu_hit.get("v_reply")
            else:
                _cited = set(_detector.detect(
                    response_text=response_text, surfaced_items=_items,
                ))
                _s_reply = None
                _e_reply = None
                _v_reply = None
            # P3 (RFP_synthesis_decision_authority) — the per-reply sovereignty
            # score S = 0.7E+0.3V + its E/V components. Prefer the values computed
            # once at the OVG boundary (stash); fall back to computing them here
            # so the meter always gets a full mark. Cheap/deterministic; never
            # blocks the path. (E/V feed the chronicle re-source 3-axis.)
            if not isinstance(_s_reply, (int, float)):
                try:
                    from titan_hcl.synthesis.sovereignty_score import (
                        compute_sovereignty_score,
                    )
                    _sov = compute_sovereignty_score(
                        response_text=response_text, surfaced_items=_items,
                        cited_item_ids=_cited, detector=_detector,
                    )
                    _s_reply, _e_reply, _v_reply = _sov.s, _sov.e, _sov.v
                except Exception:
                    _s_reply = 0.0
            # Normalize E/V to floats (absent stash / pre-E/V PostHook → 0.0).
            _e_reply = float(_e_reply) if isinstance(_e_reply, (int, float)) else 0.0
            _v_reply = float(_v_reply) if isinstance(_v_reply, (int, float)) else 0.0
            _now = time.time()
            for _it in _items:
                _send(send_queue, MEMORY_RETRIEVAL_USED, name, "all", {
                    "item_id": _it.item_id,
                    "ts": _now,
                    "used_by_llm": _it.item_id in _cited,
                })
            # Operator-closure C1 (SPEC §25.9) — ONE per-TURN knowledge-moment
            # signal: this turn needed knowledge (≥1 item surfaced) and was
            # `satisfied` iff the response cited ≥1 of them. synthesis_worker's
            # SovereigntyRatioMeter records the per-turn denominator from this,
            # replacing the per-ITEM MEMORY_RETRIEVAL_USED inflation.
            _send(send_queue, KNOWLEDGE_MOMENT, name, "all", {
                "needed": bool(_items),
                "satisfied": bool(_cited),
                "s_reply": float(_s_reply),
                "e_reply": _e_reply,
                "v_reply": _v_reply,
                "ts": _now,
                # RFP_worker_telemetry §7.C/C1 — carry THIS turn's trigger_id (minted
                # in agno's pre-hook on the shared plugin) onto the direct
                # agno→synthesis edge so `analyze --trace <id>` joins the turn's
                # agno ops to the synthesis op this event causes.
                "trigger_id": getattr(worker_plugin, "_telemetry_trigger_id", None),
                # §7.E.0 — per-Engram citation attribution: the surfaced + cited
                # tx_hash sets (item_id = tx_hash, the tx-spine recall key) so
                # synthesis_worker can resolve each → its Engram(s) and record
                # recall-citation OFF the chat hot path. INV-Syn-17 soft.
                "surfaced_tx": [_it.item_id for _it in _items],
                "cited_tx": [_it.item_id for _it in _items
                             if _it.item_id in _cited],
            })
    except Exception as _cu_err:
        logger.debug(
            "[AgnoWorker] cited-use gate error (chat unaffected): %s", _cu_err)

    # ── Operator-closure telemetry (2026-06-01) — emit the recall sample the
    # pre-hook stashed, so synthesis_worker's §18 chi/retrieval metrics reflect
    # the recall that actually ran HERE (its own evaluator/ring are idle).
    # Fire-and-forget P3, metrics-only (INV-Syn-25). Pop so it never re-emits.
    try:
        _rs = getattr(worker_plugin, "_last_retrieval_sample", None)
        worker_plugin._last_retrieval_sample = None
        if isinstance(_rs, dict):
            _send(send_queue, RETRIEVAL_SAMPLE, name, "all", {
                **_rs, "ts": time.time()})
    except Exception as _rs_err:
        logger.debug("[AgnoWorker] retrieval-sample emit failed: %s", _rs_err)

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

    # Tool-backstop activity (2026-06-01) — when a deterministic tool ran this
    # turn (PreHook force or OVG PostHook salvage), surface it so the frontend /
    # comma channel can show "Titan verified this via its sandbox" and explain
    # the extra latency. None on ordinary turns. Pop so it never leaks forward.
    tool_activity = getattr(worker_plugin, "_last_tool_activity", None)
    worker_plugin._last_tool_activity = None
    # §7.B (B.4) — the reasoning_id of a NON-verifiable turn (direct/research/IDK;
    # None otherwise). The client returns it to POST /v6/synthesis/turn_feedback so
    # a user/Maker rating attaches to THIS turn's stashed decision. Pop so it never
    # leaks forward to the next turn.
    reasoning_id = getattr(worker_plugin, "_last_reasoning_id", None)
    worker_plugin._last_reasoning_id = None

    response_payload = {
        "request_id": request_id,
        "response": response_text,
        "session_id": session_id,
        "mode": getattr(worker_plugin, "_last_execution_mode", "") or "",
        "mood": "Unknown",  # populated post-Chunk-K when mood SHM read wires in
        "state_narration": None,
        "state_snapshot": None,
        "ovg_data": ovg_data,
        "tool_activity": tool_activity,
        "reasoning_id": reasoning_id,
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

        # ── §7.B (B.4) — live progress: stash the emit-context so OUR PreHook (which
        # runs live inside arun) can stream the decided phase, and emit "thinking"
        # now. Safe metadata only (not OVG-gated); the verified content still flows
        # post-OVG below (D-SPEC-78 truth-gate untouched).
        from titan_hcl.modules.agno_hooks import _emit_stream_progress
        worker_plugin._stream_progress_ctx = {
            "send_queue": send_queue, "src": src, "rid": rid,
            "name": name, "request_id": request_id}
        _emit_stream_progress(worker_plugin, "thinking")

        # ── 1. Run agent to completion (Pre+LLM+Post-with-OVG) ──
        # Concurrent multi-user chat (RFP_concurrent_multiuser_chat): flag-ON
        # (default) = per-call agent over the shared context, NO lock; flag-OFF
        # = legacy shared agent + _route_model_for_tier lock (ζ.5, D-SPEC-79).
        if _concurrent_chat_enabled(worker_plugin) and \
                getattr(worker_plugin, "_chat_ctx", None) is not None:
            run_agent = _make_chat_agent(worker_plugin, message_text, agent)
            run_output = await run_agent.arun(
                message_text,
                session_id=session_id,
                user_id=user_id,
            )
        else:
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
        # Release the dead per-call agent + RunResponse before the post-LLM
        # processing below (see _handle_chat_request for rationale).
        run_agent = run_output = None

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

        # ── §7.B (B.4) — the turn is verified; Titan is now writing the reply. ──
        _emit_stream_progress(worker_plugin, "writing-reply")

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

        # ── §7.B (B.4) — the reasoning_id + execution mode of a NON-verifiable
        # turn (direct/research/IDK; reasoning_id None otherwise). The stream
        # done-frame carries them so the SSE relay can forward them as the
        # closing `event: meta` → the client attaches the rating footer to THIS
        # turn (POST /v6/synthesis/turn_feedback by reasoning_id) and picks the
        # per-lane scale from `mode`. Mirrors the non-stream CHAT_RESPONSE path.
        # Pop reasoning_id so it never leaks forward to the next turn.
        reasoning_id = getattr(worker_plugin, "_last_reasoning_id", None)
        worker_plugin._last_reasoning_id = None
        exec_mode = getattr(worker_plugin, "_last_execution_mode", "") or ""

        # ── 4. Final done frame with ovg_headers + reasoning_id + mode ──
        _send(send_queue, CHAT_STREAM_CHUNK, name, src, {
            "request_id": request_id,
            "chunk": "",
            "done": True,
            "ovg_headers": ovg_data or {},
            "reasoning_id": reasoning_id,
            "mode": exec_mode,
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
        worker_plugin._stream_progress_ctx = None  # §7.B (B.4) — never leak forward


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


def _build_local_tool_plugs(send_queue) -> dict:
    """Build the chat-time ToolPlug registry IN the agno process (Phase 6
    amendment, arch §11.3 / SPEC §25.5 — see HANDOFF + SPEC changelog).

    ToolPlugs are instantiated in the process that INVOKES them: chat-time
    tools live in agno_worker; dream/autonomous tools in synthesis_worker.
    There is still ONE implementation per tool (INV-12) — it is merely
    constructed where it executes. The canonical write path (INV-4) holds
    because the plug anchors its procedural TX via OuterMemoryWriter →
    TIMECHAIN_COMMIT regardless of host process.

    Rationale for local (not bus-routed to synthesis_worker):
      • cross-process — a worker cannot populate another process's plugin
        attr, so the P6.I "synthesis_worker wires this at boot" framing
        could not work for the agno-hosted chat tools;
      • G19 — the coding sandbox runs up to 30s (its own AST-gated, 256MB
        subprocess); a ≤5s work-RPC to synthesis_worker would blow the cap,
        so execution stays local;
      • §3J — the sandbox is a transient subprocess (verified torch-free at
        import + construct), so it does NOT regress agno's lean RSS.

    Scope: coding_sandbox only. `research` routes through `knowledge` when
    present but falls back to the legacy sage path, and `events_teacher` /
    `knowledge` require an injected bus-RPC invoke_fn to be functional
    (a separate gap; not wired here) — only coding_sandbox is self-contained
    and hard-fails when its plug is absent.
    """
    from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter
    from titan_hcl.synthesis.tools.coding_sandbox_tool import CodingSandboxTool

    omw = OuterMemoryWriter(send_queue, "agno_worker")

    # Operator-closure C2 (W7): the chat-time sandbox is its own truth oracle but
    # the OracleRouter lives in synthesis_worker. Ship the sandbox's pre-computed
    # verdict over the bus so it joins the dream-boundary OracleVerdictBatch flush
    # (→ §A.6 coverage). Fire-and-forget; never re-executes the sandbox.
    def _companion_verdict_sink(*, parent_tool_call_tx, oracle_id, verdict,
                                evidence_ref="", latency_ms=0,
                                parent_goal="", tool_id="",
                                decision_features=None, decision_action=None,
                                recipe_json="", result_summary="",
                                parent_skill_id=None):
        try:
            _payload = {
                "parent_tool_call_tx": parent_tool_call_tx,
                "oracle_id": oracle_id, "verdict": verdict,
                "evidence_ref": evidence_ref, "latency_ms": latency_ms,
                # EEL B1 — goal+tool so synthesis can form the (outcome,
                # task-shape) skill-score event (INV-Syn-29).
                "parent_goal": parent_goal, "tool_id": tool_id,
            }
            # §9.3 — the delegated-skill id (if this tool call executed a matched
            # skill) so synthesis fires SkillFailureTracker.record_outcome →
            # repair-fork-on-failure → bump_version. Only present on delegated runs.
            if parent_skill_id:
                _payload["parent_skill_id"] = str(parent_skill_id)
            # v1.1 — the OuterMetaPolicy decision (only set on the policy-driven
            # ToolBackstop path) → synthesis's verdict-time C1 capture writes the
            # Reasoning record + trains the policy (INV-OML-11/12).
            if decision_features is not None and decision_action is not None:
                _payload["features"] = list(decision_features)
                _payload["action"] = int(decision_action)
            # §7.E (E1.1) — carry the executable recipe to the C1 capture so the
            # leaf Reasoning(tool_use) record (and the distilled composite) can be
            # symbolically replayed by E.1 on the hot path.
            if recipe_json:
                _payload["recipe_json"] = str(recipe_json)
            if result_summary:
                _payload["result_summary"] = str(result_summary)  # §7.E E.2 literal answer
            send_queue.put_nowait({
                "type": bus.TOOL_CALL_VERDICT_RECORD,
                "src": "agno_worker", "dst": "synthesis", "ts": time.time(),
                "payload": _payload,
            })
        except Exception:
            pass

    return {"coding_sandbox": CodingSandboxTool(
        writer=omw, companion_verdict_sink=_companion_verdict_sink)}


def _init_worker_plugin_and_agent(bus_client, config: dict[str, Any], send_queue):
    """Construct WorkerPlugin + Agno Agent for this worker subprocess.

    Returns (worker_plugin, agent) tuple. Agent.arun() is the entry point
    for CHAT_REQUEST dispatch.

    Construction order matters: hooks/tools/guardrails close over plugin
    references at create_agent time, so worker_plugin MUST be fully
    initialised before create_agent runs — including synthesis_tool_plugs,
    which agno_tools.create_tools reads off the plugin (Phase 6 amendment).
    """
    from titan_hcl.modules.agno_agent_factory import (
        build_shared_chat_context, make_agent,
    )
    from titan_hcl.modules.agno_worker_plugin import WorkerPlugin

    worker_plugin = WorkerPlugin(bus_client=bus_client, config=config)
    try:
        worker_plugin.synthesis_tool_plugs = _build_local_tool_plugs(send_queue)
        logger.info(
            "[AgnoWorker] chat-time ToolPlugs wired locally: %s (Phase 6 "
            "amendment — INV-4/INV-12 preserved via OuterMemoryWriter)",
            sorted(worker_plugin.synthesis_tool_plugs.keys()),
        )
    except Exception as _tp_err:  # noqa: BLE001
        logger.warning(
            "[AgnoWorker] local ToolPlug wiring failed — coding_sandbox tool "
            "will report 'not wired': %s", _tp_err, exc_info=True,
        )
        worker_plugin.synthesis_tool_plugs = {}
    # Concurrent multi-user chat (RFP_concurrent_multiuser_chat): build the
    # heavy shared chat context ONCE (one session db + one hook/tool set), stash
    # it for the per-call chat path, and derive the shared legacy agent from the
    # SAME context (used by the flag-off path, dream replay + agent_ready).
    chat_ctx = build_shared_chat_context(
        worker_plugin, agent_config=config.get("agent"))
    worker_plugin._chat_ctx = chat_ctx
    agent = make_agent(chat_ctx, None)
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


def _keepalive_gap(in_flight: int, recent_peak: float, warm_cap: int,
                   scale: bool) -> int:
    """Pure helper (§7.A) — how many gemma warm-pings to fire this tick. We keep
    ~`recent_peak` units warm total; live chats already warm `in_flight`, so we only
    fill the GAP. `scale=False` → keep just 1 unit warm. Capped at `warm_cap`."""
    target_warm = (max(1, min(int(round(recent_peak)), warm_cap)) if scale else 1)
    return max(0, target_warm - max(0, int(in_flight)))


def _keepalive_loop(worker_plugin, stats_ref: dict,
                    stop_event: threading.Event, ka_cfg: dict) -> None:
    """Phase A — RFP_load_adaptive_inference_routing §7.A. Keeps the chat model
    (`gemma4:31b`) WARM on Ollama Cloud while the Titan is chat-active, so a
    concurrency burst never pays the ~60s cold-start tax (Ad-A verified: 180s idle →
    105s/2-timeout burst vs keepalive → 21s/0-fail). Concurrency-scaled: warms ~the
    recent-peak `in_flight` units (1 ping ≈ 1 warm unit). Off the chat hot path, soft
    (a ping failure is logged + ignored), activity-gated (no warming an idle Titan →
    no wasted cloud tokens). Runs its OWN asyncio loop — `provider.chat` is async."""
    import asyncio
    model = str(ka_cfg.get("keepalive_model", "gemma4:31b"))
    interval = float(ka_cfg.get("keepalive_interval_s", 25.0) or 25.0)
    idle_after = float(ka_cfg.get("keepalive_idle_after_s", 120.0) or 120.0)
    warm_cap = int(ka_cfg.get("ladder_warm_cap", 8) or 8)
    scale = bool(ka_cfg.get("keepalive_scale_with_load", True))
    _msgs = [{"role": "user", "content": "ping"}]

    async def _warm(provider, n):
        async def _one():
            try:
                await provider.chat(_msgs, model=model, max_tokens=8, timeout=30.0)
            except Exception:
                pass  # soft — keepalive never affects chat
        await asyncio.gather(*[_one() for _ in range(n)])

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    recent_peak = 1.0
    _fired = 0
    try:
        while not stop_event.is_set():
            try:
                provider = getattr(worker_plugin, "_inference_provider", None)
                in_flight = int(stats_ref.get("in_flight", 0) or 0)
                # decaying recent-peak of in_flight (warm for the NEXT burst, not just now)
                recent_peak = max(float(in_flight), recent_peak * 0.7)
                active = (time.time()
                          - float(stats_ref.get("last_chat_ts", 0) or 0)) < idle_after
                gap = _keepalive_gap(in_flight, recent_peak, warm_cap, scale)
                if provider is not None and active and gap > 0:
                    loop.run_until_complete(_warm(provider, gap))
                    _fired += 1
                    if _fired == 1 or _fired % 20 == 0:
                        logger.info(
                            "[AgnoWorker] keepalive #%d — warmed %d gemma unit(s) "
                            "(in_flight=%d recent_peak=%.1f)",
                            _fired, gap, in_flight, recent_peak)
            except Exception as _ka_err:  # noqa: BLE001
                logger.debug("[AgnoWorker] keepalive tick failed: %s", _ka_err)
            stop_event.wait(interval)
    finally:
        try:
            loop.close()
        except Exception:
            pass


@with_error_envelope(module_name="agno_worker", subsystem="entry", severity=_phase11_sev.FATAL)
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
    # Phase 11 §11.I.5 (Chunk 11N) — module-level readiness flags reset on
    # every entry (fork-mode re-entries inherit parent's True; spawn-mode
    # re-spawns get fresh False; explicit reset covers both).
    global _AGENT_READY, _OVG_READY, _BOOT_DEADLINE
    _AGENT_READY = False
    _OVG_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    logger.info("[AgnoWorker] Boot")
    boot_start = time.time()

    # RFP_worker_telemetry §7.C/C3+C4 — record this boot FIRST (before the
    # flusher's first 30s memory-sample → prev-run uptime + downtime come from
    # the PRIOR run) + register a size provider for the agno leak-watch (the
    # bounded recent-turns store is the natural growth companion to RssAnon's
    # +253MB/load observation). Best-effort; a fault never breaks the boot.
    try:
        from titan_hcl.logic.worker_telemetry import get_telemetry as _get_tel
        _agno_boot_tel = _get_tel("agno")
        if _agno_boot_tel is not None:
            _agno_boot_tel.record_boot("agno_boot")

            def _agno_size_provider():
                try:
                    from titan_hcl.modules import agno_hooks as _ah
                    rt = getattr(_ah, "_recent_turns", None)
                    if not rt:
                        return {"recent_turns": 0, "recent_sessions": 0}
                    return {"recent_turns": sum(len(v) for v in rt.values()),
                            "recent_sessions": len(rt)}
                except Exception:
                    return {}
            _agno_boot_tel.set_size_provider(_agno_size_provider)
    except Exception:  # noqa: BLE001
        pass

    # ── Phase 11 §11.I.5 / Chunk 11N — SHM state-slot writer (G21 per worker) ──
    # Constructed BEFORE the slow Agent + OVG init so:
    #   1. The slot publishes state="starting" immediately — titan_hcl's
    #      1Hz SHM poll sees the worker is alive while it warms.
    #   2. `state_writer.heartbeat()` keeps the slot's `last_heartbeat`
    #      fresh during the ~30s Agent build + ~30s OVG warmup so
    #      guardian_hcl's staleness detector doesn't kill the worker
    #      mid-boot (the SPEC-correct fix for the heartbeat-timeout-on-
    #      slow-boot live regression observed T3 cascade 2026-05-27).
    # The `_phase11_state` ref is also stashed below on the heartbeat thread
    # so the periodic loop can publish heartbeats to BOTH the bus (legacy)
    # and the SHM slot (Phase 11 §11.I.5 contract).
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority,
            ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name=name,
            layer="L2",
            boot_priority=BootPriority.MANDATORY,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[AgnoWorker] Phase 11 ModuleStateWriter init failed "
            "(continuing on legacy MODULE_READY path): %s", _sw_err)

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
    # NOTE: a periodic malloc_trim(0) daemon was tried here (2026-06-23) to lower
    # the RSS high-water, but malloc_trim(0) froze agno on the memory-tight devnet
    # box long enough to lapse the guardian heartbeat → shm_pid_dead restart loop
    # (every ~30s, the trim interval). REVERTED — too risky for the marginal gain
    # (RSS was bounded + the throttle is by-design back-pressure, not a kill). The
    # early-release of run_agent/run_output (below) is the kept footprint win.

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

    # ── Phase 11 §11.I.5 — start heartbeat EARLY (before Agent build) ──
    # The legacy heartbeat thread used to start AFTER Agent + OVG warmup
    # completed (~60-90s in). Under Guardian's 60s `heartbeat_timeout`
    # default that meant agno was killed on EVERY cold boot before its
    # first heartbeat fired — the heartbeat-timeout restart loop observed
    # T3 cascade 2026-05-27. Starting the heartbeat HERE (before the slow
    # init) is the SPEC-correct fix: liveness is asserted during boot,
    # state transitions to "running" only after probe passes per §11.I.2.
    # The local `stats` dict is constructed below; we forward-declare it
    # so the heartbeat closure can reference it.

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

    # ── Phase 11 §11.I.5 — heartbeat thread BEFORE the slow init ──
    # See block at line ~949 for rationale. The thread sends MODULE_HEARTBEAT
    # on the bus (legacy Guardian path) AND publishes _state_writer.heartbeat()
    # on the SHM slot (Phase 11 contract). Until _AGENT_READY + _OVG_READY
    # are True, the SHM-heartbeat is suppressed inside _heartbeat_loop so
    # the slot stays in `starting` / `booted` rather than prematurely
    # asserting `running`.
    hb_stop = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(send_queue, name, hb_stop, stats, _state_writer),
        daemon=True, name="agno-heartbeat",
    )
    hb_thread.start()
    logger.info(
        "[AgnoWorker] Heartbeat thread started EARLY (Phase 11 §11.I.5 — "
        "bus + SHM cadence covers slow Agent+OVG boot window)"
    )

    # ── WorkerPlugin + Agent construction ──
    worker_plugin: Optional[Any] = None
    agent: Optional[Any] = None
    try:
        worker_plugin, agent = _init_worker_plugin_and_agent(bus_client, config, send_queue)
        logger.info("[AgnoWorker] Agent constructed in %.0fms",
                    (time.time() - boot_start) * 1000)
        _AGENT_READY = (agent is not None and worker_plugin is not None)
    except Exception as e:
        logger.exception(
            "[AgnoWorker] Agent construction failed — chat handler will "
            "return error responses: %s", e,
        )

    # ── Phase A — keepalive warmth (RFP_load_adaptive_inference_routing §7.A) ──
    # Start the gemma-warmth daemon once the provider exists (lazily re-read each
    # tick, so boot-order is irrelevant). Flag-gated (keepalive_enabled), soft, and
    # off the chat hot path. ON for the fleet (all-flags-default-on); a single-user
    # install can disable it. The thread no-ops while the Titan is chat-idle.
    ka_stop = threading.Event()
    try:
        _ka_cfg = (get_params("inference").get("autoscale", {}) or {})
    except Exception:
        _ka_cfg = {}
    if worker_plugin is not None and bool(_ka_cfg.get("keepalive_enabled", True)):
        ka_thread = threading.Thread(
            target=_keepalive_loop,
            args=(worker_plugin, stats, ka_stop, _ka_cfg),
            daemon=True, name="agno-keepalive",
        )
        ka_thread.start()
        logger.info(
            "[AgnoWorker] keepalive daemon started (§7.A — gemma warmth, "
            "interval=%ss idle_after=%ss scale=%s)",
            _ka_cfg.get("keepalive_interval_s", 25.0),
            _ka_cfg.get("keepalive_idle_after_s", 120.0),
            _ka_cfg.get("keepalive_scale_with_load", True))

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
            _OVG_READY = True
        except Exception as _ovg_err:
            # Defense-in-depth: if eager init fails, the lazy-init path
            # still works on first chat. The warning surfaces the regression
            # without blocking boot.
            logger.warning(
                "[AgnoWorker] OVG eager-init failed (lazy retry will run "
                "on first chat — first-chat latency will spike): %s",
                _ovg_err,
            )
            # _OVG_READY stays False — probe will fail-fast rather than
            # claim a half-ready worker can serve.

    # ── Phase 13 §3J.1/§3J.2 — eager-warm the text embedder (llama.cpp) ──
    # Same anti-pattern correction as the OVG warm above (D-SPEC-138): the
    # reasoning-tier gatekeeper `state_tensor` embed (PreHook) calls
    # `worker_plugin.recorder.action_embedder.encode(...)` → get_text_embedder().
    # Warming it here keeps the embedder cold load (~3.5s settled, far worse
    # under boot contention) OFF the first reasoning chat's critical path, and
    # the fail-loud self_test ensures the zero-vector regression (§3J.1) can
    # never return silently. The runtime is now `llama-cpp-python` (2026-06-01
    # §3J.1 migration, via utils.text_embedder.get_text_embedder → LlamaCppEncoder):
    # same bge-small-en-v1.5 model + identical vectors, flat ~197 MB, torch-free.
    # (Was fastembed/ONNX, replaced because onnxruntime's CPU arena never returns
    # memory to the OS — the rss_3522mb leak. This comment was stale until
    # 2026-06-16.)
    try:
        from titan_hcl.utils.text_embedder import self_test as _emb_self_test
        _emb_t0 = time.time()
        if _emb_self_test():
            logger.info(
                "[AgnoWorker] text embedder warmed in %.0fms (llama.cpp eager "
                "init at boot — Phase 13 §3J.1)", (time.time() - _emb_t0) * 1000)
        else:
            logger.error(
                "[AgnoWorker] text embedder SELF-TEST FAILED at boot — "
                "embeddings degraded; CHECK llama-cpp-python install (§3J.1)")
    except Exception as _emb_err:  # noqa: BLE001
        logger.warning(
            "[AgnoWorker] embedder eager-warm failed (lazy retry on first "
            "chat — first reasoning-chat latency will spike): %s", _emb_err)

    # ── Phase 11 §11.I.2 — slot transition: starting → booted ──────────
    # Phase 13 §3J.3 (torch-ectomy) — boot-time guard: agno must carry NO torch
    # (the gatekeeper encode moved host-side to the recorder). If torch is
    # resident, a transitive import leaked it back in → log LOUD so we catch the
    # regression (visibility, not a hard boot-fail — a dep could pull it).
    import sys as _sys
    if "torch" in _sys.modules:
        logger.error(
            "[AgnoWorker] §3J REGRESSION — torch is resident in agno (RSS bloat "
            "+ 1GB-restart risk). The gatekeeper encode should run host-side; "
            "investigate the transitive import.")
    else:
        logger.info("[AgnoWorker] §3J torch-ectomy OK — torch not resident")

    # Both Agent + OVG eager warmup have completed (or soft-failed with
    # graceful degradation). Worker now signals "in-process scaffolding
    # done; ready to be probed". titan_hcl's 1Hz SHM poll detects this
    # transition + dispatches MODULE_PROBE_REQUEST. The handler below in
    # the recv loop runs the probe and transitions slot → "running".
    if _state_writer is not None and _AGENT_READY:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[AgnoWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[AgnoWorker] Phase 11 write_state(booted) failed "
                "(continuing on legacy MODULE_READY path): %s", _swb_err)

    # ── Phase 7 §P7.C/E — BufferCache wiring (D-SPEC-PHASE7) ─────────────
    # Per-chat in-mem cache + write-through bus emit. Sole DuckDB writer of
    # actr_buffers remains synthesis_worker (INV-Syn-16); this cache is a
    # caller-side optimization. Soft-fail mirrors the OVG warmup pattern:
    # if construction fails, the agno tools + pre-LLM goal hook degrade
    # to no-op and chat keeps working (INV-Syn-17).
    if worker_plugin is not None:
        try:
            from titan_hcl.synthesis.buffer_cache import (
                BufferCache, DEFAULT_HYDRATION_WARM_THRESHOLD_S,
            )
            from titan_hcl.bus import SYNTHESIS_BUFFER_COMMAND

            _buffers_cfg = (
                get_params("synthesis").get("buffers", {}) or {}
            )
            _warm_s = float(_buffers_cfg.get(
                "hydration_warm_threshold_s",
                DEFAULT_HYDRATION_WARM_THRESHOLD_S,
            ))
            _buffers_snapshot = os.path.join(
                os.environ.get("TITAN_DATA_DIR", "data"),
                "buffers_snapshot.json",
            )

            def _emit_buffer_command(payload: dict) -> None:
                """Publish SYNTHESIS_BUFFER_COMMAND on the kernel bus
                (dst='synthesis'). Soft-fail handled inside BufferCache."""
                _send(
                    send_queue, SYNTHESIS_BUFFER_COMMAND, name,
                    "synthesis", payload,
                )

            buffer_cache = BufferCache(
                bus_emit=_emit_buffer_command,
                snapshot_path=_buffers_snapshot,
                hydration_warm_threshold_s=_warm_s,
            )
            worker_plugin.synthesis_buffer_cache = buffer_cache
            logger.info(
                "[AgnoWorker] Phase 7 BufferCache ready — snapshot=%s "
                "warm_threshold_s=%.0f",
                _buffers_snapshot, _warm_s,
            )
        except Exception as _bc_err:
            logger.warning(
                "[AgnoWorker] Phase 7 BufferCache wiring failed (chat works "
                "but buffer tools + goal hook are no-ops): %s",
                _bc_err,
            )

    # ── Phase 8.Y fold-in: CGN lexicon loader ───────────────────────
    # Reads data/cgn_lexicon_snapshot.json into plugin.cgn_lexicon so
    # `_ground_for_goal_hook` returns real concept_ids instead of `[]`.
    # Soft-fail: missing snapshot = lexicon stays empty (no regression
    # vs pre-P8.Y).
    try:
        _lex_size = _load_cgn_lexicon(worker_plugin)
        if _lex_size > 0:
            logger.info(
                "[AgnoWorker] Phase 8.Y CGN lexicon loaded — %d entries",
                _lex_size,
            )
        else:
            logger.info(
                "[AgnoWorker] Phase 8.Y CGN lexicon not yet exported "
                "(empty/missing); _ground_for_goal_hook will return [] "
                "until cgn_worker fires CGN_LEXICON_UPDATED"
            )
    except Exception as _lex_err:
        logger.debug(
            "[AgnoWorker] Phase 8.Y CGN lexicon boot-load failed: %s",
            _lex_err,
        )

    # ── Phase 8.F — delegate_live cascade flag (D-SPEC-PHASE8) ──────
    # The `match_procedural_skill` agno tool only invokes a compiled skill
    # when delegate_live is True. Config default = false (safety); per-Titan
    # override via ~/.titan/microkernel_<id>.toml [synthesis.skill]
    # delegate_live = true (T3 canary). Wire config → plugin attr so the
    # tool's `getattr(plugin, "synthesis_delegate_live", ...)` resolves the
    # operator's intent (NOT the getattr default). Dry-run (false) means
    # the tool returns "no match" even when a skill matches.
    try:
        _skill_cfg = get_params("synthesis").get("skill", {}) or {}
        worker_plugin.synthesis_delegate_live = bool(
            _skill_cfg.get("delegate_live", False)
        )
        logger.info(
            "[AgnoWorker] Phase 8 delegate_live=%s (skill invocation %s)",
            worker_plugin.synthesis_delegate_live,
            "LIVE" if worker_plugin.synthesis_delegate_live else "DRY-RUN",
        )
    except Exception as _dl_err:
        # Conservative fallback: dry-run on any config error.
        worker_plugin.synthesis_delegate_live = False
        logger.debug(
            "[AgnoWorker] delegate_live config read failed (%s) — defaulting DRY-RUN",
            _dl_err,
        )

    # ── Synthesis EngineRecall — the CANONICAL thought-recall road (Phase E) ──
    # Build a cross-process read-only EngineRecall over the tx_hash-native
    # FaissReader (Phase A) + a lazy embedder so the chat path's SEARCH composite
    # retrieval fires. Phase E (0.24.1 / RFP §7.E): PROMOTED from flag-gated
    # augment to canonical — wired unconditionally; the PreHook runs it whenever
    # engine_recall is not None (soft-fail / block simply absent on a box without
    # the spine infra). P4 (RFP_synthesis_decision_authority): VCB and
    # memory.query BOTH enrich (disjoint by DB per §5.1); the context_assembler
    # dedups (recall-wins) + renders the one grounded block. (No
    # `synthesis_recall_augment` flag — de-flagged.)
    worker_plugin.engine_recall = None
    worker_plugin.synthesis_tx_deref = None
    try:
        _data_dir_ag = os.environ.get("TITAN_DATA_DIR", "data")

        def _agno_embedder(text: str):
            try:
                import numpy as np
                from titan_hcl.utils.text_embedder import get_text_embedder
                # Singleton returns an L2-normalized 1-D vector already.
                return np.asarray(get_text_embedder().encode(text), dtype=np.float32)
            except Exception as _emb_err:
                logger.debug("[AgnoWorker] agno_embedder failed: %s", _emb_err)
                return None

        from titan_hcl.synthesis.bridge_recall import BridgeRecall
        from titan_hcl.synthesis.recall_reader import build_recall_reader
        from titan_hcl.synthesis.snapshot_procedural_reader import (
            SnapshotProceduralReader)
        from titan_hcl.synthesis.spine_snapshot_reader import SnapshotSpineReader
        from titan_hcl.synthesis.synthesis_vector_index import FaissReader
        from titan_hcl.synthesis.tx_index_builder import TxContentDeref

        _faiss_reader = FaissReader(data_dir=_data_dir_ag)
        # RFP §7.P4 — a snapshot-backed Kuzu reader so the CHAT path runs concept-
        # AND self-granularity recall (the live Kuzu spine can't be opened here —
        # single-writer lock; this reads the atomic spine_snapshot.json, G18-pure).
        # Without it kuzu_reader=None, so BOTH concept + self recall silently fell
        # back in chat. Self-recall = the "chat resolves from the SELF node" goal.
        _snapshot_spine_reader = SnapshotSpineReader(_data_dir_ag)
        # Break F (RFP_synthesis_reuse_and_routing_revival) — the agno-side
        # procedural reader. WITHOUT it engine_recall(granularity="procedural")
        # returned None unconditionally (procedural=none in the boot log) → the
        # match_procedural_skill tool always answered "no match" AND the OML
        # skill_utility/skill_matched features never lit → skill_delegate=0
        # fleet-wide. Snapshot+faiss-backed, canonical scoring (G18-pure, no lock).
        _snapshot_procedural_reader = SnapshotProceduralReader(
            _data_dir_ag, _agno_embedder)
        worker_plugin.engine_recall = build_recall_reader(
            data_dir=_data_dir_ag,
            bridge_recall=BridgeRecall(),
            embedder=_agno_embedder,
            faiss_reader=_faiss_reader,
            kuzu_reader=_snapshot_spine_reader,
            procedural_reader=_snapshot_procedural_reader,
        )
        worker_plugin.synthesis_tx_deref = TxContentDeref(data_dir=_data_dir_ag)
        if not hasattr(worker_plugin, "_last_surfaced_items"):
            worker_plugin._last_surfaced_items = {}
        logger.info(
            "[AgnoWorker] EngineRecall wired (canonical thought-recall, Phase E; "
            "concept+self granularity via snapshot kuzu_reader) — engine=%s",
            "ready" if worker_plugin.engine_recall is not None else "none")
    except Exception as _er_err:
        logger.warning(
            "[AgnoWorker] Phase B EngineRecall wiring failed: %s — chat stays "
            "on legacy memory_context only", _er_err)
        worker_plugin.engine_recall = None

    # ── VerifiedContextBuilder — the chat path's rich inner-titan-state recall ──
    # (RFP_synthesis_decision_authority P4) The PreHook's VCB enrichment block
    # (multi-store: vocabulary/knowledge_concepts/chain_archive/meta_wisdom/…) was
    # carried over from the pre-agno chat path but its builder was NEVER wired onto
    # the agno plugin — so chat silently fell back to memory.query and the
    # sovereignty V-term (inner-state share, source=="vcb") was structurally pinned
    # at 0. Wire it here so VCB runs in chat as the RFP intends, activating V.
    # data_dir matches engine_recall; bus_emit left None (the strict Phase-9
    # cited-use gate — not VCB's soft Phase-1 emit — drives reinforcement now).
    # Soft-fail: on error chat falls back to memory.query exactly as before.
    try:
        from titan_hcl.logic.verified_context_builder import VerifiedContextBuilder
        _vcb_data_dir = os.environ.get("TITAN_DATA_DIR", "data")
        worker_plugin._verified_context_builder = VerifiedContextBuilder(
            data_dir=_vcb_data_dir)
        logger.info(
            "[AgnoWorker] VerifiedContextBuilder wired (chat inner-state recall "
            "+ sovereignty V-term; data_dir=%s)", _vcb_data_dir)
    except Exception as _vcb_err:
        worker_plugin._verified_context_builder = None
        logger.warning(
            "[AgnoWorker] VCB wiring failed: %s — chat stays on memory.query "
            "fallback (V-term inactive)", _vcb_err)

    # ── SHM publisher (G21 single-writer for agno_state.bin) ──
    try:
        publisher = AgnoStatePublisher(name=name)
        publisher.publish(stats)
        # Phase 11 §11.I.2 — MODULE_READY deleted per locked D2
        # (SHM slot state=booted is the contract now)
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

    # ── (heartbeat thread was started EARLIER above per Phase 11 §11.I.5) ──

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

    # ── Warm-infra (G20 / RFP_synthesis_decision_authority P4) ──
    # Front-load the cold-start cost of the chat hot path so the FIRST real
    # turn doesn't pay the ~712 ms cold first-query (embedder model load +
    # tx-hash-spine FAISS mmap + VCB sqlite conn-pool open). Runs in a DAEMON
    # thread kicked off AFTER the AGNO_WORKER_READY broadcast above, so it never
    # delays readiness / the boot-grace window, and it adds NO steady-state RSS
    # — it only loads, at boot, exactly what the first chat would load lazily.
    # A modest startup delay lets the co-boot burst pass first (this is a
    # daemon side-thread, never the worker/heartbeat thread). Single pass,
    # fully guarded: a warm-up failure must never affect the worker.
    def _warm_infra():
        try:
            time.sleep(3.0)   # let the co-boot CPU/IO burst subside (daemon thread)
        except Exception:
            pass
        _t0 = time.time()
        _warmed = []
        try:
            from titan_hcl.utils.text_embedder import get_text_embedder
            get_text_embedder().encode("warmup")
            _warmed.append("embedder")
        except Exception as _we:
            logger.debug("[AgnoWorker] warm embedder: %s", _we)
        try:
            _er = getattr(worker_plugin, "engine_recall", None)
            if _er is not None:
                _er.recall("warmup", k=1)        # loads spine FAISS + contract
                _warmed.append("engine_recall_faiss")
        except Exception as _we:
            logger.debug("[AgnoWorker] warm engine_recall: %s", _we)
        try:
            _vcb = getattr(worker_plugin, "_verified_context_builder", None)
            if _vcb is not None:                 # opens the persistent sqlite conn-pool
                _vcb.build(query="warmup", user_id="", max_tokens=64, max_records=1)
                _warmed.append("vcb_conn_pool")
        except Exception as _we:
            logger.debug("[AgnoWorker] warm vcb: %s", _we)
        logger.info("[AgnoWorker] warm-infra (G20) done in %.0fms: %s",
                    (time.time() - _t0) * 1000.0, ",".join(_warmed) or "none")

    try:
        import threading as _warm_threading
        _warm_threading.Thread(
            target=_warm_infra, name="agno-warm-infra", daemon=True).start()
    except Exception as _wte:
        logger.debug("[AgnoWorker] warm-infra thread spawn failed: %s", _wte)

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

            # P8.Y fold-in: cgn_worker emits this on every grounding mutation
            # + 5-min snapshot cadence. Refresh the in-process lexicon so
            # `_ground_for_goal_hook` always returns the freshest concept_ids.
            if msg_type == CGN_LEXICON_UPDATED:
                try:
                    new_size = _load_cgn_lexicon(worker_plugin)
                    logger.debug(
                        "[AgnoWorker] CGN_LEXICON_UPDATED — lexicon now %d entries",
                        new_size,
                    )
                except Exception as _lu_err:
                    logger.debug(
                        "[AgnoWorker] CGN_LEXICON_UPDATED handler failed: %s",
                        _lu_err,
                    )
                continue

            if msg_type == SAVE_NOW:
                # Republish SHM stats. Session history is already durable
                # per-write via AsyncSqliteDb, so there is no DB checkpoint to
                # force here (AUDIT §C / rFP §P2 — contract corrected). Only the
                # in-memory observability stats are ephemeral (acceptable loss).
                if publisher:
                    publisher.publish(stats)
                    last_shm_publish = time.time()
                continue

            if msg_type == KERNEL_EPOCH_TICK:
                if publisher:
                    publisher.publish(stats)
                    last_shm_publish = time.time()
                continue

            # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ─────────
            # titan_hcl dispatches this after detecting our SHM slot
            # transitioned to "booted". `handle_module_probe_request`:
            #   1. Writes state="probing" to our SHM slot
            #   2. Invokes agno_worker_probe(bus_client) which inspects
            #      _AGENT_READY + _OVG_READY module-level sentinels
            #   3. Writes state="running" + last_probe_result on pass,
            #      OR state="unhealthy" + last_error envelope on fail
            #   4. Publishes MODULE_PROBE_RESPONSE(probe_id, result) back
            from titan_hcl.bus import MODULE_PROBE_REQUEST
            if msg_type == MODULE_PROBE_REQUEST and _state_writer is not None:
                try:
                    from titan_hcl.core.probe_dispatcher import (
                        handle_module_probe_request,
                    )
                    from titan_hcl.probes.agno import agno_worker_probe
                    handle_module_probe_request(
                        msg,
                        probe_fn=agno_worker_probe,
                        send_queue=send_queue,
                        module_name=name,
                        state_writer=_state_writer,
                        bus_client=bus_client,
                    )
                except Exception as _phb_err:  # noqa: BLE001
                    logger.warning(
                        "[AgnoWorker] Phase 11 probe handler raised: %s",
                        _phb_err)
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
