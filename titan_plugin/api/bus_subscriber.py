"""
titan_plugin/api/bus_subscriber.py — translates kernel bus events into
api-side cached state.

Microkernel v2 Phase A §A.4 S5 amendment (2026-04-25).

Listens for bus messages on the api_subprocess recv_queue and updates the
CachedState dict. Per Q4 (PLAN v2): pure event-driven, no TTL — the
kernel publishes a *_UPDATED event on every relevant state change; this
subscriber is a passive consumer.

Bootstrap protocol:
  1. api_subprocess starts → sends STATE_SNAPSHOT_REQUEST via send_queue
  2. kernel observes request, gathers current plugin state, replies via
     STATE_SNAPSHOT_RESPONSE bus message (dst=`api`, payload=full state)
  3. subscriber consumes response → bulk_update on CachedState →
     marks bootstrap_done event → endpoint code unblocks

Per-event mapping (kernel emits; subscriber consumes):
  - SOLANA_BALANCE_UPDATED       → cached_state["network.balance"]
  - GUARDIAN_STATUS_UPDATED      → cached_state["guardian.status"]
  - AGENCY_STATS_UPDATED         → cached_state["agency.stats"]
  - REASONING_STATS_UPDATED      → cached_state["reasoning.stats"]
  - DREAMING_STATE_UPDATED       → cached_state["dreaming.state"]
  - CGN_STATS_UPDATED            → cached_state["cgn.stats"]
  - LANGUAGE_STATS_UPDATED       → cached_state["language.stats"]
  - META_TEACHER_STATS_UPDATED   → cached_state["meta_teacher.stats"]
  - SOCIAL_STATS_UPDATED         → cached_state["social.stats"]
  - SOUL_STATE_UPDATED           → cached_state["soul.state"]  (incl. maker_pubkey, nft_address, current_gen)
  - NETWORK_INFO_UPDATED         → cached_state["network.info"]
  - CONFIG_LOADED                → cached_state["config.full"]  (once at boot)

This file does NOT spawn its own thread. It exposes `handle_message(msg)`
that the existing api_subprocess `_bus_listener_loop` calls. Keeps the
single-listener-thread invariant.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from titan_plugin.api.cached_state import CachedState

logger = logging.getLogger(__name__)


# ── Bus message type names ────────────────────────────────────────────

STATE_SNAPSHOT_REQUEST = "STATE_SNAPSHOT_REQUEST"
STATE_SNAPSHOT_RESPONSE = "STATE_SNAPSHOT_RESPONSE"

# Mapping: (kernel-emitted event_type) → (cache key)
#
# DERIVED from cache_key_registry.REGISTRY (single source of truth) per
# rFP_observatory_data_loading_v1 Phase 1. To add or change an entry,
# edit titan_plugin/api/cache_key_registry.py — never edit this map by hand.
# Audit: `python scripts/arch_map.py cache-keys --audit`.
from titan_plugin.api.cache_key_registry import (  # noqa: E402
    EVENT_TO_CACHE_KEY as _REGISTRY_EVENT_TO_CACHE_KEY,
)
EVENT_TO_CACHE_KEY: dict[str, str] = dict(_REGISTRY_EVENT_TO_CACHE_KEY)


class BusSubscriber:
    """Passive consumer of kernel-side state-update bus messages.

    Owned by the api_subprocess; updates the CachedState that the
    TitanStateAccessor reads from.
    """

    def __init__(
        self,
        cached_state: CachedState,
        send_queue: Any | None = None,
    ) -> None:
        """
        Args:
          cached_state: target cache to update
          send_queue: api_subprocess send_queue for emitting
                      STATE_SNAPSHOT_REQUEST. None disables bootstrap
                      (test fixtures construct without a queue).
        """
        self._cache = cached_state
        self._send_queue = send_queue
        self._bootstrap_sent = False
        self._handlers: dict[str, Callable[[dict], None]] = {}
        self._install_default_handlers()

    # -- handler registration -----------------------------------------

    def _install_default_handlers(self) -> None:
        for event_type, cache_key in EVENT_TO_CACHE_KEY.items():
            self._handlers[event_type] = self._make_simple_handler(cache_key)
        # Bootstrap response handler is special — populates many keys
        self._handlers[STATE_SNAPSHOT_RESPONSE] = self._handle_snapshot_response

    def _make_simple_handler(self, cache_key: str) -> Callable[[dict], None]:
        def handler(payload: dict) -> None:
            self._cache.set(cache_key, payload)
        return handler

    def register(self, event_type: str, handler: Callable[[dict], None]) -> None:
        """Register a custom handler — overrides default. Used for tests +
        future extensions."""
        self._handlers[event_type] = handler

    # -- bootstrap ----------------------------------------------------

    def request_snapshot(self) -> None:
        """Send STATE_SNAPSHOT_REQUEST to the kernel. Idempotent — once
        sent, won't re-send unless reset_bootstrap() is called."""
        if self._bootstrap_sent:
            return
        if self._send_queue is None:
            logger.info(
                "[BusSubscriber] bootstrap skipped (no send_queue) — "
                "test fixture mode")
            return
        try:
            from titan_plugin.bus import make_msg
            msg = make_msg(
                STATE_SNAPSHOT_REQUEST,
                "api",
                "all",
                {"requested_at_ns": time.time_ns()},
            )
            self._send_queue.put(msg)
            self._bootstrap_sent = True
            logger.info("[BusSubscriber] STATE_SNAPSHOT_REQUEST sent to kernel")
        except Exception as e:
            logger.warning(
                "[BusSubscriber] failed to send snapshot request: %s", e)

    def _handle_snapshot_response(self, payload: dict) -> None:
        """Bulk-update the cache from the kernel's full-state snapshot."""
        if not isinstance(payload, dict):
            logger.warning(
                "[BusSubscriber] invalid snapshot payload type: %s",
                type(payload).__name__)
            return
        # Snapshot is a flat dict of {cache_key: value}. Kernel decides
        # which keys to include; we just trust it.
        self._cache.bulk_update(payload)
        self._cache.mark_bootstrap_done()
        logger.info(
            "[BusSubscriber] bootstrap snapshot applied (%d keys)",
            len(payload))

    # -- main entry point --------------------------------------------

    def handle_message(self, msg: dict) -> bool:
        """Process one bus message. Returns True if handled (i.e. subscriber
        owns this event_type), False otherwise (caller may dispatch
        elsewhere — e.g. OBSERVATORY_EVENT for websocket bridge).

        Called from the api_subprocess _bus_listener_loop.
        """
        if not isinstance(msg, dict):
            return False
        msg_type = msg.get("type")
        if msg_type not in self._handlers:
            return False
        payload = msg.get("payload", {})
        try:
            self._handlers[msg_type](payload)
        except Exception as e:
            logger.warning(
                "[BusSubscriber] handler error for %s: %s",
                msg_type, e)
        return True

    # -- diagnostic ---------------------------------------------------

    def event_types(self) -> list[str]:
        """List of event types this subscriber consumes."""
        return list(self._handlers.keys())
