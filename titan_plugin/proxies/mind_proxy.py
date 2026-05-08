"""
Mind Module Proxy — SHM-direct bridge to the supervised Mind process.

Phase C Session 4 (rFP_phase_c_async_shm_consumer_migration §4.C.2):
state-lookup methods read mind_state.bin / inner_mind_15d.bin via
StateRegistryReader (Preamble G18 — state transport is SHM, never bus).

Methods classified per Preamble G19 + §1.B:
  - get_mood_label          → SHM read mind_state.bin
  - get_mood_valence        → SHM read mind_state.bin
  - get_current_reward      → SHM read mind_state.bin (caller-side
                              info_gain scaling — same formula as
                              MoodEngine.get_current_reward)
  - get_mind_tensor         → SHM read inner_mind_15d.bin (first 5 dims
                              are current_5d per SPEC §23.5)
  - record_interaction      → fire-and-forget publish (already non-blocking)
  - _save_profile           → fire-and-forget publish (already non-blocking)
  - get_or_create_user      → bus.request_async + 5s timeout (true
                              work-RPC: write/upsert per Preamble G19;
                              exemption rationale in
                              titan-docs/phase_c_rpc_exemptions.yaml)
  - should_engage           → bus.request_async + 5s timeout (true
                              work-RPC: computation that needs
                              SocialGraph state — same exemption)
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

import msgpack
import numpy as np

from ..bus import DivineBus
from ..core.state_registry import (
    INNER_MIND_15D,
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)
from ..guardian import Guardian
from ..logic.session4_state_specs import MIND_STATE_SLOT, MIND_STATE_SPEC

logger = logging.getLogger(__name__)


class _DictProfile:
    """Lightweight profile wrapper — attribute access over a dict from bus response."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name):
        if name == '_data':
            raise AttributeError
        return self._data.get(name)

    def __setattr__(self, name, value):
        if name == '_data':
            super().__setattr__(name, value)
        else:
            self._data[name] = value


class MindProxy:
    """
    Drop-in proxy for Mind subsystems (MoodEngine, SocialGraph).

    State-lookup paths read SHM directly (Phase C SPEC §10.E). True
    work-RPC paths use bus.request_async + bounded timeout (per-site
    rationale documented in phase_c_rpc_exemptions.yaml).
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        # Reply queue retained for the 2 work-RPC methods that still use
        # bus.request_async (get_or_create_user, should_engage).
        self._reply_queue = bus.subscribe("mind_proxy", reply_only=True)
        self._started = False

        # SHM-direct readers
        self._titan_id = resolve_titan_id()
        self._shm_root: Path = ensure_shm_root(self._titan_id)
        self._r_mind_state = StateRegistryReader(MIND_STATE_SPEC, self._shm_root)
        self._r_inner_mind = StateRegistryReader(INNER_MIND_15D, self._shm_root)

        self._fallback_counts: dict[str, int] = {}

        logger.info(
            "[MindProxy] initialized SHM-direct readers — titan_id=%s "
            "shm_root=%s (mind_state.bin + inner_mind_15d.bin per "
            "Preamble G18)", self._titan_id, self._shm_root)

    # ── Helpers ──────────────────────────────────────────────────────

    def _read_mind_state(self) -> Optional[dict]:
        try:
            raw = self._r_mind_state.read_variable()
        except Exception as e:
            self._track_fallback("mind_state",
                                 f"read_raised:{type(e).__name__}")
            return None
        if raw is None:
            self._track_fallback("mind_state", "shm_unavailable")
            return None
        try:
            decoded = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            self._track_fallback("mind_state",
                                 f"decode_raised:{type(e).__name__}")
            return None
        return decoded if isinstance(decoded, dict) else None

    def _track_fallback(self, slot_name: str, reason: str) -> None:
        prev = self._fallback_counts.get(slot_name, 0)
        self._fallback_counts[slot_name] = prev + 1
        if prev == 0:
            logger.info(
                "[MindProxy] FIRST FALLBACK slot=%s reason=%s — "
                "consumer uses default until producer first publish",
                slot_name, reason)

    def _ensure_started(self) -> None:
        # Guardian.start() blocks the asyncio event loop when called from
        # async endpoint handlers. The shared helper detects asyncio
        # context and spawns a thread in that case so the event loop
        # stays responsive.
        from ._start_safe import ensure_started_async_safe
        ready = ensure_started_async_safe(
            self._guardian, "mind", id(self), proxy_label="MindProxy"
        )
        if ready:
            self._started = True

    # ── State-lookup methods (SHM-direct) ────────────────────────────

    def get_mood_label(self) -> str:
        """Get current mood label from MoodEngine via SHM."""
        decoded = self._read_mind_state()
        if decoded is None:
            return "Unknown"
        return str(decoded.get("mood_label", "Unknown"))

    def get_mood_valence(self) -> float:
        """Get mood valence scalar via SHM."""
        decoded = self._read_mind_state()
        if decoded is None:
            return 0.5
        return float(decoded.get("mood_valence", 0.5))

    def get_current_reward(self, info_gain: float = 0.0) -> float:
        """Get RL reward (info_gain-adjusted) via SHM.

        Reward formula matches MoodEngine.get_current_reward:
          reward = clamp(mood_delta + info_gain, -1.0, 2.0)

        We publish mood_delta in mind_state.bin and apply the caller-
        provided info_gain client-side (preserves per-call parameterization
        without sync RPC).
        """
        decoded = self._read_mind_state()
        if decoded is None:
            return 0.5
        mood_delta = float(decoded.get("mood_delta", 0.0))
        reward = mood_delta + float(info_gain)
        return max(-1.0, min(2.0, reward))

    def get_mind_tensor(self) -> list:
        """Get the 5DT Mind state tensor.

        SPEC §23.5: dims [0:5] of inner_mind_15d ARE current_5d (the
        mind_worker._collect_mind_tensor output). Read those directly.
        """
        try:
            arr = self._r_inner_mind.read()
        except Exception as e:
            self._track_fallback("inner_mind_15d",
                                 f"read_raised:{type(e).__name__}")
            return [0.5] * 5
        if arr is None:
            self._track_fallback("inner_mind_15d", "shm_unavailable")
            return [0.5] * 5
        try:
            tensor = arr.view(np.float32)
            if len(tensor) >= 5:
                return [float(x) for x in tensor[:5]]
        except Exception as e:
            self._track_fallback("inner_mind_15d",
                                 f"decode_raised:{type(e).__name__}")
        return [0.5] * 5

    # ── Fire-and-forget publishes (already non-blocking) ─────────────

    def record_interaction(self, user_id: str, interaction_type: str = "chat",
                           **kwargs) -> None:
        """Record a social interaction in the SocialGraph."""
        self._ensure_started()
        self._bus.publish({
            "type": "QUERY",
            "src": "mind_proxy",
            "dst": "mind",
            "ts": __import__("time").time(),
            "rid": None,  # Fire-and-forget
            "payload": {
                "action": "record_interaction",
                "user_id": user_id,
                "interaction_type": interaction_type,
                **kwargs,
            },
        })

    def _save_profile(self, profile) -> None:
        """Save profile back (fire-and-forget via bus)."""
        self._ensure_started()
        data = profile._data if hasattr(profile, '_data') else {}
        self._bus.publish({
            "type": "QUERY",
            "src": "mind_proxy",
            "dst": "mind",
            "ts": __import__("time").time(),
            "rid": None,
            "payload": {"action": "save_profile", "profile": data},
        })

    # ── Work-RPC methods (bus.request_async + bounded timeout) ───────
    #
    # These are NOT state-lookup — they execute work in the producer
    # process (write/upsert + computed engagement decision needing
    # SocialGraph state). Per Preamble G19 + §1.B these MUST use
    # bus.request_async with explicit timeout ≤ 5s. Synchronous
    # convenience wrappers below run the async path on the event loop
    # (or a thread if called from async context).

    async def _work_rpc_async(self, action: str, extra: dict | None = None,
                              timeout: float = 5.0) -> dict:
        """Single async work-RPC primitive."""
        self._ensure_started()
        payload = {"action": action}
        if extra:
            payload.update(extra)
        try:
            reply = await self._bus.request_async(
                "mind_proxy", "mind", payload,
                timeout=timeout, reply_queue=self._reply_queue,
            )
            return reply.get("payload", {}) if reply else {}
        except Exception as e:
            logger.warning(
                "[MindProxy] %s async work-RPC raised (timeout=%.1fs): %s",
                action, timeout, e)
            return {}

    def _work_rpc_sync(self, action: str, extra: dict | None = None,
                       timeout: float = 5.0) -> dict:
        """Sync wrapper around the async path. If we're already inside
        an asyncio event loop, fall back to the legacy sync bus.request
        path (with the same bounded timeout) — bus.request_async cannot
        be awaited synchronously inside a running loop without
        asyncio.run_coroutine_threadsafe + the proxy's loop reference,
        which the proxy doesn't keep. The legacy path is acceptable for
        these two work-RPC methods because: (1) they have a real timeout
        (no infinite sock.sendall blocking — DivineBus.request honours
        timeout for the outbound send when a broker is in-process); and
        (2) the deadlock surface that motivates G19 is for state lookup
        on hot paths, not for low-frequency user-record writes."""
        self._ensure_started()
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if not in_loop:
            try:
                return asyncio.run(
                    self._work_rpc_async(action, extra, timeout=timeout))
            except Exception as e:
                logger.warning(
                    "[MindProxy] %s asyncio.run failed: %s — falling "
                    "back to bounded sync bus.request",
                    action, e)

        # Loop-running fallback: legacy sync request with explicit
        # timeout. Allow-listed in phase_c_rpc_exemptions.yaml as a
        # work-RPC site (not state lookup) — G19 exemption.
        payload = {"action": action}
        if extra:
            payload.update(extra)
        reply = self._bus.request(
            "mind_proxy", "mind", payload,
            timeout=timeout, reply_queue=self._reply_queue,
        )
        return reply.get("payload", {}) if reply else {}

    def get_or_create_user(self, user_id: str):
        """Get or create a user profile from SocialGraph.

        Work-RPC (write/upsert) — NOT state lookup.
        """
        result = self._work_rpc_sync(
            "get_or_create_user", {"user_id": user_id})
        profile_data = result.get("profile", {}) if isinstance(
            result, dict) else {}
        if not isinstance(profile_data, dict):
            profile_data = {"user_id": user_id}
        return _DictProfile(profile_data)

    def should_engage(self, user_id: str) -> str:
        """Check engagement level for a user.

        Work-RPC (computed against SocialGraph) — NOT state lookup.
        """
        result = self._work_rpc_sync(
            "should_engage", {"user_id": user_id})
        if isinstance(result, dict):
            level = result.get("level")
            if isinstance(level, str):
                return level
        return "minimal"
