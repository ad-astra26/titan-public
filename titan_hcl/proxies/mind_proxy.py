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
                              titan-docs/specs/phase_c_rpc_exemptions.yaml)
  - should_engage           → bus.request_async + 5s timeout (true
                              work-RPC: computation that needs
                              SocialGraph state — same exemption)
"""
from __future__ import annotations

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


class MindProxy:
    """
    Drop-in proxy for Mind subsystem (MoodEngine state-lookup only).

    v1.7.1 (D-SPEC-50) — SocialGraph methods removed from this proxy;
    `_proxies["social_graph"] = SocialGraphProxy(...)` now binds to a
    dedicated proxy backed by `social_graph_worker` subprocess. See
    `rFP_titan_hcl_l2_separation_strategy §4.P`.

    All remaining state-lookup paths read SHM directly (Phase C
    SPEC §10.E + Preamble G18).
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
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

    # v1.7.1 (D-SPEC-50, rFP_titan_hcl_l2_separation_strategy §4.P):
    # The 4 SocialGraph methods (`record_interaction`, `_save_profile`,
    # `get_or_create_user`, `should_engage`) + the `_work_rpc_async` /
    # `_work_rpc_sync` helpers + the `_DictProfile` class were REMOVED
    # from MindProxy. SocialGraph is now hosted by `social_graph_worker`
    # (its own subprocess) accessed via `SocialGraphProxy` —
    # `_proxies["social_graph"] = SocialGraphProxy(self.bus, self.guardian)`
    # in `plugin.py` + `legacy_core.py` (the legacy MindProxy alias was
    # DELETED in same v1.7.1 PATCH).
