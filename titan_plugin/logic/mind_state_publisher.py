"""
mind_state_publisher — Phase C Session 4 §4.B.6.

Publishes mind_state.bin from mind_worker's MoodEngine instance.
Mirrors MoodEngine.previous_mood / _prior_mood / get_mood_label() output
so consumers (MindProxy) can derive every queryable field without a sync
RPC round-trip.

Schema per SPEC §7.1 mind_state.bin:
    {
        mood_label: str,
        mood_valence: float,         # = previous_mood
        mood_intensity: float,       # alias of mood_valence (legacy)
        previous_mood: float,        # MoodEngine.previous_mood (t-1)
        prior_mood: float,           # MoodEngine._prior_mood (t-2)
        mood_delta: float,           # = previous_mood - prior_mood
        current_reward: float,       # = clamp(mood_delta + 0.0, -1.0, 2.0) base reward
        info_gain_ema: float,        # rolling EMA of recent info_gain inputs
        mood_history_digest: list,   # last N mood_label transitions (diagnostics)
        ts: float,
    }

The reward formula is:
    reward(info_gain) = clamp(mood_delta + info_gain, -1.0, 2.0)

Consumers compute info_gain-adjusted reward client-side from the published
base. This avoids G19 sync-RPC and matches MoodEngine.get_current_reward()
exactly.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional

from titan_plugin.logic.base_state_publisher import BaseStatePublisher
from titan_plugin.logic.session4_state_specs import (
    MIND_STATE_SLOT,
    MIND_STATE_SPEC,
)


_HISTORY_DIGEST_LEN = 10


class MindStatePublisher(BaseStatePublisher):
    slot_name = MIND_STATE_SLOT
    slot_spec = MIND_STATE_SPEC

    def __init__(self, titan_id: str):
        super().__init__(titan_id)
        self._label_history: deque = deque(maxlen=_HISTORY_DIGEST_LEN)
        self._last_label: Optional[str] = None

    def _compute_payload(self, mood_engine: Any) -> dict[str, Any]:
        if mood_engine is None:
            # Cold-boot stub: SPEC-stable defaults so consumers see a
            # well-formed slot before mood_engine init completes.
            return {
                "mood_label": "Unknown",
                "mood_valence": 0.5,
                "mood_intensity": 0.5,
                "previous_mood": 0.5,
                "prior_mood": 0.5,
                "mood_delta": 0.0,
                "current_reward": 0.0,
                "info_gain_ema": 0.0,
                "mood_history_digest": [],
                "ts": time.time(),
            }

        prev = float(getattr(mood_engine, "previous_mood", 0.5) or 0.5)
        prior = float(getattr(mood_engine, "_prior_mood", prev) or prev)
        delta = prev - prior
        # Base reward (info_gain=0) — same clamp as
        # MoodEngine.get_current_reward(0.0)
        base_reward = max(-1.0, min(2.0, delta))

        try:
            label = mood_engine.get_mood_label()
        except Exception:
            label = "Unknown"

        # Track label transitions for diagnostics
        if label != self._last_label:
            self._label_history.append(
                {"label": label, "valence": prev, "ts": time.time()})
            self._last_label = label

        info_gain_ema = float(
            getattr(mood_engine, "info_gain_ema", 0.0) or 0.0)

        return {
            "mood_label": label,
            "mood_valence": prev,
            "mood_intensity": prev,
            "previous_mood": prev,
            "prior_mood": prior,
            "mood_delta": delta,
            "current_reward": base_reward,
            "info_gain_ema": info_gain_ema,
            "mood_history_digest": list(self._label_history),
            "ts": time.time(),
        }
