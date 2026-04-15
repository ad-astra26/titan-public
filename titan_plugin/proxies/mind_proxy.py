"""
Mind Module Proxy — lazy bridge to the supervised Mind process.

Covers MoodEngine, SocialGraph, and future 5DT Mind senses.
"""
import logging
from typing import Optional

from ..bus import DivineBus
from ..guardian import Guardian

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
    Routes calls through Divine Bus to the supervised Mind module.
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        self._reply_queue = bus.subscribe("mind_proxy", reply_only=True)
        self._started = False

    def _ensure_started(self) -> None:
        # Guardian.start() blocks the asyncio event loop when called from
        # async endpoint handlers (observed 2026-04-14 T1 API hang). The
        # shared helper detects asyncio context and spawns a thread in
        # that case so the event loop stays responsive.
        from ._start_safe import ensure_started_async_safe
        ready = ensure_started_async_safe(
            self._guardian, "mind", id(self), proxy_label="MindProxy"
        )
        if ready:
            self._started = True

    def get_mood_label(self) -> str:
        """Get current mood label from MoodEngine."""
        self._ensure_started()
        reply = self._bus.request(
            "mind_proxy", "mind",
            {"action": "get_mood"},
            timeout=5.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("mood", "Unknown")
        return "Unknown"

    def get_mood_valence(self) -> float:
        """Get mood valence scalar."""
        self._ensure_started()
        reply = self._bus.request(
            "mind_proxy", "mind",
            {"action": "get_valence"},
            timeout=5.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("valence", 0.5)
        return 0.5

    def record_interaction(self, user_id: str, interaction_type: str = "chat", **kwargs) -> None:
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

    def get_or_create_user(self, user_id: str):
        """Get or create a user profile from SocialGraph."""
        self._ensure_started()
        reply = self._bus.request(
            "mind_proxy", "mind",
            {"action": "get_or_create_user", "user_id": user_id},
            timeout=5.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            profile_data = reply.get("payload", {}).get("profile", {})
            return _DictProfile(profile_data)
        return _DictProfile({"user_id": user_id})

    def should_engage(self, user_id: str) -> str:
        """Check engagement level for a user."""
        self._ensure_started()
        reply = self._bus.request(
            "mind_proxy", "mind",
            {"action": "should_engage", "user_id": user_id},
            timeout=5.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("level", "minimal")
        return "minimal"

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

    def get_current_reward(self, info_gain: float = 0.0) -> float:
        """Get RL reward from MoodEngine."""
        self._ensure_started()
        reply = self._bus.request(
            "mind_proxy", "mind",
            {"action": "get_current_reward", "info_gain": info_gain},
            timeout=5.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("reward", 0.5)
        return 0.5

    def get_mind_tensor(self) -> list:
        """Get the 5DT Mind state tensor."""
        self._ensure_started()
        reply = self._bus.request(
            "mind_proxy", "mind",
            {"action": "get_tensor"},
            timeout=5.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("tensor", [0.5] * 5)
        return [0.5] * 5
