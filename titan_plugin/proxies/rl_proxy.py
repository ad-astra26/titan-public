"""
RL/Sage Module Proxy — lazy bridge to the supervised RL process.

Provides interfaces matching SageScholar, SageGatekeeper, and SageRecorder,
routing all calls through the Divine Bus to the RL module process.
This proxy alone saves ~2GB RSS by keeping TorchRL mmap out of Core.
"""
import logging
from typing import Optional

from ..bus import DivineBus
from ..guardian import Guardian

logger = logging.getLogger(__name__)


class RLProxy:
    """
    Drop-in proxy for RL subsystems (Scholar, Gatekeeper, Recorder).
    Delegates to the supervised RL module via Divine Bus.
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        self._reply_queue = bus.subscribe("rl_proxy", reply_only=True)
        self._started = False

    def _ensure_started(self) -> None:
        # Async-safe Guardian.start() — see _start_safe.py for rationale.
        from ._start_safe import ensure_started_async_safe
        if ensure_started_async_safe(
            self._guardian, "rl", id(self), proxy_label="RLProxy"
        ):
            self._started = True

    def evaluate(self, state_tensor: list, prompt: str = "") -> dict:
        """
        Gatekeeper evaluation — returns execution mode + advantage score.
        Replaces SageGatekeeper.evaluate().
        """
        self._ensure_started()
        reply = self._bus.request(
            "rl_proxy", "rl",
            {"action": "evaluate", "state": state_tensor, "prompt": prompt},
            timeout=10.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        # Fallback: Shadow mode (defer to LLM)
        return {"mode": "Shadow", "advantage": 0.0, "confidence": 0.0}

    def record_transition(self, observation: list, action: int, reward: float, next_obs: list, done: bool) -> int:
        """Record an RL transition. Replaces SageRecorder.record()."""
        self._ensure_started()
        reply = self._bus.request(
            "rl_proxy", "rl",
            {
                "action": "record",
                "observation": observation,
                "action_idx": action,
                "reward": reward,
                "next_observation": next_obs,
                "done": done,
            },
            timeout=5.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("transition_id", -1)
        return -1

    def get_stats(self) -> dict:
        """Get RL training stats."""
        self._ensure_started()
        reply = self._bus.request(
            "rl_proxy", "rl",
            {"action": "stats"},
            timeout=5.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        return {}
