"""
titan_plugin/logic/interface_advisor.py — InterfaceAdvisor (Step 7.2).

Flow controller sitting inside the Interface layer. Manages bidirectional
message rate limiting between Inner Self and Outer Self using per-message-type
sliding window counters.

When a message type exceeds its rate limit, the advisor:
  1. Lets the message through (never drops data)
  2. Sends RATE_LIMIT feedback to the source component

Source components can use RATE_LIMIT feedback to self-regulate via IQL.
This is TCP congestion control for consciousness signals.

Design:
  - O(1) per check (deque of timestamps, pop expired)
  - Zero allocation in hot path
  - Pure Python, no external dependencies
"""
import logging
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


# Default rate limits per message type (max messages per window)
INITIAL_LIMITS: dict[str, int] = {
    "IMPULSE": 1,           # Max 1 per window (avoid impulse spam)
    "INTERFACE_INPUT": 5,   # Max 5 per window (user chat bursts OK)
    "BODY_STATE": 12,       # Max 12 per window (every 5s tick)
    "MIND_STATE": 12,       # Same as body
    "SPIRIT_STATE": 2,      # Max 2 per window (60s publish)
    "ACTION_RESULT": 3,     # Max 3 per window
    "INTENT": 1,            # Max 1 per window
}

# Default window size in seconds
DEFAULT_WINDOW = 60.0


class InterfaceAdvisor:
    """
    Per-message-type sliding window rate limiter.

    Usage:
        advisor = InterfaceAdvisor()
        feedback = advisor.check("IMPULSE", "spirit")
        if feedback:
            # Rate limit exceeded — feedback dict for source component
            bus.publish(make_msg("RATE_LIMIT", "interface", "spirit", feedback))
    """

    def __init__(
        self,
        limits: Optional[dict[str, int]] = None,
        window: float = DEFAULT_WINDOW,
    ):
        self._limits = dict(limits or INITIAL_LIMITS)
        self._window = window
        # msg_type → deque of timestamps
        self._windows: dict[str, deque] = {}
        self._rate_limit_count = 0

    def check(self, msg_type: str, source: str = "") -> Optional[dict]:
        """
        Check if a message type has exceeded its rate limit.

        Args:
            msg_type: The bus message type being checked
            source: Source module name (for feedback routing)

        Returns:
            None if within limits, or a RATE_LIMIT feedback dict if exceeded.
        """
        limit = self._limits.get(msg_type)
        if limit is None:
            # Unknown message type — no limit configured, pass through
            return None

        now = time.time()
        cutoff = now - self._window

        # Get or create window deque
        if msg_type not in self._windows:
            self._windows[msg_type] = deque()
        window = self._windows[msg_type]

        # Pop expired entries
        while window and window[0] < cutoff:
            window.popleft()

        # Record this message
        window.append(now)

        # Check limit
        current_rate = len(window)
        if current_rate > limit:
            self._rate_limit_count += 1
            suggested_backoff = self._window / limit  # Ideal interval

            feedback = {
                "message_type": msg_type,
                "current_rate": current_rate,
                "limit": limit,
                "window_seconds": self._window,
                "suggested_backoff": round(suggested_backoff, 2),
                "source": source,
                "ts": now,
            }

            logger.debug(
                "[InterfaceAdvisor] RATE_LIMIT: %s rate=%d/%d (window=%.0fs) → %s",
                msg_type, current_rate, limit, self._window, source,
            )
            return feedback

        return None

    def set_limit(self, msg_type: str, limit: int) -> None:
        """Dynamically adjust a rate limit (e.g., from IQL learning)."""
        old = self._limits.get(msg_type)
        self._limits[msg_type] = max(1, limit)  # Minimum 1 per window
        logger.info("[InterfaceAdvisor] Limit updated: %s %s→%d",
                     msg_type, old, self._limits[msg_type])

    def get_current_rate(self, msg_type: str) -> int:
        """Return the current message count in the active window."""
        if msg_type not in self._windows:
            return 0
        now = time.time()
        cutoff = now - self._window
        window = self._windows[msg_type]
        while window and window[0] < cutoff:
            window.popleft()
        return len(window)

    def reset(self, msg_type: Optional[str] = None) -> None:
        """Reset rate tracking for a specific type or all types."""
        if msg_type:
            self._windows.pop(msg_type, None)
        else:
            self._windows.clear()

    def get_stats(self) -> dict:
        """Return advisor statistics."""
        now = time.time()
        cutoff = now - self._window
        rates = {}
        for msg_type, window in self._windows.items():
            while window and window[0] < cutoff:
                window.popleft()
            rates[msg_type] = len(window)

        return {
            "limits": dict(self._limits),
            "current_rates": rates,
            "window_seconds": self._window,
            "rate_limit_count": self._rate_limit_count,
        }
