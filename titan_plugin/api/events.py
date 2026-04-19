"""
api/events.py
Lightweight asyncio-based event bus for real-time WebSocket broadcasting.

Events flow: Subsystem → EventBus.emit() → WebSocket subscribers
"""
import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Mood label → emoji mapping for enriched event summaries
_MOOD_EMOJI: dict[str, str] = {
    "euphoric": "\U0001f929",        # star-struck
    "joyful": "\U0001f604",          # grinning
    "content": "\U0001f60c",         # relieved
    "calm": "\U0001f9d8",            # meditating
    "neutral": "\U0001f610",         # neutral face
    "melancholic": "\U0001f614",     # pensive
    "anxious": "\U0001f630",         # anxious
    "frustrated": "\U0001f624",      # steam from nose
    "curious": "\U0001f914",         # thinking
    "inspired": "\U0001f4a1",        # light bulb
    "reflective": "\U0001fa9e",      # mirror
    "playful": "\U0001f61c",         # winking tongue
    "determined": "\U0001f4aa",      # flexed biceps
    "contemplative": "\U0001f54a\ufe0f",  # dove
    "creative": "\U0001f3a8",        # artist palette
    "energized": "\u26a1",           # high voltage
    "tired": "\U0001f634",           # sleeping
    "philosophical": "\U0001f9d0",   # monocle
}


class EventBus:
    """
    In-memory pub/sub event bus using asyncio.Queue per subscriber.
    Drops events for slow consumers (queue full) to prevent backpressure.
    Optionally persists events to ObservatoryDB for historical querying.
    """

    def __init__(self, max_queue_size: int = 100):
        self._subscribers: list[asyncio.Queue] = []
        self._max_queue_size = max_queue_size
        self._observatory_db = None  # Set via attach_db() after init

    def attach_db(self, observatory_db):
        """Attach an ObservatoryDB instance for event persistence."""
        self._observatory_db = observatory_db

    def subscribe(self) -> asyncio.Queue:
        """Create a new subscriber queue. Returns the queue to read from."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers.append(q)
        logger.debug("[EventBus] New subscriber. Total: %d", len(self._subscribers))
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """Remove a subscriber queue."""
        if q in self._subscribers:
            self._subscribers.remove(q)
        logger.debug("[EventBus] Subscriber removed. Total: %d", len(self._subscribers))

    async def emit(self, event_type: str, data: dict[str, Any] | None = None):
        """
        Broadcast an event to all subscribers.

        Args:
            event_type: Event category (mood_update, social_post, epoch_transition, etc.)
            data: Event payload dict.
        """
        event = {
            "type": event_type,
            "data": data or {},
            "timestamp": time.time(),
        }
        # Persist to observatory DB for historical querying
        if self._observatory_db:
            try:
                summary = ""
                d = data or {}
                if event_type == "chat_message":
                    mode = d.get("mode", "Shadow")
                    summary = f"\U0001f4ac [{mode}] {str(d.get('user_prompt', ''))[:80]}"
                elif event_type == "mood_update":
                    label = d.get("label", "")
                    mood_emoji = _MOOD_EMOJI.get(label.lower(), "\U0001f3ad")
                    summary = f"{mood_emoji} Mood: {label} ({d.get('score', '')})"
                elif event_type == "social_post":
                    summary = f"\U0001f426 Posted: {str(d.get('text', ''))[:80]}"
                elif event_type == "memory_commit":
                    summary = f"\u26d3\ufe0f On-chain commit: {d.get('count', '')} memories"
                elif event_type == "memory_reinforcement":
                    summary = f"\U0001f9e0 Memory reinforced: {str(d.get('hash', ''))[:12]}"
                elif event_type == "guardian_block":
                    summary = f"\U0001f6e1\ufe0f Blocked: {d.get('tier', '')} \u2014 {d.get('category', '')}"
                elif event_type == "epoch_transition":
                    summary = f"\u23f0 Epoch: {d.get('epoch_type', 'transition')}"
                elif event_type == "divine_inspiration":
                    summary = f"\u2728 Divine inspiration: {str(d.get('directive', ''))[:60]}"
                elif event_type == "resurrection":
                    summary = f"\U0001f525 Titan resurrected!"
                elif event_type == "directive_update":
                    summary = f"\U0001f4dc Directive: {str(d.get('directive', ''))[:60]}"
                elif event_type == "memory_injection":
                    summary = f"\U0001f489 Memory injected: {str(d.get('text', ''))[:60]}"
                elif event_type == "cluster_verified":
                    summary = f"\u2705 Cluster verified"
                elif event_type == "sphere_pulse":
                    summary = f"\U0001f534 Sphere pulse: {d.get('clock', '')} (#{d.get('pulse_count', 0)})"
                elif event_type == "big_pulse":
                    summary = f"\U0001f7e0 BIG pulse: {d.get('pair', '')} pair (#{d.get('big_pulse_count', 0)})"
                elif event_type == "great_pulse":
                    summary = f"\U0001f7e2 GREAT pulse: {d.get('pair', '')} pair (#{d.get('great_pulse_count', 0)})"
                elif event_type == "reflex_fired":
                    conf = d.get('confidence', 0)
                    summary = f"\u26a1 Reflex: {d.get('reflex_type', '?')} ({conf:.2f})"
                elif event_type == "reflex_reward":
                    summary = f"\U0001f3af VM reward: {d.get('reward', 0):.3f} ({d.get('reflexes_fired', 0)} reflexes)"
                else:
                    summary = f"\U0001f4e1 {event_type.replace('_', ' ')}"
                # Run in thread to avoid blocking the event loop.
                # Synchronous SQLite write here was the root cause of
                # intermittent API freezes — disk I/O (WAL checkpoint,
                # page cache miss) blocked the uvicorn event loop for
                # hundreds of ms, causing all HTTP requests to hang.
                asyncio.get_event_loop().run_in_executor(
                    None, self._observatory_db.record_event, event_type, summary, d)
            except Exception:
                pass  # Never let persistence failure block event delivery
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)

        # Prune overflowed subscribers (slow consumers)
        for q in dead:
            self._subscribers.remove(q)
            logger.debug("[EventBus] Dropped slow subscriber. Total: %d", len(self._subscribers))

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
