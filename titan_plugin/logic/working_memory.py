"""
titan_plugin/logic/working_memory.py — Short-Term Working Memory Buffer.

Holds ~7 items currently being processed (Miller's Law).
Items decay over ~5 consciousness epochs unless refreshed.
Contents influence word selection, tool dispatch, and conversation.

Not persisted — ephemeral by design. Like human working memory,
it holds what you're thinking about RIGHT NOW.
"""
import logging
import time

logger = logging.getLogger(__name__)

# Miller's Law: 7 ± 2 items
DEFAULT_CAPACITY = 7
# Items decay after this many consciousness epochs
DEFAULT_DECAY_EPOCHS = 5


class WorkingMemoryItem:
    """A single item in working memory."""
    __slots__ = ("item_type", "key", "content", "epoch_added", "strength")

    def __init__(self, item_type: str, key: str, content: dict, epoch: int):
        self.item_type = item_type
        self.key = key
        self.content = content
        self.epoch_added = epoch
        self.strength = 1.0

    def age(self, current_epoch: int) -> int:
        return current_epoch - self.epoch_added

    def to_dict(self) -> dict:
        return {
            "type": self.item_type,
            "key": self.key,
            "content": self.content,
            "epoch_added": self.epoch_added,
            "strength": round(self.strength, 3),
        }


class WorkingMemory:
    """Short-term active processing buffer.

    Item types:
    - active_word: Currently learning/using word
    - active_goal: Current goal from hormonal drive
    - active_experience: Recalled experience from ex_mem
    - conversation_context: Who am I talking to?
    - dominant_emotion: Current strongest hormone
    - active_dream: Recently recalled dream insight
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY,
                 decay_epochs: int = DEFAULT_DECAY_EPOCHS):
        self.capacity = capacity
        self.decay_epochs = decay_epochs
        self._items: list[WorkingMemoryItem] = []

    def attend(self, item_type: str, key: str, content: dict, epoch: int):
        """Add or refresh item in working memory.

        If item already exists (same type+key), refresh its epoch.
        If at capacity, evict lowest-strength item.
        """
        # Check if already attending to this
        for item in self._items:
            if item.item_type == item_type and item.key == key:
                item.epoch_added = epoch
                item.strength = 1.0
                item.content = content
                return

        # At capacity — evict weakest
        if len(self._items) >= self.capacity:
            self._items.sort(key=lambda x: x.strength)
            evicted = self._items.pop(0)
            logger.debug("[WorkingMem] Evicted %s:%s (strength=%.2f)",
                         evicted.item_type, evicted.key, evicted.strength)

        self._items.append(WorkingMemoryItem(item_type, key, content, epoch))

    def decay(self, current_epoch: int):
        """Reduce strength based on age, remove expired items."""
        surviving = []
        for item in self._items:
            age = item.age(current_epoch)
            if age >= self.decay_epochs:
                logger.debug("[WorkingMem] Decayed %s:%s (age=%d)",
                             item.item_type, item.key, age)
                continue
            # Strength decreases linearly with age
            item.strength = max(0.0, 1.0 - (age / self.decay_epochs))
            surviving.append(item)
        self._items = surviving

    def get_context(self) -> list[dict]:
        """Current working memory items for enriching actions."""
        return [item.to_dict() for item in self._items]

    def get_items_by_type(self, item_type: str) -> list[dict]:
        """Get all items of a specific type."""
        return [item.to_dict() for item in self._items
                if item.item_type == item_type]

    def is_attended(self, item_type: str, key: str) -> bool:
        """Is this item currently in working memory?"""
        return any(
            item.item_type == item_type and item.key == key
            for item in self._items
        )

    def clear(self):
        """Clear all items (e.g., on major state transition)."""
        self._items.clear()

    @property
    def size(self) -> int:
        return len(self._items)

    def compare(self, idx_a: int, idx_b: int) -> float:
        """Compare two items by strength similarity. Returns 0-1."""
        if idx_a >= len(self._items) or idx_b >= len(self._items):
            return 0.0
        a = self._items[idx_a]
        b = self._items[idx_b]
        # Type match bonus + strength proximity
        type_match = 1.0 if a.item_type == b.item_type else 0.0
        strength_sim = 1.0 - abs(a.strength - b.strength)
        return (type_match * 0.4 + strength_sim * 0.6)

    def find(self, query_type: str = None) -> dict | None:
        """Find most relevant (highest strength) item, optionally filtered by type."""
        candidates = self._items
        if query_type:
            candidates = [i for i in candidates if i.item_type == query_type]
        if not candidates:
            return None
        best = max(candidates, key=lambda x: x.strength)
        return best.to_dict()

    def summary_vector(self, slots: int = 8) -> list[float]:
        """Compact representation: strength per slot (0.0 if empty).

        Used by reasoning policy network as working memory input.
        """
        vec = [0.0] * slots
        for i, item in enumerate(self._items[:slots]):
            vec[i] = item.strength
        return vec

    def get_stats(self) -> dict:
        return {
            "size": self.size,
            "capacity": self.capacity,
            "items": [f"{i.item_type}:{i.key}" for i in self._items],
        }
