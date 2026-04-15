"""
titan_plugin/logic/arc/state_memory.py — ARC-AGI-3 State-Action Memory Graph.

Lightweight graph of visited states and transitions that persists across
episodes. Enables cross-episode learning: Titan remembers which actions
in which states led to progress, novel states, or level completion.

Maps to INTUITION program: pattern-based replay of successful sequences.
Maps to Cognee graph memory: structured knowledge from experience.
"""
import hashlib
import json
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class StateActionMemory:
    """
    Graph of visited states and their transitions.

    Each state is a hash of the game frame. Each transition records
    which action led to which next state and what reward was received.
    Enables:
      - Suggesting known-good actions (replay successful paths)
      - Prioritizing unexplored actions (novelty detection)
      - Counting visits for exploration bonus calculation
    """

    def __init__(self, max_states: int = 50000):
        self._max_states = max_states
        # state_hash → {action_id: [(next_hash, reward), ...]}
        self._transitions: dict[str, dict[int, list[tuple[str, float]]]] = {}
        # state_hash → visit count
        self._visit_counts: dict[str, int] = {}
        # state_hash → best reward ever seen from this state
        self._best_rewards: dict[str, float] = {}
        # States that led to level completion
        self._level_completion_states: set[str] = set()
        # Action sequences that led to level completion (last N actions before level-up)
        self._winning_sequences: list[list[int]] = []
        # Stats
        self._total_records = 0
        self._new_this_episode = 0

    def hash_state(self, grid: np.ndarray) -> str:
        """Hash a grid frame to a compact state identifier."""
        return hashlib.md5(grid.tobytes()).hexdigest()[:12]

    def record(self, state_hash: str, action: int,
               next_hash: str, reward: float) -> bool:
        """
        Record a state transition.

        Returns True if this is a novel transition (new state discovered).
        """
        # Initialize state if new
        if state_hash not in self._transitions:
            if len(self._transitions) >= self._max_states:
                self._evict_oldest()
            self._transitions[state_hash] = {}

        # Record transition
        if action not in self._transitions[state_hash]:
            self._transitions[state_hash][action] = []

        transitions = self._transitions[state_hash][action]
        # Keep only last 3 outcomes per (state, action) pair
        if len(transitions) >= 3:
            transitions.pop(0)
        transitions.append((next_hash, reward))

        # Update visit counts for BOTH states
        self._visit_counts[state_hash] = self._visit_counts.get(state_hash, 0) + 1
        # Mark next_hash as seen (with 0 visits until it becomes a state_hash)
        is_novel = next_hash not in self._visit_counts
        if is_novel:
            self._visit_counts[next_hash] = 0

        # Track best reward
        if reward > self._best_rewards.get(state_hash, -999.0):
            self._best_rewards[state_hash] = reward

        self._total_records += 1
        if is_novel:
            self._new_this_episode += 1

        return is_novel

    def record_level_completion(self, state_hash: str,
                                 recent_actions: list[int]) -> None:
        """Record that a level was completed from this state."""
        self._level_completion_states.add(state_hash)
        # Save last 20 actions before level-up as a winning sequence
        if recent_actions:
            seq = recent_actions[-20:]
            self._winning_sequences.append(seq)
            # Keep only last 10 winning sequences
            if len(self._winning_sequences) > 10:
                self._winning_sequences = self._winning_sequences[-10:]

    def suggest_action(self, state_hash: str,
                       available_actions: list[int]) -> Optional[int]:
        """
        Suggest action based on accumulated knowledge.

        Priority:
        1. Action that previously led to level completion
        2. Action that led to highest-reward next state
        3. Action that led to least-visited next state (explore)
        4. None (let ActionMapper decide)
        """
        if state_hash not in self._transitions:
            return None

        known = self._transitions[state_hash]
        available_known = {a: t for a, t in known.items() if a in available_actions}

        if not available_known:
            return None

        # Priority 1: action that led to a level-completion state
        for action, transitions in available_known.items():
            for next_hash, reward in transitions:
                if next_hash in self._level_completion_states:
                    return action

        # Priority 2: action with highest average reward
        best_action = None
        best_avg_reward = -999.0
        for action, transitions in available_known.items():
            avg_r = sum(r for _, r in transitions) / len(transitions)
            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                best_action = action

        if best_avg_reward > 0.0:
            return best_action

        # Priority 3: action leading to least-visited next state
        least_action = None
        min_visits = float("inf")
        for action, transitions in available_known.items():
            if transitions:
                last_next = transitions[-1][0]
                visits = self._visit_counts.get(last_next, 0)
                if visits < min_visits:
                    min_visits = visits
                    least_action = action

        return least_action

    def suggest_sequence(self, state_hash: str,
                         available_actions: list[int]) -> Optional[list[int]]:
        """Suggest a full action SEQUENCE when near a winning path start.

        Phase A3: Instead of single actions, replay entire winning sequences
        when we recognize a state similar to where a winning sequence began.

        Returns:
            List of actions to replay sequentially, or None if no match.
            Caller should execute actions one by one and abort if grid
            diverges significantly from expected trajectory.
        """
        if not self._winning_sequences:
            return None

        # Check if current state was near the START of any winning sequence.
        # "Near" = current state has a known transition matching the first
        # action of a winning sequence.
        if state_hash not in self._transitions:
            return None

        known_actions = set(self._transitions[state_hash].keys())
        for seq in self._winning_sequences:
            if not seq:
                continue
            first_action = seq[0]
            if first_action in known_actions and first_action in available_actions:
                # Check if first action led to a positive-reward state
                transitions = self._transitions[state_hash].get(first_action, [])
                has_positive = any(r > 0.0 for _, r in transitions)
                if has_positive:
                    # Return the sequence (filtered to available actions)
                    valid_seq = [a for a in seq if a in available_actions]
                    if len(valid_seq) >= 3:  # Only replay if sequence is meaningful
                        logger.info("[StateMemory] Sequence replay: %d actions from winning path",
                                    len(valid_seq))
                        return valid_seq

        return None

    def get_visit_count(self, state_hash: str) -> int:
        """Get how many times this state has been visited."""
        return self._visit_counts.get(state_hash, 0)

    def is_novel_state(self, state_hash: str) -> bool:
        """Check if this state has never been seen."""
        return state_hash not in self._visit_counts

    def get_novelty_ratio(self, state_hash: str) -> float:
        """Get novelty score: 1.0 for never-seen, decreasing with visits."""
        visits = self._visit_counts.get(state_hash, 0)
        if visits == 0:
            return 1.0
        return 1.0 / (1.0 + visits)

    def reset_episode_counter(self) -> None:
        """Reset per-episode novelty counter."""
        self._new_this_episode = 0

    @property
    def new_transitions_this_episode(self) -> int:
        return self._new_this_episode

    def _evict_oldest(self) -> None:
        """Evict least-visited states when at capacity."""
        if not self._visit_counts:
            return
        # Remove states with lowest visit count (keep level-completion states)
        sorted_states = sorted(
            ((h, c) for h, c in self._visit_counts.items()
             if h not in self._level_completion_states),
            key=lambda x: x[1],
        )
        # Remove bottom 10%
        to_remove = max(1, len(sorted_states) // 10)
        for h, _ in sorted_states[:to_remove]:
            self._transitions.pop(h, None)
            self._visit_counts.pop(h, None)
            self._best_rewards.pop(h, None)

    def seed_from_kin(self, winning_sequences: list = None,
                      high_reward_states: list = None) -> int:
        """Seed memory from kin exchange (Phase A6).

        Args:
            winning_sequences: List of action sequences from kin's experience
            high_reward_states: List of {hash, reward} from kin's best states

        Returns:
            Number of new items seeded.
        """
        seeded = 0

        # Seed winning sequences (deduplicate)
        if winning_sequences:
            for seq in winning_sequences[:5]:
                if isinstance(seq, list) and seq not in self._winning_sequences:
                    self._winning_sequences.append(seq)
                    seeded += 1
                    # Keep last 10 sequences
                    if len(self._winning_sequences) > 10:
                        self._winning_sequences = self._winning_sequences[-10:]

        # Seed high-reward state knowledge
        if high_reward_states:
            for entry in high_reward_states[:5]:
                if isinstance(entry, dict):
                    h = entry.get("hash", "")
                    r = entry.get("reward", 0.0)
                    if h and r > self._best_rewards.get(h, 0.0):
                        self._best_rewards[h] = r
                        seeded += 1

        if seeded > 0:
            logger.info("[StateMemory] Seeded %d items from kin exchange", seeded)
        return seeded

    def save(self, path: str) -> None:
        """Save memory graph to JSON. Atomic write."""
        data = {
            "transitions": {
                h: {str(a): ts for a, ts in acts.items()}
                for h, acts in self._transitions.items()
            },
            "visit_counts": self._visit_counts,
            "best_rewards": self._best_rewards,
            "level_completion_states": list(self._level_completion_states),
            "winning_sequences": self._winning_sequences,
            "total_records": self._total_records,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
        logger.info("[StateMemory] Saved %d states, %d transitions to %s",
                    len(self._transitions), self._total_records, path)

    def load(self, path: str) -> bool:
        """Load memory graph from JSON."""
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self._transitions = {
                h: {int(a): ts for a, ts in acts.items()}
                for h, acts in data.get("transitions", {}).items()
            }
            self._visit_counts = data.get("visit_counts", {})
            self._best_rewards = data.get("best_rewards", {})
            self._level_completion_states = set(data.get("level_completion_states", []))
            self._winning_sequences = data.get("winning_sequences", [])
            self._total_records = data.get("total_records", 0)
            logger.info("[StateMemory] Loaded %d states, %d records from %s",
                        len(self._transitions), self._total_records, path)
            return True
        except Exception as e:
            logger.warning("[StateMemory] Load failed: %s", e)
            return False

    def get_stats(self) -> dict:
        """Return memory statistics."""
        return {
            "total_states": len(self._transitions),
            "total_records": self._total_records,
            "total_visits": sum(self._visit_counts.values()),
            "level_completion_states": len(self._level_completion_states),
            "winning_sequences": len(self._winning_sequences),
            "new_this_episode": self._new_this_episode,
        }
