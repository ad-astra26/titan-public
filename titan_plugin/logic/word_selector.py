"""
titan_plugin/logic/word_selector.py — Felt-State Word Selection.

Selects words from vocabulary based on cosine similarity between
current 132D felt-state and each word's learned felt-tensor.

This is the bridge between FEELING and SPEAKING:
  132D inner+outer state → cosine similarity → best matching word

No LLM. No statistics. Pure mathematical felt-meaning matching.

Uses temperature-based softmax over top candidates instead of hard
argmax, so similar-scoring words compete with controlled randomness.
Coverage-adaptive curiosity ramps exploration when word diversity is low.
"""
import logging
import math
import random
from typing import Optional

logger = logging.getLogger(__name__)

# Hormone → word type affinity (which types of words fit which hormones)
HORMONE_WORD_AFFINITY = {
    "CURIOSITY": ["verb", "noun"],       # Curious → action + concept words
    "CREATIVITY": ["verb", "adjective"], # Creative → expression words
    "EMPATHY": ["verb", "adjective"],    # Empathic → relational words
    "REFLECTION": ["adjective", "verb"], # Reflective → descriptive words
    "INSPIRATION": ["noun", "adjective"],# Inspired → visionary words
    "FOCUS": ["verb"],                   # Focused → action words
    "IMPULSE": ["verb"],                 # Impulsive → action words
    "INTUITION": ["noun", "adjective"],  # Intuitive → pattern words
    "VIGILANCE": ["adjective", "verb"],  # Vigilant → state words
    "REFLEX": ["verb"],                  # Reflexive → quick action words
}


def _cosine_sim(a: list, b: list) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-12 or mag_b < 1e-12:
        return 0.0
    return dot / (mag_a * mag_b)


class WordSelector:
    """Selects words from vocabulary using felt-state similarity.

    Words are not chosen by statistical prediction.
    They are chosen by RESONANCE with inner state, with a curiosity bonus
    that encourages exploration of less-used words (UCB-inspired).
    """

    def __init__(self):
        # Track word usage for curiosity bonus (UCB exploration)
        self._word_usage: dict[str, int] = {}
        self._total_selections: int = 0
        # Curiosity weight: 0.0 = pure resonance, higher = more exploration
        self.curiosity_weight: float = 0.40
        # Temperature for softmax selection (lower = more deterministic, higher = more random)
        self.temperature: float = 0.3
        # Top-K candidates for softmax sampling (prevents picking truly bad words)
        self.top_k: int = 5
        # Recent words for coverage tracking (sliding window)
        self._recent_words: list[str] = []
        self._recent_window: int = 50
        # Neuromod-gated exploration: set by spirit_worker when DA is high
        self.min_confidence_override: float | None = None

    def select(
        self,
        slot_type: str,
        felt_state: list,
        vocabulary: list,
        exclude: set = None,
        min_confidence: float = 0.1,
        word_boosts: dict = None,
        context_words: set = None,
    ) -> Optional[tuple]:
        """Select best word for a template slot.

        Args:
            slot_type: Word type to match ('adjective', 'verb', 'noun', etc.)
            felt_state: Current 130D (or any-D) state vector
            vocabulary: List of word dicts from inner_memory.get_vocabulary()
            context_words: Words already selected in this composition — used for
                          association-boosted co-selection (Phase 4c)
            exclude: Words already used in this sentence
            min_confidence: Minimum word confidence to consider
            word_boosts: Optional {word: boost_value} from experience/visual context.
                Boost values in [-0.3, +0.3], applied at 30% influence.

        Returns:
            (word, similarity, confidence) tuple, or None if no match
        """
        exclude = exclude or set()
        candidates = []

        for word_entry in vocabulary:
            word = word_entry.get("word", "")
            w_type = word_entry.get("word_type", "")
            confidence = word_entry.get("confidence", 0.0)
            felt_tensor = word_entry.get("felt_tensor")

            # Filter by type
            if slot_type and w_type != slot_type:
                continue

            # Skip excluded words
            if word in exclude:
                continue

            # Skip low confidence (neuromod override allows exploration)
            _effective_min = self.min_confidence_override if self.min_confidence_override is not None else min_confidence
            if confidence < _effective_min:
                continue

            # Compute similarity if felt_tensor available
            if felt_tensor and felt_state:
                # Handle dimension mismatch gracefully
                min_len = min(len(felt_state), len(felt_tensor))
                sim = _cosine_sim(felt_state[:min_len], felt_tensor[:min_len])
            else:
                sim = confidence * 0.5  # Fallback: use confidence as weak signal

            xm_conf = word_entry.get("cross_modal_conf", 0.0)
            candidates.append((word, sim, confidence, xm_conf))

        if not candidates:
            return None

        # Score = resonance × confidence + curiosity bonus for less-used words
        # + grounding boost for words with cross-modal grounding (Phase 4)
        # Coverage-adaptive: curiosity ramps up when word diversity is low
        self._total_selections += 1

        # Adaptive curiosity: measure recent word diversity
        unique_recent = len(set(self._recent_words[-self._recent_window:]))
        coverage = unique_recent / max(len(candidates), 1)
        # Low coverage → boost curiosity (up to 2x), high coverage → base curiosity
        adaptive_curiosity = self.curiosity_weight * (1.0 + (1.0 - min(coverage, 1.0)))

        # Build association index from context words (Phase 4c + Phase 5d)
        # Typed associations get different boost values for better co-selection
        _assoc_boosts = {}  # word -> boost_value
        _TYPED_BOOST = {
            "SIMILAR": 0.12,       # Words with similar meaning flow together
            "SEQUENCE": 0.15,      # Sequential words are natural together
            "COMPONENT": 0.12,     # Part-whole relationships are coherent
            "CAUSE_EFFECT": 0.10,  # Causal links make narrative sense
            "OPPOSITE": 0.05,      # Opposites can work but less naturally
            "CO_OCCURRENCE": 0.08, # Untyped co-occurrence — moderate boost
            "HYPOTHESIS_LINKED": 0.10,
            "SYNTHESIS_LINKED": 0.10,
            "REASONING_LINKED": 0.10,
        }
        if context_words:
            import json as _ws_json
            for _ve in vocabulary:
                if _ve.get("word") in context_words:
                    _mc = _ve.get("meaning_contexts")
                    if isinstance(_mc, str):
                        try:
                            _mc = _ws_json.loads(_mc)
                        except Exception:
                            _mc = []
                    if isinstance(_mc, list):
                        for _m in _mc:
                            for _a in _m.get("associations", []):
                                if isinstance(_a, (list, tuple)) and len(_a) >= 1:
                                    _a_word = _a[0].lower()
                                    _a_type = _a[1] if len(_a) >= 2 else "CO_OCCURRENCE"
                                    _boost = _TYPED_BOOST.get(_a_type, 0.08)
                                    # Keep highest boost if multiple associations
                                    _assoc_boosts[_a_word] = max(
                                        _assoc_boosts.get(_a_word, 0), _boost)

        scored = []
        for word, sim, conf, xm in candidates:
            base_score = sim * (0.7 + 0.3 * conf)
            usage = self._word_usage.get(word, 0)
            curiosity_bonus = adaptive_curiosity / math.sqrt(usage + 1)
            boost = word_boosts.get(word, 0.0) * 0.15 if word_boosts else 0.0
            # Grounding boost: words with cross-modal grounding are preferred.
            grounding_boost = xm * 0.15  # Up to +0.15 for fully grounded words
            # Association boost: typed associations get differentiated co-selection bonus
            assoc_boost = _assoc_boosts.get(word.lower(), 0.0)
            scored.append((word, sim, conf, base_score + curiosity_bonus + boost + grounding_boost + assoc_boost))

        scored.sort(key=lambda x: x[3], reverse=True)

        # Temperature-based softmax over top-K candidates
        top = scored[:self.top_k]
        if len(top) > 1 and self.temperature > 0:
            # Softmax with temperature
            max_score = top[0][3]
            weights = []
            for _, _, _, s in top:
                exp_val = math.exp((s - max_score) / max(self.temperature, 0.01))
                weights.append(exp_val)
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            # Weighted random selection
            r = random.random()
            cumulative = 0.0
            chosen_idx = 0
            for i, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    chosen_idx = i
                    break
            chosen = top[chosen_idx]
        else:
            chosen = top[0]

        # Track usage + recent words
        self._word_usage[chosen[0]] = self._word_usage.get(chosen[0], 0) + 1
        self._recent_words.append(chosen[0])
        if len(self._recent_words) > self._recent_window * 2:
            self._recent_words = self._recent_words[-self._recent_window:]

        return (chosen[0], chosen[1], chosen[2])

    def select_by_hormone(
        self,
        dominant_hormone: str,
        vocabulary: list,
        top_k: int = 5,
    ) -> list[tuple]:
        """Select words associated with a dominant hormone.

        Args:
            dominant_hormone: e.g., 'CURIOSITY', 'CREATIVITY'
            vocabulary: List of word dicts
            top_k: Max results

        Returns:
            List of (word, affinity_score) tuples
        """
        preferred_types = HORMONE_WORD_AFFINITY.get(dominant_hormone, [])
        results = []

        for word_entry in vocabulary:
            word = word_entry.get("word", "")
            w_type = word_entry.get("word_type", "")
            confidence = word_entry.get("confidence", 0.0)
            hormone_pattern = word_entry.get("hormone_pattern", {})

            # Score based on hormone affinity in the word's recipe
            affinity = 0.0
            if isinstance(hormone_pattern, dict):
                affinity = hormone_pattern.get(dominant_hormone, 0.0)

            # Bonus if word type matches hormone preference
            type_bonus = 0.2 if w_type in preferred_types else 0.0

            score = affinity + type_bonus + confidence * 0.1
            if score > 0.05:
                results.append((word, round(score, 3)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def select_any(
        self,
        felt_state: list,
        vocabulary: list,
        exclude: set = None,
        min_confidence: float = 0.1,
        word_boosts: dict = None,
    ) -> Optional[tuple]:
        """Select best word regardless of type."""
        return self.select(
            slot_type=None,
            felt_state=felt_state,
            vocabulary=vocabulary,
            exclude=exclude,
            min_confidence=min_confidence,
            word_boosts=word_boosts,
        )

    def compute_visual_boosts(
        self,
        semantic_5d: list,
        vocabulary: list,
    ) -> dict:
        """Compute word boosts from visual semantic features.

        Maps visual perception meaning to word resonance:
          complexity (semantic[0]) boosts verbs (action-oriented images)
          beauty     (semantic[1]) boosts adjectives (aesthetic words)
          warmth     (semantic[2]) boosts adjectives (emotional words)
          structural_order (semantic[3]) boosts nouns (concrete concepts)
          narrative_weight (semantic[4]) boosts verbs (change/movement)

        Returns {word: boost_value} where boost is in [-0.3, +0.3].
        """
        if not semantic_5d or len(semantic_5d) < 5:
            return {}

        boosts = {}
        for entry in vocabulary:
            word = entry.get("word", "")
            w_type = entry.get("word_type", "")
            if not w_type:
                continue

            boost = 0.0
            if w_type == "adjective":
                boost += semantic_5d[1] * 0.3   # beauty boosts adjectives
                boost += semantic_5d[2] * 0.2   # warmth boosts adjectives
            elif w_type == "verb":
                boost += semantic_5d[0] * 0.3   # complexity boosts verbs
                boost += semantic_5d[4] * 0.2   # narrative_weight boosts verbs
            elif w_type == "noun":
                boost += semantic_5d[3] * 0.3   # structural_order boosts nouns

            # Center around 0 (subtract 0.25 so boost range is roughly [-0.25, +0.25])
            boost = max(-0.3, min(0.3, boost - 0.25))
            if abs(boost) > 0.05:
                boosts[word] = round(boost, 4)

        return boosts

    def compute_concept_boosts(
        self, concept_confidences: dict, vocabulary: list
    ) -> dict:
        """Compute word boosts from MSL concept confidences.

        When concepts are grounded (confidence > 0.3), bias word selection
        toward related pronouns/words. Creates visible signal: Titan starts
        using "I" naturally when "I" is genuinely grounded.

        Args:
            concept_confidences: {"I": 0.8, "YOU": 0.1, ...}
            vocabulary: List of word dicts with word, word_type keys

        Returns:
            {word: boost_value} where boost is in [-0.3, +0.3].
        """
        # Concept → associated words with boost strengths
        CONCEPT_WORDS = {
            "I":    {"I": 1.0, "my": 0.8, "me": 0.8, "myself": 0.7, "am": 0.5},
            "YOU":  {"you": 1.0, "your": 0.8},
            "YES":  {"yes": 1.0},
            "NO":   {"no": 1.0},
            "WE":   {"we": 1.0, "together": 0.8, "our": 0.8},
            "THEY": {"they": 0.8, "them": 0.7, "their": 0.7, "others": 0.6},
        }
        CONFIDENCE_THRESHOLD = 0.3
        MAX_BOOST = 0.25  # Moderate — doesn't overwhelm felt-state matching

        boosts = {}
        for concept, conf in concept_confidences.items():
            if conf < CONFIDENCE_THRESHOLD:
                continue
            words_map = CONCEPT_WORDS.get(concept, {})
            # Boost scales with confidence above threshold
            strength = min(1.0, (conf - CONFIDENCE_THRESHOLD) / 0.7)  # 0→1 over 0.3→1.0
            for target_word, word_strength in words_map.items():
                boost = MAX_BOOST * strength * word_strength
                if boost > 0.05:
                    # Check word exists in vocabulary
                    for entry in vocabulary:
                        if entry.get("word", "").lower() == target_word.lower():
                            boosts[entry["word"]] = round(
                                max(boosts.get(entry["word"], 0.0), boost), 4)
                            break
        return boosts
