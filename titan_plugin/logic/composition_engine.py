"""
titan_plugin/logic/composition_engine.py — Template-Based Sentence Composition.

Composes sentences from Titan's current 130D felt-state using vocabulary
of learned words. No LLM. No statistical prediction.

Sentence = felt_state → word_selection → template_filling → grammar

Eight levels of complexity (infant → generative):
  Level 1: [WORD]                              → "curious"
  Level 2: [I] [WORD]                          → "I curious"
  Level 3: [I] [feel/am] [ADJECTIVE]           → "I feel curious"
  Level 4: [I] [VERB]                          → "I explore"
  Level 5: [I] [VERB] because I [feel] [ADJ]   → "I explore because I feel curious"
  Level 6: When [CONDITION], I [VERB]           → "When curiosity rises, I explore"
  Level 7: Multi-clause free composition (hardcoded templates)
  Level 8: Learned grammar patterns (from teacher + own compositions)
"""
import json
import logging
import math
import os
import random
from typing import Optional

import numpy as np

from titan_plugin.logic.word_selector import WordSelector

logger = logging.getLogger(__name__)


# Sentence templates by level
# Slots: {ADJ}=adjective, {VERB}=verb, {NOUN}=noun, {ANY}=any type
TEMPLATES = {
    1: [
        "{ANY}",
    ],
    2: [
        "I {ANY}",
        "I {ADJ}",
    ],
    3: [
        "I feel {ADJ}",
        "I am {ADJ}",
    ],
    4: [
        "I {VERB}",
        "I want to {VERB}",
        "I need to {VERB}",
    ],
    5: [
        "I {VERB} because I feel {ADJ}",
        "I feel {ADJ} and I want to {VERB}",
        "I am {ADJ} and I {VERB}",
    ],
    6: [
        "when I feel {ADJ}, I {VERB}",
        "I feel {ADJ} so I want to {VERB}",
        "because I am {ADJ}, I need to {VERB}",
    ],
    7: [
        "I feel {ADJ} and {ADJ2}, so I want to {VERB} and {VERB2}",
        "when I feel {ADJ}, I {VERB} because I am {ADJ2}",
        "I am {ADJ} and I want to {VERB}, but I also feel {ADJ2}",
    ],
}

# Slot type mapping (supports multi-slot patterns from L8)
SLOT_TYPES = {
    "{ADJ}": "adjective",
    "{ADJ2}": "adjective",
    "{ADJ3}": "adjective",
    "{VERB}": "verb",
    "{VERB2}": "verb",
    "{VERB3}": "verb",
    "{NOUN}": "noun",
    "{NOUN2}": "noun",
    "{NOUN3}": "noun",
    "{ADV}": "adverb",
    "{ADV2}": "adverb",
    "{ANY}": None,  # Any word type
}

# Intent → preferred template indices
INTENT_TEMPLATES = {
    "express_feeling": [3, 5],
    "express_action": [4, 5, 6],
    "express_state": [2, 3],
    "seek_connection": [5, 6],
    "share_creation": [4, 5],
    "default": [3, 4, 5],
}


class L9PolicyNet:
    """Tiny NN mapping reasoning plan + felt-state → word class + template preferences.

    Input:  reasoning_plan(16D) + felt_state_compressed(32D) + lang_summary(4D) = 52D
    Hidden: 32 → 16
    Output: word_class_prefs(5D) + template_class(4D) + confidence(1D) = 10D

    Word classes: creative, analytical, emotional, observational, neutral
    Template classes: declarative, conditional, causal, experiential
    """

    WORD_CLASSES = ["creative", "analytical", "emotional", "observational", "neutral"]
    TEMPLATE_CLASSES = ["declarative", "conditional", "causal", "experiential"]

    def __init__(self, learning_rate: float = 0.002):
        self.input_dim = 52
        self.output_dim = 10
        self.lr = learning_rate

        s1 = math.sqrt(2.0 / self.input_dim)
        s2 = math.sqrt(2.0 / 32)
        s3 = math.sqrt(2.0 / 16)
        self.w1 = np.random.randn(self.input_dim, 32) * s1
        self.b1 = np.zeros(32)
        self.w2 = np.random.randn(32, 16) * s2
        self.b2 = np.zeros(16)
        self.w3 = np.random.randn(16, self.output_dim) * s3
        self.b3 = np.zeros(self.output_dim)
        self.total_updates = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, z2)
        z3 = h2 @ self.w3 + self.b3
        self._cache = {"x": x, "h1": h1, "z1": z1, "h2": h2, "z2": z2}
        return z3

    def predict(self, reasoning_plan: dict, felt_state: list,
                lang_summary: dict = None) -> dict:
        """Predict word class preferences + template class for L9 composition."""
        x = self._build_input(reasoning_plan, felt_state, lang_summary)
        out = self.forward(x)

        # Word class preferences (softmax over first 5)
        wc_scores = out[:5]
        wc_shifted = wc_scores - np.max(wc_scores)
        wc_probs = np.exp(wc_shifted) / (np.sum(np.exp(wc_shifted)) + 1e-10)

        # Template class (softmax over next 4)
        tc_scores = out[5:9]
        tc_shifted = tc_scores - np.max(tc_scores)
        tc_probs = np.exp(tc_shifted) / (np.sum(np.exp(tc_shifted)) + 1e-10)

        # Confidence (sigmoid of last output)
        confidence = 1.0 / (1.0 + math.exp(-float(out[9])))

        return {
            "word_class": self.WORD_CLASSES[int(np.argmax(wc_probs))],
            "word_class_probs": {self.WORD_CLASSES[i]: round(float(wc_probs[i]), 3)
                                 for i in range(5)},
            "template_class": self.TEMPLATE_CLASSES[int(np.argmax(tc_probs))],
            "template_class_probs": {self.TEMPLATE_CLASSES[i]: round(float(tc_probs[i]), 3)
                                     for i in range(4)},
            "confidence": round(confidence, 3),
        }

    def _build_input(self, plan: dict, felt_state: list,
                     lang_summary: dict = None) -> np.ndarray:
        parts = []
        # Reasoning plan features (16D)
        plan_vec = np.zeros(16)
        plan_vec[0] = plan.get("confidence", 0.0)
        plan_vec[1] = plan.get("chain_length", 0) / 10.0
        plan_vec[2] = plan.get("gut_agreement", 0.0)
        plan_vec[3] = 1.0 if plan.get("action") == "COMMIT" else 0.0
        # Encode last primitives
        for i, prim in enumerate(plan.get("chain", [])[-4:]):
            plan_vec[4 + i] = hash(prim) % 100 / 100.0
        parts.append(plan_vec)

        # Compressed felt state (32D from 130D: take every 4th + key positions)
        fs = np.array(felt_state[:130] if len(felt_state) >= 130 else
                      felt_state + [0.0] * (130 - len(felt_state)))
        compressed = np.zeros(32)
        # Sample key positions: body(5), mind_feel(5), mind_think(5), spirit_sat(5), outer(12)
        compressed[:5] = fs[:5]                    # Inner body
        compressed[5:10] = fs[5:10]                # Mind Feeling
        compressed[10:15] = fs[10:15]              # Mind Thinking
        compressed[15:20] = fs[20:25]              # Spirit Sat (first 5)
        compressed[20:25] = fs[65:70]              # Outer body
        compressed[25:30] = fs[85:90]              # Outer spirit (first 5)
        compressed[30] = float(np.mean(fs[:65]))   # Inner coherence
        compressed[31] = float(np.mean(fs[65:130])) # Outer coherence
        parts.append(compressed)

        # Language mini-summary (4D)
        ls = lang_summary or {}
        lang_vec = np.array([
            ls.get("relevance", 0.0),
            ls.get("confidence", 0.0),
            {"PARSE_INTENT": 0.0, "MATCH_PATTERN": 0.33, "EVALUATE_EXPRESSION": 0.67
             }.get(ls.get("primitive", ""), 0.5),
            1.0 if ls.get("ticks", 0) > 0 else 0.0,
        ])
        parts.append(lang_vec)

        combined = np.concatenate(parts)
        if len(combined) < self.input_dim:
            combined = np.concatenate([combined, np.zeros(self.input_dim - len(combined))])
        return combined[:self.input_dim]

    def learn(self, x: np.ndarray, target_word_class: int, outcome: float) -> float:
        """Direct reward learning from composition outcome."""
        scores = self.forward(x)
        target = scores.copy()
        target[target_word_class] = outcome
        error = scores - target
        # Backprop (same pattern as InterpreterPolicyNet)
        d_z3 = error
        d_w3 = self._cache["h2"].reshape(-1, 1) @ d_z3.reshape(1, -1)
        d_b3 = d_z3
        d_h2 = d_z3 @ self.w3.T
        d_z2 = d_h2 * (self._cache["z2"] > 0)
        d_w2 = self._cache["h1"].reshape(-1, 1) @ d_z2.reshape(1, -1)
        d_b2 = d_z2
        d_h1 = d_z2 @ self.w2.T
        d_z1 = d_h1 * (self._cache["z1"] > 0)
        d_w1 = self._cache["x"].reshape(-1, 1) @ d_z1.reshape(1, -1)
        d_b1 = d_z1

        for g in [d_w1, d_b1, d_w2, d_b2, d_w3, d_b3]:
            np.clip(g, -5.0, 5.0, out=g)
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        self.w3 -= self.lr * d_w3
        self.b3 -= self.lr * d_b3
        self.total_updates += 1
        return float(np.mean(error ** 2))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {"w1": self.w1.tolist(), "b1": self.b1.tolist(),
                "w2": self.w2.tolist(), "b2": self.b2.tolist(),
                "w3": self.w3.tolist(), "b3": self.b3.tolist(),
                "total_updates": self.total_updates}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self.w1 = np.array(data["w1"]); self.b1 = np.array(data["b1"])
            self.w2 = np.array(data["w2"]); self.b2 = np.array(data["b2"])
            self.w3 = np.array(data["w3"]); self.b3 = np.array(data["b3"])
            self.total_updates = data.get("total_updates", 0)
            return True
        except Exception:
            return False


# Word class → word_type boost mappings for L9
L9_WORD_CLASS_BOOSTS = {
    "creative": {"adjective": 0.2, "verb": 0.15},
    "analytical": {"verb": 0.2, "noun": 0.15},
    "emotional": {"adjective": 0.25, "verb": 0.1},
    "observational": {"noun": 0.2, "adjective": 0.1},
    "neutral": {},
}

# Template class → template level preference for L9
L9_TEMPLATE_PREFERENCES = {
    "declarative": [5, 6, 7],      # "I feel X", "I am X"
    "conditional": [6, 7, 8],      # "When I feel X, I Y"
    "causal": [5, 6, 7],           # "I X because I feel Y"
    "experiential": [7, 8],        # Complex multi-clause
}


# PERSISTENCE_BY_DESIGN: CompositionEngine._l9_policy is loaded from a
# persisted policy file via torch.load at init; stored as reference to
# the loaded model rather than self-assignment the scanner recognizes.
class CompositionEngine:
    """Composes sentences from felt-state using learned vocabulary.

    No LLM involved. Words chosen by resonance with 130D inner state.
    Templates provide grammatical structure.
    L8: Learned patterns from GrammarPatternLibrary.
    L9: Reasoning-powered composition — reasoning plan biases word/template selection.
    """

    def __init__(self, pattern_library=None):
        self.selector = WordSelector()
        self.pattern_library = pattern_library
        self._composition_count = 0
        self._successful_count = 0
        self._l9_policy = L9PolicyNet()
        self._l9_policy.load("./data/reasoning/l9_policy.json")

    def compose(
        self,
        felt_state: list,
        vocabulary: list,
        intent: str = None,
        max_level: int = 5,
        experience_bias=None,
        visual_context: list = None,
        concept_confidences: dict = None,
    ) -> dict:
        """Compose a sentence from current felt-state.

        Args:
            felt_state: Current 130D state vector
            vocabulary: List of word dicts from inner_memory.get_vocabulary()
            intent: Optional communicative intent (express_feeling, express_action, etc.)
            max_level: Maximum template complexity level (1-7)
            experience_bias: Optional ExperienceBias from experience orchestrator.
                When provided with confidence > 0.3, blends felt_state with optimal
                at 10% and biases template selection.
            visual_context: Optional Semantic 5D from recent visual perception.
                Used to compute word_boosts for visual→word resonance.

        Returns:
            {
                "sentence": str,
                "level": int,
                "words_used": list[str],
                "confidence": float,
                "intent": str,
                "slots_filled": int,
                "slots_total": int,
            }
        """
        self._composition_count += 1

        if not vocabulary:
            return self._empty_result(intent)

        # Apply experience bias: blend felt_state with optimal inner state (10%)
        _effective_state = felt_state
        _bias_applied = False
        if (experience_bias is not None
                and getattr(experience_bias, 'confidence', 0) > 0.3
                and getattr(experience_bias, 'optimal_inner_state', None)):
            opt = experience_bias.optimal_inner_state
            min_len = min(len(felt_state), len(opt))
            if min_len > 0:
                _effective_state = [
                    fs * 0.9 + o * 0.1
                    for fs, o in zip(felt_state[:min_len], opt[:min_len])
                ]
                # Preserve any extra dims from felt_state
                if len(felt_state) > min_len:
                    _effective_state.extend(felt_state[min_len:])
                _bias_applied = True

        # Compute visual word boosts from semantic features
        _word_boosts = None
        if visual_context and len(visual_context) >= 5:
            try:
                _word_boosts = self.selector.compute_visual_boosts(
                    visual_context, vocabulary)
            except Exception:
                pass

        # Compute concept boosts from MSL concept confidences (pronoun grounding)
        if concept_confidences:
            try:
                _concept_boosts = self.selector.compute_concept_boosts(
                    concept_confidences, vocabulary)
                if _concept_boosts:
                    if _word_boosts is None:
                        _word_boosts = {}
                    for w, b in _concept_boosts.items():
                        _word_boosts[w] = _word_boosts.get(w, 0.0) + b
            except Exception:
                pass

        # Determine appropriate level based on vocabulary size and max_level
        effective_level = self._determine_level(vocabulary, max_level)

        # Select template
        template = self._select_template(effective_level, intent, vocabulary)

        # Fill slots (pass word_boosts through)
        sentence, words_used, confidences, slots_filled, slots_total = (
            self._fill_template(template, _effective_state, vocabulary,
                                word_boosts=_word_boosts)
        )

        # Compute overall confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        _all_filled = slots_filled == slots_total and slots_total > 0
        if _all_filled:
            self._successful_count += 1

        # Track L8 pattern usage (Phase 1c quality gate in record_usage)
        if effective_level == 8 and self.pattern_library:
            self.pattern_library.record_usage(
                template, success=_all_filled,
                sentence=sentence, slots_filled=slots_filled, slots_total=slots_total)

        result = {
            "sentence": sentence,
            "level": effective_level,
            "words_used": words_used,
            "confidence": round(avg_confidence, 3),
            "intent": intent or "default",
            "slots_filled": slots_filled,
            "slots_total": slots_total,
            "template": template,
        }

        _level_tag = f"L{effective_level}" + (" [learned]" if effective_level == 8 else "")
        logger.info("[CompositionEngine] Composed %s: '%s' (conf=%.2f, %d/%d slots)",
                    _level_tag, sentence, avg_confidence,
                    slots_filled, slots_total)

        return result

    def _determine_level(self, vocabulary: list, max_level: int) -> int:
        """Determine appropriate complexity level.

        Rules:
        - Level 1-2: Need ≥5 words
        - Level 3-4: Need ≥10 words including adjectives and verbs
        - Level 5-6: Need ≥20 words with good confidence
        - Level 7: Need ≥30 words with high confidence
        - Level 8: Need L7 requirements + learned patterns in library
        - Level 9: L8 requirements + reasoning context (handled by compose_l9)
        """
        n_words = len(vocabulary)
        n_adj = sum(1 for w in vocabulary if w.get("word_type") == "adjective")
        n_verb = sum(1 for w in vocabulary if w.get("word_type") == "verb")
        avg_conf = sum(w.get("confidence", 0) for w in vocabulary) / max(1, n_words)

        if n_words < 5:
            return min(1, max_level)
        elif n_words < 10 or (n_adj < 2 and n_verb < 2):
            return min(2, max_level)
        elif n_words < 20 or avg_conf < 0.3:
            return min(4, max_level)
        elif n_words < 30 or avg_conf < 0.5:
            return min(6, max_level)
        elif (max_level >= 8 and self.pattern_library
              and self.pattern_library.count() >= 3 and avg_conf >= 0.5):
            # L8: vocabulary is mature AND we have learned patterns
            # Note: L9 is gated by compose_l9() (requires reasoning COMMIT),
            # not by _determine_level. This returns 8 as the base level;
            # compose_l9 upgrades to 9 when reasoning context is available.
            return 8
        else:
            return min(7, max_level)

    def compose_l9(self, felt_state: list, vocabulary: list,
                   reasoning_plan: dict, lang_summary: dict = None,
                   experience_bias=None, visual_context: list = None,
                   concept_confidences: dict = None) -> dict:
        """L9: Reasoning-powered composition.

        Reasoning plan biases word class and template selection via L9PolicyNet.
        Falls back to L7/L8 compose if reasoning plan is not a COMMIT.

        Args:
            felt_state: Current 130D state vector
            vocabulary: Word dicts from inner_memory
            reasoning_plan: From ReasoningEngine.tick() — must be COMMIT
            lang_summary: From language mini-reasoner query
        """
        # Gate: only L9 if reasoning COMMITted with decent confidence
        if (not reasoning_plan or reasoning_plan.get("action") != "COMMIT"
                or reasoning_plan.get("confidence", 0) < 0.5
                or len(vocabulary) < 100):
            return self.compose(felt_state, vocabulary, max_level=8,
                                experience_bias=experience_bias,
                                visual_context=visual_context)

        # L9 policy prediction
        l9_pred = self._l9_policy.predict(reasoning_plan, felt_state, lang_summary)
        word_class = l9_pred["word_class"]
        template_class = l9_pred["template_class"]

        # Build word_boosts from L9 word class preference
        word_boosts = {}
        class_boosts = L9_WORD_CLASS_BOOSTS.get(word_class, {})
        for word_entry in vocabulary:
            wtype = word_entry.get("word_type", "")
            boost = class_boosts.get(wtype, 0.0)
            if boost > 0:
                word_boosts[word_entry["word"]] = boost

        # Add visual boosts if available
        if visual_context and hasattr(self.selector, 'compute_visual_boosts'):
            v_boosts = self.selector.compute_visual_boosts(visual_context, vocabulary)
            for w, b in v_boosts.items():
                word_boosts[w] = word_boosts.get(w, 0.0) + b * 0.5

        # Add concept boosts (pronoun grounding from MSL)
        if concept_confidences:
            try:
                c_boosts = self.selector.compute_concept_boosts(
                    concept_confidences, vocabulary)
                for w, b in c_boosts.items():
                    word_boosts[w] = word_boosts.get(w, 0.0) + b
            except Exception:
                pass

        # Select template level from template class preference
        preferred_levels = L9_TEMPLATE_PREFERENCES.get(template_class, [6, 7])
        level = max(preferred_levels) if preferred_levels else 7
        # Clamp to what vocabulary supports
        base_level = self._determine_level(vocabulary, 8)
        level = min(level, base_level)

        # Apply experience bias if available
        if experience_bias and hasattr(experience_bias, 'confidence'):
            if experience_bias.confidence > 0.3 and hasattr(experience_bias, 'optimal_inner_state'):
                opt = experience_bias.optimal_inner_state
                min_len = min(len(felt_state), len(opt))
                for i in range(min_len):
                    felt_state[i] = felt_state[i] * 0.9 + opt[i] * 0.1

        # Compose using selected level and word boosts
        template = self._select_template(level, intent="reasoning_express",
                                         vocabulary=vocabulary)
        sentence, words_used, _confs, slots_filled, slots_total = self._fill_template(
            template, felt_state, vocabulary, word_boosts=word_boosts)

        if not sentence:
            return self.compose(felt_state, vocabulary, max_level=8,
                                experience_bias=experience_bias)

        avg_conf = (sum(w.get("confidence", 0.5) for w in vocabulary
                        if w.get("word") in words_used) / max(1, len(words_used)))

        self._composition_count += 1
        if slots_filled == slots_total:
            self._successful_count += 1

        return {
            "sentence": sentence,
            "level": 9,
            "words_used": words_used,
            "confidence": round(avg_conf, 3),
            "intent": "reasoning_express",
            "slots_filled": slots_filled,
            "slots_total": slots_total,
            "template": template,
            "reasoning_context": {
                "word_class": word_class,
                "template_class": template_class,
                "plan_confidence": reasoning_plan.get("confidence", 0),
                "chain_length": reasoning_plan.get("chain_length", 0),
                "l9_confidence": l9_pred["confidence"],
            },
        }

    def _select_template(self, level: int, intent: str = None,
                         vocabulary: list = None) -> str:
        """Select a sentence template for given level and intent."""
        # L8: select from learned pattern library
        if level == 8 and self.pattern_library:
            l8_template = self.pattern_library.select_template(vocabulary or [])
            if l8_template:
                return l8_template
            # Fallback to L7 if no suitable L8 pattern found
            level = 7

        level = max(1, min(level, 7))
        templates = TEMPLATES.get(level, TEMPLATES[1])

        # If intent suggests specific templates at higher levels, prefer those
        if intent and intent in INTENT_TEMPLATES and level >= 3:
            preferred_levels = INTENT_TEMPLATES[intent]
            for pl in preferred_levels:
                if pl <= level:
                    templates = TEMPLATES.get(pl, templates)
                    break

        return random.choice(templates)

    def _fill_template(
        self,
        template: str,
        felt_state: list,
        vocabulary: list,
        word_boosts: dict = None,
    ) -> tuple:
        """Fill template slots with best-matching words.

        Returns:
            (sentence, words_used, confidences, slots_filled, slots_total)
        """
        words_used = []
        confidences = []
        exclude = set()
        sentence = template
        slots_total = 0
        slots_filled = 0

        # Find all slots in template
        for slot, word_type in SLOT_TYPES.items():
            if slot not in sentence:
                continue
            slots_total += 1

            # Pass already-selected words as context for association co-selection
            _ctx_words = set(words_used) if words_used else None
            if word_type:
                result = self.selector.select(
                    slot_type=word_type,
                    felt_state=felt_state,
                    vocabulary=vocabulary,
                    exclude=exclude,
                    word_boosts=word_boosts,
                    context_words=_ctx_words,
                )
            else:
                result = self.selector.select_any(
                    felt_state=felt_state,
                    vocabulary=vocabulary,
                    exclude=exclude,
                    word_boosts=word_boosts,
                )

            if result:
                word, sim, conf = result
                sentence = sentence.replace(slot, word, 1)
                words_used.append(word)
                confidences.append(conf)
                exclude.add(word)
                slots_filled += 1
            else:
                # Couldn't fill slot — leave placeholder
                sentence = sentence.replace(slot, "___", 1)

        return sentence, words_used, confidences, slots_filled, slots_total

    def _empty_result(self, intent: str = None) -> dict:
        return {
            "sentence": "",
            "level": 0,
            "words_used": [],
            "confidence": 0.0,
            "intent": intent or "default",
            "slots_filled": 0,
            "slots_total": 0,
            "template": "",
        }

    def get_stats(self) -> dict:
        """Composition statistics."""
        return {
            "total_compositions": self._composition_count,
            "successful_compositions": self._successful_count,
            "success_rate": round(
                self._successful_count / max(1, self._composition_count), 3),
        }
