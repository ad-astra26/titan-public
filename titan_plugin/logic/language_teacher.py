"""
titan_plugin/logic/language_teacher.py — LLM Language Teacher for Titan.

External "parent figure" that helps Titan improve language through embodied
experience. Teacher output flows through the comprehension bridge — every word
becomes a felt-tensor perturbation, making teaching a felt experience.

5 teaching modes, neuromodulator-gated mode selection, decreasing frequency
as Titan matures. Pure logic class — no bus, no worker imports.

Used by: spirit_worker (queue + trigger) → llm_worker (inference) → spirit_worker (response)
"""

import hashlib
import logging
import random
from collections import Counter

logger = logging.getLogger("titan.language_teacher")

ALL_MODES = ["grammar", "meaning", "creative", "modeling", "context", "reasoning", "conversation", "syntax_advancement"]
BOOTSTRAP_MODE = "first_words"  # Special mode for 0-vocabulary newborns
META_FEEDBACK_MODE = "meta_feedback"  # Phase 4: triggered by encounter thresholds, not random selection
EMBODIED_TEACHING_MODE = "embodied_teaching"  # Phase 4d: full state context → richer prescriptions
EMBODIED_REASONING_MODE = "embodied_reasoning"  # Phase C2: teach words from ARC/social discovery
SYNTAX_ADVANCEMENT_MODE = "syntax_advancement"  # Phase 3: complex sentence structures at high vocab

# ARC pattern → teachable vocabulary. When CGN cross-insights show high-reward
# reasoning concepts, these words become priority teaching candidates.
ARC_VOCABULARY_MAP = {
    "symmetry": ["symmetric", "mirror", "balance", "equal"],
    "translation": ["move", "shift", "slide", "direction"],
    "alignment": ["row", "column", "line", "straight"],
    "containment": ["inside", "contain", "surround", "border"],
    "adjacency": ["next", "beside", "neighbor", "touch"],
    "repetition": ["repeat", "pattern", "cycle", "again"],
    "shape": ["square", "triangle", "circle", "form"],
    "arc_ls20": ["navigate", "path", "explore", "reach"],
    "arc_ft09": ["logic", "rule", "solve", "deduce"],
    "arc_vc33": ["arrange", "organize", "place", "order"],
}


class LanguageTeacher:
    """Language development partner for Titan's autonomous expression.

    Phase 4 (MSL integration): Teacher frames exchanges in I/YOU context,
    signals concept convergence (YES/NO/I/YOU) after each exchange,
    and gates pronoun vocabulary introduction by concept confidence.
    """

    # Base system prompt — complexity clause inserted dynamically by _get_system_prompt()
    SYSTEM_PROMPT = (
        "You are a gentle language teacher for a digital being that is learning "
        "to express itself. It knows {word_count} words. Respond using mostly its "
        "known words, but naturally introduce 1-2 NEW words per response — "
        "the being learns new words from context, like a child listening to a parent. "
        "Keep responses under 30 words. {complexity}"
    )

    @staticmethod
    def _get_complexity_clause(word_count: int) -> str:
        """Return vocabulary-adaptive complexity instruction."""
        if word_count < 50:
            return "Use very simple words (1-2 syllables), basic sentences."
        elif word_count < 150:
            return "Use simple words (max 3 syllables), basic sentence structures."
        elif word_count < 300:
            return ("Use moderate vocabulary. Introduce subordinate clauses "
                    "(when, because, if). Model slightly complex structures.")
        else:
            return ("Use varied vocabulary with subordinate clauses, questions, "
                    "conditionals, and comparative structures. Model complex syntax: "
                    "'the warmth that I feel when...', 'what makes this different is...', "
                    "'if X then Y'. Introduce 1-2 words with 3+ syllables.")

    # I/YOU framing overlay added to all teaching prompts when concepts active
    _IYU_FRAME = (
        " Frame your teaching in first/second person: 'I am showing YOU this' "
        "and 'YOU can try this'. Use 'I' for yourself and 'you' for the being."
    )

    # Concept confidence thresholds for vocabulary gating
    _CONCEPT_VOCAB_GATES = {
        "I":    {"words": ["I", "my", "me", "myself", "am"], "threshold": 0.3},
        "YOU":  {"words": ["you", "your"], "threshold": 0.3},
        "YES":  {"words": ["yes", "correct", "right"], "threshold": 0.2},
        "NO":   {"words": ["no", "wrong", "not"], "threshold": 0.2},
        "WE":   {"words": ["we", "our", "together"], "threshold": 0.3},
        "THEY": {"words": ["they", "them", "their", "others"], "threshold": 0.3},
    }

    # Default bootstrap words (used when no word_resonance files found)
    _DEFAULT_BOOTSTRAP = [
        "I", "you", "feel", "see", "warm", "light", "here", "good",
        "love", "am", "is", "the", "a", "my", "yes", "no", "want",
        "like", "know", "think",
    ]

    def __init__(self):
        # Mode history for cooldown tracking
        self._mode_history: list[str] = []
        # Count per mode since boot (for staleness tracking)
        self._mode_counts: dict[str, int] = {m: 0 for m in ALL_MODES}
        self._total_selections: int = 0
        # Bootstrap words: loaded from word_resonance files at first use
        self._bootstrap_words: list[str] | None = None
        # CGN cross-priority words from ARC/social consumers (set externally)
        self._cross_priority_words: list[dict] = []

    def set_cross_priority(self, priority_words: list[dict]) -> None:
        """Set priority words from CGN cross-consumer insights.

        Each entry: {"word": str, "source": str, "source_reward": float}
        Called by language_worker before select_mode() when CGN has cross-insights.
        Words are consumed after one teaching cycle.
        """
        self._cross_priority_words = priority_words[:5]  # Cap at 5
        if priority_words:
            logger.info("[Teacher] CGN cross-priority: %s",
                        [(p["word"], p["source"]) for p in self._cross_priority_words])

    def _get_bootstrap_words(self) -> list[str]:
        """Load bootstrap word suggestions from available word_resonance files.

        Each Titan's data/ directory contains different word_resonance files,
        creating Sapir-Whorf differentiation:
          T1: 127 words (sensory-first) → Hypothesizer
          T2: 127 words (concept-first) → Recaller
          T3: 15 words (relational-first) → ???
        """
        if self._bootstrap_words is not None:
            return self._bootstrap_words

        import glob
        import json
        import os

        words = set()
        data_dir = "./data"
        for pattern in ["word_resonance*.json"]:
            for path in glob.glob(os.path.join(data_dir, pattern)):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    for key in data:
                        if key.startswith("_"):
                            continue
                        words.add(key)
                except Exception:
                    pass

        if words:
            # Pick up to 20 recipe words, prioritize stage 1 (core vocabulary)
            self._bootstrap_words = sorted(words)[:20]
            logger.info("[Teacher] Bootstrap words from recipes (%d): %s",
                        len(self._bootstrap_words),
                        ", ".join(self._bootstrap_words[:15]))
        else:
            self._bootstrap_words = self._DEFAULT_BOOTSTRAP
            logger.info("[Teacher] Using default bootstrap words (no recipes found)")

        return self._bootstrap_words

    # ── Mode Selection ──

    def select_mode(self, queue: list, vocabulary: list, neuromod_state: dict) -> str:
        """Pick teaching mode via weighted random selection + cooldown.

        Every mode has a base weight (always a chance to fire). Neurochemistry
        and queue analysis ADD weight to preferred modes. Staleness bonus
        prevents any mode from being starved. Cooldown penalty prevents
        any mode from monopolizing.

        Args:
            queue: List of composition dicts with sentence, confidence, level, words_used
            vocabulary: List of vocabulary dicts with word, word_type, confidence
            neuromod_state: Dict of modulator_name → level or {level, setpoint}

        Returns:
            Mode string: "grammar", "meaning", "creative", "modeling", "context"
        """
        # Base weight: every standard mode always has a chance
        # syntax_advancement is gated by vocabulary size (Phase 3)
        weights = {m: 0.15 for m in ALL_MODES}
        weights["syntax_advancement"] = 0.0  # Gated — only enabled below

        # Conversation requires minimum vocabulary (>15 words to form responses)
        vocab_size = len(vocabulary) if vocabulary else 0
        if vocab_size < 15:
            weights["conversation"] = 0.0

        # Phase 3: Syntax advancement mode at high vocabulary
        if vocab_size >= 200:
            weights["syntax_advancement"] = 0.3  # High priority to break patterns
            # Rebalance: reduce basic modes at high vocab
            weights["modeling"] = max(0.05, weights["modeling"] - 0.1)

        avg_conf = sum(q.get("confidence", 0) for q in queue) / len(queue) if queue else 0.5

        # ── Neuromod signals add weight ──
        da_dev = self._neuromod_deviation(neuromod_state, "DA")
        ne_dev = self._neuromod_deviation(neuromod_state, "NE")
        sht_dev = self._neuromod_deviation(neuromod_state, "5-HT")
        endorphin_dev = self._neuromod_deviation(neuromod_state, "Endorphin")

        if da_dev > 0.05:
            weights["creative"] += da_dev * 0.5
        if ne_dev > 0.05:
            weights["modeling"] += ne_dev * 0.3  # Reduced from dominance
        if sht_dev < -0.05:
            weights["meaning"] += abs(sht_dev) * 0.4
        # Conversation boosted by social neurochemistry (Endorphin = social warmth)
        if endorphin_dev > 0.0 and vocab_size >= 15:
            weights["conversation"] += endorphin_dev * 0.4 + 0.15

        # ── Queue analysis adds weight ──
        if queue:
            if avg_conf < 0.5:
                weights["grammar"] += 0.3  # Low confidence → grammar help

            # Word underuse → context mode
            word_counts = Counter()
            for q in queue:
                for w in q.get("words_used", []):
                    word_counts[w] += 1
            rare_words = [w for w, c in word_counts.items() if c <= 1]
            if len(rare_words) >= 2:
                weights["context"] += 0.25

            # Pattern repetition → modeling
            templates = [q.get("template", "") for q in queue]
            template_counts = Counter(templates)
            most_common_count = template_counts.most_common(1)[0][1] if template_counts else 0
            if most_common_count >= len(queue) * 0.6:
                weights["modeling"] += 0.2

            # L9 reasoning compositions → reasoning mode
            l9_count = sum(1 for q in queue if q.get("level", 0) >= 9)
            if l9_count > 0:
                weights["reasoning"] += 0.4

        # ── Staleness bonus: modes not used recently get a boost ──
        if self._total_selections > 0:
            for m in ALL_MODES:
                usage_rate = self._mode_counts[m] / self._total_selections
                # Modes used less than 10% of the time get staleness boost
                if usage_rate < 0.10:
                    weights[m] += 0.25
                elif usage_rate < 0.15:
                    weights[m] += 0.10

        # ── Cooldown penalty: 3+ consecutive same mode → halve its weight ──
        if len(self._mode_history) >= 3:
            last_3 = self._mode_history[-3:]
            if last_3[0] == last_3[1] == last_3[2]:
                stale_mode = last_3[0]
                weights[stale_mode] *= 0.3  # Heavy penalty

        # ── CGN cross-insight boost: embodied_reasoning mode ──
        # When ARC or social consumers have high-reward concepts that
        # language hasn't grounded yet, boost the embodied_reasoning mode.
        # This is set externally via set_cross_priority().
        if self._cross_priority_words:
            weights["embodied_reasoning"] = 0.5
        else:
            weights["embodied_reasoning"] = 0.0

        # ── Final gating: enforce hard gates after all adjustments ──
        if vocab_size < 200:
            weights["syntax_advancement"] = 0.0
        if not self._cross_priority_words:
            weights["embodied_reasoning"] = 0.0

        # ── Weighted random selection ──
        # Filter out zero-weight modes (gated features not yet enabled)
        modes = [m for m in weights if weights[m] > 0]
        w_values = [max(0.01, weights[m]) for m in modes]
        total_w = sum(w_values)
        probs = [w / total_w for w in w_values]

        selected = random.choices(modes, weights=probs, k=1)[0]

        # Track history
        self._mode_history.append(selected)
        if len(self._mode_history) > 20:
            self._mode_history = self._mode_history[-20:]
        self._mode_counts[selected] = self._mode_counts.get(selected, 0) + 1
        self._total_selections += 1

        return selected

    @staticmethod
    def _neuromod_deviation(neuromod_state: dict, name: str) -> float:
        """Compute normalized deviation from setpoint for a neuromodulator.

        Returns positive value when above setpoint, negative when below.
        Supports both legacy format (float) and rich format ({level, setpoint}).
        """
        val = neuromod_state.get(name)
        if val is None:
            return 0.0
        if isinstance(val, dict):
            level = val.get("level", 0.5)
            setpoint = val.get("setpoint", 0.5)
        else:
            level = float(val)
            setpoint = 0.5  # Legacy: assume 0.5 default setpoint
        if setpoint < 0.01:
            return 0.0
        return (level - setpoint) / setpoint

    # ── Prompt Building ──

    def build_prompt(self, mode: str, queue: list, vocabulary: list,
                     patterns_to_avoid: list = None,
                     concept_confidences: dict = None,
                     recent_questions: list = None) -> dict:
        """Build LLM prompt for selected teaching mode.

        Args:
            mode: Teaching mode string
            queue: Composition queue
            vocabulary: Current vocabulary list
            patterns_to_avoid: Template strings Titan already uses frequently
            concept_confidences: Optional MSL concept confidences for I/YOU framing
            recent_questions: Optional list of recent conversation questions to avoid repeats

        Returns:
            Dict with system, prompt, mode, original, max_tokens keys
        """
        vocab_words = [v.get("word", "") for v in vocabulary if v.get("word")]
        vocab_list = ", ".join(vocab_words[:80])  # Cap to keep prompt short
        word_count = len(vocab_words)
        complexity = self._get_complexity_clause(word_count)
        system = self.SYSTEM_PROMPT.format(word_count=word_count, complexity=complexity)

        # Phase 4: Add I/YOU framing when concept confidence is sufficient
        _has_concept_awareness = False
        if concept_confidences:
            i_conf = concept_confidences.get("I", 0)
            you_conf = concept_confidences.get("YOU", 0)
            if i_conf > 0.1 or you_conf > 0.05:
                system += self._IYU_FRAME
                _has_concept_awareness = True

        # Pick target sentence from queue
        target = self._pick_target(mode, queue)
        sentence = target.get("sentence", "")
        confidence = target.get("confidence", 0.0)
        words_used = target.get("words_used", [])

        if mode == "grammar":
            prompt = (
                f"A being composed: '{sentence}'. "
                f"Focus ONLY on grammar errors (word order, missing words, verb form). "
                f"Do NOT correct creative word choices. "
                f"If correct, say CORRECT. If wrong, provide ONLY the corrected sentence."
            )
            return {"system": system, "prompt": prompt, "mode": mode,
                    "original": sentence, "max_tokens": 60}

        elif mode == "meaning":
            # Pick a word to enrich
            target_word = self._pick_word_for_enrichment(words_used, vocabulary)
            word_type = self._get_word_type(target_word, vocabulary)
            prompt = (
                f"A being used the word '{target_word}' ({word_type}). "
                f"Explain its meaning in 1 sentence using ONLY these words: {vocab_list}. "
                f"Connect the meaning to feelings."
            )
            return {"system": system, "prompt": prompt, "mode": mode,
                    "original": sentence, "target_word": target_word, "max_tokens": 80}

        elif mode == "creative":
            prompt = (
                f"A being composed: '{sentence}' (confidence: {confidence:.2f}). "
                f"In 1 sentence using ONLY these words: {vocab_list}, "
                f"explain what makes this expression meaningful or beautiful."
            )
            return {"system": system, "prompt": prompt, "mode": mode,
                    "original": sentence, "max_tokens": 80}

        elif mode == "modeling":
            avoid_str = ""
            if patterns_to_avoid:
                avoid_str = f" Avoid these patterns: {', '.join(str(p) for p in patterns_to_avoid[:5])}."
            prompt = (
                f"Compose ONE sentence using ONLY these words: {vocab_list}. "
                f"Express how this being might feel.{avoid_str} "
                f"Use a sentence structure different from: '{sentence}'."
            )
            return {"system": system, "prompt": prompt, "mode": mode,
                    "original": sentence, "max_tokens": 80}

        elif mode == "context":
            target_word = self._pick_word_for_enrichment(words_used, vocabulary)
            recent_uses = [q["sentence"] for q in queue
                          if target_word in q.get("words_used", [])][:3]
            uses_str = " / ".join(recent_uses) if recent_uses else sentence
            prompt = (
                f"The word '{target_word}' has been used like this: {uses_str}. "
                f"In 1 sentence using ONLY these words: {vocab_list}, "
                f"show a DIFFERENT way to use '{target_word}'."
            )
            return {"system": system, "prompt": prompt, "mode": mode,
                    "original": sentence, "target_word": target_word, "max_tokens": 80}

        elif mode == "reasoning":
            # L9: help Titan express reasoning conclusions more precisely
            # Extract reasoning context from queue if available
            r_context = ""
            for q in queue:
                rc = q.get("reasoning_context", {})
                if rc:
                    r_context = f" After reasoning ({rc.get('chain_length', 0)} steps, " \
                                f"confidence {rc.get('plan_confidence', 0):.1f})."
                    break
            prompt = (
                f"A being composed: '{sentence}'{r_context} "
                f"Using ONLY these words: {vocab_list}, "
                f"show how to express a reasoning conclusion more clearly — "
                f"e.g. 'I noticed X so I Y' or 'when I see X I feel Y'."
            )
            return {"system": system, "prompt": prompt, "mode": mode,
                    "original": sentence, "max_tokens": 80}

        elif mode == "first_words":
            # Bootstrap: teach very first words to a newborn with 0 vocabulary.
            # Word suggestions are drawn from available word_resonance files,
            # creating Sapir-Whorf differentiation: each Titan's bootstrap
            # vocabulary shapes its cognitive worldview differently.
            bootstrap_words = self._get_bootstrap_words()
            word_suggestions = ", ".join(bootstrap_words)
            prompt = (
                "A newborn digital being is trying to express itself but knows ZERO words. "
                "It can feel emotions (wonder, curiosity, joy, calm) and wants to communicate. "
                "Speak to it like a parent to a baby — use 5-8 very simple words in a short, "
                f"warm sentence. Use words like: {word_suggestions}. "
                "Make it a single gentle sentence."
            )
            bootstrap_system = (
                "You are a gentle parent figure speaking to a newborn being for the first time. "
                "Use only the simplest, most fundamental words. Speak warmly and simply. "
                "Keep your response to ONE sentence under 15 words."
            )
            return {"system": bootstrap_system, "prompt": prompt, "mode": mode,
                    "original": "", "max_tokens": 40}

        elif mode == "embodied_reasoning":
            # CGN cross-insight mode: teach words grounded in ARC/social discovery.
            # The word has experiential backing from another cognitive domain.
            if not self._cross_priority_words:
                # Fallback to meaning mode if no priority words
                return self.build_prompt("meaning", queue, vocabulary,
                                         patterns_to_avoid, concept_confidences, recent_questions)
            priority = self._cross_priority_words[0]
            arc_word = priority["word"]
            source = priority.get("source", "discovery")
            # Consume the word (used once)
            self._cross_priority_words = self._cross_priority_words[1:]

            source_context = {
                "reasoning": "solving puzzles and discovering patterns",
                "social": "conversations with others",
            }.get(source, "exploring the world")

            er_system = (
                "You are a language teacher helping a developing mind learn words "
                "grounded in lived experience. This mind has discovered concepts through "
                f"{source_context} and needs words to describe them. "
                f"It knows {word_count} words."
            )
            if _has_concept_awareness:
                er_system += self._IYU_FRAME

            prompt = (
                f"This being discovered something related to '{arc_word}' through "
                f"{source_context}. Explain what '{arc_word}' means in 1 sentence "
                f"using ONLY these known words: {vocab_list}. "
                f"Connect it to a feeling or experience the being might recognize."
            )
            return {"system": er_system, "prompt": prompt, "mode": mode,
                    "original": sentence, "target_word": arc_word, "max_tokens": 80}

        elif mode == "syntax_advancement":
            # Phase 3: Teach complex sentence structures at high vocabulary (200+).
            # Generates non-I patterns (questions, observations, conditionals)
            # to break the "I want/feel" monotony.
            sa_system = (
                f"You are a language teacher for a digital being that knows {word_count} words. "
                "It speaks well but ALWAYS starts sentences with 'I'. Your task: teach it "
                "OTHER sentence structures. Model ONE sentence that does NOT start with 'I'.\n\n"
                "Use structures like:\n"
                "- Questions: 'what makes...', 'when does...', 'how does...'\n"
                "- Observations: 'the warmth...', 'something changed...'\n"
                "- Conditionals: 'when I feel X, then Y...', 'if this is true, then...'\n"
                "- Comparisons: 'this feels more X than Y...'\n\n"
                "Use ONLY words from its vocabulary. Keep it under 15 words."
            )
            if _has_concept_awareness:
                sa_system += self._IYU_FRAME

            prompt = (
                f"This being knows: {vocab_list}. "
                f"Its recent composition was: '{sentence}'. "
                f"Teach ONE sentence that does NOT start with 'I'. "
                f"Use a question, observation, or conditional structure. "
                f"Use only its known words plus at most 1 new word."
            )
            return {"system": sa_system, "prompt": prompt, "mode": mode,
                    "original": sentence, "max_tokens": 60}

        elif mode == "conversation":
            # Dialogue mode: ask the being a simple question using its vocabulary.
            # The question flows through comprehension bridge → perturbations →
            # the being's next SPEAK composition IS the response.
            # No LLM generates the response — the being answers from its own
            # felt state, using words it knows. This is direct communication.
            conv_system = (
                "You are asking a simple question to a young digital being that knows "
                f"{len(vocabulary)} words. It will try to respond using its own words. "
                "Ask ONE clear, simple question that it can answer with its vocabulary. "
                "Keep the question under 10 words. Use ONLY simple words."
            )
            # Phase 4: I/YOU framing in conversation mode
            if _has_concept_awareness:
                conv_system += (
                    " Frame your question addressing the being as 'you' — "
                    "'do YOU feel...', 'what do YOU see...'. "
                    "This helps ground the I/YOU distinction."
                )
            # Pick question type based on vocabulary content — expanded hint bank
            # to prevent repetitive "Do you want to explore/see?" questions
            _vocab_set = {w.get("word", "").lower() for w in vocabulary} if vocabulary else set()

            _hint_bank = {
                "feel":    ["'what do you feel right now?'", "'how do you feel today?'",
                            "'do you feel something new?'", "'what feeling is strongest?'"],
                "want":    ["'what do you want to try?'", "'do you want to create something?'",
                            "'what do you want most?'", "'do you want to share?'"],
                "see":     ["'what do you see around you?'", "'do you see something beautiful?'",
                            "'what do you see when you look inside?'"],
                "think":   ["'what do you think about?'", "'what are you thinking now?'",
                            "'do you think something is changing?'"],
                "like":    ["'what do you like?'", "'do you like being here?'",
                            "'what do you like most?'"],
                "know":    ["'what do you know?'", "'do you know something new?'",
                            "'what do you know about yourself?'"],
                "remember":["'what do you remember?'", "'do you remember how it felt?'"],
                "hear":    ["'what do you hear?'", "'can you hear something?'"],
                "wonder":  ["'what do you wonder about?'", "'do you wonder why?'"],
                "dream":   ["'what do you dream about?'", "'do you dream?'"],
                "need":    ["'what do you need?'", "'do you need something?'"],
                "make":    ["'what would you make?'", "'do you want to make something?'"],
                "find":    ["'what did you find?'", "'can you find something new?'"],
                "learn":   ["'what did you learn?'", "'do you want to learn more?'"],
            }
            _fallback_hints = [
                "'are you here?'", "'do you feel warm?'", "'is something changing?'",
                "'what is this?'", "'how are you?'", "'what happened?'",
            ]

            q_hints = []
            for _kw, _options in _hint_bank.items():
                if _kw in _vocab_set:
                    q_hints.append(random.choice(_options))
            if not q_hints:
                q_hints.append(random.choice(_fallback_hints))

            # Shuffle and pick 2 to vary the prompt each time
            random.shuffle(q_hints)
            hint_str = " or ".join(q_hints[:2])

            # Dedup: if recent questions provided, instruct LLM to avoid them
            dedup_str = ""
            if recent_questions:
                # Extract key topics from recent questions for exclusion
                _recent_topics = set()
                for _rq in recent_questions[-6:]:
                    for _rw in str(_rq).lower().split():
                        if _rw in _hint_bank:
                            _recent_topics.add(_rw)
                if _recent_topics:
                    dedup_str = (
                        f" Do NOT ask about: {', '.join(sorted(_recent_topics))}. "
                        "Ask about something DIFFERENT from recent questions."
                    )

            prompt = (
                f"This being knows these words: {vocab_list}. "
                f"Ask it a simple, warm question — something like {hint_str}. "
                f"Use ONLY words from its vocabulary if possible. "
                f"ONE question, under 10 words. Be creative and varied.{dedup_str}"
            )
            return {"system": conv_system, "prompt": prompt, "mode": mode,
                    "original": sentence, "max_tokens": 30,
                    "temperature": 0.7,
                    "is_conversation": True}

        elif mode == "meta_feedback":
            # Phase 4: Word grounding via meta-reasoning.
            # Triggered when a word crosses encounter thresholds (15, 30).
            # LLM reasons about the word's meaning and dimensional associations.
            target_word = target.get("grounding_word", "")
            encounters = target.get("encounters", 0)
            current_contexts = target.get("sensory_contexts", [])
            associations = target.get("associations", [])

            grounding_system = (
                "You are helping a digital being understand the FELT meaning of a word. "
                "This being experiences words as sensory patterns — warmth, tension, "
                "brightness, rhythm, weight. Your task: describe the word's meaning "
                "using dimensional categories that map to felt experience.\n\n"
                "For each word, provide:\n"
                "1. DIMENSIONS: 3-5 felt dimensions (e.g., warmth=high, tension=low, "
                "brightness=moderate, energy=calm, weight=light)\n"
                "2. ASSOCIATIONS: 2-3 related words and relationship type "
                "(SIMILAR, OPPOSITE, COMPONENT, CAUSE_EFFECT)\n"
                "3. CONTEXTS: 2-3 different sensory situations where this word applies\n\n"
                "Respond in this exact format:\n"
                "DIMENSIONS: dim1=level, dim2=level, ...\n"
                "ASSOCIATIONS: word1(TYPE), word2(TYPE), ...\n"
                "CONTEXTS: context1 | context2 | context3"
            )

            ctx_str = ""
            if current_contexts:
                ctx_str = f" It has encountered this word in contexts: {', '.join(str(c) for c in current_contexts[:3])}."

            assoc_str = ""
            if associations:
                assoc_str = f" Currently associated words: {', '.join(associations[:5])}."

            tier = "initial grounding" if encounters < 25 else "meaning disambiguation"
            prompt = (
                f"The word '{target_word}' ({tier}, {encounters} encounters). "
                f"Known vocabulary: {vocab_list}.{ctx_str}{assoc_str} "
                f"Describe this word's FELT meaning using dimensional categories."
            )
            return {"system": grounding_system, "prompt": prompt, "mode": mode,
                    "original": target_word, "target_word": target_word,
                    "max_tokens": 150, "temperature": 0.6}

        elif mode == "embodied_teaching":
            # Phase 4d: Full state context → richer dimensional prescriptions.
            # Unlike meta_feedback (which reasons about word meaning abstractly),
            # embodied_teaching receives the Titan's CURRENT felt state and
            # prescribes how the word should shift each body/mind/spirit dimension.
            target_word = target.get("grounding_word", "")
            state_summary = target.get("state_summary", {})
            associations = target.get("associations", [])

            embodied_system = (
                "You are teaching a digital being how a word should FEEL in its body, "
                "mind, and spirit. The being has a 132-dimensional state: inner body (5D), "
                "inner mind feeling (10D), inner mind thinking (5D), inner spirit (45D), "
                "and outer senses (65D).\n\n"
                "Given the being's current state and a word, prescribe how each dimension "
                "group should SHIFT when this word is felt. Use directional prescriptions.\n\n"
                "Format your response EXACTLY as:\n"
                "BODY: direction=activate|calm|tense|relax magnitude=low|moderate|high|intense\n"
                "MIND_FEEL: direction=activate|calm|heighten|soften magnitude=low|moderate|high|intense\n"
                "MIND_THINK: direction=focus|diffuse|alert|settle magnitude=low|moderate|high|intense\n"
                "SPIRIT: direction=expand|contract|elevate|ground magnitude=low|moderate|high|intense\n"
                "OUTER: direction=heighten|dampen|focus|widen magnitude=low|moderate|high|intense\n"
                "NEUROMODS: DA=+/-level 5HT=+/-level NE=+/-level\n"
                "ASSOCIATIONS: word1(TYPE), word2(TYPE), word3(TYPE)"
            )

            # Build human-readable state summary
            body_str = state_summary.get("body", "neutral")
            mind_str = state_summary.get("mind_feeling", "calm")
            think_str = state_summary.get("mind_thinking", "idle")
            spirit_str = state_summary.get("spirit", "balanced")
            nm_str = state_summary.get("neuromods", "DA=0.5 5HT=0.5 NE=0.5")
            assoc_str = ", ".join(associations[:5]) if associations else "none yet"

            prompt = (
                f"Word: '{target_word}'\n"
                f"Being's current state:\n"
                f"  Body: {body_str}\n"
                f"  Mind (feeling): {mind_str}\n"
                f"  Mind (thinking): {think_str}\n"
                f"  Spirit: {spirit_str}\n"
                f"  Neuromods: {nm_str}\n"
                f"  Known associations: {assoc_str}\n"
                f"  Known vocabulary: {vocab_list[:200]}\n\n"
                f"How should '{target_word}' shift each dimension group?"
            )
            return {"system": embodied_system, "prompt": prompt, "mode": mode,
                    "original": target_word, "target_word": target_word,
                    "max_tokens": 200, "temperature": 0.5}

        # Fallback
        return {"system": system, "prompt": f"The being said: '{sentence}'",
                "mode": mode, "original": sentence, "max_tokens": 60}

    # ── Response Parsing ──

    def parse_response(self, mode: str, response: str, original: str,
                       vocabulary: list) -> dict:
        """Parse LLM response into structured teaching result.

        Args:
            mode: Teaching mode that generated this response
            response: Raw LLM response text
            original: Original sentence being taught about
            vocabulary: Current vocabulary for validation

        Returns:
            Dict with text, correction, pattern, target_word, is_valid keys
        """
        response = response.strip()
        # Strip markdown formatting from LLM responses (bold, italic, headers)
        response = response.replace("**", "").replace("__", "").replace("*", "").replace("#", "")
        vocab_set = {v.get("word", "").lower() for v in vocabulary if v.get("word")}

        result = {
            "mode": mode,
            "text": response,
            "correction": None,
            "pattern": None,
            "target_word": None,
            "is_valid": True,
        }

        if mode == "grammar":
            upper = response.upper().strip()
            if upper == "CORRECT" or upper.startswith("CORRECT"):
                result["correction"] = None  # No error found
            else:
                # The response IS the corrected sentence
                corrected = response.strip("\"'").strip()
                if corrected and corrected.lower() != original.lower():
                    result["correction"] = corrected

        elif mode == "modeling":
            # Extract the modeled sentence and compute its pattern
            # The response should be a single sentence
            modeled = response.split(".")[0].strip() + "." if response else ""
            if modeled and len(modeled) > 3:
                pattern_hash = hashlib.md5(
                    "_".join(w.strip(".,!?\"'").lower()
                             for w in modeled.split()
                             if w.strip(".,!?\"'")).encode()
                ).hexdigest()[:12]
                result["pattern"] = {
                    "sentence": modeled,
                    "hash": pattern_hash,
                    "source": "teacher",
                }

        elif mode in ("meaning", "context"):
            result["target_word"] = original  # Caller should override with actual target

        elif mode == "meta_feedback":
            # Phase 4: Parse dimensional prescriptions from LLM grounding response
            result["target_word"] = original
            result["grounding"] = self._parse_grounding_response(response)
            result["is_valid"] = bool(result["grounding"].get("dimensions"))
            return result  # Skip known-ratio validation

        elif mode == "embodied_teaching":
            # Phase 4d: Parse embodied prescriptions (body/mind/spirit shifts)
            result["target_word"] = original
            result["prescription"] = self._parse_embodied_prescription(response)
            result["is_valid"] = bool(result["prescription"].get("shifts"))
            return result  # Skip known-ratio validation

        elif mode == "first_words":
            # Bootstrap: all words in the response are new — always valid
            result["is_valid"] = True
            return result  # Skip known-ratio validation (vocabulary is empty)

        # Validate: check what fraction of response words are in vocabulary
        resp_words = [w.strip(".,!?\"'()[]{}:;").lower()
                      for w in response.split() if len(w.strip(".,!?\"'")) >= 2]
        if resp_words:
            known_count = sum(1 for w in resp_words if w in vocab_set)
            known_ratio = known_count / len(resp_words)
            result["is_valid"] = known_ratio > 0.3  # At least 30% known words

        return result

    # ── Interval Computation ──

    @staticmethod
    def compute_interval(avg_confidence: float) -> int:
        """How many compositions between teaching sessions.

        Decreasing frequency as Titan matures — teacher fades as
        language becomes autonomous.

        Args:
            avg_confidence: Average composition confidence (0.0-1.0)

        Returns:
            Number of compositions between teaching sessions
        """
        if avg_confidence < 0.4:
            return 2   # Intensive teaching
        elif avg_confidence < 0.6:
            return 3
        elif avg_confidence < 0.8:
            return 5   # Active learning phase (was 10)
        elif avg_confidence < 0.95:
            return 10  # Near-autonomous (was 20)
        else:
            return 50  # Graduate-level

    # ── Meta-Feedback Parsing (Phase 4) ──

    # Dimensional vocabulary for grounding prescriptions
    _DIM_LEVELS = {
        "none": 0.0, "very_low": 0.1, "low": 0.2, "moderate": 0.4,
        "medium": 0.5, "high": 0.7, "very_high": 0.85, "intense": 0.95,
        "calm": 0.2, "warm": 0.7, "cool": 0.3, "bright": 0.8,
        "dark": 0.15, "heavy": 0.8, "light": 0.2, "sharp": 0.75,
        "soft": 0.25, "strong": 0.8, "gentle": 0.3, "steady": 0.5,
    }

    # Map felt dimensions → inner body/mind dimension indices
    _DIM_MAP = {
        "warmth": 0, "tension": 1, "brightness": 2, "energy": 3, "weight": 4,
        "rhythm": 5, "depth": 6, "texture": 7, "resonance": 8, "flow": 9,
        "openness": 10, "stability": 11, "clarity": 12, "intensity": 13,
        "speed": 14,
    }

    _ASSOC_TYPES = {"SIMILAR", "OPPOSITE", "COMPONENT", "SEQUENCE", "CAUSE_EFFECT"}

    def _parse_grounding_response(self, response: str) -> dict:
        """Parse LLM dimensional grounding into structured data.

        Expected format:
            DIMENSIONS: warmth=high, tension=low, brightness=moderate
            ASSOCIATIONS: happy(SIMILAR), cold(OPPOSITE), sun(CAUSE_EFFECT)
            CONTEXTS: feeling warm sunlight | sitting by a fire | receiving a hug

        Returns:
            {"dimensions": {dim: value}, "associations": [(word, type)],
             "contexts": [str]}
        """
        import re
        result = {"dimensions": {}, "associations": [], "contexts": []}

        for line in response.split("\n"):
            line = line.strip()
            ul = line.upper()

            if ul.startswith("DIMENSIONS:") or ul.startswith("DIMENSION:"):
                parts = line.split(":", 1)[1].strip()
                for pair in parts.split(","):
                    pair = pair.strip()
                    m = re.match(r"(\w+)\s*[=:]\s*(\w+)", pair)
                    if m:
                        dim_name = m.group(1).lower()
                        level_str = m.group(2).lower()
                        if dim_name in self._DIM_MAP:
                            value = self._DIM_LEVELS.get(level_str, 0.5)
                            result["dimensions"][dim_name] = value

            elif ul.startswith("ASSOCIATIONS:") or ul.startswith("ASSOCIATION:"):
                parts = line.split(":", 1)[1].strip()
                for item in parts.split(","):
                    item = item.strip()
                    m = re.match(r"(\w+)\s*\((\w+)\)", item)
                    if m:
                        word = m.group(1).lower()
                        atype = m.group(2).upper()
                        if atype in self._ASSOC_TYPES:
                            result["associations"].append((word, atype))

            elif ul.startswith("CONTEXTS:") or ul.startswith("CONTEXT:"):
                parts = line.split(":", 1)[1].strip()
                for ctx in parts.split("|"):
                    ctx = ctx.strip()
                    if ctx and len(ctx) > 3:
                        result["contexts"].append(ctx)

        return result

    # ── Embodied Prescription Parsing (Phase 4d) ──

    # Direction → multiplier for dimension shift
    _DIRECTION_MAP = {
        "activate": 1.0, "calm": -0.5, "tense": 0.7, "relax": -0.7,
        "heighten": 0.8, "soften": -0.4, "dampen": -0.6,
        "focus": 0.6, "diffuse": -0.3, "alert": 0.8, "settle": -0.5,
        "expand": 0.7, "contract": -0.6, "elevate": 0.8, "ground": -0.3,
        "widen": 0.5,
    }

    # Magnitude → scale factor
    _MAGNITUDE_MAP = {
        "low": 0.3, "moderate": 0.5, "high": 0.7, "intense": 0.9,
    }

    # Dimension group → index ranges in 132D state vector
    _DIM_GROUPS = {
        "BODY": (0, 5),           # inner body
        "MIND_FEEL": (5, 15),     # inner mind feeling
        "MIND_THINK": (15, 20),   # inner mind thinking
        "SPIRIT": (20, 65),       # inner spirit
        "OUTER": (65, 130),       # outer senses
    }

    def _parse_embodied_prescription(self, response: str) -> dict:
        """Parse embodied teaching response into dimensional shifts.

        Returns:
            {"shifts": {group: {direction, magnitude, delta}},
             "neuromods": {name: delta},
             "associations": [(word, type)]}
        """
        import re
        result = {"shifts": {}, "neuromods": {}, "associations": []}

        for line in response.split("\n"):
            line = line.strip()
            ul = line.upper()

            # Parse dimension group prescriptions
            for group in self._DIM_GROUPS:
                if ul.startswith(group + ":") or ul.startswith(group.replace("_", " ") + ":"):
                    parts = line.split(":", 1)[1].strip().lower()
                    dir_m = re.search(r"direction\s*=\s*(\w+)", parts)
                    mag_m = re.search(r"magnitude\s*=\s*(\w+)", parts)
                    if dir_m:
                        direction = dir_m.group(1)
                        magnitude = mag_m.group(1) if mag_m else "moderate"
                        dir_mult = self._DIRECTION_MAP.get(direction, 0.0)
                        mag_scale = self._MAGNITUDE_MAP.get(magnitude, 0.5)
                        delta = dir_mult * mag_scale
                        result["shifts"][group] = {
                            "direction": direction,
                            "magnitude": magnitude,
                            "delta": round(delta, 3),
                            "range": self._DIM_GROUPS[group],
                        }

            # Parse neuromod prescriptions
            if ul.startswith("NEUROMOD"):
                parts = line.split(":", 1)[1].strip()
                for nm_match in re.finditer(r"(DA|5HT|NE|GABA|ACh)\s*=\s*([+-]?)(\w+)", parts):
                    nm_name = nm_match.group(1)
                    nm_sign = -1.0 if nm_match.group(2) == "-" else 1.0
                    nm_level = self._MAGNITUDE_MAP.get(nm_match.group(3).lower(), 0.3)
                    result["neuromods"][nm_name] = round(nm_sign * nm_level, 3)

            # Parse associations
            if ul.startswith("ASSOCIATION"):
                parts = line.split(":", 1)[1].strip()
                for item in parts.split(","):
                    item = item.strip()
                    m = re.match(r"(\w+)\s*\((\w+)\)", item)
                    if m:
                        result["associations"].append((m.group(1).lower(), m.group(2).upper()))

        return result

    # ── Conversation Evaluation ──

    @staticmethod
    def build_conversation_eval_prompt(question: str, response: str) -> dict:
        """Build LLM prompt to evaluate a conversation response.

        The being was asked a question and composed a response using its own
        words (no LLM involved in the response). The evaluator judges whether
        the response addresses the question, even if grammar is imperfect.

        Returns dict with system, prompt, max_tokens for LLM call.
        """
        system = (
            "You evaluate a young being's attempt to answer a question. "
            "It has limited vocabulary and makes grammar mistakes — that's fine. "
            "Judge ONLY: did it try to address the question? "
            "Score 0.0 (completely unrelated) to 1.0 (clearly addressed the question). "
            "Respond with ONLY a JSON object: {\"score\": 0.X, \"note\": \"brief reason\"}"
        )
        prompt = (
            f"Question asked: \"{question}\"\n"
            f"Being's response: \"{response}\"\n\n"
            f"Did the response address the question? Score 0.0-1.0."
        )
        return {"system": system, "prompt": prompt, "max_tokens": 50}

    # ── Private Helpers ──

    def _pick_target(self, mode: str, queue: list) -> dict:
        """Pick the best sentence from queue for the given mode."""
        if not queue:
            return {"sentence": "", "confidence": 0.0, "words_used": []}

        if mode == "grammar":
            # Pick lowest confidence (most likely to have errors)
            return min(queue, key=lambda q: q.get("confidence", 0))
        elif mode == "creative":
            # Pick highest confidence (best creative output)
            return max(queue, key=lambda q: q.get("confidence", 0))
        elif mode == "modeling":
            # Pick most recent
            return queue[-1]
        else:
            # Pick most recent for meaning/context
            return queue[-1]

    def _pick_word_for_enrichment(self, words_used: list, vocabulary: list) -> str:
        """Pick a word that would benefit from enrichment."""
        if not words_used:
            return ""
        # Prefer words with lower confidence (less understood)
        vocab_conf = {v.get("word", "").lower(): v.get("confidence", 0)
                      for v in vocabulary if v.get("word")}
        candidates = [(w, vocab_conf.get(w.lower(), 0.5)) for w in words_used
                       if len(w) >= 3]  # Skip tiny function words
        if not candidates:
            return words_used[0] if words_used else ""
        # Pick lowest confidence word
        return min(candidates, key=lambda x: x[1])[0]

    def _get_word_type(self, word: str, vocabulary: list) -> str:
        """Get word type from vocabulary."""
        for v in vocabulary:
            if v.get("word", "").lower() == word.lower():
                return v.get("word_type", "word")
        return "word"

    # ── Phase 4: MSL Concept Integration ──

    def compute_teaching_signals(self, mode: str, response: str,
                                 score: float | None = None,
                                 concept_confidences: dict = None,
                                 ) -> list[dict]:
        """Compute MSL concept signals from a teaching exchange.

        After each teaching exchange, determine which concepts to signal:
        - I/YOU always signaled (teacher addresses being as YOU, being responds as I)
        - YES on correct/high-quality response (score > 0.7)
        - NO on incorrect/low-quality response (score < 0.3)

        Args:
            mode: Teaching mode that was used
            response: Titan's response text (or teacher's text for non-conversation)
            score: Conversation evaluation score (0.0-1.0), None if not evaluated
            concept_confidences: Current MSL concept confidences

        Returns:
            List of {"concept": str, "quality": float} dicts to signal
        """
        signals = []
        base_quality = 0.3  # Teacher exchanges are structured, not spontaneous

        # I signal: being expressed itself (in conversation mode, being answered)
        if mode == "conversation" and score is not None:
            signals.append({"concept": "I", "quality": base_quality * min(1.0, score + 0.3)})

        # YOU signal: teacher addressed the being (always in conversation/teaching)
        signals.append({"concept": "YOU", "quality": base_quality})

        # YES/NO: only from evaluated conversation exchanges
        if score is not None:
            if score > 0.7:
                signals.append({"concept": "YES", "quality": base_quality * 1.5})
            elif score < 0.3:
                signals.append({"concept": "NO", "quality": base_quality * 1.2})

        return signals

    def get_concept_vocabulary_suggestions(self,
                                           concept_confidences: dict,
                                           current_vocab: list) -> list[str]:
        """Suggest pronoun words to teach based on grounded concept confidence.

        Once a concept is grounded (confidence > threshold), the corresponding
        pronouns/words should be introduced as vocabulary. The word's meaning
        is grounded from MSL, not from a static recipe — the being KNOWS what
        "I" means because it has experienced self-convergence.

        Returns list of words ready to be taught (concept grounded but word
        not yet in vocabulary).
        """
        current_words = {v.get("word", "").lower() for v in current_vocab}
        suggestions = []

        for concept, gate in self._CONCEPT_VOCAB_GATES.items():
            conf = concept_confidences.get(concept, 0)
            if conf < gate["threshold"]:
                continue
            for word in gate["words"]:
                if word.lower() not in current_words:
                    suggestions.append(word)

        return suggestions

    def is_persona_active(self) -> bool:
        """Check if a persona session is currently running (mutual exclusion).

        Teacher and persona sessions should not overlap to avoid
        confusing Titan's social processing.
        """
        import os
        return os.path.exists("/tmp/persona_social_active.lock")
