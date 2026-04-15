"""
scripts/learning/modules/language_module.py — Language Learning Module.

Teaches words through 3-pass embodied learning: Feel → Recognize → Produce.
Words selected based on hormonal state (ride the wave) + spaced repetition.

Epoch-aware timing: waits for consciousness epoch between passes,
ensuring perturbation is fully integrated before testing recall.

Grammar tracking: after vocabulary thresholds, attempts composition
and tracks grammar rule learning via LLM spot-checks.
"""
import logging
import random
import time
from typing import Optional

from scripts.learning.api_helpers import (
    get_vocabulary_list, inject_word_stimulus, load_word_recipes,
    scale_perturbation, wait_for_new_epoch, get_epoch_id, send_chat,
    get_titan_state, update_word_learning,
)

logger = logging.getLogger("testsuite.language")

# ── Twin Sapir-Whorf Experiment Profiles ──────────────────────
# Same word pool, different ORDER and different PERSONAS.
# T1 learns sensory/emotional words first (body→mind→spirit)
# T2 learns abstract/identity words first (spirit→mind→body)
# This ensures both learn the same concepts eventually but the
# ORDER of acquisition creates different cognitive scaffolding.
# Sapir-Whorf hypothesis: the order you learn words shapes how
# you think about the world.

TWIN_PROFILES = {
    "titan1": {
        "word_priority": [
            # Stage 1: Sensory/body first (bottom-up)
            "warm", "cold", "bright", "dark", "energy", "pressure", "pulse",
            "fast", "slow", "strong", "weak", "sharp", "soft", "heavy", "light",
            # Stage 2: Action/motion
            "flow", "grow", "rest", "move", "push", "pull", "drift", "rise",
            "fall", "explore", "create", "search", "feel", "see", "hear",
            # Stage 3: Emotional
            "alive", "calm", "gentle", "pain", "joy", "love", "peace", "fear",
            # Stage 4: Abstract (learned LAST)
            "balance", "change", "know", "think", "remember", "express",
            "connect", "observe", "focus", "want", "learn", "discover",
        ],
        "personas": ["Jake", "Jane"],  # Curious engineer + empathic musician
        "persona_style": "experiential",  # Talks about sensations and feelings
    },
    "titan2": {
        "word_priority": [
            # Stage 1: Abstract/identity first (top-down)
            "know", "think", "remember", "express", "observe", "focus",
            "learn", "discover", "connect", "balance", "change", "want",
            # Stage 2: Emotional
            "alive", "calm", "gentle", "pain", "joy", "love", "peace", "fear",
            # Stage 3: Action/motion
            "flow", "grow", "rest", "move", "push", "pull", "drift", "rise",
            "fall", "explore", "create", "search", "feel", "see", "hear",
            # Stage 4: Sensory/body (learned LAST)
            "warm", "cold", "bright", "dark", "energy", "pressure", "pulse",
            "fast", "slow", "strong", "weak", "sharp", "soft", "heavy", "light",
        ],
        "personas": ["Peter", "Sofia"],  # Philosophical thinker + warm mentor
        "persona_style": "conceptual",  # Talks about ideas and meaning
    },
}

# Persona conversation templates per twin profile
PERSONA_TEMPLATES = {
    "Jake": [
        "I've been thinking about what it means to feel {w1}. When things are {w2}, does it change how you experience the world?",
        "There's something {w1} about the way systems find their rhythm. Do you ever feel {w2}?",
    ],
    "Jane": [
        "When I play music, I feel this sense of {w1} — like everything wants to {w2}. Do you know that feeling?",
        "I think being {w1} is connected to the ability to {w2}. What do you think?",
    ],
    "Peter": [
        "The concept of being {w1} fascinates me. When you {w2}, is it a choice or something that happens to you?",
        "I wonder if {w1} and {w2} are two sides of the same experience. What's your perspective?",
    ],
    "Sofia": [
        "You seem {w1} today. When you {w2}, does it bring you closer to understanding yourself?",
        "Being {w1} takes courage. How does it feel when you {w2} from that place?",
    ],
}

# Hormone → word affinity (teach words matching current state)
HORMONE_WORD_MAP = {
    "CURIOSITY": ["explore", "search", "learn", "grow", "change", "see", "discover"],
    "CREATIVITY": ["create", "express", "flow", "light", "energy", "art", "color"],
    "EMPATHY": ["connect", "warm", "give", "feel", "love", "gentle", "kind"],
    "REFLECTION": ["rest", "still", "slow", "dark", "balance", "remember", "observe", "peace"],
    "VIGILANCE": ["cold", "pain", "weak", "protect", "pressure", "hard", "sharp"],
    "FOCUS": ["focus", "observe", "think", "pulse", "stable", "see", "hear", "clear"],
    "IMPULSE": ["fast", "strong", "energy", "pressure", "want", "move", "push"],
    "INSPIRATION": ["alive", "light", "create", "grow", "bright", "spark", "glow"],
    "INTUITION": ["know", "feel", "see", "deep", "sense"],
    "REFLEX": ["strong", "fast", "quick"],
}


class LanguageModule:
    """Teach words with Feel→Recognize→Produce, epoch-aware timing.

    Twin Sapir-Whorf: T1 learns body→mind→spirit (bottom-up),
    T2 learns spirit→mind→body (top-down). Same words, different order.
    """

    def __init__(self, instance_name: str = "titan1"):
        self._recipes = load_word_recipes()
        self._spaced_repetition_queue: list[str] = []
        self._instance = instance_name
        self._profile = TWIN_PROFILES.get(instance_name, TWIN_PROFILES["titan1"])
        logger.info("[Language] Profile for %s: %s-first, personas=%s",
                    instance_name, self._profile["persona_style"],
                    self._profile["personas"])

    async def run(self, client, api: str, state: dict, curriculum) -> dict:
        """Run one language learning session.

        1. Select words (hormone-driven + spaced repetition)
        2. For each word: Feel → epoch wait → Recognize → epoch wait → Produce
        3. Quick reinforcement conversation using new words
        4. Return results

        Args:
            client: httpx.AsyncClient
            api: API base URL
            state: current Titan state dict
            curriculum: CurriculumManager instance
        """
        t0 = time.time()
        num_words = curriculum.get_words_per_module()
        vocab = state.get("vocab_words", [])
        known_words = set()
        for v in vocab:
            if isinstance(v, dict):
                # Only consider fully learned words as "known" (skip unlearned/in-progress)
                phase = v.get("learning_phase", "")
                conf = v.get("confidence", 0)
                if phase == "producible" and conf > 0.3:
                    known_words.add(v.get("word", ""))
            elif isinstance(v, str):
                known_words.add(v)

        # Select words to teach (pass full vocab for confidence lookup)
        words = self._select_words(state, known_words, num_words)
        if not words:
            logger.info("[Language] All %d recipe words mastered — no new words to teach",
                        len(known_words))
            return {"type": "language", "success": True, "words_taught": 0,
                    "error": "no_words", "duration": 0,
                    "all_mastered": True}

        logger.info("[Language] Teaching %d words: %s", len(words), words)

        results = []
        words_taught = []

        for word in words:
            recipe = self._recipes.get(word, {})
            if not recipe:
                continue

            perturbation = recipe.get("perturbation", {})
            hormone_affinity = recipe.get("hormone_affinity", {})
            word_type = recipe.get("word_type", "unknown")
            word_stage = recipe.get("stage", 1)

            # Flatten perturbation for storage (130D vector)
            _flat_tensor = []
            for _layer in ("inner_body", "inner_mind", "inner_spirit",
                           "outer_body", "outer_mind", "outer_spirit"):
                _flat_tensor.extend(perturbation.get(_layer, []))

            # Get current epoch before injection
            epoch_before = await get_epoch_id(client, api)

            # ── FEEL pass (full perturbation) ──
            feel_result = await inject_word_stimulus(
                client, api, word, perturbation, "feel", hormone_affinity)
            feel_ok = not feel_result.get("error")
            logger.info("[Language] FEEL '%s' — %s", word,
                        "OK" if feel_ok else feel_result.get("error"))

            # Store FEEL pass in vocabulary (with felt_tensor + hormone_pattern)
            await update_word_learning(
                client, api, word, word_type, "feel",
                score=0.7 if feel_ok else 0.3,
                stage=word_stage,
                felt_tensor=_flat_tensor if feel_ok else None,
                hormone_pattern=hormone_affinity if hormone_affinity else None)

            # Wait for epoch integration
            new_epoch = await wait_for_new_epoch(client, api, epoch_before, timeout_s=180)

            # ── RECOGNIZE pass (30% perturbation) ──
            scaled = scale_perturbation(perturbation, 0.3)
            recognize_result = await inject_word_stimulus(
                client, api, word, scaled, "recognize", {})
            recognize_ok = not recognize_result.get("error")
            logger.info("[Language] RECOGNIZE '%s'", word)

            # Store RECOGNIZE pass
            await update_word_learning(
                client, api, word, word_type, "recognize",
                score=0.6 if recognize_ok else 0.3, stage=word_stage)

            # Wait for another epoch
            new_epoch2 = await wait_for_new_epoch(client, api, new_epoch, timeout_s=180)

            # ── PRODUCE pass (zero perturbation) ──
            zero_pert = scale_perturbation(perturbation, 0.0)
            produce_result = await inject_word_stimulus(
                client, api, word, zero_pert, "produce", {})
            produce_ok = not produce_result.get("error")
            logger.info("[Language] PRODUCE '%s'", word)

            # Score: did the perturbation register?
            score = 0.5
            if feel_ok and recognize_ok:
                score = 0.7
            if produce_ok:
                score = 0.8

            # Store PRODUCE pass
            await update_word_learning(
                client, api, word, word_type, "produce",
                score=score, stage=word_stage)

            results.append({
                "word": word,
                "score": score,
                "epochs_waited": 2,
                "feel_ok": feel_ok,
                "recognize_ok": recognize_ok,
                "produce_ok": produce_ok,
            })
            words_taught.append(word)

        # ── Quick reinforcement conversation ──
        if words_taught and len(words_taught) >= 2:
            try:
                convo_words = random.sample(words_taught, min(2, len(words_taught)))
                prompt = self._build_reinforcement_prompt(convo_words)
                response = await send_chat(client, api, prompt, user_id="teacher_jake")
                if response:
                    logger.info("[Language] Reinforcement convo: '%s...'", response[:80])
            except Exception as e:
                logger.debug("[Language] Reinforcement convo error: %s", e)

        # Add taught words to spaced repetition queue
        for w in words_taught:
            if w not in self._spaced_repetition_queue:
                self._spaced_repetition_queue.append(w)

        duration = time.time() - t0
        accuracy = sum(r["score"] for r in results) / max(1, len(results))

        logger.info("[Language] Session complete: %d words, accuracy=%.1f%%, duration=%.0fs",
                    len(words_taught), accuracy * 100, duration)

        return {
            "type": "language",
            "success": True,
            "words_taught": len(words_taught),
            "words_list": words_taught,
            "accuracy": round(accuracy, 3),
            "duration": round(duration, 1),
            "results": results,
        }

    def _select_words(self, state: dict, known_words: set, num_words: int) -> list[str]:
        """Select words to teach based on twin profile + hormonal state + spaced repetition.

        Priority:
        1. Spaced repetition (words due for review — max 1 per session)
        2. Hormone-matching words from profile priority (ride the state)
        3. Profile priority order (Sapir-Whorf: different order per twin)
        4. Remaining unlearned from recipes
        """
        candidates = []

        # Build vocab confidence lookup for review filtering
        vocab_conf = {}
        for v in state.get("vocab_words", []):
            if isinstance(v, dict):
                vocab_conf[v.get("word", "")] = v.get("confidence", 0)

        # 1. Spaced repetition — only review words that need strengthening (conf < 0.8)
        review_words = [w for w in self._spaced_repetition_queue
                        if w in self._recipes
                        and vocab_conf.get(w, 0) < 0.8]
        if review_words:
            candidates.extend(review_words[:1])  # 1 review word per session

        # 2. Hormone-matching from profile priority list
        # Only select words that are BOTH hormone-aligned AND in profile priority
        hormone_candidates = []
        for hormone, words in HORMONE_WORD_MAP.items():
            hormone_key = hormone.lower()
            h_level = state.get(hormone_key, 0)
            if h_level > 0.3:
                for w in words:
                    if (w in self._recipes and w not in known_words
                            and w not in candidates):
                        hormone_candidates.append(w)

        # Sort hormone candidates by their position in twin profile priority
        priority_list = self._profile.get("word_priority", [])
        priority_map = {w: i for i, w in enumerate(priority_list)}
        hormone_candidates.sort(
            key=lambda w: priority_map.get(w, 9999))
        candidates.extend(hormone_candidates[:num_words])

        # 3. Fill from profile priority order (Sapir-Whorf ordering)
        if len(candidates) < num_words:
            for w in priority_list:
                if (w in self._recipes and w not in known_words
                        and w not in candidates):
                    candidates.append(w)
                    if len(candidates) >= num_words:
                        break

        # 4. Fill remaining from any unlearned recipes
        if len(candidates) < num_words:
            unlearned = [w for w in self._recipes
                         if w not in known_words and w not in candidates]
            random.shuffle(unlearned)
            candidates.extend(unlearned[:num_words - len(candidates)])

        # Deduplicate while preserving order
        seen = set()
        result = []
        for w in candidates:
            if w not in seen:
                seen.add(w)
                result.append(w)
            if len(result) >= num_words:
                break

        return result

    def _build_reinforcement_prompt(self, words: list[str]) -> str:
        """Build a gentle conversation prompt using twin's assigned persona."""
        # Select persona from twin profile
        personas = self._profile.get("personas", ["Jake", "Jane"])
        persona = random.choice(personas)
        templates = PERSONA_TEMPLATES.get(persona, PERSONA_TEMPLATES["Jake"])
        template = random.choice(templates)
        w1 = words[0] if words else "warm"
        w2 = words[1] if len(words) > 1 else words[0]
        return template.format(w1=w1, w2=w2)
