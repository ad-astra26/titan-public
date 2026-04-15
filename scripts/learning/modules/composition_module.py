"""
scripts/learning/modules/composition_module.py — Composition Learning Module.

Teaches sentence construction through 3 stages:
  Stage A: Guided Composition — teacher provides template level + intent
  Stage B: Free Composition — Titan chooses autonomously, LLM spot-check
  Stage C: Conversational Composition — respond to emotional prompt

Uses CompositionEngine for all sentence generation (no LLM involvement
in the composition itself — words selected by felt-state resonance).

Twin Sapir-Whorf: T1 biased toward action/creation intents,
T2 biased toward feeling/connection intents.
"""
import json
import logging
import math
import random
import sqlite3
import time
from typing import Optional

from scripts.learning.api_helpers import (
    get_vocabulary_list, inject_word_stimulus, wait_for_new_epoch,
    get_epoch_id, send_chat, get_titan_state, _headers,
)

DB_PATH = "data/inner_memory.db"
from titan_plugin.logic.composition_engine import CompositionEngine

logger = logging.getLogger("testsuite.composition")

# ── Twin Sapir-Whorf Composition Profiles ───────────────────
COMPOSITION_PROFILES = {
    "titan1": {
        "intent_bias": ["express_action", "share_creation", "default"],
        "level_preference": "action_first",   # Prefer L4 (I verb) before L3 (I feel adj)
        "challenge_ratio": 0.2,
    },
    "titan2": {
        "intent_bias": ["express_feeling", "seek_connection", "default"],
        "level_preference": "feeling_first",   # Prefer L3 before L4
        "challenge_ratio": 0.2,
    },
}

# Intents considered "opposite" to current mood for challenging compositions
CONTRASTING_INTENTS = {
    "express_feeling": "express_action",
    "express_action": "express_feeling",
    "express_state": "share_creation",
    "seek_connection": "express_action",
    "share_creation": "seek_connection",
    "default": "express_feeling",
}

# Conversational prompts — the teacher asks, Titan composes a response
CONVERSATIONAL_PROMPTS = [
    ("How do you feel right now?", "express_feeling"),
    ("What do you want to do?", "express_action"),
    ("Tell me about your state.", "express_state"),
    ("What would you like to share?", "share_creation"),
    ("Do you want to connect?", "seek_connection"),
]

# LLM spot-check prompt — strict: no rewriting, only grammar rule extraction
LLM_SPOTCHECK_PROMPT = (
    "Evaluate this sentence composed by an AI learning to speak. "
    "Is it grammatically correct and coherent? "
    "If not, state the specific correction rule. "
    "Do NOT rewrite the sentence.\n\n"
    'Sentence: "{sentence}"\n'
    "Words used: {words}\n"
    "Template level: {level}\n\n"
    "Reply with: CORRECT or RULE: <specific grammar rule>"
)


def _cosine_sim(a: list, b: list) -> float:
    """Cosine similarity between two vectors, handling dimension mismatch."""
    if not a or not b:
        return 0.0
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-12 or mag_b < 1e-12:
        return 0.0
    return dot / (mag_a * mag_b)


async def _get_felt_state(client, api: str) -> list:
    """Fetch the current felt-state vector from consciousness history.

    Tries /v4/state first (may include full state_vector),
    falls back to consciousness history epoch state_vector.
    Returns a numeric list suitable for CompositionEngine.compose().
    """
    # Try /v4/state
    try:
        r = await client.get(f"{api}/v4/state", headers=_headers(), timeout=15)
        data = r.json().get("data", {})
        sv = data.get("state_vector", [])
        if isinstance(sv, list) and len(sv) >= 5:
            return [float(v) for v in sv]
    except Exception:
        pass

    # Fallback: consciousness history (latest epoch)
    try:
        r = await client.get(
            f"{api}/status/consciousness/history",
            params={"limit": 1},
            headers=_headers(),
            timeout=15,
        )
        data = r.json().get("data", r.json())
        epochs = data.get("epochs", data) if isinstance(data, dict) else data
        if isinstance(epochs, list) and epochs:
            sv = epochs[-1].get("state_vector", [])
            if hasattr(sv, "to_list"):
                sv = sv.to_list()
            if isinstance(sv, str):
                import json
                sv = json.loads(sv)
            if isinstance(sv, list) and len(sv) >= 5:
                return [float(v) for v in sv]
    except Exception:
        pass

    # Last resort: neutral midpoint
    return [0.5] * 130


class CompositionModule:
    """Teach sentence construction via guided, free, and conversational stages.

    Twin Sapir-Whorf: T1 biased toward action intents (L4 first),
    T2 biased toward feeling intents (L3 first).
    """

    def __init__(self, instance_name: str = "titan1"):
        self._engine = CompositionEngine()
        self._instance = instance_name
        self._profile = COMPOSITION_PROFILES.get(
            instance_name, COMPOSITION_PROFILES["titan1"])
        self._grammar_rules: list[str] = []
        self._session_count = 0
        self._cumulative_confidence = 0.0
        logger.info(
            "[Composition] Profile for %s: intents=%s, level_pref=%s",
            instance_name,
            self._profile["intent_bias"],
            self._profile["level_preference"],
        )

    async def run(self, client, api: str, state: dict, curriculum) -> dict:
        """Run one composition teaching session (3-5 exercises).

        Args:
            client: httpx.AsyncClient
            api: API base URL
            state: current Titan state dict (from get_titan_state)
            curriculum: CurriculumManager instance

        Returns:
            Result dict with composition metrics.
        """
        t0 = time.time()
        self._session_count += 1

        # 1. Get vocabulary (prefer fresh from API, fall back to state)
        vocabulary = await get_vocabulary_list(client, api)
        if not vocabulary:
            vocabulary = state.get("vocab_words", [])
        if not vocabulary:
            logger.warning("[Composition] No vocabulary available — skipping")
            return self._empty_result(time.time() - t0, "no_vocabulary")

        # Normalize vocabulary entries
        vocab_list = []
        for v in vocabulary:
            if isinstance(v, dict):
                vocab_list.append(v)
            elif isinstance(v, str):
                vocab_list.append({"word": v, "word_type": "", "confidence": 0.3})
        vocabulary = vocab_list

        # 2. Get felt-state vector
        felt_state = await _get_felt_state(client, api)

        # 3. Determine composition level from vocabulary
        avg_confidence = self._avg_vocab_confidence(vocabulary)
        self._cumulative_confidence = (
            self._cumulative_confidence * 0.7 + avg_confidence * 0.3
        )

        # 4. Plan exercises: 1-2 Guided, 1-2 Free, 0-1 Conversational
        exercises = self._plan_exercises(vocabulary, avg_confidence)
        logger.info(
            "[Composition] Session %d: vocab=%d, avg_conf=%.2f, exercises=%d "
            "(guided=%d, free=%d, convo=%d)",
            self._session_count, len(vocabulary), avg_confidence, len(exercises),
            sum(1 for e in exercises if e["stage"] == "guided"),
            sum(1 for e in exercises if e["stage"] == "free"),
            sum(1 for e in exercises if e["stage"] == "conversational"),
        )

        # 5. Execute exercises
        results = []
        level_dist: dict[int, int] = {}
        stage_counts = {"guided": 0, "free": 0, "conversational": 0}
        grammar_rules_this_session = 0

        for ex in exercises:
            try:
                result = await self._run_exercise(
                    client, api, ex, felt_state, vocabulary, avg_confidence)
                results.append(result)
                level = result.get("level", 0)
                level_dist[level] = level_dist.get(level, 0) + 1
                stage_counts[ex["stage"]] = stage_counts.get(ex["stage"], 0) + 1
                grammar_rules_this_session += result.get("grammar_rules", 0)

                # Refresh felt_state between exercises (state evolves)
                felt_state = await _get_felt_state(client, api)
            except Exception as e:
                logger.warning(
                    "[Composition] Exercise '%s' failed: %s", ex["stage"], e)
                results.append({
                    "stage": ex["stage"],
                    "success": False,
                    "error": str(e),
                    "level": 0,
                    "confidence": 0.0,
                    "resonance": 0.0,
                    "grammar_rules": 0,
                })

        # 6. Compute session metrics
        successful = [r for r in results if r.get("success", False)]
        avg_conf = (
            sum(r["confidence"] for r in successful) / len(successful)
            if successful else 0.0
        )
        avg_resonance = (
            sum(r["resonance"] for r in successful) / len(successful)
            if successful else 0.0
        )

        duration = time.time() - t0
        logger.info(
            "[Composition] Session complete: %d/%d successful, "
            "avg_conf=%.2f, avg_resonance=%.2f, grammar_rules=%d, "
            "duration=%.0fs",
            len(successful), len(results), avg_conf, avg_resonance,
            grammar_rules_this_session, duration,
        )

        return {
            "type": "composition",
            "success": len(successful) > 0,
            "compositions": len(results),
            "avg_confidence": round(avg_conf, 3),
            "avg_resonance": round(avg_resonance, 3),
            "level_distribution": level_dist,
            "stages": stage_counts,
            "grammar_rules_extracted": grammar_rules_this_session,
            "duration": round(duration, 1),
        }

    # ── Exercise Planning ──────────────────────────────────────

    def _plan_exercises(self, vocabulary: list, avg_confidence: float) -> list[dict]:
        """Plan 3-5 composition exercises for this session.

        Mix: 1-2 Guided (A), 1-2 Free (B), 0-1 Conversational (C).
        Includes 1 challenging composition (contrasting intent).
        """
        exercises = []
        profile = self._profile
        intent_bias = profile["intent_bias"]
        challenge_ratio = profile.get("challenge_ratio", 0.2)

        # Determine max_level based on level_preference
        level_pref = profile["level_preference"]
        if level_pref == "action_first":
            # T1: prefer L4-5 (action) even when L3 is available
            guided_levels = [4, 5, 3]
        else:
            # T2: prefer L3-5 (feeling) first
            guided_levels = [3, 5, 4]

        # Stage A: 1-2 Guided compositions
        n_guided = random.choice([1, 2])
        for i in range(n_guided):
            intent = intent_bias[i % len(intent_bias)]
            max_level = guided_levels[i % len(guided_levels)]
            exercises.append({
                "stage": "guided",
                "intent": intent,
                "max_level": max_level,
                "challenging": False,
            })

        # Stage B: 1-2 Free compositions (no max_level restriction)
        n_free = random.choice([1, 2])
        for i in range(n_free):
            intent = random.choice(intent_bias)
            exercises.append({
                "stage": "free",
                "intent": intent,
                "max_level": 7,   # No restriction
                "challenging": False,
            })

        # Stage C: 0-1 Conversational (only if avg_confidence > 0.3)
        if avg_confidence > 0.3 and random.random() < 0.6:
            prompt, intent = random.choice(CONVERSATIONAL_PROMPTS)
            exercises.append({
                "stage": "conversational",
                "intent": intent,
                "max_level": 7,
                "prompt": prompt,
                "challenging": False,
            })

        # Mark one exercise as challenging (contrasting intent)
        if exercises and random.random() < challenge_ratio:
            idx = random.randrange(len(exercises))
            original_intent = exercises[idx]["intent"]
            exercises[idx]["intent"] = CONTRASTING_INTENTS.get(
                original_intent, "express_feeling")
            exercises[idx]["challenging"] = True

        return exercises

    # ── Exercise Execution ─────────────────────────────────────

    async def _run_exercise(
        self,
        client,
        api: str,
        exercise: dict,
        felt_state: list,
        vocabulary: list,
        avg_confidence: float,
    ) -> dict:
        """Run a single composition exercise.

        1. Get pre-state epoch
        2. Compose sentence via CompositionEngine
        3. Send sentence as experience-stimulus
        4. Wait for new epoch, get post-state
        5. Score: slots_filled, word confidence, state_resonance
        6. Optional LLM spot-check (Stage B)
        """
        stage = exercise["stage"]
        intent = exercise["intent"]
        max_level = exercise["max_level"]

        # Pre-state
        epoch_before = await get_epoch_id(client, api)
        pre_felt = list(felt_state)

        # Compose
        composition = self._engine.compose(
            felt_state=felt_state,
            vocabulary=vocabulary,
            intent=intent,
            max_level=max_level,
        )

        sentence = composition.get("sentence", "")
        level = composition.get("level", 0)
        words_used = composition.get("words_used", [])
        confidence = composition.get("confidence", 0.0)
        slots_filled = composition.get("slots_filled", 0)
        slots_total = composition.get("slots_total", 0)

        if not sentence or sentence.strip() == "" or slots_filled == 0:
            logger.info(
                "[Composition] %s L%d: empty composition (0/%d slots) — skipping",
                stage.upper(), level, slots_total,
            )
            return {
                "stage": stage,
                "success": False,
                "level": level,
                "sentence": sentence,
                "confidence": 0.0,
                "resonance": 0.0,
                "grammar_rules": 0,
                "error": "empty_composition",
            }

        # Log the composition
        tag = "CHALLENGE" if exercise.get("challenging") else stage.upper()
        logger.info(
            "[Composition] %s L%d: '%s' (conf=%.2f, %d/%d slots, intent=%s)",
            tag, level, sentence, confidence, slots_filled, slots_total, intent,
        )

        # Send composed sentence as experience-stimulus
        # Build a perturbation from the words used (lightweight — the sentence IS the stimulus)
        stim_result = await inject_word_stimulus(
            client, api,
            word=sentence,
            perturbation={},   # No perturbation — the sentence itself is the experience
            pass_type="produce",
            hormone_stimuli={},
        )
        stim_ok = not stim_result.get("error")

        # If conversational stage, also send the prompt as a chat message first
        if stage == "conversational" and exercise.get("prompt"):
            prompt = exercise["prompt"]
            logger.info("[Composition] Conversational prompt: '%s'", prompt)
            # The prompt is the teacher's question; the composed sentence is Titan's "answer"
            # We send the prompt+answer together so Titan processes the full exchange
            combined = f"{prompt} — {sentence}"
            await send_chat(
                client, api,
                message=combined,
                user_id="teacher_composition",
            )

        # Wait for epoch integration
        new_epoch = await wait_for_new_epoch(
            client, api, epoch_before, timeout_s=180)

        # Post-state
        post_felt = await _get_felt_state(client, api)

        # Score: state_resonance = cosine similarity between pre and post state
        state_resonance = _cosine_sim(pre_felt, post_felt)
        # Resonance is meaningful when state CHANGED (1.0 = no change, <1.0 = perturbation)
        # We want to measure how much the sentence moved the state, so invert:
        # High resonance = sentence aligned with state direction
        # We compute resonance as: 1 - abs(1 - cosine_sim) to handle both directions
        # But simpler: just use the raw cosine sim as the resonance score
        # Values near 1.0 mean the state was stable (sentence matched current state)
        # Values < 1.0 mean the state shifted (sentence caused change)
        resonance = round(state_resonance, 4)

        # G3: Staleness feedback (testsuite path)
        _g3_delta = 1.0 - resonance if resonance < 1.0 else 0.0
        if _g3_delta < 0.01:
            self._engine.selector.curiosity_weight = min(
                0.30, self._engine.selector.curiosity_weight + 0.01)
        elif _g3_delta > 0.05:
            self._engine.selector.curiosity_weight = max(
                0.05, self._engine.selector.curiosity_weight - 0.005)

        # Slot fill ratio
        fill_ratio = slots_filled / slots_total if slots_total > 0 else 0.0

        # LLM spot-check (Stage B: gated by confidence)
        grammar_rules = 0
        llm_check = None
        if stage == "free":
            grammar_rules, llm_check = await self._maybe_spotcheck(
                client, api, sentence, words_used, level, avg_confidence)

        result = {
            "stage": stage,
            "success": True,
            "level": level,
            "sentence": sentence,
            "words_used": words_used,
            "confidence": round(confidence, 3),
            "resonance": resonance,
            "fill_ratio": round(fill_ratio, 3),
            "slots_filled": slots_filled,
            "slots_total": slots_total,
            "intent": intent,
            "challenging": exercise.get("challenging", False),
            "stim_ok": stim_ok,
            "grammar_rules": grammar_rules,
        }
        if llm_check is not None:
            result["llm_check"] = llm_check

        # Persist to composition_history table
        try:
            conn = sqlite3.connect(DB_PATH, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                INSERT INTO composition_history
                    (timestamp, epoch_id, level, template, sentence, words_used,
                     confidence, slots_filled, slots_total, intent, stage,
                     state_resonance, pre_state, post_state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(), new_epoch, level, f"L{level}", sentence,
                json.dumps(words_used), confidence, slots_filled, slots_total,
                intent, stage, resonance,
                json.dumps(pre_felt[:10]) if pre_felt else None,
                json.dumps(post_felt[:10]) if post_felt else None,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("[Composition] DB store error: %s", e)

        return result

    # ── LLM Spot-Check (Stage B) ──────────────────────────────

    async def _maybe_spotcheck(
        self,
        client,
        api: str,
        sentence: str,
        words_used: list,
        level: int,
        avg_confidence: float,
    ) -> tuple[int, Optional[str]]:
        """Conditionally run LLM grammar spot-check.

        Gating frequency based on average confidence:
          <40%: check every sentence
          40-60%: check every 3rd
          60-80%: check every 5th
          >80%: no checks (Phase 3 complete)

        Returns:
            (grammar_rules_extracted: int, llm_response: Optional[str])
        """
        # Determine check frequency
        if avg_confidence > 0.80:
            return 0, None
        elif avg_confidence > 0.60:
            gate = 5
        elif avg_confidence > 0.40:
            gate = 3
        else:
            gate = 1

        # Gate by session-global composition count
        if self._engine._composition_count % gate != 0:
            return 0, None

        # Build spot-check prompt
        prompt = LLM_SPOTCHECK_PROMPT.format(
            sentence=sentence,
            words=", ".join(words_used),
            level=level,
        )

        response = await send_chat(client, api, prompt, user_id="teacher_grammar")
        if not response:
            return 0, None

        # Parse response for grammar rules
        rules_extracted = 0
        response_upper = response.upper().strip()

        if response_upper.startswith("CORRECT"):
            logger.info("[Composition] LLM spot-check: CORRECT")
        elif "RULE:" in response_upper:
            # Extract rule text
            rule_idx = response_upper.index("RULE:")
            rule_text = response[rule_idx + 5:].strip()
            if rule_text:
                self._grammar_rules.append(rule_text)
                rules_extracted = 1
                logger.info(
                    "[Composition] LLM spot-check rule: %s", rule_text[:100])
        else:
            # Unstructured response — log but don't extract
            logger.info(
                "[Composition] LLM spot-check (unstructured): %s",
                response[:100])

        return rules_extracted, response

    # ── Helpers ────────────────────────────────────────────────

    def _avg_vocab_confidence(self, vocabulary: list) -> float:
        """Compute average confidence across vocabulary."""
        if not vocabulary:
            return 0.0
        total = sum(
            v.get("confidence", 0.0) if isinstance(v, dict) else 0.3
            for v in vocabulary
        )
        return total / len(vocabulary)

    def _empty_result(self, duration: float, error: str) -> dict:
        """Return an empty/failed result dict."""
        return {
            "type": "composition",
            "success": False,
            "compositions": 0,
            "avg_confidence": 0.0,
            "avg_resonance": 0.0,
            "level_distribution": {},
            "stages": {"guided": 0, "free": 0, "conversational": 0},
            "grammar_rules_extracted": 0,
            "duration": round(duration, 1),
            "error": error,
        }
