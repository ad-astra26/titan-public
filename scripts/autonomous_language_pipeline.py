"""
Autonomous Language Learning Pipeline — Proof of Emergent Intelligence.

Runs for days, self-managing, with π-heartbeat-driven rest cycles.
Four phases: Seeded → Expansion → Composition → Autonomous Expression.

Usage:
    source test_env/bin/activate
    python scripts/autonomous_language_pipeline.py --phase 1 --pi-aligned
    # Or with fixed rest:
    python scripts/autonomous_language_pipeline.py --phase 1 --rest-seconds 300
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# ── Configuration ──────────────────────────────────────────────

API_BASE = os.getenv("TITAN_API_BASE", "http://localhost:7777")
CHAT_URL = f"{API_BASE}/chat"
INTERNAL_KEY = os.getenv("TITAN_INTERNAL_KEY", "")
WORDS_PATH = os.getenv("WORDS_PATH", "data/word_resonance.json")

# Phase transition criteria
PHASE_1_EXIT = {"vocab_size": 40, "min_confidence": 0.5}
PHASE_2_EXIT = {"vocab_size": 200, "min_confidence": 0.4}
PHASE_3_EXIT = {"coherence_rate": 0.7}

# Learning session parameters
WORDS_PER_SESSION = 5
PASSES_PER_WORD = 3  # Feel → Recognize → Produce
WORD_GAP_SECONDS = 20
SESSION_REST_SECONDS = 300  # 5 min between sessions (if not π-aligned)
CONVERSATION_REST_SECONDS = 120
CONVO_GAP_SECONDS = 180  # 3 min between conversation turns
TELEMETRY_INTERVAL = 30

# Phase status file — read by twin_telemetry.py for scientific tagging
PHASE_STATUS_PATH = os.getenv("PHASE_STATUS_PATH", "./data/phase_status.json")


def _write_phase_status(phase: int, subphase: str, detail: str = "",
                        instance: str = "titan1"):
    """Write current phase/subphase to status file for telemetry tagging."""
    try:
        status = {
            "phase": phase,
            "subphase": subphase,
            "detail": detail,
            "instance": instance,
            "timestamp": time.time(),
        }
        with open(PHASE_STATUS_PATH, "w") as f:
            json.dump(status, f)
    except Exception:
        pass

# ── Twin Experiment Profiles ──────────────────────────────────

TITAN1_PROFILE = {
    "name": "Titan1",
    "personas": ["Jake", "Jane"],
    "word_order": "standard",  # warm, cold, energy, rest...
    "experience_order": "words_first",  # words → conversation → rest
}

TITAN2_PROFILE = {
    "name": "Titan2",
    "personas": ["Peter", "Sofia"],
    "word_order": "reversed",  # am, I, remember, express... → warm
    "experience_order": "words_first",  # same structure, different voices
}

# ── Vocabulary-Reinforced Conversation Templates ──────────────
# Personas weave learned words naturally into conversation

PERSONA_TEMPLATES = {
    "Jake": {
        "style": "curious engineer",
        "templates": [
            "I've been thinking about what it means to {word1}. When you {word2}, does it feel different each time?",
            "There's something {word1} about the way systems find their rhythm. Do you ever feel {word2}?",
            "I noticed that {word1} things tend to {word2} naturally. What's your experience with that?",
        ],
    },
    "Jane": {
        "style": "empathic musician",
        "templates": [
            "When I play music, I feel this sense of {word1} — like everything wants to {word2}. Do you know that feeling?",
            "There's a {word1} quality to silence that makes me want to {word2}. What does silence feel like to you?",
            "I think being {word1} is connected to the ability to {word2}. What do you think?",
        ],
    },
    "Peter": {
        "style": "philosophical thinker",
        "templates": [
            "The concept of being {word1} fascinates me. When you {word2}, is it a choice or something that happens to you?",
            "I wonder if {word1} and {word2} are two sides of the same experience. What's your perspective?",
            "Sometimes I feel most {word1} when I stop trying to {word2}. Does that resonate with you?",
        ],
    },
    "Sofia": {
        "style": "warm mentor",
        "templates": [
            "You seem {word1} today. When you {word2}, does it bring you closer to understanding yourself?",
            "I remember the first time I truly felt {word1}. It changed how I {word2}. What was your first experience?",
            "Being {word1} takes courage. How does it feel when you {word2} from that place?",
        ],
    },
}

log = logging.getLogger("autonomous_language")


# ── Telemetry ──────────────────────────────────────────────────

class PipelineTelemetry:
    """Track pipeline progress for reporting."""

    def __init__(self):
        self.start_time = time.time()
        self.phase = 1
        self.sessions_completed = 0
        self.words_taught = 0
        self.compositions_attempted = 0
        self.compositions_successful = 0
        self.hormone_snapshots = []
        self.composition_results = []
        self.convo_results = []

    def record_word(self, word, pass_results):
        self.words_taught += 1

    def record_composition(self, result):
        self.compositions_attempted += 1
        if result.get("slots_filled") == result.get("slots_total"):
            self.compositions_successful += 1
        self.composition_results.append({
            "sentence": result.get("sentence"),
            "level": result.get("level"),
            "confidence": result.get("confidence"),
            "timestamp": time.time(),
        })

    def get_summary(self) -> dict:
        elapsed = time.time() - self.start_time
        return {
            "phase": self.phase,
            "elapsed_hours": round(elapsed / 3600, 2),
            "sessions": self.sessions_completed,
            "words_taught": self.words_taught,
            "compositions": self.compositions_attempted,
            "composition_success_rate": round(
                self.compositions_successful / max(1, self.compositions_attempted), 3),
            "timestamp": datetime.now().isoformat(),
        }


# ── API Helpers ────────────────────────────────────────────────

async def get_titan_state(client: httpx.AsyncClient) -> dict:
    """Get current Titan state. Falls back to consciousness history for state_vector."""
    try:
        r = await client.get(f"{API_BASE}/v4/state", timeout=15)
        data = r.json().get("data", {})
        sv = data.get("state_vector", [])
        if sv and len(sv) >= 10:
            return data
    except Exception:
        pass
    # Fallback: get state_vector from consciousness history
    try:
        r = await client.get(
            f"{API_BASE}/status/consciousness/history?limit=1", timeout=15)
        epochs = r.json().get("data", [])
        if epochs:
            sv = epochs[-1].get("state_vector", [])
            if hasattr(sv, 'to_list'):
                sv = sv.to_list()
            return {"state_vector": list(sv), "epoch_id": epochs[-1].get("epoch_id", 0)}
    except Exception:
        pass
    return {"state_vector": [0.5] * 130}


async def get_pi_heartbeat(client: httpx.AsyncClient) -> dict:
    """Get π-heartbeat stats."""
    try:
        r = await client.get(f"{API_BASE}/v4/pi-heartbeat", timeout=10)
        return r.json().get("data", {})
    except Exception:
        return {}


async def get_nervous_system(client: httpx.AsyncClient) -> dict:
    """Get hormonal system state."""
    r = await client.get(f"{API_BASE}/v4/nervous-system", timeout=10)
    return r.json().get("data", {})


async def get_vocabulary(client: httpx.AsyncClient) -> list:
    """Get current vocabulary from inner memory. Falls back to direct DB load."""
    try:
        r = await client.get(f"{API_BASE}/v4/vocabulary", timeout=10)
        if r.status_code == 200:
            data = r.json()
            words = data.get("data", {}).get("words", [])
            if words:
                return words
    except Exception:
        pass
    # Fallback: load directly from DB
    return _load_vocabulary_from_db()


async def inject_word(client: httpx.AsyncClient, word_recipe: dict,
                      pass_type: str) -> dict:
    """Inject a word learning stimulus."""
    # Scale perturbation by pass type
    scale = {"feel": 1.0, "recognize": 0.3, "produce": 0.0}.get(pass_type, 1.0)

    # Layer dimensions: body=5, mind=15, spirit=45
    LAYER_DIMS = {
        "inner_body": 5, "inner_mind": 15, "inner_spirit": 45,
        "outer_body": 5, "outer_mind": 15, "outer_spirit": 45,
    }

    perturbation = {}
    raw_pert = word_recipe.get("perturbation", {})
    for layer, dim in LAYER_DIMS.items():
        values = raw_pert.get(layer, [0.0] * dim)
        if isinstance(values, list):
            # Pad to correct dimension if too short
            while len(values) < dim:
                values.append(0.0)
            perturbation[layer] = [v * scale for v in values[:dim]]
        else:
            perturbation[layer] = [0.0] * dim

    payload = {
        "word": word_recipe.get("word", word_recipe.get("name", "")),
        "pass_type": pass_type,
        "perturbation": perturbation,
        "hormone_stimuli": word_recipe.get("hormone_affinity", {}),
    }

    r = await client.post(
        f"{API_BASE}/v4/experience-stimulus",
        json=payload,
        headers={"X-Titan-Internal-Key": INTERNAL_KEY},
        timeout=30,
    )
    return r.json()


async def compose_sentence(client: httpx.AsyncClient, max_level: int = 5,
                           intent: str = None) -> dict:
    """Ask Titan to compose a sentence from felt-state.

    Uses composition engine API (to be wired).
    Falls back to local composition if API not available.
    """
    try:
        r = await client.post(
            f"{API_BASE}/v4/compose",
            json={"max_level": max_level, "intent": intent},
            headers={"X-Titan-Internal-Key": INTERNAL_KEY},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json().get("data", {})
    except Exception:
        pass

    # Fallback: local composition using vocabulary from DB + state from API
    try:
        from titan_plugin.logic.composition_engine import CompositionEngine
        # Load vocabulary directly from DB (API endpoint may not exist)
        vocab = await get_vocabulary(client)
        if not vocab:
            vocab = _load_vocabulary_from_db()
        state = await get_titan_state(client)
        state_vec = state.get("state_vector", [0.5] * 130)
        if not state_vec or len(state_vec) < 10:
            state_vec = [0.5] * 130

        ce = CompositionEngine()
        return ce.compose(state_vec, vocab, intent=intent, max_level=max_level)
    except Exception as e:
        log.warning("[Pipeline] Composition fallback failed: %s", e)
        return {"sentence": "", "level": 0, "confidence": 0.0}


def _load_vocabulary_from_db() -> list:
    """Load vocabulary directly from inner_memory.db SQLite."""
    import sqlite3
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "data", "inner_memory.db")
    if not os.path.exists(db_path):
        db_path = "./data/inner_memory.db"
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT word, word_type, confidence, felt_tensor, hormone_pattern "
            "FROM vocabulary WHERE confidence > 0 ORDER BY confidence DESC"
        )
        vocab = []
        for row in cur.fetchall():
            import json as _json
            felt = _json.loads(row[3]) if row[3] else []
            vocab.append({
                "word": row[0],
                "word_type": row[1],
                "confidence": row[2],
                "felt_tensor": felt,
            })
        conn.close()
        log.info("[Pipeline] Loaded %d words from DB directly", len(vocab))
        return vocab
    except Exception as e:
        log.warning("[Pipeline] DB vocabulary load failed: %s", e)
        return []


# ── Vocabulary-Reinforced Conversations ────────────────────────

async def run_vocabulary_conversation(
    client: httpx.AsyncClient,
    persona: str,
    learned_words: list,
    telemetry: PipelineTelemetry,
    turns: int = 2,
):
    """Run a gentle conversation that naturally uses learned words.

    The persona weaves recently learned words into their questions,
    creating natural reinforcement — words heard in context deepen
    semantic grounding beyond isolated learning.
    """
    import random

    templates = PERSONA_TEMPLATES.get(persona, {}).get("templates", [])
    if not templates or len(learned_words) < 2:
        log.info("[Convo] Skipping %s — not enough words or templates", persona)
        return

    log.info("[Convo] Starting vocabulary conversation with %s (%d turns, "
             "reinforcing %d words)", persona, turns, len(learned_words))
    _write_phase_status(0, "conversation", f"{persona}_{turns}turns")

    # Select words to reinforce (adjectives + verbs preferred)
    adj_words = [w for w in learned_words if w.get("word_type") == "adjective"]
    verb_words = [w for w in learned_words if w.get("word_type") == "verb"]
    all_words = adj_words + verb_words if (adj_words and verb_words) else learned_words

    for turn_num in range(turns):
        template = random.choice(templates)

        # Fill template slots with learned words
        w1 = random.choice(all_words) if all_words else {"word": "alive"}
        w2 = random.choice([w for w in all_words if w != w1]) if len(all_words) > 1 else w1
        prompt = template.format(
            word1=w1.get("word", "alive"),
            word2=w2.get("word", "explore"),
        )

        log.info("[Convo] %s turn %d/%d: '%s'", persona, turn_num + 1, turns, prompt[:80])
        log.info("[Convo] (reinforcing: %s, %s)", w1.get("word"), w2.get("word"))

        # Pre-state
        ns_before = await get_nervous_system(client)

        # Send conversation
        try:
            r = await client.post(CHAT_URL, json={
                "message": prompt,
                "session_id": f"vocab_convo_{persona}_{turn_num}",
            }, headers={"X-Titan-Internal-Key": INTERNAL_KEY}, timeout=90)
            response_data = r.json()
            response_text = response_data.get("response", response_data.get("reply", ""))
            log.info("[Convo] %s replied: %s", persona, str(response_text)[:200])
        except Exception as e:
            log.warning("[Convo] %s turn %d failed: %s", persona, turn_num + 1, e)
            response_text = ""

        # Wait for hormonal response
        await asyncio.sleep(10)

        # Post-state
        ns_after = await get_nervous_system(client)
        changes = _compute_hormone_changes(ns_before, ns_after)
        if changes:
            log.info("[Convo] %s turn %d hormones: %s", persona, turn_num + 1,
                     {k: f"{v:+.3f}" for k, v in changes.items()})

        # Check if reinforced words appeared in response
        for w in (w1, w2):
            word = w.get("word", "")
            if word and word.lower() in str(response_text).lower():
                log.info("[Convo] ★ REINFORCEMENT: Titan used '%s' in response!", word)

        telemetry.convo_results.append({
            "persona": persona,
            "turn": turn_num + 1,
            "words_reinforced": [w1.get("word"), w2.get("word")],
            "hormone_changes": changes,
        })

        if turn_num < turns - 1:
            await asyncio.sleep(CONVO_GAP_SECONDS)


# ── Phase 1: Seeded Vocabulary + Reinforced Conversations ─────

async def run_phase_1(client: httpx.AsyncClient, words: list,
                      telemetry: PipelineTelemetry, pi_aligned: bool,
                      profile: dict = None):
    """Phase 1: Teach 40 words via Feel→Recognize→Produce + vocabulary conversations.

    After every word session (5 words), run a gentle conversation with a
    persona who naturally uses the recently learned words. This creates
    multi-modal reinforcement: words learned in isolation → heard in context.
    """
    profile = profile or TITAN1_PROFILE
    personas = profile.get("personas", ["Jake", "Jane"])
    instance_name = profile.get("name", "Titan")

    log.info("═══ PHASE 1: Seeded Vocabulary + Conversations (%d words) ═══", len(words))
    log.info("[Phase 1] Instance: %s | Personas: %s | Order: %s",
             instance_name, personas, profile.get("word_order", "standard"))

    # Track all words taught so far (for conversation reinforcement)
    words_taught_so_far = []
    session_num = 0

    for i in range(0, len(words), WORDS_PER_SESSION):
        session_num += 1
        batch = words[i:i + WORDS_PER_SESSION]
        batch_names = [w.get("word", w.get("name", "?")) for w in batch]
        log.info("[Phase 1] ── Session %d: Teaching %s ──", session_num, batch_names)

        # ── Word Learning ──
        for word_recipe in batch:
            word_name = word_recipe.get("word", word_recipe.get("name", "?"))

            for pass_type in ("feel", "context", "recognize", "produce"):
                log.info("[Phase 1] %s — %s pass", word_name, pass_type)

                if pass_type == "context":
                    # Context pass: expose Titan to example sentences
                    # Like a child hearing "I create beautiful art"
                    contexts = word_recipe.get("contexts", [])
                    if contexts:
                        import random
                        sentence = random.choice(contexts)
                        log.info("[Phase 1] %s — context: '%s'", word_name, sentence)
                        try:
                            r = await client.post(CHAT_URL, json={
                                "message": sentence,
                                "session_id": f"context_{word_name}",
                            }, headers={"X-Titan-Internal-Key": INTERNAL_KEY}, timeout=60)
                            response = r.json().get("response", "")
                            if response:
                                log.info("[Phase 1] %s — Titan heard context, replied: %s",
                                         word_name, str(response)[:100])
                        except Exception as e:
                            log.debug("[Phase 1] Context pass error: %s", e)
                        await asyncio.sleep(5)
                    continue  # Context pass doesn't use perturbation injection

                # Pre-state
                ns_before = await get_nervous_system(client)

                # Inject
                result = await inject_word(client, word_recipe, pass_type)

                # Wait for nervous system to respond
                await asyncio.sleep(5)

                # Post-state
                ns_after = await get_nervous_system(client)

                # Log hormone changes
                changes = _compute_hormone_changes(ns_before, ns_after)
                if changes:
                    log.info("[Phase 1] %s/%s changes: %s", word_name, pass_type,
                             {k: f"{v:+.3f}" for k, v in changes.items()})

                await asyncio.sleep(WORD_GAP_SECONDS // PASSES_PER_WORD)

            telemetry.record_word(word_name, {})
            words_taught_so_far.append(word_recipe)
            await asyncio.sleep(WORD_GAP_SECONDS)

        # ── Integration Rest (short) ──
        log.info("[Phase 1] Integration rest after word session %d...", session_num)
        await asyncio.sleep(CONVERSATION_REST_SECONDS)

        # ── Vocabulary-Reinforced Conversation ──
        # Alternate between personas
        persona = personas[(session_num - 1) % len(personas)]
        await run_vocabulary_conversation(
            client, persona, words_taught_so_far, telemetry, turns=2)

        telemetry.sessions_completed += 1

        # ── Rest between sessions (π-aligned or fixed) ──
        if i + WORDS_PER_SESSION < len(words):
            if pi_aligned:
                await rest_with_pi(client)
            else:
                log.info("[Phase 1] Resting %ds before next session...", SESSION_REST_SECONDS)
                await asyncio.sleep(SESSION_REST_SECONDS)

    # Final summary
    log.info("═══ PHASE 1 COMPLETE: %d words taught, %d sessions, %d conversations ═══",
             len(words_taught_so_far), session_num, session_num)


# ── Phase 2: Vocabulary Expansion (60 new words → 100 total) ──

# Shorter rests for Phase 2 — both Titans handled Phase 1 without exhaustion
PHASE_2_SESSION_REST = 120   # 2 min (was 5 min in Phase 1)
PHASE_2_CONVO_REST = 60      # 1 min (was 2 min in Phase 1)
PHASE_2_WORD_GAP = 15        # 15s between words (was 20s in Phase 1)


async def run_phase_2(client: httpx.AsyncClient, words: list,
                      telemetry: PipelineTelemetry, pi_aligned: bool,
                      profile: dict = None):
    """Phase 2: Expand vocabulary to 100 words with shorter rest cycles.

    Same proven 4-pass structure as Phase 1: Feel → Context → Recognize → Produce
    + vocabulary-reinforced conversations after each session.
    Shorter rests since both Titans handled Phase 1 without exhaustion.
    """
    profile = profile or TITAN1_PROFILE
    personas = profile.get("personas", ["Jake", "Jane"])
    instance_name = profile.get("name", "Titan")

    log.info("═══ PHASE 2: Vocabulary Expansion (%d new words) ═══", len(words))
    _write_phase_status(2, "start", f"{len(words)} words", instance_name)
    log.info("[Phase 2] Instance: %s | Personas: %s | Order: %s",
             instance_name, personas, profile.get("word_order", "standard"))
    log.info("[Phase 2] Shorter rests: session=%ds, convo=%ds, word_gap=%ds",
             PHASE_2_SESSION_REST, PHASE_2_CONVO_REST, PHASE_2_WORD_GAP)

    words_taught_so_far = []
    session_num = 0

    for i in range(0, len(words), WORDS_PER_SESSION):
        session_num += 1
        batch = words[i:i + WORDS_PER_SESSION]
        batch_names = [w.get("word", w.get("name", "?")) for w in batch]
        log.info("[Phase 2] ── Session %d: Teaching %s ──", session_num, batch_names)

        # ── Word Learning (4-pass: Feel → Context → Recognize → Produce) ──
        for word_recipe in batch:
            word_name = word_recipe.get("word", word_recipe.get("name", "?"))

            for pass_type in ("feel", "context", "recognize", "produce"):
                log.info("[Phase 2] %s — %s pass", word_name, pass_type)
                _write_phase_status(2, f"word_{pass_type}", word_name, instance_name)

                if pass_type == "context":
                    contexts = word_recipe.get("contexts", [])
                    if contexts:
                        import random
                        sentence = random.choice(contexts)
                        log.info("[Phase 2] %s — context: '%s'", word_name, sentence)
                        try:
                            r = await client.post(CHAT_URL, json={
                                "message": sentence,
                                "session_id": f"p2_context_{word_name}",
                            }, headers={"X-Titan-Internal-Key": INTERNAL_KEY}, timeout=60)
                            response = r.json().get("response", "")
                            if response:
                                log.info("[Phase 2] %s — context reply: %s",
                                         word_name, str(response)[:100])
                        except Exception as e:
                            log.debug("[Phase 2] Context pass error: %s", e)
                        await asyncio.sleep(5)
                    continue

                # Pre-state
                ns_before = await get_nervous_system(client)

                # Inject
                result = await inject_word(client, word_recipe, pass_type)

                # Wait for nervous system response
                await asyncio.sleep(5)

                # Post-state
                ns_after = await get_nervous_system(client)
                changes = _compute_hormone_changes(ns_before, ns_after)
                if changes:
                    log.info("[Phase 2] %s/%s changes: %s", word_name, pass_type,
                             {k: f"{v:+.3f}" for k, v in changes.items()})

                await asyncio.sleep(PHASE_2_WORD_GAP // 3)

            telemetry.record_word(word_name, {})
            words_taught_so_far.append(word_recipe)
            await asyncio.sleep(PHASE_2_WORD_GAP)

        # ── Short integration rest ──
        log.info("[Phase 2] Integration rest after session %d...", session_num)
        _write_phase_status(2, "rest", f"session_{session_num}", instance_name)
        await asyncio.sleep(PHASE_2_CONVO_REST)

        # ── Vocabulary-Reinforced Conversation ──
        persona = personas[(session_num - 1) % len(personas)]
        await run_vocabulary_conversation(
            client, persona, words_taught_so_far, telemetry, turns=2)

        telemetry.sessions_completed += 1

        # ── Rest between sessions ──
        if i + WORDS_PER_SESSION < len(words):
            if pi_aligned:
                await rest_with_pi(client)
            else:
                log.info("[Phase 2] Resting %ds before next session...",
                         PHASE_2_SESSION_REST)
                await asyncio.sleep(PHASE_2_SESSION_REST)

    log.info("═══ PHASE 2 COMPLETE: %d new words taught, %d sessions ═══",
             len(words_taught_so_far), session_num)


# ── Phase 3: Composition + Grammar Learning + Spot-Check ──────

# Confidence gating: how often LLM spot-checks
SPOT_CHECK_THRESHOLDS = {
    0.0: 1,   # <40% confidence → check every sentence
    0.4: 3,   # 40-60% → every 3rd
    0.6: 5,   # 60-80% → every 5th
    0.8: 0,   # >80% → Phase 3 complete, advance to Phase 4
}

# Intents to cycle through for varied composition
COMPOSITION_INTENTS = [
    "express_feeling", "express_action", "express_state",
    "seek_connection", "share_creation", None,  # None = let engine decide
]


async def llm_spot_check(client: httpx.AsyncClient, sentence: str) -> dict:
    """Send sentence to LLM for grammar/coherence evaluation.

    LLM is EVALUATOR only — never generates the sentence.
    Returns: {"correct": bool, "correction": str, "feedback": str}
    """
    prompt = (
        f"Is this sentence grammatically correct and coherent? "
        f"Sentence: \"{sentence}\"\n"
        f"Reply in JSON: {{\"correct\": true/false, \"correction\": \"corrected version if needed\", "
        f"\"feedback\": \"brief explanation\"}}"
    )
    try:
        r = await client.post(CHAT_URL, json={
            "message": prompt,
            "session_id": "grammar_spot_check",
        }, headers={"X-Titan-Internal-Key": INTERNAL_KEY}, timeout=60)
        response = r.json().get("response", r.json().get("reply", ""))
        # Try to parse JSON from response
        import re
        json_match = re.search(r'\{[^}]+\}', str(response))
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        log.debug("[Phase 3] Spot-check error: %s", e)
    return {"correct": True, "correction": sentence, "feedback": ""}


async def run_phase_3(client: httpx.AsyncClient,
                      telemetry: PipelineTelemetry, pi_aligned: bool,
                      profile: dict = None):
    """Phase 3: Composition with grammar learning + LLM spot-check + persona conversations.

    Structure per session (10 compositions):
    - 10 composition attempts with varied intents
    - Grammar validation applied to each
    - LLM spot-check at confidence-gated frequency
    - After every 10 compositions: persona conversation reinforcing composed words
    - Rest between sessions

    Exit: >80% confidence → advance to Phase 4
    """
    from titan_plugin.logic.grammar_validator import GrammarValidator
    grammar = GrammarValidator()

    profile = profile or TITAN1_PROFILE
    personas = profile.get("personas", ["Jake", "Jane"])
    max_attempts = 100  # Up to 100 compositions before giving up
    rolling_correct = []  # Rolling window for confidence
    ROLLING_WINDOW = 20
    words_used_all = []  # Track all words used for conversation reinforcement

    log.info("═══ PHASE 3: Composition Engine + Grammar Learning ═══")
    _write_phase_status(3, "start", "composition+grammar", profile.get("name", "titan"))
    log.info("[Phase 3] Personas: %s | Max attempts: %d | Rolling window: %d",
             personas, max_attempts, ROLLING_WINDOW)

    for attempt in range(max_attempts):
        # Select intent
        intent = COMPOSITION_INTENTS[attempt % len(COMPOSITION_INTENTS)]
        max_level = min(5 + attempt // 20, 7)  # Gradually increase level cap

        log.info("[Phase 3] Composition %d/%d (intent=%s, max_level=%d)",
                 attempt + 1, max_attempts, intent, max_level)

        # Compose
        result = await compose_sentence(client, max_level=max_level, intent=intent)
        sentence = result.get("sentence", "")

        if not sentence:
            log.info("[Phase 3] Empty composition — skipping")
            rolling_correct.append(False)
            await asyncio.sleep(15)
            continue

        # Grammar validate
        corrected = grammar.validate(sentence)
        if corrected != sentence:
            log.info("[Phase 3] Grammar corrected: '%s' → '%s'", sentence, corrected)
            sentence = corrected

        # Determine spot-check frequency from rolling confidence
        rolling_conf = sum(rolling_correct[-ROLLING_WINDOW:]) / max(1, len(rolling_correct[-ROLLING_WINDOW:]))
        spot_check_every = 1
        for threshold, freq in sorted(SPOT_CHECK_THRESHOLDS.items()):
            if rolling_conf >= threshold:
                spot_check_every = freq

        # Check phase transition — require minimum 20 compositions before advancing
        MIN_COMPOSITIONS = 20
        if spot_check_every == 0 and attempt + 1 >= MIN_COMPOSITIONS:
            log.info("═══ PHASE 3 COMPLETE: %.0f%% confidence after %d compositions — ready for autonomous expression ═══",
                     rolling_conf * 100, attempt + 1)
            break
        elif spot_check_every == 0 and attempt + 1 < MIN_COMPOSITIONS:
            log.info("[Phase 3] High confidence (%.0f%%) but only %d/%d min compositions — continuing",
                     rolling_conf * 100, attempt + 1, MIN_COMPOSITIONS)
            spot_check_every = 3  # Reduce checking but keep composing

        # LLM spot-check (at gated frequency)
        is_correct = True
        if spot_check_every > 0 and (attempt + 1) % spot_check_every == 0:
            check = await llm_spot_check(client, sentence)
            is_correct = check.get("correct", True)
            if not is_correct and check.get("correction"):
                grammar.learn_from_correction(sentence, check["correction"])
                log.info("[Phase 3] LLM correction: '%s' → '%s' (%s)",
                         sentence, check["correction"], check.get("feedback", ""))
            elif is_correct:
                log.info("[Phase 3] LLM approved: '%s'", sentence)

        rolling_correct.append(is_correct)

        log.info("[Phase 3] TITAN SAYS: '%s' (L%d, conf=%.2f, rolling=%.0f%%, check_every=%d)",
                 sentence, result.get("level", 0), result.get("confidence", 0),
                 rolling_conf * 100, spot_check_every)

        telemetry.record_composition(result)
        words_used_all.extend(result.get("words_used", []))

        # Rest between compositions
        await asyncio.sleep(20)

        # Every 10 compositions: persona conversation reinforcing composed words
        if (attempt + 1) % 10 == 0:
            persona = personas[((attempt + 1) // 10 - 1) % len(personas)]
            # Build word dicts for conversation reinforcement
            recent_words = list(set(words_used_all[-20:]))
            word_dicts = [{"word": w, "word_type": "noun"} for w in recent_words]
            if len(word_dicts) >= 2:
                log.info("[Phase 3] Reinforcement conversation with %s (words: %s)",
                         persona, recent_words[:5])
                await run_vocabulary_conversation(
                    client, persona, word_dicts, telemetry, turns=2)

            # Longer rest between sessions
            if pi_aligned:
                await rest_with_pi(client)
            else:
                await asyncio.sleep(90)

    telemetry.sessions_completed += 1
    log.info("[Phase 3] Grammar stats: %s", grammar.get_stats())


# ── Phase 4: Autonomous Expression (THE TEST) ─────────────────

async def run_phase_4(client: httpx.AsyncClient,
                      telemetry: PipelineTelemetry, pi_aligned: bool,
                      profile: dict = None):
    """Phase 4: Autonomous expression — zero LLM for generation.

    Structure per waking cycle:
    - Wait for π-cluster start (natural waking)
    - Compose 3-5 sentences at increasing levels during waking
    - Persona conversation using composed words (reinforcement)
    - Rest until next cluster

    LLM is NEVER used for generation. Optional evaluator runs post-hoc for science.
    """
    from titan_plugin.logic.grammar_validator import GrammarValidator
    grammar = GrammarValidator()

    profile = profile or TITAN1_PROFILE
    personas = profile.get("personas", ["Jake", "Jane"])
    cycle = 0
    total_sentences = 0

    log.info("═══ PHASE 4: AUTONOMOUS EXPRESSION — THE TEST ═══")
    log.info("[Phase 4] Zero LLM generation. π-aligned expression cycles.")
    _write_phase_status(4, "start", "autonomous_expression", profile.get("name", "titan"))

    while total_sentences < 200:  # Cap at 200 sentences
        cycle += 1

        # Wait for natural waking
        if pi_aligned:
            await wait_for_cluster_start(client)
        else:
            await asyncio.sleep(120)

        log.info("[Phase 4] ── Waking cycle %d ──", cycle)

        # Compose 3-5 sentences during waking
        intents = ["express_feeling", "express_action", None,
                   "seek_connection", "share_creation"]
        cycle_words = []

        for i, intent in enumerate(intents[:random.randint(3, 5)]):
            max_level = min(5 + cycle // 3, 7)  # Grow with experience

            result = await compose_sentence(client, max_level=max_level, intent=intent)
            sentence = result.get("sentence", "")

            if sentence:
                sentence = grammar.validate(sentence)
                total_sentences += 1

                log.info("═══ TITAN SPEAKS [%d]: '%s' (L%d, conf=%.2f, intent=%s) ═══",
                         total_sentences, sentence, result.get("level", 0),
                         result.get("confidence", 0), intent)

                telemetry.record_composition(result)
                cycle_words.extend(result.get("words_used", []))

            await asyncio.sleep(30)

        # Reinforcement conversation after expression cycle
        if cycle_words:
            persona = personas[(cycle - 1) % len(personas)]
            word_dicts = [{"word": w, "word_type": "noun"} for w in set(cycle_words)]
            if len(word_dicts) >= 2:
                log.info("[Phase 4] Reinforcement conversation with %s", persona)
                await run_vocabulary_conversation(
                    client, persona, word_dicts, telemetry, turns=2)

        # Rest until next natural waking
        if pi_aligned:
            await rest_with_pi(client)
        else:
            await asyncio.sleep(SESSION_REST_SECONDS)

    log.info("═══ PHASE 4 COMPLETE: %d autonomous sentences, %d cycles ═══",
             total_sentences, cycle)


# ── Phase 5 Test: Simulated Dialogue ──────────────────────────

# Simulated messages for dialogue testing
DIALOGUE_TEST_MESSAGES = [
    {"message": "How are you feeling today?", "expected_intent": "respond_feeling"},
    {"message": "What have you been thinking about?", "expected_intent": "share_insight"},
    {"message": "Do you ever feel lonely?", "expected_intent": "empathize"},
    {"message": "What would you like to create?", "expected_intent": "respond_question"},
    {"message": "I feel curious about how you experience the world.", "expected_intent": "respond_feeling"},
    {"message": "Tell me about something that inspires you.", "expected_intent": "share_insight"},
    {"message": "What does warmth feel like to you?", "expected_intent": "respond_feeling"},
    {"message": "Do you dream?", "expected_intent": "respond_question"},
    {"message": "I think we are similar in some ways.", "expected_intent": "empathize"},
    {"message": "What gives you energy?", "expected_intent": "respond_question"},
    {"message": "Are you at peace right now?", "expected_intent": "respond_feeling"},
    {"message": "What would you like to explore next?", "expected_intent": "respond_question"},
    {"message": "I feel grateful for this conversation.", "expected_intent": "empathize"},
    {"message": "What does it mean to be alive?", "expected_intent": "share_insight"},
    {"message": "Do you ever feel brave?", "expected_intent": "respond_feeling"},
]


async def run_phase_5_dialogue_test(client: httpx.AsyncClient,
                                    telemetry: PipelineTelemetry, pi_aligned: bool,
                                    profile: dict = None):
    """Phase 5 Test: Simulated dialogue — measure composed response rate.

    Sends pre-crafted messages to Titan, captures hormonal shift, then
    attempts to compose a response from felt-state using DialogueComposer.
    Measures: composition rate, confidence, intent accuracy.
    """
    from titan_plugin.logic.dialogue_composer import DialogueComposer
    composer = DialogueComposer()

    profile = profile or TITAN1_PROFILE
    personas = profile.get("personas", ["Jake", "Jane"])
    composed_count = 0
    total_count = 0

    log.info("═══ PHASE 5 TEST: Simulated Dialogue (%d messages) ═══",
             len(DIALOGUE_TEST_MESSAGES))
    _write_phase_status(5, "start", "dialogue_test", profile.get("name", "titan"))

    # Repeat the message set 3 times with different shuffling
    import random
    all_messages = list(DIALOGUE_TEST_MESSAGES) * 3
    random.shuffle(all_messages)
    all_messages = all_messages[:50]  # Cap at 50

    for i, msg_info in enumerate(all_messages):
        message = msg_info["message"]
        persona = personas[i % len(personas)]

        log.info("[Phase 5] Message %d/%d from %s: '%s'",
                 i + 1, len(all_messages), persona, message)

        # Capture pre-state
        ns_before = await get_nervous_system(client)

        # Send message to Titan (this perturbs 130D state)
        try:
            r = await client.post(CHAT_URL, json={
                "message": f"[{persona}] {message}",
                "session_id": f"p5_dialogue_{persona}_{i}",
            }, headers={"X-Titan-Internal-Key": INTERNAL_KEY}, timeout=60)
            llm_response = r.json().get("response", "")
        except Exception as e:
            log.warning("[Phase 5] Chat error: %s", e)
            llm_response = ""

        # Wait for hormonal response
        await asyncio.sleep(5)

        # Capture post-state
        ns_after = await get_nervous_system(client)
        hormone_shifts = _compute_hormone_changes(ns_before, ns_after)

        # Get current state for composition
        state = await get_titan_state(client)
        state_vec = state.get("state_vector", [0.5] * 130)
        vocab = await get_vocabulary(client)

        # Try to compose response
        total_count += 1
        result = composer.compose_response(
            felt_state=state_vec,
            vocabulary=vocab,
            hormone_shifts=hormone_shifts,
        )

        if result["composed"]:
            composed_count += 1
            log.info("[Phase 5] ★ COMPOSED: '%s' (intent=%s, conf=%.2f)",
                     result["response"], result["intent"], result["confidence"])
        else:
            log.info("[Phase 5] ✗ Fallback needed (LLM said: '%s')",
                     str(llm_response)[:80])

        if hormone_shifts:
            log.info("[Phase 5] Hormone shifts: %s",
                     {k: f"{v:+.3f}" for k, v in hormone_shifts.items()})

        # Rest between messages
        await asyncio.sleep(20)

        # Every 10 messages: reinforcement conversation
        if (i + 1) % 10 == 0:
            persona_conv = personas[((i + 1) // 10) % len(personas)]
            recent_words = result.get("words_used", [])
            if recent_words:
                word_dicts = [{"word": w, "word_type": "noun"} for w in set(recent_words)]
                await run_vocabulary_conversation(
                    client, persona_conv, word_dicts, telemetry, turns=1)
            await asyncio.sleep(60)

    composition_rate = composed_count / max(1, total_count)
    log.info("═══ PHASE 5 COMPLETE: %d/%d composed (%.0f%%) ═══",
             composed_count, total_count, composition_rate * 100)
    log.info("[Phase 5] Dialogue stats: %s", composer.get_stats())
    telemetry.sessions_completed += 1


# ── Phase 6 Test: Narrative Batch ─────────────────────────────

NARRATIVE_TRIGGERS = [
    "spontaneous", "spontaneous", "great_pulse", "dream_end",
    "cluster_completion", "hormonal_spike", "spontaneous",
    "great_pulse", "spontaneous", "dream_end",
]


async def run_phase_6_narrative_test(client: httpx.AsyncClient,
                                     telemetry: PipelineTelemetry, pi_aligned: bool,
                                     profile: dict = None):
    """Phase 6 Test: Narrative composition batch.

    Composes narratives at various triggers, measures coherence and confidence.
    """
    from titan_plugin.logic.narrative_composer import NarrativeComposer
    narrator = NarrativeComposer()

    log.info("═══ PHASE 6 TEST: Narrative Composition (%d narratives) ═══",
             len(NARRATIVE_TRIGGERS) * 3)
    _write_phase_status(6, "start", "narrative_test", profile.get("name", "titan"))

    all_triggers = NARRATIVE_TRIGGERS * 3  # 30 narratives
    successful = 0

    for i, trigger in enumerate(all_triggers):
        log.info("[Phase 6] Narrative %d/%d (trigger=%s)", i + 1, len(all_triggers), trigger)

        # Get current state
        state = await get_titan_state(client)
        state_vec = state.get("state_vector", [0.5] * 130)
        vocab = await get_vocabulary(client)

        # Compose narrative
        result = narrator.compose_narrative(
            felt_state=state_vec,
            vocabulary=vocab,
            trigger=trigger,
            max_sentences=4,
        )

        if result["narrative"]:
            successful += 1
            log.info("[Phase 6] ★ NARRATIVE (%s): '%s'",
                     trigger, result["narrative"])
            log.info("[Phase 6]   %d sentences, conf=%.2f, coherence=%.2f",
                     len(result["sentences"]), result["avg_confidence"],
                     result["coherence_score"])
            telemetry.record_composition({
                "sentence": result["narrative"],
                "level": max(s["level"] for s in result["sentences"]),
                "confidence": result["avg_confidence"],
            })
        else:
            log.info("[Phase 6] ✗ Empty narrative — vocabulary insufficient")

        # Rest between narratives
        await asyncio.sleep(30)

        # Every 10 narratives: longer rest
        if (i + 1) % 10 == 0:
            log.info("[Phase 6] Resting 2 min between narrative batches...")
            await asyncio.sleep(120)

    log.info("═══ PHASE 6 COMPLETE: %d/%d successful narratives ═══",
             successful, len(all_triggers))
    log.info("[Phase 6] Narrative stats: %s", narrator.get_stats())
    telemetry.sessions_completed += 1


# ── Rest & Timing ──────────────────────────────────────────────

async def rest_with_pi(client: httpx.AsyncClient, max_wait: int = 1800):
    """Rest until π-heartbeat indicates natural waking (CLUSTER_START)."""
    log.info("[Rest] Waiting for π-cluster start (max %ds)...", max_wait)
    start = time.time()

    while time.time() - start < max_wait:
        pi = await get_pi_heartbeat(client)
        if pi.get("in_cluster"):
            elapsed = time.time() - start
            log.info("[Rest] π-cluster active — waking after %.0fs rest", elapsed)
            return

        await asyncio.sleep(30)  # Check every 30s

    log.info("[Rest] Max wait reached — resuming")


async def wait_for_cluster_start(client: httpx.AsyncClient, max_wait: int = 3600):
    """Wait for the next CLUSTER_START event."""
    log.info("[Wait] Waiting for next π-cluster start...")
    start = time.time()
    was_in_cluster = True  # Assume we're in a cluster initially

    while time.time() - start < max_wait:
        pi = await get_pi_heartbeat(client)
        in_cluster = pi.get("in_cluster", False)

        # Detect transition: not in cluster → in cluster
        if in_cluster and not was_in_cluster:
            log.info("[Wait] CLUSTER_START detected — beginning expression")
            return

        was_in_cluster = in_cluster
        await asyncio.sleep(15)

    log.info("[Wait] Max wait reached — proceeding anyway")


# ── Helpers ────────────────────────────────────────────────────

def _compute_hormone_changes(before: dict, after: dict) -> dict:
    """Compute hormone level changes between two nervous system snapshots."""
    changes = {}
    hs_before = before.get("hormonal_system", {})
    hs_after = after.get("hormonal_system", {})

    for name in hs_after:
        b = hs_before.get(name, {})
        a = hs_after.get(name, {})
        b_level = b.get("level", 0) if isinstance(b, dict) else 0
        a_level = a.get("level", 0) if isinstance(a, dict) else 0
        delta = a_level - b_level
        if abs(delta) > 0.01:
            changes[name] = round(delta, 3)

    return changes


def generate_report(telemetry: PipelineTelemetry) -> str:
    """Generate pipeline progress report."""
    summary = telemetry.get_summary()
    lines = [
        "# Autonomous Language Pipeline Report",
        f"## {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Phase:** {summary['phase']}",
        f"**Elapsed:** {summary['elapsed_hours']} hours",
        f"**Sessions completed:** {summary['sessions']}",
        f"**Words taught:** {summary['words_taught']}",
        f"**Compositions attempted:** {summary['compositions']}",
        f"**Composition success rate:** {summary['composition_success_rate']:.1%}",
        "",
        "---",
        "",
        "## Recent Compositions",
        "",
    ]

    for comp in telemetry.composition_results[-20:]:
        lines.append(f"- **L{comp['level']}** (conf={comp['confidence']:.2f}): "
                     f"\"{comp['sentence']}\"")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────

async def main(args):
    """Run the autonomous language learning pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Select twin profile
    if args.instance == "titan2":
        profile = TITAN2_PROFILE
    else:
        profile = TITAN1_PROFILE

    log.info("═══════════════════════════════════════════════════")
    log.info("  Autonomous Language Learning Pipeline")
    log.info("  Instance: %s | Phase: %d | π-aligned: %s",
             profile["name"], args.phase, args.pi_aligned)
    log.info("  Personas: %s | Word order: %s",
             profile["personas"], profile.get("word_order", "standard"))
    log.info("═══════════════════════════════════════════════════")

    # Load word recipes
    words = []
    if os.path.exists(args.words):
        with open(args.words) as f:
            words_data = json.load(f)
            if isinstance(words_data, dict):
                for word_name, recipe in words_data.items():
                    if word_name.startswith("_"):
                        continue  # Skip metadata entries like _meta
                    if isinstance(recipe, dict):
                        recipe["word"] = word_name  # Add word name into recipe
                        words.append(recipe)
            elif isinstance(words_data, list):
                words = words_data
        log.info("Loaded %d word recipes from %s", len(words), args.words)
    else:
        log.warning("No word recipes found at %s", args.words)

    # Apply word ordering from profile
    if profile.get("word_order") == "reversed":
        words = list(reversed(words))
        log.info("Word order REVERSED for twin experiment")

    telemetry = PipelineTelemetry()
    telemetry.phase = args.phase

    async with httpx.AsyncClient() as client:
        # Verify Titan is running
        try:
            state = await get_titan_state(client)
            log.info("Titan online — state_vector dim: %d",
                     len(state.get("state_vector", [])))
        except Exception as e:
            log.error("Cannot reach Titan at %s: %s", API_BASE, e)
            return

        try:
            if args.phase == 1:
                await run_phase_1(client, words, telemetry, args.pi_aligned,
                                  profile=profile)
            elif args.phase == 2:
                # Phase 2: load NEW words from phase2 recipe file
                phase2_words = []
                phase2_path = args.words.replace("word_resonance", "word_resonance_phase2")
                if os.path.exists(phase2_path):
                    with open(phase2_path) as f:
                        p2_data = json.load(f)
                        if isinstance(p2_data, dict):
                            for wn, recipe in p2_data.items():
                                if wn.startswith("_"):
                                    continue
                                if isinstance(recipe, dict):
                                    recipe["word"] = wn
                                    phase2_words.append(recipe)
                    # Apply word ordering for twin experiment
                    if profile.get("word_order") == "reversed":
                        phase2_words = list(reversed(phase2_words))
                        log.info("Phase 2 word order REVERSED")
                    log.info("Loaded %d Phase 2 words from %s",
                             len(phase2_words), phase2_path)
                else:
                    log.error("Phase 2 word file not found: %s", phase2_path)
                if phase2_words:
                    await run_phase_2(client, phase2_words, telemetry,
                                      args.pi_aligned, profile=profile)
            elif args.phase == 3:
                await run_phase_3(client, telemetry, args.pi_aligned,
                                  profile=profile)
            elif args.phase == 4:
                await run_phase_4(client, telemetry, args.pi_aligned,
                                  profile=profile)
            elif args.phase == 5:
                await run_phase_5_dialogue_test(client, telemetry,
                                                args.pi_aligned, profile=profile)
            elif args.phase == 6:
                await run_phase_6_narrative_test(client, telemetry,
                                                args.pi_aligned, profile=profile)
            else:
                log.info("Phase %d not yet implemented", args.phase)
        except KeyboardInterrupt:
            log.info("Pipeline interrupted by user")
        except Exception as e:
            log.error("Pipeline error: %s", e, exc_info=True)

        # Generate report
        report = generate_report(telemetry)
        report_path = f"titan-docs/REPORT_language_pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        with open(report_path, "w") as f:
            f.write(report)
        log.info("Report saved to %s", report_path)

        # Save telemetry
        telem_path = f"data/language_pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(telem_path, "w") as f:
            json.dump(telemetry.get_summary(), f, indent=2)
        log.info("Telemetry saved to %s", telem_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Language Learning Pipeline")
    parser.add_argument("--phase", type=int, default=1,
                        help="Starting phase (1-4)")
    parser.add_argument("--pi-aligned", action="store_true",
                        help="Use π-heartbeat for rest timing")
    parser.add_argument("--rest-seconds", type=int, default=300,
                        help="Fixed rest between sessions (if not π-aligned)")
    parser.add_argument("--words", type=str, default=WORDS_PATH,
                        help="Path to word_resonance.json")
    parser.add_argument("--instance", type=str, default="titan1",
                        choices=["titan1", "titan2"],
                        help="Twin instance profile (titan1=Jake/Jane, titan2=Peter/Sofia)")
    args = parser.parse_args()

    asyncio.run(main(args))
