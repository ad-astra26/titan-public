"""
titan_plugin/logic/language_pipeline.py — Pure Language Logic Functions.

Extracted from spirit_worker.py inline code. These functions have ZERO
dependencies on bus, queues, or process infrastructure. They operate on
data passed in and return results — no side effects beyond DB writes.

Used by:
  - spirit_worker.py (Phase 0 — calls these instead of inline code)
  - language_worker.py (Phase 1+ — owns these directly)

EXTRACTION SOURCE LINES refer to spirit_worker.py as of commit 22d57d0.
"""
import json
import logging
import math
import re
import sqlite3
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ── Word Type Classification ─────────────────────────────────────────
# Moved from spirit_worker.py lines 59-81. Checked BEFORE suffix heuristics.

KNOWN_ADJECTIVES = {
    "good", "bad", "big", "small", "happy", "sad", "warm", "cold", "fast",
    "slow", "strong", "weak", "bright", "dark", "soft", "hard", "new", "old",
    "wise", "clear", "dear", "full", "strange", "peaceful", "important",
    "joyful", "thankful", "beautiful", "alive", "safe", "sure", "free",
    "gentle", "quiet", "deep", "high", "low", "sharp", "sweet", "wild",
}
KNOWN_INTERJECTIONS = {
    "hello", "yes", "no", "okay", "oh", "wow", "hey", "please", "thanks",
    "sorry", "hmm", "ah", "well",
}
KNOWN_ADVERBS = {
    "here", "there", "now", "then", "very", "always", "never", "often",
    "still", "also", "maybe", "perhaps", "together", "inside", "outside",
    "forward", "gently", "slowly", "quickly", "deeply",
}
KNOWN_PRONOUNS = {
    "i", "you", "we", "they", "he", "she", "it", "me", "us", "them",
    "my", "your", "our", "their", "this", "that", "something", "nothing",
    "everything", "myself", "yourself",
}


def classify_word_type(word: str) -> str:
    """Classify a word into its type using known-word sets + suffix heuristics.

    Source: spirit_worker.py lines 6258-6279.
    """
    w_lower = word.lower().strip()
    if w_lower in KNOWN_ADJECTIVES:
        return "adjective"
    if w_lower in KNOWN_INTERJECTIONS:
        return "interjection"
    if w_lower in KNOWN_ADVERBS:
        return "adverb"
    if w_lower in KNOWN_PRONOUNS:
        return "pronoun"
    # Suffix heuristics (fallback)
    if w_lower.endswith(("ful", "ous", "ive", "ish", "ent", "ant", "tic", "al", "ic")):
        return "adjective"
    if w_lower.endswith(("ness", "ment", "tion", "sion", "ity", "ance", "ence")):
        return "noun"
    if w_lower.endswith("ly") and len(w_lower) > 3:
        return "adverb"
    return "verb"  # Default fallback


# ── Vocabulary Loading ───────────────────────────────────────────────

def load_vocabulary(
    db_path: str = "./data/inner_memory.db",
    top_k: int = 100,
    explore_k: int = 28,
) -> list[dict]:
    """Load vocabulary from DB: top_k by confidence + explore_k random exploration words.

    Returns:
        List of {word, word_type, confidence, felt_tensor, cross_modal_conf} dicts.
    """
    vocab = []
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")

        # Top words by confidence
        rows_top = conn.execute(
            "SELECT word, word_type, confidence, felt_tensor, "
            "COALESCE(cross_modal_conf, 0.0), "
            "COALESCE(meaning_contexts, '[]') FROM vocabulary "
            "WHERE confidence >= 0.0 ORDER BY confidence DESC LIMIT ?",
            (top_k,)
        ).fetchall()

        # Exploration pool — random low-confidence words
        rows_explore = conn.execute(
            "SELECT word, word_type, confidence, felt_tensor, "
            "COALESCE(cross_modal_conf, 0.0), "
            "COALESCE(meaning_contexts, '[]') FROM vocabulary "
            "WHERE confidence > 0.0 AND confidence < 0.2 "
            "ORDER BY RANDOM() LIMIT ?",
            (explore_k,)
        ).fetchall()

        conn.close()

        seen = set()
        for row in list(rows_top) + list(rows_explore):
            if row[0] in seen:
                continue
            seen.add(row[0])
            ft = None
            if row[3]:
                try:
                    ft = json.loads(row[3]) if isinstance(row[3], str) else row[3]
                except Exception:
                    pass
            mc = row[5] if len(row) > 5 else "[]"
            vocab.append({
                "word": row[0], "word_type": row[1],
                "confidence": row[2], "felt_tensor": ft,
                "cross_modal_conf": row[4],
                "meaning_contexts": mc,
            })
    except Exception as e:
        logger.debug("[LanguagePipeline] load_vocabulary error: %s", e)

    return vocab


# ── DA-Gated Exploration ─────────────────────────────────────────────

def apply_da_exploration(
    selector,
    da_level: float,
    da_setpoint: float,
) -> None:
    """Set min_confidence_override on WordSelector based on DA level.

    Source: spirit_worker.py lines 4090-4103.

    When DA > setpoint, lower min_confidence to let exploration words in.
    """
    if da_level > da_setpoint and da_setpoint > 0:
        explore_ratio = min(1.0, (da_level - da_setpoint) / da_setpoint)
        selector.min_confidence_override = 0.1 * (1.0 - explore_ratio * 0.9)
    else:
        selector.min_confidence_override = None


# ── Sentence Composition ─────────────────────────────────────────────

def compose_sentence(
    composition_engine,
    state_vector: list,
    vocabulary: list,
    da_level: float = 0.5,
    da_setpoint: float = 0.5,
    grammar_validator=None,
    experience_bias=None,
    visual_context: list | None = None,
    concept_confidences: dict | None = None,
    max_level: int = 8,
) -> dict:
    """Compose a sentence from felt-state with DA-gated exploration + grammar.

    Source: spirit_worker.py lines 4090-4122.

    Returns:
        Composition result dict {sentence, level, confidence, words_used, template, ...}
        or empty dict on failure.
    """
    # DA-gated exploration
    apply_da_exploration(composition_engine.selector, da_level, da_setpoint)

    try:
        result = composition_engine.compose(
            state_vector, vocabulary,
            intent=None, max_level=max_level,
            experience_bias=experience_bias,
            visual_context=visual_context,
            concept_confidences=concept_confidences,
        )
    except Exception as e:
        logger.warning("[LanguagePipeline] Composition error: %s", e)
        return {}

    # Grammar validation (Broca's area)
    if grammar_validator and result.get("sentence"):
        corrected = grammar_validator.validate(result["sentence"])
        if corrected != result["sentence"]:
            logger.info("[GRAMMAR] Corrected: '%s' -> '%s'",
                        result["sentence"], corrected)
            result["sentence"] = corrected

    return result


# ── Self-Hearing Perturbation ────────────────────────────────────────

# Characters to strip from words during self-hearing
_STRIP_CHARS = ".,!?\"'()[]{}:;—–-\u201c\u201d\u2018\u2019"


def compute_perturbation_deltas(
    narrator,
    sentence: str,
) -> list[dict]:
    """Compute bone-conduction perturbation deltas for each word in sentence.

    Source: spirit_worker.py lines 4272-4298.

    Does NOT apply them — caller applies to body_state/mind_state.

    Returns:
        List of {word, inner_body: [5D], inner_mind: [15D]} dicts.
    """
    deltas = []
    if not sentence or not narrator:
        return deltas

    for raw_word in sentence.split():
        word = raw_word.strip(_STRIP_CHARS).lower()
        if not word or len(word) < 3:
            continue
        perturb = narrator.get_word_perturbation(word)
        if perturb:
            deltas.append({
                "word": word,
                "inner_body": perturb.get("inner_body", []),
                "inner_mind": perturb.get("inner_mind", []),
            })

    return deltas


def apply_perturbation_deltas(
    deltas: list[dict],
    body_values: list,
    mind_values: list,
    strength: float = 0.5,
) -> int:
    """Apply pre-computed perturbation deltas to body/mind state.

    Source: spirit_worker.py lines 4284-4296.

    Args:
        deltas: From compute_perturbation_deltas()
        body_values: body_state["values"] (5D) — modified in place
        mind_values: mind_state["values_15d"] (15D) — modified in place
        strength: Perturbation strength (0.5 = bone conduction, 0.4 = teacher)

    Returns:
        Number of words that had perturbations applied.
    """
    reinforced = 0
    for delta in deltas:
        ib = delta.get("inner_body", [])
        im = delta.get("inner_mind", [])
        applied = False
        for i, v in enumerate(ib):
            if v != 0 and i < len(body_values):
                body_values[i] = max(0.0, min(1.0, body_values[i] + v * strength))
                applied = True
        for i, v in enumerate(im):
            if v != 0 and i < len(mind_values):
                mind_values[i] = max(0.0, min(1.0, mind_values[i] + v * strength))
                applied = True
        if applied:
            reinforced += 1
    return reinforced


# ── Vocabulary Update After SPEAK ────────────────────────────────────

def update_vocabulary_after_speak(
    db_path: str,
    narrator,
    sentence: str,
) -> tuple[int, list[str]]:
    """Update vocabulary for words Titan just spoke (advance to producible).

    Source: spirit_worker.py lines 4305-4347.

    Returns:
        (words_updated, list_of_words_reinforced)
    """
    updated = 0
    words_reinforced = []

    if not sentence:
        return updated, words_reinforced

    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")

        for raw_word in sentence.split():
            word = raw_word.strip(_STRIP_CHARS).lower()
            # Strip markdown artifacts
            word = word.strip("*_`")
            if not word or len(word) < 2:
                continue
            # Reject leaked slot tags (adj4, verb2, noun3 etc.)
            if re.match(r'^(adj|verb|noun|adv)\d*$', word, re.IGNORECASE):
                continue
            # Reject single-char words except "i" and "a"
            if len(word) < 2 and word not in ("i", "a"):
                continue

            # Check if word exists
            row = conn.execute(
                "SELECT word, confidence, learning_phase FROM vocabulary WHERE word=?",
                (word,)
            ).fetchone()

            if row:
                # Advance to producible + increment times_produced
                from titan_plugin.persistence import get_client
                get_client(caller_name="language_pipeline").write(
                    "UPDATE vocabulary SET learning_phase='producible', "
                    "times_produced = times_produced + 1, "
                    "confidence = MIN(1.0, confidence + 0.02) "
                    "WHERE word=?",
                    (word,),
                    table="vocabulary",
                )
                updated += 1
                words_reinforced.append(word)
            else:
                # Auto-create word not in vocabulary
                w_type = classify_word_type(word)
                conn.execute(
                    "INSERT OR IGNORE INTO vocabulary "
                    "(word, word_type, confidence, learning_phase, "
                    "times_encountered, times_produced, created_at) "
                    "VALUES (?, ?, 0.05, 'producible', 1, 1, ?)",
                    (word, w_type, time.time())
                )
                updated += 1
                words_reinforced.append(word)
                # Register dynamic recipe if narrator available
                if narrator and hasattr(narrator, "register_dynamic_recipe"):
                    narrator.register_dynamic_recipe(word, None, word_type=w_type)

        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug("[LanguagePipeline] update_vocabulary error: %s", e)

    return updated, words_reinforced


# ── Vocabulary DB Operations (extracted from CGN for consumer client migration) ──

def load_concept_from_db(db_path: str, word: str):
    """Load a word from vocabulary DB as a dict compatible with CGNConsumerClient.ground().

    Returns dict with ConceptFeatures-compatible fields, or None.
    Extracted from ConceptGroundingNetwork.load_concept() for use without
    a local CGN instance.
    """
    import numpy as np

    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        row = conn.execute(
            "SELECT word, felt_tensor, confidence, times_encountered, "
            "times_produced, learning_phase, created_at, "
            "COALESCE(sensory_context, '[]'), "
            "COALESCE(meaning_contexts, '[]'), "
            "COALESCE(cross_modal_conf, 0.0) "
            "FROM vocabulary WHERE word=?", (word,)).fetchone()
        conn.close()

        if not row:
            return None

        # Defensive parsing
        try:
            ft_raw = row[1]
            if isinstance(ft_raw, bytes):
                ft = [0.5] * 130
            elif ft_raw:
                ft = json.loads(ft_raw)
            else:
                ft = [0.5] * 130
        except (json.JSONDecodeError, TypeError):
            ft = [0.5] * 130

        try:
            contexts = json.loads(row[7]) if row[7] and isinstance(row[7], str) else []
        except (json.JSONDecodeError, TypeError):
            contexts = []

        try:
            meanings = json.loads(row[8]) if row[8] and isinstance(row[8], str) else []
        except (json.JSONDecodeError, TypeError):
            meanings = []

        associations = {}
        for m in meanings:
            for a in m.get("associations", []):
                if isinstance(a, (list, tuple)) and len(a) >= 2:
                    associations[a[0]] = associations.get(a[0], 0) + 0.1

        age = int((time.time() - (row[6] or time.time())) / 1.15)

        return {
            "concept_id": word,
            "embedding": np.array(ft, dtype=np.float32),
            "confidence": row[2],
            "encounter_count": row[3],
            "production_count": row[4],
            "context_history": [{"ctx": c} for c in contexts[-10:]],
            "associations": associations,
            "age_epochs": max(0, age),
            "cross_modal_conf": row[9],
            "meaning_contexts": meanings,
            "extra": {"learning_phase": row[5]},
        }
    except Exception as e:
        logger.debug("[VocabDB] load_concept('%s') failed: %s", word, e)
        return None


def apply_grounding_action_to_db(db_path: str, word: str, action,
                                  state_132d=None) -> bool:
    """Apply a grounding action to a word in the vocabulary DB.

    Args:
        db_path: path to inner_memory.db
        word: vocabulary word
        action: GroundingAction or LocalGroundingResult (same field names)
        state_132d: optional 132D state vector for tensor plasticity blending

    Extracted from ConceptGroundingNetwork.apply_grounding_action().
    """
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        row = conn.execute(
            "SELECT confidence, felt_tensor, "
            "COALESCE(cross_modal_conf, 0.0), "
            "COALESCE(sensory_context, '[]') "
            "FROM vocabulary WHERE word=?", (word,)).fetchone()

        if not row:
            conn.close()
            return False

        def _db_float(v, default=0.0):
            if isinstance(v, dict):
                return float(v.get("confidence", v.get("level", default)))
            if isinstance(v, bytes):
                import struct as _st
                try: return _st.unpack('<f', v)[0]
                except Exception: return default
            try: return float(v) if v is not None else default
            except (TypeError, ValueError): return default

        old_conf = _db_float(row[0])
        try:
            ft = json.loads(row[1]) if row[1] and isinstance(row[1], str) else [0.5] * 130
        except (json.JSONDecodeError, TypeError):
            ft = [0.5] * 130
        old_xm = _db_float(row[2])

        # Apply confidence delta
        new_conf = max(0.0, min(1.0, old_conf + float(action.confidence_delta)))

        # Apply tensor plasticity
        if action.tensor_plasticity > 0.01 and state_132d is not None:
            p = action.tensor_plasticity
            for i in range(min(len(ft), len(state_132d), 130)):
                ft[i] = ft[i] * (1 - p) + float(state_132d[i]) * p

        # Update cross_modal_conf
        xm_boost = 0.0
        action_name = getattr(action, "action_name", "reinforce")
        if action_name in ("reinforce", "deepen", "consolidate"):
            xm_boost = 0.01
        elif action_name == "differentiate":
            xm_boost = 0.02
        new_xm = min(1.0, old_xm + xm_boost)

        conn.close()
        from titan_plugin.persistence import get_client
        get_client(caller_name="language_pipeline").write(
            "UPDATE vocabulary SET confidence=?, felt_tensor=?, "
            "cross_modal_conf=? WHERE word=?",
            (new_conf, json.dumps(ft), new_xm, word),
            table="vocabulary",
        )
        return True

    except Exception as e:
        logger.debug("[VocabDB] apply_grounding_action('%s') failed: %s", word, e)
        return False


# ── Language Stats ───────────────────────────────────────────────────

def update_language_stats(
    db_path: str = "./data/inner_memory.db",
    cached_vocab: list | None = None,
) -> dict:
    """Query DB for language proficiency statistics.

    Source: spirit_worker.py lines 3165-3194.

    Returns:
        Dict with vocab_total, vocab_producible, avg_confidence, etc.
    """
    now = time.time()
    stats = {
        "vocab_total": 0,
        "vocab_producible": 0,
        "vocab_contextual": 0,
        "avg_confidence": 0.0,
        "max_confidence": 0.0,
        "recent_words": [],
        "teacher_sessions_last_hour": 0,
        "composition_level": "L1",
        "_last_update": now,
    }

    try:
        conn = sqlite3.connect(db_path, timeout=2.0)
        conn.execute("PRAGMA journal_mode=WAL")

        total = conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
        prod = conn.execute(
            "SELECT COUNT(*) FROM vocabulary WHERE learning_phase='producible'"
        ).fetchone()[0]
        avg = conn.execute(
            "SELECT AVG(confidence) FROM vocabulary WHERE confidence > 0"
        ).fetchone()[0] or 0
        max_conf = conn.execute(
            "SELECT MAX(confidence) FROM vocabulary"
        ).fetchone()[0] or 0
        recent = conn.execute(
            "SELECT word, confidence FROM vocabulary ORDER BY created_at DESC LIMIT 5"
        ).fetchall()

        # Teacher sessions in last hour
        sessions = 0
        try:
            sessions = conn.execute(
                "SELECT COUNT(*) FROM teacher_sessions WHERE created_at > ?",
                (now - 3600,)
            ).fetchone()[0]
        except Exception:
            pass  # Table may not exist yet

        # Composition level from actual recent compositions (before closing conn)
        _comp_max_level = 8
        try:
            _comp_row = conn.execute(
                "SELECT MAX(level) FROM (SELECT level FROM composition_history "
                "ORDER BY rowid DESC LIMIT 50)"
            ).fetchone()
            if _comp_row and _comp_row[0]:
                _comp_max_level = _comp_row[0]
        except Exception:
            pass

        conn.close()

        stats["vocab_total"] = total
        stats["vocab_producible"] = prod
        stats["vocab_contextual"] = total - prod
        stats["avg_confidence"] = round(avg, 3)
        stats["max_confidence"] = round(max_conf, 3)
        stats["recent_words"] = [
            {"word": r[0], "confidence": round(r[1], 3)} for r in recent
        ]
        stats["teacher_sessions_last_hour"] = sessions

        # Composition level from actual recent data
        vocab_size = len(cached_vocab) if cached_vocab else total
        if vocab_size >= 50:
            stats["composition_level"] = f"L{min(9, max(1, _comp_max_level))}"
        else:
            stats["composition_level"] = f"L{min(8, max(1, vocab_size // 5))}"

    except Exception as e:
        logger.debug("[LanguagePipeline] update_language_stats error: %s", e)

    return stats


# ── Bootstrap Logic ──────────────────────────────────────────────────

def should_bootstrap(
    vocab_size: int,
    bootstrap_speak_attempts: int,
    compositions_since_teach: int,
    teacher_queue_empty: bool,
    teacher_pending: bool,
    last_bootstrap_trigger: float,
    threshold: int = 50,
    cooldown_s: float = 60,
) -> bool:
    """Determine if bootstrap (first_words) teaching should trigger.

    Source: spirit_worker.py lines 3201-3213.

    Returns True when vocabulary is small and SPEAK keeps failing.
    """
    now = time.time()

    # Not enough time since last bootstrap
    if now - last_bootstrap_trigger < cooldown_s:
        return False

    # Don't bootstrap while teacher is pending or queue has items
    if teacher_pending or not teacher_queue_empty:
        return False

    # Vocabulary above threshold — no bootstrap needed
    if vocab_size >= threshold:
        return False

    # Bootstrap conditions:
    # 1. SPEAK attempted 3+ times with no vocab
    # 2. OR: 0 compositions in 5+ min and low vocab
    speak_stuck = bootstrap_speak_attempts >= 3
    idle_stuck = (
        compositions_since_teach == 0
        and now - last_bootstrap_trigger > 300
    )

    return speak_stuck or idle_stuck


# ── Teacher Request Building ─────────────────────────────────────────

def build_teacher_request(
    teacher,
    teacher_queue: list,
    cached_vocab: list,
    neuromod_state: dict,
    concept_confidences: dict | None = None,
    recent_questions: list | None = None,
    patterns_to_avoid: list | None = None,
) -> dict | None:
    """Build a teacher request payload (mode selection + prompt).

    Source: spirit_worker.py lines 3248-3305.

    Returns:
        Dict with {prompt, system, mode, max_tokens, original, sentences, neuromod_gate}
        or None if teacher shouldn't fire.
    """
    if not teacher_queue:
        return None

    # Select teaching mode
    mode = teacher.select_mode(teacher_queue, cached_vocab, neuromod_state)
    if not mode:
        return None

    # Build prompt with concept confidences
    prompt_data = teacher.build_prompt(
        mode, teacher_queue, cached_vocab,
        patterns_to_avoid=patterns_to_avoid or [],
        concept_confidences=concept_confidences or {},
        recent_questions=recent_questions or [],
    )

    if not prompt_data or not prompt_data.get("prompt"):
        return None

    # Build sentences list for response handler
    sentences = [q.get("sentence", "") for q in teacher_queue]

    return {
        "prompt": prompt_data["prompt"],
        "system": prompt_data.get("system", ""),
        "mode": mode,
        "max_tokens": prompt_data.get("max_tokens", 100),
        "temperature": prompt_data.get("temperature", 0.4),
        "original": sentences[0] if sentences else "",
        "sentences": sentences,
        "neuromod_gate": prompt_data.get("neuromod_gate", ""),
    }


# ── Composition History Persistence ──────────────────────────────────

def persist_composition(
    db_path: str,
    sentence: str,
    level: int,
    template: str,
    words_used: list,
    confidence: float,
    slots_filled: int,
    slots_total: int,
    epoch_id: int,
    state_resonance: float = 0.0,
) -> bool:
    """Save composition to composition_history table.

    Source: spirit_worker.py lines 4142-4169.
    """
    try:
        from titan_plugin.persistence import get_client
        res = get_client(caller_name="language_pipeline").write(
            "INSERT INTO composition_history "
            "(timestamp, epoch_id, level, template, sentence, words_used, "
            "confidence, slots_filled, slots_total, intent, stage, state_resonance) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), epoch_id, level, template, sentence,
             json.dumps(words_used), confidence, slots_filled, slots_total,
             "", "", state_resonance),
            table="composition_history",
        )
        return res.ok
    except Exception as e:
        logger.debug("[LanguagePipeline] persist_composition error: %s", e)
        return False


# ── Teacher Session Persistence ──────────────────────────────────────

def persist_teacher_session(
    db_path: str,
    mode: str,
    original: str,
    teacher_response: str,
    words_recognized: int,
    correction: str | None = None,
    pattern_hash: str | None = None,
    neuromod_gate: str = "",
    epoch_id: int = 0,
) -> bool:
    """Save teacher session record to teacher_sessions table.

    Source: spirit_worker.py lines 6368-6410.
    """
    try:
        # Ensure table exists (DDL via direct, one-time)
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS teacher_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL, mode TEXT, original_sentence TEXT,
                teacher_response TEXT, words_recognized INTEGER,
                correction TEXT, pattern_hash TEXT, neuromod_gate TEXT,
                epoch_id INTEGER
            )
        """)
        conn.commit()
        conn.close()
        from titan_plugin.persistence import get_client
        res = get_client(caller_name="language_pipeline").write(
            "INSERT INTO teacher_sessions "
            "(timestamp, mode, original_sentence, teacher_response, "
            "words_recognized, correction, pattern_hash, neuromod_gate, "
            "epoch_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), mode, original, teacher_response,
             words_recognized, correction, pattern_hash, neuromod_gate,
             epoch_id),
            table="teacher_sessions",
        )
        return res.ok
    except Exception as e:
        logger.debug("[LanguagePipeline] persist_teacher_session error: %s", e)
        return False
