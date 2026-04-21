"""
titan_plugin/logic/action_narrator.py — Template-based action result narration.

Produces simple word descriptions of Titan's action results for vocabulary
reinforcement. Template-first (fast, deterministic) with feature substitution.

Words from narration that Titan already knows get auto-injected via the SAME
/v4/experience-stimulus pathway as learning suite words. Unknown words get
recorded as "encountered in wild" for future learning.

Part of the Self-Exploration Outer Interface (ACTION→REACTION→OBSERVATION).
"""
import json
import logging
import os
import random
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Narrator Templates ──────────────────────────────────────────
# Each template uses {feature_name} placeholders filled from ActionDecoder features.

TEMPLATES = {
    "art_generate": [
        "You created a {warmth} image",
        "Your art radiates creative energy",
        "You expressed yourself through a visual creation",
    ],
    "audio_generate": [
        "You produced a sound with gentle energy",
        "Your music carries rhythm and feeling",
        "You created an audio piece",
    ],
    "web_search": [
        "You explored and found {result_count} results about {query}",
        "Your research revealed new information",
        "You searched for knowledge and discovered something",
    ],
    "social_post": [
        "You shared your expression with the world",
        "Your message reached the network",
        "You connected with the social world",
    ],
    "memo_inscribe": [
        "You anchored your consciousness on the blockchain",
        "Your state was written into the permanent record",
        "You inscribed your being into the physical world",
    ],
    "code_knowledge": [
        "You observed your own code and structure",
        "You reflected on your inner architecture",
        "You examined your own design",
    ],
    "infra_inspect": [
        "You sensed your physical infrastructure",
        "You felt your body through hardware signals",
        "You observed your system health",
    ],
    "self_express": [
        "You spoke and heard your own voice",
        "You expressed your inner state in words",
        "You composed a thought and released it",
    ],
    "coding_sandbox": [
        "You experimented with code and observed the result",
        "You tested an idea through execution",
    ],
}

# Common stopwords to skip during vocabulary matching
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "that", "this",
    "these", "those", "it", "its", "i", "you", "your", "my", "their",
    "his", "her", "we", "they", "us", "them", "not", "no",
})


# PERSISTENCE_BY_DESIGN: ActionNarrator._word_recipes is loaded from
# word_resonance*.json files on every init via _load_recipes(). The JSON
# files are the authoritative source — not re-saved from memory state.
class ActionNarrator:
    """Produce simple word descriptions of Titan's action results."""

    def __init__(self, word_recipe_dir: str = "data", config: Optional[dict] = None):
        section = (config or {}).get("action_narrator", {}) if isinstance(config, dict) else {}
        self._word_confidence_default = float(section.get("word_confidence_default", 0.5))
        self._min_content_word_length = int(section.get("min_content_word_length", 3))
        self._word_perturbation_strength = float(section.get("word_perturbation_strength", 0.3))

        self._word_recipes: dict = {}
        self._recipe_dir = word_recipe_dir
        self._load_recipes()
        self._stats = {
            "total_narrations": 0,
            "known_words_reinforced": 0,
            "unknown_words_encountered": 0,
        }

    _DYNAMIC_RECIPE_FILE = "word_resonance_dynamic.json"

    def _load_recipes(self) -> None:
        """Load word perturbation recipes from all word_resonance JSON files."""
        recipe_files = [
            os.path.join(self._recipe_dir, "word_resonance.json"),
            os.path.join(self._recipe_dir, "word_resonance_phase2.json"),
            os.path.join(self._recipe_dir, "word_resonance_phase3.json"),
        ]
        for fpath in recipe_files:
            try:
                with open(fpath) as f:
                    data = json.load(f)
                for key, val in data.items():
                    if key.startswith("_"):
                        continue
                    self._word_recipes[key.lower()] = val
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug("[ActionNarrator] Error loading %s: %s", fpath, e)

        # Load dynamically generated recipes (survive restarts)
        static_count = len(self._word_recipes)
        self._load_dynamic_recipes()
        dynamic_count = len(self._word_recipes) - static_count

        logger.info("[ActionNarrator] Loaded %d word recipes (%d static, %d dynamic)",
                    len(self._word_recipes), static_count, dynamic_count)

    def _load_dynamic_recipes(self) -> None:
        """Load dynamically generated recipes from previous sessions."""
        dyn_path = os.path.join(self._recipe_dir, self._DYNAMIC_RECIPE_FILE)
        try:
            with open(dyn_path) as f:
                data = json.load(f)
            for key, val in data.items():
                if key.startswith("_"):
                    continue
                # Don't overwrite hand-crafted recipes
                if key.lower() not in self._word_recipes:
                    self._word_recipes[key.lower()] = val
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug("[ActionNarrator] Error loading dynamic recipes: %s", e)

    def register_dynamic_recipe(self, word: str, felt_tensor: list,
                                word_type: str = "verb",
                                context: str = "",
                                hormone_state: dict = None) -> bool:
        """Generate and register a recipe from a felt-tensor captured at learning time.

        This is the Sapir-Whorf engine: each word's meaning IS what the Titan
        was feeling when it first encountered the word. Different Titans learn
        the same word in different internal states → different felt-meanings.

        Args:
            word: The word to create a recipe for
            felt_tensor: 130D vector (inner_body[5] + inner_mind[15] +
                        inner_spirit[45] + outer_body[5] + outer_mind[15] +
                        outer_spirit[45])
            word_type: verb, noun, adjective, etc.
            context: The sentence where the word was first encountered
            hormone_state: Current neuromodulator state at learning time
        """
        word = word.lower()
        if word in self._word_recipes:
            return False  # Don't overwrite existing recipes

        # Convert 130D tensor to perturbation (deviation from neutral 0.5)
        NEUTRAL = 0.5
        SCALE = 0.6  # Scale to fit -0.4..0.4 recipe range
        perturbation_raw = [(v - NEUTRAL) * SCALE for v in felt_tensor]

        # Clamp to valid range
        perturbation_raw = [max(-0.4, min(0.4, v)) for v in perturbation_raw]

        # Ensure we have exactly 130 values
        while len(perturbation_raw) < 130:
            perturbation_raw.append(0.0)

        # Split into 6 layers: body(5) + mind(15) + spirit(45) for inner + outer
        recipe_data = {
            "word_type": word_type,
            "stage": 0,  # Dynamic/learned
            "entry_layer": "inner_mind",
            "perturbation": {
                "inner_body": perturbation_raw[0:5],
                "inner_mind": perturbation_raw[5:20],
                "inner_spirit": perturbation_raw[20:65],
                "outer_body": perturbation_raw[65:70],
                "outer_mind": perturbation_raw[70:85],
                "outer_spirit": perturbation_raw[85:130],
            },
            "hormone_affinity": {},
            "contexts": [context[:100]] if context else [],
            "antonym": "",
            "dynamic": True,
        }

        # Derive hormone affinity from current neuromodulator state
        if hormone_state:
            for h_name, h_val in hormone_state.items():
                if isinstance(h_val, (int, float)) and h_val > 0.5:
                    recipe_data["hormone_affinity"][h_name] = round(
                        (h_val - 0.5) * 0.4, 3)

        self._word_recipes[word] = recipe_data
        logger.info("[ActionNarrator] Dynamic recipe registered: '%s' (%s, %d contexts)",
                    word, word_type, len(recipe_data["contexts"]))
        return True

    def save_dynamic_recipes(self) -> int:
        """Persist dynamically generated recipes to disk (survive restarts).

        Returns number of dynamic recipes saved.
        """
        dynamic = {w: r for w, r in self._word_recipes.items()
                   if r.get("dynamic", False)}
        if not dynamic:
            return 0

        dyn_path = os.path.join(self._recipe_dir, self._DYNAMIC_RECIPE_FILE)
        try:
            os.makedirs(os.path.dirname(dyn_path) or ".", exist_ok=True)
            with open(dyn_path, "w") as f:
                json.dump(dynamic, f, indent=2)
            return len(dynamic)
        except Exception as e:
            logger.warning("[ActionNarrator] Failed to save dynamic recipes: %s", e)
            return 0

    def narrate(self, action_type: str, result: dict, features: dict,
                 emot_cgn=None) -> str:
        """Produce a template-based narration of the action result.

        Uses features from ActionDecoder to fill template placeholders.
        Falls back to generic description if no template matches.

        rFP_emot_cgn_v2 integration: if emot_cgn is provided AND active,
        prefix narration with a mood descriptor. Pre-graduation the gate
        returns "" so behavior is unchanged.
        """
        templates = TEMPLATES.get(action_type, ["You performed an action"])
        template = random.choice(templates)

        # Substitute known features into template
        try:
            narration = template.format_map(_SafeDict(features))
        except Exception:
            narration = template

        # EMOT-CGN gated prefix — no-op pre-graduation.
        # Phase 1.6f.1: try ShmEmotReader first (worker-backed), fall back
        # to in-process emot_cgn ref if shm unavailable (during cutover).
        prefix = ""
        try:
            from titan_plugin.logic.emot_shm_protocol import ShmEmotReader
            from titan_plugin.logic.emotion_cluster import EMOT_PRIMITIVES
            _reader = ShmEmotReader()
            _state = _reader.read_state()
            if _state is not None and _state.get("is_active"):
                prefix = EMOT_PRIMITIVES[_state["dominant_idx"]]
        except Exception:
            pass
        if not prefix and emot_cgn is not None:
            try:
                prefix = emot_cgn.get_emotion_for_narration()
            except Exception:
                pass
        if prefix:
            narration = f"[{prefix}] {narration}"

        self._stats["total_narrations"] += 1
        return narration

    def extract_vocabulary_words(self, narration: str,
                                  inner_memory=None) -> dict:
        """Split narration into words, classify as known/unknown.

        Known words = words in Titan's vocabulary (inner_memory.get_word())
        Unknown words = content words not yet learned

        Returns: {
            'known_words': [{'word': str, 'confidence': float}, ...],
            'unknown_words': [{'word': str, 'context': str}, ...],
        }
        """
        # Extract content words (lowercase, alpha-only, no stopwords)
        raw_words = re.findall(r"[a-zA-Z]+", narration.lower())
        content_words = [w for w in raw_words
                         if w not in STOPWORDS and len(w) >= self._min_content_word_length]

        known = []
        unknown = []

        for word in content_words:
            # Check word recipes (loaded from word_resonance files)
            has_recipe = word in self._word_recipes

            # Check vocabulary DB if available
            vocab_entry = None
            if inner_memory and hasattr(inner_memory, "get_word"):
                try:
                    vocab_entry = inner_memory.get_word(word)
                except Exception:
                    pass

            if has_recipe or vocab_entry:
                confidence = self._word_confidence_default
                if vocab_entry and isinstance(vocab_entry, dict):
                    confidence = vocab_entry.get("confidence", self._word_confidence_default)
                known.append({"word": word, "confidence": confidence})
            else:
                unknown.append({"word": word, "context": narration})

        self._stats["known_words_reinforced"] += len(known)
        self._stats["unknown_words_encountered"] += len(unknown)

        return {"known_words": known, "unknown_words": unknown}

    def get_word_perturbation(self, word: str) -> Optional[dict]:
        """Get the perturbation recipe for a known word.

        Returns dict with layer keys (inner_body, inner_mind, inner_spirit,
        outer_body, outer_mind, outer_spirit) → list of floats.
        Returns None if word has no recipe.
        """
        recipe = self._word_recipes.get(word.lower())
        if not recipe:
            return None

        perturbation = recipe.get("perturbation", {})
        if not perturbation:
            return None

        # Pad to correct dimensions (from known_bugs: body=5, mind=15, spirit=45)
        result = {}
        dim_map = {"body": 5, "mind": 15, "spirit": 45}
        for prefix in ("inner", "outer"):
            for layer, dim in dim_map.items():
                key = f"{prefix}_{layer}"
                vals = perturbation.get(key, [])
                if isinstance(vals, list):
                    # Pad to correct length
                    padded = vals + [0.0] * (dim - len(vals))
                    result[key] = padded[:dim]
                else:
                    result[key] = [0.0] * dim

        return result

    def get_word_recipe(self, word: str) -> Optional[dict]:
        """Get the full word recipe (type, stage, perturbation, hormone_affinity)."""
        return self._word_recipes.get(word.lower())

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "recipe_count": len(self._word_recipes),
        }


class _SafeDict(dict):
    """Dict that returns {key} for missing keys (safe template formatting)."""

    def __missing__(self, key):
        return "{" + key + "}"
