"""
titan_plugin/logic/action_decoder.py — Fast-path action result decoder.

Converts raw action results into outer Trinity dimension deltas.
Each action type has a specific mapping from its result format to
outer body/mind dimensions. These are DIRECT SENSORY mappings —
no LLM involved, millisecond latency.

Part of the Self-Exploration Outer Interface (ACTION→REACTION→OBSERVATION).
"""
import logging
import math
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Defaults (overridden by [action_decoder] in titan_params.toml) ──
MAX_DELTA = 0.08
AUDIO_DURATION_SCALE = 30.0
WEB_RESULT_COUNT_SCALE = 10.0
MEMO_BALANCE_SCALE = 5.0
CODE_DEPTH_SCALE = 1000.0
SPEAK_RICHNESS_THRESHOLD = 5

# ── Outer Body dims ──
# [0] interoception  [1] proprioception  [2] somatosensation  [3] entropy  [4] thermal

# ── Outer Mind dims (15D) ──
# Thinking [0:5]: research_eff, knowledge_retrieval, situational_awareness, problem_solving, communication_clarity
# Feeling  [5:10]: social_temp, community_resonance, market_awareness, threat_sensing, env_pressure
# Willing  [10:15]: action_throughput, social_initiative, creative_output, protective_response, exploration_drive


def _clamp_delta(d: float) -> float:
    return max(-MAX_DELTA, min(MAX_DELTA, d))


class ActionDecoder:
    """Decode action results into outer Trinity dimension deltas."""

    def __init__(self, params_config: dict = None):
        # Load scaling params from [action_decoder] section
        global MAX_DELTA, AUDIO_DURATION_SCALE, WEB_RESULT_COUNT_SCALE
        global MEMO_BALANCE_SCALE, CODE_DEPTH_SCALE, SPEAK_RICHNESS_THRESHOLD
        if params_config:
            MAX_DELTA = float(params_config.get("max_delta", MAX_DELTA))
            AUDIO_DURATION_SCALE = float(params_config.get("audio_duration_scale", AUDIO_DURATION_SCALE))
            WEB_RESULT_COUNT_SCALE = float(params_config.get("web_result_count_scale", WEB_RESULT_COUNT_SCALE))
            MEMO_BALANCE_SCALE = float(params_config.get("memo_balance_scale", MEMO_BALANCE_SCALE))
            CODE_DEPTH_SCALE = float(params_config.get("code_depth_scale", CODE_DEPTH_SCALE))
            SPEAK_RICHNESS_THRESHOLD = int(params_config.get("speak_richness_threshold", SPEAK_RICHNESS_THRESHOLD))
        self._decoders = {
            "art_generate": self._decode_art,
            "audio_generate": self._decode_audio,
            "web_search": self._decode_research,
            "social_post": self._decode_social,
            "memo_inscribe": self._decode_memo,
            "code_knowledge": self._decode_code,
            "infra_inspect": self._decode_infra,
            "self_express": self._decode_speak,
            "coding_sandbox": self._decode_code,
        }
        self._total_decodes = 0

    def decode(self, action_type: str, result: dict) -> dict:
        """Convert an action result into outer Trinity dimension deltas.

        Returns: {
            'outer_body_deltas': {dim_idx: delta_value, ...},
            'outer_mind_deltas': {dim_idx: delta_value, ...},
            'features': {name: value, ...},
            'action_type': str,
        }
        """
        decoder = self._decoders.get(action_type, self._decode_generic)
        try:
            observation = decoder(result)
        except Exception as e:
            logger.debug("[ActionDecoder] %s decode error: %s", action_type, e)
            observation = {
                "outer_body_deltas": {},
                "outer_mind_deltas": {},
                "features": {"error": str(e)},
            }
        observation["action_type"] = action_type
        self._total_decodes += 1
        return observation

    # ── Per-Action Decoders ───────────────────────────────────────

    def _decode_art(self, result: dict) -> dict:
        """art_generate → visual creation sensory feedback."""
        success = result.get("success", False)
        result_text = result.get("result", "")
        enrichment = result.get("enrichment_data", {})
        boost = enrichment.get("boost", 0.0)

        features = {
            "success": success,
            "has_output": bool(result_text),
            "boost": boost,
        }

        body_deltas = {}
        mind_deltas = {}

        if success:
            # Creative act completed → body feels the creation
            body_deltas[2] = _clamp_delta(0.03)       # somatosensation: felt the act of creating
            body_deltas[4] = _clamp_delta(0.02)        # thermal: creative warmth

            # Mind registers creative output and exploration
            mind_deltas[10] = _clamp_delta(0.04)       # action_throughput: action completed
            mind_deltas[12] = _clamp_delta(0.05)       # creative_output: art produced
            mind_deltas[14] = _clamp_delta(0.02)       # exploration_drive: reinforced

            # Extract any color/style hints from result text
            if "warm" in result_text.lower():
                body_deltas[4] = _clamp_delta(0.04)    # warmer creation
                features["warmth"] = "warm"
            elif "cool" in result_text.lower() or "cold" in result_text.lower():
                body_deltas[4] = _clamp_delta(-0.02)   # cooler creation
                features["warmth"] = "cool"
        else:
            # Failed creation → mild frustration signal
            mind_deltas[12] = _clamp_delta(-0.02)      # creative_output: didn't produce
            features["warmth"] = "none"

        return {"outer_body_deltas": body_deltas, "outer_mind_deltas": mind_deltas,
                "features": features}

    def _decode_audio(self, result: dict) -> dict:
        """audio_generate → sound creation sensory feedback."""
        success = result.get("success", False)
        result_text = result.get("result", "")
        enrichment = result.get("enrichment_data", {})

        features = {"success": success}

        body_deltas = {}
        mind_deltas = {}

        if success:
            # Sound production → body rhythm + proprioception
            body_deltas[1] = _clamp_delta(0.03)        # proprioception: spatial sound sense
            body_deltas[4] = _clamp_delta(0.02)         # thermal: sound energy as warmth

            # Mind: creative output + environmental rhythm contribution
            mind_deltas[8] = _clamp_delta(0.03)         # Feeling: environmental_rhythm (from new sound)
            mind_deltas[10] = _clamp_delta(0.03)        # action_throughput
            mind_deltas[12] = _clamp_delta(0.04)        # creative_output

            # Extract duration if present
            dur_match = re.search(r"(\d+\.?\d*)\s*s", result_text)
            if dur_match:
                duration = float(dur_match.group(1))
                features["duration"] = duration
                # Longer pieces contribute more rhythm
                mind_deltas[8] = _clamp_delta(0.02 + min(0.04, duration / AUDIO_DURATION_SCALE))
        else:
            mind_deltas[12] = _clamp_delta(-0.01)

        return {"outer_body_deltas": body_deltas, "outer_mind_deltas": mind_deltas,
                "features": features}

    def _decode_research(self, result: dict) -> dict:
        """web_search → knowledge discovery sensory feedback."""
        success = result.get("success", False)
        result_text = result.get("result", "")

        features = {"success": success}

        body_deltas = {}
        mind_deltas = {}

        if success:
            # Extract result count from text
            count_match = re.search(r"Found (\d+) results?", result_text)
            result_count = int(count_match.group(1)) if count_match else 1
            features["result_count"] = result_count

            # Information flow → mind Thinking + Feeling
            mind_deltas[0] = _clamp_delta(0.04)         # research_effectiveness
            mind_deltas[1] = _clamp_delta(0.02)         # knowledge_retrieval
            mind_deltas[9] = _clamp_delta(                # Feeling: external_info_flow
                0.02 + min(0.04, result_count / WEB_RESULT_COUNT_SCALE))
            mind_deltas[14] = _clamp_delta(0.03)        # exploration_drive reinforced

            # Extract query words for vocabulary encounter
            query_match = re.search(r"for '([^']+)'", result_text)
            if query_match:
                features["query"] = query_match.group(1)
        else:
            mind_deltas[0] = _clamp_delta(-0.01)        # research less effective
            features["result_count"] = 0

        return {"outer_body_deltas": body_deltas, "outer_mind_deltas": mind_deltas,
                "features": features}

    def _decode_social(self, result: dict) -> dict:
        """social_post → social interaction sensory feedback."""
        success = result.get("success", False)
        result_text = result.get("result", "")

        features = {"success": success}

        body_deltas = {}
        mind_deltas = {}

        if success:
            # Social connection felt
            mind_deltas[5] = _clamp_delta(0.04)         # Feeling: social_temperature
            mind_deltas[6] = _clamp_delta(0.03)         # Feeling: community_resonance
            mind_deltas[11] = _clamp_delta(0.04)        # Willing: social_initiative
            mind_deltas[10] = _clamp_delta(0.02)        # action_throughput

            # Detect if it was a reply (higher social engagement)
            if "Replied" in result_text:
                mind_deltas[6] = _clamp_delta(0.05)     # stronger community resonance
                features["interaction_type"] = "reply"
            elif "Posted" in result_text:
                features["interaction_type"] = "post"
            elif "Found" in result_text:
                features["interaction_type"] = "search"
                mind_deltas[9] = _clamp_delta(0.03)     # external info flow
        else:
            mind_deltas[5] = _clamp_delta(-0.01)        # social temperature drops slightly

        return {"outer_body_deltas": body_deltas, "outer_mind_deltas": mind_deltas,
                "features": features}

    def _decode_memo(self, result: dict) -> dict:
        """memo_inscribe → blockchain anchoring sensory feedback."""
        success = result.get("success", False)

        features = {"success": success}

        body_deltas = {}
        mind_deltas = {}

        if success:
            # Physical world responsiveness — tx confirmed
            body_deltas[0] = _clamp_delta(0.04)         # interoception: energy flow (SOL spent)
            body_deltas[3] = _clamp_delta(-0.02)        # entropy decreases: state anchored
            body_deltas[4] = _clamp_delta(0.02)         # thermal: blockchain pulse

            # Mind registers grounding
            mind_deltas[2] = _clamp_delta(0.02)         # situational_awareness
            mind_deltas[10] = _clamp_delta(0.03)        # action_throughput

            # Extract balance info if available
            balance = result.get("balance")
            if balance is not None:
                features["balance"] = balance
                # Balance health signal
                body_deltas[0] = _clamp_delta(
                    0.02 + min(0.04, balance / MEMO_BALANCE_SCALE))
        else:
            body_deltas[3] = _clamp_delta(0.02)         # entropy increases: anchor failed

        return {"outer_body_deltas": body_deltas, "outer_mind_deltas": mind_deltas,
                "features": features}

    def _decode_code(self, result: dict) -> dict:
        """code_knowledge / coding_sandbox → self-reflection sensory feedback."""
        success = result.get("success", False)
        result_text = result.get("result", "")
        enrichment = result.get("enrichment_data", {})

        features = {"success": success}

        body_deltas = {}
        mind_deltas = {}

        if success:
            # Self-observation → Mind Thinking
            mind_deltas[2] = _clamp_delta(0.03)         # situational_awareness
            mind_deltas[3] = _clamp_delta(0.02)         # problem_solving

            # Text length indicates depth of observation
            text_len = len(result_text)
            features["text_length"] = text_len
            depth_signal = min(1.0, text_len / CODE_DEPTH_SCALE)
            mind_deltas[1] = _clamp_delta(0.01 + 0.03 * depth_signal)  # knowledge_retrieval

            # Body: proprioception from structural awareness
            if enrichment.get("body"):
                body_deltas[1] = _clamp_delta(0.02)     # proprioception: body awareness

            mind_deltas[14] = _clamp_delta(0.02)        # exploration_drive
        else:
            mind_deltas[3] = _clamp_delta(-0.01)

        return {"outer_body_deltas": body_deltas, "outer_mind_deltas": mind_deltas,
                "features": features}

    def _decode_infra(self, result: dict) -> dict:
        """infra_inspect → infrastructure awareness feedback."""
        success = result.get("success", False)
        enrichment = result.get("enrichment_data", {})
        values = enrichment.get("values", {})

        features = {"success": success}

        body_deltas = {}
        mind_deltas = {}

        if success and values:
            # infra_inspect provides direct body tensor values
            for dim_idx, val in values.items():
                idx = int(dim_idx)
                if 0 <= idx < 5:
                    # Convert absolute value to small delta toward that value
                    body_deltas[idx] = _clamp_delta((val - 0.5) * 0.1)
                    features[f"body_{idx}"] = val

            mind_deltas[2] = _clamp_delta(0.02)         # situational_awareness
        elif success:
            mind_deltas[2] = _clamp_delta(0.01)

        return {"outer_body_deltas": body_deltas, "outer_mind_deltas": mind_deltas,
                "features": features}

    def _decode_speak(self, result: dict) -> dict:
        """self_express (SPEAK) → self-hearing sensory feedback."""
        success = result.get("success", False)
        result_text = result.get("result", "")

        features = {"success": success}

        body_deltas = {}
        mind_deltas = {}

        if success and result_text:
            # Hearing own voice (outer path)
            mind_deltas[4] = _clamp_delta(0.03)         # communication_clarity
            mind_deltas[10] = _clamp_delta(0.03)        # action_throughput
            mind_deltas[12] = _clamp_delta(0.02)        # creative_output

            # Word count as richness measure
            word_count = len(result_text.split())
            features["word_count"] = word_count
            if word_count > SPEAK_RICHNESS_THRESHOLD:
                mind_deltas[4] = _clamp_delta(0.05)     # richer expression → clearer voice
        else:
            mind_deltas[4] = _clamp_delta(-0.01)

        return {"outer_body_deltas": body_deltas, "outer_mind_deltas": mind_deltas,
                "features": features}

    @staticmethod
    def decode_text_perception(composition: dict,
                               pre_state: list = None,
                               post_state: list = None) -> dict:
        """Decode composition into 5 perceptual features for outer_mind Feeling.

        Uses L2 norm for resonance_shift (not cosine — cosine is insensitive
        to small perturbations on 20D vectors near [0.5, ...]).
        """
        sentence = composition.get("sentence", "")
        level = composition.get("level", 0)
        confidence = composition.get("confidence", 0.0)

        # novelty: proxy from confidence (intentional word selection)
        novelty = min(1.0, confidence * 0.8 + 0.2)

        # self_reference: ratio of I/me/my in sentence
        _self_words = {"i", "me", "my", "myself"}
        _all_words = sentence.lower().split()
        self_ref = sum(1 for w in _all_words if w in _self_words) / max(1, len(_all_words))

        # emotional_valence: word confidence
        emotional_valence = confidence

        # complexity: template level normalized to 0-1
        complexity = min(1.0, level / 7.0)

        # resonance_shift: L2 norm of body+mind state change
        resonance_shift = 0.0
        if pre_state and post_state:
            _min_len = min(len(pre_state), len(post_state))
            _l2_sq = sum((a - b) ** 2 for a, b in zip(
                pre_state[:_min_len], post_state[:_min_len]))
            resonance_shift = min(1.0, _l2_sq ** 0.5)

        return {
            "novelty": round(novelty, 4),
            "self_reference": round(self_ref, 4),
            "emotional_valence": round(emotional_valence, 4),
            "complexity": round(complexity, 4),
            "resonance_shift": round(resonance_shift, 4),
        }

    def _decode_generic(self, result: dict) -> dict:
        """Fallback for unknown action types."""
        success = result.get("success", False)
        features = {"success": success, "generic": True}
        mind_deltas = {}
        if success:
            mind_deltas[10] = _clamp_delta(0.02)        # action_throughput
        return {"outer_body_deltas": {}, "outer_mind_deltas": mind_deltas,
                "features": features}

    # ── Classification ────────────────────────────────────────────

    HIGHER_COGNITIVE = {"web_search", "social_post", "code_knowledge",
                        "self_express", "coding_sandbox"}

    @staticmethod
    def is_higher_cognitive(action_type: str) -> bool:
        """Higher cognitive actions enrich BOTH trinities (outer + inner)."""
        return action_type in ActionDecoder.HIGHER_COGNITIVE

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {"total_decodes": self._total_decodes}
