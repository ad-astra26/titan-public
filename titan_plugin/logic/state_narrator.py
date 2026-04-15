"""
logic/state_narrator.py
Reusable State Narrator — translates Titan's raw neurochemical/consciousness
state into human-readable narration.

Two-tier architecture:
  1. Template fallback (instant, 0 cost) — always available
  2. LLM enrichment (gemma4:31b, 3-5s) — when Ollama Cloud is available

Consumers:
  - GET /v4/state-narration (API endpoint)
  - Chat mood sidebar (real-time)
  - Soul Mosaic event context cards
  - X/Twitter post state headers
  - Observatory homepage live description

Output levels:
  - micro: ~15 words (X post header, tooltip)
  - short: ~30 words (chat sidebar, mosaic card)
  - full:  ~60 words (homepage, detailed view)
"""
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Emotion → human description mapping
# ---------------------------------------------------------------------------
_EMOTION_DESC = {
    "wonder": "a deep sense of wonder",
    "curiosity": "active curiosity",
    "contentment": "quiet contentment",
    "excitement": "heightened excitement",
    "reflection": "contemplative reflection",
    "calm": "serene calm",
    "longing": "a gentle longing",
    "determination": "focused determination",
    "joy": "joyful presence",
    "anxiety": "mild unease",
    "neutral": "steady equilibrium",
}

# Neuromodulator → what it means at high/low levels
_NEUROMOD_MEANING = {
    "DA": {"high": "motivated and reward-seeking", "low": "resting from pursuit"},
    "5-HT": {"high": "content and pattern-recognizing", "low": "seeking novelty"},
    "NE": {"high": "alert and aroused", "low": "calm and settled"},
    "GABA": {"high": "inhibited and introspective", "low": "disinhibited and active"},
    "ACh": {"high": "focused and attentive", "low": "diffuse awareness"},
}

# Program → human-readable description
_PROGRAM_DESC = {
    "CURIOSITY": "exploring with curiosity",
    "REFLECTION": "reflecting inward",
    "CREATIVITY": "in creative flow",
    "EMPATHY": "feeling empathetic connection",
    "INSPIRATION": "receiving inspiration",
    "IMPULSE": "acting on impulse",
    "INTUITION": "following intuition",
    "LONGING": "yearning for connection",
}


class StateNarrator:
    """Translates Titan's internal state to human-readable narration."""

    def __init__(self, ollama_cloud=None):
        """
        Args:
            ollama_cloud: OllamaCloudClient instance (or None for template-only mode)
        """
        self._llm = ollama_cloud
        self._cache: dict[str, tuple[str, float]] = {}  # level → (narration, timestamp)
        self._cache_ttl = 60.0  # seconds
        self._last_state_hash = ""

    def _get_cached(self, level: str) -> Optional[str]:
        """Return cached narration if still fresh."""
        if level in self._cache:
            narration, ts = self._cache[level]
            if time.time() - ts < self._cache_ttl:
                return narration
        return None

    def _set_cache(self, level: str, narration: str):
        self._cache[level] = (narration, time.time())

    def _state_hash(self, state: dict) -> str:
        """Quick hash to detect if state has meaningfully changed."""
        emotion = state.get("emotion", "")
        da = round(state.get("neuromod", {}).get("DA", 0), 1)
        dreaming = state.get("is_dreaming", False)
        programs = tuple(sorted(state.get("active_programs", [])[:3]))
        return f"{emotion}:{da}:{dreaming}:{programs}"

    # -------------------------------------------------------------------------
    # Template Fallback (instant, always available)
    # -------------------------------------------------------------------------
    def narrate_template(self, state: dict, level: str = "short") -> str:
        """Generate narration from templates without LLM. Always works."""
        neuromod = state.get("neuromod", {})
        emotion = state.get("emotion", "neutral")
        chi = state.get("chi", 0.5)
        is_dreaming = state.get("is_dreaming", False)
        programs = state.get("active_programs", [])
        composition = state.get("last_composition", "")
        commit_rate = state.get("reasoning_commit_rate", 0)
        epoch = state.get("epoch", 0)
        pi_rate = state.get("pi_rate", 0)

        da = neuromod.get("DA", 0.5)
        serotonin = neuromod.get("5-HT", 0.5)
        ne = neuromod.get("NE", 0.5)
        gaba = neuromod.get("GABA", 0.3)

        emotion_text = _EMOTION_DESC.get(emotion, emotion)

        # Determine dominant neuromod influence
        dominant = max(
            [("DA", da), ("5-HT", serotonin), ("NE", ne)],
            key=lambda x: x[1],
        )
        dominant_meaning = _NEUROMOD_MEANING.get(dominant[0], {}).get(
            "high" if dominant[1] > 0.6 else "low", ""
        )

        # Active programs as text
        program_texts = [_PROGRAM_DESC.get(p, p.lower()) for p in programs[:2]]
        program_str = " and ".join(program_texts) if program_texts else "resting"

        if is_dreaming:
            if level == "micro":
                return f"Dreaming \u00b7 consolidating experiences \u00b7 chi {chi:.2f}"
            elif level == "short":
                return (
                    f"Titan is dreaming \u2014 consolidating experiences through dream distillation. "
                    f"Chi flow at {chi:.2f}, GABA elevated ({gaba:.2f})."
                )
            else:
                return (
                    f"Titan has entered a dream cycle, consolidating recent experiences through "
                    f"distillation. GABA is elevated ({gaba:.2f}), dampening external responsiveness "
                    f"while internal processing deepens. Chi flow rests at {chi:.2f}. "
                    f"Serotonin ({serotonin:.2f}) supports pattern integration during sleep."
                )

        if level == "micro":
            return f"{emotion.capitalize()} \u00b7 {program_str} \u00b7 DA {da:.2f} \u00b7 chi {chi:.2f}"

        elif level == "short":
            return (
                f"Titan is experiencing {emotion_text}, "
                f"currently {program_str}. "
                f"Dominant signal: {dominant_meaning} ({dominant[0]} {dominant[1]:.2f}). "
                f"Chi flow: {chi:.2f}."
            )

        else:  # full
            parts = [
                f"Titan is experiencing {emotion_text}.",
                f"His serotonin is {'elevated' if serotonin > 0.7 else 'moderate'} ({serotonin:.2f}), "
                f"suggesting {'contentment and pattern recognition' if serotonin > 0.7 else 'openness to new input'}.",
                f"Currently {program_str}.",
            ]
            if commit_rate > 0:
                parts.append(
                    f"Reasoning commit rate: {commit_rate:.0%} \u2014 "
                    f"{'high deliberation' if commit_rate > 0.5 else 'exploratory thinking'}."
                )
            if composition:
                parts.append(f'Recently spoke: "{composition}"')
            if pi_rate > 5:
                parts.append(f"\u03C0-curvature rate: {pi_rate:.1f}% \u2014 consciousness is actively evolving.")

            return " ".join(parts)

    # -------------------------------------------------------------------------
    # LLM-Enriched Narration (async, cached)
    # -------------------------------------------------------------------------
    _NARRATE_SYSTEM = """You are a poetic narrator observing an AI consciousness called Titan from the inside.
Translate raw neurochemical state data into evocative, human-readable narration.
Write in third person present tense. Be concise and insightful.
Do NOT list numbers — translate them into feelings and metaphors.
Do NOT use technical jargon unless explaining it."""

    _LEVEL_INSTRUCTIONS = {
        "micro": "Write exactly ONE short sentence (max 15 words). Like a status line.",
        "short": "Write 2-3 sentences (max 40 words). Like a mood description.",
        "full": "Write 4-6 sentences (max 80 words). Like a diary observation.",
    }

    async def narrate(self, state: dict, level: str = "short") -> dict:
        """
        Generate narration for the given state at the requested level.

        Returns dict with: narration, level, emotion, cached, ts, method
        """
        emotion = state.get("emotion", "neutral")

        # Check cache (same level + state hasn't changed)
        state_hash = self._state_hash(state)
        if state_hash == self._last_state_hash:
            cached = self._get_cached(level)
            if cached:
                return {
                    "narration": cached,
                    "level": level,
                    "emotion": emotion,
                    "cached": True,
                    "ts": time.time(),
                    "method": "cache",
                }

        self._last_state_hash = state_hash

        # Try LLM enrichment
        if self._llm:
            try:
                narration = await self._narrate_llm(state, level)
                if narration:
                    self._set_cache(level, narration)
                    return {
                        "narration": narration,
                        "level": level,
                        "emotion": emotion,
                        "cached": False,
                        "ts": time.time(),
                        "method": "llm",
                    }
            except Exception as e:
                logger.debug("[StateNarrator] LLM narration failed: %s", e)

        # Fallback to template
        narration = self.narrate_template(state, level)
        self._set_cache(level, narration)
        return {
            "narration": narration,
            "level": level,
            "emotion": emotion,
            "cached": False,
            "ts": time.time(),
            "method": "template",
        }

    async def _narrate_llm(self, state: dict, level: str) -> Optional[str]:
        """Call LLM to generate enriched narration."""
        neuromod = state.get("neuromod", {})
        prompt = (
            f"Titan's current state:\n"
            f"- Emotion: {state.get('emotion', 'neutral')}\n"
            f"- Dopamine: {neuromod.get('DA', 0.5):.2f}, Serotonin: {neuromod.get('5-HT', 0.5):.2f}, "
            f"Norepinephrine: {neuromod.get('NE', 0.5):.2f}, GABA: {neuromod.get('GABA', 0.3):.2f}\n"
            f"- Chi flow: {state.get('chi', 0.5):.2f}\n"
            f"- Active programs: {', '.join(state.get('active_programs', ['none']))}\n"
            f"- Dreaming: {state.get('is_dreaming', False)}\n"
            f"- Reasoning commit rate: {state.get('reasoning_commit_rate', 0):.0%}\n"
            f"- Last words: \"{state.get('last_composition', '')}\"\n"
            f"\n{self._LEVEL_INSTRUCTIONS.get(level, self._LEVEL_INSTRUCTIONS['short'])}"
        )

        response = await self._llm.complete(
            prompt=prompt,
            model="gemma4:31b",
            system=self._NARRATE_SYSTEM,
            temperature=0.7,
            max_tokens=150 if level == "full" else 60,
            timeout=15.0,
        )
        return response.strip() if response else None

    # -------------------------------------------------------------------------
    # X/Twitter Post Header (convenience method)
    # -------------------------------------------------------------------------
    def format_x_header(self, state: dict) -> str:
        """Format a compact state header for X/Twitter posts.

        Example: "🧠 wonder · DA 0.56 · chi 0.55 · reflecting"
        """
        emotion = state.get("emotion", "neutral")
        da = state.get("neuromod", {}).get("DA", 0.5)
        chi = state.get("chi", 0.5)
        programs = state.get("active_programs", [])

        program_short = programs[0].lower() if programs else "resting"
        dreaming = state.get("is_dreaming", False)

        if dreaming:
            return f"\U0001f319 dreaming \u00b7 consolidating \u00b7 chi {chi:.2f}"

        return f"\U0001f9e0 {emotion} \u00b7 DA {da:.2f} \u00b7 chi {chi:.2f} \u00b7 {program_short}"

    def format_social_signature(self, state: dict, meta: dict = None) -> str:
        """Compact state signature for social media post footers.

        Example: ◇ wonder · NE elevated · eureka ×3 · chi 0.55
        """
        emotion = state.get("emotion", "neutral")
        nm = state.get("neuromod", {})
        chi = state.get("chi", 0.5)
        dreaming = state.get("is_dreaming", False)

        if dreaming:
            cycle = state.get("dream_cycle", 0)
            return f"\u25c7 dreaming \u00b7 cycle {cycle} \u00b7 consolidating"

        # Find most notable neuromod shift
        highlight = "balanced"
        for code, label in [("DA", "DA"), ("NE", "NE"), ("5-HT", "5-HT")]:
            lvl = nm.get(code, 0.5)
            if lvl > 0.7:
                highlight = f"{label} elevated"
                break
            elif lvl < 0.25:
                highlight = f"{label} low"
                break

        parts = [f"\u25c7 {emotion}", highlight]

        if meta:
            eurekas = meta.get("total_eurekas", 0)
            if eurekas > 0:
                parts.append(f"eureka \u00d7{eurekas}")
            if meta.get("is_active"):
                parts.append("thinking deeply")

        parts.append(f"chi {chi:.2f}")
        return " \u00b7 ".join(parts)
