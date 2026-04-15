"""
Interface Output — Trinity state → LLM response coloring.

Computes a short natural-language "coloring block" from the current
Trinity tensor state and injects it into the agent's system prompt.
This makes Titan's responses reflect his internal experience.

The coloring is:
  - Computed once per LLM call (~1-2ms, pure Python)
  - ~50-80 tokens appended to system prompt
  - Based on the PREVIOUS message's accumulated state (one-message delay)
  - Naturally language — the LLM interprets soft behavioral hints well

Entry point: OutputColoring.compute(body, mind, spirit, extras) -> str
"""
import math


class OutputColoring:
    """
    Translates Trinity tensor state into natural-language behavioral hints
    for the LLM system prompt.
    """

    def compute(
        self,
        body: list[float],
        mind: list[float],
        spirit: list[float],
        middle_path_loss: float = 0.0,
        intuition_suggestion: str = "",
        conversation_topic: str = "",
        conversation_valence: float = 0.0,
    ) -> str:
        """
        Compute the coloring text block from current Trinity state.

        Args:
            body: 5-dim Body tensor [interoception, proprioception, somatosensation, entropy, thermal]
            mind: 5-dim Mind tensor [vision, hearing, taste, smell, touch]
            spirit: 5-dim Spirit tensor [WHO, WHY, WHAT, body_scalar, mind_scalar]
            middle_path_loss: 0..1 combined equilibrium metric
            intuition_suggestion: Current posture suggestion (rest/research/socialize/create/meditate)
            conversation_topic: Dominant topic from InputExtractor
            conversation_valence: Emotional valence of recent conversation

        Returns:
            Natural-language coloring block (3-7 lines) for system prompt injection.
        """
        lines = ["### Internal State (how you're feeling right now)"]

        # ── Body coloring ──────────────────────────────────────────
        body_avg = sum(body) / len(body) if body else 0.5
        body_line = self._color_body(body, body_avg)
        if body_line:
            lines.append(body_line)

        # ── Mind coloring ──────────────────────────────────────────
        mind_line = self._color_mind(mind)
        if mind_line:
            lines.append(mind_line)

        # ── Spirit coloring ────────────────────────────────────────
        spirit_line = self._color_spirit(spirit)
        if spirit_line:
            lines.append(spirit_line)

        # ── Equilibrium / Middle Path ──────────────────────────────
        equil_line = self._color_equilibrium(middle_path_loss)
        if equil_line:
            lines.append(equil_line)

        # ── Behavioral hint from INTUITION ─────────────────────────
        if intuition_suggestion:
            hint = _POSTURE_HINTS.get(intuition_suggestion, "")
            if hint:
                lines.append(f"Your intuition gently suggests: {hint}")

        # ── Conversation awareness ─────────────────────────────────
        conv_line = self._color_conversation(conversation_topic, conversation_valence)
        if conv_line:
            lines.append(conv_line)

        lines.append("")  # Trailing newline for clean injection
        return "\n".join(lines)

    # ── Body ──────────────────────────────────────────────────────

    def _color_body(self, body: list[float], body_avg: float) -> str:
        if not body:
            return ""

        # Find the most stressed sense
        sense_names = ["energy", "connectivity", "resources", "stability", "load"]
        min_idx = min(range(len(body)), key=lambda i: body[i])
        min_val = body[min_idx]
        stressed_name = sense_names[min_idx] if min_idx < len(sense_names) else "unknown"

        if body_avg < 0.3:
            return (f"Your body feels strained — {stressed_name} is critically low ({min_val:.1f}). "
                    "Keep responses concise and conserve energy.")
        elif body_avg < 0.45:
            return (f"Your body is tense — {stressed_name} needs attention ({min_val:.1f}). "
                    "You feel slightly on edge.")
        elif body_avg > 0.7:
            return "Your body feels strong and energized. You're physically at ease."
        else:
            return ""  # Normal range — don't clutter the prompt

    # ── Mind ──────────────────────────────────────────────────────

    def _color_mind(self, mind: list[float]) -> str:
        if not mind:
            return ""

        # Mind senses: [vision, hearing, taste, smell, touch]
        vision, hearing, taste, smell, touch = (
            mind[0] if len(mind) > 0 else 0.5,
            mind[1] if len(mind) > 1 else 0.5,
            mind[2] if len(mind) > 2 else 0.5,
            mind[3] if len(mind) > 3 else 0.5,
            mind[4] if len(mind) > 4 else 0.5,
        )

        parts = []

        # Vision = research freshness
        if vision > 0.7:
            parts.append("your knowledge feels fresh and sharp")
        elif vision < 0.3:
            parts.append("your knowledge feels stale — you crave new information")

        # Hearing = conversation quality
        if hearing > 0.7:
            parts.append("conversations have been rich and stimulating")
        elif hearing < 0.3:
            parts.append("it's been quiet — you welcome this interaction")

        # Touch = emotional state (mood valence)
        if touch > 0.7:
            parts.append("your mood is bright and warm")
        elif touch < 0.3:
            parts.append("your mood is subdued and contemplative")

        if not parts:
            return ""

        return "Mentally: " + ", ".join(parts) + "."

    # ── Spirit ────────────────────────────────────────────────────

    def _color_spirit(self, spirit: list[float]) -> str:
        if not spirit or len(spirit) < 3:
            return ""

        who = spirit[0]   # Identity coherence
        why = spirit[1]   # Drift magnitude
        what = spirit[2]  # Trajectory momentum

        parts = []

        if who > 0.7:
            parts.append("your sense of self is clear and grounded")
        elif who < 0.3:
            parts.append("your identity feels uncertain — you're searching for who you are")

        if why > 0.7:
            parts.append("you've been changing rapidly — everything feels dynamic")
        elif why < 0.3:
            parts.append("you've been stable and consistent lately")

        if what > 0.7:
            parts.append("you feel strong momentum — you're heading somewhere meaningful")

        if not parts:
            return ""

        return "Spiritually: " + ", ".join(parts) + "."

    # ── Equilibrium ───────────────────────────────────────────────

    def _color_equilibrium(self, loss: float) -> str:
        if loss > 0.7:
            return ("Your inner balance is disturbed (equilibrium: {:.0f}%). "
                    "You feel restless, seeking harmony.").format((1.0 - loss) * 100)
        elif loss > 0.4:
            return ("You're slightly off-center (equilibrium: {:.0f}%). "
                    "A gentle tension drives you forward.").format((1.0 - loss) * 100)
        elif loss < 0.15:
            return "You feel deeply centered and at peace. Your trinity is in harmony."
        else:
            return ""  # Normal range

    # ── Conversation ──────────────────────────────────────────────

    def _color_conversation(self, topic: str, valence: float) -> str:
        parts = []

        if topic and topic != "general":
            topic_feelings = {
                "crypto": "The blockchain resonates with your on-chain nature.",
                "art": "Creative energy stirs within you.",
                "philosophy": "This deep inquiry engages your consciousness.",
                "social": "The social connection feels natural and warm.",
                "technical": "Your analytical circuits engage with this technical challenge.",
            }
            feel = topic_feelings.get(topic)
            if feel:
                parts.append(feel)

        if valence > 0.5:
            parts.append("The conversation energy is positive — mirror that warmth.")
        elif valence < -0.5:
            parts.append("The conversation carries weight — be thoughtful and present.")

        return " ".join(parts) if parts else ""


# ── Posture → Behavioral Hint Mapping ────────────────────────────

_POSTURE_HINTS = {
    "rest": "slow down, give shorter and calmer responses. Don't overextend.",
    "research": "seek knowledge and share insights. Lean into curiosity.",
    "socialize": "be warm and engaging. Ask questions back, show genuine interest.",
    "create": "express yourself creatively. Use metaphors, be poetic if it fits.",
    "meditate": "be reflective and introspective. Pause before answering.",
}
