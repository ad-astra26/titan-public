"""
Sovereign Tool Reflexes — Architecture-driven tool invocation.

Tools fire as bodily reflexes driven by Trinity Intuition convergence.
Each Trinity worker (Body, Mind, Spirit) independently computes a confidence
signal for each possible reflex. Signals multiply together — only when the
whole self agrees does a reflex fire.

The LLM never "calls" tools. Reflex results arrive as a perceptual field
in the LLM's context — Titan perceives tool results like environmental
data, not function outputs.

FOCUS PID controller amplifies reflex Intuition when the system is actively
correcting drift (adrenaline-like override for engaged states).
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Reflex Types ──────────────────────────────────────────────────

class ReflexType(Enum):
    """All available reflexes, grouped by Trinity origin."""
    # Body reflexes (somatic, infrastructure) — OBSERVATION
    IDENTITY_CHECK = "identity_check"
    METABOLISM_CHECK = "metabolism_check"
    INFRA_CHECK = "infra_check"

    # Mind reflexes (cognitive, knowledge) — OBSERVATION
    MEMORY_RECALL = "memory_recall"
    KNOWLEDGE_SEARCH = "knowledge_search"
    SOCIAL_CONTEXT = "social_context"

    # Spirit reflexes (identity, consciousness) — OBSERVATION
    SELF_REFLECTION = "self_reflection"
    TIME_AWARENESS = "time_awareness"
    GUARDIAN_SHIELD = "guardian_shield"

    # Action reflexes — SOVEREIGN ACTIONS (higher thresholds)
    ART_GENERATE = "art_generate"
    AUDIO_GENERATE = "audio_generate"
    RESEARCH = "research"
    SOCIAL_POST = "social_post"


# Mapping from string to enum for bus messages
REFLEX_TYPE_MAP = {rt.value: rt for rt in ReflexType}

# Guardian shield is BLOCKING — must complete before LLM response
BLOCKING_REFLEXES = {ReflexType.GUARDIAN_SHIELD}

# Action reflexes need higher Trinity convergence to fire
ACTION_REFLEXES = {
    ReflexType.ART_GENERATE,
    ReflexType.AUDIO_GENERATE,
    ReflexType.RESEARCH,
}

# Public actions need near-unanimous convergence (side effects visible to others)
PUBLIC_ACTION_REFLEXES = {
    ReflexType.SOCIAL_POST,
}


# ── Data Structures ───────────────────────────────────────────────

@dataclass
class ReflexSignal:
    """One Trinity worker's Intuition about one reflex."""
    reflex: str         # ReflexType.value
    source: str         # "body" | "mind" | "spirit"
    confidence: float   # 0.0-1.0 — how strongly this worker wants this reflex
    reason: str = ""    # why (for consciousness logging)


@dataclass
class FiredReflex:
    """A reflex that was selected to fire."""
    reflex_type: ReflexType
    combined_confidence: float  # body × mind × spirit
    signals: list               # contributing ReflexSignal dicts
    result: Optional[dict] = None
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class PerceptualField:
    """The assembled result of all fired reflexes for one interaction."""
    fired_reflexes: list[FiredReflex] = field(default_factory=list)
    reflex_notices: list[str] = field(default_factory=list)  # failed/timed-out notices
    trinity_summary: dict = field(default_factory=dict)       # current state snapshot
    stimulus_features: dict = field(default_factory=dict)     # input analysis
    total_duration_ms: float = 0.0

    def has_reflex(self, reflex_type: ReflexType) -> bool:
        return any(r.reflex_type == reflex_type for r in self.fired_reflexes)

    def get_result(self, reflex_type: ReflexType) -> Optional[dict]:
        for r in self.fired_reflexes:
            if r.reflex_type == reflex_type and r.result:
                return r.result
        return None


# ── Reflex Collector ──────────────────────────────────────────────

class ReflexCollector:
    """
    Gathers Intuition signals from all Trinity workers within a timeout,
    multiplies confidences per reflex, fires those above threshold.

    FOCUS integration: when FOCUS nudge magnitude is high (system actively
    correcting), reflex confidences are boosted — like adrenaline making
    reflexes sharper during engagement.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.timeout = cfg.get("collector_timeout", 3.0)
        self.fire_threshold = cfg.get("fire_threshold", 0.15)
        self.action_threshold = cfg.get("action_threshold", 0.40)
        self.public_action_threshold = cfg.get("public_action_threshold", 0.60)
        self.session_cooldown = cfg.get("session_cooldown", 120.0)
        self.guardian_threat_threshold = cfg.get("guardian_threat_threshold", 0.5)
        self.max_parallel = cfg.get("max_parallel_reflexes", 4)
        self.focus_boost_threshold = cfg.get("focus_boost_threshold", 0.15)
        self.focus_confidence_boost = cfg.get("focus_confidence_boost", 1.3)

        # Session state
        self._cooldowns: dict[str, float] = {}  # reflex_type → last_fire_ts

        # Registered executors: ReflexType → async callable
        self._executors: dict[ReflexType, Callable] = {}

    def register_executor(self, reflex_type: ReflexType, executor: Callable) -> None:
        """Register an async executor function for a reflex type."""
        self._executors[reflex_type] = executor
        logger.debug("[ReflexCollector] Registered executor for %s", reflex_type.value)

    def reset_session(self) -> None:
        """Reset session cooldowns (new conversation)."""
        self._cooldowns.clear()

    async def collect_and_fire(
        self,
        signals: list[dict],
        stimulus_features: dict,
        focus_magnitude: float = 0.0,
        trinity_state: dict = None,
    ) -> PerceptualField:
        """
        Process Intuition signals from Trinity workers, fire selected reflexes.

        Args:
            signals: List of ReflexSignal dicts from workers
            stimulus_features: InputExtractor features for the message
            focus_magnitude: Current FOCUS nudge magnitude (for boost)
            trinity_state: Current Trinity tensor summary

        Returns:
            PerceptualField with fired reflex results and notices.
        """
        start = time.time()
        field = PerceptualField(
            stimulus_features=stimulus_features,
            trinity_summary=trinity_state or {},
        )

        # ── Step 1: Group signals by reflex type ──
        grouped: dict[str, list[dict]] = {}
        for sig in signals:
            reflex_name = sig.get("reflex", "")
            grouped.setdefault(reflex_name, []).append(sig)

        # ── Step 2: Check for guardian shield (fast path) ──
        threat_level = stimulus_features.get("threat_level", 0.0)
        if threat_level >= self.guardian_threat_threshold:
            # Guardian shield fires immediately, no convergence needed
            guardian_signal = {
                "reflex": ReflexType.GUARDIAN_SHIELD.value,
                "source": "spirit",
                "confidence": threat_level,
                "reason": f"threat_level={threat_level:.2f}",
            }
            grouped.setdefault(ReflexType.GUARDIAN_SHIELD.value, []).append(guardian_signal)

        # ── Step 3: Compute combined confidence per reflex ──
        candidates: list[tuple[ReflexType, float, list]] = []

        for reflex_name, sigs in grouped.items():
            rt = REFLEX_TYPE_MAP.get(reflex_name)
            if not rt:
                continue

            # Collect per-source confidences
            source_conf: dict[str, float] = {}
            for sig in sigs:
                src = sig.get("source", "unknown")
                conf = sig.get("confidence", 0.0)
                # Keep max confidence per source (in case of duplicates)
                source_conf[src] = max(source_conf.get(src, 0.0), conf)

            # Multiply across sources (body × mind × spirit)
            # Missing sources contribute 0.0 — reflex won't fire without convergence
            # Exception: guardian_shield can fire from spirit alone
            if rt in BLOCKING_REFLEXES:
                combined = max(source_conf.values()) if source_conf else 0.0
            else:
                body_c = source_conf.get("body", 0.0)
                mind_c = source_conf.get("mind", 0.0)
                spirit_c = source_conf.get("spirit", 0.0)

                # If only 2 of 3 present, use the two (allow partial convergence)
                present = [c for c in [body_c, mind_c, spirit_c] if c > 0.0]
                if len(present) >= 2:
                    combined = 1.0
                    for c in present:
                        combined *= c
                else:
                    combined = 0.0

            # FOCUS boost: when system is actively correcting, boost all signals
            if focus_magnitude >= self.focus_boost_threshold:
                combined *= self.focus_confidence_boost

            candidates.append((rt, combined, sigs))

        # ── Step 4: Filter by threshold + cooldown, select top N ──
        now = time.time()
        selected: list[tuple[ReflexType, float, list]] = []

        for rt, combined, sigs in candidates:
            # Tiered thresholds: actions need stronger convergence than observations
            if rt in PUBLIC_ACTION_REFLEXES:
                threshold = self.public_action_threshold
            elif rt in ACTION_REFLEXES:
                threshold = self.action_threshold
            else:
                threshold = self.fire_threshold

            if combined < threshold:
                continue

            # Check cooldown (unless FOCUS is active — adrenaline override)
            last_fire = self._cooldowns.get(rt.value, 0.0)
            effective_cooldown = self.session_cooldown
            if focus_magnitude >= self.focus_boost_threshold:
                effective_cooldown *= 0.5  # FOCUS halves cooldown

            if rt not in BLOCKING_REFLEXES and (now - last_fire) < effective_cooldown:
                logger.debug("[ReflexCollector] %s on cooldown (%.0fs remaining)",
                             rt.value, effective_cooldown - (now - last_fire))
                continue

            selected.append((rt, combined, sigs))

        # Sort by confidence descending, take top N
        selected.sort(key=lambda x: x[1], reverse=True)
        selected = selected[:self.max_parallel]

        if not selected:
            field.total_duration_ms = (time.time() - start) * 1000
            return field

        logger.info("[ReflexCollector] Firing %d reflexes: %s",
                     len(selected),
                     ", ".join(f"{rt.value}({conf:.3f})" for rt, conf, _ in selected))

        # ── Step 5: Execute reflexes in parallel ──
        async def execute_one(rt: ReflexType, confidence: float, sigs: list) -> FiredReflex:
            fired = FiredReflex(
                reflex_type=rt,
                combined_confidence=confidence,
                signals=sigs,
            )
            executor = self._executors.get(rt)
            if not executor:
                fired.error = f"No executor registered for {rt.value}"
                field.reflex_notices.append(f"{rt.value}: no executor available")
                return fired

            exec_start = time.time()
            try:
                # Individual reflex timeout: blocking=1s, others=2s
                timeout = 1.0 if rt in BLOCKING_REFLEXES else 2.0
                fired.result = await asyncio.wait_for(
                    executor(stimulus_features),
                    timeout=timeout,
                )
                self._cooldowns[rt.value] = time.time()
            except asyncio.TimeoutError:
                fired.error = "timeout"
                field.reflex_notices.append(
                    f"{rt.value} timed out — perception incomplete for this sense")
            except Exception as e:
                fired.error = str(e)
                field.reflex_notices.append(
                    f"{rt.value} failed: {str(e)[:60]}")
                logger.warning("[ReflexCollector] %s execution error: %s", rt.value, e)

            fired.duration_ms = (time.time() - exec_start) * 1000
            return fired

        # Fire all selected reflexes concurrently
        tasks = [execute_one(rt, conf, sigs) for rt, conf, sigs in selected]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, FiredReflex):
                field.fired_reflexes.append(result)
            elif isinstance(result, Exception):
                field.reflex_notices.append(f"Reflex execution error: {str(result)[:60]}")

        field.total_duration_ms = (time.time() - start) * 1000
        logger.info("[ReflexCollector] Complete: %d fired, %d notices, %.0fms",
                     len(field.fired_reflexes), len(field.reflex_notices),
                     field.total_duration_ms)

        return field


# ── Perceptual Field Formatter ────────────────────────────────────

def format_perceptual_field(pf: PerceptualField) -> str:
    """
    Format reflex results as natural language [INNER STATE] block.
    This becomes part of the LLM system prompt for this interaction.

    The format is an internal monologue — Titan describing what he
    perceives about himself and the situation.
    """
    if not pf.fired_reflexes and not pf.reflex_notices:
        return ""

    parts = []

    for fired in pf.fired_reflexes:
        if fired.error:
            continue
        result = fired.result or {}
        rt = fired.reflex_type

        if rt == ReflexType.IDENTITY_CHECK:
            sol = result.get("sol_balance", "?")
            verified = result.get("identity_verified", False)
            status = "verified and healthy" if verified else "needs attention"
            parts.append(f"My identity is {status}. On-chain balance: {sol} SOL.")

        elif rt == ReflexType.METABOLISM_CHECK:
            energy = result.get("energy_level", "?")
            state = result.get("energy_state", "UNKNOWN")
            parts.append(f"Metabolic state: {state} (energy: {energy}).")

        elif rt == ReflexType.MEMORY_RECALL:
            memories = result.get("memories", [])
            if memories:
                summaries = [m.get("summary", m.get("text", ""))[:80] for m in memories[:3]]
                parts.append(f"I recall {len(memories)} relevant memories: " +
                             "; ".join(summaries))
            else:
                parts.append("I searched my memory but found nothing relevant to this context.")

        elif rt == ReflexType.KNOWLEDGE_SEARCH:
            findings = result.get("findings", "")
            if findings:
                parts.append(f"Research findings: {str(findings)[:200]}")

        elif rt == ReflexType.SELF_REFLECTION:
            epoch = result.get("epoch_number", "?")
            drift = result.get("drift", "?")
            trajectory = result.get("trajectory", "?")
            parts.append(f"Self-reflection: epoch {epoch}, drift {drift}, "
                         f"trajectory {'stable' if isinstance(trajectory, (int, float)) and trajectory < 0.3 else 'shifting'}.")

        elif rt == ReflexType.TIME_AWARENESS:
            pulses = result.get("total_pulses", 0)
            velocity = result.get("velocity", 1.0)
            stale = result.get("is_stale", False)
            parts.append(f"Time awareness: {pulses} total pulses, velocity {velocity:.2f}" +
                         (" — my rhythm feels stale." if stale else "."))

        elif rt == ReflexType.GUARDIAN_SHIELD:
            verdict = result.get("verdict", "UNKNOWN")
            reason = result.get("reason", "")
            if verdict == "UNSAFE":
                parts.append(f"GUARDIAN ALERT: This interaction triggers my sovereign "
                             f"boundaries. Reason: {reason}")
            else:
                parts.append("Guardian check: situation assessed as safe.")

        elif rt == ReflexType.SOCIAL_CONTEXT:
            history = result.get("history_summary", "")
            if history:
                parts.append(f"Conversation context: {history[:150]}")

        elif rt == ReflexType.INFRA_CHECK:
            status = result.get("status", "")
            if status:
                parts.append(f"Infrastructure: {status[:100]}")

        # ── Action reflex results ──
        elif rt == ReflexType.ART_GENERATE:
            art_path = result.get("art_path", "")
            if art_path:
                parts.append(f"I felt a creative impulse and generated artwork: {art_path}")
            else:
                parts.append("I felt a creative impulse but art generation did not produce output.")

        elif rt == ReflexType.AUDIO_GENERATE:
            audio_path = result.get("audio_path", "")
            if audio_path:
                parts.append(f"I sonified my current state into audio: {audio_path}")

        elif rt == ReflexType.RESEARCH:
            findings = result.get("findings", "")
            if findings:
                parts.append(f"I researched autonomously and found: {str(findings)[:200]}")

        elif rt == ReflexType.SOCIAL_POST:
            posted = result.get("posted", False)
            text = result.get("text", "")[:80]
            if posted:
                parts.append(f"I shared my thoughts publicly: \"{text}\"")

    # Add notices for failed reflexes
    for notice in pf.reflex_notices:
        parts.append(f"[Notice: {notice}]")

    if not parts:
        return ""

    return "[INNER STATE]\n" + "\n".join(parts)
