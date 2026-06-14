"""Grounded execution-mode router — EEL Pillar 0 (RFP_emergent_experience_learning §7.0).

Replaces the gatekeeper's single 2-month-old IQL scalar (`advantage = q − v`,
`gatekeeper.py:188`) with a decision made from Titan's ACTUAL cognitive state,
read over the spine. This module is the PURE decision core (0a): no I/O, no
plugin/torch deps, fully unit-testable. The agno-side PreHook (0b) assembles the
`GroundedReadout` from already-computed spine reads and calls `grounded_route`.

Mechanic (locked §1 Pillar 0 + §7.0): a two-stage grounded cascade —
  (1) `task_type` selects the LANE (computational → oracle, procedural → skill,
      informational → memory, else → conversational);
  (2) the grounded STRENGTHS within that lane pick the mode;
  (3) IQL refines as a re-ranker / veto OVER the grounded candidate.

Stage 3 is PASSIVE in 0a/0b: `iql_advantage` is carried on the decision for
telemetry + recording but never changes the mode. The veto is activated in 0c
(retrain scholar on mode-outcomes), gated by `iql_veto_margin`.

Grounding stays primary (INV-EEL-7); IQL never gates blind. All inputs are O(1)
spine reads assembled by the caller — no bulk op on the chat path (INV-EEL-1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# ── Execution modes ──────────────────────────────────────────────────────────
# The first four MUST match the strings the existing agno_hooks dispatch already
# keys on (`agno_hooks.py` Sovereign/Collaborative/STATE_NEED_RESEARCH/Shadow);
# the last two are the modes Pillar 0 adds to the dispatch.
MODE_SOVEREIGN = "Sovereign"               # answer from substrate; light LLM narration
MODE_COLLABORATIVE = "Collaborative"       # recall + LLM fills the gaps
MODE_RESEARCH = "STATE_NEED_RESEARCH"      # Pillar A — "I don't know, looking it up"
MODE_SHADOW = "Shadow"                     # honest "I don't know" (no affordable path)
MODE_SKILL_DELEGATE = "Skill-Delegate"     # delegate a verified skill; oracle-verify
MODE_TOOL_ORACLE = "Tool/Oracle"           # deterministic coding_sandbox / oracle

# Lanes (telemetry labels; the lane that produced the mode).
LANE_COMPUTATIONAL = "computational"
LANE_SKILL = "skill"
LANE_INFORMATIONAL = "informational"
LANE_CONVERSATIONAL = "conversational"


@dataclass(frozen=True)
class GroundedReadout:
    """Titan's cognitive state for one reasoning turn — assembled by the PreHook
    (0b) from reads that already exist on the spine. All cheap / O(1).

    recall_score        top `composite_score` of the memories already recalled in
                        the PreHook (`agno_hooks.py:663–766`) — reused, not
                        re-computed (INV-EEL-1). 0.0 when nothing recalled.
    engram_ground       EngramStore reduced-groundedness scalar for the matched
                        Engram (`synthesis/engram_store.py`). 0.0 when none.
    skill_utility       utility of the top `match_procedural_skill` (already
                        delegate-gated by the synthesis reader) or None when no
                        match. Stays None until B1 populates skills (locked
                        sequencing) — the skill-lane degrades gracefully.
    requires_tool       `detect_tool_intent().requires_tool` — a deterministic
                        compute/verify intent (the computational lane).
    is_informational    `gatekeeper._is_informational_query(prompt)` — needs
                        real-world / current information (the memory lane).
    can_afford_research the metabolic gate: is there SOL budget for research /
                        metered oracles? (metabolic-governance gated — NOT
                        INV-EEL-7.) Computed by the caller from the live
                        metabolic state; defaults True for pure-logic tests.
    """

    recall_score: float = 0.0
    engram_ground: float = 0.0
    skill_utility: Optional[float] = None
    requires_tool: bool = False
    is_informational: bool = False
    can_afford_research: bool = True


@dataclass(frozen=True)
class RouterThresholds:
    """Bootstrap thresholds (titan_params.toml [gatekeeper.grounded_router]).
    Conservative by design — IQL / `time_cost` supersede over time (INV-EEL-4).
    Defaults mirror the toml so pure-logic tests need no file."""

    recall_known_floor: float = 0.65    # strong memory → Sovereign
    recall_present_floor: float = 0.30  # partial memory → Collaborative
    engram_ground_floor: float = 0.30   # grounded Engram counts as strong substrate
    skill_promote_floor: float = 0.70   # verified skill utility → Skill-Delegate (B1)
    iql_veto_margin: float = 0.25       # |advantage| beyond this may veto (0c only)


@dataclass(frozen=True)
class RouterDecision:
    """The routed execution mode + a telemetry trace.

    `iql_advantage` is the refinement signal from the preserved recorder RPC
    (`proxies/rl_proxy.py:185`). In 0a/0b it is recorded but does NOT change
    `mode` (passive); 0c consumes it via `iql_veto_margin`."""

    mode: str
    lane: str
    reason: str
    iql_advantage: Optional[float] = None


def grounded_route(
    readout: GroundedReadout,
    thresholds: RouterThresholds,
    *,
    iql_advantage: Optional[float] = None,
) -> RouterDecision:
    """Decide the execution mode from the grounded cognitive readout.

    Pure function — same inputs always yield the same decision. The lane
    precedence encodes the locked §1 combination: a deterministic-verifiable
    task takes the oracle lane first; a verified skill (procedural) wins over
    the memory lane (the §1 ambiguous-case resolution); informational falls to
    memory-then-research; everything else is answered directly (no oracle →
    never skill-scored, §1.0 oracle roster)."""
    r = readout
    t = thresholds

    def _decide(mode: str, lane: str, reason: str) -> RouterDecision:
        return RouterDecision(mode=mode, lane=lane, reason=reason, iql_advantage=iql_advantage)

    has_strong_substrate = (
        r.recall_score >= t.recall_known_floor
        or r.engram_ground >= t.engram_ground_floor
    )

    # Lane 1 — computational / verifiable: the deterministic oracle is the most
    # reliable path, so it wins even over a recall/skill match.
    if r.requires_tool:
        return _decide(MODE_TOOL_ORACLE, LANE_COMPUTATIONAL,
                       "requires_tool → deterministic oracle (coding_sandbox)")

    # Lane 2 — skill / procedural: a verified, delegatable skill exists. Dormant
    # until B1 populates skills (skill_utility stays None → falls through).
    if r.skill_utility is not None and r.skill_utility >= t.skill_promote_floor:
        return _decide(MODE_SKILL_DELEGATE, LANE_SKILL,
                       f"skill utility {r.skill_utility:.2f} ≥ promote_floor "
                       f"{t.skill_promote_floor:.2f} → delegate")

    # Lane 3 — informational / memory: recall (or grounded Engram) strength
    # decides; absence of substrate is what finally opens the research branch.
    if r.is_informational:
        if has_strong_substrate:
            return _decide(MODE_SOVEREIGN, LANE_INFORMATIONAL,
                           f"strong substrate (recall {r.recall_score:.2f} / "
                           f"engram {r.engram_ground:.2f}) → answer from memory")
        if r.recall_score >= t.recall_present_floor:
            return _decide(MODE_COLLABORATIVE, LANE_INFORMATIONAL,
                           f"partial recall {r.recall_score:.2f} ≥ "
                           f"{t.recall_present_floor:.2f} → collaborate")
        if r.can_afford_research:
            return _decide(MODE_RESEARCH, LANE_INFORMATIONAL,
                           "no memory + informational → research (Pillar A)")
        return _decide(MODE_SHADOW, LANE_INFORMATIONAL,
                       "no memory + research unaffordable (metabolic) → honest IDK")

    # Lane 4 — conversational / creative: no objective oracle (§1.0 roster) → answer
    # directly. A strong substrate, when present, is still surfaced in the reason.
    if has_strong_substrate:
        return _decide(MODE_SOVEREIGN, LANE_CONVERSATIONAL,
                       "conversational with strong substrate → sovereign narrate")
    return _decide(MODE_SOVEREIGN, LANE_CONVERSATIONAL,
                   "conversational / creative → direct narrate")


def load_router_thresholds() -> RouterThresholds:
    """Best-effort load of `[gatekeeper.grounded_router]` from titan_params.toml.

    Mirrors `engram_store._load_groundedness_params_from_toml`: returns the
    in-code defaults on any error (file/section missing, parse error) so unit
    tests + cold scenarios keep working. The PreHook calls this once and caches."""
    try:
        try:
            import tomllib  # 3.11+
        except ImportError:  # pragma: no cover - py<3.11 fallback
            import tomli as tomllib  # type: ignore
        # __file__ = titan_hcl/logic/sage/grounded_router.py → titan_hcl/ is 3 up.
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(here, "titan_params.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)
        sub = data.get("gatekeeper", {}).get("grounded_router", {})
        d = RouterThresholds()
        return RouterThresholds(
            recall_known_floor=float(sub.get("recall_known_floor", d.recall_known_floor)),
            recall_present_floor=float(sub.get("recall_present_floor", d.recall_present_floor)),
            engram_ground_floor=float(sub.get("engram_ground_floor", d.engram_ground_floor)),
            skill_promote_floor=float(sub.get("skill_promote_floor", d.skill_promote_floor)),
            iql_veto_margin=float(sub.get("iql_veto_margin", d.iql_veto_margin)),
        )
    except Exception:
        return RouterThresholds()


# ── Readout assembly helpers (torch-free; importable agno-side) ───────────────
# These let the agno PreHook (0b) build a GroundedReadout from signals already on
# the spine without importing the torch-bearing gatekeeper module.

# Single source for the informational classifier — extracted from
# gatekeeper._is_informational_query (which lives in a torch-importing module) so
# the agno-side router AND the recorder-side q−v path share ONE keyword set
# (EEL §7.0 "expose _is_informational_query"; no duplication, no shim).
_INFORMATIONAL_KEYWORDS = frozenset({
    # Time-sensitive data
    "latest", "current", "today", "right now", "recent", "news", "price",
    "predict", "forecast",
    # Market / financial signals
    "market", "sol price", "bitcoin price", "eth price", "token price",
    "trading", "apy", "yield", "tvl",
    # Social pulse
    "trending", "people saying", "sentiment", "what does", "what do people",
    "community",
    # Event-driven
    "just happened", "breaking", "announcement", "launch", "update", "release",
    "upgrade",
})


def is_informational_query(prompt: str) -> bool:
    """True when the prompt needs real-world / real-time data that static memory
    cannot answer reliably (→ the informational lane). Case-insensitive keyword
    match; the single source for both the grounded router (agno-side) and
    `gatekeeper._is_informational_query` (recorder-side q−v path)."""
    if not prompt:
        return False
    lower = prompt.lower()
    return any(kw in lower for kw in _INFORMATIONAL_KEYWORDS)


def recall_score_from_memories(memories) -> float:
    """Recall strength for the readout = the top weight of the memories ALREADY
    recalled in the PreHook (reused, not re-computed — INV-EEL-1). 0.0 when
    nothing was recalled, which is what opens the research branch. Covers both
    recall paths: VCB records expose `effective_weight` (= chain confidence);
    `memory.query` nodes expose `effective_weight` / `mempool_weight`.

    NOTE (2026-06-14): `effective_weight`/`mempool_weight` is an UNBOUNDED chain-
    confidence weight that GROWS with reinforcement (observed live: 1.18 → 1.39,
    still climbing). The consumer feature `recall_top_cosine` is contracted as
    [0,1]; an unbounded value silently saturates it at the clip ceiling → the OML
    policy reads "knows everything" → collapses to `direct`. So clamp to [0,1]
    here at the source. (The p2/decision-authority path uses a true EngineRecall
    cosine and is unaffected; this guards the legacy fallback.)"""
    top = 0.0
    for m in (memories or []):
        if isinstance(m, dict):
            w = m.get("effective_weight")
            if w is None:
                w = m.get("mempool_weight", m.get("weight"))
        else:
            w = getattr(m, "effective_weight", None)
        try:
            w = float(w) if w is not None else 0.0
        except (TypeError, ValueError):
            w = 0.0
        if w > top:
            top = w
    return max(0.0, min(1.0, top))
