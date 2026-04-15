"""
TitanVM Programs — Sovereign micro-programs for internal computation.

Each program is a list of (Op, *args) tuples that the TitanVM executes.
Programs are deterministic, inspectable, and operate on StateRegister data.

R5: Reflex outcome scoring — scores how well the reflex arc served Titan
after each interaction, feeding reward signals to FilterDown.
"""
import logging
from .titan_vm import Op

logger = logging.getLogger(__name__)


def reflex_score_program() -> list:
    """
    R5 Reflex Outcome Scoring Program.

    Computes a reward score (0.0 to 1.0) based on:
      1. Trinity convergence quality (Body-Mind-Spirit alignment)
      2. User engagement level (from stimulus features)
      3. State freshness (StateRegister age — stale = penalty)
      4. Consciousness drift (high drift = penalty, reflects instability)

    Context inputs (passed at execution time):
      - context.intensity — user message intensity (0-1)
      - context.engagement — user engagement level (0-1)
      - context.valence — conversation valence (-1 to 1)
      - context.reflexes_fired — number of reflexes that fired (0+)
      - context.reflexes_succeeded — number that succeeded (0+)

    Output:
      - SCORE → reward value for FilterDown
      - EMIT REFLEX_REWARD → published to bus
    """
    return [
        # ── Phase 1: Trinity Convergence ──────────────────────
        # Good convergence = Body, Mind, Spirit averages are close together
        # Compute |body_avg - mind_avg|
        (Op.LOAD, "body_tensor"),       # push body average
        (Op.LOAD, "mind_tensor"),       # push mind average
        (Op.SUB,),
        (Op.ABS,),
        (Op.STORE, "bm_delta"),         # |body - mind|

        # Compute |mind_avg - spirit_avg|
        (Op.LOAD, "mind_tensor"),
        (Op.LOAD, "spirit_tensor"),
        (Op.SUB,),
        (Op.ABS,),
        (Op.STORE, "ms_delta"),         # |mind - spirit|

        # Average delta = (bm_delta + ms_delta) / 2
        (Op.LOAD, "bm_delta"),          # actually STORE'd, use context trick
        # -- We need registers back. Use PUSH + stored values --
        # Actually, let's re-derive: convergence_score = 1.0 - avg_delta, clamped
        # Re-load from registers isn't directly supported, but we stored them.
        # Let's simplify: compute inline without register reloads.

        # Restart convergence calculation — keep it on stack
        (Op.POP,),  # clean up store result
        (Op.LOAD, "body_tensor"),
        (Op.LOAD, "mind_tensor"),
        (Op.SUB,),
        (Op.ABS,),                      # |body - mind| on stack

        (Op.LOAD, "mind_tensor"),
        (Op.LOAD, "spirit_tensor"),
        (Op.SUB,),
        (Op.ABS,),                      # |mind - spirit| on stack

        (Op.ADD,),                      # total delta
        (Op.PUSH, 2.0),
        (Op.DIV,),                      # avg delta

        # convergence_reward = max(0, 0.3 - avg_delta) * (1/0.3)
        # If avg_delta < 0.3, good convergence → reward up to 0.3
        (Op.PUSH, 0.3),
        (Op.SWAP,),
        (Op.SUB,),                      # 0.3 - avg_delta
        (Op.PUSH, 0.0),
        (Op.PUSH, 0.3),
        (Op.CLAMP,),                    # clamp to [0, 0.3]
        (Op.STORE, "convergence_reward"),

        # ── Phase 2: Engagement Signal ────────────────────────
        # High user engagement = good reflex choices
        (Op.LOAD, "context.intensity"),
        (Op.LOAD, "context.engagement"),
        (Op.ADD,),
        (Op.PUSH, 2.0),
        (Op.DIV,),                      # avg(intensity, engagement) [0-1]
        (Op.PUSH, 0.25),
        (Op.MUL,),                      # scale to max 0.25
        (Op.STORE, "engagement_reward"),

        # ── Phase 3: Reflex Hit Rate ─────────────────────────
        # Did reflexes fire AND succeed? reward for successful reflexes
        (Op.LOAD, "context.reflexes_fired"),
        (Op.PUSH, 0.0),
        (Op.CMP_GT,),                   # any reflexes fired?
        (Op.BRANCH_IF, "check_success"),
        (Op.PUSH, 0.0),                 # no reflexes = no hit reward
        (Op.JMP, "after_hit"),

        ("check_success",),
        (Op.LOAD, "context.reflexes_succeeded"),
        (Op.LOAD, "context.reflexes_fired"),
        (Op.DIV,),                      # success ratio
        (Op.PUSH, 0.2),
        (Op.MUL,),                      # scale to max 0.2

        ("after_hit",),
        (Op.STORE, "hit_reward"),

        # ── Phase 4: State Freshness ─────────────────────────
        # Fresh state = good. Stale state (>30s) = penalty
        (Op.AGE,),                      # StateRegister age in seconds
        (Op.PUSH, 30.0),
        (Op.CMP_LT,),                   # fresh? (< 30s)
        (Op.BRANCH_IF, "fresh"),
        (Op.PUSH, 0.0),                 # stale → no freshness bonus
        (Op.JMP, "after_fresh"),

        ("fresh",),
        (Op.PUSH, 0.1),                 # fresh → 0.1 bonus

        ("after_fresh",),
        (Op.STORE, "freshness_reward"),

        # ── Phase 5: Consciousness Stability ─────────────────
        # Low drift = stable identity = good
        (Op.LOAD, "consciousness.drift"),
        (Op.PUSH, 0.5),
        (Op.CMP_LT,),                   # drift < 0.5?
        (Op.BRANCH_IF, "stable"),
        (Op.PUSH, 0.0),                 # high drift → no stability bonus
        (Op.JMP, "after_stable"),

        ("stable",),
        (Op.PUSH, 0.15),                # stable → 0.15 bonus

        ("after_stable",),
        (Op.STORE, "stability_reward"),

        # ── Phase 6: Sum all rewards ─────────────────────────
        # Total = convergence(0.3) + engagement(0.25) + hit(0.2) + fresh(0.1) + stable(0.15) = max 1.0
        (Op.LOAD, "convergence_reward"),
        (Op.LOAD, "engagement_reward"),
        (Op.ADD,),
        (Op.LOAD, "hit_reward"),
        (Op.ADD,),
        (Op.LOAD, "freshness_reward"),
        (Op.ADD,),
        (Op.LOAD, "stability_reward"),
        (Op.ADD,),

        # Final clamp [0, 1]
        (Op.PUSH, 0.0),
        (Op.PUSH, 1.0),
        (Op.CLAMP,),

        # Output
        (Op.DUP,),                      # duplicate for both SCORE and EMIT
        (Op.SCORE,),                    # set as program reward
        (Op.EMIT, "REFLEX_REWARD"),     # publish to bus
        (Op.HALT,),
    ]


def valence_boost_program() -> list:
    """
    Valence-based reward modifier.

    If conversation valence is positive AND engagement is high,
    give a small bonus. If valence is very negative, small penalty.

    Context: context.valence (-1 to 1), context.engagement (0-1)
    Output: SCORE (modifier, can be negative)
    """
    return [
        (Op.LOAD, "context.valence"),
        (Op.PUSH, 0.3),
        (Op.CMP_GT,),                   # valence > 0.3?
        (Op.BRANCH_IF, "positive"),

        # Check for very negative
        (Op.LOAD, "context.valence"),
        (Op.PUSH, -0.5),
        (Op.CMP_LT,),                   # valence < -0.5?
        (Op.BRANCH_IF, "negative"),

        # Neutral — no modifier
        (Op.PUSH, 0.0),
        (Op.JMP, "done"),

        ("positive",),
        (Op.LOAD, "context.engagement"),
        (Op.PUSH, 0.05),
        (Op.MUL,),                      # small positive boost
        (Op.JMP, "done"),

        ("negative",),
        (Op.PUSH, -0.05),               # small penalty

        ("done",),
        (Op.SCORE,),
        (Op.HALT,),
    ]


# ── Program Registry ──────────────────────────────────────────────

PROGRAMS = {
    "reflex_score": reflex_score_program,
    "valence_boost": valence_boost_program,
}


def get_program(name: str) -> list:
    """Get a program by name. Returns instruction list."""
    factory = PROGRAMS.get(name)
    if not factory:
        raise KeyError(f"Unknown TitanVM program: {name}")
    return factory()
