"""IDK-oracle — P6 (RFP_emergent_mastery_curriculum §7.P6; ARCHITECTURE_mastery_leveling.md §4).

A live per-turn CORRECTNESS verdict for the `IDK` routing action. It promotes the
EXISTING structural IDK oracle (`outer_meta_policy.structural_target_action`: "does
NOT know" ⇔ `recall_top_cosine < know_threshold`) to a per-turn reward signal — so
`IDK` becomes a recall-VERIFIABLE action, NOT a quality-judged one (INV-MC-5 /
INV-MC-8). A quality judge scoring IDK is a single-action collapse vector
(`reference_oml_reward_must_be_correctness_aware_not_quality_aware`); the honest
"I don't know" is scored on whether memory was *genuinely* empty, never assumed.

  • verified  (recall genuinely empty)  → reward slightly > a damped `direct`
              (honest hard self-knowledge — `idk_verified_reward`).
  • unverified (recall was strong — he KNEW and declined) → penalize
              (`idk_unverified_penalty`).

Pure + deterministic (no I/O): the recall signal is already in the decision's
feature vector (`φ[1] = recall_top_cosine`). The gap-fill (fire a DEFERRED research
IMPULSE on a verified-IDK so the unknown becomes known) is wired at the call site
(the agno PreHook), not here — this module is just the verdict kernel.
"""
from __future__ import annotations


def idk_verdict(*, recall_top: float, know_threshold: float,
                verified_reward: float, unverified_penalty: float) -> dict:
    """Return `{verified: bool, reward: float}` for an `IDK` turn.

    `verified` = recall was genuinely empty (`recall_top < know_threshold`) → the
    "I don't know" was honest (INV-MC-5). Otherwise a dereferenceable memory existed
    and the agent still declined → unverified → penalized."""
    verified = float(recall_top) < float(know_threshold)
    reward = float(verified_reward) if verified else float(unverified_penalty)
    return {"verified": verified, "reward": reward}
