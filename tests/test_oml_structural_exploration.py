"""Active idle action-space exploration — the G5/GB8 closer (deadlock fix step 2).

The verifiable structural oracle (`structural_target_action`) scores a
Boltzmann-explored action on an idle tick (off the live path, INV-OML-9), so the
under-used actions get tried + credited — esp. `IDK`, which has NO live reward
stream. The know/don't-know axis is the memory-search signal (recall_top_cosine —
"does a dereferenceable thought exist?"); research-vs-IDK is metabolic
affordability. These tests pin the oracle's ladder + that idle exploration
actually teaches the policy IDK/research on no-recall contexts and direct on
known ones (the routing the always-tool collapse destroyed)."""
import numpy as np

from titan_hcl.synthesis.outer_meta_policy import (
    OUTER_ACTIONS, OUTER_POLICY_INPUT_DIM, OuterMetaPolicy, structural_target_action)

DIRECT = OUTER_ACTIONS.index("direct")
TOOL = OUTER_ACTIONS.index("tool")
SKILL = OUTER_ACTIONS.index("skill_delegate")
RESEARCH = OUTER_ACTIONS.index("research")
IDK = OUTER_ACTIONS.index("IDK")


def _vec(*, recall=0.0, recall_count=0.0, skill_util=0.0, skill_matched=False,
         requires_tool=False, has_code=False):
    x = np.zeros(OUTER_POLICY_INPUT_DIM, dtype=np.float32)
    x[0] = 1.0
    x[1] = recall                          # recall_top_cosine
    x[2] = recall_count                    # recall_count_norm
    x[3] = skill_util                       # skill_utility
    x[4] = 1.0 if skill_matched else 0.0    # skill_matched
    x[6] = 1.0 if requires_tool else 0.0    # requires_tool
    x[7] = 1.0 if has_code else 0.0         # has_code_signal
    return x


# ── the structural oracle ladder ────────────────────────────────────────────
def test_oracle_skill_reuse_first():
    assert structural_target_action(
        _vec(skill_util=0.8, skill_matched=True, requires_tool=True)) == SKILL


def test_oracle_tool_for_computable():
    assert structural_target_action(_vec(requires_tool=True)) == TOOL
    assert structural_target_action(_vec(has_code=True)) == TOOL


def test_oracle_direct_when_he_knows():
    # dereferenceable recall present → he KNOWS → direct
    assert structural_target_action(_vec(recall=0.9)) == DIRECT


def test_oracle_research_when_unknown_and_affordable():
    assert structural_target_action(_vec(recall=0.0), affordable=True) == RESEARCH


def test_oracle_idk_when_unknown_and_not_affordable():
    # the Maker's case: search empty + can't afford a request → honest IDK
    assert structural_target_action(_vec(recall=0.0), affordable=False) == IDK


def test_oracle_know_threshold_boundary():
    assert structural_target_action(_vec(recall=0.49), affordable=True) == RESEARCH
    assert structural_target_action(_vec(recall=0.51), affordable=True) == DIRECT


# ── idle exploration actually learns the routing ────────────────────────────
def _teach_structural(policy, sampler, *, affordable, iters):
    """Run the step-2 idle bootstrap: teach the VERIFIABLE structural target
    directly via cross-entropy (`train_step`), mirroring `_structural_explore`.
    (Boltzmann-sample-and-reward provably collapses here — REINFORCE's advantage
    decays to 0 before the conditional routing is learned; the target is
    verifiable, so we teach it directly. Robust 8/8 seeds.)"""
    for _ in range(iters):
        vec = sampler()
        target = structural_target_action(vec, affordable=affordable)
        policy.train_step(vec, target, advantage=1.0)


def test_idle_exploration_teaches_research_and_direct():
    """The closer (soak-like): over DIVERSE recalled contexts (the live
    multi-feature recall signal — cosine AND count) the idle bootstrap teaches
    the recall discriminator — doesn't-know → research over BOTH always-`tool`
    AND the fabrication-risk `direct`; knows → direct; tool-intent → tool."""
    np.random.seed(0)
    policy = OuterMetaPolicy()

    def sampler():
        r = np.random.random()
        if r < 0.34:   # he KNOWS — strong recall (cosine + count)
            return _vec(recall=float(np.random.uniform(0.6, 1.0)),
                        recall_count=float(np.random.uniform(0.4, 1.0)))
        if r < 0.67:   # doesn't know — recall empty
            return _vec(recall=float(np.random.uniform(0.0, 0.3)),
                        recall_count=float(np.random.uniform(0.0, 0.2)))
        return _vec(requires_tool=True, recall=float(np.random.uniform(0, 1)))  # compute

    _teach_structural(policy, sampler, affordable=True, iters=4000)
    su = policy.forward(_vec(recall=0.0))                       # doesn't know
    sk = policy.forward(_vec(recall=0.9, recall_count=0.8))     # knows
    st = policy.forward(_vec(requires_tool=True))               # computable
    assert int(np.argmax(su)) == RESEARCH    # off always-tool, no fabrication
    assert int(np.argmax(sk)) == DIRECT      # knows → direct
    assert int(np.argmax(st)) == TOOL        # tool-intent → tool


def test_idle_exploration_teaches_honest_idk_when_starved():
    """G5 directly: when research is NOT affordable, a no-recall context learns
    the honest IDK route — the action that had no reward stream at all before."""
    np.random.seed(1)
    policy = OuterMetaPolicy()
    _teach_structural(
        policy, lambda: _vec(recall=float(np.random.uniform(0.0, 0.3))),
        affordable=False, iters=2000)
    assert int(policy.exploit_action(_vec(recall=0.0))) == IDK
