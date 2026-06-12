"""Fix 1 (§24.10) — correctness-aware credit.

When the OuterMetaPolicy chose a NON-tool action whose numeric claim the OVG
POST-backstop had to SALVAGE (verdict "false" → the chosen action's answer was
WRONG; `coding_sandbox` computed the right value), the reward must follow the
verification OUTCOME, NOT the turn-judge's QUALITY score: credit `tool` (+1) AND
penalize the chosen action (−1). These tests pin the producer payloads, the
consumer training effect, the behavioral routing shift, and the `oracle` source
authority — the live T3 soak proved the judge-quality reward alone collapses the
policy to `direct` (2026-06-12)."""
import numpy as np

from titan_hcl.modules.self_learning_worker import (
    _REWARD_SOURCE_RANK, _SelfLearningStore, _cfg, _handle_reward)
from titan_hcl.modules.synthesis_worker import _correctness_aware_reward_msgs
from titan_hcl.synthesis.outer_meta_policy import (
    OUTER_ACTIONS, OuterFeatures, OuterMetaPolicy)

TOOL = OUTER_ACTIONS.index("tool")
DIRECT = OUTER_ACTIONS.index("direct")


class _Q:
    def __init__(self):
        self.items = []

    def put(self, msg):
        self.items.append(msg)


def _feat(**kw):
    return OuterFeatures(**kw).to_vector().tolist()


# ── producer: the two payloads ──────────────────────────────────────────────
def test_producer_emits_tool_credit_and_chosen_penalty():
    feats = _feat(has_code_signal=True, requires_tool=True)
    msgs = _correctness_aware_reward_msgs(
        decision_feats=feats, policy_action=DIRECT, tool_action=TOOL,
        goal_class="general-compute", oracle_id="ora1", parent_tool_call_tx="tx1")
    assert len(msgs) == 2
    credit, penalty = msgs
    # tool credited +1
    assert credit["action"] == TOOL and credit["reward"] == 1.0
    assert credit["source"] == "oracle" and credit["goal_class"] == "general-compute"
    # the chosen (direct) penalized −1
    assert penalty["action"] == DIRECT and penalty["reward"] == -1.0
    assert penalty["source"] == "oracle"
    # both carry the decision features (so the policy trains on THIS prompt-shape)
    assert credit["features"] == feats and penalty["features"] == feats


# ── consumer: each payload trains the right action ──────────────────────────
def test_consumer_credit_trains_tool_penalty_trains_chosen(tmp_path):
    store = _SelfLearningStore(path=str(tmp_path / "sl.duckdb"))
    policy = OuterMetaPolicy(lr=0.05)
    feats = _feat(has_code_signal=True, requires_tool=True)
    msgs = _correctness_aware_reward_msgs(
        decision_feats=feats, policy_action=DIRECT, tool_action=TOOL,
        goal_class="general-compute", oracle_id="", parent_tool_call_tx="tx1")
    updates0 = policy.total_updates
    for m in msgs:
        assert _handle_reward(m, store, policy, None, _cfg({}), _Q(),
                              "self_learning") is True
    # both rewards trained (direct path — features+action present, no join)
    assert policy.total_updates == updates0 + 2
    # two reward tuples recorded (tool +1, direct −1)
    tuples = store.recent_reward_tuples(10)
    assert len(tuples) == 2


def test_routing_shifts_toward_tool_after_repeated_salvage(tmp_path):
    """The behavioral outcome: a feature-shape the policy keeps routing wrong
    (its answer salvaged) is pushed toward `tool` over repeated corrections —
    even starting from a policy that argmaxes `direct` on it."""
    store = _SelfLearningStore(path=str(tmp_path / "sl.duckdb"))
    np.random.seed(0)
    policy = OuterMetaPolicy(lr=0.05)
    feats = _feat(requires_tool=False, has_code_signal=False, recall_top_cosine=0.7)
    # force a starting bias toward direct on this vector
    policy.seed_prior(DIRECT, strength=3.0)
    assert int(policy.exploit_action(np.asarray(feats, dtype=np.float32))) == DIRECT
    cfg = _cfg({})
    for _ in range(40):
        for m in _correctness_aware_reward_msgs(
                decision_feats=feats, policy_action=DIRECT, tool_action=TOOL,
                goal_class="general-compute", oracle_id="", parent_tool_call_tx="tx"):
            _handle_reward(m, store, policy, None, cfg, _Q(), "self_learning")
    s = policy.forward(np.asarray(feats, dtype=np.float32))
    # tool now out-scores the (penalized) chosen action for this shape
    assert s[TOOL] > s[DIRECT], f"tool={s[TOOL]:.2f} !> direct={s[DIRECT]:.2f}"


# ── the oracle source is the top authority ──────────────────────────────────
def test_oracle_outranks_quality_judge_and_human():
    # objective numeric correctness > subjective quality/preference
    assert _REWARD_SOURCE_RANK["oracle"] > _REWARD_SOURCE_RANK["llm_judge"]
    assert _REWARD_SOURCE_RANK["oracle"] > _REWARD_SOURCE_RANK["maker"]
    assert _REWARD_SOURCE_RANK["oracle"] == max(_REWARD_SOURCE_RANK.values())
