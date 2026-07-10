"""P8.3 — skill_delegate reuse reward (RFP_titan_research_agent §7.P8.3).

Closes the dead skill_delegate loop: the policy routed skill_delegate 954x/27d but
produced 0 reward tuples ever, because the action fell in a dead zone (excluded from
the non-verifiable stash AND from the tool-only ToolBackstop verdict). A KNOWLEDGE /
ground-concept skill match is an oracle-VERIFIED reuse (the delegate-gate itself is
the oracle, INV-Syn-20) → a SYNCHRONOUS intrinsic reward (weight*match*util) on the
v1.1 direct path (mirrors idk_oracle), so the policy finally learns delegate. Coding
skills (oracle=coding_sandbox) are RESERVED for the Lane-2 sandbox-verdict path so the
two never double-credit.
"""
from titan_hcl import bus
from titan_hcl.modules.agno_hooks import (
    _skill_reuse_cfg, _emit_skill_reuse_reward)
from titan_hcl.modules.self_learning_worker import (
    _REWARD_SOURCE_RANK, _SelfLearningStore, _cfg, _handle_reward)
from titan_hcl.synthesis.outer_meta_policy import (
    OUTER_ACTIONS, OuterFeatures, OuterMetaPolicy)
from titan_hcl.synthesis.recall import RecallResult, EngineRecall

SKILL_DELEGATE = OUTER_ACTIONS.index("skill_delegate")


# ── config gate ─────────────────────────────────────────────────────────────
class _Plugin:
    def __init__(self, full_config=None, bus_obj=None):
        self._full_config = full_config or {}
        self.bus = bus_obj


def test_cfg_enabled_default_and_weight():
    sc = _skill_reuse_cfg(_Plugin({"synthesis": {"self_learning": {}}}))
    assert sc is not None
    assert sc["weight"] == 0.5   # default matches self_learning_worker._DEFAULTS


def test_cfg_disabled_returns_none():
    sc = _skill_reuse_cfg(_Plugin(
        {"synthesis": {"self_learning": {"skill_reuse_reward_enabled": False}}}))
    assert sc is None   # kill-switch off → delegate stays unrewarded (pre-P8.3 dead loop)


def test_cfg_custom_weight():
    sc = _skill_reuse_cfg(_Plugin(
        {"synthesis": {"self_learning": {"curiosity_reuse_weight": 0.8}}}))
    assert sc["weight"] == 0.8


# ── the agno emit: intrinsic reward on the v1.1 direct path ─────────────────
class _FakeBus:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


def _feat(**kw):
    return OuterFeatures(**kw).to_vector().tolist()


def test_emit_rewards_delegate_intrinsic_to_match_quality():
    fb = _FakeBus()
    plugin = _Plugin(bus_obj=fb)
    feats = _feat(skill_matched=1.0, skill_utility=0.8)
    _emit_skill_reuse_reward(plugin, feats, SKILL_DELEGATE, "who am i to you",
                             match_score=0.9, utility=0.8, cfg={"weight": 0.5})
    rewards = [m for m in fb.msgs if m["type"] == bus.SELF_LEARN_REWARD]
    assert len(rewards) == 1
    p = rewards[0]["payload"]
    assert p["action"] == SKILL_DELEGATE
    assert p["source"] == "skill_reuse_oracle"
    # reward is INTRINSIC to the verified match quality, not a flat constant.
    assert abs(p["reward"] - (0.5 * 0.9 * 0.8)) < 1e-9
    assert p["features"] == feats and "parent_tool_call_tx" not in p  # v1.1 direct path


def test_emit_reward_clamps_and_scales():
    fb = _FakeBus()
    _emit_skill_reuse_reward(_Plugin(bus_obj=fb), _feat(), SKILL_DELEGATE, "q",
                             match_score=1.5, utility=1.2, cfg={"weight": 0.5})
    # match/util clamp to [0,1] → reward = 0.5*1*1 = 0.5 (never runs away)
    p = [m for m in fb.msgs if m["type"] == bus.SELF_LEARN_REWARD][0]["payload"]
    assert abs(p["reward"] - 0.5) < 1e-9


def test_emit_zero_when_no_utility():
    fb = _FakeBus()
    _emit_skill_reuse_reward(_Plugin(bus_obj=fb), _feat(), SKILL_DELEGATE, "q",
                             match_score=0.9, utility=0.0, cfg={"weight": 0.5})
    p = [m for m in fb.msgs if m["type"] == bus.SELF_LEARN_REWARD][0]["payload"]
    assert p["reward"] == 0.0


def test_emit_never_raises_without_bus():
    # hot path: a missing bus must never break chat (INV-OML-7)
    _emit_skill_reuse_reward(_Plugin(bus_obj=None), _feat(), SKILL_DELEGATE, "q",
                             match_score=0.9, utility=0.8, cfg={"weight": 0.5})


# ── the reward trains skill_delegate via the v1.1 direct path ───────────────
class _Q:
    def __init__(self):
        self.items = []

    def put(self, msg):
        self.items.append(msg)


def test_skill_reuse_reward_trains_delegate_via_direct_path(tmp_path):
    store = _SelfLearningStore(path=str(tmp_path / "sl.duckdb"))
    policy = OuterMetaPolicy(lr=0.05)
    feats = _feat(skill_matched=1.0, skill_utility=0.8)
    payload = {"features": feats, "action": SKILL_DELEGATE, "reward": 0.36,
               "goal_class": "self-query", "source": "skill_reuse_oracle"}
    trained = _handle_reward(payload, store, policy, None,
                             _cfg({"synthesis": {"self_learning": {}}}),
                             _Q(), "self_learning")
    assert trained is True
    tuples = store.recent_reward_tuples(10)
    assert len(tuples) == 1                       # a delegate experience recorded (was 0 all-time)
    assert int(tuples[0][1]) == SKILL_DELEGATE    # trained the skill_delegate action


def test_skill_reuse_oracle_registered_at_oracle_tier():
    # INV-MC-8 landmine: an unregistered src defaults to rank 0 → dropped on a
    # contended join. skill_reuse_oracle must sit at the oracle (top) tier.
    assert _REWARD_SOURCE_RANK["skill_reuse_oracle"] == _REWARD_SOURCE_RANK["oracle"]
    assert _REWARD_SOURCE_RANK["skill_reuse_oracle"] == max(_REWARD_SOURCE_RANK.values())


# ── the discriminator: RecallResult carries oracle_id (knowledge vs coding split) ──
def test_recallresult_has_oracle_id_default_blank():
    r = RecallResult(tx_hash="x", score=0.5)
    assert r.oracle_id == ""      # non-procedural results carry no oracle


def test_procedural_recall_maps_oracle_id():
    # A stub procedural_reader returning a knowledge-skill row → the mapping must
    # surface oracle_id on the RecallResult so the agno decide can split lanes.
    class _StubProcReader:
        def recall(self, query_text, *, k=5):
            return [{"skill_id": "s1", "match_score": 0.9, "utility_score": 0.8,
                     "cosine_surrogate": 0.9, "name": "ground-concept:self",
                     "oracle_id": "web_api_oracle"}]

    er = EngineRecall(rule_evaluator=None, activation_lookup=lambda ids: {},
                      procedural_reader=_StubProcReader())
    res = er.recall("who am i", granularity="procedural", k=1)
    assert res and res[0].oracle_id == "web_api_oracle"
