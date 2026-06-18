"""P6 — IDK-oracle + gap-fill (RFP_emergent_mastery_curriculum §7.P6).

IDK becomes a recall-VERIFIABLE action (off the quality-judge lane, INV-MC-8/5):
verified-empty recall ⇒ honest IDK rewarded slightly > direct; recall strong ⇒
he KNEW & bailed ⇒ penalized. A verified-empty IDK fires a DEFERRED gap-fill
(CGN_KNOWLEDGE_REQ → knowledge pipeline). The reward travels the v1.1 direct path.
"""
import numpy as np

from titan_hcl import bus
from titan_hcl.synthesis.idk_oracle import idk_verdict
from titan_hcl.modules.agno_hooks import (
    _idk_oracle_cfg, _emit_idk_oracle_reward, _fire_idk_gap_fill)
from titan_hcl.modules.self_learning_worker import (
    _REWARD_SOURCE_RANK, _SelfLearningStore, _cfg, _handle_reward)
from titan_hcl.synthesis.outer_meta_policy import (
    OUTER_ACTIONS, OuterFeatures, OuterMetaPolicy)

IDK = OUTER_ACTIONS.index("IDK")


# ── the verdict kernel ──────────────────────────────────────────────────────
def test_verdict_verified_when_recall_empty():
    v = idk_verdict(recall_top=0.10, know_threshold=0.65,
                    verified_reward=0.15, unverified_penalty=-0.5)
    assert v["verified"] is True and v["reward"] == 0.15


def test_verdict_unverified_when_recall_strong():
    v = idk_verdict(recall_top=0.82, know_threshold=0.65,
                    verified_reward=0.15, unverified_penalty=-0.5)
    assert v["verified"] is False and v["reward"] == -0.5


def test_verdict_boundary_is_unverified():
    # recall_top == know_threshold ⇒ NOT < threshold ⇒ unverified (he had a hit).
    v = idk_verdict(recall_top=0.65, know_threshold=0.65,
                    verified_reward=0.15, unverified_penalty=-0.5)
    assert v["verified"] is False


# ── config gate ─────────────────────────────────────────────────────────────
class _Plugin:
    def __init__(self, full_config=None, bus_obj=None):
        self._full_config = full_config or {}
        self.bus = bus_obj


def test_cfg_enabled_default_and_values():
    oc = _idk_oracle_cfg(_Plugin({"synthesis": {"self_learning": {}}}))
    assert oc is not None
    assert oc["know_threshold"] == 0.65 and oc["verified_reward"] == 0.15
    assert oc["unverified_penalty"] == -0.5


def test_cfg_disabled_returns_none():
    oc = _idk_oracle_cfg(_Plugin(
        {"synthesis": {"self_learning": {"idk_oracle_enabled": False}}}))
    assert oc is None   # flag-off → caller keeps IDK on the quality-judge stash


# ── the agno emit: reward + gap-fill ────────────────────────────────────────
class _FakeBus:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


def _feat(**kw):
    return OuterFeatures(**kw).to_vector().tolist()


def _oc():
    return {"know_threshold": 0.65, "verified_reward": 0.15, "unverified_penalty": -0.5}


def test_emit_verified_idk_rewards_and_fires_gapfill():
    fb = _FakeBus()
    plugin = _Plugin(bus_obj=fb)
    feats = _feat(recall_top_cosine=0.10)   # empty recall → verified
    _emit_idk_oracle_reward(plugin, feats, IDK, "GDP of Tuvalu 1997", "maker", _oc())
    rewards = [m for m in fb.msgs if m["type"] == bus.SELF_LEARN_REWARD]
    gapfills = [m for m in fb.msgs if m["type"] == bus.CGN_KNOWLEDGE_REQ]
    assert len(rewards) == 1
    p = rewards[0]["payload"]
    assert p["action"] == IDK and p["reward"] == 0.15 and p["source"] == "idk_oracle"
    assert p["features"] == feats and "parent_tool_call_tx" not in p  # v1.1 direct path
    assert len(gapfills) == 1                      # verified → deferred research fired
    assert gapfills[0]["payload"]["topic"] == "GDP of Tuvalu 1997"
    assert gapfills[0]["payload"]["requestor"] == "idk_oracle.gap_fill"


def test_emit_unverified_idk_penalizes_and_no_gapfill():
    fb = _FakeBus()
    plugin = _Plugin(bus_obj=fb)
    feats = _feat(recall_top_cosine=0.85)   # strong recall → he knew
    _emit_idk_oracle_reward(plugin, feats, IDK, "what is 2+2", "maker", _oc())
    rewards = [m for m in fb.msgs if m["type"] == bus.SELF_LEARN_REWARD]
    gapfills = [m for m in fb.msgs if m["type"] == bus.CGN_KNOWLEDGE_REQ]
    assert len(rewards) == 1 and rewards[0]["payload"]["reward"] == -0.5
    assert len(gapfills) == 0                      # he knew → no gap-fill


def test_gapfill_skips_empty_topic():
    fb = _FakeBus()
    _fire_idk_gap_fill(_Plugin(bus_obj=fb), "   ")
    assert [m for m in fb.msgs if m["type"] == bus.CGN_KNOWLEDGE_REQ] == []


def test_emit_never_raises_without_bus():
    # hot path: a missing bus must never break chat (INV-OML-7)
    _emit_idk_oracle_reward(_Plugin(bus_obj=None), _feat(), IDK, "q", "u", _oc())


# ── the reward trains the IDK action via the v1.1 direct path ───────────────
class _Q:
    def __init__(self):
        self.items = []

    def put(self, msg):
        self.items.append(msg)


def test_idk_oracle_reward_trains_idk_via_direct_path(tmp_path):
    store = _SelfLearningStore(path=str(tmp_path / "sl.duckdb"))
    policy = OuterMetaPolicy(lr=0.05)
    feats = _feat(recall_top_cosine=0.10)
    # the exact payload _emit_idk_oracle_reward publishes (features+action+reward).
    payload = {"features": feats, "action": IDK, "reward": 0.15,
               "goal_class": "trivia", "source": "idk_oracle"}
    trained = _handle_reward(payload, store, policy, None,
                             _cfg({"synthesis": {"self_learning": {}}}),
                             _Q(), "self_learning")
    assert trained is True
    tuples = store.recent_reward_tuples(10)
    assert len(tuples) == 1                          # IDK experience recorded
    assert int(tuples[0][1]) == IDK                  # trained the IDK action


def test_idk_oracle_registered_in_rank():
    # cold-review landmine: a new src defaults to rank 0 → dropped on a contended
    # join. idk_oracle must be registered at the oracle (top) tier.
    assert _REWARD_SOURCE_RANK["idk_oracle"] == _REWARD_SOURCE_RANK["oracle"]
    assert _REWARD_SOURCE_RANK["idk_oracle"] == max(_REWARD_SOURCE_RANK.values())
