"""P8.1+P8.2 — engagement-independent experience (RFP_emergent_mastery_curriculum §7.P8).

Agency's AUTONOMOUS (no-chat) routing-helper outcomes train the OUTER routing IQL,
correctness-grounded by the TaskCompletionJudge (NOT the quality TurnJudge — INV-MC-8)
with solve-until-correct. The reward rides the v1.1 DIRECT path (features present).
"""
import json

import numpy as np

from titan_hcl import bus
from titan_hcl.synthesis.task_completion_judge import TaskCompletionJudge
from titan_hcl.synthesis.outer_meta_policy import (
    autonomous_features_for_helper, OUTER_ACTIONS, OuterMetaPolicy)
from titan_hcl.modules.agency_worker import _maybe_emit_autonomous_experience
from titan_hcl.modules.self_learning_worker import (
    _REWARD_SOURCE_RANK, _SelfLearningStore, _cfg, _handle_reward)

TOOL = OUTER_ACTIONS.index("tool")
RESEARCH = OUTER_ACTIONS.index("research")


# ── TaskCompletionJudge (distinct from TurnJudge) ───────────────────────────
def _judge(solved, correction="", confidence=1.0):
    def _p(prompt, timeout_s):
        return json.dumps({"solved": solved, "correction": correction,
                           "confidence": confidence})
    return TaskCompletionJudge(llm_provider=_p)


def test_judge_solved():
    out = _judge(True, "", 0.9).judge(problem="compute phi", action="tool",
                                      evidence="phi=1.618")
    assert out["solved"] is True and out["confidence"] == 0.9


def test_judge_not_solved_carries_correction():
    out = _judge(False, "wrong formula").judge(problem="p", action="tool", evidence="e")
    assert out["solved"] is False and out["correction"] == "wrong formula"


def test_judge_none_on_empty_problem_or_evidence():
    assert _judge(True).judge(problem="", action="tool", evidence="e") is None
    assert _judge(True).judge(problem="p", action="tool", evidence="") is None


def test_judge_tolerates_string_bool():
    def _p(prompt, t):
        return '{"solved": "true", "correction": "", "confidence": 0.7}'
    out = TaskCompletionJudge(llm_provider=_p).judge(
        problem="p", action="tool", evidence="e")
    assert out["solved"] is True


def test_judge_is_distinct_module_from_turnjudge():
    # INV-MC-8: the autonomous correctness judge is NOT the quality TurnJudge.
    from titan_hcl.synthesis import task_completion_judge as tcj
    from titan_hcl.synthesis import turn_judge as tj
    assert tcj.JUDGE_PROMPT_TEMPLATE != tj.JUDGE_PROMPT_TEMPLATE
    assert "CORRECTNESS" in tcj.JUDGE_PROMPT_TEMPLATE


# ── the φ-builder ───────────────────────────────────────────────────────────
def test_features_coding_sandbox_is_tool_shaped():
    feats, idx = autonomous_features_for_helper("coding_sandbox")
    assert idx == TOOL
    v = feats.to_vector()
    assert v[6] == 1.0 and v[7] == 1.0   # requires_tool, has_code_signal lit


def test_features_research_helpers():
    for h in ("web_search", "code_knowledge"):
        feats, idx = autonomous_features_for_helper(h)
        assert idx == RESEARCH
        v = feats.to_vector()
        assert v[6] == 0.0 and v[7] == 0.0   # research lane: not tool/code-shaped


def test_features_non_routing_helper_is_none():
    feats, idx = autonomous_features_for_helper("memo_inscribe")
    assert feats is None and idx is None


# ── the orchestration (fakes for judge / agency / send_queue) ───────────────
class _FakeJudge:
    def __init__(self, verdicts):
        self._v = list(verdicts)
        self.calls = 0

    def judge(self, *, problem, action, evidence):
        v = self._v[min(self.calls, len(self._v) - 1)]
        self.calls += 1
        return v


class _FakeAgency:
    def __init__(self, rerun_results=None):
        self._r = list(rerun_results or [])
        self.rerun_calls = 0

    async def p8_rerun(self, helper_name, intent, correction):
        r = self._r[min(self.rerun_calls, len(self._r) - 1)] if self._r else None
        self.rerun_calls += 1
        return r


class _FakeSendQ:
    def __init__(self):
        self.items = []

    def put(self, msg):
        self.items.append(msg)


def _emit(sq, judge, agency, helper="coding_sandbox", intent=None, ar=None):
    intent = intent or {"posture": "meditate"}
    ar = ar or {"helper": helper, "action_type": "tool", "result": "out", "reasoning": "r"}
    _maybe_emit_autonomous_experience(sq, "agency", "test", intent, ar, agency, judge)


def _rewards(sq):
    return [m for m in sq.items if m["type"] == bus.SELF_LEARN_REWARD]


def test_solved_first_attempt_positive_reward_no_retry():
    sq = _FakeSendQ()
    agency = _FakeAgency()
    _emit(sq, _FakeJudge([{"solved": True, "correction": "", "confidence": 0.9}]), agency)
    r = _rewards(sq)
    assert len(r) == 1
    p = r[0]["payload"]
    assert p["action"] == TOOL and p["reward"] == 0.9
    assert p["source"] == "task_completion" and "parent_tool_call_tx" not in p
    assert len(p["features"]) == 30 and agency.rerun_calls == 0


def test_retry_then_solved_rewards_converged():
    sq = _FakeSendQ()
    agency = _FakeAgency([{"result": "ratio=1.618", "success": True}])
    judge = _FakeJudge([
        {"solved": False, "correction": "use the ratio", "confidence": 0.3},
        {"solved": True, "correction": "", "confidence": 0.85},
    ])
    _emit(sq, judge, agency)
    r = _rewards(sq)
    assert len(r) == 1 and r[0]["payload"]["reward"] == 0.85
    assert agency.rerun_calls == 1            # one correction+retry


def test_exhaustion_emits_negative_reward():
    sq = _FakeSendQ()
    # default p8_max_attempts=3 → 1 initial + 2 retries; judge never solves.
    agency = _FakeAgency([{"result": "x"}, {"result": "y"}, {"result": "z"}])
    judge = _FakeJudge([{"solved": False, "correction": "still wrong", "confidence": 0.2}])
    _emit(sq, judge, agency)
    r = _rewards(sq)
    assert len(r) == 1 and r[0]["payload"]["reward"] == -0.5
    assert agency.rerun_calls == 2            # bounded at max_attempts=3 total


def test_non_routing_helper_emits_nothing():
    sq = _FakeSendQ()
    _emit(sq, _FakeJudge([{"solved": True, "confidence": 1.0}]), _FakeAgency(),
          helper="memo_inscribe",
          ar={"helper": "memo_inscribe", "result": "x", "reasoning": "r"})
    assert _rewards(sq) == []


def test_judge_miss_emits_nothing():
    sq = _FakeSendQ()
    _emit(sq, _FakeJudge([None]), _FakeAgency())   # judge LLM miss → untrained
    assert _rewards(sq) == []


# ── Unified failure-replay (EEL-B2 / mastery §7.P9) — a REVISIT must ALWAYS emit
#    exactly one terminal RESULT so the stored problem progresses, never strands.
def _results(sq):
    return [m for m in sq.items if m["type"] == bus.FAILED_ATTEMPT_REVISIT_RESULT]


_REVISIT_INTENT = {"posture": "meditate",
                   "_revisit": {"problem_id": "fa_x", "goal_class": "autonomous:research",
                                "helper": "code_knowledge"}}


def test_revisit_solved_emits_resolved_result():
    sq = _FakeSendQ()
    _emit(sq, _FakeJudge([{"solved": True, "correction": "", "confidence": 0.8}]),
          _FakeAgency(), intent=dict(_REVISIT_INTENT))
    res = _results(sq)
    assert len(res) == 1 and res[0]["payload"]["solved"] is True
    assert res[0]["payload"]["problem_id"] == "fa_x"
    # solved revisit → boosted reward at rank-3 source
    assert _rewards(sq)[0]["payload"]["source"] == "task_completion"


def test_revisit_judge_miss_still_emits_unsolved_result():
    # The judge LLM miss path emitted NOTHING before — a revisit would strand
    # in_progress. The backstop must now emit an unsolved RESULT (→ bump→abandon).
    sq = _FakeSendQ()
    _emit(sq, _FakeJudge([None]), _FakeAgency(), intent=dict(_REVISIT_INTENT))
    res = _results(sq)
    assert len(res) == 1 and res[0]["payload"]["solved"] is False
    assert res[0]["payload"]["problem_id"] == "fa_x"
    assert _rewards(sq) == []   # no false-positive reward, but the problem progresses


def test_revisit_non_routing_helper_still_emits_unsolved_result():
    sq = _FakeSendQ()
    intent = {"posture": "meditate",
              "_revisit": {"problem_id": "fa_y", "goal_class": "g", "helper": "memo_inscribe"}}
    _emit(sq, _FakeJudge([{"solved": True, "confidence": 1.0}]), _FakeAgency(),
          helper="memo_inscribe",
          ar={"helper": "memo_inscribe", "result": "x", "reasoning": "r"}, intent=intent)
    res = _results(sq)
    assert len(res) == 1 and res[0]["payload"]["solved"] is False   # backstop fired
    assert _rewards(sq) == []


def test_non_revisit_bail_emits_no_result():
    # A NON-revisit bail must NOT emit a revisit result (only revisits do).
    sq = _FakeSendQ()
    _emit(sq, _FakeJudge([None]), _FakeAgency())   # normal autonomous, judge miss
    assert _results(sq) == []


# ── Fix #2 (EEL-B2/P9): a research-helper exhaustion enqueues the ORIGINAL input
#    params (helper_params) so a revisit faithfully replays the real attempt.
def test_enqueue_carries_helper_params_for_faithful_replay():
    sq = _FakeSendQ()
    agency = _FakeAgency([{"result": "x"}, {"result": "y"}, {"result": "z"}])
    judge = _FakeJudge([{"solved": False, "correction": "nope", "confidence": 0.2}])
    ar = {"helper": "web_search", "action_type": "research", "result": "out",
          "reasoning": "r", "helper_params": {"query": "what is the TVL of Jupiter?"}}
    _emit(sq, judge, agency, helper="web_search", ar=ar)
    enq = [m for m in sq.items if m["type"] == bus.FAILED_ATTEMPT_ENQUEUE]
    assert len(enq) == 1
    assert enq[0]["payload"]["intent_seed"].get("helper_params") == {
        "query": "what is the TVL of Jupiter?"}


def test_kill_switch_off_emits_nothing(monkeypatch):
    import titan_hcl.modules.agency_worker as aw
    monkeypatch.setattr(aw, "get_params", lambda s: (
        {"self_learning": {"oml_autonomous_experience_enabled": False}}
        if s == "synthesis" else {}))
    sq = _FakeSendQ()
    _emit(sq, _FakeJudge([{"solved": True, "confidence": 1.0}]), _FakeAgency())
    assert _rewards(sq) == []


# ── the emitted reward trains the action via the v1.1 direct path ───────────
def test_autonomous_reward_trains_via_direct_path(tmp_path):
    sq = _FakeSendQ()
    _emit(sq, _FakeJudge([{"solved": True, "correction": "", "confidence": 0.8}]),
          _FakeAgency())
    payload = _rewards(sq)[0]["payload"]
    store = _SelfLearningStore(path=str(tmp_path / "sl.duckdb"))
    policy = OuterMetaPolicy(lr=0.05)
    trained = _handle_reward(payload, store, policy, None,
                             _cfg({"synthesis": {"self_learning": {}}}),
                             _FakeSendQ(), "self_learning")
    assert trained is True
    tuples = store.recent_reward_tuples(10)
    assert len(tuples) == 1 and int(tuples[0][1]) == TOOL


def test_new_sources_registered_at_correct_ranks():
    # autonomous_oracle = deterministic top tier; task_completion = llm_judge tier.
    assert _REWARD_SOURCE_RANK["autonomous_oracle"] == _REWARD_SOURCE_RANK["oracle"]
    assert _REWARD_SOURCE_RANK["task_completion"] == _REWARD_SOURCE_RANK["llm_judge"]
