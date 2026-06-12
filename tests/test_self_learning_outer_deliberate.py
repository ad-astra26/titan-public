"""Offline tests for OML Phase C piece 6 — the explore-tick deliberative path.

Verifies: (a) candidate_macro_classes draws verified-win, not-yet-emitted classes;
(b) _emit_macro produces the exact SELF_LEARN_MACRO_READY S2 payload + marks emitted;
(c) _outer_deliberate emits ONLY on a verified composite + always trains the outer
policies; (d) the real OuterMeta+OuterReasoning integrate end-to-end through the
deliberation with the in-worker win-oracle (no crash, policies advance).
"""
import numpy as np
import pytest

from titan_hcl.modules.self_learning_worker import (
    _DEFAULTS,
    _SelfLearningStore,
    _emit_macro,
    _outer_deliberate,
)
from titan_hcl.synthesis.outer_meta_policy import OUTER_POLICY_INPUT_DIM
from titan_hcl.synthesis.outer_meta_reasoning import OuterMetaReasoningEngine
from titan_hcl.synthesis.outer_reasoning import OuterReasoningEngine


def _store():
    return _SelfLearningStore(path=":memory:")


def _seed_wins(store, goal_class, action, n):
    for _ in range(n):
        store.record_reward_tuple(
            features=[0.3] * OUTER_POLICY_INPUT_DIM, action=action,
            reward=1.0, goal_class=goal_class)


class _Q:
    def __init__(self):
        self.puts = []

    def put(self, msg):
        self.puts.append(msg)


class _StubMeta:
    def __init__(self, verified):
        self._v = verified
        self.trained = []
        self.last_problem = None

    def run_chain(self, problem, reasoning_engine):
        self.last_problem = problem
        return {"verified": self._v, "reward": 1.0 if self._v else -0.5,
                "chain_length": 5, "chain": ["FORMULATE.define", "RECALL.topic"]}

    def train_terminal(self, r):
        self.trained.append(r)

    def save_all(self):
        pass


class _StubReason:
    def __init__(self):
        self.sig = None

    def set_problem(self, sig):
        self.sig = list(sig)

    def save_all(self):
        pass


def _cfg(**over):
    c = dict(_DEFAULTS)
    c.update(over)
    return c


def test_candidate_macro_classes_filters_correctly():
    s = _store()
    _seed_wins(s, "combinatorics", 1, 6)   # qualifies
    _seed_wins(s, "arithmetic", 2, 5)      # qualifies
    _seed_wins(s, "trivia", 0, 2)          # below min_wins
    s.mark_macro_emitted("combinatorics", 1)  # already emitted → excluded
    out = s.candidate_macro_classes(min_wins=5, limit=8)
    assert ("arithmetic", 2) in out
    assert ("combinatorics", 1) not in out   # emitted
    assert ("trivia", 0) not in out          # too few wins


def test_emit_macro_payload_and_dedup():
    s = _store()
    _seed_wins(s, "combinatorics", 1, 6)
    q = _Q()
    _emit_macro("combinatorics", 1, s, q, "self_learning")
    assert len(q.puts) == 1
    msg = q.puts[0]
    assert msg["type"] == "SELF_LEARN_MACRO_READY"
    assert msg["dst"] == "synthesis"
    p = msg["payload"]
    assert p["goal_class"] == "combinatorics"
    assert p["action_name"]  # mapped name
    assert len(p["signature"]) == OUTER_POLICY_INPUT_DIM
    assert p["verified"] is True
    assert p["use_count"] == 6
    assert p["label"] == f"macro::combinatorics::{p['action_name']}"
    assert s.macro_already_emitted("combinatorics", 1) is True


def test_outer_deliberate_emits_only_when_verified():
    s = _store()
    _seed_wins(s, "combinatorics", 1, 6)
    cfg = _cfg(outer_meta_enabled=True, macro_min_wins=5)

    # verified=True → emits + trains
    q1, reason1, meta1 = _Q(), _StubReason(), _StubMeta(verified=True)
    _outer_deliberate(cfg, s, q1, "self_learning", reason1, meta1)
    assert meta1.trained == [1.0]
    assert reason1.sig is not None and len(reason1.sig) == OUTER_POLICY_INPUT_DIM
    assert meta1.last_problem["goal_class"] == "combinatorics"
    assert len(q1.puts) == 1 and q1.puts[0]["type"] == "SELF_LEARN_MACRO_READY"
    assert s.macro_already_emitted("combinatorics", 1) is True


def test_outer_deliberate_no_emit_when_unverified():
    s = _store()
    _seed_wins(s, "arithmetic", 2, 6)
    cfg = _cfg(outer_meta_enabled=True, macro_min_wins=5)
    q, reason, meta = _Q(), _StubReason(), _StubMeta(verified=False)
    _outer_deliberate(cfg, s, q, "self_learning", reason, meta)
    assert meta.trained == [-0.5]          # trained regardless
    assert len(q.puts) == 0                 # NOT emitted
    assert s.macro_already_emitted("arithmetic", 2) is False


def test_outer_deliberate_noop_without_candidates():
    s = _store()  # no wins
    cfg = _cfg(outer_meta_enabled=True, macro_min_wins=5)
    q, reason, meta = _Q(), _StubReason(), _StubMeta(verified=True)
    _outer_deliberate(cfg, s, q, "self_learning", reason, meta)
    assert meta.trained == []   # never ran
    assert len(q.puts) == 0


def test_real_engines_integrate_end_to_end():
    """The REAL OuterMeta + OuterReasoning run through a deliberation with the
    in-worker win-oracle — proving the piece-3/4/5/6 stack composes. Emit is
    stochastic (depends on the chain delegating), so we assert it RUNS + TRAINS."""
    np.random.seed(17)
    s = _store()
    _seed_wins(s, "combinatorics", 1, 6)
    cfg = _cfg(outer_meta_enabled=True, macro_min_wins=5, outer_meta_max_steps=16)

    reason = OuterReasoningEngine()
    meta = OuterMetaReasoningEngine(config={"max_steps": 16})

    def win_oracle(problem, _dr, _store=s):
        return 1.0 if _store.win_count(str(problem.get("goal_class", "")),
                                       int(problem.get("action", 0))) > 0 else -1.0
    meta.set_oracle(win_oracle)

    q = _Q()
    _outer_deliberate(cfg, s, q, "self_learning", reason, meta)
    # the meta policy was trained on the concluded chain (>=1 transition)
    assert meta.meta_policy.total_updates >= 1
    assert meta._total_meta_chains == 1
    # whatever it emitted is a well-formed macro
    for msg in q.puts:
        assert msg["type"] == "SELF_LEARN_MACRO_READY"
        assert msg["payload"]["goal_class"] == "combinatorics"
        assert msg["payload"]["verified"] is True
