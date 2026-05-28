"""Phase 9 — UserFeedbackOverride (Tier-2 override, INV-Syn-24)."""

from titan_hcl.synthesis.user_feedback import UserFeedbackOverride


class _FakeOMW:
    def __init__(self):
        self.patches = []
        self._n = 0

    def write_scored_by_patch(self, *, entries, **kw):
        self.patches.append(entries)
        self._n += 1
        return f"patch_tx_{self._n}"


class _FakeSkillStore:
    def __init__(self, util=0.7):
        self.util = util
        self.deltas = []

    def apply_utility_delta(self, skill_id, delta):
        self.deltas.append((skill_id, delta))
        self.util = max(-1.0, min(1.0, self.util + delta))
        return self.util


def test_positive_feedback_patches_scored_by_user_and_bumps_utility():
    omw, ss = _FakeOMW(), _FakeSkillStore(util=0.7)
    o = UserFeedbackOverride(outer_memory_writer=omw, skill_store=ss,
                             user_feedback_delta=0.15)
    res = o.apply(tool_call_tx="tx_1", verdict="positive", skill_id="skill_a")
    assert res["scored_by"] == "user"
    assert res["patch_tx"] == "patch_tx_1"
    assert res["new_utility"] == 0.85
    # patch carries the supersede marker + verdict
    entry = omw.patches[0][0]
    assert entry["scored_by"] == "user"
    assert entry["verdict"] == "positive"
    assert entry["parent_tool_call_tx"] == "tx_1"
    assert ss.deltas == [("skill_a", 0.15)]


def test_negative_feedback_lowers_utility():
    omw, ss = _FakeOMW(), _FakeSkillStore(util=0.5)
    o = UserFeedbackOverride(outer_memory_writer=omw, skill_store=ss,
                             user_feedback_delta=0.15)
    res = o.apply(tool_call_tx="tx_2", verdict="negative", skill_id="skill_b")
    assert ss.deltas == [("skill_b", -0.15)]
    assert abs(res["new_utility"] - 0.35) < 1e-9


def test_no_skill_path_only_anchors_patch():
    omw = _FakeOMW()
    o = UserFeedbackOverride(outer_memory_writer=omw, skill_store=None)
    res = o.apply(tool_call_tx="tx_3", verdict="positive")
    assert res["new_utility"] is None
    assert res["patch_tx"] == "patch_tx_1"


def test_implicit_source_ignored():
    omw = _FakeOMW()
    o = UserFeedbackOverride(outer_memory_writer=omw)
    assert o.apply(tool_call_tx="tx", verdict="positive", source="implicit") is None
    assert omw.patches == []


def test_bad_verdict_rejected():
    omw = _FakeOMW()
    o = UserFeedbackOverride(outer_memory_writer=omw)
    assert o.apply(tool_call_tx="tx", verdict="meh") is None
    assert o.apply(tool_call_tx="", verdict="positive") is None
    assert omw.patches == []


def test_anchor_failure_is_soft():
    class _Boom:
        def write_scored_by_patch(self, **kw):
            raise RuntimeError("chain down")
    o = UserFeedbackOverride(outer_memory_writer=_Boom(), skill_store=_FakeSkillStore())
    res = o.apply(tool_call_tx="tx", verdict="positive", skill_id="s")
    # utility still applied; patch_tx None; no raise
    assert res["patch_tx"] is None
    assert res["scored_by"] == "user"


def test_provenance_supersede_marker_present():
    omw = _FakeOMW()
    o = UserFeedbackOverride(outer_memory_writer=omw)
    o.apply(tool_call_tx="tx", verdict="negative")
    assert omw.patches[0][0]["supersedes"] == "oracle_or_llm"
