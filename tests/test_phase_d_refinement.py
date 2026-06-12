"""§7.D D.4c — composite refinement (mutate-not-update / INV-OML-5).

A (goal_class, action) whose verified wins GROW past a threshold since its last
macro emit yields a SUCCESSOR composite `::v{n+1}` that cites its predecessor as
lineage — the prior macro record is never overwritten. Covers the store
accounting (version/wins_at_emit, refinement_candidates) + the _emit_macro
successor payload + the _outer_deliberate refinement branch.
"""
from titan_hcl.modules.self_learning_worker import (
    _DEFAULTS,
    _SelfLearningStore,
    _emit_macro,
    _outer_deliberate,
)
from titan_hcl.synthesis.outer_meta_policy import OUTER_POLICY_INPUT_DIM


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
    def __init__(self, verified=True):
        self._v = verified
        self.trained = []

    def run_chain(self, problem, reasoning_engine):
        return {"verified": self._v, "reward": 1.0, "chain_length": 4}

    def train_terminal(self, r):
        self.trained.append(r)

    def save_all(self):
        pass


class _StubReason:
    def set_problem(self, sig):
        pass

    def save_all(self):
        pass


def _cfg(**over):
    c = dict(_DEFAULTS)
    c.update(over)
    return c


# ── store accounting ────────────────────────────────────────────────────
def test_mark_macro_emitted_records_version_and_wins():
    s = _store()
    s.mark_macro_emitted("x", 1, version=1, wins_at_emit=6)
    assert s.macro_version("x", 1) == (1, 6)
    # a successor UPDATEs the row in place (the accounting; the macro RECORDS are
    # distinct ::v ids on the synthesis side).
    s.mark_macro_emitted("x", 1, version=2, wins_at_emit=11)
    assert s.macro_version("x", 1) == (2, 11)
    assert s.macro_version("never", 0) == (0, 0)


def test_refinement_candidates_surfaces_only_grown_classes():
    s = _store()
    _seed_wins(s, "grown", 1, 11)
    s.mark_macro_emitted("grown", 1, version=1, wins_at_emit=6)   # +5 since → candidate
    _seed_wins(s, "flat", 2, 6)
    s.mark_macro_emitted("flat", 2, version=1, wins_at_emit=6)    # +0 since → not
    out = s.refinement_candidates(min_growth=5, limit=4)
    classes = [(gc, a) for gc, a, _v, _w in out]
    assert ("grown", 1) in classes
    assert ("flat", 2) not in classes
    # the version to mint next is prev+1
    grown = [r for r in out if r[0] == "grown"][0]
    assert grown[2] == 1                       # prev_version


# ── _emit_macro successor ───────────────────────────────────────────────
def test_emit_macro_successor_label_and_lineage():
    s = _store()
    _seed_wins(s, "combinatorics", 1, 12)
    q = _Q()
    _emit_macro("combinatorics", 1, s, q, "self_learning", version=2)
    p = q.puts[0]["payload"]
    base = f"macro::combinatorics::{p['action_name']}"
    assert p["label"] == f"{base}::v2"          # successor id, NOT the v1 id
    assert p["version"] == 2
    assert base in p["composed_from"]           # cites its predecessor (lineage)
    assert s.macro_version("combinatorics", 1) == (2, 12)


def test_emit_macro_v1_keeps_canonical_label():
    s = _store()
    _seed_wins(s, "x", 1, 6)
    q = _Q()
    _emit_macro("x", 1, s, q, "self_learning")   # version defaults to 1
    p = q.puts[0]["payload"]
    assert p["label"] == f"macro::x::{p['action_name']}"   # no ::v suffix (back-compat)
    assert p["version"] == 1
    assert "composed_from" not in p or f"macro::x::{p['action_name']}" not in p.get(
        "composed_from", [])                     # v1 does not cite itself


# ── _outer_deliberate refinement branch ─────────────────────────────────
def test_outer_deliberate_refines_when_no_fresh_candidates():
    s = _store()
    _seed_wins(s, "combinatorics", 1, 6)
    q = _Q()
    # emit v1 (marks emitted, wins_at_emit=6) → now NOT a fresh candidate
    _emit_macro("combinatorics", 1, s, q, "self_learning")
    assert s.candidate_macro_classes(min_wins=5) == []
    _seed_wins(s, "combinatorics", 1, 5)        # grow +5 → refinement candidate

    cfg = _cfg(outer_meta_enabled=True, macro_min_wins=5, macro_refine_min_growth=5)
    q2, reason, meta = _Q(), _StubReason(), _StubMeta(verified=True)
    _outer_deliberate(cfg, s, q2, "self_learning", reason, meta)
    assert len(q2.puts) == 1
    p = q2.puts[0]["payload"]
    assert p["version"] == 2                     # a successor was minted
    assert p["label"].endswith("::v2")
    assert s.macro_version("combinatorics", 1) == (2, 11)


def test_outer_deliberate_no_refine_below_growth():
    s = _store()
    _seed_wins(s, "x", 1, 6)
    q = _Q()
    _emit_macro("x", 1, s, q, "self_learning")
    _seed_wins(s, "x", 1, 2)                     # +2 only — below growth
    cfg = _cfg(outer_meta_enabled=True, macro_min_wins=5, macro_refine_min_growth=5)
    q2, reason, meta = _Q(), _StubReason(), _StubMeta(verified=True)
    _outer_deliberate(cfg, s, q2, "self_learning", reason, meta)
    assert q2.puts == []                         # nothing fresh, nothing grown enough
