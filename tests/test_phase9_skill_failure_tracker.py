"""Phase 9 — SkillFailureTracker (repair-fork-on-failure, arch §9.3 / rFP §11.5)."""

import pytest

from titan_hcl.synthesis.skill_failure_tracker import SkillFailureTracker


class _FakeForkStore:
    def __init__(self):
        self.calls = []
        self._n = 0

    def create_fork(self, *, intent, root_anchor=None, parent_concept_id=None):
        self.calls.append({"intent": intent, "root_anchor": root_anchor,
                           "parent_concept_id": parent_concept_id})
        self._n += 1
        return f"fork_{self._n}"


def _emits():
    out = []
    return out, (lambda topic, payload: out.append((topic, payload)))


def test_failure_below_threshold_no_fork():
    fs = _FakeForkStore()
    t = SkillFailureTracker(fork_store=fs, concept_resolver=lambda s: None,
                            failure_threshold=3)
    assert t.record_outcome("skill_a", success=False) is None
    assert t.record_outcome("skill_a", success=False) is None
    assert fs.calls == []
    assert t.consecutive_failures("skill_a") == 2


def test_threshold_spawns_repair_fork():
    fs = _FakeForkStore()
    resolver = lambda s: ("tx_root_hash", "concept_metaplex")
    out, emit = _emits()
    t = SkillFailureTracker(fork_store=fs, concept_resolver=resolver,
                            bus_emit=emit, failure_threshold=3)
    assert t.record_outcome("skill_a", success=False) is None
    assert t.record_outcome("skill_a", success=False) is None
    fork_id = t.record_outcome("skill_a", success=False)
    assert fork_id == "fork_1"
    assert fs.calls[0]["root_anchor"] == "tx_root_hash"
    assert fs.calls[0]["parent_concept_id"] == "concept_metaplex"
    # counter reset after spawn
    assert t.consecutive_failures("skill_a") == 0
    # bus event emitted
    assert out and out[0][0] == "SKILL_REPAIR_FORK_SPAWNED"
    assert out[0][1]["kind"] == "repair"


def test_success_resets_counter():
    fs = _FakeForkStore()
    t = SkillFailureTracker(fork_store=fs, concept_resolver=lambda s: None,
                            failure_threshold=3)
    t.record_outcome("skill_a", success=False)
    t.record_outcome("skill_a", success=False)
    t.record_outcome("skill_a", success=True)
    assert t.consecutive_failures("skill_a") == 0
    # next two failures should NOT yet trigger
    t.record_outcome("skill_a", success=False)
    assert t.record_outcome("skill_a", success=False) is None
    assert fs.calls == []


def test_no_spine_concept_spawns_net_new():
    fs = _FakeForkStore()
    t = SkillFailureTracker(fork_store=fs, concept_resolver=lambda s: None,
                            failure_threshold=1)
    fork_id = t.record_outcome("skill_x", success=False)
    assert fork_id == "fork_1"
    assert fs.calls[0]["root_anchor"] is None
    assert fs.calls[0]["parent_concept_id"] is None


def test_idempotency_no_double_spawn_while_live():
    fs = _FakeForkStore()
    resolver = lambda s: ("tx_root", "concept_a")
    t = SkillFailureTracker(fork_store=fs, concept_resolver=resolver,
                            failure_threshold=1)
    f1 = t.record_outcome("skill_a", success=False)
    assert f1 == "fork_1"
    # another failure while the repair fork is live → no new spawn
    f2 = t.record_outcome("skill_a", success=False)
    assert f2 is None
    assert len(fs.calls) == 1
    # resolving the repair fork clears the guard → next threshold spawns again
    t.resolve_repair_fork("skill_a")
    f3 = t.record_outcome("skill_a", success=False)
    assert f3 == "fork_2"


def test_half_anchor_resolver_falls_back_to_net_new():
    fs = _FakeForkStore()
    # resolver returns a root but no concept → inconsistent → net-new
    t = SkillFailureTracker(fork_store=fs, concept_resolver=lambda s: ("tx", None),
                            failure_threshold=1)
    fork_id = t.record_outcome("skill_a", success=False)
    assert fork_id == "fork_1"
    assert fs.calls[0]["root_anchor"] is None
    assert fs.calls[0]["parent_concept_id"] is None


def test_fork_store_raise_is_soft():
    class _Boom:
        def create_fork(self, **kw):
            raise RuntimeError("kuzu down")
    t = SkillFailureTracker(fork_store=_Boom(), concept_resolver=lambda s: None,
                            failure_threshold=1)
    # must not raise; returns None; counter reset
    assert t.record_outcome("skill_a", success=False) is None
    assert t.consecutive_failures("skill_a") == 0


def test_empty_skill_id_ignored():
    fs = _FakeForkStore()
    t = SkillFailureTracker(fork_store=fs, concept_resolver=lambda s: None,
                            failure_threshold=1)
    assert t.record_outcome("", success=False) is None
    assert fs.calls == []
