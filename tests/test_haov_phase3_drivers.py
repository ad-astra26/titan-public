"""Phase 3A/3B tests for rFP_haov_efficacy_closure — restored dropped loops.

Covers the self_reflection_worker drivers that commit 72f95a6b dropped:
  - coding_explorer.explore() driver (3A): Layer-A gap-driven (introspect
    coherence gaps) + Layer-B 6h time-fallback in the epoch tick.
  - self_reasoning.check_predictions() driver + self_model CGN producer (3B):
    the producer rFP §1.3 flagged as entirely MISSING.

Asserts the faithful-restore contract (Maker 2026-05-30, option A):
  • coding explore() is driven (feeds the `coding` HAOV consumer);
  • self_model CGN_TRANSITION emitted with legacy reward (0.5 / -0.1);
  • the dead SELF_PREDICTION_VERIFIED route is NOT recreated (F4 discipline);
  • G18 SHM decoders (_read_neuromods / _read_epoch) parse the slot shapes.

Run: python -m pytest tests/test_haov_phase3_drivers.py -v -p no:anchorpy --tb=short
"""
import numpy as np
import pytest

import titan_hcl.modules.self_reflection_worker as w


# ── fakes ────────────────────────────────────────────────────────────────────

class FakeQueue:
    def __init__(self):
        self.msgs = []

    def put(self, m):
        self.msgs.append(m)

    def put_nowait(self, m):
        self.msgs.append(m)

    def of_type(self, t):
        return [m for m in self.msgs if m.get("type") == t]


class FakeReader:
    def __init__(self, arr):
        self._arr = arr

    def read(self):
        return self._arr


class FakeCodingExplorer:
    def __init__(self, *, can_explore=True, fallback=False, result=None):
        self._can_explore = can_explore
        self._fallback = fallback
        self._result = result
        self._time_fallback_seconds = 21600.0
        self.explore_calls = []

    @property
    def can_explore(self):
        return self._can_explore

    def tick_cooldown(self):
        pass

    def should_fire_fallback(self, now):
        return self._fallback

    def build_fallback_trigger(self, now):
        return {"action": "scheduled_novelty", "gap_metric": "time_fallback",
                "reason": "silence", "urgency": 0.5}

    def explore(self, *, trigger, epoch, neuromods, context):
        self.explore_calls.append(
            {"trigger": trigger, "epoch": epoch,
             "neuromods": neuromods, "context": context})
        return self._result


class FakeResult:
    action = "compose"
    concept = "symmetry"
    sandbox_success = True
    tests_passed = 2
    tests_total = 3
    reward = 0.6


class FakeSelfReasoning:
    def __init__(self, *, verifications=None, triggers=None, ema=0.5):
        self._verifications = verifications or []
        self._triggers = triggers or []
        self._prediction_accuracy_ema = ema
        self.check_calls = []

    def tick_cooldown(self):
        pass

    def check_predictions(self, epoch, neuromods, msl=None, language=None):
        self.check_calls.append(epoch)
        # mimic the EMA move on a real verification
        if self._verifications:
            self._prediction_accuracy_ema = 0.42
        return self._verifications

    def get_exploration_triggers(self):
        return self._triggers


# ── G18 SHM decoders ──────────────────────────────────────────────────────────

def test_read_neuromods_decodes_6x4():
    arr = np.zeros((6, 4), dtype=np.float32)
    for i in range(6):
        arr[i, 0] = 0.1 * (i + 1)  # level field
    state_refs = {"_shm_readers": {"neuromod": FakeReader(arr)}}
    nm = w._read_neuromods(state_refs)
    assert set(nm) == set(w._NEUROMOD_NAMES)
    assert nm["DA"] == pytest.approx(0.1)
    assert nm["GABA"] == pytest.approx(0.6)


def test_read_neuromods_bad_shape_returns_empty():
    state_refs = {"_shm_readers": {"neuromod": FakeReader(np.zeros((3, 3)))}}
    assert w._read_neuromods(state_refs) == {}
    assert w._read_neuromods({}) == {}


def test_read_epoch_decodes():
    state_refs = {"_shm_readers": {"epoch": FakeReader(np.array([1234, 0, 0]))}}
    assert w._read_epoch(state_refs) == 1234
    assert w._read_epoch({}) == 0


# ── context builder ─────────────────────────────────────────────────────────

def test_build_context_assembles_from_sources():
    sr = FakeSelfReasoning(ema=0.73)
    state_refs = {"_last_reasoning_stats": {"total_chains": 40,
                                            "commit_rate": 0.25}}
    ctx = w._build_coding_explore_context(
        state_refs, sr, {"chi_coherence": 0.8, "i_confidence": 0.6})
    assert ctx["chi_coherence"] == pytest.approx(0.8)
    assert ctx["i_confidence"] == pytest.approx(0.6)
    assert ctx["total_chains"] == 40
    assert ctx["commit_rate"] == pytest.approx(0.25)
    assert ctx["prediction_accuracy"] == pytest.approx(0.73)


def test_build_context_degrades_gracefully():
    ctx = w._build_coding_explore_context({}, None, None)
    assert ctx == {}


# ── coding exploration driver (3A) ──────────────────────────────────────────

def test_drive_coding_exploration_calls_explore():
    ce = FakeCodingExplorer(result=FakeResult())
    q = FakeQueue()
    state_refs = {"coding_explorer": ce}
    w._drive_coding_exploration(
        state_refs, q, "self_reflection_worker",
        trigger={"action": "seek_novelty", "gap_metric": "coherence"},
        epoch=500, neuromods={"DA": 0.5}, context={"total_chains": 10})
    assert len(ce.explore_calls) == 1
    call = ce.explore_calls[0]
    assert call["epoch"] == 500
    assert call["trigger"]["gap_metric"] == "coherence"


def test_drive_coding_exploration_respects_cooldown():
    ce = FakeCodingExplorer(can_explore=False)
    w._drive_coding_exploration(
        {"coding_explorer": ce}, FakeQueue(), "n",
        trigger={}, epoch=1, neuromods={}, context={})
    assert ce.explore_calls == []


def test_drive_coding_exploration_handles_explore_exception():
    ce = FakeCodingExplorer()
    def _boom(**kw):
        raise RuntimeError("sandbox down")
    ce.explore = _boom
    # must not raise
    w._drive_coding_exploration(
        {"coding_explorer": ce}, FakeQueue(), "n",
        trigger={}, epoch=1, neuromods={}, context={})


# ── self-prediction check + self_model producer (3B) ─────────────────────────

def _verif(target, confirmed):
    return {"prediction_id": "p1", "target": target, "predicted": 0.5,
            "predicted_direction": "up", "actual": 0.6 if confirmed else 0.1,
            "error": 0.1, "confirmed": confirmed, "horizon_epochs": 100}


def test_self_prediction_emits_self_model_transition_confirmed():
    sr = FakeSelfReasoning(verifications=[_verif("i_confidence", True)])
    q = FakeQueue()
    w._drive_self_prediction_check(
        {"self_reasoning": sr}, q, "self_reflection_worker",
        epoch=200, neuromods={"DA": 0.5})
    txs = q.of_type("CGN_TRANSITION")
    assert len(txs) == 1
    p = txs[0]["payload"]
    assert p["consumer"] == "self_model"
    assert p["concept_id"] == "self_pred_i_confidence"
    assert p["reward"] == 0.5            # legacy contract: confirmed
    assert p["outcome_context"]["confirmed"] is True
    # provenance commit to a LIVE dst (timechain_worker)
    assert len(q.of_type("TIMECHAIN_COMMIT")) == 1


def test_self_prediction_falsified_reward():
    sr = FakeSelfReasoning(verifications=[_verif("vocab_total", False)])
    q = FakeQueue()
    w._drive_self_prediction_check(
        {"self_reasoning": sr}, q, "n", epoch=200, neuromods={})
    p = q.of_type("CGN_TRANSITION")[0]["payload"]
    assert p["reward"] == -0.1           # legacy contract: falsified


def test_self_prediction_no_dead_route_emitted():
    """The dead SELF_PREDICTION_VERIFIED constant must NOT be re-emitted
    (would recreate the F4 dead-route anti-pattern)."""
    sr = FakeSelfReasoning(verifications=[_verif("x", True)])
    q = FakeQueue()
    w._drive_self_prediction_check(
        {"self_reasoning": sr}, q, "n", epoch=200, neuromods={})
    assert q.of_type("SELF_PREDICTION_VERIFIED") == []


def test_self_prediction_no_verifications_no_emit():
    sr = FakeSelfReasoning(verifications=[])
    q = FakeQueue()
    w._drive_self_prediction_check(
        {"self_reasoning": sr}, q, "n", epoch=200, neuromods={})
    assert q.msgs == []


# ── epoch-tick integration: Layer-B fallback + 100-epoch self-pred cadence ───

def test_epoch_tick_fires_fallback_when_due():
    ce = FakeCodingExplorer(fallback=True, result=FakeResult())
    sr = FakeSelfReasoning()
    q = FakeQueue()
    state_refs = {
        "coding_explorer": ce, "self_reasoning": sr,
        "_shm_readers": {"epoch": FakeReader(np.array([10])),
                         "neuromod": FakeReader(np.zeros((6, 4)))},
        "_last_self_pred_check_epoch": 0,
    }
    w._handle_epoch_tick({}, state_refs, q, "n")
    assert len(ce.explore_calls) == 1
    assert ce.explore_calls[0]["trigger"]["gap_metric"] == "time_fallback"


def test_epoch_tick_no_fallback_when_not_due():
    ce = FakeCodingExplorer(fallback=False)
    sr = FakeSelfReasoning()
    state_refs = {
        "coding_explorer": ce, "self_reasoning": sr,
        "_shm_readers": {"epoch": FakeReader(np.array([10])),
                         "neuromod": FakeReader(np.zeros((6, 4)))},
        "_last_self_pred_check_epoch": 0,
    }
    w._handle_epoch_tick({}, state_refs, FakeQueue(), "n")
    assert ce.explore_calls == []


def test_epoch_tick_self_pred_cadence_100():
    ce = FakeCodingExplorer(fallback=False)
    sr = FakeSelfReasoning(verifications=[_verif("x", True)])
    q = FakeQueue()
    state_refs = {
        "coding_explorer": ce, "self_reasoning": sr,
        "_shm_readers": {"epoch": FakeReader(np.array([100])),
                         "neuromod": FakeReader(np.zeros((6, 4)))},
        "_last_self_pred_check_epoch": 0,
    }
    # epoch 100, last=0 → delta 100 ≥ 100 → fires
    w._handle_epoch_tick({}, state_refs, q, "n")
    assert sr.check_calls == [100]
    assert state_refs["_last_self_pred_check_epoch"] == 100
    # immediately again at epoch 100 → delta 0 → does NOT re-fire
    sr.check_calls.clear()
    w._handle_epoch_tick({}, state_refs, q, "n")
    assert sr.check_calls == []
