"""Tests for titan_hcl.modules.self_reflection_worker (Track 2 of
rFP_phase_c_self_improvement_subsystem_migration, SPEC v1.2.1 §9.B
D-SPEC-38).

Bus-independent tests covering chunks B3 + B5 + B6 + B7 + B8:
- Module identity (MODULE_NAME, subscribe topics list)
- B5 dispatcher routing — each handler is called with correct payload
- B5 dream-cycle hook — dream_start → on_dream_start; dream_end →
  consolidate_training + _last_dream_profile
- B6 sandbox lifecycle — health check restart attempt + disable on 2nd fail
- B7 publishers — SELF_REFLECTION_STATS_UPDATED + CODING_EXPLORER_STATS_UPDATED
  + PREDICTION_STATS_UPDATED + PREDICTION_GENERATED on counter delta
- B8 drift correction (cognitive_worker no longer has prediction_engine in
  state_refs + BOOT_DRIVER_PARITY)
- SPEC parity — new bus constants present in bus.py + bus_specs.py;
  cache_key_registry contains all 3 new keys

Note: subprocess boot + full Guardian stack is covered by the integration
suite at session-close gate. Here we exercise the pure-Python handler
logic with mock engines + a list-backed send_queue.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from titan_hcl import bus
from titan_hcl.modules import self_reflection_worker as srw


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_self_reasoning():
    sr = MagicMock()
    sr.tick_cooldown = MagicMock()
    sr.consolidate_training = MagicMock()
    sr.get_stats = MagicMock(return_value={
        "total_predictions": 12,
        "total_verified": 5,
        "total_correct": 4,
    })
    sr._last_dream_profile = None
    return sr


@pytest.fixture
def mock_coding_explorer():
    ce = MagicMock()
    ce.tick_cooldown = MagicMock()
    ce.on_dream_start = MagicMock()
    ce.get_stats = MagicMock(return_value={
        "total_exercises": 3,
        "total_successes": 2,
    })
    # Mock sandbox with status() method
    ce._sandbox = MagicMock()
    ce._sandbox.status = MagicMock(return_value="available")
    # _cgn_client absent by default — handler should no-op
    del ce._cgn_client
    return ce


@pytest.fixture
def mock_prediction_engine():
    pe = MagicMock()
    pe.get_stats = MagicMock(return_value={
        "total_predictions": 100,
        "total_surprises": 8,
        "novelty": 0.12,
        "novelty_ema": 0.11,
    })
    pe._total_predictions = 100
    pe._total_surprises = 8
    pe._last_prediction = [0.1, 0.2, 0.3]
    return pe


@pytest.fixture
def state_refs(mock_self_reasoning, mock_coding_explorer, mock_prediction_engine):
    return {
        "self_reasoning": mock_self_reasoning,
        "coding_explorer": mock_coding_explorer,
        "prediction_engine": mock_prediction_engine,
        "_db_path": "data/inner_memory.db",
        "_last_dream_state": "awake",
        "_last_dream_profile": None,
        "_sandbox_disabled": False,
        "_sandbox_restart_attempts": 0,
    }


@pytest.fixture
def send_queue():
    class _Q:
        def __init__(self):
            self.items: list = []
        def put(self, msg):
            self.items.append(msg)
    return _Q()


# ── B3 — module identity ────────────────────────────────────────────────────


def test_module_name_matches_spec():
    """SPEC v1.2.1 §9.B self_reflection_worker row."""
    assert srw.MODULE_NAME == "self_reflection_worker"


def test_entry_function_present_and_callable():
    assert hasattr(srw, "self_reflection_worker_main")
    assert callable(srw.self_reflection_worker_main)


def test_subscribe_topics_canonical_set():
    """SPEC §9.B + rFP §2.B.3 subscribe topics list — 8 events expected."""
    topics = srw._SELF_REFLECTION_WORKER_SUBSCRIBE_TOPICS
    assert len(topics) >= 8
    for t in ("REASONING_STATS_UPDATED", "META_REASONING_STATS_UPDATED",
              "EXPERIENCE_STIMULUS", "DREAMING_STATE_UPDATED",
              "CGN_CROSS_INSIGHT", "KERNEL_EPOCH_TICK",
              "MODULE_SHUTDOWN", "SAVE_NOW"):
        assert t in topics, f"Missing subscribe topic {t}"


def test_default_cadences_in_range():
    assert 1.0 <= srw.PUBLISHER_DEFAULT_S <= 10.0
    assert 2.0 <= srw.CODING_EXPLORER_PUBLISHER_S <= 30.0
    assert 10.0 <= srw.SANDBOX_HEALTH_CHECK_S <= 120.0
    assert 30.0 <= srw.ORPHAN_CHECK_S <= 300.0


# ── B5 — dispatcher routing ─────────────────────────────────────────────────


def test_dispatcher_caches_reasoning_stats(state_refs, send_queue):
    msg = {"type": bus.REASONING_STATS_UPDATED, "payload": {"total_chains": 42}}
    srw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "self_reflection_worker")
    assert state_refs["_last_reasoning_stats"]["total_chains"] == 42


def test_dispatcher_caches_meta_reasoning_stats(state_refs, send_queue):
    msg = {"type": bus.META_REASONING_STATS_UPDATED, "payload": {"total_eurekas": 7}}
    srw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "self_reflection_worker")
    assert state_refs["_last_meta_reasoning_stats"]["total_eurekas"] == 7


def test_dispatcher_caches_experience_stimulus(state_refs, send_queue):
    msg = {"type": bus.EXPERIENCE_STIMULUS, "payload": {"trace_id": "x"}}
    srw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "self_reflection_worker")
    assert state_refs["_last_experience_stimulus"]["trace_id"] == "x"


def test_dispatcher_epoch_tick_calls_tick_cooldown(state_refs, send_queue):
    msg = {"type": bus.KERNEL_EPOCH_TICK, "payload": {}}
    srw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "self_reflection_worker")
    state_refs["self_reasoning"].tick_cooldown.assert_called_once()
    state_refs["coding_explorer"].tick_cooldown.assert_called_once()


def test_dispatcher_save_now_calls_save_state(state_refs, send_queue):
    msg = {"type": bus.SAVE_NOW, "payload": {}}
    # _save_state_on_shutdown calls prediction_engine._save_state (the real
    # method name — fixed from the nonexistent save_state in commit 55e3cdce;
    # this assertion was stale residue of the old name). Verify the real method.
    state_refs["prediction_engine"]._save_state = MagicMock()
    srw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "self_reflection_worker")
    state_refs["prediction_engine"]._save_state.assert_called_once()


# ── B5 — dream-cycle hook (load-bearing per rFP §2.B.5) ─────────────────────


def test_dream_start_calls_coding_explorer_on_dream_start(state_refs, send_queue):
    msg = {"type": bus.DREAMING_STATE_UPDATED, "payload": {"state": "dream_start"}}
    srw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "self_reflection_worker")
    state_refs["coding_explorer"].on_dream_start.assert_called_once()
    # consolidate_training NOT called on dream_start
    state_refs["self_reasoning"].consolidate_training.assert_not_called()


def test_dream_end_calls_consolidate_training_and_sets_dream_profile(state_refs, send_queue):
    profile = {"intensity": 0.7, "kind": "deep"}
    msg = {"type": bus.DREAMING_STATE_UPDATED, "payload": {
        "state": "dream_end", "dream_profile": profile,
    }}
    srw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "self_reflection_worker")
    state_refs["self_reasoning"].consolidate_training.assert_called_once()
    assert state_refs["self_reasoning"]._last_dream_profile == profile
    # on_dream_start NOT called on dream_end
    state_refs["coding_explorer"].on_dream_start.assert_not_called()


def test_dream_state_cached_regardless(state_refs, send_queue):
    for state in ("dreaming", "awake"):
        msg = {"type": bus.DREAMING_STATE_UPDATED, "payload": {"state": state}}
        srw._dispatch_msg(msg, msg["type"], state_refs, send_queue, "self_reflection_worker")
        assert state_refs["_last_dream_state"] == state


# ── B6 — sandbox lifecycle ──────────────────────────────────────────────────


def test_sandbox_health_check_passes_when_available(state_refs):
    state_refs["coding_explorer"]._sandbox.status.return_value = "available"
    srw._check_sandbox_health(state_refs)
    # _sandbox_disabled stays False; restart_attempts stays 0
    assert state_refs["_sandbox_disabled"] is False
    assert state_refs["_sandbox_restart_attempts"] == 0


def test_sandbox_health_check_attempts_restart_on_unavailable(state_refs):
    """First unavailable → attempt restart workflow. On a test machine with
    python3 on PATH, restart succeeds (re-instantiated CodingSandboxHelper
    reports "available" via shutil.which) → counter resets to 0 + sandbox
    not disabled. Workflow ran without crashing is the assertion here."""
    state_refs["coding_explorer"]._sandbox.status.return_value = "unavailable"
    srw._check_sandbox_health(state_refs)
    # The original _sandbox MagicMock has been replaced with a real
    # CodingSandboxHelper instance.
    from titan_hcl.logic.agency.helpers.coding_sandbox import CodingSandboxHelper
    assert isinstance(state_refs["coding_explorer"]._sandbox, CodingSandboxHelper)
    # On test machines python3 is on PATH → restart succeeded → counter reset.
    # If python3 happens to be unavailable, counter would be 1 + disabled=True
    # — both branches are valid post-conditions for "workflow ran".
    assert (state_refs["_sandbox_restart_attempts"] == 0
            or state_refs["_sandbox_disabled"] is True)


def test_sandbox_disabled_after_second_failure(state_refs):
    """Second unavailable status with restart_attempts already 1 → disable."""
    state_refs["_sandbox_restart_attempts"] = 1
    state_refs["coding_explorer"]._sandbox.status.return_value = "unavailable"
    srw._check_sandbox_health(state_refs)
    assert state_refs["_sandbox_disabled"] is True


# ── B7 — periodic publishers ────────────────────────────────────────────────


def test_publish_self_reflection_stats_emits_payload(state_refs, send_queue):
    srw._publish_self_reflection_stats(
        state_refs, send_queue, "self_reflection_worker", "T1")
    msgs = [m for m in send_queue.items
            if m.get("type") == bus.SELF_REFLECTION_STATS_UPDATED]
    assert len(msgs) == 1
    payload = msgs[0]["payload"]
    assert payload["titan_id"] == "T1"
    assert payload["stats"]["total_predictions"] == 12
    assert "last_dream_state" in payload


def test_publish_coding_explorer_stats_emits_payload(state_refs, send_queue):
    srw._publish_coding_explorer_stats(
        state_refs, send_queue, "self_reflection_worker", "T1")
    msgs = [m for m in send_queue.items
            if m.get("type") == bus.CODING_EXPLORER_STATS_UPDATED]
    assert len(msgs) == 1
    payload = msgs[0]["payload"]
    assert payload["titan_id"] == "T1"
    assert payload["stats"]["total_exercises"] == 3
    assert payload["sandbox_disabled"] is False


def test_publish_prediction_stats_emits_payload(state_refs, send_queue):
    srw._publish_prediction_stats(
        state_refs, send_queue, "self_reflection_worker", "T1")
    msgs = [m for m in send_queue.items
            if m.get("type") == bus.PREDICTION_STATS_UPDATED]
    assert len(msgs) == 1
    payload = msgs[0]["payload"]
    assert payload["titan_id"] == "T1"
    assert payload["stats"]["novelty"] == pytest.approx(0.12)


def test_publish_prediction_generated_payload(state_refs, send_queue):
    """PREDICTION_GENERATED carries the counter + last_prediction surrogate
    for cognitive_worker's novelty consumer (per drift correction B8)."""
    srw._publish_prediction_generated(
        state_refs["prediction_engine"], state_refs, send_queue,
        "self_reflection_worker", "T1", total=100)
    msgs = [m for m in send_queue.items
            if m.get("type") == bus.PREDICTION_GENERATED]
    assert len(msgs) == 1
    p = msgs[0]["payload"]
    assert p["titan_id"] == "T1"
    assert p["total_predictions"] == 100
    assert p["total_surprises"] == 8
    assert p["last_prediction"] == [0.1, 0.2, 0.3]


# ── B8 — drift correction verification ──────────────────────────────────────


def test_cognitive_worker_no_longer_has_prediction_engine_in_state_refs():
    """Track 2 drift correction (commit B8): cognitive_worker must NOT have
    a 'prediction_engine' key in its state_refs anymore. Test parses
    the source of _init_cognitive_engines and asserts the key is absent."""
    import inspect
    from titan_hcl.modules import cognitive_worker
    src = inspect.getsource(cognitive_worker._init_cognitive_engines)
    # The string '"prediction_engine"' (as a dict key) must NOT appear in
    # the state_refs construction.
    assert '"prediction_engine":' not in src, (
        "cognitive_worker._init_cognitive_engines still has a prediction_engine "
        "entry in state_refs — Track 2 B8 drift correction incomplete")


def test_cognitive_worker_subscribes_to_prediction_generated():
    """B8 consumer side — cognitive_worker subscribes to PREDICTION_GENERATED
    bus event (replacing the in-process predict_next driver)."""
    from titan_hcl.modules.cognitive_worker import _COGNITIVE_WORKER_SUBSCRIBE_TOPICS
    assert bus.PREDICTION_GENERATED in _COGNITIVE_WORKER_SUBSCRIBE_TOPICS


def test_boot_driver_parity_no_longer_has_prediction_engine_row():
    """test_cognitive_worker_boot_driver_parity.BOOT_DRIVER_PARITY must NOT
    have prediction_engine row anymore (Track 2 B8 corrected)."""
    from tests.test_cognitive_worker_boot_driver_parity import BOOT_DRIVER_PARITY
    keys = {k for k, _ in BOOT_DRIVER_PARITY}
    assert "prediction_engine" not in keys, (
        "BOOT_DRIVER_PARITY still has prediction_engine — Track 2 B8 incomplete")


# ── SPEC parity ─────────────────────────────────────────────────────────────


def test_new_bus_constants_consistent_with_specs():
    """Every new bus constant ships in bus.py + bus_specs.py."""
    from titan_hcl.bus_specs import MSG_SPECS
    for const in ("SELF_REFLECTION_STATS_UPDATED", "SELF_REASONING_INSIGHT",
                  "CODING_EXPLORER_STATS_UPDATED", "CODING_INSIGHT",
                  "PREDICTION_STATS_UPDATED", "PREDICTION_GENERATED"):
        assert hasattr(bus, const), f"bus.{const} missing"
        assert getattr(bus, const) == const, f"bus.{const} value drift"
        assert const in MSG_SPECS, f"bus_specs.MSG_SPECS missing {const}"


def test_bus_specs_audit_clean():
    """audit_against_bus_constants + all_priorities_in_range clean after B7."""
    from titan_hcl.bus_specs import audit_against_bus_constants, all_priorities_in_range
    assert audit_against_bus_constants() == []
    assert all_priorities_in_range() == []


# NOTE: cache_key_registry was RETIRED in Phase D D-SPEC-80 along with the
# BusSubscriber + CachedState pipeline. self_reflection.state +
# coding_explorer.state + prediction.state are SHM-published; SHM-direct
# reads replace the cache key registry per Preamble G18.


def test_spec_version_at_or_above_1_3_1():
    """SPEC v1.3.1 — Track 2 originally bumped to v1.2.1 in chunk A1, but
    rebased to v1.3.1 at merge time because v1.3.0 (BUS_SUBSCRIBE multi-name
    semantics) shipped to titan-v6 in parallel and was merged first. See
    SPEC §21 Decision Log D-SPEC-38 + D-SPEC-39.

    Test refactored 2026-05-15 during v1.8.2 §4.I dream_state_worker carve —
    exact-match `== "1.3.1"` was too narrow once SPEC_VERSION advanced. The
    durable claim is "v1.3.1 shipped or later"; verify via SemVer-ordered
    comparison so subsequent bumps don't break this regression check.
    """
    from titan_hcl._phase_c_constants import SPEC_VERSION
    cur_tuple = tuple(int(p) for p in SPEC_VERSION.split("."))
    assert cur_tuple >= (1, 3, 1), (
        f"SPEC_VERSION {SPEC_VERSION} is below 1.3.1 — Track 2 D-SPEC-38/39 "
        f"baseline regression")
