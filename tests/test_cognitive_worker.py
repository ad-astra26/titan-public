"""Unit tests for cognitive_worker.py — chunks 8E + 8F + 8G + 8H.

Covers:
  - SPEC §8.5 src-disambiguation: 3 trinity event types × payload.src ∈
    {inner, outer} → 6 first-class internal cache slots (G1 doctrinal
    symmetry).
  - Dispatcher wiring (PLAN §3.1): trinity, dream consolidate, stimulus,
    meditation, save_now handlers route to the correct engine methods.
  - Subscribe topics list matches PLAN §11 acceptance criterion #3.
  - Cognitive epoch constants (SPEC v0.2.0 [domains.COGNITIVE]) match
    expected values per Maker D4 (a) Schumann × {1, 9, 27} multiples.
  - Engine init shape (state_refs dict canonical keys).
  - Persist + neuromod reader graceful-failure (return None on missing
    deps; don't raise).

Per PLAN §7.1 — these are the unit tests. Per PLAN §7.2 + §7.6 the
integration tests (consciousness_epoch real subprocess + guardian
restart) need live engines + 30+s runtime; they live in chunk 8J's
companion files OR are verified at chunk 8L T3 deploy gate.

These tests do NOT spawn subprocesses or require an active titan_HCL.
They exercise pure-Python helpers with in-memory state.
"""
from __future__ import annotations

from queue import Queue

import pytest


# ── SPEC §8.5 src-disambiguation (the core 4B contract) ─────────────


class TestTrinityStateDispatcher:
    """SPEC §8.5: 3 event types × payload.src ∈ {inner, outer} → 6 cache slots."""

    def setup_method(self):
        from titan_hcl.modules.cognitive_worker import _dispatch_trinity_state
        self.dispatch = _dispatch_trinity_state
        self.state_refs = {
            "_inner_body_state": [0.5] * 5,
            "_outer_body_state": [0.5] * 5,
            "_inner_mind_state": [0.5] * 15,
            "_outer_mind_state": [0.5] * 15,
            "_inner_spirit_state": [0.5] * 45,
            "_outer_spirit_state": [0.5] * 45,
        }

    def test_body_state_src_inner_routes_to_inner_slot(self):
        self.dispatch(
            self.state_refs,
            {"src": "inner", "values": [0.1, 0.2, 0.3, 0.4, 0.5]},
            dim=5, inner_key="_inner_body_state",
            outer_key="_outer_body_state", type_label="BODY_STATE")
        assert self.state_refs["_inner_body_state"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        # outer must not change — G1 doctrinal symmetry preserved
        assert self.state_refs["_outer_body_state"] == [0.5] * 5

    def test_body_state_src_outer_routes_to_outer_slot(self):
        self.dispatch(
            self.state_refs,
            {"src": "outer", "values": [0.9, 0.8, 0.7, 0.6, 0.5]},
            dim=5, inner_key="_inner_body_state",
            outer_key="_outer_body_state", type_label="BODY_STATE")
        assert self.state_refs["_outer_body_state"] == [0.9, 0.8, 0.7, 0.6, 0.5]
        # inner must not change
        assert self.state_refs["_inner_body_state"] == [0.5] * 5

    def test_inner_outer_independent_under_concurrent_updates(self):
        """Per SPEC §8.5 coalesce=("src","type"): inner-BODY and outer-BODY
        occupy SEPARATE coalesce slots and both survive backpressure."""
        self.dispatch(
            self.state_refs,
            {"src": "inner", "values": [0.1, 0.1, 0.1, 0.1, 0.1]},
            dim=5, inner_key="_inner_body_state",
            outer_key="_outer_body_state", type_label="BODY_STATE")
        self.dispatch(
            self.state_refs,
            {"src": "outer", "values": [0.9, 0.9, 0.9, 0.9, 0.9]},
            dim=5, inner_key="_inner_body_state",
            outer_key="_outer_body_state", type_label="BODY_STATE")
        # Both survived — neither overwrote the other
        assert self.state_refs["_inner_body_state"] == [0.1] * 5
        assert self.state_refs["_outer_body_state"] == [0.9] * 5

    def test_missing_src_defaults_to_inner_legacy_compat(self):
        """Legacy 67D-only consciousness epoch publisher had no src field;
        default to inner so we don't drop tensors."""
        self.dispatch(
            self.state_refs,
            {"values": [0.0, 0.1, 0.2, 0.3, 0.4]},
            dim=5, inner_key="_inner_body_state",
            outer_key="_outer_body_state", type_label="BODY_STATE")
        assert self.state_refs["_inner_body_state"] == [0.0, 0.1, 0.2, 0.3, 0.4]

    def test_bad_payload_wrong_dim_preserves_prior_value(self):
        """Defense: malformed payload doesn't blank cache slot."""
        self.state_refs["_inner_body_state"] = [0.42] * 5
        self.dispatch(
            self.state_refs,
            {"src": "inner", "values": [0.0, 0.1]},  # too short
            dim=5, inner_key="_inner_body_state",
            outer_key="_outer_body_state", type_label="BODY_STATE")
        assert self.state_refs["_inner_body_state"] == [0.42] * 5

    def test_bad_payload_no_values_preserves_prior_value(self):
        self.state_refs["_inner_body_state"] = [0.7] * 5
        self.dispatch(
            self.state_refs,
            {"src": "inner"},  # no values key
            dim=5, inner_key="_inner_body_state",
            outer_key="_outer_body_state", type_label="BODY_STATE")
        assert self.state_refs["_inner_body_state"] == [0.7] * 5

    def test_mind_state_15d_dispatch(self):
        values_15 = [round(0.01 * i, 2) for i in range(15)]
        self.dispatch(
            self.state_refs,
            {"src": "outer", "values": values_15},
            dim=15, inner_key="_inner_mind_state",
            outer_key="_outer_mind_state", type_label="MIND_STATE")
        assert self.state_refs["_outer_mind_state"] == values_15
        assert len(self.state_refs["_outer_mind_state"]) == 15

    def test_spirit_state_45d_dispatch(self):
        values_45 = [round(0.001 * i, 3) for i in range(45)]
        self.dispatch(
            self.state_refs,
            {"src": "inner", "values": values_45},
            dim=45, inner_key="_inner_spirit_state",
            outer_key="_outer_spirit_state", type_label="SPIRIT_STATE")
        assert self.state_refs["_inner_spirit_state"] == values_45
        assert len(self.state_refs["_inner_spirit_state"]) == 45


# ── PLAN §3.1 driver table — non-trinity dispatchers ────────────────


class TestNonTrinityDispatchers:
    """Dispatchers for non-trinity events (PLAN §3.1)."""

    def test_dispatch_dream_consolidate_calls_coordinator_dreaming(self):
        from titan_hcl.modules.cognitive_worker import _dispatch_dream_consolidate

        class FakeDreaming:
            def __init__(self):
                self.calls = []
            def consolidate_pending(self, payload):
                self.calls.append(payload)

        class FakeCoordinator:
            def __init__(self):
                self.dreaming = FakeDreaming()

        coord = FakeCoordinator()
        state_refs = {"coordinator": coord}
        _dispatch_dream_consolidate(state_refs, {"insight": "test"})
        assert coord.dreaming.calls == [{"insight": "test"}]

    def test_dispatch_dream_consolidate_no_coordinator_is_noop(self):
        from titan_hcl.modules.cognitive_worker import _dispatch_dream_consolidate
        state_refs = {"coordinator": None}
        _dispatch_dream_consolidate(state_refs, {"foo": "bar"})  # no raise

    def test_dispatch_stimulus_calls_reasoning_observe(self):
        from titan_hcl.modules.cognitive_worker import _dispatch_stimulus
        from titan_hcl import bus

        class FakeReasoning:
            def __init__(self):
                self.observed = []
            def observe_stimulus(self, payload):
                self.observed.append(payload)

        re = FakeReasoning()
        state_refs = {"reasoning_engine": re}
        _dispatch_stimulus(state_refs, bus.CONVERSATION_STIMULUS,
                           {"text": "hello"})
        assert len(re.observed) == 1
        # Source tag should be derived from msg type
        assert re.observed[0].get("source") == "conversation"
        assert re.observed[0].get("text") == "hello"

    def test_dispatch_stimulus_experience_source_tag(self):
        from titan_hcl.modules.cognitive_worker import _dispatch_stimulus
        from titan_hcl import bus

        class FakeReasoning:
            def __init__(self):
                self.observed = []
            def observe_stimulus(self, payload):
                self.observed.append(payload)

        re = FakeReasoning()
        state_refs = {"reasoning_engine": re}
        _dispatch_stimulus(state_refs, bus.EXPERIENCE_STIMULUS, {"replay": True})
        assert re.observed[0].get("source") == "experience"

    def test_dispatch_meditation_complete_calls_coordinator(self):
        from titan_hcl.modules.cognitive_worker import _dispatch_meditation_complete

        class FakeCoordinator:
            def __init__(self):
                self.observed = []
            def meditation_observe(self, payload):
                self.observed.append(payload)

        coord = FakeCoordinator()
        state_refs = {"coordinator": coord}
        _dispatch_meditation_complete(state_refs, {"phase": "deep"})
        assert coord.observed == [{"phase": "deep"}]


# ── Persistence ─────────────────────────────────────────────────────


class TestPersistence:
    """_persist_engine_state graceful failure."""

    def test_persist_no_engines_is_noop(self):
        from titan_hcl.modules.cognitive_worker import _persist_engine_state
        _persist_engine_state({})  # no raise

    def test_persist_calls_each_engine_save(self):
        from titan_hcl.modules.cognitive_worker import _persist_engine_state

        calls = []

        class FakeReasoning:
            def save_state(self): calls.append("reasoning")

        class FakePiMonitor:
            def _save_state(self): calls.append("pi_monitor")

        class FakeNS:
            def save_all(self): calls.append("ns")

        class FakeDreaming:
            def _persist(self): calls.append("dreaming")

        class FakeCoordinator:
            def __init__(self):
                self.dreaming = FakeDreaming()

        state_refs = {
            "reasoning_engine": FakeReasoning(),
            "pi_monitor": FakePiMonitor(),
            "neural_nervous_system": FakeNS(),
            "coordinator": FakeCoordinator(),
        }
        _persist_engine_state(state_refs)
        assert set(calls) == {"reasoning", "pi_monitor", "ns", "dreaming"}

    def test_persist_isolates_failures(self):
        """One engine raising on save shouldn't block the others."""
        from titan_hcl.modules.cognitive_worker import _persist_engine_state

        calls = []

        class FakeReasoning:
            def save_state(self):
                raise RuntimeError("disk full")

        class FakePiMonitor:
            def _save_state(self): calls.append("pi_monitor")

        class FakeNS:
            def save_all(self): calls.append("ns")

        state_refs = {
            "reasoning_engine": FakeReasoning(),
            "pi_monitor": FakePiMonitor(),
            "neural_nervous_system": FakeNS(),
        }
        _persist_engine_state(state_refs)
        # pi_monitor + ns still saved despite reasoning failure
        assert "pi_monitor" in calls
        assert "ns" in calls


# ── SPEC v0.2.0 [domains.COGNITIVE] constants ──────────────────────


class TestCognitiveConstants:
    """Verify chunk 8B constants match SPEC v0.2.0 [domains.COGNITIVE].

    Per Maker D4 (a) the integer multiples of Schumann body period
    (1.15s) define the floor / default / ceiling of the adaptive
    epoch driver. Drift here = rebuild-Rust + Python out-of-sync.
    """

    def test_min_interval_is_1x_schumann_body(self):
        from titan_hcl._phase_c_constants import COGNITIVE_EPOCH_MIN_INTERVAL_S
        assert COGNITIVE_EPOCH_MIN_INTERVAL_S == 1.15  # 1× Schumann body

    def test_default_interval_is_9x_schumann_body(self):
        from titan_hcl._phase_c_constants import COGNITIVE_EPOCH_DEFAULT_INTERVAL_S
        assert COGNITIVE_EPOCH_DEFAULT_INTERVAL_S == 10.35  # 9× Schumann body

    def test_max_interval_is_27x_schumann_body(self):
        from titan_hcl._phase_c_constants import COGNITIVE_EPOCH_MAX_INTERVAL_S
        assert COGNITIVE_EPOCH_MAX_INTERVAL_S == 31.05  # 27× Schumann body

    def test_persist_every_n_epochs(self):
        from titan_hcl._phase_c_constants import COGNITIVE_PERSIST_EVERY_N_EPOCHS
        assert COGNITIVE_PERSIST_EVERY_N_EPOCHS == 100

    def test_min_default_max_are_ordered(self):
        from titan_hcl._phase_c_constants import (
            COGNITIVE_EPOCH_MIN_INTERVAL_S,
            COGNITIVE_EPOCH_DEFAULT_INTERVAL_S,
            COGNITIVE_EPOCH_MAX_INTERVAL_S,
        )
        assert (COGNITIVE_EPOCH_MIN_INTERVAL_S
                < COGNITIVE_EPOCH_DEFAULT_INTERVAL_S
                < COGNITIVE_EPOCH_MAX_INTERVAL_S)


# ── Subscribe topics list (PLAN §11 acceptance #3) ──────────────────


class TestSubscribeTopics:
    """PLAN §11 acceptance criterion #3: cognitive_worker subscribes to
    exactly these bus topics. Drift here breaks the journalctl
    assertion at chunk 8L T3 deploy gate.

    Topic-count drift history:
      - chunk 8E (v0.1.8): 10 topics (PLAN §11 canonical)
      - Track 2 (v1.2.1): +2 → 12 (ADVISOR_REFRACTORY_STATE, PREDICTION_GENERATED)
      - §4.B Track 3 (v1.7.4 D-SPEC-53): +2 → 14 (SPEAK_REQUEST_PENDING, NS_REWARD)
      - §4.Q (v1.8.0 D-SPEC-54): +4 → 18 (PREDICTION_STATS_UPDATED,
        EXPRESSION_COMPOSITES_UPDATED, KIN_SIGNATURE_UPDATED, NEUROMOD_STATS_UPDATED)
      - RFP_meta-reasoning_CGN_FIX Chunk B.7b: +5 → 23 (CGN_KNOWLEDGE_REQ,
        META_REASON_REQUEST, META_REASON_OUTCOME, CGN_KNOWLEDGE_RESP,
        TIMECHAIN_QUERY_RESP — MetaService relocated to cognitive_worker)
      - §4.I (v1.8.2 D-SPEC-56): +1 → 24 (DREAM_WAKE_FORWARD —
        dream_state_worker forwards wake requests here)
    """

    def test_subscribe_topics_list_canonical(self):
        from titan_hcl.modules.cognitive_worker import _COGNITIVE_WORKER_SUBSCRIBE_TOPICS
        from titan_hcl import bus

        expected = {
            # Chunk 8E canonical (PLAN §11):
            bus.BODY_STATE,                # 5D × src ∈ {inner, outer}
            bus.MIND_STATE,                # 15D × src ∈ {inner, outer}
            bus.SPIRIT_STATE,              # 45D × src ∈ {inner, outer}
            bus.KERNEL_EPOCH_TICK,         # circadian phase
            bus.CGN_DREAM_CONSOLIDATE,     # dream consolidation trigger
            bus.CONVERSATION_STIMULUS,     # reasoning input (chat)
            bus.EXPERIENCE_STIMULUS,       # reasoning input (experience replay)
            bus.MEDITATION_COMPLETE,       # meditation phase tracking
            bus.MODULE_SHUTDOWN,           # lifecycle
            bus.SAVE_NOW,                  # B.1 persistence trigger
            # Track 2 (v1.2.1) — SPEAK gate cache feed from
            # outer_interface_worker. Per SPEC §8.5 D-SPEC-38.
            bus.ADVISOR_REFRACTORY_STATE,  # SPEAK refractory gate
            # Track 2 (v1.2.1 commit B8) — prediction_engine drift correction.
            bus.PREDICTION_GENERATED,
            # §4.B Track 3 (v1.7.4 D-SPEC-53) — expression_worker bridges.
            bus.SPEAK_REQUEST_PENDING,
            bus.NS_REWARD,
            # §4.Q (v1.8.0 D-SPEC-54) — neuromod_inputs.bin builder feeds.
            bus.PREDICTION_STATS_UPDATED,
            bus.EXPRESSION_COMPOSITES_UPDATED,
            bus.KIN_SIGNATURE_UPDATED,
            bus.NEUROMOD_STATS_UPDATED,
            # RFP_meta-reasoning_CGN_FIX Chunk B.7b — MetaService relocation.
            bus.CGN_KNOWLEDGE_REQ,
            bus.META_REASON_REQUEST,
            bus.META_REASON_OUTCOME,
            bus.CGN_KNOWLEDGE_RESP,
            bus.TIMECHAIN_QUERY_RESP,
            # §4.I (v1.8.2 D-SPEC-56) — dream_state_worker → cognitive_worker.
            bus.DREAM_WAKE_FORWARD,
            # §4.I post-cleanup — FORCE_DREAM_REQUEST orphan handler closure.
            bus.FORCE_DREAM_REQUEST,
            # D-SPEC-64 v1.10.0 PLAN §1.6 — kin_resonance catalyst emit
            # (D8-3 catalyst-producer site #7 closure).
            bus.KIN_SIGNAL,
            # D-SPEC-103 (v1.41.0) — Record stage of the ExperienceOrchestrator
            # loop; _dispatch_experience_record (dispatcher handler at
            # cognitive_worker.py msg_type == bus.EXPERIENCE_RECORD).
            bus.EXPERIENCE_RECORD,
            # rFP_subsystem_reward_refresh_restore — CONTRACT_LIST_RESP feeds
            # meta_engine.update_subsystem_cache (dispatcher handler present).
            bus.CONTRACT_LIST_RESP,
            # D-SPEC-116 (Phase D) — spirit_worker retirement re-homed 3 flows:
            bus.MEMORY_RECALL_PERTURBATION,  # → msl.i_depth + working_mem.attend
            bus.TEACHER_SIGNALS,             # → msl.concept_grounder + neuromod nudge
            bus.OUTER_OBSERVATION,           # → msl.signal_engagement
        }
        actual = set(_COGNITIVE_WORKER_SUBSCRIBE_TOPICS)
        assert actual == expected, (
            f"Subscribe topics drifted from PLAN §11(3) + SPEC §8.5 D-SPEC-38 "
            f"+ §4.B/§4.Q/§4.I/Meta-Reasoning/D-SPEC-64 lineage. "
            f"Missing: {expected - actual}; Extra: {actual - expected}"
        )

    def test_subscribe_topics_count_is_thirty_one(self):
        """Topic-count walk: 10 → 12 (Track 2) → 14 (§4.B Track 3)
        → 18 (§4.Q) → 23 (Meta-Reasoning B.7b) → 24 (§4.I dream_state)
        → 25 (§4.I FORCE_DREAM_REQUEST orphan closure) → 26 (D-SPEC-64
        v1.10.0 KIN_SIGNAL for #7 kin_resonance migration) → 27 (D-SPEC-103
        EXPERIENCE_RECORD) → 28 (rFP_subsystem_reward_refresh CONTRACT_LIST_RESP)
        → 31 (D-SPEC-116 spirit_worker retirement: MEMORY_RECALL_PERTURBATION +
        TEACHER_SIGNALS + OUTER_OBSERVATION re-homed)."""
        from titan_hcl.modules.cognitive_worker import _COGNITIVE_WORKER_SUBSCRIBE_TOPICS
        assert len(_COGNITIVE_WORKER_SUBSCRIBE_TOPICS) == 31


# ── Neuromod shm reader factory ─────────────────────────────────────


class TestNeuromodReader:
    """_make_neuromod_reader graceful failure."""

    def test_reader_returns_callable_or_none(self):
        from titan_hcl.modules.cognitive_worker import _make_neuromod_reader
        reader = _make_neuromod_reader()
        # Either a callable (shm available) or None (shm disabled / no Rust)
        assert reader is None or callable(reader)

    def test_reader_does_not_raise(self):
        """Even when shm slot isn't initialized, factory must not raise."""
        from titan_hcl.modules.cognitive_worker import _make_neuromod_reader
        try:
            _make_neuromod_reader()
        except Exception as e:
            pytest.fail(f"_make_neuromod_reader raised: {e}")
