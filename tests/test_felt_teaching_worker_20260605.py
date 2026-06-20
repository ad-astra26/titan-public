"""RFP_inner_outer_felt_teaching_bridge §7.4 — Phase 4: the felt_teaching consumer.

Covers `titan_hcl/modules/felt_teaching_worker.py`:
  • felt_evidence_strength (Q3 reward) — mean lived-neuromod magnitude × recurrence,
    bounded [0,1]
  • _FeltTeachingStore — the worker's OWN store (status lifecycle, dedup, recurrence,
    mark_matured); separate from synthesis.duckdb (G21)
  • LanguageTeacher.build_felt_perturbation — prompt-build only
  • _process_candidate — drives CGN via record_felt_experience (PROPOSE-ONLY, a complete
    value-net "experience" transition; INV-Syn-ENG-4); status lifecycle; frame_dependent
    when the Object is already in the grounded-view (F⊗C, never overwrite); dedup against
    the own store; reward bounded

The worker drives CGN through CGNConsumerClient.record_felt_experience / emit_cross_insight
ONLY (the fakes expose exactly those) — proving propose-only by construction.
"""
from __future__ import annotations

import json

import pytest

from titan_hcl.modules.felt_teaching_worker import (
    _FeltTeachingStore,
    _process_candidate,
    felt_evidence_strength,
    _felt_magnitude,
)
from titan_hcl.logic.language_teacher import LanguageTeacher


class _FakeClient:
    """CGN consumer surface the worker is allowed to use — record_felt_experience
    (complete value-net "experience" transition; INV-Syn-ENG-4) + emit_cross_insight
    (peer learning). NO grounding-write method exists, so any non-propose-only attempt
    would AttributeError (propose-only by construction)."""

    def __init__(self):
        self.experiences = []
        self.insights = []

    def record_felt_experience(self, *, concept_id, neuromods, reward,
                               action=0, encounter_type="teaching",
                               outcome_context=None, metadata=None):
        # Mirror the production capture: (concept_id, reward, outcome_context) is
        # the propose-only evidence; neuromods builds the 30D state internally.
        self.experiences.append((concept_id, reward, outcome_context or {},
                                 neuromods or {}))

    def emit_cross_insight(self, reward, ctx=None):
        self.insights.append((reward, ctx or {}))
        return True


@pytest.fixture()
def store(tmp_path):
    return _FeltTeachingStore(str(tmp_path / "ft.duckdb"))


def _payload(label="microbe", felt=None, engram="glacier_x", version=2,
             domain="biology"):
    return {
        "object_label": label,
        "felt_state": felt if felt is not None else {"curiosity": 0.7, "awe": 0.4},
        "source_engram": engram, "source_version": version, "domain_hint": domain,
    }


# ── reward ─────────────────────────────────────────────────────────────────
def test_felt_magnitude_range():
    assert _felt_magnitude({}) == 0.0
    assert _felt_magnitude({"curiosity": 0.5, "awe": 0.5}) == 0.0  # at centre
    m = _felt_magnitude({"curiosity": 1.0, "awe": 0.0})
    assert 0.0 < m <= 1.0


def test_felt_evidence_strength_bounded_and_scales():
    felt = {"curiosity": 0.9, "awe": 0.1}
    r1 = felt_evidence_strength(felt, recurrence=1)
    r3 = felt_evidence_strength(felt, recurrence=3)
    r9 = felt_evidence_strength(felt, recurrence=9)
    assert 0.0 <= r1 <= 1.0 and 0.0 <= r9 <= 1.0
    assert r1 < r3                 # recurrence increases evidence...
    assert r3 == r9                # ...capped at the recurrence norm
    assert felt_evidence_strength({}, recurrence=5) == 0.0  # no felt → no evidence


# ── the worker's own store ────────────────────────────────────────────────
def test_store_lifecycle(store):
    assert store.status_of("microbe", "e1", 1) is None
    assert store.recurrence("microbe") == 0
    store.upsert(label="microbe", engram="e1", version=1, felt_json="{}",
                 domain_hint="biology", reward=0.2, status="grounding")
    store.upsert(label="microbe", engram="e2", version=1, felt_json="{}",
                 domain_hint="biology", reward=0.3, status="grounding")
    assert store.status_of("microbe", "e1", 1) == "grounding"
    assert store.recurrence("microbe") == 2          # distinct source engrams
    store.mark_matured("microbe")
    assert store.status_of("microbe", "e1", 1) == "matured"


# ── perturbation prompt ───────────────────────────────────────────────────
def test_build_felt_perturbation_shape():
    spec = LanguageTeacher.build_felt_perturbation(
        "microbe", {"curiosity": 0.7}, "biology")
    assert set(spec) == {"system", "prompt", "max_tokens"}
    assert "microbe" in spec["prompt"]
    assert "biology" in spec["prompt"]
    assert "curiosity=0.70" in spec["prompt"]


# ── _process_candidate ────────────────────────────────────────────────────
def test_process_grounding_drives_cgn(store):
    client = _FakeClient()
    _process_candidate(_payload(), store, client, LanguageTeacher(), None, set())
    assert len(client.experiences) == 1
    concept_id, reward, ctx, _nm = client.experiences[0]
    assert concept_id == "microbe"
    assert 0.0 <= reward <= 1.0
    assert ctx["source_engram"] == "glacier_x" and ctx["source_version"] == 2
    assert ctx["domain_hint"] == "biology"
    assert "frame" not in ctx                       # not frame_dependent
    assert len(client.insights) == 1               # peer cross-insight emitted
    assert store.status_of("microbe", "glacier_x", 2) == "grounding"


def test_process_frame_dependent_scopes_to_frame(store):
    client = _FakeClient()
    grounded_view = {"microbe"}                     # already grounded (race/stale gap)
    _process_candidate(_payload(), store, client, LanguageTeacher(), None,
                       grounded_view)
    _cid, _r, ctx, _nm = client.experiences[0]
    assert ctx["frame"] == "biology"               # F⊗C — domain-scoped, never base
    assert store.status_of("microbe", "glacier_x", 2) == "frame_dependent"


def test_process_dedups_against_own_store(store):
    client = _FakeClient()
    _process_candidate(_payload(), store, client, LanguageTeacher(), None, set())
    _process_candidate(_payload(), store, client, LanguageTeacher(), None, set())
    assert len(client.experiences) == 1            # 2nd is skipped (already grounding)


def test_process_empty_label_noop(store):
    client = _FakeClient()
    _process_candidate(_payload(label="   "), store, client, LanguageTeacher(),
                       None, set())
    assert client.experiences == []


def test_process_reward_persisted_matches(store):
    client = _FakeClient()
    _process_candidate(_payload(), store, client, LanguageTeacher(), None, set())
    _cid, reward, _ctx, _nm = client.experiences[0]
    row = store._conn.execute(
        "SELECT reward, felt_state_json FROM felt_candidates").fetchone()
    assert row[0] == reward
    assert json.loads(row[1]) == {"awe": 0.4, "curiosity": 0.7}
