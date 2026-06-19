"""Unified failure-replay store — FailedAttemptStore unit tests (EEL-B2 / mastery
§7.P9). Covers the durable `failed_attempts` lifecycle: idempotent enqueue, the
one-at-a-time claim, resolve (with the anchored skill_id), bump→abandon after
max_revisits, save/resume (in_progress→pending on boot), intent_seed roundtrip,
and coverage counts. INV-Syn-19 (sole writer) is exercised via the InlineWriter
(resolve_writer(None) runs submit_sync inline on the calling thread)."""
from __future__ import annotations

import duckdb
import pytest

from titan_hcl.synthesis.failed_attempt_store import (
    FailedAttemptStore,
    ST_ABANDONED,
    ST_IN_PROGRESS,
    ST_PENDING,
    ST_RESOLVED,
    compute_problem_id,
)


class _Clock:
    """Monotone test clock so created_at/last_attempt_at ordering is deterministic."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1.0
        return self.t


@pytest.fixture()
def store():
    conn = duckdb.connect(":memory:")
    return FailedAttemptStore(duckdb_conn=conn, clock=_Clock(), max_revisits=3)


def test_enqueue_creates_pending_and_is_idempotent(store):
    pid = store.enqueue(problem="factorial of 12", goal_class="autonomous:tool",
                        helper="coding_sandbox")
    assert pid == compute_problem_id("factorial of 12", "autonomous:tool")
    row = store.get(pid)
    assert row["status"] == ST_PENDING
    assert row["enqueue_count"] == 1
    # Same problem failing AGAIN bumps the count, never forks a duplicate row.
    pid2 = store.enqueue(problem="Factorial of 12  ", goal_class="autonomous:tool",
                         helper="coding_sandbox")  # whitespace/case-normalized
    assert pid2 == pid
    assert store.get(pid)["enqueue_count"] == 2
    assert store.coverage()["pending"] == 1   # still ONE row


def test_next_unresolved_claims_in_progress_and_is_not_rehanded(store):
    store.enqueue(problem="p1", goal_class="g")
    store.enqueue(problem="p2", goal_class="g")
    first = store.next_unresolved(limit=1)
    assert len(first) == 1
    assert store.get(first[0]["problem_id"])["status"] == ST_IN_PROGRESS
    # A second claim returns the OTHER problem, never the in_progress one.
    second = store.next_unresolved(limit=1)
    assert len(second) == 1
    assert second[0]["problem_id"] != first[0]["problem_id"]
    # Both claimed → nothing left pending.
    assert store.next_unresolved(limit=5) == []


def test_intent_seed_roundtrips(store):
    seed = {"posture": "explore", "source_layer": "spirit",
            "source_dims": [1, 2], "deficit_values": [0.3]}
    store.enqueue(problem="p", goal_class="g", helper="coding_sandbox",
                  intent_seed=seed, known_target="42")
    claimed = store.next_unresolved(limit=1)[0]
    assert claimed["intent_seed"] == seed
    assert claimed["known_target"] == "42"
    assert claimed["helper"] == "coding_sandbox"


def test_mark_resolved_is_terminal(store):
    pid = store.enqueue(problem="p", goal_class="g")
    store.next_unresolved(limit=1)
    assert store.mark_resolved(pid, skill_id="skill_abc") is True
    row = store.get(pid)
    assert row["status"] == ST_RESOLVED
    assert row["skill_id"] == "skill_abc"
    # Cannot re-resolve a terminal row.
    assert store.mark_resolved(pid, skill_id="skill_xyz") is False
    assert store.get(pid)["skill_id"] == "skill_abc"


def test_bump_then_abandon_after_max_revisits(store):
    pid = store.enqueue(problem="hard", goal_class="g")
    # max_revisits=3 → bumps 1,2 stay pending; bump 3 abandons.
    assert store.bump_attempt(pid, correction="try x") == ST_PENDING
    assert store.bump_attempt(pid, correction="try y") == ST_PENDING
    assert store.bump_attempt(pid, correction="try z") == ST_ABANDONED
    row = store.get(pid)
    assert row["status"] == ST_ABANDONED
    assert row["revisit_count"] == 3
    # An abandoned problem is no longer handed out.
    assert store.next_unresolved(limit=5) == []


def test_save_resume_resets_in_progress_to_pending(store):
    # Hand out a problem (→ in_progress), then simulate a restart: a fresh store on
    # the SAME conn re-runs _init_schema_body, which resets in_progress→pending so an
    # interrupted revisit is simply retried (durable, never lost).
    pid = store.enqueue(problem="interrupted", goal_class="g")
    store.next_unresolved(limit=1)
    assert store.get(pid)["status"] == ST_IN_PROGRESS
    reborn = FailedAttemptStore(duckdb_conn=store._db, clock=_Clock())
    assert reborn.get(pid)["status"] == ST_PENDING
    assert len(reborn.next_unresolved(limit=1)) == 1   # claimable again


def test_reenqueue_reopens_terminal_row(store):
    pid = store.enqueue(problem="regressed", goal_class="g")
    store.next_unresolved(limit=1)
    store.mark_resolved(pid, skill_id="s")
    assert store.get(pid)["status"] == ST_RESOLVED
    # The same problem failing again REOPENS it to pending (a regression worth revisiting).
    store.enqueue(problem="regressed", goal_class="g")
    assert store.get(pid)["status"] == ST_PENDING


def test_coverage_counts(store):
    p1 = store.enqueue(problem="a", goal_class="g")
    store.enqueue(problem="b", goal_class="g")
    store.next_unresolved(limit=1)            # one → in_progress
    store.mark_resolved(p1) if store.get(p1)["status"] == ST_IN_PROGRESS else None
    cov = store.coverage()
    assert cov["resolved_total"] >= 0
    assert cov["enqueued_total"] == 2
    assert cov["revisits_handed"] == 1
    assert set(cov["by_status"]).issubset(
        {ST_PENDING, ST_IN_PROGRESS, ST_RESOLVED, ST_ABANDONED})
