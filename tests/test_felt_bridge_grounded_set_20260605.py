"""RFP_inner_outer_felt_teaching_bridge §7.2 — Phase 2: event-sourced CGN grounded-set.

Covers `titan_hcl/synthesis/felt_bridge.py` grounded-set:
  • record_grounded → is_object_grounded (the event→set path)
  • shared key-space normalization on both write + query
  • durability: a fresh FeltBridge boot-seeds the in-memory mirror from the table
  • idempotency (ON CONFLICT DO NOTHING)
  • G18: is_object_grounded reads the in-memory mirror ONLY — no DB/RPC at query time
  • soft-fail + empty-label guards

This is the seam that replaces cgn_bridge.ensure_grounded for the Object-level
grounding query. The producer (Phase 3) consumes is_object_grounded; the live event
source is cgn_worker's CGN_CONCEPT_GROUNDED (offline we drive record_grounded directly,
per the §7.2 data-flow).
"""
from __future__ import annotations

import duckdb
import pytest

from titan_hcl.synthesis.felt_bridge import FeltBridge


class _DirectWriter:
    def submit(self, fn):
        return fn()

    def submit_sync(self, fn):
        return fn()


@pytest.fixture()
def bridge():
    conn = duckdb.connect(":memory:")
    fb = FeltBridge(conn, _DirectWriter())
    assert fb.ensure_schema() is True
    return fb, conn


# ── event → set ───────────────────────────────────────────────────────────
def test_record_then_grounded(bridge):
    fb, _ = bridge
    assert fb.is_object_grounded("microbe") is False
    fb.record_grounded("microbe")
    assert fb.is_object_grounded("microbe") is True


def test_normalization_on_write_and_query(bridge):
    fb, _ = bridge
    fb.record_grounded("  Glacier   Microbe ")  # mixed case + extra whitespace
    assert fb.is_object_grounded("glacier microbe") is True
    assert fb.is_object_grounded("GLACIER  microbe") is True
    assert fb.is_object_grounded("glacier") is False  # different Object


def test_empty_label_noop(bridge):
    fb, _ = bridge
    fb.record_grounded("")
    fb.record_grounded("   ")
    assert fb.is_object_grounded("") is False


def test_idempotent_single_row(bridge):
    fb, conn = bridge
    fb.record_grounded("microbe")
    fb.record_grounded("Microbe")  # normalizes to same key → ON CONFLICT DO NOTHING
    rows = conn.execute(
        "SELECT object_label FROM cgn_grounded_objects").fetchall()
    assert rows == [("microbe",)]


# ── durability (boot-seed from the table) ─────────────────────────────────
def test_durability_boot_seed():
    conn = duckdb.connect(":memory:")
    fb1 = FeltBridge(conn, _DirectWriter())
    assert fb1.ensure_schema() is True
    fb1.record_grounded("microbe")
    fb1.record_grounded("altitude")
    # A *fresh* FeltBridge on the same durable store (simulates restart) must
    # rebuild its in-memory mirror from the table — no event replay, no RPC.
    fb2 = FeltBridge(conn, _DirectWriter())
    assert fb2.ensure_schema() is True
    assert fb2.is_object_grounded("microbe") is True
    assert fb2.is_object_grounded("altitude") is True
    assert fb2.is_object_grounded("glacier") is False


# ── G18: reads are in-memory only (no DB/RPC at query time) ───────────────
def test_g18_query_does_not_touch_writer():
    class _ReadFailWriter(_DirectWriter):
        def __init__(self):
            self.fail = False

        def submit_sync(self, fn):
            if self.fail:
                raise AssertionError("is_object_grounded must NOT read via the "
                                     "writer/DB at query time (G18)")
            return fn()

    w = _ReadFailWriter()
    fb = FeltBridge(duckdb.connect(":memory:"), w)
    assert fb.ensure_schema() is True
    fb.record_grounded("microbe")
    w.fail = True  # any submit_sync from here on is a G18 violation
    assert fb.is_object_grounded("microbe") is True   # served from in-memory mirror
    assert fb.is_object_grounded("ghost") is False


# ── soft-fail ─────────────────────────────────────────────────────────────
def test_record_grounded_soft_fail_no_raise():
    class _BrokenWriter:
        def submit(self, fn):
            raise RuntimeError("boom")

        def submit_sync(self, fn):
            raise RuntimeError("boom")

    fb = FeltBridge(duckdb.connect(":memory:"), _BrokenWriter())
    assert fb.ensure_schema() is False
    fb.record_grounded("microbe")              # no raise (in-memory add still works)
    assert fb.is_object_grounded("microbe") is True  # in-memory mirror updated
