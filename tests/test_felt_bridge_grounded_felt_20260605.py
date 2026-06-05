"""Phase B — felt centroid on the CGN grounded-set (RFP_cgn_felt_state_exposure §7.B).

Covers `synthesis/felt_bridge.py`:
  • record_grounded(label, felt_centroid) → grounded_felt(label) (store + read the centroid)
  • backward-compat: label-only emit → grounded_felt None, is_object_grounded still True
  • felt_json column migration onto a pre-existing (pre-felt) table
  • durability: a fresh FeltBridge boot-seeds the felt mirror from felt_json (G2)
  • centroid refresh on re-emit (ON CONFLICT DO UPDATE)
  • G18: grounded_felt reads the in-memory mirror ONLY — no DB/RPC at query time

Run: python -m pytest tests/test_felt_bridge_grounded_felt_20260605.py -v -p no:anchorpy
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


# ── store + read the centroid ───────────────────────────────────────────────
def test_record_with_felt_then_grounded_felt(bridge):
    fb, _ = bridge
    fb.record_grounded("microbe", felt_centroid={"DA": 0.55, "NE": 0.45})
    assert fb.is_object_grounded("microbe") is True
    assert fb.grounded_felt("microbe") == {"DA": 0.55, "NE": 0.45}


def test_label_normalized_for_felt(bridge):
    fb, _ = bridge
    fb.record_grounded("  Glacier  Microbe ", felt_centroid={"DA": 0.5})
    assert fb.grounded_felt("glacier microbe") == {"DA": 0.5}


def test_felt_json_persisted_in_column(bridge):
    fb, conn = bridge
    fb.record_grounded("microbe", felt_centroid={"DA": 0.55})
    row = conn.execute(
        "SELECT felt_json FROM cgn_grounded_objects WHERE object_label='microbe'"
    ).fetchone()
    assert row is not None and '"DA": 0.55' in row[0]


# ── backward compatibility (pre-felt emitters) ──────────────────────────────
def test_label_only_grounded_felt_is_none(bridge):
    fb, _ = bridge
    fb.record_grounded("microbe")  # no felt_centroid (old emitter / no felt yet)
    assert fb.is_object_grounded("microbe") is True
    assert fb.grounded_felt("microbe") is None


def test_empty_felt_treated_as_none(bridge):
    fb, _ = bridge
    fb.record_grounded("microbe", felt_centroid={})
    assert fb.is_object_grounded("microbe") is True
    assert fb.grounded_felt("microbe") is None


def test_grounded_felt_none_when_ungrounded(bridge):
    fb, _ = bridge
    assert fb.grounded_felt("ghost") is None
    assert fb.grounded_felt("") is None


# ── refresh on re-emit ──────────────────────────────────────────────────────
def test_centroid_refreshes_on_conflict(bridge):
    fb, conn = bridge
    fb.record_grounded("microbe", felt_centroid={"DA": 0.5})
    fb.record_grounded("microbe", felt_centroid={"DA": 0.8})  # re-emit → refresh
    assert fb.grounded_felt("microbe") == {"DA": 0.8}
    rows = conn.execute(
        "SELECT object_label FROM cgn_grounded_objects").fetchall()
    assert rows == [("microbe",)]  # still one row


def test_relabel_only_does_not_wipe_existing_felt(bridge):
    # A later label-only emit (no felt) must NOT clobber a stored centroid.
    fb, _ = bridge
    fb.record_grounded("microbe", felt_centroid={"DA": 0.6})
    fb.record_grounded("microbe")  # label-only → ON CONFLICT DO NOTHING (keeps felt)
    assert fb.grounded_felt("microbe") == {"DA": 0.6}


# ── migration onto a pre-existing (pre-felt) table ──────────────────────────
def test_migration_adds_felt_column():
    conn = duckdb.connect(":memory:")
    # Simulate a table created before the felt column existed.
    conn.execute(
        "CREATE TABLE cgn_grounded_objects ("
        " object_label VARCHAR PRIMARY KEY, first_seen_ts DOUBLE DEFAULT 0)")
    conn.execute(
        "INSERT INTO cgn_grounded_objects (object_label, first_seen_ts) "
        "VALUES ('legacy', 1.0)")
    fb = FeltBridge(conn, _DirectWriter())
    assert fb.ensure_schema() is True  # ALTER ADD COLUMN IF NOT EXISTS
    # legacy row grounded but felt-less; new row carries a centroid.
    assert fb.is_object_grounded("legacy") is True
    assert fb.grounded_felt("legacy") is None
    fb.record_grounded("microbe", felt_centroid={"DA": 0.5})
    assert fb.grounded_felt("microbe") == {"DA": 0.5}


# ── durability: boot-seed the felt mirror from felt_json (G2) ────────────────
def test_durability_boot_seed_restores_felt_mirror():
    conn = duckdb.connect(":memory:")
    fb1 = FeltBridge(conn, _DirectWriter())
    assert fb1.ensure_schema() is True
    fb1.record_grounded("microbe", felt_centroid={"DA": 0.55, "NE": 0.45})
    fb1.record_grounded("altitude")  # label-only

    fb2 = FeltBridge(conn, _DirectWriter())  # simulates restart
    assert fb2.ensure_schema() is True
    assert fb2.is_object_grounded("microbe") is True
    assert fb2.grounded_felt("microbe") == {"DA": 0.55, "NE": 0.45}  # restored
    assert fb2.is_object_grounded("altitude") is True
    assert fb2.grounded_felt("altitude") is None  # was felt-less


# ── G18: grounded_felt reads in-memory only ─────────────────────────────────
def test_g18_grounded_felt_no_writer_at_query():
    class _ReadFailWriter(_DirectWriter):
        def __init__(self):
            self.fail = False

        def submit_sync(self, fn):
            if self.fail:
                raise AssertionError("grounded_felt must NOT read via the writer/DB "
                                     "at query time (G18)")
            return fn()

    w = _ReadFailWriter()
    fb = FeltBridge(duckdb.connect(":memory:"), w)
    assert fb.ensure_schema() is True
    fb.record_grounded("microbe", felt_centroid={"DA": 0.5})
    w.fail = True  # any submit_sync from here is a G18 violation
    assert fb.grounded_felt("microbe") == {"DA": 0.5}  # served from in-memory mirror
    assert fb.grounded_felt("ghost") is None
