"""SpiritState.snapshot() msgpack-round-trip safety.

Regression test for the 2026-04-28 malformed-frame incident, where Fix B's
hex-dump diagnostic in titan_plugin/core/bus_socket.py caught spirit's
RESPONSE payloads being unpackable with `strict_map_key=True` because the
nested `topology["distance_matrix"]` dict used tuple keys (msgpack
serializes tuples as arrays — "list as map key" — which strict unpack
rejects).

Fix: SpiritState.snapshot() now sanitizes dict keys recursively.
"""
from __future__ import annotations

import msgpack
import pytest

from titan_plugin.logic.spirit_state import SpiritState


def _pack_then_strict_unpack(snap: dict) -> dict:
    """Helper: pack with default msgpack, unpack with broker's strict mode."""
    packed = msgpack.packb(snap, use_bin_type=True)
    return msgpack.unpackb(packed, strict_map_key=True, raw=False)


def test_empty_state_round_trips():
    s = SpiritState()
    snap = s.snapshot()
    out = _pack_then_strict_unpack(snap)
    assert out["assembly_count"] == 0


def test_topology_with_tuple_keys_round_trips():
    """Reproduces the 2026-04-28 bug — without the fix this raises:
    'list is not allowed for map key when strict_map_key=True'."""
    s = SpiritState()
    s.topology = {
        "distance_matrix": {
            ("a", "b"): 0.5,
            ("a", "c"): 0.3,
            ("b", "c"): 0.7,
        },
        "great_pulse": {"phase": "rest"},
    }
    snap = s.snapshot()
    out = _pack_then_strict_unpack(snap)
    # Tuple keys converted to "a:b" string form (same as inner_coordinator.get_stats)
    assert "a:b" in out["topology"]["distance_matrix"]
    assert out["topology"]["distance_matrix"]["a:b"] == 0.5
    assert out["topology"]["great_pulse"]["phase"] == "rest"


def test_observables_with_nested_tuple_keys_round_trips():
    s = SpiritState()
    s.observables = {
        "outer_body": {
            "neighbors": {("body", "mind"): 0.8, ("body", "spirit"): 0.6},
            "coherence": 0.9,
        }
    }
    snap = s.snapshot()
    out = _pack_then_strict_unpack(snap)
    assert "body:mind" in out["observables"]["outer_body"]["neighbors"]


def test_topology_with_3tuple_keys_round_trips():
    s = SpiritState()
    s.topology = {
        "triplet_distances": {
            ("a", "b", "c"): 1.5,
            ("x", "y", "z"): 2.5,
        }
    }
    snap = s.snapshot()
    out = _pack_then_strict_unpack(snap)
    # 3-tuple should serialize as "a:b:c"
    assert "a:b:c" in out["topology"]["triplet_distances"]


def test_full_assembly_round_trips():
    """End-to-end: assemble() → snapshot() → pack+strict-unpack."""
    s = SpiritState()
    outer = {
        "body_tensor": [0.1, 0.2, 0.3, 0.4, 0.5],
        "mind_tensor": [0.5, 0.4, 0.3, 0.2, 0.1],
        "spirit_tensor": [0.0] * 5,
        "outer_body": [0.5] * 5,
        "outer_mind": [0.5] * 5,
        "outer_spirit": [0.5] * 5,
    }
    inner = {"topology": {"distance_matrix": {("body", "mind"): 0.4}}}
    obs = {
        "outer_body": {
            "coherence": 0.95,
            "neighbors": {("self", "kin"): 0.7},
        }
    }
    s.assemble(outer_snapshot=outer, inner_snapshot=inner, observables=obs)
    snap = s.snapshot()
    # Must round-trip without raising
    out = _pack_then_strict_unpack(snap)
    assert out["assembly_count"] == 1
    assert "body:mind" in out["topology"]["distance_matrix"]
    assert "self:kin" in out["observables"]["outer_body"]["neighbors"]
    assert len(out["full_30dt"]) == 30


def test_string_keys_unchanged():
    """Sanitizer must NOT alter normal str keys."""
    s = SpiritState()
    s.topology = {
        "great_pulse": {"phase": "rest", "amplitude": 0.7},
        "string_keyed": {"a": 1, "b": 2},
    }
    snap = s.snapshot()
    out = _pack_then_strict_unpack(snap)
    assert out["topology"]["great_pulse"]["phase"] == "rest"
    assert out["topology"]["string_keyed"]["a"] == 1


def test_int_keys_coerced_to_strings():
    """msgpack strict_map_key=True rejects int keys too — sanitizer
    coerces them to str() form so the snapshot survives round-trip."""
    s = SpiritState()
    s.topology = {"int_keyed": {1: "one", 2: "two"}}
    snap = s.snapshot()
    out = _pack_then_strict_unpack(snap)
    assert out["topology"]["int_keyed"]["1"] == "one"
    assert out["topology"]["int_keyed"]["2"] == "two"


def test_lists_inside_dicts_preserved():
    """Lists as VALUES (vs keys) must be preserved as lists."""
    s = SpiritState()
    s.topology = {
        "trajectory": [0.1, 0.2, 0.3],
        "neighbors": [("a", "b"), ("c", "d")],  # list of tuples — values, OK
    }
    snap = s.snapshot()
    out = _pack_then_strict_unpack(snap)
    assert out["topology"]["trajectory"] == [0.1, 0.2, 0.3]
    # Tuples-in-lists become lists-in-lists (msgpack doesn't preserve tuple)
    assert len(out["topology"]["neighbors"]) == 2
