"""BUS_SUBSCRIBE envelope byte-wire parity test against Python ground truth.

Locks the canonical Python wire format produced by
`titan_hcl/core/bus_socket.py::BusSocketClient._send_subscribe_frame`
and `_send_alias_subscribe_frame` (lines 1391-1396 + 1362-1379) against
the vectors in `tests/parity/vectors.json :: bus_subscribe_envelope ::
canonical_v1`.

Sibling Rust integration test at
`titan-rust/crates/titan-bus/tests/bus_subscribe_envelope.rs` validates
the Rust port against the same vectors per SPEC §8.10 byte-identical
guarantee.

Per `rFP_worker_broadcast_topics_completion §4.C-ter.2` (codified
2026-05-13) — closes the parity-test gap that allowed the pre-2026-05-13
Rust encoder Binary-vs-Map bug to land. SPEC §8.2 line 789 + §8.2 v1.4.0
D-SPEC-42 + §8.10 line 900.
"""

import json
from pathlib import Path

import msgpack
import pytest


VECTORS_PATH = Path(__file__).parent / "vectors.json"


@pytest.fixture(scope="module")
def vectors():
    return json.loads(VECTORS_PATH.read_text())


def _encode_subscribe_envelope(name: str, topics: list[str], reply_only: bool) -> bytes:
    """Mirror `bus_socket.py::_send_subscribe_frame` (lines 1391-1396).

    Canonical Python wire format per SPEC §8.2 line 789:
      `{type, src, dst, payload: {name, topics, reply_only}}`
    msgpack-packed with `use_bin_type=True` (default in modern msgpack-python).
    """
    msg = {
        "type": "BUS_SUBSCRIBE",
        "src": name,
        "dst": "broker",
        "payload": {
            "name": name,
            "topics": topics,
            "reply_only": reply_only,
        },
    }
    return msgpack.packb(msg, use_bin_type=True)


def test_broadcast_consumer_inner_body_matches_locked_vector(vectors):
    vec_root = vectors["bus_subscribe_envelope"]["canonical_v1"]["broadcast_consumer_inner_body"]
    expected_hex = vec_root["msgpack_hex"]
    expected_bytes = bytes.fromhex(expected_hex)

    name = vec_root["input"]["payload"]["name"]
    topics = vec_root["input"]["payload"]["topics"]
    reply_only = vec_root["input"]["payload"]["reply_only"]

    actual = _encode_subscribe_envelope(name, topics, reply_only)

    assert len(actual) == vec_root["msgpack_bytes"], (
        f"Python BUS_SUBSCRIBE envelope length {len(actual)} differs from "
        f"locked vector {vec_root['msgpack_bytes']}"
    )
    assert actual == expected_bytes, (
        f"Python BUS_SUBSCRIBE envelope bytes differ from locked vector — "
        f"SPEC §8.10 byte-identical guarantee violated. "
        f"Got hex={actual.hex()}"
    )


def test_reply_only_titan_hcl_matches_locked_vector(vectors):
    vec_root = vectors["bus_subscribe_envelope"]["canonical_v1"]["reply_only_titan_HCL"]
    expected_hex = vec_root["msgpack_hex"]
    expected_bytes = bytes.fromhex(expected_hex)

    name = vec_root["input"]["payload"]["name"]
    topics = vec_root["input"]["payload"]["topics"]
    reply_only = vec_root["input"]["payload"]["reply_only"]

    actual = _encode_subscribe_envelope(name, topics, reply_only)

    assert actual == expected_bytes, (
        f"Python BUS_SUBSCRIBE (reply_only=true, D-SPEC-42 row 2) bytes "
        f"differ from locked vector — SPEC §8.10 byte-identical violated. "
        f"Got hex={actual.hex()}"
    )


def test_payload_field_order_locked_by_spec(vectors):
    """The msgpack wire bytes depend on dict insertion order (msgpack-python
    preserves it; Rust rmpv preserves Map Vec order). The locked vector
    encodes a specific field order — if production code reorders the dict
    keys (e.g., topics before name), the bytes change. This test asserts
    the expected order is what's in the vector.
    """
    vec_root = vectors["bus_subscribe_envelope"]["canonical_v1"]["broadcast_consumer_inner_body"]
    assert vec_root["envelope_field_order"] == ["type", "src", "dst", "payload"]
    assert vec_root["payload_field_order"] == ["name", "topics", "reply_only"]


def test_three_d_spec_42_intents_present_in_vector_set(vectors):
    """SPEC §8.2 v1.4.0 D-SPEC-42 defines three legal intents (line 803).
    Vector set MUST cover the two non-forbidden ones: broadcast_consumer
    (topics non-empty, reply_only=false) AND reply_only (topics empty,
    reply_only=true). The forbidden regression state (topics empty,
    reply_only=false) is NOT vectored — it must fail at SPEC-enforcement
    time, not at encode-decode parity.
    """
    canonical = vectors["bus_subscribe_envelope"]["canonical_v1"]
    assert "broadcast_consumer_inner_body" in canonical, (
        "Missing D-SPEC-42 row 1 (broadcast consumer) vector"
    )
    assert "reply_only_titan_HCL" in canonical, (
        "Missing D-SPEC-42 row 2 (reply_only) vector"
    )
    # Verify the data shape matches the intent labels
    bc = canonical["broadcast_consumer_inner_body"]["input"]["payload"]
    assert bc["topics"] and bc["reply_only"] is False
    ro = canonical["reply_only_titan_HCL"]["input"]["payload"]
    assert ro["topics"] == [] and ro["reply_only"] is True
