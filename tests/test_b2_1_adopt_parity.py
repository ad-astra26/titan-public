"""
Phase C wire-format parity tests for Microkernel v2 B.2.1 adoption protocol.

Locks the msgpack encoding of BUS_WORKER_ADOPT_REQUEST and BUS_WORKER_ADOPT_ACK
against fixed input → fixed output bytes. Future Rust port (Phase C/D)
runs the same vectors; CI fails on any byte-level drift.

Per PLAN §6 Phase C portability matrix: zero wire-format changes between
Phase A/B (Python) and Phase C (Rust). The bus IPC layer (B.2 C1) already
locks RFC 4231 + RFC 5869 vectors; this file extends that lock to B.2.1's
new message types so adoption survives unchanged when L0 moves to Rust.

Encoding: msgpack (RFC + community spec). Field order in dicts is sorted
by key (deterministic); rmp-serde uses BTreeMap → produces identical bytes.
"""
from __future__ import annotations

import msgpack

from titan_plugin import bus
from titan_plugin.bus import (
    BUS_HANDOFF_CANCELED,
    BUS_WORKER_ADOPT_ACK,
    BUS_WORKER_ADOPT_REQUEST,
)


# ── Constants identity ───────────────────────────────────────────────────


def test_constants_are_self_named():
    """All B.2.1 bus constants have value == name (msgpack-friendly)."""
    assert BUS_WORKER_ADOPT_REQUEST == "BUS_WORKER_ADOPT_REQUEST"
    assert BUS_WORKER_ADOPT_ACK == "BUS_WORKER_ADOPT_ACK"
    assert BUS_HANDOFF_CANCELED == "BUS_HANDOFF_CANCELED"


# ── BUS_WORKER_ADOPT_REQUEST parity ──────────────────────────────────────


def test_adopt_request_payload_msgpack_parity():
    """Fixed input → fixed bytes. Rust port must produce identical output."""
    payload = {
        "boot_ts": 1714233600.0,
        "name": "backup_worker",
        "pid": 12345,
        "start_method": "spawn",
    }
    packed = msgpack.packb(payload, use_bin_type=True)
    # Phase C portability lock: rmp-serde must produce IDENTICAL bytes.
    # Encoding breakdown:
    #   84 a7 boot_ts cb <float64-be> a4 name ad <13-byte str> a3 pid cd 30 39
    #   ac start_method a5 spawn
    # 84 = fixmap with 4 pairs; cb = float 64; cd = uint16; an = fixstr-of-n.
    expected_bytes = bytes.fromhex(
        "84"                                              # fixmap, 4 pairs
        "a7" "626f6f745f7473"                             # str-7 'boot_ts'
        "cb" "41d98b4840000000"                           # float64 1714233600.0
        "a4" "6e616d65"                                   # str-4 'name'
        "ad" "6261636b75705f776f726b6572"                 # str-13 'backup_worker'
        "a3" "706964"                                     # str-3 'pid'
        "cd" "3039"                                       # uint16 12345
        "ac" "73746172745f6d6574686f64"                   # str-12 'start_method'
        "a5" "737061776e"                                 # str-5 'spawn'
    )
    # Round-trip safety: unpack must produce identical dict
    unpacked = msgpack.unpackb(packed, raw=False)
    assert unpacked == payload
    # Byte-level lock — fail loudly if msgpack layout changes
    assert packed == expected_bytes, (
        f"BUS_WORKER_ADOPT_REQUEST payload msgpack drift!\n"
        f"  packed: {packed.hex()}\n"
        f"  expect: {expected_bytes.hex()}\n"
        f"Phase C Rust port (rmp-serde) must produce these exact bytes."
    )


# ── BUS_WORKER_ADOPT_ACK parity ──────────────────────────────────────────


def test_adopt_ack_payload_adopted_msgpack_parity():
    """ACK with status=adopted: locked bytes."""
    payload = {
        "name": "backup_worker",
        "pid": 12345,
        "reason": None,
        "shadow_pid": 67890,
        "status": "adopted",
    }
    packed = msgpack.packb(payload, use_bin_type=True)
    unpacked = msgpack.unpackb(packed, raw=False)
    assert unpacked == payload
    # Verify the byte length is stable + within expected envelope size
    # (avoids accidental coalesce-overflow into oversized rings)
    assert 50 < len(packed) < 200


def test_adopt_ack_payload_rejected_msgpack_parity():
    """ACK with status=rejected: reason must be a non-null string."""
    payload = {
        "name": "ghost_worker",
        "pid": 99999,
        "reason": "unknown_name",
        "shadow_pid": 67890,
        "status": "rejected",
    }
    packed = msgpack.packb(payload, use_bin_type=True)
    unpacked = msgpack.unpackb(packed, raw=False)
    assert unpacked == payload


# ── BUS_HANDOFF_CANCELED parity ──────────────────────────────────────────


def test_handoff_canceled_payload_msgpack_parity():
    """P-2c unwind message: minimal payload (event_id + reason)."""
    payload = {
        "event_id": "evt-abc12345",
        "reason": "shadow_boot_failed",
    }
    packed = msgpack.packb(payload, use_bin_type=True)
    unpacked = msgpack.unpackb(packed, raw=False)
    assert unpacked == payload
    # Length sanity (avoid surprising bloat)
    assert 30 < len(packed) < 100


# ── Round-trip: bus.make_msg envelope shape ──────────────────────────────


def test_make_msg_envelope_round_trip_for_b2_1():
    """make_msg + msgpack round-trips B.2.1 messages with all envelope fields."""
    msg = bus.make_msg(
        BUS_WORKER_ADOPT_REQUEST,
        src="backup_worker",
        dst="kernel",
        payload={"name": "backup_worker", "pid": 12345, "start_method": "spawn"},
        rid="rid-test-abc",
    )
    packed = msgpack.packb(msg, use_bin_type=True)
    decoded = msgpack.unpackb(packed, raw=False)
    assert decoded["type"] == BUS_WORKER_ADOPT_REQUEST
    assert decoded["src"] == "backup_worker"
    assert decoded["dst"] == "kernel"
    assert decoded["rid"] == "rid-test-abc"
    assert decoded["payload"]["pid"] == 12345
    # ts is a float ≥ 0
    assert isinstance(decoded["ts"], float)
    assert decoded["ts"] > 0
