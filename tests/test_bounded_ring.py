"""BoundedRing unit tests — the per-subscriber ring buffer with P0 reserve.

`BoundedRing` is live code: `BrokerSubscriber.ring` still uses it
(`titan_hcl/core/bus_socket.py:428`). These tests were preserved when the
surrounding `BusSocketServer`-dependent tests were retired (the Python broker
was deleted in D8-1 / commit 69035803 — titan-kernel-rs owns the bus broker
under fleet-wide Phase C per rFP §12.1; the live broker fanout/filter/skip
behavior is tested in `titan-rust/crates/titan-bus/src/broker.rs`). Carried
over verbatim from the retired `tests/test_bus_socket_server.py` BoundedRing
block (Phase B.2 C4).
"""

from __future__ import annotations

import pytest

from titan_hcl.core.bus_socket import BoundedRing


def test_boundedring_basic_append_and_pop():
    r = BoundedRing(capacity=10, p0_reserve=2)
    assert r.is_empty()
    assert r.append_main({"id": 1}) is True
    assert r.append_main({"id": 2}) is True
    assert len(r) == 2
    out = r.pop_for_send(max_msgs=10)
    assert [m["id"] for m in out] == [1, 2]
    assert r.is_empty()


def test_boundedring_p0_reserve_separate_from_main():
    r = BoundedRing(capacity=10, p0_reserve=2)
    # Fill main fully
    for i in range(8):
        assert r.append_main({"id": i}) is True
    assert r.main_is_full()
    # P0 still has room
    assert r.append_p0({"id": "p0_a"}) is True
    assert r.append_p0({"id": "p0_b"}) is True
    out = r.pop_for_send(max_msgs=99)
    # P0 drains FIRST
    assert out[0]["id"] == "p0_a"
    assert out[1]["id"] == "p0_b"
    assert [m["id"] for m in out[2:]] == list(range(8))


def test_boundedring_main_eviction_returns_false():
    r = BoundedRing(capacity=4, p0_reserve=1)  # main maxlen=3
    assert r.append_main({"id": 1}) is True
    assert r.append_main({"id": 2}) is True
    assert r.append_main({"id": 3}) is True
    assert r.append_main({"id": 4}) is False  # evicts 1
    out = r.pop_for_send(max_msgs=10)
    assert [m["id"] for m in out] == [2, 3, 4]


def test_boundedring_invalid_p0_reserve_raises():
    with pytest.raises(ValueError):
        BoundedRing(capacity=10, p0_reserve=10)
    with pytest.raises(ValueError):
        BoundedRing(capacity=10, p0_reserve=99)
