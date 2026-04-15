"""Tests for SharedBlackboard and DivineBus v2 dual-channel (Phase S1+S2)."""
import threading
import time
import pytest
from titan_plugin.shared_blackboard import SharedBlackboard
from titan_plugin.bus import DivineBus, BODY_STATE, MIND_STATE, IMPULSE


class TestSharedBlackboard:
    def test_write_and_read(self):
        bb = SharedBlackboard()
        bb.write("body", {"tensor": [0.5, 0.5, 0.5]})
        data, ts = bb.read("body")
        assert data == {"tensor": [0.5, 0.5, 0.5]}
        assert ts > 0

    def test_read_returns_latest(self):
        bb = SharedBlackboard()
        bb.write("body", {"v": 1})
        bb.write("body", {"v": 2})
        bb.write("body", {"v": 3})
        data, _ = bb.read("body")
        assert data["v"] == 3
        assert bb.stats["writes"] == 3

    def test_read_if_newer(self):
        bb = SharedBlackboard()
        bb.write("body", {"v": 1})
        _, ts = bb.read("body")
        time.sleep(0.01)
        bb.write("body", {"v": 2})
        result = bb.read_if_newer("body", ts)
        assert result is not None
        assert result["v"] == 2

    def test_read_if_newer_returns_none_when_stale(self):
        bb = SharedBlackboard()
        bb.write("body", {"v": 1})
        future = time.time() + 100
        result = bb.read_if_newer("body", future)
        assert result is None

    def test_read_nonexistent_key(self):
        bb = SharedBlackboard()
        data, ts = bb.read("nonexistent")
        assert data is None
        assert ts == 0.0

    def test_thread_safety(self):
        bb = SharedBlackboard()
        errors = []

        def writer(key, count):
            try:
                for i in range(count):
                    bb.write(key, {"i": i})
            except Exception as e:
                errors.append(e)

        def reader(key, count):
            try:
                for _ in range(count):
                    bb.read(key)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=("k1", 100)),
            threading.Thread(target=writer, args=("k2", 100)),
            threading.Thread(target=reader, args=("k1", 100)),
            threading.Thread(target=reader, args=("k2", 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert bb.stats["writes"] == 200

    def test_stats(self):
        bb = SharedBlackboard()
        bb.write("a", {"v": 1})
        bb.write("b", {"v": 2})
        bb.read("a")
        stats = bb.stats
        assert stats["key_count"] == 2
        assert stats["writes"] == 2
        assert stats["reads"] == 1


class TestDivineBusBlackboard:
    def test_state_msg_written_to_blackboard(self):
        bus = DivineBus(maxsize=100)
        bus.subscribe("test")
        msg = {"type": BODY_STATE, "src": "body", "dst": "all",
               "ts": time.time(), "payload": {"tensor": [0.5]*5}}
        bus.publish(msg)
        data, ts = bus.blackboard.read("body_BODY_STATE")
        assert data is not None
        assert data["payload"]["tensor"] == [0.5]*5

    def test_event_msg_not_on_blackboard(self):
        bus = DivineBus(maxsize=100)
        bus.subscribe("test")
        msg = {"type": IMPULSE, "src": "spirit", "dst": "all",
               "ts": time.time(), "payload": {"impulse_id": 1}}
        bus.publish(msg)
        data, _ = bus.blackboard.read("spirit_IMPULSE")
        assert data is None  # IMPULSE is event, not state

    def test_blackboard_accessible_via_bus(self):
        bus = DivineBus(maxsize=100)
        assert bus.blackboard is not None
        assert isinstance(bus.blackboard, SharedBlackboard)

    def test_backward_compatible_queue_routing(self):
        """State messages still go to queues (dual-write)."""
        bus = DivineBus(maxsize=100)
        q = bus.subscribe("consumer")
        msg = {"type": BODY_STATE, "src": "body", "dst": "all",
               "ts": time.time(), "payload": {"tensor": [0.5]*5}}
        bus.publish(msg)
        # Should be on blackboard AND queue
        bb_data, _ = bus.blackboard.read("body_BODY_STATE")
        assert bb_data is not None
        # Queue should also have it
        from queue import Empty
        try:
            q_msg = q.get_nowait()
            assert q_msg["type"] == BODY_STATE
        except Empty:
            pytest.fail("State message should also be on queue (dual-write)")
