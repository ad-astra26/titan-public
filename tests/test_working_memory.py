"""Tests for Working Memory — short-term processing buffer."""
import pytest
from titan_plugin.logic.working_memory import WorkingMemory


class TestWorkingMemory:

    def test_attend_and_retrieve(self):
        wm = WorkingMemory()
        wm.attend("active_word", "warm", {"word": "warm", "confidence": 0.8}, epoch=1)
        ctx = wm.get_context()
        assert len(ctx) == 1
        assert ctx[0]["key"] == "warm"

    def test_capacity_limit(self):
        wm = WorkingMemory(capacity=3)
        for i in range(5):
            wm.attend("active_word", f"word_{i}", {}, epoch=1)
        assert wm.size == 3  # Only 3 survive

    def test_decay_removes_old(self):
        wm = WorkingMemory(decay_epochs=3)
        wm.attend("active_word", "old", {}, epoch=1)
        wm.attend("active_word", "new", {}, epoch=5)
        wm.decay(current_epoch=5)
        assert wm.size == 1
        assert wm.is_attended("active_word", "new")
        assert not wm.is_attended("active_word", "old")

    def test_refresh_extends_lifetime(self):
        wm = WorkingMemory(decay_epochs=3)
        wm.attend("active_word", "warm", {}, epoch=1)
        # Refresh at epoch 3
        wm.attend("active_word", "warm", {}, epoch=3)
        # Decay at epoch 4 — should survive (refreshed at 3, decay at 3+3=6)
        wm.decay(current_epoch=4)
        assert wm.is_attended("active_word", "warm")

    def test_get_items_by_type(self):
        wm = WorkingMemory()
        wm.attend("active_word", "warm", {}, epoch=1)
        wm.attend("dominant_emotion", "CURIOSITY", {}, epoch=1)
        wm.attend("active_word", "cold", {}, epoch=1)

        words = wm.get_items_by_type("active_word")
        assert len(words) == 2

    def test_is_attended(self):
        wm = WorkingMemory()
        wm.attend("active_word", "warm", {}, epoch=1)
        assert wm.is_attended("active_word", "warm")
        assert not wm.is_attended("active_word", "cold")

    def test_clear(self):
        wm = WorkingMemory()
        wm.attend("active_word", "warm", {}, epoch=1)
        wm.clear()
        assert wm.size == 0

    def test_strength_decay(self):
        wm = WorkingMemory(decay_epochs=4)
        wm.attend("active_word", "warm", {}, epoch=1)
        wm.decay(current_epoch=3)  # age=2 out of 4
        ctx = wm.get_context()
        assert ctx[0]["strength"] == pytest.approx(0.5)

    def test_stats(self):
        wm = WorkingMemory()
        wm.attend("active_word", "warm", {}, epoch=1)
        stats = wm.get_stats()
        assert stats["size"] == 1
        assert stats["capacity"] == 7
