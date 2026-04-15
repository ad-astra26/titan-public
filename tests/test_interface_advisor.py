"""Tests for Step 7.2 — InterfaceAdvisor."""
import time
import pytest


class TestInterfaceAdvisorRateLimiting:
    """Per-message-type sliding window rate limiting."""

    def test_within_limit_no_feedback(self):
        from titan_plugin.logic.interface_advisor import InterfaceAdvisor
        advisor = InterfaceAdvisor(limits={"IMPULSE": 3}, window=60.0)
        # First message — within limit
        result = advisor.check("IMPULSE", "spirit")
        assert result is None

    def test_exceeds_limit_returns_feedback(self):
        from titan_plugin.logic.interface_advisor import InterfaceAdvisor
        advisor = InterfaceAdvisor(limits={"IMPULSE": 2}, window=60.0)
        advisor.check("IMPULSE", "spirit")
        advisor.check("IMPULSE", "spirit")
        # Third message exceeds limit of 2
        result = advisor.check("IMPULSE", "spirit")
        assert result is not None
        assert result["message_type"] == "IMPULSE"
        assert result["current_rate"] == 3
        assert result["limit"] == 2
        assert result["source"] == "spirit"

    def test_unknown_type_passes_through(self):
        from titan_plugin.logic.interface_advisor import InterfaceAdvisor
        advisor = InterfaceAdvisor(limits={"IMPULSE": 1})
        # Unknown type has no limit
        result = advisor.check("CUSTOM_TYPE", "test")
        assert result is None

    def test_window_expiry_resets_count(self):
        from titan_plugin.logic.interface_advisor import InterfaceAdvisor
        # Very short window (0.1s)
        advisor = InterfaceAdvisor(limits={"IMPULSE": 1}, window=0.1)
        advisor.check("IMPULSE", "spirit")
        # Wait for window to expire
        time.sleep(0.15)
        # Should be within limits again
        result = advisor.check("IMPULSE", "spirit")
        assert result is None

    def test_per_type_isolation(self):
        from titan_plugin.logic.interface_advisor import InterfaceAdvisor
        advisor = InterfaceAdvisor(limits={"IMPULSE": 1, "BODY_STATE": 5}, window=60.0)
        # Exhaust IMPULSE limit
        advisor.check("IMPULSE", "spirit")
        feedback = advisor.check("IMPULSE", "spirit")
        assert feedback is not None
        # BODY_STATE should still be within limit
        result = advisor.check("BODY_STATE", "body")
        assert result is None


class TestInterfaceAdvisorDynamic:
    """Dynamic limit adjustment."""

    def test_set_limit(self):
        from titan_plugin.logic.interface_advisor import InterfaceAdvisor
        advisor = InterfaceAdvisor(limits={"IMPULSE": 1})
        advisor.set_limit("IMPULSE", 5)
        # Now 5 messages should be allowed
        for _ in range(5):
            assert advisor.check("IMPULSE", "spirit") is None
        # 6th should trigger feedback
        assert advisor.check("IMPULSE", "spirit") is not None

    def test_minimum_limit_is_one(self):
        from titan_plugin.logic.interface_advisor import InterfaceAdvisor
        advisor = InterfaceAdvisor(limits={"IMPULSE": 1})
        advisor.set_limit("IMPULSE", 0)  # Try to set to 0
        # Should clamp to 1
        assert advisor.check("IMPULSE", "spirit") is None
        assert advisor.check("IMPULSE", "spirit") is not None


class TestInterfaceAdvisorStats:
    """Statistics and introspection."""

    def test_get_current_rate(self):
        from titan_plugin.logic.interface_advisor import InterfaceAdvisor
        advisor = InterfaceAdvisor(limits={"IMPULSE": 10}, window=60.0)
        advisor.check("IMPULSE", "spirit")
        advisor.check("IMPULSE", "spirit")
        assert advisor.get_current_rate("IMPULSE") == 2
        assert advisor.get_current_rate("UNKNOWN") == 0

    def test_stats_structure(self):
        from titan_plugin.logic.interface_advisor import InterfaceAdvisor
        advisor = InterfaceAdvisor()
        stats = advisor.get_stats()
        assert "limits" in stats
        assert "current_rates" in stats
        assert "window_seconds" in stats
        assert "rate_limit_count" in stats

    def test_reset_clears_window(self):
        from titan_plugin.logic.interface_advisor import InterfaceAdvisor
        advisor = InterfaceAdvisor(limits={"IMPULSE": 1}, window=60.0)
        advisor.check("IMPULSE", "spirit")
        advisor.check("IMPULSE", "spirit")
        assert advisor.get_current_rate("IMPULSE") == 2
        advisor.reset("IMPULSE")
        assert advisor.get_current_rate("IMPULSE") == 0
