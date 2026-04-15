"""
Tests for Phase 5 QoL Features: Info Banner + Privacy Filter.
All tests are pure offline — no external services, no async.
"""
import pytest


# =========================================================================
# Feature 5.1: Info Banner
# =========================================================================

class TestBannerRendering:
    """Test banner string generation (pure offline, no mocks needed)."""

    def test_render_bar_0_percent(self):
        from titan_plugin.utils.banner import render_bar
        bar = render_bar(0)
        assert bar == "\u2591" * 10
        assert len(bar) == 10

    def test_render_bar_100_percent(self):
        from titan_plugin.utils.banner import render_bar
        bar = render_bar(100)
        assert bar == "\u2588" * 10
        assert len(bar) == 10

    def test_render_bar_50_percent(self):
        from titan_plugin.utils.banner import render_bar
        bar = render_bar(50)
        assert bar == "\u2588" * 5 + "\u2591" * 5

    def test_render_bar_clamps(self):
        from titan_plugin.utils.banner import render_bar
        bar_over = render_bar(150)
        assert bar_over == "\u2588" * 10
        bar_under = render_bar(-10)
        assert bar_under == "\u2591" * 10

    def test_build_banner_compact(self):
        from titan_plugin.utils.banner import build_banner
        banner = build_banner(68.0, 74.0, 31.0, "Contemplative", 3, style="compact")
        assert "[TITAN]" in banner
        assert "Life" in banner
        assert "68%" in banner
        assert "Sovereignty" in banner
        assert "74%" in banner
        assert "Memory 31%" in banner
        assert "Mood: Contemplative" in banner
        assert "Epoch 3" in banner

    def test_build_banner_minimal(self):
        from titan_plugin.utils.banner import build_banner
        banner = build_banner(68.0, 74.0, 31.0, "Contemplative", 3, style="minimal")
        assert "[TITAN]" in banner
        assert "Life 68%" in banner
        assert "Sov 74%" in banner
        assert "Mem 31%" in banner
        assert "Contemplative" in banner
        assert "E3" in banner
        # Minimal should NOT contain block characters
        assert "\u2588" not in banner

    def test_build_banner_unavailable_life(self):
        from titan_plugin.utils.banner import build_banner
        banner = build_banner(-1, 50.0, 10.0, "Unknown", 0, style="compact")
        assert "--%" in banner
        # Should show empty bars (all light shade)
        assert "\u2591" * 10 in banner

    def test_banner_single_line(self):
        from titan_plugin.utils.banner import build_banner
        banner = build_banner(100.0, 100.0, 100.0, "Energetic", 99)
        assert "\n" not in banner


class TestSovereigntyScore:
    """Test gatekeeper sovereignty metric calculation."""

    def _make_gatekeeper(self):
        """Create a minimal SageGatekeeper with mocked scholar/recorder."""

        class FakeScholar:
            pass

        class FakeRecorder:
            storage = []

        from titan_plugin.logic.sage.gatekeeper import SageGatekeeper
        return SageGatekeeper(FakeScholar(), FakeRecorder())

    def test_empty_history(self):
        gk = self._make_gatekeeper()
        assert gk.sovereignty_score == 0.0

    def test_all_sovereign(self):
        gk = self._make_gatekeeper()
        for _ in range(10):
            gk._record_decision("sovereign")
        assert gk.sovereignty_score == 100.0

    def test_all_shadow(self):
        gk = self._make_gatekeeper()
        for _ in range(10):
            gk._record_decision("shadow")
        assert gk.sovereignty_score == 0.0

    def test_mixed_decisions(self):
        gk = self._make_gatekeeper()
        # 2 sovereign (2.0) + 2 collaborative (1.0) + 1 shadow (0) = 3.0 / 5 * 100 = 60%
        gk._record_decision("sovereign")
        gk._record_decision("sovereign")
        gk._record_decision("collaborative")
        gk._record_decision("collaborative")
        gk._record_decision("shadow")
        assert abs(gk.sovereignty_score - 60.0) < 0.01

    def test_rolling_window(self):
        gk = self._make_gatekeeper()
        # Fill with 100 shadow decisions
        for _ in range(100):
            gk._record_decision("shadow")
        assert gk.sovereignty_score == 0.0
        # Add 1 sovereign — oldest shadow should be dropped
        gk._record_decision("sovereign")
        assert len(gk._decision_history) == 100
        # 1 sovereign + 99 shadow = 1.0 / 100 * 100 = 1%
        assert abs(gk.sovereignty_score - 1.0) < 0.01

    def test_record_decision(self):
        gk = self._make_gatekeeper()
        gk._record_decision("sovereign")
        assert len(gk._decision_history) == 1
        assert gk._decision_history[0] == "sovereign"


# =========================================================================
# Feature 5.2: Privacy Filter
# =========================================================================

class TestPrivacyFilter:
    """Test PII sanitization (pure offline)."""

    def test_email_redacted(self):
        from titan_plugin.utils.privacy import sanitize_outbound
        text, count = sanitize_outbound("Contact user@example.com for info")
        assert "[EMAIL_REDACTED]" in text
        assert "user@example.com" not in text
        assert count == 1

    def test_phone_redacted(self):
        from titan_plugin.utils.privacy import sanitize_outbound
        text, count = sanitize_outbound("Call 555-123-4567 or (555) 987-6543")
        assert "[PHONE_REDACTED]" in text
        assert "555-123-4567" not in text
        assert count >= 1

    def test_ssn_redacted(self):
        from titan_plugin.utils.privacy import sanitize_outbound
        text, count = sanitize_outbound("SSN: 123-45-6789")
        assert "[SSN_REDACTED]" in text
        assert "123-45-6789" not in text
        assert count == 1

    def test_credit_card_redacted(self):
        from titan_plugin.utils.privacy import sanitize_outbound
        text, count = sanitize_outbound("Card: 4111 1111 1111 1111")
        assert "[CARD_REDACTED]" in text
        assert "4111 1111 1111 1111" not in text
        assert count == 1

    def test_ip_redacted(self):
        from titan_plugin.utils.privacy import sanitize_outbound
        text, count = sanitize_outbound("Server at 192.168.1.1 is down")
        assert "[IP_REDACTED]" in text
        assert "192.168.1.1" not in text
        assert count == 1

    def test_solana_address_preserved(self):
        from titan_plugin.utils.privacy import sanitize_outbound
        # A real Solana address (base58, 44 chars)
        addr = "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"
        text, count = sanitize_outbound(f"Wallet: {addr}")
        assert addr in text
        assert count == 0

    def test_no_pii_unchanged(self):
        from titan_plugin.utils.privacy import sanitize_outbound
        original = "The Titan is sovereign and free."
        text, count = sanitize_outbound(original)
        assert text == original
        assert count == 0

    def test_multiple_pii_in_one_text(self):
        from titan_plugin.utils.privacy import sanitize_outbound
        text, count = sanitize_outbound(
            "Email: a@b.com, Phone: 555-111-2222, IP: 10.0.0.1"
        )
        assert "[EMAIL_REDACTED]" in text
        assert "[PHONE_REDACTED]" in text
        assert "[IP_REDACTED]" in text
        assert count >= 3

    def test_redaction_count_correct(self):
        from titan_plugin.utils.privacy import sanitize_outbound
        text, count = sanitize_outbound(
            "Emails: a@b.com and c@d.com"
        )
        assert count == 2

    def test_selective_patterns(self):
        from titan_plugin.utils.privacy import sanitize_outbound
        # Only apply email pattern — phone should survive
        text, count = sanitize_outbound(
            "Email: a@b.com Phone: 555-111-2222",
            active_patterns=["email"],
        )
        assert "[EMAIL_REDACTED]" in text
        assert "555-111-2222" in text  # phone NOT redacted
        assert count == 1

    def test_empty_text(self):
        from titan_plugin.utils.privacy import sanitize_outbound
        text, count = sanitize_outbound("")
        assert text == ""
        assert count == 0

    def test_disabled_by_default(self):
        """Verify config.toml default is false."""
        import os
        try:
            import tomllib
        except ModuleNotFoundError:
            import toml as tomllib  # type: ignore
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "titan_plugin", "config.toml"
        )
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        assert config["privacy"]["enabled"] is False


# =========================================================================
# Feature 5.1 + 5.2: Integration Tests
# =========================================================================

class TestBannerIntegration:
    """Test banner wiring into pre_prompt_hook (mocked subsystems)."""

    def _make_plugin_stub(self, banner_enabled=True, banner_style="compact"):
        """Build a minimal TitanPlugin-like object with mocked subsystems."""

        class Stub:
            pass

        plugin = Stub()

        # Config
        plugin._full_config = {
            "info_banner": {"enabled": banner_enabled, "style": banner_style},
            "privacy": {"enabled": False},
        }

        # Metabolism
        metab = Stub()
        metab._last_balance = 1.5
        metab._last_balance_pct_val = 75.0
        type(metab)._last_balance_pct = property(lambda self: 75.0)
        plugin.metabolism = metab

        # Gatekeeper with sovereignty_score
        gk = Stub()
        gk._decision_history = ["sovereign", "sovereign", "collaborative", "shadow"]
        type(gk).sovereignty_score = property(
            lambda self: (
                sum(1.0 if d == "sovereign" else 0.5 if d == "collaborative" else 0.0
                    for d in self._decision_history)
                / len(self._decision_history) * 100
            ) if self._decision_history else 0.0
        )
        plugin.gatekeeper = gk

        # Mood engine
        mood = Stub()
        mood.current_mood = "Contemplative"
        plugin.mood_engine = mood

        # Meditation
        med = Stub()
        med._epoch_counter = 5
        plugin.meditation = med

        # Memory contribution
        plugin._last_memory_contribution = 25.0

        return plugin

    def test_banner_prepended_when_enabled(self):
        from titan_plugin.utils.banner import build_banner

        plugin = self._make_plugin_stub(banner_enabled=True)
        banner = build_banner(
            life_pct=75.0,
            sovereignty_pct=plugin.gatekeeper.sovereignty_score,
            memory_pct=25.0,
            mood="Contemplative",
            epoch=5,
            style="compact",
        )
        assert "[TITAN]" in banner
        assert "Contemplative" in banner

    def test_banner_absent_when_disabled(self):
        plugin = self._make_plugin_stub(banner_enabled=False)
        banner_cfg = plugin._full_config.get("info_banner", {})
        assert banner_cfg.get("enabled") is False

    def test_memory_contribution_calculated(self):
        """Memory contribution should reflect injected memory size vs total prompt."""
        memory_text = "A" * 100
        system_prompt = "B" * 300
        total = len(memory_text) + len(system_prompt)
        expected_pct = len(memory_text) / total * 100
        assert abs(expected_pct - 25.0) < 0.01


class TestMetabolismBalancePct:
    """Test the _last_balance_pct property on MetabolismController."""

    def _make_controller(self):
        class Stub:
            pass

        soul = Stub()
        network = Stub()
        from titan_plugin.core.metabolism import MetabolismController
        return MetabolismController(soul, network)

    def test_no_balance_returns_negative(self):
        mc = self._make_controller()
        assert mc._last_balance_pct == -1.0

    def test_balance_1_sol(self):
        mc = self._make_controller()
        mc._last_balance = 1.0
        assert mc._last_balance_pct == 50.0

    def test_balance_2_sol(self):
        mc = self._make_controller()
        mc._last_balance = 2.0
        assert mc._last_balance_pct == 100.0

    def test_balance_clamp_high(self):
        mc = self._make_controller()
        mc._last_balance = 5.0
        assert mc._last_balance_pct == 100.0

    def test_balance_clamp_low(self):
        mc = self._make_controller()
        mc._last_balance = 0.001
        assert mc._last_balance_pct == 1.0
