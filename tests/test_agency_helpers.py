"""Tests for Step 7.5 — Helper implementations (BaseHelper protocol compliance)."""
import pytest
from titan_plugin.logic.agency.registry import BaseHelper


class TestWebSearchHelper:
    """WebSearchHelper protocol compliance and status."""

    def test_implements_protocol(self):
        from titan_plugin.logic.agency.helpers.web_search import WebSearchHelper
        helper = WebSearchHelper()
        assert isinstance(helper, BaseHelper)
        assert helper.name == "web_search"
        assert "search" in helper.capabilities
        assert helper.enriches == ["mind"]
        assert helper.requires_sandbox is False


class TestInfraInspectHelper:
    """InfraInspectHelper protocol compliance and execution."""

    def test_implements_protocol(self):
        from titan_plugin.logic.agency.helpers.infra_inspect import InfraInspectHelper
        helper = InfraInspectHelper()
        assert isinstance(helper, BaseHelper)
        assert helper.name == "infra_inspect"
        assert "system_stats" in helper.capabilities
        assert helper.enriches == ["body"]

    def test_always_available(self):
        from titan_plugin.logic.agency.helpers.infra_inspect import InfraInspectHelper
        helper = InfraInspectHelper()
        assert helper.status() == "available"

    @pytest.mark.asyncio
    async def test_system_inspection(self):
        from titan_plugin.logic.agency.helpers.infra_inspect import InfraInspectHelper
        helper = InfraInspectHelper()
        result = await helper.execute({"what": "system"})
        assert result["success"] is True
        assert "CPU" in result["result"]
        assert "RAM" in result["result"]
        assert result["enrichment_data"].get("body") is not None


class TestSocialPostHelper:
    """SocialPostHelper protocol compliance and status."""

    def test_implements_protocol(self):
        from titan_plugin.logic.agency.helpers.social_post import SocialPostHelper
        helper = SocialPostHelper()
        assert isinstance(helper, BaseHelper)
        assert helper.name == "social_post"
        assert "post" in helper.capabilities
        assert helper.enriches == ["mind"]

    def test_unavailable_without_api_key(self):
        from titan_plugin.logic.agency.helpers.social_post import SocialPostHelper
        helper = SocialPostHelper(api_key="")
        assert helper.status() == "unavailable"

    def test_available_with_api_key(self):
        from titan_plugin.logic.agency.helpers.social_post import SocialPostHelper
        helper = SocialPostHelper(api_key="test_key_123")
        assert helper.status() == "available"


class TestArtGenerateHelper:
    """ArtGenerateHelper protocol compliance."""

    def test_implements_protocol(self):
        from titan_plugin.logic.agency.helpers.art_generate import ArtGenerateHelper
        helper = ArtGenerateHelper()
        assert isinstance(helper, BaseHelper)
        assert helper.name == "art_generate"
        assert "image_generation" in helper.capabilities
        assert "mind" in helper.enriches
        assert "spirit" in helper.enriches


class TestAudioGenerateHelper:
    """AudioGenerateHelper protocol compliance and execution."""

    def test_implements_protocol(self):
        from titan_plugin.logic.agency.helpers.audio_generate import AudioGenerateHelper
        helper = AudioGenerateHelper()
        assert isinstance(helper, BaseHelper)
        assert helper.name == "audio_generate"
        assert "audio_generation" in helper.capabilities
        assert "trinity_sonification" in helper.capabilities
        assert "mind" in helper.enriches
        assert "spirit" in helper.enriches
        assert helper.requires_sandbox is False

    def test_always_available(self):
        from titan_plugin.logic.agency.helpers.audio_generate import AudioGenerateHelper
        helper = AudioGenerateHelper()
        assert helper.status() == "available"

    @pytest.mark.asyncio
    async def test_trinity_sonification(self, tmp_path):
        from titan_plugin.logic.agency.helpers.audio_generate import AudioGenerateHelper
        helper = AudioGenerateHelper(
            output_dir=str(tmp_path / "audio"),
            media_queue_dir=str(tmp_path / "queue"),
            max_duration=3,
        )
        result = await helper.execute({
            "mode": "trinity",
            "body": [0.8, 0.5, 0.3, 0.1, 0.6],
            "mind": [0.7, 0.4, 0.5, 0.2, 0.9],
            "spirit": [0.6, 0.3, 0.8, 0.5, 0.5],
            "middle_path_loss": 0.2,
            "duration": 3,
        })
        assert result["success"] is True
        assert "trinity" in result["result"].lower()
        assert result["enrichment_data"]["mind"] == [1]
        assert result["enrichment_data"]["spirit"] == [2]
        # Verify WAV file was created
        import os
        audio_files = list((tmp_path / "audio").glob("*.wav"))
        assert len(audio_files) == 1
        # Verify media queue copy for enrichment loop
        queue_files = list((tmp_path / "queue").glob("*.wav"))
        assert len(queue_files) == 1

    @pytest.mark.asyncio
    async def test_blockchain_sonification(self, tmp_path):
        from titan_plugin.logic.agency.helpers.audio_generate import AudioGenerateHelper
        helper = AudioGenerateHelper(
            output_dir=str(tmp_path / "audio"),
            media_queue_dir=str(tmp_path / "queue"),
        )
        result = await helper.execute({
            "mode": "blockchain",
            "tx_signature": "abc123def456789012345678901234567890abcdef1234567890abcdef12345678",
            "sol_balance": 5.0,
        })
        assert result["success"] is True
        assert "blockchain" in result["result"].lower()

    @pytest.mark.asyncio
    async def test_duration_cap(self, tmp_path):
        from titan_plugin.logic.agency.helpers.audio_generate import AudioGenerateHelper
        helper = AudioGenerateHelper(
            output_dir=str(tmp_path / "audio"),
            media_queue_dir=str(tmp_path / "queue"),
            max_duration=5,
        )
        result = await helper.execute({
            "mode": "trinity",
            "duration": 999,  # should be capped to 5
            "body": [0.5] * 5,
            "mind": [0.5] * 5,
            "spirit": [0.5] * 5,
        })
        assert result["success"] is True
        # File should be ~5s not 999s
        import os
        audio_files = list((tmp_path / "audio").glob("*.wav"))
        # 5s at 44100Hz * 2 bytes = ~441KB, not ~88MB
        assert audio_files[0].stat().st_size < 500_000

    @pytest.mark.asyncio
    async def test_extreme_tensor_values(self, tmp_path):
        """Test with extreme tensor values (all 0s and all 1s)."""
        from titan_plugin.logic.agency.helpers.audio_generate import AudioGenerateHelper
        helper = AudioGenerateHelper(
            output_dir=str(tmp_path / "audio"),
            media_queue_dir=str(tmp_path / "queue"),
            max_duration=2,
        )
        # All zeros — stressed/depleted state
        result = await helper.execute({
            "mode": "trinity",
            "body": [0.0] * 5,
            "mind": [0.0] * 5,
            "spirit": [0.0] * 5,
            "middle_path_loss": 1.0,
            "duration": 2,
        })
        assert result["success"] is True

        # All ones — peak state
        result = await helper.execute({
            "mode": "trinity",
            "body": [1.0] * 5,
            "mind": [1.0] * 5,
            "spirit": [1.0] * 5,
            "middle_path_loss": 0.0,
            "duration": 2,
        })
        assert result["success"] is True


class TestHelperRegistration:
    """Verify all helpers can register in the registry."""

    def test_register_all_helpers(self):
        from titan_plugin.logic.agency.registry import HelperRegistry
        from titan_plugin.logic.agency.helpers import (
            WebSearchHelper, InfraInspectHelper, SocialPostHelper,
            ArtGenerateHelper, AudioGenerateHelper,
        )

        registry = HelperRegistry()
        helpers = [
            WebSearchHelper(),
            InfraInspectHelper(),
            SocialPostHelper(api_key="test"),
            ArtGenerateHelper(),
            AudioGenerateHelper(),
        ]
        for h in helpers:
            registry.register(h)

        assert len(registry.list_all_names()) == 5
        manifest = registry.list_available()
        # InfraInspect is always available, others depend on status
        assert "infra_inspect" in manifest
        assert "audio_generate" in manifest
