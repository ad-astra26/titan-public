"""Tests for chat_tier_config + provider resolve_model_class (D-SPEC-79)."""
from __future__ import annotations

import pytest

from titan_hcl.modules.chat_tier_config import (
    ChatTierClassifier,
    ClassifyResult,
    TierConfig,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def full_config() -> dict:
    """Mirrors the [chat] section of config.toml (subset)."""
    return {
        "chat": {
            "default_tier": "casual",
            "default_model_class": "heavy",
            "classifier_log_decisions": False,
            "features": {
                "known": [
                    "directives", "felt_state", "history",
                    "cgn_grounding", "cgn_social_action",
                    "user_recognition", "topic_memory",
                    "reasoning_chain", "gatekeeper_state", "tools",
                ],
            },
            # Mirrors config.toml order: specific-first, fallback-last.
            "tiers": [
                {
                    "name": "greeting",
                    "model_class": "fast",
                    "max_tokens": 80,
                    "detect": {
                        "max_chars": 50,
                        "regex_any": [
                            r"^\s*(hi|hello|hey|yo|sup|greetings|good (morning|afternoon|evening))\b",
                            r"^\s*(how are you|what's up|wassup)\??\s*$",
                        ],
                    },
                    "features": ["directives"],
                },
                {
                    "name": "personal",
                    "model_class": "heavy",
                    "detect": {
                        "regex_any": [
                            r"\b(do you (know|remember)|have we (met|talked|spoken)|recognize me)\b",
                            r"\b(my name is|i'?m\s+\w+|remember me)\b",
                        ],
                        "topic_memory_regex": [
                            r"\b(remember (talking|discussing|chatting) (about|with))\b",
                            r"\b(what (did|do) (we|you) (talk|discuss|say) about)\b",
                        ],
                    },
                    "features": [
                        "directives", "felt_state", "history",
                        "user_recognition", "cgn_social_action",
                    ],
                },
                {
                    "name": "reasoning",
                    "model_class": "heavy",
                    "detect": {
                        "regex_any": [
                            r"\b(why|explain|how does|how do|analyze|compare|reason)\b",
                        ],
                    },
                    "features": [
                        "directives", "felt_state", "history",
                        "cgn_grounding", "cgn_social_action",
                        "user_recognition", "topic_memory",
                        "reasoning_chain", "gatekeeper_state", "tools",
                    ],
                },
                {
                    "name": "casual",
                    "model_class": "heavy",
                    "detect": {"max_chars": 200},
                    "features": ["directives", "felt_state", "history"],
                },
            ],
        },
    }


@pytest.fixture
def classifier(full_config) -> ChatTierClassifier:
    return ChatTierClassifier.from_config(full_config)


# ── Tier classification ────────────────────────────────────────────


class TestTierClassification:
    def test_hi_greeting(self, classifier):
        result = classifier.classify("Hi")
        assert result.tier.name == "greeting"
        assert result.tier.model_class == "fast"
        assert result.active_features == frozenset({"directives"})

    def test_hello_titan_greeting(self, classifier):
        result = classifier.classify("Hello Titan")
        assert result.tier.name == "greeting"

    def test_how_are_you_greeting(self, classifier):
        result = classifier.classify("How are you?")
        assert result.tier.name == "greeting"

    def test_short_message_casual(self, classifier):
        result = classifier.classify("Tell me about your day.")
        assert result.tier.name == "casual"
        assert result.tier.model_class == "heavy"
        assert "felt_state" in result.active_features
        assert "user_recognition" not in result.active_features
        assert "reasoning_chain" not in result.active_features

    def test_do_you_know_me_personal(self, classifier):
        result = classifier.classify("Do you know me?")
        assert result.tier.name == "personal"
        assert "user_recognition" in result.active_features
        assert "topic_memory" not in result.active_features  # no "about X"

    def test_do_you_remember_talking_about_personal_with_topic(self, classifier):
        result = classifier.classify(
            "Do you remember talking about quantum computing with me?"
        )
        assert result.tier.name == "personal"
        assert "user_recognition" in result.active_features
        assert "topic_memory" in result.active_features

    def test_why_question_reasoning(self, classifier):
        prompt = (
            "Why does the cognitive worker handle hormonal events "
            "differently from sensorimotor inputs in the trinity architecture?"
        )
        result = classifier.classify(prompt)
        assert result.tier.name == "reasoning"
        assert "reasoning_chain" in result.active_features
        assert "tools" in result.active_features

    def test_long_no_keyword_falls_through_to_reasoning(self, classifier):
        prompt = "x" * 250 + " explain"
        result = classifier.classify(prompt)
        assert result.tier.name == "reasoning"

    def test_empty_prompt_defaults_to_casual(self, classifier):
        result = classifier.classify("")
        # Empty matches casual (max_chars=200, no regex required)
        assert result.tier.name == "casual"


class TestDefaultTier:
    def test_default_tier_is_casual(self, classifier):
        assert classifier.default_tier.name == "casual"


class TestEmptyConfig:
    def test_no_tiers_falls_back_to_passthrough(self):
        cls = ChatTierClassifier.from_config({"chat": {}})
        result = cls.classify("Anything")
        assert result.tier.name == "passthrough"
        # Passthrough enables all features (safety net)
        assert "directives" in result.active_features
        assert "reasoning_chain" in result.active_features

    def test_missing_chat_section_passthrough(self, monkeypatch):
        # No `chat` key in the passed config → from_config falls back to the
        # live SHM read; with no chat config anywhere → passthrough safety net.
        monkeypatch.setattr(
            "titan_hcl.modules.chat_tier_config.get_params",
            lambda section=None: {},
        )
        cls = ChatTierClassifier.from_config({})
        result = cls.classify("Hi")
        assert result.tier.name == "passthrough"


class TestBadRegex:
    def test_invalid_regex_skipped_not_crash(self):
        cfg = {
            "chat": {
                "default_tier": "t1",
                "default_model_class": "heavy",
                "tiers": [
                    {
                        "name": "t1",
                        "model_class": "heavy",
                        "detect": {
                            "max_chars": 100,
                            "regex_any": ["[unclosed", "valid.*"],
                        },
                        "features": ["directives"],
                    },
                ],
            },
        }
        cls = ChatTierClassifier.from_config(cfg)
        # Invalid regex was dropped; valid one still matches.
        result = cls.classify("validword here")
        assert result.tier.name == "t1"


# ── Provider resolve_model_class ───────────────────────────────────


class TestProviderResolveModelClass:
    def test_ollama_resolves_fast_light_heavy(self):
        from titan_hcl.inference.ollama_cloud import OllamaCloudProvider
        cfg = {
            "ollama_cloud_api_key": "test",
            "ollama_cloud_fast_model": "gemma3:4b",
            "ollama_cloud_light_model": "gemma4:31b",
            "ollama_cloud_heavy_model": "deepseek-v3.1:671b",
        }
        p = OllamaCloudProvider(cfg)
        assert p.resolve_model_class("fast") == "gemma3:4b"
        assert p.resolve_model_class("light") == "gemma4:31b"
        assert p.resolve_model_class("heavy") == "deepseek-v3.1:671b"

    def test_ollama_unknown_class_falls_back_to_default(self):
        from titan_hcl.inference.ollama_cloud import OllamaCloudProvider
        cfg = {"ollama_cloud_api_key": "test"}
        p = OllamaCloudProvider(cfg)
        # Unknown class → falls back to provider's default chat model (self.id).
        assert p.resolve_model_class("nonexistent") == p.id

    def test_venice_resolves_classes(self):
        from titan_hcl.inference.venice import VeniceProvider
        cfg = {
            "venice_api_key": "test",
            "venice_fast_model": "llama-3.1-8b",
            "venice_heavy_model": "deepseek-r1-671b",
        }
        p = VeniceProvider(cfg)
        assert p.resolve_model_class("fast") == "llama-3.1-8b"
        assert p.resolve_model_class("heavy") == "deepseek-r1-671b"

    def test_openrouter_resolves_classes(self):
        from titan_hcl.inference.openrouter import OpenRouterProvider
        cfg = {
            "openrouter_api_key": "test",
            "openrouter_fast_model": "google/gemma-2-9b-it:free",
            "openrouter_heavy_model": "deepseek/deepseek-r1",
        }
        p = OpenRouterProvider(cfg)
        assert p.resolve_model_class("fast") == "google/gemma-2-9b-it:free"
        assert p.resolve_model_class("heavy") == "deepseek/deepseek-r1"

    def test_custom_resolves_classes(self):
        from titan_hcl.inference.custom import CustomProvider
        cfg = {
            "custom_llm_api_key": "test",
            "custom_fast_model": "fast-model",
            "custom_heavy_model": "heavy-model",
        }
        p = CustomProvider(cfg)
        assert p.resolve_model_class("fast") == "fast-model"
        assert p.resolve_model_class("heavy") == "heavy-model"


# ── ζ.1 PreHook feature-gate source assertions ─────────────────────


class TestPreHookFeatureGates:
    """Verify create_pre_hook wires every documented feature gate.

    These are source-level assertions (not runtime) because constructing a
    full TitanHCL to invoke the hook requires ~40 attached subsystems.
    The gates are critical contract surfaces — if a future edit drops one,
    these tests fail before the cascade ever sees a regression.
    """

    @pytest.fixture
    def hook_source(self) -> str:
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        return (repo_root / "titan_hcl" / "modules" / "agno_hooks.py").read_text()

    def test_classifier_constructed_lazily(self, hook_source):
        assert "ChatTierClassifier.from_config" in hook_source
        # Only built once per closure (lazy)
        assert "_tier_classifier = None" in hook_source

    def test_active_features_extracted(self, hook_source):
        assert "active_features = _cr.active_features" in hook_source
        assert "plugin._current_chat_tier" in hook_source
        assert "plugin._current_chat_features" in hook_source
        assert "plugin._current_chat_model_class" in hook_source

    def test_user_recognition_gate(self, hook_source):
        # Both social_graph branch + KnownUserResolver branch must check feature
        assert hook_source.count('"user_recognition" in active_features') >= 3

    def test_topic_memory_gate_on_vcb(self, hook_source):
        # VCB skip-reason set when feature missing
        assert "tier_no_topic_memory" in hook_source
        # user_memories also gated
        assert '"topic_memory" in active_features' in hook_source

    def test_gatekeeper_state_gate(self, hook_source):
        # The torch sync embedder.encode block must be gated
        assert '"gatekeeper_state" in active_features' in hook_source

    def test_v5_outer_skip(self, hook_source):
        # _SkipV5Block sentinel must exist + outer feature set check
        assert "_SkipV5Block" in hook_source
        assert "_v5_active = bool(active_features & _v5_features)" in hook_source

    def test_cgn_grounding_gate(self, hook_source):
        assert '"cgn_grounding" in active_features' in hook_source

    def test_cgn_social_action_gate(self, hook_source):
        assert '"cgn_social_action" in active_features' in hook_source

    def test_reasoning_chain_gates(self, hook_source):
        # [16] meta-reasoning, [21] reasoning, [22] experience narrative,
        # [24] knowledge gap = 4+ gates
        assert hook_source.count('"reasoning_chain" in active_features') >= 4

    def test_felt_state_gates(self, hook_source):
        # [10-15] + [17] + [18] + [20] + [23] = many gates
        assert hook_source.count('"felt_state" in active_features') >= 8

    def test_directives_gate(self, hook_source):
        # Directive fetch must be gated (every tier has it, but check defensive)
        assert '"directives" in active_features' in hook_source


# ── ζ.5 per-tier model routing ─────────────────────────────────────


class TestModelClassRouting:
    """Verify _route_model_for_tier swaps agent.model.id per tier."""

    @pytest.mark.asyncio
    async def test_route_swaps_to_fast_for_greeting(self, full_config):
        from titan_hcl.modules.agno_worker import _route_model_for_tier
        from titan_hcl.modules.chat_tier_config import ChatTierClassifier

        # Fake provider with a model_class map
        class _FakeProvider:
            id = "deepseek-v3.1:671b"
            _model_class_map = {
                "fast":  "gemma3:4b",
                "light": "gemma4:31b",
                "heavy": "deepseek-v3.1:671b",
            }
            def resolve_model_class(self, name):
                return self._model_class_map.get(name, self.id)

        class _FakeModel:
            id = "deepseek-v3.1:671b"

        class _FakeAgent:
            model = _FakeModel()

        class _FakePlugin:
            _full_config = full_config
            _inference_provider = _FakeProvider()
            _tier_classifier_cache = ChatTierClassifier.from_config(full_config)

        agent = _FakeAgent()
        plugin = _FakePlugin()
        observed = []
        async with _route_model_for_tier(agent, plugin, "Hi"):
            observed.append(agent.model.id)
        # During the block, model.id should be fast model
        assert observed == ["gemma3:4b"]
        # Restored after exit
        assert agent.model.id == "deepseek-v3.1:671b"

    @pytest.mark.asyncio
    async def test_route_noop_when_same_class(self, full_config):
        from titan_hcl.modules.agno_worker import _route_model_for_tier
        from titan_hcl.modules.chat_tier_config import ChatTierClassifier

        class _FakeProvider:
            id = "deepseek-v3.1:671b"
            _model_class_map = {"heavy": "deepseek-v3.1:671b"}
            def resolve_model_class(self, name):
                return self._model_class_map.get(name, self.id)

        class _FakeModel:
            id = "deepseek-v3.1:671b"

        class _FakeAgent:
            model = _FakeModel()

        class _FakePlugin:
            _full_config = full_config
            _inference_provider = _FakeProvider()
            _tier_classifier_cache = ChatTierClassifier.from_config(full_config)

        agent = _FakeAgent()
        plugin = _FakePlugin()
        # Reasoning prompt → heavy → already the agent's id → no swap
        async with _route_model_for_tier(agent, plugin, "Why does X work?"):
            assert agent.model.id == "deepseek-v3.1:671b"

    @pytest.mark.asyncio
    async def test_route_swaps_max_tokens_for_capped_tier(self, full_config):
        """ζ.6: greeting tier has max_tokens=80 — should swap + restore."""
        from titan_hcl.modules.agno_worker import _route_model_for_tier
        from titan_hcl.modules.chat_tier_config import ChatTierClassifier

        class _FakeProvider:
            id = "deepseek-v3.1:671b"
            _model_class_map = {
                "fast":  "gemma3:4b",
                "heavy": "deepseek-v3.1:671b",
            }
            def resolve_model_class(self, name):
                return self._model_class_map.get(name, self.id)

        class _FakeModel:
            id = "deepseek-v3.1:671b"
            max_tokens = 4096  # default

        class _FakeAgent:
            model = _FakeModel()

        class _FakePlugin:
            _full_config = full_config
            _inference_provider = _FakeProvider()
            _tier_classifier_cache = ChatTierClassifier.from_config(full_config)

        agent = _FakeAgent()
        plugin = _FakePlugin()
        captured = []
        async with _route_model_for_tier(agent, plugin, "Hi"):
            captured.append((agent.model.id, agent.model.max_tokens))
        # During: model=gemma3:4b, max_tokens=80 (per fixture)
        assert captured == [("gemma3:4b", 80)]
        # Restored:
        assert agent.model.id == "deepseek-v3.1:671b"
        assert agent.model.max_tokens == 4096

    @pytest.mark.asyncio
    async def test_route_noop_when_provider_missing(self, full_config):
        from titan_hcl.modules.agno_worker import _route_model_for_tier

        class _FakeModel:
            id = "default-model"

        class _FakeAgent:
            model = _FakeModel()

        class _FakePlugin:
            _full_config = full_config
            _inference_provider = None  # not stashed

        agent = _FakeAgent()
        plugin = _FakePlugin()
        # No provider → context manager yields without mutation
        async with _route_model_for_tier(agent, plugin, "Hi"):
            assert agent.model.id == "default-model"
        assert agent.model.id == "default-model"
