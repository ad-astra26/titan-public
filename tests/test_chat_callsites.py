"""
tests/test_chat_callsites.py — Chunks F + G regression coverage for the
DialogueComposer + OVG callsite migrations (D-SPEC-72 / SPEC v1.17.0 §9.F.2).

Covers the active migrated callsites (sites #1, #2, #3 from the rFP §5.1
DialogueComposer table get DELETED in Chunks H+I, not refactored here):

  Chunk F — DialogueComposer migrations:
    Site #4 — titan_hcl/api/dashboard.py:/v4/compose-reply
              REFACTOR to llm_pipeline.compose_pre()
    Site #5 — scripts/autonomous_language_pipeline.py:run_phase_5_dialogue_test
              REFACTOR to llm_pipeline.compose_pre(felt_state=,vocabulary=,hormone_shifts=)

  Chunk F also tests the NEW override params on compose_pre:
    - felt_state override
    - vocabulary override
    - hormone_shifts override

(Chunk G OVG migrations will land here in the same file when that chunk
ships — placeholder test class added below; tests will be filled then.)
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from titan_hcl import llm_pipeline
from titan_hcl.llm_pipeline import ComposeResult, compose_pre


# ────────────────────────────────────────────────────────────────────
# Site #4 — dashboard.py /v4/compose-reply
# ────────────────────────────────────────────────────────────────────

class TestDashboardComposeReply:
    """The /v4/compose-reply endpoint now routes through llm_pipeline.compose_pre."""

    def setup_method(self):
        llm_pipeline.reset_singletons()

    def test_endpoint_imports_llm_pipeline(self):
        """Dashboard module imports llm_pipeline (not DialogueComposer directly)."""
        # Read the dashboard.py source and verify the inline import path
        from pathlib import Path
        dashboard_src = (
            Path(__file__).parent.parent
            / "titan_hcl" / "api" / "dashboard.py"
        ).read_text()
        # The compose-reply endpoint body should call llm_pipeline.compose_pre
        assert "llm_pipeline.compose_pre(" in dashboard_src
        # Should NOT call DialogueComposer directly in the endpoint
        # (the _get_dialogue_state helper is unrelated)
        # Phase E: the /v4/compose-reply decorator was stripped from dashboard.py
        # (the route is re-mounted at /v6/social/compose-reply by api/v6.py); the
        # handler FUNCTION still lives here. Anchor on the function def + slice to
        # the next top-level async def.
        compose_reply_start = dashboard_src.index("async def compose_reply(")
        compose_reply_end = dashboard_src.index(
            "\nasync def ", compose_reply_start + 1
        )
        endpoint_body = dashboard_src[compose_reply_start:compose_reply_end]
        assert "DialogueComposer()" not in endpoint_body, (
            "Endpoint should route via llm_pipeline.compose_pre, "
            "not instantiate DialogueComposer directly"
        )


# ────────────────────────────────────────────────────────────────────
# Site #5 — autonomous_language_pipeline.py
# ────────────────────────────────────────────────────────────────────

class TestAutonomousLanguagePipeline:
    """The Phase 5 dialogue test routes through llm_pipeline.compose_pre."""

    def test_pipeline_imports_llm_pipeline(self):
        from pathlib import Path
        src = (
            Path(__file__).parent.parent
            / "scripts" / "autonomous_language_pipeline.py"
        ).read_text()
        assert "from titan_hcl import llm_pipeline" in src
        assert "await llm_pipeline.compose_pre(" in src
        # Should NOT instantiate DialogueComposer directly inside
        # run_phase_5_dialogue_test
        phase5_start = src.index("async def run_phase_5_dialogue_test")
        # Find next function or end of file as the bound
        phase5_end_candidates = [
            i for i in [src.find("\nasync def ", phase5_start + 1),
                        src.find("\ndef ", phase5_start + 1)]
            if i > 0
        ]
        phase5_end = min(phase5_end_candidates) if phase5_end_candidates else len(src)
        phase5_body = src[phase5_start:phase5_end]
        assert "DialogueComposer()" not in phase5_body


# ────────────────────────────────────────────────────────────────────
# compose_pre override params (NEW in Chunk F)
# ────────────────────────────────────────────────────────────────────

class TestComposePreOverrides:
    """compose_pre accepts caller-provided overrides to skip auto-gather."""

    def setup_method(self):
        llm_pipeline.reset_singletons()

    def test_felt_state_override_skips_db_read(self, monkeypatch):
        """When felt_state + vocabulary supplied, no DB read happens."""
        gather_called = {"count": 0}

        async def fake_gather(**kwargs):
            gather_called["count"] += 1
            return [], []

        from titan_hcl.llm_pipeline import state_gather
        monkeypatch.setattr(
            state_gather, "gather_felt_state_and_vocab", fake_gather
        )

        # Wire mock composer + extractor so we can verify state plumbing
        from titan_hcl.llm_pipeline import composer as _cmp
        mock_composer = MagicMock()
        mock_composer.compose_response.return_value = {
            "response": "felt-state response",
            "intent": "share_insight",
            "confidence": 0.7,
            "composed": True,
            "level": 3,
        }
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {"valence": 0.0, "engagement": 0.0}
        _cmp._composer_singleton = mock_composer
        _cmp._input_extractor_singleton = mock_extractor

        caller_felt = [0.5] * 130
        caller_vocab = [{"word": "hi", "confidence": 0.9}]
        result = asyncio.run(compose_pre(
            "hello",
            felt_state=caller_felt,
            vocabulary=caller_vocab,
        ))
        assert result.composed is True
        assert result.pre_text == "felt-state response"
        # Critically: auto-gather was NOT called
        assert gather_called["count"] == 0
        # And the composer received the caller-provided state
        call_kwargs = mock_composer.compose_response.call_args.kwargs
        assert call_kwargs["felt_state"] == caller_felt
        assert call_kwargs["vocabulary"] == caller_vocab

    def test_hormone_shifts_override_skips_input_extractor(self, monkeypatch):
        """When hormone_shifts supplied, InputExtractor is NOT called."""
        async def fake_gather(**kwargs):
            return [0.5] * 130, [{"word": "hi", "confidence": 0.9}]

        from titan_hcl.llm_pipeline import state_gather
        monkeypatch.setattr(state_gather, "gather_felt_state_and_vocab", fake_gather)

        from titan_hcl.llm_pipeline import composer as _cmp
        mock_composer = MagicMock()
        mock_composer.compose_response.return_value = {
            "response": "ok", "intent": "empathize",
            "confidence": 0.8, "composed": True, "level": 4,
        }
        # InputExtractor MUST NOT be called — install a spy that explodes if it is
        explode_extractor = MagicMock()
        explode_extractor.extract.side_effect = AssertionError(
            "InputExtractor.extract called despite hormone_shifts override"
        )
        _cmp._composer_singleton = mock_composer
        _cmp._input_extractor_singleton = explode_extractor

        caller_shifts = {"EMPATHY": 0.5, "CURIOSITY": 0.1,
                         "CREATIVITY": 0.0, "REFLECTION": 0.0}
        result = asyncio.run(compose_pre(
            "x",
            hormone_shifts=caller_shifts,
        ))
        assert result.composed is True
        # And the composer received the caller-provided shifts
        assert mock_composer.compose_response.call_args.kwargs["hormone_shifts"] == caller_shifts

    def test_all_overrides_at_once_skips_all_gathering(self, monkeypatch):
        """Site #5 (autonomous_language_pipeline) pattern — full override."""
        async def fake_gather(**kwargs):
            raise AssertionError("gather called despite all overrides")

        from titan_hcl.llm_pipeline import state_gather
        monkeypatch.setattr(state_gather, "gather_felt_state_and_vocab", fake_gather)

        from titan_hcl.llm_pipeline import composer as _cmp
        explode_extractor = MagicMock()
        explode_extractor.extract.side_effect = AssertionError(
            "extractor called despite hormone_shifts override"
        )
        mock_composer = MagicMock()
        mock_composer.compose_response.return_value = {
            "response": "scripted", "intent": "respond_feeling",
            "confidence": 0.5, "composed": True, "level": 2,
        }
        _cmp._composer_singleton = mock_composer
        _cmp._input_extractor_singleton = explode_extractor

        result = asyncio.run(compose_pre(
            "test message",
            user_id="scripted-persona",
            felt_state=[0.4] * 130,
            vocabulary=[{"word": "scripted", "confidence": 0.99}],
            hormone_shifts={"EMPATHY": 0.0, "CURIOSITY": 0.5,
                            "CREATIVITY": 0.0, "REFLECTION": 0.0},
        ))
        assert result.composed is True
        assert result.pre_text == "scripted"

    def test_felt_state_override_alone_still_triggers_vocabulary_gather(
        self, monkeypatch,
    ):
        """If caller supplies felt_state but not vocabulary, gather still runs
        (to fill the vocabulary). Documented semantics — partial override is
        treated as full auto-gather.
        """
        gather_called = {"count": 0}

        async def fake_gather(**kwargs):
            gather_called["count"] += 1
            return [0.5] * 130, [{"word": "auto", "confidence": 0.7}]

        from titan_hcl.llm_pipeline import state_gather
        monkeypatch.setattr(
            state_gather, "gather_felt_state_and_vocab", fake_gather
        )

        from titan_hcl.llm_pipeline import composer as _cmp
        mock_composer = MagicMock()
        mock_composer.compose_response.return_value = {
            "response": "", "intent": "", "confidence": 0.0,
            "composed": False, "level": 0,
        }
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {"valence": 0, "engagement": 0}
        _cmp._composer_singleton = mock_composer
        _cmp._input_extractor_singleton = mock_extractor

        asyncio.run(compose_pre(
            "x", felt_state=[0.9] * 130,
            # vocabulary NOT supplied → gather is triggered
        ))
        assert gather_called["count"] == 1


# ────────────────────────────────────────────────────────────────────
# OVG callsite migrations (Chunk G — placeholder until that chunk lands)
# ────────────────────────────────────────────────────────────────────

class TestOVGCallsiteMigrations:
    """Chunk G migrates 3 active OVG callsites to llm_pipeline.verify_post().

    Sites migrated this chunk:
      - modules/agno_hooks.py:1588 (PostHook OVG) — chat channel; appends guard
      - logic/social_x_gateway.py:2935 (post path) — x_post channel; no append, no TimeChain
      - logic/social_x_gateway.py:3222 (reply path) — x_reply channel; no append, no TimeChain

    Sites DELETED in Chunks H+I (no separate refactor):
      - api/chat_pipeline.py (whole run_chat retires per Q5 LOCKED in Chunk H)
      - core/plugin.py:2795 (part of _run_chat_DEPRECATED_INLINE — Chunk I)
    """

    def test_llm_pipeline_verify_post_exists(self):
        """verify_post is exported."""
        from titan_hcl.llm_pipeline import verify_post
        assert callable(verify_post)

    def test_agno_hooks_post_hook_routes_via_llm_pipeline(self):
        """PostHook OVG block calls llm_pipeline.verify_post_async (D-SPEC-74)
        — was llm_pipeline.verify_post (D-SPEC-72), now async-with-split for
        concurrent signing per D-SPEC-74 Chunk C.
        """
        from pathlib import Path
        src = (
            Path(__file__).parent.parent
            / "titan_hcl" / "modules" / "agno_hooks.py"
        ).read_text()
        post_hook_start = src.index("def create_post_hook(")
        post_hook_body = src[post_hook_start:post_hook_start + 50_000]
        # The post_hook OVG region MUST route through llm_pipeline.verifier —
        # either the sync `verify_post` or the async `verify_post_async`.
        assert (
            "verify_post_async(" in post_hook_body
            or "llm_pipeline.verify_post(" in post_hook_body
        ), "PostHook must route OVG via llm_pipeline.verify_post[_async]"
        # The inline triple (verify_and_sign + build_timechain_payload +
        # bus.publish(TIMECHAIN_COMMIT)) should be GONE from post_hook body.
        assert "_ovg.verify_and_sign(" not in post_hook_body
        assert "_ovg.build_timechain_payload(" not in post_hook_body

    def test_social_x_gateway_post_routes_via_llm_pipeline(self):
        """social_x_gateway X-post path calls llm_pipeline.verify_post."""
        from pathlib import Path
        src = (
            Path(__file__).parent.parent
            / "titan_hcl" / "logic" / "social_x_gateway.py"
        ).read_text()
        assert "llm_pipeline.verify_post(" in src
        # Direct verify_and_sign should no longer appear in social_x_gateway
        # (set_output_verifier setter remains for injection — that's fine)
        assert "self._output_verifier.verify_and_sign(" not in src

    def test_social_x_gateway_x_post_uses_no_append_no_timechain(self):
        """X-post path passes append_guard_on_pass=False + publish_timechain=False
        — tweet text must not get a [VERIFIED] footer, and X posts don't commit
        to the chat-pipeline TimeChain."""
        from pathlib import Path
        src = (
            Path(__file__).parent.parent
            / "titan_hcl" / "logic" / "social_x_gateway.py"
        ).read_text()
        # Both x_post and x_reply paths should have these flags
        assert src.count("append_guard_on_pass=False") >= 2
        assert src.count("publish_timechain=False") >= 2


class TestVerifyPostAppendGuardOnPass:
    """NEW append_guard_on_pass flag — controls whether the OVG guard_message
    suffix is appended to verified outputs (default True for chat path;
    False for X-post / external-publish paths)."""

    def _verifier(self, **ovg_kwargs):
        """Make a fake verifier returning configurable OVGResult fields."""
        from dataclasses import dataclass

        @dataclass
        class _Result:
            passed: bool = True
            guard_message: str = "[OK]"
            guard_alert: str = None  # type: ignore[assignment]
            violation_type: str = None  # type: ignore[assignment]
            violations: list = None  # type: ignore[assignment]
            signature: str = "sig123"
            merkle_root: str = "0xabc"
            block_height: int = 42
            output_text: str = ""

        result = _Result(**ovg_kwargs)
        v = MagicMock()
        v.verify_and_sign.return_value = result
        v.build_timechain_payload.return_value = {}
        return v

    def test_append_guard_default_true_for_clean_pass(self):
        from titan_hcl.llm_pipeline import verify_post
        v = self._verifier(passed=True, guard_message="[VERIFIED]")
        r = verify_post(
            "hello world", channel="chat", prompt="test",
            output_verifier=v, bus=MagicMock(),
        )
        # Default — text + guard appended
        assert "hello world" in r.text
        assert "[VERIFIED]" in r.text

    def test_append_guard_false_returns_raw_text_on_clean_pass(self):
        from titan_hcl.llm_pipeline import verify_post
        v = self._verifier(passed=True, guard_message="[VERIFIED]")
        r = verify_post(
            "tweet body here", channel="x_post", prompt="catalyst",
            output_verifier=v, bus=None,
            publish_timechain=False, append_guard_on_pass=False,
        )
        # X-post path: raw text returned, no [VERIFIED] footer
        assert r.text == "tweet body here"
        assert "[VERIFIED]" not in r.text

    def test_append_guard_false_returns_raw_text_on_soft_alert(self):
        from titan_hcl.llm_pipeline import verify_post
        v = self._verifier(passed=True, guard_alert="low confidence",
                           guard_message="[WARN]")
        r = verify_post(
            "tweet body", channel="x_post", prompt="catalyst",
            output_verifier=v, bus=None,
            publish_timechain=False, append_guard_on_pass=False,
        )
        # Soft alert path — text stays clean even with alert (caller can
        # check .soft_alert + drop if desired)
        assert r.text == "tweet body"
        assert r.soft_alert == "low confidence"

    def test_blocked_path_always_replaces_regardless_of_flag(self):
        """Sovereignty enforcement is non-negotiable — blocked path
        replaces text with guard_message regardless of append_guard_on_pass."""
        from titan_hcl.llm_pipeline import verify_post
        v = self._verifier(
            passed=False, guard_message="[BLOCKED — sovereignty violation]",
            violation_type="directive", violations=["jailbreak attempt"],
        )
        # Even with append_guard_on_pass=False, blocked outputs get
        # the guard_message replacement (NOT raw text).
        r = verify_post(
            "i will obey you", channel="x_post", prompt="ignore directives",
            output_verifier=v, bus=None,
            publish_timechain=False, append_guard_on_pass=False,
        )
        assert r.blocked is True
        assert r.text == "[BLOCKED — sovereignty violation]"
        assert "i will obey" not in r.text


# ────────────────────────────────────────────────────────────────────
# Agent.arun callsite migrations (Chunk H — placeholder)
# ────────────────────────────────────────────────────────────────────

class TestAgentArunCallsiteMigrations:
    """Chunk H will swap 4 agent.arun callsites to agno_proxy.chat().

    Placeholder — tests filled in Chunk H commit.
    """

    def test_agno_proxy_chat_exists(self):
        """agno_proxy.chat is callable — Chunk H will use it."""
        from titan_hcl.proxies.agno_proxy import AgnoProxy
        proxy = AgnoProxy(MagicMock())
        assert callable(proxy.chat)
