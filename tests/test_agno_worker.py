"""
tests/test_agno_worker.py — Chunk C2 regression tests for agno_worker
with Agno Agent integration (D-SPEC-72 / SPEC v1.17.0).

Supersedes tests/test_agno_worker_scaffold.py (C1, deleted in C2). Covers:
  - WorkerPlugin construction + state cache surface
  - WorkerPlugin proxy property accessors (lazy, cached)
  - WorkerPlugin _extract_sources_from_findings helper
  - agno_agent_factory.create_agent end-to-end with mocked Agno Agent class
  - _init_worker_plugin_and_agent integration
  - _handle_chat_request: dispatches via agent.arun, emits CHAT_RESPONSE
  - _handle_chat_stream_request: yields CHAT_STREAM_CHUNK then done=True
  - Per-request state caching (user_id propagates pre→post hook)
  - Error paths: agent.arun raises → CHAT_RESPONSE with error field
  - _WorkerBusClient publish + subscribe contract
  - extract_sources_from_findings utility (lifted from TitanHCL)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from queue import Queue
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from titan_hcl import bus


# ────────────────────────────────────────────────────────────────────
# WorkerPlugin
# ────────────────────────────────────────────────────────────────────

class TestWorkerPlugin:
    """WorkerPlugin shim exposes the surface hooks expect."""

    def _make_plugin(self):
        from titan_hcl.modules.agno_worker_plugin import WorkerPlugin
        fake_bus = MagicMock()
        return WorkerPlugin(bus_client=fake_bus, config={"agent": {"agent_name": "T"}})

    def test_construction_initializes_state_caches(self):
        p = self._make_plugin()
        assert p._limbo_mode is False
        assert p._current_user_id is None
        assert p._current_engagement_level == 0.0
        assert p._pre_chat_user_id == ""
        assert p._pre_chat_neuromods == {}
        assert p._last_research_sources == []
        assert p._last_observation_vector is None
        assert p._last_execution_mode == ""
        assert p._last_transition_id == -1
        assert p._last_ovg_result is None
        assert p._last_sol_balance is None
        assert p._last_energy_state == "UNKNOWN"
        assert p._pending_self_composed == ""
        assert p._pending_self_composed_confidence == 0.0
        assert p._known_user_resolver is None
        assert p._verified_context_builder is None

    def test_bus_attribute_passed_through(self):
        p = self._make_plugin()
        assert p.bus is not None

    def test_full_config_stored(self):
        p = self._make_plugin()
        assert p._full_config == {"agent": {"agent_name": "T"}}

    def test_inline_tools_initially_none(self):
        p = self._make_plugin()
        assert p.maker_engine is None
        assert p._skill_registry is None

    def test_v3_core_points_to_self(self):
        p = self._make_plugin()
        assert p.v3_core is p

    def test_reflex_collector_initially_none(self):
        p = self._make_plugin()
        assert p.reflex_collector is None
        # (state_register proxy retired from WorkerPlugin — no longer asserted)

    def test_state_writes_persist(self):
        """Hook pattern: pre_hook sets _current_user_id → post_hook reads it."""
        p = self._make_plugin()
        p._current_user_id = "alice"
        p._pre_chat_neuromods = {"DA": 0.9}
        p._last_research_sources = ["Web", "X"]
        # Same instance, later access
        assert p._current_user_id == "alice"
        assert p._pre_chat_neuromods == {"DA": 0.9}
        assert p._last_research_sources == ["Web", "X"]


class TestWorkerPluginProxies:
    """Proxy attributes — lazy construction + cache."""

    def _make_plugin(self):
        from titan_hcl.modules.agno_worker_plugin import WorkerPlugin
        fake_bus = MagicMock()
        return WorkerPlugin(bus_client=fake_bus)

    def test_proxies_dict_back_compat(self):
        """agno_hooks.py does plugin._proxies.get("spirit") — must work.
        (recorder/gatekeeper keys RETIRED with the offline-RL subsystem,
        RFP_synthesis_decision_authority P1.)"""
        p = self._make_plugin()
        d = p._proxies
        for key in ["memory", "social_graph", "mind",
                    "spirit", "studio", "metabolism", "agency",
                    "soul", "mood_engine", "neuromod"]:
            assert key in d

    def test_mood_engine_aliases_mind(self):
        p = self._make_plugin()
        assert p.mood_engine is p.mind

    def test_no_retired_recorder_gatekeeper_proxies(self):
        """recorder/gatekeeper proxies are gone (offline-RL retired, P1)."""
        p = self._make_plugin()
        assert "recorder" not in p._proxies
        assert "gatekeeper" not in p._proxies
        assert not hasattr(p, "gatekeeper")

    def test_consciousness_returns_none(self):
        p = self._make_plugin()
        assert p.consciousness is None

    def test_sage_researcher_constructed_in_process(self):
        """2026-06-15 — sage_researcher is wired in-process for the chat
        research lane (was hardcoded None, which silently degraded every
        STATE_NEED_RESEARCH to empty). Real StealthSageResearcher; falls back
        to None only on init failure."""
        p = self._make_plugin()
        sr = p.sage_researcher
        assert sr is not None
        from titan_hcl.logic.sage.researcher import StealthSageResearcher
        assert isinstance(sr, StealthSageResearcher)

    def test_output_verifier_is_constructed(self):
        p = self._make_plugin()
        ov = p._output_verifier
        # Should be an OutputVerifierProxy-like object (not None)
        assert ov is not None

    def test_soul_shim_get_active_directives(self):
        """plugin.soul.get_active_directives() — used at agno_hooks.py:651."""
        p = self._make_plugin()
        soul = p.soul
        assert soul is not None
        assert callable(soul.get_active_directives)

    def test_neuromod_shim_get_stats(self):
        p = self._make_plugin()
        nm = p.neuromod
        assert nm is not None
        assert callable(nm.get_stats)
        assert nm.get_stats() == {}


class TestExtractSourcesFromFindings:
    """_extract_sources_from_findings — lifted verbatim from TitanHCL."""

    def test_empty_findings_returns_empty(self):
        from titan_hcl.modules.agno_worker_plugin import extract_sources_from_findings
        assert extract_sources_from_findings("") == []
        assert extract_sources_from_findings(None) == []

    def test_non_empty_returns_web_base(self):
        from titan_hcl.modules.agno_worker_plugin import extract_sources_from_findings
        assert extract_sources_from_findings("some research text") == ["Web"]

    def test_detects_x_search_results(self):
        from titan_hcl.modules.agno_worker_plugin import extract_sources_from_findings
        result = extract_sources_from_findings("findings [X_SEARCH_RESULTS...]")
        assert "X" in result
        assert "Web" in result

    def test_detects_document_topic(self):
        from titan_hcl.modules.agno_worker_plugin import extract_sources_from_findings
        result = extract_sources_from_findings("findings Document Topic: stuff")
        assert "Document" in result

    def test_all_sources_additive(self):
        from titan_hcl.modules.agno_worker_plugin import extract_sources_from_findings
        result = extract_sources_from_findings(
            "[X_SEARCH_RESULTS results] Document Topic: x"
        )
        assert set(result) == {"Web", "X", "Document"}

    def test_workerplugin_static_method_delegates(self):
        """plugin._extract_sources_from_findings should match the module helper."""
        from titan_hcl.modules.agno_worker_plugin import WorkerPlugin
        result = WorkerPlugin._extract_sources_from_findings("test [X_SEARCH_RESULTS...]")
        assert "X" in result
        assert "Web" in result


# ────────────────────────────────────────────────────────────────────
# agno_agent_factory.create_agent
# ────────────────────────────────────────────────────────────────────

class TestCreateAgentFactory:
    """Factory constructs an Agno Agent with the right wiring."""

    def test_create_agent_calls_agno_with_correct_wiring(self):
        """Mock out Agent class + verify create_agent invokes it with hooks/tools."""
        from titan_hcl.modules.agno_worker_plugin import WorkerPlugin

        fake_bus = MagicMock()
        plugin = WorkerPlugin(
            bus_client=fake_bus,
            config={
                "agent": {"agent_name": "TitanTest"},
                "inference": {
                    "inference_provider": "venice",
                    "venice_api_key": "key",
                },
            },
        )
        plugin.guardian = MagicMock()  # GuardianGuardrail takes plugin.guardian

        with patch("agno.agent.Agent") as MockAgent, \
             patch("agno.db.sqlite.async_sqlite.AsyncSqliteDb") as MockDb, \
             patch("titan_hcl.modules.agno_hooks.create_pre_hook") as MockPre, \
             patch("titan_hcl.modules.agno_hooks.create_post_hook") as MockPost, \
             patch("titan_hcl.modules.agno_tools.create_tools") as MockTools, \
             patch("titan_hcl.modules.agno_guardrails.GuardianGuardrail") as MockGuard:
            MockPre.return_value = "pre_hook"
            MockPost.return_value = "post_hook"
            MockTools.return_value = ["tool1", "tool2"]
            MockGuard.return_value = "guardrail"

            from titan_hcl.modules.agno_agent_factory import create_agent
            agent = create_agent(plugin)

            # Agent class was called once
            assert MockAgent.called
            kwargs = MockAgent.call_args.kwargs
            assert kwargs["pre_hooks"] == ["guardrail", "pre_hook"]
            assert kwargs["post_hooks"] == ["post_hook"]
            assert kwargs["tools"] == ["tool1", "tool2"]
            assert kwargs["name"] == "TitanTest"
            # D-SPEC-159 — agno per-run history bypass defaults ON
            # (agno_history_bypass=True) ⇒ add_history_to_context is False.
            # (create_agent now flows through build_shared_chat_context +
            # make_agent for concurrent chat — the wiring above is unchanged.)
            assert kwargs["add_history_to_context"] is False

    def test_create_agent_raises_on_unknown_provider_no_venice_fallback(self):
        """An unknown/failing inference provider must surface LOUDLY — it must
        NOT silently fall back to a hardcoded 'venice'.

        The venice fallback was REMOVED 2026-06-24 (Maker rule: no hardcoded
        inference provider) because it masked the real provider failure AND hit
        depleted Venice credits. `agno_agent_factory.build_shared_chat_context`
        now re-raises the provider-construction error. (This test previously
        asserted the old fall-back-to-venice behavior — updated to the new
        contract.)
        """
        from titan_hcl.modules.agno_worker_plugin import WorkerPlugin
        plugin = WorkerPlugin(
            bus_client=MagicMock(),
            config={
                "inference": {"inference_provider": "definitely-not-real"},
                "agent": {},
            },
        )
        plugin.guardian = MagicMock()
        with patch("agno.agent.Agent"), patch("agno.db.sqlite.async_sqlite.AsyncSqliteDb"):
            from titan_hcl.modules.agno_agent_factory import create_agent
            with pytest.raises(ValueError, match="[Uu]nknown inference provider"):
                create_agent(plugin)


# ────────────────────────────────────────────────────────────────────
# Chat dispatch via Agno Agent (C2 replaces stub)
# ────────────────────────────────────────────────────────────────────

class TestChatRequestHandler:
    """_handle_chat_request dispatches via agent.arun, emits CHAT_RESPONSE."""

    def test_chat_request_calls_agent_arun(self):
        from titan_hcl.modules.agno_worker import _handle_chat_request

        send_q = Queue()
        stats = {"in_flight": 0, "total_chats_24h": 0, "last_chat_ts": 0.0}

        agent = MagicMock()
        run_output = MagicMock()
        run_output.content = "Hello, I am Titan."
        agent.arun = AsyncMock(return_value=run_output)

        worker_plugin = MagicMock()
        worker_plugin._last_ovg_result = None
        worker_plugin._last_execution_mode = "Collaborative"

        msg = {
            "type": bus.CHAT_REQUEST,
            "src": "api_subproc",
            "rid": "rid-001",
            "payload": {
                "request_id": "req-001",
                "message": "hello",
                "user_id": "alice",
                "session_id": "s1",
            },
        }
        asyncio.run(_handle_chat_request(
            msg, agent, worker_plugin, send_q, "agno_worker", stats,
        ))

        agent.arun.assert_called_once()
        call_args = agent.arun.call_args
        assert call_args.args[0] == "hello"
        assert call_args.kwargs["session_id"] == "s1"
        assert call_args.kwargs["user_id"] == "alice"

    def test_chat_request_updates_worker_plugin_state(self):
        """The handler sets _current_user_id + _pre_chat_user_id so the hooks +
        arun see this turn's identity.

        RFP §7.B0 (B0-state): the per-turn fields are now REQUEST-SCOPED via a
        ContextVar bag, so they are visible DURING the chat task (where arun +
        the hooks run) but do NOT leak out of it afterward (that leakage is the
        exact cross-contamination B0 fixes). So we capture the values at
        arun-time (inside the task) and assert isolation after the task returns.
        """
        from titan_hcl.modules.agno_worker import _handle_chat_request

        send_q = Queue()
        stats = {"in_flight": 0, "total_chats_24h": 0, "last_chat_ts": 0.0}
        agent = MagicMock()
        run_output = MagicMock()
        run_output.content = "hi"

        # Use real WorkerPlugin so state writes actually land
        from titan_hcl.modules.agno_worker_plugin import WorkerPlugin
        worker_plugin = WorkerPlugin(bus_client=MagicMock())

        seen = {}

        async def _capturing_arun(message, session_id=None, user_id=None):
            # Runs INSIDE the chat task → must observe this turn's state.
            seen["current_user_id"] = worker_plugin._current_user_id
            seen["pre_chat_user_id"] = worker_plugin._pre_chat_user_id
            return run_output

        agent.arun = _capturing_arun

        msg = {
            "type": bus.CHAT_REQUEST,
            "src": "api_subproc",
            "payload": {"message": "hi", "user_id": "bob", "session_id": "s"},
        }
        asyncio.run(_handle_chat_request(
            msg, agent, worker_plugin, send_q, "agno_worker", stats,
        ))
        # set correctly DURING the turn (what the hooks/arun actually see)
        assert seen["current_user_id"] == "bob"
        assert seen["pre_chat_user_id"] == "bob"
        # B0-state isolation: NOT leaked to the process-global scope after the task
        assert worker_plugin._current_user_id is None
        assert worker_plugin._pre_chat_user_id == ""

    def test_chat_request_emits_chat_response(self):
        from titan_hcl.modules.agno_worker import _handle_chat_request
        send_q = Queue()
        stats = {"in_flight": 0, "total_chats_24h": 0, "last_chat_ts": 0.0}
        agent = MagicMock()
        run_output = MagicMock()
        run_output.content = "Titan response"
        agent.arun = AsyncMock(return_value=run_output)
        worker_plugin = MagicMock()
        worker_plugin._last_ovg_result = None
        worker_plugin._last_execution_mode = "Verified"

        msg = {
            "type": bus.CHAT_REQUEST,
            "src": "api_subproc",
            "rid": "rid-002",
            "payload": {"request_id": "r2", "message": "x", "session_id": "s2"},
        }
        asyncio.run(_handle_chat_request(
            msg, agent, worker_plugin, send_q, "agno_worker", stats,
        ))

        out = send_q.get(timeout=1.0)
        assert out["type"] == bus.CHAT_RESPONSE
        assert out["dst"] == "api_subproc"
        body = out["payload"]
        assert body["request_id"] == "r2"
        assert body["response"] == "Titan response"
        assert body["session_id"] == "s2"
        assert body["mode"] == "Verified"
        assert body["error"] is None

    def test_chat_request_error_in_arun_returns_error_response(self):
        from titan_hcl.modules.agno_worker import _handle_chat_request
        send_q = Queue()
        stats = {"in_flight": 0, "total_chats_24h": 0, "last_chat_ts": 0.0}
        agent = MagicMock()
        agent.arun = AsyncMock(side_effect=RuntimeError("LLM upstream down"))
        worker_plugin = MagicMock()
        worker_plugin._last_ovg_result = None
        worker_plugin._last_execution_mode = ""

        msg = {
            "type": bus.CHAT_REQUEST,
            "src": "api_subproc",
            "payload": {"request_id": "r3", "message": "x", "session_id": "s"},
        }
        asyncio.run(_handle_chat_request(
            msg, agent, worker_plugin, send_q, "agno_worker", stats,
        ))

        out = send_q.get(timeout=1.0)
        body = out["payload"]
        assert body["error"] == "LLM upstream down"
        assert body["response"] == ""
        # On error, last_chat_ts should NOT update
        assert stats["last_chat_ts"] == 0.0

    def test_chat_request_dream_state_gate_buffers_and_returns_dream_mode(self):
        """D-SPEC-56 + Chunk H: when dream_state.bin.is_dreaming=True, the
        worker emits DREAM_INBOX_ENQUEUE + returns a dream-mode CHAT_RESPONSE
        instead of calling agent.arun()."""
        from titan_hcl.modules.agno_worker import _handle_chat_request

        send_q = Queue()
        stats = {"in_flight": 0, "total_chats_24h": 0, "last_chat_ts": 0.0}

        # Real WorkerPlugin so we can attach a dream_reader mock
        from titan_hcl.modules.agno_worker_plugin import WorkerPlugin
        bus_mock = MagicMock()
        worker_plugin = WorkerPlugin(bus_client=bus_mock)

        # Mock dream_reader to report is_dreaming=True
        dream_reader_mock = MagicMock()
        dream_reader_mock.read.return_value = {
            "is_dreaming": True,
            "recovery_pct": 42.0,
            "remaining_epochs": 100,
            "wake_transition": False,
        }
        worker_plugin._dream_reader = dream_reader_mock

        # Agent should NOT be called in dream mode
        agent = MagicMock()
        agent.arun = AsyncMock(return_value=MagicMock(content="should not arrive"))

        msg = {
            "type": bus.CHAT_REQUEST,
            "src": "api_subproc",
            "rid": "rid-dream",
            "payload": {
                "request_id": "req-dream",
                "message": "hello during dream",
                "user_id": "alice",
                "session_id": "s",
                "channel": "web",
                "is_maker": False,
            },
        }
        asyncio.run(_handle_chat_request(
            msg, agent, worker_plugin, send_q, "agno_worker", stats,
        ))

        # Agent.arun MUST NOT have been called
        agent.arun.assert_not_called()

        # bus.publish should have emitted DREAM_INBOX_ENQUEUE
        publish_calls = bus_mock.publish.call_args_list
        assert len(publish_calls) >= 1
        first_msg = publish_calls[0].args[0]
        assert first_msg["type"] == bus.DREAM_INBOX_ENQUEUE
        assert first_msg["payload"]["message"] == "hello during dream"
        assert first_msg["payload"]["priority"] == 1  # not maker

        # CHAT_RESPONSE should be dream-mode
        out = send_q.get(timeout=1.0)
        assert out["type"] == bus.CHAT_RESPONSE
        body = out["payload"]
        assert body["mode"] == "dreaming"
        assert body["mood"] == "sleeping"
        assert body["state_snapshot"]["is_dreaming"] is True
        assert body["state_snapshot"]["recovery_pct"] == 42.0

    def test_chat_request_dream_state_maker_emits_wake_request(self):
        """Maker messages during dream emit DREAM_WAKE_REQUEST + DREAM_INBOX_ENQUEUE."""
        from titan_hcl.modules.agno_worker import _handle_chat_request

        send_q = Queue()
        stats = {"in_flight": 0, "total_chats_24h": 0, "last_chat_ts": 0.0}

        from titan_hcl.modules.agno_worker_plugin import WorkerPlugin
        bus_mock = MagicMock()
        worker_plugin = WorkerPlugin(bus_client=bus_mock)
        dream_reader_mock = MagicMock()
        dream_reader_mock.read.return_value = {
            "is_dreaming": True, "recovery_pct": 30.0,
            "remaining_epochs": 50, "wake_transition": True,
        }
        worker_plugin._dream_reader = dream_reader_mock

        agent = MagicMock()
        agent.arun = AsyncMock()

        msg = {
            "type": bus.CHAT_REQUEST, "src": "api_subproc", "rid": "rid-maker-dream",
            "payload": {
                "request_id": "req-maker-dream",
                "message": "wake up",
                "user_id": "maker",
                "session_id": "s",
                "is_maker": True,
            },
        }
        asyncio.run(_handle_chat_request(
            msg, agent, worker_plugin, send_q, "agno_worker", stats,
        ))

        # Should have published 2 events: DREAM_INBOX_ENQUEUE + DREAM_WAKE_REQUEST
        msg_types = [c.args[0]["type"] for c in bus_mock.publish.call_args_list]
        assert bus.DREAM_INBOX_ENQUEUE in msg_types
        assert bus.DREAM_WAKE_REQUEST in msg_types
        # Maker priority should be 0 (vs 1 for non-maker)
        enqueue_call = next(c for c in bus_mock.publish.call_args_list
                            if c.args[0]["type"] == bus.DREAM_INBOX_ENQUEUE)
        assert enqueue_call.args[0]["payload"]["priority"] == 0

    def test_chat_request_propagates_ovg_data_from_plugin(self):
        from titan_hcl.modules.agno_worker import _handle_chat_request
        send_q = Queue()
        stats = {"in_flight": 0, "total_chats_24h": 0, "last_chat_ts": 0.0}
        agent = MagicMock()
        run_output = MagicMock()
        run_output.content = "ok"
        agent.arun = AsyncMock(return_value=run_output)

        # Plugin has an OVG result from the post-hook
        fake_ovg = MagicMock()
        fake_ovg.passed = True
        fake_ovg.guard_alert = None
        fake_ovg.guard_message = "[VERIFIED]"
        fake_ovg.block_height = 42
        fake_ovg.merkle_root = "0xabc"
        fake_ovg.signature = "sig123"

        worker_plugin = MagicMock()
        worker_plugin._last_ovg_result = fake_ovg
        worker_plugin._last_execution_mode = "Verified"

        msg = {
            "type": bus.CHAT_REQUEST,
            "src": "api_subproc",
            "payload": {"request_id": "r4", "message": "x", "session_id": "s"},
        }
        asyncio.run(_handle_chat_request(
            msg, agent, worker_plugin, send_q, "agno_worker", stats,
        ))

        out = send_q.get(timeout=1.0)
        ovg = out["payload"]["ovg_data"]
        assert ovg is not None
        assert ovg["verified"] is True
        assert ovg["block_height"] == 42
        assert ovg["merkle_root"] == "0xabc"
        assert ovg["signature"] == "sig123"


# ────────────────────────────────────────────────────────────────────
# Stream chat handler
# ────────────────────────────────────────────────────────────────────

class TestChatStreamRequestHandler:
    """_handle_chat_stream_request yields chunks then done=True."""

    def test_stream_handler_emits_chunks_then_done(self):
        """D-SPEC-78 Chunk δ: agno_worker runs agent.arun() to completion
        (PostHook does OVG), then chunks the verified response into
        ~200-char segments for SSE delivery. Final chunk carries ovg_headers.
        """
        from titan_hcl.modules.agno_worker import _handle_chat_stream_request

        send_q = Queue()
        stats = {"in_flight": 0, "total_chats_24h": 0, "last_chat_ts": 0.0}
        worker_plugin = MagicMock()
        worker_plugin._chat_ctx = None  # legacy path → uses the passed agent mock
        # OVG result with ovg_data dict (D-SPEC-74 VerifiedResult shape)
        worker_plugin._last_ovg_result.ovg_data = {
            "verified": True, "signature": "sig123",
            "block_height": 42, "merkle_root": "mr"}

        # Short response → single chunk + done marker
        run_output = MagicMock()
        run_output.content = "hello world"
        agent = MagicMock()
        agent.arun = AsyncMock(return_value=run_output)

        msg = {
            "type": bus.CHAT_STREAM_REQUEST,
            "src": "api_subproc",
            "rid": "stream-rid",
            "payload": {"request_id": "rs1", "message": "hi", "session_id": "s"},
        }
        asyncio.run(_handle_chat_stream_request(
            msg, agent, worker_plugin, send_q, "agno_worker", stats,
        ))

        chunks: list = []
        while not send_q.empty():
            chunks.append(send_q.get_nowait())
        # §7.B (B.4) live-progress frames (carry a `phase`: thinking/
        # writing-reply) precede the SSE delivery — filter to the real
        # content/done frames (phase is None).
        sse = [c for c in chunks if not c["payload"].get("phase")]
        # Short response: 1 content segment + 1 done marker
        assert len(sse) == 2
        assert sse[0]["payload"]["chunk"] == "hello world"
        assert sse[0]["payload"]["done"] is False
        assert sse[1]["payload"]["done"] is True
        # Done frame carries ovg_headers
        assert sse[1]["payload"]["ovg_headers"]["verified"] is True
        assert sse[1]["payload"]["ovg_headers"]["signature"] == "sig123"

    def test_stream_handler_segments_long_response(self):
        """Long responses (>200 chars) are split at sentence/word boundaries."""
        from titan_hcl.modules.agno_worker import (
            _handle_chat_stream_request, _segment_for_stream)

        # Direct test of segmenter
        long_text = ("This is sentence one. " * 20)  # ~440 chars
        segs = _segment_for_stream(long_text)
        assert len(segs) >= 2, f"expected multi-segment, got {len(segs)}"
        assert "".join(segs) == long_text, "lossless reconstruction"

        # Short text → single segment
        assert _segment_for_stream("short") == ["short"]
        assert _segment_for_stream("") == [""]

    def test_stream_handler_error_emits_done_with_error_field(self):
        from titan_hcl.modules.agno_worker import _handle_chat_stream_request

        send_q = Queue()
        stats = {"in_flight": 0, "total_chats_24h": 0, "last_chat_ts": 0.0}
        worker_plugin = MagicMock()
        worker_plugin._chat_ctx = None  # legacy path → uses the passed agent mock

        agent = MagicMock()
        agent.arun = AsyncMock(side_effect=RuntimeError("stream upstream fail"))

        msg = {
            "type": bus.CHAT_STREAM_REQUEST,
            "src": "api_subproc",
            "payload": {"request_id": "rs-err", "message": "x"},
        }
        asyncio.run(_handle_chat_stream_request(
            msg, agent, worker_plugin, send_q, "agno_worker", stats,
        ))
        # Drain the queue (live-progress frames may precede the error/done
        # frame) and assert on the terminal done frame.
        frames = []
        while not send_q.empty():
            frames.append(send_q.get_nowait())
        done_frames = [f for f in frames if f["payload"].get("done") is True]
        assert len(done_frames) == 1
        assert done_frames[0]["payload"]["error"] == "stream upstream fail"


# ────────────────────────────────────────────────────────────────────
# Public surface
# ────────────────────────────────────────────────────────────────────

class TestPublicSurface:
    """agno_worker module exports the canonical entry function + helpers."""

    def test_agno_worker_main_is_callable(self):
        from titan_hcl.modules.agno_worker import agno_worker_main
        assert callable(agno_worker_main)

    def test_init_worker_plugin_and_agent_callable(self):
        from titan_hcl.modules.agno_worker import _init_worker_plugin_and_agent
        assert callable(_init_worker_plugin_and_agent)

    def test_state_publisher_is_importable(self):
        from titan_hcl.modules.agno_worker import AgnoStatePublisher
        assert AgnoStatePublisher is not None

    def test_worker_plugin_is_importable_from_modules(self):
        from titan_hcl.modules.agno_worker_plugin import WorkerPlugin
        assert WorkerPlugin is not None

    def test_create_agent_factory_is_importable(self):
        from titan_hcl.modules.agno_agent_factory import create_agent
        assert callable(create_agent)

    def test_agno_hooks_in_modules(self):
        from titan_hcl.modules.agno_hooks import create_pre_hook, create_post_hook
        assert callable(create_pre_hook)
        assert callable(create_post_hook)

    def test_agno_tools_in_modules(self):
        from titan_hcl.modules.agno_tools import create_tools
        assert callable(create_tools)

    def test_agno_guardrails_in_modules(self):
        from titan_hcl.modules.agno_guardrails import GuardianGuardrail
        assert GuardianGuardrail is not None

    def test_legacy_agent_module_deleted(self):
        """titan_hcl.agent module no longer exists (moved to modules/)."""
        with pytest.raises(ImportError):
            import titan_hcl.agent  # noqa: F401


# ═════════════════════════════════════════════════════════════════════
# D-SPEC-76 (SPEC v1.18.0) — agno session pre-warm LRU
# ═════════════════════════════════════════════════════════════════════

class TestSessionLRUCache:
    """_handle_chat_request maintains per-(user_id, session_id) LRU cache
    with hit/miss counters surfaced in stats_ref → agno_state.bin."""

    def _make_stats(self, capacity: int = 3):
        return {
            "in_flight": 0,
            "total_chats_24h": 0,
            "last_chat_ts": 0.0,
            "_session_cache_capacity": capacity,
            "session_cache_size": 0,
            "session_hits": 0,
            "session_misses": 0,
        }

    def _make_msg(self, user, session):
        return {
            "type": bus.CHAT_REQUEST,
            "src": "api_subproc",
            "payload": {
                "request_id": f"r-{user}-{session}",
                "message": "hi",
                "user_id": user,
                "session_id": session,
            },
        }

    def _run(self, msg, stats):
        from titan_hcl.modules.agno_worker import _handle_chat_request
        agent = MagicMock()
        run_output = MagicMock()
        run_output.content = "ok"
        agent.arun = AsyncMock(return_value=run_output)
        worker_plugin = MagicMock()
        worker_plugin._last_ovg_result = None
        send_q = Queue()
        asyncio.run(_handle_chat_request(
            msg, agent, worker_plugin, send_q, "agno_worker", stats,
        ))
        send_q.get(timeout=1.0)

    def test_first_request_is_miss(self):
        stats = self._make_stats()
        self._run(self._make_msg("alice", "s1"), stats)
        assert stats["session_misses"] == 1
        assert stats["session_hits"] == 0
        assert stats["session_cache_size"] == 1

    def test_same_session_is_hit(self):
        stats = self._make_stats()
        self._run(self._make_msg("alice", "s1"), stats)
        self._run(self._make_msg("alice", "s1"), stats)
        assert stats["session_misses"] == 1
        assert stats["session_hits"] == 1
        assert stats["session_cache_size"] == 1

    def test_different_user_is_miss(self):
        """LRU keyed by (user_id, session_id) — same session_id from
        different user does NOT hit."""
        stats = self._make_stats()
        self._run(self._make_msg("alice", "s1"), stats)
        self._run(self._make_msg("bob", "s1"), stats)
        assert stats["session_misses"] == 2
        assert stats["session_cache_size"] == 2

    def test_eviction_on_capacity_exceeded(self):
        """When capacity=3 and 4 distinct sessions arrive, oldest evicted."""
        stats = self._make_stats(capacity=3)
        for i in range(4):
            self._run(self._make_msg("user", f"s{i}"), stats)
        # Cap holds 3; first one (s0) evicted
        assert stats["session_cache_size"] == 3
        assert stats["session_misses"] == 4
        # Re-touch s0 → should be a miss (it was evicted)
        self._run(self._make_msg("user", "s0"), stats)
        assert stats["session_misses"] == 5

    def test_recency_promotes_session(self):
        """LRU touch on hit moves the entry to most-recent; the eviction
        target on next insert is the LEAST-recently-accessed entry."""
        stats = self._make_stats(capacity=2)
        # Setup: cache [a, b]
        self._run(self._make_msg("u", "a"), stats)   # miss
        self._run(self._make_msg("u", "b"), stats)   # miss
        # Touch a → cache [b, a] (a most-recent, b least-recent)
        self._run(self._make_msg("u", "a"), stats)   # hit
        # Add c → evicts b (least recent), cache becomes [a, c]
        self._run(self._make_msg("u", "c"), stats)   # miss
        # a should STILL be cached because we touched it before c was added
        self._run(self._make_msg("u", "a"), stats)   # hit (a survived)
        # 3 misses, 2 hits
        assert stats["session_hits"] == 2
        assert stats["session_misses"] == 3
