"""
test_sage_subprocess_migration.py — Microkernel v2 Layer 2 (2026-04-28)

Tests for the Sage subprocess migration:
  - SageEncoder is a lightweight parent-safe encoder (no LazyMemmapStorage)
  - SAGE_RECORD_TRANSITION bus constant + payload shape
  - rl_worker handles SAGE_RECORD_TRANSITION via _handle_sage_record_transition
  - SageGuardian routes divine trauma through the injected callable when
    provided, falls back to recorder.record_transition otherwise
  - TitanPlugin._publish_sage_record_transition: bus path when bus attached,
    local fallback when bus is None

Closes BUG-SAGE-INSTANTIATED-IN-PARENT (architectural code closure;
runtime impact in V6 prod is zero — V6 plugin path doesn't load Sage in
parent — but this fix makes the legacy/MCP path lean and cements the
contract for L3 Scholar/Gatekeeper extraction).
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest


def test_sage_encoder_lightweight() -> None:
    """SageEncoder constructs without scratch_dir / LazyMemmapStorage."""
    from titan_plugin.core.sage.recorder import SageEncoder

    enc = SageEncoder(config={"sage_memory": {"embedding_dim": 3072}})

    assert enc.dynamic_embedding_dim == 3072
    assert not hasattr(enc, "buffer"), "SageEncoder must not own a ReplayBuffer"
    assert not hasattr(enc, "storage"), "SageEncoder must not own LazyMemmapStorage"


def test_sage_encoder_default_dim() -> None:
    """Empty config → 3072 default embedding_dim."""
    from titan_plugin.core.sage.recorder import SageEncoder

    enc = SageEncoder()
    assert enc.dynamic_embedding_dim == 3072


def test_bus_constant_registered() -> None:
    """SAGE_RECORD_TRANSITION constant exists and is the canonical name."""
    from titan_plugin.bus import SAGE_RECORD_TRANSITION

    assert SAGE_RECORD_TRANSITION == "SAGE_RECORD_TRANSITION"


def test_make_msg_payload_shape() -> None:
    """Payload envelope matches SageRecorder.record_transition kwargs."""
    from titan_plugin.bus import SAGE_RECORD_TRANSITION, make_msg

    payload = {
        "observation_vector": [0.1, 0.2, 0.3],
        "action": "test_action",
        "reward": 0.5,
        "trauma_metadata": None,
        "research_metadata": None,
        "session_id": "unit_test",
    }
    msg = make_msg(SAGE_RECORD_TRANSITION, "titan_plugin", "rl", payload)

    assert msg["type"] == "SAGE_RECORD_TRANSITION"
    assert msg["src"] == "titan_plugin"
    assert msg["dst"] == "rl"
    assert msg["payload"] == payload


def test_rl_worker_handler_calls_recorder_with_kwargs() -> None:
    """_handle_sage_record_transition unpacks payload into recorder.record_transition."""
    from titan_plugin.modules.rl_worker import _handle_sage_record_transition

    captured: dict = {}

    async def _fake_record(**kwargs) -> None:
        captured.update(kwargs)

    fake_recorder = MagicMock()
    fake_recorder.record_transition = _fake_record

    msg = {
        "type": "SAGE_RECORD_TRANSITION",
        "src": "titan_plugin",
        "dst": "rl",
        "payload": {
            "observation_vector": [0.0, 1.0, 2.0],
            "action": "respond",
            "reward": 1.5,
            "trauma_metadata": None,
            "research_metadata": {"research_used": False, "transition_id": -1},
            "session_id": "openclaw_session",
        },
    }
    send_queue = MagicMock()
    _handle_sage_record_transition(msg, fake_recorder, send_queue, "rl")

    assert captured["observation_vector"] == [0.0, 1.0, 2.0]
    assert captured["action"] == "respond"
    assert captured["reward"] == 1.5
    assert captured["session_id"] == "openclaw_session"


def test_rl_worker_handler_swallows_exceptions() -> None:
    """Handler logs and swallows exceptions — never crashes the rl_worker loop."""
    from titan_plugin.modules.rl_worker import _handle_sage_record_transition

    async def _broken_record(**kwargs) -> None:
        raise RuntimeError("simulated failure")

    fake_recorder = MagicMock()
    fake_recorder.record_transition = _broken_record

    msg = {"type": "SAGE_RECORD_TRANSITION", "payload": {}}
    # Must not raise.
    _handle_sage_record_transition(msg, fake_recorder, MagicMock(), "rl")


def test_sage_guardian_uses_callable_when_injected() -> None:
    """SageGuardian with `record_transition_callable` routes through it."""
    from titan_plugin.logic.sage.guardian import SageGuardian

    captured: dict = {}

    def _fake_publish(**kwargs) -> int:
        captured.update(kwargs)
        return 42

    fake_recorder = MagicMock()
    # If the callable is used, the recorder MUST NOT be touched for record_transition.
    fake_recorder.record_transition = MagicMock(
        side_effect=AssertionError("recorder.record_transition should not be called when callable is injected")
    )

    g = SageGuardian(
        fake_recorder,
        config={},
        record_transition_callable=_fake_publish,
    )

    asyncio.run(g._trigger_trauma(action_intent="rm -rf /", veto_logic="DIRECTIVE_VIOLATION"))

    assert captured["action"] == "rm -rf /"
    assert captured["reward"] == -5.0
    assert captured["trauma_metadata"]["is_violation"] is True


def test_sage_guardian_falls_back_to_recorder_when_no_callable() -> None:
    """SageGuardian without callable uses recorder.record_transition directly."""
    from titan_plugin.logic.sage.guardian import SageGuardian

    captured: dict = {}

    async def _fake_record(**kwargs) -> None:
        captured.update(kwargs)

    fake_recorder = MagicMock()
    fake_recorder.record_transition = _fake_record

    g = SageGuardian(fake_recorder, config={})  # no callable

    asyncio.run(g._trigger_trauma(action_intent="bad", veto_logic="LOGIC"))

    assert captured["action"] == "bad"
    assert captured["reward"] == -5.0


def test_publish_helper_uses_bus_when_attached() -> None:
    """TitanPlugin._publish_sage_record_transition publishes via bus when attached."""
    # We test the helper at the function level without booting the full TitanPlugin
    # (which requires a Solana wallet etc.). Build a minimal stub that has
    # the helper bound to it.
    from titan_plugin import TitanPlugin
    from titan_plugin.bus import SAGE_RECORD_TRANSITION

    # Construct a bare instance (skip __init__ — test the method in isolation).
    plugin = TitanPlugin.__new__(TitanPlugin)
    plugin._sage_transition_counter = 0
    plugin.bus = MagicMock()
    plugin.bus.publish = MagicMock()
    # recorder won't be touched on the bus path
    plugin.recorder = MagicMock()

    counter = plugin._publish_sage_record_transition(
        observation_vector=[0.0] * 5,
        action="test",
        reward=2.0,
        trauma_metadata=None,
        research_metadata=None,
        session_id="t",
    )

    assert counter == 1
    plugin.bus.publish.assert_called_once()
    msg = plugin.bus.publish.call_args.args[0]
    assert msg["type"] == SAGE_RECORD_TRANSITION
    assert msg["dst"] == "rl"
    assert msg["payload"]["action"] == "test"
    # Critical: recorder.record_transition NOT called when bus attached.
    plugin.recorder.record_transition.assert_not_called()


def test_publish_helper_falls_back_when_bus_none() -> None:
    """Without a bus, the helper falls back to recorder.record_transition."""
    from titan_plugin import TitanPlugin

    plugin = TitanPlugin.__new__(TitanPlugin)
    plugin._sage_transition_counter = 0
    plugin.bus = None

    captured: dict = {}

    async def _fake_record(**kwargs) -> None:
        captured.update(kwargs)

    plugin.recorder = MagicMock()
    plugin.recorder.record_transition = _fake_record

    # Run inside an event loop so asyncio.create_task works.
    async def _run() -> None:
        plugin._publish_sage_record_transition(
            observation_vector=[1.0],
            action="fallback",
            reward=0.1,
        )
        # Yield to let the created task run.
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert captured["action"] == "fallback"
    assert captured["reward"] == 0.1
