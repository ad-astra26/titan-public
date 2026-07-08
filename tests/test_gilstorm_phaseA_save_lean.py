"""RFP_chat_concurrency_serialization_gil_storm — Phase A (INV-SAVE-LEAN).

Verifies the per-turn agno session save no longer serializes the big
V5-enriched SYSTEM prompt (the GIL-storm payload — live-measured 92% of saved
message bytes), while keeping user/assistant text + all other fields
byte-identical to stock agno serialization (INV-NO-BEHAVIOUR-CHANGE).

The elision lives in the process-global `RunOutput.to_dict` monkeypatch
installed by the sole agno agent factory, so it covers EVERY subsystem that
persists an agno session (keyed on role=='system', content-agnostic).

Run isolated: python -m pytest tests/test_gilstorm_phaseA_save_lean.py -v -p no:anchorpy
"""
import json

import pytest

agno = pytest.importorskip("agno")
from agno.run.agent import RunOutput            # noqa: E402
from agno.session.agent import AgentSession     # noqa: E402
from agno.models.message import Message         # noqa: E402

from titan_hcl.modules.agno_agent_factory import (  # noqa: E402
    _install_agno_to_dict_fastpath,
    _ELIDED_SYSTEM_MARKER,
)

_BIG_SYS = (
    "You are Titan, a sovereign AI entity...\n"
    "### Internal State (how you're feeling right now)\n"
    + ("felt-state and VCB recall payload " * 400)   # ~20KB, like production
)


@pytest.fixture(autouse=True)
def _install():
    _install_agno_to_dict_fastpath()
    yield


def _make_run():
    return RunOutput(
        run_id="r1", agent_id="a",
        messages=[
            Message(role="system", content=_BIG_SYS),
            Message(role="user", content="hey Titan, how are you?"),
            Message(role="assistant", content="I feel resonant today."),
        ],
        metadata={"turn": 1},
    )


def test_system_message_elided_from_saved_run():
    d = _make_run().to_dict()
    msgs = d["messages"]
    assert msgs[0]["role"] == "system"
    # the fat enriched prompt is GONE, replaced by the lean marker
    assert msgs[0]["content"] == _ELIDED_SYSTEM_MARKER
    assert "Internal State" not in json.dumps(d)
    assert "VCB recall payload" not in json.dumps(d)
    # serialized payload is tiny now, not ~20KB
    assert len(json.dumps(d)) < 2000


def test_user_and_assistant_text_intact():
    d = _make_run().to_dict()
    msgs = d["messages"]
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "hey Titan, how are you?"
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == "I feel resonant today."


def test_nonsystem_run_content_untouched():
    """A run with NO system message keeps all content verbatim — the patch
    only touches system-role content (stock byte-identity for non-system
    fields is separately guaranteed by the install-time self-check)."""
    run = RunOutput(
        run_id="r2", agent_id="a",
        messages=[Message(role="user", content="x"),
                  Message(role="assistant", content="y")],
        metadata={"k": 1})
    # Build the same run and compare structure/keys — content untouched.
    d = run.to_dict()
    assert d["messages"][0]["content"] == "x"
    assert d["messages"][1]["content"] == "y"
    assert all(m["role"] != "system" for m in d["messages"])
    # no elision marker anywhere
    assert _ELIDED_SYSTEM_MARKER not in json.dumps(d)


def test_agentsession_nesting_also_elided():
    """AgentSession.to_dict → run.to_dict per run → elision must propagate to
    the whole persisted session (this is what upsert_session serializes)."""
    run = _make_run()
    session = AgentSession(
        session_id="s1", user_id="u",
        session_data={"session_state": {"k": 1}},
        metadata={"m": 2},
        runs=[run])
    sd = session.to_dict()
    # ensure_ascii=False so the marker's em-dash/§ match literally
    blob = json.dumps(sd, ensure_ascii=False)
    assert _ELIDED_SYSTEM_MARKER in blob
    assert sd["runs"][0]["messages"][0]["content"] == _ELIDED_SYSTEM_MARKER
    assert "Internal State" not in blob
    # the session_data / user text survive (INV-NO-BEHAVIOUR-CHANGE)
    assert sd["runs"][0]["messages"][1]["content"] == "hey Titan, how are you?"


def test_elision_is_save_only_live_messages_untouched():
    """to_dict must NOT mutate the live Message objects (they drive the next
    turn's cognition — INV-NO-BEHAVIOUR-CHANGE)."""
    run = _make_run()
    _ = run.to_dict()
    # the live system message still carries the full enriched prompt
    assert run.messages[0].content == _BIG_SYS
    assert "Internal State" in run.messages[0].content
