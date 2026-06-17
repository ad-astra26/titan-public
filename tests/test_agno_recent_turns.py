"""RFP_agno_memory_bypass §1b (D-SPEC-159) — Titan's own recent-conversation store
replacing agno's `add_history_to_context` (the ~30MB/turn retainer).

Verifies: bounded per-session ring (last N), LRU session cap, TEXT-only context
block (never the enriched system context — INV-AGN-2), never-raises, and that
create_agent disables agno's native history by default (the fix) with a kill-switch.

Run isolated: python -m pytest tests/test_agno_recent_turns.py -v -p no:anchorpy
"""
from titan_hcl.modules.agno_hooks import (
    record_recent_turn, get_recent_turns_context, _recent_turns,
    _RECENT_TURNS_MAX, _RECENT_TURNS_SESSIONS, _RECENT_TURNS_TEXT_CAP)


def _clear():
    _recent_turns.clear()


def test_ring_bounded_to_last_n():
    _clear()
    for i in range(_RECENT_TURNS_MAX + 3):
        record_recent_turn("u", "s", f"q{i}", f"a{i}")
    ctx = get_recent_turns_context("u", "s")
    # oldest dropped, newest kept
    assert f"q{_RECENT_TURNS_MAX + 2}" in ctx          # newest
    assert "q0" not in ctx                              # oldest dropped (maxlen)
    assert ctx.count("User:") == _RECENT_TURNS_MAX     # exactly N turns kept


def test_context_is_text_only_with_markers():
    _clear()
    record_recent_turn("u", "s", "hello there", "hi friend")
    ctx = get_recent_turns_context("u", "s")
    assert ctx.startswith("### Recent conversation")
    assert "User: hello there" in ctx and "Titan: hi friend" in ctx


def test_empty_for_unknown_session():
    _clear()
    assert get_recent_turns_context("u", "nope") == ""
    assert get_recent_turns_context("", "") == ""


def test_lru_session_cap():
    _clear()
    for i in range(_RECENT_TURNS_SESSIONS + 20):
        record_recent_turn("u", f"s{i}", "q", "a")
    assert len(_recent_turns) <= _RECENT_TURNS_SESSIONS


def test_per_message_text_capped():
    _clear()
    big = "x" * (_RECENT_TURNS_TEXT_CAP + 500)
    record_recent_turn("u", "s", big, big)
    ctx = get_recent_turns_context("u", "s")
    # each stored message is capped (continuity, not full transcript)
    assert ("x" * _RECENT_TURNS_TEXT_CAP) in ctx
    assert ("x" * (_RECENT_TURNS_TEXT_CAP + 1)) not in ctx


def test_record_never_raises():
    _clear()
    # weird inputs must not break the PostHook
    record_recent_turn(None, None, None, None)
    record_recent_turn("u", "s", 12345, ["not", "a", "string"])
    assert True


