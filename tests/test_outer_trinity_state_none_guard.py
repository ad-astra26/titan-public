"""Regression tests for _merge_outer_trinity_payload None-guard.

Bug surfaced 2026-04-23 during Phase 1 sensory wiring smoke test:
spirit_worker's OUTER_TRINITY_STATE handler unconditionally assigned
`outer_state["outer_spirit_45d"] = payload.get("outer_spirit_45d")` which
wrote None whenever OuterTrinityCollector._collect_extended failed silently.
That flipped `has_outer_extended` False in spirit_loop.py and collapsed the
130D state_vector to 67D until the next successful extended tick.

These tests verify the fix: missing / None extended dims in an incoming
payload preserve the prior rich extended state instead of overwriting it.
"""
from titan_plugin.modules.spirit_worker import _merge_outer_trinity_payload


def _fresh_outer_state() -> dict:
    return {
        "outer_body": [0.5] * 5,
        "outer_mind": [0.5] * 5,
        "outer_spirit": [0.5] * 5,
    }


def test_5d_tensors_always_applied():
    st = _fresh_outer_state()
    payload = {
        "outer_body": [0.1, 0.2, 0.3, 0.4, 0.5],
        "outer_mind": [0.9] * 5,
        "outer_spirit": [0.7] * 5,
    }
    _merge_outer_trinity_payload(st, payload)
    assert st["outer_body"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert st["outer_mind"] == [0.9] * 5
    assert st["outer_spirit"] == [0.7] * 5


def test_5d_tensors_default_to_neutral_when_missing():
    st = _fresh_outer_state()
    _merge_outer_trinity_payload(st, {})
    assert st["outer_body"] == [0.5] * 5
    assert st["outer_mind"] == [0.5] * 5
    assert st["outer_spirit"] == [0.5] * 5


def test_extended_dims_set_on_first_rich_tick():
    """First OUTER_TRINITY_STATE with extended dims → fresh state populated."""
    st = _fresh_outer_state()
    payload = {
        "outer_body": [0.5] * 5,
        "outer_mind": [0.5] * 5,
        "outer_spirit": [0.5] * 5,
        "outer_mind_15d": list(range(15)),  # mock 15D vector
        "outer_spirit_45d": list(range(45)),
    }
    _merge_outer_trinity_payload(st, payload)
    assert st["outer_mind_15d"] == list(range(15))
    assert st["outer_spirit_45d"] == list(range(45))


def test_missing_extended_dims_preserve_prior_rich_state():
    """THE BUG FIX: payload without extended dims must NOT clear prior state."""
    st = _fresh_outer_state()
    st["outer_mind_15d"] = [0.8] * 15
    st["outer_spirit_45d"] = [0.6] * 45

    # Simulate a tick where _collect_extended failed (payload lacks extended)
    payload = {
        "outer_body": [0.5] * 5,
        "outer_mind": [0.5] * 5,
        "outer_spirit": [0.5] * 5,
        # no outer_mind_15d, no outer_spirit_45d
    }
    _merge_outer_trinity_payload(st, payload)

    # Prior rich extended state must survive intact
    assert st["outer_mind_15d"] == [0.8] * 15, \
        "outer_mind_15d was cleared by a payload missing the key — regression"
    assert st["outer_spirit_45d"] == [0.6] * 45, \
        "outer_spirit_45d was cleared by a payload missing the key — regression"


def test_none_extended_dims_preserve_prior_rich_state():
    """Same fix, but payload contains the keys with None values (e.g. from
    payload.get('outer_mind_15d') when _collect_extended partial-failed)."""
    st = _fresh_outer_state()
    st["outer_mind_15d"] = [0.8] * 15
    st["outer_spirit_45d"] = [0.6] * 45

    payload = {
        "outer_body": [0.5] * 5,
        "outer_mind": [0.5] * 5,
        "outer_spirit": [0.5] * 5,
        "outer_mind_15d": None,
        "outer_spirit_45d": None,
    }
    _merge_outer_trinity_payload(st, payload)

    assert st["outer_mind_15d"] == [0.8] * 15
    assert st["outer_spirit_45d"] == [0.6] * 45


def test_outer_mind_15d_merge_preserves_willing_octave():
    """Incoming payload contributes Thinking[0:5] + Feeling[5:10];
    existing Willing[10:15] must be preserved (GROUND_UP enrichment owns it)."""
    st = _fresh_outer_state()
    # Prior state with Willing carrying specific values from GROUND_UP
    existing = [0.1] * 5 + [0.2] * 5 + [0.9, 0.8, 0.7, 0.6, 0.5]
    st["outer_mind_15d"] = list(existing)

    # Incoming rich tick: new Thinking + Feeling, but we don't want its Willing to win
    incoming = [0.3] * 5 + [0.4] * 5 + [0.0] * 5
    payload = {
        "outer_body": [0.5] * 5,
        "outer_mind": [0.5] * 5,
        "outer_spirit": [0.5] * 5,
        "outer_mind_15d": incoming,
    }
    _merge_outer_trinity_payload(st, payload)

    result = st["outer_mind_15d"]
    # Thinking [0:5] taken from incoming
    assert result[0:5] == [0.3] * 5
    # Feeling [5:10] taken from incoming
    assert result[5:10] == [0.4] * 5
    # Willing [10:15] preserved from prior (NOT zeroed by incoming)
    assert result[10:15] == [0.9, 0.8, 0.7, 0.6, 0.5], \
        "Willing octave was clobbered by incoming payload — regression"


def test_first_15d_payload_when_existing_is_5d_fallback():
    """If prior outer_mind_15d is missing, the first full 15D payload
    populates it wholesale (no merge since there's no existing 15D to merge with)."""
    st = _fresh_outer_state()
    # Note: no outer_mind_15d key in the fresh state

    payload = {
        "outer_body": [0.5] * 5,
        "outer_mind": [0.5] * 5,
        "outer_spirit": [0.5] * 5,
        "outer_mind_15d": list(range(15)),
    }
    _merge_outer_trinity_payload(st, payload)

    assert st["outer_mind_15d"] == list(range(15))


def test_flip_flop_regression_scenario():
    """Reproduce the exact scenario that caused the 132D↔67D flip-flop
    observed during 2026-04-23 Phase 1 smoke test.

    Sequence:
      tick 1: rich extended dims → state goes 132D-capable
      tick 2: _collect_extended failed → payload lacks extended dims
      tick 3: rich extended again

    Invariant: between ticks 1 and 3, extended state must stay rich
    (preserving 132D state_vector assembly)."""
    st = _fresh_outer_state()

    # Tick 1: rich extended
    _merge_outer_trinity_payload(st, {
        "outer_body": [0.23, 0.85, 0.73, 0.05, 0.52],
        "outer_mind": [0.5] * 5,
        "outer_spirit": [0.5] * 5,
        "outer_mind_15d": [0.4] * 15,
        "outer_spirit_45d": [0.3] * 45,
    })
    assert st["outer_mind_15d"] == [0.4] * 15
    assert st["outer_spirit_45d"] == [0.3] * 45

    # Tick 2: extended computation failed silently (no extended keys)
    _merge_outer_trinity_payload(st, {
        "outer_body": [0.30, 0.80, 0.70, 0.10, 0.50],
        "outer_mind": [0.5] * 5,
        "outer_spirit": [0.5] * 5,
    })
    # Extended state must survive — the whole point of the fix
    assert st["outer_mind_15d"] == [0.4] * 15, "tick 2 cleared outer_mind_15d (flip-flop bug)"
    assert st["outer_spirit_45d"] == [0.3] * 45, "tick 2 cleared outer_spirit_45d (flip-flop bug)"
    # 5D dims updated from tick 2 payload
    assert st["outer_body"] == [0.30, 0.80, 0.70, 0.10, 0.50]

    # Tick 3: rich extended again — merge replaces Thinking[0:5] + Feeling[5:10]
    # from incoming, but preserves Willing[10:15] from prior (tick 1 = 0.4)
    _merge_outer_trinity_payload(st, {
        "outer_body": [0.25, 0.82, 0.71, 0.08, 0.55],
        "outer_mind": [0.5] * 5,
        "outer_spirit": [0.5] * 5,
        "outer_mind_15d": [0.6] * 15,
        "outer_spirit_45d": [0.7] * 45,
    })
    # outer_spirit_45d is full-overwrite (no octave-preservation for spirit)
    assert st["outer_spirit_45d"] == [0.7] * 45
    # outer_mind_15d: Thinking+Feeling from incoming, Willing preserved from tick 1
    assert st["outer_mind_15d"][0:10] == [0.6] * 10
    assert st["outer_mind_15d"][10:15] == [0.4] * 5, \
        "Willing octave should be preserved from prior state during merge"
