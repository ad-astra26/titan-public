"""§7.E (GE7) — flag-off parity. When OML is off (or no composite matched / no prompt
vec stashed / the E flags are disabled), every E tier is INERT — the tool-backstop
falls through to its normal router path byte-identically. This is the safety contract:
Phase E never changes behaviour unless a real verified library entry matches.

Run: python -m pytest tests/test_e_flag_off_parity.py -v -p no:anchorpy
"""
import asyncio

from titan_hcl.modules.agno_hooks import _e3_research_tool_fetch
from titan_hcl.synthesis.tool_backstop import _e1_recipe_replay, _e2_literal_lookup


class _BarePlugin:
    """A plugin with NO OML state (the flag-off / cold-start reality): no composite
    match, no stashed prompt vec — exactly what the backstop sees when self-learning
    is off (the decide block that stashes these never ran)."""
    def __init__(self, e_cfg=None):
        self._full_config = {"synthesis": {"tool_backstop": e_cfg or {}}}
    # note: NO _last_composite_match, NO _last_prompt_vec attributes


def _cfg(p):
    return p._full_config["synthesis"]["tool_backstop"]


def test_e1_inert_without_composite_match():
    p = _BarePlugin({"e_compose": {"enabled": True, "floor": 0.85}})
    assert _e1_recipe_replay(p, "what is 17 times 23", _cfg(p)) == ""  # → LLM router (normal)


def test_e2_inert_without_prompt_vec():
    p = _BarePlugin({"e_prompt_cache": {"enabled": True}})
    assert _e2_literal_lookup(p, "what is 17 times 23", _cfg(p)) is None  # → E.1/normal


def test_e3_inert_without_composite_match():
    p = _BarePlugin({"e_research_tool": {"enabled": True}})
    p.sage_researcher = None
    p._last_composite_match = None
    assert asyncio.run(_e3_research_tool_fetch(p, "current sol price")) is None  # → full research


def test_all_tiers_inert_with_flags_disabled():
    # even WITH a (hypothetical) match, the per-tier disable flags make each inert
    p = _BarePlugin({"e_compose": {"enabled": False},
                     "e_prompt_cache": {"enabled": False},
                     "e_research_tool": {"enabled": False}})
    p._last_composite_match = {"score": 0.99, "action": "tool",
                               "recipe_json": '{"tool_id":"coding_sandbox",'
                               '"code_template":"math.factorial({p0})",'
                               '"param_kinds":["number"],"captured_params":["8"]}',
                               "source": "https://api.x/y", "goal_class": "g"}
    p._last_prompt_vec = [0.0] * 384
    p.sage_researcher = None
    assert _e1_recipe_replay(p, "order 10 routes", _cfg(p)) == ""
    assert _e2_literal_lookup(p, "order 10 routes", _cfg(p)) is None
    assert asyncio.run(_e3_research_tool_fetch(p, "x")) is None
