"""§7.E (E.3) — research→tool crystallization: a matched crystallized research::{gc}
recipe's source is fetched DIRECTLY (bypassing the full multi-URL research lane), with
a switchable source-allowlist guard (default OFF). Covers hit / non-research / no-source
/ allowlist on+off / disabled / empty-content fall-through.

Run: python -m pytest tests/test_e3_research_tool_call.py -v -p no:anchorpy
"""
import asyncio

from titan_hcl.modules.agno_hooks import _e3_research_tool_fetch


class _Researcher:
    def __init__(self, content="PoH + PoS consensus, $71 last price"):
        self.content = content
        self.fetched = []

    async def _scrape_single(self, url):
        self.fetched.append(url)
        return self.content


class _Plugin:
    def __init__(self, match, researcher=None, e_research_tool=None):
        self._last_composite_match = match
        self.sage_researcher = researcher
        self._full_config = {"synthesis": {"tool_backstop": {
            "e_research_tool": e_research_tool if e_research_tool is not None
            else {"enabled": True}}}}


def _match(action="research", source="https://api.x.com/sol/LastPrice"):
    return {"score": 0.95, "action": action, "goal_class": "sol_price",
            "recipe_json": "", "source": source, "reasoning_id": "research::sol_price"}


def _run(plugin):
    return asyncio.run(_e3_research_tool_fetch(plugin, "what is the current sol price?"))


def test_direct_fetch_hit():
    rsr = _Researcher()
    p = _Plugin(_match(), researcher=rsr)
    out = _run(p)
    assert out is not None
    assert "api.x.com/sol/LastPrice" in out
    assert "PoH + PoS" in out
    assert rsr.fetched == ["https://api.x.com/sol/LastPrice"]   # direct single fetch


def test_non_research_action_falls_through():
    p = _Plugin(_match(action="tool"), researcher=_Researcher())
    assert _run(p) is None


def test_no_source_falls_through():
    p = _Plugin(_match(source=""), researcher=_Researcher())
    assert _run(p) is None


def test_non_url_source_falls_through():
    p = _Plugin(_match(source="not a url"), researcher=_Researcher())
    assert _run(p) is None


def test_no_match_falls_through():
    p = _Plugin(None, researcher=_Researcher())
    assert _run(p) is None


def test_disabled_flag():
    p = _Plugin(_match(), researcher=_Researcher(), e_research_tool={"enabled": False})
    assert _run(p) is None


def test_empty_content_falls_through():
    p = _Plugin(_match(), researcher=_Researcher(content=""))
    assert _run(p) is None


# ── source-allowlist guard (switchable, default OFF) ────────────────────────
def test_allowlist_off_serves_any_source():
    # default OFF → any source is fetched (the latency-measurement posture)
    rsr = _Researcher()
    p = _Plugin(_match(source="https://random.example/x"), researcher=rsr)
    assert _run(p) is not None and rsr.fetched == ["https://random.example/x"]


def test_allowlist_on_blocks_non_allowlisted():
    p = _Plugin(_match(source="https://evil.example/x"), researcher=_Researcher(),
                e_research_tool={"enabled": True, "source_allowlist_enabled": True,
                                 "source_allowlist": ["api.x.com"]})
    assert _run(p) is None  # host not allowlisted → full research


def test_allowlist_on_allows_listed_host_and_subdomain():
    rsr = _Researcher()
    p = _Plugin(_match(source="https://data.api.x.com/sol"), researcher=rsr,
                e_research_tool={"enabled": True, "source_allowlist_enabled": True,
                                 "source_allowlist": ["api.x.com"]})
    out = _run(p)
    assert out is not None and rsr.fetched == ["https://data.api.x.com/sol"]  # subdomain ok
