"""Tests for the soul_diary_worker pipeline logic (RFP §1.0 P1).

Exercises the self-contained compose (own LLM + OVG), the gather, and the
full _author_daily_entry flow (authored persist · OVG soft-fail to minimal ·
daily-latch skip) with fakes — no bus, no real LLM, no real chronicle write.
"""
import asyncio
from unittest.mock import MagicMock

from titan_hcl.core import soul_diary_chain
from titan_hcl.core.soul_diary import SoulDiaryOrchestrator
from titan_hcl.modules import soul_diary_worker as sdw


class _FakeProvider:
    def __init__(self, text):
        self._text = text

    async def complete(self, prompt, *, system="", **kw):
        return self._text


def _verifier(passed):
    v = MagicMock()
    v.verify_safety.return_value = MagicMock(passed=passed)
    return v


_PROMPTS = {"system_prompt": "s", "user_prompt": "u"}


def test_compose_pass():
    text, ok = asyncio.run(sdw._compose_diary(_FakeProvider("authored entry"),
                                              _verifier(True), _PROMPTS))
    assert text == "authored entry" and ok is True


def test_compose_ovg_block():
    text, ok = asyncio.run(sdw._compose_diary(_FakeProvider("bad"),
                                              _verifier(False), _PROMPTS))
    assert text == "bad" and ok is False


def test_compose_empty_text():
    text, ok = asyncio.run(sdw._compose_diary(_FakeProvider("   "),
                                              _verifier(True), _PROMPTS))
    assert text == "" and ok is False


def test_compose_no_provider():
    text, ok = asyncio.run(sdw._compose_diary(None, _verifier(True), _PROMPTS))
    assert text == "" and ok is False


def test_gather_bundle_maps_neuromod():
    orch = SoulDiaryOrchestrator()
    shm = MagicMock()
    shm.read_neuromod.return_value = {"dopamine": 0.6, "cortisol": 0.2,
                                      "valence": 0.3, "arousal": 0.4}
    bundle = sdw._gather_bundle({"promoted": 7, "pruned": 2, "epoch": 5}, shm, orch)
    assert bundle["outcome"]["promoted"] == 7
    assert bundle["felt"]["dominant"] == "dopamine"  # highest level


def _orch(tmp_path):
    return SoulDiaryOrchestrator(state_path=str(tmp_path / "state.json"),
                                 ledger_path=str(tmp_path / "chain.json"))


def test_author_persists_authored_text(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    seen = {}
    monkeypatch.setattr(orch, "persist", lambda text, **kw: seen.update(text=text))
    monkeypatch.setattr(sdw, "_gather_bundle", lambda p, s, o: o.build_bundle(
        sovereignty={"s": 0.58, "replies": 22}, outcome={"promoted": 7, "pruned": 2},
        felt={}, engrams_today=["Glacier"], memory={}, social={}, onchain={}))
    ok = sdw._author_daily_entry({"promoted": 7}, orchestrator=orch,
                                 provider=_FakeProvider("Today I reflected."),
                                 verifier=_verifier(True), shm_reader=None)
    assert ok is True
    assert seen["text"] == "Today I reflected."          # authored text persisted
    assert len(soul_diary_chain.load_chain(path=str(tmp_path / "chain.json"))) == 1
    assert orch.should_author(sdw._utc_today()) is False  # latched


def test_author_softfails_to_minimal_on_ovg_block(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    seen = {}
    monkeypatch.setattr(orch, "persist", lambda text, **kw: seen.update(text=text))
    monkeypatch.setattr(sdw, "_gather_bundle", lambda p, s, o: o.build_bundle(
        sovereignty={"s": 0.5, "replies": 1}, outcome={"promoted": 3, "pruned": 0},
        felt={}, engrams_today=[], memory={}, social={}, onchain={}))
    ok = sdw._author_daily_entry({}, orchestrator=orch,
                                 provider=_FakeProvider("hallucinated text"),
                                 verifier=_verifier(False), shm_reader=None)
    assert ok is True
    assert "hallucinated" not in seen["text"]   # OVG-blocked text NOT persisted
    assert "0.50" in seen["text"]               # minimal grounded entry instead


def test_author_latch_skips_second_same_day(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    orch.mark_authored(sdw._utc_today())  # already wrote today
    called = {"persist": False}
    monkeypatch.setattr(orch, "persist", lambda *a, **k: called.update(persist=True))
    ok = sdw._author_daily_entry({}, orchestrator=orch,
                                 provider=_FakeProvider("x"),
                                 verifier=_verifier(True), shm_reader=None)
    assert ok is True and called["persist"] is False  # latch closed → no second entry


def test_author_noop_day_latches_without_entry(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    called = {"persist": False}
    monkeypatch.setattr(orch, "persist", lambda *a, **k: called.update(persist=True))
    monkeypatch.setattr(sdw, "_gather_bundle", lambda p, s, o: o.build_bundle(
        sovereignty={"replies": 0}, outcome={"promoted": 0, "pruned": 0},
        felt={}, engrams_today=[], memory={}, social={}, onchain={}))
    ok = sdw._author_daily_entry({}, orchestrator=orch, provider=_FakeProvider("x"),
                                 verifier=_verifier(True), shm_reader=None)
    assert ok is True and called["persist"] is False      # no-op day → no entry
    assert orch.should_author(sdw._utc_today()) is False    # but latched
