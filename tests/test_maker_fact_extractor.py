"""Tests for the Maker-fact extractor (RFP_missions_and_the_maker_model §7.1) — the
pre-filter gate + JSON parse + LLM-injected extraction. No network."""
from titan_hcl.synthesis.maker_fact_extractor import (
    looks_like_self_disclosure, build_extraction_prompt, extract_maker_facts,
    maker_fact_loop, _parse_facts,
)


# ── pre-filter gate ──────────────────────────────────────────────────

def test_pre_filter_fires_on_self_disclosure():
    assert looks_like_self_disclosure("I'm a software architect")
    assert looks_like_self_disclosure("I love climbing on weekends")
    assert looks_like_self_disclosure("my job is pretty demanding lately")
    assert looks_like_self_disclosure("I grew up in Prague")


def test_pre_filter_skips_non_disclosure():
    assert not looks_like_self_disclosure("what's the weather today?")
    assert not looks_like_self_disclosure("restart the synthesis worker please")
    assert not looks_like_self_disclosure("that function looks buggy")
    assert not looks_like_self_disclosure("")


# ── JSON parse robustness ────────────────────────────────────────────

def test_parse_clean_array():
    facts = _parse_facts('[{"category":"occupation","value":"architect","confidence":0.9}]')
    assert facts == [{"category": "occupation", "value": "architect", "confidence": 0.9}]


def test_parse_code_fenced_with_prose():
    raw = 'Sure! Here you go:\n```json\n[{"category":"hobby","value":"climbing"}]\n```'
    facts = _parse_facts(raw)
    assert len(facts) == 1
    assert facts[0]["category"] == "hobby"
    assert facts[0]["confidence"] == 0.7          # default when omitted


def test_parse_empty_and_malformed():
    assert _parse_facts("[]") == []
    assert _parse_facts("no json here") == []
    assert _parse_facts('[{"category":"x"}]') == []   # missing value → dropped
    assert _parse_facts("") == []


def test_confidence_clamped():
    facts = _parse_facts('[{"category":"c","value":"v","confidence":5}]')
    assert facts[0]["confidence"] == 1.0


# ── extraction (LLM injected) ────────────────────────────────────────

def test_build_prompt_contains_text():
    assert "I'm an architect" in build_extraction_prompt("I'm an architect")


def test_extract_with_fake_llm():
    def fake_llm(prompt):
        assert "software architect" in prompt
        return '[{"category":"occupation","value":"software architect","confidence":0.9}]'
    facts = extract_maker_facts("I'm a software architect", fake_llm)
    assert facts == [{"category": "occupation", "value": "software architect",
                      "confidence": 0.9}]


def test_extract_soft_on_llm_error():
    def boom(prompt):
        raise RuntimeError("provider down")
    assert extract_maker_facts("I'm a chef", boom) == []


# ── loop integration (fake store + queue + stop after one pass) ──────

class _FakeStore:
    def __init__(self):
        self.calls = []

    def record_fact(self, *, category, value, provenance, confidence, source_turn=""):
        self.calls.append((category, value, provenance, round(confidence, 3)))
        return f"maker:{category}:v1"


class _OneShotStop:
    """wait() returns False once (run a pass), then True (exit)."""
    def __init__(self):
        self._n = 0

    def wait(self, _t):
        self._n += 1
        return self._n > 1


def test_loop_filters_extracts_and_records():
    import collections
    q = collections.deque()
    q.append({"prompt": "I'm a software architect and I love climbing"})
    q.append({"prompt": "what's the weather?"})          # filtered out (no LLM)
    store = _FakeStore()

    def fake_llm(prompt):
        return '[{"category":"occupation","value":"software architect","confidence":0.9}]'

    maker_fact_loop(q, store, fake_llm, _OneShotStop(), interval_s=0.0, per_pass_cap=8)
    assert len(store.calls) == 1
    cat, val, prov, conf = store.calls[0]
    assert cat == "occupation" and val == "software architect"
    assert prov == "maker-told"                          # learned-from-chat provenance
