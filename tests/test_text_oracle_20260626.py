"""Phase A — the deterministic text-extraction oracle (RFP_text_extraction_introspection §7.A).

Covers G-ORACLE: determinism, each query kind, the safety caps, and typed errors.
"""
import os

os.environ.setdefault("TITAN_CONFIG_SHM_READ", "0")

import pytest  # noqa: E402

from titan_hcl.synthesis.text_oracle import (  # noqa: E402
    extract, ExtractError, ExtractResult, MAX_PATTERN_CHARS, MAX_MATCHES_CAP,
)

_LOG = """\
ts=100.0 action=direct src=llm_judge
ts=150.0 action=research src=curiosity
ts=200.0 action=direct src=llm_judge
ts=250.0 action=tool src=oracle
ts=300.0 action=research src=curiosity
"""


# ── determinism (the whole point — a re-checkable fact) ──────────────────────
def test_determinism_same_in_same_out():
    q = {"kind": "regex", "pattern": r"action=\w+"}
    a, b = extract(_LOG, q), extract(_LOG, q)
    assert a.to_dict() == b.to_dict()
    assert a.corpus_sha == b.corpus_sha and len(a.corpus_sha) == 16


# ── regex ────────────────────────────────────────────────────────────────────
def test_regex_matches():
    r = extract(_LOG, {"kind": "regex", "pattern": r"action=(\w+)"})
    assert r.matches == ["action=direct", "action=research", "action=direct",
                         "action=tool", "action=research"]
    assert r.n == 5


def test_regex_max_matches_cap_honored():
    r = extract(_LOG, {"kind": "regex", "pattern": r"action=\w+", "max_matches": 2})
    assert r.n == 2 and len(r.matches) == 2


# ── count (total + grouped) ──────────────────────────────────────────────────
def test_count_total():
    r = extract(_LOG, {"kind": "count", "pattern": r"action=\w+"})
    assert r.counts == {"_total": 5} and r.n == 5


def test_count_group_by_named_group():
    r = extract(_LOG, {"kind": "count", "pattern": r"action=(?P<act>\w+)",
                       "group_by": "act"})
    assert r.counts == {"direct": 2, "research": 2, "tool": 1}


def test_count_group_by_index():
    r = extract(_LOG, {"kind": "count", "pattern": r"src=(\w+)", "group_by": 1})
    assert r.counts == {"llm_judge": 2, "curiosity": 2, "oracle": 1}


# ── window (timestamp band) ──────────────────────────────────────────────────
def test_window_filters_by_timestamp():
    q = {"kind": "window", "pattern": r"ts=([\d.]+) action=\w+",
         "ts_group": 1, "since": 150.0, "until": 250.0}
    r = extract(_LOG, q)
    # the 150/200/250 rows fall inside [150, 250]
    assert r.n == 3
    assert all("ts=" in m for m in r.matches)


# ── fields (named captures → dict) ───────────────────────────────────────────
def test_fields_named_captures():
    r = extract(_LOG, {"kind": "fields",
                       "pattern": r"action=(?P<action>\w+) src=(?P<src>\w+)"})
    assert r.fields == {"action": "direct", "src": "llm_judge"}  # first match


# ── safety + typed errors ────────────────────────────────────────────────────
def test_empty_corpus_safe():
    r = extract("", {"kind": "regex", "pattern": r"x"})
    assert isinstance(r, ExtractResult) and r.n == 0 and r.matches == []


def test_truncation_flagged():
    from titan_hcl.synthesis.text_oracle import MAX_CORPUS_CHARS
    big = "a" * (MAX_CORPUS_CHARS + 10)
    r = extract(big, {"kind": "count", "pattern": "a"})
    assert r.truncated is True
    assert r.counts["_total"] <= MAX_CORPUS_CHARS


def test_nested_quantifier_rejected():
    with pytest.raises(ExtractError, match="catastrophic"):
        extract("aaaa", {"kind": "regex", "pattern": r"(a+)+"})


def test_pattern_too_long_rejected():
    with pytest.raises(ExtractError, match="too long"):
        extract("x", {"kind": "regex", "pattern": "a" * (MAX_PATTERN_CHARS + 1)})


def test_bad_kind_rejected():
    with pytest.raises(ExtractError, match="kind"):
        extract("x", {"kind": "frobnicate", "pattern": "a"})


def test_bad_regex_rejected():
    with pytest.raises(ExtractError, match="invalid regex"):
        extract("x", {"kind": "regex", "pattern": "("})


def test_non_string_corpus_rejected():
    with pytest.raises(ExtractError, match="corpus_text must be a string"):
        extract(12345, {"kind": "regex", "pattern": "a"})


def test_unknown_flag_rejected():
    with pytest.raises(ExtractError, match="unknown regex flag"):
        extract("x", {"kind": "regex", "pattern": "a", "flags": ["bogus"]})


def test_max_matches_cap_is_hard():
    # even if a query asks for more than the hard ceiling, it's clamped
    r = extract("a " * (MAX_MATCHES_CAP + 50), {"kind": "regex", "pattern": "a",
                                                "max_matches": MAX_MATCHES_CAP + 1000})
    assert r.n <= MAX_MATCHES_CAP


def test_ignorecase_flag():
    r = extract("Action ACTION action", {"kind": "count", "pattern": "action",
                                         "flags": ["i"]})
    assert r.counts == {"_total": 3}
