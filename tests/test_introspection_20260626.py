"""Phase B core — introspection damper + orchestration (RFP_text_extraction_introspection §7.B).

Covers the navel-gaze damper (INV-TX-6) + run_introspection over the text oracle.
The live agency wiring (should_fire trigger + the curiosity grounding ride) is verified
separately; here we prove the faculty's logic + that it can't loop/collapse.
"""
import os

os.environ.setdefault("TITAN_CONFIG_SHM_READ", "0")

from titan_hcl.synthesis.introspection import (  # noqa: E402
    IntrospectionDamper, run_introspection,
)
from titan_hcl.modules.self_learning_worker import _REWARD_SOURCE_RANK  # noqa: E402

# a tiny self-telemetry corpus (routing rows)
_CORPUS = ("ts=100 action=direct\nts=110 action=research\n"
           "ts=120 action=direct\nts=130 action=tool\n")
_Q = {"kind": "count", "pattern": r"action=(?P<a>\w+)", "group_by": "a"}


def _provider_const(_aspect):
    return _CORPUS


# ── reward source registered (INV-MC-8) ──────────────────────────────────────
def test_introspection_reward_source_registered_rank3():
    assert _REWARD_SOURCE_RANK.get("introspection") == 3
    assert _REWARD_SOURCE_RANK["introspection"] > _REWARD_SOURCE_RANK["llm_judge"]


# ── damper: self-referential refusal (INV-TX-6c) ─────────────────────────────
def test_damper_refuses_self_referential_aspects():
    d = IntrospectionDamper()
    for a in ("introspection_habits", "my_introspection", "navel_gazing"):
        allowed, why = d.allow(a)
        assert not allowed and "self-referential" in why


# ── damper: per-window budget (INV-TX-6a) ────────────────────────────────────
def test_damper_budget_bounds_per_window():
    d = IntrospectionDamper(max_per_window=2, window_s=1000.0)
    t = 1000.0
    assert d.allow("routing", now=t)[0]
    d.commit("routing", "sha1", now=t)
    assert d.allow("mood", now=t)[0]
    d.commit("mood", "sha2", now=t)
    # third in the same window → refused
    ok, why = d.allow("skills", now=t)
    assert not ok and "budget" in why
    # window rolls → allowed again
    assert d.allow("skills", now=t + 1001.0)[0]


# ── damper: novelty gate (INV-TX-6b) ─────────────────────────────────────────
def test_damper_novelty_gate():
    d = IntrospectionDamper()
    assert d.is_novel("routing", "shaA") is True
    d.commit("routing", "shaA")
    assert d.is_novel("routing", "shaA") is False   # unchanged → not novel
    assert d.is_novel("routing", "shaB") is True     # changed → novel again


# ── orchestration: a substantive novel read grounds a SELF concept ───────────
def test_run_introspection_grounds_novel_self_concept():
    d = IntrospectionDamper()
    r = run_introspection("routing", _Q, _provider_const, d, now=1000.0)
    assert r.grounded is True
    assert r.research_target["concept_id"] == "SELF:routing"
    assert r.research_target["source"] == "introspection"
    assert r.research_target["domain_hint"] == "self"
    assert "direct=2" in r.observation and "SELF:routing" in r.observation
    assert r.extract["counts"] == {"direct": 2, "research": 1, "tool": 1}


# ── orchestration: unchanged read does NOT re-ground (converges, not loops) ───
def test_run_introspection_skips_unchanged():
    d = IntrospectionDamper()
    r1 = run_introspection("routing", _Q, _provider_const, d, now=1000.0)
    assert r1.grounded
    r2 = run_introspection("routing", _Q, _provider_const, d, now=1001.0)
    assert r2.grounded is False and "novelty gate" in r2.reason


# ── orchestration: empty telemetry / bad query / self-aspect → not grounded ──
def test_run_introspection_empty_corpus():
    d = IntrospectionDamper()
    r = run_introspection("routing", _Q, lambda _a: "", d)
    assert r.grounded is False and "no telemetry" in r.reason


def test_run_introspection_bad_query_safe():
    d = IntrospectionDamper()
    r = run_introspection("routing", {"kind": "regex", "pattern": "("},
                          _provider_const, d)
    assert r.grounded is False and "bad query" in r.reason


def test_run_introspection_self_referential_refused():
    d = IntrospectionDamper()
    r = run_introspection("introspection", _Q, _provider_const, d)
    assert r.grounded is False and "self-referential" in r.reason


def test_run_introspection_flaky_provider_never_crashes():
    d = IntrospectionDamper()
    def _boom(_a):
        raise RuntimeError("reader down")
    r = run_introspection("routing", _Q, _boom, d)
    assert r.grounded is False and "corpus unavailable" in r.reason
