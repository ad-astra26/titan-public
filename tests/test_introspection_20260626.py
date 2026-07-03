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


# ── IntrospectHelper end-to-end (mocked lock-safe corpus) ────────────────────
def test_introspect_helper_grounds_and_novelty_gates():
    import asyncio
    from titan_hcl.logic.agency.helpers.introspect import IntrospectHelper
    h = IntrospectHelper(damper=IntrospectionDamper())
    corpus = ('{"skills":[{"goal_class":"ground-concept:self"},'
              '{"goal_class":"ground-concept:self"},{"goal_class":"coding"}]}')
    async def _mock(_a):
        return corpus
    h._read_corpus = _mock

    async def _go():
        r1 = await h.execute({"aspect": "skills"})
        assert r1["introspection_grounded"] is True
        rt = r1["helper_params"]["_research_target"]
        assert rt["concept_id"] == "SELF:skills" and rt["source"] == "introspection"
        assert r1["helper_params"]["query"] == "introspect:skills"   # string for 3a
        assert len(r1["result"]) >= 40 and "SELF:skills" in r1["result"]
        # novelty gate — same corpus must NOT re-ground
        r2 = await h.execute({"aspect": "skills"})
        assert r2["introspection_grounded"] is False
        assert "novelty gate" in r2["introspection_reason"]
    asyncio.run(_go())


def test_introspect_helper_empty_endpoint_safe():
    import asyncio
    from titan_hcl.logic.agency.helpers.introspect import IntrospectHelper
    h = IntrospectHelper(damper=IntrospectionDamper())
    async def _empty(_a):
        return ""
    h._read_corpus = _empty
    async def _go():
        r = await h.execute({"aspect": "skills"})
        assert r["introspection_grounded"] is False
        assert "_research_target" not in r["helper_params"]
    asyncio.run(_go())


# ── the autonomous grounding event (the fix: introspect is non-routing, so the P8
#    routing-coupled 3a emit BAILS for it → it needs its OWN RESEARCH_CURIOSITY_GROUNDED
#    emit; this asserts the event the agency INTROSPECT_REQUEST handler builds is a
#    well-formed 3b payload the memory consumer accepts) ──────────────────────────
def test_autonomous_grounding_event_is_valid_3b_payload():
    import asyncio
    from titan_hcl.logic.agency.helpers.introspect import IntrospectHelper
    h = IntrospectHelper(damper=IntrospectionDamper())
    corpus = ('{"skills":[{"goal_class":"ground-concept:self"},'
              '{"goal_class":"ground-concept:biology"}]}')
    async def _mock(_a):
        return corpus
    h._read_corpus = _mock

    async def _go():
        r = await h.execute({"aspect": "skills"})
        assert r["introspection_grounded"] is True
        # build the event EXACTLY as agency_worker's INTROSPECT_REQUEST handler does
        hp = r["helper_params"]
        event_payload = {
            "query": str(hp.get("query", "")),
            "content": str(r.get("result", "")),
            "_research_target": hp.get("_research_target"),
        }
        # the 3b memory consumer requires non-empty (query+"\n"+content) + a target
        assert (event_payload["query"] + "\n" + event_payload["content"]).strip()
        rt = event_payload["_research_target"]
        assert rt and rt["concept_id"] == "SELF:skills"
        assert rt["source"] == "introspection" and rt["domain_hint"] == "self"
    asyncio.run(_go())


# ── the bus trigger constant exists (the §5 shared should_fire wire) ──────────
def test_introspect_request_bus_constant_exists():
    from titan_hcl import bus
    assert bus.INTROSPECT_REQUEST == "INTROSPECT_REQUEST"


# ── the standalone-helper manifest contract (HelperRegistry calls these
#    synchronously — a missing status()/capabilities/etc. logged a live warning
#    on T3 boot + excluded introspect from list_available; regression guard) ──
def test_introspect_helper_satisfies_registry_manifest_contract():
    from titan_hcl.logic.agency.helpers.introspect import IntrospectHelper
    h = IntrospectHelper()
    assert h.status() == "available"           # must not raise (sync /health path)
    assert isinstance(h.capabilities, list) and h.capabilities
    assert isinstance(h.enriches, list) and "self" in h.enriches
    assert isinstance(h.description, str) and h.description
    assert h.resource_cost in ("low", "medium", "high")
    assert h.latency in ("low", "medium", "high")
    assert h.requires_sandbox is False         # read-only own telemetry (INV-TX-2)
    assert h.name == "introspect" and h.action_type == "introspect"


# ── §7.P-B AUTONOMOUS-TRIGGER TRANSPORT FIX (2026-07-03) ────────────────────────
# Regression for the dead-letter bug: should_fire emitted INTROSPECT_REQUEST
# dst="agency" ~38x/10min but 0 were delivered — the agency A.8.6 subprocess
# ("agency_worker") never receives raw bus app-messages (they go to the PARENT
# "agency" loop) AND is reply_only (its drain drops all but SHUTDOWN/QUERY). The
# fix: the parent forwards INTROSPECT_REQUEST into the subprocess as a fire-and-
# forget QUERY action="introspect" (QUERY passes the drain), reusing the already-
# correct subprocess grounding logic.
def test_parent_forwards_introspect_request_as_query_to_subprocess():
    from types import SimpleNamespace
    from titan_hcl.core.plugin import TitanHCL
    from titan_hcl import bus

    published = []
    fake = SimpleNamespace(bus=SimpleNamespace(publish=published.append))
    msg = {"type": bus.INTROSPECT_REQUEST, "src": "self_learning", "dst": "agency",
           "payload": {"src_gp": 4242, "aspect": "skills"}}
    TitanHCL._forward_introspect_request(fake, msg)

    assert len(published) == 1
    out = published[0]
    assert out["type"] == bus.QUERY                # QUERY passes the reply_only drain
    assert out["dst"] == "agency_worker"           # the SUBPROCESS name, not "agency"
    assert out["payload"]["action"] == "introspect"
    assert out["payload"]["src_gp"] == 4242        # original payload carried through
    assert out["payload"]["aspect"] == "skills"
    assert out.get("rid") is None                  # fire-and-forget → no RESPONSE


def test_forward_introspect_request_never_raises_on_bus_error():
    from types import SimpleNamespace
    from titan_hcl.core.plugin import TitanHCL
    from titan_hcl import bus

    def _boom(_m):
        raise RuntimeError("bus down")
    fake = SimpleNamespace(bus=SimpleNamespace(publish=_boom))
    # Must swallow the error — a forward failure can never crash the agency loop.
    TitanHCL._forward_introspect_request(
        fake, {"type": bus.INTROSPECT_REQUEST, "payload": {}})
