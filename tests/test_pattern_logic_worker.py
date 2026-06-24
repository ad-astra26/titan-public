"""Offline tests for pattern_logic_worker passes — the cross-substrate lifecycle on
a REAL store with a deterministic embedder (no llama.cpp). INV-NO-STUBS: asserts a
real verdict+HAOV stream produces a promoted MODEL + an OFFER event + a sig cache.
"""
import json
import os
import tempfile

import numpy as np

from titan_hcl.synthesis.pattern_particle_store import PatternParticleStore
from titan_hcl.modules import pattern_logic_worker as plw


class FakeEmbedder:
    """Deterministic 8-d embedder: maps a context to a fixed direction by keyword so
    same-context strings cluster and different ones don't."""
    _DIRS = {
        "freshness": [1, 0, 0, 0, 0, 0, 0, 0],
        "coding": [0, 1, 0, 0, 0, 0, 0, 0],
        "other": [0, 0, 1, 0, 0, 0, 0, 0],
    }

    def encode(self, text):
        t = text.lower()
        key = "other"
        if "tvl" in t or "price" in t or "current" in t or "fresh" in t:
            key = "freshness"
        elif "code" in t or "coding" in t:
            key = "coding"
        v = np.asarray(self._DIRS[key], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v


def _cfg(**over):
    cfg = dict(plw._DEFAULTS)
    cfg.update({"min_cluster_size": 2, "min_sources": 2, "min_transitions": 5,
                "promote_floor": 0.85, "f_floor": 0.7, "cos_thresh": 0.85, "c0": 1.0})
    cfg.update(over)
    return cfg


def _store(d):
    cfg = _cfg()
    return PatternParticleStore(os.path.join(d, "pattern_logic.duckdb"),
                                c0=cfg["c0"], promote_floor=cfg["promote_floor"],
                                min_transitions=cfg["min_transitions"], f_floor=cfg["f_floor"])


def test_record_outer_transition():
    with tempfile.TemporaryDirectory() as d:
        s, e = _store(d), FakeEmbedder()
        tid = plw.record_outer_transition(s, e, {
            "context": "current TVL of Jupiter?", "oracle_id": "web_search",
            "frame": "general-lookup", "verdict": True, "source": "tool_verdict"})
        assert tid is not None
        rows = s.recent_transitions()
        assert rows[0]["operation"] == "RESEARCH"  # web_search → RESEARCH
        assert rows[0]["substrate"] == "outer" and rows[0]["verdict"] is True
        # empty context → skipped (no stub embedding).
        assert plw.record_outer_transition(s, e, {"context": "", "verdict": True}) is None
        s.close()


def test_ingest_inner_snapshot_deltas():
    with tempfile.TemporaryDirectory() as d:
        s, e = _store(d), FakeEmbedder()
        snap_path = os.path.join(d, "haov.json")
        with open(snap_path, "w") as f:
            json.dump({"hypotheses": [
                {"rule": "research_beats_direct_on_fresh", "action_context": {"x": 1},
                 "predicted_effect": "research wins", "confirmations": 2,
                 "falsifications": 1, "source": "impasse"}]}, f)
        seen = {}
        n = plw.ingest_inner_snapshot(s, e, snap_path, seen)
        assert n == 3  # 2 TRUE + 1 FALSE
        # idempotent — no new deltas on re-read.
        assert plw.ingest_inner_snapshot(s, e, snap_path, seen) == 0
        rows = s.recent_transitions()
        assert all(r["substrate"] == "inner" for r in rows)
        assert {r["operation"] for r in rows} == {"RESEARCH"}  # keyword 'research'
        s.close()


def test_cross_substrate_promotion_and_offer():
    """The §1.3 shape: RESEARCH→TRUE recurs across OUTER verdicts + INNER HAOV →
    cross-substrate PATTERN → promotes to MODEL → emits a PATTERN_MODEL_READY OFFER."""
    with tempfile.TemporaryDirectory() as d:
        s, e = _store(d), FakeEmbedder()
        cfg = _cfg()
        # OUTER: 4 RESEARCH→TRUE on freshness contexts.
        for i in range(4):
            plw.record_outer_transition(s, e, {
                "context": f"current price now {i}", "oracle_id": "web_search",
                "frame": "general-lookup", "verdict": True, "source": "tool_verdict"})
        # INNER: 3 RESEARCH→TRUE from the reasoning_strategy HAOV (different substrate).
        snap = os.path.join(d, "haov.json")
        with open(snap, "w") as f:
            json.dump({"hypotheses": [
                {"rule": "research wins on fresh tvl", "action_context": {},
                 "predicted_effect": "research", "confirmations": 3,
                 "falsifications": 0, "source": "impasse"}]}, f)
        plw.ingest_inner_snapshot(s, e, snap, {})

        offered = []  # offer_sink receives the MODEL dict; the worker wraps it.
        stats = plw.recognise_and_construct(s, cfg, offered.append)
        # A cross-substrate PATTERN formed (outer + inner, same op+region).
        assert stats["patterns"] >= 1, stats
        # Enough oracle-grounded evidence (7 TRUE) → promoted to MODEL.
        assert stats["models"] >= 1, stats
        assert len(offered) >= 1
        ev = plw.build_offer_event(offered[0], "pattern_logic")
        assert ev["type"].endswith("PATTERN_MODEL_READY")
        assert ev["dst"] == "synthesis"
        assert ev["payload"]["action"] == "research"  # RESEARCH → research lane
        assert ev["payload"]["goal_class"] == "general-lookup"  # outer frame, not inner
        models = s.get_models(min_c=0.85)
        assert len(models) == 1 and models[0]["operation"] == "RESEARCH"
        s.close()


def test_single_substrate_single_domain_not_proposed():
    """INV-PL-4: a single-substrate single-domain cluster is NOT pattern_logic's job
    (it belongs to the existing miners) — must NOT propose."""
    with tempfile.TemporaryDirectory() as d:
        s, e = _store(d), FakeEmbedder()
        cfg = _cfg()
        for i in range(5):  # all OUTER, all same frame → not cross-substrate/domain
            plw.record_outer_transition(s, e, {
                "context": f"current tvl {i}", "oracle_id": "web_search",
                "frame": "general-lookup", "verdict": True, "source": "tool_verdict"})
        stats = plw.recognise_and_construct(s, cfg, lambda m: None)
        assert stats["patterns"] == 0, stats  # fenced out by INV-PL-4
        s.close()


def test_model_reuse_citation():
    with tempfile.TemporaryDirectory() as d:
        s, e = _store(d), FakeEmbedder()
        cfg = _cfg()
        for i in range(4):
            plw.record_outer_transition(s, e, {
                "context": f"current price {i}", "oracle_id": "web_search",
                "frame": "general-lookup", "verdict": True, "source": "tool_verdict"})
        snap = os.path.join(d, "haov.json")
        with open(snap, "w") as f:
            json.dump({"hypotheses": [{"rule": "research fresh", "action_context": {},
                       "predicted_effect": "research", "confirmations": 3,
                       "falsifications": 0, "source": "impasse"}]}, f)
        plw.ingest_inner_snapshot(s, e, snap, {})
        plw.recognise_and_construct(s, cfg, lambda m: None)
        assert s.get_stats()["models_active"] == 1

        # A new matching RESEARCH→TRUE arrives → reused MODEL is cited (G-REUSE).
        plw.record_outer_transition(s, e, {
            "context": "current price again", "oracle_id": "web_search",
            "frame": "general-lookup", "verdict": True, "source": "tool_verdict"})
        plw.recognise_and_construct(s, cfg, lambda m: None)
        assert s.get_stats()["models_cited"] == 1
        s.close()


def test_inner_offer_corroboration_events():
    """OFFER-inner: a promoted MODEL with inner HAOV provenance emits a
    CGN_MODEL_CORROBORATION naming the contributing rule(s) (rule-keyed, §VC-2)."""
    with tempfile.TemporaryDirectory() as d:
        s, e = _store(d), FakeEmbedder()
        cfg = _cfg()
        for i in range(4):
            plw.record_outer_transition(s, e, {
                "context": f"current price {i}", "oracle_id": "web_search",
                "frame": "general-lookup", "verdict": True, "source": "tool_verdict"})
        snap = os.path.join(d, "haov.json")
        with open(snap, "w") as f:
            json.dump({"hypotheses": [{"rule": "research_beats_direct_on_fresh",
                       "action_context": {}, "predicted_effect": "research wins",
                       "confirmations": 3, "falsifications": 0, "source": "impasse"}]}, f)
        plw.ingest_inner_snapshot(s, e, snap, {})

        offered = []
        plw.recognise_and_construct(s, cfg, offered.append)
        assert offered, "a model should have been promoted"
        model = offered[0]
        # provenance: the inner rule name is recorded as the inner transition source.
        rules = s.inner_rules_for_particle(model["parent_id"])
        assert ("reasoning_strategy", "research_beats_direct_on_fresh") in rules

        evs = plw.build_corroboration_events(s, model, "pattern_logic")
        assert len(evs) == 1
        ev = evs[0]
        assert ev["type"].endswith("CGN_MODEL_CORROBORATION")
        assert ev["dst"] == "cgn"
        assert ev["payload"]["consumer"] == "reasoning_strategy"
        assert "research_beats_direct_on_fresh" in ev["payload"]["rules"]
        assert 0.85 <= ev["payload"]["strength"] <= 1.0
        s.close()


def test_haov_corroborate_boosts_confidence_only():
    """The CGN side of OFFER-inner: corroborate() raises a hypothesis's confidence by
    rule name, records it in action_context, and leaves the in-process tally honest."""
    from titan_hcl.logic.cgn_types import GeneralizedHAOVTracker
    t = GeneralizedHAOVTracker("reasoning_strategy")
    h = t.hypothesize(action_context={}, observation={
        "effect": "research", "magnitude": 0.5,
        "rule_name": "research_beats_direct_on_fresh", "source": "impasse"})
    assert h is not None
    c0 = h.confidence
    assert t.corroborate("research_beats_direct_on_fresh", 1.0) is True
    assert h.confidence > c0
    assert h.confirmations == 0 and h.falsifications == 0  # in-process tally untouched
    assert h.action_context["corroborations"] == 1
    assert t.corroborate("nonexistent_rule", 1.0) is False     # unknown rule → no-op
    c1 = h.confidence
    assert t.corroborate("research_beats_direct_on_fresh", 0.0) is False  # zero strength
    assert h.confidence == c1


def test_purely_outer_model_no_corroboration():
    """A model with NO inner provenance emits no corroboration (validity guard)."""
    with tempfile.TemporaryDirectory() as d:
        s, e = _store(d), FakeEmbedder()
        # Two distinct OUTER frames (cross-DOMAIN, not cross-substrate) so a pattern
        # can form without any inner transition.
        for i in range(3):
            plw.record_outer_transition(s, e, {
                "context": f"current price {i}", "oracle_id": "web_search",
                "frame": "general-lookup", "verdict": True, "source": "tool_verdict"})
        for i in range(3):
            plw.record_outer_transition(s, e, {
                "context": f"current price time {i}", "oracle_id": "web_search",
                "frame": "time-lookup", "verdict": True, "source": "skill_score"})
        offered = []
        plw.recognise_and_construct(s, _cfg(), offered.append)
        if offered:  # if a cross-domain model formed, it must have no inner provenance
            assert plw.build_corroboration_events(s, offered[0], "pattern_logic") == []
        s.close()
