"""Offline tests for PatternParticleStore — the proto-BRAIN particle lifecycle.

Asserts a REAL evidence stream drives propose→merge→promote (mutate-not-update)
and the G-REUSE citation path — NOT empty-input stubs (INV-NO-STUBS).
"""
import os
import tempfile

import numpy as np

from titan_hcl.synthesis.pattern_particle_store import (
    PatternParticleStore, beta_to_f, beta_to_c,
)


def _store(tmpdir):
    return PatternParticleStore(
        os.path.join(tmpdir, "pattern_logic.duckdb"),
        c0=1.0, promote_floor=0.85, min_transitions=5, f_floor=0.7,
    )


def test_beta_helpers():
    assert beta_to_f(1.0, 1.0) == 0.5
    assert beta_to_f(9.0, 1.0) == 0.9
    # c saturates with evidence mass n=(α+β)-2.
    assert beta_to_c(1.0, 1.0, 1.0) == 0.0          # no evidence
    assert abs(beta_to_c(6.0, 1.0, 1.0) - 5.0 / 6.0) < 1e-9   # n=5
    assert beta_to_c(7.0, 1.0, 1.0) > 0.85          # n=6 → promotable


def test_transition_log_and_cluster_feed():
    with tempfile.TemporaryDirectory() as d:
        s = _store(d)
        sig = [0.1, 0.2, 0.3, 0.4]
        for i in range(3):
            s.record_transition(signature=sig, operation="RESEARCH", frame="general-lookup",
                                 verdict=True, substrate="outer", source="oracle", ts=100.0 + i)
        rows = s.recent_transitions(only_unclustered=True)
        assert len(rows) == 3
        assert isinstance(rows[0]["signature"], np.ndarray)
        assert rows[0]["operation"] == "RESEARCH"
        s.close()


def test_freshness_routing_lifecycle():
    """The §1.3 worked example shape: RESEARCH reliably TRUE in a context region →
    PATTERN forms low-c → evidence accrues → promotes to MODEL (mutate-not-update) →
    cited (G-REUSE)."""
    with tempfile.TemporaryDirectory() as d:
        s = _store(d)
        sig = [1.0, 0.0, 0.0]
        # OBSERVE: RESEARCH→TRUE across two substrates (outer verdict + inner HAOV).
        tx_ids = []
        evidence = []
        for i in range(2):
            tx_ids.append(s.record_transition(
                signature=sig, operation="RESEARCH", frame="general-lookup",
                verdict=True, substrate="outer", source="oracle", ts=10.0 + i))
            evidence.append({"verdict": True, "source": "oracle", "ts": 10.0 + i})
        tx_ids.append(s.record_transition(
            signature=sig, operation="RESEARCH", frame="general-lookup",
            verdict=True, substrate="inner", source="reasoning_strategy", ts=12.0))
        evidence.append({"verdict": True, "source": "reasoning_strategy", "ts": 12.0})

        # RECOGNISE: propose a PATTERN (≥2 sources: oracle + reasoning_strategy).
        pid = s.propose_pattern(signature=sig, operation="RESEARCH", frame="general-lookup",
                                evidence=evidence, n_sources=2, tx_ids=tx_ids)
        p = s.get_particle(pid)
        assert p["kind"] == "PATTERN" and p["status"] == "ACTIVE"
        # 3 TRUE, 0 FALSE over uniform prior → α=1+3=4, β=1 → f=4/5=0.8
        assert abs(p["f"] - 0.8) < 1e-9
        assert not s.eligible_for_promotion(pid)  # n=3 < 5 transitions yet

        # CONSTRUCT: cheap-oracle re-tests accrue more TRUE evidence.
        for i in range(4):
            s.merge_evidence(pid, verdict=True, source="oracle_router", ts=20.0 + i)
        p = s.get_particle(pid)
        # α=4+4=8, β=1 → n=7 → c=7/8=0.875 ≥0.85; f=8/9≈0.889 ≥0.7
        assert s.eligible_for_promotion(pid), p

        # PROMOTE: mutate-not-update successor.
        mid = s.promote_to_model(pid)
        assert mid != pid
        model = s.get_particle(mid)
        parent = s.get_particle(pid)
        assert model["kind"] == "MODEL" and model["status"] == "ACTIVE"
        assert model["parent_id"] == pid
        assert parent["status"] == "SUPERSEDED"  # lineage preserved, not deleted

        # OFFER / cache: get_models surfaces the high-c model.
        models = s.get_models(min_c=0.85)
        assert len(models) == 1 and models[0]["id"] == mid
        assert models[0]["operation"] == "RESEARCH"

        # G-REUSE: cite it.
        assert s.cite_model(mid) == 1
        stats = s.get_stats()
        assert stats["models_active"] == 1 and stats["models_cited"] == 1
        assert stats["patterns_active"] == 0 and stats["superseded"] == 1
        s.close()


def test_low_f_pattern_not_promoted():
    """A reliably-FALSE op (e.g. DIRECT→FALSE) is a valid PATTERN but NOT a MODEL
    you 'apply' (f below floor) — the anti-pattern half of freshness routing."""
    with tempfile.TemporaryDirectory() as d:
        s = _store(d)
        sig = [0.0, 1.0]
        ev = [{"verdict": False, "source": "oracle", "ts": float(i)} for i in range(8)]
        pid = s.propose_pattern(signature=sig, operation="RECALL", frame="general-lookup",
                                evidence=ev, n_sources=2)
        p = s.get_particle(pid)
        assert p["f"] < 0.3  # mostly FALSE
        # c is high (lots of evidence) but f below floor → not promotion-eligible.
        assert not s.eligible_for_promotion(pid)
        s.close()


def test_persistence_reopen():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pattern_logic.duckdb")
        s = PatternParticleStore(path, c0=1.0)
        tid = s.record_transition(signature=[0.5, 0.5], operation="TOOL", frame="coding-query",
                                  verdict=True, substrate="outer", source="oracle")
        pid = s.propose_pattern(signature=[0.5, 0.5], operation="TOOL", frame="coding-query",
                                evidence=[{"verdict": True, "source": "oracle", "ts": 1.0}],
                                n_sources=2, tx_ids=[tid])
        s.close()
        # Reopen — ids must not collide, data must persist.
        s2 = PatternParticleStore(path, c0=1.0)
        assert s2.get_particle(pid) is not None
        new_tid = s2.record_transition(signature=[0.5, 0.5], operation="TOOL",
                                       frame="coding-query", verdict=True,
                                       substrate="outer", source="oracle")
        assert new_tid > tid  # counter resumed past the persisted max
        s2.close()
