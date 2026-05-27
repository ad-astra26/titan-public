"""Phase 6 — CGNMeaningOracle tests (§P6.H; INV-Syn-1 / INV-1).

Covers `titan_hcl/synthesis/cgn_meaning_oracle.py`:

- meaning_of populated from injected concept_reader
- meaning_of soft-fails to empty strands when reader returns None /
  raises / yields wrong type
- ground delegates to injected cgn_grounder
- ground returns degraded Grounding (strength=0, grounding_id="")
  when grounder returns None / raises / yields wrong shape
- ground clamps strength outside [0,1] (degraded as defense against
  CGN schema drift)
- ground NEVER invents a grounding_id when CGN didn't supply one
  (INV-Syn-1 enforcement)
- oracle_id, cost_class identity
- FeltContext neuromods passed through to the grounder
"""
from __future__ import annotations

import time

import pytest

from titan_hcl.synthesis.cgn_meaning_oracle import CGNMeaningOracle
from titan_hcl.synthesis.plugs import ConceptRef, FeltContext


# ─────────────────────────────────────────────────────────────────────────
# Identity
# ─────────────────────────────────────────────────────────────────────────


def test_oracle_id_and_cost_class():
    o = CGNMeaningOracle()
    assert o.oracle_id == "cgn"
    assert o.cost_class == "free"


# ─────────────────────────────────────────────────────────────────────────
# meaning_of
# ─────────────────────────────────────────────────────────────────────────


def test_meaning_of_returns_populated_strands_when_reader_supplies():
    def reader(concept_id, version):
        assert concept_id == "metaplex_nft"
        assert version == 3
        return {
            "declarative_anchors": ["d1", "d2"],
            "procedural_anchors": ["p1"],
            "episodic_anchors": ["e1", "e2", "e3"],
            "felt_anchors": [],
        }

    o = CGNMeaningOracle(concept_reader=reader)
    strand = o.meaning_of(ConceptRef(concept_id="metaplex_nft", version=3))

    assert strand.concept.concept_id == "metaplex_nft"
    assert strand.concept.version == 3
    assert strand.declarative_anchors == ["d1", "d2"]
    assert strand.procedural_anchors == ["p1"]
    assert strand.episodic_anchors == ["e1", "e2", "e3"]
    assert strand.felt_anchors == []


def test_meaning_of_reader_returns_none_yields_empty_strands():
    o = CGNMeaningOracle(concept_reader=lambda cid, v: None)
    strand = o.meaning_of(ConceptRef(concept_id="x", version=1))
    assert strand.declarative_anchors == []
    assert strand.procedural_anchors == []
    assert strand.episodic_anchors == []
    assert strand.felt_anchors == []


def test_meaning_of_reader_raises_yields_empty_strands():
    def bad_reader(cid, v):
        raise RuntimeError("kuzu down")

    o = CGNMeaningOracle(concept_reader=bad_reader)
    strand = o.meaning_of(ConceptRef(concept_id="x", version=1))
    assert strand.declarative_anchors == []
    assert strand.felt_anchors == []


def test_meaning_of_reader_returns_wrong_type_yields_empty_strands():
    """A reader returning a list or string (schema drift) must soft-fail."""
    o = CGNMeaningOracle(concept_reader=lambda cid, v: ["wrong shape"])
    strand = o.meaning_of(ConceptRef(concept_id="x", version=1))
    assert strand.declarative_anchors == []


def test_meaning_of_missing_strand_keys_defaults_to_empty():
    def reader(cid, v):
        return {"declarative_anchors": ["d1"]}   # only one strand key

    o = CGNMeaningOracle(concept_reader=reader)
    strand = o.meaning_of(ConceptRef(concept_id="x", version=1))
    assert strand.declarative_anchors == ["d1"]
    assert strand.procedural_anchors == []
    assert strand.episodic_anchors == []
    assert strand.felt_anchors == []


def test_meaning_of_handles_null_strand_values():
    """JSON null in a strand key (e.g. from a wire round-trip) → []."""
    def reader(cid, v):
        return {
            "declarative_anchors": None,
            "procedural_anchors": ["p1"],
            "episodic_anchors": None,
            "felt_anchors": None,
        }

    o = CGNMeaningOracle(concept_reader=reader)
    strand = o.meaning_of(ConceptRef(concept_id="x", version=1))
    assert strand.declarative_anchors == []
    assert strand.procedural_anchors == ["p1"]


# ─────────────────────────────────────────────────────────────────────────
# ground — delegation + INV-Syn-1 enforcement
# ─────────────────────────────────────────────────────────────────────────


def test_ground_delegates_to_cgn_grounder():
    calls = []

    def grounder(concept_id, version, valence, arousal, neuromods):
        calls.append({
            "concept_id": concept_id,
            "version": version,
            "valence": valence,
            "arousal": arousal,
            "neuromods": dict(neuromods),
        })
        return {
            "grounding_id": "g-abc-123",
            "strength": 0.75,
            "ts": 999.0,
        }

    o = CGNMeaningOracle(cgn_grounder=grounder)
    felt = FeltContext(valence=0.3, arousal=0.6, neuromods={"DA": 0.4})
    grounding = o.ground(
        ConceptRef(concept_id="metaplex_nft", version=3), felt,
    )
    assert grounding.concept.concept_id == "metaplex_nft"
    assert grounding.concept.version == 3
    assert grounding.grounding_id == "g-abc-123"
    assert grounding.strength == 0.75
    assert grounding.ts == 999.0
    # Grounder received the felt context verbatim.
    assert calls[0]["valence"] == 0.3
    assert calls[0]["arousal"] == 0.6
    assert calls[0]["neuromods"] == {"DA": 0.4}


def test_ground_cgn_degraded_returns_degraded_grounding():
    """CGN grounder returns None → degraded Grounding, strength=0, id="".

    INV-Syn-1 / INV-1: we NEVER invent a real grounding when CGN can't
    author one. The consumer detects strength=0 and treats it as
    "no felt strand yet" (same shape as the P4 stub returned)."""
    o = CGNMeaningOracle(cgn_grounder=lambda *args, **kw: None)
    grounding = o.ground(
        ConceptRef(concept_id="x", version=1),
        FeltContext(valence=0.0, arousal=0.5),
    )
    assert grounding.grounding_id == ""
    assert grounding.strength == 0.0
    assert grounding.ts > 0  # timestamp populated


def test_ground_cgn_raises_returns_degraded_grounding():
    def bad_grounder(*args, **kw):
        raise RuntimeError("CGN crashed")

    o = CGNMeaningOracle(cgn_grounder=bad_grounder)
    grounding = o.ground(
        ConceptRef(concept_id="x", version=1),
        FeltContext(valence=0.0, arousal=0.5),
    )
    assert grounding.grounding_id == ""
    assert grounding.strength == 0.0


def test_ground_cgn_returns_wrong_type_yields_degraded():
    """A grounder returning a list / string (schema drift) → degraded."""
    o = CGNMeaningOracle(cgn_grounder=lambda *a, **k: ["wrong shape"])
    grounding = o.ground(
        ConceptRef(concept_id="x", version=1),
        FeltContext(valence=0.0, arousal=0.5),
    )
    assert grounding.grounding_id == ""
    assert grounding.strength == 0.0


def test_ground_does_not_invent_grounding_id_when_cgn_omits_one():
    """INV-Syn-1: if CGN didn't supply grounding_id, we leave it empty.
    The consumer treats this as "CGN didn't grant; retry later"."""
    grounder = lambda *a, **k: {"strength": 0.5, "ts": 100.0}
    o = CGNMeaningOracle(cgn_grounder=grounder)
    grounding = o.ground(
        ConceptRef(concept_id="x", version=1),
        FeltContext(valence=0.0, arousal=0.5),
    )
    assert grounding.grounding_id == ""
    # Note: strength can still propagate even without an id — caller decides.
    assert grounding.strength == 0.5


def test_ground_clamps_strength_outside_unit_interval_to_zero():
    """CGN schema drift returning strength=1.5 or -0.3 → degraded."""
    for bad_strength in (1.5, -0.3, 2.0, -1.0):
        grounder = lambda *a, **k: {
            "grounding_id": "id",
            "strength": bad_strength,
            "ts": 1.0,
        }
        o = CGNMeaningOracle(cgn_grounder=grounder)
        g = o.ground(
            ConceptRef(concept_id="x", version=1),
            FeltContext(valence=0.0, arousal=0.5),
        )
        assert g.strength == 0.0, (
            f"strength={bad_strength} should clamp to 0.0 (degraded)"
        )


def test_ground_accepts_strength_at_boundaries():
    """0.0 and 1.0 are valid."""
    for s in (0.0, 0.5, 1.0):
        grounder = lambda *a, **k: {
            "grounding_id": "id",
            "strength": s,
            "ts": 1.0,
        }
        o = CGNMeaningOracle(cgn_grounder=grounder)
        g = o.ground(
            ConceptRef(concept_id="x", version=1),
            FeltContext(valence=0.0, arousal=0.5),
        )
        assert g.strength == s


def test_ground_now_fn_used_when_cgn_omits_ts():
    """If CGN didn't supply ts, we stamp with the injected now_fn."""
    fixed_now = 12345.6789
    grounder = lambda *a, **k: {"grounding_id": "id", "strength": 0.5}  # no ts
    o = CGNMeaningOracle(cgn_grounder=grounder, now_fn=lambda: fixed_now)
    g = o.ground(
        ConceptRef(concept_id="x", version=1),
        FeltContext(valence=0.0, arousal=0.5),
    )
    assert g.ts == fixed_now


def test_ground_default_unconfigured_grounder_degrades_quietly():
    """Default no-op grounder returns None → degraded grounding (no crash)."""
    o = CGNMeaningOracle()  # both reader + grounder defaults
    g = o.ground(
        ConceptRef(concept_id="x", version=1),
        FeltContext(valence=0.0, arousal=0.5),
    )
    assert g.strength == 0.0
    assert g.grounding_id == ""


def test_meaning_of_default_unconfigured_reader_returns_empty_strand():
    o = CGNMeaningOracle()
    strand = o.meaning_of(ConceptRef(concept_id="x", version=1))
    assert strand.declarative_anchors == []
    assert strand.felt_anchors == []
