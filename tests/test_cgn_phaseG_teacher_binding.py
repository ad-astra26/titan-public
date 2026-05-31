"""Phase G (RFP_cgn_enhancements §9.3 / proto-SPEC §9.5c) — meta-teacher → policy binding.

The meta-teacher accumulates a curriculum of ReasoningBindings ("in inner context X,
the proper next primitive is Y"); the meta-reasoning policy reads them READ-ONLY and
adds a logit bias toward the taught primitive in matching contexts — the strong
cross-process teacher→policy channel the pre-G scalar reward_bonus (≤0.05) never gave.

Covers:
  1. context_signature geometry + determinism + drift-guard vs META_PRIMITIVES.
  2. store mint / refine-merge / distinct-primitive / eviction.
  3. retrieve top-k + similarity floor.
  4. recognized/produced counters + curriculum promotion.
  5. read-only reader parity (cross-process channel).
  6. engine: _current_context_signature + _compute_teacher_bias biases the right primitive.

Run: python -m pytest tests/test_cgn_phaseG_teacher_binding.py -v -p no:anchorpy
"""
import os
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from titan_hcl.logic.reasoning_binding import (
    BINDING_PRIMITIVES, SIG_DIM, ReasoningBindingStore, build_context_signature,
    MINT_MERGE_THRESHOLD,
)


# ── 1. signature geometry ────────────────────────────────────────────────
def test_signature_geometry_and_determinism():
    s1 = build_context_signature(
        "concept_grounding", "FLOW", ["FORMULATE.define", "FORMULATE.define"], "language")
    s2 = build_context_signature(
        "concept_grounding", "FLOW", ["FORMULATE.define", "FORMULATE.define"], "language")
    assert s1.shape == (SIG_DIM,)
    assert s1.dtype == np.float32
    assert abs(float(np.linalg.norm(s1)) - 1.0) < 1e-5     # L2-normalized
    assert np.allclose(s1, s2)                              # deterministic (process-stable hash)
    # Different context → lower cosine.
    s3 = build_context_signature("experience_pressure", "COLD", [], "knowledge")
    assert float(s1 @ s3) < float(s1 @ s2)


def test_signature_drift_guard():
    """BINDING_PRIMITIVES must mirror meta_reasoning.META_PRIMITIVES (same order)."""
    from titan_hcl.logic.meta_reasoning import META_PRIMITIVES
    assert BINDING_PRIMITIVES == tuple(META_PRIMITIVES)


def test_concept_aware_signature():
    """Concept-aware (Maker 2026-05-31): same context but different grounding_concept
    must diverge; same concept must match; empty concept = the legacy context-only key."""
    base = dict(trigger_reason="concept_grounding", dominant_emotion="FLOW",
                chain_so_far=["FORMULATE.define"], domain="language")
    a = build_context_signature(**base, grounding_concept="warmth")
    a2 = build_context_signature(**base, grounding_concept="warmth")
    b = build_context_signature(**base, grounding_concept="music")
    empty = build_context_signature(**base, grounding_concept="")
    assert np.allclose(a, a2)                                # same concept → identical
    assert float(a @ b) < float(a @ a2)                      # different concept → diverges
    assert float(a @ empty) < 1.0                            # concept-keyed ≠ context-only
    # Two DIFFERENT concepts still share the rest of the context (sim stays > 0).
    assert float(a @ b) > 0.5


def test_empty_chain_signature_driven_by_context():
    """At chain entry (empty chain) the signature is driven by trigger/emotion/domain."""
    s = build_context_signature("concept_grounding", "FLOW", [], "language")
    assert abs(float(np.linalg.norm(s)) - 1.0) < 1e-5
    # primitive-histogram block is all zero when chain empty
    assert float(np.linalg.norm(s[:len(BINDING_PRIMITIVES)])) == 0.0


# ── 2/3. store mint / refine / retrieve ──────────────────────────────────
def _fresh_store():
    tmp = tempfile.mkdtemp()
    return ReasoningBindingStore(db_path=os.path.join(tmp, "rb.db")), tmp


def test_mint_then_refine_merges():
    st, _ = _fresh_store()
    sig = build_context_signature("concept_grounding", "FLOW",
                                  ["FORMULATE.define", "FORMULATE.define"], "language")
    bid = st.mint_or_refine(sig, "EVALUATE", "peer_cgn", "post_formulate_loop_breaker")
    assert bid >= 1 and st.count() == 1
    # near-identical context + same primitive → MERGE (same id, n_taught bumped).
    bid2 = st.mint_or_refine(sig, "EVALUATE", "peer_cgn", "post_formulate_loop_breaker")
    assert bid2 == bid and st.count() == 1
    b = st.all()[0]
    assert b.n_taught == 2
    assert b.confidence > 0.30                              # corroboration raised it


def test_distinct_primitive_makes_new_binding():
    st, _ = _fresh_store()
    sig = build_context_signature("concept_grounding", "FLOW", ["FORMULATE.define"], "language")
    st.mint_or_refine(sig, "EVALUATE", "", "p1")
    st.mint_or_refine(sig, "SYNTHESIZE", "", "p2")          # same ctx, different primitive
    assert st.count() == 2
    prims = {b.recommended_primitive for b in st.all()}
    assert prims == {"EVALUATE", "SYNTHESIZE"}


def test_retrieve_topk_and_floor():
    st, _ = _fresh_store()
    sig_a = build_context_signature("concept_grounding", "FLOW", ["FORMULATE.define"], "language")
    sig_b = build_context_signature("experience_pressure", "COLD", ["RECALL.semantic"], "knowledge")
    st.mint_or_refine(sig_a, "EVALUATE", "", "pa")
    st.mint_or_refine(sig_b, "BREAK", "", "pb")
    hits = st.retrieve_topk(sig_a, k=3, sim_floor=0.6)
    assert hits and hits[0][0].recommended_primitive == "EVALUATE"
    assert hits[0][1] >= 0.99                                # exact context → sim≈1
    # A wholly unrelated query clears the floor for nothing.
    far = build_context_signature("periodic", "DARK",
                                  ["INTROSPECT.state", "BREAK.reset"], "social")
    assert all(s >= 0.6 for _, s in st.retrieve_topk(far, k=3, sim_floor=0.6))


def test_unknown_primitive_rejected():
    st, _ = _fresh_store()
    sig = build_context_signature("periodic", "FLOW", [], "general")
    try:
        st.mint_or_refine(sig, "NOT_A_PRIMITIVE")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ── 4. counters + curriculum ─────────────────────────────────────────────
def test_counters_and_curriculum_promotion():
    st, _ = _fresh_store()
    sig = build_context_signature("concept_grounding", "FLOW", ["FORMULATE.define"], "language")
    bid = st.mint_or_refine(sig, "EVALUATE", "", "p")
    b0 = st.all()[0]
    assert b0.level == 0 and b0.n_recognized == 0 and b0.n_produced == 0
    # produced bumps both produced + recognized and triggers promotion.
    st.record_produced(bid)
    st.refresh(force=True)
    b1 = next(x for x in st.all() if x.binding_id == bid)
    assert b1.n_produced == 1 and b1.n_recognized == 1 and b1.level >= 1
    # recognized-only does not raise produced.
    st.record_recognized(bid)
    st.refresh(force=True)
    b2 = next(x for x in st.all() if x.binding_id == bid)
    assert b2.n_produced == 1 and b2.n_recognized == 2


# ── 5. read-only reader parity (the cross-process channel) ────────────────
def test_readonly_reader_parity():
    st, tmp = _fresh_store()
    sig = build_context_signature("concept_grounding", "FLOW", ["FORMULATE.define"], "language")
    st.mint_or_refine(sig, "EVALUATE", "peer_cgn", "p")
    ro = ReasoningBindingStore(db_path=os.path.join(tmp, "rb.db"), read_only=True)
    assert ro.count() == 1
    hits = ro.retrieve_topk(sig, k=1, sim_floor=0.6)
    assert hits and hits[0][0].recommended_primitive == "EVALUATE"


# ── 6. engine integration ────────────────────────────────────────────────
def _engine_with_store(db_path):
    from titan_hcl.logic.meta_reasoning import MetaReasoningEngine
    eng = MetaReasoningEngine(config={}, send_queue=None)
    eng._binding_store = ReasoningBindingStore(db_path=db_path, read_only=True)
    return eng


def test_engine_signature_and_bias_prefers_binding():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "rb.db")
    writer = ReasoningBindingStore(db_path=db)
    # Teach: in (concept_grounding / FLOW / language) after a FORMULATE loop → EVALUATE.
    ctx_chain = ["FORMULATE.define", "FORMULATE.define"]
    sig = build_context_signature("concept_grounding", "FLOW", ctx_chain, "language")
    bid = writer.mint_or_refine(sig, "EVALUATE", "peer_cgn", "post_formulate_loop_breaker")
    # Drive confidence up so the bias is meaningful.
    for _ in range(8):
        writer.record_produced(bid)

    eng = _engine_with_store(db)
    # Put the engine into the taught context.
    eng.state.trigger_reason = "concept_grounding"
    eng._emot_dom_at_chain_start = "FLOW"
    eng.state.grounding_consumer = "language"
    eng.state.chain = list(ctx_chain)

    # Signature the engine computes must match the taught one (one cosine space).
    qsig = eng._current_context_signature()
    assert qsig is not None and np.allclose(qsig, sig)

    bias, top = eng._compute_teacher_bias()
    from titan_hcl.logic.meta_reasoning import META_PRIMITIVES
    eval_idx = META_PRIMITIVES.index("EVALUATE")
    assert top is not None and top[0].recommended_primitive == "EVALUATE"
    assert bias[eval_idx] > 0.0                              # EVALUATE is biased up
    assert bias.argmax() == eval_idx                         # and it dominates the bias
    # Non-matching context yields no bias.
    eng.state.trigger_reason = "periodic"
    eng.state.grounding_consumer = "social"
    eng.state.chain = ["INTROSPECT.state"]
    bias2, top2 = eng._compute_teacher_bias()
    assert float(np.linalg.norm(bias2)) == 0.0


def test_engine_without_store_is_inert():
    from titan_hcl.logic.meta_reasoning import MetaReasoningEngine
    eng = MetaReasoningEngine(config={}, send_queue=None)
    eng._binding_store = None
    bias, top = eng._compute_teacher_bias()
    assert float(np.linalg.norm(bias)) == 0.0 and top is None


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("OK — all Phase G unit checks passed")
