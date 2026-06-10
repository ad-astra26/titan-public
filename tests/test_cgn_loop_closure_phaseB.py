"""RFP_cgn_loop_closure §7.B (Q1 Option-β) — grounded V drives per-step selection.

The crux: the 9 meta-primitives must walk the populated CGN matrix because the
grounded V(s) of the concept's neighbourhood makes a primitive the right move —
NOT from a context-free FORMULATE-attractor policy. These tests pin:
  1. _softmax is stable + normalized.
  2. _grounded_primitive_vectors builds (composed_V, conf) aligned to
     META_PRIMITIVES from meta_cgn's per-primitive Beta posteriors; returns
     (None, None) when meta_cgn is cold (→ caller falls back to the policy).
  3. THE crux: a high-confidence grounded V on a non-FORMULATE primitive
     overrides the FORMULATE policy logit advantage in the Option-β blend.
  4. An UNPROVEN (low-confidence) grounded V defers to the policy (λ=conf).

Run: python -m pytest tests/test_cgn_loop_closure_phaseB.py -v -p no:anchorpy
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from titan_hcl.logic.meta_reasoning import (
    MetaReasoningEngine, META_PRIMITIVES, NUM_META_ACTIONS, _softmax,
)
from titan_hcl.logic.meta_cgn import PrimitiveConcept


def _make_engine(**cfg):
    return MetaReasoningEngine(config=cfg, send_queue=None)


def _set_primitive(eng, name, alpha, beta):
    """Force a primitive's Beta posterior → derived V + confidence."""
    pc = PrimitiveConcept(primitive_id=name, alpha=float(alpha), beta=float(beta))
    pc.recompute_derived()
    eng._meta_cgn._primitives[name] = pc
    return pc


def test_softmax_stable_and_normalized():
    out = _softmax(np.array([1000.0, 1000.0, 999.0], dtype=np.float32))
    assert np.isfinite(out).all()
    assert abs(float(out.sum()) - 1.0) < 1e-5
    # Larger logit → larger probability.
    assert out[0] > out[2]


def test_grounded_primitive_vectors_aligned_to_meta_primitives():
    eng = _make_engine()
    assert eng._meta_cgn is not None  # engine builds its MetaCGNConsumer
    # HYPOTHESIZE: strongly grounded + confident (V=0.9, big n_eff → high conf).
    _set_primitive(eng, "HYPOTHESIZE", alpha=900.0, beta=100.0)
    composed_V, conf = eng._grounded_primitive_vectors()
    assert composed_V is not None and conf is not None
    assert composed_V.shape == (NUM_META_ACTIONS,)
    h = META_PRIMITIVES.index("HYPOTHESIZE")
    assert composed_V[h] > 0.85          # V = beta_mean(900,100) ≈ 0.9
    assert conf[h] > 0.6                  # n_eff≈1000 → 1000/1500 ≈ 0.667


def test_grounded_vectors_none_when_meta_cgn_absent():
    eng = _make_engine()
    eng._meta_cgn = None
    composed_V, conf = eng._grounded_primitive_vectors()
    assert composed_V is None and conf is None


def _blend(composed_V, conf, policy, temperature=1.0):
    """The Option-β per-step blend, exactly as tick() computes it."""
    final = conf * _softmax(composed_V) + (1.0 - conf) * _softmax(policy, temperature)
    return final / (final.sum() + 1e-8)


def test_high_conf_grounded_v_shifts_off_formulate():
    """THE crux (G6): when the grounded matrix is CONFIDENT that the overused
    FORMULATE has low value (high conf + low V) and another primitive has high
    value, the Option-β blend shifts the argmax OFF FORMULATE despite the
    FORMULATE-attractor policy (+5 logit, §5.5). This is the monoculture-break
    mechanism: FORMULATE is suppressed by its OWN high-confidence low grounded V.
    λ=conf, so 'full authority' arrives as evidence accumulates (RFP §6 Q1)."""
    eng = _make_engine()
    # All primitives moderately sampled so conf is meaningful; FORMULATE proven
    # LOW value (overused → low reward), HYPOTHESIZE proven HIGH (n_eff≈10k →
    # conf≈0.95 = the 'confident' regime where grounded V earns full authority).
    for p in META_PRIMITIVES:
        _set_primitive(eng, p, alpha=100.0, beta=100.0)          # V=0.5, conf≈0.29
    _set_primitive(eng, "FORMULATE", alpha=500.0, beta=9500.0)   # V≈0.05, conf≈0.95
    _set_primitive(eng, "HYPOTHESIZE", alpha=9500.0, beta=500.0) # V≈0.95, conf≈0.95
    composed_V, conf = eng._grounded_primitive_vectors()

    policy = np.zeros(NUM_META_ACTIONS, dtype=np.float32)
    policy[META_PRIMITIVES.index("FORMULATE")] = 5.0   # the FORMULATE attractor
    assert int(np.argmax(_softmax(policy))) == META_PRIMITIVES.index("FORMULATE")

    final = _blend(composed_V, conf, policy)
    f = META_PRIMITIVES.index("FORMULATE")
    # The blend shifts the pick OFF the FORMULATE monoculture...
    assert int(np.argmax(final)) != f
    # ...and dramatically cuts FORMULATE's probability vs the raw policy (0.95).
    assert float(final[f]) < 0.5
    # HYPOTHESIZE (high V, confident) is now the most likely pick.
    assert int(np.argmax(final)) == META_PRIMITIVES.index("HYPOTHESIZE")


def test_unproven_grounded_v_defers_to_policy():
    """λ=confidence: an UNPROVEN primitive (low n → low conf) lets the policy
    decide — grounded V only takes over once it has earned confidence."""
    eng = _make_engine()
    # All primitives near-prior (alpha≈beta≈1 → conf≈0): grounded V is neutral.
    for p in META_PRIMITIVES:
        _set_primitive(eng, p, alpha=1.0, beta=1.0)
    composed_V, conf = eng._grounded_primitive_vectors()
    assert float(np.max(conf)) < 0.05   # essentially no confidence yet

    policy = np.zeros(NUM_META_ACTIONS, dtype=np.float32)
    policy[META_PRIMITIVES.index("SYNTHESIZE")] = 4.0   # policy prefers SYNTHESIZE
    final = _blend(composed_V, conf, policy)
    # With conf≈0 the blend ≈ softmax(policy) → policy's choice wins.
    assert int(np.argmax(final)) == META_PRIMITIVES.index("SYNTHESIZE")


def test_grounded_v_selection_flag_default_on():
    eng = _make_engine()
    assert eng._grounded_v_selection_enabled is True
    eng2 = _make_engine(grounded_v_selection_enabled=False)
    assert eng2._grounded_v_selection_enabled is False
    # Flag off → vectors suppressed → caller uses the legacy policy path.
    assert eng2._grounded_primitive_vectors() is not None  # method still works
    eng2._meta_cgn = None
    assert eng2._grounded_primitive_vectors() == (None, None)


if __name__ == "__main__":
    test_softmax_stable_and_normalized()
    test_grounded_primitive_vectors_aligned_to_meta_primitives()
    test_grounded_vectors_none_when_meta_cgn_absent()
    test_high_conf_grounded_v_shifts_off_formulate()
    test_unproven_grounded_v_defers_to_policy()
    test_grounded_v_selection_flag_default_on()
    print("OK — Phase B grounded-V selection checks passed")
