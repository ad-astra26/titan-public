"""§6 regression — β-dispersion EMA sourced from cross-primitive V spread.

RFP_meta-reasoning_CGN_FIX.md §6 option (b) fix: shadow_mode deadlock — rerank-based
β-dispersion EMA at meta_cgn.py:2514-2519 requires β authority that shadow_mode denies
(w_grounded=0 → composed_V dispersion=0 → EMA=0 → E3 threshold never crossed). New
fix: also update _beta_dispersion_ema inside update_primitive_V() from max-min of V
across the 9 primitives — non-zero even in shadow_mode (T1 live 2026-05-16: 0.036).

Run isolated:
    python -m pytest tests/test_beta_dispersion_cross_primitive.py -v -p no:anchorpy --tb=short
"""
import tempfile


def test_beta_dispersion_ema_starts_at_zero():
    """Fresh MetaCGNConsumer must have β-dispersion EMA == 0."""
    from titan_hcl.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp, titan_id="T1")
        assert c._beta_dispersion_ema == 0.0


def test_beta_dispersion_ema_moves_off_zero_under_v_spread():
    """Differentiated update_primitive_V calls must push EMA off 0.

    Calling update_primitive_V with HIGH quality for FORMULATE and LOW quality
    for INTROSPECT differentiates their α/β posteriors → V spread grows →
    cross-primitive max(V)-min(V) is non-zero → EMA moves off 0.
    """
    from titan_hcl.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp, titan_id="T1")
        assert c._beta_dispersion_ema == 0.0
        # Differentiate primitives: FORMULATE biased high, INTROSPECT biased low
        for i in range(50):
            c.update_primitive_V("FORMULATE", quality=0.9, chain_id=i)
            c.update_primitive_V("INTROSPECT", quality=0.1, chain_id=i)
        # After ≥50 differentiated updates, V spread > 0 → EMA moved off 0
        v_vals = [p.V for p in c._primitives.values()]
        spread = max(v_vals) - min(v_vals)
        assert spread > 0.05, f"expected V spread > 0.05, got {spread:.4f}"
        assert c._beta_dispersion_ema > 0.0, (
            f"expected β-dispersion EMA > 0, got {c._beta_dispersion_ema:.4f}")


def test_beta_dispersion_ema_survives_shadow_mode_status():
    """Cross-primitive V-spread path must update EMA regardless of MetaCGN
    status (the whole point of §6 — bypass the shadow_mode deadlock).
    """
    from titan_hcl.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp, titan_id="T1")
        # Force shadow_mode (the deadlock state)
        c._status = "shadow_mode"
        ema_before = c._beta_dispersion_ema
        for i in range(30):
            c.update_primitive_V("FORMULATE", quality=0.85, chain_id=i)
            c.update_primitive_V("INTROSPECT", quality=0.15, chain_id=i)
        assert c._beta_dispersion_ema > ema_before, (
            f"shadow_mode must NOT block §6 cross-primitive EMA update; "
            f"got {c._beta_dispersion_ema:.4f} ≤ {ema_before:.4f}")


def test_beta_dispersion_ema_stable_under_uniform_quality():
    """If every primitive gets the same quality, V spread stays ~0 and EMA
    decays toward 0 (sanity — the signal must be discriminating)."""
    from titan_hcl.logic.meta_cgn import MetaCGNConsumer
    with tempfile.TemporaryDirectory() as tmp:
        c = MetaCGNConsumer(send_queue=None, save_dir=tmp, titan_id="T1")
        # Seed EMA at some non-zero starting value, then verify uniform updates
        # decay it (EMA = (1-α)·ema + α·spread; uniform quality → spread→0).
        c._beta_dispersion_ema = 0.05
        for i in range(200):
            for p in list(c._primitives.keys()):
                c.update_primitive_V(p, quality=0.5, chain_id=i)
        # After 200 rounds, spread asymptotes near 0 → EMA decays
        v_vals = [p.V for p in c._primitives.values()]
        spread = max(v_vals) - min(v_vals)
        assert spread < 0.02, f"expected spread → 0 under uniform quality, got {spread:.4f}"
        # EMA decayed toward 0 (started at 0.05, α=0.01, 200 steps would bring
        # close to spread which is ~0); allow generous tolerance
        assert c._beta_dispersion_ema < 0.05, (
            f"EMA must decay under uniform quality; got {c._beta_dispersion_ema:.4f}")
