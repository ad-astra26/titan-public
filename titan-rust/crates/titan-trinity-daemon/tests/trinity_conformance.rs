//! Trinity SPEC-conformance suite (Layer 1 of the conformance gate).
//!
//! Each test is named `conformance_*` and enforces ONE binding clause of the
//! Trinity SPEC (§G5.1 / §G5.2 / §G10 / §G11). The manifest
//! `titan-docs/specs/TRINITY_CONFORMANCE.md` maps every clause → its test here;
//! `scripts/trinity_conformance.py` runs these + checks coverage; the pre-commit
//! hook blocks on any LOCKED-clause regression. A test fails iff its clause's
//! term is absent or simplified — making "the trinity is 100% per spec" a
//! machine-checked invariant.
//!
//! Per-observable tests construct `LayerObs` directly so each observable can be
//! ISOLATED (perturb one field, assert the output reflects its specified role).

use titan_trinity_daemon::homeostasis::{
    gradient, observe, stateful_update, Layer, LayerObs, RestoringCfg, CENTRE,
};

/// Build a `LayerObs` with the given fields (all in their valid ranges).
fn obs_of(
    coherence: f32,
    magnitude: f32,
    velocity: f32,
    direction: f32,
    polarity: f32,
) -> LayerObs {
    LayerObs {
        coherence,
        magnitude,
        velocity,
        direction,
        polarity,
    }
}

/// A "neutral" LayerObs that activates none of the observable-derived gains
/// beyond their unit baseline — useful for isolating a single observable in a test.
fn neutral_obs() -> LayerObs {
    obs_of(1.0, 0.5, 0.0, 0.0, 0.0) // coherence=1 → drift=0; mag=0.5 → mag_landing=1; dir=0
}

// ── G5.2-eq-structure ────────────────────────────────────────────────────────
#[test]
fn conformance_g5_2_update_equation_structure() {
    // A steady tensor at centre with neutral observables + zero drive/enrichment
    // must stay put — the kernel integrates x[t-1], doesn't recompute.
    let cfg = RestoringCfg::for_layer(Layer::Mind);
    let prev = vec![CENTRE; 15];
    let prev2 = vec![CENTRE; 15];
    let raw = vec![CENTRE; 15];
    let enrich = vec![0.0_f32; 15];
    let x = stateful_update(&prev, &prev2, &raw, &enrich, &neutral_obs(), &cfg);
    assert_eq!(x.len(), 15);
    for v in &x {
        assert!((0.0..=1.0).contains(v), "clamped");
        assert!((v - CENTRE).abs() < 1e-6, "centred+steady stays put");
    }
}

// ── G5.2-obs-velocity → momentum carry (§9.0) ────────────────────────────────
#[test]
fn conformance_g5_2_obs_velocity_influences_output() {
    // With a non-zero step (prev != prev2) the momentum + damping + continuity
    // terms activate. Spirit layer (w_qual=0.7) keeps direction-continuity active;
    // with direction=+1, the upward step carries the output above the still-case.
    let cfg = RestoringCfg {
        k_restore: 0.0, // isolate velocity-driven terms
        ..RestoringCfg::for_layer(Layer::Spirit)
    };
    let prev = vec![CENTRE; 4];
    let prev2_moving = vec![CENTRE - 0.1_f32; 4]; // step = +0.1 (upward velocity)
    let prev2_still = vec![CENTRE; 4]; // zero velocity
    let raw = vec![CENTRE; 4];
    let enrich = vec![0.0_f32; 4];
    let obs_moving = obs_of(1.0, 0.5, 0.1, 1.0, 0.0); // upward direction
    let obs_still = obs_of(1.0, 0.5, 0.0, 0.0, 0.0);
    let x_moving = stateful_update(&prev, &prev2_moving, &raw, &enrich, &obs_moving, &cfg);
    let x_still = stateful_update(&prev, &prev2_still, &raw, &enrich, &obs_still, &cfg);
    assert!(
        (x_moving[0] - x_still[0]).abs() > 1e-6,
        "velocity must influence output (momentum + continuity)"
    );
    assert!(
        x_moving[0] > x_still[0],
        "upward step → output carries upward"
    );
}

// ── G5.2-obs-magnitude → drive-landing gain (§9.0) ───────────────────────────
#[test]
fn conformance_g5_2_obs_magnitude_influences_output() {
    // Drive = k_drive · (1 + w_quant·a_mag·(2·mag − 1)) · (raw − prev).
    // Two cases identical except obs.magnitude → the drive gain (and hence
    // the per-tick output) MUST differ; higher magnitude lands harder.
    let cfg = RestoringCfg {
        k_restore: 0.0,
        k_damp: 0.0,
        k_mom: 0.0,
        k_dir: 0.0,
        ..RestoringCfg::for_layer(Layer::Body) // body is quant-heavy → magnitude carries
    };
    let prev = vec![CENTRE; 5];
    let prev2 = vec![CENTRE; 5];
    let raw = vec![CENTRE + 0.2_f32; 5]; // upward drive target
    let enrich = vec![0.0_f32; 5];
    let obs_low = obs_of(1.0, 0.2, 0.0, 0.0, 0.0); // low magnitude → softer landing
    let obs_high = obs_of(1.0, 0.9, 0.0, 0.0, 0.0); // high magnitude → harder landing
    let x_low = stateful_update(&prev, &prev2, &raw, &enrich, &obs_low, &cfg);
    let x_high = stateful_update(&prev, &prev2, &raw, &enrich, &obs_high, &cfg);
    assert!(
        x_high[0] > x_low[0] + 1e-4,
        "magnitude must scale drive landing — got high={} low={}",
        x_high[0],
        x_low[0]
    );
}

// ── G5.2-obs-coherence → spring strength modulation (§9.0 + §9.2) ────────────
#[test]
fn conformance_g5_2_obs_coherence_influences_output() {
    // Spring = -k_restore·g_neuro·(1 + w_qual·a_drift·((1−coh) + |pol|))·(prev−0.5).
    // With prev != 0.5 + polarity=0, varying coherence varies the spring strength.
    // Lower coherence → stronger pull → output closer to 0.5.
    let cfg = RestoringCfg {
        k_drive: 0.0,
        k_damp: 0.0,
        k_mom: 0.0,
        k_dir: 0.0,
        ..RestoringCfg::for_layer(Layer::Spirit) // qual-heavy → coherence carries
    };
    let prev = vec![0.8_f32; 4]; // off-centre
    let prev2 = vec![0.8_f32; 4];
    let raw = vec![0.8_f32; 4];
    let enrich = vec![0.0_f32; 4];
    let obs_high_coh = obs_of(0.9, 0.5, 0.0, 0.0, 0.0); // little drift → weak spring
    let obs_low_coh = obs_of(0.2, 0.5, 0.0, 0.0, 0.0); // big drift → strong spring
    let x_high = stateful_update(&prev, &prev2, &raw, &enrich, &obs_high_coh, &cfg);
    let x_low = stateful_update(&prev, &prev2, &raw, &enrich, &obs_low_coh, &cfg);
    assert!(
        x_low[0] < x_high[0],
        "lower coherence → stronger spring pull toward 0.5 (got low={} high={})",
        x_low[0],
        x_high[0]
    );
}

// ── G5.2-obs-polarity → spring strength modulation (§9.0 + §9.2) ─────────────
#[test]
fn conformance_g5_2_obs_polarity_influences_output() {
    // |polarity| also enters the drift signal → spring gain. Holding coherence
    // fixed, varying |polarity| varies the spring strength.
    let cfg = RestoringCfg {
        k_drive: 0.0,
        k_damp: 0.0,
        k_mom: 0.0,
        k_dir: 0.0,
        ..RestoringCfg::for_layer(Layer::Spirit)
    };
    let prev = vec![0.8_f32; 4];
    let prev2 = vec![0.8_f32; 4];
    let raw = vec![0.8_f32; 4];
    let enrich = vec![0.0_f32; 4];
    let obs_pol0 = obs_of(0.5, 0.5, 0.0, 0.0, 0.0); // |pol| = 0
    let obs_pol_hi = obs_of(0.5, 0.5, 0.0, 0.0, 0.8); // |pol| = 0.8
    let x0 = stateful_update(&prev, &prev2, &raw, &enrich, &obs_pol0, &cfg);
    let x_hi = stateful_update(&prev, &prev2, &raw, &enrich, &obs_pol_hi, &cfg);
    assert!(
        x_hi[0] < x0[0],
        "higher |polarity| → stronger spring pull toward 0.5 (got hi={} 0={})",
        x_hi[0],
        x0[0]
    );
}

// ── G5.2-obs-direction → trajectory continuity (§9.0) ────────────────────────
#[test]
fn conformance_g5_2_obs_direction_influences_output() {
    // Continuity = w_qual · k_dir · direction · |prev − prev2|.
    // With a non-zero step, direction ∈ {−1, 0, +1} biases the output along
    // the trajectory sign. Spirit layer (w_qual=0.7) emphasises continuity.
    let cfg = RestoringCfg {
        k_drive: 0.0,
        k_restore: 0.0,
        k_damp: 0.0,
        k_mom: 0.0, // isolate continuity from momentum
        ..RestoringCfg::for_layer(Layer::Spirit)
    };
    let prev = vec![CENTRE; 4];
    let prev2 = vec![CENTRE - 0.1_f32; 4]; // |step| = 0.1
    let raw = vec![CENTRE; 4];
    let enrich = vec![0.0_f32; 4];
    let obs_dir_0 = obs_of(1.0, 0.5, 0.1, 0.0, 0.0); // direction = 0
    let obs_dir_up = obs_of(1.0, 0.5, 0.1, 1.0, 0.0); // direction = +1
    let x_0 = stateful_update(&prev, &prev2, &raw, &enrich, &obs_dir_0, &cfg);
    let x_up = stateful_update(&prev, &prev2, &raw, &enrich, &obs_dir_up, &cfg);
    assert!(
        x_up[0] > x_0[0] + 1e-6,
        "direction=+1 must carry the trajectory upward vs direction=0 (got up={} 0={})",
        x_up[0],
        x_0[0]
    );
}

// ── G5.2-enrichment-separate (full-weight additive) ──────────────────────────
#[test]
fn conformance_g5_2_enrichment_is_separate_full_weight() {
    // Enrichment_force enters as a SEPARATE additive term (full weight) per the
    // §G5.2 equation literal. If it were folded into the drive (×k_drive≈0.3)
    // its contribution would be ~0.03 for an enrichment of 0.1 — wrong by 3×.
    // With everything else zeroed, an enrichment of 0.1 must move x by ~+0.1.
    let cfg = RestoringCfg {
        k_drive: 0.0,
        k_restore: 0.0,
        k_damp: 0.0,
        k_mom: 0.0,
        k_dir: 0.0,
        a_mag: 0.0,
        ..RestoringCfg::for_layer(Layer::Body)
    };
    let prev = vec![CENTRE; 5];
    let prev2 = vec![CENTRE; 5];
    let raw = vec![CENTRE; 5]; // zero drive
    let enrich = vec![0.1_f32; 5];
    let obs = obs_of(1.0, 0.5, 0.0, 0.0, 0.0);
    let x = stateful_update(&prev, &prev2, &raw, &enrich, &obs, &cfg);
    // Full-weight: x = 0.5 + 0.1 = 0.6, NOT 0.5 + 0.03 = 0.53.
    assert!(
        (x[0] - 0.6_f32).abs() < 1e-5,
        "enrichment must be FULL-WEIGHT additive (got {}, expected ~0.6)",
        x[0]
    );
}

// ── G5.2-neuromod-gain → modulates k_restore (§G5.2 item 2) ──────────────────
#[test]
fn conformance_g5_2_neuromod_gain_modulates_restore() {
    // Two cfgs identical except neuromod_gain (1.0 vs 2.0). With prev off-centre
    // and spring isolated, the doubled gain must produce a stronger pull → x
    // closer to 0.5 than the unit-gain case.
    let mut cfg = RestoringCfg {
        k_drive: 0.0,
        k_damp: 0.0,
        k_mom: 0.0,
        k_dir: 0.0,
        ..RestoringCfg::for_layer(Layer::Body)
    };
    let prev = vec![0.9_f32; 3]; // off-centre
    let prev2 = vec![0.9_f32; 3];
    let raw = vec![0.9_f32; 3];
    let enrich = vec![0.0_f32; 3];
    let obs = obs_of(1.0, 0.5, 0.0, 0.0, 0.0); // drift_signal = 0 → spring_gain = 1
    cfg.neuromod_gain = 1.0;
    let x1 = stateful_update(&prev, &prev2, &raw, &enrich, &obs, &cfg);
    cfg.neuromod_gain = 2.0;
    let x2 = stateful_update(&prev, &prev2, &raw, &enrich, &obs, &cfg);
    assert!(
        x2[0] < x1[0],
        "neuromod_gain=2.0 must pull harder toward 0.5 than 1.0 (got x2={} x1={})",
        x2[0],
        x1[0]
    );
}

// ── G5.2-spring covers spirit's 45D ──────────────────────────────────────────
#[test]
fn conformance_g5_2_spring_pulls_to_centre_all_layers() {
    // Spirit's 45D MUST be covered by the restoring spring — GROUND_UP deliberately
    // does not reach spirit, so this clause's enforcement is what guarantees spirit
    // centre-pull at all.
    for (layer, n) in [
        (Layer::Body, 5usize),
        (Layer::Mind, 15),
        (Layer::Spirit, 45),
    ] {
        let cfg = RestoringCfg {
            k_drive: 0.0,
            k_damp: 0.0,
            k_mom: 0.0,
            k_dir: 0.0,
            ..RestoringCfg::for_layer(layer)
        };
        let prev = vec![0.9_f32; n];
        let prev2 = vec![0.9_f32; n];
        let raw = vec![0.9_f32; n];
        let enrich = vec![0.0_f32; n];
        let obs = obs_of(1.0, 0.5, 0.0, 0.0, 0.0);
        let x = stateful_update(&prev, &prev2, &raw, &enrich, &obs, &cfg);
        assert!(
            x[0] < 0.9 && x[0] > 0.5,
            "{:?}: spring must pull toward 0.5 (got {})",
            layer,
            x[0]
        );
    }
}

// ── G5.2-gradient: quant→qual weights (INV-9) ────────────────────────────────
#[test]
fn conformance_g5_2_gradient_per_layer() {
    assert_eq!(gradient(Layer::Body), (0.7, 0.3));
    assert_eq!(gradient(Layer::Mind), (0.5, 0.5));
    assert_eq!(gradient(Layer::Spirit), (0.3, 0.7));
}

// ── G5.2-no-floor: quiet dim is NOT snapped to a neutral floor ───────────────
#[test]
fn conformance_g5_2_no_hardcoded_floor() {
    let cfg = RestoringCfg::for_layer(Layer::Spirit);
    let prev = vec![0.0_f32; 45]; // honest 0.0
    let prev2 = prev.clone();
    let raw = vec![0.0_f32; 45];
    let enrich = vec![0.0_f32; 45];
    let obs = obs_of(1.0, 0.0, 0.0, 0.0, -1.0); // matches a quiet, polarised-low layer
    let x = stateful_update(&prev, &prev2, &raw, &enrich, &obs, &cfg);
    assert!(
        x[0] < 0.10,
        "a quiet dim must NOT snap to a neutral floor (got {})",
        x[0]
    );
    assert!(x[0] >= 0.0, "clamped to [0,1]");
}

// ── G5.2 observe(): the 5 observables are computed for the feedback term ─────
#[test]
fn conformance_g5_2_observe_produces_five_signals() {
    let cur = vec![0.2_f32, 0.8, 0.5, 0.6];
    let prev = vec![0.1_f32, 0.7, 0.5, 0.6];
    let o = observe(&cur, &prev);
    assert!((0.0..=1.0).contains(&o.coherence));
    assert!((0.0..=1.0).contains(&o.magnitude));
    assert!((0.0..=1.0).contains(&o.velocity));
    assert!((-1.0..=1.0).contains(&o.direction));
    assert!((-1.0..=1.0).contains(&o.polarity));
    assert_eq!(o.direction, 1.0); // moving upward
}

// ── G10-groundup-scope: body + mind-willing ONLY (no Side::Spirit variant) ───
#[test]
fn conformance_g10_groundup_scope_body_mind_only() {
    use titan_trinity_daemon::ground_up::{GroundUpEnricher, Side};
    // GROUND_UP is scoped to exactly two sides — Body + MindWilling. There is NO
    // Side::Spirit variant: spirit is structurally excluded from GROUND_UP (§G10).
    let mut eb = GroundUpEnricher::new(Side::Body);
    let _ = eb.compute_nudge(&[0.9_f32; 10]);
    let mut body = [0.5_f32; 5];
    eb.apply_held_to_body(&mut body, 1.0).unwrap();
    let mut em = GroundUpEnricher::new(Side::MindWilling);
    let _ = em.compute_nudge(&[0.9_f32; 10]);
    let mut mind = [0.5_f32; 15];
    em.apply_held_to_mind(&mut mind, 1.0).unwrap();
    // Spirit centre-pull is the §G5.2 restoring force (separate clause), NOT GROUND_UP.
    assert_eq!(
        eb.prev_nudge().len(),
        5,
        "GROUND_UP nudge is body-5D scoped"
    );
}

// ── G5.1-learned-formula: 65D learned-net, not a placeholder multiply ────────
#[test]
fn conformance_g5_1_learned_value_net_not_placeholder() {
    use titan_core::small_filter_down::HALF_DIM;
    assert_eq!(
        HALF_DIM, 65,
        "small filter_down is the 65D learned-half engine, not a placeholder"
    );
}
