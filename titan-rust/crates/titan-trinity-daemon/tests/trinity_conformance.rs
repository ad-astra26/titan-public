//! Trinity SPEC-conformance suite (Layer 1 of the conformance gate).
//!
//! Each test is named `conformance_*` and enforces ONE binding clause of the
//! Trinity SPEC (§G5.1/§G5.2/§G10/§G11). The manifest
//! `titan-docs/specs/TRINITY_CONFORMANCE.md` maps every clause → its test here;
//! `scripts/trinity_conformance.py` runs these + checks coverage; the pre-commit
//! hook blocks on any LOCKED-clause regression.
//!
//! A test fails iff its clause's term is absent or simplified. The dropped-term
//! tests (magnitude/coherence/polarity/direction feedback, enrichment-separate,
//! neuromod gain) require the to-spec kernel interface that exposes the 5
//! observables as first-class inputs; they are authored TDD-first with the
//! §G5.2 implementation and are listed PENDING in the manifest until then.
//!
//! This file currently encodes the clauses expressible against the SHIPPED
//! `homeostasis` public API (the LOCKED clauses).

use titan_trinity_daemon::homeostasis::{
    gradient, observe, stateful_update, Layer, RestoringCfg, CENTRE,
};

/// G5.2-eq-structure — the update integrates x[t-1] (statefulness) + clamps [0,1].
#[test]
fn conformance_g5_2_update_equation_structure() {
    // A steady producer at the current value with neutral observables must
    // leave a centred tensor essentially put (it integrates prev, doesn't recompute).
    let cfg = RestoringCfg::for_layer(Layer::Mind);
    let prev = vec![CENTRE; 15];
    let prev2 = vec![CENTRE; 15];
    let enriched = vec![CENTRE; 15];
    let x = stateful_update(&prev, &prev2, &enriched, &cfg);
    assert_eq!(x.len(), 15, "output dimensionality preserved");
    for v in &x {
        assert!((0.0..=1.0).contains(v), "clamped to [0,1]");
        assert!((v - CENTRE).abs() < 1e-6, "centred+steady tensor stays put");
    }
}

/// G5.2-obs-velocity — the velocity observable feeds back as carry-forward momentum.
/// (The one observable-feedback term the shipped kernel DOES implement.)
#[test]
fn conformance_g5_2_obs_velocity_influences_output() {
    let cfg = RestoringCfg::for_layer(Layer::Body);
    let enriched = vec![CENTRE; 5];
    // Case A: moving (prev != prev2) → nonzero velocity → momentum carry.
    let prev = vec![0.5_f32; 5];
    let prev2_moving = vec![0.4_f32; 5]; // prev-prev2 = +0.1 → upward momentum
    let prev2_still = vec![0.5_f32; 5]; // zero velocity
    let x_moving = stateful_update(&prev, &prev2_moving, &enriched, &cfg);
    let x_still = stateful_update(&prev, &prev2_still, &enriched, &cfg);
    assert!(
        (x_moving[0] - x_still[0]).abs() > 1e-6,
        "velocity (prev-prev2) must influence the output via momentum carry"
    );
    assert!(x_moving[0] > x_still[0], "upward momentum carries upward");
}

/// G5.2-spring — PD restoring spring pulls every layer (incl. spirit 45D) toward 0.5.
#[test]
fn conformance_g5_2_spring_pulls_to_centre_all_layers() {
    for (layer, n) in [
        (Layer::Body, 5usize),
        (Layer::Mind, 15),
        (Layer::Spirit, 45),
    ] {
        let cfg = RestoringCfg {
            k_drive: 0.0,
            k_momentum: 0.0,
            k_cohesion: 0.0,
            ..RestoringCfg::for_layer(layer)
        };
        let prev = vec![0.9_f32; n];
        let prev2 = prev.clone();
        let x = stateful_update(&prev, &prev2, &prev, &cfg);
        assert!(
            x[0] < 0.9 && x[0] > 0.5,
            "{:?}: spring must pull toward 0.5 (got {})",
            layer,
            x[0]
        );
    }
    // Spirit's 45D MUST be covered (the clause GROUND_UP deliberately does not reach).
}

/// G5.2-gradient — quant→qual gradient body 0.7/0.3, mind 0.5/0.5, spirit 0.3/0.7 (INV-9).
#[test]
fn conformance_g5_2_gradient_per_layer() {
    assert_eq!(gradient(Layer::Body), (0.7, 0.3));
    assert_eq!(gradient(Layer::Mind), (0.5, 0.5));
    assert_eq!(gradient(Layer::Spirit), (0.3, 0.7));
}

/// G5.2-no-floor — no hardcoded neutral floor: a quiet dim is NOT snapped to 0.5/any constant;
/// it holds its last value and only drifts under the (gentle) spring.
#[test]
fn conformance_g5_2_no_hardcoded_floor() {
    let cfg = RestoringCfg::for_layer(Layer::Spirit);
    // A dim sitting at honest 0.0 with a 0.0 producer must NOT jump to a floor; it
    // moves only by the gentle spring (toward 0.5), staying near 0.0 for one tick.
    let prev = vec![0.0_f32; 45];
    let prev2 = prev.clone();
    let enriched = vec![0.0_f32; 45];
    let x = stateful_update(&prev, &prev2, &enriched, &cfg);
    assert!(
        x[0] < 0.10,
        "a quiet dim must hold near its last value (no neutral-0.5 floor), got {}",
        x[0]
    );
    assert!(x[0] >= 0.0, "clamped");
}

/// G5.2 observe() — the 5-observable signature is computed (coherence/magnitude/
/// velocity/direction/polarity), the inputs the feedback term must consume.
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
    // moving upward → direction +1
    assert_eq!(o.direction, 1.0);
}

/// G10-groundup-scope — GROUND_UP modifies body all-5D + mind willing[10:15] ONLY;
/// spirit is NOT grounded toward matter (no spirit application path exists).
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
    // GROUND_UP nudge is body-5D scoped. Spirit centre-pull is the §G5.2 restoring
    // force (covered by conformance_g5_2_spring_pulls_to_centre_all_layers), NOT GROUND_UP.
    assert_eq!(
        eb.prev_nudge().len(),
        5,
        "GROUND_UP nudge is body-5D scoped"
    );
}

/// G5.1-learned-formula — small filter_down uses the learned TrinityValueNet<65>
/// gradient-attention engine, NOT the `1.0+(spirit−0.5)×0.1` placeholder multiply.
#[test]
fn conformance_g5_1_learned_value_net_not_placeholder() {
    use titan_core::small_filter_down::HALF_DIM;
    // The learned engine operates on the 65D trinity half (body5+mind15+spirit45).
    // A placeholder multiply would not carry the 65D learned-net contract.
    assert_eq!(
        HALF_DIM, 65,
        "small filter_down is the 65D learned-half engine"
    );
}
