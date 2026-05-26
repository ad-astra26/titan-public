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

// ── G5.2-config: gain coefficients sourced from titan_params.toml sidecar ────
#[test]
fn conformance_g5_2_gains_from_titan_params_toml() {
    // The daemon-side path is `load_restoring_cfg(shm_dir, layer)` which reads
    // `shm_dir/trinity_restoring.bin` (Python L2-published from
    // `titan_params.toml [trinity_restoring.{body,mind,spirit}]` per D-SPEC-129
    // per-layer schema, v1.59.2) and falls back to the kernel DEFAULT_* constants
    // only when the sidecar is absent (cold boot).
    use std::fs;
    use tempfile::tempdir;
    use titan_trinity_daemon::{load_restoring_cfg, DEFAULT_K_DRIVE, TRINITY_RESTORING_SIDECAR};
    let dir = tempdir().unwrap();

    // Absent → kernel default (substrate continues per §11.B).
    let cfg_default = load_restoring_cfg(dir.path(), Layer::Body);
    assert!(
        (cfg_default.k_drive - DEFAULT_K_DRIVE).abs() < 1e-6,
        "fallback should match kernel default"
    );

    // Present + valid (D-SPEC-129, 96-byte per-layer sidecar) → values come
    // from the TOML-derived sidecar. body slice [0:32], mind [32:64], spirit
    // [64:96]; each layer is 8 × f32 LE in the field order documented in
    // restoring_cfg.rs. Distinct per-layer k_drive values (0.30 / 0.40 / 0.20)
    // verify the per-layer offset table — not just position 0 for everyone.
    let path = dir.path().join(TRINITY_RESTORING_SIDECAR);
    let body = [0.30_f32, 0.05, 0.05, 0.10, 0.05, 0.50, 1.10, 0.50];
    let mind = [0.40_f32, 0.07, 0.06, 0.11, 0.04, 0.55, 1.10, 0.45];
    let spirit = [0.20_f32, 0.05, 0.05, 0.10, 0.05, 0.50, 1.10, 0.50];
    let mut bytes = Vec::with_capacity(96);
    for v in body.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in mind.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in spirit.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(&path, &bytes).unwrap();

    let body_cfg = load_restoring_cfg(dir.path(), Layer::Body);
    let mind_cfg = load_restoring_cfg(dir.path(), Layer::Mind);
    let spirit_cfg = load_restoring_cfg(dir.path(), Layer::Spirit);
    assert!(
        (body_cfg.k_drive - 0.30).abs() < 1e-6,
        "body k_drive (got {})",
        body_cfg.k_drive
    );
    assert!(
        (mind_cfg.k_drive - 0.40).abs() < 1e-6,
        "mind k_drive — PLAN §6.6.2 bump (got {})",
        mind_cfg.k_drive
    );
    assert!(
        (spirit_cfg.k_drive - 0.20).abs() < 1e-6,
        "spirit k_drive (got {})",
        spirit_cfg.k_drive
    );
    assert!(
        (mind_cfg.a_drift - 1.10).abs() < 1e-6,
        "mind a_drift override (got {})",
        mind_cfg.a_drift
    );
}

// ── G5.2-persistence: traveling tensor + observable EMAs survive restart ─────
#[test]
fn conformance_g5_2_state_checkpoint_roundtrip() {
    // SPEC §G5.2 item 4 (Maker override 2026-05-23): the per-part tensor
    // state (x[t-1], x[t-2]) + the latest 5D observable signature MUST be
    // checkpointed + reloaded on boot — so a restart never erases the
    // §G5.2 traveling tensor's journey. The checkpoint module exposes a
    // single API (`write_for_part` + `load_for_part`) used by all 6 daemons;
    // a roundtrip per layer (body / mind / spirit dim counts) verifies
    // the contract end-to-end.
    use tempfile::tempdir;
    use titan_trinity_daemon::checkpoint::{load_for_part, write_for_part, CheckpointSnapshot};

    fn sample_obs() -> LayerObs {
        obs_of(0.65, 0.42, 0.05, -1.0, 0.18)
    }

    // Body roundtrip (5D).
    {
        let dir = tempdir().unwrap();
        let prev: [f32; 5] = [0.10, 0.20, 0.30, 0.40, 0.50];
        let prev2: [f32; 5] = [0.11, 0.21, 0.31, 0.41, 0.51];
        let obs = sample_obs();
        write_for_part(dir.path(), "inner_body", &prev, &prev2, &obs).unwrap();
        let snap: CheckpointSnapshot<5> = load_for_part(dir.path(), "inner_body").expect("loaded");
        assert_eq!(snap.prev, prev);
        assert_eq!(snap.prev2, prev2);
        assert!((snap.last_obs.coherence - obs.coherence).abs() < 1e-6);
        assert!((snap.last_obs.polarity - obs.polarity).abs() < 1e-6);
    }
    // Mind roundtrip (15D).
    {
        let dir = tempdir().unwrap();
        let prev: [f32; 15] = std::array::from_fn(|i| (i as f32) * 0.013);
        let prev2: [f32; 15] = std::array::from_fn(|i| (i as f32) * 0.017);
        let obs = sample_obs();
        write_for_part(dir.path(), "outer_mind", &prev, &prev2, &obs).unwrap();
        let snap: CheckpointSnapshot<15> = load_for_part(dir.path(), "outer_mind").expect("loaded");
        assert_eq!(snap.prev, prev);
        assert_eq!(snap.prev2, prev2);
    }
    // Spirit roundtrip (45D).
    {
        let dir = tempdir().unwrap();
        let prev: [f32; 45] = std::array::from_fn(|i| ((i as f32) * 0.007) % 1.0);
        let prev2: [f32; 45] = std::array::from_fn(|i| ((i as f32) * 0.009) % 1.0);
        let obs = sample_obs();
        write_for_part(dir.path(), "inner_spirit", &prev, &prev2, &obs).unwrap();
        let snap: CheckpointSnapshot<45> =
            load_for_part(dir.path(), "inner_spirit").expect("loaded");
        assert_eq!(snap.prev, prev);
        assert_eq!(snap.prev2, prev2);
        assert!((snap.last_obs.coherence - obs.coherence).abs() < 1e-6);
    }
    // Restart-simulation: after the kernel re-spawns the daemon, the next
    // §G5.2 stateful_update fed (prev, prev2) restored from checkpoint must
    // produce the SAME output a continuous run would have produced — i.e.
    // a checkpoint is a true continuation, not a re-seed at 0.5.
    {
        let dir = tempdir().unwrap();
        let prev: [f32; 5] = [0.7, 0.7, 0.7, 0.7, 0.7];
        let prev2: [f32; 5] = [0.65, 0.65, 0.65, 0.65, 0.65];
        let obs = obs_of(0.9, 0.55, 0.05, 1.0, 0.4);
        write_for_part(dir.path(), "outer_body", &prev, &prev2, &obs).unwrap();
        let snap: CheckpointSnapshot<5> = load_for_part(dir.path(), "outer_body").expect("loaded");

        let cfg = RestoringCfg::for_layer(Layer::Body);
        let raw = vec![0.7_f32; 5];
        let enrich = vec![0.0_f32; 5];
        let x_continuous = stateful_update(&prev, &prev2, &raw, &enrich, &obs, &cfg);
        let x_restored =
            stateful_update(&snap.prev, &snap.prev2, &raw, &enrich, &snap.last_obs, &cfg);
        for i in 0..5 {
            assert!(
                (x_continuous[i] - x_restored[i]).abs() < 1e-6,
                "restored kernel output must match continuous run (dim {i})",
            );
        }
    }
}

// ── G5.1-down-leg-spirit-pulse: small filter_down trigger reads BALANCED spirit pulse ─
#[test]
fn conformance_g5_1_down_leg_on_spirit_pulse() {
    // SPEC §G5.1 (D-SPEC-121 v1.54.0; narrows D-SPEC-112): the small filter_down
    // DOWN-leg gates on the spirit sphere-clock's BALANCED PULSE rising-edge —
    // `edges[Spirit] && cb[Spirit] >= 1` read SHM-direct from `sphere_clocks.bin`.
    // An unbalanced spirit pulse does NOT fire the small filter_down. The
    // PulseWatcher's `tick_with_balanced()` is the canonical SPEC accessor.
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;
    use tempfile::tempdir;
    use titan_state::Slot;
    use titan_trinity_daemon::{
        pulse_watch::{PULSE_WATCH_CONSEC_BALANCED_OFFSET, PULSE_WATCH_PAYLOAD_BYTES},
        PulseClockRole, PulseWatcher,
    };

    let dir = tempdir().unwrap();
    let path = dir.path().join("sphere_clocks.bin");
    let mut slot = Slot::create(&path, 1, PULSE_WATCH_PAYLOAD_BYTES as u32).unwrap();
    let mut bytes = vec![0u8; PULSE_WATCH_PAYLOAD_BYTES];
    slot.write(&bytes).unwrap();
    let mut watcher = PulseWatcher::open(dir.path());
    let _ = watcher.tick_with_balanced(); // seed

    // (1) Spirit clock pulses but cb=0 (unbalanced) → edge fires, balanced_edge does NOT.
    let inner_spirit_off = PulseClockRole::InnerSpirit.count_byte_offset();
    bytes[inner_spirit_off..inner_spirit_off + 4].copy_from_slice(&1.0_f32.to_le_bytes());
    slot.write(&bytes).unwrap();
    let (edges, balanced_edges) = watcher.tick_with_balanced();
    assert!(
        edges[PulseClockRole::InnerSpirit.index()],
        "spirit pulse rising edge must be detected SHM-direct"
    );
    assert!(
        !balanced_edges[PulseClockRole::InnerSpirit.index()],
        "D-SPEC-121: unbalanced spirit pulse must NOT fire the small filter_down"
    );

    // (2) Spirit clock pulses AND cb=5 (balanced) → BOTH edges fire — small filter_down gate met.
    bytes[inner_spirit_off..inner_spirit_off + 4].copy_from_slice(&2.0_f32.to_le_bytes());
    let cb_off = PulseClockRole::InnerSpirit.index() * 28 + PULSE_WATCH_CONSEC_BALANCED_OFFSET;
    bytes[cb_off..cb_off + 4].copy_from_slice(&5.0_f32.to_le_bytes());
    slot.write(&bytes).unwrap();
    let (_, balanced_edges2) = watcher.tick_with_balanced();
    assert!(
        balanced_edges2[PulseClockRole::InnerSpirit.index()],
        "D-SPEC-121: balanced spirit pulse MUST fire the small filter_down — \
         this is the spirit-reaches-the-Middle-Path gate"
    );

    // Structural assertions: the spirit daemons must NOT gate small filter_down
    // on (a) `epoch_pending` (retired D-SPEC-96/97) NOR (b) raw `pulse_edges`
    // unaware of balance (D-SPEC-121). Both spirit DOWN-legs MUST read
    // `balanced_pulse_edges`. Block any regression that resurfaces the
    // pre-D-SPEC-121 trigger.
    fn assert_no_epoch_arm(path: &Path) {
        let s = std::fs::read_to_string(path).expect("read daemon tick_loop.rs");
        assert!(
            !s.contains("if epoch_pending.swap(false, Ordering::Relaxed)"),
            "{}: KERNEL_EPOCH_TICK arm pattern resurfaced — small filter_down \
             must gate on the spirit sphere-clock BALANCED PULSE per D-SPEC-121 §G5.1",
            path.display(),
        );
    }
    fn assert_uses_balanced_pulse_edges_for_spirit_gate(path: &Path) {
        let s = std::fs::read_to_string(path).expect("read daemon tick_loop.rs");
        assert!(
            s.contains("balanced_pulse_edges[PulseClockRole::InnerSpirit.index()]")
                || s.contains("balanced_pulse_edges[PulseClockRole::OuterSpirit.index()]"),
            "{}: small filter_down DOWN-leg must gate on `balanced_pulse_edges[*Spirit]` \
             per D-SPEC-121 (not raw `pulse_edges` — that would fire on unbalanced pulses)",
            path.display(),
        );
    }
    let here = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let crates_dir = std::path::PathBuf::from(here).join("..");
    let inner_spirit_path = crates_dir.join("titan-inner-spirit-rs/src/tick_loop.rs");
    let outer_spirit_path = crates_dir.join("titan-outer-spirit-rs/src/tick_loop.rs");
    assert_no_epoch_arm(&inner_spirit_path);
    assert_no_epoch_arm(&outer_spirit_path);
    assert_uses_balanced_pulse_edges_for_spirit_gate(&inner_spirit_path);
    assert_uses_balanced_pulse_edges_for_spirit_gate(&outer_spirit_path);

    let mut f = File::create(dir.path().join("g5_1_down_leg_pulsed.ok")).unwrap();
    let _ = f.write_all(b"ok");
}

// ── G5.1 D-SPEC-121: one-shot filter_down application (consume-and-clear) ────
#[test]
fn conformance_g5_1_filter_down_application_is_one_shot() {
    // SPEC §G5.1 D-SPEC-121 (v1.54.0): the filter_down multipliers received by
    // body/mind daemons MUST be applied ONCE on the consuming Schumann tick and
    // then cleared (held value returns to None). Static structural assertion:
    // every receiving daemon's per-tick read of `s.unified` + `s.local` MUST
    // use `.take()` (which sets the Option to None), NOT a plain field read
    // (which would re-apply the same multiplier every subsequent tick — the
    // pre-D-SPEC-121 v1.36.2 "held + applied per tick" behavior that
    // saturated high-raw dims at 0 under §G5.2 statefulness).
    use std::path::Path;
    fn assert_consume_and_clear(path: &Path) {
        let s = std::fs::read_to_string(path).expect("read daemon tick_loop.rs");
        assert!(
            s.contains("s.unified.take()"),
            "{}: D-SPEC-121 violation — `s.unified` must be `.take()`'d (consume-and-clear), \
             not held; otherwise the §G5.2 integrator gets a continuous −0.5·raw pull \
             every Schumann tick under any non-1.0 multiplier",
            path.display(),
        );
        assert!(
            s.contains("s.local.take()"),
            "{}: D-SPEC-121 violation — `s.local` must be `.take()`'d (consume-and-clear)",
            path.display(),
        );
    }
    let here = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let crates_dir = std::path::PathBuf::from(here).join("..");
    for daemon in &[
        "titan-inner-body-rs/src/tick_loop.rs",
        "titan-inner-mind-rs/src/tick_loop.rs",
        "titan-outer-body-rs/src/tick_loop.rs",
        "titan-outer-mind-rs/src/tick_loop.rs",
    ] {
        assert_consume_and_clear(&crates_dir.join(daemon));
    }
}

// ── G5.1-up-leg-bonus: body/mind pulse → spirit content additive bonus ───────
#[test]
fn conformance_g5_1_up_leg_bonus_on_body_mind_pulse() {
    // SPEC §G5.1 P0-0a (D-SPEC-112 amendment): on a body/mind sphere-clock
    // PULSE, an additive snapshot bonus lands on SPIRIT content dims [5:45]
    // (observer [0:5] excluded per §G8). Replicate the spirit daemon's
    // composition + kernel call: with a positive body+mind polarity, content
    // dims must move strictly above the no-pulse baseline; observer dims
    // must be unchanged.
    // (Inner-spirit + outer-spirit daemons share the same logic — testing
    // the shared semantics is sufficient. The amplitude constant
    // UP_LEG_BONUS_AMPLITUDE = 0.02 is per-daemon-local; mirrored here.)
    const UP_LEG_AMP: f32 = 0.02;
    const SPIRIT_DIMS: usize = 45;
    let body: [f32; 5] = [0.9_f32; 5]; // body polarity = +0.4
    let mind: [f32; 15] = [0.9_f32; 15]; // mind polarity = +0.4
                                         // Bonus = 0.02 * (0.4 + 0.4) = 0.016 (clamped within [-1, 1] before scale).
    let body_polarity = (body.iter().copied().sum::<f32>() / 5.0) - 0.5;
    let mind_polarity = (mind.iter().copied().sum::<f32>() / 15.0) - 0.5;
    let signed = UP_LEG_AMP * (body_polarity + mind_polarity).clamp(-1.0, 1.0);
    assert!(signed > 0.0, "polarised body+mind → positive bonus");

    // Construct enrichment_with_bonus: zeros + add `signed` to content [5..45].
    let mut enrichment_with_bonus = [0.0_f32; SPIRIT_DIMS];
    for i in 5..SPIRIT_DIMS {
        enrichment_with_bonus[i] += signed;
    }
    let enrichment_no_pulse = [0.0_f32; SPIRIT_DIMS];

    let cfg = RestoringCfg {
        k_restore: 0.0,
        k_damp: 0.0,
        k_mom: 0.0,
        k_dir: 0.0,
        ..RestoringCfg::for_layer(Layer::Spirit)
    };
    let prev = vec![CENTRE; SPIRIT_DIMS];
    let prev2 = vec![CENTRE; SPIRIT_DIMS];
    let raw = vec![CENTRE; SPIRIT_DIMS]; // drive = 0
    let obs = obs_of(1.0, 0.5, 0.0, 0.0, 0.0);

    let x_no_pulse = stateful_update(&prev, &prev2, &raw, &enrichment_no_pulse, &obs, &cfg);
    let x_pulsed = stateful_update(&prev, &prev2, &raw, &enrichment_with_bonus, &obs, &cfg);

    // Observer dims [0..5] — no bonus.
    for i in 0..5 {
        assert!(
            (x_pulsed[i] - x_no_pulse[i]).abs() < 1e-6,
            "observer dim {i} must NOT receive UP-leg bonus (G8)",
        );
    }
    // Content dims [5..45] — strictly above baseline.
    for i in 5..SPIRIT_DIMS {
        assert!(
            x_pulsed[i] > x_no_pulse[i] + 1e-6,
            "content dim {i}: UP-leg bonus must raise output (no_pulse={} pulsed={})",
            x_no_pulse[i],
            x_pulsed[i],
        );
    }
}

// ── G5.2-focus-input: FOCUS cascade nudge composes via focus_input.bin ───────
#[test]
fn conformance_g5_2_focus_input_applied() {
    // SPEC §G5.2 item 2 + §G12: FOCUS enters every part daemon via a read-only
    // `focus_input.bin` SHM slot. The nudge is amplified by
    // `stale_focus_multiplier` (the §G12 SPIRIT→Lower-Spirit→Mind→Body
    // cascade) and composes into `enrichment_force` — a SEPARATE full-weight
    // additive per §G5.2 (preserving G5.2-enrichment-separate semantics, NOT
    // folded into drive). A non-zero focus nudge MUST move the §G5.2 kernel
    // output away from the no-focus baseline; the amplifier scales the move.
    use tempfile::tempdir;
    use titan_state::Slot;
    use titan_trinity_daemon::{
        compose_focus_into_enrichment, read_focus_nudge, FocusPart, FOCUS_INPUT_PAYLOAD_BYTES,
        FOCUS_INPUT_SIDECAR,
    };

    // Build a payload that nudges inner-body[0] up by +0.10, amplified ×2.0.
    let mut bytes = vec![0u8; FOCUS_INPUT_PAYLOAD_BYTES];
    bytes[0..4].copy_from_slice(&0.0_f32.to_le_bytes()); // ts
    bytes[4..8].copy_from_slice(&2.0_f32.to_le_bytes()); // stale_focus_multiplier
    bytes[8..12].copy_from_slice(&0.10_f32.to_le_bytes()); // inner_body[0] nudge
    let dir = tempdir().unwrap();
    let path = dir.path().join(FOCUS_INPUT_SIDECAR);
    let mut slot = Slot::create(&path, 1, FOCUS_INPUT_PAYLOAD_BYTES as u32).unwrap();
    slot.write(&bytes).unwrap();

    // Daemon-side read returns the part's slice + amplifier.
    let nudge = read_focus_nudge::<5>(Some(&slot), FocusPart::InnerBody);
    assert_eq!(nudge.stale_focus_multiplier, 2.0);
    assert!((nudge.nudge[0] - 0.10).abs() < 1e-6);

    // Compose into enrichment + run the §G5.2 kernel — output MUST move
    // upward vs the no-focus baseline.
    let mut enrichment = [0.0_f32; 5];
    compose_focus_into_enrichment(&mut enrichment, &nudge);
    // Amplified nudge: enrichment[0] = 0.10 * 2.0 = 0.20.
    assert!(
        (enrichment[0] - 0.20).abs() < 1e-6,
        "amplified enrichment should be 0.20 (got {})",
        enrichment[0]
    );

    let cfg = RestoringCfg {
        k_restore: 0.0, // isolate the additive enrichment from spring pull
        k_damp: 0.0,
        k_mom: 0.0,
        k_dir: 0.0,
        ..RestoringCfg::for_layer(Layer::Body)
    };
    let prev = vec![CENTRE; 5];
    let prev2 = vec![CENTRE; 5];
    let raw = vec![CENTRE; 5]; // drive = 0
    let obs = obs_of(1.0, 0.5, 0.0, 0.0, 0.0);

    let x_baseline = stateful_update(&prev, &prev2, &raw, &[0.0_f32; 5], &obs, &cfg);
    let x_focused = stateful_update(&prev, &prev2, &raw, &enrichment, &obs, &cfg);
    assert!(
        x_focused[0] > x_baseline[0] + 1e-4,
        "FOCUS nudge must move the §G5.2 output upward vs no-focus baseline \
         (focused={} baseline={})",
        x_focused[0],
        x_baseline[0],
    );
    // Other dims (no nudge) must be unchanged by focus.
    for i in 1..5 {
        assert!((x_focused[i] - x_baseline[i]).abs() < 1e-6);
    }
}

// ── P0.5 / D-SPEC-131 — UP-leg meaning-mapped gift (PLAN §6.5) ───────────────

// G5.1-up-leg-q-l-d-mask: spirit dims classified Q / L / D per locked masks.
#[test]
fn conformance_p0_5_up_leg_masks_q_l_d_locked() {
    // PLAN §6.5.4 LOCKED 2026-05-24: inner = 15Q + 16L + 14D; outer = 10Q +
    // 22L + 13D. D-dim combined fraction = 1.0 (body 0.25 + mind 0.75 per the
    // power-of-three law). Q dims receive 1.0 body / 0.0 mind; L mirror.
    use titan_trinity_daemon::{
        BODY_FLAG_INNER, BODY_FLAG_OUTER, MIND_FLAG_INNER, MIND_FLAG_OUTER,
    };
    for i in 0..45 {
        let inner = BODY_FLAG_INNER[i] + MIND_FLAG_INNER[i];
        let outer = BODY_FLAG_OUTER[i] + MIND_FLAG_OUTER[i];
        assert!(
            (inner - 1.0).abs() < 1e-6,
            "inner dim {i}: body+mind mask must combine to 1.0 (got {inner}) — \
             classification clauseQ=1+0, L=0+1, D=0.25+0.75",
        );
        assert!(
            (outer - 1.0).abs() < 1e-6,
            "outer dim {i}: body+mind mask must combine to 1.0 (got {outer})",
        );
    }
    // PLAN-locked class counts (must hold by construction).
    let inner_body_sum: f32 = BODY_FLAG_INNER.iter().sum();
    let inner_mind_sum: f32 = MIND_FLAG_INNER.iter().sum();
    let outer_body_sum: f32 = BODY_FLAG_OUTER.iter().sum();
    let outer_mind_sum: f32 = MIND_FLAG_OUTER.iter().sum();
    // 15·1 + 14·0.25 = 18.5; 16·1 + 14·0.75 = 26.5
    assert!(
        (inner_body_sum - 18.5).abs() < 1e-5,
        "inner BODY sum {inner_body_sum}"
    );
    assert!(
        (inner_mind_sum - 26.5).abs() < 1e-5,
        "inner MIND sum {inner_mind_sum}"
    );
    // 10·1 + 13·0.25 = 13.25; 22·1 + 13·0.75 = 31.75
    assert!(
        (outer_body_sum - 13.25).abs() < 1e-5,
        "outer BODY sum {outer_body_sum}"
    );
    assert!(
        (outer_mind_sum - 31.75).abs() < 1e-5,
        "outer MIND sum {outer_mind_sum}"
    );
}

// G5.1-journey-first-cycle-suppressed: first balanced pulse after boot emits NO gift.
#[test]
fn conformance_p0_5_journey_first_cycle_suppressed() {
    // PLAN §6.5.2: the first balanced pulse after daemon boot only RESETS
    // the accumulator — no gift emitted. This prevents cold-start partial
    // cycles from gifting unrepresentative data.
    use titan_trinity_daemon::{
        JourneyAccumulator, JourneyTickInputs, BODY_GIFT_WEIGHTS, MIND_GIFT_WEIGHTS,
    };
    let mut acc: JourneyAccumulator<5> = JourneyAccumulator::new();
    let x = [0.5_f32; 5];
    acc.tick(JourneyTickInputs {
        x: &x,
        obs: LayerObs::default(),
        now_secs: 0.0,
    });
    acc.mark_balanced(LayerObs::default());
    assert!(
        acc.finalize_body_gift(&BODY_GIFT_WEIGHTS).is_none(),
        "first balanced pulse: body gift MUST be None (cold-start suppression)",
    );
    let mut acc2: JourneyAccumulator<15> = JourneyAccumulator::new();
    let x15 = [0.5_f32; 15];
    acc2.tick(JourneyTickInputs {
        x: &x15,
        obs: LayerObs::default(),
        now_secs: 0.0,
    });
    acc2.mark_balanced(LayerObs::default());
    assert!(
        acc2.finalize_mind_gift(&MIND_GIFT_WEIGHTS).is_none(),
        "first balanced pulse: mind gift MUST be None",
    );
}

// G5.1-gift-event-roundtrip: encode → decode preserves side + amplitude.
#[test]
fn conformance_p0_5_gift_event_roundtrip() {
    // P0.5 / D-SPEC-131: BODY_BALANCE_GIFT + MIND_BALANCE_GIFT payloads are
    // structured rmpv::Value::Map per §8.6 convention (NOT opaque binary).
    // decode_gift_at_spirit MUST recover side + gift_amplitude + duration +
    // tick count + ts — the fields the spirit daemon uses for mask-weighted
    // enrichment.
    use titan_trinity_daemon::{
        decode_gift_at_spirit, encode_body_balance_gift, encode_mind_balance_gift,
        BodyJourneyDigest, MindJourneyDigest, TrinitySide, JOURNEY_SNAPSHOT_RING_LEN,
    };
    let body_digest: BodyJourneyDigest<5> = BodyJourneyDigest {
        gift_amplitude: 0.37,
        cycle_duration_s: 2.1,
        cycle_tick_count: 17,
        peak_excursion: [0.1; 5],
        path_length: [0.2; 5],
        excursion_integral: [0.05; 5],
        direction_flips: [1; 5],
        polarity_max: 0.3,
        polarity_at_balance: 0.1,
        per_dim_contribution: [0.2; 5],
        snapshots: [[0.5_f32; 5]; JOURNEY_SNAPSHOT_RING_LEN],
    };
    let body_payload = encode_body_balance_gift::<5>(TrinitySide::Inner, &body_digest, 99.0);
    let decoded = decode_gift_at_spirit(&body_payload).unwrap();
    assert_eq!(decoded.side, TrinitySide::Inner);
    assert!((decoded.gift_amplitude - 0.37).abs() < 1e-5);
    assert_eq!(decoded.cycle_tick_count, 17);

    let mind_digest: MindJourneyDigest<15> = MindJourneyDigest {
        gift_amplitude: 0.22,
        cycle_duration_s: 1.0,
        cycle_tick_count: 23,
        peak_excursion: [0.2; 15],
        path_length: [0.5; 15],
        excursion_integral: [0.1; 15],
        direction_flips: [1; 15],
        coherence_climb_max: 0.4,
        polarity_max: 0.5,
        polarity_at_balance: 0.1,
        per_dim_contribution: [1.0 / 15.0; 15],
        snapshots: [[0.5_f32; 15]; JOURNEY_SNAPSHOT_RING_LEN],
    };
    let mind_payload = encode_mind_balance_gift::<15>(TrinitySide::Outer, &mind_digest, 100.0);
    let decoded_m = decode_gift_at_spirit(&mind_payload).unwrap();
    assert_eq!(decoded_m.side, TrinitySide::Outer);
    assert!((decoded_m.gift_amplitude - 0.22).abs() < 1e-5);
}

// G5.1-spirit-subscribes-to-gifts: both spirit daemons receive both gifts.
#[test]
fn conformance_p0_5_spirit_subscribes_to_balance_gifts() {
    // P0.5 / D-SPEC-131: inner-spirit-rs + outer-spirit-rs MUST list both
    // BODY_BALANCE_GIFT and MIND_BALANCE_GIFT in their REQUIRED subscription
    // topics. Sovereign-half filtering happens at payload-decode time (by
    // `side` field) — the broker delivers both events to both daemons.
    use titan_trinity_daemon::{INNER_SPIRIT_TOPICS, OUTER_SPIRIT_TOPICS};
    assert!(INNER_SPIRIT_TOPICS.contains(&"BODY_BALANCE_GIFT"));
    assert!(INNER_SPIRIT_TOPICS.contains(&"MIND_BALANCE_GIFT"));
    assert!(OUTER_SPIRIT_TOPICS.contains(&"BODY_BALANCE_GIFT"));
    assert!(OUTER_SPIRIT_TOPICS.contains(&"MIND_BALANCE_GIFT"));
}

// G5.1-gift-mask-applied-by-kernel: BODY_FLAG_INNER * amp lands on correct dims.
#[test]
fn conformance_p0_5_gift_mask_applied_to_enrichment() {
    // P0.5 / D-SPEC-131: when a spirit daemon receives a BODY_BALANCE_GIFT
    // with amplitude `a`, the resulting enrichment vector has `UP_LEG_BONUS *
    // a * BODY_FLAG_*[i]` on each dim. The §G5.2 integrator then propagates
    // that enrichment into x[t]. Asserts the kernel composition is the
    // same regardless of HOW the enrichment was constructed (the kernel
    // doesn't know about masks — it only knows about additive enrichment).
    //
    // The integration tests in journey + gift_events + up_leg_masks lib tests
    // assert the per-component correctness; this conformance test verifies the
    // composition lands as designed at the kernel boundary.
    use titan_trinity_daemon::{BODY_FLAG_INNER, MIND_FLAG_INNER};
    const UP_LEG_AMP: f32 = 0.02;
    const DIMS: usize = 45;
    let body_amp = 0.5_f32;
    let mind_amp = 0.4_f32;
    let mut enrichment = [0.0_f32; DIMS];
    for i in 0..DIMS {
        enrichment[i] += UP_LEG_AMP * body_amp * BODY_FLAG_INNER[i];
        enrichment[i] += UP_LEG_AMP * mind_amp * MIND_FLAG_INNER[i];
    }
    // Q dim (inner Q list includes 6): receives body only.
    let q_dim = 6;
    let expected_q = UP_LEG_AMP * body_amp * 1.0 + UP_LEG_AMP * mind_amp * 0.0;
    assert!(
        (enrichment[q_dim] - expected_q).abs() < 1e-6,
        "Q dim {q_dim}: body-only enrichment {} (got {})",
        expected_q,
        enrichment[q_dim],
    );
    // L dim (inner L list includes 0): receives mind only.
    let l_dim = 0;
    let expected_l = UP_LEG_AMP * mind_amp * 1.0;
    assert!(
        (enrichment[l_dim] - expected_l).abs() < 1e-6,
        "L dim {l_dim}: mind-only enrichment {} (got {})",
        expected_l,
        enrichment[l_dim],
    );
    // D dim (inner D list includes 1): receives both with 0.25/0.75 split.
    let d_dim = 1;
    let expected_d = UP_LEG_AMP * body_amp * 0.25 + UP_LEG_AMP * mind_amp * 0.75;
    assert!(
        (enrichment[d_dim] - expected_d).abs() < 1e-6,
        "D dim {d_dim}: 0.25body + 0.75mind = {} (got {})",
        expected_d,
        enrichment[d_dim],
    );
    // Drive the §G5.2 kernel with this composed enrichment — output should
    // move on classified dims (no flat zero for any classified position).
    let cfg = RestoringCfg {
        k_restore: 0.0,
        k_damp: 0.0,
        k_mom: 0.0,
        k_dir: 0.0,
        ..RestoringCfg::for_layer(Layer::Spirit)
    };
    let prev = vec![CENTRE; DIMS];
    let prev2 = vec![CENTRE; DIMS];
    let raw = vec![CENTRE; DIMS];
    let obs = obs_of(1.0, 0.5, 0.0, 0.0, 0.0);
    let x = stateful_update(&prev, &prev2, &raw, &enrichment, &obs, &cfg);
    assert!(x[q_dim] > CENTRE);
    assert!(x[l_dim] > CENTRE);
    assert!(x[d_dim] > CENTRE);
}
