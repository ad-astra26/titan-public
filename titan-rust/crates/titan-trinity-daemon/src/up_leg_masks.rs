//! up_leg_masks — Q/L/D classification const masks for the §G5.1 UP-leg gift.
//!
//! PLAN_trinity_homeostasis_p0 §6.5.4 (LOCKED 2026-05-24). For every position
//! in the 45D inner_spirit / outer_spirit tensors, classifies whether that
//! dim is enriched primarily by the quantitative-essence body gift (Q),
//! the qualitative-essence mind gift (L), or by both (D = dual).
//!
//! Mask values land on the per-position additive term as:
//!
//! ```text
//! spirit_45d[i] += UP_LEG_BONUS_AMPLITUDE
//!                * gift.gift_amplitude
//!                * mask[i]
//! ```
//!
//! Per the §6.5.4 power-of-three law (`body=5/(5+15)=0.25`, `mind=15/20=0.75`)
//! a D dim's body-fraction is 0.25 and mind-fraction is 0.75 — so if both
//! BodyBalanceGift and MindBalanceGift fire on the same Schumann tick (a
//! sub-1% coincidence) the D dim receives `0.25*body + 0.75*mind = 1.0×` the
//! combined gift; a Q dim receives a 1.0× body gift only; an L dim receives
//! a 1.0× mind gift only.
//!
//! Per `feedback_spec_bound_work_zero_simplification_clause_by_clause`: these
//! arrays implement the PLAN's locked classification clause-by-clause without
//! simplification. Each mask sums to:
//!   * inner BODY: `15·1.0 + 16·0.0 + 14·0.25 = 18.5`
//!   * inner MIND: `15·0.0 + 16·1.0 + 14·0.75 = 26.5`
//!   * outer BODY: `10·1.0 + 22·0.0 + 13·0.25 = 13.25`
//!   * outer MIND: `10·0.0 + 22·1.0 + 13·0.75 = 31.75`
//!
//! These sums are NOT normalised — the §6.5.5 spirit applicator multiplies
//! by the per-event `gift.gift_amplitude` which absorbs scale, and the §G5.2
//! integrator clamps to [0, 1] (the natural energy sink for over-amplitude
//! gifts).
//!
//! D dims (PLAN §6.5.4 LOCKED 2026-05-24):
//!   inner: 1, 17, 20, 22, 24, 28, 31, 32, 35, 36, 38, 39, 41, 43  (14 dims)
//!   outer: 0, 1, 3, 5, 9, 13, 20, 25, 32, 36, 38, 39, 43  (13 dims)
//!
//! Q dims (body-enriched):
//!   inner: 6, 7, 9, 11, 14, 21, 23, 25, 26, 27, 29, 37, 40, 42, 44  (15 dims)
//!   outer: 4, 7, 14, 15, 21, 29, 30, 34, 37, 41  (10 dims)
//!
//! L dims (mind-enriched):
//!   inner: 0, 2, 3, 4, 5, 8, 10, 12, 13, 15, 16, 18, 19, 30, 33, 34  (16 dims)
//!   outer: 2, 6, 8, 10, 11, 12, 16, 17, 18, 19, 22, 23, 24, 26, 27, 28, 31, 33, 35, 40, 42, 44  (22 dims)
//!
//! Class checksums (must hold at compile-time): 15+16+14=45 inner; 10+22+13=45 outer.

/// Per-position body-fraction for inner_spirit_45d. Q=1.0, D=0.25, L=0.0.
pub const BODY_FLAG_INNER: [f32; 45] = {
    let mut m = [0.0_f32; 45];
    // Q dims — full body weight (1.0).
    m[6] = 1.0;
    m[7] = 1.0;
    m[9] = 1.0;
    m[11] = 1.0;
    m[14] = 1.0;
    m[21] = 1.0;
    m[23] = 1.0;
    m[25] = 1.0;
    m[26] = 1.0;
    m[27] = 1.0;
    m[29] = 1.0;
    m[37] = 1.0;
    m[40] = 1.0;
    m[42] = 1.0;
    m[44] = 1.0;
    // D dims — 0.25 body share per the power-of-three law (5/(5+15)).
    m[1] = 0.25;
    m[17] = 0.25;
    m[20] = 0.25;
    m[22] = 0.25;
    m[24] = 0.25;
    m[28] = 0.25;
    m[31] = 0.25;
    m[32] = 0.25;
    m[35] = 0.25;
    m[36] = 0.25;
    m[38] = 0.25;
    m[39] = 0.25;
    m[41] = 0.25;
    m[43] = 0.25;
    m
};

/// Per-position mind-fraction for inner_spirit_45d. L=1.0, D=0.75, Q=0.0.
pub const MIND_FLAG_INNER: [f32; 45] = {
    let mut m = [0.0_f32; 45];
    // L dims — full mind weight (1.0).
    m[0] = 1.0;
    m[2] = 1.0;
    m[3] = 1.0;
    m[4] = 1.0;
    m[5] = 1.0;
    m[8] = 1.0;
    m[10] = 1.0;
    m[12] = 1.0;
    m[13] = 1.0;
    m[15] = 1.0;
    m[16] = 1.0;
    m[18] = 1.0;
    m[19] = 1.0;
    m[30] = 1.0;
    m[33] = 1.0;
    m[34] = 1.0;
    // D dims — 0.75 mind share per the power-of-three law (15/(5+15)).
    m[1] = 0.75;
    m[17] = 0.75;
    m[20] = 0.75;
    m[22] = 0.75;
    m[24] = 0.75;
    m[28] = 0.75;
    m[31] = 0.75;
    m[32] = 0.75;
    m[35] = 0.75;
    m[36] = 0.75;
    m[38] = 0.75;
    m[39] = 0.75;
    m[41] = 0.75;
    m[43] = 0.75;
    m
};

/// Per-position body-fraction for outer_spirit_45d. Q=1.0, D=0.25, L=0.0.
pub const BODY_FLAG_OUTER: [f32; 45] = {
    let mut m = [0.0_f32; 45];
    // Q dims — full body weight (1.0).
    m[4] = 1.0;
    m[7] = 1.0;
    m[14] = 1.0;
    m[15] = 1.0;
    m[21] = 1.0;
    m[29] = 1.0;
    m[30] = 1.0;
    m[34] = 1.0;
    m[37] = 1.0;
    m[41] = 1.0;
    // D dims — 0.25 body share.
    m[0] = 0.25;
    m[1] = 0.25;
    m[3] = 0.25;
    m[5] = 0.25;
    m[9] = 0.25;
    m[13] = 0.25;
    m[20] = 0.25;
    m[25] = 0.25;
    m[32] = 0.25;
    m[36] = 0.25;
    m[38] = 0.25;
    m[39] = 0.25;
    m[43] = 0.25;
    m
};

/// Per-position mind-fraction for outer_spirit_45d. L=1.0, D=0.75, Q=0.0.
pub const MIND_FLAG_OUTER: [f32; 45] = {
    let mut m = [0.0_f32; 45];
    // L dims — full mind weight (1.0).
    m[2] = 1.0;
    m[6] = 1.0;
    m[8] = 1.0;
    m[10] = 1.0;
    m[11] = 1.0;
    m[12] = 1.0;
    m[16] = 1.0;
    m[17] = 1.0;
    m[18] = 1.0;
    m[19] = 1.0;
    m[22] = 1.0;
    m[23] = 1.0;
    m[24] = 1.0;
    m[26] = 1.0;
    m[27] = 1.0;
    m[28] = 1.0;
    m[31] = 1.0;
    m[33] = 1.0;
    m[35] = 1.0;
    m[40] = 1.0;
    m[42] = 1.0;
    m[44] = 1.0;
    // D dims — 0.75 mind share.
    m[0] = 0.75;
    m[1] = 0.75;
    m[3] = 0.75;
    m[5] = 0.75;
    m[9] = 0.75;
    m[13] = 0.75;
    m[20] = 0.75;
    m[25] = 0.75;
    m[32] = 0.75;
    m[36] = 0.75;
    m[38] = 0.75;
    m[39] = 0.75;
    m[43] = 0.75;
    m
};

#[cfg(test)]
mod tests {
    use super::*;

    fn class_counts(body: &[f32; 45], mind: &[f32; 45]) -> (usize, usize, usize) {
        let mut q = 0;
        let mut l = 0;
        let mut d = 0;
        for i in 0..45 {
            match (body[i], mind[i]) {
                (1.0, 0.0) => q += 1,
                (0.0, 1.0) => l += 1,
                (0.25, 0.75) => d += 1,
                (0.0, 0.0) => panic!("dim {i} unclassified: body=0 mind=0"),
                other => panic!("dim {i} bad pair: {:?}", other),
            }
        }
        (q, l, d)
    }

    #[test]
    fn inner_class_counts_match_plan_lock() {
        let (q, l, d) = class_counts(&BODY_FLAG_INNER, &MIND_FLAG_INNER);
        assert_eq!((q, l, d), (15, 16, 14), "inner: 15Q + 16L + 14D = 45");
    }

    #[test]
    fn outer_class_counts_match_plan_lock() {
        let (q, l, d) = class_counts(&BODY_FLAG_OUTER, &MIND_FLAG_OUTER);
        assert_eq!((q, l, d), (10, 22, 13), "outer: 10Q + 22L + 13D = 45");
    }

    #[test]
    fn d_dim_body_plus_mind_equals_one() {
        // §6.5.4 power-of-three law: when BOTH gifts arrive on the same tick,
        // the D dim's combined fraction must be 0.25 + 0.75 = 1.0.
        for masks in [
            (&BODY_FLAG_INNER, &MIND_FLAG_INNER),
            (&BODY_FLAG_OUTER, &MIND_FLAG_OUTER),
        ] {
            for i in 0..45 {
                let combined = masks.0[i] + masks.1[i];
                assert!(
                    (combined - 1.0).abs() < 1e-6,
                    "dim {i} combined = {combined} (expected 1.0)",
                );
            }
        }
    }

    #[test]
    fn inner_q_dims_match_plan_lock() {
        const Q: [usize; 15] = [6, 7, 9, 11, 14, 21, 23, 25, 26, 27, 29, 37, 40, 42, 44];
        for &i in &Q {
            assert_eq!(BODY_FLAG_INNER[i], 1.0, "inner Q dim {i} body != 1.0");
            assert_eq!(MIND_FLAG_INNER[i], 0.0, "inner Q dim {i} mind != 0.0");
        }
    }

    #[test]
    fn inner_l_dims_match_plan_lock() {
        const L: [usize; 16] = [0, 2, 3, 4, 5, 8, 10, 12, 13, 15, 16, 18, 19, 30, 33, 34];
        for &i in &L {
            assert_eq!(BODY_FLAG_INNER[i], 0.0, "inner L dim {i} body != 0.0");
            assert_eq!(MIND_FLAG_INNER[i], 1.0, "inner L dim {i} mind != 1.0");
        }
    }

    #[test]
    fn inner_d_dims_match_plan_lock() {
        const D: [usize; 14] = [1, 17, 20, 22, 24, 28, 31, 32, 35, 36, 38, 39, 41, 43];
        for &i in &D {
            assert_eq!(BODY_FLAG_INNER[i], 0.25, "inner D dim {i} body != 0.25");
            assert_eq!(MIND_FLAG_INNER[i], 0.75, "inner D dim {i} mind != 0.75");
        }
    }

    #[test]
    fn outer_q_dims_match_plan_lock() {
        const Q: [usize; 10] = [4, 7, 14, 15, 21, 29, 30, 34, 37, 41];
        for &i in &Q {
            assert_eq!(BODY_FLAG_OUTER[i], 1.0);
            assert_eq!(MIND_FLAG_OUTER[i], 0.0);
        }
    }

    #[test]
    fn outer_l_dims_match_plan_lock() {
        const L: [usize; 22] = [
            2, 6, 8, 10, 11, 12, 16, 17, 18, 19, 22, 23, 24, 26, 27, 28, 31, 33, 35, 40, 42, 44,
        ];
        for &i in &L {
            assert_eq!(BODY_FLAG_OUTER[i], 0.0);
            assert_eq!(MIND_FLAG_OUTER[i], 1.0);
        }
    }

    #[test]
    fn outer_d_dims_match_plan_lock() {
        const D: [usize; 13] = [0, 1, 3, 5, 9, 13, 20, 25, 32, 36, 38, 39, 43];
        for &i in &D {
            assert_eq!(BODY_FLAG_OUTER[i], 0.25);
            assert_eq!(MIND_FLAG_OUTER[i], 0.75);
        }
    }

    #[test]
    fn body_flag_sum_invariant_inner() {
        let s: f32 = BODY_FLAG_INNER.iter().sum();
        // 15·1.0 + 14·0.25 + 16·0.0 = 18.5
        assert!((s - 18.5).abs() < 1e-5, "BODY_FLAG_INNER sum = {s}");
    }

    #[test]
    fn mind_flag_sum_invariant_inner() {
        let s: f32 = MIND_FLAG_INNER.iter().sum();
        // 16·1.0 + 14·0.75 + 15·0.0 = 26.5
        assert!((s - 26.5).abs() < 1e-5, "MIND_FLAG_INNER sum = {s}");
    }

    #[test]
    fn body_flag_sum_invariant_outer() {
        let s: f32 = BODY_FLAG_OUTER.iter().sum();
        // 10·1.0 + 13·0.25 + 22·0.0 = 13.25
        assert!((s - 13.25).abs() < 1e-5, "BODY_FLAG_OUTER sum = {s}");
    }

    #[test]
    fn mind_flag_sum_invariant_outer() {
        let s: f32 = MIND_FLAG_OUTER.iter().sum();
        // 22·1.0 + 13·0.75 + 10·0.0 = 31.75
        assert!((s - 31.75).abs() < 1e-5, "MIND_FLAG_OUTER sum = {s}");
    }
}
