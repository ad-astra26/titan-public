//! observer_mask — Phase C C-S6 outer-spirit observer-dim masking.
//!
//! Per SPEC G8 + §7.1 + §10.F step 3 + outer_spirit_tensor.py:9 OBSERVER
//! PRINCIPLE: outer_spirit dims `[0:5]` (= TITAN_SELF absolute `[85:90]`)
//! are the **observer dims**. They are computed and STORED in the slot
//! (so downstream consumers like CGN, reasoning, MSL see all 45D), but
//! they are **MASKED at filter_down OUTPUT** — the
//! `OUTER_SPIRIT_FILTER_DOWN.outer_spirit_content[40]` payload field
//! contains exactly `outer_spirit_45d[5:45]`, never the observer dims.
//!
//! "Spirit observer is a reflection surface, not a target of filtering."
//!
//! # Why mask at output, not at compute
//!
//! Masking at compute would zero the observer dims in the slot, which
//! would (a) lie to L2 consumers that read the slot for self-perception
//! and (b) break the 45D shape contract. Masking at the FILTER_DOWN emit
//! site preserves the slot's full 45D semantics + enforces the doctrine
//! that the spirit's observer eye is never a vector consumers can bias.
//!
//! # Constants
//!
//! Bound to TOML constants (auto-generated from SPEC):
//! - `OUTER_SPIRIT_OBSERVER_DIM_START` = 0
//! - `OUTER_SPIRIT_OBSERVER_DIM_END` = 5
//! - `OUTER_SPIRIT_CONTENT_DIM_START` = 5
//! - `OUTER_SPIRIT_CONTENT_DIM_END` = 45
//!
//! Sourced via `titan-core::constants` (auto-regen from
//! `SPEC_titan_architecture_constants.toml`).
//!
//! # API surface
//!
//! - [`extract_outer_spirit_content`]: returns the 40D content slice
//!   `outer_spirit_45d[5:45]` for use in `OUTER_SPIRIT_FILTER_DOWN`.
//! - [`mask_observer_dims_in_place`]: zeros `[0:5]` of any 45D slice in
//!   place (defensive helper for callers that need to pre-mask before
//!   serialization).

use titan_core::constants::{
    OUTER_SPIRIT_CONTENT_DIM_END, OUTER_SPIRIT_CONTENT_DIM_START, OUTER_SPIRIT_OBSERVER_DIM_END,
    OUTER_SPIRIT_OBSERVER_DIM_START,
};

/// Number of observer dims to mask. Bound to SPEC constants at compile
/// time so any TOML drift surfaces as a build error (not a runtime bug).
pub const OBSERVER_DIM_COUNT: usize =
    (OUTER_SPIRIT_OBSERVER_DIM_END - OUTER_SPIRIT_OBSERVER_DIM_START) as usize;

/// Number of content dims published in OUTER_SPIRIT_FILTER_DOWN.
pub const CONTENT_DIM_COUNT: usize =
    (OUTER_SPIRIT_CONTENT_DIM_END - OUTER_SPIRIT_CONTENT_DIM_START) as usize;

/// Byte-level start offset of the observer range in the 45D float32-LE
/// payload. Useful for parity tests + low-level byte readers.
pub const OBSERVER_BYTE_START: usize = (OUTER_SPIRIT_OBSERVER_DIM_START as usize) * 4;

/// Byte-level end (exclusive) offset of the observer range.
pub const OBSERVER_BYTE_END: usize = (OUTER_SPIRIT_OBSERVER_DIM_END as usize) * 4;

/// Extract the 40D content slice `outer_spirit_45d[5:45]` from a full 45D
/// outer-spirit tensor. This is exactly what
/// `OUTER_SPIRIT_FILTER_DOWN.multipliers.outer_spirit_content` contains
/// per SPEC §8.6 row 4.
///
/// # Panics
///
/// Cannot panic — both ranges are compile-time SPEC-bound and the input
/// is a fixed-size 45-element array. Index errors would surface at the
/// `cargo build` step.
pub fn extract_outer_spirit_content(outer_spirit_45d: &[f32; 45]) -> [f32; CONTENT_DIM_COUNT] {
    let mut content = [0.0_f32; CONTENT_DIM_COUNT];
    let start = OUTER_SPIRIT_CONTENT_DIM_START as usize;
    let end = OUTER_SPIRIT_CONTENT_DIM_END as usize;
    content.copy_from_slice(&outer_spirit_45d[start..end]);
    content
}

/// Zero out the observer dims `[0:5]` of a 45D outer-spirit slice in
/// place. Defensive helper — most callers should use
/// [`extract_outer_spirit_content`] which structurally cannot leak the
/// observer dims into the output (different array length).
pub fn mask_observer_dims_in_place(outer_spirit_45d: &mut [f32; 45]) {
    let start = OUTER_SPIRIT_OBSERVER_DIM_START as usize;
    let end = OUTER_SPIRIT_OBSERVER_DIM_END as usize;
    for x in outer_spirit_45d[start..end].iter_mut() {
        *x = 0.0;
    }
}

/// Sanity-check helper for tests + debug assertions: returns true iff
/// the first `OBSERVER_DIM_COUNT` elements of a 45D slice are exactly
/// zero.
pub fn observer_dims_are_zero(outer_spirit_45d: &[f32; 45]) -> bool {
    let start = OUTER_SPIRIT_OBSERVER_DIM_START as usize;
    let end = OUTER_SPIRIT_OBSERVER_DIM_END as usize;
    outer_spirit_45d[start..end].iter().all(|&x| x == 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 45D vector where dim `i` has value `i as f32 + 0.5` so each
    /// dim is uniquely identifiable in slice tests.
    fn make_distinct_45d() -> [f32; 45] {
        let mut v = [0.0_f32; 45];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = i as f32 + 0.5;
        }
        v
    }

    #[test]
    fn constants_match_spec() {
        // Compile-time: G8 + G9 + SPEC §7.1 row 7
        assert_eq!(OUTER_SPIRIT_OBSERVER_DIM_START, 0);
        assert_eq!(OUTER_SPIRIT_OBSERVER_DIM_END, 5);
        assert_eq!(OUTER_SPIRIT_CONTENT_DIM_START, 5);
        assert_eq!(OUTER_SPIRIT_CONTENT_DIM_END, 45);
        assert_eq!(OBSERVER_DIM_COUNT, 5);
        assert_eq!(CONTENT_DIM_COUNT, 40);
        assert_eq!(OBSERVER_BYTE_START, 0);
        assert_eq!(OBSERVER_BYTE_END, 20); // 5 dims × 4 bytes/f32
    }

    #[test]
    fn extract_content_returns_exactly_40_dims() {
        let v = make_distinct_45d();
        let content = extract_outer_spirit_content(&v);
        // length is locked at compile time via const
        let _: &[f32; 40] = &content;
        // First content dim must be original [5] (= 5.5)
        assert_eq!(content[0], 5.5);
        // Last content dim must be original [44] (= 44.5)
        assert_eq!(content[39], 44.5);
    }

    #[test]
    fn extract_content_excludes_observer_dims() {
        let v = make_distinct_45d();
        let content = extract_outer_spirit_content(&v);
        // The observer values 0.5, 1.5, 2.5, 3.5, 4.5 must NOT appear in
        // any content slot (each is unique by construction).
        for &observer_val in &[0.5_f32, 1.5, 2.5, 3.5, 4.5] {
            assert!(
                !content.contains(&observer_val),
                "observer dim value {} leaked into content slice",
                observer_val,
            );
        }
    }

    #[test]
    fn extract_content_byte_identical_to_45d_slice_5_to_45() {
        // The output of extract_outer_spirit_content MUST be byte-identical
        // (modulo array vs. slice type) to outer_spirit_45d[5:45]. This is
        // the parity contract for OUTER_SPIRIT_FILTER_DOWN.
        let v = make_distinct_45d();
        let content = extract_outer_spirit_content(&v);
        for (i, &x) in content.iter().enumerate() {
            assert_eq!(x, v[5 + i], "mismatch at content[{}] vs 45d[{}]", i, 5 + i);
        }
    }

    #[test]
    fn mask_observer_dims_zeros_first_five_only() {
        let mut v = make_distinct_45d();
        mask_observer_dims_in_place(&mut v);
        // First 5 zeroed
        for i in 0..5 {
            assert_eq!(v[i], 0.0, "dim {} should be 0 after mask", i);
        }
        // Remaining 40 preserved (dim i has value i + 0.5)
        for i in 5..45 {
            assert_eq!(v[i], i as f32 + 0.5, "dim {} should be unchanged", i);
        }
    }

    #[test]
    fn mask_idempotent() {
        let mut v = make_distinct_45d();
        mask_observer_dims_in_place(&mut v);
        let snapshot = v;
        mask_observer_dims_in_place(&mut v);
        assert_eq!(v, snapshot, "mask should be idempotent");
    }

    #[test]
    fn observer_dims_are_zero_check() {
        let mut v = make_distinct_45d();
        assert!(!observer_dims_are_zero(&v));
        mask_observer_dims_in_place(&mut v);
        assert!(observer_dims_are_zero(&v));
    }
}
