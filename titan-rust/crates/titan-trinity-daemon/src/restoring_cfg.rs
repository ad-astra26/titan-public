//! restoring_cfg — boot-time loader for the §G5.2 restoring-force gains.
//!
//! Per SPEC §G5.2 item 5 + PLAN §1.5 / §6.6.2 + D-SPEC-129 (per-layer
//! extension, v1.59.2, 2026-05-25): the gain coefficients live in
//! `titan_hcl/titan_params.toml [trinity_restoring.{body,mind,spirit}]`
//! (per-Titan, per-layer tunable, NOT hardcoded). Inner+outer of a layer
//! share one tuning unit. Rather than parse TOML inside the daemon (would
//! add a runtime dep + path-discovery), Python L2 reads the three subsections
//! at startup and writes a fixed-layout SHM sidecar `trinity_restoring.bin`
//! (3 layers × 8 floats × 4 bytes LE = 96 bytes) in `shm_dir/`. Daemons
//! retry-load it at boot — if absent or short, they fall back to the
//! crate-default `DEFAULT_*` constants in [`homeostasis`].
//!
//! Layout (must match `titan_hcl/logic/trinity_restoring_publisher.py`):
//! ```text
//!   bytes [ 0:32) — body  (8 × f32 LE)
//!   bytes [32:64) — mind  (8 × f32 LE)
//!   bytes [64:96) — spirit(8 × f32 LE)
//! ```
//!
//! Per-layer field order:
//! ```text
//!   [0] k_drive       [4] k_dir
//!   [1] k_restore     [5] a_mag
//!   [2] k_damp        [6] a_drift
//!   [3] k_mom         [7] a_dmag
//! ```
//!
//! The per-layer quant→qual gradient (INV-9 body 0.7/0.3, mind 0.5/0.5,
//! spirit 0.3/0.7) is structural — fixed in [`homeostasis::gradient`], not in
//! the TOML.

use std::path::Path;

use tracing::{info, warn};

use crate::homeostasis::{Layer, RestoringCfg};

/// Per-layer slice size: 8 × float32 LE.
pub const TRINITY_RESTORING_PER_LAYER_BYTES: usize = 32;
/// Full sidecar payload: 3 layers × 32 bytes (D-SPEC-129).
pub const TRINITY_RESTORING_PAYLOAD_BYTES: usize = 96;
/// Sidecar file name under `shm_dir/`.
pub const TRINITY_RESTORING_SIDECAR: &str = "trinity_restoring.bin";

/// Offset (in bytes) of a layer's 32-byte slice within the sidecar.
/// Must match the order Python writes (body → mind → spirit).
const fn layer_offset(layer: Layer) -> usize {
    match layer {
        Layer::Body => 0,
        Layer::Mind => TRINITY_RESTORING_PER_LAYER_BYTES,
        Layer::Spirit => 2 * TRINITY_RESTORING_PER_LAYER_BYTES,
    }
}

/// Read this layer's 8 floats from `shm_dir/trinity_restoring.bin` and merge
/// into a per-layer [`RestoringCfg`] (gradient comes from [`Layer`]). Falls
/// back to crate defaults on any read/parse failure — substrate continues
/// (`feedback_no_hardcoded_values_emergence_over_determinism` — config
/// gains OK, defaults are a safe baseline, not a signal floor).
pub fn load_for_layer(shm_dir: &Path, layer: Layer) -> RestoringCfg {
    let mut cfg = RestoringCfg::for_layer(layer);
    let path = shm_dir.join(TRINITY_RESTORING_SIDECAR);
    match std::fs::read(&path) {
        Ok(bytes) if bytes.len() >= TRINITY_RESTORING_PAYLOAD_BYTES => {
            let off = layer_offset(layer);
            let mut g = [0.0_f32; 8];
            for i in 0..8 {
                let pos = off + i * 4;
                g[i] = f32::from_le_bytes(
                    bytes[pos..pos + 4]
                        .try_into()
                        .expect("4 bytes after length check"),
                );
            }
            cfg.k_drive = g[0];
            cfg.k_restore = g[1];
            cfg.k_damp = g[2];
            cfg.k_mom = g[3];
            cfg.k_dir = g[4];
            cfg.a_mag = g[5];
            cfg.a_drift = g[6];
            cfg.a_dmag = g[7];
            info!(
                event = "TRINITY_RESTORING_CFG_LOADED",
                layer = ?layer,
                slice_offset = off,
                k_drive = cfg.k_drive,
                k_restore = cfg.k_restore,
                k_damp = cfg.k_damp,
                k_mom = cfg.k_mom,
                k_dir = cfg.k_dir,
                a_mag = cfg.a_mag,
                a_drift = cfg.a_drift,
                a_dmag = cfg.a_dmag,
                source = "trinity_restoring.bin",
            );
        }
        Ok(_short) => {
            warn!(
                event = "TRINITY_RESTORING_CFG_FALLBACK",
                reason = "sidecar payload < 96 bytes (D-SPEC-129 per-layer)",
                path = %path.display(),
                layer = ?layer,
                "falling back to crate-default §G5.2 gains"
            );
        }
        Err(e) => {
            // Absence at boot is normal (Python sidecar may not have
            // landed yet); not an error. Log at info so operators can see
            // the resolution path but it doesn't pollute warn-level.
            info!(
                event = "TRINITY_RESTORING_CFG_DEFAULT",
                reason = format!("{e}"),
                path = %path.display(),
                layer = ?layer,
                "using crate-default §G5.2 gains until sidecar present"
            );
        }
    }
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    /// Pack a per-layer 8-float slice in the sidecar field order.
    fn pack_slice(gains: [f32; 8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(TRINITY_RESTORING_PER_LAYER_BYTES);
        for v in gains.iter() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    /// Build a full 96-byte D-SPEC-129 sidecar from per-layer gains.
    fn pack_full(body: [f32; 8], mind: [f32; 8], spirit: [f32; 8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(TRINITY_RESTORING_PAYLOAD_BYTES);
        bytes.extend(pack_slice(body));
        bytes.extend(pack_slice(mind));
        bytes.extend(pack_slice(spirit));
        bytes
    }

    #[test]
    fn fallback_to_defaults_when_absent() {
        let dir = tempdir().unwrap();
        let cfg = load_for_layer(dir.path(), Layer::Body);
        // Crate-default k_drive = DEFAULT_K_DRIVE = 0.30 — substrate continues.
        assert!((cfg.k_drive - crate::homeostasis::DEFAULT_K_DRIVE).abs() < 1e-6);
    }

    #[test]
    fn loads_overrides_from_sidecar() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(TRINITY_RESTORING_SIDECAR);
        // Spirit slice gets distinct values from body/mind so we know the
        // offset table is correct (not just reading position 0).
        let body = [0.30_f32, 0.05, 0.05, 0.10, 0.05, 0.50, 1.00, 0.50];
        let mind = [0.40_f32, 0.05, 0.05, 0.10, 0.05, 0.50, 1.00, 0.50];
        let spirit = [0.42_f32, 0.07, 0.06, 0.11, 0.04, 0.55, 1.10, 0.45];
        fs::write(&path, pack_full(body, mind, spirit)).unwrap();

        let cfg = load_for_layer(dir.path(), Layer::Spirit);
        assert!((cfg.k_drive - 0.42).abs() < 1e-6, "spirit k_drive");
        assert!((cfg.k_restore - 0.07).abs() < 1e-6);
        assert!((cfg.k_damp - 0.06).abs() < 1e-6);
        assert!((cfg.k_mom - 0.11).abs() < 1e-6);
        assert!((cfg.k_dir - 0.04).abs() < 1e-6);
        assert!((cfg.a_mag - 0.55).abs() < 1e-6);
        assert!((cfg.a_drift - 1.10).abs() < 1e-6);
        assert!((cfg.a_dmag - 0.45).abs() < 1e-6);
        // Gradient is structural per INV-9 — Spirit = (0.3, 0.7).
        assert!((cfg.w_quant - 0.3).abs() < 1e-6);
        assert!((cfg.w_qual - 0.7).abs() < 1e-6);
    }

    #[test]
    fn per_layer_slices_resolve_independently() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(TRINITY_RESTORING_SIDECAR);
        let body = [0.30_f32, 0.05, 0.05, 0.10, 0.05, 0.50, 1.00, 0.50];
        let mind = [0.40_f32, 0.05, 0.05, 0.10, 0.05, 0.50, 1.00, 0.50];
        let spirit = [0.20_f32, 0.05, 0.05, 0.10, 0.05, 0.50, 1.00, 0.50];
        fs::write(&path, pack_full(body, mind, spirit)).unwrap();

        // Each layer's loader returns its OWN k_drive, not body's. This
        // catches the offset-table regression class (reading slice 0 for
        // every layer = the pre-D-SPEC-129 bug).
        let body_cfg = load_for_layer(dir.path(), Layer::Body);
        let mind_cfg = load_for_layer(dir.path(), Layer::Mind);
        let spirit_cfg = load_for_layer(dir.path(), Layer::Spirit);
        assert!(
            (body_cfg.k_drive - 0.30).abs() < 1e-6,
            "body got body slice"
        );
        assert!(
            (mind_cfg.k_drive - 0.40).abs() < 1e-6,
            "mind got mind slice"
        );
        assert!(
            (spirit_cfg.k_drive - 0.20).abs() < 1e-6,
            "spirit got spirit slice"
        );
    }

    #[test]
    fn short_sidecar_falls_back() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(TRINITY_RESTORING_SIDECAR);
        // A pre-D-SPEC-129 32-byte sidecar is now considered too-short —
        // daemons fall back to crate defaults rather than risk reading
        // garbage past the end of a stale file.
        fs::write(&path, vec![0_u8; 32]).unwrap();
        let cfg = load_for_layer(dir.path(), Layer::Mind);
        assert!((cfg.k_drive - crate::homeostasis::DEFAULT_K_DRIVE).abs() < 1e-6);
    }
}
