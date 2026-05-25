//! restoring_cfg — boot-time loader for the §G5.2 restoring-force gains.
//!
//! Per SPEC §G5.2 item 5 + PLAN §1.5: the gain coefficients live in
//! `titan_hcl/titan_params.toml [trinity_restoring]` (per-Titan tunable, NOT
//! hardcoded). Rather than parse TOML inside the daemon (would add a runtime
//! dep + path-discovery), Python L2 reads the TOML at startup and writes a
//! fixed-layout SHM sidecar `trinity_restoring.bin` (8 × float32 LE = 32 bytes)
//! in `shm_dir/`. Daemons retry-load it at boot — if absent or stale, they
//! fall back to the crate-default `DEFAULT_*` constants in [`homeostasis`].
//!
//! Field order (matches `titan_hcl/logic/trinity_restoring_publisher.py`):
//! ```text
//!   [0] k_drive
//!   [1] k_restore
//!   [2] k_damp
//!   [3] k_mom
//!   [4] k_dir
//!   [5] a_mag
//!   [6] a_drift
//!   [7] a_dmag
//! ```
//!
//! The per-layer quant→qual gradient (INV-9 body 0.7/0.3, mind 0.5/0.5,
//! spirit 0.3/0.7) is structural — fixed in [`homeostasis::gradient`], not in
//! the TOML.

use std::path::Path;

use tracing::{info, warn};

use crate::homeostasis::{Layer, RestoringCfg};

/// Fixed-layout payload size: 8 × float32 LE.
pub const TRINITY_RESTORING_PAYLOAD_BYTES: usize = 32;
/// Sidecar file name under `shm_dir/`.
pub const TRINITY_RESTORING_SIDECAR: &str = "trinity_restoring.bin";

/// Read 8 floats from `shm_dir/trinity_restoring.bin` and merge into a
/// per-layer [`RestoringCfg`] (gradient comes from [`Layer`]). Falls back
/// to crate defaults on any read/parse failure — substrate continues
/// (`feedback_no_hardcoded_values_emergence_over_determinism` — config
/// gains OK, defaults are a safe baseline, not a signal floor).
pub fn load_for_layer(shm_dir: &Path, layer: Layer) -> RestoringCfg {
    let mut cfg = RestoringCfg::for_layer(layer);
    let path = shm_dir.join(TRINITY_RESTORING_SIDECAR);
    match std::fs::read(&path) {
        Ok(bytes) if bytes.len() >= TRINITY_RESTORING_PAYLOAD_BYTES => {
            let mut g = [0.0_f32; 8];
            for i in 0..8 {
                let off = i * 4;
                g[i] = f32::from_le_bytes(
                    bytes[off..off + 4]
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
                reason = "sidecar payload < 32 bytes",
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
        let gains = [0.42_f32, 0.07, 0.06, 0.11, 0.04, 0.55, 1.10, 0.45];
        let mut bytes = Vec::with_capacity(32);
        for v in gains.iter() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        fs::write(&path, &bytes).unwrap();
        let cfg = load_for_layer(dir.path(), Layer::Spirit);
        assert!((cfg.k_drive - 0.42).abs() < 1e-6);
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
    fn short_sidecar_falls_back() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(TRINITY_RESTORING_SIDECAR);
        fs::write(&path, vec![0_u8; 16]).unwrap();
        let cfg = load_for_layer(dir.path(), Layer::Mind);
        assert!((cfg.k_drive - crate::homeostasis::DEFAULT_K_DRIVE).abs() < 1e-6);
    }
}
