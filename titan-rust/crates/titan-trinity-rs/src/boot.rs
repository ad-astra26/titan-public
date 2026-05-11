//! boot — Substrate boot sequence per SPEC §10.A B8 (substrate side) + PLAN
//! §7.1 (S0–S14).
//!
//! C-S3 chunk C3-3 ships the SCAFFOLDING for the boot sequence. Subsequent
//! chunks fill in:
//!   - C3-4: filter_down + ground_up coordination primitive init
//!   - C3-5: topology engine + sphere clocks + chi state init + tick_loop
//!   - C3-6: fastbus consumer attach + main bus client connect + MODULE_READY

use std::path::PathBuf;

use thiserror::Error;
use tracing::{info, warn};

use crate::cli::Cli;

/// Substrate boot errors. Each maps to a [`crate::exit::SubstrateExitCode`]
/// per SPEC §15.
#[derive(Debug, Error)]
pub enum BootError {
    /// Required env var or CLI flag missing.
    #[error("substrate config error: {0}")]
    ConfigError(String),

    /// `prctl(PR_SET_PDEATHSIG, SIGTERM)` failed.
    #[error("prctl PDEATHSIG failed: {0}")]
    PdeathSigFailure(String),

    /// `prctl(PR_SET_CHILD_SUBREAPER, 1)` failed (non-fatal, but logged).
    #[error("prctl CHILD_SUBREAPER failed: {0}")]
    SubreaperFailure(String),
}

impl BootError {
    /// Map to canonical exit code per SPEC §15.
    pub fn to_exit_code(&self) -> crate::exit::SubstrateExitCode {
        use crate::exit::SubstrateExitCode;
        match self {
            BootError::ConfigError(_) => SubstrateExitCode::ConfigError,
            BootError::PdeathSigFailure(_) => SubstrateExitCode::PdeathSigFailure,
            BootError::SubreaperFailure(_) => SubstrateExitCode::Generic,
        }
    }
}

/// Resolved boot configuration after CLI parsing + env resolution.
///
/// Captured ONCE at boot from the env snapshot — supervise.rs uses these
/// fields directly rather than re-reading global env at spawn time. Avoids
/// test-parallelism flakes where another test mutates the global env.
#[derive(Debug, Clone)]
pub struct BootConfig {
    /// Titan ID (T1 / T2 / T3).
    pub titan_id: String,
    /// Path to fastbus.bin (kernel-pre-created).
    pub fastbus_path: PathBuf,
    /// Shm dir (`/dev/shm/titan_<id>/`).
    pub shm_dir: PathBuf,
    /// Data dir (default `data/`).
    pub data_dir: Option<PathBuf>,
    /// Optional main bus socket path (used in C3-6).
    pub bus_socket: Option<PathBuf>,
    /// Boot generation (monotonic counter from kernel via env).
    pub boot_generation: u64,
    /// Kernel's PID (substrate's parent).
    pub parent_pid: Option<u32>,
    /// HKDF-derived bus authkey hex (kernel passes via env per SPEC §3 D08).
    /// `None` when running standalone in tests.
    pub authkey_hex: Option<String>,
    /// Log level passed by kernel (`info` / `debug` / etc.).
    pub log_level: Option<String>,
    /// Path to `titan-unified-spirit-rs` binary (resolved
    /// relative to substrate binary if not provided).
    pub unified_spirit_binary: PathBuf,
    /// Whether to spawn the unified-spirit-placeholder child.
    pub spawn_unified_spirit: bool,
}

impl BootConfig {
    /// Resolve from CLI + env. Per PLAN §7.1 step S1.
    pub fn from_cli(cli: &Cli) -> Result<Self, BootError> {
        let titan_id = cli
            .resolve_titan_id()
            .ok_or_else(|| BootError::ConfigError("missing TITAN_KERNEL_TITAN_ID".into()))?;
        let fastbus_path = cli
            .resolve_fastbus_path()
            .ok_or_else(|| BootError::ConfigError("missing TITAN_KERNEL_FASTBUS_PATH".into()))?;
        let shm_dir = cli
            .resolve_shm_dir()
            .ok_or_else(|| BootError::ConfigError("missing TITAN_KERNEL_SHM_DIR".into()))?;
        let bus_socket = cli.resolve_bus_socket();
        let boot_generation = std::env::var("TITAN_KERNEL_BOOT_GENERATION")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let parent_pid = std::env::var("TITAN_DAEMON_PARENT_PID")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());
        let authkey_hex = std::env::var("TITAN_AUTHKEY_HEX").ok();
        let log_level = std::env::var("TITAN_KERNEL_LOG_LEVEL").ok();
        let data_dir = cli.data_dir.clone().or_else(|| {
            std::env::var("TITAN_KERNEL_DATA_DIR")
                .ok()
                .map(PathBuf::from)
        });
        let unified_spirit_binary = cli
            .unified_spirit_binary
            .clone()
            .or_else(|| {
                std::env::current_exe()
                    .ok()
                    .and_then(|exe| exe.parent().map(|p| p.join("titan-unified-spirit-rs")))
            })
            .unwrap_or_else(|| PathBuf::from("titan-unified-spirit-rs"));
        let spawn_unified_spirit = !cli.skip_unified_spirit_spawn;

        Ok(BootConfig {
            titan_id,
            fastbus_path,
            shm_dir,
            data_dir,
            bus_socket,
            boot_generation,
            parent_pid,
            authkey_hex,
            log_level,
            unified_spirit_binary,
            spawn_unified_spirit,
        })
    }
}

/// Step S2: install `prctl(PR_SET_PDEATHSIG, SIGTERM)` + (best-effort)
/// `prctl(PR_SET_CHILD_SUBREAPER, 1)` so substrate dies if kernel dies and
/// can reap orphaned descendants.
///
/// Per SPEC §11.C(1) cascade rules + PLAN §7.1 step S2.
pub fn install_prctl() -> Result<(), BootError> {
    use nix::sys::signal::Signal;

    titan_core::supervisor::prctl_unix::set_pdeathsig(Signal::SIGTERM)
        .map_err(|e| BootError::PdeathSigFailure(format!("{e:?}")))?;

    // CHILD_SUBREAPER is best-effort; failure is logged but non-fatal in C-S3
    // because substrate's only child is the unified-spirit-placeholder and we
    // wait on it directly. C-S5+ daemons need this to be reliable.
    if let Err(e) = titan_core::supervisor::prctl_unix::set_child_subreaper(true) {
        warn!(
            err = ?e,
            "prctl(PR_SET_CHILD_SUBREAPER, 1) failed — non-fatal in C-S3"
        );
    }
    Ok(())
}

/// Logging banner emitted at boot start. Per PLAN §7.1 step S3 + SPEC §16.
pub fn log_boot_starting(cfg: &BootConfig) {
    info!(
        event = "BOOT_STARTING",
        binary = "trinity-substrate",
        titan_id = cfg.titan_id.as_str(),
        boot_generation = cfg.boot_generation,
        parent_pid = ?cfg.parent_pid,
        cargo_version = crate::version::CARGO_VERSION,
        git_sha = crate::version::GIT_SHA,
        spec_version = titan_core::constants::SPEC_VERSION,
        full_version = %crate::version::full_version(),
        "titan-trinity-rs boot starting"
    );
    info!(
        fastbus_path = ?cfg.fastbus_path,
        shm_dir = ?cfg.shm_dir,
        bus_socket = ?cfg.bus_socket,
        unified_spirit_binary = ?cfg.unified_spirit_binary,
        spawn_unified_spirit = cfg.spawn_unified_spirit,
        "boot configuration resolved"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn make_cli(args: &[&str]) -> Cli {
        let mut full = vec!["titan-trinity-rs"];
        full.extend(args);
        Cli::try_parse_from(full).expect("parse")
    }

    #[test]
    fn boot_config_requires_titan_id() {
        // Clear env so env fallback can't supply it
        std::env::remove_var("TITAN_KERNEL_TITAN_ID");
        std::env::remove_var("TITAN_KERNEL_FASTBUS_PATH");
        std::env::remove_var("TITAN_KERNEL_SHM_DIR");
        let cli = make_cli(&[
            "--fastbus-path",
            "/tmp/ignored",
            "--shm-dir",
            "/tmp/ignored",
        ]);
        let err = BootConfig::from_cli(&cli).unwrap_err();
        assert!(matches!(err, BootError::ConfigError(_)));
        assert_eq!(err.to_exit_code() as u8, 2);
    }

    #[test]
    fn boot_config_resolves_from_cli() {
        std::env::remove_var("TITAN_KERNEL_TITAN_ID");
        std::env::remove_var("TITAN_KERNEL_FASTBUS_PATH");
        std::env::remove_var("TITAN_KERNEL_SHM_DIR");
        std::env::remove_var("TITAN_KERNEL_BUS_SOCKET_PATH");
        std::env::remove_var("TITAN_KERNEL_BOOT_GENERATION");
        std::env::remove_var("TITAN_DAEMON_PARENT_PID");
        let cli = make_cli(&[
            "--titan-id",
            "T1",
            "--fastbus-path",
            "/tmp/fastbus.bin",
            "--shm-dir",
            "/tmp/shm",
        ]);
        let cfg = BootConfig::from_cli(&cli).expect("config");
        assert_eq!(cfg.titan_id, "T1");
        assert_eq!(cfg.fastbus_path, PathBuf::from("/tmp/fastbus.bin"));
        assert_eq!(cfg.shm_dir, PathBuf::from("/tmp/shm"));
        assert_eq!(cfg.boot_generation, 0);
        assert!(cfg.parent_pid.is_none());
        assert!(cfg.spawn_unified_spirit);
    }

    #[test]
    fn boot_config_skip_unified_spirit_flag_honored() {
        std::env::remove_var("TITAN_KERNEL_TITAN_ID");
        std::env::remove_var("TITAN_KERNEL_FASTBUS_PATH");
        std::env::remove_var("TITAN_KERNEL_SHM_DIR");
        let cli = make_cli(&[
            "--titan-id",
            "T2",
            "--fastbus-path",
            "/tmp/fastbus.bin",
            "--shm-dir",
            "/tmp/shm",
            "--skip-unified-spirit-spawn",
        ]);
        let cfg = BootConfig::from_cli(&cli).expect("config");
        assert!(!cfg.spawn_unified_spirit);
    }

    #[test]
    fn install_prctl_is_idempotent_on_linux() {
        // Best-effort — don't fail the test on non-Linux dev machines.
        let r1 = install_prctl();
        let r2 = install_prctl();
        // Either both succeed or both are PdeathSigFailure (non-Linux dev).
        match (r1, r2) {
            (Ok(()), Ok(())) => {}
            (Err(BootError::PdeathSigFailure(_)), Err(BootError::PdeathSigFailure(_))) => {}
            other => panic!("unexpected install_prctl result pair: {other:?}"),
        }
    }
}
