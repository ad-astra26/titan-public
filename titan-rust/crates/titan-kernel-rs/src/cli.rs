//! cli — `clap`-derived CLI per SPEC §13 (per-binary CLI contract).
//!
//! Mandatory flags every Rust binary accepts (SPEC §13):
//! - `--titan-id <id>`  (free-form, path-safe `[A-Za-z0-9_-]`; fleet uses
//!   `T1`/`T2`/`T3`, a sovereign user's Titan picks its own, e.g. `titan`)
//! - `--shm-dir <path>`
//! - `--bus-socket <path>`
//! - `--log-level <debug|info|warn|error>`
//! - `--config <path>`
//! - `--version`
//! - `--help`
//!
//! Plus kernel-specific:
//! - `--kernel-rpc-socket <path>` — defaults to `/tmp/titan_kernel_<id>.sock`
//! - `--data-dir <path>` — defaults to `data/`

use std::path::PathBuf;

use clap::Parser;

/// Log level, mirrors `tracing::Level`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "lowercase")]
pub enum LogLevel {
    /// Maximum verbosity.
    Debug,
    /// Default — informational + warnings + errors.
    Info,
    /// Warnings + errors only.
    Warn,
    /// Errors only.
    Error,
}

impl LogLevel {
    /// Convert to `tracing::Level`.
    pub fn to_tracing(self) -> tracing::Level {
        match self {
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Info => tracing::Level::INFO,
            LogLevel::Warn => tracing::Level::WARN,
            LogLevel::Error => tracing::Level::ERROR,
        }
    }
}

/// Top-level CLI per SPEC §13.
///
/// `--version` prints the cargo crate version. The kernel emits a richer
/// version line at boot via [`crate::version::full_version`] (includes git
/// SHA + SPEC version + SPEC SHA prefix) for ops/observability.
#[derive(Debug, Parser)]
#[command(
    name = "titan-kernel-rs",
    about = "Titan microkernel v2 Phase C — Rust L0 kernel binary",
    long_about = None,
    version,
)]
pub struct Cli {
    /// Titan ID — the fleet's `T1`/`T2`/`T3` or a sovereign user's own id
    /// (the installer defaults to `titan`). 1–32 of `[A-Za-z0-9_-]`. Required
    /// either via this flag OR the `TITAN_KERNEL_TITAN_ID` env.
    #[arg(
        long,
        env = "TITAN_KERNEL_TITAN_ID",
        value_parser = titan_core::identity::validate_titan_id
    )]
    pub titan_id: String,

    /// Override `TITAN_KERNEL_SHM_DIR` (default: `/dev/shm/titan_<id>/`).
    #[arg(long, env = "TITAN_KERNEL_SHM_DIR")]
    pub shm_dir: Option<PathBuf>,

    /// Override main bus socket path (default: `/tmp/titan_bus_<id>.sock`).
    #[arg(long, env = "TITAN_KERNEL_BUS_SOCKET_PATH")]
    pub bus_socket: Option<PathBuf>,

    /// Override kernel RPC socket path (default: `/tmp/titan_kernel_<id>.sock`).
    ///
    /// **Reserved for future use (Phase C C-S7 Gap 9):** the Rust kernel
    /// does NOT bind a kernel_rpc listener yet. The Python plugin
    /// (titan_HCL) keeps owning `/tmp/titan_kernel_<id>.sock` for the
    /// API subprocess to connect to. Pass this flag for forward-compat
    /// only; it is currently a no-op aside from path resolution. Full
    /// Rust kernel_rpc server is scheduled for Phase C C-S8 / Phase D.
    #[arg(long)]
    pub kernel_rpc_socket: Option<PathBuf>,

    /// Override data directory (default: `data/`).
    #[arg(long, env = "TITAN_KERNEL_DATA_DIR", default_value = "data")]
    pub data_dir: PathBuf,

    /// Logging verbosity.
    #[arg(long, env = "TITAN_KERNEL_LOG_LEVEL", value_enum, default_value_t = LogLevel::Info)]
    pub log_level: LogLevel,

    /// Path to `titan_hcl/config.toml`.
    #[arg(long, default_value = "titan_hcl/config.toml")]
    pub config: PathBuf,
}

impl Cli {
    /// Resolve `shm_dir` to its effective path: if not provided,
    /// `/dev/shm/titan_<id>/`.
    pub fn effective_shm_dir(&self) -> PathBuf {
        self.shm_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("/dev/shm/titan_{}/", self.titan_id.as_str())))
    }

    /// Resolve `bus_socket` to its effective path: if not provided,
    /// `/tmp/titan_bus_<id>.sock`.
    pub fn effective_bus_socket(&self) -> PathBuf {
        self.bus_socket.clone().unwrap_or_else(|| {
            PathBuf::from(format!("/tmp/titan_bus_{}.sock", self.titan_id.as_str()))
        })
    }

    /// Resolve `kernel_rpc_socket` to its effective path: if not provided,
    /// `/tmp/titan_kernel_<id>.sock`.
    pub fn effective_kernel_rpc_socket(&self) -> PathBuf {
        self.kernel_rpc_socket.clone().unwrap_or_else(|| {
            PathBuf::from(format!("/tmp/titan_kernel_{}.sock", self.titan_id.as_str()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimum_required_args() {
        let cli = Cli::try_parse_from(["titan-kernel-rs", "--titan-id", "T1"]).unwrap();
        assert_eq!(cli.titan_id, "T1");
        assert_eq!(cli.log_level, LogLevel::Info);
    }

    #[test]
    fn parses_sovereign_titan_id() {
        // A sovereign user's Titan id (the installer default) must parse —
        // this is the case the closed T1/T2/T3 enum rejected (kernel exit 2).
        let cli = Cli::try_parse_from(["titan-kernel-rs", "--titan-id", "titan"]).unwrap();
        assert_eq!(cli.titan_id, "titan");
        assert_eq!(
            cli.effective_shm_dir(),
            PathBuf::from("/dev/shm/titan_titan/")
        );
        assert_eq!(
            cli.effective_bus_socket(),
            PathBuf::from("/tmp/titan_bus_titan.sock")
        );
    }

    #[test]
    fn parses_explicit_overrides() {
        let cli = Cli::try_parse_from([
            "titan-kernel-rs",
            "--titan-id",
            "T2",
            "--shm-dir",
            "/tmp/test_shm",
            "--bus-socket",
            "/tmp/test_bus.sock",
            "--log-level",
            "debug",
            "--data-dir",
            "/tmp/data",
        ])
        .unwrap();
        assert_eq!(cli.titan_id, "T2");
        assert_eq!(cli.shm_dir, Some(PathBuf::from("/tmp/test_shm")));
        assert_eq!(cli.bus_socket, Some(PathBuf::from("/tmp/test_bus.sock")));
        assert_eq!(cli.log_level, LogLevel::Debug);
        assert_eq!(cli.data_dir, PathBuf::from("/tmp/data"));
    }

    #[test]
    fn rejects_invalid_titan_id() {
        // Path-unsafe / empty ids are rejected by the shared value_parser.
        // (NB: "T9" is now a VALID free-form id — the old closed-enum
        // rejection of it was the fleet-ism this fix removes.)
        for bad in &["bad/id", "has space", "a.b", ""] {
            let result = Cli::try_parse_from(["titan-kernel-rs", "--titan-id", bad]);
            assert!(result.is_err(), "expected {bad:?} to be rejected");
        }
    }

    #[test]
    fn rejects_missing_titan_id() {
        let result = Cli::try_parse_from(["titan-kernel-rs"]);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_invalid_log_level() {
        let result = Cli::try_parse_from([
            "titan-kernel-rs",
            "--titan-id",
            "T1",
            "--log-level",
            "trace", // not in our enum
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn effective_shm_dir_default() {
        let cli = Cli::try_parse_from(["titan-kernel-rs", "--titan-id", "T2"]).unwrap();
        assert_eq!(cli.effective_shm_dir(), PathBuf::from("/dev/shm/titan_T2/"));
    }

    #[test]
    fn effective_shm_dir_override() {
        let cli = Cli::try_parse_from([
            "titan-kernel-rs",
            "--titan-id",
            "T1",
            "--shm-dir",
            "/custom/path",
        ])
        .unwrap();
        assert_eq!(cli.effective_shm_dir(), PathBuf::from("/custom/path"));
    }

    #[test]
    fn effective_bus_socket_default() {
        let cli = Cli::try_parse_from(["titan-kernel-rs", "--titan-id", "T3"]).unwrap();
        assert_eq!(
            cli.effective_bus_socket(),
            PathBuf::from("/tmp/titan_bus_T3.sock")
        );
    }

    #[test]
    fn effective_kernel_rpc_socket_default() {
        let cli = Cli::try_parse_from(["titan-kernel-rs", "--titan-id", "T1"]).unwrap();
        assert_eq!(
            cli.effective_kernel_rpc_socket(),
            PathBuf::from("/tmp/titan_kernel_T1.sock")
        );
    }

    #[test]
    fn log_level_to_tracing() {
        assert_eq!(LogLevel::Debug.to_tracing(), tracing::Level::DEBUG);
        assert_eq!(LogLevel::Info.to_tracing(), tracing::Level::INFO);
        assert_eq!(LogLevel::Warn.to_tracing(), tracing::Level::WARN);
        assert_eq!(LogLevel::Error.to_tracing(), tracing::Level::ERROR);
    }
}
