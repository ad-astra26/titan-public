//! cli — `clap`-derived CLI per SPEC §13 + §6 of PLAN_microkernel_phase_c_s4_unified_spirit.md.
//!
//! Mandatory flags every Rust binary accepts (SPEC §13):
//! - `--titan-id <T1|T2|T3>`
//! - `--shm-dir <path>`
//! - `--bus-socket <path>`
//! - `--log-level <debug|info|warn|error>`
//! - `--config <path>`
//! - `--version`
//! - `--help`
//!
//! Plus unified-spirit-specific:
//! - `--data-dir <path>` — defaults to `data/` (atomic_write critical-data
//!   files: unified_spirit_state.json, resonance_state.json,
//!   filter_down_v5_{weights,buffer,state}.json)
//! - `--daemon-binary-dir <path>` — where to find 6 trinity daemon binaries
//!   (default `/usr/local/bin`); C-S5/C-S6 ship the real ones, C-S4 uses
//!   placeholders per the `--use-placeholder-daemons` flag.
//! - `--use-placeholder-daemons` — bool; spawn 6 instances of
//!   `titan-trinity-rs-placeholder` in lieu of real daemons (C-S4 test mode
//!   per PLAN §3.3 Option A).
//! - `--self-assembly-cadence-ms <u64>` — defaults to
//!   `BODY_CYCLE_INTERVAL_MS=1150` per SPEC §10.G.

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

/// Titan ID — restricted to canonical T1/T2/T3 set per SPEC §1 glossary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "UPPERCASE")]
pub enum TitanIdArg {
    /// T1 (origin).
    T1,
    /// T2 (mid-cluster).
    T2,
    /// T3 (T3 cluster).
    T3,
}

impl TitanIdArg {
    /// Canonical short form (`"T1"`, etc.).
    pub fn as_str(&self) -> &'static str {
        match self {
            TitanIdArg::T1 => "T1",
            TitanIdArg::T2 => "T2",
            TitanIdArg::T3 => "T3",
        }
    }
}

/// Top-level CLI per SPEC §13.
#[derive(Debug, Parser)]
#[command(
    name = "titan-unified-spirit-rs",
    about = "Titan microkernel v2 Phase C — Rust L1b unified-spirit binary (SELF orchestrator)",
    long_about = None,
    version,
)]
pub struct Cli {
    /// Titan ID (T1/T2/T3). Required either via this flag OR
    /// `TITAN_KERNEL_TITAN_ID` env.
    #[arg(long, env = "TITAN_KERNEL_TITAN_ID", value_enum)]
    pub titan_id: TitanIdArg,

    /// Override `TITAN_KERNEL_SHM_DIR` (default: `/dev/shm/titan_<id>/`).
    #[arg(long, env = "TITAN_KERNEL_SHM_DIR")]
    pub shm_dir: Option<PathBuf>,

    /// Override main bus socket path (default: `/tmp/titan_bus_<id>.sock`).
    #[arg(long, env = "TITAN_KERNEL_BUS_SOCKET_PATH")]
    pub bus_socket: Option<PathBuf>,

    /// Override data directory (default: `data/`).
    #[arg(long, env = "TITAN_KERNEL_DATA_DIR", default_value = "data")]
    pub data_dir: PathBuf,

    /// Daemon binary search directory.
    ///
    /// Phase C C-S7 (2026-05-05): added `env = "TITAN_DAEMON_BINARY_DIR"`
    /// so the systemd unit / kernel-rs spawn pipeline can override the
    /// /usr/local/bin default without per-Titan symlinks. Per Titan can
    /// have its own bin/ (T1: /home/youruser/projects/titan/bin/,
    /// T3: /home/youruser/projects/titan3/bin/) and the env var
    /// flows kernel → substrate → unified-spirit unchanged.
    #[arg(
        long,
        env = "TITAN_DAEMON_BINARY_DIR",
        default_value = "/usr/local/bin"
    )]
    pub daemon_binary_dir: PathBuf,

    /// If set, spawn 6 instances of `titan-trinity-rs-placeholder` instead
    /// of the real daemon binaries. C-S4 test mode (PLAN §3.3 Option A);
    /// the real daemons ship in C-S5 + C-S6.
    #[arg(long, default_value_t = false)]
    pub use_placeholder_daemons: bool,

    /// Body cycle cadence in milliseconds (default
    /// `BODY_CYCLE_INTERVAL_MS=1150`).
    #[arg(long, default_value_t = titan_core::constants::BODY_CYCLE_INTERVAL_MS)]
    pub self_assembly_cadence_ms: u64,

    /// Log level (`debug` / `info` / `warn` / `error`).
    #[arg(long, env = "TITAN_KERNEL_LOG_LEVEL", value_enum, default_value_t = LogLevel::Info)]
    pub log_level: LogLevel,

    /// Config file path (default: `titan_hcl/config.toml`).
    #[arg(long, default_value = "titan_hcl/config.toml")]
    pub config: PathBuf,
}

impl Cli {
    /// Effective shm directory — `--shm-dir` override OR default
    /// `/dev/shm/titan_<id>/`.
    pub fn effective_shm_dir(&self) -> PathBuf {
        self.shm_dir.clone().unwrap_or_else(|| {
            PathBuf::from(format!(
                "/dev/shm/titan_{}/",
                self.titan_id.as_str().to_lowercase()
            ))
        })
    }

    /// Effective main bus socket path — `--bus-socket` override OR default
    /// `/tmp/titan_bus_<id>.sock`.
    pub fn effective_bus_socket(&self) -> PathBuf {
        self.bus_socket.clone().unwrap_or_else(|| {
            PathBuf::from(format!(
                "/tmp/titan_bus_{}.sock",
                self.titan_id.as_str().to_lowercase()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn cli_parses_minimal_required_flags() {
        // C4-1 test 1: minimum required flag (--titan-id) parses cleanly.
        let cli = Cli::try_parse_from(["titan-unified-spirit-rs", "--titan-id", "T1"]).unwrap();
        assert_eq!(cli.titan_id, TitanIdArg::T1);
        assert_eq!(cli.titan_id.as_str(), "T1");
    }

    #[test]
    fn cli_rejects_invalid_titan_id() {
        // C4-1 test 2: T4/T0/T1A all rejected.
        for invalid in &["T0", "T4", "T1A", "x"] {
            let result = Cli::try_parse_from(["titan-unified-spirit-rs", "--titan-id", invalid]);
            assert!(result.is_err(), "should reject invalid titan_id={invalid}");
        }
    }

    #[test]
    fn cli_titan_id_canonical_short_form() {
        // C4-1 test 3: as_str() returns canonical UPPERCASE short form.
        assert_eq!(TitanIdArg::T1.as_str(), "T1");
        assert_eq!(TitanIdArg::T2.as_str(), "T2");
        assert_eq!(TitanIdArg::T3.as_str(), "T3");
    }

    #[test]
    fn cli_default_paths_per_titan() {
        // C4-1 test 4: default shm + bus paths derive from --titan-id.
        let cli = Cli::try_parse_from(["titan-unified-spirit-rs", "--titan-id", "T2"]).unwrap();
        assert_eq!(cli.effective_shm_dir(), PathBuf::from("/dev/shm/titan_t2/"));
        assert_eq!(
            cli.effective_bus_socket(),
            PathBuf::from("/tmp/titan_bus_t2.sock")
        );
    }

    #[test]
    fn cli_shm_dir_override() {
        // C4-1 test 5: --shm-dir overrides default.
        let cli = Cli::try_parse_from([
            "titan-unified-spirit-rs",
            "--titan-id",
            "T1",
            "--shm-dir",
            "/tmp/test_shm",
        ])
        .unwrap();
        assert_eq!(cli.effective_shm_dir(), PathBuf::from("/tmp/test_shm"));
    }

    #[test]
    fn cli_bus_socket_override() {
        // C4-1 test 6: --bus-socket overrides default.
        let cli = Cli::try_parse_from([
            "titan-unified-spirit-rs",
            "--titan-id",
            "T1",
            "--bus-socket",
            "/tmp/test_bus.sock",
        ])
        .unwrap();
        assert_eq!(
            cli.effective_bus_socket(),
            PathBuf::from("/tmp/test_bus.sock")
        );
    }

    #[test]
    fn cli_log_level_default_is_info() {
        // C4-1 test 7: log-level default is Info per SPEC §16.
        let cli = Cli::try_parse_from(["titan-unified-spirit-rs", "--titan-id", "T1"]).unwrap();
        assert_eq!(cli.log_level, LogLevel::Info);
        assert_eq!(cli.log_level.to_tracing(), tracing::Level::INFO);
    }

    #[test]
    fn cli_self_assembly_cadence_default_matches_spec() {
        // C4-1 test 8: --self-assembly-cadence-ms default = BODY_CYCLE_INTERVAL_MS
        // (SPEC v0.1.3 constant from TOML, SPEC §10.G body publish rate).
        let cli = Cli::try_parse_from(["titan-unified-spirit-rs", "--titan-id", "T1"]).unwrap();
        assert_eq!(
            cli.self_assembly_cadence_ms,
            titan_core::constants::BODY_CYCLE_INTERVAL_MS
        );
        // Sanity: the SPEC constant is 1150 (Schumann/9 ≈ 1.15s).
        assert_eq!(cli.self_assembly_cadence_ms, 1150);
    }
}
