//! cli — Substrate command-line interface per SPEC §13 + PLAN §7.3.
//!
//! Substrate is mostly env-driven (kernel passes everything via env at
//! spawn-time), but exposes the canonical CLI flags so it can be invoked
//! standalone for debugging + smoke testing.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};

/// Substrate CLI per SPEC §13.
#[derive(Debug, Parser)]
#[command(name = "titan-trinity-rs")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Trinity substrate L1a binary — supervised by titan-kernel-rs")]
pub struct Cli {
    /// Override `TITAN_KERNEL_TITAN_ID` env (T1 / T2 / T3).
    #[arg(long, env = "TITAN_KERNEL_TITAN_ID", default_value = "")]
    pub titan_id: String,

    /// Override `TITAN_KERNEL_BUS_SOCKET_PATH` env.
    #[arg(long, env = "TITAN_KERNEL_BUS_SOCKET_PATH")]
    pub bus_socket: Option<PathBuf>,

    /// Override `TITAN_KERNEL_FASTBUS_PATH` env (kernel-pre-created
    /// `/dev/shm/titan_<id>/fastbus.bin`).
    #[arg(long, env = "TITAN_KERNEL_FASTBUS_PATH")]
    pub fastbus_path: Option<PathBuf>,

    /// Override `TITAN_KERNEL_SHM_DIR` env.
    #[arg(long, env = "TITAN_KERNEL_SHM_DIR")]
    pub shm_dir: Option<PathBuf>,

    /// Override `TITAN_KERNEL_DATA_DIR` env (default `data/`).
    #[arg(long, env = "TITAN_KERNEL_DATA_DIR")]
    pub data_dir: Option<PathBuf>,

    /// Path to the supervised `titan-unified-spirit-rs` binary.
    /// Default: same dir as substrate binary.
    #[arg(long, env = "TITAN_SUBSTRATE_UNIFIED_SPIRIT_BINARY")]
    pub unified_spirit_binary: Option<PathBuf>,

    /// Logging level — overrides `TITAN_KERNEL_LOG_LEVEL`.
    #[arg(long, env = "TITAN_KERNEL_LOG_LEVEL", default_value = "info")]
    pub log_level: LogLevel,

    /// Disable spawning the unified-spirit-placeholder child (smoke testing).
    /// Substrate's supervisor primitive remains wired but managed-children = 0.
    #[arg(long, default_value_t = false)]
    pub skip_unified_spirit_spawn: bool,

    /// Auto-shutdown after this many seconds. `0` = wait for SIGTERM.
    /// Used by integration tests; not a SPEC §13 flag.
    #[arg(long, default_value_t = 0)]
    pub auto_shutdown_after_s: u64,
}

/// Log level — mirrors `titan-kernel-rs::cli::LogLevel`.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LogLevel {
    /// `debug` — verbose, dev only.
    Debug,
    /// `info` — default production level.
    Info,
    /// `warn` — anomalies only.
    Warn,
    /// `error` — failures only.
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

    /// Canonical lowercase string.
    pub fn as_str(self) -> &'static str {
        match self {
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

impl Cli {
    /// Resolve `--titan-id` falling back to env. Returns `None` if neither
    /// CLI nor env provides one — caller emits the SPEC §15 ConfigError.
    pub fn resolve_titan_id(&self) -> Option<String> {
        if !self.titan_id.is_empty() {
            return Some(self.titan_id.clone());
        }
        // clap pulls env automatically if `env=` is set, but our default_value
        // is empty so this only triggers if env is also unset.
        std::env::var("TITAN_KERNEL_TITAN_ID")
            .ok()
            .filter(|s| !s.is_empty())
    }

    /// Resolve fastbus path. Required (caller emits ConfigError if missing).
    pub fn resolve_fastbus_path(&self) -> Option<PathBuf> {
        self.fastbus_path.clone().or_else(|| {
            std::env::var("TITAN_KERNEL_FASTBUS_PATH")
                .ok()
                .map(PathBuf::from)
        })
    }

    /// Resolve shm dir. Required (caller emits ConfigError if missing).
    pub fn resolve_shm_dir(&self) -> Option<PathBuf> {
        self.shm_dir.clone().or_else(|| {
            std::env::var("TITAN_KERNEL_SHM_DIR")
                .ok()
                .map(PathBuf::from)
        })
    }

    /// Resolve bus socket path. Optional in C-S3 (substrate connects in C3-6).
    pub fn resolve_bus_socket(&self) -> Option<PathBuf> {
        self.bus_socket.clone().or_else(|| {
            std::env::var("TITAN_KERNEL_BUS_SOCKET_PATH")
                .ok()
                .map(PathBuf::from)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_command_compiles() {
        Cli::command().debug_assert();
    }

    #[test]
    fn log_level_to_tracing_round_trip() {
        assert_eq!(LogLevel::Debug.to_tracing(), tracing::Level::DEBUG);
        assert_eq!(LogLevel::Info.to_tracing(), tracing::Level::INFO);
        assert_eq!(LogLevel::Warn.to_tracing(), tracing::Level::WARN);
        assert_eq!(LogLevel::Error.to_tracing(), tracing::Level::ERROR);
    }

    #[test]
    fn log_level_as_str() {
        assert_eq!(LogLevel::Debug.as_str(), "debug");
        assert_eq!(LogLevel::Info.as_str(), "info");
        assert_eq!(LogLevel::Warn.as_str(), "warn");
        assert_eq!(LogLevel::Error.as_str(), "error");
    }
}
