//! spawn — Spawn `titan-trinity-rs` + `python -m titan_hcl` per SPEC §10.A
//! B8 + B9. Sets all canonical env vars per §3 D18 + §5.
//!
//! C-S2 shipped the spawn pathway against `titan-trinity-rs-placeholder`;
//! C-S3 chunk C3-3 renamed the placeholder to `titan-trinity-rs` and filled in
//! the real substrate body. Python child is optional — controlled by
//! [`SpawnConfig::spawn_python_main`].

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::process::{Child, Command};
use tracing::{info, warn};

/// Errors during child spawn.
#[derive(Debug, thiserror::Error)]
pub enum SpawnError {
    /// Failed to spawn the child process (binary not found, perms, etc.).
    #[error("spawn {binary} failed: {source}")]
    SpawnFailed {
        /// Binary path attempted.
        binary: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },
}

/// Spawn-time configuration.
#[derive(Debug, Clone)]
pub struct SpawnConfig {
    /// Titan ID (T1/T2/T3).
    pub titan_id: String,
    /// Boot generation (monotonic counter passed to children).
    pub boot_generation: u64,
    /// Main bus socket path.
    pub bus_socket: PathBuf,
    /// Fast bus path.
    pub fastbus_path: PathBuf,
    /// Shm root dir.
    pub shm_dir: PathBuf,
    /// Data dir.
    pub data_dir: PathBuf,
    /// HKDF-derived authkey hex (32 bytes encoded as 64 hex chars).
    pub authkey_hex: String,
    /// Log level passed to children via env.
    pub log_level: String,
    /// Path to `titan-trinity-rs` binary (renamed from
    /// `titan-trinity-rs-placeholder` in C-S3 chunk C3-3).
    pub substrate_binary: PathBuf,
    /// Path to Python executable (for `python -m titan_hcl`).
    pub python_executable: Option<PathBuf>,
    /// CWD for Python child (where `titan_hcl/` lives).
    pub python_cwd: Option<PathBuf>,
    /// `false` = skip spawning Python (used by tests).
    pub spawn_python_main: bool,
}

impl SpawnConfig {
    /// Compute the canonical env var set per SPEC §5 + §3 D18 (sets BOTH
    /// new + legacy names during Phase C transition).
    pub fn build_child_env(&self, daemon_name: &str) -> HashMap<String, String> {
        let mut env = HashMap::new();
        // Canonical names per SPEC §5
        env.insert("TITAN_KERNEL_TITAN_ID".into(), self.titan_id.clone());
        env.insert(
            "TITAN_KERNEL_BOOT_GENERATION".into(),
            self.boot_generation.to_string(),
        );
        env.insert(
            "TITAN_KERNEL_BUS_SOCKET_PATH".into(),
            self.bus_socket.to_string_lossy().into_owned(),
        );
        env.insert(
            "TITAN_KERNEL_FASTBUS_PATH".into(),
            self.fastbus_path.to_string_lossy().into_owned(),
        );
        env.insert(
            "TITAN_KERNEL_SHM_DIR".into(),
            self.shm_dir.to_string_lossy().into_owned(),
        );
        env.insert(
            "TITAN_KERNEL_DATA_DIR".into(),
            self.data_dir.to_string_lossy().into_owned(),
        );
        env.insert("TITAN_KERNEL_LOG_LEVEL".into(), self.log_level.clone());
        env.insert("TITAN_AUTHKEY_HEX".into(), self.authkey_hex.clone());
        env.insert("TITAN_DAEMON_NAME".into(), daemon_name.into());
        env.insert(
            "TITAN_DAEMON_PARENT_PID".into(),
            std::process::id().to_string(),
        );

        // Legacy aliases per SPEC §3 D18 (kernel sets BOTH; consumers can
        // read either; old names removed in C-S8)
        env.insert("TITAN_BUS_TITAN_ID".into(), self.titan_id.clone());
        env.insert("TITAN_ID".into(), self.titan_id.clone());
        env.insert(
            "TITAN_SHM_ROOT".into(),
            self.shm_dir.to_string_lossy().into_owned(),
        );
        env.insert(
            "TITAN_DATA_DIR".into(),
            self.data_dir.to_string_lossy().into_owned(),
        );

        // glibc arena cap per CLAUDE.md / 2026-04-27 fix
        env.insert("MALLOC_ARENA_MAX".into(), "2".into());

        // Phase C C-S7 (2026-05-05) — forward TITAN_DAEMON_BINARY_DIR if
        // systemd unit set it (Environment=TITAN_DAEMON_BINARY_DIR=...).
        // Substrate forwards this onward to unified-spirit-rs which reads
        // it via clap env attribute. Removes /usr/local/bin symlink dep.
        if let Ok(daemon_bin_dir) = std::env::var("TITAN_DAEMON_BINARY_DIR") {
            env.insert("TITAN_DAEMON_BINARY_DIR".into(), daemon_bin_dir);
        }

        env
    }
}

/// Tracks the spawned children so shutdown can SIGTERM them.
pub struct SpawnedChildren {
    /// Substrate (placeholder in C-S2; real titan-trinity-rs in C-S3).
    pub substrate: Mutex<Option<Child>>,
    /// Python `titan_hcl` (optional; when `spawn_python_main=true`).
    pub python_main: Mutex<Option<Child>>,
}

impl SpawnedChildren {
    /// New empty registry.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            substrate: Mutex::new(None),
            python_main: Mutex::new(None),
        })
    }

    /// Send SIGTERM to all live children (best-effort).
    pub async fn sigterm_all(&self) {
        if let Some(mut child) = self.substrate.lock().take() {
            let pid = child.id();
            info!(pid, "kernel: sending SIGTERM to substrate");
            let _ = child.start_kill();
        }
        if let Some(mut child) = self.python_main.lock().take() {
            let pid = child.id();
            info!(pid, "kernel: sending SIGTERM to python_main");
            let _ = child.start_kill();
        }
    }
}

impl Default for SpawnedChildren {
    fn default() -> Self {
        Self {
            substrate: Mutex::new(None),
            python_main: Mutex::new(None),
        }
    }
}

/// Spawn the substrate placeholder per SPEC §10.A B8.
pub fn spawn_substrate(config: &SpawnConfig) -> Result<Child, SpawnError> {
    let env = config.build_child_env("trinity-substrate");
    let mut cmd = Command::new(&config.substrate_binary);
    cmd.env_clear().envs(env).kill_on_drop(false);

    info!(
        binary = ?config.substrate_binary,
        titan_id = %config.titan_id,
        "kernel: spawning substrate placeholder per SPEC §10.A B8"
    );

    cmd.spawn().map_err(|source| SpawnError::SpawnFailed {
        binary: config.substrate_binary.to_string_lossy().into_owned(),
        source,
    })
}

/// Spawn `python -m titan_hcl` per SPEC §10.A B9. Returns `Ok(None)` if
/// `spawn_python_main=false` (tests).
pub fn spawn_python_main(config: &SpawnConfig) -> Result<Option<Child>, SpawnError> {
    if !config.spawn_python_main {
        return Ok(None);
    }
    let python = config
        .python_executable
        .as_ref()
        .ok_or_else(|| SpawnError::SpawnFailed {
            binary: "python".into(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "python_executable not configured",
            ),
        })?;
    let cwd = config
        .python_cwd
        .as_ref()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    let mut env = config.build_child_env("titan_HCL");
    // Phase C C-S7 (2026-05-05): forward a small set of parent env vars
    // Python needs to function. PATH for subprocess() finding bash/cargo/etc;
    // HOME for ~/.config/* lookups (Solana CLI fallback paths, ~/.cache for
    // pip / Hugging Face / etc); USER for any module that introspects it.
    // We don't forward EVERYTHING (env_clear remains semantically intact;
    // the kernel deliberately scopes child env per SPEC §3 D18) — just the
    // minimum that Python needs to behave like a normal process.
    for key in ["PATH", "HOME", "USER", "LANG", "LC_ALL", "TZ"] {
        if let Ok(v) = std::env::var(key) {
            env.entry(key.into()).or_insert(v);
        }
    }

    let mut cmd = Command::new(python);
    // Phase C C-S7 (2026-05-05): invoke as a script (`python -u
    // scripts/titan_hcl.py --server`) to match the legacy boot path
    // (t3_manage.sh / titan_watchdog.sh). `-u` = unbuffered stdout/stderr
    // so journald + tee'd log file receive lines as they're written.
    // `--server` = non-interactive mode (no stdin prompt). Path-based
    // invocation avoids the PYTHONPATH coordination needed for the
    // module form (`-m titan_hcl` requires `titan_hcl` to be on
    // sys.path).
    cmd.arg("-u")
        .arg("scripts/titan_hcl.py")
        .arg("--server")
        .current_dir(&cwd)
        .env_clear()
        .envs(env)
        .kill_on_drop(false);

    info!(
        python = ?python,
        cwd = ?cwd,
        titan_id = %config.titan_id,
        "kernel: spawning python -u scripts/titan_hcl.py --server per SPEC §10.A B9"
    );

    let child = cmd.spawn().map_err(|source| {
        warn!(err = ?source, "python_main spawn failed");
        SpawnError::SpawnFailed {
            binary: python.to_string_lossy().into_owned(),
            source,
        }
    })?;
    Ok(Some(child))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SpawnConfig {
        SpawnConfig {
            titan_id: "T1".into(),
            boot_generation: 1,
            bus_socket: PathBuf::from("/tmp/test_bus.sock"),
            fastbus_path: PathBuf::from("/tmp/test_fastbus.bin"),
            shm_dir: PathBuf::from("/tmp/test_shm"),
            data_dir: PathBuf::from("/tmp/test_data"),
            authkey_hex: "00".repeat(32),
            log_level: "info".into(),
            substrate_binary: PathBuf::from("/nonexistent/binary"),
            python_executable: None,
            python_cwd: None,
            spawn_python_main: false,
        }
    }

    #[test]
    fn build_child_env_sets_canonical_and_legacy_names() {
        let cfg = test_config();
        let env = cfg.build_child_env("trinity-substrate");
        assert_eq!(env.get("TITAN_KERNEL_TITAN_ID"), Some(&"T1".to_string()));
        assert_eq!(env.get("TITAN_ID"), Some(&"T1".to_string())); // legacy
        assert_eq!(env.get("TITAN_AUTHKEY_HEX"), Some(&"00".repeat(32)));
        assert_eq!(env.get("MALLOC_ARENA_MAX"), Some(&"2".to_string()));
        assert_eq!(
            env.get("TITAN_DAEMON_NAME"),
            Some(&"trinity-substrate".to_string())
        );
    }

    #[test]
    fn build_child_env_includes_parent_pid() {
        let cfg = test_config();
        let env = cfg.build_child_env("x");
        assert_eq!(
            env.get("TITAN_DAEMON_PARENT_PID"),
            Some(&std::process::id().to_string())
        );
    }

    #[test]
    fn spawn_substrate_fails_for_missing_binary() {
        let cfg = test_config();
        let result = spawn_substrate(&cfg);
        assert!(matches!(result, Err(SpawnError::SpawnFailed { .. })));
    }

    #[test]
    fn spawn_python_main_returns_none_when_disabled() {
        let cfg = test_config();
        let result = spawn_python_main(&cfg).unwrap();
        assert!(result.is_none());
    }
}
