//! spawn — Spawn `titan-trinity-rs` + `python -u scripts/guardian_hcl.py`
//! per SPEC §10.A B8 + §11.B.4 INV-PROC-3. Sets all canonical env vars
//! per §3 D18 + §5.
//!
//! C-S2 shipped the spawn pathway against `titan-trinity-rs-placeholder`;
//! C-S3 chunk C3-3 renamed the placeholder to `titan-trinity-rs` and filled in
//! the real substrate body. Phase 6 (D-SPEC-135 / v1.62.0) renamed the
//! Python child entry from `scripts/titan_hcl.py --server` to
//! `scripts/guardian_hcl.py` — kernel-rs now spawns the L1 supervisor
//! (guardian_hcl) which in turn supervises L2 (titan_hcl) + L3
//! (titan_hcl_api) via the module catalog (INV-PROC-3). The field
//! [`SpawnConfig::spawn_guardian_hcl`] gates this child.

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
    /// Path to Python executable (for `python -u scripts/guardian_hcl.py`).
    pub python_executable: Option<PathBuf>,
    /// CWD for Python child (where `scripts/guardian_hcl.py` lives).
    pub python_cwd: Option<PathBuf>,
    /// `false` = skip spawning the L1 supervisor (used by tests).
    /// Phase 6 / D-SPEC-135: gates `scripts/guardian_hcl.py` (was
    /// `spawn_python_main` gating `scripts/titan_hcl.py --server`).
    pub spawn_guardian_hcl: bool,
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
    /// Python L1 supervisor `guardian_hcl` (optional; when
    /// `spawn_guardian_hcl=true`). Phase 6 / D-SPEC-135 / v1.62.0 — was
    /// `python_main` gating `titan_hcl.py --server`. Field name retained
    /// for binary-on-disk handle compatibility.
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

/// Spawn `python -u scripts/guardian_hcl.py` per SPEC §11.B.4 INV-PROC-3
/// (Phase 6 / D-SPEC-135 / v1.62.0). Returns `Ok(None)` if
/// `spawn_guardian_hcl=false` (tests).
///
/// Pre-Phase-6 kernel-rs spawned `titan_hcl.py --server` directly. Phase 6
/// inserts `guardian_hcl` as the L1 supervisor between kernel-rs (L0) and
/// titan_hcl (L2). guardian_hcl in turn spawns titan_hcl and titan_hcl_api
/// as Guardian-supervised module children — see
/// `titan_hcl/module_catalog.py:build_catalog`.
pub fn spawn_guardian_hcl(config: &SpawnConfig) -> Result<Option<Child>, SpawnError> {
    if !config.spawn_guardian_hcl {
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
    let mut env = config.build_child_env("guardian_hcl");
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
    // Phase 6 (D-SPEC-135 / v1.62.0): kernel-rs spawns the L1 supervisor
    // (`scripts/guardian_hcl.py`) per SPEC §11.B.4 INV-PROC-3. guardian_hcl
    // boots BEFORE titan_hcl and titan_hcl_api and spawns both as
    // Guardian-supervised module children. No `--server` flag —
    // guardian_hcl is non-interactive by default. `-u` = unbuffered
    // stdout/stderr so journald + tee'd log files receive lines as they're
    // written. Path-based invocation avoids PYTHONPATH coordination.
    cmd.arg("-u")
        .arg("scripts/guardian_hcl.py")
        .current_dir(&cwd)
        .env_clear()
        .envs(env)
        .kill_on_drop(false);

    info!(
        python = ?python,
        cwd = ?cwd,
        titan_id = %config.titan_id,
        "kernel: spawning python -u scripts/guardian_hcl.py per Phase 6 §11.B.4 INV-PROC-3 (D-SPEC-135 / v1.62.0)"
    );

    let child = cmd.spawn().map_err(|source| {
        warn!(err = ?source, "guardian_hcl spawn failed");
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
            spawn_guardian_hcl: false,
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
    fn spawn_guardian_hcl_returns_none_when_disabled() {
        let cfg = test_config();
        let result = spawn_guardian_hcl(&cfg).unwrap();
        assert!(result.is_none());
    }
}
