//! config_daemon — in-kernel config-as-SHM-state daemon
//! (RFP_config_as_shm_state_canonical_integrity §7.A / SPEC §14.A.1).
//!
//! Merges the config files (`titan_params.toml ⊎ config.toml` + `~/.titan/secrets.toml`
//! overlay — disjoint by Phase 0, so the params/config part is a UNION), validates
//! against `config_schema.toml` fail-closed, and writes ONE per-top-level-section SHM
//! slot (`/dev/shm/titan_<id>/config/<section>.bin`, `slot.rs` triple-buffer SeqLock,
//! mode 0600). Seeded at boot (B3) before workers spawn; an mtime-poll watch loop
//! re-applies on file change. Single writer = this daemon (G21).
//!
//! Phase A is WRITE-ONLY: no worker reads these slots yet (Phase B repoints reads).
//! So an invalid edit retains last-good slots + emits a loud `[ERR]`, and a boot-time
//! schema failure is logged-but-non-fatal (`fatal_on_invalid=false`) — it cannot break
//! the fleet because nothing depends on the slots. Phase B flips `fatal_on_invalid` on.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use titan_state::Slot;
use tracing::{error, info, warn};

use crate::config_schema::{ConfigError, ConfigSchema};

/// Per-section slot payload capacity (matches the 64KB spirit_supplemental precedent).
pub const CONFIG_SLOT_BYTES: u32 = 65536;
/// Watch-loop mtime poll cadence.
pub const CONFIG_WATCH_POLL_MS: u64 = 1000;

/// The three config source files (microkernel override retired in Phase 0).
#[derive(Debug, Clone)]
pub struct SourcePaths {
    /// `titan_params.toml` (DNA layer).
    pub params: PathBuf,
    /// `config.toml` (user-tunable layer).
    pub config: PathBuf,
    /// `~/.titan/secrets.toml` (the only runtime overlay).
    pub secrets: PathBuf,
}

impl SourcePaths {
    /// Resolve from the kernel's `--config` path: params = sibling
    /// `titan_params.toml`; secrets = `~/.titan/secrets.toml`.
    pub fn resolve(config_path: &Path) -> Self {
        let dir = config_path.parent().unwrap_or_else(|| Path::new("."));
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/root"));
        Self {
            params: dir.join("titan_params.toml"),
            config: config_path.to_path_buf(),
            secrets: home.join(".titan").join("secrets.toml"),
        }
    }

    fn mtimes(&self) -> Vec<Option<SystemTime>> {
        [&self.params, &self.config, &self.secrets]
            .iter()
            .map(|p| std::fs::metadata(p).ok().and_then(|m| m.modified().ok()))
            .collect()
    }
}

/// Recursively merge `overlay` into `base` (later wins), table-deep — mirrors
/// `titan_hcl/config_loader.py:_deep_merge`.
fn deep_merge(base: &mut toml::Value, overlay: &toml::Value) {
    match (base, overlay) {
        (toml::Value::Table(b), toml::Value::Table(o)) => {
            for (k, ov) in o {
                match b.get_mut(k) {
                    Some(bv) if bv.is_table() && ov.is_table() => deep_merge(bv, ov),
                    _ => {
                        b.insert(k.clone(), ov.clone());
                    }
                }
            }
        }
        (b, o) => *b = o.clone(),
    }
}

fn read_toml(path: &Path) -> Result<toml::Value, String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    txt.parse::<toml::Value>()
        .map_err(|e| format!("parse {}: {e}", path.display()))
}

/// Merge {params ⊎ config} + secrets overlay. params + config are disjoint
/// (Phase 0), so the first merge is a union; secrets overlays named keys.
/// A missing secrets file is non-fatal (returns the params⊎config merge).
pub fn merge_config(sources: &SourcePaths) -> Result<toml::Value, String> {
    let mut merged = read_toml(&sources.params).unwrap_or_else(|e| {
        warn!(err = %e, "config_daemon: titan_params.toml unreadable — Layer-1 empty");
        toml::Value::Table(Default::default())
    });
    let config = read_toml(&sources.config)?; // config.toml is required
    deep_merge(&mut merged, &config);
    if sources.secrets.exists() {
        match read_toml(&sources.secrets) {
            Ok(secrets) => deep_merge(&mut merged, &secrets),
            Err(e) => warn!(err = %e, "config_daemon: secrets.toml unreadable — using base config"),
        }
    }
    Ok(merged)
}

/// msgpack-encode every top-level section of the merged config → its slot payload.
fn section_payloads(merged: &toml::Value) -> HashMap<String, Vec<u8>> {
    let mut out = HashMap::new();
    if let toml::Value::Table(t) = merged {
        for (section, value) in t {
            match rmp_serde::to_vec_named(value) {
                Ok(bytes) => {
                    out.insert(section.clone(), bytes);
                }
                Err(e) => {
                    error!(section = %section, err = %e, "config_daemon: msgpack encode failed")
                }
            }
        }
    }
    out
}

/// Dotted keys whose value differs between two merged configs (added/removed/changed).
fn changed_keys(old: &toml::Value, new: &toml::Value) -> Vec<String> {
    let mut o = HashMap::new();
    let mut n = HashMap::new();
    flatten(old, "", &mut o);
    flatten(new, "", &mut n);
    let mut keys: Vec<String> = Vec::new();
    for (k, nv) in &n {
        if o.get(k) != Some(nv) {
            keys.push(k.clone());
        }
    }
    for k in o.keys() {
        if !n.contains_key(k) {
            keys.push(k.clone());
        }
    }
    keys.sort();
    keys.dedup();
    keys
}

fn flatten(val: &toml::Value, prefix: &str, out: &mut HashMap<String, toml::Value>) {
    match val {
        toml::Value::Table(t) => {
            for (k, v) in t {
                let key = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                flatten(v, &key, out);
            }
        }
        toml::Value::Array(arr)
            if !arr.is_empty() && arr.iter().all(|e| matches!(e, toml::Value::Table(_))) =>
        {
            for e in arr {
                flatten(e, &format!("{prefix}[]"), out);
            }
        }
        leaf => {
            out.insert(prefix.to_string(), leaf.clone());
        }
    }
}

fn section_of(dotted_key: &str) -> &str {
    dotted_key.split(['.', '[']).next().unwrap_or(dotted_key)
}

/// Outcome of a (re)apply: what changed + what was rejected.
#[derive(Debug, Default, Clone)]
pub struct ApplyResult {
    /// Top-level sections whose slot was (re)written this apply.
    pub changed_sections: Vec<String>,
    /// Changed keys whose reload class is `restart_required` (surfaced, not hot).
    pub restart_required_keys: Vec<String>,
    /// Validation violations (non-empty ⇒ fail-closed: nothing was written).
    pub errors: Vec<ConfigError>,
}

/// The per-section slot set owned by the kernel. Single-writer.
pub struct ConfigSlotSet {
    config_dir: PathBuf,
    slots: HashMap<String, Slot>,
    last_merged: toml::Value,
    sources: SourcePaths,
    last_mtimes: Vec<Option<SystemTime>>,
}

impl ConfigSlotSet {
    /// Number of per-section slots currently managed.
    pub fn section_count(&self) -> usize {
        self.slots.len()
    }
    /// The `/dev/shm/titan_<id>/config/` directory holding the slot files.
    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }
    /// The resolved source-file paths this set merges from.
    pub fn sources(&self) -> &SourcePaths {
        &self.sources
    }

    /// BOOT SEED (B3): merge → validate → create + write one slot per section.
    /// `fatal_on_invalid` ⇒ propagate a schema failure as Err (Phase B); Phase A
    /// passes `false` (log loudly, still seed — nothing reads the slots yet).
    /// Resolves sources from the kernel `--config` path (params sibling + ~/.titan/secrets.toml).
    pub fn seed(
        shm_dir: &Path,
        config_path: &Path,
        schema: &ConfigSchema,
        fatal_on_invalid: bool,
    ) -> Result<Self, String> {
        Self::seed_with_sources(
            shm_dir,
            SourcePaths::resolve(config_path),
            schema,
            fatal_on_invalid,
        )
    }

    /// Seed from explicit source paths (testable; the public `seed` resolves them).
    pub fn seed_with_sources(
        shm_dir: &Path,
        sources: SourcePaths,
        schema: &ConfigSchema,
        fatal_on_invalid: bool,
    ) -> Result<Self, String> {
        let merged = merge_config(&sources)?;

        let errs = schema.validate(&merged);
        if !errs.is_empty() {
            log_config_errors("seed", &errs);
            if fatal_on_invalid {
                return Err(format!("{} config-schema violation(s) at boot", errs.len()));
            }
        }

        let config_dir = shm_dir.join("config");
        std::fs::create_dir_all(&config_dir)
            .map_err(|e| format!("mkdir {}: {e}", config_dir.display()))?;

        let payloads = section_payloads(&merged);
        let mut slots = HashMap::new();
        for (section, payload) in &payloads {
            let path = config_dir.join(format!("{section}.bin"));
            // Fresh boot: /dev/shm is tmpfs, recreated each boot — create new.
            // If a stale file exists (warm restart in tests), open + overwrite.
            let mut slot = match Slot::create(&path, 1, CONFIG_SLOT_BYTES) {
                Ok(s) => s,
                Err(_) => {
                    Slot::open(&path).map_err(|e| format!("open {}: {e:?}", path.display()))?
                }
            };
            slot.write(payload)
                .map_err(|e| format!("write {}: {e:?}", path.display()))?;
            slots.insert(section.clone(), slot);
        }

        let last_mtimes = sources.mtimes();
        info!(
            event = "CONFIG_SEED_DONE",
            sections = slots.len(),
            invalid = errs.len(),
            "config_daemon: seeded per-section config slots"
        );
        Ok(Self {
            config_dir,
            slots,
            last_merged: merged,
            sources,
            last_mtimes,
        })
    }

    /// Re-merge + validate + write changed slots. Validation errors ⇒ NO write
    /// (last-good retained) + errors returned. Hard IO/parse failure ⇒ Err.
    pub fn reapply(&mut self, schema: &ConfigSchema) -> Result<ApplyResult, String> {
        let merged = merge_config(&self.sources)?;
        let errs = schema.validate(&merged);
        if !errs.is_empty() {
            return Ok(ApplyResult {
                errors: errs,
                ..Default::default()
            });
        }

        let changed = changed_keys(&self.last_merged, &merged);
        if changed.is_empty() {
            return Ok(ApplyResult::default());
        }

        let payloads = section_payloads(&merged);
        let mut changed_sections: Vec<String> =
            changed.iter().map(|k| section_of(k).to_string()).collect();
        changed_sections.sort();
        changed_sections.dedup();

        for section in &changed_sections {
            if let (Some(slot), Some(payload)) =
                (self.slots.get_mut(section), payloads.get(section))
            {
                slot.write(payload)
                    .map_err(|e| format!("write {section}.bin: {e:?}"))?;
            } else if let Some(payload) = payloads.get(section) {
                // a brand-new section appeared at runtime — create its slot
                let path = self.config_dir.join(format!("{section}.bin"));
                let mut slot = Slot::create(&path, 1, CONFIG_SLOT_BYTES)
                    .map_err(|e| format!("create {}: {e:?}", path.display()))?;
                slot.write(payload)
                    .map_err(|e| format!("write {}: {e:?}", path.display()))?;
                self.slots.insert(section.clone(), slot);
            }
        }

        let restart_required_keys: Vec<String> = changed
            .iter()
            .filter(|k| {
                schema
                    .get(k)
                    .map(|s| s.reload == "restart_required")
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        self.last_merged = merged;
        Ok(ApplyResult {
            changed_sections,
            restart_required_keys,
            errors: Vec::new(),
        })
    }
}

fn log_config_errors(phase: &str, errs: &[ConfigError]) {
    for e in errs {
        // Greppable cascade per §7.F taxonomy.
        error!(
            event = "CONFIG_INVALID",
            phase = phase,
            key = %e.key,
            detail = %e.detail,
            "[ERR][config_daemon][CONFIG_INVALID][FATAL] {}={}", e.key, e.detail
        );
    }
}

/// Watch loop: poll source mtimes; on change re-apply; publish notices/errors via
/// the broker. Runs until `shutdown` fires. `broker` is a `publish_local`-capable
/// async sink; we keep it generic over a closure so the loop is unit-testable.
pub async fn run_config_watch_loop<P, Fut>(
    mut slots: ConfigSlotSet,
    schema: Arc<ConfigSchema>,
    shutdown: Arc<tokio::sync::Notify>,
    mut publish: P,
) where
    P: FnMut(&'static str, Vec<u8>) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let mut ticker = tokio::time::interval(Duration::from_millis(CONFIG_WATCH_POLL_MS));
    info!(
        event = "CONFIG_WATCH_START",
        "config_daemon: watch loop running"
    );
    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let now = slots.sources.mtimes();
                if now == slots.last_mtimes {
                    continue;
                }
                slots.last_mtimes = now;
                match slots.reapply(&schema) {
                    Ok(r) if !r.errors.is_empty() => {
                        log_config_errors("watch", &r.errors);
                        let payload = rmp_serde::to_vec_named(&ErrEvent {
                            module: "config_daemon",
                            count: r.errors.len(),
                        }).unwrap_or_default();
                        publish("MODULE_ERROR", payload).await;
                    }
                    Ok(r) => {
                        if !r.changed_sections.is_empty() {
                            info!(event = "CONFIG_APPLIED", sections = ?r.changed_sections,
                                "config_daemon: re-applied changed sections");
                        }
                        if !r.restart_required_keys.is_empty() {
                            warn!(event = "CONFIG_RESTART_REQUIRED", keys = ?r.restart_required_keys,
                                "config_daemon: restart_required keys changed (written, applies next restart)");
                            let payload = rmp_serde::to_vec_named(&RestartEvent {
                                keys: r.restart_required_keys.clone(),
                            }).unwrap_or_default();
                            publish("CONFIG_RESTART_REQUIRED", payload).await;
                        }
                    }
                    Err(e) => error!(err = %e, "config_daemon: reapply hard error — keeping last-good"),
                }
            }
            _ = shutdown.notified() => {
                info!(event = "CONFIG_WATCH_STOP", "config_daemon: watch loop shutdown");
                break;
            }
        }
    }
}

#[derive(serde::Serialize)]
struct ErrEvent {
    module: &'static str,
    count: usize,
}
#[derive(serde::Serialize)]
struct RestartEvent {
    keys: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str = r#"
[config]
"api.port" = {type="int", constraint="1..65535", reload="restart_required", owner=["api"], sensitive=false}
"social_x.tick" = {type="float", constraint=">0", reload="hot", owner=["social_worker"], sensitive=false}
"social_x.name" = {type="str", constraint="", reload="hot", owner=["social_worker"], sensitive=false}
[params]
"syn.threshold" = {type="float", constraint="0..1", reload="hot", owner=["synthesis"], sensitive=false}
"#;

    fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    /// Isolated sources: secrets points at a NONEXISTENT path so the test merge
    /// never picks up the real ~/.titan/secrets.toml on the box.
    fn base_sources(dir: &Path) -> (SourcePaths, ConfigSchema) {
        let params = write(dir, "titan_params.toml", "[syn]\nthreshold = 0.7\n");
        let config = write(
            dir,
            "config.toml",
            "[api]\nport = 7777\n[social_x]\ntick = 30.0\nname = \"t\"\n",
        );
        let sources = SourcePaths {
            params,
            config,
            secrets: dir.join("no_secrets.toml"),
        };
        (sources, ConfigSchema::from_toml_str(SCHEMA).unwrap())
    }

    #[test]
    fn deep_merge_overrides() {
        let mut base: toml::Value = toml::from_str("[a]\nx=1\ny=2\n").unwrap();
        let over: toml::Value = toml::from_str("[a]\ny=9\n[b]\nz=3\n").unwrap();
        deep_merge(&mut base, &over);
        assert_eq!(base["a"]["x"].as_integer(), Some(1));
        assert_eq!(base["a"]["y"].as_integer(), Some(9));
        assert_eq!(base["b"]["z"].as_integer(), Some(3));
    }

    #[test]
    fn seed_creates_one_slot_per_section() {
        let tmp = tempfile::tempdir().unwrap();
        let shm = tmp.path().join("shm");
        std::fs::create_dir_all(&shm).unwrap();
        let (sources, schema) = base_sources(tmp.path());
        let set = ConfigSlotSet::seed_with_sources(&shm, sources, &schema, true).unwrap();
        // sections: syn (params), api + social_x (config) = 3
        assert_eq!(set.section_count(), 3);
        assert!(shm.join("config/api.bin").exists());
        assert!(shm.join("config/social_x.bin").exists());
        // slot payload round-trips to msgpack map
        let bytes = Slot::open(shm.join("config/social_x.bin"))
            .unwrap()
            .read()
            .unwrap();
        let val: rmpv::Value = rmp_serde::from_slice(&bytes).unwrap();
        assert!(val.is_map());
    }

    #[test]
    fn valid_hot_edit_bumps_only_its_section() {
        let tmp = tempfile::tempdir().unwrap();
        let shm = tmp.path().join("shm");
        std::fs::create_dir_all(&shm).unwrap();
        let (sources, schema) = base_sources(tmp.path());
        let cfg = sources.config.clone();
        let mut set = ConfigSlotSet::seed_with_sources(&shm, sources, &schema, true).unwrap();
        std::fs::write(
            &cfg,
            "[api]\nport = 7777\n[social_x]\ntick = 15.0\nname = \"t\"\n",
        )
        .unwrap();
        let r = set.reapply(&schema).unwrap();
        assert_eq!(r.changed_sections, vec!["social_x".to_string()]);
        assert!(r.errors.is_empty());
        assert!(r.restart_required_keys.is_empty());
    }

    #[test]
    fn invalid_edit_keeps_last_good_no_write() {
        let tmp = tempfile::tempdir().unwrap();
        let shm = tmp.path().join("shm");
        std::fs::create_dir_all(&shm).unwrap();
        let (sources, schema) = base_sources(tmp.path());
        let cfg = sources.config.clone();
        let mut set = ConfigSlotSet::seed_with_sources(&shm, sources, &schema, true).unwrap();
        std::fs::write(
            &cfg,
            "[api]\nport = 70000\n[social_x]\ntick = 30.0\nname = \"t\"\n",
        )
        .unwrap();
        let r = set.reapply(&schema).unwrap();
        assert!(!r.errors.is_empty());
        assert!(r.changed_sections.is_empty()); // fail-closed: nothing written
    }

    #[test]
    fn restart_required_key_surfaced() {
        let tmp = tempfile::tempdir().unwrap();
        let shm = tmp.path().join("shm");
        std::fs::create_dir_all(&shm).unwrap();
        let (sources, schema) = base_sources(tmp.path());
        let cfg = sources.config.clone();
        let mut set = ConfigSlotSet::seed_with_sources(&shm, sources, &schema, true).unwrap();
        std::fs::write(
            &cfg,
            "[api]\nport = 8888\n[social_x]\ntick = 30.0\nname = \"t\"\n",
        )
        .unwrap();
        let r = set.reapply(&schema).unwrap();
        assert_eq!(r.restart_required_keys, vec!["api.port".to_string()]);
        assert_eq!(r.changed_sections, vec!["api".to_string()]);
    }
}
