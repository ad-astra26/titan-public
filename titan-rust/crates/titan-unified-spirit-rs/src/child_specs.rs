//! child_specs — 6 trinity daemon `ChildSpec` definitions for
//! unified-spirit's supervisor (C4-4).
//!
//! Per SPEC §9.A unified-spirit-rs row + §11.G dependency-aware supervision.
//! Each daemon declares:
//! - REQUIRED kernel-binary dependency (the kernel must be alive — its
//!   shm slots + bus broker are everyone's foundation).
//! - REQUIRED its own `sensor_cache_<name>.bin` slot (Python sensor
//!   refresh sidecar populates this in C-S5/C-S6; substrate +
//!   unified-spirit do NOT depend on the cache file existing during
//!   C-S4, so this is gated soft until C-S5 ships the writers).
//! - SOFT `topology_30d.bin` slot — body+mind daemons read it for
//!   ground_up nudges; spirit daemons don't.
//!
//! In C-S4, the actual daemon binaries don't exist yet — C-S5/C-S6
//! ship `titan-inner-body-rs`, etc. Until then, the supervisor's
//! `--use-placeholder-daemons` flag (CLI arg from C4-1) routes spawn
//! attempts to a generic placeholder OR `/bin/true` for tests.

use titan_core::supervisor::{ChildSpec, Dependency};

/// Default max-age (seconds) for sensor_cache slot freshness checks per
/// SPEC §11.G shm slot dep — generous to allow Python sidecar warm-up.
const SENSOR_CACHE_MAX_AGE_S: f32 = 30.0;
/// Default max-age (seconds) for topology slot freshness — body cycle
/// is ~1.15s, so 5s allows substantial cushion.
const TOPOLOGY_MAX_AGE_S: f32 = 5.0;

/// Canonical 6-daemon name set per SPEC §9.A. Order is `(inner|outer) × (body|mind|spirit)`.
pub const DAEMON_NAMES: [&str; 6] = [
    "titan-inner-body-rs",
    "titan-inner-mind-rs",
    "titan-inner-spirit-rs",
    "titan-outer-body-rs",
    "titan-outer-mind-rs",
    "titan-outer-spirit-rs",
];

/// Build the 6 trinity daemon `ChildSpec` instances. Each carries the
/// canonical dependency set per SPEC §9.A daemon rows.
pub fn build_daemon_specs() -> Vec<ChildSpec> {
    DAEMON_NAMES.iter().map(|&n| build_daemon_spec(n)).collect()
}

/// Build one ChildSpec by canonical daemon name.
pub fn build_daemon_spec(name: &str) -> ChildSpec {
    let is_body_or_mind = matches!(
        name,
        "titan-inner-body-rs"
            | "titan-inner-mind-rs"
            | "titan-outer-body-rs"
            | "titan-outer-mind-rs"
    );
    let cache_file = sensor_cache_filename(name);

    let mut spec = ChildSpec::new(name)
        // Soft: own sensor_cache slot exists. C-S5/C-S6 sidecars write
        // these; in C-S4 / pre-C-S5 the cache file is absent at boot.
        // SOFT severity → supervisor logs DEPENDENCY_DEGRADED + respawns
        // the daemon anyway (per SPEC §11.G); will be flipped to CRITICAL
        // once C-S5 ships sensor sidecars.
        .with_dependency(
            Dependency::critical_shm_slot(cache_file.clone(), SENSOR_CACHE_MAX_AGE_S).soft(),
        );

    // Body + mind daemons read topology_30d.bin for ground_up nudges
    // (SPEC §10.G step 6). Spirit daemons don't ground_up.
    if is_body_or_mind {
        spec = spec.with_dependency(
            Dependency::critical_shm_slot("topology_30d.bin", TOPOLOGY_MAX_AGE_S).soft(),
        );
    }

    spec
}

/// Map daemon name to its `sensor_cache_*.bin` slot per SPEC §9.A.
pub fn sensor_cache_filename(daemon_name: &str) -> String {
    let suffix = match daemon_name {
        "titan-inner-body-rs" => "inner_body",
        "titan-inner-mind-rs" => "inner_mind",
        "titan-inner-spirit-rs" => "inner_spirit",
        "titan-outer-body-rs" => "outer_body",
        "titan-outer-mind-rs" => "outer_mind",
        "titan-outer-spirit-rs" => "outer_spirit",
        _ => "unknown",
    };
    format!("sensor_cache_{suffix}.bin")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn six_daemon_names_canonical() {
        // C4-4 test 1: name set matches SPEC §9.A
        assert_eq!(DAEMON_NAMES.len(), 6);
        for n in [
            "titan-inner-body-rs",
            "titan-inner-mind-rs",
            "titan-inner-spirit-rs",
            "titan-outer-body-rs",
            "titan-outer-mind-rs",
            "titan-outer-spirit-rs",
        ] {
            assert!(DAEMON_NAMES.contains(&n));
        }
    }

    #[test]
    fn build_daemon_specs_returns_six() {
        // C4-4 test 2: 6 specs built
        let specs = build_daemon_specs();
        assert_eq!(specs.len(), 6);
        for spec in &specs {
            assert!(spec.restart_on_crash, "default restart_on_crash=true");
        }
    }

    #[test]
    fn body_and_mind_daemons_depend_on_topology() {
        // C4-4 test 3: body + mind daemons declare topology_30d.bin dep
        for name in [
            "titan-inner-body-rs",
            "titan-inner-mind-rs",
            "titan-outer-body-rs",
            "titan-outer-mind-rs",
        ] {
            let spec = build_daemon_spec(name);
            assert!(
                spec.dependencies
                    .iter()
                    .any(|d| d.name == "topology_30d.bin"),
                "{name} should declare topology_30d.bin dep"
            );
        }
    }

    #[test]
    fn spirit_daemons_do_not_depend_on_topology() {
        // C4-4 test 4: spirit daemons skip topology dep (don't ground_up)
        for name in ["titan-inner-spirit-rs", "titan-outer-spirit-rs"] {
            let spec = build_daemon_spec(name);
            assert!(
                !spec
                    .dependencies
                    .iter()
                    .any(|d| d.name == "topology_30d.bin"),
                "{name} should NOT declare topology_30d.bin dep"
            );
        }
    }

    #[test]
    fn each_daemon_declares_own_sensor_cache_dep() {
        // C4-4 test 5: per-daemon sensor_cache file present in deps
        for &name in DAEMON_NAMES.iter() {
            let spec = build_daemon_spec(name);
            let cache = sensor_cache_filename(name);
            assert!(
                spec.dependencies.iter().any(|d| d.name == cache),
                "{name} must depend on {cache}"
            );
        }
    }

    #[test]
    fn sensor_cache_filename_canonical() {
        // C4-4 test 6: name → filename mapping
        assert_eq!(
            sensor_cache_filename("titan-inner-body-rs"),
            "sensor_cache_inner_body.bin"
        );
        assert_eq!(
            sensor_cache_filename("titan-outer-spirit-rs"),
            "sensor_cache_outer_spirit.bin"
        );
        assert_eq!(
            sensor_cache_filename("unknown-binary"),
            "sensor_cache_unknown.bin"
        );
    }
}
