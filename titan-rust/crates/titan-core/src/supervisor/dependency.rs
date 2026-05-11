//! dependency — `Dependency` declaration + `DepProbe` trait + DAG cycle check.
//!
//! Per SPEC §11.G.1 (declaration) + §11.G.2 (pre-respawn check) + §11.G.7
//! (`arch_map phase-c verify --check-deps` cycle detection at SPEC-load
//! time).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::supervisor::types::{DependencyKind, DependencySeverity};

/// Declarative dependency for a supervised child.
#[derive(Debug, Clone)]
pub struct Dependency {
    /// Human/log-friendly name (e.g. `"social_x_db"`, `"x_api_reachable"`).
    pub name: String,
    /// What kind of resource this is.
    pub kind: DependencyKind,
    /// Critical → block respawn; soft → log + respawn anyway.
    pub severity: DependencySeverity,
    /// Type-specific check parameters.
    pub check_spec: DepCheckSpec,
}

impl Dependency {
    /// New critical sibling-module dependency.
    pub fn critical_module(name: impl Into<String>) -> Self {
        let module_name: String = name.into();
        Self {
            name: module_name.clone(),
            kind: DependencyKind::Module,
            severity: DependencySeverity::Critical,
            check_spec: DepCheckSpec::Module(module_name),
        }
    }

    /// New critical Rust binary dependency.
    pub fn critical_binary(name: impl Into<String>) -> Self {
        let binary_name: String = name.into();
        Self {
            name: binary_name.clone(),
            kind: DependencyKind::Binary,
            severity: DependencySeverity::Critical,
            check_spec: DepCheckSpec::Binary(binary_name),
        }
    }

    /// New critical shm-slot freshness dependency.
    pub fn critical_shm_slot(slot_name: impl Into<String>, max_age_s: f32) -> Self {
        let slot_name: String = slot_name.into();
        Self {
            name: slot_name.clone(),
            kind: DependencyKind::ShmSlot,
            severity: DependencySeverity::Critical,
            check_spec: DepCheckSpec::ShmSlot {
                name: slot_name,
                max_age_s,
            },
        }
    }

    /// Convert to soft (informational only — respawn proceeds even if
    /// dep fails).
    pub fn soft(mut self) -> Self {
        self.severity = DependencySeverity::Soft;
        self
    }
}

/// Type-specific check parameters per `DependencyKind`.
#[derive(Debug, Clone)]
pub enum DepCheckSpec {
    /// Sibling Python module name.
    Module(String),
    /// Rust binary name.
    Binary(String),
    /// Shm slot freshness check.
    ShmSlot {
        /// Slot filename.
        name: String,
        /// Max age before considered stale.
        max_age_s: f32,
    },
    /// External-service probe.
    ExternalService {
        /// Probe identifier (e.g. `"solana_rpc_get_health"`).
        probe_kind: String,
        /// Endpoint URL.
        url: String,
    },
    /// DB file presence + freshness.
    DbFile {
        /// Path to the DB.
        path: PathBuf,
        /// Max acceptable age since last write.
        max_write_age_s: f32,
    },
    /// HTTP endpoint status check.
    Endpoint {
        /// URL to probe.
        url: String,
        /// Probe timeout (seconds).
        timeout_s: f32,
    },
}

/// Outcome of a dependency check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyCheckOutcome {
    /// Dep is healthy.
    Ok,
    /// Dep is down (with reason for logs).
    Down {
        /// Why the check failed.
        reason: String,
    },
}

/// Trait abstracting how dependencies are probed at runtime.
///
/// Real impl in C2-6 walks the supervisor's child registry + `/dev/shm/`
/// + makes HTTP probes. Tests use [`MockDepProbe`] for determinism.
pub trait DepProbe: Send + Sync {
    /// Check one dependency. Returns `Ok` (healthy) or `Down { reason }`.
    fn check(&self, dep: &Dependency) -> DependencyCheckOutcome;
}

/// Mock dependency probe — record-and-replay state for tests.
pub struct MockDepProbe {
    states: parking_lot::Mutex<HashMap<String, DependencyCheckOutcome>>,
}

impl MockDepProbe {
    /// New probe with no dependencies set (all checks fail with
    /// `Down { reason: "not configured" }`).
    pub fn new() -> Self {
        Self {
            states: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    /// Set a dep's outcome. Subsequent `check()` calls return this.
    pub fn set(&self, dep_name: &str, outcome: DependencyCheckOutcome) {
        self.states.lock().insert(dep_name.to_string(), outcome);
    }

    /// Mark a dep as healthy.
    pub fn healthy(&self, dep_name: &str) {
        self.set(dep_name, DependencyCheckOutcome::Ok);
    }

    /// Mark a dep as down with a reason.
    pub fn down(&self, dep_name: &str, reason: impl Into<String>) {
        self.set(
            dep_name,
            DependencyCheckOutcome::Down {
                reason: reason.into(),
            },
        );
    }
}

impl Default for MockDepProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl DepProbe for MockDepProbe {
    fn check(&self, dep: &Dependency) -> DependencyCheckOutcome {
        self.states
            .lock()
            .get(&dep.name)
            .cloned()
            .unwrap_or(DependencyCheckOutcome::Down {
                reason: "not configured in MockDepProbe".into(),
            })
    }
}

/// Errors during DAG cycle detection.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DagCheckError {
    /// Dependency cycle detected. Path lists the cycle members in order.
    #[error("dependency cycle: {path:?}")]
    Cycle {
        /// Names of children involved in the cycle, in detection order.
        path: Vec<String>,
    },
    /// Reference to an unknown child name.
    #[error("dep {dep_name} referenced by {child} but no such child registered")]
    UnknownReference {
        /// Child that has the bad reference.
        child: String,
        /// Dep name that doesn't resolve.
        dep_name: String,
    },
}

/// Verify the dependency graph forms a DAG (no cycles).
///
/// Per SPEC §11.G.7 — `arch_map phase-c verify --check-deps` runs this at
/// SPEC-load time (compile-time-equivalent at boot). Used by the
/// `Supervisor::register_child` path in C2-6 to fail-fast on cyclic specs.
///
/// Only `DependencyKind::Module` and `DependencyKind::Binary` deps are
/// checked for cycles — other kinds reference external resources, not
/// other supervised children.
pub fn check_dag(children: &HashMap<String, Vec<Dependency>>) -> Result<(), DagCheckError> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut on_stack: HashSet<String> = HashSet::new();
    let mut path: Vec<String> = Vec::new();

    for child in children.keys() {
        if !visited.contains(child) {
            visit(child, children, &mut visited, &mut on_stack, &mut path)?;
        }
    }
    Ok(())
}

fn visit(
    name: &str,
    children: &HashMap<String, Vec<Dependency>>,
    visited: &mut HashSet<String>,
    on_stack: &mut HashSet<String>,
    path: &mut Vec<String>,
) -> Result<(), DagCheckError> {
    visited.insert(name.to_string());
    on_stack.insert(name.to_string());
    path.push(name.to_string());

    if let Some(deps) = children.get(name) {
        for dep in deps {
            // Only traverse deps that point to other supervised children
            // (Module / Binary). Other kinds reference external resources.
            let dep_target = match &dep.check_spec {
                DepCheckSpec::Module(n) | DepCheckSpec::Binary(n) => Some(n.as_str()),
                _ => None,
            };
            if let Some(target) = dep_target {
                if !children.contains_key(target) {
                    return Err(DagCheckError::UnknownReference {
                        child: name.to_string(),
                        dep_name: target.to_string(),
                    });
                }
                if on_stack.contains(target) {
                    // Cycle: collect the on-stack tail starting from target
                    let mut cycle = Vec::new();
                    let mut started = false;
                    for p in path.iter() {
                        if p == target {
                            started = true;
                        }
                        if started {
                            cycle.push(p.clone());
                        }
                    }
                    cycle.push(target.to_string());
                    return Err(DagCheckError::Cycle { path: cycle });
                }
                if !visited.contains(target) {
                    visit(target, children, visited, on_stack, path)?;
                }
            }
        }
    }

    path.pop();
    on_stack.remove(name);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deps(items: &[Dependency]) -> Vec<Dependency> {
        items.to_vec()
    }

    #[test]
    fn dependency_critical_module_constructor() {
        let d = Dependency::critical_module("memory_module");
        assert_eq!(d.name, "memory_module");
        assert_eq!(d.severity, DependencySeverity::Critical);
        assert_eq!(d.kind, DependencyKind::Module);
    }

    #[test]
    fn dependency_soft_builder() {
        let d = Dependency::critical_module("mem").soft();
        assert_eq!(d.severity, DependencySeverity::Soft);
    }

    #[test]
    fn mock_probe_default_is_down() {
        let probe = MockDepProbe::new();
        let dep = Dependency::critical_module("x");
        let outcome = probe.check(&dep);
        assert!(matches!(outcome, DependencyCheckOutcome::Down { .. }));
    }

    #[test]
    fn mock_probe_can_be_set_healthy() {
        let probe = MockDepProbe::new();
        probe.healthy("memory_module");
        let dep = Dependency::critical_module("memory_module");
        let outcome = probe.check(&dep);
        assert_eq!(outcome, DependencyCheckOutcome::Ok);
    }

    #[test]
    fn mock_probe_can_be_set_down() {
        let probe = MockDepProbe::new();
        probe.down("x_api", "connection refused");
        let dep = Dependency {
            name: "x_api".into(),
            kind: DependencyKind::ExternalService,
            severity: DependencySeverity::Critical,
            check_spec: DepCheckSpec::ExternalService {
                probe_kind: "x_api_health".into(),
                url: "https://x.com/api/health".into(),
            },
        };
        match probe.check(&dep) {
            DependencyCheckOutcome::Down { reason } => {
                assert!(reason.contains("connection refused"));
            }
            _ => panic!("expected Down"),
        }
    }

    // ── DAG check tests ───────────────────────────────────────────────

    #[test]
    fn dag_empty_graph_ok() {
        let m: HashMap<String, Vec<Dependency>> = HashMap::new();
        assert!(check_dag(&m).is_ok());
    }

    #[test]
    fn dag_single_child_no_deps_ok() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), Vec::new());
        assert!(check_dag(&m).is_ok());
    }

    #[test]
    fn dag_simple_chain_ok() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), deps(&[Dependency::critical_module("b")]));
        m.insert("b".to_string(), deps(&[Dependency::critical_module("c")]));
        m.insert("c".to_string(), Vec::new());
        assert!(check_dag(&m).is_ok());
    }

    #[test]
    fn dag_self_cycle_detected() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), deps(&[Dependency::critical_module("a")]));
        let result = check_dag(&m);
        match result {
            Err(DagCheckError::Cycle { path }) => assert!(path.contains(&"a".to_string())),
            other => panic!("expected Cycle, got {other:?}"),
        }
    }

    #[test]
    fn dag_two_node_cycle_detected() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), deps(&[Dependency::critical_module("b")]));
        m.insert("b".to_string(), deps(&[Dependency::critical_module("a")]));
        assert!(matches!(check_dag(&m), Err(DagCheckError::Cycle { .. })));
    }

    #[test]
    fn dag_three_node_cycle_detected() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), deps(&[Dependency::critical_module("b")]));
        m.insert("b".to_string(), deps(&[Dependency::critical_module("c")]));
        m.insert("c".to_string(), deps(&[Dependency::critical_module("a")]));
        assert!(matches!(check_dag(&m), Err(DagCheckError::Cycle { .. })));
    }

    #[test]
    fn dag_unknown_reference_detected() {
        let mut m = HashMap::new();
        m.insert(
            "a".to_string(),
            deps(&[Dependency::critical_module("ghost")]),
        );
        let result = check_dag(&m);
        assert!(matches!(
            result,
            Err(DagCheckError::UnknownReference { .. })
        ));
    }

    #[test]
    fn dag_external_deps_not_traversed() {
        // ExternalService deps reference URLs, not children — should NOT
        // produce UnknownReference errors.
        let mut m = HashMap::new();
        m.insert(
            "a".to_string(),
            vec![Dependency {
                name: "x_api".into(),
                kind: DependencyKind::ExternalService,
                severity: DependencySeverity::Critical,
                check_spec: DepCheckSpec::ExternalService {
                    probe_kind: "test".into(),
                    url: "http://x".into(),
                },
            }],
        );
        assert!(check_dag(&m).is_ok());
    }

    #[test]
    fn dag_diamond_pattern_ok() {
        // a → b, a → c, b → d, c → d (no cycle, just diamond)
        let mut m = HashMap::new();
        m.insert(
            "a".to_string(),
            deps(&[
                Dependency::critical_module("b"),
                Dependency::critical_module("c"),
            ]),
        );
        m.insert("b".to_string(), deps(&[Dependency::critical_module("d")]));
        m.insert("c".to_string(), deps(&[Dependency::critical_module("d")]));
        m.insert("d".to_string(), Vec::new());
        assert!(check_dag(&m).is_ok());
    }
}
