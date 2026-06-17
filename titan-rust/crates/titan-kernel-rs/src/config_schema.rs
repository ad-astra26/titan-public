//! config_schema — load + validate against the canonical `config_schema.toml`
//! (RFP_config_as_shm_state_canonical_integrity §7.A / SPEC §14.A.1).
//!
//! Kept in kernel-rs (not titan-core) because schema validation is kernel-only:
//! the in-kernel config daemon is the single validator (G21 single-writer). This
//! avoids adding a `toml` dependency to titan-core. Per-key spec mirrors the
//! Phase-0 generator: `type · constraint · reload · owner(set) · sensitive`.
//!
//! Format (one stanza/key, quoted dotted literal keys, two namespace tables):
//! ```toml
//! [config]
//! "social_x.post_dispatch_tick_interval_seconds" = {type="float", constraint=">0", reload="hot", owner=["social_worker"], sensitive=false}
//! [params]
//! "synthesis.decision_authority.threshold" = {type="float", constraint="0..1", reload="hot", owner=["synthesis"], sensitive=false}
//! ```

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

/// Per-key declaration from `config_schema.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct KeySpec {
    /// Declared value type: `bool` | `int` | `float` | `str` | `list`.
    #[serde(rename = "type")]
    pub ty: String,
    /// Value constraint (e.g. `a..b`, `>0`, `>=0`) or empty for none.
    #[serde(default)]
    pub constraint: String,
    /// Reload class: `hot` (re-applied live) or `restart_required` (surfaced only).
    pub reload: String,
    /// Set of real registered modules (or `kernel`) that read this key.
    #[serde(default)]
    pub owner: Vec<String>,
    /// Whether the value is secret material (controls handling/redaction).
    #[serde(default)]
    pub sensitive: bool,
}

#[derive(Debug, Deserialize)]
struct RawSchema {
    #[serde(default)]
    config: HashMap<String, KeySpec>,
    #[serde(default)]
    params: HashMap<String, KeySpec>,
}

/// The loaded schema: one flat map of dotted-key → spec (config ⊎ params; the
/// two namespaces are disjoint by INV-CFG-6, so a single map is unambiguous).
#[derive(Debug, Clone)]
pub struct ConfigSchema {
    entries: HashMap<String, KeySpec>,
}

/// A single validation violation (collected; the daemon reports ALL of them).
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigError {
    /// The offending dotted config key.
    pub key: String,
    /// What kind of violation.
    pub kind: ConfigErrorKind,
    /// Human-readable detail (expected vs got, range, etc.).
    pub detail: String,
}

/// Category of a [`ConfigError`].
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigErrorKind {
    /// Key present in the merged config but absent from the schema (non-canonical).
    Undeclared,
    /// Value's TOML type does not match the declared `type`.
    TypeMismatch,
    /// Value violates the declared `constraint`.
    ConstraintViolation,
}

impl ConfigSchema {
    /// Load + parse `config_schema.toml`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let txt = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("read {}: {e}", path.as_ref().display()))?;
        Self::from_toml_str(&txt)
    }

    /// Parse a schema from a TOML string (the `load` body, split out for tests).
    pub fn from_toml_str(txt: &str) -> Result<Self, String> {
        let raw: RawSchema = toml::from_str(txt).map_err(|e| format!("parse schema: {e}"))?;
        let mut entries = raw.config;
        // params ⊎ config — disjoint, so no clobber expected; if a dup ever appears
        // it's a schema bug (caught by the Phase-0 disjointness test), keep config's.
        for (k, v) in raw.params {
            entries.entry(k).or_insert(v);
        }
        Ok(Self { entries })
    }

    /// Number of declared keys (config ⊎ params).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the schema declares no keys.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a key's declaration by its full dotted name.
    pub fn get(&self, key: &str) -> Option<&KeySpec> {
        self.entries.get(key)
    }

    /// Validate the merged config against the schema, collecting EVERY violation.
    /// Empty result ⇒ canonical. Each leaf must be declared + type-ok + constraint-ok.
    pub fn validate(&self, merged: &toml::Value) -> Vec<ConfigError> {
        let mut flat = HashMap::new();
        flatten(merged, "", &mut flat);
        let mut errs = Vec::new();
        for (key, val) in &flat {
            match self.entries.get(key) {
                None => errs.push(ConfigError {
                    key: key.clone(),
                    kind: ConfigErrorKind::Undeclared,
                    detail: "key not present in config_schema.toml".into(),
                }),
                Some(spec) => {
                    if !type_matches(&spec.ty, val) {
                        errs.push(ConfigError {
                            key: key.clone(),
                            kind: ConfigErrorKind::TypeMismatch,
                            detail: format!("expected {}, got {}", spec.ty, toml_type_name(val)),
                        });
                    } else if let Some(why) = constraint_violation(&spec.constraint, val) {
                        errs.push(ConfigError {
                            key: key.clone(),
                            kind: ConfigErrorKind::ConstraintViolation,
                            detail: format!("constraint `{}`: {}", spec.constraint, why),
                        });
                    }
                }
            }
        }
        errs.sort_by(|a, b| a.key.cmp(&b.key));
        errs
    }
}

/// Flatten a TOML table to dotted keys; array-of-tables → `<key>[].<leaf>`
/// (element shape declared once, matching the Phase-0 generator + schema).
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
            if arr.iter().all(|e| matches!(e, toml::Value::Table(_))) && !arr.is_empty() =>
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

fn toml_type_name(v: &toml::Value) -> &'static str {
    match v {
        toml::Value::Boolean(_) => "bool",
        toml::Value::Integer(_) => "int",
        toml::Value::Float(_) => "float",
        toml::Value::String(_) => "str",
        toml::Value::Array(_) => "list",
        toml::Value::Table(_) => "table",
        toml::Value::Datetime(_) => "datetime",
    }
}

fn type_matches(declared: &str, v: &toml::Value) -> bool {
    match declared {
        "bool" => matches!(v, toml::Value::Boolean(_)),
        // an int value satisfies a `float` declaration (TOML 1 vs 1.0 drift)
        "float" => matches!(v, toml::Value::Float(_) | toml::Value::Integer(_)),
        "int" => matches!(v, toml::Value::Integer(_)),
        "str" => matches!(v, toml::Value::String(_)),
        "list" => matches!(v, toml::Value::Array(_)),
        _ => true, // unknown declared type → don't block (schema-author error, not config error)
    }
}

/// Return `Some(reason)` if `value` violates `constraint`, else `None`.
/// Supported: "" (none) · "a..b" (inclusive numeric range) · ">0" · ">=0".
fn constraint_violation(constraint: &str, v: &toml::Value) -> Option<String> {
    let c = constraint.trim();
    if c.is_empty() {
        return None;
    }
    let num = match v {
        toml::Value::Integer(i) => Some(*i as f64),
        toml::Value::Float(f) => Some(*f),
        _ => None,
    }?;
    if let Some((lo, hi)) = c.split_once("..") {
        let lo: f64 = lo.trim().parse().ok()?;
        let hi: f64 = hi.trim().parse().ok()?;
        if num < lo || num > hi {
            return Some(format!("{num} not in {lo}..{hi}"));
        }
    } else if c == ">0" && num <= 0.0 {
        return Some(format!("{num} not > 0"));
    } else if c == ">=0" && num < 0.0 {
        return Some(format!("{num} not >= 0"));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str = r#"
[config]
"api.port" = {type="int", constraint="1..65535", reload="restart_required", owner=["api"], sensitive=false}
"social_x.tick" = {type="float", constraint=">0", reload="hot", owner=["social_worker"], sensitive=false}
"x.flag" = {type="bool", constraint="", reload="hot", owner=["kernel"], sensitive=false}
[params]
"syn.threshold" = {type="float", constraint="0..1", reload="hot", owner=["synthesis"], sensitive=false}
"chat.tiers[].temp" = {type="float", constraint="0..1", reload="hot", owner=["agno_worker"], sensitive=false}
"#;

    fn schema() -> ConfigSchema {
        ConfigSchema::from_toml_str(SCHEMA).unwrap()
    }

    #[test]
    fn loads_both_namespaces() {
        let s = schema();
        assert_eq!(s.len(), 5);
        assert_eq!(s.get("api.port").unwrap().reload, "restart_required");
        assert_eq!(s.get("syn.threshold").unwrap().ty, "float");
    }

    #[test]
    fn valid_config_passes() {
        let cfg: toml::Value = toml::from_str(
            "[api]\nport = 7777\n[social_x]\ntick = 15.0\n[x]\nflag = true\n[syn]\nthreshold = 0.7\n[[chat.tiers]]\ntemp = 0.5\n",
        )
        .unwrap();
        assert!(schema().validate(&cfg).is_empty());
    }

    #[test]
    fn out_of_range_fails() {
        let cfg: toml::Value = toml::from_str("[api]\nport = 70000\n").unwrap();
        let errs = schema().validate(&cfg);
        assert!(errs
            .iter()
            .any(|e| e.key == "api.port" && e.kind == ConfigErrorKind::ConstraintViolation));
    }

    #[test]
    fn type_mismatch_fails() {
        let cfg: toml::Value = toml::from_str("[x]\nflag = \"yes\"\n").unwrap();
        let errs = schema().validate(&cfg);
        assert!(errs
            .iter()
            .any(|e| e.key == "x.flag" && e.kind == ConfigErrorKind::TypeMismatch));
    }

    #[test]
    fn undeclared_key_fails() {
        let cfg: toml::Value = toml::from_str("[api]\nport = 7777\nbogus = 1\n").unwrap();
        let errs = schema().validate(&cfg);
        assert!(errs
            .iter()
            .any(|e| e.key == "api.bogus" && e.kind == ConfigErrorKind::Undeclared));
    }

    #[test]
    fn array_of_tables_normalized() {
        let cfg: toml::Value =
            toml::from_str("[[chat.tiers]]\ntemp = 0.3\n[[chat.tiers]]\ntemp = 0.9\n").unwrap();
        // both elements map to chat.tiers[].temp → declared, in range
        assert!(schema().validate(&cfg).is_empty());
    }

    #[test]
    fn int_satisfies_float_decl() {
        let cfg: toml::Value = toml::from_str("[social_x]\ntick = 15\n").unwrap();
        assert!(schema().validate(&cfg).is_empty());
    }
}
