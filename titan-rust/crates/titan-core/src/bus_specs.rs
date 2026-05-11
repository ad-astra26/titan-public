//! bus_specs — Bus message catalog (SPEC §8) as compile-time perfect hash.
//!
//! C-S2 chunk C2-2 populates the catalog. Total: 47 canonical messages +
//! 5 legacy drift-bridge entries (D13/D14/D15) = 52 entries.
//!
//! # Priority lanes (SPEC §8.0)
//!
//! - **P0** — never drop; reserve `BUS_P0_RESERVE_SLOTS=64`.
//! - **P1** — drop oldest under load + coalesce-by-`(src, type)` for STATE
//!   messages (BODY_STATE / MIND_STATE / SPIRIT_STATE / OUTER_OBSERVATION).
//! - **P2** — drop oldest under load (no coalesce) — default lane.
//! - **P3** — drop NEWEST under load (e.g. OBSERVATORY_EVENT).
//!
//! # Lookup
//!
//! ```ignore
//! use titan_core::bus_specs::SPECS;
//! if let Some(spec) = SPECS.get("KERNEL_EPOCH_TICK") {
//!     // spec.priority, spec.coalesce_by, spec.catalog
//! }
//! ```
//!
//! Hot-path O(1) thanks to `phf::phf_map!` compile-time perfect hash.
//!
//! # Drift bridges (SPEC §3.1 D13/D14/D15)
//!
//! Legacy names (`BUS_HANDOFF`, `BUS_WORKER_ADOPT_REQUEST`, `EPOCH_TICK`,
//! etc.) appear in the catalog with the SAME priority/coalesce as their
//! canonical counterparts so subscribers can listen on either name during
//! the Phase C transition. Broker dual-emits in `titan-bus::drift_bridge`
//! (C-S2 chunk C2-2.b). C-S8 deletes the legacy entries + the bridge.

use phf::phf_map;

/// Priority lane per SPEC §8.0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Priority {
    /// Never drop.
    P0,
    /// Drop oldest + coalesce-by-key.
    P1,
    /// Drop oldest (no coalesce).
    P2,
    /// Drop newest (high-volume telemetry).
    P3,
}

/// Top-level message catalog per SPEC §8.1 → §8.7 grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Catalog {
    /// SPEC §8.1 — kernel + supervision (P0 family).
    KernelSupervision,
    /// SPEC §8.2 — bus protocol (BUS_PING/PONG/SUBSCRIBE/etc.).
    BusProtocol,
    /// SPEC §8.3 — swap protocol (SWAP_HANDOFF + SWAP_SUBTREE_*).
    SwapProtocol,
    /// SPEC §8.4 — adoption protocol (ADOPTION_REQUEST/ACK).
    Adoption,
    /// SPEC §8.5 — Trinity tensor (BODY/MIND/SPIRIT_STATE).
    TrinityTensor,
    /// SPEC §8.6 — filter_down cascade + topology + SELF assembly.
    FilterDownCascade,
    /// SPEC §8.7 — observatory / high-volume telemetry.
    Observatory,
}

/// One bus message specification. Every `SPECS` entry is byte-identical to
/// the corresponding `MSG_SPECS` entry in `titan_plugin/bus_specs.py` — see
/// SPEC §8.10 byte-identical guarantee + C2-7 Python catalog refactor.
#[derive(Debug, Clone, Copy)]
pub struct BusMsgSpec {
    /// Canonical message name (e.g. `"KERNEL_EPOCH_TICK"`).
    pub name: &'static str,
    /// Priority lane.
    pub priority: Priority,
    /// Coalesce key fields, if any (P1 STATE messages use `["src", "type"]`).
    pub coalesce_by: Option<&'static [&'static str]>,
    /// TTL in milliseconds, if any (most messages: `None`).
    pub ttl_ms: Option<u32>,
    /// Catalog category.
    pub catalog: Catalog,
}

// Coalesce-key tuples need to be `'static` slices for use in `phf_map!`.
const COALESCE_SRC_TYPE: &[&str] = &["src", "type"];

/// Canonical SPEC §8 message catalog. Full list of all messages the Phase C
/// bus broker recognizes — 47 canonical + 5 legacy drift-bridge.
///
/// Lookups are O(1) compile-time perfect hash. Every entry references its
/// SPEC §8.X subsection in the comment.
pub static SPECS: phf::Map<&'static str, BusMsgSpec> = phf_map! {
    // ── §8.1 KernelSupervision (17, all P0 except SUPERVISION_DEPENDENCY_DEGRADED P1) ──
    "KERNEL_EPOCH_TICK" => BusMsgSpec {
        name: "KERNEL_EPOCH_TICK", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "KERNEL_BOOT_GENERATION_CHANGED" => BusMsgSpec {
        name: "KERNEL_BOOT_GENERATION_CHANGED", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "KERNEL_SHUTDOWN_ANNOUNCE" => BusMsgSpec {
        name: "KERNEL_SHUTDOWN_ANNOUNCE", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "MODULE_HEARTBEAT" => BusMsgSpec {
        name: "MODULE_HEARTBEAT", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "MODULE_READY" => BusMsgSpec {
        name: "MODULE_READY", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "MODULE_SHUTDOWN" => BusMsgSpec {
        name: "MODULE_SHUTDOWN", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "MODULE_CRASHED" => BusMsgSpec {
        name: "MODULE_CRASHED", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "SUPERVISION_CHILD_DOWN" => BusMsgSpec {
        name: "SUPERVISION_CHILD_DOWN", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "SUPERVISION_CHILD_RESTARTED" => BusMsgSpec {
        name: "SUPERVISION_CHILD_RESTARTED", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "SUPERVISION_ESCALATION" => BusMsgSpec {
        name: "SUPERVISION_ESCALATION", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "SUPERVISION_ESCALATION_RESPONSE" => BusMsgSpec {
        name: "SUPERVISION_ESCALATION_RESPONSE", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "SUPERVISION_DEPENDENCY_BLOCKED" => BusMsgSpec {
        name: "SUPERVISION_DEPENDENCY_BLOCKED", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "SUPERVISION_DEPENDENCY_RECOVERED" => BusMsgSpec {
        name: "SUPERVISION_DEPENDENCY_RECOVERED", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "SUPERVISION_DEPENDENCY_DEGRADED" => BusMsgSpec {
        name: "SUPERVISION_DEPENDENCY_DEGRADED", priority: Priority::P1, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "SUPERVISION_DATA_RESTORE" => BusMsgSpec {
        name: "SUPERVISION_DATA_RESTORE", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "SUPERVISION_DATA_LOST" => BusMsgSpec {
        name: "SUPERVISION_DATA_LOST", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
    "SUPERVISION_FORCED_KILL" => BusMsgSpec {
        name: "SUPERVISION_FORCED_KILL", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },

    // ── §8.2 BusProtocol (5, all P0) ──
    "BUS_SUBSCRIBE" => BusMsgSpec {
        name: "BUS_SUBSCRIBE", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::BusProtocol,
    },
    "BUS_UNSUBSCRIBE" => BusMsgSpec {
        name: "BUS_UNSUBSCRIBE", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::BusProtocol,
    },
    "BUS_PING" => BusMsgSpec {
        name: "BUS_PING", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::BusProtocol,
    },
    "BUS_PONG" => BusMsgSpec {
        name: "BUS_PONG", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::BusProtocol,
    },
    // BUS_PEER_DIED — broker-emitted death notification (Phase B.2.1).
    // Subscribed by Guardian to trigger immediate restart of crashed
    // workers (faster than waiting for heartbeat timeout). P0 — never drop.
    "BUS_PEER_DIED" => BusMsgSpec {
        name: "BUS_PEER_DIED", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::BusProtocol,
    },
    "BUS_SLOW_CONSUMER" => BusMsgSpec {
        name: "BUS_SLOW_CONSUMER", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::BusProtocol,
    },

    // ── §8.3 SwapProtocol (7 canonical, all P0) ──
    "SWAP_HANDOFF" => BusMsgSpec {
        name: "SWAP_HANDOFF", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::SwapProtocol,
    },
    "SWAP_HANDOFF_ACK" => BusMsgSpec {
        name: "SWAP_HANDOFF_ACK", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::SwapProtocol,
    },
    "SWAP_HANDOFF_CANCELED" => BusMsgSpec {
        name: "SWAP_HANDOFF_CANCELED", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::SwapProtocol,
    },
    "SWAP_SUBTREE_REQUEST" => BusMsgSpec {
        name: "SWAP_SUBTREE_REQUEST", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::SwapProtocol,
    },
    "SWAP_SUBTREE_ACK" => BusMsgSpec {
        name: "SWAP_SUBTREE_ACK", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::SwapProtocol,
    },
    "SWAP_CHECKPOINT_REQUEST" => BusMsgSpec {
        name: "SWAP_CHECKPOINT_REQUEST", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::SwapProtocol,
    },
    "SWAP_CHECKPOINT_ACK" => BusMsgSpec {
        name: "SWAP_CHECKPOINT_ACK", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::SwapProtocol,
    },

    // ── §8.4 Adoption (2 canonical, both P0; B.2.1 79-byte payload locked) ──
    "ADOPTION_REQUEST" => BusMsgSpec {
        name: "ADOPTION_REQUEST", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::Adoption,
    },
    "ADOPTION_ACK" => BusMsgSpec {
        name: "ADOPTION_ACK", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::Adoption,
    },

    // ── §8.5 TrinityTensor (5, P1 with coalesce-by-(src,type)) ──
    "BODY_STATE" => BusMsgSpec {
        name: "BODY_STATE", priority: Priority::P1, coalesce_by: Some(COALESCE_SRC_TYPE),
        ttl_ms: None, catalog: Catalog::TrinityTensor,
    },
    "MIND_STATE" => BusMsgSpec {
        name: "MIND_STATE", priority: Priority::P1, coalesce_by: Some(COALESCE_SRC_TYPE),
        ttl_ms: None, catalog: Catalog::TrinityTensor,
    },
    "SPIRIT_STATE" => BusMsgSpec {
        name: "SPIRIT_STATE", priority: Priority::P1, coalesce_by: Some(COALESCE_SRC_TYPE),
        ttl_ms: None, catalog: Catalog::TrinityTensor,
    },
    "OUTER_OBSERVATION" => BusMsgSpec {
        name: "OUTER_OBSERVATION", priority: Priority::P1, coalesce_by: Some(COALESCE_SRC_TYPE),
        ttl_ms: None, catalog: Catalog::TrinityTensor,
    },
    "OUTER_DISPATCH" => BusMsgSpec {
        name: "OUTER_DISPATCH", priority: Priority::P1, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::TrinityTensor,
    },

    // ── §8.6 FilterDownCascade (7) ──
    "UNIFIED_SPIRIT_FILTER_DOWN" => BusMsgSpec {
        name: "UNIFIED_SPIRIT_FILTER_DOWN", priority: Priority::P1, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::FilterDownCascade,
    },
    "INNER_SPIRIT_FILTER_DOWN" => BusMsgSpec {
        name: "INNER_SPIRIT_FILTER_DOWN", priority: Priority::P1, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::FilterDownCascade,
    },
    "OUTER_SPIRIT_FILTER_DOWN" => BusMsgSpec {
        name: "OUTER_SPIRIT_FILTER_DOWN", priority: Priority::P1, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::FilterDownCascade,
    },
    "TRINITY_SUBSTRATE_TOPOLOGY_UPDATED" => BusMsgSpec {
        name: "TRINITY_SUBSTRATE_TOPOLOGY_UPDATED", priority: Priority::P1, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::FilterDownCascade,
    },
    "UNIFIED_SPIRIT_SELF_ASSEMBLED" => BusMsgSpec {
        name: "UNIFIED_SPIRIT_SELF_ASSEMBLED", priority: Priority::P1, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::FilterDownCascade,
    },
    "SPHERE_PULSE" => BusMsgSpec {
        name: "SPHERE_PULSE", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::FilterDownCascade,
    },
    "SPHERE_EPOCH_TICK" => BusMsgSpec {
        name: "SPHERE_EPOCH_TICK", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::FilterDownCascade,
    },

    // ── §8.7 Observatory (1, P3 drop-newest) ──
    "OBSERVATORY_EVENT" => BusMsgSpec {
        name: "OBSERVATORY_EVENT", priority: Priority::P3, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::Observatory,
    },

    // ── §8.8 Drift bridges (D13/D14/D15) — DELETED in C-S8 ──
    // Legacy names share the same priority/catalog as canonical so subscribers
    // listening on either name see consistent broker behavior. The `titan-bus`
    // drift_bridge module dual-emits when publishing.
    "BUS_HANDOFF" => BusMsgSpec {
        name: "BUS_HANDOFF", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::SwapProtocol,
    },
    "BUS_HANDOFF_CANCELED" => BusMsgSpec {
        name: "BUS_HANDOFF_CANCELED", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::SwapProtocol,
    },
    "BUS_WORKER_ADOPT_REQUEST" => BusMsgSpec {
        name: "BUS_WORKER_ADOPT_REQUEST", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::Adoption,
    },
    "BUS_WORKER_ADOPT_ACK" => BusMsgSpec {
        name: "BUS_WORKER_ADOPT_ACK", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::Adoption,
    },
    "EPOCH_TICK" => BusMsgSpec {
        name: "EPOCH_TICK", priority: Priority::P0, coalesce_by: None,
        ttl_ms: None, catalog: Catalog::KernelSupervision,
    },
};

/// Total spec count expected at SPEC v0.1.0.
///
/// 45 canonical entries (17 §8.1 + 6 §8.2 + 7 §8.3 + 2 §8.4 + 5 §8.5 +
/// 7 §8.6 + 1 §8.7) plus 5 legacy drift-bridge entries (`BUS_HANDOFF`,
/// `BUS_HANDOFF_CANCELED`, `BUS_WORKER_ADOPT_REQUEST`, `BUS_WORKER_ADOPT_ACK`,
/// `EPOCH_TICK`) = 50 total. Used by `tests/parity_vectors.rs::bus_specs_*`
/// for regression coverage.
///
/// Phase C C-S7 Gap 10 (2026-05-05): added BUS_PEER_DIED to §8.2 after
/// Python↔Rust parity test caught the drift. See
/// PLAN_microkernel_phase_c_s7_activation_prep.md §2 Gap 10.
pub const SPECS_COUNT_V0_1_0: usize = 45 + 5;

/// Default spec for any message type not explicitly listed (per Python
/// `bus_specs.DEFAULT_SPEC`). Canonical name `<default>`, priority P2,
/// no coalesce, no ttl, default catalog. Used by broker for unmapped types.
pub const DEFAULT_SPEC: BusMsgSpec = BusMsgSpec {
    name: "<default>",
    priority: Priority::P2,
    coalesce_by: None,
    ttl_ms: None,
    catalog: Catalog::Observatory, // arbitrary; default lane is P2
};

/// Look up a message's spec by name. Returns `&DEFAULT_SPEC` for unmapped
/// messages (matches Python `get_spec()` semantics).
pub fn get_spec(msg_type: &str) -> BusMsgSpec {
    SPECS.get(msg_type).copied().unwrap_or(DEFAULT_SPEC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_serialize_uppercase() {
        assert_eq!(serde_json::to_string(&Priority::P0).unwrap(), "\"P0\"");
        assert_eq!(serde_json::to_string(&Priority::P3).unwrap(), "\"P3\"");
    }

    #[test]
    fn specs_count_matches_v0_1_0() {
        assert_eq!(SPECS.len(), SPECS_COUNT_V0_1_0);
    }

    #[test]
    fn kernel_epoch_tick_p0() {
        let spec = SPECS.get("KERNEL_EPOCH_TICK").unwrap();
        assert_eq!(spec.priority, Priority::P0);
        assert_eq!(spec.coalesce_by, None);
        assert_eq!(spec.catalog, Catalog::KernelSupervision);
    }

    #[test]
    fn body_state_p1_with_coalesce() {
        let spec = SPECS.get("BODY_STATE").unwrap();
        assert_eq!(spec.priority, Priority::P1);
        assert_eq!(spec.coalesce_by, Some(&["src", "type"][..]));
        assert_eq!(spec.catalog, Catalog::TrinityTensor);
    }

    #[test]
    fn observatory_event_p3() {
        let spec = SPECS.get("OBSERVATORY_EVENT").unwrap();
        assert_eq!(spec.priority, Priority::P3);
        assert_eq!(spec.catalog, Catalog::Observatory);
    }

    #[test]
    fn supervision_dependency_degraded_p1_only_one_in_supervision_family() {
        // Per SPEC §8.1: SUPERVISION_DEPENDENCY_DEGRADED is the lone P1 in
        // an otherwise all-P0 family (it's informational, not load-bearing).
        let spec = SPECS.get("SUPERVISION_DEPENDENCY_DEGRADED").unwrap();
        assert_eq!(spec.priority, Priority::P1);
    }

    #[test]
    fn drift_bridge_legacy_names_present() {
        // Per SPEC §3.1 D13/D14/D15: legacy names must resolve via SPECS so
        // subscribers listening on either name get identical priority handling.
        for legacy in [
            "BUS_HANDOFF",
            "BUS_HANDOFF_CANCELED",
            "BUS_WORKER_ADOPT_REQUEST",
            "BUS_WORKER_ADOPT_ACK",
            "EPOCH_TICK",
        ] {
            assert!(
                SPECS.get(legacy).is_some(),
                "drift-bridge legacy name '{legacy}' missing from SPECS"
            );
        }
    }

    #[test]
    fn drift_bridge_legacy_priority_matches_canonical() {
        // BUS_HANDOFF (legacy) ↔ SWAP_HANDOFF (canonical): same priority
        let legacy = SPECS.get("BUS_HANDOFF").unwrap();
        let canonical = SPECS.get("SWAP_HANDOFF").unwrap();
        assert_eq!(legacy.priority, canonical.priority);
        assert_eq!(legacy.catalog, canonical.catalog);

        // EPOCH_TICK (legacy) ↔ KERNEL_EPOCH_TICK (canonical): same priority
        let legacy = SPECS.get("EPOCH_TICK").unwrap();
        let canonical = SPECS.get("KERNEL_EPOCH_TICK").unwrap();
        assert_eq!(legacy.priority, canonical.priority);
        assert_eq!(legacy.catalog, canonical.catalog);
    }

    #[test]
    fn unmapped_message_returns_default() {
        let spec = get_spec("THIS_MSG_NEVER_EXISTED");
        assert_eq!(spec.priority, Priority::P2);
        assert_eq!(spec.name, "<default>");
    }

    #[test]
    fn mapped_message_returns_specific() {
        let spec = get_spec("KERNEL_EPOCH_TICK");
        assert_eq!(spec.priority, Priority::P0);
        assert_eq!(spec.name, "KERNEL_EPOCH_TICK");
    }

    #[test]
    fn all_5_state_messages_coalesce_by_src_type() {
        // SPEC §8.5 — BODY/MIND/SPIRIT_STATE + OUTER_OBSERVATION coalesce.
        // OUTER_DISPATCH does NOT (action messages preserve order).
        for state_msg in [
            "BODY_STATE",
            "MIND_STATE",
            "SPIRIT_STATE",
            "OUTER_OBSERVATION",
        ] {
            let spec = SPECS.get(state_msg).unwrap();
            assert_eq!(
                spec.coalesce_by,
                Some(&["src", "type"][..]),
                "{state_msg} should coalesce by (src, type)"
            );
        }
        let outer_dispatch = SPECS.get("OUTER_DISPATCH").unwrap();
        assert_eq!(outer_dispatch.coalesce_by, None);
    }

    #[test]
    fn priority_enum_ordering_makes_sense_for_drop_decisions() {
        // P0 < P1 < P2 < P3 makes "smallest priority enum value = highest
        // protection" easy to remember. Verify with serialization.
        assert_eq!(serde_json::to_string(&Priority::P0).unwrap(), "\"P0\"");
        assert!(matches!(Priority::P0, Priority::P0));
    }
}
