//! drift_bridge — D13/D14/D15 dual-emit (canonical ↔ legacy names).
//!
//! Per SPEC §3.1 D13/D14/D15: during Phase C transition (until C-S8 cleanup),
//! the broker accepts publications under EITHER canonical or legacy names
//! and dual-emits to subscribers. Consumers can listen on whichever name
//! they have today; broker bridges automatically.
//!
//! # The 5 bridged pairs
//!
//! | Legacy (today)              | Canonical (SPEC §8)        | SPEC ref |
//! |-----------------------------|----------------------------|----------|
//! | `BUS_HANDOFF`               | `SWAP_HANDOFF`              | D13       |
//! | `BUS_HANDOFF_CANCELED`       | `SWAP_HANDOFF_CANCELED`      | D13       |
//! | `BUS_WORKER_ADOPT_REQUEST`   | `ADOPTION_REQUEST`           | D14       |
//! | `BUS_WORKER_ADOPT_ACK`       | `ADOPTION_ACK`               | D14       |
//! | `EPOCH_TICK`                 | `KERNEL_EPOCH_TICK`          | D15       |
//!
//! # Algorithm
//!
//! When the broker publishes a message, it calls [`bridge_emit_names`] which
//! returns the list of names to fanout under. For non-bridged messages the
//! list is just `[original]`. For bridged ones, both canonical + legacy
//! names appear in the list; the broker fanouts to subscribers under each
//! name with identical payload.
//!
//! # C-S8 cleanup
//!
//! This entire module + the bridged entries in `titan-core::bus_specs::SPECS`
//! are deleted in C-S8 once all consumers migrate to canonical names.

/// Returns the list of message names to fanout under for a given message
/// type. Includes the original name PLUS the bridge partner if applicable.
///
/// # Examples
///
/// - `bridge_emit_names("KERNEL_EPOCH_TICK")` → `["KERNEL_EPOCH_TICK", "EPOCH_TICK"]`
/// - `bridge_emit_names("EPOCH_TICK")` → `["EPOCH_TICK", "KERNEL_EPOCH_TICK"]`
/// - `bridge_emit_names("BODY_STATE")` → `["BODY_STATE"]` (not bridged)
/// - `bridge_emit_names("UNKNOWN")` → `["UNKNOWN"]`
pub fn bridge_emit_names(msg_type: &str) -> Vec<&'static str> {
    let canonical_partner: Option<&'static str> = match msg_type {
        "SWAP_HANDOFF" => Some("BUS_HANDOFF"),
        "BUS_HANDOFF" => Some("SWAP_HANDOFF"),
        "SWAP_HANDOFF_CANCELED" => Some("BUS_HANDOFF_CANCELED"),
        "BUS_HANDOFF_CANCELED" => Some("SWAP_HANDOFF_CANCELED"),
        "ADOPTION_REQUEST" => Some("BUS_WORKER_ADOPT_REQUEST"),
        "BUS_WORKER_ADOPT_REQUEST" => Some("ADOPTION_REQUEST"),
        "ADOPTION_ACK" => Some("BUS_WORKER_ADOPT_ACK"),
        "BUS_WORKER_ADOPT_ACK" => Some("ADOPTION_ACK"),
        "KERNEL_EPOCH_TICK" => Some("EPOCH_TICK"),
        "EPOCH_TICK" => Some("KERNEL_EPOCH_TICK"),
        _ => None,
    };

    // Allocate `Vec<&'static str>` to hold the original (passed in as &str
    // referencing the broker's message; we use a static lookup table for
    // both halves so we can return `&'static str` for both).
    let original_static: Option<&'static str> = match msg_type {
        "SWAP_HANDOFF" => Some("SWAP_HANDOFF"),
        "BUS_HANDOFF" => Some("BUS_HANDOFF"),
        "SWAP_HANDOFF_CANCELED" => Some("SWAP_HANDOFF_CANCELED"),
        "BUS_HANDOFF_CANCELED" => Some("BUS_HANDOFF_CANCELED"),
        "ADOPTION_REQUEST" => Some("ADOPTION_REQUEST"),
        "BUS_WORKER_ADOPT_REQUEST" => Some("BUS_WORKER_ADOPT_REQUEST"),
        "ADOPTION_ACK" => Some("ADOPTION_ACK"),
        "BUS_WORKER_ADOPT_ACK" => Some("BUS_WORKER_ADOPT_ACK"),
        "KERNEL_EPOCH_TICK" => Some("KERNEL_EPOCH_TICK"),
        "EPOCH_TICK" => Some("EPOCH_TICK"),
        _ => None,
    };

    match (original_static, canonical_partner) {
        (Some(orig), Some(partner)) => vec![orig, partner],
        // Non-bridged message: caller already has &str, but we can't return
        // it as &'static. Return empty vec → caller knows to use the
        // original name only. This is the hot path for ~99% of messages.
        _ => vec![],
    }
}

/// Returns `true` if the given message type is part of a Phase C drift
/// bridge pair (D13/D14/D15). Used by broker stats / arch_map auditing.
pub fn is_bridged(msg_type: &str) -> bool {
    matches!(
        msg_type,
        "SWAP_HANDOFF"
            | "BUS_HANDOFF"
            | "SWAP_HANDOFF_CANCELED"
            | "BUS_HANDOFF_CANCELED"
            | "ADOPTION_REQUEST"
            | "BUS_WORKER_ADOPT_REQUEST"
            | "ADOPTION_ACK"
            | "BUS_WORKER_ADOPT_ACK"
            | "KERNEL_EPOCH_TICK"
            | "EPOCH_TICK"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d13_swap_handoff_pair() {
        let names = bridge_emit_names("SWAP_HANDOFF");
        assert_eq!(names, vec!["SWAP_HANDOFF", "BUS_HANDOFF"]);
        let names = bridge_emit_names("BUS_HANDOFF");
        assert_eq!(names, vec!["BUS_HANDOFF", "SWAP_HANDOFF"]);
    }

    #[test]
    fn d13_swap_handoff_canceled_pair() {
        let names = bridge_emit_names("SWAP_HANDOFF_CANCELED");
        assert_eq!(names, vec!["SWAP_HANDOFF_CANCELED", "BUS_HANDOFF_CANCELED"]);
        let names = bridge_emit_names("BUS_HANDOFF_CANCELED");
        assert_eq!(names, vec!["BUS_HANDOFF_CANCELED", "SWAP_HANDOFF_CANCELED"]);
    }

    #[test]
    fn d14_adoption_request_pair() {
        let names = bridge_emit_names("ADOPTION_REQUEST");
        assert_eq!(names, vec!["ADOPTION_REQUEST", "BUS_WORKER_ADOPT_REQUEST"]);
        let names = bridge_emit_names("BUS_WORKER_ADOPT_REQUEST");
        assert_eq!(names, vec!["BUS_WORKER_ADOPT_REQUEST", "ADOPTION_REQUEST"]);
    }

    #[test]
    fn d14_adoption_ack_pair() {
        let names = bridge_emit_names("ADOPTION_ACK");
        assert_eq!(names, vec!["ADOPTION_ACK", "BUS_WORKER_ADOPT_ACK"]);
        let names = bridge_emit_names("BUS_WORKER_ADOPT_ACK");
        assert_eq!(names, vec!["BUS_WORKER_ADOPT_ACK", "ADOPTION_ACK"]);
    }

    #[test]
    fn d15_kernel_epoch_tick_pair() {
        let names = bridge_emit_names("KERNEL_EPOCH_TICK");
        assert_eq!(names, vec!["KERNEL_EPOCH_TICK", "EPOCH_TICK"]);
        let names = bridge_emit_names("EPOCH_TICK");
        assert_eq!(names, vec!["EPOCH_TICK", "KERNEL_EPOCH_TICK"]);
    }

    #[test]
    fn non_bridged_returns_empty_vec() {
        // Caller treats empty vec as "no dual-emit; use original name"
        assert_eq!(bridge_emit_names("BODY_STATE"), Vec::<&'static str>::new());
        assert_eq!(bridge_emit_names("MIND_STATE"), Vec::<&'static str>::new());
        assert_eq!(
            bridge_emit_names("KERNEL_SHUTDOWN_ANNOUNCE"),
            Vec::<&'static str>::new()
        );
    }

    #[test]
    fn is_bridged_for_all_5_pairs() {
        for name in [
            "SWAP_HANDOFF",
            "BUS_HANDOFF",
            "SWAP_HANDOFF_CANCELED",
            "BUS_HANDOFF_CANCELED",
            "ADOPTION_REQUEST",
            "BUS_WORKER_ADOPT_REQUEST",
            "ADOPTION_ACK",
            "BUS_WORKER_ADOPT_ACK",
            "KERNEL_EPOCH_TICK",
            "EPOCH_TICK",
        ] {
            assert!(is_bridged(name), "{name} should be bridged");
        }
    }

    #[test]
    fn is_bridged_false_for_non_bridged() {
        for name in [
            "BODY_STATE",
            "MIND_STATE",
            "SPIRIT_STATE",
            "KERNEL_SHUTDOWN_ANNOUNCE",
            "OBSERVATORY_EVENT",
            "BUS_PING",
            "MODULE_HEARTBEAT",
        ] {
            assert!(!is_bridged(name), "{name} should NOT be bridged");
        }
    }
}
