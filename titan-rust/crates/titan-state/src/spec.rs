//! spec — Declarative slot specifications mirroring SPEC §7.1.
//!
//! Each [`SlotSpec`] entry declares a shm slot's name, schema version,
//! payload size, and creator. The kernel boot path walks `SLOT_SPECS` and
//! creates every entry whose `creator == SlotCreator::Kernel`. Other entries
//! are listed for documentation + arch_map cross-checking but created
//! lazily by their Python writers.
//!
//! Per `feedback_phase_c_spec_enforcement.md` Rule 1: every value in this
//! table comes from `titan-core::constants` (auto-generated from SPEC TOML)
//! or directly from SPEC §7.1 byte-counted payload sizes.

use titan_core::constants::*;

/// Who creates the slot file at boot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotCreator {
    /// `titan-kernel-rs` creates this slot at boot per SPEC §10.A B3 step.
    Kernel,
    /// `titan-trinity-rs` substrate creates this slot (C-S3 onward).
    Substrate,
    /// `titan-unified-spirit-rs` creates this slot (C-S4 onward).
    UnifiedSpirit,
    /// Created lazily by a Python sensor refresh task in `titan_HCL`.
    PythonSensorRefresh,
    /// Created lazily by a named Python L2/L3 module.
    PythonModule(&'static str),
}

/// One slot specification.
#[derive(Debug, Clone, Copy)]
pub struct SlotSpec {
    /// Filename within `/dev/shm/titan_<id>/` (e.g. `"self_162d.bin"`).
    pub name: &'static str,
    /// Per-slot schema version constant (per SPEC §3.1 D05).
    pub schema_version: u32,
    /// Payload size in bytes (`0` = variable; `max_payload_bytes` is the cap).
    pub payload_bytes: u32,
    /// Maximum payload size (for variable-size slots like `cgn_live_weights`).
    pub max_payload_bytes: u32,
    /// Who creates the file at boot.
    pub creator: SlotCreator,
    /// Documentation: who is the canonical writer (single-writer per slot).
    pub writer: &'static str,
}

impl SlotSpec {
    /// Total bytes for the `fastbus.bin` self-contained SPSC ring file.
    /// Per SPEC §7.1: fastbus.bin is EXCLUDED from the §7.0 universal
    /// triple-buffer header — it is a flat file with the titan-fastbus
    /// crate's own ring header at offset 24 + ring slots, sized by
    /// `FASTBUS_FILE_TOTAL_BYTES` (currently 262232 = 24 legacy prefix +
    /// 64 ring header + 1024 × 256-byte slots).
    ///
    /// This helper is callable on any SlotSpec but only meaningful for
    /// `name == "fastbus.bin"`.
    pub fn fastbus_total_bytes(&self) -> u64 {
        // The 24-byte legacy prefix is preserved as unused space ahead of
        // the fastbus ring header so titan-fastbus's existing offset math
        // (header_offset = 24) keeps working without changes. The new
        // §7.0 universal-header format does not apply to fastbus.bin —
        // see SPEC §7.1 fastbus.bin row note.
        24 + self.payload_bytes as u64
    }
}

/// All Phase C shm slots per SPEC §7.1 + §9.D sensor caches.
///
/// **Total: 19 slots.** The kernel creates 16 of them at boot (every
/// `SlotCreator::Kernel` row); the other 3 (neuromod_state, titanvm_registers,
/// 5 sensor caches) are created lazily by their Python writers. The
/// `cgn_live_weights.bin` slot lives separately in the `titan-cgn` crate's
/// spec (per SPEC §18.2: kernel creates ONLY this one CGN slot).
pub const SLOT_SPECS: &[SlotSpec] = &[
    // ── Tensor slots (kernel creates; daemons + L2 read+write) ──────────
    SlotSpec {
        name: "self_162d.bin",
        schema_version: SELF_162D_SCHEMA_VERSION as u32,
        payload_bytes: 162 * 4, // 162 × float32 LE = 648
        max_payload_bytes: 162 * 4,
        creator: SlotCreator::Kernel,
        writer: "titan-unified-spirit-rs",
    },
    SlotSpec {
        name: "inner_body_5d.bin",
        schema_version: INNER_BODY_5D_SCHEMA_VERSION as u32,
        payload_bytes: 5 * 4,
        max_payload_bytes: 5 * 4,
        creator: SlotCreator::Kernel,
        writer: "titan-inner-body-rs",
    },
    SlotSpec {
        name: "inner_mind_15d.bin",
        schema_version: INNER_MIND_15D_SCHEMA_VERSION as u32,
        payload_bytes: 15 * 4,
        max_payload_bytes: 15 * 4,
        creator: SlotCreator::Kernel,
        writer: "titan-inner-mind-rs",
    },
    SlotSpec {
        name: "inner_spirit_45d.bin",
        schema_version: INNER_SPIRIT_45D_SCHEMA_VERSION as u32,
        payload_bytes: 45 * 4,
        max_payload_bytes: 45 * 4,
        creator: SlotCreator::Kernel,
        writer: "titan-inner-spirit-rs",
    },
    SlotSpec {
        name: "outer_body_5d.bin",
        schema_version: OUTER_BODY_5D_SCHEMA_VERSION as u32,
        payload_bytes: 5 * 4,
        max_payload_bytes: 5 * 4,
        creator: SlotCreator::Kernel,
        writer: "titan-outer-body-rs",
    },
    SlotSpec {
        name: "outer_mind_15d.bin",
        schema_version: OUTER_MIND_15D_SCHEMA_VERSION as u32,
        payload_bytes: 15 * 4,
        max_payload_bytes: 15 * 4,
        creator: SlotCreator::Kernel,
        writer: "titan-outer-mind-rs",
    },
    SlotSpec {
        name: "outer_spirit_45d.bin",
        schema_version: OUTER_SPIRIT_45D_SCHEMA_VERSION as u32,
        payload_bytes: 45 * 4,
        max_payload_bytes: 45 * 4,
        creator: SlotCreator::Kernel,
        writer: "titan-outer-spirit-rs",
    },
    SlotSpec {
        name: "topology_30d.bin",
        schema_version: TOPOLOGY_30D_SCHEMA_VERSION as u32,
        payload_bytes: 30 * 4,
        max_payload_bytes: 30 * 4,
        creator: SlotCreator::Kernel,
        writer: "titan-trinity-rs",
    },
    SlotSpec {
        name: "unified_spirit_132d.bin",
        schema_version: UNIFIED_SPIRIT_132D_SCHEMA_VERSION as u32,
        payload_bytes: 132 * 4,
        max_payload_bytes: 132 * 4,
        creator: SlotCreator::Kernel,
        writer: "titan-unified-spirit-rs",
    },
    // ── Control + state slots (kernel writes) ─────────────────────────────
    SlotSpec {
        name: "epoch_counter.bin",
        schema_version: EPOCH_COUNTER_SCHEMA_VERSION as u32,
        payload_bytes: 8, // 1 × uint64 LE
        max_payload_bytes: 8,
        creator: SlotCreator::Kernel,
        writer: "titan-kernel-rs",
    },
    SlotSpec {
        name: "circadian.bin",
        schema_version: CIRCADIAN_SCHEMA_VERSION as u32,
        payload_bytes: 12, // f32 phase + f32 day_progress + f32 reserved
        max_payload_bytes: 12,
        creator: SlotCreator::Kernel,
        writer: "titan-kernel-rs",
    },
    SlotSpec {
        name: "pi_heartbeat.bin",
        schema_version: PI_HEARTBEAT_SCHEMA_VERSION as u32,
        payload_bytes: 12, // f32 phase + uint64 pulse_count
        max_payload_bytes: 12,
        creator: SlotCreator::Kernel,
        writer: "titan-kernel-rs",
    },
    SlotSpec {
        name: "sphere_clocks.bin",
        schema_version: SPHERE_CLOCKS_SCHEMA_VERSION as u32,
        payload_bytes: 6 * 7 * 4, // 6 clocks × 7 fields × float32 = 168
        max_payload_bytes: 6 * 7 * 4,
        creator: SlotCreator::Kernel,
        writer: "titan-trinity-rs",
    },
    SlotSpec {
        name: "chi_state.bin",
        schema_version: CHI_STATE_SCHEMA_VERSION as u32,
        payload_bytes: 6 * 4, // 6 × float32
        max_payload_bytes: 6 * 4,
        creator: SlotCreator::Kernel,
        writer: "titan-trinity-rs",
    },
    SlotSpec {
        name: "identity.bin",
        schema_version: IDENTITY_SCHEMA_VERSION as u32,
        payload_bytes: 96, // 32 + 32 + 32
        max_payload_bytes: 96,
        creator: SlotCreator::Kernel,
        writer: "titan-kernel-rs",
    },
    SlotSpec {
        name: "fastbus.bin",
        schema_version: FASTBUS_SCHEMA_VERSION as u32,
        // 64-byte ring header + 1024 × 256-byte slots = 262208 bytes
        payload_bytes: FASTBUS_HEADER_BYTES as u32
            + (FASTBUS_RING_CAPACITY_SLOTS as u32 * FASTBUS_SLOT_BYTES as u32),
        max_payload_bytes: FASTBUS_HEADER_BYTES as u32
            + (FASTBUS_RING_CAPACITY_SLOTS as u32 * FASTBUS_SLOT_BYTES as u32),
        creator: SlotCreator::Kernel,
        writer: "titan-trinity-rs",
    },
    // ── Domain registry (created lazily by Python modules; documented here for arch_map) ──
    SlotSpec {
        name: "neuromod_state.bin",
        schema_version: NEUROMOD_SCHEMA_VERSION as u32,
        payload_bytes: 6 * 4, // 6 neuromods × float32
        max_payload_bytes: 6 * 4,
        creator: SlotCreator::PythonModule("neuromod_module"),
        writer: "neuromod_module",
    },
    SlotSpec {
        name: "titanvm_registers.bin",
        schema_version: TITANVM_REGISTERS_SCHEMA_VERSION as u32,
        payload_bytes: 11 * 4 * 4, // 11 NS programs × 4 fields × float32 = 176
        max_payload_bytes: 11 * 4 * 4,
        creator: SlotCreator::PythonModule("ns_module"),
        writer: "ns_module",
    },
    SlotSpec {
        name: "hormonal_state.bin",
        schema_version: HORMONAL_STATE_SCHEMA_VERSION as u32,
        payload_bytes: 11 * 4 * 4, // 11 hormones × 4 fields × float32 = 176
        max_payload_bytes: 11 * 4 * 4,
        creator: SlotCreator::PythonModule("hormonal_module"),
        writer: "hormonal_module",
    },
    // ── Sensor cache slots (Python sensor refresh tasks write; Rust daemons read) ──
    SlotSpec {
        name: "sensor_cache_inner_body.bin",
        schema_version: 1,
        payload_bytes: 0, // variable
        max_payload_bytes: 4096,
        creator: SlotCreator::PythonSensorRefresh,
        writer: "python_inner_body_sensor_refresh",
    },
    SlotSpec {
        name: "sensor_cache_inner_mind.bin",
        schema_version: 1,
        payload_bytes: 0,
        max_payload_bytes: 4096,
        creator: SlotCreator::PythonSensorRefresh,
        writer: "python_inner_mind_sensor_refresh",
    },
    SlotSpec {
        name: "sensor_cache_outer_body.bin",
        schema_version: 1,
        payload_bytes: 0,
        max_payload_bytes: OUTER_SENSOR_CACHE_BODY_MAX_BYTES as u32,
        creator: SlotCreator::PythonSensorRefresh,
        writer: "python_outer_body_sensor_refresh",
    },
    SlotSpec {
        name: "sensor_cache_outer_mind.bin",
        schema_version: 1,
        payload_bytes: 0,
        max_payload_bytes: OUTER_SENSOR_CACHE_MIND_MAX_BYTES as u32,
        creator: SlotCreator::PythonSensorRefresh,
        writer: "python_outer_mind_sensor_refresh",
    },
    SlotSpec {
        name: "sensor_cache_outer_spirit.bin",
        schema_version: 1,
        payload_bytes: 0,
        max_payload_bytes: OUTER_SENSOR_CACHE_SPIRIT_MAX_BYTES as u32,
        creator: SlotCreator::PythonSensorRefresh,
        writer: "python_outer_spirit_sensor_refresh",
    },
];

/// Number of slots that the kernel creates at boot (per SPEC §10.A B3).
///
/// 9 tensor slots (`self_162d` + 6 trinity + `topology_30d` + `unified_spirit_132d`)
/// plus 7 control/state slots (`epoch_counter`, `circadian`, `pi_heartbeat`,
/// `sphere_clocks`, `chi_state`, `identity`, `fastbus`) = 16 total.
pub const KERNEL_CREATED_COUNT: usize = 16;

/// Total slot count across all creators.
///
/// 16 kernel-created + 3 Python-managed (`neuromod_state`, `titanvm_registers`,
/// `hormonal_state` — added in SPEC v0.1.4 per master plan §10 D22) plus 5
/// sensor cache slots (Python sensor refresh tasks) = 24 total.
/// Note that `cgn_live_weights.bin` lives in `titan-cgn` instead, per SPEC §18.2.
pub const TOTAL_SLOT_COUNT: usize = 24;

/// Iterate over slots whose creator is `Kernel` — used by `SlotRegistry::create_all`.
pub fn kernel_slots() -> impl Iterator<Item = &'static SlotSpec> {
    SLOT_SPECS
        .iter()
        .filter(|s| matches!(s.creator, SlotCreator::Kernel))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_count_matches_constant() {
        assert_eq!(SLOT_SPECS.len(), TOTAL_SLOT_COUNT);
    }

    #[test]
    fn kernel_slot_count_matches_constant() {
        assert_eq!(kernel_slots().count(), KERNEL_CREATED_COUNT);
    }

    #[test]
    fn self_162d_byte_layout_matches_spec_7_1() {
        // SPEC §7.1: 162 × float32 LE = 648 payload bytes
        let spec = SLOT_SPECS
            .iter()
            .find(|s| s.name == "self_162d.bin")
            .unwrap();
        assert_eq!(spec.payload_bytes, 648);
    }

    #[test]
    fn fastbus_size_matches_spec_9_2() {
        // SPEC §9.2: 64B header + 1024 × 256B = 262208 bytes total payload
        let spec = SLOT_SPECS.iter().find(|s| s.name == "fastbus.bin").unwrap();
        assert_eq!(spec.payload_bytes, 262208);
    }

    #[test]
    fn identity_slot_is_sacred_size() {
        // SPEC §7.1: identity.bin = 96 bytes (32 titan_id + 32 maker + 32 nonce)
        let spec = SLOT_SPECS
            .iter()
            .find(|s| s.name == "identity.bin")
            .unwrap();
        assert_eq!(spec.payload_bytes, 96);
        assert_eq!(spec.creator, SlotCreator::Kernel);
    }

    #[test]
    fn cgn_slot_not_in_titan_state_specs() {
        // Per SPEC §18.2 + PLAN §13.1 chunk C2-3: cgn_live_weights lives in
        // titan-cgn, NOT here. Verifies architectural separation.
        assert!(!SLOT_SPECS.iter().any(|s| s.name.contains("cgn")));
    }

    #[test]
    fn no_kernel_slot_has_zero_payload() {
        // Variable-size (payload_bytes=0) slots must NOT be kernel-created
        // — the kernel allocates exact byte counts at boot. Variable slots
        // are Python-managed (sensor caches, cgn_live_weights).
        for spec in kernel_slots() {
            assert_ne!(
                spec.payload_bytes, 0,
                "kernel-created slot {} has zero payload_bytes",
                spec.name
            );
        }
    }

    #[test]
    fn schema_versions_all_v1_at_v0_1_0() {
        // Per SPEC §3.1 D05: every per-slot schema version starts at 1.
        for spec in SLOT_SPECS.iter() {
            assert_eq!(
                spec.schema_version, 1,
                "{} schema_version should be 1 at SPEC v0.1.0",
                spec.name
            );
        }
    }
}
