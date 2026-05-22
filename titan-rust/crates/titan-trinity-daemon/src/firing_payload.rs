//! firing_payload — Per-block firing-slot writer for `<block>_firing.bin`.
//!
//! Per `rFP_phase_c_130d_rust_l1_port.md` §4.7 + Phase 2.5.A.2 schema:
//! each Rust trinity daemon writes its block's `<block>_firing.bin` slot
//! after every successful canonical slot write. The slot carries the
//! diagnostic firing payload that `arch_map dim-live` reads to classify
//! each dim as ALIVE / ALIVE_AT_DEFAULT / PARTIAL / SILENT (vs the
//! pre-port GHOST artifact that occurred when dim-live's classifier
//! couldn't see a Phase C producer firing because only Python's
//! `DimFiringTracker.record_block` wrote the slot).
//!
//! # Schema parity with Python writer
//!
//! 1:1 byte-compatible with [`titan_hcl/api/dim_registry.py:444`]
//! `_block_payload_locked` msgpack encoding:
//!
//! ```text
//! {
//!   "block": "inner_body" | "inner_mind" | "inner_spirit"
//!          | "outer_body" | "outer_mind" | "outer_spirit",
//!   "block_calls_total": u64,
//!   "block_last_call_ts": f64 | nil,
//!   "inputs_state": {input_name: "real" | "default" | "absent"},
//!   "dims": [{"v": f64 | nil, "ts": f64 | nil}, ...],   # length = block size
//!   "ts": f64,
//! }
//! ```
//!
//! Rust producers always emit f64 values (never nil) for `dims[i].v` and
//! `block_last_call_ts` because the Rust tick path always computes the
//! full block tensor — the nil branches only arose in the Python tracker
//! when a producer hadn't yet fired (cold-start). Per-dim `ts` is the
//! same wall-clock as the block `ts` (Rust writes the whole block at once).
//!
//! # Single-writer per G21
//!
//! Each `<block>_firing.bin` slot has exactly one writer post-port:
//! the Rust trinity daemon that owns the block's canonical output slot.
//! Python `DimFiringTracker._publish_block_shm` becomes no-op under
//! `microkernel.l0_rust_enabled=true` (handled in companion Python edit).
//!
//! # Failure semantics
//!
//! `record_tick` failures are non-fatal — counters increment, throttled
//! WARN logs fire, but the daemon's tick path is not blocked. Mirrors the
//! Python tracker's defensive policy: firing-slot writes are diagnostic,
//! never load-bearing for cognition.

use std::path::{Path, PathBuf};

use rmpv::Value;
use titan_state::Slot;
use tracing::warn;

/// Throttle WARN logs after this many consecutive failures.
const WARN_THROTTLE_EVERY: u64 = 100;

/// Per-block firing-slot writer. One instance per Rust trinity daemon.
///
/// Open the slot via [`FiringSlotWriter::open`] at boot (after the
/// canonical block slot is open + the kernel has created the firing slot
/// file). Call [`FiringSlotWriter::record_tick`] after every successful
/// canonical slot write.
pub struct FiringSlotWriter {
    /// Block name used in the payload `block` field. Static lifetime
    /// because each daemon knows its block at compile time.
    block_name: &'static str,

    /// Slot path — kept for retry-open / create if the slot file isn't
    /// yet on disk at daemon boot.
    slot_path: PathBuf,

    /// Schema version for the firing slot (per-block constant from
    /// `titan_core::constants::*_FIRING_SCHEMA_VERSION`). Used when the
    /// slot file is missing and the daemon needs to create it.
    schema_version: u32,

    /// Max payload bytes for the firing slot (per-block constant from
    /// `titan_core::constants::*_FIRING_MAX_BYTES`). Slot capacity for
    /// `Slot::create` when slot file is missing.
    max_bytes: u32,

    /// Open slot handle. `None` when retry-open / create is still pending.
    slot: Option<Slot>,

    /// Tick count since daemon boot — monotonic. Mirrors Python
    /// `BlockFiringRecord.calls_total`.
    block_calls_total: u64,

    /// Wall-clock of the most recent successful `record_tick` call.
    /// Initialized to 0.0; first call sets it.
    last_call_ts: f64,

    /// Diagnostic — number of `record_tick` failures since boot.
    write_failures: u64,
}

impl FiringSlotWriter {
    /// Construct a writer for the given block. Does NOT open / create
    /// the slot yet — first `record_tick` call lazy-initializes it.
    ///
    /// `block_name` MUST be one of:
    ///   - `"inner_body"` / `"inner_mind"` / `"inner_spirit"`
    ///   - `"outer_body"` / `"outer_mind"` / `"outer_spirit"`
    ///
    /// `shm_dir` is the resolved per-Titan SHM root (e.g.
    /// `/dev/shm/titan_T3`); the slot file lives at
    /// `shm_dir/<block>_firing.bin`.
    ///
    /// `schema_version` + `max_bytes` are the per-block constants from
    /// `titan_core::constants::*_FIRING_SCHEMA_VERSION` /
    /// `*_FIRING_MAX_BYTES`. Used when the slot file is missing and
    /// this writer must create it (Phase C: Rust trinity daemons own
    /// the firing-slot lifecycle under `l0_rust_enabled=true`; Python
    /// tracker no-ops the SHM publish per single-writer G21).
    pub fn new(
        block_name: &'static str,
        shm_dir: &Path,
        schema_version: u32,
        max_bytes: u32,
    ) -> Self {
        let slot_path = shm_dir.join(format!("{block_name}_firing.bin"));
        Self {
            block_name,
            slot_path,
            schema_version,
            max_bytes,
            slot: None,
            block_calls_total: 0,
            last_call_ts: 0.0,
            write_failures: 0,
        }
    }

    /// Try to open the slot. If the file doesn't exist, create it with
    /// the per-block schema/capacity. Idempotent. Failure is non-fatal;
    /// caller continues, next `record_tick` retries.
    ///
    /// Slot lifecycle ownership follows G21 single-writer per slot:
    /// Phase C T3 = Rust daemon owns; Phase A+B T1/T2 = Python
    /// `DimFiringTracker._publish_block_shm` owns. The Python tracker's
    /// `_l0_rust_enabled()` gate ensures only one process tries to
    /// create/write per slot.
    fn ensure_slot(&mut self) {
        if self.slot.is_some() {
            return;
        }
        if let Ok(slot) = Slot::open(&self.slot_path) {
            self.slot = Some(slot);
            return;
        }
        // Slot file doesn't exist — Rust daemon creates it.
        if let Ok(slot) = Slot::create(&self.slot_path, self.schema_version, self.max_bytes) {
            self.slot = Some(slot);
        }
        // Create failed — possibly a race with another writer that just
        // created it. Next tick will retry the open path.
    }

    /// Record one tick. Encodes the firing payload + writes via SeqLock.
    /// Non-fatal on failure: increments `write_failures`, throttled WARN.
    ///
    /// `dims` is the canonical block tensor written this tick (length
    /// must equal the block's dim count: 5 / 15 / 45).
    ///
    /// `inputs_state` is the per-input classification — `(name,
    /// "real" | "default" | "absent")`. Matches the Python
    /// `_classify_input` heuristic. Pass `&[]` if the daemon doesn't
    /// yet derive input classifications (PARTIAL vs ALIVE_AT_DEFAULT
    /// distinction lost, but ALIVE/SILENT classification preserved).
    ///
    /// `ts` is the wall-clock at tick boundary (seconds since UNIX
    /// epoch, f64). Pass `now_secs()` from the caller's clock.
    pub fn record_tick(&mut self, dims: &[f32], inputs_state: &[(&str, &str)], ts: f64) {
        self.block_calls_total = self.block_calls_total.saturating_add(1);
        self.last_call_ts = ts;

        self.ensure_slot();
        let slot = match self.slot.as_mut() {
            Some(s) => s,
            None => return, // open still pending — silent skip; counter advances
        };

        let payload = encode_firing_payload(
            self.block_name,
            self.block_calls_total,
            self.last_call_ts,
            inputs_state,
            dims,
            ts,
        );

        if let Err(e) = slot.write(&payload) {
            self.write_failures = self.write_failures.saturating_add(1);
            if self.write_failures <= 5 || self.write_failures.is_multiple_of(WARN_THROTTLE_EVERY) {
                warn!(
                    block = self.block_name,
                    failures = self.write_failures,
                    err = %e,
                    "firing slot write failed (non-fatal)",
                );
            }
        }
    }

    /// Diagnostic — total successful + attempted ticks recorded.
    pub fn calls_total(&self) -> u64 {
        self.block_calls_total
    }

    /// Diagnostic — number of write failures since daemon boot.
    pub fn write_failures(&self) -> u64 {
        self.write_failures
    }
}

/// Encode the firing payload as msgpack — 1:1 byte-compatible with the
/// Python `_block_payload_locked` writer in
/// `titan_hcl/api/dim_registry.py:444`.
///
/// Public for parity-test reuse.
pub fn encode_firing_payload(
    block: &str,
    block_calls_total: u64,
    block_last_call_ts: f64,
    inputs_state: &[(&str, &str)],
    dims: &[f32],
    ts: f64,
) -> Vec<u8> {
    let inputs_map = Value::Map(
        inputs_state
            .iter()
            .map(|(name, state)| {
                (
                    Value::String((*name).into()),
                    Value::String((*state).into()),
                )
            })
            .collect(),
    );

    let dims_array = Value::Array(
        dims.iter()
            .map(|v| {
                Value::Map(vec![
                    (Value::String("v".into()), Value::F64(*v as f64)),
                    (Value::String("ts".into()), Value::F64(ts)),
                ])
            })
            .collect(),
    );

    let payload = Value::Map(vec![
        (Value::String("block".into()), Value::String(block.into())),
        (
            Value::String("block_calls_total".into()),
            Value::Integer(block_calls_total.into()),
        ),
        (
            Value::String("block_last_call_ts".into()),
            Value::F64(block_last_call_ts),
        ),
        (Value::String("inputs_state".into()), inputs_map),
        (Value::String("dims".into()), dims_array),
        (Value::String("ts".into()), Value::F64(ts)),
    ]);

    let mut out = Vec::with_capacity(256 + dims.len() * 32);
    rmpv::encode::write_value(&mut out, &payload)
        .expect("rmpv encode never fails on well-formed Value");
    out
}

/// Wall-clock seconds since UNIX epoch as f64 (helper for daemons that
/// don't already compute this for their canonical slot write).
pub fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn decode_map(bytes: &[u8]) -> Vec<(Value, Value)> {
        let v = rmpv::decode::read_value(&mut std::io::Cursor::new(bytes)).unwrap();
        match v {
            Value::Map(items) => items,
            _ => panic!("expected map"),
        }
    }

    fn lookup<'a>(map: &'a [(Value, Value)], key: &str) -> &'a Value {
        for (k, v) in map.iter() {
            if let Value::String(s) = k {
                if s.as_str() == Some(key) {
                    return v;
                }
            }
        }
        panic!("key {key} not found");
    }

    #[test]
    fn encode_payload_top_level_keys_present() {
        let bytes = encode_firing_payload(
            "inner_body",
            42,
            1234.5,
            &[("body_state", "real")],
            &[0.1, 0.2, 0.3, 0.4, 0.5],
            1235.0,
        );
        let map = decode_map(&bytes);
        // All 6 top-level keys present
        for key in [
            "block",
            "block_calls_total",
            "block_last_call_ts",
            "inputs_state",
            "dims",
            "ts",
        ] {
            let _ = lookup(&map, key);
        }
    }

    #[test]
    fn encode_payload_block_field_round_trips() {
        let bytes = encode_firing_payload("outer_spirit", 0, 0.0, &[], &[], 0.0);
        let map = decode_map(&bytes);
        let block = lookup(&map, "block");
        assert_eq!(block.as_str(), Some("outer_spirit"));
    }

    #[test]
    fn encode_payload_block_calls_total_round_trips_u64() {
        let bytes = encode_firing_payload("inner_mind", 1_234_567_890, 0.0, &[], &[], 0.0);
        let map = decode_map(&bytes);
        let v = lookup(&map, "block_calls_total");
        assert_eq!(v.as_u64(), Some(1_234_567_890));
    }

    #[test]
    fn encode_payload_dims_array_length_matches() {
        let bytes = encode_firing_payload("inner_spirit", 1, 0.0, &[], &[0.0_f32; 45], 0.0);
        let map = decode_map(&bytes);
        let dims = lookup(&map, "dims");
        match dims {
            Value::Array(items) => assert_eq!(items.len(), 45),
            _ => panic!("dims not array"),
        }
    }

    #[test]
    fn encode_payload_dim_entries_have_v_and_ts() {
        let bytes =
            encode_firing_payload("inner_body", 1, 42.0, &[], &[0.7, 0.8, 0.9, 1.0, 1.1], 42.0);
        let map = decode_map(&bytes);
        let dims = lookup(&map, "dims");
        let items = match dims {
            Value::Array(items) => items,
            _ => panic!(),
        };
        assert_eq!(items.len(), 5);
        for (i, entry) in items.iter().enumerate() {
            let entry_map = match entry {
                Value::Map(m) => m,
                _ => panic!("dim entry not map"),
            };
            let v = lookup(entry_map, "v").as_f64().unwrap();
            let ts = lookup(entry_map, "ts").as_f64().unwrap();
            // Float32→f64 lossless for these values
            let expected = [0.7_f64, 0.8, 0.9, 1.0, 1.1][i];
            assert!((v - expected as f32 as f64).abs() < 1e-6);
            assert!((ts - 42.0).abs() < 1e-9);
        }
    }

    #[test]
    fn encode_payload_inputs_state_round_trips() {
        let bytes = encode_firing_payload(
            "outer_body",
            1,
            0.0,
            &[
                ("agency_stats", "real"),
                ("anchor_state", "absent"),
                ("hormone_levels", "default"),
            ],
            &[0.0; 5],
            0.0,
        );
        let map = decode_map(&bytes);
        let inputs = lookup(&map, "inputs_state");
        let items = match inputs {
            Value::Map(m) => m,
            _ => panic!("inputs_state not map"),
        };
        assert_eq!(items.len(), 3);
        // Verify each (key, value) pair
        let expected: &[(&str, &str)] = &[
            ("agency_stats", "real"),
            ("anchor_state", "absent"),
            ("hormone_levels", "default"),
        ];
        for (k, v) in items {
            let key = k.as_str().unwrap();
            let value = v.as_str().unwrap();
            let exp_value = expected
                .iter()
                .find(|(ek, _)| *ek == key)
                .map(|(_, ev)| *ev)
                .unwrap_or_else(|| panic!("unexpected key {key}"));
            assert_eq!(value, exp_value);
        }
    }

    #[test]
    fn writer_creates_slot_when_missing_then_records_tick() {
        let dir = tempdir().unwrap();
        // Don't pre-create the slot file — writer must create it on first tick.
        let mut w = FiringSlotWriter::new("inner_body", dir.path(), 1, 1024);
        w.record_tick(&[0.1, 0.2, 0.3, 0.4, 0.5], &[], 1.0);
        // Counter advances + slot file now exists.
        assert_eq!(w.calls_total(), 1);
        assert_eq!(w.write_failures(), 0);
        assert!(dir.path().join("inner_body_firing.bin").exists());

        // Re-open from outside the writer + verify payload decodes.
        let reopened = Slot::open(dir.path().join("inner_body_firing.bin")).unwrap();
        let payload = reopened.read().unwrap();
        let map = decode_map(&payload);
        assert_eq!(lookup(&map, "block").as_str(), Some("inner_body"));
        assert_eq!(lookup(&map, "block_calls_total").as_u64(), Some(1));
    }

    #[test]
    fn writer_records_tick_when_slot_pre_existing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("inner_body_firing.bin");
        // Pre-create slot with capacity matching INNER_BODY_FIRING_MAX_BYTES (1024).
        let _slot = Slot::create(&path, 1, 1024).unwrap();
        drop(_slot); // close so writer can re-open

        let mut w = FiringSlotWriter::new("inner_body", dir.path(), 1, 1024);
        w.record_tick(&[0.1, 0.2, 0.3, 0.4, 0.5], &[("body_state", "real")], 1.0);
        assert_eq!(w.calls_total(), 1);
        assert_eq!(w.write_failures(), 0);

        // Re-open the slot from outside the writer + verify payload decodes.
        let reopened = Slot::open(&path).unwrap();
        let payload = reopened.read().unwrap();
        let map = decode_map(&payload);
        assert_eq!(lookup(&map, "block").as_str(), Some("inner_body"));
        assert_eq!(lookup(&map, "block_calls_total").as_u64(), Some(1));
    }

    #[test]
    fn writer_calls_total_increments_per_tick() {
        let dir = tempdir().unwrap();
        let mut w = FiringSlotWriter::new("outer_mind", dir.path(), 1, 2048);
        for i in 1..=10 {
            w.record_tick(&[0.0_f32; 15], &[], i as f64);
            assert_eq!(w.calls_total(), i as u64);
        }
    }

    #[test]
    fn now_secs_returns_positive_value() {
        let t = now_secs();
        assert!(t > 1_700_000_000.0); // sanity: post-2023 wall clock
    }
}
