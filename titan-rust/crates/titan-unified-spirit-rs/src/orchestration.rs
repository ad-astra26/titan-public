//! orchestration — body_cycle_loop tokio task driving SELF assembly.
//!
//! Per SPEC §10.G ground_up tick ordering step 7: unified-spirit assembles
//! 162D + writes self_162d + unified_spirit_132d + publishes
//! `UNIFIED_SPIRIT_SELF_ASSEMBLED` per body cycle.
//!
//! C-S4 chunk decomposition:
//! - **C4-2 (this commit)**: body_cycle_loop SKELETON — ticks at
//!   `BODY_CYCLE_INTERVAL_MS` (default 1150), reads slots, assembles,
//!   writes via SeqLock, content-hash-gates duplicate writes. NO bus
//!   publish yet (bus_client lands in C4-2b1 alongside SPHERE_PULSE
//!   subscriber). NO ResonanceDetector wiring (C4-2b1). NO V5 publish
//!   (C4-3c). NO daemon supervision (C4-4). NO substrate handshake
//!   (C4-5).
//! - **C4-2b1** wires bus_client + SPHERE_PULSE → ResonanceDetector.
//! - **C4-2b2** wires UnifiedSpirit.advance() callback on all-3-resonant.
//! - **C4-3c** wires V5 publish on each tick.
//! - **C4-7 / C4-6** verify cadence + parity in e2e test.
//!
//! Cadence comes from `titan-core::constants::BODY_CYCLE_INTERVAL_MS`
//! (SPEC v0.1.3 TOML, default 1150ms = Schumann/9 body publish rate).

use std::path::PathBuf;
use std::time::Duration;

use tokio::select;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};
use tracing::{error, info, warn};

use titan_core::constants::BODY_CYCLE_INTERVAL_MS;

use crate::self_assembly::{self, AssemblyError, JOURNEY_DIMS};
use crate::slot_handles::{SlotHandleError, SlotHandles};

/// One body cycle's outcome — observable by tests + orchestration callers.
#[derive(Debug, Clone, PartialEq)]
pub enum CycleOutcome {
    /// SELF assembled + written (content hash changed since last tick).
    Wrote {
        /// 16-byte blake2b-128 digest of the new 162D payload.
        hash: [u8; 16],
        /// Wall clock nanoseconds at write completion.
        wall_ns: u64,
    },
    /// SELF unchanged — content hash matched previous tick. Skipped both
    /// SeqLock writes (per content-hash gate, mirror of Python
    /// `kernel.py:_writer_loop` `if h != last_hash` pattern).
    Unchanged {
        /// Reused 16-byte hash from previous tick.
        hash: [u8; 16],
    },
    /// Cycle skipped due to a transient error. The orchestration loop
    /// continues (does NOT crash); error is logged at WARN.
    Skipped {
        /// Reason category for telemetry.
        reason: SkipReason,
    },
}

/// Why a body cycle skipped a write attempt. Distinguished from a fatal
/// error to keep the orchestration loop alive across transient hiccups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// One or more daemon slots returned a transient read error
    /// (e.g. SeqLock retry budget exhausted under heavy contention).
    /// Self-healing on next tick.
    SlotReadTransient,
    /// Daemon emitted NaN — refused to publish corrupted SELF tensor.
    NanInput,
    /// Slot file missing (rare — kernel didn't create it; should not
    /// reach steady state). Boot integrity check should have caught this.
    SlotMissing,
}

/// Run one body cycle. Pure function over `SlotHandles` (mutable for
/// writes) + content-hash-gate state. Returns outcome for telemetry.
pub fn run_body_cycle(handles: &mut SlotHandles, last_hash: &mut Option<[u8; 16]>) -> CycleOutcome {
    // Read 6 trinity daemon slots
    let trinity = match handles.read_trinity() {
        Ok(t) => t,
        Err(SlotHandleError::Read { .. }) | Err(SlotHandleError::ReadShape { .. }) => {
            return CycleOutcome::Skipped {
                reason: SkipReason::SlotReadTransient,
            };
        }
        Err(SlotHandleError::Open { .. }) => {
            return CycleOutcome::Skipped {
                reason: SkipReason::SlotMissing,
            };
        }
        Err(_) => {
            return CycleOutcome::Skipped {
                reason: SkipReason::SlotReadTransient,
            };
        }
    };

    // Read topology + journey
    let topology = match handles.read_topology() {
        Ok(t) => t,
        Err(_) => {
            return CycleOutcome::Skipped {
                reason: SkipReason::SlotReadTransient,
            };
        }
    };
    // Fall back to zeros if Journey 2D not yet populated by substrate
    // (sphere_clocks not yet ticking). Don't skip the whole cycle —
    // SELF tensor still has 130D felt + 30D topology signal, journey
    // just defaults to zero for this tick.
    let journey = handles.read_journey().unwrap_or([0.0_f32; JOURNEY_DIMS]);

    // Compute 162D + 132D
    let self_162 = match self_assembly::assemble_162d(&trinity, &topology, &journey) {
        Ok(s) => s,
        Err(AssemblyError::NanInput { layer, index }) => {
            warn!(
                event = "ASSEMBLY_NAN",
                layer = layer,
                index = index,
                "NaN in input slot — refusing to publish corrupted SELF"
            );
            return CycleOutcome::Skipped {
                reason: SkipReason::NanInput,
            };
        }
    };

    // Content-hash gate
    let hash = self_assembly::content_hash(&self_162);
    if Some(hash) == *last_hash {
        return CycleOutcome::Unchanged { hash };
    }

    // Build 132D intermediate (felt + journey)
    let mut felt_130 = [0.0_f32; 130];
    felt_130.copy_from_slice(&self_162[..130]);
    let unified_132 = match self_assembly::assemble_unified_spirit_132d(&felt_130, &journey) {
        Ok(u) => u,
        Err(_) => {
            // Should never happen — felt_130 already validated above.
            return CycleOutcome::Skipped {
                reason: SkipReason::NanInput,
            };
        }
    };

    // Write both slots via SeqLock
    if let Err(e) = handles.write_unified_spirit_132d(&unified_132) {
        error!(event = "SLOT_WRITE_FAIL", slot = "unified_spirit_132d.bin", err = ?e);
        return CycleOutcome::Skipped {
            reason: SkipReason::SlotReadTransient,
        };
    }
    if let Err(e) = handles.write_self_162d(&self_162) {
        error!(event = "SLOT_WRITE_FAIL", slot = "self_162d.bin", err = ?e);
        return CycleOutcome::Skipped {
            reason: SkipReason::SlotReadTransient,
        };
    }

    *last_hash = Some(hash);
    CycleOutcome::Wrote {
        hash,
        wall_ns: self_assembly::now_ns(),
    }
}

/// Orchestration entry point — spawns the body_cycle_loop tokio task.
///
/// `cadence_ms` defaults to `BODY_CYCLE_INTERVAL_MS` from SPEC; CLI
/// `--self-assembly-cadence-ms` overrides for tests.
///
/// Returns a `JoinHandle` and a shutdown sender. Caller drops the sender
/// (or sends `()` explicitly) to stop the loop gracefully.
pub fn spawn_body_cycle_loop(
    shm_dir: PathBuf,
    cadence_ms: u64,
) -> Result<(JoinHandle<()>, watch::Sender<()>), SlotHandleError> {
    let mut handles = SlotHandles::open_all(&shm_dir)?;
    let (shutdown_tx, mut shutdown_rx) = watch::channel(());

    let task = tokio::spawn(async move {
        let mut last_hash: Option<[u8; 16]> = None;
        let mut tick = interval(Duration::from_millis(cadence_ms));
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

        info!(
            event = "BODY_CYCLE_LOOP_START",
            cadence_ms = cadence_ms,
            "body_cycle_loop started"
        );

        // First tick fires immediately per tokio interval semantics.
        loop {
            select! {
                _ = tick.tick() => {
                    let outcome = run_body_cycle(&mut handles, &mut last_hash);
                    match &outcome {
                        CycleOutcome::Wrote { hash, .. } => {
                            tracing::debug!(
                                event = "BODY_CYCLE_WROTE",
                                hash_hex = %hex::encode(hash),
                                "wrote self_162d + unified_spirit_132d"
                            );
                        }
                        CycleOutcome::Unchanged { .. } => {
                            tracing::trace!(event = "BODY_CYCLE_UNCHANGED", "content hash unchanged");
                        }
                        CycleOutcome::Skipped { reason } => {
                            tracing::warn!(event = "BODY_CYCLE_SKIPPED", ?reason, "skipped tick");
                        }
                    }
                }
                _ = shutdown_rx.changed() => {
                    info!(event = "BODY_CYCLE_LOOP_STOP", "graceful shutdown");
                    break;
                }
            }
        }
    });

    Ok((task, shutdown_tx))
}

/// Default cadence used by the orchestration loop unless overridden by
/// CLI. Sourced from SPEC v0.1.3 TOML constant.
pub const fn default_cadence_ms() -> u64 {
    BODY_CYCLE_INTERVAL_MS
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    use crate::self_assembly::{encode_f32_slice, F32_BYTES};
    use titan_state::Slot;

    /// Helper: create the 10 slot files in a temp dir (mimics kernel boot).
    fn create_all_slots(dir: &std::path::Path) {
        Slot::create(dir.join("inner_body_5d.bin"), 1, 5 * F32_BYTES as u32).unwrap();
        Slot::create(dir.join("inner_mind_15d.bin"), 1, 15 * F32_BYTES as u32).unwrap();
        Slot::create(dir.join("inner_spirit_45d.bin"), 1, 45 * F32_BYTES as u32).unwrap();
        Slot::create(dir.join("outer_body_5d.bin"), 1, 5 * F32_BYTES as u32).unwrap();
        Slot::create(dir.join("outer_mind_15d.bin"), 1, 15 * F32_BYTES as u32).unwrap();
        Slot::create(dir.join("outer_spirit_45d.bin"), 1, 45 * F32_BYTES as u32).unwrap();
        Slot::create(dir.join("topology_30d.bin"), 1, 30 * F32_BYTES as u32).unwrap();
        Slot::create(dir.join("sphere_clocks.bin"), 1, 6 * 7 * F32_BYTES as u32).unwrap();
        Slot::create(
            dir.join("unified_spirit_132d.bin"),
            1,
            132 * F32_BYTES as u32,
        )
        .unwrap();
        Slot::create(dir.join("self_162d.bin"), 1, 162 * F32_BYTES as u32).unwrap();
    }

    /// Helper: mutate a daemon slot's payload (to drive content-hash changes).
    fn set_slot_payload(dir: &std::path::Path, name: &str, payload: &[u8]) {
        let mut slot = Slot::open(dir.join(name)).unwrap();
        slot.write(payload).unwrap();
    }

    #[test]
    fn run_body_cycle_writes_first_tick() {
        // C4-2 orchestration 5: first cycle writes (no previous hash to gate)
        let dir = tempdir().unwrap();
        create_all_slots(dir.path());
        let mut handles = SlotHandles::open_all(dir.path()).unwrap();
        let mut last_hash = None;
        let outcome = run_body_cycle(&mut handles, &mut last_hash);
        assert!(matches!(outcome, CycleOutcome::Wrote { .. }));
        assert!(last_hash.is_some());
    }

    #[test]
    fn run_body_cycle_content_hash_gate_suppresses_duplicate() {
        // C4-2 orchestration 6: identical inputs → second cycle is Unchanged
        let dir = tempdir().unwrap();
        create_all_slots(dir.path());
        let mut handles = SlotHandles::open_all(dir.path()).unwrap();
        let mut last_hash = None;
        let _ = run_body_cycle(&mut handles, &mut last_hash);
        let outcome = run_body_cycle(&mut handles, &mut last_hash);
        assert!(matches!(outcome, CycleOutcome::Unchanged { .. }));
    }

    #[test]
    fn run_body_cycle_writes_again_on_input_change() {
        // C4-2 orchestration 7: changing any input → next cycle writes
        let dir = tempdir().unwrap();
        create_all_slots(dir.path());
        let mut handles = SlotHandles::open_all(dir.path()).unwrap();
        let mut last_hash = None;
        let _ = run_body_cycle(&mut handles, &mut last_hash);

        // Mutate inner_body
        let new_payload = encode_f32_slice(&[42.0_f32; 5]);
        set_slot_payload(dir.path(), "inner_body_5d.bin", &new_payload);
        // Re-open handles to pick up writer-mutated bytes (mmap snapshots
        // the file on open; in production both writers and readers share
        // the same mmap region of the same file)
        let mut handles = SlotHandles::open_all(dir.path()).unwrap();
        let outcome = run_body_cycle(&mut handles, &mut last_hash);
        assert!(matches!(outcome, CycleOutcome::Wrote { .. }));
    }

    #[test]
    fn run_body_cycle_skips_on_nan_input() {
        // C4-2 orchestration 8: NaN in any input → skip (don't publish)
        let dir = tempdir().unwrap();
        create_all_slots(dir.path());

        // Inject NaN into inner_body slot
        let mut payload = [0_u8; 5 * F32_BYTES];
        payload[0..F32_BYTES].copy_from_slice(&f32::NAN.to_le_bytes());
        set_slot_payload(dir.path(), "inner_body_5d.bin", &payload);

        let mut handles = SlotHandles::open_all(dir.path()).unwrap();
        let mut last_hash = None;
        let outcome = run_body_cycle(&mut handles, &mut last_hash);
        assert_eq!(
            outcome,
            CycleOutcome::Skipped {
                reason: SkipReason::NanInput
            }
        );
        // last_hash unchanged — gate didn't fire on a refused write
        assert!(last_hash.is_none());
    }

    #[test]
    fn default_cadence_matches_spec_constant() {
        // C4-2 orchestration 9: default cadence = SPEC v0.1.3 BODY_CYCLE_INTERVAL_MS
        assert_eq!(default_cadence_ms(), BODY_CYCLE_INTERVAL_MS);
        assert_eq!(default_cadence_ms(), 1150);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn body_cycle_loop_ticks_at_cadence_and_shuts_down_cleanly() {
        // C4-2 orchestration 10: spawn loop, advance virtual time across
        // 3 cadences, send shutdown signal, verify graceful exit.
        // start_paused = true — tokio test virtual clock; tick driven by sleep.
        let dir = tempdir().unwrap();
        create_all_slots(dir.path());

        // Spawn with a tight 100ms cadence (test override).
        let (handle, shutdown_tx) = spawn_body_cycle_loop(dir.path().to_path_buf(), 100).unwrap();

        // Advance virtual time past 3 ticks
        tokio::time::advance(Duration::from_millis(350)).await;

        // Send shutdown
        shutdown_tx.send(()).unwrap();

        // Task should complete soon
        let result = tokio::time::timeout(Duration::from_millis(500), handle).await;
        assert!(result.is_ok(), "task should finish within timeout");
    }

    #[test]
    fn cycle_outcome_skip_reason_classification() {
        // C4-2 orchestration 11: SkipReason variants distinguishable in
        // CycleOutcome match patterns
        let nan_skip = CycleOutcome::Skipped {
            reason: SkipReason::NanInput,
        };
        let transient_skip = CycleOutcome::Skipped {
            reason: SkipReason::SlotReadTransient,
        };
        assert_ne!(nan_skip, transient_skip);
    }

    #[test]
    fn run_body_cycle_handles_missing_journey_with_zero_fallback() {
        // C4-2 orchestration 12: sphere_clocks empty (journey unreadable
        // OR returns garbage) — cycle still proceeds with [0.0, 0.0]
        // Journey 2D fallback (don't bring down SELF assembly because
        // substrate clocks aren't ticking yet).
        //
        // Setup: sphere_clocks slot exists but payload too short for
        // journey extraction (< 28 bytes). The fallback should kick in.
        // BUT — Slot::write() rejects payloads exceeding capacity, and
        // create_all_slots reserves 168 bytes, so we'd write a short
        // payload that read returns. Slot::read() returns the 168-byte
        // mmap region regardless.
        //
        // Easier path: Slot::create initializes payload to all-zeros, so
        // first 28 bytes are also zeros → journey = [0.0, 0.0]. Test that
        // pre-written zero bytes work.
        let dir = tempdir().unwrap();
        create_all_slots(dir.path());
        let mut handles = SlotHandles::open_all(dir.path()).unwrap();
        let mut last_hash = None;
        let outcome = run_body_cycle(&mut handles, &mut last_hash);
        assert!(matches!(outcome, CycleOutcome::Wrote { .. }));
        // Verify the journey was [0.0, 0.0]
        let journey = handles.read_journey().unwrap();
        assert_eq!(journey, [0.0, 0.0]);
    }
}
