"""
_phase_c_constants.py — AUTO-GENERATED from titan-docs/SPEC_titan_architecture_constants.toml.

DO NOT EDIT BY HAND. Edit the TOML, then run:
    python scripts/generate_phase_c_constants.py

SPEC version: 1.2.0
Source SHA-256: dfb308daa1f0d8794d2bd7f1eac0297b638adbdd96046af6594c1687cc8c505d

Per SPEC §19 + §2.6: hand-editing this file is a SPEC violation flagged by
`arch_map phase-c verify`.
"""
from __future__ import annotations
from typing import Final

# ── SPEC version metadata ──────────────────────────────────────────────
SPEC_VERSION: Final[str] = "1.2.0"
SPEC_SOURCE_SHA256: Final[str] = "dfb308daa1f0d8794d2bd7f1eac0297b638adbdd96046af6594c1687cc8c505d"


# ── KERNEL ────────────────────────────────────────────────────────────────
# titan-kernel-rs internals (boot, snapshot, signal handling, persistence)

# Total budget for kernel boot to steady state
KERNEL_BOOT_TIMEOUT_S: Final[float] = 30.0
# Kernel SIGTERM→SIGKILL grace window
KERNEL_SHUTDOWN_GRACE_S: Final[float] = 10.0
# L0 persistence atomic-snapshot cadence
KERNEL_SNAPSHOT_INTERVAL_S: Final[float] = 1.0
# Circadian clock tick cadence (1 Hz logical) per SPEC §10.H
KERNEL_CIRCADIAN_TICK_INTERVAL_S: Final[float] = 1.0
# Circadian clock full-cycle period (24h)
KERNEL_CIRCADIAN_PERIOD_S: Final[float] = 86400.0
# Pi-heartbeat tick cadence (~3 Hz target) — drives KERNEL_EPOCH_TICK publish per SPEC §10.H + §8.1
KERNEL_PI_HEARTBEAT_INTERVAL_S: Final[float] = 0.333333333


# ── TRINITY_SUBSTRATE ─────────────────────────────────────────────────────
# titan-trinity-rs internals (Schumann generators, fast bus, topology)

# Single-linkage clustering distance threshold for body-part observable vectors
TOPOLOGY_CLUSTER_THRESHOLD: Final[float] = 0.3
# Rolling window length for topology curvature computation (delta-volume across history)
TOPOLOGY_VOLUME_HISTORY_SIZE: Final[int] = 20
# topology_30d.bin payload size: 30 × float32 LE = 120 bytes (header excluded)
TOPOLOGY_30D_PAYLOAD_BYTES: Final[int] = 120
# Conservative grounding force magnitude — daemons multiply nudge by this per body cycle (G5 + G10 invariants)
GROUND_UP_DEFAULT_STRENGTH: Final[float] = 0.1
# Damping factor preventing nudge oscillation overshoot (smoothing of prev_nudge with raw signal)
GROUND_UP_DEFAULT_DAMPING: Final[float] = 0.95
# Per-tick safety clamp on absolute nudge magnitude per dimension — prevents runaway grounding
GROUND_UP_MAX_NUDGE: Final[float] = 0.05
# Number of spirit Schumann ticks per substrate body cycle (= SCHUMANN_SPIRIT_HZ / SCHUMANN_BODY_HZ ratio)
SUBSTRATE_BODY_CYCLE_SCHUMANN_TICKS: Final[int] = 9
# Substrate body cycle period (1.0 / SCHUMANN_BODY_HZ × 9 / 9 = 1/7.83 × 9 ≈ 1.149425287 s; exposes derived value for telemetry)
SUBSTRATE_BODY_CYCLE_S: Final[float] = 1.149425287
# chi_state.bin field count (total, spirit, mind, body, coherence, urgency)
CHI_STATE_FIELD_COUNT: Final[int] = 6
# chi_state.bin payload size: 6 × float32 LE = 24 bytes (header excluded)
CHI_STATE_PAYLOAD_BYTES: Final[int] = 24


# ── UNIFIED_SPIRIT ────────────────────────────────────────────────────────
# titan-unified-spirit-rs internals (162D SELF assembly, filter_down origination)

# Lower clamp for FILTER_DOWN V5 multipliers applied by daemons (Preamble G7 LOCKED)
FILTER_DOWN_MULTIPLIER_FLOOR: Final[float] = 0.3
# Upper clamp for FILTER_DOWN V5 multipliers (Preamble G7 LOCKED)
FILTER_DOWN_MULTIPLIER_CEIL: Final[float] = 3.0
# Gentle-filter multiplier applied to spirit content multipliers — 'Spirit modulates slowly' (Preamble G9)
FILTER_DOWN_SPIRIT_STRENGTH_MULT: Final[float] = 0.3
# Until reached, V5 publishes all-1.0 multipliers (no modulation); network needs ~2000 epochs of TD(0) training before producing meaningful gradients
FILTER_DOWN_COLD_START_FLOOR_EPOCHS: Final[int] = 2000
# Lower clamp for filter_down multipliers (UNIFIED_SPIRIT_FILTER_DOWN + INNER_SPIRIT_FILTER_DOWN + OUTER_SPIRIT_FILTER_DOWN payloads); applied at consume site per G7 + filter_down.py:464-498
UNIFIED_SPIRIT_MULTIPLIER_FLOOR: Final[float] = 0.3
# Upper clamp for filter_down multipliers
UNIFIED_SPIRIT_MULTIPLIER_CEIL: Final[float] = 3.0
# TrinityValueNet input dim — 130D felt + 30D topology + 2D journey
FILTER_DOWN_INPUT_DIM: Final[int] = 162
# TrinityValueNet hidden layer 1 width
FILTER_DOWN_HIDDEN_1: Final[int] = 128
# TrinityValueNet hidden layer 2 width
FILTER_DOWN_HIDDEN_2: Final[int] = 64
# Multiplier output dim — 5+15+40+5+15+40 (observer 10 dims masked per G8)
FILTER_DOWN_OUTPUT_DIM: Final[int] = 120
# TD(0) learning rate
FILTER_DOWN_LR: Final[float] = 0.001
# TD(0) discount factor — target = r + GAMMA × V(s')
FILTER_DOWN_GAMMA: Final[float] = 0.95
# TD(0) mini-batch size
FILTER_DOWN_BATCH_SIZE: Final[int] = 16
# TransitionBuffer ring capacity
FILTER_DOWN_BUFFER_MAX: Final[int] = 2000
# Minimum transitions buffered before training begins
FILTER_DOWN_MIN_TRANSITIONS: Final[int] = 32
# Train every N new transitions
FILTER_DOWN_TRAIN_EVERY_N: Final[int] = 5
# Rolling-window size (epochs) for SPIRIT velocity computation
UNIFIED_SPIRIT_VELOCITY_WINDOW: Final[int] = 10
# Velocity below this = SPIRIT IS_STALE (not growing enough)
UNIFIED_SPIRIT_STALE_THRESHOLD: Final[float] = 0.8
# Base FOCUS cascade multiplier when SPIRIT IS_STALE; escalates by 0.2 × consecutive_stale, capped at 3.0
UNIFIED_SPIRIT_STALE_FOCUS_MULTIPLIER: Final[float] = 1.5
# Max GreatEpoch records held in-memory before rotation to archive file (Rust port — Python had no cap)
UNIFIED_SPIRIT_EPOCHS_HISTORY_CAP: Final[int] = 4096
# Max phase difference for resonance (π/6 = 30°) — Proof of Harmony
RESONANCE_PHASE_THRESHOLD_RAD: Final[float] = 0.5235987755982988
# Consecutive resonant cycles required for BIG PULSE per pair
RESONANCE_CYCLES_REQUIRED: Final[int] = 3
# Max time between counterpart sphere pulses for resonance candidacy
RESONANCE_PULSE_WINDOW_S: Final[float] = 120.0
# Body publish rate ≈ Schumann/9 — unified-spirit body_cycle_loop tick interval
BODY_CYCLE_INTERVAL_MS: Final[int] = 1150
# Min interval between consecutive body_cycle_loop ticks (debounce TRINITY_SUBSTRATE_TOPOLOGY_UPDATED early-wakes)
BODY_CYCLE_DEBOUNCE_MS: Final[int] = 200


# ── DAEMON ────────────────────────────────────────────────────────────────
# Shared trinity-daemon library + per-daemon overrides

# Trinity daemon SIGTERM→SIGKILL grace
DAEMON_SHUTDOWN_GRACE_S: Final[float] = 3.0


# ── SUPERVISION ───────────────────────────────────────────────────────────
# OTP-style supervision contract: max_restarts, intensity_window, backoff, dependency-aware respawn, escalation handshake

# Max restarts per intensity window before escalation handshake fires
SUPERVISION_MAX_RESTARTS: Final[int] = 5
# Rolling window for restart counting (OTP standard 60s, NOT today's Python 600s)
SUPERVISION_INTENSITY_WINDOW_S: Final[float] = 60.0
# Initial backoff before first restart attempt (OTP ladder 100→200→400→800ms)
SUPERVISION_RESTART_BACKOFF_INITIAL_MS: Final[int] = 100
# Backoff ceiling
SUPERVISION_RESTART_BACKOFF_MAX_S: Final[float] = 2.0
# ± jitter on restart backoff to prevent thundering herd
SUPERVISION_RESTART_JITTER_PCT: Final[int] = 25
# Stable uptime threshold to reset restart counter
SUPERVISION_SUSTAINED_UPTIME_RESET_S: Final[float] = 300.0
# Max wait for kernel ESCALATION_RESPONSE before defaulting to terminate
SUPERVISION_ESCALATION_TIMEOUT_S: Final[float] = 10.0
# How often to recheck blocked dependencies in respawn_blocked state
SUPERVISION_DEPENDENCY_RECHECK_INTERVAL_S: Final[float] = 10.0
# Time blocked-respawn waits before escalating to kernel for halt decision
SUPERVISION_DEPENDENCY_BLOCKED_TIMEOUT_S: Final[float] = 300.0
# HTTP/RPC probe timeout for external_service dependency check
SUPERVISION_DEPENDENCY_PROBE_TIMEOUT_S: Final[float] = 5.0
# Single supervision log file (kernel writes; arch_map reads)
SUPERVISION_LOG_PATH: Final[str] = "data/supervision.jsonl"
# Max size before rotation (100 MB)
SUPERVISION_LOG_MAX_BYTES: Final[int] = 104857600
# Archive files to keep (.1 ... .10)
SUPERVISION_LOG_ARCHIVE_COUNT: Final[int] = 10
# Grace period before EMPTY classification fires (module alive but unpopulated)
SUPERVISION_EMPTY_GRACE_S: Final[float] = 60.0


# ── BUS ───────────────────────────────────────────────────────────────────
# Main bus broker behavior (rings, ping, slow-consumer, accept rate, reconnect backoff)

# Bus broker ping interval
BUS_PING_INTERVAL_S: Final[float] = 5.0
# Bus broker drops connection after this much silence (3 missed pings)
BUS_PING_TIMEOUT_S: Final[float] = 15.0
# Per-subscriber bounded ring buffer size
BUS_RING_CAPACITY_SLOTS: Final[int] = 1024
# P0 priority lane reserve (never dropped)
BUS_P0_RESERVE_SLOTS: Final[int] = 64
# Token-bucket accept() rate limit
BUS_ACCEPT_RATE_LIMIT_PER_S: Final[int] = 50
# Drop-rate threshold to fire BUS_SLOW_CONSUMER (5%)
BUS_SLOW_CONSUMER_DROP_RATE_RATIO: Final[float] = 0.05
# Throttle slow-consumer warnings
BUS_SLOW_CONSUMER_WARN_INTERVAL_S: Final[float] = 60.0
# Batch-send flush timeout (50ms)
BUS_SEND_FLUSH_TIMEOUT_S: Final[float] = 0.05
# Client reconnect initial backoff
BUS_RECONNECT_BACKOFF_INITIAL_MS: Final[int] = 100
# Client reconnect backoff ceiling
BUS_RECONNECT_BACKOFF_MAX_S: Final[float] = 2.0
# Default FastAPI subprocess listen port
BUS_API_HTTP_PORT_DEFAULT: Final[int] = 7777


# ── FASTBUS ───────────────────────────────────────────────────────────────
# Kernel↔Substrate lock-free shm ring buffer

# Lock-free SPSC shm ring buffer slot count
FASTBUS_RING_CAPACITY_SLOTS: Final[int] = 1024
# Per-slot byte size in fastbus ring
FASTBUS_SLOT_BYTES: Final[int] = 256
# Fast bus ring header layout (v0.1.4): magic[8] + read_idx[8] + write_idx[8] + version[4] + mask[4] + reserved[32] = 64 bytes. AtomicU64 fields at 8-byte aligned offsets (read_idx@8, write_idx@16) for portable lock-free atomics.
FASTBUS_HEADER_BYTES: Final[int] = 64
# Ring header version field — bumped on layout changes; substrate refuses to attach if version > current
FASTBUS_RING_VERSION: Final[int] = 1
# 8-byte magic identifier at offset 0 of fastbus ring header — substrate verifies on attach (8 ASCII bytes; UTF-8 byte length must equal 8)
FASTBUS_MAGIC_BYTES: Final[bytes] = b"TITANFB1"


# ── FRAME ─────────────────────────────────────────────────────────────────
# Wire-format primitives (length-prefix, msgpack, HMAC challenge)

# uint32 little-endian frame length prefix
FRAME_LENGTH_PREFIX_BYTES: Final[int] = 4
# Max frame size (16 MB hard ceiling)
FRAME_MAX_FRAME_BYTES: Final[int] = 16777216
# Server's random nonce per connection (handshake)
FRAME_CHALLENGE_BYTES: Final[int] = 32
# HMAC-SHA256 output size
FRAME_AUTH_TAG_BYTES: Final[int] = 32


# ── AUTHKEY ───────────────────────────────────────────────────────────────
# HKDF-derived bus + RPC authkey (one per Titan, env-passed)

# 256-bit HMAC key size (HKDF-derived from Ed25519 identity)
AUTHKEY_BYTES: Final[int] = 32
# Version-bumpable HKDF salt (interpreted as UTF-8 bytes)
AUTHKEY_HKDF_SALT: Final[bytes] = b"titan-bus-v1"
# HKDF info field — domain separation constant (NOT titan_id-derived; per-Titan isolation comes from per-Titan identity keypair → different IKM → different authkey). Constant prevents the runtime call-site drift class that broke Phase C C-S7 on 2026-05-05 (Rust kernel passed 'titan_T3' while Python worker passed 'T3' → different authkeys → 100% handshake failure). See titan-docs/rFP_phase_c_bus_authkey_contract_fix.md.
AUTHKEY_HKDF_INFO: Final[bytes] = b"titan-bus"


# ── SHM ───────────────────────────────────────────────────────────────────
# Shared-memory triple-buffer slot wire format (universal §7.0 header v1.0.0)

# Fixed slot header size (§7.0 v1.0.0): header_seq(8 atomic) + schema_version(4 constant) + payload_capacity(4 constant). Schema + capacity are set at Slot::create and never updated by writer.
SHM_HEADER_BYTES: Final[int] = 16
# Per-buffer metadata size (§7.0 v1.0.0): wall_ns(8) + payload_bytes(4) + buffer_crc32(4). One block per buffer — co-located with payload so the entire buffer state (metadata + payload + CRC) is published atomically by the header_seq Release-store.
SHM_BUFFER_META_BYTES: Final[int] = 16
# Triple-buffer count per slot — writer rotates 0→1→2, reader picks ready_idx; race-elimination requires N+1 buffers where N=max writer publishes during reader memcpy = 2
SHM_BUFFER_COUNT: Final[int] = 3
# Python struct format for §7.0 v1.0.0 fixed header (LE: u64 header_seq, u32 schema_version, u32 payload_capacity)
SHM_HEADER_STRUCT: Final[str] = "<QII"
# Python struct format for §7.0 v1.0.0 per-buffer metadata (LE: u64 wall_ns, u32 payload_bytes, u32 buffer_crc32)
SHM_BUFFER_META_STRUCT: Final[str] = "<QII"


# ── REGISTRY ──────────────────────────────────────────────────────────────
# Per-slot schema versions + registry lifecycle primitives

# unified_spirit_132d.bin slot schema version (Trinity 130D + Journey 2D intermediate before SELF assembly)
UNIFIED_SPIRIT_132D_SCHEMA_VERSION: Final[int] = 1
# Schema version for self_162d.bin slot (162D TITAN_SELF tensor)
SELF_162D_SCHEMA_VERSION: Final[int] = 1
# Schema version for inner_body_5d.bin slot
INNER_BODY_5D_SCHEMA_VERSION: Final[int] = 1
# Schema version for inner_mind_15d.bin slot
INNER_MIND_15D_SCHEMA_VERSION: Final[int] = 1
# Schema version for inner_spirit_45d.bin slot (Schumann × 9 = 70.47 Hz)
INNER_SPIRIT_45D_SCHEMA_VERSION: Final[int] = 1
# Schema version for outer_body_5d.bin slot
OUTER_BODY_5D_SCHEMA_VERSION: Final[int] = 1
# Schema version for outer_mind_15d.bin slot
OUTER_MIND_15D_SCHEMA_VERSION: Final[int] = 1
# Schema version for outer_spirit_45d.bin slot
OUTER_SPIRIT_45D_SCHEMA_VERSION: Final[int] = 1
# Schema version for topology_30d.bin slot ([0:10] outer_lower + [10:20] inner_lower + [20:30] whole)
TOPOLOGY_30D_SCHEMA_VERSION: Final[int] = 1
# Schema version for neuromod_state.bin slot (DA, 5HT, NE, ACh, Endorphin, GABA)
NEUROMOD_SCHEMA_VERSION: Final[int] = 1
# Number of neuromodulator fields persisted in neuromod_state.bin (DA, 5HT, NE, ACh, Endorphin, GABA)
NEUROMOD_FIELD_COUNT: Final[int] = 6
# Total payload bytes for neuromod_state.bin = NEUROMOD_FIELD_COUNT × 4 (f32 LE) = 6 × 4 = 24 bytes
NEUROMOD_PAYLOAD_BYTES: Final[int] = 24
# Schema version for epoch_counter.bin slot
EPOCH_COUNTER_SCHEMA_VERSION: Final[int] = 1
# Schema version for sphere_clocks.bin slot (6 × 7 fields)
SPHERE_CLOCKS_SCHEMA_VERSION: Final[int] = 1
# Schema version for chi_state.bin slot
CHI_STATE_SCHEMA_VERSION: Final[int] = 1
# Schema version for titanvm_registers.bin slot (11 NS programs × 4 fields)
TITANVM_REGISTERS_SCHEMA_VERSION: Final[int] = 1
# Schema version for hormonal_state.bin slot (11 hormones × 4 fields × float32 = 176 bytes payload). Canonical hormone order matches NS_PROGRAMS in emot_bundle_protocol.py (REFLEX, FOCUS, INTUITION, IMPULSE, METABOLISM, CREATIVITY, CURIOSITY, EMPATHY, REFLECTION, INSPIRATION, VIGILANCE). Per-hormone fields: level, threshold, refractory, peak_level (read-mostly state surfaced from HormonalPressure class in titan_plugin/logic/hormonal_pressure.py).
HORMONAL_STATE_SCHEMA_VERSION: Final[int] = 1
# Schema version for identity.bin slot (kernel identity + per-boot nonce)
IDENTITY_SCHEMA_VERSION: Final[int] = 1
# Schema version for cgn_live_weights.bin slot (variable-size, ≤256 KB)
CGN_LIVE_WEIGHTS_SCHEMA_VERSION: Final[int] = 1
# Schema version for circadian.bin slot
CIRCADIAN_SCHEMA_VERSION: Final[int] = 1
# Schema version for pi_heartbeat.bin slot
PI_HEARTBEAT_SCHEMA_VERSION: Final[int] = 1
# Schema version for fastbus.bin lock-free shm ring
FASTBUS_SCHEMA_VERSION: Final[int] = 1
# Schema version for hormone_fires.bin slot — variable msgpack {hormone_name → fire_count} for 8 canonical hormones + ts. Owned by spirit_worker (titan_HCL). Closes spirit_proxy.get_trinity sync-RPC deadlock (rFP §4.B.1).
HORMONE_FIRES_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for hormone_fires.bin (8 hormones × small int + ts → ~150B typical, 1024B cap).
HORMONE_FIRES_MAX_BYTES: Final[int] = 1024
# Schema version for impulse_engine_state.bin slot — variable msgpack {hormones: {name → {impulse_value, last_fire_ts, threshold}}, total_fires, last_observe_ts, ts}. Owned by spirit_worker (ImpulseEngine.get_stats output).
IMPULSE_ENGINE_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for impulse_engine_state.bin (~600B typical, 2048B cap).
IMPULSE_ENGINE_STATE_MAX_BYTES: Final[int] = 2048
# Schema version for consciousness_state.bin slot — variable msgpack {epoch_id, density, curvature, dream_quality, fatigue, trajectory_magnitude, latest_epoch, ts}. Extracts from consciousness['latest_epoch'] per _run_consciousness_epoch output. Owned by spirit_worker.
CONSCIOUSNESS_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for consciousness_state.bin (latest_epoch dict can include nested state vector ~2KB, 4096B cap).
CONSCIOUSNESS_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for assessment_state.bin slot — variable msgpack {average_score, total, recent[10], trend, score_variance, research_avg_score, ts}. Owned by Python assessment module. (Producer ships Session 2; slot declared now per rFP §4.A.1.)
ASSESSMENT_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for assessment_state.bin.
ASSESSMENT_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for agency_state.bin slot — variable msgpack {total_actions, actions_this_hour, success_rate, llm_calls_this_hour, helper_statuses, last_action_ts, posture_history_digest, ts}. Owned by agency_module. (Session 2.)
AGENCY_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for agency_state.bin (helper_statuses dict can be large).
AGENCY_STATE_MAX_BYTES: Final[int] = 8192
# Schema version for social_perception_state.bin slot — variable msgpack {sentiment_ema, interaction_rate, social_activity, last_interaction_ts, ts}. Owned by spirit_worker. (Session 2.)
SOCIAL_PERCEPTION_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for social_perception_state.bin.
SOCIAL_PERCEPTION_STATE_MAX_BYTES: Final[int] = 2048
# Schema version for rl_state.bin slot — variable msgpack {programs[], current_program_id, dream_quality, training_loss_ema, transitions, last_train_ts, ts}. Owned by rl_worker. (Session 2.)
RL_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for rl_state.bin.
RL_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for memory_state.bin slot — variable msgpack {persistent_count, mempool_size, learning_velocity, directive_alignment, effective_nodes_24h, high_quality_count, kg_node_count, kg_edge_count, topology_clusters_summary, ts}. Owned by memory_worker. (Session 2.)
MEMORY_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for memory_state.bin (topology_clusters_summary can include cluster digests).
MEMORY_STATE_MAX_BYTES: Final[int] = 8192
# Schema version for timechain_state.bin slot — variable msgpack {tx_latency_norm, block_delta_norm, recent_anchor_age_s, fork_summary[7], integrity_status, total_blocks, chi_spent_total, ts}. Owned by timechain_worker. (Session 2.)
TIMECHAIN_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for timechain_state.bin.
TIMECHAIN_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for reflex_state.bin slot — variable msgpack {reflex_name → {fire_count, total_updates, last_loss, fire_threshold}} + ts. Owned by reflex_worker. (Session 2.)
REFLEX_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for reflex_state.bin.
REFLEX_STATE_MAX_BYTES: Final[int] = 2048
# Schema version for output_verifier_state.bin slot — variable msgpack {verified_count, rejected_count, sovereignty_score, threats_24h{directive,injection,consistency,identity,qualia}, recent_rejections_digest, ts}. Owned by output_verifier_worker. (Session 2.)
OUTPUT_VERIFIER_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for output_verifier_state.bin.
OUTPUT_VERIFIER_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for resonance_state.bin slot — variable msgpack {pairs, resonant_count, all_resonant, great_pulse_count, last_great_pulse_ts, config, ts}. Direct ResonanceDetector.get_stats() output. Owned by spirit_worker (titan_HCL). Added 2026-05-07 to complete rFP_phase_c_async_shm_consumer_migration §4.C.1 spirit_proxy.get_resonance migration (Session 1).
RESONANCE_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for resonance_state.bin (3 pairs × per-pair stats + counters + config — typical ~1KB, 4096B cap).
RESONANCE_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for unified_spirit_metadata.bin slot — variable msgpack UnifiedSpirit.get_stats() output (epoch_count, current_epoch_id, velocity, is_stale, consecutive_stale, stale_focus_multiplier, tensor_magnitude, tensor_sum, latest_epoch, cumulative_quality, micro_tick_count, last_alignment, enrichment_rate, full_130dt[130], config, ts). Pairs with existing unified_spirit_132d.bin (raw tensor) — metadata slot carries every queryable field. Owned by spirit_worker (titan_HCL). Added 2026-05-07 to complete rFP §4.C.1 spirit_proxy.get_unified_spirit migration (Session 1).
UNIFIED_SPIRIT_METADATA_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for unified_spirit_metadata.bin (full_130dt 130 floats + latest_epoch with 130D state vector + scalars — typical ~4-6KB, 8192B cap).
UNIFIED_SPIRIT_METADATA_MAX_BYTES: Final[int] = 8192
# Schema version for mind_state.bin slot — variable msgpack {mood_label, mood_valence, mood_intensity, current_reward, info_gain_ema, mood_history_digest, ts}. Owned by mind_worker (MoodEngine + reward telemetry). Supplements Rust-owned inner_mind_15d.bin tensor slot. Closes mind_proxy.get_mood_label/get_mood_valence/get_current_reward sync-RPC (rFP §4.B.6 + §4.C.2).
MIND_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for mind_state.bin (mood scalars + small history digest, typical ~500B, 4096B cap).
MIND_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for body_state.bin slot — variable msgpack {interoception, proprioception, somatosensation, entropy, thermal, sol_balance, sol_norm, block_delta_norm, anchor_fresh, body_health, body_details, ts}. Owned by body_worker. Supplements Rust-owned inner_body_5d.bin tensor slot with queryable body-detail metadata. Closes body_proxy.get_body_details sync-RPC (rFP §4.B.6 + §4.C.3).
BODY_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for body_state.bin (body_details dict + scalars, typical ~1KB, 4096B cap).
BODY_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for language_state.bin slot — variable msgpack {vocab_total, vocab_producible, vocab_contextual, avg_confidence, max_confidence, recent_words[], teacher_sessions_last_hour, composition_level, teacher_compositions_since, teacher_last_fire_time, ts}. Mirrors language_pipeline.update_language_stats() output (the same payload as LANGUAGE_STATS_UPDATE bus event). Owned by language_worker. Closes LANGUAGE_STATS_UPDATE RPC path (rFP §4.B.7 + §23.13 row 10).
LANGUAGE_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for language_state.bin (small scalar set, 4096B cap).
LANGUAGE_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for events_teacher_state.bin slot — variable msgpack {fingerprints_count, last_run_time, window_count, perception_buffer_size, follower_rotation_idx, mode_stats, felt_experiences, followers_tracked, windows_completed, ts}. Owned by language_worker (1Hz polling thread reads EventsTeacher JSON state + DB.get_stats — EventsTeacher itself is cron-based). Separate slot from cgn_live_weights.bin per G21. Closes events_teacher RPC path (rFP §4.B.7 + §23.13 row 9).
EVENTS_TEACHER_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for events_teacher_state.bin (curated-signal scalar telemetry, 4096B cap).
EVENTS_TEACHER_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for spirit_supplemental_state.bin slot — variable msgpack {filter_down_status, meditation_health, coordinator, nervous_system, ts}. Owned by spirit_worker. Covers the 4 spirit_loop handlers Session 1 retained sync (filter_down_status / meditation_health / coordinator / nervous_system) until producer migrated. Closes spirit_proxy fully (rFP §4.C.1 expansion — Session 4 greenlight 2026-05-07).
SPIRIT_SUPPLEMENTAL_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for spirit_supplemental_state.bin. Bumped 8192→65536 2026-05-07 after T3 deploy showed live payload at 58106B (coordinator section carries every spirit subsystem snapshot). 64KB cap covers production worst-case + 10% margin.
SPIRIT_SUPPLEMENTAL_STATE_MAX_BYTES: Final[int] = 65536
# Schema version for inner_body_firing.bin dim-firing diagnostic slot. Producer: body_worker via DimFiringTracker.record_block. Reader: api_subprocess /v4/debug/dim-sources endpoint.
INNER_BODY_FIRING_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for inner_body_firing.bin (5 dims × per-dim {v, ts} + block metadata + inputs_state).
INNER_BODY_FIRING_MAX_BYTES: Final[int] = 1024
# Schema version for inner_mind_firing.bin dim-firing diagnostic slot. Producer: mind_worker via DimFiringTracker.record_block.
INNER_MIND_FIRING_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for inner_mind_firing.bin (15 dims × per-dim {v, ts} + block metadata + 6 inputs_state).
INNER_MIND_FIRING_MAX_BYTES: Final[int] = 2048
# Schema version for inner_spirit_firing.bin dim-firing diagnostic slot. Producer: spirit_worker via DimFiringTracker.record_block.
INNER_SPIRIT_FIRING_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for inner_spirit_firing.bin (45 dims × per-dim {v, ts} + block metadata + 10 inputs_state).
INNER_SPIRIT_FIRING_MAX_BYTES: Final[int] = 4096
# Schema version for outer_body_firing.bin dim-firing diagnostic slot. Producer: outer_body_worker via DimFiringTracker.record_block.
OUTER_BODY_FIRING_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for outer_body_firing.bin (5 dims × per-dim {v, ts} + block metadata + 7 inputs_state).
OUTER_BODY_FIRING_MAX_BYTES: Final[int] = 1024
# Schema version for outer_mind_firing.bin dim-firing diagnostic slot. Producer: outer_mind_worker via DimFiringTracker.record_block.
OUTER_MIND_FIRING_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for outer_mind_firing.bin (15 dims × per-dim {v, ts} + block metadata + 12 inputs_state).
OUTER_MIND_FIRING_MAX_BYTES: Final[int] = 2048
# Schema version for outer_spirit_firing.bin dim-firing diagnostic slot. Producer: outer_spirit_worker via DimFiringTracker.record_block.
OUTER_SPIRIT_FIRING_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for outer_spirit_firing.bin (45 dims × per-dim {v, ts} + block metadata + 25 inputs_state).
OUTER_SPIRIT_FIRING_MAX_BYTES: Final[int] = 4096


# ── SCHUMANN ──────────────────────────────────────────────────────────────
# 7.83 / 23.49 / 70.47 Hz frequencies (locked by biology, NOT tunable)

# Schumann fundamental — body tick frequency (period 128 ms). LOCKED BY BIOLOGY (G13).
SCHUMANN_BODY_HZ: Final[float] = 7.83
# Schumann × 3 — mind tick frequency (period 43 ms). LOCKED BY BIOLOGY.
SCHUMANN_MIND_HZ: Final[float] = 23.49
# Schumann × 9 — spirit tick frequency (period 14 ms). LOCKED BY BIOLOGY.
SCHUMANN_SPIRIT_HZ: Final[float] = 70.47
# Substrate Schumann generator drift target over 24h — OBS-c-s3-schumann-precision pass criteria (drift < 0.1%)
SCHUMANN_DRIFT_TARGET_PCT: Final[float] = 0.1
# Substrate Schumann generator per-tick jitter p99 target — OBS-c-s3-schumann-precision pass criteria
SCHUMANN_JITTER_P99_MS: Final[float] = 1.0


# ── CLOCK ─────────────────────────────────────────────────────────────────
# Circadian + π-heartbeat + sphere clocks

# Per-clock field count: radius, scalar_position, phase, contraction_velocity, pulse_count, consecutive_balanced, last_pulse_age_s
SPHERE_CLOCK_FIELD_COUNT: Final[int] = 7
# Sphere clock count (3 inner + 3 outer = 6 trinity components)
SPHERE_CLOCK_COUNT: Final[int] = 6
# sphere_clocks.bin payload size: 6 clocks × 7 fields × float32 LE = 168 bytes (header excluded)
SPHERE_CLOCKS_PAYLOAD_BYTES: Final[int] = 168


# ── SWAP ──────────────────────────────────────────────────────────────────
# Shadow-swap orchestration (B.2.1 spawn-mode + Phase-C subtree)

# Pre-swap data checkpoint coordination timeout
SWAP_CHECKPOINT_TIMEOUT_S: Final[float] = 10.0
# Pre-swap backup creation verification timeout
SWAP_BACKUP_VERIFY_TIMEOUT_S: Final[float] = 5.0
# Post-swap integrity verification timeout (across all critical-data files)
SWAP_INTEGRITY_VERIFY_TIMEOUT_S: Final[float] = 10.0


# ── ADOPTION ──────────────────────────────────────────────────────────────
# ADOPTION_REQUEST/ACK protocol (B.2.1)

# Per-module budget for full B.2.1 adoption protocol
ADOPTION_TIMEOUT_S: Final[float] = 30.0
# Worker self-SIGTERM timeout if bus unreachable during swap_pending state
ADOPTION_SUPERVISION_TIMEOUT_S: Final[float] = 30.0


# ── MODULE ────────────────────────────────────────────────────────────────
# guardian_HCL ↔ Python module liveness (heartbeat, RSS, restart)

# Python L2/L3 module heartbeat publish interval (NOT bus-level keepalive)
MODULE_HEARTBEAT_INTERVAL_S: Final[float] = 10.0
# guardian_HCL marks module dead after this much heartbeat silence
MODULE_HEARTBEAT_TIMEOUT_S: Final[float] = 90.0
# Default per-module RSS limit (overridable via ModuleSpec.rss_limit_mb)
MODULE_DEFAULT_RSS_LIMIT_MB: Final[int] = 1500


# ── GUARDIAN_HCL ──────────────────────────────────────────────────────────
# guardian_HCL Python supervisor internals (Python-only — Rust supervision uses shm slot freshness, not heartbeat)

# Min CPU growth per heartbeat to count as alive (Python-only; Rust uses shm freshness)
GUARDIAN_HCL_MIN_CPU_DELTA_S: Final[float] = 1.0
# Consecutive starved heartbeats before force-restart
GUARDIAN_HCL_MAX_STARVED_CYCLES: Final[int] = 5
# Stable uptime resets module's restart counter (Python-only mirror of SUPERVISION_*)
GUARDIAN_HCL_SUSTAINED_UPTIME_RESET_S: Final[float] = 300.0
# Cooldown before auto-re-enable of disabled module
GUARDIAN_HCL_REENABLE_COOLDOWN_S: Final[float] = 600.0


# ── KERNEL_RPC ────────────────────────────────────────────────────────────
# Kernel↔Python RPC over /tmp/titan_kernel_<id>.sock (msgpack+HMAC, 525µs/call)

# Default RPC call timeout (kernel ↔ Python via Unix socket, msgpack+HMAC)
KERNEL_RPC_CALL_TIMEOUT_S: Final[float] = 5.0


# ── SOLANA_RPC ────────────────────────────────────────────────────────────
# Solana network RPC client (mainnet/devnet) — Python-only, named distinctly from KERNEL_RPC

# SOL balance polling cadence (mainnet/devnet)
SOLANA_RPC_BALANCE_POLL_INTERVAL_S: Final[float] = 60.0


# ── DATA ──────────────────────────────────────────────────────────────────
# Data integrity primitives — atomic-write helpers, DB checkpoint, backup retention, integrity check (G16)

# Periodic auto-checkpoint for DuckDB/SQLite connections
DATA_CHECKPOINT_INTERVAL_S: Final[float] = 60.0
# Wait for in-flight DB queries during graceful shutdown
DATA_QUERY_DRAIN_TIMEOUT_S: Final[float] = 2.0
# Backup file retention (.bak + .bak.prev — 2 generations)
DATA_BACKUP_RETENTION_GENERATIONS: Final[int] = 2
# Per-file integrity-check timeout at boot (G16 invariant 5)
DATA_INTEGRITY_CHECK_TIMEOUT_S: Final[float] = 5.0


# ── WORKER ────────────────────────────────────────────────────────────────
# Generic Python worker shutdown grace (used by all L2/L3 modules under guardian_HCL)

# Python L2/L3 module SIGTERM→SIGKILL grace
WORKER_SHUTDOWN_GRACE_S: Final[float] = 5.0


# ── OUTER ─────────────────────────────────────────────────────────────────
# Outer trinity daemon cadences (NOT Schumann-locked; per SPEC §18.1)

# Outer-body Python sensor sidecar refresh period; stale-threshold = 3× this. Post-A.S8 the Rust daemon ticks at SCHUMANN_BODY_HZ (7.83 Hz) — this constant defines sidecar cadence + sensor staleness, NOT daemon tick rate.
OUTER_BODY_TICK_BASE_S: Final[float] = 10.0
# Outer-body sensor sidecar jitter (±). DEPRECATED for daemon tick — daemon now uses SchumannGenerator (no jitter).
OUTER_BODY_TICK_JITTER_PCT: Final[int] = 20
# Outer-body daemon bus publish throttle interval. Daemon ticks at Schumann body (7.83 Hz) but throttles MIND_STATE/SPIRIT_STATE bus publishes to this cadence. Body-slowest G13 invariant: this > OUTER_MIND_BUS_PUBLISH_INTERVAL_S > OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S.
OUTER_BODY_BUS_PUBLISH_INTERVAL_S: Final[float] = 45.0
# Outer-mind Python sensor sidecar refresh period; stale-threshold = 3× this. Post-A.S8 the Rust daemon ticks at SCHUMANN_MIND_HZ (23.49 Hz) — this constant defines sidecar cadence + sensor staleness, NOT daemon tick rate.
OUTER_MIND_TICK_BASE_S: Final[float] = 5.0
# Outer-mind sensor sidecar jitter (±). DEPRECATED for daemon tick.
OUTER_MIND_TICK_JITTER_PCT: Final[int] = 20
# Outer-mind daemon bus publish throttle interval. Daemon ticks at Schumann mind (23.49 Hz) but throttles MIND_STATE bus publishes to this cadence.
OUTER_MIND_BUS_PUBLISH_INTERVAL_S: Final[float] = 15.0
# Outer-spirit daemon bus publish throttle interval. Daemon ticks at Schumann spirit (70.47 Hz) but throttles SPIRIT_STATE / OUTER_SPIRIT_FILTER_DOWN bus publishes to this cadence. Spirit-fastest at bus layer (mirrors inner spirit publish rate).
OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S: Final[float] = 5.0
# Outer-spirit Python sensor sidecar refresh period; stale-threshold = 3× this. Post-A.S8 the Rust daemon ticks at SCHUMANN_SPIRIT_HZ (70.47 Hz) — this constant defines sidecar cadence + sensor staleness, NOT daemon tick rate.
OUTER_SPIRIT_TICK_BASE_S: Final[float] = 30.0
# Outer-spirit sensor sidecar jitter (±). DEPRECATED for daemon tick.
OUTER_SPIRIT_TICK_JITTER_PCT: Final[int] = 10
# Start dim of outer_mind willing range (ground_up applies here per G10 ground_up_mind_range=10:15)
OUTER_MIND_WILLING_DIM_START: Final[int] = 10
# End dim (exclusive) of outer_mind willing range
OUTER_MIND_WILLING_DIM_END: Final[int] = 15
# Sensor cache staleness threshold = N × daemon's natural cadence (wall_ns < now − N×cadence → cache stale, daemon writes last-known with confidence=0.0 log)
OUTER_CACHE_STALE_CADENCE_MULTIPLIER: Final[int] = 3
# Max payload bytes for sensor_cache_outer_body.bin (msgpack source dict per O4 lock — agency_stats/helper_statuses/bus_stats/system_sensor_stats/network_monitor_stats/tx_latency_stats/block_delta_stats/anchor_state/sol_balance + Step 3 hormone_levels). Bumped 8192→65536 2026-05-10 after Step 3 §3.1 SOURCE_KEYS extension pushed observed payloads past 8KB on T3 (live max ~33KB pre-cap).
OUTER_SENSOR_CACHE_BODY_MAX_BYTES: Final[int] = 65536
# Max payload bytes for sensor_cache_outer_mind.bin (msgpack source dict — persona narrative + social context + creative_stats + memory_stats + agency_stats + Step 3 meta_cgn/cgn/memory/language/events_teacher/social_x/output_verifier/jailbreak). Bumped 8192→65536 2026-05-10 (live oversize observed at 32970B).
OUTER_SENSOR_CACHE_MIND_MAX_BYTES: Final[int] = 65536
# Max payload bytes for sensor_cache_outer_spirit.bin (msgpack pre-aggregated outer-state — action_stats/sovereignty_ratio/uptime_ratio/social_stats/solana_stats/hormone_levels/recovery_stats/creative_stats/memory_stats/assessment_stats/history + Step 3 meta_cgn/cgn/memory/KG/events_teacher/jailbreak/output_verifier/anchor/bus/expression/osh/community_engagement/inner_memory). Bumped 8192→65536 2026-05-10 (live oversize observed at 32994B).
OUTER_SENSOR_CACHE_SPIRIT_MAX_BYTES: Final[int] = 65536


# ── OUTER_SPIRIT ──────────────────────────────────────────────────────────
# Outer-spirit local filter_down (Phase C 3-level cascade addition)

# Start dim of outer_spirit_45d local-frame observer range (= TITAN_SELF absolute [85:90] per G8 outer_spirit_observer_dims_masked=85:90); MASKED from filter_down output only — slot itself contains all 45D
OUTER_SPIRIT_OBSERVER_DIM_START: Final[int] = 0
# End dim (exclusive) of outer_spirit_45d local-frame observer range
OUTER_SPIRIT_OBSERVER_DIM_END: Final[int] = 5
# Start dim of outer_spirit_45d local-frame content range (= TITAN_SELF absolute [90:130] per G9 outer_spirit_content_range=90:130); 40 dims [5:45] = SAT[5:15]+CHIT[15:30]+ANANDA[30:45] minus observer; this is the slice published in OUTER_SPIRIT_FILTER_DOWN.outer_spirit_content[40]
OUTER_SPIRIT_CONTENT_DIM_START: Final[int] = 5
# End dim (exclusive) of outer_spirit_45d local-frame content range
OUTER_SPIRIT_CONTENT_DIM_END: Final[int] = 45


# ── COGNITIVE ─────────────────────────────────────────────────────────────
# L3 cognitive engine cluster owned by cognitive_worker L2 module (NEW in v0.2.0 per PLAN_microkernel_phase_c_s8_cognitive_worker_extraction.md). Adaptive consciousness epoch driver constants — 1-30s tick interval bounded by Schumann body period. NOT coupled to Rust trinity-rs Schumann publish rates per Maker D4 (a) — cognitive_worker has its own adaptive timer that fires on max-interval, resonance transition, or hormonal urgency.

# Cognitive epoch tick floor — never tick faster than 1× Schumann body period (1.15s). Prevents over-driving reasoning_engine + meta_engine in pathological resonance cascades.
COGNITIVE_EPOCH_MIN_INTERVAL_S: Final[float] = 1.15
# Cognitive epoch tick default cadence — 9× Schumann body (10.35s). Matches legacy spirit_loop._run_consciousness_epoch cadence so chain commit / dream insert / π-heartbeat observation rates stay constant across the 4A→4B cutover.
COGNITIVE_EPOCH_DEFAULT_INTERVAL_S: Final[float] = 10.35
# Cognitive epoch tick ceiling — 27× Schumann body (31.05s). Force epoch fire if no resonance transition / hormonal urgency / max-interval trigger arrived. Bounds the worst-case staleness of /v4/reasoning + /v4/dreaming + /v4/pi-heartbeat under low-arousal idle.
COGNITIVE_EPOCH_MAX_INTERVAL_S: Final[float] = 31.05
# Engine state persistence cadence — every 100 epochs (≈10–30 min wall time at default 10.35s, max 51 min at 31.05s ceiling). Persists reasoning_totals.json + dreaming_state.json + pi_heartbeat_state.json + neural_ns/* + msl/* atomically per G16 invariants. Intermediate crash recovers from last checkpoint without losing chain history.
COGNITIVE_PERSIST_EVERY_N_EPOCHS: Final[int] = 100
