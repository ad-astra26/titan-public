"""
_phase_c_constants.py — AUTO-GENERATED from titan-docs/specs/SPEC_titan_architecture_constants.toml.

DO NOT EDIT BY HAND. Edit the TOML, then run:
    python scripts/generate_phase_c_constants.py

SPEC version: 1.11.4
Source SHA-256: 4570bb68d59d5d6861d6c69aef324c956c92a2760a2fead44d15e7eee17cfeaa

Per SPEC §19 + §2.6: hand-editing this file is a SPEC violation flagged by
`arch_map phase-c verify`.
"""
from __future__ import annotations
from typing import Final

# ── SPEC version metadata ──────────────────────────────────────────────
SPEC_VERSION: Final[str] = "1.11.4"
SPEC_SOURCE_SHA256: Final[str] = "4570bb68d59d5d6861d6c69aef324c956c92a2760a2fead44d15e7eee17cfeaa"


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
# SPEC §11.G.2.5 (D-SPEC-90, v1.29.0) — max wait for an ENSURE_RUNNING dep to reach state=RUNNING + emit MODULE_READY before the dependent's start gives up. Sized to comfortably exceed memory_worker's empirical cold-boot time on devnet T3 (~52s for FAISS+Kuzu+DuckDB hot init observed 2026-05-19) plus headroom; below the 120s module heartbeat_timeout for L2/L3 workers.
SUPERVISION_DEPENDENCY_ACTIVATION_TIMEOUT_S: Final[float] = 90.0
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
# Per-client outbound buffer depth at which a rate-limited WARN fires (SPEC §8.0.ter). NOT a cap — per §8.0 P0 never-drop, frames stay queued; this is operator-visible backpressure signal surfaced via /v4/warning-monitor under key bus_client.<name>.
OUTBOUND_BUFFER_HIGH_WATER: Final[int] = 1000
# Per-client rate-limit on the SPEC §8.0.ter high-water WARN — one log line per client per minute. Prevents sustained-backpressure log flood.
OUTBOUND_BUFFER_HIGH_WATER_WARN_INTERVAL_S: Final[float] = 60.0
# BusSocketClient writer thread defensive idle-poll interval. Writer wakes on _outbound_event signal OR every N seconds. Idle polling lets the writer recover if the event was somehow missed (defense-in-depth; signal is set on every enqueue + on every (re)connect). SPEC §8.0.ter.
OUTBOUND_WRITER_IDLE_POLL_S: Final[float] = 1.0
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
# Schema version for neuromod_state.bin slot. v1=(6,) levels only. v2=(6,4) (level,gain,phasic,tonic) per modulator — added 2026-05-15 with §4.Q neuromod_worker.evaluate migration.
NEUROMOD_SCHEMA_VERSION: Final[int] = 2
# Number of neuromodulators in neuromod_state.bin (DA, 5HT, NE, ACh, Endorphin, GABA)
NEUROMOD_FIELD_COUNT: Final[int] = 6
# Per-modulator fields in neuromod_state.bin v2 layout (level, gain, phasic, tonic) — required by cognitive_worker for cross-process modulation reconstruction via compute_modulation_from_state().
NEUROMOD_FIELDS_PER_MOD: Final[int] = 4
# Total payload bytes for neuromod_state.bin v2 = NEUROMOD_FIELD_COUNT × NEUROMOD_FIELDS_PER_MOD × 4 (f32 LE) = 6 × 4 × 4 = 96 bytes
NEUROMOD_PAYLOAD_BYTES: Final[int] = 96
# Schema version for neuromod_inputs.bin slot — cognitive_worker writes the 11 emergent inputs + chi_health + kin signal + topology_velocity as msgpack; neuromod_worker reads per KERNEL_EPOCH_TICK. Added 2026-05-15 with §4.Q.
NEUROMOD_INPUTS_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload size for neuromod_inputs.bin. Typical payload ≈400-700 bytes (11 emergent inputs × ~20 bytes each + small DNA section).
NEUROMOD_INPUTS_MAX_BYTES: Final[int] = 4096
# Schema version for epoch_counter.bin slot
EPOCH_COUNTER_SCHEMA_VERSION: Final[int] = 1
# Schema version for sphere_clocks.bin slot (6 × 7 fields)
SPHERE_CLOCKS_SCHEMA_VERSION: Final[int] = 1
# Schema version for chi_state.bin slot
CHI_STATE_SCHEMA_VERSION: Final[int] = 1
# Schema version for titanvm_registers.bin slot (11 NS programs × 4 fields)
TITANVM_REGISTERS_SCHEMA_VERSION: Final[int] = 1
# Schema version for ns_program_urgencies_input.bin slot (11 × float32 = 44 bytes payload — cross-process bridge from cognitive_worker canonical NS evaluator to ns_worker titanvm_registers.bin urgency-column writer + NS_URGENCIES_UPDATE → emot_cgn ns_urgencies substrate cache). NS_PROGRAMS row order from emot_bundle_protocol.py. Closes the load-bearing wire-up gap from ns_worker L2 carve-out (Phase C). Pattern mirrors NEUROMOD_INPUTS (§4.Q D-SPEC-57) + LIFE_FORCE_INPUTS (§4.G D-SPEC-57).
NS_PROGRAM_URGENCIES_INPUT_SCHEMA_VERSION: Final[int] = 1
# Schema version for trajectory_state.bin slot (2 × float32 = 8 bytes payload — curvature + density global meta-scalars from state_132d[130:132] per consciousness.py:46). cognitive_worker writes from coordinator's freshly-computed values per _run_consciousness_epoch (bypassing the broken consciousness.latest_epoch.state_vector snapshot pipe). Reader: emot_cgn_worker (substrate trajectory_2d key in bundle). Retires TRAJECTORY_UPDATE bus event (PART B §8 D-SPEC-65 v1.9.6) per G18.
TRAJECTORY_STATE_SCHEMA_VERSION: Final[int] = 1
# Schema version for cgn_beta_state.bin slot (8 × float32 = 32 bytes payload — per-consumer reward_ema in CGN_CONSUMERS order: language, social, knowledge, reasoning, coding, self_model, reasoning_strategy, meta from emot_bundle_protocol.py:172-175). cgn_worker writes from live cgn_consumer._reward_ema attribute (NOT the snapshot that defaults to 0.5). Reader: emot_cgn_worker (substrate cgn_beta_states key in bundle). Retires CGN_BETA_SNAPSHOT bus event (§23.6a) per G18.
CGN_BETA_STATE_SCHEMA_VERSION: Final[int] = 1
# Schema version for hormonal_state.bin slot (11 hormones × 4 fields × float32 = 176 bytes payload). Canonical hormone order matches NS_PROGRAMS in emot_bundle_protocol.py (REFLEX, FOCUS, INTUITION, IMPULSE, METABOLISM, CREATIVITY, CURIOSITY, EMPATHY, REFLECTION, INSPIRATION, VIGILANCE). Per-hormone fields: level, threshold, refractory, peak_level (read-mostly state surfaced from HormonalPressure class in titan_hcl/logic/hormonal_pressure.py).
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
# Schema version for recorder_state.bin slot — variable msgpack {programs[], current_program_id, dream_quality, training_loss_ema, transitions, last_train_ts, ts}. Owned by recorder_worker (was rl_worker prior to v1.8.4 §4.N rename, D-SPEC-58). (Session 2.)
RL_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for recorder_state.bin.
RL_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for interface_advisor_state.bin slot — variable msgpack {rates: {msg_type → current_rate_in_window}, limits: dict[msg_type → limit], window_s: float, rate_limit_count: int, schema_version: int, ts: float}. Owned by interface_advisor_worker (G21 single-writer). NEW v1.8.5 §4.H (D-SPEC-59).
INTERFACE_ADVISOR_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for interface_advisor_state.bin. NEW v1.8.5 §4.H (D-SPEC-59).
INTERFACE_ADVISOR_STATE_MAX_BYTES: Final[int] = 512
# Minimum cadence between SHM republishes from interface_advisor_worker after IMPULSE_RECEIVED activity. Caps publish rate to 10Hz to avoid SHM thrash under burst. NEW v1.8.5 §4.H (D-SPEC-59).
INTERFACE_ADVISOR_RATE_REFRESH_CADENCE_S: Final[float] = 0.1
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
# Schema version for resonance_metadata.bin slot — variable msgpack ResonanceDetector::get_stats() Rust-side output (pairs, resonant_count, all_resonant, great_pulse_count, last_great_pulse_ts, config, schema_version, ts). Rust-owned by titan-unified-spirit-rs (G21 single-writer). Replaces Python wrapper resonance_state.bin per rFP_phase_c_state_read_unification Phase B.
RESONANCE_METADATA_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for resonance_metadata.bin (3 pairs × per-pair stats + counters + config — typical ~1.5KB, 4096B cap).
RESONANCE_METADATA_MAX_BYTES: Final[int] = 4096
# Schema version for filter_down_state.bin slot — variable msgpack FilterDownV5Engine::get_stats() output (version, input_dim, output_dim, buffer_size, total_train_steps, last_loss, publish_enabled, spirit_filter_strength, cold_start_floor, multipliers_mean, multipliers, schema_version, ts). Rust-owned by titan-unified-spirit-rs (G21 single-writer). NEW slot replacing the bus-event-only V5 publish path with G18-compliant SHM read.
FILTER_DOWN_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for filter_down_state.bin (120 multipliers + 6 means + scalars — typical ~4KB, 8192B cap).
FILTER_DOWN_STATE_MAX_BYTES: Final[int] = 8192
# Schema version for mind_state.bin slot — variable msgpack {mood_label, mood_valence, mood_intensity, current_reward, info_gain_ema, mood_history_digest, ts}. Owned by mind_worker (MoodEngine + reward telemetry). Supplements Rust-owned inner_mind_15d.bin tensor slot. Closes mind_proxy.get_mood_label/get_mood_valence/get_current_reward sync-RPC (rFP §4.B.6 + §4.C.2).
MIND_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for mind_state.bin (mood scalars + small history digest, typical ~500B, 4096B cap).
MIND_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for expression_state.bin slot — variable msgpack {sovereignty_ratio, learned_actions, llm_actions, total_actions, posture_authenticity_ratio_30, top_mappings[], total_learned_pairs, intensity, composites{name → {urge, threshold, fire_count}}, ts}. Owned by main plugin (ExpressionTranslator.get_stats() + ExpressionManager.get_stats()). Closes rFP_phase_c_130d_rust_l1_port §3.2 — feeds inner_spirit SAT[2] sovereignty + CHIT[28] causal_understanding + ANANDA[8] expression_quality + outer_spirit SAT[1] expressive_authenticity. Sprint 7 §4.6 closure 2026-05-12.
EXPRESSION_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for expression_state.bin (sovereignty stats + ~15 top_mappings + composites dict, typical ~2-3KB, 8192B cap).
EXPRESSION_STATE_MAX_BYTES: Final[int] = 8192
# Schema version for inner_perception_state.bin slot — variable msgpack {audio_state, visual_state, ambient_change, last_create_ts, ts}. Owned by main plugin (InnerPerception is parent-resident hardware ambient sampler). Phase C dissolution 2026-05-22: replaces the OUTER_SOURCES_SNAPSHOT.inner_perception_stats bus delivery to mind_worker (G18). Feeds inner_mind feeling[5] inner_hearing / [7] inner_sight / [9] inner_smell + outer_spirit ANANDA[41] creative_tension.
INNER_PERCEPTION_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for inner_perception_state.bin (audio_state + visual_state dicts + 2 scalars, typical <1KB, 4096B cap).
INNER_PERCEPTION_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for body_state.bin slot — variable msgpack {interoception, proprioception, somatosensation, entropy, thermal, sol_balance, sol_norm, block_delta_norm, anchor_fresh, body_health, body_details, ts}. Owned by body_worker. Supplements Rust-owned inner_body_5d.bin tensor slot with queryable body-detail metadata. Closes body_proxy.get_body_details sync-RPC (rFP §4.B.6 + §4.C.3).
BODY_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for body_state.bin (body_details dict + scalars, typical ~1KB, 4096B cap).
BODY_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for social_x_state.bin slot — variable msgpack {recent_posts, current_urge, post_threshold, next_allowed_post_ts, posts_this_hour, posts_today, catalysts_pending, last_archetype_fired, archetype_dispatch_state, last_post_ts, is_canonical_poller, titan_id, ts}. Owned by social_worker (Phase C-S9 §4.C — chunk 9E SHIPPED 2026-05-12). 1 Hz publisher. Consumed by /v4/social Observatory route + dim-live producers (ANANDA[11]/[36]/[38]). Restored 2026-05-12 after auto-regen sweep dropped the entries — same pattern as f58d998e (EXPRESSION_STATE).
SOCIAL_X_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for social_x_state.bin (recent_posts list ~5 × 200B + scalars + archetype_dispatch_state dict, typical 10-20KB on T1 with active orchestrator, 32KB cap covers production worst-case + 10% margin).
SOCIAL_X_STATE_MAX_BYTES: Final[int] = 32768
# Schema version for language_state.bin slot — variable msgpack {vocab_total, vocab_producible, vocab_contextual, avg_confidence, max_confidence, recent_words[], teacher_sessions_last_hour, composition_level, teacher_compositions_since, teacher_last_fire_time, ts}. Mirrors language_pipeline.update_language_stats() output (the same payload as LANGUAGE_STATS_UPDATE bus event). Owned by language_worker. Closes LANGUAGE_STATS_UPDATE RPC path (rFP §4.B.7 + §23.13 row 10).
LANGUAGE_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for language_state.bin (small scalar set, 4096B cap).
LANGUAGE_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for events_teacher_state.bin slot — variable msgpack {fingerprints_count, last_run_time, window_count, perception_buffer_size, follower_rotation_idx, mode_stats, felt_experiences, followers_tracked, windows_completed, ts}. Owned by language_worker (1Hz polling thread reads EventsTeacher JSON state + DB.get_stats — EventsTeacher itself is cron-based). Separate slot from cgn_live_weights.bin per G21. Closes events_teacher RPC path (rFP §4.B.7 + §23.13 row 9).
EVENTS_TEACHER_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for events_teacher_state.bin (curated-signal scalar telemetry, 4096B cap).
EVENTS_TEACHER_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for spirit_supplemental_state.bin slot — variable msgpack {filter_down_status, meditation_health, coordinator, nervous_system, post_context, ts}. Owned by spirit_worker (l0_rust_enabled=false) or cognitive_worker (l0_rust_enabled=true) — G21 single-writer holds per-Titan via mutually-exclusive flag gating. Covers the 4 spirit_loop handlers Session 1 retained sync (filter_down_status / meditation_health / coordinator / nervous_system) plus Phase C-S9 chunk 9Q-1 post_context section (pi_ratio + expression_composites_fire_counts + social_contagion_latest) consumed by social_worker's post-dispatch orchestration tick for PostContext construction. Schema bump 1→2 is backward-compatible — pre-9Q-1 consumers ignore the new key.
SPIRIT_SUPPLEMENTAL_STATE_SCHEMA_VERSION: Final[int] = 2
# Max msgpack payload bytes for spirit_supplemental_state.bin. Bumped 8192→65536 2026-05-07 after T3 deploy showed live payload at 58106B (coordinator section carries every spirit subsystem snapshot). 64KB cap covers production worst-case + 10% margin. Phase C-S9 chunk 9Q-1 post_context section adds ~150 bytes (3 small subfields); no further bump needed for v2.
SPIRIT_SUPPLEMENTAL_STATE_MAX_BYTES: Final[int] = 65536
# Schema version for social_graph_state.bin slot — variable msgpack {users: int, edges: int, donations: int, total_donated_sol: float, inspirations: int, engagement_ledger_today: int, schema_version: int, ts: float}. Owned by social_graph_worker (G21 single-writer). Closes G22 violation: mind_worker `_sense_taste` formerly called `social_graph.get_stats()` over a bus QUERY (the `get_social_stats` orphan-handler on phase_c_rpc_exemptions.yaml::orphan_handler_allowlist rationale: 'SocialGraph stats; full migration deferred'). Now reads SHM directly per G18 (state via SHM, never bus).
SOCIAL_GRAPH_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for social_graph_state.bin. Small fixed payload (~200 bytes nominal: 6 ints + 1 float + ts; cap at 8KB for headroom on future small extensions). 1Hz publisher cadence; G21 single-writer (social_graph_worker).
SOCIAL_GRAPH_STATE_MAX_BYTES: Final[int] = 8192
# Schema version for metabolism_state.bin slot — variable msgpack {tier: str, balance_pct: float, gates_enforced: bool, last_gate_decision_reason: str, tier_info: dict, last_tier_change_ts: float, social_gravity_score: float, schema_version: int, ts: float}. Owned by metabolism_worker (G21 single-writer). Hot-path tier + gates_enforced reads (Soul NFT mint, memo_inscribe, dashboard /status, kernel `metabolism.get_metabolic_tier` proxy) bypass bus entirely via this slot per G18+G20.
METABOLISM_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for metabolism_state.bin. Small fixed payload (~400 bytes nominal: 1 str tier + 2 floats + 1 bool + 1 str reason + tier_info dict (~6 keys × small values) + 2 floats + 1 int schema + 1 float ts; cap at 2KB for headroom on future small extensions). 1Hz publisher cadence; G21 single-writer (metabolism_worker).
METABOLISM_STATE_MAX_BYTES: Final[int] = 2048
# Schema version for dream_state.bin slot — variable msgpack {is_dreaming: bool, state: str (∈ {"awake", "dreaming", "dream_start", "dream_end"}), recovery_pct: float, remaining_epochs: int, wake_transition: bool, just_woke: bool, wake_ts: float, dream_started_ts: float, last_transition_ts: float (freshness probe), schema_version: int, ts: float}. Owned by dream_state_worker (G21 single-writer; sole writer under Phase C). Dual-trigger republish: on every KERNEL_EPOCH_TICK (1.0 Hz adaptive) + on every DREAMING_STATE_UPDATED arrival per D-SPEC-56 Maker Q6 greenlight. Hot-path is_dreaming reads (plugin chat-during-dream buffer decision, api_subprocess chat-bridge buffer decision, spirit_worker _read_is_dreaming_from_shm helper replacing the deleted _shared_is_dreaming module-level flag + 20+ readers, expression_worker tick-gate cache, timechain_worker dream-hook) bypass bus entirely via this slot per G18+G20. Closes the latent fleet-wide Phase C DREAM_STATE_CHANGED silent-emit bug.
DREAM_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for dream_state.bin. Base payload (~180 bytes: is_dreaming + state ∈ 4-set + recovery_pct + remaining_epochs + 2 bools + 3 floats + schema_version + ts) PLUS circadian telemetry (cycle_count + fatigue + developmental_age + epochs_since_dream) consumed by the Observatory DreamingTab/DreamingIndicator/CircadianClock — those pushed the payload to ~268B, overflowing the prior 256B cap (oversize guard rejected every write → dream_state.bin never created → /v4/dreaming lost cycle_count/epochs_since_dream/state). Bumped 256 → 512 for the new fields + headroom. Slot total = 64 + 3·512 = 1600 bytes. 1Hz dual-trigger publisher cadence; G21 single-writer (dream_state_worker).
DREAM_STATE_MAX_BYTES: Final[int] = 512
# Schema version for life_force_state.bin slot — variable msgpack {total: float ∈ [0,1], spirit: ChiLayer, mind: ChiLayer, body: ChiLayer (each: {raw, effective, weight, thinking, feeling, willing, components: dict}), circulation: float, weights: {spirit, mind, body}, state: str ∈ {FLOURISHING/HEALTHY/CONSERVING/SURVIVAL/STARVATION}, developmental_phase: str ∈ {BIRTH/YOUTH/MATURE}, contemplation: {active, phase ∈ [0,4], phase_name, conviction, conviction_threshold, mature_enough, survival_mode?, action?}, metabolic_drain: float ∈ [0,0.8], is_dreaming: bool, schema_version: int, ts: float}. Owned by life_force_worker (G21 single-writer). Hot-path readers: cognitive_worker (MSL static_context chi_total + reasoning body_state + hormonal_pressure inputs + ground_up_enricher chi_overlay + NN modulation cap), api_subprocess (/v4/chi route via chi.state cache key — populated by CHI_UPDATED bus event whose producer is now life_force_worker), metabolism_worker (soft-dep tier weighting per metabolism_worker.py:93-95 NULL-safe subscriber, wired in v1.7.2 awaiting this slot). 1 Hz cadence; republish-on-change content-hash gated.
LIFE_FORCE_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for life_force_state.bin. Typical 1200-1600 bytes (3 ChiLayer dicts × ~400 bytes each + circulation + weights + state + contemplation + drain + ts; cap at 4096B for headroom on future components-dict extensions per Maker Q2 lock 2026-05-15). 1Hz publisher cadence; G21 single-writer (life_force_worker).
LIFE_FORCE_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for life_force_inputs.bin slot — variable msgpack {pi_heartbeat_ratio, developmental_age: int, sovereignty_index: int, spirit_coherence, vocabulary_size: int, learning_rate_gain, emotional_coherence, neuromodulator_homeostasis, mind_coherence, expression_fire_rate, sol_balance, anchor_freshness, hormonal_vitality, body_coherence, topology_grounding, infrastructure_health: float (each input × float), schema_version: int, ts: float}. Cross-process bridge feeding life_force_worker.evaluate per KERNEL_EPOCH_TICK — cognitive_worker aggregates the 16 emergent inputs from its in-process state via compute_life_force_inputs(...) and writes the result here. Mirrors §4.Q neuromod_inputs.bin pattern (same writer/reader roles, same cadence). G21 single-writer = cognitive_worker.
LIFE_FORCE_INPUTS_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for life_force_inputs.bin. Typical 300-450 bytes (16 floats + 3 ints + schema + ts; cap at 1024B for headroom). 1Hz cadence matching KERNEL_EPOCH_TICK (adaptive 1-30s); G21 single-writer (cognitive_worker).
LIFE_FORCE_INPUTS_MAX_BYTES: Final[int] = 1024
# Schema version for meditation_state.bin slot — variable msgpack {tracker: {last_epoch: int, count: int, count_since_nft: int, last_ts: float, in_meditation: bool, current_phase: str (∈ {"idle", "entering", "deep", "exiting"})}, watchdog: {last_check_ts: float, gap_samples: int, expected_interval_hours: float, in_meditation_since_ts: float, consecutive_zero_promoted: int, selftest_done: bool, selftest_pass: bool}, last_alert: {severity, failure_mode, detail, ts} | null, last_completion: {epoch, promoted, pruned, trigger, success, ts} | null, schema_version: int, ts: float}. Owned by meditation_worker (G21 single-writer; sole writer under Phase C). Dual-trigger republish: on every KERNEL_EPOCH_TICK (1.0 Hz adaptive floor) + on every transition (in_meditation flip, phase change, watchdog alert, completion) per D-SPEC-57 Maker Q1/Q3 greenlight. Hot-path reads (/v4/meditation/health dashboard, daily_nft trigger, soul-NFT mint cron) bypass bus entirely via this slot per G18+G20. Closes the cross-process state_refs['meditation_tracker'] direct dict reference + spirit_supplemental_state.bin meditation_health section indirection (G21 violation).
MEDITATION_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for meditation_state.bin. Fixed small payload (~400 bytes nominal: tracker section ~60B + watchdog section ~80B + last_alert ~80B + last_completion ~100B + framing ~80B; cap at 1024B for headroom on long failure_mode/detail strings). 1Hz + on-transition publisher cadence; G21 single-writer (meditation_worker).
MEDITATION_STATE_MAX_BYTES: Final[int] = 1024
# Schema version for studio_state.bin slot — variable msgpack payload (matches the Python L2 slot family pattern: metabolism_state.bin D-SPEC-51 / social_graph_state.bin D-SPEC-50 / dream_state.bin D-SPEC-56). Schema: {schema_version: int, meditation_count: int, epoch_count: int, eureka_count: int, last_render_ts: float (unix epoch, 0 if no renders yet), last_render_type: str ∈ {"none", "meditation", "epoch", "eureka"}, output_root: str (output_path config — readers can hash for config-drift detection), default_resolution: int, highres_resolution: int, nft_composite_enabled: bool, ts: float}. Owned by studio_worker (G21 single-writer; sole writer under Phase C). Dual-trigger republish on every KERNEL_EPOCH_TICK (1.0 Hz adaptive cadence — keeps counts fresh against external dir scans) + immediately after each successful render (updates counts + last_render_ts / last_render_type). Hot-path stats reads (/v4/studio/stats Observatory route, future dashboards) bypass bus entirely via this slot per G18+G20.
STUDIO_STATE_SCHEMA_VERSION: Final[int] = 1
# Max msgpack payload bytes for studio_state.bin. Fixed small payload (~180 bytes nominal: ~11 keys, all primitives + 2 short strings); cap at 512B for headroom. 1Hz dual-trigger publisher cadence; G21 single-writer (studio_worker).
STUDIO_STATE_MAX_BYTES: Final[int] = 512
# Schema version for agno_state.bin slot — variable msgpack. v1 (1.17.0): {schema_version, session_count, last_chat_ts, total_chats_24h, provider_stats, dream_inbox_size, ts}. v2 (1.18.0, D-SPEC-76): adds {session_cache_size, session_hits, session_misses} for agno session pre-warm LRU observability. Owned by agno_worker (G21 single-writer). Dual-trigger republish: every KERNEL_EPOCH_TICK (1.0 Hz adaptive floor) + immediately after every chat completion.
AGNO_STATE_SCHEMA_VERSION: Final[int] = 2
# Max msgpack payload bytes for agno_state.bin. Fixed small payload (~250 bytes nominal: 5 primitive fields + provider_stats dict per active provider ~50B each + 3 LRU counters @v2); cap at 512B for headroom. 1Hz + on-completion publisher cadence; G21 single-writer (agno_worker).
AGNO_STATE_MAX_BYTES: Final[int] = 512
# Default capacity for agno_worker's (user_id, session_id) LRU pre-warm cache (D-SPEC-76). Overridable via [agno_worker].session_cache_capacity in titan_hcl/config.toml. Hits/misses/size surfaced in agno_state.bin schema v2 for Observatory + health monitor.
AGNO_SESSION_CACHE_DEFAULT_CAPACITY: Final[int] = 32
# Maximum age of a safety_verdict_token (HMAC issued by OutputVerifier.verify_safety) consumable by OutputVerifier.sign_and_commit (D-SPEC-74). Matches the 90s phase_c_rpc_exemptions.yaml allowlist for agno_proxy → agno_worker work-RPC ceiling — no signing of in-flight requests that would have already returned 504 to the user. Defense-in-depth: prevents replay of stale safety verdicts; combined with HMAC binding to (prompt, response, channel, ts), forging signing without a paired in-window safety check is cryptographically prevented.
OVG_SAFETY_VERDICT_TOKEN_TTL_S: Final[float] = 90.0
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
# Schema version for soul_state.bin slot — variable msgpack {maker_pubkey: str, nft_address: str, current_gen: int, active_directives: list[dict], directives_count: int, last_directive_ts: float, soul_initialized: bool, schema_version: int, ts: float}. Owned by sovereignty_worker (G21 single-writer). Reader: api_subprocess StateAccessor.soul + IdentityAccessor.maker_pubkey fallback per Phase A.4 closing the soul.state bus-cache lookup.
SOUL_STATE_SCHEMA_VERSION: Final[int] = 1
# Max bytes for soul_state.bin payload. Typical ≈ 600-1200 bytes (depends on active_directives count). Bounded 2KB cap.
SOUL_STATE_MAX_BYTES: Final[int] = 2048
# Schema version for cgn_engine_state.bin slot — variable msgpack {consumers: dict[str→dict], total_transitions: int, buffer_size: int, consolidations: int, anchor_count: int, sigma_updates: int, soar_impasses: int, haov_stats: dict, schema_version: int, ts: float}. Owned by cgn_worker (G21 single-writer). Sibling to existing cgn_live_weights.bin (tensor) + cgn_beta_state.bin (8-float per-consumer reward EMA) — this slot carries the engine-level stats previously surfaced via cgn.stats bus-cache key. Reader: api_subprocess StateAccessor.cgn.
CGN_ENGINE_STATE_SCHEMA_VERSION: Final[int] = 1
# Max bytes for cgn_engine_state.bin payload. CGN has 9 consumers × ~10 fields each plus aggregates → typically 3-5KB. Bounded 8KB cap.
CGN_ENGINE_STATE_MAX_BYTES: Final[int] = 8192
# Schema version for reasoning_state.bin slot — variable msgpack {total_chains: int, total_commits: int, commit_rate: float, avg_chain_length: float, buffer_size: int, current_active: bool, last_action: str, last_outcome: str, action_distribution: dict[str→int], schema_version: int, ts: float}. Owned by cognitive_worker (engine lives there per SPEC §1 glossary). Reader: api_subprocess StateAccessor.reasoning + memory.get_reasoning_state.
REASONING_STATE_SCHEMA_VERSION: Final[int] = 1
# Max bytes for reasoning_state.bin payload. Typical ≈ 800-1500 bytes. Bounded 4KB cap.
REASONING_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for meta_reasoning_state.bin slot. v2 (D-SPEC-91, v1.30.0) — additive extension: {total_meta_chains: int, total_introspect_picks: int, total_introspect_executions: int, monoculture_score: float, primitive_distribution: dict[str→float], last_chain_id: int, last_chain_reason: str, last_chain_succeeded: bool, subsystem_signals_status: dict, meta_cgn: dict, schema_version: int, ts: float}. meta_cgn = {status, graduation: {progress, rolled_back_count}, primitives_well_sampled, haov: {by_status: {confirmed}}, updates_applied, ready_to_graduate, primitive_v_summary: dict[str→float], failsafe: {status, last_check_ts, recent_failures}}. failsafe nested under meta_cgn matching MetaCGNConsumer.get_stats() native shape + /v4/meta-cgn/failsafe-status drill path. v1 readers tolerate missing meta_cgn key (defaults to empty dict). Owned by cognitive_worker (MetaReasoningEngine lives there). Reader: api_subprocess StateAccessor.spirit.get_coordinator overlay (meta_reasoning key) feeding /v4/meta-cgn + /v4/meta-cgn/failsafe-status + /v4/meta-cgn/graduation-readiness dashboard endpoints.
META_REASONING_STATE_SCHEMA_VERSION: Final[int] = 2
# Max bytes for meta_reasoning_state.bin payload. Typical ≈ 1-2KB (subsystem signals + primitive distribution). Bounded 4KB cap.
META_REASONING_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for consciousness_age.bin slot — variable msgpack {age_epochs: int, schema_version: int, ts: float}. Owned by cognitive_worker (Consciousness lives in spirit_loop under cognitive_worker per SPEC §1 glossary). G21 single-writer. Reader: api_subprocess + post_dispatch footer (canonical source for Titan's 'main age' — the fast cognitive self-observation tick counter, distinct from unified_spirit_metadata.epoch_count GreatEpoch counter). Per D-SPEC-85 v1.25.0.
CONSCIOUSNESS_AGE_SCHEMA_VERSION: Final[int] = 1
# Max bytes for consciousness_age.bin payload. Tiny dict (~30-50 bytes). Bounded 256B cap; fits trivially with msgpack overhead.
CONSCIOUSNESS_AGE_MAX_BYTES: Final[int] = 256
# Schema version for meta_teacher_state.bin slot — variable msgpack {total_critiques: int, voice_tuning_enabled: bool, peer_exchange_enabled: bool, last_critique_ts: float, per_domain_critiques: dict[str→int], adoption_rate: float, schema_version: int, ts: float}. Owned by cognitive_worker (MetaTeacherEngine instance). Reader: api_subprocess StateAccessor.meta_teacher.
META_TEACHER_STATE_SCHEMA_VERSION: Final[int] = 1
# Max bytes for meta_teacher_state.bin payload. Typical ≈ 600-1500 bytes. Bounded 4KB cap.
META_TEACHER_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for experience_stats.bin slot — variable msgpack {total_records: int, undistilled: int, total_wisdom: int, by_domain: dict[str→{count: int, avg_score: float, success_rate: float}], schema_version: int, ts: float}. Owned by cognitive_worker (ExperienceOrchestrator instance; G21 single-writer). Aggregated from the INCREMENTAL action_stats table (running avg/success maintained O(1) in _update_action_stats) + experience_records counts — NEVER a per-read GROUP BY. Reader: api_subprocess StateAccessor.experience + in-proc coord-snapshot build + neuromod_inputs_builder. §3L Phase 15 chunk 15.1 / D-SPEC-PHASE15: retires the frozen ExperienceMemory recompute-on-read path (spirit_worker write loop dropped in D8-3/72f95a6b).
EXPERIENCE_STATS_SCHEMA_VERSION: Final[int] = 1
# Max bytes for experience_stats.bin payload. by_domain has ~7-12 domains × ~80 bytes → typically 600-1000 bytes. Bounded 4KB cap.
EXPERIENCE_STATS_MAX_BYTES: Final[int] = 4096
# Schema version for guardian_state.bin slot — variable msgpack {modules: dict[str→{state: str, pid: int, rss_mb: float, uptime: float, restart_count: int, restarts_in_window: int, last_heartbeat_age: float, layer: str, start_method: str, adopted: bool, adopt_ts: float}], total_modules: int, modules_by_layer: dict[str→list[str]], escalation_count: int, schema_version: int, ts: float}. Owned by guardian (Python L1 supervisor — `titan_hcl/guardian.py`). Reader: api_subprocess StateAccessor.guardian.get_status + get_modules_by_layer.
GUARDIAN_STATE_SCHEMA_VERSION: Final[int] = 1
# Max bytes for guardian_state.bin payload. ~30-40 modules × ~200 bytes each → typically 8-10KB. Bounded 16KB cap accommodates fleet growth.
GUARDIAN_STATE_MAX_BYTES: Final[int] = 16384
# Schema version for llm_state.bin slot — variable msgpack {provider: str, model: str, total_completions: int, completions_this_hour: int, avg_latency_ms: float, p99_latency_ms: float, total_input_tokens: int, total_output_tokens: int, last_completion_ts: float, last_error: str, error_rate: float, schema_version: int, ts: float}. Owned by llm_worker (G21 single-writer). Reader: api_subprocess StateAccessor.llm.
LLM_STATE_SCHEMA_VERSION: Final[int] = 1
# Max bytes for llm_state.bin payload. Typical ≈ 600-900 bytes. Bounded 4KB cap.
LLM_STATE_MAX_BYTES: Final[int] = 4096
# Schema version for media_state.bin slot — variable msgpack {meditation_render_count: int, epoch_render_count: int, eureka_render_count: int, last_render_ts: float, last_render_type: str, total_disk_mb: float, nft_composite_count: int, schema_version: int, ts: float}. Owned by studio_worker (G21 single-writer). Distinct from existing studio_state.bin in that this slot carries media-pipeline counters consumed by dashboard's MediaAccessor (studio_state.bin is owned by StudioCoordinator with broader render lifecycle data). Reader: api_subprocess StateAccessor.media.
MEDIA_STATE_SCHEMA_VERSION: Final[int] = 1
# Max bytes for media_state.bin payload. Typical ≈ 400-700 bytes. Bounded 2KB cap.
MEDIA_STATE_MAX_BYTES: Final[int] = 2048
# Schema version for msl_state.bin slot — variable msgpack {synthesis_count: int, novel_associations: int, cross_modal_bindings: int, decay_rate: float, current_capacity: int, schema_version: int, ts: float}. Owned by cognitive_worker (MSL — Multisensory Synthesis Layer engine — lives there per SPEC §1 glossary preamble_extensions_pending). Reader: api_subprocess StateAccessor.spirit.get_coordinator overlay (msl key).
MSL_STATE_SCHEMA_VERSION: Final[int] = 1
# Max bytes for msl_state.bin payload. Typical ≈ 300-600 bytes. Bounded 2KB cap.
MSL_STATE_MAX_BYTES: Final[int] = 2048
# Schema version for network_state.bin slot — variable msgpack {balance_sol: float, pubkey: str, premium_rpc: str|None, rpc_urls: list[str], rpc_endpoint: str, recent_account_data: dict[str→dict], last_balance_update_ts: float, last_info_update_ts: float, network_available: bool, schema_version: int, ts: float}. Owned by network_state writer in titan_HCL kernel monitor_tick loop (G21 single-writer). Reader: api_subprocess StateAccessor.network. Closes the network.balance / network.info / network.account.* bus-cache state-lookups per Preamble G18. Note: this slot exposes the Solana RPC client's cached state, not the live RPC (each request to /v4/wallet etc. still hits the RPC if cache is stale).
NETWORK_STATE_SCHEMA_VERSION: Final[int] = 1
# Max bytes for network_state.bin payload. Typical ≈ 600-1200 bytes (depends on cached account_data entries). Bounded 4KB cap.
NETWORK_STATE_MAX_BYTES: Final[int] = 4096


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
# Maximum end-to-end elapsed time for Guardian.reload_module() to reach MODULE_RELOAD_ACK status=ready. Acceptance gate §4.6 #1. Sized to exceed L2 worker boot (2-8s) + adoption (~1-3s) + margin.
MODULE_RELOAD_HAPPY_PATH_S: Final[float] = 10.0
# Default caller-side timeout on Guardian.reload_module(). Larger than MODULE_RELOAD_HAPPY_PATH_S to allow rollback path to complete cleanly. Also bounds the §11.B.3 supervision-suppression window.
MODULE_RELOAD_DEFAULT_TIMEOUT_S: Final[float] = 30.0


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
# Monthly baseline rebase cadence — full snapshot on 1st of UTC month OR when chain depth cap hit (first-wins per SPEC §24.2)
BACKUP_BASELINE_CADENCE_DAYS: Final[int] = 30
# Max incrementals before forcing baseline rebase (chain-depth cap per SPEC §24.2; pairs with monthly date-trigger first-wins)
BACKUP_INCREMENTAL_MAX_CHAIN_DEPTH: Final[int] = 30
# Weekly mandatory restore-test (full byte-for-byte) per SPEC §24.12 — every Sunday after soul upload
BACKUP_RESTORE_TEST_CADENCE_DAYS: Final[int] = 7
# Per-tarball Arweave fetch + Merkle verification timeout during restore (mismatch emits BACKUP_MERKLE_MISMATCH P0)
BACKUP_VERIFY_MERKLE_TIMEOUT_S: Final[float] = 30.0
# Daily-incremental FAISS index ships full only when content_hash changes by >5% (signal of major retrain); otherwise FAISS ships only at weekly cadence per SPEC §24.5
BACKUP_FAISS_FULL_SHIP_DELTA_THRESHOLD_PCT: Final[float] = 5.0


# ── WORKER ────────────────────────────────────────────────────────────────
# Generic Python worker shutdown grace (used by all L2/L3 modules under guardian_HCL)

# Python L2/L3 module SIGTERM→SIGKILL grace
WORKER_SHUTDOWN_GRACE_S: Final[float] = 5.0


# ── OUTER ─────────────────────────────────────────────────────────────────
# Outer trinity daemon cadences (NOT Schumann-locked; per SPEC §18.1)

# Outer-body Python sensor sidecar source-refresh period; stale-threshold = 3× this (135s). Post-A.S8 the Rust daemon ticks at SCHUMANN_BODY_HZ (7.83 Hz) — this constant defines sidecar cadence + sensor staleness, NOT daemon tick rate. G13 body-slowest: 45s = strict 1:3:9 with mind 15s / spirit 5s; mirrors OUTER_BODY_BUS_PUBLISH_INTERVAL_S (45s) and the 5/15/45 dim counts (body 5D reads slowest). Was 10s (1:2:6 inverted) pre-D-SPEC-100.
OUTER_BODY_TICK_BASE_S: Final[float] = 45.0
# Outer-body sensor sidecar jitter (±). DEPRECATED for daemon tick — daemon now uses SchumannGenerator (no jitter).
OUTER_BODY_TICK_JITTER_PCT: Final[int] = 20
# Outer-body daemon bus publish throttle interval. Daemon ticks at Schumann body (7.83 Hz) but throttles MIND_STATE/SPIRIT_STATE bus publishes to this cadence. Body-slowest G13 invariant: this > OUTER_MIND_BUS_PUBLISH_INTERVAL_S > OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S.
OUTER_BODY_BUS_PUBLISH_INTERVAL_S: Final[float] = 45.0
# Outer-mind Python sensor sidecar source-refresh period; stale-threshold = 3× this (45s). Post-A.S8 the Rust daemon ticks at SCHUMANN_MIND_HZ (23.49 Hz) — this constant defines sidecar cadence + sensor staleness, NOT daemon tick rate. G13: 15s = strict 1:3:9 (spirit 5s / mind 15s / body 45s); mirrors OUTER_MIND_BUS_PUBLISH_INTERVAL_S (15s) and the 15D dim count. Was 5s pre-D-SPEC-100.
OUTER_MIND_TICK_BASE_S: Final[float] = 15.0
# Outer-mind sensor sidecar jitter (±). DEPRECATED for daemon tick.
OUTER_MIND_TICK_JITTER_PCT: Final[int] = 20
# Outer-mind daemon bus publish throttle interval. Daemon ticks at Schumann mind (23.49 Hz) but throttles MIND_STATE bus publishes to this cadence.
OUTER_MIND_BUS_PUBLISH_INTERVAL_S: Final[float] = 15.0
# Outer-spirit daemon bus publish throttle interval. Daemon ticks at Schumann spirit (70.47 Hz) but throttles SPIRIT_STATE / OUTER_SPIRIT_FILTER_DOWN bus publishes to this cadence. Spirit-fastest at bus layer (mirrors inner spirit publish rate).
OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S: Final[float] = 5.0
# Outer-spirit Python sensor sidecar source-refresh period; stale-threshold = 3× this (15s). Post-A.S8 the Rust daemon ticks at SCHUMANN_SPIRIT_HZ (70.47 Hz) — this constant defines sidecar cadence + sensor staleness, NOT daemon tick rate. G13 spirit-fastest: 5s = strict 1:3:9 (spirit 5s / mind 15s / body 45s); mirrors OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S (5s) and the 45D dim count (spirit carries the most dims and reads fastest). Was 30s (slowest — inverted) pre-D-SPEC-100. Load-safe: _gather_outer_sources reads in-process + bus-cached stats only; separate _heavy_stats_refresher owns DB/RPC cadence.
OUTER_SPIRIT_TICK_BASE_S: Final[float] = 5.0
# Outer-spirit sensor sidecar jitter (±). DEPRECATED for daemon tick.
OUTER_SPIRIT_TICK_JITTER_PCT: Final[int] = 10
# Start dim of outer_mind willing range (ground_up applies here per G10 ground_up_mind_range=10:15)
OUTER_MIND_WILLING_DIM_START: Final[int] = 10
# End dim (exclusive) of outer_mind willing range
OUTER_MIND_WILLING_DIM_END: Final[int] = 15
# Sensor cache staleness threshold = N × daemon's natural cadence (wall_ns < now − N×cadence → cache stale, daemon writes last-known with confidence=0.0 log)
OUTER_CACHE_STALE_CADENCE_MULTIPLIER: Final[int] = 3
# Max payload bytes for sensor_cache_outer_body.bin (msgpack source dict per O4 lock — agency_stats/helper_statuses/bus_stats/system_sensor_stats/network_monitor_stats/tx_latency_stats/block_delta_stats/anchor_state/sol_balance). Bumped 8192→65536 (v1.36.3): restores commit dd7e1d91's intent, which edited only the generated _phase_c_constants.py + constants.rs and was reverted by the next regen — Step 3 §4.3 P3 SOURCE_KEYS extension produces payloads up to ~33KB.
OUTER_SENSOR_CACHE_BODY_MAX_BYTES: Final[int] = 65536
# Max payload bytes for sensor_cache_outer_mind.bin (msgpack source dict — persona narrative + social context + creative_stats + memory_stats + agency_stats). Bumped 8192→65536 (v1.36.3): restores commit dd7e1d91's intent (lost on regen — generated-files-only edit).
OUTER_SENSOR_CACHE_MIND_MAX_BYTES: Final[int] = 65536
# Max payload bytes for sensor_cache_outer_spirit.bin (msgpack pre-aggregated outer-state — action_stats/sovereignty_ratio/uptime_ratio/social_stats/solana_stats/hormone_levels/recovery_stats/creative_stats/memory_stats/assessment_stats/history). Bumped 8192→65536 (v1.36.3): restores commit dd7e1d91's intent (lost on regen). Step 3 §4.3 P3 SOURCE_KEYS extension produces ~33KB payloads on T3; the 8192 cap silently rejected oversize writes, freezing outer_spirit dims on stale data.
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
# Maximum entries in dream_state_worker's _dream_inbox chat-during-dream buffer (deque maxlen). Matches the existing plugin.py:2270 429-error threshold ('Titan is dreaming and message queue is full (50)') preserved verbatim through the carve. Chat handlers (plugin + api_subprocess) check this cap before emitting DREAM_INBOX_ENQUEUE; queue-full → return standard 429 to client. Queue is volatile by design (worker crash forfeits messages, same as today's plugin._dream_inbox in-memory behavior).
DREAM_INBOX_MAX_ENTRIES: Final[int] = 50
# Upward-crossing threshold for FATIGUE_LEVEL_CRITICAL emission — when life_force_engine._metabolic_drain ≥ 0.7 (87.5% of 0.8 cap), life_force_worker emits the P1 single-shot event. Edge-debounced: re-emission requires drain to drop below LIFE_FORCE_FATIGUE_RESET (0.6) first, preventing threshold-edge oscillation. Publish-only per Maker Q6 2026-05-15; consumer wiring (cognitive_worker epoch-cadence reducer) deferred to follow-up rFP.
LIFE_FORCE_FATIGUE_THRESHOLD: Final[float] = 0.7
# Hysteresis reset threshold for FATIGUE_LEVEL_CRITICAL — once emitted at drain≥0.7, the single-shot edge re-arms only after drain falls back ≤0.6. Prevents oscillation when drain hovers near the threshold.
LIFE_FORCE_FATIGUE_RESET: Final[float] = 0.6
# Proportional drain-recovery factor applied to LifeForceEngine._metabolic_drain on every MEDITATION_COMPLETE event (drain *= 0.85 → ~15% reduction). Matches the *= 0.93 dreaming-recovery precedent at life_force.py:303 (7%/eval) in proportional shape. Maker Q4 lock 2026-05-15.
LIFE_FORCE_MEDITATION_RECOVERY_FACTOR: Final[float] = 0.85
# Max chat message length buffered into dream_state_worker's _dream_inbox during dream. Matches the existing plugin.py:2280 truncation (`message[:500]`) preserved verbatim through the carve. Chat handlers truncate before emitting DREAM_INBOX_ENQUEUE.
DREAM_INBOX_MAX_MESSAGE_CHARS: Final[int] = 500
# Cadence at which dream_state_worker republishes dream_state.bin SHM slot on every KERNEL_EPOCH_TICK (dual-trigger pattern: on tick + on DREAMING_STATE_UPDATED arrival per Maker Q6 greenlight 2026-05-15). Last_transition_ts field in the payload is the freshness probe — readers detect staleness if (time.time() - last_transition_ts) > DREAM_STATE_REPUBLISH_CADENCE_S * 5. Prevents readers from seeing infinite-stale SHM if cognitive_worker hangs (same staleness-detection pattern as metabolism_worker per D-SPEC-51).
DREAM_STATE_REPUBLISH_CADENCE_S: Final[float] = 1.0


# ── SOCIAL ────────────────────────────────────────────────────────────────
# L3 social-presence + X-posting cluster owned by social_worker L2 module (Phase C-S9 §4.C SHIPPED 2026-05-12 per PLAN_microkernel_phase_c_s9_social_worker_extraction.md). Houses SocialXGateway + SocialPressureMeter + ArchetypeDispatcher + PostDispatchOrchestrator + per-Titan polling-autonomy gates (canonical_poller_titan_id). Tick cadence + bus event priorities here decouple posting policy from underlying engine state (which flows via SHM).

# Cadence at which social_worker's PostDispatchOrchestrator drives one orchestration tick (build PostContext from SHM → optional delegate-rotation → F-phase pre-post → gateway.post → outcome emit → mention-cycle 30-min-gated → polling broadcasts on canonical poller). Matches legacy spirit_worker:7772 which gated the same block on `_msl_tick_count % 30 == 0` at 1 Hz tick (~30s). Configurable via [social_x].post_dispatch_tick_interval_seconds in titan_hcl/config.toml — read once at worker boot per `feedback_no_quick_patches_only_spec_correct_solutions` (set up front, not per-tick).
POST_DISPATCH_TICK_INTERVAL_SECONDS: Final[float] = 30.0
