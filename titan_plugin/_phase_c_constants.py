"""
_phase_c_constants.py — AUTO-GENERATED from titan-docs/SPEC_titan_architecture_constants.toml.

DO NOT EDIT BY HAND. Edit the TOML, then run:
    python scripts/generate_phase_c_constants.py

SPEC version: 0.1.2
Source SHA-256: bdbd30e94aea257eff8c1d7f51207464d67770e61685d43f5de62e48dceb6e0e

Per SPEC §19 + §2.6: hand-editing this file is a SPEC violation flagged by
`arch_map phase-c verify`.
"""
from __future__ import annotations
from typing import Final

# ── SPEC version metadata ──────────────────────────────────────────────
SPEC_VERSION: Final[str] = "0.1.2"
SPEC_SOURCE_SHA256: Final[str] = "bdbd30e94aea257eff8c1d7f51207464d67770e61685d43f5de62e48dceb6e0e"


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
# Fast bus ring header layout: magic[8] + version[4] + read_idx[8] + write_idx[8] + mask[4] + reserved[32] = 64 bytes
FASTBUS_HEADER_BYTES: Final[int] = 64


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


# ── REGISTRY ──────────────────────────────────────────────────────────────
# Per-slot schema versions + registry lifecycle primitives

# unified_spirit_132d.bin slot schema version (Trinity 130D + Journey 2D intermediate before SELF assembly)
UNIFIED_SPIRIT_132D_SCHEMA_VERSION: Final[int] = 1
# SeqLock header size: seq(4) + schema(4) + wall_ns(8) + payload(4) + crc(4)
REGISTRY_HEADER_BYTES: Final[int] = 24
# Python struct format for header (LE: uint32, uint32, uint64, uint32, uint32)
REGISTRY_HEADER_STRUCT: Final[str] = "<IIQII"
# SeqLock reader retry budget on torn read
REGISTRY_MAX_READ_RETRIES: Final[int] = 3
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
# Schema version for epoch_counter.bin slot
EPOCH_COUNTER_SCHEMA_VERSION: Final[int] = 1
# Schema version for sphere_clocks.bin slot (6 × 7 fields)
SPHERE_CLOCKS_SCHEMA_VERSION: Final[int] = 1
# Schema version for chi_state.bin slot
CHI_STATE_SCHEMA_VERSION: Final[int] = 1
# Schema version for titanvm_registers.bin slot (11 NS programs × 4 fields)
TITANVM_REGISTERS_SCHEMA_VERSION: Final[int] = 1
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


# ── SCHUMANN ──────────────────────────────────────────────────────────────
# 7.83 / 23.49 / 70.47 Hz frequencies (locked by biology, NOT tunable)

# Schumann fundamental — body tick frequency (period 128 ms). LOCKED BY BIOLOGY (G13).
SCHUMANN_BODY_HZ: Final[float] = 7.83
# Schumann × 3 — mind tick frequency (period 43 ms). LOCKED BY BIOLOGY.
SCHUMANN_MIND_HZ: Final[float] = 23.49
# Schumann × 9 — spirit tick frequency (period 14 ms). LOCKED BY BIOLOGY.
SCHUMANN_SPIRIT_HZ: Final[float] = 70.47


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

# Outer-body daemon base tick cadence (Solana RPC + slow files)
OUTER_BODY_TICK_BASE_S: Final[float] = 10.0
# Outer-body cadence jitter (±)
OUTER_BODY_TICK_JITTER_PCT: Final[int] = 20
# Outer-mind daemon base tick cadence (persona, social — moderately fresh)
OUTER_MIND_TICK_BASE_S: Final[float] = 5.0
# Outer-mind cadence jitter (±)
OUTER_MIND_TICK_JITTER_PCT: Final[int] = 20
# Outer-spirit daemon base tick cadence (narrative — slow-changing)
OUTER_SPIRIT_TICK_BASE_S: Final[float] = 30.0
# Outer-spirit cadence jitter (±)
OUTER_SPIRIT_TICK_JITTER_PCT: Final[int] = 10
