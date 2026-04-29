"""Phase C drift-rename bridge aliases — DELETED IN C-S8.

Re-exports OLD legacy constant names → NEW canonical names from
`_phase_c_constants.py`. Per SPEC §3.0 Running-Titans Safety Rule:
existing modules MAY continue to import old names from here.
NEW modules MUST import canonical names from `_phase_c_constants.py`.

Coverage: drift items D01-D11/D13-D15/D18-D29 from
`titan-docs/SPEC_titan_architecture.md §3` whose canonical name has
landed in `_phase_c_constants.py` (auto-generated from
`SPEC_titan_architecture_constants.toml`). Drift items whose canonical
name is not yet TOML-sourced (e.g. SWAP_AUDIT_LOG_PATH, SWAP_ACTIVE_PORT_PATH,
BUS_P0_RESERVE_SLOTS) keep their legacy definition in the original module
unchanged — they will be aliased here once the TOML lands the canonical name
in a future C-S* chunk.

C-S8 deletes this module + bulk-renames all import sites in one commit.

Per PLAN_microkernel_phase_c_s2_kernel.md §12.4.
"""

from __future__ import annotations

from titan_plugin._phase_c_constants import (
    ADOPTION_TIMEOUT_S,
    AUTHKEY_BYTES,
    AUTHKEY_HKDF_SALT,
    BUS_ACCEPT_RATE_LIMIT_PER_S,
    BUS_API_HTTP_PORT_DEFAULT,
    BUS_PING_INTERVAL_S,
    BUS_PING_TIMEOUT_S,
    BUS_RECONNECT_BACKOFF_INITIAL_MS,
    BUS_RECONNECT_BACKOFF_MAX_S,
    BUS_RING_CAPACITY_SLOTS,
    BUS_SEND_FLUSH_TIMEOUT_S,
    BUS_SLOW_CONSUMER_DROP_RATE_RATIO,
    BUS_SLOW_CONSUMER_WARN_INTERVAL_S,
    FRAME_AUTH_TAG_BYTES,
    FRAME_CHALLENGE_BYTES,
    FRAME_LENGTH_PREFIX_BYTES,
    FRAME_MAX_FRAME_BYTES,
    GUARDIAN_HCL_MAX_STARVED_CYCLES,
    GUARDIAN_HCL_MIN_CPU_DELTA_S,
    GUARDIAN_HCL_REENABLE_COOLDOWN_S,
    GUARDIAN_HCL_SUSTAINED_UPTIME_RESET_S,
    MODULE_DEFAULT_RSS_LIMIT_MB,
    MODULE_HEARTBEAT_INTERVAL_S,
    MODULE_HEARTBEAT_TIMEOUT_S,
    REGISTRY_HEADER_BYTES,
    REGISTRY_HEADER_STRUCT,
    REGISTRY_MAX_READ_RETRIES,
    SUPERVISION_INTENSITY_WINDOW_S,
    SUPERVISION_MAX_RESTARTS,
    SUPERVISION_RESTART_BACKOFF_MAX_S,
)

# D01: guardian.py — restart backoff (legacy alias points to MAX_S cap; values in seconds)
RESTART_BACKOFF_BASE = SUPERVISION_RESTART_BACKOFF_MAX_S

# D02: guardian.py — restart-window counters (OTP rename: 600s legacy → 60s canonical)
MAX_RESTARTS_IN_WINDOW = SUPERVISION_MAX_RESTARTS
RESTART_WINDOW_SECONDS = SUPERVISION_INTENSITY_WINDOW_S

# D03: guardian.py — heartbeat
HEARTBEAT_INTERVAL = MODULE_HEARTBEAT_INTERVAL_S
HEARTBEAT_TIMEOUT = MODULE_HEARTBEAT_TIMEOUT_S

# D06: bus_authkey.py — HKDF derivation parameters
BUS_AUTHKEY_SALT = AUTHKEY_HKDF_SALT
BUS_AUTHKEY_LEN = AUTHKEY_BYTES

# D07: _frame.py — handshake byte sizes
CHALLENGE_SIZE = FRAME_CHALLENGE_BYTES
AUTH_TAG_SIZE = FRAME_AUTH_TAG_BYTES

# D09: bus_socket.py — ring capacity
DEFAULT_RING_CAPACITY = BUS_RING_CAPACITY_SLOTS

# D10: bus_socket.py — heartbeat
PING_INTERVAL_S = BUS_PING_INTERVAL_S
PING_TIMEOUT_S = BUS_PING_TIMEOUT_S

# D19: bus_socket.py — reconnect backoff (note ms → s conversion semantics)
RECONNECT_BACKOFF_BASE_S = BUS_RECONNECT_BACKOFF_INITIAL_MS / 1000.0
RECONNECT_BACKOFF_MAX_S = BUS_RECONNECT_BACKOFF_MAX_S

# D20: guardian.py — HCL CPU/starved cycles
MIN_CPU_DELTA_FOR_ALIVE = GUARDIAN_HCL_MIN_CPU_DELTA_S
MAX_STARVED_CYCLES = GUARDIAN_HCL_MAX_STARVED_CYCLES

# D21: bus_socket.py — accept rate limit
ACCEPT_RATE_LIMIT_PER_S = BUS_ACCEPT_RATE_LIMIT_PER_S

# D22: bus_socket.py — flush + slow-consumer thresholds
SEND_FLUSH_TIMEOUT_S = BUS_SEND_FLUSH_TIMEOUT_S
SLOW_CONSUMER_WARN_INTERVAL_S = BUS_SLOW_CONSUMER_WARN_INTERVAL_S
SLOW_CONSUMER_DROP_RATE_THRESHOLD = BUS_SLOW_CONSUMER_DROP_RATE_RATIO

# D23: _frame.py — wire frame byte sizes
LENGTH_PREFIX_SIZE = FRAME_LENGTH_PREFIX_BYTES
MAX_FRAME_SIZE = FRAME_MAX_FRAME_BYTES

# D24: state_registry.py — header layout
HEADER_SIZE = REGISTRY_HEADER_BYTES
HEADER_STRUCT = REGISTRY_HEADER_STRUCT
MAX_READ_RETRIES = REGISTRY_MAX_READ_RETRIES

# D25: worker_lifecycle.py — adoption supervision timeout
B2_1_DEFAULT_SUPERVISION_TIMEOUT_S = ADOPTION_TIMEOUT_S

# D26: guardian.py — sustained-uptime reset + re-enable cooldown
SUSTAINED_UPTIME_RESET = GUARDIAN_HCL_SUSTAINED_UPTIME_RESET_S
REENABLE_COOLDOWN_S = GUARDIAN_HCL_REENABLE_COOLDOWN_S

# D27: guardian.py — RSS limit default
DEFAULT_RSS_LIMIT_MB = MODULE_DEFAULT_RSS_LIMIT_MB

# D28: shadow_orchestrator.py — API port (only canonical so far; AUDIT_LOG_PATH +
# ACTIVE_PORT_PATH stay in legacy module until SPEC TOML lands them).
DEFAULT_API_PORT = BUS_API_HTTP_PORT_DEFAULT


__all__ = [
    "ACCEPT_RATE_LIMIT_PER_S",
    "AUTH_TAG_SIZE",
    "B2_1_DEFAULT_SUPERVISION_TIMEOUT_S",
    "BUS_AUTHKEY_LEN",
    "BUS_AUTHKEY_SALT",
    "CHALLENGE_SIZE",
    "DEFAULT_API_PORT",
    "DEFAULT_RING_CAPACITY",
    "DEFAULT_RSS_LIMIT_MB",
    "HEADER_SIZE",
    "HEADER_STRUCT",
    "HEARTBEAT_INTERVAL",
    "HEARTBEAT_TIMEOUT",
    "LENGTH_PREFIX_SIZE",
    "MAX_FRAME_SIZE",
    "MAX_READ_RETRIES",
    "MAX_RESTARTS_IN_WINDOW",
    "MAX_STARVED_CYCLES",
    "MIN_CPU_DELTA_FOR_ALIVE",
    "PING_INTERVAL_S",
    "PING_TIMEOUT_S",
    "RECONNECT_BACKOFF_BASE_S",
    "RECONNECT_BACKOFF_MAX_S",
    "REENABLE_COOLDOWN_S",
    "RESTART_BACKOFF_BASE",
    "RESTART_WINDOW_SECONDS",
    "SEND_FLUSH_TIMEOUT_S",
    "SLOW_CONSUMER_DROP_RATE_THRESHOLD",
    "SLOW_CONSUMER_WARN_INTERVAL_S",
    "SUSTAINED_UPTIME_RESET",
]
