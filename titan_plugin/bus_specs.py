"""
bus_specs — declarative priority + coalesce table for every bus message type.

Microkernel v2 Phase B.2 §D6 (Maker-locked 2026-04-27): single source of truth
for how the bus_socket broker treats each message type under load. Replaces
ad-hoc per-emit configuration with one auditable table.

Two attributes per message type:

  • priority (int 0–3) — drives broker behavior under hard backpressure:
      P0  l0.* — clock ticks, guardian, identity, BUS_HANDOFF.
          NEVER drop. Broker reserves 64 ring slots exclusively for P0.
      P1  l1.* — Trinity tensors (BODY_STATE/MIND_STATE/SPIRIT_STATE),
          NS, neuromod. Drop OLDEST non-P0 of same type under pressure.
      P2  l2.* — reasoning, CGN, MSL, episodic. Drop oldest. (Default.)
      P3  l3.* — social, expression, OBSERVATORY_EVENT, LLM. Drop NEWEST
          under pressure (existing work gets priority to land; reject the
          newest publisher).

  • coalesce: tuple[str, ...] | None — when set, broker overwrites a pending
    message with the same coalesce-key tuple instead of appending. This is
    the GRACEFUL piece: state-update messages don't accumulate stale; the
    consumer always gets the freshest state. Default None (events preserved).

The table is consulted only by the bus_socket broker. mp.Queue mode (legacy,
default) is unaffected — get_spec returns DEFAULT_SPEC for unmapped types
with no behavioral change.

Phase C: this table maps cleanly to a Rust `phf::Map<&str, BusMsgSpec>` for
zero-runtime-cost lookups in the broker hot path.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BusMsgSpec:
    """Declarative spec for a bus message type.

    Frozen + hashable so specs are safe to share across threads and to use as
    dict keys if needed. Keep this dataclass small — every broker enqueue
    consults it on the hot path.
    """

    name: str
    priority: int = 2  # P2 default (drop-oldest under pressure)
    coalesce: tuple[str, ...] | None = None


# Default spec for any message type not explicitly listed.
DEFAULT_SPEC = BusMsgSpec(name="<default>", priority=2, coalesce=None)


# ── The canonical spec table ────────────────────────────────────────────────
# Populated incrementally; B.2 ships this initial set. Future PRs refine as
# data-driven decisions land (e.g., observed coalesce-replace rates).
#
# Keys MUST match the message type constants in titan_plugin/bus.py. The
# audit_against_bus_constants() function below verifies this at startup.

MSG_SPECS: dict[str, BusMsgSpec] = {
    # ── P0 — never drop (kernel-critical) ─────────────────────────────────
    # Lifecycle / clock / health.
    "EPOCH_TICK":         BusMsgSpec("EPOCH_TICK",         priority=0),
    "MODULE_HEARTBEAT":   BusMsgSpec("MODULE_HEARTBEAT",   priority=0),
    "MODULE_READY":       BusMsgSpec("MODULE_READY",       priority=0),
    "MODULE_SHUTDOWN":    BusMsgSpec("MODULE_SHUTDOWN",    priority=0),
    "MODULE_CRASHED":     BusMsgSpec("MODULE_CRASHED",     priority=0),
    # Bus protocol messages (B.2) — broker ↔ worker control plane.
    "BUS_SUBSCRIBE":      BusMsgSpec("BUS_SUBSCRIBE",      priority=0),
    "BUS_UNSUBSCRIBE":    BusMsgSpec("BUS_UNSUBSCRIBE",    priority=0),
    "BUS_PING":           BusMsgSpec("BUS_PING",           priority=0),
    "BUS_PONG":           BusMsgSpec("BUS_PONG",           priority=0),
    "BUS_SLOW_CONSUMER":  BusMsgSpec("BUS_SLOW_CONSUMER",  priority=0),
    "BUS_HANDOFF":        BusMsgSpec("BUS_HANDOFF",        priority=0),
    # Phase B.2.1 supervision-transfer protocol — kernel-critical, never-drop.
    "BUS_WORKER_ADOPT_REQUEST": BusMsgSpec("BUS_WORKER_ADOPT_REQUEST", priority=0),
    "BUS_WORKER_ADOPT_ACK":     BusMsgSpec("BUS_WORKER_ADOPT_ACK",     priority=0),
    "BUS_HANDOFF_CANCELED":     BusMsgSpec("BUS_HANDOFF_CANCELED",     priority=0),

    # ── P1 — Trinity state updates (coalesce by source+type) ───────────────
    # Body/Mind/Spirit emit at Schumann rate (7.83/23.49/70.47 Hz). Under
    # backpressure, freshest tensor wins; older same-type messages are
    # overwritten in place with no drop counter increment.
    "BODY_STATE":         BusMsgSpec("BODY_STATE",        priority=1, coalesce=("src", "type")),
    "MIND_STATE":         BusMsgSpec("MIND_STATE",        priority=1, coalesce=("src", "type")),
    "SPIRIT_STATE":       BusMsgSpec("SPIRIT_STATE",      priority=1, coalesce=("src", "type")),

    # ── P1 — observation / outer dispatch ─────────────────────────────────
    "OUTER_OBSERVATION":  BusMsgSpec("OUTER_OBSERVATION", priority=1, coalesce=("src", "type")),
    "OUTER_DISPATCH":     BusMsgSpec("OUTER_DISPATCH",    priority=1),

    # ── P3 — high-volume / drop-newest under pressure ─────────────────────
    # OBSERVATORY_EVENT can burst (websocket fan-out); under pressure we
    # prefer to keep older queued events flowing rather than crowd them out.
    "OBSERVATORY_EVENT":  BusMsgSpec("OBSERVATORY_EVENT", priority=3),

    # All other types fall through to DEFAULT_SPEC (P2, no coalesce).
    # That is the safe default for events: drop-oldest under pressure,
    # preserve event ordering otherwise.
}


# ── Public API ─────────────────────────────────────────────────────────────


def get_spec(msg_type: str) -> BusMsgSpec:
    """Look up the spec for a message type. Returns DEFAULT_SPEC if unmapped.

    Hot-path call — the broker calls this for every published message. Keep
    the lookup a single dict.get; do not add validation or normalization here.
    """
    return MSG_SPECS.get(msg_type, DEFAULT_SPEC)


def coalesce_key(spec: BusMsgSpec, msg: dict) -> tuple | None:
    """Compute the coalesce-key tuple for a message, given its spec.

    Returns None if spec.coalesce is None (no coalesce); otherwise a tuple
    formed by reading each named field from msg in order. Missing fields
    contribute None — never raises (broker hot path must not raise).

    Example:
      spec.coalesce = ("src", "type")
      msg = {"src": "body_worker", "type": "BODY_STATE", ...}
      → ("body_worker", "BODY_STATE")
    """
    if spec.coalesce is None:
        return None
    return tuple(msg.get(field) for field in spec.coalesce)


# ── Drift detection ────────────────────────────────────────────────────────


def audit_against_bus_constants() -> list[str]:
    """Verify every spec.name corresponds to a constant defined in bus.py.

    Returns a list of issues (empty list = clean). Called at kernel boot
    when bus_ipc_socket_enabled=true so any rename/typo surfaces immediately
    rather than silently dropping messages.

    Lazy-imports bus to avoid circularity (bus may import bus_specs in the
    future). bus.py constants are uppercase strings whose value matches the
    constant name (e.g., BODY_STATE = "BODY_STATE").
    """
    from titan_plugin import bus  # lazy

    issues: list[str] = []
    for spec_name in MSG_SPECS:
        # The constant exists in bus.py iff hasattr(bus, spec_name) and the
        # constant's value is the spec_name string.
        const_value = getattr(bus, spec_name, None)
        if const_value is None:
            issues.append(
                f"bus_specs.MSG_SPECS has '{spec_name}' but bus.py has no constant of that name"
            )
        elif const_value != spec_name:
            issues.append(
                f"bus.{spec_name} = {const_value!r} but bus_specs key is '{spec_name}' — drift"
            )
    return issues


def all_priorities_in_range() -> list[str]:
    """Check every spec's priority is in [0, 3]."""
    issues: list[str] = []
    for name, spec in MSG_SPECS.items():
        if not (0 <= spec.priority <= 3):
            issues.append(
                f"bus_specs.{name}: priority={spec.priority} out of [0, 3]"
            )
    return issues
