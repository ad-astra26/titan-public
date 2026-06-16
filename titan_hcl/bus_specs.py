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
# Keys MUST match the message type constants in titan_hcl/bus.py. The
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
    # Microkernel v2 Phase B.2 §D9 (2026-05-02) — peer-process death signal
    # from broker to Guardian. Kernel-critical: must never drop, since it
    # triggers immediate restart for crashed workers.
    "BUS_PEER_DIED":            BusMsgSpec("BUS_PEER_DIED",            priority=0),

    # Phase C C-S7 (2026-05-05) — supervision messages emitted by Python
    # guardian + Rust supervisors. Per SPEC §8.1: P0 (never drop) for the
    # state-changing events; SUPERVISION_DEPENDENCY_DEGRADED is P1 (informational
    # only — soft dep failed but respawn continued).
    "SUPERVISION_CHILD_DOWN":          BusMsgSpec("SUPERVISION_CHILD_DOWN",          priority=0),
    "SUPERVISION_CHILD_RESTARTED":     BusMsgSpec("SUPERVISION_CHILD_RESTARTED",     priority=0),
    "SUPERVISION_ESCALATION":          BusMsgSpec("SUPERVISION_ESCALATION",          priority=0),
    "SUPERVISION_ESCALATION_RESPONSE": BusMsgSpec("SUPERVISION_ESCALATION_RESPONSE", priority=0),
    "SUPERVISION_DEPENDENCY_BLOCKED":  BusMsgSpec("SUPERVISION_DEPENDENCY_BLOCKED",  priority=0),
    "SUPERVISION_DEPENDENCY_RECOVERED": BusMsgSpec("SUPERVISION_DEPENDENCY_RECOVERED", priority=0),
    "SUPERVISION_DEPENDENCY_DEGRADED": BusMsgSpec("SUPERVISION_DEPENDENCY_DEGRADED", priority=1),

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

    # ── Track 2 SPEAK gating (v1.2.1) — SPEC §8.5 D-SPEC-38 ───────────────
    # outer_interface_worker ↔ cognitive_worker ↔ language_worker.
    # ADVISOR_REFRACTORY_STATE: per-titan refractory map, coalesce so freshest
    # state wins (rare emission — only on advisor cooldown change).
    # SPEAK_REQUEST_PENDING: cognitive_worker → outer_interface_worker
    # precursor — coalesce-by-request_id so if cognitive_worker re-emits before
    # outer_interface_worker has consumed, the freshest candidate-words win.
    # WORD_PERTURBATION_HINT: outer_interface_worker → language_worker — short
    # TTL (≤200ms enforced consumer-side). P2 drop-oldest is correct here
    # under pressure: the freshest hint is the one that matters within TTL;
    # stale queued hints would just get TTL-filtered downstream anyway.
    "ADVISOR_REFRACTORY_STATE": BusMsgSpec("ADVISOR_REFRACTORY_STATE", priority=1, coalesce=("titan_id",)),
    "SPEAK_REQUEST_PENDING":    BusMsgSpec("SPEAK_REQUEST_PENDING",    priority=2, coalesce=("request_id",)),
    "WORD_PERTURBATION_HINT":   BusMsgSpec("WORD_PERTURBATION_HINT",   priority=2),

    # Track 2 outer_interface_worker periodic stats publishers (v1.2.1).
    # Coalesce by titan_id so under backpressure the freshest snapshot wins
    # (drops nothing — old snapshot is overwritten in place by the new one).
    "OUTER_INTERFACE_STATS_UPDATED": BusMsgSpec("OUTER_INTERFACE_STATS_UPDATED", priority=2, coalesce=("titan_id",)),
    "KIN_SIGNATURE_UPDATED":         BusMsgSpec("KIN_SIGNATURE_UPDATED",         priority=2, coalesce=("titan_id",)),
    "KIN_SOCIETY_UPDATED":           BusMsgSpec("KIN_SOCIETY_UPDATED",           priority=2, coalesce=("titan_id",)),

    # Track 2 self_reflection_worker publishers (v1.2.1) — SPEC §9.B
    # self_reflection_worker Bus publications row. Cadence *_STATS_UPDATED
    # coalesce-by-titan_id; on-event insight + prediction events stay
    # ordered (P2 default, no coalesce).
    "SELF_REFLECTION_STATS_UPDATED": BusMsgSpec("SELF_REFLECTION_STATS_UPDATED", priority=2, coalesce=("titan_id",)),
    "SELF_REASONING_INSIGHT":        BusMsgSpec("SELF_REASONING_INSIGHT",        priority=2),
    "CODING_EXPLORER_STATS_UPDATED": BusMsgSpec("CODING_EXPLORER_STATS_UPDATED", priority=2, coalesce=("titan_id",)),
    "CODING_INSIGHT":                BusMsgSpec("CODING_INSIGHT",                priority=2),
    "PREDICTION_STATS_UPDATED":      BusMsgSpec("PREDICTION_STATS_UPDATED",      priority=2, coalesce=("titan_id",)),
    "PREDICTION_GENERATED":          BusMsgSpec("PREDICTION_GENERATED",          priority=2),

    # Phase C-S9 social_worker (rFP §4.C + PLAN §11.2). 12 new types — all
    # P3 (social tier), no coalesce (each is a distinct signal, not a
    # state update). See bus.py for direction comments per type.
    "KIN_SIGNAL":                       BusMsgSpec("KIN_SIGNAL",                       priority=3),
    "SOCIAL_RECEIVED":                  BusMsgSpec("SOCIAL_RECEIVED",                  priority=3),
    # ONE generic catalyst event (type in payload, see bus.py comment)
    "SOCIAL_CATALYST":                  BusMsgSpec("SOCIAL_CATALYST",                  priority=3),
    "X_POST_PUBLISHED":                 BusMsgSpec("X_POST_PUBLISHED",                 priority=3),
    "SOCIAL_GRAPH_UPDATE":              BusMsgSpec("SOCIAL_GRAPH_UPDATE",              priority=3),
    "MENTION_RECEIVED":                 BusMsgSpec("MENTION_RECEIVED",                 priority=3),
    "FELT_EXPERIENCE_CAPTURED":         BusMsgSpec("FELT_EXPERIENCE_CAPTURED",         priority=3),
    "ENGAGEMENT_SNAPSHOT_TAKEN":        BusMsgSpec("ENGAGEMENT_SNAPSHOT_TAKEN",        priority=3),
    "KNOWLEDGE_REUSE_HIT":              BusMsgSpec("KNOWLEDGE_REUSE_HIT",              priority=3),

    # rFP_titan_hcl_l2_separation_strategy §4.P + D-SPEC-50 (v1.7.1, 2026-05-14).
    # social_graph_worker publishers. READY is P1 one-shot lifecycle (never drop);
    # *_RECORDED events are P3 telemetry (drop-newest under load); STATS_UPDATED is
    # P3 with coalesce-by-type (latest-wins notification, bulk via SHM per
    # rFP_bus_payload_contracts §3.1).
    "SOCIAL_GRAPH_READY":               BusMsgSpec("SOCIAL_GRAPH_READY",               priority=1),
    "SOCIAL_GRAPH_STATS_UPDATED":       BusMsgSpec("SOCIAL_GRAPH_STATS_UPDATED",       priority=3, coalesce=("type",)),
    "SOCIAL_INTERACTION_RECORDED":      BusMsgSpec("SOCIAL_INTERACTION_RECORDED",      priority=3),
    "SOCIAL_DONATION_RECORDED":         BusMsgSpec("SOCIAL_DONATION_RECORDED",         priority=3),
    "SOCIAL_INSPIRATION_RECORDED":      BusMsgSpec("SOCIAL_INSPIRATION_RECORDED",      priority=3),

    # rFP_titan_hcl_l2_separation_strategy §4.J + D-SPEC-51 (v1.7.2, 2026-05-14).
    # metabolism_worker publishers. METABOLIC_TIER_CHANGED is P1 (every tier
    # transition is load-bearing for life_force_worker future + Soul mint gate
    # cache + dashboard). GATE_DECISION_RECORDED is P3 coalesce-by-feature
    # (latest-decision-per-feature; ring buffer is authoritative inside worker).
    # METABOLIC_STATS_UPDATED is P3 coalesce-by-type (1Hz notification; bulk via
    # metabolism_state.bin SHM slot per rFP_bus_payload_contracts §3.1).
    "METABOLIC_TIER_CHANGED":           BusMsgSpec("METABOLIC_TIER_CHANGED",           priority=1),
    "GATE_DECISION_RECORDED":           BusMsgSpec("GATE_DECISION_RECORDED",           priority=3, coalesce=("feature",)),
    "METABOLIC_STATS_UPDATED":          BusMsgSpec("METABOLIC_STATS_UPDATED",          priority=3, coalesce=("type",)),

    # Phase C rFP §3.4.1 Phase B (v1.3.4) — memory_worker event protocol
    # replacing work-RPC `add`. MEMORY_INGEST_REQUEST = producers → memory
    # worker (one-way, no rid). MEMORY_INGEST_COMPLETED = memory worker →
    # all (broadcast, filtered by request_id). Both P2, no coalesce — every
    # request is distinct work; every completion is a distinct ack with a
    # unique node_id payload.
    "MEMORY_INGEST_REQUEST":   BusMsgSpec("MEMORY_INGEST_REQUEST",   priority=2),
    "MEMORY_INGEST_COMPLETED": BusMsgSpec("MEMORY_INGEST_COMPLETED", priority=2),

    # rFP_titan_hcl_l2_separation_strategy §4.K + D-SPEC-57 (v1.8.3, 2026-05-15).
    # studio_worker publishers. READY is P1 one-shot lifecycle. RENDER_REQUEST +
    # RENDER_COMPLETED are P3 with coalesce-by-request_id (each render is a
    # distinct work unit; coalesce protects against accidental double-publish on
    # the same request_id by the same caller). Adopts D-SPEC-46 (memory_worker
    # Phase B) event-driven pattern for slow renders — ALL work-RPC paths stay
    # ≤5s per G19 strict; ZERO new phase_c_rpc_exemptions.yaml entries.
    "STUDIO_WORKER_READY":     BusMsgSpec("STUDIO_WORKER_READY",     priority=1),
    "STUDIO_RENDER_REQUEST":   BusMsgSpec("STUDIO_RENDER_REQUEST",   priority=3, coalesce=("request_id",)),
    "STUDIO_RENDER_COMPLETED": BusMsgSpec("STUDIO_RENDER_COMPLETED", priority=3, coalesce=("request_id",)),

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
    from titan_hcl import bus  # lazy

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
