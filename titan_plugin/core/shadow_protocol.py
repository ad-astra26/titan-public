"""
Microkernel v2 Phase B.1 — Shadow Core Swap protocol module.

State-machine support for the shadow-swap orchestrator (scripts/shadow_swap.py)
and worker hibernate/readiness handlers.

This module owns:
  - Dataclasses for the three protocol artifacts (ReadinessReport,
    HibernateAck, RuntimeSnapshot) + their helpers (HardBlocker, SoftBlocker)
  - msgpack-based serialization of RuntimeSnapshot (used by kernel.hibernate
    and titan_main.py --restore-from)
  - Compatibility verification (schema version + titan_id + module roster)
  - Constants: timeouts, grace period, snapshot path
  - event_id generation (UUID4 — links system fork ↔ episodic fork blocks
    per Q7 dual-write design)

It does NOT own:
  - Bus message types (those live in bus.py — see B.1 §1)
  - Hibernate execution logic (that lives in each worker — see B.1 §6)
  - Orchestration loop (scripts/shadow_swap.py — see B.1 §7)

rFP: titan-docs/rFP_microkernel_v2_shadow_core.md §347-357
PLAN: titan-docs/PLAN_microkernel_phase_b1_shadow_swap.md §2

Schema version 1 (initial). Bump on incompatible field changes; the
verify_compatible() check refuses to restore from a snapshot with a
different schema version (orchestrator falls back to clean restart).
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import msgpack


# ── Constants ───────────────────────────────────────────────────────

#: Schema version for RuntimeSnapshot. Bumping requires a fall-through
#: in verify_compatible() — old snapshots are refused, orchestrator does
#: clean-restart instead. Version 1 = initial B.1.
SNAPSHOT_SCHEMA_VERSION = 1

#: Default location for runtime snapshot files (overridable per-call).
#: Lives on /tmp because we want it gone after reboot — it's a transient
#: handoff between the hibernating kernel and the shadow boot.
DEFAULT_SNAPSHOT_DIR = Path("/tmp/titan/shadow")
DEFAULT_SNAPSHOT_NAME = "runtime.msgpack"

#: Auto-readiness wait — orchestrator polls every 1s for up to this many
#: seconds. Beyond grace, orchestrator publishes SYSTEM_UPGRADE_PENDING_DEFERRED
#: and exits with code 2. NO force-flag exists per Maker's design call —
#: forcing Titan defeats the cognitive-respect purpose.
READINESS_GRACE_SECONDS = 120.0
READINESS_POLL_INTERVAL = 1.0
READINESS_REPORT_TIMEOUT = 5.0  # per-poll wait for worker reports
READINESS_PENDING_NOTIFY_INTERVAL = 5.0  # how often we re-publish PENDING during wait

#: Per-layer hibernate-ack timeout. Tightened 2026-04-27 (Bonus #5):
#: workers typically ack in 1-5s in steady state; legacy 30s L3 timeout
#: was conservative for the 13-min broken swap pattern. After fix #1
#: (pause replaces stop_all) + fix #2 (proxy interlock prevents lock
#: contention), most workers ack <2s. Tighter timeouts make slow-ack
#: failures fail-fast instead of dragging the swap.
HIBERNATE_ACK_TIMEOUT_BY_LAYER: dict[str, float] = {
    "L0": 5.0,
    "L1": 5.0,
    "L2": 8.0,
    "L3": 10.0,
}

#: Shadow boot — how long we wait for new kernel's /health to come up
#: before declaring a boot timeout and rolling back via HIBERNATE_CANCEL.
#:
#: 2026-04-27 retuning: tried 30s (Bonus #7) but swap #5 confirmed it's
#: too aggressive — spirit (V4 consciousness imports + 5 state-paths
#: save/load) and memory (DuckDB + faiss imports) take longer than 30s
#: to reach state=running on a fresh shadow. 45s gives them headroom
#: while still failing-fast on degraded boots. 60s was the original
#: B.1 conservative choice; 45s was sweet spot pre-A.8.X.
#: 2026-04-28 PM raise to 120.0 — A.8.3/4/5/6/7 workers + autostart rl
#: (LazyMemmapStorage 3-5s) + 9 spawn-graduated workers add ~30-60s to
#: cold-boot health gate. T1 first attempt at 45s timed out at
#: shadow_boot phase with health_endpoint connection refused (shadow
#: process didn't bind port 7779 within window). 120s gives clean
#: headroom while still failing-fast on actually-broken boots.
SHADOW_BOOT_TIMEOUT = 120.0

#: Dream cycle special soft-blocker handling — wait this long for natural
#: completion before gentle wake (sets fatigue → wake threshold).
DREAM_MAX_WAIT_SECONDS = 180.0

#: Ping-pong port pair for shadow boot (Q6 confirmed). Orchestrator picks
#: whichever is currently free (i.e. the one not listed in active_api_port).
PING_PONG_PORTS = (7777, 7779)


# ── Blocker dataclasses ─────────────────────────────────────────────

@dataclass(frozen=True)
class HardBlocker:
    """A cognitive activity that MUST complete before hibernate can fire.

    Interruption causes data loss or breaks an external commitment
    (HTTP request mid-flight, LLM call mid-stream, backup mid-write).

    Fields:
      name: stable identifier matching the 10-cognitive-activity list
            (e.g. "x_post_in_flight", "chat_session_responding")
      eta_seconds: estimated seconds until this blocker clears (orchestrator
                   uses this to decide whether to keep waiting)
      since: timestamp the blocker started (for staleness detection)
    """
    name: str
    eta_seconds: float
    since: float

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "eta_seconds": float(self.eta_seconds), "since": float(self.since)}


@dataclass(frozen=True)
class SoftBlocker:
    """A cognitive activity that CAN be paused at a natural boundary.

    Examples: reasoning chain (commit/abandon at any step), dream cycle
    (wait for wake), expression render (wait for fire).

    Fields:
      name: stable identifier
      eta_seconds: estimated seconds until natural pause point
      metadata: optional dict with activity-specific detail (e.g.
                {"chain_id": 4521, "step": 3, "of": 7}) — surfaced to
                Maker UX for status-table refresh
    """
    name: str
    eta_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "eta_seconds": float(self.eta_seconds), "metadata": dict(self.metadata)}


# ── Readiness report ────────────────────────────────────────────────

@dataclass
class ReadinessReport:
    """Worker's response to UPGRADE_READINESS_QUERY.

    A worker is `ready=True` only when both lists are empty. Any HARD
    blocker forces wait. Any SOFT blocker ALSO forces wait (until its
    eta_seconds elapses, then the worker should self-pause and re-report
    ready).

    Fields:
      src: worker name (matches its bus subscriber name)
      ready: True iff hard==[] and soft==[]
      hard: HARD blockers (must wait for completion)
      soft: SOFT blockers (wait for natural pause)
      module_health: free-form one-word status ("ok", "degraded", "starting")
                     for orchestrator's decision-making (degraded modules
                     should not block — they may be the reason for the upgrade)
    """
    src: str
    ready: bool
    hard: list[HardBlocker] = field(default_factory=list)
    soft: list[SoftBlocker] = field(default_factory=list)
    module_health: str = "ok"

    def __post_init__(self) -> None:
        # Invariant: ready ⇔ no blockers
        computed_ready = (not self.hard) and (not self.soft)
        if self.ready != computed_ready:
            # Trust the lists; ready is derived
            object.__setattr__(self, "ready", computed_ready)

    def to_payload(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "ready": self.ready,
            "hard": [b.to_dict() for b in self.hard],
            "soft": [b.to_dict() for b in self.soft],
            "module_health": self.module_health,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ReadinessReport:
        hard = [HardBlocker(**b) for b in payload.get("hard", [])]
        soft = [SoftBlocker(**b) for b in payload.get("soft", [])]
        return cls(
            src=payload["src"],
            ready=payload.get("ready", False),
            hard=hard,
            soft=soft,
            module_health=payload.get("module_health", "ok"),
        )


# ── Hibernate ack ──────────────────────────────────────────────────

@dataclass
class HibernateAck:
    """Worker's response to HIBERNATE.

    Fields:
      src: worker name
      layer: "L0" | "L1" | "L2" | "L3" (per ModuleSpec.layer)
      state_paths: list of disk paths the worker wrote during hibernate
                   (orchestrator can checksum these for OBS-mkernel-hibernate-fidelity)
      state_checksum: SHA-256 hex of concatenated file contents (for the
                      hibernate-fidelity OBS gate — pre/post must match)
      elapsed_ms: how long the hibernate save took (for tuning per-layer timeouts)
    """
    src: str
    layer: str
    state_paths: list[str] = field(default_factory=list)
    state_checksum: str = ""
    elapsed_ms: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> HibernateAck:
        return cls(
            src=payload["src"],
            layer=payload["layer"],
            state_paths=list(payload.get("state_paths", [])),
            state_checksum=payload.get("state_checksum", ""),
            elapsed_ms=float(payload.get("elapsed_ms", 0.0)),
        )


# ── Runtime snapshot (kernel state across the swap) ─────────────────

@dataclass
class RuntimeSnapshot:
    """Kernel-side state serialized to disk during HIBERNATE phase.

    Used by the shadow boot to verify compatibility + restore continuity.
    /dev/shm/titan/*.bin files survive the swap separately (they're files
    in shared memory; new kernel re-opens via mmap). This snapshot only
    holds metadata that's NOT in /dev/shm.

    Fields:
      kernel_version: short identifier — git rev (first 8) of the kernel
                      that wrote this snapshot. The shadow kernel's version
                      is logged in SystemForkBlock for the upgrade record.
      soul_current_gen: SovereignSoul.current_gen at hibernate-time (must
                        match shadow's loaded value or upgrade refuses)
      titan_id: T1/T2/T3 identifier (refuses cross-Titan restore by mistake)
      registry_seqs: name → last-seq dict for each /dev/shm registry. Shadow
                     kernel verifies its mmap reads return seqs >= these
                     values (proves /dev/shm continuity).
      guardian_modules: list of module names registered before hibernate.
                        Shadow must register a superset (new modules ok,
                        removed modules force compat-check failure).
      bus_subscriber_count: informational — how many subscribers existed
                            before hibernate. Shadow workers re-subscribe
                            on boot; this lets us spot subscriber drops.
      written_at: hibernate timestamp (orchestrator uses for staleness
                  check — refuse if older than 5 minutes by default)
      schema_version: bump on incompatible changes. Snapshot from older
                      schema → compat-check fails → orchestrator does
                      clean-restart instead of shadow swap.
      event_id: UUID4 linking this swap's system-fork + episodic-fork
                TimeChain blocks (Q7 Option C dual-write). Same event_id
                appears in every per-worker HibernateAck and the
                SYSTEM_RESUMED published-by-shadow event.
    """
    kernel_version: str
    soul_current_gen: int
    titan_id: str
    registry_seqs: dict[str, int]
    guardian_modules: list[str]
    bus_subscriber_count: int
    written_at: float
    event_id: str
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Serialization ───────────────────────────────────────────────────

def default_snapshot_path(snapshot_dir: Path | str | None = None) -> Path:
    """Return the canonical snapshot path, ensuring its parent directory exists.

    Reads DEFAULT_SNAPSHOT_DIR lazily (not as a default-arg binding) so
    monkeypatched module-level constants take effect for tests.
    """
    if snapshot_dir is None:
        snapshot_dir = DEFAULT_SNAPSHOT_DIR
    p = Path(snapshot_dir) / DEFAULT_SNAPSHOT_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def serialize_snapshot(snap: RuntimeSnapshot, path: Path | str) -> Path:
    """Write `snap` to `path` atomically using msgpack.

    Atomic-rename so a torn write can't leave a half-snapshot that the
    shadow kernel would fail to deserialize (defaulting to clean restart
    is fine; corrupted half-snapshot is not).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = msgpack.packb(snap.to_dict(), use_bin_type=True)
    tmp.write_bytes(payload)
    tmp.replace(path)
    return path


def deserialize_snapshot(path: Path | str) -> RuntimeSnapshot:
    """Read a RuntimeSnapshot from disk.

    Raises FileNotFoundError, msgpack.exceptions.UnpackException, or
    KeyError if the file is missing/corrupt/wrong-shape — orchestrator
    catches all three and falls back to clean restart.
    """
    path = Path(path)
    raw = path.read_bytes()
    payload = msgpack.unpackb(raw, raw=False)
    return RuntimeSnapshot(
        kernel_version=payload["kernel_version"],
        soul_current_gen=int(payload["soul_current_gen"]),
        titan_id=payload["titan_id"],
        registry_seqs=dict(payload.get("registry_seqs", {})),
        guardian_modules=list(payload.get("guardian_modules", [])),
        bus_subscriber_count=int(payload.get("bus_subscriber_count", 0)),
        written_at=float(payload.get("written_at", 0.0)),
        event_id=str(payload.get("event_id", "")),
        schema_version=int(payload.get("schema_version", 1)),
    )


# ── Compatibility verification ──────────────────────────────────────

def verify_compatible(
    snap: RuntimeSnapshot,
    target_titan_id: str,
    target_modules: list[str],
    *,
    max_age_seconds: float = 300.0,
    now: float | None = None,
) -> tuple[bool, str]:
    """Check whether `snap` can safely be restored on this kernel.

    Returns (True, "ok") if compatible, else (False, "<reason>").
    Reasons surface to the orchestrator log for Maker review.

    Compat rules:
      1. Schema version must match (no migration support — bump = clean restart)
      2. titan_id must match (refuse cross-Titan restore — would crash soul)
      3. Snapshot must not be stale (default 5min — older means kernel
         was hibernated long ago, state on disk diverged)
      4. Target module roster must be a superset of snapshot's roster
         (new modules ok; removed modules = workers without targets,
         which would crash on boot)
    """
    now = now if now is not None else time.time()

    if snap.schema_version != SNAPSHOT_SCHEMA_VERSION:
        return (
            False,
            f"schema_version mismatch: snapshot={snap.schema_version} "
            f"target={SNAPSHOT_SCHEMA_VERSION}",
        )

    if snap.titan_id != target_titan_id:
        return (
            False,
            f"titan_id mismatch: snapshot={snap.titan_id!r} target={target_titan_id!r}",
        )

    age = now - snap.written_at
    if age > max_age_seconds:
        return (
            False,
            f"snapshot stale: age={age:.1f}s > max_age={max_age_seconds:.1f}s",
        )

    snap_modules = set(snap.guardian_modules)
    target_set = set(target_modules)
    missing_in_target = snap_modules - target_set
    if missing_in_target:
        return (
            False,
            f"target missing modules from snapshot: {sorted(missing_in_target)}",
        )

    return True, "ok"


# ── Event ID + checksum helpers ─────────────────────────────────────

def new_event_id() -> str:
    """Generate a fresh UUID4 string for an upgrade event.

    Used by orchestrator at queue time to link this swap's
    SystemForkBlock + EpisodicForkBlock + per-worker HibernateAck records.
    Format: 32-char hex (no dashes) for compactness in TimeChain.
    """
    return uuid.uuid4().hex


def sha256_of_files(paths: list[str | Path]) -> str:
    """Compute a stable SHA-256 over the contents of multiple files in order.

    Used by worker hibernate handlers to compute state_checksum for
    the OBS-mkernel-hibernate-fidelity gate (pre-swap and post-swap
    checksums must match — divergence means data integrity bug).

    Files are hashed in the given order. Missing files contribute their
    name + b"<MISSING>" so re-ordering / removal is detectable.
    """
    h = hashlib.sha256()
    for p in paths:
        path = Path(p)
        h.update(str(path).encode())
        h.update(b"\x00")
        if path.is_file():
            h.update(path.read_bytes())
        else:
            h.update(b"<MISSING>")
        h.update(b"\x00")
    return h.hexdigest()
