"""supervision — Python-side data model for SPEC §11 cross-language unification.

Mirrors Rust's `titan_core::supervisor` types so Python guardian's
SUPERVISION_* events have wire-compatible payloads with the Rust
supervisors (kernel + substrate + unified-spirit). Per SPEC §11.G.4 the
kernel writes a single supervision.jsonl from these events; payload
discipline matters.

Phase C C-S7 (2026-05-05) — initial cross-language unification per
PLAN_microkernel_phase_c_s7_activation_prep.md §2 Gap 11.
"""
from __future__ import annotations

import enum
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional


# ─── Reason classification (mirrors Rust SupervisionReason) ───────────


class SupervisionReason(str, enum.Enum):
    """Mirror of `titan_core::supervisor::types::SupervisionReason`.

    Wire format = canonical SCREAMING_SNAKE_CASE string. Used in:
      - Rolling reason buffer per child (last 16)
      - SUPERVISION_ESCALATION payload `most_common_reason`
      - data/supervision.jsonl `reason` field
    """

    OOM = "OOM"
    PANIC = "PANIC"
    SEGV = "SEGV"
    HANG = "HANG"
    EMPTY = "EMPTY"
    DEPENDENCY_BLOCKED = "DEPENDENCY_BLOCKED"
    CONFIG_ERROR = "CONFIG_ERROR"
    BOOT_FAILURE = "BOOT_FAILURE"
    CLEAN_EXIT = "CLEAN_EXIT"
    KILLED = "KILLED"
    OTHER = "OTHER"


def classify_exit_code(exit_code: Optional[int]) -> SupervisionReason:
    """Map a child's exit code to a `SupervisionReason`.

    Mirror of `titan_core::supervisor::restart::classify_exit`. Same
    canonical mapping per SPEC §15.
    """
    if exit_code is None:
        return SupervisionReason.KILLED
    if exit_code == 0:
        return SupervisionReason.CLEAN_EXIT
    if exit_code == 1:
        return SupervisionReason.PANIC
    if 2 <= exit_code <= 6:
        return SupervisionReason.CONFIG_ERROR
    if exit_code in (7, 8):
        return SupervisionReason.OTHER
    if exit_code == 137:  # 128 + 9 = SIGKILL
        return SupervisionReason.KILLED
    if exit_code == 139:  # 128 + 11 = SIGSEGV
        return SupervisionReason.SEGV
    if exit_code == 134:  # 128 + 6 = SIGABRT
        return SupervisionReason.SEGV
    if exit_code == 135:  # 128 + 7 = SIGBUS
        return SupervisionReason.SEGV
    if exit_code == 143:  # 128 + 15 = SIGTERM (graceful)
        return SupervisionReason.CLEAN_EXIT
    if 64 <= exit_code <= 127:
        return SupervisionReason.OTHER
    return SupervisionReason.OTHER


# ─── Escalation decision (mirrors Rust EscalationDecision) ─────────────


class EscalationDecision(str, enum.Enum):
    """Mirror of `titan_core::supervisor::types::EscalationDecision`.

    Wire format = canonical lowercase string for SUPERVISION_ESCALATION_RESPONSE
    payload `decision` field.
    """

    CONTINUE = "continue"
    TERMINATE = "terminate"
    HALT = "halt"


def kernel_default_decision(reason: SupervisionReason) -> EscalationDecision:
    """Mirror of `titan_core::supervisor::escalation::kernel_default_decision`.

    Per SPEC §11.B.2 — simplest viable policy that surfaces real cascades.
    Same mapping as the Rust kernel so Python guardian's in-process
    short-circuit reaches the same decision the Rust kernel would.
    """
    mapping = {
        SupervisionReason.OOM: EscalationDecision.TERMINATE,
        SupervisionReason.PANIC: EscalationDecision.TERMINATE,
        SupervisionReason.SEGV: EscalationDecision.TERMINATE,
        SupervisionReason.HANG: EscalationDecision.TERMINATE,
        SupervisionReason.EMPTY: EscalationDecision.HALT,
        SupervisionReason.DEPENDENCY_BLOCKED: EscalationDecision.CONTINUE,
        SupervisionReason.CONFIG_ERROR: EscalationDecision.HALT,
        SupervisionReason.BOOT_FAILURE: EscalationDecision.HALT,
        SupervisionReason.CLEAN_EXIT: EscalationDecision.CONTINUE,
        SupervisionReason.KILLED: EscalationDecision.TERMINATE,
        SupervisionReason.OTHER: EscalationDecision.TERMINATE,
    }
    return mapping[reason]


# ─── Dependency declaration (SPEC §11.G.1) ─────────────────────────────


class DependencyKind(str, enum.Enum):
    """Mirror of `titan_core::supervisor::types::DependencyKind`."""

    MODULE = "module"  # sibling Python L2/L3 module (must be RUNNING in guardian_HCL)
    BINARY = "binary"  # Rust binary (must be alive in supervisor child registry)
    SHM_SLOT = "shm_slot"  # /dev/shm/titan_<id>/<slot>.bin exists + populated + fresh
    EXTERNAL_SVC = "external_service"  # Solana RPC, X API, Ollama, etc.
    DB_FILE = "db_file"  # data/*.db exists + readable + schema OK
    ENDPOINT = "endpoint"  # HTTP 200 on URL within timeout


class DependencySeverity(str, enum.Enum):
    """Mirror of `titan_core::supervisor::types::DependencySeverity`."""

    CRITICAL = "critical"  # refuse to respawn; escalate to kernel
    SOFT = "soft"  # log warning; respawn anyway; module degrades


@dataclass(frozen=True)
class Dependency:
    """A declarative dependency for a Python L2/L3 module.

    Per SPEC §11.G.1 — used by guardian_HCL's pre-respawn dependency check
    (§11.G.2). When a child crashes, Guardian iterates `dependencies` and
    skips respawn if any critical dep is down (emits SUPERVISION_DEPENDENCY_BLOCKED
    instead). Soft deps log + continue with respawn.

    `check` is a callable returning True iff the dependency is currently
    healthy. Examples (declared per-module in titan_plugin/core/plugin.py):

        Dependency(
            name="x_api_reachable",
            kind=DependencyKind.EXTERNAL_SVC,
            severity=DependencySeverity.SOFT,
            check=lambda: requests.get("https://api.x.com/2/users/me",
                                       timeout=5.0).status_code == 200,
        )
    """

    name: str
    kind: DependencyKind
    severity: DependencySeverity
    check: Optional[Callable[[], bool]] = None


# ─── Per-child rolling reason buffer (SPEC §11.B step 3) ───────────────


@dataclass
class ReasonRecord:
    """One entry in a child's rolling reason buffer (last 16)."""

    reason: SupervisionReason
    detail: str  # capped 256 chars per SPEC
    ts: float = field(default_factory=time.time)
    exit_code: Optional[int] = None

    @staticmethod
    def make(
        reason: SupervisionReason,
        detail: str,
        exit_code: Optional[int] = None,
    ) -> "ReasonRecord":
        # Cap detail to 256 chars per SPEC §11.B step 2.
        if len(detail) > 256:
            detail = detail[:256]
        return ReasonRecord(
            reason=reason, detail=detail, ts=time.time(), exit_code=exit_code,
        )


def most_common_reason(buffer: deque) -> SupervisionReason:
    """Return the mode of a reason buffer, falling back to OTHER if empty.

    Used in SUPERVISION_ESCALATION payload's `most_common_reason` field per
    SPEC §11.B.1 step 1.
    """
    if not buffer:
        return SupervisionReason.OTHER
    counts: dict[SupervisionReason, int] = {}
    for record in buffer:
        counts[record.reason] = counts.get(record.reason, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


__all__ = [
    "SupervisionReason",
    "classify_exit_code",
    "EscalationDecision",
    "kernel_default_decision",
    "DependencyKind",
    "DependencySeverity",
    "Dependency",
    "ReasonRecord",
    "most_common_reason",
]
