"""titan_hcl/errors.py — Phase 11 ModuleError typed-error envelope (D-SPEC-141 §11.I.4).

Single structured-error path between workers and the orchestrator/supervisor +
warning_monitor + observatory_worker. Replaces the historical "untyped exception
escaping + heartbeat_timeout + exit-code-only" failure surface that drove the
2026-05-27 py-spy-journal-grep firefighting cycle (per AUDIT_agno_chat_hang_diagnosis).

Per SPEC §11.I.4: every worker, helper, provider, tool that raises must either
(a) be wrapped in `@with_error_envelope(...)` (Chunk 11C) which auto-publishes
    `ModuleError(severity=ERROR, ...)` before re-raising, OR
(b) explicitly call `publish_module_error(bus_or_send_queue, ModuleError(...))`
    before raising / before returning a soft failure.

Uncaught exceptions escaping a worker entry function MUST be caught by the
spawn-side supervision watcher (titan_hcl) and converted to
`ModuleError(severity=FATAL, error_code=ModuleErrorCode.UNCAUGHT_EXCEPTION, ...)`
before `MODULE_CRASHED` is emitted by guardian_hcl.

Wire format: msgpack-serializable dict envelope on bus topic `MODULE_ERROR` (P1,
dst="all", non-blocking per §8.0.ter).
"""
from __future__ import annotations

import enum
import time
import traceback as _tb
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ── Limits (SPEC §11.I.4) ──────────────────────────────────────────────────────
MESSAGE_MAX_LEN: int = 200       # human-short summary
DETAIL_MAX_LEN: int = 1024       # full context payload
TRACEBACK_MAX_FRAMES: int = 10   # first 10 frames if exception caught


# ── Severity ──────────────────────────────────────────────────────────────────
class Severity(str, enum.Enum):
    """ModuleError severity levels per SPEC §11.I.4.

    guardian_hcl restart-strategy selector uses severity + error_code:
      FATAL + recoverable error_code   → MODULE_RESTART_REQUEST → titan_hcl
      FATAL + unrecoverable error_code → state=DISABLED + escalate (D4)
      ERROR (non-fatal)                → counted toward MAX_RESTARTS_IN_WINDOW
      WARN / INFO / DEBUG              → informational (no restart impact)
    """
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"


# ── Canonical error codes (registry — callers may also pass raw strings) ──────
class ModuleErrorCode(str, enum.Enum):
    """Canonical short-id registry for `error_code` field.

    Per SPEC §11.I.4 the field type is `str`, not this enum — callers may pass
    enum values (via `.value` or `str(enum_member)`) OR raw strings for codes
    not yet registered. Add new well-known codes here as the contract matures.
    """
    # Generic
    UNCAUGHT_EXCEPTION = "UNCAUGHT_EXCEPTION"
    BOOT_TIMEOUT = "BOOT_TIMEOUT"
    SHUTDOWN_GRACE_EXCEEDED = "SHUTDOWN_GRACE_EXCEEDED"

    # Probe (§11.I.3)
    PROBE_TIMEOUT = "PROBE_TIMEOUT"
    PROBE_FAILED = "PROBE_FAILED"
    PROBE_BUDGET_EXCEEDED = "PROBE_BUDGET_EXCEEDED"

    # Dependency (§11.G + §11.I.1)
    DEPENDENCY_NOT_READY = "DEPENDENCY_NOT_READY"
    DEPENDENCY_TIMEOUT = "DEPENDENCY_TIMEOUT"

    # Transport (§8.0.ter + §11.I.5)
    BUS_PUBLISH_FAILED = "BUS_PUBLISH_FAILED"
    BUS_RECV_QUEUE_FULL = "BUS_RECV_QUEUE_FULL"
    SHM_WRITE_FAILED = "SHM_WRITE_FAILED"
    SHM_READ_FAILED = "SHM_READ_FAILED"

    # Upstream LLM / external
    LLM_REQUEST_FAILED = "LLM_REQUEST_FAILED"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    EXTERNAL_SVC_UNAVAILABLE = "EXTERNAL_SVC_UNAVAILABLE"

    # OutputVerifier / TimeChain (agno hot path)
    OVG_TIMECHAIN_OPEN_FAILED = "OVG_TIMECHAIN_OPEN_FAILED"
    OVG_WARMUP_FAILED = "OVG_WARMUP_FAILED"
    TIMECHAIN_SCAN_TIMEOUT = "TIMECHAIN_SCAN_TIMEOUT"
    # Storage health — a DuckDB store degrading (e.g. ART-index churn slowdown,
    # the early-warning precursor to the actr_buffers FATAL crash-loop, D-SPEC-154).
    STORAGE_DEGRADED = "STORAGE_DEGRADED"


# ── Recoverability classification (RFP_supervision_lifecycle §7.F) ────────────
# The Severity docstring above specifies the guardian selector:
#   FATAL + recoverable   error_code → restart (retry may help)
#   FATAL + unrecoverable error_code → DISABLE immediately (restart is futile)
# This set makes that selector EXECUTABLE. An error_code is "unrecoverable" when
# respawning the SAME code from disk cannot plausibly fix it — the fault is in the
# module's contract/config/boot, not a transient runtime condition. Conservative
# by design (false-"recoverable" just means the crash-loop path disables it a few
# restarts later via §7.A; false-"unrecoverable" would disable a module that a
# restart could have saved). A FATAL code NOT in this set still disables on the
# *repeated* path (N FATAL in window) per the guardian gate — the crash-loop case.
UNRECOVERABLE_CODES: frozenset[str] = frozenset({
    ModuleErrorCode.BOOT_TIMEOUT.value,        # can't finish boot → restart re-loops the same boot
})


def is_unrecoverable(error_code: str) -> bool:
    """True iff a FATAL error with this code should DISABLE immediately rather
    than restart. Accepts a raw string or a ModuleErrorCode value. Unknown codes
    are treated as RECOVERABLE (fail-open to the restart path; the repeated-FATAL
    crash-loop gate still disables a genuinely-broken module)."""
    code = error_code.value if isinstance(error_code, ModuleErrorCode) else str(error_code)
    return code in UNRECOVERABLE_CODES


def _truncate(s: str, limit: int) -> str:
    """Truncate to `limit` chars, replacing the tail with ' …[truncated]' if cut."""
    if len(s) <= limit:
        return s
    suffix = " …[truncated]"
    return s[: max(0, limit - len(suffix))] + suffix


@dataclass(frozen=True)
class ModuleError:
    """Phase 11 typed-error envelope (SPEC §11.I.4 / RFP §3H.3).

    Fields:
      module_name           — producer module (e.g. "agno_worker")
      subsystem             — sub-area within module (e.g. "ovg.warmup", "llm.client")
      error_code            — canonical short id; prefer ModuleErrorCode values
      severity              — Severity enum
      message               — human-short, ≤200 chars (auto-truncated)
      detail                — full context, ≤1024 chars (auto-truncated)
      traceback_top10       — first 10 frames if exception caught (auto-trimmed)
      context               — arbitrary JSON-serializable dict
      suggested_remediation — operator hint, optional
      ts                    — wall-clock seconds (defaults to time.time())
      correlation_id        — bus correlation_id for request tracing, optional

    The dataclass is frozen for safety across IPC boundaries. The list / dict
    members are *references*, not deep-copied — callers should not mutate them
    after construction. Use `as_wire_dict()` to serialize for bus publication.
    """
    module_name: str
    subsystem: str
    error_code: str
    severity: Severity
    message: str
    detail: str = ""
    traceback_top10: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    suggested_remediation: Optional[str] = None
    ts: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None

    def __post_init__(self) -> None:
        # Enforce length limits per SPEC §11.I.4. Frozen → object.__setattr__.
        object.__setattr__(self, "message", _truncate(str(self.message), MESSAGE_MAX_LEN))
        object.__setattr__(self, "detail", _truncate(str(self.detail), DETAIL_MAX_LEN))
        if len(self.traceback_top10) > TRACEBACK_MAX_FRAMES:
            object.__setattr__(self, "traceback_top10", self.traceback_top10[:TRACEBACK_MAX_FRAMES])

    # ── Construction helpers ──────────────────────────────────────────────────
    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        module_name: str,
        subsystem: str,
        error_code: str | ModuleErrorCode = ModuleErrorCode.UNCAUGHT_EXCEPTION,
        severity: Severity = Severity.ERROR,
        message: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        suggested_remediation: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> "ModuleError":
        """Build envelope from a caught exception. Extracts top-10 traceback frames."""
        tb_lines = _tb.format_exception(type(exc), exc, exc.__traceback__)
        # Each traceback.format_exception entry can span multiple lines; keep them as-is
        # but cap the COUNT to TRACEBACK_MAX_FRAMES per SPEC.
        frames = tb_lines[-TRACEBACK_MAX_FRAMES:] if len(tb_lines) > TRACEBACK_MAX_FRAMES else list(tb_lines)
        return cls(
            module_name=module_name,
            subsystem=subsystem,
            error_code=str(error_code.value if isinstance(error_code, ModuleErrorCode) else error_code),
            severity=severity,
            message=message or f"{type(exc).__name__}: {exc}",
            detail="".join(tb_lines),
            traceback_top10=frames,
            context=context or {},
            suggested_remediation=suggested_remediation,
            correlation_id=correlation_id,
        )

    # ── Serialization ─────────────────────────────────────────────────────────
    def as_wire_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for msgpack/json serialization on the bus.

        Severity enum is converted to its string value; correlation_id and
        suggested_remediation pass through as None | str.
        """
        d = asdict(self)
        # asdict() preserves Severity as the StrEnum instance; ensure plain str on wire.
        d["severity"] = self.severity.value
        return d

    @classmethod
    def from_wire_dict(cls, d: dict[str, Any]) -> "ModuleError":
        """Reconstruct from a wire dict (e.g. received on bus subscription)."""
        return cls(
            module_name=d["module_name"],
            subsystem=d["subsystem"],
            error_code=d["error_code"],
            severity=Severity(d["severity"]),
            message=d["message"],
            detail=d.get("detail", ""),
            traceback_top10=list(d.get("traceback_top10", []) or []),
            context=dict(d.get("context", {}) or {}),
            suggested_remediation=d.get("suggested_remediation"),
            ts=float(d.get("ts", time.time())),
            correlation_id=d.get("correlation_id"),
        )
