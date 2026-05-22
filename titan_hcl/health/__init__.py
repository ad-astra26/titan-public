"""titan_hcl/health/ — pluggable L3 health-check plugin framework.

Per rFP_health_monitor_worker.md + SPEC v1.12.0 §9.B `health_monitor_worker`
block + D-SPEC-67 (2026-05-17).

This package holds:
  - The plugin CONTRACT (HealthResult dataclass + HealthCheckPlugin abstract
    base) in this __init__.py.
  - One CONCRETE PLUGIN per submodule (e.g. `social_x.py`).
  - health_monitor_worker discovers every concrete subclass at boot,
    filters by `applies_on` against this Titan's role, and schedules each
    plugin at its declared `cadence_s`.

SOLE sanctioned heal path = bus → owning worker.
  - `check()` runs in-proc inside health_monitor_worker (side-effect-free).
  - `heal()` returns only a (action, details) descriptor — never executes
    the action.
  - health_monitor_worker emits `HEAL_REQUEST(dst=plugin.owning_worker)`;
    the owning worker (e.g. social_worker for social_x) executes the action
    against its live in-proc state and replies `HEAL_RESULT`.

This preserves SOLE-sanctioned-X-path
(`feedback_social_x_gateway_post_is_sole_sanctioned_x_path.md`): health_monitor
never instantiates a second SocialXGateway with no cookie/session state.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Literal

# Default cap + cooldown values — plugins may override per-class.
DEFAULT_MAX_HEAL_ATTEMPTS_PER_24H = 6
DEFAULT_HEAL_COOLDOWN_AFTER_SUCCESS_S = 3600.0   # 1h
DEFAULT_HEAL_COOLDOWN_AFTER_FAILURE_S = 1800.0   # 30min

# Reply timeout for HEAL_REQUEST → HEAL_RESULT round-trip. Enforced by the
# dispatcher; on timeout the attempt is counted as a failure toward both
# daily-cap and consecutive-failure escalation.
HEALTH_HEAL_REPLY_TIMEOUT_S = 60.0

# Consecutive HEAL_RESULT(success=False) or timeouts that trigger
# HEALTH_HEAL_FAILED P1 escalation. Resets on first success.
HEALTH_HEAL_CONSECUTIVE_FAILURE_THRESHOLD = 3

# Per-check timeout (ThreadPoolExecutor.Future.result enforcement). Plugin
# contract: check() MUST be fast (<30s) + side-effect-free.
HEALTH_CHECK_TIMEOUT_S = 30.0


@dataclass
class HealthResult:
    """One outcome of a check() pass for a single (plugin, layer) tuple.

    Multiple HealthResults per check() pass are normal — one per layer
    (e.g. social_x emits pipeline + posting).
    """
    plugin: str
    layer: str
    status: Literal["OK", "DEGRADED", "DOWN"]
    reason: str
    ts: float = field(default_factory=time.time)
    details: dict = field(default_factory=dict)
    heal_recommended: bool = False

    def to_dict(self) -> dict:
        return {
            "plugin": self.plugin,
            "layer": self.layer,
            "status": self.status,
            "reason": self.reason,
            "ts": self.ts,
            "details": dict(self.details),
            "heal_recommended": self.heal_recommended,
        }


class HealthCheckPlugin(abc.ABC):
    """Abstract base for a subsystem health-monitor plugin.

    Subclasses MUST set class-level attributes `name`, `applies_on`,
    `owning_worker`, and (optionally override) cadence + cap + cooldown
    defaults, then implement `check()`. `heal()` is optional — default
    returns (None, {}) meaning "no heal recommended".

    See SPEC v1.12.0 §9.B `health_monitor_worker` block for the runtime
    contract the worker enforces around these methods.
    """

    # ── REQUIRED class attributes (subclass MUST set) ─────────────────
    name: str                                                   # unique
    applies_on: Literal["all", "canonical_poller", "mainnet_only"]
    owning_worker: str   # bus dst for HEAL_REQUEST (e.g. "social")

    # ── OPTIONAL — sensible defaults; override per-class ──────────────
    cadence_s: float = 14400.0                                  # 4h
    max_heal_attempts_per_24h: int = DEFAULT_MAX_HEAL_ATTEMPTS_PER_24H
    heal_cooldown_after_success_s: float = (
        DEFAULT_HEAL_COOLDOWN_AFTER_SUCCESS_S)
    heal_cooldown_after_failure_s: float = (
        DEFAULT_HEAL_COOLDOWN_AFTER_FAILURE_S)

    def __init__(self, config: dict | None = None) -> None:
        """Plugin instances are constructed once at worker boot by the
        HealthCheckRegistry. `config` is the full plugin config dict from
        titan_hcl/config.toml [health_monitor.<plugin_name>] section
        (may be empty). Subclasses MUST tolerate empty config."""
        self.config = dict(config or {})

    @abc.abstractmethod
    def check(self) -> list[HealthResult]:
        """One health-check pass. Returns 1+ HealthResults (typically one
        per layer). MUST be:

          - FAST (<30s — enforced via ThreadPoolExecutor timeout).
          - SIDE-EFFECT-FREE (no DB writes, no posts, no Arweave uploads).
          - SAFE to call from any thread.

        Network probes ARE permitted (e.g. GET to twitterapi.io). Reading
        SHM, reading local DB rows, reading state files = permitted.

        Returning an empty list is invalid — plugins MUST emit at least
        one HealthResult per call (use status=DOWN reason=exception when
        the check itself failed)."""
        raise NotImplementedError

    def heal(self, last_result: HealthResult
             ) -> tuple[str | None, dict]:
        """OPTIONAL self-heal hook. Returns (action_name, details).

        Return (None, {}) when no heal is recommended (e.g. the failure
        is not auto-recoverable, like a network outage). Return a non-None
        `action_name` when health_monitor_worker should emit a
        HEAL_REQUEST(dst=self.owning_worker) with that action and details.

        The owning worker dispatches by `action` to its own
        `_handle_heal_request(action, details) → (success, reason)`
        method. The plugin MUST NOT execute the action in-proc — this
        preserves SOLE-sanctioned-X-path and avoids cross-process
        session-state duplication.

        Default = no-heal."""
        return None, {}


__all__ = (
    "DEFAULT_MAX_HEAL_ATTEMPTS_PER_24H",
    "DEFAULT_HEAL_COOLDOWN_AFTER_SUCCESS_S",
    "DEFAULT_HEAL_COOLDOWN_AFTER_FAILURE_S",
    "HEALTH_HEAL_REPLY_TIMEOUT_S",
    "HEALTH_HEAL_CONSECUTIVE_FAILURE_THRESHOLD",
    "HEALTH_CHECK_TIMEOUT_S",
    "HealthResult",
    "HealthCheckPlugin",
)
