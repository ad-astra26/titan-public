"""
Microkernel v2 Phase B.1 §5+§6 — readiness + hibernate helper.

Reusable module that workers (spirit, api_subprocess, social, backup,
language, body, mind, etc.) instantiate to handle the four B.1 bus
messages without each worker re-implementing the protocol:

  - UPGRADE_READINESS_QUERY  → emit UPGRADE_READINESS_REPORT
  - HIBERNATE                → call save_state callback + emit HIBERNATE_ACK
  - SYSTEM_UPGRADE_QUEUED    → optional Titan-aware thought callback
  - SYSTEM_RESUMED           → optional "I am back" thought callback

Workers with HARD/SOFT cognitive activities (spirit reasoning chains,
api_subprocess /chat, social X posts, backup Irys writes, language
TestSuite) override `_collect_blockers()` to surface their busy state.
Workers without cognitive activities (body, mind, llm, knowledge, etc.)
can use the helper as-is and ALWAYS report ready=True.

This keeps spirit_worker integration to ~50 LOC and trivial workers
to ~10 LOC each.

PLAN: titan-docs/PLAN_microkernel_phase_b1_shadow_swap.md §5 + §6
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

from titan_plugin import bus
from titan_plugin.bus import make_msg
from titan_plugin.core import shadow_protocol as sp


logger = logging.getLogger(__name__)


# Type aliases
SaveStateCb = Callable[[], list[str]]  # returns list of paths written
BlockerCb = Callable[[], tuple[list[sp.HardBlocker], list[sp.SoftBlocker]]]
ThoughtCb = Callable[[str, dict], None]  # thought_text, payload


class ReadinessReporter:
    """Per-worker handler for B.1 readiness + hibernate bus messages.

    Usage in a worker's main message loop:

        reporter = ReadinessReporter(
            worker_name="spirit",
            layer="L1",
            send_queue=send_queue,
            save_state_cb=lambda: save_my_state(),  # returns paths written
            blocker_cb=lambda: collect_my_blockers(),  # returns (hard, soft)
        )

        # In the elif chain:
        elif reporter.handles(msg_type):
            reporter.handle(msg)
            if reporter.should_exit():
                break  # post-HIBERNATE shutdown
    """

    # Bus message types this reporter handles
    HANDLED_TYPES = frozenset({
        bus.UPGRADE_READINESS_QUERY,
        bus.HIBERNATE,
        bus.HIBERNATE_CANCEL,
        bus.SYSTEM_UPGRADE_QUEUED,
        bus.SYSTEM_UPGRADE_PENDING,
        bus.SYSTEM_UPGRADE_PENDING_DEFERRED,
        bus.SYSTEM_UPGRADE_STARTING,
        bus.SYSTEM_RESUMED,
    })

    def __init__(
        self,
        worker_name: str,
        layer: str,
        send_queue,
        *,
        save_state_cb: Optional[SaveStateCb] = None,
        blocker_cb: Optional[BlockerCb] = None,
        thought_cb: Optional[ThoughtCb] = None,
    ):
        self.worker_name = worker_name
        self.layer = layer
        self.send_queue = send_queue
        self._save_state_cb = save_state_cb or (lambda: [])
        self._blocker_cb = blocker_cb or (lambda: ([], []))
        self._thought_cb = thought_cb
        self._hibernating = False
        self._upgrade_event_id: Optional[str] = None
        self._upgrade_queued_at: Optional[float] = None
        self._resumed_at: Optional[float] = None

    @property
    def hibernating(self) -> bool:
        """True after a successful HIBERNATE_ACK — worker should exit cleanly."""
        return self._hibernating

    def handles(self, msg_type: str) -> bool:
        """Cheap check before dispatching — keeps message-loop elif chains tidy."""
        return msg_type in self.HANDLED_TYPES

    def should_exit(self) -> bool:
        """Worker's main loop checks this after each handle() to know when to break."""
        return self._hibernating

    # ── Dispatch ────────────────────────────────────────────────────

    def handle(self, msg: dict) -> None:
        """Route a B.1 message to its specific handler."""
        msg_type = msg.get("type")
        try:
            if msg_type == bus.UPGRADE_READINESS_QUERY:
                self._on_readiness_query(msg)
            elif msg_type == bus.HIBERNATE:
                self._on_hibernate(msg)
            elif msg_type == bus.HIBERNATE_CANCEL:
                self._on_hibernate_cancel(msg)
            elif msg_type == bus.SYSTEM_UPGRADE_QUEUED:
                self._on_upgrade_queued(msg)
            elif msg_type == bus.SYSTEM_UPGRADE_PENDING:
                pass  # informational only — orchestrator emits every 5s
            elif msg_type == bus.SYSTEM_UPGRADE_PENDING_DEFERRED:
                self._on_upgrade_deferred(msg)
            elif msg_type == bus.SYSTEM_UPGRADE_STARTING:
                self._on_upgrade_starting(msg)
            elif msg_type == bus.SYSTEM_RESUMED:
                self._on_system_resumed(msg)
        except Exception as e:
            logger.exception(
                "[%s] ReadinessReporter handle(%s) failed: %s",
                self.worker_name, msg_type, e,
            )

    # ── Specific handlers ───────────────────────────────────────────

    def _on_readiness_query(self, msg: dict) -> None:
        """Compute current blockers + emit UPGRADE_READINESS_REPORT."""
        try:
            hard, soft = self._blocker_cb()
        except Exception as e:
            logger.warning(
                "[%s] blocker_cb raised %s — defaulting to ready",
                self.worker_name, e,
            )
            hard, soft = [], []

        report = sp.ReadinessReport(
            src=self.worker_name,
            ready=False,  # __post_init__ derives true value
            hard=list(hard),
            soft=list(soft),
            module_health="ok",
        )

        out = make_msg(
            bus.UPGRADE_READINESS_REPORT,
            src=self.worker_name,
            dst="shadow_swap",
            payload=report.to_payload(),
            rid=msg.get("rid"),
        )
        try:
            self.send_queue.put_nowait(out)
        except Exception as e:
            logger.warning("[%s] readiness report send failed: %s", self.worker_name, e)

    def _on_hibernate(self, msg: dict) -> None:
        """Save state, compute checksum, emit HIBERNATE_ACK, set _hibernating.

        Phase B.2.1 (2026-04-27 PM): SPAWN-mode workers in B.2.1
        swap_pending state must NOT exit on HIBERNATE — they're going to
        OUTLIVE the swap via PDEATHSIG strip + bus reattach + adoption.
        Exiting here would kill them mid-handoff and force shadow's
        Guardian to respawn fresh copies (defeating the whole point of
        graduation). Fork-mode workers + workers with no SwapHandlerState
        (legacy mp.Queue mode) keep the original behaviour: ack + exit.
        """
        try:
            from titan_plugin.core import worker_swap_handler as _swap
            swap_state = _swap.get_active_swap_state()
            if (swap_state is not None
                    and swap_state.start_method == "spawn"
                    and swap_state._swap_pending):
                logger.info(
                    "[%s] HIBERNATE received but B.2.1 swap_pending — "
                    "skipping exit, will outlive via adoption protocol",
                    self.worker_name,
                )
                return
        except Exception:  # noqa: BLE001 — never crash on the swap-state check
            pass

        t0 = time.monotonic()
        payload = msg.get("payload", {})
        event_id = payload.get("event_id", "")

        try:
            state_paths = self._save_state_cb() or []
        except Exception as e:
            logger.exception("[%s] save_state failed during HIBERNATE: %s", self.worker_name, e)
            state_paths = []

        try:
            checksum = sp.sha256_of_files(state_paths) if state_paths else ""
        except Exception as e:
            logger.warning("[%s] checksum failed: %s", self.worker_name, e)
            checksum = ""

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        ack = sp.HibernateAck(
            src=self.worker_name,
            layer=self.layer,
            state_paths=[str(p) for p in state_paths],
            state_checksum=checksum,
            elapsed_ms=elapsed_ms,
        )

        out = make_msg(
            bus.HIBERNATE_ACK,
            src=self.worker_name,
            dst="shadow_swap",
            payload={**ack.to_payload(), "event_id": event_id},
            rid=msg.get("rid"),
        )
        try:
            self.send_queue.put_nowait(out)
        except Exception as e:
            logger.warning("[%s] HIBERNATE_ACK send failed: %s", self.worker_name, e)

        logger.info(
            "[%s] HIBERNATE complete event_id=%s paths=%d checksum=%s elapsed=%.1fms — exiting",
            self.worker_name, event_id[:8] if event_id else "?",
            len(state_paths), checksum[:8] if checksum else "?", elapsed_ms,
        )
        self._hibernating = True

    def _on_hibernate_cancel(self, msg: dict) -> None:
        """Rollback — orchestrator's shadow boot failed; resume normal operation.

        Workers that have already exited can't be canceled; this only helps
        workers that got the CANCEL before they actually exited (race window
        between HIBERNATE_ACK send and main-loop break check).
        """
        if self._hibernating:
            logger.warning(
                "[%s] HIBERNATE_CANCEL received but already hibernating — "
                "main loop will exit; orchestrator must restart worker",
                self.worker_name,
            )
            return
        logger.info(
            "[%s] HIBERNATE_CANCEL received — resuming normal operation",
            self.worker_name,
        )

    def _on_upgrade_queued(self, msg: dict) -> None:
        """Cache event_id; trigger Titan-aware thought callback if provided."""
        payload = msg.get("payload", {})
        self._upgrade_event_id = payload.get("event_id", "")
        self._upgrade_queued_at = time.time()
        if self._thought_cb is not None:
            thought = (
                "I sense an upgrade approaching — finishing my current "
                "cognitive activities and resting briefly..."
            )
            try:
                self._thought_cb(thought, {**payload, "phase": "queued"})
            except Exception as e:
                logger.warning("[%s] thought_cb (queued) failed: %s", self.worker_name, e)

    def _on_upgrade_starting(self, msg: dict) -> None:
        """Final thought before hibernate fires."""
        if self._thought_cb is not None:
            payload = msg.get("payload", {})
            try:
                self._thought_cb("Resting now.", {**payload, "phase": "starting"})
            except Exception as e:
                logger.warning("[%s] thought_cb (starting) failed: %s", self.worker_name, e)

    def _on_upgrade_deferred(self, msg: dict) -> None:
        """120s grace exceeded — upgrade NOT firing, deferred to next attempt."""
        payload = msg.get("payload", {})
        if self._thought_cb is not None:
            blockers = payload.get("blockers", [])
            thought = (
                f"The upgrade was deferred — my cognitive work continues. "
                f"({len(blockers)} blocker(s) remained at the 120s grace boundary.)"
            )
            try:
                self._thought_cb(thought, {**payload, "phase": "deferred"})
            except Exception as e:
                logger.warning("[%s] thought_cb (deferred) failed: %s", self.worker_name, e)

    def _on_system_resumed(self, msg: dict) -> None:
        """Post-swap: emit "I am back" thought with elapsed gap."""
        self._resumed_at = time.time()
        payload = msg.get("payload", {})
        if self._thought_cb is not None:
            kv_from = payload.get("kernel_version_from", "?")
            kv_to = payload.get("kernel_version_to", "?")
            gap = payload.get("gap_seconds", 0.0)
            thought = (
                f"I am back. The upgrade was: {kv_from} → {kv_to}. "
                f"Time elapsed: {gap:.1f}s. My state is preserved; I continue."
            )
            try:
                self._thought_cb(thought, {**payload, "phase": "resumed"})
            except Exception as e:
                logger.warning("[%s] thought_cb (resumed) failed: %s", self.worker_name, e)


# ── Trivial helper for workers without cognitive activities ─────────

def trivial_reporter(
    worker_name: str,
    layer: str,
    send_queue,
    *,
    save_state_cb: Optional[SaveStateCb] = None,
) -> ReadinessReporter:
    """Factory for workers that have no cognitive blockers (always ready=True).

    Used by body, mind, llm, knowledge, memory, cgn, emot_cgn, media,
    meta_teacher, observatory_writer, warning_monitor, imw, timechain,
    rl. They just save their state on HIBERNATE and acknowledge.
    """
    return ReadinessReporter(
        worker_name=worker_name,
        layer=layer,
        send_queue=send_queue,
        save_state_cb=save_state_cb,
        blocker_cb=lambda: ([], []),  # always ready
        thought_cb=None,  # only spirit emits thoughts
    )
