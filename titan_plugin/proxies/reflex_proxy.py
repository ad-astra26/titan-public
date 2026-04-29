"""
Reflex Proxy — bus-routed bridge to reflex_worker (L3 §A.8.5).

Drop-in subclass of ReflexCollector. Inherits register_executor,
_executors, _cooldowns, reset_session, and the executor-side step
(_execute_selected). Overrides collect_and_fire so aggregation
(group → guardian-shield → combine → threshold + cooldown filter →
top-N) runs inside reflex_worker via bus.request(QUERY action="aggregate"),
while executors still run locally in parent (they reference parent
state — plugin.soul / plugin.metabolism / plugin.memory_proxy etc.
— and cannot trivially move to a subprocess).

When `microkernel.a8_reflex_subprocess_enabled=false` (default), parent
boots the regular ReflexCollector (no proxy) and behavior is byte-
identical to pre-A.8.5.

When the flag flips, parent boots ReflexProxy(ReflexCollector). Caller
contract is unchanged: agno_hooks._run_reflex_arc still calls
`await collector.collect_and_fire(signals, features, focus, trinity_state)`.

See: titan-docs/rFP_microkernel_phase_a8_l2_l3_residency_completion.md §A.8.5
"""
from __future__ import annotations

import logging
import time

from ..bus import DivineBus
from ..logic.reflexes import (
    PerceptualField,
    REFLEX_TYPE_MAP,
    ReflexCollector,
)

logger = logging.getLogger(__name__)


class ReflexProxy(ReflexCollector):
    """ReflexCollector subclass whose aggregation runs in reflex_worker.

    Constructor mirrors ReflexCollector(config: dict). Adds bus + timeout.
    """

    def __init__(self, bus: DivineBus, config: dict = None,
                 request_timeout_s: float = 2.0):
        super().__init__(config)
        self._bus = bus
        self._timeout = float(request_timeout_s)
        self._reply_queue = bus.subscribe("reflex_proxy", reply_only=True)

    async def collect_and_fire(
        self,
        signals: list[dict],
        stimulus_features: dict,
        focus_magnitude: float = 0.0,
        trinity_state: dict = None,
    ) -> PerceptualField:
        """Override: aggregation via bus, execution local."""
        start = time.time()
        field = PerceptualField(
            stimulus_features=stimulus_features,
            trinity_summary=trinity_state or {},
        )

        # Aggregation step delegated to worker. Parent owns cooldown
        # state (it writes them in _execute_selected on successful fire),
        # so we send the current cooldown snapshot in each request.
        payload = {
            "action": "aggregate",
            "signals": list(signals or []),
            "stimulus_features": dict(stimulus_features or {}),
            "focus_magnitude": float(focus_magnitude),
            "cooldowns": dict(self._cooldowns),
        }
        try:
            reply = self._bus.request(
                src="reflex_proxy",
                dst="reflex",
                payload=payload,
                timeout=self._timeout,
                reply_queue=self._reply_queue,
            )
        except Exception as e:
            logger.warning("[ReflexProxy] bus.request raised: %s", e)
            reply = None

        if reply is None:
            # Worker unavailable / timeout. No reflexes selected this turn —
            # parent's _run_reflex_arc handles empty PerceptualField (falls
            # back to state_register.format_minimal_state()).
            logger.warning(
                "[ReflexProxy] aggregate timeout (%.2fs) — no reflexes selected",
                self._timeout,
            )
            field.reflex_notices.append("reflex_aggregate_timeout")
            field.total_duration_ms = (time.time() - start) * 1000
            return field

        body = reply.get("payload") or {}
        if "error" in body:
            logger.warning("[ReflexProxy] worker reported error: %s", body["error"])
            field.reflex_notices.append(f"reflex_aggregate_error: {body['error']}")
            field.total_duration_ms = (time.time() - start) * 1000
            return field

        # Reconstruct selected list. Worker sends serial form
        # [{"reflex_type": str, "combined_confidence": float, "signals": list}]
        selected_serial = body.get("selected_serial") or []
        for note in body.get("notices") or []:
            field.reflex_notices.append(note)

        selected: list[tuple] = []
        for item in selected_serial:
            rt_name = item.get("reflex_type")
            rt = REFLEX_TYPE_MAP.get(rt_name)
            if rt is None:
                # Unknown reflex type from worker — should never happen if
                # both sides import the same enum. Skip with notice.
                field.reflex_notices.append(
                    f"reflex_unknown_type: {rt_name}")
                continue
            selected.append((
                rt,
                float(item.get("combined_confidence", 0.0)),
                list(item.get("signals") or []),
            ))

        if not selected:
            field.total_duration_ms = (time.time() - start) * 1000
            return field

        logger.info(
            "[ReflexProxy] Firing %d reflexes (worker-aggregated): %s",
            len(selected),
            ", ".join(f"{rt.value}({conf:.3f})" for rt, conf, _ in selected),
        )

        return await self._execute_selected(
            selected, field, stimulus_features, start,
        )
