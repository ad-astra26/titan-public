"""
body_state_publisher — Phase C Session 4 §4.B.6.

Publishes body_state.bin from body_worker's _collect_body_tensor() output
(the `details` half — the tensor itself lives in Rust-owned
inner_body_5d.bin).

Schema per SPEC §7.1 body_state.bin:
    {
        # 5DT body tensor (mirrored from Python compute; T3 readers
        # may prefer inner_body_5d.bin Rust slot, but this slot retains
        # the Python computation for T1+T2 + diagnostic comparison)
        interoception: float,
        proprioception: float,
        somatosensation: float,
        entropy: float,
        thermal: float,

        # Outer body context (sol balance + timechain) — stable reference
        # for body_proxy callers that want one-shot body context
        sol_balance: float,
        sol_norm: float,
        block_delta_norm: float,
        anchor_fresh: float,
        body_health: float,        # = mean(1 - severity_per_sense)

        # Per-sense urgency breakdown (the "details" payload from
        # _collect_body_tensor — what get_body_details actually wants)
        body_details: dict,

        ts: float,
    }
"""
from __future__ import annotations

import time
from typing import Any

from titan_plugin.logic.base_state_publisher import BaseStatePublisher
from titan_plugin.logic.session4_state_specs import (
    BODY_STATE_SLOT,
    BODY_STATE_SPEC,
)


class BodyStatePublisher(BaseStatePublisher):
    slot_name = BODY_STATE_SLOT
    slot_spec = BODY_STATE_SPEC

    def _compute_payload(self, body_state: Any) -> dict[str, Any]:
        """
        Args:
          body_state — dict produced by body_worker each tick:
            {
              "tensor": [5 floats],
              "details": {...},
              "history_size": {...},
              "severity_multipliers": [...],
              "focus_nudges": [...],
              "outer_context": {sol_balance, sol_norm, block_delta_norm,
                                anchor_fresh},  # optional
            }
        """
        if not isinstance(body_state, dict):
            return {
                "interoception": 0.5,
                "proprioception": 0.5,
                "somatosensation": 0.5,
                "entropy": 0.5,
                "thermal": 0.5,
                "sol_balance": 0.0,
                "sol_norm": 0.0,
                "block_delta_norm": 0.0,
                "anchor_fresh": 0.0,
                "body_health": 0.5,
                "body_details": {},
                "ts": time.time(),
            }

        tensor = body_state.get("tensor") or [0.5] * 5
        if len(tensor) < 5:
            tensor = list(tensor) + [0.5] * (5 - len(tensor))
        tensor = [float(x) for x in tensor[:5]]

        details = body_state.get("details") or {}
        # body_health = mean(1 - urgency) over present senses (cap at [0,1])
        urgencies = []
        for v in details.values():
            if isinstance(v, dict):
                u = v.get("urgency")
                if isinstance(u, (int, float)):
                    urgencies.append(float(u))
        if urgencies:
            mean_urgency = sum(urgencies) / len(urgencies)
            body_health = max(0.0, min(1.0, 1.0 - mean_urgency))
        else:
            body_health = 0.5

        outer = body_state.get("outer_context") or {}
        return {
            "interoception": tensor[0],
            "proprioception": tensor[1],
            "somatosensation": tensor[2],
            "entropy": tensor[3],
            "thermal": tensor[4],
            "sol_balance": float(outer.get("sol_balance", 0.0) or 0.0),
            "sol_norm": float(outer.get("sol_norm", 0.0) or 0.0),
            "block_delta_norm": float(
                outer.get("block_delta_norm", 0.0) or 0.0),
            "anchor_fresh": float(outer.get("anchor_fresh", 0.0) or 0.0),
            "body_health": body_health,
            "body_details": details,
            "ts": time.time(),
        }
