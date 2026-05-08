"""
rl_state_publisher — Phase C Session 3 §4.B.5.

Publishes rl_state.bin from rl_worker's `recorder` (SageRecorder) +
`gatekeeper` (SageGatekeeper) instances. Mirrors `_build_sage_stats()`
output (rl_worker.py:209-224) so consumers see the same schema as the
existing SAGE_STATS bus broadcast.

Source schema: { buffer_len, storage_len, buffer_size, sovereignty_score,
                 decision_history_len } + ts + extended fields where
available (training_loss_ema, transitions, last_train_ts).
"""
from __future__ import annotations

from typing import Any

from titan_plugin.logic.base_state_publisher import BaseStatePublisher
from titan_plugin.logic.session3_state_specs import (
    RL_STATE_SLOT,
    RL_STATE_SPEC,
)


class RLStatePublisher(BaseStatePublisher):
    slot_name = RL_STATE_SLOT
    slot_spec = RL_STATE_SPEC

    def _compute_payload(self, recorder: Any, gatekeeper: Any) -> dict[str, Any]:
        import time
        # Mirror _build_sage_stats from rl_worker.py for cross-consistency
        if recorder is None:
            buffer_len = 0
            storage_len = 0
            buffer_size = 0
        else:
            buf = getattr(recorder, "buffer", None)
            buffer_len = int(len(buf)) if buf is not None else 0
            storage = getattr(recorder, "storage", None)
            storage_len = int(len(storage)) if storage is not None else 0
            buffer_size = int(getattr(recorder, "buffer_size", 0) or 0)

        sovereignty_score = 0.0
        decision_history_len = 0
        if gatekeeper is not None:
            sovereignty_score = float(
                getattr(gatekeeper, "sovereignty_score", 0.0) or 0.0)
            decision_history_len = int(len(
                getattr(gatekeeper, "_decision_history", []) or []))

        # Optional extended fields (best-effort — not all rl_worker
        # variants expose these; defaults are SPEC-stable)
        last_train_ts = float(
            getattr(recorder, "last_train_ts", 0.0) or 0.0)
        training_loss_ema = float(
            getattr(recorder, "training_loss_ema", 0.0) or 0.0)
        transitions = int(
            getattr(recorder, "total_transitions", buffer_len) or buffer_len)

        return {
            "buffer_len": buffer_len,
            "storage_len": storage_len,
            "buffer_size": buffer_size,
            "sovereignty_score": sovereignty_score,
            "decision_history_len": decision_history_len,
            "training_loss_ema": training_loss_ema,
            "transitions": transitions,
            "last_train_ts": last_train_ts,
            "ts": time.time(),
        }
