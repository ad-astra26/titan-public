"""
reflex_state_publisher — Phase C Session 3 §4.B.10.

Publishes reflex_state.bin from a ``NeuralReflexNet`` instance owned by
``NeuralNervousSystem._reflex_net`` (lives inside spirit_worker per
SPEC §23.1 producer 15). Mirrors the per-reflex stats schema:

  { reflex_name: str → { fire_count, total_updates, last_loss,
                          fire_threshold } } + ts

Owner per G21: spirit_worker (only — instantiated alongside the existing
SpiritStatePublisher in the snapshot-builder thread).
"""
from __future__ import annotations

from typing import Any

from titan_plugin.logic.base_state_publisher import BaseStatePublisher
from titan_plugin.logic.session3_state_specs import (
    REFLEX_STATE_SLOT,
    REFLEX_STATE_SPEC,
)


class ReflexStatePublisher(BaseStatePublisher):
    slot_name = REFLEX_STATE_SLOT
    slot_spec = REFLEX_STATE_SPEC

    def _compute_payload(self, neural_nervous_system: Any) -> dict[str, Any]:
        import time
        reflexes: dict[str, dict[str, float]] = {}
        if neural_nervous_system is not None:
            try:
                # NeuralReflexNet instances may be exposed as `_reflex_net`
                # (single shared net across reflex types) OR as a dict of
                # per-reflex networks. Defensive: try both shapes.
                reflex_net = getattr(neural_nervous_system, "_reflex_net", None)
                reflex_dict = getattr(neural_nervous_system, "_reflexes", None)

                # Single-net case — get_stats returns aggregate or per-reflex
                if reflex_net is not None and hasattr(reflex_net, "get_stats"):
                    stats = reflex_net.get_stats() or {}
                    if isinstance(stats, dict):
                        # If get_stats returns per-reflex dict, use directly.
                        # Otherwise wrap as a single "default" entry.
                        if all(isinstance(v, dict) for v in stats.values()):
                            for n, s in stats.items():
                                reflexes[str(n)] = self._coerce_entry(s)
                        else:
                            reflexes["default"] = self._coerce_entry(stats)

                # Dict-of-nets case
                if isinstance(reflex_dict, dict):
                    for n, r in reflex_dict.items():
                        if hasattr(r, "get_stats"):
                            try:
                                s = r.get_stats() or {}
                                reflexes[str(n)] = self._coerce_entry(s)
                            except Exception:
                                continue
            except Exception:
                # Defensive — neural_nervous_system shape variance tolerated
                pass

        return {
            "reflexes": reflexes,
            "reflex_count": len(reflexes),
            "ts": time.time(),
        }

    @staticmethod
    def _coerce_entry(s: dict[str, Any]) -> dict[str, float]:
        return {
            "fire_count": int(s.get("fire_count", 0) or 0),
            "total_updates": int(s.get("total_updates", 0) or 0),
            "last_loss": float(s.get("last_loss", 0.0) or 0.0),
            "fire_threshold": float(s.get("fire_threshold", 0.0) or 0.0),
        }
