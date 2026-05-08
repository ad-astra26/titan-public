"""
output_verifier_state_publisher — Phase C Session 3 §4.B.11.

Publishes output_verifier_state.bin from an OutputVerifier instance owned
by output_verifier_worker. Mirrors the `OUTPUT_VERIFIER_STATS` bus event
schema (output_verifier_worker.py:116-118 + 177-179) so consumers see
the same shape via SHM.

Schema per SPEC §7.1:
  { verified_count, rejected_count, sovereignty_score,
    threats_24h: { directive, injection, consistency, identity, qualia },
    recent_rejections_digest, ts }
"""
from __future__ import annotations

from typing import Any

from titan_plugin.logic.base_state_publisher import BaseStatePublisher
from titan_plugin.logic.session3_state_specs import (
    OUTPUT_VERIFIER_STATE_SLOT,
    OUTPUT_VERIFIER_STATE_SPEC,
)


class OutputVerifierStatePublisher(BaseStatePublisher):
    slot_name = OUTPUT_VERIFIER_STATE_SLOT
    slot_spec = OUTPUT_VERIFIER_STATE_SPEC

    def _compute_payload(self, verifier: Any) -> dict[str, Any]:
        import time
        if verifier is None:
            return self._cold_boot_payload()

        verified_count = int(getattr(verifier, "verified_count", 0) or 0)
        rejected_count = int(getattr(verifier, "rejected_count", 0) or 0)
        sovereignty_score = float(
            getattr(verifier, "sovereignty_score", 0.0) or 0.0)

        # threats_24h — best-effort (verifier may track aggregate or per-
        # category counts under different attribute names; defensive walk)
        threats: dict[str, int] = {
            "directive": 0,
            "injection": 0,
            "consistency": 0,
            "identity": 0,
            "qualia": 0,
        }
        try:
            t24 = getattr(verifier, "threats_24h", None)
            if isinstance(t24, dict):
                for k in threats.keys():
                    threats[k] = int(t24.get(k, 0) or 0)
            else:
                # Fallback: per-category counters as direct attrs
                for k in threats.keys():
                    val = getattr(verifier, f"{k}_violations_24h", None)
                    if val is not None:
                        threats[k] = int(val or 0)
        except Exception:
            pass

        # recent_rejections_digest — best-effort tail-list of recent
        # rejections (cap at 10 entries)
        recent_rejections: list[dict[str, Any]] = []
        try:
            recent = getattr(verifier, "_recent_rejections", None)
            if isinstance(recent, (list, tuple)):
                for entry in list(recent)[-10:]:
                    if isinstance(entry, dict):
                        recent_rejections.append({
                            "category": str(entry.get("category", "")),
                            "ts": float(entry.get("ts", 0.0) or 0.0),
                            "score": float(entry.get("score", 0.0) or 0.0),
                        })
        except Exception:
            pass

        return {
            "verified_count": verified_count,
            "rejected_count": rejected_count,
            "sovereignty_score": sovereignty_score,
            "threats_24h": threats,
            "recent_rejections_digest": recent_rejections,
            "ts": time.time(),
        }

    def _cold_boot_payload(self) -> dict[str, Any]:
        import time
        return {
            "verified_count": 0,
            "rejected_count": 0,
            "sovereignty_score": 0.0,
            "threats_24h": {
                "directive": 0, "injection": 0, "consistency": 0,
                "identity": 0, "qualia": 0,
            },
            "recent_rejections_digest": [],
            "ts": time.time(),
        }
