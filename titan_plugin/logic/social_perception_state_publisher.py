"""
social_perception_state_publisher — Phase C Session 3 §4.B.4.

Publishes social_perception_state.bin from spirit_worker context.
Source: ``inner_state.observables`` or ``spirit_proxy._bus.request("get_social_perception_stats")``-style
data which lives inside spirit_loop's state_refs at runtime. Per
SPEC §23.13 producer 22, this is the per-titan social perception layer
sourced from inner_state observables (sentiment_ema, interaction_rate,
social_activity).

Owner per G21: spirit_worker (only). Replaces the
``spirit_proxy._bus.request(action="get_social_perception_stats")`` call
chain in `_gather_outer_sources` (plugin.py:3323) which is currently
sync-RPC and contributes to the broker pressure on T3.

Schema:
  { sentiment_ema, interaction_rate, social_activity,
    last_interaction_ts, ts }
"""
from __future__ import annotations

from typing import Any

from titan_plugin.logic.base_state_publisher import BaseStatePublisher
from titan_plugin.logic.session3_state_specs import (
    SOCIAL_PERCEPTION_STATE_SLOT,
    SOCIAL_PERCEPTION_STATE_SPEC,
)


class SocialPerceptionStatePublisher(BaseStatePublisher):
    slot_name = SOCIAL_PERCEPTION_STATE_SLOT
    slot_spec = SOCIAL_PERCEPTION_STATE_SPEC

    def _compute_payload(self, state_refs: dict[str, Any]) -> dict[str, Any]:
        """Extract social-perception fields from spirit_loop's state_refs.

        Source priority (each step is best-effort, falls through to next
        on missing/partial):
          1. ``inner_state.observables`` dict — owned by spirit_loop's
             InnerState container (canonical source per SPEC §23.13).
          2. Direct attrs on inner_state (`sentiment_ema`,
             `interaction_rate`, `social_activity`).
          3. Cold-boot stub with neutral defaults.
        """
        import time

        sentiment_ema = 0.0
        interaction_rate = 0.0
        social_activity = 0.0
        last_interaction_ts = 0.0

        if isinstance(state_refs, dict):
            inner = state_refs.get("inner_state")
            if inner is not None:
                # Try observables dict first
                obs = getattr(inner, "observables", None)
                if isinstance(obs, dict):
                    sentiment_ema = float(
                        obs.get("sentiment_ema", 0.0) or 0.0)
                    interaction_rate = float(
                        obs.get("interaction_rate", 0.0) or 0.0)
                    social_activity = float(
                        obs.get("social_activity", 0.0) or 0.0)
                    last_interaction_ts = float(
                        obs.get("last_interaction_ts", 0.0) or 0.0)
                # Direct attrs as fallback (only if observables dict
                # didn't supply the field)
                if sentiment_ema == 0.0:
                    sentiment_ema = float(
                        getattr(inner, "sentiment_ema", 0.0) or 0.0)
                if interaction_rate == 0.0:
                    interaction_rate = float(
                        getattr(inner, "interaction_rate", 0.0) or 0.0)
                if social_activity == 0.0:
                    social_activity = float(
                        getattr(inner, "social_activity", 0.0) or 0.0)
                if last_interaction_ts == 0.0:
                    last_interaction_ts = float(
                        getattr(inner, "last_interaction_ts", 0.0) or 0.0)

        return {
            "sentiment_ema": sentiment_ema,
            "interaction_rate": interaction_rate,
            "social_activity": social_activity,
            "last_interaction_ts": last_interaction_ts,
            "ts": time.time(),
        }
