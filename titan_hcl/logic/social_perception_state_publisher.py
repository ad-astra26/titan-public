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

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.session3_state_specs import (
    SOCIAL_PERCEPTION_STATE_SLOT,
    SOCIAL_PERCEPTION_STATE_SPEC,
)


class SocialPerceptionStatePublisher(BaseStatePublisher):
    slot_name = SOCIAL_PERCEPTION_STATE_SLOT
    slot_spec = SOCIAL_PERCEPTION_STATE_SPEC

    def _compute_payload(self, state_refs: dict[str, Any]) -> dict[str, Any]:
        """Compute social-perception fields from real felt_experiences data.

        Wave 4a Cat A fix (D-SPEC-89, 2026-05-18): the original `inner_state.
        observables` path was a dead end — InnerState's `observables` dict
        carries ObservableEngine output (coherence/magnitude/velocity/direction
        /polarity per body part) NOT sentiment fields. The sentiment field
        path was never wired → sentiment_ema=0 fleet-wide despite 100+ persona
        social sessions per Titan.

        Real source: `events_teacher.db.felt_experiences` table written by
        social_worker on each FELT_EXPERIENCE_CAPTURED event. Columns:
        sentiment, arousal, relevance, created_at, contagion_type.

        Compute:
        - sentiment_ema   = avg(sentiment) of felt_experiences in last 24h
        - interaction_rate = count of felt_experiences in last 24h / 24
        - social_activity = clamp(count_24h / 50, 1.0)
        - last_interaction_ts = max(created_at)

        Falls back to neutral defaults on DB-missing / no-rows.
        """
        import os
        import sqlite3
        import time

        sentiment_ema = 0.0
        interaction_rate = 0.0
        social_activity = 0.0
        last_interaction_ts = 0.0

        try:
            db_path = os.path.join(
                os.path.dirname(__file__), "..", "..",
                "data", "events_teacher.db")
            if os.path.exists(db_path):
                now = time.time()
                con = sqlite3.connect(db_path, timeout=2.0)
                try:
                    # Cascade through windows — use the freshest window that
                    # has data. Active Titans hit 24h; quiet ones use 7d/all.
                    # Rate calculations scale by the window denominator so
                    # comparable across windows.
                    for window_s, scale_h in (
                        (86400, 24.0),       # 24h
                        (3 * 86400, 72.0),   # 72h
                        (7 * 86400, 168.0),  # 7d
                        (0, 1.0),             # all-time (window_s=0)
                    ):
                        cutoff = now - window_s if window_s > 0 else 0.0
                        row = con.execute(
                            "SELECT AVG(sentiment), COUNT(*), MAX(created_at) "
                            "FROM felt_experiences WHERE created_at >= ?",
                            (cutoff,)
                        ).fetchone()
                        if row and row[1] and row[1] > 0:
                            raw_sentiment = float(row[0] or 0.0)
                            # Map sentiment ∈ [-1, 1] → [0, 1] for downstream
                            # consumers (mind_worker inner_touch expects [0,1]).
                            sentiment_ema = max(0.0,
                                                min(1.0,
                                                    (raw_sentiment + 1.0) / 2.0))
                            cnt = int(row[1])
                            interaction_rate = float(cnt) / scale_h
                            social_activity = min(1.0, cnt / 50.0)
                            last_interaction_ts = float(row[2] or 0.0)
                            break  # use first window that yields data
                finally:
                    con.close()
        except Exception:
            pass  # fall through to neutral defaults

        return {
            "sentiment_ema": sentiment_ema,
            "interaction_rate": interaction_rate,
            "social_activity": social_activity,
            "last_interaction_ts": last_interaction_ts,
            "ts": time.time(),
        }
