"""social_x_state_publisher — Phase C-S9 §4.C / PLAN §2.5.

Publishes social_x_state.bin from social_worker context. Single writer per
G21 (only social_worker mutates this slot — confirmed in PLAN §0.5 +
SPEC §G1). Consumers:
  - /v4/social Observatory route (chunk 9H — reads SHM, returns JSON)
  - dim-live producers (ANANDA[11] willing/social_initiative + ANANDA[36]
    community_connection + ANANDA[38] expression_reach — Trinity 130D)

Schema (variable msgpack):
  {
    titan_id: str,                       # which Titan owns this slot
    current_urge: float,                 # SocialPressureMeter.urge_accumulator
    post_threshold: float,               # SocialPressureMeter.post_threshold
    posts_this_hour: int,                # rate-limit window state
    posts_today: int,                    # daily budget state
    next_allowed_post_ts: float,         # 0 if eligible now, else unix epoch
    catalysts_pending: int,              # len(meter.catalyst_events)
    last_archetype_fired: str,           # post_type of most recent verified post
    last_post_ts: float,                 # unix epoch of last verified post
    recent_posts: list[dict],            # last 5 posts {post_type, tweet_id, ts}
    is_canonical_poller: bool,           # canonical_poller_titan_id == titan_id
    boot_grace_remaining_s: float,       # 0 once boot grace expires
    ts: float,                           # publish timestamp (writer side)
  }

Inline spec (not bundled into session3_state_specs.py — this is a
session 9 slot owned by a different worker; future session-9 bundle
file would group social + per-Titan-polling slots).
"""
from __future__ import annotations

import sqlite3
import time
from typing import Any

import numpy as np

from titan_hcl._phase_c_constants import (
    SOCIAL_X_STATE_MAX_BYTES,
    SOCIAL_X_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec
from titan_hcl.logic.base_state_publisher import BaseStatePublisher


SOCIAL_X_STATE_SLOT = "social_x_state"

SOCIAL_X_STATE_SPEC = RegistrySpec(
    name=SOCIAL_X_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(SOCIAL_X_STATE_MAX_BYTES,),
    schema_version=SOCIAL_X_STATE_SCHEMA_VERSION,
    variable_size=True,
)


class SocialXStatePublisher(BaseStatePublisher):
    """Publishes social_x_state.bin at 1 Hz from social_worker.

    Reads from the worker's state_refs dict (gateway, meter, polling-mode
    flag) + the local social_x.db actions table for recent-posts list.
    """

    slot_name = SOCIAL_X_STATE_SLOT
    slot_spec = SOCIAL_X_STATE_SPEC

    def _compute_payload(self, state_refs: dict[str, Any]) -> dict[str, Any]:
        meter = state_refs.get("social_pressure_meter")
        gateway = state_refs.get("social_x_gateway")
        titan_id = state_refs.get("titan_id", "")
        is_canonical_poller = bool(state_refs.get("is_canonical_poller", False))
        boot_ts = float(state_refs.get("boot_ts", 0.0))

        # Default values for cold-boot / partial init.
        current_urge = 0.0
        post_threshold = 50.0
        posts_this_hour = 0
        posts_today = 0
        next_allowed_post_ts = 0.0
        catalysts_pending = 0

        if meter is not None:
            current_urge = float(getattr(meter, "urge_accumulator", 0.0) or 0.0)
            post_threshold = float(getattr(meter, "post_threshold", 50.0) or 50.0)
            posts_this_hour = int(getattr(meter, "posts_this_hour", 0) or 0)
            posts_today = int(getattr(meter, "posts_today", 0) or 0)
            last_post = float(getattr(meter, "_last_post_time", 0.0) or 0.0)
            min_interval = int(getattr(meter, "min_post_interval", 0) or 0)
            if last_post > 0 and min_interval > 0:
                eligible_at = last_post + min_interval
                if eligible_at > time.time():
                    next_allowed_post_ts = eligible_at
            catalysts_pending = len(getattr(meter, "catalyst_events", []) or [])

        # Boot-grace remaining: read from gateway if present; meter has
        # its own boot-settle (separate concept, see SocialPressureMeter
        # _BOOT_SETTLE_SECONDS).
        boot_grace_remaining_s = 0.0
        if gateway is not None and hasattr(gateway, "_boot_grace_remaining"):
            try:
                boot_grace_remaining_s = float(gateway._boot_grace_remaining())
            except Exception:
                pass

        # Last archetype + recent-posts list — read from local social_x.db.
        # Best-effort; cold-boot returns empty.
        last_archetype_fired = ""
        last_post_ts = 0.0
        recent_posts: list[dict] = []
        db_path = "./data/social_x.db"
        if gateway is not None and hasattr(gateway, "_db_path"):
            db_path = gateway._db_path
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
            try:
                rows = conn.execute(
                    "SELECT post_type, tweet_id, COALESCE(posted_at, created_at) "
                    "FROM actions WHERE action_type='post' "
                    "AND status IN ('verified','posted') AND titan_id=? "
                    "ORDER BY COALESCE(posted_at, created_at) DESC LIMIT 5",
                    (titan_id,),
                ).fetchall()
            finally:
                conn.close()
            recent_posts = [
                {"post_type": r[0] or "", "tweet_id": r[1] or "", "ts": float(r[2] or 0.0)}
                for r in rows
            ]
            if recent_posts:
                last_archetype_fired = recent_posts[0]["post_type"]
                last_post_ts = recent_posts[0]["ts"]
        except Exception:
            pass

        return {
            "titan_id": titan_id,
            "current_urge": current_urge,
            "post_threshold": post_threshold,
            "posts_this_hour": posts_this_hour,
            "posts_today": posts_today,
            "next_allowed_post_ts": next_allowed_post_ts,
            "catalysts_pending": catalysts_pending,
            "last_archetype_fired": last_archetype_fired,
            "last_post_ts": last_post_ts,
            "recent_posts": recent_posts,
            "is_canonical_poller": is_canonical_poller,
            "boot_grace_remaining_s": boot_grace_remaining_s,
            "boot_ts": boot_ts,
            "ts": time.time(),
        }
