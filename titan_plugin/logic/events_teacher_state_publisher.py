"""
events_teacher_state_publisher — Phase C Session 4 §4.B.7.

Publishes events_teacher_state.bin from EventsTeacher's JSON state file
+ EventsTeacherDB. Polled at 1Hz from inside language_worker (co-located
because EventsTeacher itself is cron-based and language_worker is the
educator-family supervised worker most likely already loading
events_teacher.py).

Schema per SPEC §7.1 events_teacher_state.bin:
    {
        fingerprints_count: int,
        last_run_time: float,
        window_count: int,
        perception_buffer_size: int,
        follower_rotation_idx: int,
        mode_stats: dict,
        felt_experiences: int,
        followers_tracked: int,
        windows_completed: int,
        ts: float,
    }

Resilience:
  - JSON file missing → cold-boot stub (zeros + ts), publisher does NOT
    raise. Common at boot before first cron run.
  - DB missing → felt_experiences/followers_tracked/windows_completed
    default to 0.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from titan_plugin.logic.base_state_publisher import BaseStatePublisher
from titan_plugin.logic.session4_state_specs import (
    EVENTS_TEACHER_STATE_SLOT,
    EVENTS_TEACHER_STATE_SPEC,
)


_DEFAULT_STATE_PATH = "data/events_teacher_state.json"
_DEFAULT_DB_PATH = "data/events_teacher.db"


class EventsTeacherStatePublisher(BaseStatePublisher):
    slot_name = EVENTS_TEACHER_STATE_SLOT
    slot_spec = EVENTS_TEACHER_STATE_SPEC

    def _compute_payload(
        self,
        titan_id: str,
        state_path: str = _DEFAULT_STATE_PATH,
        db_path: str = _DEFAULT_DB_PATH,
    ) -> dict[str, Any]:
        # JSON state read (best-effort)
        fingerprints_count = 0
        last_run_time = 0.0
        window_count = 0
        perception_buffer_size = 0
        follower_rotation_idx = 0
        mode_stats: dict = {}

        try:
            p = Path(state_path)
            if p.exists():
                state = json.loads(p.read_text())
                fingerprints_count = len(state.get("fingerprints", {}) or {})
                last_run_time = float(state.get("last_run_time", 0.0) or 0.0)
                window_count = int(state.get("window_count", 0) or 0)
                perception_buffer_size = len(
                    state.get("perception_buffer", []) or [])
                follower_rotation_idx = int(
                    state.get("follower_rotation_idx", 0) or 0)
                mode_stats = state.get("mode_stats", {}) or {}
        except Exception:
            # Cold-boot fallback — defaults already set
            pass

        # DB get_stats (best-effort — defaults if DB missing or unreachable)
        felt_experiences = 0
        followers_tracked = 0
        windows_completed = 0
        try:
            from titan_plugin.logic.events_teacher import EventsTeacherDB
            if Path(db_path).exists():
                db = EventsTeacherDB(db_path)
                db_stats = db.get_stats(titan_id)
                felt_experiences = int(
                    db_stats.get("felt_experiences", 0) or 0)
                followers_tracked = int(
                    db_stats.get("followers_tracked", 0) or 0)
                windows_completed = int(
                    db_stats.get("windows_completed", 0) or 0)
        except Exception:
            # Cold-boot fallback — defaults already set
            pass

        return {
            "fingerprints_count": fingerprints_count,
            "last_run_time": last_run_time,
            "window_count": window_count,
            "perception_buffer_size": perception_buffer_size,
            "follower_rotation_idx": follower_rotation_idx,
            "mode_stats": mode_stats,
            "felt_experiences": felt_experiences,
            "followers_tracked": followers_tracked,
            "windows_completed": windows_completed,
            "ts": time.time(),
        }
