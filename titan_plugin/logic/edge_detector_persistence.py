"""Shared EdgeDetector persistence helpers.

Hoisted out of spirit_worker.py 2026-05-10 so cognitive_worker can use the
same canonical JSON checkpoint when the composite META-CGN EdgeDetector
and the P14 coherence EdgeDetector migrate from spirit_worker to
cognitive_worker (l0_rust_enabled=true path).

Persistence preserves "once per lifetime" / "once per threshold crossing"
semantics across worker restarts. Without this, Producer #1 (sphere_clock)
re-emitted milestones after each hot-restart (T1 observed 26 sphere
emissions vs the 16-per-lifetime budget across 4 restarts).

Single canonical JSON keyed by detector name. Saved on the existing 5-min
checkpoint cadence alongside NS SQLite backup. Loaded on detector init.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time

logger = logging.getLogger(__name__)


EDGE_DETECTOR_STATE_PATH = "./data/edge_detector_state.json"


def load_edge_detector_state(path: str = EDGE_DETECTOR_STATE_PATH) -> dict:
    """Read the persisted EdgeDetector state file. Returns {} on any error
    (fresh state) — fail-open is safe because missing state just means the
    producer will re-emit on first observation post-restart."""
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("schema_version") != 1:
            logger.warning(
                "[EdgeDetectorPersistence] Unknown schema version %s; ignoring",
                data.get("schema_version"))
            return {}
        return data.get("detectors", {}) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning("[EdgeDetectorPersistence] state load failed: %s", e)
        return {}


def save_edge_detector_state(detectors: dict,
                             path: str = EDGE_DETECTOR_STATE_PATH) -> None:
    """Atomically write EdgeDetector state (tmpfile + os.replace). Best-effort:
    WARN on failure because silent failure would hide a persistence gap.
    `detectors` is {name: EdgeDetector-instance}."""
    payload = {
        "schema_version": 1,
        "saved_at": time.time(),
        "detectors": {name: det.to_dict() for name, det in detectors.items()
                      if det is not None and hasattr(det, "to_dict")},
    }
    try:
        _dir = os.path.dirname(path) or "."
        if not os.path.isdir(_dir):
            os.makedirs(_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=_dir, prefix="edge_detector_state.", suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("[EdgeDetectorPersistence] state save failed: %s", e)
