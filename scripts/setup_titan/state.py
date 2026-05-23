"""Install-state marker — idempotency + resume across runs.

Lives at ~/.titan/install_state.json (alongside per-Titan config — see
reference_t2_t3_deploy_mechanic_kernel_binary_and_rename for the convention
that per-Titan runtime config lives in ~/.titan/, not in the repo).

Stores: schema version, the chosen mode, which phases have completed, the
timestamp of last action, the install_root path, and the version of
setup_titan that wrote it. Used by `setup_titan repair` to know what to
re-run, by `install --resume`, and by every phase to skip itself if already done.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

STATE_DIR = Path.home() / ".titan"
STATE_PATH = STATE_DIR / "install_state.json"
SCHEMA_VERSION = 1


def _default_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "setup_titan_version": None,   # filled by caller (avoids circular import)
        "mode": None,                  # mainnet | devnet | local
        "install_root": None,          # absolute path to the cloned titan repo
        "phases": {},                  # phase_name -> {status: done|failed, ts: epoch}
        "last_action": None,
        "last_action_ts": None,
    }


def load() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return _default_state()
    try:
        raw = json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        # corrupt or unreadable — start fresh rather than crash the wizard
        return _default_state()
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        # forward/back schema drift — keep the file as a sibling, return fresh
        STATE_PATH.rename(STATE_PATH.with_suffix(".incompatible.json"))
        return _default_state()
    # fill in any keys added since
    base = _default_state()
    base.update(raw)
    return base


def save(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["last_action_ts"] = int(time.time())
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, STATE_PATH)


def mark_phase(state: dict[str, Any], phase: str, status: str, *, action: str | None = None) -> None:
    """Record a phase result and persist."""
    state["phases"][phase] = {"status": status, "ts": int(time.time())}
    if action:
        state["last_action"] = action
    save(state)


def phase_done(state: dict[str, Any], phase: str) -> bool:
    p = state.get("phases", {}).get(phase)
    return bool(p) and p.get("status") == "done"


def exists() -> bool:
    return STATE_PATH.exists()
