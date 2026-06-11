"""API handlers for `/v6/synthesis/metrics/*` (Phase 10 §P10.C, D-SPEC-PHASE10).

Backs the Observatory "Synthesis" panel: the headline sovereignty ratio,
groundedness heatmap, skill-library stats, retrieval p99 + chi compliance, and
chain-growth trend.

**Read source (INV-Syn-25, observation-only):** synthesis_worker (the metrics
aggregator) exports `data/synthesis_metrics_snapshot.json` atomically at the tail
of each 60s recompute pass. The api process reads this JSON only — never the
canonical stores.

Soft-fail contract: missing / stale / corrupt snapshot → 200 with
`{"ok": true, "snapshot": "missing|stale|corrupt", ...}`.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from fastapi import Request

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_NAME = "synthesis_metrics_snapshot.json"
SNAPSHOT_STALENESS_SECONDS = 600  # 10× the 60s recompute heartbeat

_SNAPSHOT_CACHE: dict[str, dict] = {}


def _resolve_snapshot_path() -> str:
    data_dir = os.environ.get("TITAN_DATA_DIR", "data")
    return os.path.join(data_dir, DEFAULT_SNAPSHOT_NAME)


def _load_snapshot(path: Optional[str] = None) -> Optional[dict]:
    if path is None:
        path = _resolve_snapshot_path()
    if not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached = _SNAPSHOT_CACHE.get(path)
    if cached is not None and cached.get("mtime") == mtime:
        return cached["data"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("[synthesis_metrics_handlers] snapshot parse failed (%s): %s", path, e)
        return None
    if not isinstance(data, dict):
        return None
    _SNAPSHOT_CACHE[path] = {"mtime": mtime, "data": data}
    return data


def _load_with_status() -> tuple[Optional[dict], str]:
    path = _resolve_snapshot_path()
    if not os.path.exists(path):
        return None, "missing"
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None, "missing"
    age = time.time() - mtime
    payload = _load_snapshot(path)
    if payload is None:
        return None, "corrupt"
    if age > SNAPSHOT_STALENESS_SECONDS:
        return payload, "stale"
    return payload, "ok"


def _sub(section: str, request: Request) -> dict:
    payload, status = _load_with_status()
    if payload is None:
        return {"ok": True, "snapshot": status, "ts": 0.0, section: {"available": False}}
    return {
        "ok": True,
        "snapshot": status,
        "ts": float(payload.get("ts") or 0.0),
        section: payload.get(section, {"available": False}),
    }


# ── GET handlers ──────────────────────────────────────────────────

async def get_v6_synthesis_metrics(request: Request):
    """Full metrics bundle."""
    payload, status = _load_with_status()
    if payload is None:
        return {"ok": True, "snapshot": status, "ts": 0.0, "metrics": {}}
    return {"ok": True, "snapshot": status, "ts": float(payload.get("ts") or 0.0),
            "metrics": payload}


async def get_v6_synthesis_metrics_sovereignty(request: Request):
    """Headline sovereignty ratio + windows + trend (B.6 readout)."""
    return _sub("sovereignty", request)


async def get_v6_synthesis_metrics_groundedness(request: Request):
    """Per-concept groundedness heatmap."""
    return _sub("groundedness", request)


async def get_v6_synthesis_metrics_retrieval(request: Request):
    """Retrieval p50/p95/p99 (B.4) + chi compliance (B.5)."""
    payload, status = _load_with_status()
    if payload is None:
        return {"ok": True, "snapshot": status, "ts": 0.0,
                "retrieval": {"available": False}, "chi": {"available": False}}
    return {
        "ok": True, "snapshot": status, "ts": float(payload.get("ts") or 0.0),
        "retrieval": payload.get("retrieval", {"available": False}),
        "chi": payload.get("chi", {"available": False}),
    }


async def get_v6_synthesis_metrics_chain_growth(request: Request):
    """Chain-growth bytes-per-fork + total (B.7 bounded-growth readout)."""
    return _sub("chain_growth", request)


# ── POST — Tier-2 explicit user feedback (INV-Syn-24 producer) ────

VALID_FEEDBACK_VERDICTS = ("positive", "negative")


async def post_v6_synthesis_feedback(request: Request):
    """POST /v6/synthesis/feedback — explicit Tier-2 user feedback (INV-Syn-24).

    Body: {tool_call_tx: str, verdict: "positive"|"negative", skill_id?: str}

    Publishes USER_FEEDBACK_SIGNAL{source:"explicit"} to the bus → synthesis_worker
    (sole consumer) applies the override via UserFeedbackOverride: patches
    scored_by="user" (supersedes oracle/llm, provenance-preserved) + adjusts the
    skill utility. Fire-and-forget (HTTP 202-style): returns {ok, accepted}.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    tool_call_tx = body.get("tool_call_tx")
    verdict = body.get("verdict")
    if not isinstance(tool_call_tx, str) or not tool_call_tx.strip():
        return {"ok": False, "error": "tool_call_tx_required"}
    if verdict not in VALID_FEEDBACK_VERDICTS:
        return {"ok": False, "error": "verdict_must_be_positive_or_negative"}

    from titan_hcl import bus
    from titan_hcl.bus import make_msg
    payload = {
        "tool_call_tx": tool_call_tx.strip(),
        "verdict": verdict,
        "source": "explicit",
        "skill_id": body.get("skill_id"),
        "ts": time.time(),
    }
    try:
        titan_state = getattr(request.app.state, "titan_state", None)
        if titan_state is None:
            titan_state = getattr(request.app.state, "titan", None)
        if titan_state is None:
            return {"ok": False, "error": "titan_state_unavailable"}
        titan_state.bus.publish(make_msg(
            bus.USER_FEEDBACK_SIGNAL, "dashboard", "synthesis", payload))
    except Exception as e:
        logger.warning("[synthesis_metrics_handlers] feedback publish failed: %s", e)
        return {"ok": False, "error": str(e)}
    return {"ok": True, "accepted": True}
