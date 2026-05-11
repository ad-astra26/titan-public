"""
api/pitch_chat.py — Wallet-less chat endpoint for the /v/<token>/pitch
VC + hackathon route.

Per rFP_observatory_pitch_route.md §5 + §11 (v2 locked 2026-05-11). This
is the backend the frontend Pitch page (Compare-mode default + single-
Titan toggle) calls. Each Titan instance serves only itself — Compare
mode is fan-out on the frontend across /t2/v4/pitch-chat and
/t3/v4/pitch-chat (nginx prefix routing).

Differences from the wallet-gated /chat endpoint:
  1. No Privy auth. Visitor identity is the X-Pitch-Token header +
     a client-generated thread_id. Synthetic claims are constructed
     `{"sub": f"pitch-visitor-{hash(thread_id)[:16]}"}` so the
     persona pipeline sees a stable per-thread visitor without us
     persisting any PII.
  2. Per-token + per-Titan rate limits (rFP §5).
  3. Server-side conversation recording to data/pitch_sessions/
     for Maker review (rFP §5.5 improvement #8).
  4. Response envelope adds "internal_time" metadata — epoch,
     dream phase, fatigue, mood — for the "I am here" timestamps
     (rFP §4 improvement #4).
  5. Educational failure modes (rFP §5.6 improvement #9): rate-limit
     and dream-phase rejections return a structured reason the
     frontend can render as a "why declined" card instead of a
     generic apology.

NOT implemented this session (deferred to v1.1 per rFP §10):
  - SSE streaming. v1 returns one JSON envelope per request; the
    frontend uses optimistic "Titan is thinking…" UX while it waits.
  - Compare-mode SSE fan-out at the backend. Compare is frontend-side.
  - /admin/pitch-sessions review route (recording is captured; the
    review UI ships next session).
  - Witness mode endpoint.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional, Deque, DefaultDict

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v4", tags=["pitch"])


# ── Rate limits (rFP §5) ──────────────────────────────────────────────
#
# Per-Titan, fleet-wide cap defends our daily LLM budget from a leaked
# token. Per-token cap defends from a single visitor monopolizing the
# fleet cap. Burst cap defends from accidental UI loops.
#
# These are intentionally generous for a hackathon — judges + early
# investors are the audience, not a public launch.
PITCH_BURST_LIMIT = 4         # messages in the last 60 seconds
PITCH_HOUR_LIMIT = 60         # messages in the last 60 minutes (per token)
PITCH_DAY_LIMIT = 200         # messages in the last 24 hours (per token)
PITCH_FLEET_DAY_LIMIT = 600   # messages per Titan per 24h, fleet-wide


# ── Recording (rFP §5.5) ──────────────────────────────────────────────
#
# data/pitch_sessions/<thread_id>.jsonl — one line per direction.
# No PII captured beyond the visitor's own typed messages (no IPs, no
# headers, no fingerprints). Retained 90 days then purged by an external
# cleanup script (TODO: cron).
PITCH_RECORDING_DIR = Path("data/pitch_sessions")


# ── In-process rate-limit state ───────────────────────────────────────
#
# Two ring buffers per token: short (last 60s) + medium (last hour).
# Long (24h) is computed on demand from the medium buffer's age.
# Restart resets the windows — acceptable for a hackathon demo route.
_token_timestamps: DefaultDict[str, Deque[float]] = defaultdict(deque)
_fleet_day_counter: Deque[float] = deque()  # this Titan's day counter
_rate_lock = asyncio.Lock()


# ── Pitch token validation ────────────────────────────────────────────


_PITCH_TOKEN_FILE = Path("data/pitch_token")


def _expected_pitch_token() -> Optional[str]:
    """Source of truth for the server-side pitch token. Read order:
      1. PITCH_TOKEN env var (preferred — set at titan_main startup).
      2. data/pitch_token file (single-line fallback so the token can
         be installed without restarting; gitignored).
    Minimum length 24 — anything shorter fails closed.
    """
    tok = os.environ.get("PITCH_TOKEN", "").strip()
    if not tok:
        try:
            if _PITCH_TOKEN_FILE.exists():
                tok = _PITCH_TOKEN_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            logger.debug("[PitchChat] pitch token file read failed", exc_info=True)
    if len(tok) < 24:
        return None
    return tok


def _constant_time_equal(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    diff = 0
    for x, y in zip(a, b):
        diff |= ord(x) ^ ord(y)
    return diff == 0


# ── Models ────────────────────────────────────────────────────────────


class PitchChatRequest(BaseModel):
    titan: str = Field(..., description="Titan id the visitor selected (T1/T2/T3); informational — each instance serves only itself.")
    thread_id: str = Field(..., min_length=8, max_length=64, description="Client-generated UUID-ish; scopes the conversation.")
    message: str = Field(..., min_length=1, max_length=500)


class InternalTime(BaseModel):
    """The "I am here" timestamp rendered next to every Titan reply
    (rFP §4 improvement #4). Demonstrates lived continuity."""
    epoch: Optional[int] = None
    phase: Optional[str] = None  # "awake" | "dreaming" | "meditating"
    fatigue: Optional[float] = None  # 0..1
    emotion: Optional[str] = None


class PitchChatResponse(BaseModel):
    response: str
    titan: str
    thread_id: str
    internal_time: InternalTime
    # When the reply was declined by the persona pipeline, an educational
    # "why" card is shown instead of a generic apology (rFP §5.6).
    declined: bool = False
    decline_reason: Optional[str] = None
    decline_explanation: Optional[str] = None


# ── Recording ─────────────────────────────────────────────────────────


def _safe_thread_id(s: str) -> str:
    # Defense in depth: only alnum + dash + underscore. Refuse anything
    # that could traverse the filesystem.
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", s):
        raise HTTPException(status_code=400, detail="invalid thread_id")
    return s


def _append_recording(thread_id: str, direction: str, payload: dict) -> None:
    try:
        PITCH_RECORDING_DIR.mkdir(parents=True, exist_ok=True)
        path = PITCH_RECORDING_DIR / f"{thread_id}.jsonl"
        line = {
            "ts": time.time(),
            "direction": direction,
            **payload,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        # Recording failure must never break the chat path.
        logger.exception("[PitchChat] recording write failed (thread=%s)", thread_id)


# ── Rate limiting ─────────────────────────────────────────────────────


def _prune_old(buf: Deque[float], cutoff: float) -> None:
    while buf and buf[0] < cutoff:
        buf.popleft()


async def _check_and_record_rate(token: str, now: float) -> Optional[dict]:
    async with _rate_lock:
        bucket = _token_timestamps[token]
        _prune_old(bucket, now - 86400.0)
        _prune_old(_fleet_day_counter, now - 86400.0)

        # Fleet day cap first — protects this Titan's budget.
        if len(_fleet_day_counter) >= PITCH_FLEET_DAY_LIMIT:
            return {"reason": "daily_cap_fleet", "limit": PITCH_FLEET_DAY_LIMIT}

        # Per-token day cap.
        if len(bucket) >= PITCH_DAY_LIMIT:
            return {"reason": "daily_cap_token", "limit": PITCH_DAY_LIMIT}

        # Per-token hourly + burst computed from the same bucket.
        hour_count = sum(1 for t in bucket if t > now - 3600.0)
        if hour_count >= PITCH_HOUR_LIMIT:
            return {"reason": "hour_cap_token", "limit": PITCH_HOUR_LIMIT}

        burst_count = sum(1 for t in bucket if t > now - 60.0)
        if burst_count >= PITCH_BURST_LIMIT:
            return {"reason": "burst_cap_token", "limit": PITCH_BURST_LIMIT}

        bucket.append(now)
        _fleet_day_counter.append(now)
        return None


# ── Internal-time snapshot ────────────────────────────────────────────


def _snapshot_internal_time(plugin, result_body: Optional[dict] = None) -> InternalTime:
    """Best-effort read of the Titan's current epoch + phase + fatigue +
    emotion. Failures yield empty fields rather than breaking the reply.

    Reads (in order, first non-empty wins per field):
      1. `result_body` from a successful run_chat — has mood + state_snapshot
         already populated by the pipeline.
      2. plugin._gather_current_state() if available — same source the
         pipeline uses internally.
      3. plugin._dream_state for fatigue / is_dreaming.
      4. epoch from plugin.kernel / plugin._epoch counter.
    """
    epoch = None
    phase = None
    fatigue = None
    emotion = None

    body = result_body or {}
    snap = body.get("state_snapshot") if isinstance(body, dict) else None
    if isinstance(snap, dict):
        if snap.get("is_dreaming"):
            phase = "dreaming"
        elif snap.get("emotion") in ("meditating", "meditation"):
            phase = "meditating"
        else:
            phase = "awake"
        em = snap.get("emotion")
        if isinstance(em, str) and em:
            emotion = em
    if isinstance(body, dict) and not emotion:
        mood = body.get("mood")
        if isinstance(mood, str) and mood:
            emotion = mood

    try:
        gather = getattr(plugin, "_gather_current_state", None)
        if callable(gather):
            state = gather()
            if phase is None:
                if state.get("is_dreaming"):
                    phase = "dreaming"
                else:
                    phase = "awake"
            if not emotion:
                em = state.get("emotion")
                if isinstance(em, str):
                    emotion = em
            f = state.get("fatigue")
            if isinstance(f, (int, float)):
                fatigue = float(max(0.0, min(1.0, f)))
    except Exception:
        logger.debug("[PitchChat] gather_current_state partial", exc_info=True)

    try:
        dream = getattr(plugin, "_dream_state", {}) or {}
        if fatigue is None:
            f = dream.get("fatigue")
            if isinstance(f, (int, float)):
                fatigue = float(max(0.0, min(1.0, f)))
        if phase is None:
            phase = "dreaming" if dream.get("is_dreaming") else "awake"
    except Exception:
        logger.debug("[PitchChat] dream_state partial", exc_info=True)

    try:
        for attr in ("_current_epoch", "_epoch", "epoch", "epoch_count"):
            v = getattr(plugin, attr, None)
            if callable(v):
                v = v()
            if isinstance(v, (int, float)):
                epoch = int(v)
                break
        if epoch is None:
            kernel = getattr(plugin, "kernel", None) or getattr(plugin, "_kernel", None)
            if kernel is not None:
                for attr in ("epoch", "current_epoch", "epoch_count"):
                    v = getattr(kernel, attr, None)
                    if callable(v):
                        v = v()
                    if isinstance(v, (int, float)):
                        epoch = int(v)
                        break
    except Exception:
        logger.debug("[PitchChat] epoch read partial", exc_info=True)

    return InternalTime(epoch=epoch, phase=phase, fatigue=fatigue, emotion=emotion)


# ── Decline classification ────────────────────────────────────────────


_DECLINE_PATTERNS = [
    ("jailbreak", re.compile(r"jailbreak|prompt.injection|policy.violation", re.I)),
    ("dream_phase", re.compile(r"dreaming|dream.phase|asleep", re.I)),
    ("persona_reject", re.compile(r"persona.rejected|off.policy|cannot.respond", re.I)),
]


def _classify_decline(body: dict) -> Optional[tuple[str, str]]:
    """If the chat pipeline declined, return (reason_code, explanation)
    or None if the response is a normal reply.

    Detection is heuristic — the pipeline doesn't expose a structured
    decline channel today (rFP §10 follow-up). For now we look at HTTP
    status + error/reason strings.
    """
    status = body.get("status_code") or 200
    if status == 200:
        return None
    body_inner = body.get("body") or {}
    err = (body_inner.get("error") or body_inner.get("reason") or "").strip()
    if status == 429:
        return ("budget_exhausted", err or "Titan has spoken extensively today.")
    if status == 503:
        return ("agent_unavailable", err or "Titan is currently unreachable.")
    for code, pat in _DECLINE_PATTERNS:
        if err and pat.search(err):
            return (code, err)
    return ("declined", err or f"declined (HTTP {status})")


# ── Endpoint ──────────────────────────────────────────────────────────


@router.post("/pitch-chat", response_model=PitchChatResponse)
async def pitch_chat(
    req: PitchChatRequest,
    request: Request,
    x_pitch_token: Optional[str] = Header(None, alias="X-Pitch-Token"),
):
    """Send a message to this Titan via the wallet-less pitch route.
    Validates the X-Pitch-Token against PITCH_TOKEN (server-side env);
    bad tokens 404 — we never confirm the route exists.

    Per rFP §5. Recording (§5.5) + decline classification (§5.6) are
    additive layers around the call to plugin.run_chat().
    """
    # ── Token gate ────────────────────────────────────────────────
    expected = _expected_pitch_token()
    if not expected:
        # Server misconfigured (no PITCH_TOKEN set). Fail closed.
        raise HTTPException(status_code=404)
    if not x_pitch_token or not _constant_time_equal(x_pitch_token.strip(), expected):
        raise HTTPException(status_code=404)

    thread_id = _safe_thread_id(req.thread_id)

    # ── Rate limit ────────────────────────────────────────────────
    now = time.time()
    rl = await _check_and_record_rate(expected, now)
    if rl is not None:
        # Educational failure card (rFP §5.6 / improvement #9).
        explanation_map = {
            "daily_cap_fleet": "Daily reflection budget reached fleet-wide. This Titan has spoken extensively today across all visitors. Try tomorrow.",
            "daily_cap_token": "Your pitch session has reached its daily message cap. Try tomorrow.",
            "hour_cap_token": "Your pitch session has reached its hourly message cap. Pause briefly.",
            "burst_cap_token": "You're sending messages faster than the rate limit allows. Slow down a moment.",
        }
        explanation = explanation_map.get(rl["reason"], "Rate limit exceeded.")
        _append_recording(thread_id, "system", {"event": "rate_limited", **rl})
        return PitchChatResponse(
            response="",
            titan=req.titan,
            thread_id=thread_id,
            internal_time=InternalTime(),
            declined=True,
            decline_reason=rl["reason"],
            decline_explanation=explanation,
        )

    # ── Visitor identity (synthetic) ──────────────────────────────
    visitor_hash = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:16]
    claims = {"sub": f"pitch-visitor-{visitor_hash}"}

    payload = {
        "message": req.message,
        "session_id": f"pitch-{thread_id}",
        "user_id": f"pitch-visitor-{visitor_hash}",
    }

    # ── Recording: inbound ────────────────────────────────────────
    _append_recording(thread_id, "in", {
        "message": req.message,
        "titan_requested": req.titan,
    })

    # ── Call plugin pipeline ──────────────────────────────────────
    plugin = getattr(request.app.state, "titan_plugin", None)
    agent = getattr(request.app.state, "titan_agent", None)
    chat_bridge_bus = getattr(request.app.state, "chat_bridge_bus", None)
    if plugin is None:
        raise HTTPException(status_code=503, detail="plugin not initialized")

    try:
        # plugin.run_chat is exposed via KERNEL_RPC_EXPOSED_METHODS (commit
        # 52ae3607) so this call works under api_process_separation_enabled
        # in BOTH modes:
        #   - In-process (legacy): direct Python call.
        #   - Subprocess (current): kernel_rpc proxy executes run_chat in the
        #     parent process where the AGNO agent + memory pipeline live.
        # We deliberately bypass the chat_bridge_bus QUERY/RESPONSE path —
        # under socket mode the api_subprocess is not BUS_SUBSCRIBE'd to its
        # own name, so RESPONSE-rid routed replies never deliver and every
        # request times out (regression vs the bridge's design intent;
        # diagnosed 2026-05-11 via /v4/pitch-chat smoke tests + cross-checked
        # against /chat which exhibits the identical symptom).
        result = await plugin.run_chat(payload, claims, headers={"X-Titan-Channel": "web"})
    except asyncio.TimeoutError:
        explanation = "Titan's reply did not arrive within 60 seconds. Try again, or rephrase."
        _append_recording(thread_id, "system", {"event": "timeout"})
        return PitchChatResponse(
            response="",
            titan=req.titan,
            thread_id=thread_id,
            internal_time=_snapshot_internal_time(plugin),
            declined=True,
            decline_reason="timeout",
            decline_explanation=explanation,
        )
    except Exception as e:
        logger.exception("[PitchChat] run_chat raised: %s", e)
        _append_recording(thread_id, "system", {"event": "exception", "error": str(e)})
        return PitchChatResponse(
            response="",
            titan=req.titan,
            thread_id=thread_id,
            internal_time=_snapshot_internal_time(plugin),
            declined=True,
            decline_reason="exception",
            decline_explanation=f"Pipeline error. Maker has been notified.",
        )

    # ── Classify decline / extract reply ──────────────────────────
    if not isinstance(result, dict):
        result = {"status_code": 500, "body": {"error": "malformed result"}}
    decline = _classify_decline(result)
    body_inner = result.get("body") or {}
    reply_text = ""
    if decline is None:
        reply_text = body_inner.get("response") or ""

    # state_snapshot lives at result["body"]["state_snapshot"] (chat_pipeline
    # composes the dict at chat_pipeline.py:402). Pass body_inner so the
    # snapshot helper can read mood + emotion + is_dreaming directly.
    internal = _snapshot_internal_time(plugin, body_inner if isinstance(body_inner, dict) else None)

    # ── Recording: outbound ───────────────────────────────────────
    _append_recording(thread_id, "out", {
        "response": reply_text,
        "declined": decline is not None,
        "decline_reason": decline[0] if decline else None,
        "internal_time": internal.model_dump(),
    })

    return PitchChatResponse(
        response=reply_text,
        titan=req.titan,
        thread_id=thread_id,
        internal_time=internal,
        declined=decline is not None,
        decline_reason=decline[0] if decline else None,
        decline_explanation=decline[1] if decline else None,
    )


@router.get("/pitch-chat/health")
async def pitch_chat_health() -> JSONResponse:
    """Tiny health probe. Does NOT require X-Pitch-Token so the frontend
    can use it as a connection-banner signal. Returns no narrative.
    """
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "rate_limits": {
                "burst_per_minute": PITCH_BURST_LIMIT,
                "per_hour": PITCH_HOUR_LIMIT,
                "per_day": PITCH_DAY_LIMIT,
                "fleet_day": PITCH_FLEET_DAY_LIMIT,
            },
        },
    )
