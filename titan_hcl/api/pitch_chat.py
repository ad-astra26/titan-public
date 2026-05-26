"""
api/pitch_chat.py — Wallet-less chat endpoint for the /v/<token>/pitch
VC + hackathon route.

Per rFP_observatory_pitch_route.md §5 + §11 (v2 locked 2026-05-11). This
is the backend the frontend Pitch page (Compare-mode default + single-
Titan toggle) calls. Each Titan instance serves only itself — Compare
mode is fan-out on the frontend across /t2/v6/pitch/chat and
/t3/v6/pitch/chat (nginx prefix routing).

Phase E migration (D-SPEC-115, 2026-05-26): the legacy `/v4/pitch-chat`
+ `/v4/pitch-chat/health` paths are 308/301 redirects to `/v6/pitch/chat`
+ `/v6/pitch/chat/health` (built automatically by v6_deprecation.build()
from the `replaces=` field of the manifest rows registered at module
bottom). Old judge/VC share-URLs continue to work — the migration is
backend-only; the frontend `/v/<TOKEN>/pitch` route is unchanged.

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
import inspect
import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional, Deque, DefaultDict

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .auth import verify_maker_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v6/pitch", tags=["pitch"])


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
      1. PITCH_TOKEN env var (preferred — set at titan_hcl startup).
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


class PitchChainProof(BaseModel):
    """Chain-of-evidence proof attached to a pitch reply (rFP §4 #5 Pitch
    half — chain-proof drawer per reply).

    JSON shape matches the frontend discriminated union in
    `components/pitch/ChainProofDrawer.tsx` (ChainProof type). v1 only
    emits the two kinds whose fields are snake_case-compatible across
    both sides (no alias plumbing): `memo` (Solana tx signature) and
    `timechain_block` (TimeChain height + merkle root). Adding `arweave`
    / `titan_identity` later requires camelCase aliases.
    """
    kind: str  # "memo" | "timechain_block"
    signature: Optional[str] = None
    height: Optional[int] = None
    merkle: Optional[str] = None
    label: Optional[str] = None


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
    # Chain-proof references attached to successful replies (rFP §4 #5
    # Pitch half — chain-proof drawer per reply, memory references only).
    # Empty on decline. Frontend renders each entry as a chevron via
    # ChainProofDrawer.
    proofs: list[PitchChainProof] = Field(default_factory=list)


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


# ── Chain-proof annotation (rFP §4 #5 Pitch — memory references) ──────


def _compute_pitch_proofs(request: Request) -> list[PitchChainProof]:
    """Derive 0-1 chain-proof references for a successful pitch reply.

    The Maker direction (Stage 5 design call 2026-05-26) was "memory
    references only". Every pitch reply is informed by Titan's memory
    pipeline (agno_worker memory recall), and that memory itself is
    anchored to the TimeChain on a rolling basis — so the cleanest
    proof-of-memory each reply can carry is "here's where my memory
    chain is right now": the most recent Solana memo signature (when
    the TimeChain has been anchored on-chain) or the local TimeChain
    block height + merkle root (when only the local chain has
    advanced past genesis).

    Prefers the Solana memo proof — judges click → Solscan → see
    on-chain evidence. Falls back to the local block proof so the
    chevron still has something verifiable when on-chain anchoring is
    lagging. Returns [] on any read failure so a stale Titan never
    fails a reply (rFP §8 risk: never lie).

    SHM-direct via TimechainAccessor (G19, no sync bus.request).
    """
    state = getattr(request.app.state, "titan_state", None)
    tc = getattr(state, "timechain", None)
    if tc is None:
        return []
    try:
        stats = tc.get_stats()
    except Exception:
        logger.debug("[PitchChat] timechain.get_stats failed", exc_info=True)
        return []
    if not isinstance(stats, dict) or not stats:
        return []

    # Prefer Solana memo anchor — strongest "on-chain memory" claim.
    anchor = stats.get("anchor")
    if isinstance(anchor, dict):
        sig = anchor.get("last_tx_sig")
        if isinstance(sig, str) and len(sig) >= 32:
            return [PitchChainProof(kind="memo", signature=sig, label="memory chain")]

    # Fallback: local TimeChain tip. `forks["0"]` is the canonical main
    # fork (timechain_state_publisher.py:Phase C §4.B.9). Height + merkle
    # gives the visitor a verifiable local-chain pointer even when no
    # mainnet anchor has landed yet.
    forks = stats.get("forks")
    if isinstance(forks, dict):
        main = forks.get("0") or forks.get(0)
        if isinstance(main, dict):
            height = main.get("tip_height")
            if isinstance(height, int) and height > 0:
                merkle = stats.get("merkle_root")
                return [PitchChainProof(
                    kind="timechain_block",
                    height=height,
                    merkle=merkle if isinstance(merkle, str) else None,
                    label="memory chain",
                )]

    return []


# ── Decline classification ────────────────────────────────────────────


_DECLINE_PATTERNS = [
    ("jailbreak", re.compile(r"jailbreak|prompt.injection|policy.violation", re.I)),
    ("dream_phase", re.compile(r"dreaming|dream.phase|asleep", re.I)),
    ("persona_reject", re.compile(r"persona.rejected|off.policy|cannot.respond", re.I)),
]

# Dream-phase polite-decline pattern. Matches the prefix produced by the
# chat pipeline when the Titan is asleep — that path returns status_code=200
# with the dream message embedded in body.response. Without surfacing this
# as a structured decline, the UI renders the dream text as if it were the
# Titan's regular answer (confusing in Compare mode where one Titan replies
# normally and another shows this).
_DREAM_REPLY_RE = re.compile(
    r"^\s*Titan is currently dreaming|"
    r"^\s*Your message has been queued.*\(position #\d+\)|"
    r"^\s*Estimated wake:",
    re.I,
)


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
    if status == 504:
        # chat_pipeline.run_chat returns 504 with error="llm_timeout" when
        # agent.arun exceeds _AGENT_ARUN_TIMEOUT_S (Layer 1 closure of
        # BUG-CHAT-AGENT-ARUN-HANG-T3-PHASE-C). The "reason" field carries
        # a Phase-aware human-readable explanation already.
        if err.lower().startswith("llm_timeout") or "llm_timeout" in (body_inner.get("error") or "").lower():
            return ("llm_timeout", body_inner.get("reason") or err
                    or "Agent reasoning timed out. Try again in a moment.")
        return ("gateway_timeout", err or "Titan's reply timed out at the gateway.")
    for code, pat in _DECLINE_PATTERNS:
        if err and pat.search(err):
            return (code, err)
    return ("declined", err or f"declined (HTTP {status})")


# ── Endpoint ──────────────────────────────────────────────────────────


@router.post("/chat", response_model=PitchChatResponse)
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

    # ── Call agno_proxy (D-SPEC-72 — was plugin.run_chat → chat_pipeline pre-v1.17.0) ──
    plugin = getattr(request.app.state, "titan_hcl", None)
    if plugin is None:
        raise HTTPException(status_code=503, detail="plugin not initialized")
    # Resolve AgnoProxy via the shared helper from api.chat
    from titan_hcl.api.chat import _get_agno_proxy
    agno_proxy = _get_agno_proxy(request)
    if agno_proxy is None:
        raise HTTPException(
            status_code=503,
            detail="agno_proxy not installed (chat service unavailable)",
        )

    try:
        # AgnoProxy.chat() returns the same run_chat-shaped dict
        # {status_code, body, extra_headers} — drop-in compat with the
        # downstream classification logic below.
        result = await agno_proxy.chat(
            message=req.message,
            user_id=payload["user_id"],
            session_id=payload["session_id"],
            channel="pitch",
            is_maker=False,
            claims_sub=claims.get("sub", ""),
        )
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
        # OVG guard messages — pipeline returned 200 with response_text
        # replaced by "[VERIFICATION UNAVAILABLE — request was not processed]"
        # or "[VERIFICATION ERROR: …]" when the output_verifier worker
        # times out / errors via bus. Surface as a substrate-visible
        # decline card instead of letting the bracket-string render as
        # the Titan's reply (rFP §5.6 educational failure modes).
        if reply_text.startswith("[VERIFICATION UNAVAILABLE") or reply_text.startswith("[VERIFICATION ERROR"):
            decline = (
                "verification_unavailable",
                f"{req.titan}'s output-verifier worker is offline (Phase C "
                "bus regression on this instance). The Titan generated a "
                "reply but the cryptographic verification gate couldn't "
                "co-sign it — defense-in-depth chose not to ship an "
                "unverified response. Try again in a moment.",
            )
            reply_text = ""
        # Dream-phase reply — pipeline returns 200 with a polite "Titan is
        # currently dreaming" message + queue position when the Titan is
        # asleep. Without classifying this, the UI renders the dream text
        # as if it were a normal answer. Surface as decline_reason so the
        # UI can show a dream-state card (rFP §5.6 educational failure
        # modes; in Compare mode this is critical so one awake Titan's
        # answer doesn't sit next to another's dream-text reply).
        elif _DREAM_REPLY_RE.search(reply_text):
            decline = (
                "dream_phase",
                f"{req.titan} is currently in a dream cycle — its reasoning "
                "is offline while it consolidates today's experiences into "
                "memory. Your message was received and queued; it will be "
                "answered after wake. (Phase B/C dream cycles run ~5–15 "
                "minutes; check internal time stamp for current phase.)",
            )
            # Preserve the original dream text in decline_explanation context;
            # blank reply_text so the UI doesn't double-render it.
            reply_text = ""

    # state_snapshot lives at result["body"]["state_snapshot"] (chat_pipeline
    # composes the dict at chat_pipeline.py:402). Pass body_inner so the
    # snapshot helper can read mood + emotion + is_dreaming directly.
    internal = _snapshot_internal_time(plugin, body_inner if isinstance(body_inner, dict) else None)

    # Chain-of-evidence (rFP §4 #5 Pitch). Only attach to non-declined
    # replies — a "why declined" card already carries its substrate
    # explanation; chevroning a decline would be visual noise.
    proofs: list[PitchChainProof] = []
    if decline is None and reply_text:
        proofs = _compute_pitch_proofs(request)

    # ── Recording: outbound ───────────────────────────────────────
    _append_recording(thread_id, "out", {
        "response": reply_text,
        "declined": decline is not None,
        "decline_reason": decline[0] if decline else None,
        "internal_time": internal.model_dump(),
        "proofs": [p.model_dump() for p in proofs],
    })

    return PitchChatResponse(
        response=reply_text,
        titan=req.titan,
        thread_id=thread_id,
        internal_time=internal,
        declined=decline is not None,
        decline_reason=decline[0] if decline else None,
        decline_explanation=decline[1] if decline else None,
        proofs=proofs,
    )


@router.get("/chat/health")
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


# ── Pitch tail endpoints (Stage 2 of §4 v2 closeout, 2026-05-26) ──────
#
# Three new readouts + two admin endpoints round out the v2 design:
#   - GET  /witness-tail   — substrate viewer aux signals for Witness mode
#                            (rFP §4.5 #3). Token-gated. Trinity tensors
#                            themselves are NOT included here — the
#                            WitnessPanel reads them via the existing
#                            /v6/trinity/* surface (useTitanSelf hook), so
#                            we only ship the supplementary signals: a
#                            filtered bus-event tail (last 20), a CGN
#                            snapshot, and the last meditation commit
#                            (chain-proof linkable).
#   - GET  /thinking-tail  — minimal event tail for the always-visible
#                            footer strip on the Pitch page (rFP §4 #7).
#                            Lighter than witness-tail; the frontend polls
#                            this at 1Hz and shows the last 3.
#   - GET  /sessions       — list of recorded JSONLs (rFP §5.5).
#                            Maker-Ed25519-auth gated via
#                            api.auth.verify_maker_auth.
#   - GET  /sessions/{thread_id} — read one JSONL. Maker-auth gated.
#
# SPEC compliance (Preamble G18–G22):
#   - All state reads SHM-direct via TitanStateAccessor (no sync
#     bus.request for state; G19).
#   - Event tail reads observatory.db.event_log directly — the same
#     pattern /v6/system/timeseries + /v6/expression/state-narration use
#     (producer = observatory_worker, no SHM slot, no kernel-RPC).
#   - Readouts only — no LLM calls, no bus publishes (INV-9 provably
#     non-blocking; §8.0.ter satisfied trivially).


# High-signal event types we surface in the tail endpoints. The hot ones
# (sphere_pulse @ 5.7M, expression_fired @ 5M, neuromod_update @ 1.3M)
# would drown the strip in noise; the curated set below is what judges
# read as "the Titan is thinking right now."
_THINKING_TAIL_EVENT_TYPES: tuple[str, ...] = (
    "chat_message", "big_pulse", "great_pulse", "dream_state",
    "reflex_reward", "hormone_fired",
)
# Witness mode tolerates more — adds expression + filtered neuromod
# transitions so the viewer can see the inner state shifting.
_WITNESS_TAIL_EVENT_TYPES: tuple[str, ...] = _THINKING_TAIL_EVENT_TYPES + (
    "expression_fired",
)
# Hard cap so a malformed `limit` query can never sweep the whole table.
_TAIL_HARD_MAX = 50


def _validate_pitch_token_or_404(x_pitch_token: Optional[str]) -> str:
    """Shared token gate for the tail readouts. Returns the validated
    token (callers don't need it but we surface it for symmetry with the
    rate-limit path on /chat). Raises 404 on miss — never a 401 or 403,
    we don't confirm the route exists.
    """
    expected = _expected_pitch_token()
    if not expected:
        raise HTTPException(status_code=404)
    if not x_pitch_token or not _constant_time_equal(x_pitch_token.strip(), expected):
        raise HTTPException(status_code=404)
    return expected


def _resolve_titan_id(request: Request) -> str:
    """Best-effort read of this Titan's id (T1/T2/T3) for response stamping.
    Falls back to '?' if identity isn't available — never raises.
    """
    state = getattr(request.app.state, "titan_state", None)
    identity = getattr(state, "identity", None)
    if identity is not None:
        try:
            tid = identity.titan_id()
            if isinstance(tid, str) and tid:
                return tid
        except Exception:
            pass
    plugin = getattr(request.app.state, "titan_hcl", None)
    for attr in ("titan_id", "_titan_id"):
        v = getattr(plugin, attr, None)
        if isinstance(v, str) and v:
            return v
    return "?"


def _read_event_tail(event_types: tuple[str, ...], limit: int) -> list[dict]:
    """Pull the last `limit` rows from observatory.db.event_log filtered
    to the given event_types. Returns [{ts, event_type, summary}] in
    most-recent-first order.

    Read pattern matches dashboard.py:1130 (direct sqlite3 read against
    data/observatory.db). The DB is owned by observatory_worker; we open
    it read-only here so writer contention is impossible. Failures yield
    an empty list — degraded-but-honest (rFP §8 risk: "DEGRADED state to
    a VC" → never lie; surface as empty).
    """
    import os
    import sqlite3
    if limit <= 0:
        return []
    limit = min(limit, _TAIL_HARD_MAX)
    db_path = os.path.join("data", "observatory.db")
    if not os.path.exists(db_path):
        return []
    placeholders = ",".join("?" for _ in event_types)
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error:
        logger.debug("[PitchChat] event_log open failed", exc_info=True)
        return []
    try:
        rows = con.execute(
            f"SELECT ts, event_type, summary FROM event_log "
            f"WHERE event_type IN ({placeholders}) "
            f"ORDER BY ts DESC LIMIT ?",
            (*event_types, limit),
        ).fetchall()
    except sqlite3.Error:
        logger.debug("[PitchChat] event_log query failed", exc_info=True)
        con.close()
        return []
    con.close()
    return [
        {"ts": float(ts) if ts is not None else 0.0,
         "event_type": str(et or ""),
         "summary": str(s or "")}
        for (ts, et, s) in rows
    ]


def _read_last_meditation(request: Request) -> Optional[dict]:
    """Read meditation_state.bin SHM slot via MeditationStateReader — the
    same path /v6/dreaming/meditation-health uses (dashboard.py:3999).
    Returns a slim chain-proof dict {ts, signature, arweave, summary} or
    None if no commit exists or the slot isn't published yet.
    """
    plugin = getattr(request.app.state, "titan_hcl", None)
    try:
        from titan_hcl.core.state_registry import resolve_titan_id
        from titan_hcl.logic.meditation_state_reader import MeditationStateReader
        titan_id = resolve_titan_id(getattr(plugin, "titan_id", None) if plugin else None)
        reader = MeditationStateReader(titan_id)
        ms = reader.read()
    except Exception:
        logger.debug("[PitchChat] meditation_state read failed", exc_info=True)
        return None
    if not isinstance(ms, dict) or not ms:
        return None
    out: dict = {}
    for k in ("ts", "signature", "arweave_url", "arweave_id", "summary",
              "meditation_id", "type", "committed", "last_commit_ts"):
        v = ms.get(k)
        if v is not None:
            out[k] = v
    return out or None


def _read_cgn_snapshot(request: Request) -> Optional[dict]:
    """Slim CGN snapshot via the accessor. Returns the most useful fields
    for Witness mode without dumping the whole stats blob (which can be
    multiple kilobytes). None if CGN isn't initialized.
    """
    state = getattr(request.app.state, "titan_state", None)
    cgn = getattr(state, "cgn", None)
    if cgn is None:
        return None
    try:
        stats = cgn.get_stats()
    except Exception:
        logger.debug("[PitchChat] cgn.get_stats failed", exc_info=True)
        return None
    if not isinstance(stats, dict):
        return None
    out: dict = {}
    for k in ("last_advisor", "last_decision", "haov_consensus",
              "frame_count", "by_domain_summary", "graduation_status"):
        v = stats.get(k)
        if v is not None:
            out[k] = v
    return out or None


# ── Tail response models ──────────────────────────────────────────────


class TailEvent(BaseModel):
    ts: float
    event_type: str
    summary: str


class WitnessTailResponse(BaseModel):
    titan_id: str
    ts: float
    events: list[TailEvent] = Field(default_factory=list)
    cgn: Optional[dict] = None
    last_meditation: Optional[dict] = None


class ThinkingTailResponse(BaseModel):
    titan_id: str
    ts: float
    events: list[TailEvent] = Field(default_factory=list)


# ── Session-recording models ──────────────────────────────────────────


class PitchSessionMeta(BaseModel):
    thread_id: str
    first_ts: Optional[float]
    last_ts: Optional[float]
    message_count: int
    declined_count: int
    bytes: int


class PitchSessionsResponse(BaseModel):
    sessions: list[PitchSessionMeta]
    total: int


class PitchSessionDetailResponse(BaseModel):
    thread_id: str
    lines: list[dict]


def _safe_thread_id_strict(s: str) -> str:
    """Stricter wrapper around _safe_thread_id that raises 400 on the
    sanitization fallback (used by the admin GET-by-thread_id route so
    "/sessions/../etc" type attempts surface a clear 400 rather than a
    silent 'invalid' canonical id)."""
    raw = (s or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="empty thread_id")
    if not re.match(r"^[A-Za-z0-9_-]{1,64}$", raw):
        raise HTTPException(status_code=400, detail="invalid thread_id")
    return raw


def _scan_sessions(limit: int) -> list[PitchSessionMeta]:
    """Walk PITCH_RECORDING_DIR and produce session metadata. Sorted by
    last-modified DESC. Quietly tolerates malformed JSONL lines (skip).
    """
    if not PITCH_RECORDING_DIR.exists():
        return []
    files = []
    for p in PITCH_RECORDING_DIR.glob("*.jsonl"):
        try:
            files.append((p.stat().st_mtime, p))
        except OSError:
            continue
    files.sort(key=lambda t: t[0], reverse=True)
    out: list[PitchSessionMeta] = []
    for _, p in files[:max(0, min(limit, 500))]:
        thread_id = p.stem
        if not re.match(r"^[A-Za-z0-9_-]{1,64}$", thread_id):
            continue
        first_ts: Optional[float] = None
        last_ts: Optional[float] = None
        msg_count = 0
        declined = 0
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        try:
            with p.open("r", encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        row = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    ts = row.get("ts")
                    if isinstance(ts, (int, float)):
                        if first_ts is None or ts < first_ts:
                            first_ts = float(ts)
                        if last_ts is None or ts > last_ts:
                            last_ts = float(ts)
                    if row.get("direction") in ("in", "out"):
                        msg_count += 1
                    if row.get("declined") is True:
                        declined += 1
        except OSError:
            continue
        out.append(PitchSessionMeta(
            thread_id=thread_id,
            first_ts=first_ts,
            last_ts=last_ts,
            message_count=msg_count,
            declined_count=declined,
            bytes=size,
        ))
    return out


def _read_session_lines(thread_id: str) -> list[dict]:
    """Read the full JSONL for one session. Empty list if no file."""
    path = PITCH_RECORDING_DIR / f"{thread_id}.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
    except OSError:
        logger.exception("[PitchChat] session read failed (thread=%s)", thread_id)
        return []
    return out


# ── Tail endpoints (token-gated, no rate-limit — readouts only) ───────


@router.get("/witness-tail", response_model=WitnessTailResponse)
async def witness_tail(
    request: Request,
    x_pitch_token: Optional[str] = Header(None, alias="X-Pitch-Token"),
    events_limit: int = 20,
) -> JSONResponse:
    """Substrate viewer aux signals for Witness mode (rFP §4.5).

    Token-gated. Aggregates a filtered bus-event tail, a slim CGN
    snapshot, and the last meditation chain-proof. Trinity tensors are
    NOT included — the frontend reads them via the existing
    /v6/trinity/* surface (useTitanSelf hook) for visual coherence with
    the rest of the Observatory.
    """
    _validate_pitch_token_or_404(x_pitch_token)
    events = _read_event_tail(_WITNESS_TAIL_EVENT_TYPES, max(1, min(events_limit, _TAIL_HARD_MAX)))
    cgn = _read_cgn_snapshot(request)
    meditation = _read_last_meditation(request)
    body = WitnessTailResponse(
        titan_id=_resolve_titan_id(request),
        ts=time.time(),
        events=[TailEvent(**e) for e in events],
        cgn=cgn,
        last_meditation=meditation,
    )
    return JSONResponse(content=body.model_dump(), headers={"Cache-Control": "no-store"})


@router.get("/thinking-tail", response_model=ThinkingTailResponse)
async def thinking_tail(
    request: Request,
    x_pitch_token: Optional[str] = Header(None, alias="X-Pitch-Token"),
    limit: int = 5,
) -> JSONResponse:
    """Minimal event tail for the Pitch page's always-visible footer
    strip (rFP §4 #7 — live thinking strip).

    Token-gated. Frontend polls at 1Hz and renders the last 3 to keep the
    strip readable. We return up to `limit` (default 5; hard-capped 50).
    """
    _validate_pitch_token_or_404(x_pitch_token)
    events = _read_event_tail(_THINKING_TAIL_EVENT_TYPES, max(1, min(limit, _TAIL_HARD_MAX)))
    body = ThinkingTailResponse(
        titan_id=_resolve_titan_id(request),
        ts=time.time(),
        events=[TailEvent(**e) for e in events],
    )
    return JSONResponse(content=body.model_dump(), headers={"Cache-Control": "no-store"})


# ── Session recordings (Maker-auth gated, admin-kind) ─────────────────


@router.get("/sessions", response_model=PitchSessionsResponse,
            dependencies=[Depends(verify_maker_auth)])
async def list_pitch_sessions(limit: int = 50) -> JSONResponse:
    """List the most-recently-modified pitch session recordings
    (rFP §5.5). Maker-Ed25519-auth gated via api.auth.verify_maker_auth
    — same path as /v6/maker/proposals.

    Filesystem read of PITCH_RECORDING_DIR (data/pitch_sessions/*.jsonl);
    no DB or bus involvement.
    """
    sessions = _scan_sessions(max(1, min(limit, 500)))
    body = PitchSessionsResponse(sessions=sessions, total=len(sessions))
    return JSONResponse(content=body.model_dump(), headers={"Cache-Control": "no-store"})


@router.get("/sessions/{thread_id}", response_model=PitchSessionDetailResponse,
            dependencies=[Depends(verify_maker_auth)])
async def get_pitch_session(thread_id: str) -> JSONResponse:
    """Read the full JSONL for one pitch session (rFP §5.5). Maker-auth
    gated. Returns [{ts, direction, titan_id?, message, ...}] in append
    order — same shape the recording writer in _append_recording emits.
    """
    safe = _safe_thread_id_strict(thread_id)
    lines = _read_session_lines(safe)
    body = PitchSessionDetailResponse(thread_id=safe, lines=lines)
    return JSONResponse(content=body.model_dump(), headers={"Cache-Control": "no-store"})


# ── v6 manifest registration (Phase E, D-SPEC-115) ────────────────────
#
# Each route on this router gets a RouteSpec row in v6_manifest.REGISTRY
# so the /v6/manifest cross-check stays in_sync and v6_deprecation.build()
# auto-generates the legacy /v4 → /v6 redirects (308 for POST,
# 301 for GET) from the `replaces=` field.
#
# Loaded at module import; titan_hcl/api/__init__.py guarantees this
# module loads AFTER v6_manifest in both the create_app and reload paths
# so REGISTRY isn't wiped after we register.
from . import v6_manifest as _m
from .v6_manifest import RouteSpec

_m.register(
    RouteSpec(
        path="/v6/pitch/chat",
        method="POST",
        group="pitch",
        kind="mutation",
        summary="Wallet-less pitch chat (X-Pitch-Token gated, persona pipeline, per-token + fleet rate-limited, recorded to data/pitch_sessions/).",
        command="commands.pitch_chat",
        replaces=("/v4/pitch-chat",),
    ),
    RouteSpec(
        path="/v6/pitch/chat/health",
        method="GET",
        group="pitch",
        kind="readout",
        summary="Pitch chat health probe — no auth, returns rate-limit constants for the frontend connection banner.",
        replaces=("/v4/pitch-chat/health",),
    ),
    RouteSpec(
        path="/v6/pitch/witness-tail",
        method="GET",
        group="pitch",
        kind="readout",
        summary="Witness-mode aux signals (rFP §4.5): filtered bus-event tail + slim CGN snapshot + last meditation chain-proof. Trinity tensors live in /v6/trinity/*.",
        accessor=None,
        shm_slots=("cgn_engine_state.bin", "meditation_state.bin"),
        producers=("cgn_worker", "meditation_worker", "observatory_worker"),
    ),
    RouteSpec(
        path="/v6/pitch/thinking-tail",
        method="GET",
        group="pitch",
        kind="readout",
        summary="Live thinking strip event tail (rFP §4 #7): last N high-signal bus events from observatory.db.event_log.",
        producers=("observatory_worker",),
    ),
    RouteSpec(
        path="/v6/pitch/sessions",
        method="GET",
        group="pitch",
        kind="admin",
        summary="List recorded pitch session JSONLs (rFP §5.5). Maker-Ed25519-auth gated via api.auth.verify_maker_auth.",
        producers=(),
    ),
    RouteSpec(
        path="/v6/pitch/sessions/{thread_id}",
        method="GET",
        group="pitch",
        kind="admin",
        summary="Read one pitch session's JSONL (rFP §5.5). Maker-auth gated; thread_id strictly validated against [A-Za-z0-9_-]{1,64}.",
        producers=(),
    ),
)
