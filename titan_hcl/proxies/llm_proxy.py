"""
LLM/Inference Module Proxy — lazy bridge to the supervised LLM process.

D-SPEC-77 (SPEC v1.18.0): post-D-SPEC-72 carve, `llm_worker` no longer
handles chat (agno_worker imports inference module directly). The
remaining surface is background work — distill / score / teacher — and
those routes now use EXPLICIT topic-named bus events:

  - `LLM_DISTILL_REQUEST` ↔ `LLM_DISTILL_RESPONSE`  (this proxy)
  - `LLM_SCORE_REQUEST`   ↔ `LLM_SCORE_RESPONSE`    (this proxy)
  - `LLM_TEACHER_REQUEST` ↔ `LLM_TEACHER_RESPONSE`  (spirit_worker direct)

Both `distill` and `score` accept optional `pre_hook` / `post_hook` flags
so llm_worker can pipe the call through `llm_pipeline.compose_pre /
verify_post_async` — making it the canonical "talk through the truth-gate"
surface for non-agno generation (X replies via social_x_gateway,
autonomous language pipeline, future avatars).

The legacy `LLMProxy.chat()` method was DELETED in D-SPEC-77 (no shim per
`feedback_no_shim_old_path_must_be_deleted`) — no live caller; agno_worker
handles all chat now.
"""
import logging
import uuid
from typing import Optional

from ..bus import (
    DivineBus,
    LLM_DISTILL_REQUEST,
    LLM_DISTILL_RESPONSE,
    LLM_SCORE_REQUEST,
    LLM_SCORE_RESPONSE,
    make_msg,
)
from ..guardian import Guardian

logger = logging.getLogger(__name__)


class LLMProxy:
    """
    Drop-in proxy for the LLM/Inference module.
    Routes distillation + score inference through Divine Bus via explicit
    topics (D-SPEC-77). Chat path retired — handled by agno_worker.
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        self._reply_queue = bus.subscribe("llm_proxy", reply_only=True)
        self._started = False

    def _ensure_started(self) -> None:
        # Async-safe Guardian.start() — see _start_safe.py for rationale.
        from ._start_safe import ensure_started_async_safe
        if ensure_started_async_safe(
            self._guardian, "llm", id(self), proxy_label="LLMProxy"
        ):
            self._started = True

    async def distill(self, text: str,
                      instruction: str = "Summarize concisely",
                      *,
                      model: Optional[str] = None,
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      pre_hook: bool = False,
                      post_hook: bool = False,
                      channel: str = "agent",
                      timeout: float = 30.0) -> str:
        """Distill/summarize text via the supervised llm_worker.

        D-SPEC-77: publishes LLM_DISTILL_REQUEST explicit topic (replaces
        the legacy `bus.QUERY action="distill"` muxed path).

        Phase 3 Chunk ω (D-SPEC-88, 2026-05-18): added `max_tokens` +
        `temperature` overrides so callers that previously did direct
        httpx.post (events_teacher.py:1573 with max_tokens=800, temp=0.3)
        can migrate without losing behavior parity.

        Optional flags:
            model:        Provider model override.
            max_tokens:   Cap response length. None → provider default.
            temperature:  Sampling temperature 0.0-2.0. None → provider default.
            pre_hook:     If True, llm_worker wraps the prompt with
                          llm_pipeline.compose_pre (felt-state + primedirectives).
            post_hook:    If True, llm_worker runs llm_pipeline.verify_post_async
                          on the output (OVG truth gate).
            channel:      OVG channel when post_hook=True (default "agent").

        Returns the result text, or empty string on timeout / error.
        Phase C exemption: 30s timeout > §1.B 5s default — allowlisted in
        phase_c_rpc_exemptions.yaml as LLM work-RPC.
        """
        self._ensure_started()
        payload = {
            "text": text,
            "instruction": instruction,
            "pre_hook": pre_hook,
            "post_hook": post_hook,
            "channel": channel,
        }
        if model:
            payload["model"] = model
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        # SPEC §8.2 D-SPEC-65 + bus.py:1446-1471 canonical pattern: assign a
        # uuid rid per request so the response demuxer in _await_response can
        # match LLM_DISTILL_RESPONSE messages back to the correct caller when
        # multiple distill requests are in flight on the shared reply queue
        # (in-kernel social_x_composer + HTTP `/v4/llm-distill` from
        # events_teacher / persona_endurance both land here). Without rid
        # filtering the first matching-type response was returned to ANY
        # waiter — caused the 2026-05-23 T2/T3 events_teacher JSON leak into
        # X posts (root cause documented in BUGS section A.10).
        rid = uuid.uuid4().hex
        try:
            self._bus.publish(make_msg(
                LLM_DISTILL_REQUEST,
                "llm_proxy", "llm",
                payload,
                rid=rid,
            ))
        except Exception as e:
            logger.warning("[LLMProxy] distill publish raised: %s", e)
            return ""
        return await self._await_response(
            LLM_DISTILL_RESPONSE, timeout, result_key="result",
            expected_rid=rid)

    async def score(self, prompt: str,
                    *,
                    model: Optional[str] = None,
                    timeout: float = 30.0) -> float:
        """Score a prompt via llm_worker.

        D-SPEC-77: publishes LLM_SCORE_REQUEST. Used by meta_teacher_worker
        for gating + future meta-cognitive scoring paths.
        """
        self._ensure_started()
        payload = {"prompt": prompt, "timeout": timeout}
        if model:
            payload["model"] = model
        # Per-request rid for response demux — see distill() comment for
        # SPEC anchor + the 2026-05-23 leak that motivated the fix.
        rid = uuid.uuid4().hex
        try:
            self._bus.publish(make_msg(
                LLM_SCORE_REQUEST,
                "llm_proxy", "llm",
                payload,
                rid=rid,
            ))
        except Exception as e:
            logger.warning("[LLMProxy] score publish raised: %s", e)
            return 0.0
        result = await self._await_response(
            LLM_SCORE_RESPONSE, timeout, result_key="score",
            expected_rid=rid)
        try:
            return float(result) if result != "" else 0.0
        except (TypeError, ValueError):
            return 0.0

    async def _await_response(self, expected_type: str, timeout: float,
                              *, result_key: str,
                              expected_rid: Optional[str] = None):
        """Drain reply_queue for a message matching expected_type AND
        expected_rid. Returns the value at payload[result_key] or "" / 0.0
        on timeout.

        SPEC anchors:
          - §8.2 D-SPEC-65: "Response arrives async via correlation_id-keyed
            cache" — multi-request demux requires rid matching, not type
            matching alone.
          - bus.py:1446-1471 (DivineBus.request canonical pattern): filter
            replies by `type == expected AND rid == expected_rid`; put back
            non-matching responses so concurrent waiters can find theirs.

        2026-05-23 bug fix: prior to this method's rid-filter, multiple
        in-flight distill calls (in-kernel social_x_composer + HTTP
        `/v4/llm-distill` from events_teacher / persona_endurance) shared
        one `_reply_queue` and only filtered by type — first matching-type
        response was returned to ANY waiter, causing events_teacher's JSON
        distillation to surface in X posts as a tweet body (T2 ids
        339/337/335 + T3 ids 367/365/363/361 redacted same day).
        """
        import asyncio
        import time
        deadline = time.time() + float(timeout)
        loop = asyncio.get_event_loop()
        while time.time() < deadline:
            try:
                reply = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, lambda: self._reply_queue.get(True, 0.1)),
                    timeout=0.5,
                )
            except (asyncio.TimeoutError, Exception):
                continue
            if not reply:
                continue
            if reply.get("type") != expected_type:
                # Wrong message type entirely (e.g. a stray broadcast that
                # leaked past reply_only filtering) — drop, do NOT put back.
                continue
            if expected_rid is not None and reply.get("rid") != expected_rid:
                # Right type, wrong rid — another caller's response. Put
                # back so that caller's _await_response can claim it.
                # Mirrors DivineBus.request at bus.py:1467-1468.
                try:
                    self._reply_queue.put_nowait(reply)
                except Exception as _putback_err:
                    logger.warning(
                        "[LLMProxy] put-back failed for rid=%s "
                        "(queue full?): %s — response lost",
                        reply.get("rid"), _putback_err)
                # Tiny yield so other waiters' executor tasks can run and
                # claim the put-back reply before we re-grab it ourselves.
                await asyncio.sleep(0.01)
                continue
            body = reply.get("payload") or {}
            if "error" in body:
                logger.warning(
                    "[LLMProxy] %s returned error: %s",
                    expected_type, body["error"])
            return body.get(result_key, "")
        logger.warning("[LLMProxy] %s timed out (%.0fs, rid=%s)",
                       expected_type, timeout,
                       (expected_rid or "")[:8])
        return ""
