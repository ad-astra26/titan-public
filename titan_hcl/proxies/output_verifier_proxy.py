"""
Output Verifier Proxy — bus-routed bridge to output_verifier_worker (L3 §A.8.3).

Drop-in interface match for `OutputVerifier`. When flag
`microkernel.a8_output_verifier_subprocess_enabled=true`, parent's
`_output_verifier` becomes this proxy; calls translate to bus
QUERY/RESPONSE round-trips against the output_verifier worker.

When flag is false (default), parent retains a local OutputVerifier
instance — no proxy is created, no behavior change.

Stats attributes (verified_count, rejected_count)
are kept cached locally and refreshed when OUTPUT_VERIFIER_STATS
broadcasts arrive. While the cache is empty (worker hasn't booted yet
or initial broadcast pending), they default to 0 — kernel.py:1082's
snapshot reader uses `getattr(..., default)` so missing values are tolerated.

See: titan-docs/rFP_microkernel_phase_a8_l2_l3_residency_completion.md §A.8.3
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import msgpack

from ..bus import DivineBus
from ..core.state_registry import (
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)
from ..logic.session3_state_specs import OUTPUT_VERIFIER_STATE_SPEC

logger = logging.getLogger(__name__)


class OutputVerifierProxy:
    """Bus-routed proxy that mirrors OutputVerifier's public surface.

    Phase C Session 4 (rFP §4.C.14):
      - verify_and_sign / build_timechain_payload — true work-RPC
        (signing, hashing, timechain payload assembly). Migrated from
        sync bus.request to async bus.request_async per Preamble G19
        with explicit ≤5s timeout. Allowlist entry in
        phase_c_rpc_exemptions.yaml.
      - get_stats — state lookup, migrated to SHM read of
        output_verifier_state.bin (Session 3 publisher).

    Attributes (cached, updated by OUTPUT_VERIFIER_STATS broadcast):
        verified_count, rejected_count
    """

    def __init__(self, bus: DivineBus, request_timeout_s: float = 5.0):
        self._bus = bus
        # G19 §1.B: timeout ≤ 5s for work-RPC.
        self._timeout = min(float(request_timeout_s), 5.0)
        self._reply_queue = bus.subscribe("output_verifier_proxy", reply_only=True)
        # Cached stats — updated from OUTPUT_VERIFIER_STATS broadcasts.
        # (The legacy always-0 `sovereignty_score` field was removed — sovereignty
        # is the ONE synthesis S, INV-SDA-3; the OVG owns output_integrity, not a
        # second sovereignty number.)
        self.verified_count: int = 0
        self.rejected_count: int = 0

        # Phase C Session 4 (rFP §4.C.14) — SHM-direct reader for
        # output_verifier_state.bin (Session 3 publisher).
        self._titan_id = resolve_titan_id()
        self._shm_root: Path = ensure_shm_root(self._titan_id)
        self._r_ovg_state = StateRegistryReader(
            OUTPUT_VERIFIER_STATE_SPEC, self._shm_root)
        self._fallback_counts: dict[str, int] = {}

    async def _request_async(self, payload: dict) -> Optional[dict]:
        """Single async work-RPC primitive."""
        try:
            reply = await self._bus.request_async(
                "output_verifier_proxy", "output_verifier",
                payload, self._timeout, self._reply_queue,
            )
        except Exception as e:
            logger.warning("[OutputVerifierProxy] bus.request_async raised: %s", e)
            return None
        return reply

    def _request_sync_fallback(self, payload: dict) -> Optional[dict]:
        """Sync wrapper around the async path. Used when caller is not
        async-aware (most legacy verify_and_sign call sites). When inside
        a running event loop we degrade to the legacy sync request path
        with the same bounded timeout (G19 work-RPC exemption — explicit
        timeout, not state lookup)."""
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if not in_loop:
            try:
                return asyncio.run(self._request_async(payload))
            except Exception as e:
                logger.warning("[OutputVerifierProxy] asyncio.run failed: %s — "
                               "falling back to bounded sync bus.request", e)
        # Loop-running fallback — bounded sync (work-RPC exemption).
        return self._bus.request(
            src="output_verifier_proxy", dst="output_verifier",
            payload=payload, timeout=self._timeout,
            reply_queue=self._reply_queue,
        )

    def _track_fallback(self, slot_name: str, reason: str) -> None:
        prev = self._fallback_counts.get(slot_name, 0)
        self._fallback_counts[slot_name] = prev + 1
        if prev == 0:
            logger.info(
                "[OutputVerifierProxy] FIRST FALLBACK slot=%s reason=%s",
                slot_name, reason)

    # ── D-SPEC-74 (SPEC v1.18.0) — safety/signing split ─────────────

    async def verify_safety_async(self, output_text: str, channel: str,
                                  injected_context: str = "",
                                  prompt_text: str = "",
                                  chain_state: Optional[dict] = None):
        """Phase 1 — async work-RPC for the deterministic truth gate.

        Returns SafetyResult (dataclass) — use this method in async hot
        paths (agno_worker, social_x_gateway). For sync callers, use
        `verify_safety()` which dispatches via asyncio.run when no loop
        is running.
        """
        from ..logic.output_verifier import SafetyResult
        reply = await self._request_async({
            "action": "verify_safety",
            "output_text": output_text,
            "channel": channel,
            "injected_context": injected_context,
            "prompt_text": prompt_text,
            "chain_state": chain_state,
        })
        return _decode_safety_result(reply, output_text, channel)

    async def sign_and_commit_async(self, output_text: str, channel: str,
                                    prompt_text: str = "",
                                    chain_state: Optional[dict] = None,
                                    safety_verdict_token: str = "",
                                    verdict_ts: float = 0.0):
        """Phase 2 — async work-RPC for Ed25519 sign + TimeChain commit.

        Spawned as `asyncio.create_task(...)` by agno_worker concurrent
        with SSE drain (rFP Chunk C). Returns SignedResult on completion.
        """
        from ..logic.output_verifier import SignedResult
        reply = await self._request_async({
            "action": "sign_and_commit",
            "output_text": output_text,
            "channel": channel,
            "prompt_text": prompt_text,
            "chain_state": chain_state,
            "safety_verdict_token": safety_verdict_token,
            "verdict_ts": verdict_ts,
        })
        return _decode_signed_result(reply)

    def verify_safety(self, output_text: str, channel: str,
                      injected_context: str = "",
                      prompt_text: str = "",
                      chain_state: Optional[dict] = None):
        """Sync entry — dispatches to async via asyncio.run when NO loop.

        D-SPEC-74 RETIREMENT: this method does NOT fall back to bus.request
        when inside a running loop (the legacy verify_and_sign sync fallback
        path called bus.request which `_WorkerBusClient` does NOT implement,
        breaking OVG inside worker subprocesses — surfaced on T1 2026-05-17).
        Async callers MUST use `verify_safety_async`; sync-from-loop is a
        contract error and is logged loudly.
        """
        try:
            asyncio.get_running_loop()
            logger.error(
                "[OutputVerifierProxy] verify_safety() called from inside a "
                "running event loop — caller MUST use verify_safety_async(). "
                "Returning hard-fail SafetyResult.")
            from ..logic.output_verifier import SafetyResult
            return SafetyResult(
                passed=False, output_text=output_text, channel=channel,
                violation_type="proxy_sync_in_loop",
                violations=["sync verify_safety from running loop — use async"],
                guard_message="[VERIFICATION CONTRACT ERROR — use async]",
            )
        except RuntimeError:
            return asyncio.run(self.verify_safety_async(
                output_text=output_text, channel=channel,
                injected_context=injected_context, prompt_text=prompt_text,
                chain_state=chain_state,
            ))

    def sign_and_commit(self, output_text: str, channel: str,
                        prompt_text: str = "",
                        chain_state: Optional[dict] = None,
                        safety_verdict_token: str = "",
                        verdict_ts: float = 0.0):
        """Sync entry to sign_and_commit. Mirrors verify_safety dispatch."""
        try:
            asyncio.get_running_loop()
            logger.error(
                "[OutputVerifierProxy] sign_and_commit() called from inside "
                "a running event loop — caller MUST use sign_and_commit_async()."
            )
            from ..logic.output_verifier import SignedResult
            return SignedResult(signed=False, signature=None,
                                error="proxy_sync_in_loop")
        except RuntimeError:
            return asyncio.run(self.sign_and_commit_async(
                output_text=output_text, channel=channel,
                prompt_text=prompt_text, chain_state=chain_state,
                safety_verdict_token=safety_verdict_token,
                verdict_ts=verdict_ts,
            ))

    # ── Legacy combined entry (back-compat during migration) ─────────

    def verify_and_sign(self, output_text: str, channel: str,
                        injected_context: str = "",
                        prompt_text: str = "",
                        chain_state: Optional[dict] = None):
        """Mirror of OutputVerifier.verify_and_sign — returns OVGResult.

        Sync entry point — uses async work-RPC underneath."""
        # Avoid circular import at module load — OutputVerifier ships here only
        # so callers needing OVGResult parsing can deserialize the dict response.
        from ..logic.output_verifier import OVGResult
        payload = {
            "action": "verify_and_sign",
            "output_text": output_text,
            "channel": channel,
            "injected_context": injected_context,
            "prompt_text": prompt_text,
            "chain_state": chain_state,
        }
        reply = self._request_sync_fallback(payload)
        if reply is None:
            # Worker unavailable / timeout. Return a hard-fail OVGResult so
            # callers' downstream "if passed:" branches behave correctly.
            logger.warning("[OutputVerifierProxy] verify_and_sign timeout — returning hard-fail OVGResult")
            return OVGResult(
                passed=False,
                output_text=output_text,
                signature=None,
                violation_type="proxy_unavailable",
                violations=["output_verifier worker unavailable (bus request timeout)"],
                channel=channel,
                guard_message="[VERIFICATION UNAVAILABLE — request was not processed]",
            )
        body = reply.get("payload") or {}
        if "error" in body:
            logger.warning("[OutputVerifierProxy] worker reported error: %s", body["error"])
            return OVGResult(
                passed=False,
                output_text=output_text,
                signature=None,
                violation_type="proxy_error",
                violations=[str(body["error"])],
                channel=channel,
                guard_message=f"[VERIFICATION ERROR: {body['error']}]",
            )
        # Reconstruct OVGResult from dataclass dict.
        try:
            return OVGResult(**body)
        except TypeError as e:
            logger.warning("[OutputVerifierProxy] OVGResult deserialize failed: %s", e)
            return OVGResult(
                passed=False, output_text=output_text, signature=None,
                violation_type="proxy_decode_error",
                violations=[str(e)], channel=channel,
                guard_message="[VERIFICATION DECODE ERROR]",
            )

    def build_timechain_payload(self, result, **kwargs) -> dict:
        """Mirror of OutputVerifier.build_timechain_payload.

        Sync entry point — uses async work-RPC underneath."""
        import dataclasses
        result_dict = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else dict(result)
        payload = {
            "action": "build_timechain_payload",
            "result_dict": result_dict,
            "kwargs": kwargs,
        }
        reply = self._request_sync_fallback(payload)
        if reply is None:
            return {}
        body = reply.get("payload") or {}
        return body.get("timechain_payload") or {}

    def get_stats(self) -> dict:
        """SHM read of output_verifier_state.bin (Session 3 publisher).
        Preamble G18 — state transport is SHM, never bus.

        Most callers should use the cached attribute access (verified_count
        etc.) which refreshes via OUTPUT_VERIFIER_STATS broadcasts.
        """
        try:
            raw = self._r_ovg_state.read_variable()
        except Exception as e:
            self._track_fallback("output_verifier_state",
                                 f"read_raised:{type(e).__name__}")
            return {}
        if raw is None:
            self._track_fallback("output_verifier_state", "shm_unavailable")
            return {}
        try:
            decoded = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            self._track_fallback("output_verifier_state",
                                 f"decode_raised:{type(e).__name__}")
            return {}
        return decoded if isinstance(decoded, dict) else {}

    def update_cached_stats(self, payload: dict) -> None:
        """Called by parent's bus subscription when OUTPUT_VERIFIER_STATS arrives."""
        try:
            self.verified_count = int(payload.get("verified_count", 0) or 0)
            self.rejected_count = int(payload.get("rejected_count", 0) or 0)
        except Exception:
            pass


# ── D-SPEC-74 reply decoders ─────────────────────────────────────────

def _decode_safety_result(reply, output_text: str, channel: str):
    """Reply envelope → SafetyResult. Hard-fail on worker unavailable."""
    from ..logic.output_verifier import SafetyResult
    if reply is None:
        logger.warning(
            "[OutputVerifierProxy] verify_safety_async timeout — "
            "returning hard-fail SafetyResult")
        return SafetyResult(
            passed=False, output_text=output_text, channel=channel,
            violation_type="proxy_unavailable",
            violations=["output_verifier worker unavailable"],
            guard_message="[VERIFICATION UNAVAILABLE]",
        )
    body = reply.get("payload") or {}
    if "error" in body:
        logger.warning(
            "[OutputVerifierProxy] verify_safety worker error: %s",
            body["error"])
        return SafetyResult(
            passed=False, output_text=output_text, channel=channel,
            violation_type="proxy_error",
            violations=[str(body["error"])],
            guard_message=f"[VERIFICATION ERROR: {body['error']}]",
        )
    try:
        return SafetyResult(**body)
    except TypeError as e:
        logger.warning(
            "[OutputVerifierProxy] SafetyResult deserialize failed: %s", e)
        return SafetyResult(
            passed=False, output_text=output_text, channel=channel,
            violation_type="proxy_decode_error",
            violations=[str(e)],
            guard_message="[VERIFICATION DECODE ERROR]",
        )


def _decode_signed_result(reply):
    """Reply envelope → SignedResult."""
    from ..logic.output_verifier import SignedResult
    if reply is None:
        return SignedResult(signed=False, signature=None,
                            error="proxy_unavailable")
    body = reply.get("payload") or {}
    if "error" in body:
        return SignedResult(signed=False, signature=None,
                            error=str(body["error"]))
    try:
        return SignedResult(**body)
    except TypeError as e:
        return SignedResult(signed=False, signature=None,
                            error=f"decode_error:{e}")
