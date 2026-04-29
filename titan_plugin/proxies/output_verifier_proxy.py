"""
Output Verifier Proxy — bus-routed bridge to output_verifier_worker (L3 §A.8.3).

Drop-in interface match for `OutputVerifier`. When flag
`microkernel.a8_output_verifier_subprocess_enabled=true`, parent's
`_output_verifier` becomes this proxy; calls translate to bus
QUERY/RESPONSE round-trips against the output_verifier worker.

When flag is false (default), parent retains a local OutputVerifier
instance — no proxy is created, no behavior change.

Stats attributes (sovereignty_score, verified_count, rejected_count)
are kept cached locally and refreshed when OUTPUT_VERIFIER_STATS
broadcasts arrive. While the cache is empty (worker hasn't booted yet
or initial broadcast pending), they default to 0 — kernel.py:1082's
snapshot reader uses `getattr(..., default)` so missing values are tolerated.

See: titan-docs/rFP_microkernel_phase_a8_l2_l3_residency_completion.md §A.8.3
"""
from __future__ import annotations

import logging
from typing import Optional

from ..bus import DivineBus

logger = logging.getLogger(__name__)


class OutputVerifierProxy:
    """Bus-routed proxy that mirrors OutputVerifier's public surface.

    Methods:
        verify_and_sign(...)          → bus QUERY action="verify_and_sign"
        build_timechain_payload(...)  → bus QUERY action="build_timechain_payload"
        get_stats()                   → bus QUERY action="stats"

    Attributes (cached, updated by OUTPUT_VERIFIER_STATS broadcast):
        sovereignty_score, verified_count, rejected_count
    """

    def __init__(self, bus: DivineBus, request_timeout_s: float = 10.0):
        self._bus = bus
        self._timeout = float(request_timeout_s)
        self._reply_queue = bus.subscribe("output_verifier_proxy", reply_only=True)
        # Cached stats — updated from OUTPUT_VERIFIER_STATS broadcasts.
        self.sovereignty_score: float = 0.0
        self.verified_count: int = 0
        self.rejected_count: int = 0

    def verify_and_sign(self, output_text: str, channel: str,
                        injected_context: str = "",
                        prompt_text: str = "",
                        chain_state: Optional[dict] = None):
        """Mirror of OutputVerifier.verify_and_sign — returns OVGResult."""
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
        reply = self._bus.request(
            src="output_verifier_proxy",
            dst="output_verifier",
            payload=payload,
            timeout=self._timeout,
            reply_queue=self._reply_queue,
        )
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
        """Mirror of OutputVerifier.build_timechain_payload."""
        import dataclasses
        result_dict = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else dict(result)
        payload = {
            "action": "build_timechain_payload",
            "result_dict": result_dict,
            "kwargs": kwargs,
        }
        reply = self._bus.request(
            src="output_verifier_proxy",
            dst="output_verifier",
            payload=payload,
            timeout=self._timeout,
            reply_queue=self._reply_queue,
        )
        if reply is None:
            return {}
        body = reply.get("payload") or {}
        return body.get("timechain_payload") or {}

    def get_stats(self) -> dict:
        """Force-fetch fresh stats (synchronous round-trip).

        Most callers should use the cached attribute access (sovereignty_score
        etc.) which refreshes via OUTPUT_VERIFIER_STATS broadcasts.
        """
        reply = self._bus.request(
            src="output_verifier_proxy",
            dst="output_verifier",
            payload={"action": "stats"},
            timeout=self._timeout,
            reply_queue=self._reply_queue,
        )
        if reply is None:
            return {}
        return reply.get("payload") or {}

    def update_cached_stats(self, payload: dict) -> None:
        """Called by parent's bus subscription when OUTPUT_VERIFIER_STATS arrives."""
        try:
            self.sovereignty_score = float(payload.get("sovereignty_score", 0.0) or 0.0)
            self.verified_count = int(payload.get("verified_count", 0) or 0)
            self.rejected_count = int(payload.get("rejected_count", 0) or 0)
        except Exception:
            pass
