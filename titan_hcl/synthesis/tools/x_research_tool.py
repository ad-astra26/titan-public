"""Phase 6 — `x_research` ToolPlug (§P6.I).

Wraps the existing `SocialXGateway` (sole sanctioned X path per
`feedback_social_x_gateway_post_is_sole_sanctioned_x_path`) as a
`ToolPlug`. Doubles as `TruthOraclePlug` via the P6.E `x_oracle`
wrapper (same gateway instance, two surfaces).

Capabilities: ``["post", "fetch_thread", "fetch_topic", "fetch_account"]``.

Per arch §11.3 + INV-12: this is the engine's only sanctioned write
path to X. Pre-P6 the gateway existed but was invoked from multiple
producers; P6 routes all those invocations through this ToolPlug so
every X action becomes a procedural-fork TX (auditable) per INV-12
"every tool call anchored + skill-compilable".
"""
from __future__ import annotations

import logging
from typing import Optional

from titan_hcl.synthesis.oracles.x_oracle import GatewayLike, XOracle
from titan_hcl.synthesis.plugs import ToolCall, TruthOraclePlug
from titan_hcl.synthesis.tools.base import ToolPlugBase

logger = logging.getLogger(__name__)


SUPPORTED_CAPABILITIES = frozenset(
    {"post", "fetch_thread", "fetch_topic", "fetch_account"}
)


class XResearchTool(ToolPlugBase):
    tool_id: str = "x_research"
    _capabilities = ("post", "fetch_thread", "fetch_topic", "fetch_account")

    def __init__(
        self,
        *,
        writer,
        router=None,
        gateway: GatewayLike,
        api_key: str = "",
        x_oracle: Optional[XOracle] = None,
    ):
        super().__init__(writer=writer, router=router)
        self._gateway = gateway
        self._api_key = api_key
        # Share the same gateway with the oracle wrapper so one process,
        # two surfaces (arch §11.3).
        self._oracle = x_oracle or XOracle(gateway, api_key=api_key)

    def oracle(self) -> Optional[TruthOraclePlug]:
        """Doubling-as-oracle surface (arch §11.1 / §11.3)."""
        return self._oracle

    def _execute(self, call: ToolCall) -> dict:
        capability = str(call.args.get("capability", "")).strip()
        if not capability:
            return {"success": False, "result_summary": "missing 'capability' arg"}
        if capability not in SUPPORTED_CAPABILITIES:
            return {
                "success": False,
                "result_summary": (
                    f"unsupported capability {capability!r} (want one of "
                    f"{sorted(SUPPORTED_CAPABILITIES)})"
                ),
            }
        try:
            if capability == "post":
                return self._do_post(call)
            if capability == "fetch_thread":
                return self._do_fetch_thread(call)
            if capability == "fetch_topic":
                return self._do_fetch_topic(call)
            if capability == "fetch_account":
                return self._do_fetch_account(call)
        except Exception as exc:
            logger.exception("[x_research_tool] capability %s raised", capability)
            return {
                "success": False,
                "result_summary": f"{capability} raised: {exc}",
                "exception": str(exc),
            }
        return {"success": False, "result_summary": "unreachable"}  # pragma: no cover

    # ── capability handlers ────────────────────────────────────────────

    def _do_post(self, call: ToolCall) -> dict:
        """Delegate to SocialXGateway's sanctioned post path.

        SocialXGateway is the only allowed X writer per memory rule;
        the gateway returns a dict with status + tweet_id on success.
        """
        text = str(call.args.get("text", "")).strip()
        if not text:
            return {"success": False, "result_summary": "missing 'text' arg for post"}
        # Use whatever post method the gateway exposes (production path is
        # `post()` with a Candidate object; tests inject a simpler shape).
        post_fn = getattr(self._gateway, "post", None) or getattr(
            self._gateway, "post_text", None,
        )
        if post_fn is None:
            return {
                "success": False,
                "result_summary": "gateway has no post / post_text method",
            }
        result = post_fn(text=text, api_key=self._api_key)
        if not isinstance(result, dict):
            return {"success": False, "result_summary": "post returned non-dict"}
        status = str(result.get("status", "")).lower()
        success = status in ("ok", "posted", "success")
        return {
            "success": success,
            "result_summary": (
                f"posted: tweet_id={result.get('tweet_id', '')}" if success
                else f"post failed: {result.get('message', status)}"
            ),
        }

    def _do_fetch_thread(self, call: ToolCall) -> dict:
        thread_root = str(call.args.get("thread_root_id", "")).strip()
        if not thread_root:
            return {"success": False, "result_summary": "missing 'thread_root_id'"}
        resp = self._gateway.search_tweets(
            query=thread_root, query_type="Latest", count=20, api_key=self._api_key,
        )
        if not isinstance(resp, dict) or resp.get("status", "").lower() in ("error", "failed"):
            return {"success": False, "result_summary": f"gateway error: {resp}"}
        tweets = resp.get("tweets") or resp.get("data") or []
        return {
            "success": True,
            "result_summary": f"thread {thread_root}: {len(tweets)} replies",
            "result_full_payload": str(resp),
        }

    def _do_fetch_topic(self, call: ToolCall) -> dict:
        topic = str(call.args.get("topic", "")).strip()
        if not topic:
            return {"success": False, "result_summary": "missing 'topic'"}
        resp = self._gateway.search_tweets(
            query=topic, query_type="Latest", count=20, api_key=self._api_key,
        )
        if not isinstance(resp, dict) or resp.get("status", "").lower() in ("error", "failed"):
            return {"success": False, "result_summary": f"gateway error: {resp}"}
        tweets = resp.get("tweets") or resp.get("data") or []
        return {
            "success": True,
            "result_summary": f"topic {topic!r}: {len(tweets)} tweets",
            "result_full_payload": str(resp),
        }

    def _do_fetch_account(self, call: ToolCall) -> dict:
        handle = str(call.args.get("handle", "")).strip().lstrip("@")
        if not handle:
            return {"success": False, "result_summary": "missing 'handle'"}
        resp = self._gateway.fetch_recent_tweets(
            user_name=handle, count=int(call.args.get("count", 10)), api_key=self._api_key,
        )
        if not isinstance(resp, dict) or resp.get("status", "").lower() in ("error", "failed"):
            return {"success": False, "result_summary": f"gateway error: {resp}"}
        tweets = resp.get("tweets") or []
        return {
            "success": True,
            "result_summary": f"account @{handle}: {len(tweets)} recent tweets",
            "result_full_payload": str(resp),
        }


__all__ = ("XResearchTool", "SUPPORTED_CAPABILITIES")
