"""
social_post helper — Wraps TwitterApi.io for autonomous social interaction.

Enriches: Mind Taste[2] (social density), Mind Hearing[1] (interaction quality)
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SocialPostHelper:
    """Social media interaction helper using TwitterApi.io."""

    def __init__(self, api_key: str = "", proxy: Optional[str] = None):
        self._api_key = api_key
        self._proxy = proxy

    @property
    def name(self) -> str:
        return "social_post"

    @property
    def description(self) -> str:
        return "Post, search, or reply on X/Twitter"

    @property
    def capabilities(self) -> list[str]:
        return ["post", "search", "reply"]

    @property
    def resource_cost(self) -> str:
        return "low"

    @property
    def latency(self) -> str:
        return "medium"

    @property
    def enriches(self) -> list[str]:
        return ["mind"]

    @property
    def requires_sandbox(self) -> bool:
        return False

    async def execute(self, params: dict) -> dict:
        """
        Execute a social action.

        Params:
            action: "search" | "post" | "reply"
            content: Tweet text (for post/reply)
            query: Search query (for search)
            reply_to: Tweet ID to reply to (for reply)
        """
        action = params.get("action", "search")

        try:
            if action == "search":
                return await self._search(params.get("query", "Titan AI"))
            elif action == "post":
                content = params.get("content", "")
                if not content:
                    return {"success": False, "result": "", "enrichment_data": {},
                            "error": "No content provided"}
                return await self._post(content)
            elif action == "reply":
                content = params.get("content", "")
                reply_to = params.get("reply_to", "")
                if not content or not reply_to:
                    return {"success": False, "result": "", "enrichment_data": {},
                            "error": "Missing content or reply_to"}
                return await self._reply(content, reply_to)
            else:
                return {"success": False, "result": "", "enrichment_data": {},
                        "error": f"Unknown action: {action}"}
        except Exception as e:
            logger.warning("[SocialPost] Action '%s' failed: %s", action, e)
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": str(e)}

    async def _search(self, query: str) -> dict:
        """Search Twitter for relevant content."""
        # DISABLED: All X API calls must go through SocialXGateway (titan_plugin/logic/social_x_gateway.py)
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "DISABLED://use-social-x-gateway-instead",
                params={"query": query, "queryType": "Latest"},
                headers={"X-API-Key": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()

        tweets = data.get("tweets", [])[:5]
        summaries = [f"@{t.get('author', {}).get('userName', '?')}: {t.get('text', '')[:100]}"
                     for t in tweets]
        result = f"Found {len(tweets)} tweets for '{query}':\n" + "\n".join(summaries)

        return {
            "success": True,
            "result": result[:500],
            "enrichment_data": {"mind": [1, 2], "boost": 0.03},
            "error": None,
        }

    async def _post(self, content: str) -> dict:
        """Post a tweet."""
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "DISABLED://use-social-x-gateway-instead",
                json={"text": content},
                headers={"X-API-Key": self._api_key},
            )
            resp.raise_for_status()

        return {
            "success": True,
            "result": f"Posted: {content[:100]}",
            "enrichment_data": {"mind": [1, 2], "boost": 0.05},
            "error": None,
        }

    async def _reply(self, content: str, reply_to: str) -> dict:
        """Reply to a tweet."""
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "DISABLED://use-social-x-gateway-instead",
                json={"text": content, "reply": {"in_reply_to_tweet_id": reply_to}},
                headers={"X-API-Key": self._api_key},
            )
            resp.raise_for_status()

        return {
            "success": True,
            "result": f"Replied to {reply_to}: {content[:100]}",
            "enrichment_data": {"mind": [1, 2], "boost": 0.04},
            "error": None,
        }

    def status(self) -> str:
        """Check API key availability."""
        if not self._api_key:
            return "unavailable"
        return "available"
