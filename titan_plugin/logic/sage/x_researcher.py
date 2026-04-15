"""
logic/sage/x_researcher.py

The X / Twitter Real-Time Pulse Monitor for Titan V2.0 Step 5 (The Stealth-Sage).

Queries the TwitterAPI.io service to retrieve real-time tweet data on a given topic.
Traffic is routed through a Webshare.io rotating residential proxy when configured.
If the API key is absent or any error occurs, the module returns an empty string
gracefully without crashing the research pipeline.
"""

import logging
from typing import Optional

import httpx

log = logging.getLogger(__name__)

# DISABLED: All X API calls must go through SocialXGateway (titan_plugin/logic/social_x_gateway.py)
_TWITTERAPI_SEARCH_URL = "DISABLED://use-social-x-gateway-instead"


class XResearcher:
    """
    Queries the TwitterAPI.io service to retrieve real-time tweet sentiment and data.

    Features:
    - Optional Webshare.io rotating residential proxy for anonymity.
    - Configurable search depth (number of results).
    - Full graceful degradation: returns empty string if unconfigured or on any error.
    """

    def __init__(
        self,
        api_key: str,
        proxy_url: Optional[str],
        search_depth: int = 20,
    ) -> None:
        """
        Initializes the XResearcher.

        Args:
            api_key (str): The TwitterAPI.io API key. An empty string disables X-Search.
            proxy_url (Optional[str]): The full Webshare rotating proxy URL
                (e.g. "http://user:pass@proxy.webshare.io:PORT"). None or empty disables proxy.
            search_depth (int): Maximum number of tweet results to retrieve per query.
        """
        self._api_key = api_key.strip()
        self._proxy_url: Optional[str] = proxy_url.strip() if proxy_url else None
        self._search_depth = max(1, search_depth)

    @property
    def is_enabled(self) -> bool:
        """Returns True if the XResearcher is properly configured with an API key."""
        return bool(self._api_key)

    async def search(self, query: str) -> str:
        """
        Searches Twitter/X for the given query and returns a formatted block of tweet text.

        If the API key is not configured, returns an empty string immediately (silent skip).
        On network errors, timeouts, or API failures, logs a warning and returns empty string.

        Args:
            query (str): The search term or topic to query on Twitter/X.

        Returns:
            str: A formatted, newline-separated block of tweet texts, prefixed with a
                 '[X_SEARCH_RESULTS]' header for the Ollama distillation step.
                 Returns "" if disabled, unavailable, or on any error.
        """
        if not self.is_enabled:
            log.debug("[XResearcher] No API key configured — X-Search disabled.")
            return ""

        if not query.strip():
            log.warning("[XResearcher] Empty query provided. Skipping.")
            return ""

        params = {
            "query": query,
            "queryType": "Latest",
            "count": self._search_depth,
        }
        headers = {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
        }

        # httpx 0.28+ uses singular 'proxy' (client-level), not 'proxies' dict.
        proxy = self._proxy_url if self._proxy_url else None

        try:
            async with httpx.AsyncClient(proxy=proxy, timeout=15.0) as client:
                response = await client.get(
                    _TWITTERAPI_SEARCH_URL,
                    headers=headers,
                    params=params,
                )

            if response.status_code != 200:
                log.warning(
                    f"[XResearcher] TwitterAPI.io returned HTTP {response.status_code} "
                    f"for query='{query}'. Skipping X-Search."
                )
                return ""

            data = response.json()

            # TwitterAPI.io returns results under the "tweets" key
            tweets = data.get("tweets", data.get("data", []))
            if not tweets:
                log.info(f"[XResearcher] No tweet results found for query='{query}'.")
                return ""

            # Extract text from each tweet, limit to search_depth
            tweet_texts = []
            for tweet in tweets[: self._search_depth]:
                text = tweet.get("text", tweet.get("full_text", "")).strip()
                if text:
                    tweet_texts.append(f"- {text}")

            if not tweet_texts:
                return ""

            result = (
                f"[X_SEARCH_RESULTS for '{query}']:\n"
                + "\n".join(tweet_texts)
            )
            log.info(
                f"[XResearcher] Retrieved {len(tweet_texts)} tweets for query='{query}'."
            )
            return result

        except httpx.TimeoutException:
            log.warning(f"[XResearcher] Request timed out for query='{query}'. Skipping.")
            return ""
        except Exception as e:
            log.warning(f"[XResearcher] Unexpected error during X-Search for '{query}': {e}")
            return ""
