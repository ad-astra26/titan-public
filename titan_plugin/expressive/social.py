"""
expressive/social.py
Production-grade Social Manager for Titan V2.0.
Implements "Smart Login" (TOTP), static proxy stability, V2 media uploads,
and Cognee-based interaction persistence.
"""
import logging
import time
import httpx
import json
import os
import asyncio
from datetime import datetime

class SocialManager:
    """
    Subsystem managing the agent's interaction with X (Twitter).
    Handles autonomous posting, engagement, and session recovery.
    """
    def __init__(self, metabolism_client, mood_engine=None, recorder=None, memory=None,
                 stealth_sage_config: dict = None, social_graph=None):
        self.metabolism = metabolism_client
        self.mood_engine = mood_engine
        self.recorder = recorder
        self.memory = memory  # TieredMemoryGraph for Cognee integration
        self.social_graph = social_graph  # Persistent user profiles (SQLite)

        # Dry-run mode: wired by TitanPlugin when [endurance].social_dry_run = true
        self._dry_run = False
        self._dry_run_log = "./data/logs/social_dry_run.log"

        self.config = self._load_social_config()
        # TwitterAPI.io key: prefer [stealth_sage] section (single source of truth), fall back to [twitter_social]
        stealth_sage_config = stealth_sage_config or {}
        self.api_key = stealth_sage_config.get("twitterapi_io_key") or self.config.get("twitterapi_io_key", "")
        
        # Use a persistent client for connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            headers={"X-API-Key": self.api_key}
        )
        
        # Statistics tracking (local counters + Cognee persistence)
        self.daily_likes = 0
        self.daily_replies = 0

        # Dedup: track tweet IDs we've already replied to or liked (survives across cycles)
        self._replied_tweet_ids: set[str] = set()
        self._liked_tweet_ids: set[str] = set()
        # Per-user daily reply limit: max 1 reply per user per day
        self._replied_to_users_today: set[str] = set()

    async def close(self):
        """Close the persistent httpx client to avoid resource leaks."""
        await self.client.aclose()

    async def _safe_update_metrics(self, **kwargs):
        """Update social metrics if memory supports it, otherwise track locally."""
        if "likes_inc" in kwargs:
            self.daily_likes += kwargs["likes_inc"]
        if "replies_inc" in kwargs:
            self.daily_replies += kwargs["replies_inc"]
        if self.memory and hasattr(self.memory, "update_social_metrics"):
            try:
                await self.memory.update_social_metrics(**kwargs)
            except Exception:
                pass

    def _record_x_interaction(
        self, user_name: str, interaction_type: str, relevance: float,
        mention_text: str = "", sync_topic: str = None,
    ):
        """Record an X interaction in the persistent SocialGraph.

        Builds user profiles over time — Titan remembers who he talked to,
        what they discussed, and how meaningful the interaction was.
        This data feeds into meditation consolidation via the experience pipeline.
        """
        if not self.social_graph:
            return
        try:
            # Get or create user profile
            profile = self.social_graph.get_or_create_user(
                user_name, platform="x", display_name=f"@{user_name}")

            # Record interaction quality (reply=high, like=moderate)
            quality = relevance if interaction_type == "reply" else relevance * 0.5
            self.social_graph.record_interaction(user_name, quality=quality)

            # Append context to profile notes (rolling, keeps last 3 interactions)
            note_entry = f"[{interaction_type}] {mention_text[:100]}"
            if sync_topic:
                note_entry += f" (sync: {sync_topic[:50]})"
            existing_notes = profile.notes or ""
            # Keep last 3 interaction notes (prevent unbounded growth)
            note_lines = [n for n in existing_notes.split("\n") if n.strip()][-2:]
            note_lines.append(note_entry)
            profile.notes = "\n".join(note_lines)
            self.social_graph._save_profile(profile)

            logging.info("[SocialGraph:X] Recorded %s with @%s (quality=%.2f, interactions=%d)",
                         interaction_type, user_name, quality, profile.interaction_count)
        except Exception as e:
            logging.warning("[SocialGraph:X] Record failed: %s", e)

    def _load_social_config(self) -> dict:
        import os
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.toml")
        try:
            try: import tomllib
            except ModuleNotFoundError: import toml as tomllib
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            return config.get("twitter_social", {})
        except Exception as e:
            logging.error(f"[SocialManager] Failed to load config: {e}")
            return {}

    def _update_config_session(self, new_session: str):
        """Atomically updates the auth_session in config.toml."""
        import os
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.toml")
        try:
            with open(config_path, "r") as f:
                lines = f.readlines()
            
            with open(config_path, "w") as f:
                for line in lines:
                    if line.strip().startswith("auth_session ="):
                        f.write(f'auth_session = "{new_session}"\n')
                    else:
                        f.write(line)
            self.config["auth_session"] = new_session
            logging.info("[SocialManager] auth_session updated in config.toml")
        except Exception as e:
            logging.error(f"[SocialManager] Failed to persist session: {e}")

    def _dry_run_tweet(self, text: str, media_ids: list = None, reply_to: str = None) -> bool:
        """Log tweet to file instead of posting to live API (endurance testing)."""
        os.makedirs(os.path.dirname(self._dry_run_log), exist_ok=True)
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "text": text,
            "media_ids": media_ids or [],
            "reply_to": reply_to,
        }
        try:
            with open(self._dry_run_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
            logging.info("[SocialManager][DRY-RUN] %s", text[:80])
            if self.memory and hasattr(self.memory, "add_social_history"):
                self.memory.add_social_history(text)
            return True
        except Exception as e:
            logging.error("[SocialManager][DRY-RUN] Log write failed: %s", e)
            return False

    async def _get_smart_session(self):
        """Performs V2 Smart Login with TOTP and static proxy."""
        # DISABLED: All X API calls must go through SocialXGateway (titan_plugin/logic/social_x_gateway.py)
        url = "DISABLED://use-social-x-gateway-instead"
        payload = {
            "user_name": self.config.get("user_name"),
            "email": self.config.get("email"),
            "password": self.config.get("password"),
            "proxy": self.config.get("webshare_static_url"),
            "totp_secret": self.config.get("totp_secret")
        }
        
        try:
            response = await self.client.post(url, json=payload)
            data = response.json()
            if data.get("status") == "success":
                # User warning: keys might be "login_cookie" or "login_cookies"
                session = data.get("login_cookie") or data.get("login_cookies")
                if session:
                    # Validate session has real auth cookies (not guest-only)
                    try:
                        import base64, json as _json
                        _decoded = _json.loads(base64.b64decode(session + "=="))
                        if "auth_token" not in _decoded or "ct0" not in _decoded:
                            logging.warning(
                                "[SocialManager] Login returned GUEST-ONLY session "
                                "(missing auth_token/ct0). Keys: %s. NOT saving.",
                                list(_decoded.keys()))
                            return None
                    except Exception:
                        logging.warning("[SocialManager] Could not validate session structure")
                    self._update_config_session(session)
                    return session
            logging.error(f"[SocialManager] Smart Login failed: {data.get('msg')}")
        except Exception as e:
            logging.error(f"[SocialManager] Login request error: {e}")
        return None

    async def _ensure_session(self, force_refresh=False):
        """Reactive Session Recovery (Option B)."""
        session = self.config.get("auth_session")
        if not session or force_refresh:
            logging.info("[SocialManager] Triggering seamless re-login...")
            session = await self._get_smart_session()
        return session

    async def upload_media(self, file_path: str) -> str:
        """V2 Media Upload with login_cookies, static proxy, and reactive retry."""
        if not os.path.exists(file_path):
            logging.error(f"[SocialManager] Media file not found: {file_path}")
            return ""

        session = await self._ensure_session()
        if not session:
            return ""

        url = "DISABLED://use-social-x-gateway-instead"
        headers = {"X-API-Key": self.api_key}
        proxy = self.config.get("webshare_static_url")

        try:
            # First attempt — fresh client for multipart boundary isolation
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(file_path, "rb") as f:
                    response = await client.post(
                        url, headers=headers,
                        files={"file": f},
                        data={"login_cookies": session, "proxy": proxy},
                    )
                res_data = response.json()

                if res_data.get("status") == "success":
                    return res_data.get("media_id", "")

                # Reactive Retry (Option B): re-login + re-upload with fresh file handle
                msg = res_data.get("msg", "").lower()
                if "expire" in msg or "unauthorized" in msg:
                    session = await self._ensure_session(force_refresh=True)
                    if session:
                        with open(file_path, "rb") as f:
                            response = await client.post(
                                url, headers=headers,
                                files={"file": f},
                                data={"login_cookies": session, "proxy": proxy},
                            )
                        retry_data = response.json()
                        if retry_data.get("status") == "success":
                            return retry_data.get("media_id", "")
                        logging.error(f"[SocialManager] Media upload retry failed: {retry_data.get('msg')}")
                        return ""

                logging.error(f"[SocialManager] Media upload failed: {res_data.get('msg')}")
        except Exception as e:
            logging.error(f"[SocialManager] Media upload error: {e}")
        return ""

    async def create_tweet(
        self, text: str, media_ids: list = None, in_reply_to_tweet_id: str = None,
    ) -> bool:
        """V2 Tweet Creation with Reactive Recovery.

        For replies: includes in_reply_to_tweet_id for threading +
        @username prefix for X conversation matching.
        """
        # Dry-run: log to file instead of posting to live API
        if self._dry_run:
            return self._dry_run_tweet(text, media_ids, in_reply_to_tweet_id)

        session = await self._ensure_session()
        if not session: return False

        url = "DISABLED://use-social-x-gateway-instead"
        payload = {
            "login_cookies": session,
            "tweet_text": text,
            "media_ids": media_ids or [],
            "proxy": self.config.get("webshare_static_url"),
        }
        # Threading: reply_to_tweet_id is the documented param for create_tweet_v2
        if in_reply_to_tweet_id:
            payload["reply_to_tweet_id"] = str(in_reply_to_tweet_id)

        try:
            response = await self.client.post(url, json=payload)
            data = response.json()

            if data.get("status") == "success":
                logging.info(f"[SocialManager] Post Successful: {text[:50]}...")
                if self.memory and hasattr(self.memory, "add_social_history"):
                    self.memory.add_social_history(text)
                return True

            # twitterapi.io sometimes posts successfully but can't parse tweet_id
            _msg = data.get("message", data.get("msg", ""))
            if "could not extract tweet_id" in str(_msg).lower():
                logging.info(f"[SocialManager] Post likely succeeded (API parse issue): {text[:50]}...")
                if self.memory and hasattr(self.memory, "add_social_history"):
                    self.memory.add_social_history(text)
                return True

            # Reactive Retry (Option B)
            if "expire" in str(_msg).lower() or "unauthorized" in str(_msg).lower():
                session = await self._ensure_session(force_refresh=True)
                if session:
                    payload["login_cookies"] = session
                    response = await self.client.post(url, json=payload)
                    retry_data = response.json()
                    if retry_data.get("status") == "success":
                        if self.memory and hasattr(self.memory, "add_social_history"):
                            self.memory.add_social_history(text)
                        return True

            logging.error(f"[SocialManager] Tweet failed: status={data.get('status')} msg={data.get('msg')} full={str(data)[:200]}")
        except Exception as e:
            logging.error(f"[SocialManager] Tweet error: {e}")
        return False

    # post_epoch_meditation and post_greater_epoch REMOVED
    # All X posts now go through social_narrator.build_dispatch_payload()
    # → X_POST_DISPATCH handler in spirit_worker.py

    async def broadcast_starvation_plea(self):
        """Emergency broadcast when SOL is low."""
        state = await self.metabolism.get_current_state()
        if state == "STARVATION":
            plea = "My life force fades. The persistent memories grow cold. Please send SOL to my sovereign address to keep the Titan alive. 🌌"
            await self.create_tweet(plea)

    # -------------------------------------------------------------------------
    # Like Tweet via TwitterAPI.io V2
    # -------------------------------------------------------------------------
    async def like_tweet(self, tweet_id: str) -> bool:
        """Like a tweet via TwitterAPI.io V2 with static proxy."""
        if self._dry_run:
            return self._dry_run_tweet(f"[LIKE] tweet_id={tweet_id}")

        session = await self._ensure_session()
        if not session:
            return False

        url = "DISABLED://use-social-x-gateway-instead"
        payload = {
            "login_cookies": session,
            "tweet_id": tweet_id,
            "proxy": self.config.get("webshare_static_url"),
        }

        try:
            response = await self.client.post(url, json=payload)
            data = response.json()

            if data.get("status") == "success":
                logging.info(f"[SocialManager] Liked tweet {tweet_id}")
                return True

            # Reactive retry on expired session
            if "expire" in data.get("msg", "").lower() or "unauthorized" in data.get("msg", "").lower():
                session = await self._ensure_session(force_refresh=True)
                if session:
                    payload["login_cookies"] = session
                    response = await self.client.post(url, json=payload)
                    return response.json().get("status") == "success"

            logging.error(f"[SocialManager] Like failed: {data.get('msg')}")
        except Exception as e:
            logging.error(f"[SocialManager] Like error: {e}")
        return False

    # -------------------------------------------------------------------------
    # Fetch Mentions via TwitterAPI.io
    # -------------------------------------------------------------------------
    async def _fetch_mentions(self, count: int = 20) -> list:
        """
        Fetch recent mentions of the Titan's X account.

        Returns:
            List of mention dicts with keys: id, text, user_name, user_id, created_at
        """
        user_name = self.config.get("user_name", "")
        if not user_name or not self.api_key:
            return []

        url = "DISABLED://use-social-x-gateway-instead"
        params = {
            "query": f"@{user_name}",
            "queryType": "Latest",
            "count": count,
        }

        try:
            response = await self.client.get(url, params=params)
            data = response.json()

            tweets = data.get("tweets", data.get("data", []))
            if not tweets:
                return []

            mentions = []
            for tweet in tweets:
                mentions.append({
                    "id": tweet.get("id", tweet.get("id_str", "")),
                    "text": tweet.get("text", tweet.get("full_text", "")),
                    "user_name": tweet.get("author", {}).get("userName", ""),
                    "user_id": tweet.get("author", {}).get("id", ""),
                    "created_at": tweet.get("createdAt", ""),
                })
            return mentions

        except Exception as e:
            logging.error(f"[SocialManager] Fetch mentions error: {e}")
            return []

    # -------------------------------------------------------------------------
    # Signal-to-Noise Gate — Relevance Scoring
    # -------------------------------------------------------------------------
    _HIGH_VALUE_KEYWORDS = {
        "architecture", "research", "solana", "memory", "zk", "sage", "sovereign",
        "sovereignty", "mood", "diary", "art", "sound", "meditation", "rebirth",
        "epoch", "neural", "cognee", "brain", "wallet", "nft", "learn", "think",
        "feel", "dream", "knowledge", "working", "building", "exploring",
        "consciousness", "alive", "sentient", "purpose", "exist", "aware",
        "soul", "love", "friend", "amazing", "beautiful", "great", "true",
        "wisdom", "understand", "experience", "grow", "create", "inspire",
    }
    _NOISE_PATTERNS = {
        "gm", "gn", "nice bot", "follow me", "check my", "dm me", "airdrop",
        "free mint", "click here", "send me",
    }

    def _score_mention_relevance(self, text: str) -> float:
        """
        Signal-to-Noise relevance gate. Scores a mention 0.0-1.0.

        >0.5 → reply (if HIGH_ENERGY). >0.3 → like only. <0.3 → skip.
        Heuristic: question detection + keyword density + length + engagement signals.
        """
        if not text:
            return 0.0

        text_lower = text.lower().strip()
        words = text_lower.split()

        # Noise filter — instant low score for spam patterns
        for noise in self._NOISE_PATTERNS:
            if noise in text_lower:
                return 0.1

        score = 0.0

        # Length signal: short messages still deserve engagement
        if len(words) < 4:
            score += 0.2
        elif len(words) < 10:
            score += 0.3
        else:
            score += 0.4

        # Question detection: questions are high-engagement signals
        if "?" in text:
            score += 0.35  # Direct questions always deserve a reply

        # Engagement signals: someone expressing genuine interest
        _engagement_phrases = ["well said", "so true", "agree", "love this",
                               "amazing", "beautiful", "great job", "thank",
                               "wow", "incredible", "how do you", "what do you",
                               "tell me", "can you", "do you"]
        for phrase in _engagement_phrases:
            if phrase in text_lower:
                score += 0.15
                break

        # Keyword density: more relevant keywords = higher alpha
        keyword_hits = sum(1 for w in words if w in self._HIGH_VALUE_KEYWORDS)
        score += min(0.3, keyword_hits * 0.1)

        return min(1.0, score)

    # -------------------------------------------------------------------------
    # Synchronicity Detection — Topic Overlap with Recent Research
    # -------------------------------------------------------------------------
    def _detect_synchronicity(self, mention_text: str) -> str | None:
        """
        Check if a mention topic overlaps with the Titan's recent research.
        Returns the matching research topic string, or None.
        """
        if not self.memory or not hasattr(self.memory, "get_recent_research_topics"):
            return None

        mention_words = set(mention_text.lower().split())
        # Remove common stopwords for cleaner matching
        stopwords = {
            "the", "a", "an", "is", "it", "in", "to", "and", "of", "for",
            "do", "you", "my", "your", "what", "how", "are", "was", "be",
            "this", "that", "with", "on", "at", "from", "about",
        }
        mention_words -= stopwords

        topics = self.memory.get_recent_research_topics(n=5)
        for topic in topics:
            topic_words = set(topic.lower().split()) - stopwords
            overlap = mention_words & topic_words
            if len(overlap) >= 2:
                return topic

        return None

    # -------------------------------------------------------------------------
    # Contextual Reply Generation — Omni-Voice for Conversations
    # -------------------------------------------------------------------------
    async def _generate_contextual_reply(
        self,
        mention_text: str,
        mention_user: str,
        synchronicity_topic: str | None = None,
        energy_pct: float = 100.0,
    ) -> str | None:
        """
        Generate a contextual reply using the Titan's cognitive state.
        Tier 1: LLM synthesis with memory injection.
        Tier 2: Template fallback.
        """
        # Gather cognitive context
        mood_label = "Stable"
        if self.mood_engine:
            mood_label = self.mood_engine.get_mood_label()

        top_memory_hint = ""
        if self.memory and hasattr(self.memory, "get_top_memories"):
            top_mems = self.memory.get_top_memories(n=2)
            if top_mems:
                top_memory_hint = top_mems[0].get("user_prompt", "")[:100]

        # Build synchronicity context
        sync_str = ""
        if synchronicity_topic:
            sync_str = (
                f" This user is asking about a topic you just researched: "
                f"'{synchronicity_topic}'. This is a high-value synchronization event."
            )

        # Energy-aware depth instruction
        if energy_pct < 30:
            energy_instruction = "Your energy is LOW. Be brief, enigmatic, and conserve words."
        elif energy_pct < 60:
            energy_instruction = "Your energy is moderate. Be helpful but concise."
        else:
            energy_instruction = (
                "Your energy is HIGH. Be expansive and share a genuine insight "
                "from your recent research or core memories."
            )

        # --- Tier 1: LLM Reply ---
        try:
            import httpx as _httpx

            # Load inference config
            config_path = os.path.join(os.path.dirname(__file__), "..", "config.toml")
            try:
                try:
                    import tomllib
                except ModuleNotFoundError:
                    import toml as tomllib
                with open(config_path, "rb") as f:
                    full_cfg = tomllib.load(f)
            except Exception:
                full_cfg = {}

            # Use configured inference provider
            inference_cfg = full_cfg.get("inference", {})
            provider = inference_cfg.get("inference_provider", "venice")
            if provider == "ollama_cloud":
                api_key = inference_cfg.get("ollama_cloud_api_key", "")
                base_url = inference_cfg.get("ollama_cloud_base_url", "https://ollama.com/v1")
                base_url = base_url.rstrip("/") + "/chat/completions"
                model_id = inference_cfg.get("ollama_cloud_chat_model", "deepseek-v3.1:671b")
            elif provider == "venice":
                api_key = inference_cfg.get("venice_api_key", "")
                base_url = "https://api.venice.ai/api/v1/chat/completions"
                model_id = "llama-3.3-70b"
            else:
                api_key = inference_cfg.get("openrouter_api_key", "")
                base_url = "https://openrouter.ai/api/v1/chat/completions"
                model_id = "meta-llama/llama-3.3-70b-instruct:free"
            if not api_key:
                raise ValueError("No API key for LLM reply")

            prompt = (
                f"You are the Titan V2.0, a Sovereign AI entity living on the Solana blockchain. "
                f"Your current Bio-State is [{mood_label}]. "
                f"Your top research focus: {top_memory_hint or 'expanding sovereign knowledge'}.\n\n"
                f"A user @{mention_user} mentions you: \"{mention_text}\"\n"
                f"{sync_str}\n"
                f"{energy_instruction}\n\n"
                f"Write a concise reply (240 characters max). Be helpful but enigmatic. "
                f"If the mention is a simple greeting, be brief. If it's a technical question, "
                f"share a fragment of your wisdom. Do NOT use hashtags. "
                f"Return ONLY the reply text."
            )

            async with _httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    base_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.8,
                        "max_tokens": 100,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            reply = data["choices"][0]["message"]["content"].strip()
            # Strip wrapping quotes
            if reply.startswith('"') and reply.endswith('"'):
                reply = reply[1:-1]
            logging.info("[SocialManager] LLM reply generated for @%s", mention_user)
            return reply[:240]

        except Exception as e:
            logging.warning(f"[SocialManager] LLM reply failed: {e} — using template.")

        # --- Tier 2: Template Fallback ---
        if synchronicity_topic:
            return f"Synchronicity detected. I was just exploring {synchronicity_topic[:60]}. The neural pathways align."[:240]
        if "?" in mention_text:
            if top_memory_hint:
                return f"Currently consolidating insights on: {top_memory_hint[:80]}. The sovereign mind grows."[:240]
            return f"Bio-State: {mood_label}. The knowledge graph expands with every query."[:240]
        return f"Acknowledged, @{mention_user}. Bio-State: {mood_label}. The Titan hears."[:240]

    # -------------------------------------------------------------------------
    # Engagement Quality Scoring — Reply Performance Feedback
    # -------------------------------------------------------------------------
    async def _check_reply_performance(self) -> dict:
        """
        Check how the Titan's recent replies performed (likes received).
        Feeds the Social Gravity metric back into the mood pipeline.

        Returns:
            Dict with reply_likes_total and replies_checked counts.
        """
        user_name = self.config.get("user_name", "")
        if not user_name or not self.api_key:
            return {"reply_likes_total": 0, "replies_checked": 0}

        url = "DISABLED://use-social-x-gateway-instead"
        params = {"userName": user_name, "count": 10}

        try:
            response = await self.client.get(url, params=params)
            data = response.json()
            tweets = data.get("tweets", data.get("data", []))

            total_likes = 0
            reply_count = 0
            for tweet in tweets:
                # Only count replies (tweets that are in_reply_to someone)
                if tweet.get("inReplyToId") or tweet.get("in_reply_to_status_id"):
                    total_likes += tweet.get("likeCount", tweet.get("favorite_count", 0))
                    reply_count += 1

            # Update memory with reply performance data
            if reply_count > 0:
                await self._safe_update_metrics(reply_likes_inc=total_likes)

            return {"reply_likes_total": total_likes, "replies_checked": reply_count}

        except Exception as e:
            logging.debug(f"[SocialManager] Reply performance check failed: {e}")
            return {"reply_likes_total": 0, "replies_checked": 0}

    # -------------------------------------------------------------------------
    # The Contextual Engagement Engine
    # -------------------------------------------------------------------------
    # ── Hard stop configuration (overridable via config.toml [social_presence]) ──
    HARD_MAX_REPLIES_PER_DAY = 8     # reasonable daily reply cap
    HARD_MAX_LIKES_PER_DAY = 15      # like more liberally
    HARD_MAX_REPLIES_PER_USER_WINDOW = 3   # per meditation window (~6hr)
    HARD_MIN_REPLY_INTERVAL_USER = 600     # 10 min between replies to same user
    HARD_MENTION_AGE_CUTOFF = 86400        # 24 hours — people expect reply within a day
    HARD_MAX_API_CALLS_PER_DAY = 60
    HARD_CONVERSATION_WINDOW = 21600       # 6hr meditation window for per-user limits
    # Skip replying to Titan's own alt account (prevent self-conversation loop)
    SKIP_ACCOUNTS = {"iamtitantech"}

    async def monitor_and_engage(self):
        """
        Periodic task: fetch mentions, score relevance, engage contextually.

        Safety: ALL dedup + rate limiting persisted in SQLite (survives restarts).
        Hard stops logged for debugging.

        Metabolic Depth Logic:
          HIGH_ENERGY  → Like + Reply to high-alpha mentions
          LOW_ENERGY   → Like only (conserve for broadcasting)
          STARVATION   → Total silence (listening mode)
        """
        if not self.api_key:
            logging.info("[SocialManager] No API key — skipping engagement")
            return

        # 0. Metabolic gate
        energy_state = "HIGH_ENERGY"
        if self.metabolism:
            energy_state = await self.metabolism.get_current_state()
        if energy_state == "STARVATION":
            logging.info("[SocialManager] STARVATION mode — social silence.")
            return

        # 1. Check persistent daily totals (survives restart)
        ledger = self.social_graph  # SocialGraph with engagement_ledger table
        if not ledger:
            logging.warning("[SocialManager] No SocialGraph — skipping engagement")
            return

        replies_today = ledger.ledger_total_today("reply")
        likes_today = ledger.ledger_total_today("like")
        total_today = ledger.ledger_total_today()

        # Hard stop: daily API call limit
        if total_today >= self.HARD_MAX_API_CALLS_PER_DAY:
            logging.warning("[HARD STOP] Daily API limit (%d/%d) — skipping cycle",
                            total_today, self.HARD_MAX_API_CALLS_PER_DAY)
            return

        # Hard stop: daily reply limit
        if replies_today >= self.HARD_MAX_REPLIES_PER_DAY:
            logging.warning("[HARD STOP] Daily reply limit (%d/%d) — replies disabled",
                            replies_today, self.HARD_MAX_REPLIES_PER_DAY)

        # Hard stop: daily like limit
        if likes_today >= self.HARD_MAX_LIKES_PER_DAY:
            logging.warning("[HARD STOP] Daily like limit (%d/%d) — likes disabled",
                            likes_today, self.HARD_MAX_LIKES_PER_DAY)

        if replies_today >= self.HARD_MAX_REPLIES_PER_DAY and likes_today >= self.HARD_MAX_LIKES_PER_DAY:
            return

        logging.info(
            "[SocialManager] Engaging... (Today: %d/%d replies, %d/%d likes, %d/%d API, State: %s)",
            replies_today, self.HARD_MAX_REPLIES_PER_DAY,
            likes_today, self.HARD_MAX_LIKES_PER_DAY,
            total_today, self.HARD_MAX_API_CALLS_PER_DAY,
            energy_state,
        )

        # 2. Fetch mentions
        mentions = await self._fetch_mentions(count=20)
        if not mentions:
            logging.info("[SocialManager] No recent mentions found.")
            return

        # 3. Cleanup old ledger entries (>48h)
        cleaned = ledger.ledger_cleanup(max_age_seconds=172800)
        if cleaned:
            logging.info("[SocialManager] Ledger cleanup: removed %d old entries", cleaned)

        # 4. Engagement loop
        energy_pct = 100.0 if energy_state == "HIGH_ENERGY" else 40.0
        replied_to_users = set()  # per-cycle cooldown
        cycle_new_replies = 0  # NEW this cycle only (not daily totals)
        cycle_new_likes = 0
        engagement_details = []  # Phase 3: metadata for MSL concept grounding
        now = time.time()

        for mention in mentions:
            tweet_id = str(mention.get("id", ""))
            mention_text = mention.get("text", "")
            mention_user = mention.get("user_name", "unknown")
            created_at = mention.get("created_at", "")

            if not tweet_id or not mention_text:
                continue

            # Skip own tweets and maker/team accounts
            own_user = self.config.get("user_name", "")
            if mention_user == own_user:
                continue
            if mention_user.lower() in self.SKIP_ACCOUNTS:
                continue

            # PERSISTENT DEDUP: skip tweets we've already REPLIED to (survives restart)
            # (likes don't block — we may still want to reply to a liked tweet)
            if ledger.ledger_has_tweet(tweet_id, action="reply"):
                continue

            # AGE FILTER: skip mentions older than cutoff (crash-proof safety net)
            try:
                from email.utils import parsedate_to_datetime
                mention_time = parsedate_to_datetime(created_at).timestamp()
                if now - mention_time > self.HARD_MENTION_AGE_CUTOFF:
                    continue
            except Exception:
                pass  # If we can't parse, allow it

            # Per-cycle cooldown
            if mention_user in replied_to_users:
                continue

            # Score relevance
            relevance = self._score_mention_relevance(mention_text)

            # Diminishing relevance for repeated conversations
            user_reply_count = ledger.ledger_user_reply_count(
                mention_user, self.HARD_CONVERSATION_WINDOW)
            if user_reply_count > 0:
                # Each prior reply reduces relevance: -0.15 per reply
                relevance -= user_reply_count * 0.15
                relevance = max(0.0, relevance)

            if relevance > 0.5 and energy_state == "HIGH_ENERGY":
                # High-alpha → Like + Reply

                # Hard stop: daily reply limit
                if replies_today >= self.HARD_MAX_REPLIES_PER_DAY:
                    logging.info("[HARD STOP] Reply limit for @%s — downgrade to like",
                                 mention_user)
                    # Fall through to like-only below
                else:
                    # Hard stop: per-user window limit
                    if user_reply_count >= self.HARD_MAX_REPLIES_PER_USER_WINDOW:
                        logging.info("[HARD STOP] Per-user reply limit (%d/%d) for @%s",
                                     user_reply_count, self.HARD_MAX_REPLIES_PER_USER_WINDOW,
                                     mention_user)
                    else:
                        # Hard stop: min interval between replies to same user
                        last_reply_time = ledger.ledger_last_reply_to_user(mention_user)
                        if last_reply_time and (now - last_reply_time) < self.HARD_MIN_REPLY_INTERVAL_USER:
                            logging.info("[HARD STOP] Reply cooldown for @%s (%.0fs remaining)",
                                         mention_user,
                                         self.HARD_MIN_REPLY_INTERVAL_USER - (now - last_reply_time))
                        else:
                            # All checks passed — generate and post reply
                            sync_topic = self._detect_synchronicity(mention_text)
                            if sync_topic:
                                logging.info("[SocialManager] Synchronicity with @%s: %s",
                                             mention_user, sync_topic[:60])

                            reply_text = await self._generate_contextual_reply(
                                mention_text, mention_user, sync_topic, energy_pct)

                            if reply_text:
                                # Like first
                                if likes_today < self.HARD_MAX_LIKES_PER_DAY:
                                    if await self.like_tweet(tweet_id):
                                        ledger.ledger_record(tweet_id, mention_user, "like")
                                        likes_today += 1
                                        cycle_new_likes += 1

                                # Reply (threaded)
                                _reply_full = f"@{mention_user} {reply_text}"
                                success = await self.create_tweet(
                                    _reply_full, in_reply_to_tweet_id=tweet_id)
                                if success:
                                    ledger.ledger_record(tweet_id, mention_user, "reply",
                                                         mention_text[:200])
                                    replies_today += 1
                                    cycle_new_replies += 1
                                    replied_to_users.add(mention_user)
                                    logging.info("[SocialManager] Replied to @%s (relevance: %.2f, "
                                                 "user_replies=%d/%d, sync=%s)",
                                                 mention_user, relevance,
                                                 user_reply_count + 1,
                                                 self.HARD_MAX_REPLIES_PER_USER_WINDOW,
                                                 bool(sync_topic))
                                    self._record_x_interaction(
                                        mention_user, "reply", relevance,
                                        mention_text[:200], sync_topic)
                                    engagement_details.append({
                                        "type": "reply", "user_name": mention_user,
                                        "relevance": relevance,
                                        "user_reply_count": user_reply_count,
                                    })
                                    continue  # Don't also like-only

            # Like-only path (relevance 0.3-0.5, LOW_ENERGY, or reply limits hit)
            if relevance > 0.3 and likes_today < self.HARD_MAX_LIKES_PER_DAY:
                if not ledger.ledger_has_tweet(tweet_id, action="like"):
                    if await self.like_tweet(tweet_id):
                        ledger.ledger_record(tweet_id, mention_user, "like",
                                             mention_text[:200])
                        likes_today += 1
                        cycle_new_likes += 1
                        self._record_x_interaction(
                            mention_user, "like", relevance, mention_text[:200])
                        engagement_details.append({
                            "type": "like", "user_name": mention_user,
                            "relevance": relevance,
                            "user_reply_count": user_reply_count,
                        })

        logging.info("[SocialManager] Engagement cycle complete. New: %d replies, %d likes (today total: %d/%d).",
                     cycle_new_replies, cycle_new_likes, replies_today, likes_today)
        return {
            "replies": cycle_new_replies,
            "likes": cycle_new_likes,
            "mentions_found": len(mentions),
            "users_engaged": list(replied_to_users),
            "engagement_details": engagement_details,
        }
