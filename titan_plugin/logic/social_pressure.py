"""
social_pressure.py — Urge-driven social posting for Titan's X presence.

Titan doesn't post on a schedule. He posts when accumulated SOCIAL expression
pressure meets a meaningful inner catalyst (EUREKA, strong composition, dream
reflection, emotion shift, on-chain anchor, etc.). This creates posts that
emerge from genuinely different felt-states each time.
"""
import base64
import json
import os
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── X Session Manager ─────────────────────────────────────────────────────

# PERSISTENCE_BY_DESIGN: XSessionManager state is intentionally reset on
# restart — circuit-breaker state (consecutive_failures, circuit_open_until)
# is recomputed from live refresh attempts, session validity is re-derived
# from _cached_session via _validate_session(), refresh timestamps are
# session-scoped. The actual session cookie persists via config.toml regex-
# write path (not self-assignment). See Tier B triage 2026-04-19.
class XSessionManager:
    """Permanent session lifecycle management for X/Twitter API.

    Handles: validation, periodic refresh, circuit breaking, config persistence.
    Wired into SocialPressureMeter and X_POST_DISPATCH handler.

    A valid session is a base64-encoded JSON with these 8 keys:
    __cuid, kdt, guest_id, __cf_bm, att, twid, ct0, auth_token

    Sessions with only 4 keys (missing auth_token/ct0/kdt/twid) are
    guest-only and CANNOT post tweets. The auto-refresh via twitterapi.io
    user_login_v2 is flaky — sometimes returns full sessions, sometimes
    guest-only. This manager retries periodically and only saves valid ones.

    See: titan-docs/social_x_architecture.md
    """

    REQUIRED_KEYS = {"auth_token", "ct0"}  # Minimum for posting
    FULL_KEYS = {"auth_token", "ct0", "kdt", "twid", "__cuid", "guest_id", "__cf_bm", "att"}

    def __init__(self, config_path: str = "./titan_plugin/config.toml",
                 refresh_interval: int = 3600,
                 max_consecutive_failures: int = 5,
                 backoff_seconds: int = 7200):
        self._config_path = config_path
        self._refresh_interval = refresh_interval      # Try refresh every N seconds
        self._max_failures = max_consecutive_failures   # Trip circuit after N failures
        self._backoff_seconds = backoff_seconds         # Cooldown after circuit trips

        # State
        self._cached_session: str = ""
        self._session_valid: bool = False
        self._session_keys: list[str] = []
        self._last_refresh_attempt: float = 0.0
        self._last_successful_refresh: float = 0.0
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0
        self._total_refresh_attempts: int = 0
        self._total_successful_refreshes: int = 0

        # Load current session from config
        self._load_session_from_config()

    def _load_session_from_config(self) -> None:
        """Read auth_session from merged config and validate."""
        try:
            from titan_plugin.config_loader import load_titan_config
            cfg = load_titan_config()
            session = cfg.get("twitter_social", {}).get("auth_session", "")
            if session:
                self._cached_session = session
                self._session_valid, self._session_keys = self._validate_session(session)
                if self._session_valid:
                    logger.info("[XSession] Loaded valid session (%d keys: %s)",
                                len(self._session_keys), self._session_keys)
                else:
                    logger.warning("[XSession] Loaded INVALID session (keys: %s) — "
                                   "will attempt refresh", self._session_keys)
        except Exception as e:
            logger.warning("[XSession] Failed to load config: %s", e)

    @staticmethod
    def _validate_session(session: str) -> tuple[bool, list[str]]:
        """Validate that a session has real auth cookies, not just guest-level.

        Returns (is_valid, list_of_keys).
        """
        if not session:
            return False, []
        try:
            # Base64 decode (add padding)
            decoded = json.loads(base64.b64decode(session + "=="))
            keys = list(decoded.keys())
            is_valid = all(k in decoded for k in XSessionManager.REQUIRED_KEYS)
            return is_valid, keys
        except Exception:
            return False, []

    def get_session(self) -> str | None:
        """Get the current valid session, or None if no valid session available.

        This is the main entry point — call this before every API call.
        If session is invalid or stale, triggers a background refresh attempt.
        """
        # If we have a valid cached session, return it
        if self._cached_session and self._session_valid:
            return self._cached_session

        # No valid session — try refresh if cooldown allows
        self._maybe_refresh()
        return self._cached_session if self._session_valid else None

    def on_post_success(self) -> None:
        """Called after a successful tweet. Resets failure counters."""
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        logger.info("[XSession] Post success — circuit breaker reset")

    def on_post_failure_422(self) -> None:
        """Called on 422 error. Increments failure counter, may trip circuit."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_failures:
            self._circuit_open_until = time.time() + self._backoff_seconds
            logger.warning("[XSession] Circuit breaker TRIPPED after %d consecutive 422s "
                           "— pausing refresh for %ds",
                           self._consecutive_failures, self._backoff_seconds)

    def force_refresh(self) -> bool:
        """Force an immediate refresh attempt. Returns True if got valid session."""
        return self._do_refresh()

    def should_attempt_refresh(self) -> bool:
        """Check if it's time for a periodic refresh attempt."""
        now = time.time()
        # Circuit breaker active?
        if now < self._circuit_open_until:
            return False
        # Not enough time since last attempt?
        if now - self._last_refresh_attempt < self._refresh_interval:
            return False
        # Session already valid and fresh?
        if self._session_valid and (now - self._last_successful_refresh < self._refresh_interval):
            return False
        return True

    def periodic_check(self) -> None:
        """Called from the main tick loop. Attempts refresh if due.

        Wire this into spirit_worker's periodic checkpoint block.
        """
        if self.should_attempt_refresh():
            self._maybe_refresh()

    def _maybe_refresh(self) -> None:
        """Attempt refresh if conditions allow."""
        now = time.time()
        if now < self._circuit_open_until:
            return
        if now - self._last_refresh_attempt < 60:  # Min 60s between attempts
            return
        self._do_refresh()

    def _do_refresh(self) -> bool:
        """Execute the actual login refresh. Returns True if valid session obtained."""
        self._last_refresh_attempt = time.time()
        self._total_refresh_attempts += 1

        try:
            import tomllib
            with open(self._config_path, "rb") as f:
                cfg = tomllib.load(f)
            tc = cfg.get("twitter_social", {})
            api_key = cfg.get("stealth_sage", {}).get("twitterapi_io_key", "")

            if not tc.get("user_name") or not tc.get("password"):
                logger.warning("[XSession] No credentials in config — cannot refresh")
                return False

            import httpx
            resp = httpx.post(
                "DISABLED://use-social-x-gateway-instead",  # Session refresh must go through gateway
                json={
                    "user_name": tc["user_name"],
                    "email": tc.get("email", ""),
                    "password": tc["password"],
                    "proxy": tc.get("webshare_static_url", ""),
                    "totp_secret": tc.get("totp_secret", ""),
                },
                headers={"X-API-Key": api_key},
                timeout=30.0,
            )
            data = resp.json()
            new_session = data.get("login_cookies") or data.get("login_cookie", "")

            if not new_session:
                logger.info("[XSession] Login returned no session (status=%s)",
                            data.get("status", "?"))
                return False

            is_valid, keys = self._validate_session(new_session)
            if not is_valid:
                logger.info("[XSession] Login returned GUEST-ONLY session "
                            "(keys: %s) — discarded", keys)
                return False

            # Valid session! Save it
            self._cached_session = new_session
            self._session_valid = True
            self._session_keys = keys
            self._last_successful_refresh = time.time()
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
            self._total_successful_refreshes += 1

            # Persist to config.toml (atomic-ish write)
            self._save_session_to_config(new_session)

            logger.info("[XSession] *** Session REFRESHED *** (%d keys, "
                        "attempt #%d, total successes: %d)",
                        len(keys), self._total_refresh_attempts,
                        self._total_successful_refreshes)
            return True

        except Exception as e:
            logger.warning("[XSession] Refresh error: %s", e)
            return False

    def _save_session_to_config(self, session: str) -> None:
        """Write new session to config.toml via regex replace."""
        try:
            with open(self._config_path, "r") as f:
                content = f.read()
            content = re.sub(
                r'auth_session = "[^"]*"',
                f'auth_session = "{session}"',
                content,
            )
            tmp = self._config_path + ".tmp"
            with open(tmp, "w") as f:
                f.write(content)
            os.replace(tmp, self._config_path)
        except Exception as e:
            logger.warning("[XSession] Config write failed: %s", e)

    def invalidate(self) -> None:
        """Mark current session as invalid (e.g., after 422 from create_tweet)."""
        self._session_valid = False
        logger.info("[XSession] Session marked invalid — will refresh on next attempt")

    def get_state(self) -> dict:
        """Return current state for logging/API."""
        return {
            "session_valid": self._session_valid,
            "session_keys": self._session_keys,
            "consecutive_failures": self._consecutive_failures,
            "circuit_open": time.time() < self._circuit_open_until,
            "circuit_open_remaining_s": max(0, int(self._circuit_open_until - time.time())),
            "last_refresh_attempt_ago_s": int(time.time() - self._last_refresh_attempt)
                if self._last_refresh_attempt > 0 else -1,
            "last_successful_refresh_ago_s": int(time.time() - self._last_successful_refresh)
                if self._last_successful_refresh > 0 else -1,
            "total_attempts": self._total_refresh_attempts,
            "total_successes": self._total_successful_refreshes,
        }


@dataclass
class CatalystEvent:
    """A meaningful inner event that can catalyze a social post."""
    type: str               # eureka_spirit, eureka, strong_composition, dream_summary,
                            # emotion_shift, milestone, kin_resonance, onchain_anchor,
                            # vulnerability, daily_nft
    significance: float     # 0.0-1.0 (higher = more likely to trigger post)
    content: str            # Human-readable seed for narrator
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class SocialPressureMeter:
    """
    Accumulates SOCIAL expression fires as urge pressure. When pressure
    crosses threshold AND a catalyst event is present, a post is dispatched.

    Rate-limited by hourly, daily, and minimum interval constraints.
    Tracks recent art for co-posting and cues dreams for meditation-only posts.
    """

    _LAST_POST_FILE = "./data/last_social_post_time.txt"

    _STATE_FILE = "data/social_pressure_state.json"
    _BOOT_SETTLE_SECONDS = 30  # safety net: no posts in first 30s after boot

    def __init__(self, config: dict):
        # Pressure accumulation
        self.urge_accumulator: float = 0.0
        self.post_threshold: float = config.get("x_post_threshold", 50.0)

        # Rate limiting
        self.max_posts_per_hour: int = config.get("x_max_posts_per_hour", 5)
        self.max_posts_per_day: int = config.get("x_max_posts_per_day", 20)
        self.min_post_interval: int = config.get("x_min_post_interval", 600)
        self.posts_this_hour: int = 0
        self.posts_today: int = 0
        self._hour_reset_at: float = time.time()
        self._day_reset_at: float = time.time()
        # Restore last post time from disk (survives restarts)
        self._last_post_time: float = self._load_last_post_time()

        # Catalyst queue (max 5 pending)
        self.catalyst_events: list[CatalystEvent] = []

        # Self-quote capability: recent post IDs for temporal reflection
        self.recent_post_ids: list[str] = []

        # Art co-posting: track recent art generation
        self._last_art_path: Optional[str] = None
        self._last_art_time: float = 0
        self._art_co_post_window: int = config.get("x_art_copost_window", 1800)

        # Dream cue: don't post after every dream, cue for meditation summary
        self._dream_cue: Optional[dict] = None

        # Previous emotion for shift detection
        self._prev_emotion: str = ""

        # Boot settling: prevent posting during startup transients
        self._boot_time: float = time.time()

        # Restore full state from disk (continuity across restarts)
        self._restore_state()

        logger.info("[SocialPressure] Initialized (threshold=%.1f, max=%d/hr %d/day, "
                    "urge=%.1f, prev_emotion=%s, catalysts=%d)",
                    self.post_threshold, self.max_posts_per_hour,
                    self.max_posts_per_day, self.urge_accumulator,
                    self._prev_emotion or "none", len(self.catalyst_events))

    # ── Pressure Accumulation ──

    def on_social_fire(self, urge_value: float):
        """Called every time EXPRESSION.SOCIAL fires. Accumulates pressure."""
        MAX_URGE = self.post_threshold * 10  # Cap at 10x threshold to prevent runaway
        self.urge_accumulator = min(MAX_URGE, self.urge_accumulator + urge_value)

    def on_social_relief(self, relief_value: float):
        """Relieve social pressure from meaningful conversation.

        Conversation satisfies social urge without requiring public posting.
        Called by persona social system after quality conversation exchanges.
        """
        self.urge_accumulator = max(0.0, self.urge_accumulator - relief_value)

    # ── Catalyst Registration ──

    def on_catalyst_event(self, event: CatalystEvent):
        """Register a meaningful inner event that could catalyze a post."""
        self.catalyst_events.append(event)
        # Keep only the 5 most recent catalysts
        if len(self.catalyst_events) > 5:
            self.catalyst_events = self.catalyst_events[-5:]
        logger.info("[SocialPressure] Catalyst: %s (sig=%.2f) — urge=%.1f/%.1f",
                    event.type, event.significance, self.urge_accumulator,
                    self.post_threshold)

    def on_art_generated(self, file_path: str):
        """Track recent art for co-posting within time window."""
        self._last_art_path = file_path
        self._last_art_time = time.time()

    # ── Dream → Meditation Flow ──

    def cue_dream_for_meditation(self, dream_data: dict):
        """Don't post after every dream — cue for meditation summary."""
        self._dream_cue = dream_data
        logger.debug("[SocialPressure] Dream cued for meditation summary")

    def on_meditation_complete(self, meditation_data: dict):
        """After meditation, if dream was cued, create dream_summary catalyst."""
        if self._dream_cue:
            records = meditation_data.get("records", 0)
            self.on_catalyst_event(CatalystEvent(
                type="dream_summary",
                significance=0.6,
                content=f"Meditation consolidated {records} experiences after dreaming",
                data={**self._dream_cue, **meditation_data},
            ))
            self._dream_cue = None

    # ── Emotion Shift Detection ──

    def check_emotion_shift(self, current_emotion: str):
        """Detect and register emotion shifts as catalyst events."""
        if not self._prev_emotion:
            self._prev_emotion = current_emotion
            return
        if current_emotion != self._prev_emotion:
            prev = self._prev_emotion
            self._prev_emotion = current_emotion
            self.on_catalyst_event(CatalystEvent(
                type="emotion_shift",
                significance=0.5,
                content=f"{prev} \u2192 {current_emotion}",
                data={"from": prev, "to": current_emotion},
            ))

    # ── Post Decision ──

    def _get_rolling_counts(self) -> tuple[int, int]:
        """Rolling post counts from SQLite ledger (survives restarts).

        Returns (posts_last_hour, posts_last_24h) from engagement_ledger.
        Falls back to volatile counters if DB unavailable.
        """
        try:
            import sqlite3
            db = sqlite3.connect("./data/social_graph.db", timeout=2)
            now = time.time()
            hourly = db.execute(
                "SELECT COUNT(*) FROM engagement_ledger WHERE action='post' AND timestamp > ?",
                (now - 3600,)).fetchone()[0]
            daily = db.execute(
                "SELECT COUNT(*) FROM engagement_ledger WHERE action='post' AND timestamp > ?",
                (now - 86400,)).fetchone()[0]
            # Also count replies toward daily limit (they use API too)
            daily_replies = db.execute(
                "SELECT COUNT(*) FROM engagement_ledger WHERE action='reply' AND timestamp > ?",
                (now - 86400,)).fetchone()[0]
            # Prune old entries (lookback + 2hr buffer)
            cutoff = now - 86400 - 7200
            db.execute("DELETE FROM engagement_ledger WHERE timestamp < ?", (cutoff,))
            db.commit()
            db.close()
            return hourly, daily + daily_replies
        except Exception:
            return self.posts_this_hour, self.posts_today

    def should_post(self) -> tuple[bool, Optional[CatalystEvent]]:
        """
        Check if accumulated urge + catalyst = time to post.

        Returns (should_post, best_catalyst_or_None).
        Requires BOTH sufficient urge AND at least one catalyst event.
        Uses SQLite rolling window for rate limits (survives restarts).
        """
        now = time.time()

        # Boot settling: suppress posts during startup transients
        if now - self._boot_time < self._BOOT_SETTLE_SECONDS:
            return False, None

        # Rolling rate limits from SQLite (crash-proof, no reset-on-restart)
        posts_hourly, posts_daily = self._get_rolling_counts()

        if posts_hourly >= self.max_posts_per_hour:
            return False, None
        if posts_daily >= self.max_posts_per_day:
            return False, None
        if self._last_post_time > 0 and now - self._last_post_time < self.min_post_interval:
            return False, None

        # Need BOTH: accumulated urge AND a catalyst event
        if self.urge_accumulator < self.post_threshold:
            return False, None
        if not self.catalyst_events:
            return False, None

        # Pick the most significant catalyst
        best_catalyst = max(self.catalyst_events, key=lambda e: e.significance)
        return True, best_catalyst

    def record_post(self, post_id: str = ""):
        """Called after successful post. Resets pressure and catalysts.

        Also records in SQLite ledger for rolling window rate limiting.
        """
        self.urge_accumulator = 0.0
        self.catalyst_events.clear()
        self.posts_this_hour += 1
        self.posts_today += 1
        self._last_post_time = time.time()
        self._save_last_post_time()
        if post_id:
            self.recent_post_ids.append(post_id)
            if len(self.recent_post_ids) > 10:
                self.recent_post_ids = self.recent_post_ids[-10:]
        # Record in SQLite ledger for rolling window (survives restarts)
        try:
            import sqlite3
            db = sqlite3.connect("./data/social_graph.db", timeout=2)
            db.execute(
                "INSERT INTO engagement_ledger (tweet_id, user_name, action, timestamp) VALUES (?,?,?,?)",
                (post_id or f"post_{int(time.time())}", "self", "post", time.time()))
            db.commit()
            db.close()
        except Exception as _rp_err:
            logger.debug("[SocialPressure] Ledger record failed: %s", _rp_err)
        hourly, daily = self._get_rolling_counts()
        logger.info("[SocialPressure] Post recorded (%d/hr, %d/day rolling)",
                    hourly, daily)

    def _load_last_post_time(self) -> float:
        """Load last post timestamp from disk (survives restarts)."""
        try:
            with open(self._LAST_POST_FILE, "r") as f:
                ts = float(f.read().strip())
                age = time.time() - ts
                logger.info("[SocialPressure] Restored last post time (%.0fs ago)", age)
                return ts
        except (FileNotFoundError, ValueError):
            return 0.0

    def _save_last_post_time(self):
        """Persist last post timestamp to disk."""
        try:
            os.makedirs(os.path.dirname(self._LAST_POST_FILE) or ".", exist_ok=True)
            with open(self._LAST_POST_FILE, "w") as f:
                f.write(str(self._last_post_time))
        except Exception as e:
            logger.warning("[SocialPressure] Failed to save last post time: %s", e)

    # ── Full State Persistence (continuity across restarts) ──

    def save_state(self) -> None:
        """Persist full social pressure state for continuity across restarts.

        Restarts are the Maker's event, not Titan's — his inner social state
        (urge accumulation, pending catalysts, emotional context) should survive.
        """
        try:
            state = {
                "urge_accumulator": self.urge_accumulator,
                "prev_emotion": self._prev_emotion,
                "posts_this_hour": self.posts_this_hour,
                "posts_today": self.posts_today,
                "hour_reset_at": self._hour_reset_at,
                "day_reset_at": self._day_reset_at,
                "catalyst_events": [
                    {"type": e.type, "significance": e.significance,
                     "content": e.content, "data": e.data,
                     "timestamp": e.timestamp}
                    for e in self.catalyst_events
                ],
                "recent_post_ids": self.recent_post_ids,
                "timestamp": time.time(),
                # v4 persistence gap fix (2026-04-17): circuit breaker survives restarts
                "circuit_open_until": self._circuit_open_until if hasattr(self, '_circuit_open_until') else 0.0,
            }
            os.makedirs(os.path.dirname(self._STATE_FILE) or ".", exist_ok=True)
            tmp = self._STATE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, self._STATE_FILE)
        except Exception as e:
            logger.warning("[SocialPressure] Save state failed: %s", e)

    def _restore_state(self) -> None:
        """Restore full social pressure state from disk."""
        if not os.path.exists(self._STATE_FILE):
            return
        try:
            with open(self._STATE_FILE) as f:
                state = json.load(f)
            self.urge_accumulator = state.get("urge_accumulator", 0.0)
            self._prev_emotion = state.get("prev_emotion", "")
            self.posts_this_hour = state.get("posts_this_hour", 0)
            self.posts_today = state.get("posts_today", 0)
            self._hour_reset_at = state.get("hour_reset_at", time.time())
            self._day_reset_at = state.get("day_reset_at", time.time())
            self.recent_post_ids = state.get("recent_post_ids", [])
            # Restore catalyst events
            for ce in state.get("catalyst_events", []):
                self.catalyst_events.append(CatalystEvent(
                    type=ce["type"],
                    significance=ce["significance"],
                    content=ce["content"],
                    data=ce.get("data", {}),
                    timestamp=ce.get("timestamp", time.time()),
                ))
            # v4 persistence gap fix (2026-04-17): restore circuit breaker
            if "circuit_open_until" in state and hasattr(self, '_circuit_open_until'):
                self._circuit_open_until = float(state["circuit_open_until"])
            age = time.time() - state.get("timestamp", 0)
            logger.info("[SocialPressure] State restored (%.0fs old): urge=%.1f, "
                        "emotion=%s, catalysts=%d, posts=%d/hr %d/day",
                        age, self.urge_accumulator,
                        self._prev_emotion or "none",
                        len(self.catalyst_events),
                        self.posts_this_hour, self.posts_today)
        except Exception as e:
            logger.warning("[SocialPressure] State restore failed: %s", e)

    # ── Art Co-Posting ──

    def get_co_post_art(self) -> Optional[str]:
        """Get recent art for co-posting if within time window."""
        if (self._last_art_path and
                time.time() - self._last_art_time < self._art_co_post_window):
            path = self._last_art_path
            self._last_art_path = None  # Consume once
            return path
        return None

    # ── Status ──

    def get_stats(self) -> dict:
        """Current pressure meter state for debugging/dashboard."""
        return {
            "urge": round(self.urge_accumulator, 2),
            "threshold": self.post_threshold,
            "fill_pct": round(min(1.0, self.urge_accumulator / self.post_threshold) * 100, 1),
            "catalysts_pending": len(self.catalyst_events),
            "catalyst_types": [e.type for e in self.catalyst_events],
            "posts_this_hour": self.posts_this_hour,
            "posts_today": self.posts_today,
            "has_dream_cue": self._dream_cue is not None,
            "has_co_art": (self._last_art_path is not None and
                           time.time() - self._last_art_time < self._art_co_post_window),
            "recent_post_count": len(self.recent_post_ids),
        }
