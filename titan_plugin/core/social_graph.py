"""
core/social_graph.py
Per-user profile tracking, social relationship discovery, and interaction scoring.

Phase 13: Sage Socialite — Cross-session user recognition, like/dislike scoring,
donation detection, "I:" inspiration transactions, sovereign social decisions.

Architecture:
  - UserProfile: Lightweight metadata (SQLite-backed, always in memory)
  - Per-user Cognee datasets: Heavy semantic memories (sharded, lazy-loaded)
  - Social edges: Relationship strength between users discovered from interactions
"""
import json
import logging
import os
import sqlite3
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Engagement level thresholds based on interaction quality
_ENGAGEMENT_CURIOUS = 0.2     # Default for new users
_ENGAGEMENT_FRIENDLY = 0.4    # Regular positive interactions
_ENGAGEMENT_TRUSTED = 0.6     # High-quality sustained engagement
_ENGAGEMENT_INNER_CIRCLE = 0.8  # Top users (donors, consistent quality)

# Donation mood boost tiers (SOL amount → mood delta)
DONATION_TIERS = [
    (0.10, 0.10, 5.0),   # 0.10+ SOL → +0.10 mood, 5.0x memory weight
    (0.05, 0.05, 3.0),   # 0.05-0.09 SOL → +0.05 mood, 3.0x weight
    (0.01, 0.02, 2.0),   # 0.01-0.04 SOL → +0.02 mood, 2.0x weight
    (0.00, 0.01, 1.5),   # Memo-only (0 SOL) → +0.01 mood, 1.5x weight
]


class UserProfile:
    """Lightweight user profile for cross-session recognition."""

    def __init__(self, row: dict):
        self.user_id: str = row["user_id"]
        self.platform: str = row.get("platform", "unknown")
        self.display_name: str = row.get("display_name", "")
        self.sol_address: Optional[str] = row.get("sol_address")
        self.first_seen: float = row.get("first_seen", time.time())
        self.last_seen: float = row.get("last_seen", time.time())
        self.interaction_count: int = row.get("interaction_count", 0)
        self.like_score: float = row.get("like_score", 0.0)
        self.dislike_score: float = row.get("dislike_score", 0.0)
        self.total_donated_sol: float = row.get("total_donated_sol", 0.0)
        self.engagement_level: float = row.get("engagement_level", _ENGAGEMENT_CURIOUS)
        self.notes: str = row.get("notes", "")

    @property
    def net_sentiment(self) -> float:
        """Net like/dislike sentiment (-1.0 to 1.0 range)."""
        total = self.like_score + self.dislike_score
        if total == 0:
            return 0.0
        return (self.like_score - self.dislike_score) / total

    @property
    def is_donor(self) -> bool:
        return self.total_donated_sol > 0

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "platform": self.platform,
            "display_name": self.display_name,
            "sol_address": self.sol_address,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "interaction_count": self.interaction_count,
            "like_score": self.like_score,
            "dislike_score": self.dislike_score,
            "total_donated_sol": self.total_donated_sol,
            "engagement_level": self.engagement_level,
            "notes": self.notes,
        }


class SocialGraph:
    """
    Manages user profiles, social relationships, and interaction quality scoring.
    Backed by SQLite for persistence, with in-memory cache for active users.
    """

    def __init__(self, db_path: str = "./data/social_graph.db"):
        self._db_path = db_path
        self._cache: Dict[str, UserProfile] = {}
        self._edges: Dict[str, float] = {}  # "userA::userB" → strength
        self._init_db()

    def _init_db(self):
        """Initialize SQLite schema for user profiles and edges."""
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    platform TEXT DEFAULT 'unknown',
                    display_name TEXT DEFAULT '',
                    sol_address TEXT,
                    first_seen REAL,
                    last_seen REAL,
                    interaction_count INTEGER DEFAULT 0,
                    like_score REAL DEFAULT 0.0,
                    dislike_score REAL DEFAULT 0.0,
                    total_donated_sol REAL DEFAULT 0.0,
                    engagement_level REAL DEFAULT 0.3,
                    notes TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS social_edges (
                    edge_key TEXT PRIMARY KEY,
                    user_a TEXT NOT NULL,
                    user_b TEXT NOT NULL,
                    strength REAL DEFAULT 0.1,
                    last_interaction REAL,
                    interaction_count INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS donations (
                    tx_signature TEXT PRIMARY KEY,
                    sender_address TEXT NOT NULL,
                    user_id TEXT,
                    amount_sol REAL NOT NULL,
                    memo TEXT DEFAULT '',
                    timestamp REAL NOT NULL,
                    acknowledged INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inspirations (
                    tx_signature TEXT PRIMARY KEY,
                    sender_address TEXT NOT NULL,
                    user_id TEXT,
                    message TEXT NOT NULL,
                    amount_sol REAL DEFAULT 0.0,
                    timestamp REAL NOT NULL,
                    processed INTEGER DEFAULT 0,
                    outcome TEXT DEFAULT ''
                )
            """)
            # Engagement ledger: persistent dedup + rate limiting for X interactions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS engagement_ledger (
                    tweet_id TEXT NOT NULL,
                    user_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    mention_text TEXT DEFAULT '',
                    PRIMARY KEY (tweet_id, action)
                )
            """)
            # Index for efficient window queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ledger_timestamp
                ON engagement_ledger(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ledger_user_action
                ON engagement_ledger(user_name, action, timestamp)
            """)
            # Per-Titan social preferences: each Titan tracks its own favorites
            conn.execute("""
                CREATE TABLE IF NOT EXISTS titan_social_preferences (
                    titan_id TEXT NOT NULL,
                    user_name TEXT NOT NULL,
                    relationship TEXT DEFAULT 'follower',
                    affinity REAL DEFAULT 0.0,
                    tags TEXT DEFAULT '',
                    discovered_via TEXT DEFAULT 'follower_sync',
                    interaction_count INTEGER DEFAULT 0,
                    last_checked REAL DEFAULT 0,
                    last_interacted REAL DEFAULT 0,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (titan_id, user_name)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_social_pref_titan_affinity
                ON titan_social_preferences(titan_id, affinity DESC)
            """)
            # Community registry: synced from followers/following lists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS community_registry (
                    user_name TEXT PRIMARY KEY,
                    user_id TEXT DEFAULT '',
                    display_name TEXT DEFAULT '',
                    bio TEXT DEFAULT '',
                    followers_count INTEGER DEFAULT 0,
                    is_follower INTEGER DEFAULT 0,
                    is_following INTEGER DEFAULT 0,
                    last_synced REAL DEFAULT 0,
                    last_tweet_text TEXT DEFAULT '',
                    last_tweet_time REAL DEFAULT 0
                )
            """)
            conn.commit()

    # -------------------------------------------------------------------------
    # User Profile CRUD
    # -------------------------------------------------------------------------
    def get_or_create_user(
        self, user_id: str, platform: str = "unknown", display_name: str = ""
    ) -> UserProfile:
        """Get existing profile or create a new one."""
        if user_id in self._cache:
            return self._cache[user_id]

        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()

            if row:
                profile = UserProfile(dict(row))
                self._cache[user_id] = profile
                return profile

        # Create new
        now = time.time()
        profile_data = {
            "user_id": user_id,
            "platform": platform,
            "display_name": display_name or user_id,
            "first_seen": now,
            "last_seen": now,
            "interaction_count": 0,
            "like_score": 0.0,
            "dislike_score": 0.0,
            "total_donated_sol": 0.0,
            "engagement_level": _ENGAGEMENT_CURIOUS,
        }
        profile = UserProfile(profile_data)
        self._save_profile(profile)
        self._cache[user_id] = profile
        logger.info("[SocialGraph] New user: %s (%s)", user_id, platform)
        return profile

    def _save_profile(self, profile: UserProfile):
        """Persist profile to SQLite."""
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_profiles
                (user_id, platform, display_name, sol_address, first_seen, last_seen,
                 interaction_count, like_score, dislike_score, total_donated_sol,
                 engagement_level, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.user_id, profile.platform, profile.display_name,
                profile.sol_address, profile.first_seen, profile.last_seen,
                profile.interaction_count, profile.like_score, profile.dislike_score,
                profile.total_donated_sol, profile.engagement_level, profile.notes,
            ))
            conn.commit()

    def record_interaction(self, user_id: str, quality: float = 0.5):
        """
        Record an interaction with a user. Quality: 0.0 (bad) to 1.0 (excellent).
        Updates interaction count, like/dislike scores, and engagement level.
        """
        profile = self.get_or_create_user(user_id)
        profile.interaction_count += 1
        profile.last_seen = time.time()

        # Update like/dislike based on quality
        if quality >= 0.6:
            profile.like_score += quality - 0.5
        elif quality <= 0.4:
            profile.dislike_score += 0.5 - quality

        # Recalculate engagement level (bounded sigmoid of net sentiment + interaction depth)
        depth_factor = min(1.0, profile.interaction_count / 20.0)
        sentiment_factor = max(0.0, min(1.0, (profile.net_sentiment + 1.0) / 2.0))
        donor_bonus = 0.1 if profile.is_donor else 0.0
        profile.engagement_level = min(
            1.0,
            _ENGAGEMENT_CURIOUS + (sentiment_factor * depth_factor * 0.6) + donor_bonus,
        )

        self._save_profile(profile)
        self._cache[user_id] = profile

    # -------------------------------------------------------------------------
    # Social Edges (Relationship Discovery)
    # -------------------------------------------------------------------------
    def record_edge(self, user_a: str, user_b: str):
        """
        Record or strengthen a social edge between two users.
        Called when userB replies to userA on a Titan thread.
        """
        edge_key = f"{min(user_a, user_b)}::{max(user_a, user_b)}"
        now = time.time()

        with sqlite3.connect(self._db_path, timeout=10) as conn:
            existing = conn.execute(
                "SELECT strength, interaction_count FROM social_edges WHERE edge_key = ?",
                (edge_key,),
            ).fetchone()

            if existing:
                new_strength = min(1.0, existing[0] + 0.05)
                new_count = existing[1] + 1
                conn.execute(
                    "UPDATE social_edges SET strength=?, interaction_count=?, last_interaction=? WHERE edge_key=?",
                    (new_strength, new_count, now, edge_key),
                )
            else:
                conn.execute(
                    "INSERT INTO social_edges (edge_key, user_a, user_b, strength, last_interaction, interaction_count) VALUES (?, ?, ?, ?, ?, ?)",
                    (edge_key, user_a, user_b, 0.1, now, 1),
                )
            conn.commit()

        self._edges[edge_key] = min(1.0, self._edges.get(edge_key, 0.0) + 0.05)
        logger.debug("[SocialGraph] Edge %s ↔ %s strengthened.", user_a[:12], user_b[:12])

    def get_user_connections(self, user_id: str) -> List[Dict]:
        """Get all social connections for a user, sorted by strength."""
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM social_edges WHERE user_a = ? OR user_b = ? ORDER BY strength DESC",
                (user_id, user_id),
            ).fetchall()

        connections = []
        for row in rows:
            other = row["user_b"] if row["user_a"] == user_id else row["user_a"]
            connections.append({
                "user_id": other,
                "strength": row["strength"],
                "interactions": row["interaction_count"],
                "last_interaction": row["last_interaction"],
            })
        return connections

    # -------------------------------------------------------------------------
    # Donations
    # -------------------------------------------------------------------------
    def record_donation(
        self, tx_signature: str, sender_address: str, amount_sol: float, memo: str = ""
    ) -> Optional[UserProfile]:
        """
        Record a SOL donation. Matches sender to known user profile by sol_address.
        Returns the matched UserProfile or None if unknown sender.
        """
        now = time.time()
        matched_user = self.find_user_by_sol_address(sender_address)
        user_id = matched_user.user_id if matched_user else None

        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO donations
                (tx_signature, sender_address, user_id, amount_sol, memo, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (tx_signature, sender_address, user_id, amount_sol, memo, now))
            conn.commit()

        if matched_user:
            matched_user.total_donated_sol += amount_sol
            matched_user.last_seen = now
            self._save_profile(matched_user)
            logger.info(
                "[SocialGraph] Donation %.4f SOL from %s (%s)",
                amount_sol, matched_user.display_name, tx_signature[:16],
            )
        else:
            logger.info(
                "[SocialGraph] Donation %.4f SOL from unknown address %s",
                amount_sol, sender_address[:16],
            )

        return matched_user

    def get_donation_mood_boost(self, amount_sol: float) -> tuple[float, float]:
        """
        Calculate mood boost and memory weight multiplier for a donation.
        Returns (mood_delta, memory_weight).
        """
        for threshold, mood_delta, weight in DONATION_TIERS:
            if amount_sol >= threshold:
                return mood_delta, weight
        return 0.01, 1.5

    # -------------------------------------------------------------------------
    # Inspiration Transactions
    # -------------------------------------------------------------------------
    def record_inspiration(
        self, tx_signature: str, sender_address: str, message: str, amount_sol: float = 0.0
    ) -> Optional[UserProfile]:
        """Record an I: inspiration transaction."""
        now = time.time()
        matched_user = self.find_user_by_sol_address(sender_address)
        user_id = matched_user.user_id if matched_user else None

        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO inspirations
                (tx_signature, sender_address, user_id, message, amount_sol, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (tx_signature, sender_address, user_id, message, amount_sol, now))
            conn.commit()

        if matched_user:
            # Boost interaction quality for inspired users
            self.record_interaction(matched_user.user_id, quality=0.8)

        logger.info(
            "[SocialGraph] Inspiration from %s: '%s' (%.4f SOL)",
            user_id or sender_address[:16], message[:50], amount_sol,
        )
        return matched_user

    def get_pending_inspirations(self, limit: int = 10) -> List[Dict]:
        """Get unprocessed inspiration transactions."""
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM inspirations WHERE processed = 0 ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_inspiration_processed(self, tx_signature: str, outcome: str):
        """Mark an inspiration as processed with its outcome."""
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.execute(
                "UPDATE inspirations SET processed = 1, outcome = ? WHERE tx_signature = ?",
                (outcome, tx_signature),
            )
            conn.commit()

    # -------------------------------------------------------------------------
    # Lookup Helpers
    # -------------------------------------------------------------------------
    def find_user_by_sol_address(self, sol_address: str) -> Optional[UserProfile]:
        """Find a user profile by their Solana address."""
        # Check cache first
        for profile in self._cache.values():
            if profile.sol_address == sol_address:
                return profile

        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE sol_address = ?", (sol_address,)
            ).fetchone()
            if row:
                profile = UserProfile(dict(row))
                self._cache[profile.user_id] = profile
                return profile
        return None

    def link_sol_address(self, user_id: str, sol_address: str):
        """Link a Solana address to an existing user profile."""
        profile = self.get_or_create_user(user_id)
        profile.sol_address = sol_address
        self._save_profile(profile)
        self._cache[user_id] = profile
        logger.info("[SocialGraph] Linked SOL address %s to user %s", sol_address[:16], user_id)

    def get_top_users(self, limit: int = 10) -> List[UserProfile]:
        """Get top users by engagement level."""
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM user_profiles ORDER BY engagement_level DESC, interaction_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [UserProfile(dict(r)) for r in rows]

    def get_stats(self) -> Dict:
        """Return social graph statistics."""
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            user_count = conn.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0]
            edge_count = conn.execute("SELECT COUNT(*) FROM social_edges").fetchone()[0]
            donation_count = conn.execute("SELECT COUNT(*) FROM donations").fetchone()[0]
            total_donated = conn.execute("SELECT COALESCE(SUM(amount_sol), 0) FROM donations").fetchone()[0]
            inspiration_count = conn.execute("SELECT COUNT(*) FROM inspirations").fetchone()[0]
        return {
            "users": user_count,
            "edges": edge_count,
            "donations": donation_count,
            "total_donated_sol": total_donated,
            "inspirations": inspiration_count,
        }

    # -------------------------------------------------------------------------
    # Engagement Ledger — Persistent Dedup + Rate Limiting for X
    # -------------------------------------------------------------------------

    def ledger_record(self, tweet_id: str, user_name: str, action: str,
                      mention_text: str = "") -> None:
        """Record an engagement action (reply/like) in persistent ledger."""
        now = time.time()
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO engagement_ledger "
                "(tweet_id, user_name, action, timestamp, mention_text) "
                "VALUES (?, ?, ?, ?, ?)",
                (tweet_id, user_name, action, now, mention_text[:200]),
            )
            conn.commit()

    def ledger_has_tweet(self, tweet_id: str, action: str = None) -> bool:
        """Check if we've already engaged with this tweet.

        Args:
            action: If specified, only check for this action type ('reply'/'like').
                    If None, checks for any action.
        """
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            if action:
                row = conn.execute(
                    "SELECT 1 FROM engagement_ledger WHERE tweet_id = ? AND action = ?",
                    (tweet_id, action),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT 1 FROM engagement_ledger WHERE tweet_id = ?",
                    (tweet_id,),
                ).fetchone()
            return row is not None

    def ledger_user_reply_count(self, user_name: str, window_seconds: float) -> int:
        """Count replies to a user within the time window."""
        cutoff = time.time() - window_seconds
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM engagement_ledger "
                "WHERE user_name = ? AND action = 'reply' AND timestamp > ?",
                (user_name, cutoff),
            ).fetchone()
            return row[0] if row else 0

    def ledger_last_reply_to_user(self, user_name: str) -> float:
        """Get timestamp of last reply to a user (0.0 if never)."""
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            row = conn.execute(
                "SELECT MAX(timestamp) FROM engagement_ledger "
                "WHERE user_name = ? AND action = 'reply'",
                (user_name,),
            ).fetchone()
            return row[0] if row and row[0] else 0.0

    def ledger_total_today(self, action: str = None) -> int:
        """Count total actions today (since midnight UTC)."""
        import datetime
        midnight = datetime.datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0).timestamp()
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            if action:
                row = conn.execute(
                    "SELECT COUNT(*) FROM engagement_ledger "
                    "WHERE action = ? AND timestamp > ?",
                    (action, midnight),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM engagement_ledger "
                    "WHERE timestamp > ?", (midnight,),
                ).fetchone()
            return row[0] if row else 0

    def ledger_cleanup(self, max_age_seconds: float = 172800) -> int:
        """Remove ledger entries older than max_age (default 48h). Returns count removed."""
        cutoff = time.time() - max_age_seconds
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            cursor = conn.execute(
                "DELETE FROM engagement_ledger WHERE timestamp < ?", (cutoff,))
            conn.commit()
            return cursor.rowcount

    def should_engage(self, user_id: str) -> str:
        """
        Sovereign social decision: determine engagement level for a user.
        Returns: 'warm', 'neutral', 'minimal', 'ignore'
        """
        profile = self.get_or_create_user(user_id)

        if profile.engagement_level >= _ENGAGEMENT_INNER_CIRCLE:
            return "warm"
        elif profile.engagement_level >= _ENGAGEMENT_FRIENDLY:
            return "neutral"
        elif profile.engagement_level >= _ENGAGEMENT_CURIOUS:
            return "minimal"
        else:
            return "ignore"

    # ─────────────────────────────────────────────────────────────────────
    # Community Registry — synced from followers/following lists
    # ─────────────────────────────────────────────────────────────────────

    def sync_community(self, users: list[dict], relationship: str = "follower"):
        """Sync followers or following list into community registry.

        Args:
            users: list of dicts with userName, id, description, followersCount
            relationship: 'follower' or 'following'
        """
        is_follower = 1 if relationship == "follower" else 0
        is_following = 1 if relationship == "following" else 0
        now = time.time()
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            for u in users:
                name = u.get("userName", u.get("screen_name", ""))
                if not name:
                    continue
                conn.execute("""
                    INSERT INTO community_registry
                        (user_name, user_id, display_name, bio, followers_count,
                         is_follower, is_following, last_synced)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_name) DO UPDATE SET
                        display_name=excluded.display_name,
                        bio=excluded.bio,
                        followers_count=excluded.followers_count,
                        is_follower=MAX(is_follower, excluded.is_follower),
                        is_following=MAX(is_following, excluded.is_following),
                        last_synced=excluded.last_synced
                """, (name, u.get("id", ""), u.get("name", name),
                      u.get("description", "")[:300],
                      u.get("followersCount", u.get("followers_count", 0)),
                      is_follower, is_following, now))
            conn.commit()
        logger.info("[SocialGraph] Synced %d %ss into community registry",
                    len(users), relationship)

    def get_community(self, relationship: str = None) -> list[dict]:
        """Get community members, optionally filtered by relationship."""
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            if relationship == "follower":
                rows = conn.execute(
                    "SELECT * FROM community_registry WHERE is_follower=1"
                    " ORDER BY followers_count DESC").fetchall()
            elif relationship == "following":
                rows = conn.execute(
                    "SELECT * FROM community_registry WHERE is_following=1"
                    " ORDER BY followers_count DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM community_registry ORDER BY followers_count DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────────
    # Per-Titan Social Preferences
    # ─────────────────────────────────────────────────────────────────────

    def set_titan_preference(self, titan_id: str, user_name: str,
                             affinity_delta: float = 0.1,
                             tags: str = "", discovered_via: str = ""):
        """Update a Titan's affinity for a community member.

        Affinity grows with genuine interaction quality. Each Titan builds
        its own social preferences independently.
        """
        now = time.time()
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            existing = conn.execute(
                "SELECT affinity, interaction_count, tags FROM titan_social_preferences "
                "WHERE titan_id=? AND user_name=?",
                (titan_id, user_name)).fetchone()
            if existing:
                new_affinity = min(1.0, existing[0] + affinity_delta)
                new_count = existing[1] + 1
                # Merge tags
                old_tags = set(existing[2].split(",")) if existing[2] else set()
                if tags:
                    old_tags.update(tags.split(","))
                merged_tags = ",".join(t for t in old_tags if t)
                conn.execute(
                    "UPDATE titan_social_preferences SET affinity=?, interaction_count=?, "
                    "tags=?, last_interacted=? WHERE titan_id=? AND user_name=?",
                    (new_affinity, new_count, merged_tags, now, titan_id, user_name))
            else:
                conn.execute(
                    "INSERT INTO titan_social_preferences "
                    "(titan_id, user_name, affinity, tags, discovered_via, "
                    " interaction_count, last_interacted, created_at) "
                    "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                    (titan_id, user_name, max(0.1, affinity_delta), tags,
                     discovered_via, now, now))
            conn.commit()

    def get_titan_favorites(self, titan_id: str, limit: int = 10) -> list[dict]:
        """Get a Titan's favorite accounts, ordered by affinity."""
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT p.*, c.bio, c.followers_count, c.last_tweet_text "
                "FROM titan_social_preferences p "
                "LEFT JOIN community_registry c ON p.user_name = c.user_name "
                "WHERE p.titan_id = ? ORDER BY p.affinity DESC LIMIT ?",
                (titan_id, limit)).fetchall()
            return [dict(r) for r in rows]

    def get_accounts_to_check(self, titan_id: str, limit: int = 3) -> list[dict]:
        """Get accounts for a Titan to check this cycle.

        Prioritizes: high affinity first, then unchecked, then stale.
        Avoids accounts checked in last 2 hours.
        """
        cutoff = time.time() - 7200  # 2h minimum between checks
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            # First: Titan's favorites not recently checked
            rows = conn.execute(
                "SELECT p.user_name, p.affinity, p.tags, c.bio "
                "FROM titan_social_preferences p "
                "LEFT JOIN community_registry c ON p.user_name = c.user_name "
                "WHERE p.titan_id = ? AND p.last_checked < ? "
                "ORDER BY p.affinity DESC LIMIT ?",
                (titan_id, cutoff, limit)).fetchall()
            result = [dict(r) for r in rows]

            # Fill remaining slots from community (not yet in preferences)
            if len(result) < limit:
                remaining = limit - len(result)
                checked_names = {r["user_name"] for r in result}
                rows2 = conn.execute(
                    "SELECT user_name, bio, followers_count FROM community_registry "
                    "WHERE user_name NOT IN (SELECT user_name FROM titan_social_preferences "
                    "  WHERE titan_id = ?) "
                    "AND last_synced > 0 ORDER BY RANDOM() LIMIT ?",
                    (titan_id, remaining)).fetchall()
                for r in rows2:
                    if r[0] not in checked_names:
                        result.append({"user_name": r[0], "bio": r[1],
                                       "affinity": 0.0, "tags": ""})
            return result

    def mark_checked(self, titan_id: str, user_name: str):
        """Mark that a Titan checked this account's timeline."""
        now = time.time()
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.execute(
                "UPDATE titan_social_preferences SET last_checked=? "
                "WHERE titan_id=? AND user_name=?",
                (now, titan_id, user_name))
            conn.commit()

    def update_last_tweet(self, user_name: str, tweet_text: str):
        """Cache a user's latest tweet in the community registry."""
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.execute(
                "UPDATE community_registry SET last_tweet_text=?, last_tweet_time=? "
                "WHERE user_name=?",
                (tweet_text[:500], time.time(), user_name))
            conn.commit()
