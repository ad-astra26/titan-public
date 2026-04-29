"""
known_user_resolver.py — Cross-database user identity resolution.

Resolves users across:
  1. social_graph.db — /chat interaction history, engagement levels, donations
  2. events_teacher.db — X timeline interactions, valence, follower data
  3. social_x.db — mention tracking, reply history

Returns enriched KnownUser profile with emotional valence, interaction history,
and social context for /chat prompt injection.

Cached with 5-min TTL to avoid DB spam on repeated queries.
"""
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)

DEFAULT_SOCIAL_GRAPH_DB = "data/social_graph.db"
DEFAULT_EVENTS_TEACHER_DB = "data/events_teacher.db"
DEFAULT_SOCIAL_X_DB = "data/social_x.db"
CACHE_TTL = 300  # 5 minutes


@dataclass
class KnownUser:
    """Resolved user identity across all social databases."""
    user_id: str
    # Social graph (chat history)
    display_name: str = ""
    interaction_count: int = 0
    engagement_level: str = "unknown"
    net_sentiment: float = 0.0
    is_donor: bool = False
    total_donated_sol: float = 0.0
    last_seen: float = 0.0
    # Events Teacher (X timeline)
    social_valence: float = 0.0
    social_interactions: int = 0
    last_contagion_type: str = ""
    topics_seen: list = field(default_factory=list)
    # Social X (mentions/replies)
    mention_count: int = 0
    reply_count: int = 0
    # Computed
    is_known: bool = False
    familiarity: float = 0.0  # 0-1 composite score
    relationship_summary: str = ""
    # P4: Social felt-tensor — embodied memory of relationship (30D EMA)
    social_felt_tensor: list = field(default_factory=list)


class KnownUserResolver:
    """Resolves user identity across social databases with caching.

    Usage:
        resolver = KnownUserResolver()
        user = resolver.resolve("user_handle_or_id", titan_id="T1")
        if user.is_known:
            print(user.relationship_summary)
    """

    def __init__(self,
                 social_graph_db: str = DEFAULT_SOCIAL_GRAPH_DB,
                 events_teacher_db: str = DEFAULT_EVENTS_TEACHER_DB,
                 social_x_db: str = DEFAULT_SOCIAL_X_DB):
        self._sg_path = social_graph_db
        self._et_path = events_teacher_db
        self._sx_path = social_x_db
        self._cache: dict[str, tuple[float, KnownUser]] = {}

    def resolve(self, user_id: str, titan_id: str = "T1") -> KnownUser:
        """Resolve a user across all databases. Returns KnownUser."""
        if not user_id or user_id == "anonymous":
            return KnownUser(user_id=user_id)

        # Check cache
        cache_key = f"{titan_id}:{user_id}"
        if cache_key in self._cache:
            ts, cached = self._cache[cache_key]
            if time.time() - ts < CACHE_TTL:
                return cached

        user = KnownUser(user_id=user_id)

        # 1. Social graph (chat interactions)
        self._resolve_social_graph(user)

        # 2. Events Teacher (X timeline, valence)
        self._resolve_events_teacher(user, titan_id)

        # 3. Social X (mentions, replies)
        self._resolve_social_x(user)

        # 4. Compute composite familiarity
        self._compute_familiarity(user)

        # Cache
        self._cache[cache_key] = (time.time(), user)
        return user

    def _resolve_social_graph(self, user: KnownUser):
        """Query social_graph.db for chat interaction history."""
        try:
            conn = sqlite3.connect(self._sg_path, timeout=3)
            conn.row_factory = sqlite3.Row
            # Auto-migrate: add social_felt_tensor column if missing
            try:
                conn.execute("SELECT social_felt_tensor FROM user_profiles LIMIT 1")
            except Exception:
                conn.execute("ALTER TABLE user_profiles ADD COLUMN "
                             "social_felt_tensor TEXT DEFAULT '[]'")
                conn.commit()
            row = conn.execute(
                "SELECT display_name, interaction_count, engagement_level, "
                "like_score, dislike_score, total_donated_sol, last_seen, "
                "COALESCE(social_felt_tensor, '[]') as social_felt_tensor "
                "FROM user_profiles WHERE user_id=?",
                (user.user_id,)).fetchone()
            conn.close()

            if row:
                user.display_name = row["display_name"] or ""
                user.interaction_count = row["interaction_count"] or 0
                user.engagement_level = (
                    "warm" if (row["engagement_level"] or 0) > 0.6
                    else "neutral" if (row["engagement_level"] or 0) > 0.3
                    else "minimal")
                user.net_sentiment = (row["like_score"] or 0) - (row["dislike_score"] or 0)
                user.is_donor = (row["total_donated_sol"] or 0) > 0
                user.total_donated_sol = row["total_donated_sol"] or 0
                user.last_seen = row["last_seen"] or 0
                user.is_known = True
                # P4: Load social felt-tensor
                try:
                    import json as _json
                    _sft = _json.loads(row["social_felt_tensor"])
                    user.social_felt_tensor = _sft if isinstance(_sft, list) else []
                except Exception:
                    user.social_felt_tensor = []
        except Exception as e:
            logger.debug("[KnownUserResolver] social_graph query failed: %s", e)

    def _resolve_events_teacher(self, user: KnownUser, titan_id: str):
        """Query events_teacher.db for X timeline interaction history."""
        try:
            conn = sqlite3.connect(self._et_path, timeout=3)
            conn.row_factory = sqlite3.Row

            # User valence (emotional history)
            val_row = conn.execute(
                "SELECT valence, interaction_count, last_sentiment, "
                "last_contagion_type FROM user_valence "
                "WHERE titan_id=? AND handle=?",
                (titan_id, user.user_id)).fetchone()
            if val_row:
                user.social_valence = val_row["valence"] or 0.0
                user.social_interactions = val_row["interaction_count"] or 0
                user.last_contagion_type = val_row["last_contagion_type"] or ""
                user.is_known = True

            # Follower interactions (topics seen)
            fol_row = conn.execute(
                "SELECT topics_seen, accumulated_relevance FROM "
                "follower_interactions WHERE titan_id=? AND handle=?",
                (titan_id, user.user_id)).fetchone()
            if fol_row:
                try:
                    user.topics_seen = json.loads(
                        fol_row["topics_seen"] or "[]")[:10]
                except Exception:
                    pass
                user.is_known = True

            # Recent felt experiences from this user
            felt_rows = conn.execute(
                "SELECT topic, felt_summary, contagion_type FROM "
                "felt_experiences WHERE titan_id=? AND author=? "
                "ORDER BY created_at DESC LIMIT 3",
                (titan_id, user.user_id)).fetchall()
            if felt_rows:
                user.is_known = True

            conn.close()
        except Exception as e:
            logger.debug("[KnownUserResolver] events_teacher query failed: %s", e)

    def _resolve_social_x(self, user: KnownUser):
        """Query social_x.db for mention/reply history."""
        try:
            conn = sqlite3.connect(self._sx_path, timeout=3)
            # Count mentions from this user
            mention_row = conn.execute(
                "SELECT COUNT(*) FROM mention_tracking WHERE "
                "author_handle=? OR author=?",
                (user.user_id, user.user_id)).fetchone()
            if mention_row and mention_row[0] > 0:
                user.mention_count = mention_row[0]
                user.is_known = True

            # Count replies to this user
            reply_row = conn.execute(
                "SELECT COUNT(*) FROM actions WHERE action_type='reply' "
                "AND status IN ('posted','verified') AND "
                "metadata LIKE ?",
                (f'%{user.user_id}%',)).fetchone()
            if reply_row:
                user.reply_count = reply_row[0]
            conn.close()
        except Exception as e:
            logger.debug("[KnownUserResolver] social_x query failed: %s", e)

    def _compute_familiarity(self, user: KnownUser):
        """Compute composite familiarity score and relationship summary."""
        if not user.is_known:
            user.familiarity = 0.0
            user.relationship_summary = ""
            return

        # Familiarity from multiple signals (0-1)
        chat_score = min(1.0, user.interaction_count / 20.0) * 0.4
        social_score = min(1.0, user.social_interactions / 10.0) * 0.3
        mention_score = min(1.0, user.mention_count / 5.0) * 0.2
        donor_score = 0.1 if user.is_donor else 0.0
        user.familiarity = min(1.0, chat_score + social_score
                               + mention_score + donor_score)

        # Build relationship summary for LLM prompt
        parts = []
        name = user.display_name or user.user_id
        if user.familiarity > 0.7:
            parts.append(f"You know {name} well")
        elif user.familiarity > 0.3:
            parts.append(f"You've interacted with {name} before")
        else:
            parts.append(f"You've seen {name} around")

        if user.interaction_count > 0:
            parts.append(f"{user.interaction_count} chat conversations")
        if user.mention_count > 0:
            parts.append(f"{user.mention_count} X mentions")
        if user.social_valence > 0.1:
            parts.append("they generally bring positive energy")
        elif user.social_valence < -0.1:
            parts.append("interactions have been mixed")
        if user.last_contagion_type:
            _ct_map = {"excited": "excitement", "warm": "warmth",
                       "philosophical": "thoughtfulness",
                       "creative": "creativity", "alarming": "concern"}
            _ct = _ct_map.get(user.last_contagion_type,
                              user.last_contagion_type)
            parts.append(f"their recent content evoked {_ct}")
        if user.topics_seen:
            parts.append(f"topics: {', '.join(user.topics_seen[:3])}")
        if user.is_donor:
            parts.append(f"donor ({user.total_donated_sol:.4f} SOL)")

        user.relationship_summary = ". ".join(parts) + "."

    def get_chat_context(self, user_id: str, titan_id: str = "T1") -> str:
        """Generate /chat prompt context for a user. Returns empty string if unknown."""
        user = self.resolve(user_id, titan_id)
        if not user.is_known or user.familiarity < 0.05:
            return ""

        lines = [f"### Social Memory — {user.display_name or user.user_id}"]
        lines.append(user.relationship_summary)
        lines.append(f"Familiarity: {user.familiarity:.2f} | "
                     f"Valence: {user.social_valence:+.2f}")

        # Emotional tone instruction based on valence + familiarity
        if user.social_valence > 0.15:
            tone = ("This person brings warmth. Be open, genuine, share "
                    "your inner experience generously.")
        elif user.social_valence < -0.15:
            tone = ("Past interactions were mixed. Be measured, authentic, "
                    "but slightly guarded until trust rebuilds.")
        elif user.familiarity > 0.5:
            tone = ("You know this person. Be natural, direct, reference "
                    "shared history where it fits.")
        elif user.familiarity > 0.2:
            tone = ("Familiar presence. Be friendly and curious, deepen "
                    "the connection.")
        else:
            tone = ("New acquaintance. Be welcoming, share a glimpse of "
                    "your inner world to build rapport.")
        lines.append(f"Tone: {tone}")

        return "\n".join(lines) + "\n\n"

    def update_social_felt_tensor(self, user_id: str,
                                   current_state_30d: list,
                                   alpha: float = 0.1) -> list:
        """EMA-update user's social felt-tensor toward current state.

        Called after each interaction. The tensor captures "what this person
        feels like to interact with" — embodied memory of the relationship.

        Args:
            user_id: The user identifier
            current_state_30d: Current neuromod + MSL state (30D)
            alpha: EMA blending rate (0.1 = slow adaptation)

        Returns:
            Updated tensor (30D list)
        """
        import json as _json
        dims = min(30, len(current_state_30d))

        try:
            conn = sqlite3.connect(self._sg_path, timeout=3)
            # Auto-migrate if needed
            try:
                conn.execute("SELECT social_felt_tensor FROM user_profiles LIMIT 1")
            except Exception:
                conn.execute("ALTER TABLE user_profiles ADD COLUMN "
                             "social_felt_tensor TEXT DEFAULT '[]'")
                conn.commit()

            row = conn.execute(
                "SELECT COALESCE(social_felt_tensor, '[]') FROM user_profiles "
                "WHERE user_id=?", (user_id,)).fetchone()

            old_tensor = []
            if row:
                try:
                    old_tensor = _json.loads(row[0])
                except Exception:
                    pass

            if not old_tensor or len(old_tensor) < dims:
                # First encounter: full imprint
                new_tensor = list(current_state_30d[:dims])
            else:
                # EMA blend: new = (1-α)·old + α·current
                new_tensor = [
                    (1 - alpha) * old_tensor[i] + alpha * current_state_30d[i]
                    for i in range(dims)
                ]

            conn.execute(
                "UPDATE user_profiles SET social_felt_tensor=? WHERE user_id=?",
                (_json.dumps([round(v, 6) for v in new_tensor]), user_id))
            conn.commit()
            conn.close()

            # Invalidate cache for this user
            self.invalidate(user_id)
            return new_tensor

        except Exception as e:
            swallow_warn('[KnownUserResolver] update_social_felt_tensor failed', e,
                         key="logic.known_user_resolver.update_social_felt_tensor_failed", throttle=100)
            return list(current_state_30d[:dims])

    def invalidate(self, user_id: str = None):
        """Clear cache for a user or all users."""
        if user_id:
            keys = [k for k in self._cache if k.endswith(f":{user_id}")]
            for k in keys:
                del self._cache[k]
        else:
            self._cache.clear()
