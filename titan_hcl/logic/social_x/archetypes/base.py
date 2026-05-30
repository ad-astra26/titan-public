"""Archetype base class + shared helpers.

rFP_x_voice_enrichment §4.3. An archetype is a layer combination that fires
on a specific *trigger* (an outer-world event, an inner crystallization, a
self-insight, etc.), not on felt-state alone. The 9 Phase 1 archetypes plug
into the gateway through this contract:

    archetype = ConcreteArchetype(gateway, ctx_helpers...)
    candidate = archetype.find_candidate(context)   # None or ArchetypeCandidate
    if candidate:
        text   = archetype.render_prompt(candidate, context)
        layers = candidate.layers          # for context-builder integration
        media  = candidate.media_ids       # for create_tweet_v2 attachment
        archetype.record_post(candidate, tweet_id, context)

Universal invariants enforced here:
  * Cross-archetype 4 h spacing (§4.3 universal — except where archetypes
    override, e.g. PROOF_DAY's must-post slot).
  * Idempotency via `actions.metadata` JSON column with `<archetype>_source_id`.
  * Universal footer: gateway already auto-appends state signature; no work
    here.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

logger = logging.getLogger(__name__)


# Cross-archetype spacing default (§4.3 universal). Specific archetypes
# may bypass via `bypass_spacing=True` on their candidate.
DEFAULT_CROSS_ARCHETYPE_SPACING_S = 4 * 3600

# Outer-world engagement archetypes — those that publicly reference / engage a
# specific external account. They share a single per-AUTHOR cooldown so one
# account (especially a large one like @lopp) is never reflected on / amplified
# more than once per cooldown window across ANY of these archetypes. Maker rule
# 2026-05-30: "we cannot have <big account> notified for days talking about the
# same post — makes zero sense / unprofessional."
OUTER_ENGAGEMENT_POST_TYPES = (
    "world_mirror", "outer_inner_bridge", "outer_rumination", "amplify",
)
DEFAULT_AUTHOR_COOLDOWN_S = 7 * 86400  # ≥ 1 week per Maker


def ensure_handle_mention(text: str, handle: str) -> str:
    """Guarantee the cited account's literal @handle is present in `text`.

    The LLM routinely references an account by display name and drops the '@'
    ("Lopp's warning..."), which does NOT notify the person on X. If the
    @handle (case-insensitive, word-boundary) is missing, prepend it so the
    mention is real and the account is notified. No-op when handle is empty or
    already present. (Maker 2026-05-30.)
    """
    if not text or not handle:
        return text
    import re
    h = handle.lstrip("@").strip()
    if not h:
        return text
    if re.search(r"@" + re.escape(h) + r"\b", text, re.IGNORECASE):
        return text
    return f"@{h} {text}"


@dataclass(slots=True)
class ArchetypeCandidate:
    """A concrete decision to fire this archetype at this moment."""
    archetype: str                 # e.g. "world_mirror", "grounded_today"
    pool: str = ""                 # e.g. "A_x_content" / "" if single-pool
    source_id: str = ""            # the cited entity's identifier
    layers: list[str] = field(default_factory=list)
    layer_values: dict[str, str] = field(default_factory=dict)
    prompt_template: str = ""      # archetype-specific prompt body
    prompt_values: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    media_ids: list[str] = field(default_factory=list)
    quoted_tweet_id: str = ""
    relevance: float = 0.0         # for selection tie-breaks
    salience: float = 0.0          # for adaptive scoring inputs
    bypass_spacing: bool = False   # PROOF_DAY's must-post slot
    bypass_rate_limit: bool = False  # PROOF_DAY only

    def render_prompt(self) -> str:
        if not self.prompt_template:
            return ""
        try:
            return self.prompt_template.format(**self.prompt_values)
        except KeyError as e:
            logger.warning(
                "[archetype %s] prompt template missing key %s; raw template returned",
                self.archetype, e,
            )
            return self.prompt_template


class ArchetypeBase:
    """Shared logic for every Phase 1 archetype.

    Subclasses implement `find_candidate(context)` and call helpers from
    here for idempotency, dedup, and cross-archetype spacing.
    """

    name: str = "base"             # override in subclass
    metadata_key: str = ""         # e.g. "world_mirror_source_id"
    cross_archetype_spacing_s: float = DEFAULT_CROSS_ARCHETYPE_SPACING_S

    def __init__(self, *, gateway, social_x_db_path: str):
        self.gateway = gateway
        self.db_path = social_x_db_path

    # ── Subclass API ────────────────────────────────────────────────

    def find_candidate(self, context) -> ArchetypeCandidate | None:
        """Return an `ArchetypeCandidate` if this archetype should fire now,
        or None to abstain. Override in subclass."""
        raise NotImplementedError

    # ── Shared helpers ──────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=5)
        c.row_factory = sqlite3.Row
        return c

    def is_already_cited(
        self,
        source_id: str,
        *,
        titan_id: str,
        window_seconds: float | None = None,
        metadata_key: str | None = None,
    ) -> bool:
        """True if this `<source_id>` was already cited by this archetype.

        - `window_seconds=None` → lifetime dedup.
        - `metadata_key` defaults to `<self.metadata_key>`.
        """
        key = metadata_key or self.metadata_key
        if not key or not source_id:
            return False
        conn = self._conn()
        try:
            params: list = [titan_id, self.name, f'%"{key}":"{source_id}"%']
            sql = (
                "SELECT 1 FROM actions WHERE titan_id=? AND post_type=? "
                "AND metadata LIKE ?"
            )
            if window_seconds is not None:
                sql += " AND created_at >= ?"
                params.append(time.time() - window_seconds)
            sql += " LIMIT 1"
            row = conn.execute(sql, tuple(params)).fetchone()
            return row is not None
        finally:
            conn.close()

    def cited_set(
        self,
        *,
        titan_id: str,
        window_seconds: float | None = None,
        metadata_key: str | None = None,
    ) -> set[str]:
        """Return every `source_id` previously cited by this archetype.

        Useful for SQL `NOT IN` filters when iterating large candidate pools.
        """
        key = metadata_key or self.metadata_key
        if not key:
            return set()
        conn = self._conn()
        try:
            params: list = [titan_id, self.name, f'%"{key}":%']
            sql = (
                "SELECT metadata FROM actions WHERE titan_id=? AND post_type=? "
                "AND metadata LIKE ?"
            )
            if window_seconds is not None:
                sql += " AND created_at >= ?"
                params.append(time.time() - window_seconds)
            rows = conn.execute(sql, tuple(params)).fetchall()
        finally:
            conn.close()
        out: set[str] = set()
        for r in rows:
            try:
                meta = json.loads(r["metadata"] or "{}")
            except Exception:
                continue
            sid = meta.get(key)
            if sid:
                out.add(str(sid))
        return out

    def authors_on_cooldown(
        self,
        *,
        titan_id: str,
        now: float | None = None,
        window_seconds: float = DEFAULT_AUTHOR_COOLDOWN_S,
    ) -> set[str]:
        """Lowercased handles engaged by ANY outer-engagement archetype within
        the cooldown window (cross-archetype per-author dedup, Maker 2026-05-30).

        Checks BOTH `metadata.author` (world_mirror, outer_inner_bridge,
        amplify) and `metadata.handle` (outer_rumination) since the archetypes
        historically used different metadata keys for the cited account.
        """
        cutoff = (now if now is not None else time.time()) - window_seconds
        placeholders = ",".join("?" * len(OUTER_ENGAGEMENT_POST_TYPES))
        conn = self._conn()
        try:
            rows = conn.execute(
                f"SELECT metadata FROM actions WHERE titan_id=? "
                f"AND post_type IN ({placeholders}) "
                f"AND status NOT IN ('failed','error') "
                f"AND created_at >= ?",
                (titan_id, *OUTER_ENGAGEMENT_POST_TYPES, cutoff),
            ).fetchall()
        except Exception:
            return set()
        finally:
            conn.close()
        out: set[str] = set()
        for r in rows:
            try:
                m = json.loads(r["metadata"] or "{}")
            except Exception:
                continue
            for key in ("author", "handle"):
                v = m.get(key)
                if v:
                    out.add(str(v).lower())
        return out

    def author_on_cooldown(
        self,
        author: str,
        *,
        titan_id: str,
        now: float | None = None,
        window_seconds: float = DEFAULT_AUTHOR_COOLDOWN_S,
    ) -> bool:
        """True if `author` was engaged by any outer archetype within window."""
        if not author:
            return False
        return str(author).lower() in self.authors_on_cooldown(
            titan_id=titan_id, now=now, window_seconds=window_seconds)

    def cross_archetype_blocked(
        self,
        *,
        titan_id: str,
        now: float | None = None,
        spacing_seconds: float | None = None,
    ) -> bool:
        """True if any *other* archetype posted within the spacing window.

        Archetypes that legitimately need to bypass (e.g. PROOF_DAY's
        must-post slot) set `bypass_spacing=True` on their candidate.
        """
        spacing = spacing_seconds if spacing_seconds is not None else self.cross_archetype_spacing_s
        cutoff = (now if now is not None else time.time()) - spacing
        archetype_post_types = (
            "proof_day", "world_mirror", "outer_rumination",
            "outer_inner_bridge", "grounded_today", "practiced_response",
            "reflection", "composed_thought", "self_watching",
        )
        placeholders = ",".join("?" * len(archetype_post_types))
        conn = self._conn()
        try:
            row = conn.execute(
                f"SELECT 1 FROM actions WHERE titan_id=? "
                f"AND post_type IN ({placeholders}) AND post_type != ? "
                f"AND created_at >= ? LIMIT 1",
                (titan_id, *archetype_post_types, self.name, cutoff),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def per_titan_count_today(
        self,
        *,
        titan_id: str,
        now: float | None = None,
        utc_day_aligned: bool = False,
    ) -> int:
        """Posts of this archetype already made today (UTC midnight) or in
        the rolling 24h window."""
        n = now if now is not None else time.time()
        if utc_day_aligned:
            import datetime as _dt
            today = _dt.datetime.fromtimestamp(n, _dt.timezone.utc).date()
            cutoff = _dt.datetime(today.year, today.month, today.day, tzinfo=_dt.timezone.utc).timestamp()
        else:
            cutoff = n - 86400
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM actions "
                "WHERE titan_id=? AND post_type=? AND created_at >= ?",
                (titan_id, self.name, cutoff),
            ).fetchone()
            return int(row["n"]) if row else 0
        finally:
            conn.close()

    def record_metadata_for_post(
        self,
        candidate: ArchetypeCandidate,
        tweet_id: str,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        """Build the actions.metadata JSON payload to persist alongside the
        post-write. Caller hands this back to the gateway's actions.metadata
        column (it's the gateway that owns the actions row write)."""
        meta: dict[str, Any] = dict(candidate.metadata)
        if self.metadata_key:
            meta[self.metadata_key] = candidate.source_id
        meta["archetype"] = candidate.archetype
        if candidate.pool:
            meta["pool"] = candidate.pool
        meta["tweet_id"] = tweet_id
        if extra:
            meta.update(extra)
        return json.dumps(meta, separators=(",", ":"), sort_keys=True)


__all__ = (
    "ArchetypeBase",
    "ArchetypeCandidate",
    "DEFAULT_CROSS_ARCHETYPE_SPACING_S",
)
