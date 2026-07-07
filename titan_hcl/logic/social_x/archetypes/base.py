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

import hashlib
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

# Same-archetype (intra) spacing default — minimum gap between two posts of
# the SAME archetype. The cross-archetype gate above deliberately excludes
# self (`post_type != ?`), so without this guard an archetype whose daily cap
# is ≥2 re-fires on the very next ~2h posting tick → two near-identical posts
# a couple of hours apart on the public timeline (Maker 2026-06-02). WORLD_MIRROR
# and AMPLIFY already enforced their own 6h intra-spacing; this default
# generalizes that fix to every archetype. PROOF_DAY opts out (it has its own
# once-per-UTC-day must-post slot).
DEFAULT_SAME_ARCHETYPE_SPACING_S = 6 * 3600

# Outer-world engagement archetypes — those that publicly reference / engage a
# specific external account. They share a single per-AUTHOR cooldown so one
# account (especially a large one like @lopp) is never reflected on / amplified
# more than once per cooldown window across ANY of these archetypes. Maker rule
# 2026-05-30: "we cannot have <big account> notified for days talking about the
# same post — makes zero sense / unprofessional."
OUTER_ENGAGEMENT_POST_TYPES = (
    "world_mirror", "outer_inner_bridge", "outer_rumination", "amplify",
)
DEFAULT_AUTHOR_COOLDOWN_S = 24 * 3600  # 24h (Maker 2026-06-13, RFP_fleet_x_
# engagement_coordination INV-FX-2 — the owned-author re-engage floor). Was 48h.
# Under the deterministic author-hash partition (INV-FX-1) a given author is
# only ever engaged by ONE Titan, so this LOCAL window is sufficient to bound
# the SHARED @your_x_handle account to ≤1 engagement/author/24h fleet-wide.
# Config-overridable via [social_x].author_cooldown_seconds.

# Fleet engagement roster default (RFP_fleet_x_engagement_coordination Q6).
# Config [social_x].engagement_fleet overrides; MUST be identical across boxes.
DEFAULT_ENGAGEMENT_FLEET = ("T1", "T2", "T3")


def normalize_handle(author: str) -> str:
    """Canonical handle form for hashing/matching (matches _self_handles)."""
    return str(author or "").strip().lstrip("@").lower()


def engagement_owner_for(author: str, roster) -> str:
    """The single Titan owning engagement for `author`, via a STABLE
    cross-process hash (sha256 — NOT builtin hash(), which is per-process
    salted → would differ across boxes/restarts → silent multi-tag). Pure;
    shared by ArchetypeBase (proactive engagement) and the reply cycle
    (mention-replies, INV-FX-7). Returns '' when undeterminable."""
    roster = tuple(str(t).strip() for t in (roster or ()) if str(t).strip())
    norm = normalize_handle(author)
    if not norm or not roster:
        return ""
    h = int(hashlib.sha256(norm.encode("utf-8")).hexdigest(), 16)
    return roster[h % len(roster)]


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

    def must_post_hard_capped(self, *, titan_id: str,
                              now: float | None = None) -> bool:
        """True if this must-post archetype is at ``MUST_POST_DAILY_HARD_CAP`` for
        the rolling 24h (counts **ANY** status incl ``failed`` — mirrors the gateway
        runaway backstop ``_check_rate_limits``). A hard-capped must-post MUST abstain
        in ``find_candidate`` so it does NOT monopolise dispatch and starve every
        other archetype — the failure that silenced a whole Titan for a day when its
        soul_diary X-post ``failed`` 3× during the 2026-07 proxy outage (the capped
        must-post kept being re-selected + blocked, so no other archetype ever got a
        turn). RFP_social_x §5.FX.4. Fail-OPEN (never wrongly abstain on a read error).
        """
        try:
            cfg = self.gateway._load_config() if self.gateway else {}
            hard_cap = int(cfg.get("must_post_daily_hard_cap", 3) or 0)
        except Exception:
            hard_cap = 3
        if hard_cap <= 0:
            return False
        n = now if now is not None else time.time()
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM actions WHERE action_type='post' "
                "AND post_type=? AND titan_id=? AND created_at > ?",
                (self.name, titan_id, n - 86400.0),
            ).fetchone()
            return bool(row) and int(row[0]) >= hard_cap
        except Exception:
            return False
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

    # Owned-account FLOOR — Titan's own X accounts, ALWAYS excluded regardless
    # of per-box config. INV-XENG-1 is a hard safety invariant ("never engage
    # your own account"); it must NOT depend on a config key being present.
    # 2026-06-13: T2+T3 config.toml had drifted WITHOUT `self_handles` under
    # [social_x], so _self_handles returned only {your_x_handle} (from user_name)
    # and the old "fallback only when the set is empty" never triggered →
    # @iamtitantech leaked and world_mirror quote-posted our own automation
    # account (twice). The floor is now always unioned, not a fallback.
    _OWNED_ACCOUNT_FLOOR = frozenset({"your_x_handle", "iamtitantech"})

    def _self_handles(self) -> set[str]:
        """Titan's OWN X handles — PERMANENTLY ineligible for outer engagement
        (never reply to / mirror / amplify our own account). Seeds the cooldown
        set so every outer archetype's existing `if author in cooldown` check
        skips them in one place. The owned-account floor is ALWAYS applied and
        unioned with config ([social_x] `self_handles` + `user_name`), so config
        drift cannot silently disable the exclusion. Cached per instance.
        (rFP X-post PART B / INV-XENG-1, 2026-06-03; floor-hardened 2026-06-13.)"""
        cached = getattr(self, "_self_handles_cache", None)
        if cached is not None:
            return cached
        # Start from the hard floor — never empty, never config-dependent.
        handles: set[str] = set(self._OWNED_ACCOUNT_FLOOR)
        try:
            cfg = self.gateway._load_config() if self.gateway else {}
            for raw in [cfg.get("user_name", "")] + list(cfg.get("self_handles", []) or []):
                h = str(raw or "").strip().lstrip("@").lower()
                if h:
                    handles.add(h)
        except Exception:
            pass
        self._self_handles_cache = handles
        return handles

    # ── Fleet engagement partition (RFP_fleet_x_engagement_coordination) ──
    # The three Titans share ONE X account (@your_x_handle). A deterministic
    # author-hash partition assigns each author to exactly ONE owning Titan, so
    # a human is engaged by ≤1 Titan/24h (INV-FX-1) with ZERO cross-box
    # coordination — every box computes identical ownership from the config
    # roster. Replaces the broken per-box-cooldown model for cross-Titan dedup.
    def engagement_roster(self) -> tuple[str, ...]:
        """Ordered fleet roster ([social_x].engagement_fleet). MUST match across
        boxes. Cached per instance."""
        cached = getattr(self, "_eng_roster_cache", None)
        if cached is not None:
            return cached
        roster: tuple[str, ...] = DEFAULT_ENGAGEMENT_FLEET
        try:
            cfg = self.gateway._load_config() if self.gateway else {}
            r = cfg.get("engagement_fleet") or []
            cleaned = tuple(str(x).strip() for x in r if str(x).strip())
            if cleaned:
                roster = cleaned
        except Exception:
            pass
        self._eng_roster_cache = roster
        return roster

    def engagement_owner(self, author: str) -> str:
        """The single Titan that owns engagement for `author` (delegates to the
        pure module-level `engagement_owner_for`). '' when undeterminable."""
        return engagement_owner_for(author, self.engagement_roster())

    def is_my_engagement_partition(self, author: str, titan_id: str) -> bool:
        """True if THIS Titan owns engagement for `author` (or partitioning is
        disabled). Used as an ADDITIVE candidate filter alongside the existing
        author-cooldown / self-handle skips. Fail-CLOSED on an undeterminable
        owner under an empty/misconfigured roster (safety > availability — a
        roster mismatch is surfaced by the startup WARN)."""
        try:
            cfg = self.gateway._load_config() if self.gateway else {}
            if not cfg.get("engagement_partition_enabled", True):
                return True
        except Exception:
            pass
        owner = self.engagement_owner(author)
        if not owner:
            # empty author → nothing to engage anyway; empty roster → misconfig
            # → fail-closed (don't risk the whole fleet engaging everything).
            return False
        return owner == str(titan_id or "")

    def authors_on_cooldown(
        self,
        *,
        titan_id: str,
        now: float | None = None,
        window_seconds: float | None = None,
    ) -> set[str]:
        """Lowercased handles engaged by ANY outer-engagement archetype within
        the cooldown window (cross-archetype per-author dedup, Maker 2026-05-30).

        Checks BOTH `metadata.author` (world_mirror, outer_inner_bridge,
        amplify) and `metadata.handle` (outer_rumination) since the archetypes
        historically used different metadata keys for the cited account.

        `window_seconds=None` → resolve from config [social_x].author_cooldown_
        seconds (default 24h, DEFAULT_AUTHOR_COOLDOWN_S). Under the fleet author
        partition this LOCAL window bounds the shared account to ≤1/author/24h.
        """
        if window_seconds is None:
            window_seconds = DEFAULT_AUTHOR_COOLDOWN_S
            try:
                cfg = self.gateway._load_config() if self.gateway else {}
                window_seconds = float(cfg.get(
                    "author_cooldown_seconds", DEFAULT_AUTHOR_COOLDOWN_S))
            except Exception:
                pass
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
            return set(self._self_handles())
        finally:
            conn.close()
        # B1 (INV-XENG-1): own handles are PERMANENTLY on cooldown — never engage
        # our own account. Seeds the set so all 4 outer archetypes' cooldown check
        # skips them; recently-engaged external authors are added below.
        out: set[str] = set(self._self_handles())
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

    def same_archetype_blocked(
        self,
        *,
        titan_id: str,
        now: float | None = None,
        spacing_seconds: float = DEFAULT_SAME_ARCHETYPE_SPACING_S,
    ) -> bool:
        """True if THIS archetype posted within the intra-spacing window.

        Enforces a minimum gap between two posts of the same archetype so a
        ≥2/day cap doesn't collapse into a back-to-back pair on the ~2h posting
        tick (Maker 2026-06-02). Counts only posts that actually reached — or
        are reaching — the timeline (`posted`/`verified`/`pending`/`unverified`);
        a `failed` attempt never appeared publicly, so it must not block the next
        try. (`unverified` = soft-failed but the tweet likely landed, 2026-06-13.)
        """
        cutoff = (now if now is not None else time.time()) - spacing_seconds
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM actions WHERE titan_id=? AND post_type=? "
                "AND status IN ('posted','verified','pending','unverified') "
                "AND created_at >= ? LIMIT 1",
                (titan_id, self.name, cutoff),
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
    "DEFAULT_SAME_ARCHETYPE_SPACING_S",
)
