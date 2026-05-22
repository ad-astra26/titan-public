"""Adaptive engagement-attribution scoring substrate.

rFP_x_voice_enrichment §4.7. Used by every multi-pool archetype (OUTER_RUMINATION,
GROUNDED_TODAY, PRACTICED_RESPONSE, COMPOSED_THOUGHT, REFLECTION).

Mechanism
---------
1. On archetype post: write a *pending* row (`score = 0`) with the cited
   pool + source_id + ts. Pending rows are ignored by selection logic.
2. Reaper runs lazily (on every gateway tick that triggers a post attempt):
   for any pending row older than `OBSERVATION_WINDOW_SECONDS`, pull
   engagement from `engagement_snapshots` (events_teacher's existing
   capture — no new API call), score `+1` if landed, `-1` otherwise.
3. Selection (`select_pool`):
   * 5-day anti-starvation — any pool with no observed firing in the last
     `STARVATION_WINDOW_SECONDS` wins outright (forced rotation).
   * Otherwise: rolling-7-post sum per pool + candidate salience.
   * Tie-break: highest raw `relevance_score` of candidate.
   * Cold-start: every pool starts at 0, so first ~7 cycles are uniform.

The scoring table lives in `social_x.db` (created by `schema_migrations`).
The engagement source table lives in `events_teacher.db` — the events_teacher
worker is what writes engagement_snapshots, and this module READS it.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)


OBSERVATION_WINDOW_SECONDS = 12 * 3600        # 12 h
STARVATION_WINDOW_SECONDS = 5 * 86400         # 5 days
ROLLING_WINDOW_POSTS = 7

# rFP §4.7 "Landed" thresholds — tunable later via config.
LANDED_LIKES_THRESHOLD = 5
LANDED_REPLIES_THRESHOLD = 1
LANDED_REPOSTS_THRESHOLD = 1


# ── Write hooks ──────────────────────────────────────────────────────

def record_pending_post(
    db_path: str,
    *,
    titan_id: str,
    archetype: str,
    pool: str,
    source_id: str,
    tweet_id: str,
    ts: float | None = None,
) -> int | None:
    """Record a pending pool-scoring observation after an archetype post.

    The `tweet_id` is stored in `engagement_signals` so the reaper can
    look it up in events_teacher's engagement_snapshots later.
    """
    ts = ts if ts is not None else time.time()
    payload = json.dumps({"tweet_id": tweet_id, "status": "pending"},
                         separators=(",", ":"), sort_keys=True)
    conn = sqlite3.connect(db_path, timeout=5)
    try:
        cur = conn.execute(
            "INSERT INTO archetype_pool_scores "
            "(titan_id, archetype, pool, score, source_id, "
            "engagement_signals, ts) VALUES (?, ?, ?, 0, ?, ?, ?)",
            (titan_id, archetype, pool, source_id, payload, ts),
        )
        conn.commit()
        return cur.lastrowid
    except Exception as e:
        logger.warning("[pool_scoring] record_pending_post failed: %s", e)
        return None
    finally:
        conn.close()


def reap_pending(
    social_x_db: str,
    events_teacher_db: str,
    *,
    now: float | None = None,
) -> dict:
    """Resolve every pending row whose observation window has elapsed.

    Reads `engagement_snapshots` from events_teacher.db (latest snapshot per
    tweet_id), computes "landed"/"not landed", and updates the row.
    Returns a summary `{"observed": N, "landed": N, "lost": N, "skipped": N}`.
    """
    now = now if now is not None else time.time()
    summary = {"observed": 0, "landed": 0, "lost": 0, "skipped": 0}
    cutoff = now - OBSERVATION_WINDOW_SECONDS
    conn = sqlite3.connect(social_x_db, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, engagement_signals FROM archetype_pool_scores "
            "WHERE score = 0 AND ts <= ?",
            (cutoff,),
        ).fetchall()
        if not rows:
            return summary

        # Open events_teacher in read-only mode so we don't accidentally write.
        et = None
        try:
            et = sqlite3.connect(
                f"file:{events_teacher_db}?mode=ro", uri=True, timeout=5,
            )
            et.row_factory = sqlite3.Row
        except Exception as e:
            logger.warning("[pool_scoring] events_teacher.db unreachable: %s", e)
            summary["skipped"] = len(rows)
            return summary

        for row in rows:
            try:
                signals = json.loads(row["engagement_signals"] or "{}")
            except Exception:
                signals = {}
            tweet_id = signals.get("tweet_id", "")
            if not tweet_id:
                summary["skipped"] += 1
                continue
            snap = et.execute(
                "SELECT likes, replies, quotes, checked_at "
                "FROM engagement_snapshots WHERE tweet_id=? "
                "ORDER BY checked_at DESC LIMIT 1",
                (tweet_id,),
            ).fetchone()
            if not snap:
                summary["skipped"] += 1
                continue
            landed = (
                (snap["likes"] or 0)   >= LANDED_LIKES_THRESHOLD
                or (snap["replies"] or 0) >= LANDED_REPLIES_THRESHOLD
                or (snap["quotes"] or 0)  >= LANDED_REPOSTS_THRESHOLD
            )
            score = 1 if landed else -1
            engagement_signals = json.dumps({
                "tweet_id": tweet_id,
                "likes": int(snap["likes"] or 0),
                "replies": int(snap["replies"] or 0),
                "reposts": int(snap["quotes"] or 0),
                "observed_at": float(snap["checked_at"] or now),
                "status": "observed",
            }, separators=(",", ":"), sort_keys=True)
            conn.execute(
                "UPDATE archetype_pool_scores "
                "SET score=?, engagement_signals=? WHERE id=?",
                (score, engagement_signals, row["id"]),
            )
            summary["observed"] += 1
            summary["landed"] += 1 if landed else 0
            summary["lost"] += 0 if landed else 1
        conn.commit()
        if et is not None:
            et.close()
    finally:
        conn.close()
    return summary


# ── Selection ───────────────────────────────────────────────────────

def select_pool(
    db_path: str,
    *,
    titan_id: str,
    archetype: str,
    candidates: Mapping[str, Mapping[str, float]],
    now: float | None = None,
) -> str | None:
    """Pick the winning pool for this archetype cycle.

    Args:
        candidates: {pool_name: {"salience": float in [0,1], "relevance": float}}.
            Only pools that actually have a candidate this cycle should appear.
            If empty → returns None.

    Returns the chosen pool name, or None if no candidates.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return next(iter(candidates))

    now = now if now is not None else time.time()
    pools = list(candidates.keys())

    # Pull observed rows for this archetype.
    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT pool, score, ts FROM archetype_pool_scores "
            "WHERE titan_id=? AND archetype=? AND score != 0 "
            "ORDER BY ts DESC LIMIT 50",
            (titan_id, archetype),
        ).fetchall()
    finally:
        conn.close()

    # 5-day anti-starvation: a pool with NO observed firing in the last 5d
    # wins outright (forced rotation). Two pools both starved → use salience.
    starvation_cutoff = now - STARVATION_WINDOW_SECONDS
    last_observed: dict[str, float] = {}
    for r in rows:
        if r["pool"] not in last_observed:
            last_observed[r["pool"]] = float(r["ts"])
    starved = [
        p for p in pools
        if last_observed.get(p, 0.0) < starvation_cutoff
    ]
    if len(starved) == len(pools):
        # All starved (likely cold-start) — fall through to salience.
        pass
    elif starved:
        # Pick the most-starved pool with the highest salience.
        starved.sort(
            key=lambda p: (
                last_observed.get(p, 0.0),
                -candidates[p].get("salience", 0.0),
            )
        )
        return starved[0]

    # Rolling-7-post sum per pool.
    rolling: dict[str, int] = {p: 0 for p in pools}
    seen_per_pool: dict[str, int] = {p: 0 for p in pools}
    for r in rows:
        pool = r["pool"]
        if pool not in rolling:
            continue
        if seen_per_pool[pool] >= ROLLING_WINDOW_POSTS:
            continue
        rolling[pool] += int(r["score"])
        seen_per_pool[pool] += 1

    # priority = rolling-7-sum + salience (normalized [0,1])
    best_pool = pools[0]
    best_score = -1e9
    best_relevance = -1.0
    for p in pools:
        sal = candidates[p].get("salience", 0.0)
        priority = rolling[p] + sal
        rel = candidates[p].get("relevance", 0.0)
        if (priority, rel) > (best_score, best_relevance):
            best_score = priority
            best_relevance = rel
            best_pool = p
    return best_pool


# ── Read-only stats (for tests + observability) ─────────────────────

def get_stats(
    db_path: str,
    *,
    titan_id: str | None = None,
    archetype: str | None = None,
) -> dict:
    """Aggregate scoring stats for tests / observatory."""
    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        clauses, params = [], []
        if titan_id:
            clauses.append("titan_id=?")
            params.append(titan_id)
        if archetype:
            clauses.append("archetype=?")
            params.append(archetype)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"SELECT archetype, pool, score, COUNT(*) AS n "
            f"FROM archetype_pool_scores {where} "
            f"GROUP BY archetype, pool, score",
            tuple(params),
        ).fetchall()
        out: dict = {}
        for r in rows:
            arc = out.setdefault(r["archetype"], {})
            pool = arc.setdefault(r["pool"], {"pending": 0, "landed": 0, "lost": 0, "total": 0})
            n = int(r["n"])
            if r["score"] == 0:
                pool["pending"] = n
            elif r["score"] > 0:
                pool["landed"] = n
            else:
                pool["lost"] = n
            pool["total"] += n
        return out
    finally:
        conn.close()


__all__ = (
    "OBSERVATION_WINDOW_SECONDS",
    "STARVATION_WINDOW_SECONDS",
    "ROLLING_WINDOW_POSTS",
    "record_pending_post",
    "reap_pending",
    "select_pool",
    "get_stats",
)
