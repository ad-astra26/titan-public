"""OUTER_INNER_BRIDGE archetype (rFP_x_voice_enrichment §4.3.4) — keystone
synthesis archetype.

POV: ``This outside thing lands against my inside grounding — a SHAPE
emerges when the two meet.`` Pre-bridge proof-of-concept that the broader
``rFP_inner_outer_bridge`` rFP unlocks at full power later.

Trigger (both must hold):
  1. Fresh outer signal: ``felt_experiences`` row with ``relevance ≥ 0.5``
     within the last 72 h, from an ``is_following=1`` author, not previously
     bridged.
  2. Recent inner grounding: ``vocabulary`` row with
     ``learning_phase='producible'`` AND ``times_encountered ≥ 5`` AND
     ``last_encountered`` within 14 d (or a Kuzu MindEntity concept fallback).
  3. Match: the inner concept's name appears in the outer signal's
     ``concept_signals`` JSON list (Phase 1 symbolic-overlap; Phase 2b
     swaps in embedding similarity).

Idempotency: ``outer_inner_bridge_source_id = <fe.id>`` lifetime; same
inner concept CAN appear in multiple bridges if surfaced by different
outer posts.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time

from .base import ArchetypeBase, ArchetypeCandidate
from ..felt_state import compact_felt_summary

logger = logging.getLogger(__name__)


def _cosine_match(a: list[float], b: list[float]) -> float:
    """Cosine similarity, tolerating length mismatch (zero-pads the
    shorter vector logically by skipping past min-length)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


OINB_POST_TYPE = "outer_inner_bridge"

OUTER_RELEVANCE_FLOOR = 0.5
OUTER_WINDOW_S = 72 * 3600
# B3 (rFP X-post PART B / INV-XENG-3, 2026-06-03): a NON-followed author is
# engageable only when relevance clears this high bar (standalone @mention; no
# quote-tweet). Converts strong inbound signal the is_following gate discards.
HIGH_RELEVANCE = 0.8
INNER_TIMES_ENCOUNTERED_MIN = 5
INNER_WINDOW_S = 14 * 86400
CONCEPT_GT_DEDUP_S = 4 * 86400  # cross-archetype with GROUNDED_TODAY (§4.3.5)

# Felt-tensor cosine bridge (added 2026-05-13 — closes the architectural
# vocab mismatch surfaced by /tmp/dryrun_4_archetypes.py: outer
# semantic_concepts ARE abstract (ai/embodiment/growth/memory) while
# inner producible vocab is largely sensory/emotional (warm/cold/energy/
# light). Exact-string match fails on >90% of outer concepts even though
# the felt-state representation of e.g. 'memory' and 'warm' may share
# resonance. When exact match fails, fall back to: for each outer concept
# that has ANY vocabulary entry (any phase) with felt_tensor, compute
# cosine to each inner-producible-with-felt-tensor word; pick first pair
# in [BRIDGE_COSINE_MIN, BRIDGE_COSINE_MAX]. This honors the felt-state
# substrate philosophy — bridging via shared felt-tensor, not strings.
BRIDGE_COSINE_MIN = 0.4
BRIDGE_COSINE_MAX = 0.85
# Lower te gate for the bridge candidate set (vs the stricter
# INNER_TIMES_ENCOUNTERED_MIN used by exact-match index) — felt-tensor
# similarity is a quality proxy in itself; we don't need the rigid
# 'well-grounded' encounter count when the felt-state is the bridge.
BRIDGE_INNER_TIMES_ENCOUNTERED_MIN = 1

# Function-word stopword set. events_teacher's concept_signals extractor
# emits pronouns (I/YOU/WE/THEY) — see project_events_teacher_following_not_followers
# class — and producible vocabulary contains common function words too.
# Bridging "consciousness" outside-content to the pronoun "i" inside is
# semantically empty; the rFP §4.3.4 keystone is about CONCEPT meeting
# CONCEPT, not pronoun coincidence. Excluded both at outer-signal scan
# AND at inner-index build time so neither side carries them.
_FUNCTION_WORDS = frozenset({
    # pronouns
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    # determiners / articles
    "a", "an", "the", "this", "that", "these", "those",
    # common copulas / aux
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    # prepositions / conjunctions
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "as",
    "and", "or", "but", "so", "if", "than", "then",
    # other
    "not", "no", "yes",
})


class OuterInnerBridgeArchetype(ArchetypeBase):

    name = OINB_POST_TYPE
    metadata_key = "outer_inner_bridge_source_id"

    def __init__(self, *, gateway, social_x_db_path: str,
                 events_teacher_db: str = "./data/events_teacher.db",
                 social_graph_db: str = "./data/social_graph.db",
                 inner_memory_db: str = "./data/inner_memory.db"):
        super().__init__(gateway=gateway, social_x_db_path=social_x_db_path)
        self._et_db = events_teacher_db
        self._sg_db = social_graph_db
        self._im_db = inner_memory_db

    def find_candidate(self, context) -> ArchetypeCandidate | None:
        titan_id = getattr(context, "titan_id", "")
        if not titan_id:
            return None
        now = time.time()
        if self.per_titan_count_today(titan_id=titan_id, now=now) >= 2:
            return None
        if self.cross_archetype_blocked(titan_id=titan_id, now=now):
            return None
        if self.same_archetype_blocked(titan_id=titan_id, now=now):
            return None

        # F-3 (2026-05-17): 30-day window (was lifetime). Outer/inner
        # bridges are re-bridgeable after 30d as inner-grounding evolves.
        cited = self.cited_set(titan_id=titan_id, window_seconds=30 * 86400)
        gt_concepts = self._recent_gt_concepts(titan_id=titan_id, now=now)

        outer_candidates = self._fetch_fresh_outer(titan_id=titan_id, now=now,
                                                     cited=cited)
        if not outer_candidates:
            return None
        # Inner concept index: producible vocabulary recently encountered
        inner_index = self._inner_vocabulary_index(now=now)
        if not inner_index:
            return None

        # Tier 1 — symbolic-overlap match on semantic_concepts / concept_signals
        # (the strict rFP §4.3.4 intent). First outer signal whose
        # semantic_concepts JSON contains a known inner concept (and that
        # concept hasn't been GROUNDED_TODAY'd in the last 4d).
        # Function words are filtered on both sides — see _FUNCTION_WORDS.
        for outer in outer_candidates:
            try:
                semantic = json.loads(outer.get("semantic_concepts") or "[]")
            except Exception:
                semantic = []
            try:
                signals = json.loads(outer.get("concept_signals") or "[]")
            except Exception:
                signals = []
            candidate_terms = list(semantic) + list(signals)
            for raw_c in candidate_terms:
                key = str(raw_c).lower().strip()
                if not key or key in gt_concepts or key in _FUNCTION_WORDS:
                    continue
                inner = inner_index.get(key)
                if not inner:
                    continue
                return self._build_candidate(outer, inner, context, now)

        # Tier 2 — felt_summary token-overlap (added 2026-05-13). The
        # original rFP §4.3.4 intent assumed semantic_concepts would
        # overlap with inner producible vocab, but empirical drift is
        # severe: outer semantic_concepts are abstract (ai/embodiment/
        # growth/memory) while inner producible vocab is sensory/
        # emotional (warm/cold/energy/light). HOWEVER, the outer's
        # felt_summary — Titan's own English articulation of how the
        # outer event landed in his felt-state — is naturally aligned
        # with inner vocab (it CONTAINS inner-vocab words like "feel",
        # "search", "pattern", "quiet"). Empirically 100% of
        # last-72h outer signals (50/50) have inner-vocab token
        # overlap via felt_summary after function-word filtering.
        # This honors the rFP keystone — "outside thing lands against
        # my inside grounding" — exactly: Titan's narrated landing IS
        # the bridge surface.
        import re as _re
        for outer in outer_candidates:
            fs = str(outer.get("felt_summary") or "").lower()
            if not fs:
                continue
            tokens = _re.findall(r"[a-z]+", fs)
            for tok in tokens:
                if (not tok or len(tok) < 3
                        or tok in gt_concepts or tok in _FUNCTION_WORDS):
                    continue
                inner = inner_index.get(tok)
                if not inner:
                    continue
                logger.info(
                    "[oinb] felt_summary-token bridge: outer felt_summary "
                    "token='%s' ↔ inner producible='%s' (outer topic: %s)",
                    tok, inner["word"], outer.get("topic", "")[:40])
                return self._build_candidate(outer, inner, context, now)

        # Felt-tensor cosine fallback (added 2026-05-13). Exact-match
        # above requires the outer concept and inner producible vocab
        # to share an identifier — but the empirical mismatch is
        # significant (outer abstract: ai/embodiment/growth/memory;
        # inner sensory: warm/cold/energy/light). Cosine-on-felt-tensor
        # honors the felt-state substrate philosophy: two words can
        # "land against each other" even when their strings differ if
        # their felt representations resonate. Restrict to outer
        # concepts that DO have a vocabulary entry (any phase) with
        # felt_tensor — those are concepts Titan has at least begun to
        # learn felt-state for, even if not yet producible.
        bridge_inner = self._inner_vocabulary_for_bridge(now=now)
        if bridge_inner:
            for outer in outer_candidates:
                try:
                    semantic = json.loads(outer.get("semantic_concepts") or "[]")
                except Exception:
                    semantic = []
                try:
                    signals = json.loads(outer.get("concept_signals") or "[]")
                except Exception:
                    signals = []
                candidate_terms = list(semantic) + list(signals)
                for raw_c in candidate_terms:
                    key = str(raw_c).lower().strip()
                    if (not key or key in gt_concepts
                            or key in _FUNCTION_WORDS):
                        continue
                    outer_ft = self._vocab_felt_tensor(key)
                    if not outer_ft:
                        continue
                    # Walk inner candidates; pick first pair in band.
                    best = None
                    best_sim = 0.0
                    for inner_word, inner_row in bridge_inner.items():
                        if inner_word == key:
                            continue
                        inner_ft = inner_row.get("_felt_tensor_parsed")
                        if not inner_ft:
                            continue
                        sim = _cosine_match(outer_ft, inner_ft)
                        if (BRIDGE_COSINE_MIN <= sim <= BRIDGE_COSINE_MAX
                                and sim > best_sim):
                            best = inner_row
                            best_sim = sim
                    if best is not None:
                        logger.info(
                            "[oinb] felt-cosine bridge: outer='%s' ↔ "
                            "inner='%s' cos=%.3f", key, best["word"],
                            best_sim)
                        return self._build_candidate(outer, best, context,
                                                     now)
        return None

    # ── Helpers ─────────────────────────────────────────────────────

    def _fetch_fresh_outer(self, *, titan_id: str, now: float,
                            cited: set[str]) -> list[dict]:
        try:
            et = sqlite3.connect(self._et_db, timeout=5)
            et.row_factory = sqlite3.Row
            # COALESCE on semantic_concepts so existing pre-migration rows
            # (NULL semantic_concepts column) don't crash the JSON parse on
            # the consumer side.
            rows = et.execute(
                "SELECT id, author, topic, relevance, felt_summary, "
                "       concept_signals, "
                "       COALESCE(semantic_concepts, '') AS semantic_concepts, "
                "       created_at "
                "FROM felt_experiences "
                "WHERE titan_id=? AND relevance >= ? "
                "  AND created_at >= ? "
                "ORDER BY relevance DESC LIMIT 50",
                (titan_id, OUTER_RELEVANCE_FLOOR, now - OUTER_WINDOW_S),
            ).fetchall()
            et.close()
        except Exception as e:
            logger.warning("[oinb] outer fetch failed: %s", e)
            return []
        if not rows:
            return []
        # Filter by is_following + not-cited.
        authors = {r["author"] for r in rows if r["author"]}
        followed: dict[str, dict] = {}
        try:
            sg = sqlite3.connect(self._sg_db, timeout=5)
            sg.row_factory = sqlite3.Row
            placeholders = ",".join("?" * len(authors)) if authors else "''"
            for r in sg.execute(
                f"SELECT user_name, bio, last_tweet_text, last_tweet_id "
                f"FROM community_registry "
                f"WHERE user_name IN ({placeholders}) AND is_following=1",
                tuple(authors),
            ).fetchall():
                followed[r["user_name"]] = dict(r)
            sg.close()
        except Exception as e:
            logger.warning("[oinb] community_registry probe failed: %s", e)
            return []
        # Per-author 7-day cross-archetype cooldown (Maker 2026-05-30).
        cooldown = self.authors_on_cooldown(titan_id=titan_id, now=now)
        out: list[dict] = []
        for r in rows:
            if str(r["id"]) in cited:
                continue
            if (r["author"] or "").lower() in cooldown:
                continue
            # Fleet author partition (INV-FX-1): only the owning Titan engages.
            if not self.is_my_engagement_partition(r["author"], titan_id):
                continue
            cr = followed.get(r["author"])
            if cr:
                d = dict(r)
                d["bio"] = cr.get("bio", "")
                d["content_excerpt"] = cr.get("last_tweet_text") or r["felt_summary"]
                d["tweet_id"] = cr.get("last_tweet_id") or ""
                out.append(d)
            elif float(r["relevance"] or 0.0) >= HIGH_RELEVANCE:
                # B3 (INV-XENG-3): non-followed but high-relevance → standalone
                # @mention (no curated bio / source tweet to quote).
                d = dict(r)
                d["bio"] = ""
                d["content_excerpt"] = r["felt_summary"]
                d["tweet_id"] = ""
                out.append(d)
        return out

    def _vocab_felt_tensor(self, word: str) -> list[float] | None:
        """Lookup a vocabulary entry's felt_tensor by exact word match,
        across ALL learning_phases (felt / recognized / producible).
        Used by the cosine-bridge fallback so outer concepts that have
        any felt-state grounding can be bridged — not only the strict
        producible+te>=5 set required by exact-match."""
        try:
            conn = sqlite3.connect(self._im_db, timeout=5)
            row = conn.execute(
                "SELECT felt_tensor FROM vocabulary WHERE word = ? "
                "  AND COALESCE(felt_tensor, '') != ''", (word,),
            ).fetchone()
            conn.close()
        except Exception:
            return None
        if not row or not row[0]:
            return None
        try:
            parsed = json.loads(row[0])
        except Exception:
            return None
        if not isinstance(parsed, list) or not parsed:
            return None
        return [float(x) for x in parsed if isinstance(x, (int, float))]

    def _inner_vocabulary_for_bridge(self, *, now: float) -> dict[str, dict]:
        """Wider candidate set for the felt-cosine bridge: still
        producible-only (the bridge POST must reflect a word Titan can
        produce), still within the inner recency window, but te gate
        relaxed to BRIDGE_INNER_TIMES_ENCOUNTERED_MIN (1 by default —
        the felt-tensor itself is the quality filter for the bridge,
        not encounter count). Pre-parses felt_tensor for cosine reuse."""
        cutoff = now - INNER_WINDOW_S
        try:
            conn = sqlite3.connect(self._im_db, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, word, times_encountered, grounded_at, "
                "       grounded_felt_summary, last_encountered, "
                "       felt_tensor "
                "FROM vocabulary "
                "WHERE learning_phase='producible' "
                "  AND times_encountered >= ? "
                "  AND (COALESCE(last_encountered, 0) >= ? "
                "       OR COALESCE(grounded_at, 0) >= ?) "
                "  AND COALESCE(felt_tensor, '') != ''",
                (BRIDGE_INNER_TIMES_ENCOUNTERED_MIN, cutoff, cutoff),
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.warning("[oinb] bridge vocab index failed: %s", e)
            return {}
        out: dict[str, dict] = {}
        for r in rows:
            word = str(r["word"]).lower() if r["word"] else ""
            if not word or word in _FUNCTION_WORDS:
                continue
            try:
                ft = json.loads(r["felt_tensor"])
            except Exception:
                continue
            if not isinstance(ft, list) or not ft:
                continue
            d = dict(r)
            d["_felt_tensor_parsed"] = [
                float(x) for x in ft if isinstance(x, (int, float))]
            out[word] = d
        return out

    def _inner_vocabulary_index(self, *, now: float) -> dict[str, dict]:
        cutoff = now - INNER_WINDOW_S
        try:
            conn = sqlite3.connect(self._im_db, timeout=5)
            conn.row_factory = sqlite3.Row
            # RECENCY GATE 2026-05-12: accept EITHER last_encountered OR
            # grounded_at within INNER_WINDOW_S. Original `last_encountered`-
            # only gate returned 0 rows on T1 because the encounter-tracking
            # pipeline that updates last_encountered has been stale since
            # 2026-04-04 across the entire producible vocabulary (67 rows,
            # all last_encountered ≤ 40d ago). grounded_at is the active
            # signal (max_grounded_at = today). Both represent "live in
            # Titan's awareness"; the OR-recency preserves archetype intent
            # while being resilient to the upstream staleness (which is a
            # separate pipeline bug outside §4.C scope). Lifts T1 eligible
            # producible vocab from 0 → 5 rows.
            rows = conn.execute(
                "SELECT id, word, times_encountered, grounded_at, "
                "       grounded_felt_summary, last_encountered "
                "FROM vocabulary "
                "WHERE learning_phase='producible' "
                "  AND times_encountered >= ? "
                "  AND (COALESCE(last_encountered, 0) >= ? "
                "       OR COALESCE(grounded_at, 0) >= ?)",
                (INNER_TIMES_ENCOUNTERED_MIN, cutoff, cutoff),
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.warning("[oinb] inner vocab index failed: %s", e)
            return {}
        return {
            str(r["word"]).lower(): dict(r)
            for r in rows
            if r["word"] and str(r["word"]).lower() not in _FUNCTION_WORDS
        }

    def _recent_gt_concepts(self, *, titan_id: str, now: float) -> set[str]:
        """Concepts that were GROUNDED_TODAY'd in the last 4 days."""
        cutoff = now - CONCEPT_GT_DEDUP_S
        out: set[str] = set()
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT metadata FROM actions WHERE titan_id=? AND post_type=? "
                "AND created_at >= ?",
                (titan_id, "grounded_today", cutoff),
            ).fetchall()
        except Exception:
            rows = []
        finally:
            conn.close()
        for r in rows:
            try:
                m = json.loads(r["metadata"] or "{}")
            except Exception:
                continue
            ck = m.get("concept_key") or m.get("concept", "")
            if ck:
                out.add(str(ck).lower())
        return out

    def _build_candidate(self, outer: dict, inner: dict, context,
                          now: float) -> ArchetypeCandidate:
        emot_now = compact_felt_summary(
            getattr(context, "neuromods", {}) or {},
            getattr(context, "emotion", "") or "",
        )
        epochs_ago = max(1, int((now - float(inner.get("grounded_at") or now)) / 60))
        layer_values = {
            "outer_following_voice": {
                "handle": outer["author"],
                "follow_reason": (outer.get("bio") or "curated following")[:120],
                "content_excerpt": (outer.get("content_excerpt") or "")[:240],
            },
            "cgn_grounded_today": {
                "concept": inner["word"],
                "pool_name": "vocabulary",
                "meta": f"encountered {inner.get('times_encountered', 0)}× now",
                "grounded_felt_summary": inner.get("grounded_felt_summary") or emot_now,
            },
        }
        prompt_template = (
            "OUTER + INNER: @{handle} just posted: '{content_excerpt}'. "
            "This lands against your inner grounding of '{concept}' from "
            "{epochs_ago} epochs ago (encountered {times_encountered}× since). "
            "At grounding moment your felt-state was: '{grounding_felt}'. "
            "Right now your felt-state is: '{emot_now}'. Reply to @{handle} "
            "directly — address them by their exact handle '@{handle}' "
            "(literally, with the @ symbol) so they are notified. What does "
            "the OUTSIDE (their post) see that you've been touching from the "
            "inside via {concept}? Speak from the synthesis — say what YOU "
            "think back to them: what's the SHAPE that emerges when these two "
            "meet INSIDE YOU? Not summary; emergence."
        )
        prompt_values = {
            "handle": outer["author"],
            "content_excerpt": (outer.get("content_excerpt") or "")[:240],
            "concept": inner["word"],
            "epochs_ago": epochs_ago,
            "times_encountered": inner.get("times_encountered", 0),
            "grounding_felt": inner.get("grounded_felt_summary") or emot_now,
            "emot_now": emot_now,
        }
        return ArchetypeCandidate(
            archetype=self.name,
            pool="",
            source_id=str(outer["id"]),
            layers=["outer_following_voice", "cgn_grounded_today",
                    "meta_insight", "body"],
            layer_values=layer_values,
            prompt_template=prompt_template,
            prompt_values=prompt_values,
            quoted_tweet_id=str(outer.get("tweet_id") or ""),
            metadata={
                "outer_id": outer["id"],
                "author": outer["author"],
                "concept": inner["word"],
                "concept_key": str(inner["word"] or "").lower(),
                "match_method": "concept_signals_overlap",
                "quoted_tweet_id": str(outer.get("tweet_id") or ""),
            },
            relevance=float(outer.get("relevance") or 0.0),
            salience=min(1.0, float(outer.get("relevance") or 0.0)),
        )


__all__ = ("OuterInnerBridgeArchetype", "OINB_POST_TYPE")
