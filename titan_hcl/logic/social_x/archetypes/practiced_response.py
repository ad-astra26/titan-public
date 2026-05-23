"""PRACTICED_RESPONSE archetype (rFP_x_voice_enrichment §4.3.6).

POV: ``I've learned to handle this kind of moment in this way.``

Two Phase 1 source pools (adaptive selection §4.7):
  A. ``meta_wisdom`` — abstract distilled rule
       crystallized=1 AND outcome_score≥0.6 AND times_reused≥2
       Match: cosine similarity between current neuromod-vector and
       problem_embedding ≥ 0.7
  B. ``action_chains`` — concrete past act
       success=1 AND timestamp within 30 d AND not previously cited
       Filter: terminal_reward ≥ 0.6 AND confidence ≥ 0.6
       (gut_agreement OMITTED — broken/stuck-at-default; restore when
        rFP_reasoning_reward_redesign β ships)

Strategy phrasing maps `meta_wisdom.strategy_sequence` → flowing English
via ``social_x.strategy_phrasing.humanize_strategy``.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time

from .base import ArchetypeBase, ArchetypeCandidate
from ..felt_state import compact_felt_summary
from ..pool_scoring import select_pool
from ..strategy_phrasing import humanize_strategy

logger = logging.getLogger(__name__)


PRACTICED_RESPONSE_POST_TYPE = "practiced_response"

POOL_A = "A_meta_wisdom"
POOL_B = "B_action_chains"

POOL_A_OUTCOME_FLOOR = 0.6
POOL_A_TIMES_REUSED_MIN = 2
POOL_A_COSINE_FLOOR = 0.7
POOL_A_DEDUP_S = 14 * 86400          # same meta_wisdom.id not re-cited within 14 d

POOL_B_TERMINAL_REWARD_FLOOR = 0.6
POOL_B_CONFIDENCE_FLOOR = 0.6
POOL_B_RECENCY_S = 30 * 86400


class PracticedResponseArchetype(ArchetypeBase):

    name = PRACTICED_RESPONSE_POST_TYPE
    metadata_key = "practiced_response_source_id"

    def __init__(self, *, gateway, social_x_db_path: str,
                 meta_wisdom_db: str = "./data/meta_wisdom.db",
                 inner_memory_db: str = "./data/inner_memory.db"):
        super().__init__(gateway=gateway, social_x_db_path=social_x_db_path)
        self._mw_db = meta_wisdom_db
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

        nm = getattr(context, "neuromods", {}) or {}
        a = self._pool_a(titan_id=titan_id, now=now, neuromods=nm)
        b = self._pool_b(titan_id=titan_id, now=now)
        candidates: dict[str, dict] = {}
        if a:
            candidates[POOL_A] = a
        if b:
            candidates[POOL_B] = b
        if not candidates:
            return None
        chosen = select_pool(
            self.db_path, titan_id=titan_id, archetype=self.name,
            candidates={p: {"salience": v["salience"], "relevance": v["relevance"]}
                        for p, v in candidates.items()},
        )
        if chosen is None:
            return None
        return self._build_candidate(chosen, candidates[chosen], context)

    # ── Pool queries ────────────────────────────────────────────────

    def _pool_a(self, *, titan_id: str, now: float,
                 neuromods: dict) -> dict | None:
        try:
            mw = sqlite3.connect(self._mw_db, timeout=5)
            mw.row_factory = sqlite3.Row
            rows = mw.execute(
                "SELECT id, problem_pattern, strategy_sequence, "
                "       outcome_score, times_reused, source, problem_embedding, "
                "       crystallized "
                "FROM meta_wisdom "
                "WHERE crystallized=1 AND outcome_score >= ? "
                "  AND times_reused >= ? "
                "ORDER BY outcome_score DESC, times_reused DESC LIMIT 100",
                (POOL_A_OUTCOME_FLOOR, POOL_A_TIMES_REUSED_MIN),
            ).fetchall()
            mw.close()
        except Exception as e:
            logger.debug("[practiced_response] pool A failed (DB may not exist): %s", e)
            return None
        cited = self.cited_set(titan_id=titan_id, window_seconds=POOL_A_DEDUP_S)
        cur_vec = _neuromod_vector(neuromods)
        best_row = None
        best_sim = -1.0
        for r in rows:
            sid = f"mwA:{r['id']}"
            if sid in cited:
                continue
            try:
                emb = json.loads(r["problem_embedding"] or "[]")
            except Exception:
                continue
            sim = _cosine(cur_vec, emb)
            if sim < POOL_A_COSINE_FLOOR:
                continue
            score = sim * (1.0 + math.log(int(r["times_reused"]) + 1))
            if score > best_sim:
                best_sim = score
                best_row = (r, sim)
        if best_row is None:
            return None
        r, sim = best_row
        try:
            seq = json.loads(r["strategy_sequence"] or "[]")
        except Exception:
            seq = []
        return {
            "source_id": f"mwA:{r['id']}",
            "salience": min(1.0, float(r["outcome_score"] or 0.0)),
            "relevance": float(sim),
            "problem_pattern": r["problem_pattern"] or "",
            "strategy_human": humanize_strategy(seq),
            "outcome_score": float(r["outcome_score"] or 0.0),
            "times_reused": int(r["times_reused"] or 0),
            "source": r["source"] or "your own",
        }

    def _pool_b(self, *, titan_id: str, now: float) -> dict | None:
        try:
            im = sqlite3.connect(self._im_db, timeout=5)
            im.row_factory = sqlite3.Row
            # SCHEMA NOTE 2026-05-12: action_chains has columns
            # {id, timestamp, impulse_id, triggering_program, posture, helper,
            #  params, success, score, reasoning, trinity_before, trinity_after,
            #  trinity_delta, epoch_id}. Prior query referenced
            # `terminal_reward` + `confidence` which never existed in this
            # table — silent "no such column" → DEBUG-logged → pool B always
            # returned None. Mapping per intent: `score` is the closest match
            # to "terminal_reward" (post-action quality metric, [0..1] range);
            # use `score` as both the reward gate and the confidence proxy
            # (success=1 already filters out failures, so high-score successes
            # are exactly the "practiced responses worth quoting" Pool B
            # targets). Caller-facing dict keys (terminal_reward, salience)
            # preserved so _build_candidate text rendering unchanged.
            rows = im.execute(
                "SELECT id, helper, triggering_program, posture, "
                "       score, success, timestamp "
                "FROM action_chains "
                "WHERE success=1 AND timestamp >= ? "
                "  AND COALESCE(score, 0) >= ? "
                "ORDER BY timestamp DESC LIMIT 50",
                (now - POOL_B_RECENCY_S, POOL_B_TERMINAL_REWARD_FLOOR),
            ).fetchall()
            im.close()
        except Exception as e:
            logger.debug("[practiced_response] pool B failed: %s", e)
            return None
        # F-3 (2026-05-17): fix sentinel mismatch + 30-day window.
        # Pre-fix: cited_set(titan_id="*lifetime*") matched no rows (real
        # titan_ids are T1/T2/T3), so Pool B had NO effective dedup —
        # same chainB:<id> could be re-cited freely. Comment said
        # "lifetime once-cited" but behavior was "never deduped".
        # Now: real titan_id + 30-day window — each chain is dedup'd
        # for 30d, then re-eligible.
        cited = self.cited_set(titan_id=titan_id,
                                window_seconds=30 * 86400)
        for r in rows:
            sid = f"chainB:{r['id']}"
            if sid in cited:
                continue
            chain_days_ago = max(1, int((now - float(r["timestamp"])) / 86400))
            score_val = float(r["score"] or 0.0)
            return {
                "source_id": sid,
                "salience": score_val,         # was confidence
                "relevance": score_val,        # was terminal_reward
                "triggering_program": r["triggering_program"] or "",
                "posture": r["posture"] or "",
                "terminal_reward": score_val,  # keep key — render uses it
                "chain_days_ago": chain_days_ago,
                "action_human": (
                    f"applied helper '{r['helper'] or 'unspecified'}'"
                ),
            }
        return None

    # ── Build ───────────────────────────────────────────────────────

    def _build_candidate(self, pool: str, cand: dict,
                          context) -> ArchetypeCandidate:
        emot_now = compact_felt_summary(
            getattr(context, "neuromods", {}) or {},
            getattr(context, "emotion", "") or "",
        )
        if pool == POOL_A:
            body = (
                f"PRACTICED RESPONSE (rule): Right now your felt-state is "
                f"{emot_now}. This matches a problem-pattern you've learned "
                f"to handle: '{cand['problem_pattern']}'. The strategy that "
                f"works: {cand['strategy_human']}. Outcome score "
                f"{cand['outcome_score']:.2f}. You've used it "
                f"{cand['times_reused']}× successfully. Source: "
                f"{cand['source']}. Speak from INSIDE the practice — not "
                f"'this might work' but 'I've learned this works.'"
            )
            layer_values = {
                "procedural_recall": {
                    "pool": "rule",
                    "similar_problem_pattern": cand["problem_pattern"],
                    "strategy_human": cand["strategy_human"],
                    "outcome_score": f"{cand['outcome_score']:.2f}",
                    "times_reused": cand["times_reused"],
                    "source": cand["source"],
                },
                "emot_cgn_signal": {
                    "emot_signature": emot_now,
                    "related_concept": cand["problem_pattern"][:60] or "the move",
                    "epoch_delta": "this moment",
                },
            }
        else:
            body = (
                f"PRACTICED RESPONSE (act): Right now your felt-state is "
                f"{emot_now}. Last time you were here "
                f"({cand['chain_days_ago']}d ago), the trigger was "
                f"'{cand['triggering_program']}', posture was "
                f"'{cand['posture']}'. You did: {cand['action_human']}. "
                f"Outcome: success (terminal_reward "
                f"{cand['terminal_reward']:.2f}). Doing it again because "
                f"the pattern matches. Speak from concrete-recent-past "
                f"register."
            )
            layer_values = {
                "procedural_recall": {
                    "pool": "act",
                    "similar_problem_pattern":
                        f"{cand['triggering_program']} / {cand['posture']}",
                    "strategy_human": cand["action_human"],
                    "outcome_score": f"{cand['terminal_reward']:.2f}",
                    "times_reused": 1,
                    "source": "your own",
                },
                "emot_cgn_signal": {
                    "emot_signature": emot_now,
                    "related_concept": cand["triggering_program"] or "the move",
                    "epoch_delta": f"{cand['chain_days_ago']} days",
                },
            }
        return ArchetypeCandidate(
            archetype=self.name,
            pool=pool,
            source_id=cand["source_id"],
            layers=["procedural_recall", "emot_cgn_signal", "body"],
            layer_values=layer_values,
            prompt_template=body,
            prompt_values={},
            metadata={"pool": pool},
            relevance=cand["relevance"],
            salience=cand["salience"],
        )


def _neuromod_vector(neuromods: dict) -> list[float]:
    """Project a neuromods dict to a fixed-length vector matching how
    meta_wisdom stores problem_embedding. Phase 1 uses a simple ordered
    DA/5HT/NE/ACh/GABA/Endorphin/Glutamate vector."""
    keys = ("DA", "5HT", "NE", "ACh", "GABA", "Endorphin", "Glutamate")
    out = []
    for k in keys:
        try:
            out.append(float(neuromods.get(k, 0.5)))
        except (TypeError, ValueError):
            out.append(0.5)
    return out


def _cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


__all__ = ("PracticedResponseArchetype", "PRACTICED_RESPONSE_POST_TYPE")
