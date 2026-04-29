"""
titan_plugin/logic/meta_teacher_memory.py — Phase B of rFP_meta_teacher_v2.

Two-tier memory for the Meta-Teacher:

  Hot tier — in-memory deque of full critique records (default 1000), with
  384D embedding index over topic_key for hybrid similarity×importance
  retrieval. Reuses `sentence_transformers` all-MiniLM-L6-v2 (already in
  use by sage.recorder + sage.guardian → no new dep).

  Cold tier — per-topic-key distilled journal at
  `data/meta_teacher/teaching_journal.jsonl`. Each entry carries first_seen,
  last_seen, critique_count, adoption_trajectory, quality_trajectory,
  quality_delta, still_needs_push flag, last_voice_applied, summary_cache.
  Never rolls off by age; inactive ≥90d → compressed archive
  (`teaching_journal.archive.jsonl.gz`).

  Retrieval scoring (rFP §2 Phase B):
    score = cos_similarity × importance_weight
    importance_weight =
        1.0
      + 0.3 · recency_boost (0..1 by ts age mapped onto 7d window)
      + 0.5 · adoption_rate  ∈ [0, 1]
      + 0.3 · quality_delta_sign  (+1 / 0 / -1)
      + 0.4 · still_needs_push_flag  (0 or 1)
    clamped to [0.5, 3.0]

  Top-3 hot-tier hits above similarity threshold (default 0.6) + top-1
  cold-tier exact topic_key match enter the teacher's user prompt as
  "similar past critiques you issued for this topic."

See rFP_meta_teacher_v2_content_awareness_memory.md §2 Phase B for full
design rationale and guardrails (still_needs_push surfacing, Maker INFO
cadence, topic_key stability rules).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import time
from collections import deque
from typing import Any, Optional

logger = logging.getLogger("titan.meta_teacher_memory")


# ── Module-level constants ─────────────────────────────────────────────────

JOURNAL_FILENAME = "teaching_journal.jsonl"
ARCHIVE_FILENAME = "teaching_journal.archive.jsonl.gz"
ST_MODEL_NAME = "all-MiniLM-L6-v2"   # 384D, matches other consumers
ST_EMBED_DIM = 384                     # explicit for test fixtures + type checks


# ── Topic-key canonicalization ─────────────────────────────────────────────

def canonical_topic_key(
    outer_summary: Optional[dict], primitives_used: Optional[list] = None,
    domain: str = "", first_step_arg: Optional[str] = None,
) -> str:
    """Derive the topic_key for a critique.

    Post-outer-layer (rFP §2 Phase B): topics are REAL strings. Prefer
    "<current_topic>|person=<primary_person>" when both are present.
    Fallback ladder:
        1. current_topic + primary_person
        2. current_topic only
        3. primary_person only
        4. inner::<first_primitive>::<domain>  (inner-state-only chains)
        5. inner::unknown::<domain>             (ultimate fallback)
    """
    os_ = outer_summary or {}
    topic = (os_.get("current_topic") or "").strip().lower()
    person = (os_.get("primary_person") or "").strip()
    if person and not person.startswith("@"):
        person = "@" + person.lower()
    elif person:
        person = "@" + person[1:].lower()

    if topic and person:
        return f"{topic}|person={person}"
    if topic:
        return topic
    if person:
        return f"person={person}"
    # Inner-state-only fallback: primitive + domain
    prims = list(primitives_used or [])
    first_prim = prims[0] if prims else "unknown"
    dom = (domain or "general").strip().lower()
    label = f"inner::{first_prim}::{dom}"
    if first_step_arg:
        label += f"::{str(first_step_arg)[:30].strip().lower()}"
    return label


# ── Embedding index (lazy ST load) ─────────────────────────────────────────

class _EmbedIndex:
    """Tiny in-memory embedding index. Caller supplies text, we vectorize +
    cosine-match. Backed by sentence_transformers (lazy import on first use)
    — None-returns are a soft failure that reduces retrieval to zero hits
    rather than crashing the teacher.
    """

    def __init__(self):
        self._model = None
        self._np = None
        self._init_attempted = False
        self._available = False

    def _lazy_init(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            from sentence_transformers import SentenceTransformer  # lazy
            import numpy as _np
            self._model = SentenceTransformer(ST_MODEL_NAME)
            self._np = _np
            self._available = True
            logger.info("[TeacherMemory] Embedding model loaded: %s (dim=%d)",
                        ST_MODEL_NAME, ST_EMBED_DIM)
        except Exception as e:
            logger.warning(
                "[TeacherMemory] Embedding init failed (retrieval disabled): %s",
                e)
            self._available = False

    @property
    def available(self) -> bool:
        if not self._init_attempted:
            self._lazy_init()
        return self._available

    def embed(self, text: str):
        """Return 384D unit-normalized vector or None if unavailable."""
        if not self.available:
            return None
        try:
            vec = self._model.encode(text or "", convert_to_numpy=True)
            if vec is None or getattr(vec, "size", 0) == 0:
                return None
            # L2-normalize for cosine = dot product
            norm = float(self._np.linalg.norm(vec))
            if norm < 1e-9:
                return None
            return (vec / norm).astype(self._np.float32)
        except Exception as e:
            logger.debug("[TeacherMemory] embed failed: %s", e)
            return None

    def cosine(self, a, b) -> float:
        if a is None or b is None or self._np is None:
            return 0.0
        try:
            return float(self._np.dot(a, b))
        except Exception:
            return 0.0


# ── TeacherMemory ──────────────────────────────────────────────────────────

class TeacherMemory:
    """Two-tier memory for the Meta-Teacher.

    Public surface (worker):
      - add_critique(entry, outer_summary)          → hot add + cold upsert
      - retrieve_similar(topic_key, outer_summary)  → list of up to top_k hits
      - still_needs_push_list(limit=10)              → for Maker INFO
      - snapshot()                                    → telemetry dict
      - archive_inactive()                           → cold-tier 90d sweep

    Threading: worker is single-threaded (one process per module). No locks.
    Persistence is best-effort — failures log at DEBUG and do not raise.
    """

    RECENCY_WINDOW_S = 7 * 86400.0            # recency_boost saturates past 7d
    QUALITY_BASELINE_N = 5                      # "first N avg" for quality_delta
    DEFAULT_HOT_SIZE = 1000
    DEFAULT_TOP_K = 3
    DEFAULT_SIM_THRESHOLD = 0.6
    DEFAULT_STILL_NEEDS_PUSH_N = 3
    DEFAULT_QUALITY_DELTA_EPSILON = 0.05
    DEFAULT_ARCHIVAL_DAYS = 90

    def __init__(self, config: Optional[dict] = None, data_dir: str = "./data"):
        cfg = config or {}
        self._enabled = bool(cfg.get("teaching_memory_enabled", False))
        self._hot_size = int(cfg.get("memory_buffer_hot_size", self.DEFAULT_HOT_SIZE))
        self._top_k = int(cfg.get("retrieval_top_k", self.DEFAULT_TOP_K))
        self._sim_threshold = float(cfg.get(
            "retrieval_similarity_threshold", self.DEFAULT_SIM_THRESHOLD))
        self._still_needs_n = int(cfg.get(
            "still_needs_push_critique_threshold",
            self.DEFAULT_STILL_NEEDS_PUSH_N))
        self._quality_delta_eps = float(cfg.get(
            "still_needs_push_quality_delta_epsilon",
            self.DEFAULT_QUALITY_DELTA_EPSILON))
        self._archival_days = int(cfg.get(
            "cold_tier_archival_days", self.DEFAULT_ARCHIVAL_DAYS))

        self._data_dir = os.path.join(data_dir, "meta_teacher")
        self._journal_path = os.path.join(self._data_dir, JOURNAL_FILENAME)
        self._archive_path = os.path.join(self._data_dir, ARCHIVE_FILENAME)

        # Hot tier — deque of {topic_key, embedding, critique, ts}
        self._hot: deque = deque(maxlen=self._hot_size)
        # Cold tier — in-memory dict keyed by topic_key; persisted to jsonl.
        self._cold: dict[str, dict] = {}

        self._embed_index = _EmbedIndex()

        # Retrieval / hit-rate telemetry
        self._retrievals: int = 0
        self._retrieval_hits: int = 0
        self._critiques_absorbed: int = 0

        self._loaded = False

    # ── Boot-time load ────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return self._enabled

    def load(self) -> None:
        """Load cold journal from disk. Idempotent; safe to call at boot.

        Hot tier is not persisted — it's rebuilt from live critiques after
        restart. This matches the rFP intent: hot = "recent conversational
        memory", cold = "durable learned topics".
        """
        if self._loaded:
            return
        self._loaded = True
        try:
            os.makedirs(self._data_dir, exist_ok=True)
        except Exception as e:
            logger.debug("[TeacherMemory] mkdir %s failed: %s", self._data_dir, e)
        if not os.path.exists(self._journal_path):
            logger.info("[TeacherMemory] No existing journal at %s (fresh start)",
                         self._journal_path)
            return
        loaded = 0
        corrupt = 0
        try:
            with open(self._journal_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        corrupt += 1
                        continue
                    tk = row.get("topic_key")
                    if not isinstance(tk, str):
                        corrupt += 1
                        continue
                    # Keep the LATEST row per topic_key (journal is append-only,
                    # so later = more recent).
                    self._cold[tk] = row
                    loaded += 1
        except Exception as e:
            logger.warning("[TeacherMemory] journal read failed: %s", e)
        logger.info("[TeacherMemory] Cold tier loaded: %d topic_keys (%d rows, "
                     "%d corrupt)", len(self._cold), loaded, corrupt)

    # ── add_critique ───────────────────────────────────────────────────
    def add_critique(
        self, entry: dict, outer_summary: Optional[dict] = None,
    ) -> Optional[str]:
        """Absorb a new critique into hot + cold tiers.

        `entry` is the same dict the worker appends to critiques.jsonl —
        has chain_id, ts, domain, quality_score, critique_text,
        suggested_primitives, adopted (if known). outer_summary is the
        Phase A distilled content (may be None for inner-state chains).

        Returns the topic_key used (for caller telemetry); None if disabled.
        """
        if not self._enabled:
            return None
        if not self._loaded:
            self.load()
        topic_key = canonical_topic_key(
            outer_summary,
            primitives_used=entry.get("suggested_primitives"),
            domain=entry.get("domain", ""),
        )
        ts = float(entry.get("ts") or time.time())

        # Embedding: compute if available; None → retrieval will skip.
        embed_text = self._embed_text(topic_key, outer_summary)
        emb = self._embed_index.embed(embed_text) if embed_text else None

        # Hot tier push
        hot_entry = {
            "topic_key": topic_key,
            "embedding": emb,
            "critique": entry,
            "ts": ts,
        }
        self._hot.append(hot_entry)

        # Cold tier upsert
        cold = self._cold.get(topic_key)
        if cold is None:
            cold = {
                "topic_key": topic_key,
                "first_seen": ts,
                "last_seen": ts,
                "critique_count": 0,
                "adoption_trajectory": [],
                "quality_trajectory": [],
                "quality_delta": 0.0,
                "still_needs_push": False,
                "last_voice_applied": entry.get("prompt_version"),
                "summary_cache": "",
            }
            self._cold[topic_key] = cold
        cold["last_seen"] = ts
        cold["critique_count"] = int(cold.get("critique_count", 0)) + 1
        q = float(entry.get("quality_score", 0.5))
        cold["quality_trajectory"].append({"ts": ts, "chain_quality": q})
        adopted = entry.get("adopted")
        if adopted is not None:
            cold["adoption_trajectory"].append({
                "ts": ts,
                "adopted_bool": bool(adopted),
                "suggested_list": list(entry.get("suggested_primitives") or []),
            })
        cold["last_voice_applied"] = entry.get("prompt_version", cold.get(
            "last_voice_applied"))
        # Recompute delta + still_needs_push
        self._refresh_cold_entry(cold)
        # Cache 2-sentence summary (very simple: dominant primitive + last
        # critique_text). Regenerate sparingly.
        critique_text = str(entry.get("critique_text") or "")[:160]
        cold["summary_cache"] = critique_text

        # Persist one journal line per update (append-only; load() keeps last).
        self._append_journal(cold)
        self._critiques_absorbed += 1
        return topic_key

    def record_adoption(
        self, topic_key: str, adopted: bool,
        suggested_primitives: Optional[list] = None, ts: Optional[float] = None,
    ) -> bool:
        """Retroactively record whether Titan adopted a prior suggestion.

        Called by the worker when a subsequent chain's primitives land; the
        worker remembers (topic_key, suggested_primitives) at critique time
        and feeds adopted=True/False here once the adoption window resolves.

        Returns True when a cold entry was updated, False when no matching
        topic_key existed (e.g. first critique on a new topic pushes the
        adoption signal out of scope).
        """
        if not self._enabled:
            return False
        if not self._loaded:
            self.load()
        cold = self._cold.get(topic_key)
        if cold is None:
            return False
        entry_ts = float(ts) if ts is not None else time.time()
        cold["adoption_trajectory"].append({
            "ts": entry_ts,
            "adopted_bool": bool(adopted),
            "suggested_list": list(suggested_primitives or []),
        })
        self._refresh_cold_entry(cold)
        self._append_journal(cold)
        return True

    def _refresh_cold_entry(self, cold: dict) -> None:
        """Update derived fields: quality_delta, still_needs_push."""
        q_traj = cold.get("quality_trajectory") or []
        count = len(q_traj)
        if count == 0:
            cold["quality_delta"] = 0.0
            cold["still_needs_push"] = False
            return
        baseline_n = min(self.QUALITY_BASELINE_N, count)
        baseline_avg = (
            sum(float(r.get("chain_quality", 0.5)) for r in q_traj[:baseline_n])
            / baseline_n)
        most_recent = float(q_traj[-1].get("chain_quality", 0.5))
        cold["quality_delta"] = round(most_recent - baseline_avg, 4)
        cold["still_needs_push"] = bool(
            cold["critique_count"] >= self._still_needs_n
            and cold["quality_delta"] <= self._quality_delta_eps)

    # ── retrieve_similar ───────────────────────────────────────────────
    def retrieve_similar(
        self, topic_key: str, outer_summary: Optional[dict] = None,
    ) -> list[dict]:
        """Hybrid similarity×importance retrieval over hot tier + cold match.

        Returns up to `retrieval_top_k` records. Each hit is a dict:
            {
              "topic_key": ...,
              "source": "hot" | "cold",
              "similarity": float,
              "importance_weight": float,
              "score": float,
              "critique": <hot-tier critique dict>  (hot hits only),
              "cold_entry": <cold-tier row>         (cold hits only),
            }

        Returns [] when teacher's memory is disabled, or when embedding is
        unavailable and no exact topic_key match exists.
        """
        if not self._enabled:
            return []
        if not self._loaded:
            self.load()
        self._retrievals += 1

        embed_text = self._embed_text(topic_key, outer_summary)
        query_vec = self._embed_index.embed(embed_text) if embed_text else None
        results: list[dict] = []

        # Hot-tier: cosine × importance_weight above threshold
        if query_vec is not None:
            for item in self._hot:
                if item.get("embedding") is None:
                    continue
                if item.get("topic_key") == topic_key:
                    # Exact hot hit — always count even when similarity tops out
                    pass
                sim = self._embed_index.cosine(query_vec, item["embedding"])
                if sim < self._sim_threshold:
                    continue
                tk = item["topic_key"]
                cold = self._cold.get(tk, {})
                iw = self._importance_weight(cold)
                results.append({
                    "topic_key": tk,
                    "source": "hot",
                    "similarity": round(sim, 4),
                    "importance_weight": round(iw, 3),
                    "score": round(sim * iw, 4),
                    "critique": item["critique"],
                })

        # Cold-tier: exact topic_key match (always considered, even without emb)
        cold_exact = self._cold.get(topic_key)
        if cold_exact is not None:
            iw = self._importance_weight(cold_exact)
            # Similarity = 1.0 (exact topic_key match)
            results.append({
                "topic_key": topic_key,
                "source": "cold",
                "similarity": 1.0,
                "importance_weight": round(iw, 3),
                "score": round(1.0 * iw, 4),
                "cold_entry": cold_exact,
            })

        if not results:
            return []
        # Dedup by (topic_key, source) — prefer highest score
        dedup: dict[tuple, dict] = {}
        for r in results:
            key = (r["topic_key"], r["source"])
            existing = dedup.get(key)
            if existing is None or r["score"] > existing["score"]:
                dedup[key] = r
        sorted_hits = sorted(dedup.values(), key=lambda r: -r["score"])
        top_hot = [r for r in sorted_hits if r["source"] == "hot"][: self._top_k]
        top_cold_exact = [r for r in sorted_hits if r["source"] == "cold"][:1]
        out = top_hot + top_cold_exact
        if out:
            self._retrieval_hits += 1
        return out

    def _importance_weight(self, cold: dict) -> float:
        """Importance formula per rFP §2 Phase B.

        Clamped to [0.5, 3.0]. Missing cold entry → 1.0 baseline (no
        adjustments).
        """
        if not cold:
            return 1.0
        now = time.time()
        last_seen = float(cold.get("last_seen") or now)
        age_s = max(0.0, now - last_seen)
        recency_boost = max(0.0, 1.0 - age_s / self.RECENCY_WINDOW_S)
        # adoption_rate: fraction of adoption_trajectory entries that were True
        adop = cold.get("adoption_trajectory") or []
        if adop:
            adopted_count = sum(1 for r in adop if r.get("adopted_bool"))
            adoption_rate = adopted_count / len(adop)
        else:
            adoption_rate = 0.5   # neutral when unknown
        qd = float(cold.get("quality_delta", 0.0))
        qd_sign = 1 if qd > self._quality_delta_eps else (
            -1 if qd < -self._quality_delta_eps else 0)
        snp = 1.0 if cold.get("still_needs_push") else 0.0
        weight = (
            1.0
            + 0.3 * recency_boost
            + 0.5 * adoption_rate
            + 0.3 * qd_sign
            + 0.4 * snp
        )
        return max(0.5, min(3.0, weight))

    # ── Maker INFO surfacing ──────────────────────────────────────────
    def still_needs_push_list(self, limit: int = 10) -> list[dict]:
        """Return top-N topic_keys flagged still_needs_push.

        Used by worker to emit INFO messages to Maker on 24h cadence.
        Ordered by critique_count desc (stuck topics Titan has hit most).
        """
        if not self._loaded:
            self.load()
        items = [c for c in self._cold.values() if c.get("still_needs_push")]
        items.sort(key=lambda c: -int(c.get("critique_count", 0)))
        out: list[dict] = []
        for c in items[:limit]:
            out.append({
                "topic_key": c.get("topic_key"),
                "critique_count": int(c.get("critique_count", 0)),
                "quality_delta": round(float(c.get("quality_delta", 0.0)), 4),
                "last_seen": float(c.get("last_seen") or 0.0),
                "summary_cache": c.get("summary_cache") or "",
            })
        return out

    def still_needs_push_hash(self, limit: int = 10) -> str:
        """Stable hash of the current still_needs_push list.

        Worker uses this to emit INFO only when the list *changes* across
        24h windows (per rFP §2 Phase B cadence). Hash over ordered
        (topic_key, critique_count) pairs — order already fixed by
        still_needs_push_list.
        """
        items = self.still_needs_push_list(limit=limit)
        joined = "|".join(
            f"{i['topic_key']}:{i['critique_count']}" for i in items)
        return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]

    # ── Archival ──────────────────────────────────────────────────────
    def archive_inactive(self, now: Optional[float] = None) -> int:
        """Move topic_keys inactive ≥ cold_tier_archival_days to the archive.

        Writes gzipped jsonl append. Removes archived rows from in-memory
        cold dict. Returns number archived. Safe to call periodically.
        """
        if not self._loaded:
            self.load()
        now = float(now) if now is not None else time.time()
        cutoff = now - (self._archival_days * 86400.0)
        archived = [c for tk, c in self._cold.items()
                    if float(c.get("last_seen") or now) < cutoff]
        if not archived:
            return 0
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            with gzip.open(self._archive_path, "at") as f:
                for c in archived:
                    f.write(json.dumps(c) + "\n")
        except Exception as e:
            logger.warning("[TeacherMemory] archive write failed: %s", e)
            return 0
        for c in archived:
            tk = c.get("topic_key")
            if tk in self._cold:
                self._cold.pop(tk, None)
        # Rebuild journal without archived entries (append-only semantics,
        # but on archival we prune + rewrite).
        self._rewrite_journal()
        logger.info("[TeacherMemory] Archived %d inactive topic_keys (≥%dd)",
                     len(archived), self._archival_days)
        return len(archived)

    # ── Snapshot / telemetry ──────────────────────────────────────────
    def snapshot(self) -> dict:
        """Status dict for /v4/meta-teacher/memory."""
        retr = max(1, self._retrievals)
        return {
            "enabled": self._enabled,
            "hot_tier_size": len(self._hot),
            "hot_tier_max": self._hot_size,
            "cold_tier_topics": len(self._cold),
            "critiques_absorbed": self._critiques_absorbed,
            "retrievals": self._retrievals,
            "retrieval_hits": self._retrieval_hits,
            "retrieval_hit_rate": round(self._retrieval_hits / retr, 4),
            "embedding_available": self._embed_index.available,
            "still_needs_push_count": sum(
                1 for c in self._cold.values() if c.get("still_needs_push")),
            "archival_days": self._archival_days,
            "similarity_threshold": self._sim_threshold,
            "top_k": self._top_k,
            "journal_path": self._journal_path,
        }

    # ── Persistence internals ─────────────────────────────────────────
    def _append_journal(self, cold_row: dict) -> None:
        """Append one row to the journal (single latest state per topic_key).

        Load() keeps only the LAST row per topic_key, so append-only is
        fine for durability. Compaction happens at archival time.
        """
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            with open(self._journal_path, "a") as f:
                f.write(json.dumps(cold_row) + "\n")
        except Exception as e:
            logger.debug("[TeacherMemory] journal append failed: %s", e)

    def _rewrite_journal(self) -> None:
        """Compact the journal: rewrite one row per current cold entry.

        Used after archival. Atomic via temp-file + rename.
        """
        tmp_path = self._journal_path + ".tmp"
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            with open(tmp_path, "w") as f:
                for row in self._cold.values():
                    f.write(json.dumps(row) + "\n")
            os.replace(tmp_path, self._journal_path)
        except Exception as e:
            logger.warning("[TeacherMemory] journal rewrite failed: %s", e)
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _embed_text(
        self, topic_key: str, outer_summary: Optional[dict],
    ) -> str:
        """Compose a small text payload for the embedding model.

        topic_key carries the canonical referent; adding top 1-2 felt
        summaries lifts retrieval quality for conceptually-similar topics.
        """
        parts = [str(topic_key or "")]
        if outer_summary:
            pp = outer_summary.get("primary_person")
            if pp:
                parts.append(str(pp))
            ct = outer_summary.get("current_topic")
            if ct:
                parts.append(str(ct))
            fs = outer_summary.get("felt_summaries") or []
            for s in fs[:2]:
                parts.append(str(s)[:40])
        return " | ".join(p for p in parts if p)
