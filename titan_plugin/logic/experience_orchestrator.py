"""
Experience Orchestrator — Close the Learning Loop

General-purpose experiential learning module with pluggable domain plugins.
Three-phase cycle: Record → Distill (during dreaming) → Bias (before action)

Domain-agnostic: works for puzzles, language, art, communication, anything.
Perception-keyed: experiences indexed by perception features, not task strings.
"""

import json
import hashlib
import logging
import math
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("titan.experience_orchestrator")


# ── Plugin Interface ───────────────────────────────────────────────

class ExperiencePlugin(ABC):
    """Each domain registers a plugin that knows how to extract
    perception features and interpret outcomes for that domain."""

    @property
    @abstractmethod
    def domain(self) -> str:
        """Unique domain identifier: 'arc_puzzle', 'language', etc."""

    @abstractmethod
    def extract_perception_key(self, context: dict) -> list[float]:
        """Extract domain-specific perception features from context.
        Returns a fixed-length float vector that indexes this experience."""

    @abstractmethod
    def compute_outcome_score(self, result: dict) -> float:
        """Domain-specific outcome scoring (0.0 to 1.0)."""

    @abstractmethod
    def summarize_for_distillation(self, experiences: list[dict]) -> dict:
        """Compress N raw experiences into a distilled insight.
        Returns: {'pattern': str, ...plugin-specific fields}"""

    def bias_weight(self) -> float:
        """How much this domain's experience should influence action selection."""
        return 1.0


# ── Experience Bias (return type for Phase 3) ──────────────────────

@dataclass
class ExperienceBias:
    """Result of experience recall — used to influence action selection."""

    action_scores: dict = field(default_factory=dict)
    optimal_inner_state: list = field(default_factory=list)
    inner_similarity: float = 0.0
    success_rate: float = 0.0
    confidence: float = 0.0
    relevant_experiences: int = 0
    insights: list = field(default_factory=list)

    def apply_to_threshold(self, base_threshold: float) -> float:
        """Modulate a composite threshold based on experience.
        High confidence + high success → lower threshold (fire more easily).
        High confidence + low success → raise threshold (fire less easily).
        """
        if self.confidence < 0.2 or self.relevant_experiences < 3:
            return base_threshold
        adjustment = (self.success_rate - 0.5) * self.confidence * 0.2
        return max(0.1, min(2.0, base_threshold - adjustment))


# ── Helpers ────────────────────────────────────────────────────────

def _cosine_sim(a: list, b: list) -> float:
    """Cosine similarity between two float vectors."""
    if not a or not b:
        return 0.0
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    mag_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (mag_a * mag_b)


def _state_hash(state: list) -> str:
    """MD5 hash of a state vector for fast lookup."""
    raw = json.dumps([round(v, 4) for v in state[:32]])
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ── Core Orchestrator ──────────────────────────────────────────────

class ExperienceOrchestrator:
    """
    Closes the learn-from-experience loop via a four-phase cycle:
      RECORD → DISTILL (dreaming) → GRAPH PROMOTE → BIAS (before action)
    """

    def __init__(
        self,
        ex_mem=None,
        e_mem=None,
        cognee_memory=None,
        db_path: str = "./data/experience_orchestrator.db",
    ):
        self._ex_mem = ex_mem
        self._e_mem = e_mem
        self._cognee = cognee_memory
        self._plugins: dict[str, ExperiencePlugin] = {}
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._record_count = 0
        self._on_wisdom_commit = None  # Callback: fn(domain, pattern, confidence, wisdom_id)

        logger.info("[ExperienceOrch] Initialized (db=%s)", db_path)

    def _init_schema(self):
        with self._lock:
            c = self._conn.cursor()
            c.executescript("""
                CREATE TABLE IF NOT EXISTS experience_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    perception_key TEXT NOT NULL,
                    inner_state_hash TEXT NOT NULL,
                    inner_state TEXT NOT NULL,
                    hormonal_snapshot TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    outcome_score REAL NOT NULL,
                    outcome_delta TEXT,
                    context TEXT,
                    epoch_id INTEGER,
                    distilled INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_exp_domain
                    ON experience_records(domain);
                CREATE INDEX IF NOT EXISTS idx_exp_distilled
                    ON experience_records(distilled);

                CREATE TABLE IF NOT EXISTS distilled_wisdom (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    perception_centroid TEXT NOT NULL,
                    optimal_conditions TEXT NOT NULL,
                    action_preferences TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    experience_count INTEGER NOT NULL,
                    dream_cycle INTEGER NOT NULL,
                    recall_count INTEGER NOT NULL DEFAULT 0,
                    last_recalled REAL,
                    promoted_to_graph INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_wisdom_domain
                    ON distilled_wisdom(domain);

                CREATE TABLE IF NOT EXISTS action_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    action TEXT NOT NULL,
                    total_attempts INTEGER NOT NULL DEFAULT 0,
                    total_successes INTEGER NOT NULL DEFAULT 0,
                    avg_score REAL NOT NULL DEFAULT 0.0,
                    last_updated REAL NOT NULL,
                    UNIQUE(domain, action)
                );
            """)
            self._conn.commit()

            # Migration: add pending_distillation column (waking=1, dreaming=0)
            try:
                self._conn.execute(
                    "ALTER TABLE experience_records "
                    "ADD COLUMN pending_distillation INTEGER NOT NULL DEFAULT 1"
                )
                self._conn.commit()
                logger.info("[ExperienceOrch] Added pending_distillation column")
            except Exception:
                pass  # Column already exists

            # Index on pending_distillation (must be after migration adds column)
            try:
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_exp_pending "
                    "ON experience_records(pending_distillation)"
                )
                self._conn.commit()
            except Exception:
                pass

    # ── Plugin Management ──────────────────────────────────────────

    def register_plugin(self, plugin: ExperiencePlugin) -> None:
        # Validate: test perception key extraction with dummy context
        try:
            _test_ctx = {
                "inner_state": [0.5] * 132,
                "felt_tensor": [0.5] * 65,
                "inner_body": [0.5] * 5,
                "inner_mind": [0.5] * 15,
                "inner_spirit": [0.5] * 45,
                "hormonal_snapshot": {},
                "intent_hormones": {},
                "spatial_features": [0.5] * 30,
            }
            _test_key = plugin.extract_perception_key(_test_ctx)
            _dim = len(_test_key) if _test_key else 0
            if _dim < 10:
                logger.warning(
                    "[ExperienceOrch] Plugin '%s' perception key is only %dD — "
                    "minimum recommended 10D for meaningful matching",
                    plugin.domain, _dim)
        except Exception as _val_err:
            _dim = -1
            logger.warning(
                "[ExperienceOrch] Plugin '%s' validation failed: %s",
                plugin.domain, _val_err)
        self._plugins[plugin.domain] = plugin
        logger.info("[ExperienceOrch] Registered plugin: %s (%dD perception)",
                    plugin.domain, _dim)

    # ── Phase 1: RECORD ────────────────────────────────────────────

    def record_outcome(
        self,
        domain: str,
        perception_features: list,
        inner_state_132d: list,
        hormonal_snapshot: dict,
        action_taken: str,
        outcome_score: float,
        outcome_delta: dict = None,
        context: dict = None,
        epoch_id: int = 0,
        is_dreaming: bool = False,
    ) -> int:
        """Record an action outcome with full context.

        is_dreaming: If True, record is tagged pending_distillation=0
            (will be promoted to pending=1 after dream ends, consolidated in next dream).
        """
        # During waking: pending_distillation=1 (ready for next dream)
        # During dreaming: pending_distillation=0 (wait for next cycle)
        pending = 0 if is_dreaming else 1

        with self._lock:
            row_id = self._conn.execute(
                "INSERT INTO experience_records "
                "(domain, perception_key, inner_state_hash, inner_state, "
                " hormonal_snapshot, action_taken, outcome_score, outcome_delta, "
                " context, epoch_id, pending_distillation, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    domain,
                    json.dumps(perception_features),
                    _state_hash(inner_state_132d),
                    json.dumps(inner_state_132d[:132]),
                    json.dumps(hormonal_snapshot),
                    action_taken,
                    outcome_score,
                    json.dumps(outcome_delta) if outcome_delta else "{}",
                    json.dumps(context) if context else "{}",
                    epoch_id,
                    pending,
                    time.time(),
                ),
            ).lastrowid
            self._conn.commit()
        self._record_count += 1

        if self._record_count % 50 == 0:
            logger.info("[ExperienceOrch] %d records stored (%s domain)",
                        self._record_count, domain)
        return row_id

    # ── Phase 2: DISTILL ───────────────────────────────────────────

    def distill_cycle(
        self, dream_cycle: int, current_epoch_id: int
    ) -> list[dict]:
        """Process undistilled PRE-DREAM records into wisdom during dreaming.

        Only processes records with pending_distillation=1 (tagged during waking).
        Records created during THIS dream (pending_distillation=0) are skipped —
        they'll be consolidated in the next dream cycle.
        """
        rows = self._conn.execute(
            "SELECT * FROM experience_records WHERE distilled = 0 "
            "AND pending_distillation = 1 "
            "ORDER BY domain, created_at"
        ).fetchall()

        if not rows:
            return []

        by_domain: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_domain[r["domain"]].append(dict(r))

        insights = []
        for domain, records in by_domain.items():
            plugin = self._plugins.get(domain)

            # --- Compute perception centroid ---
            p_vectors = [json.loads(r["perception_key"]) for r in records]
            if p_vectors and p_vectors[0]:
                centroid = [sum(c) / len(c) for c in zip(*p_vectors)]
            else:
                centroid = []

            # --- Compute action preferences ---
            action_scores: dict[str, list[float]] = defaultdict(list)
            for r in records:
                action_scores[r["action_taken"]].append(r["outcome_score"])
            action_prefs = {
                a: sum(s) / len(s) for a, s in action_scores.items()
            }

            # --- Compute optimal inner state (centroid of successes) ---
            successful = [r for r in records if r["outcome_score"] > 0.6]
            optimal = []
            if successful:
                inner_vecs = [json.loads(r["inner_state"]) for r in successful]
                min_len = min(len(v) for v in inner_vecs) if inner_vecs else 0
                if min_len > 0:
                    optimal = [
                        sum(v[i] for v in inner_vecs) / len(inner_vecs)
                        for i in range(min_len)
                    ]

            # --- Confidence = f(N, variance) ---
            scores = [r["outcome_score"] for r in records]
            n = len(scores)
            mean_score = sum(scores) / n
            variance = (
                sum((s - mean_score) ** 2 for s in scores) / max(1, n - 1)
                if n > 1 else 1.0
            )
            confidence = min(1.0, n / 20.0) * max(0.0, 1.0 - variance)

            # --- Plugin summary ---
            pattern = f"{domain}_pattern_cycle{dream_cycle}"
            if plugin:
                try:
                    summary = plugin.summarize_for_distillation(records)
                    pattern = summary.get("pattern", pattern)
                except Exception as e:
                    logger.debug("[ExperienceOrch] Plugin summarize error: %s", e)

            # --- Store wisdom ---
            with self._lock:
                wisdom_id = self._conn.execute(
                    "INSERT INTO distilled_wisdom "
                    "(domain, pattern, perception_centroid, optimal_conditions, "
                    " action_preferences, confidence, experience_count, "
                    " dream_cycle, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        domain, pattern,
                        json.dumps(centroid), json.dumps(optimal),
                        json.dumps(action_prefs), confidence, n,
                        dream_cycle, time.time(),
                    ),
                ).lastrowid

            # --- Outer Trinity analysis (Tier 2: separate inner/outer centroids) ---
            outer_analysis = {}
            if len(optimal) >= 70:
                outer_analysis["outer_body"] = [round(v, 4) for v in optimal[65:70]]
            if len(optimal) >= 85:
                outer_analysis["outer_mind"] = [round(v, 4) for v in optimal[70:85]]

            insights.append({
                "domain": domain,
                "pattern": pattern,
                "confidence": confidence,
                "experience_count": n,
                "wisdom_id": wisdom_id,
                "outer_analysis": outer_analysis,
            })

            # TimeChain callback for distilled wisdom
            if self._on_wisdom_commit:
                try:
                    self._on_wisdom_commit(domain, pattern, confidence, wisdom_id)
                except Exception:
                    pass

            # --- Update running action_stats ---
            for action, s_list in action_scores.items():
                self._update_action_stats(domain, action, s_list)

            # --- Bridge to e_mem (inner + outer analysis) ---
            if confidence > 0.5 and self._e_mem:
                try:
                    inner_tensor = optimal[:65] if len(optimal) >= 65 else optimal
                    outer_tensor = optimal[65:132] if len(optimal) >= 132 else []
                    self._e_mem.store_insight(
                        {
                            "significance": confidence,
                            "felt_tensor": inner_tensor,
                            "outer_tensor": outer_tensor,
                            "epoch_id": current_epoch_id,
                            "hormones": {},
                        },
                        dream_cycle=dream_cycle,
                    )
                except Exception as e:
                    logger.debug("[ExperienceOrch] e_mem bridge error: %s", e)

            logger.info(
                "[ExperienceOrch] Distilled %s: %d records → confidence=%.3f, "
                "pattern=%s, actions=%s",
                domain, n, confidence, pattern, list(action_prefs.keys()),
            )

        # Mark all as distilled
        record_ids = [r["id"] for r in rows]
        if record_ids:
            placeholders = ",".join("?" * len(record_ids))
            with self._lock:
                self._conn.execute(
                    f"UPDATE experience_records SET distilled = 1 "
                    f"WHERE id IN ({placeholders})",
                    record_ids,
                )
                self._conn.commit()

        return insights

    def _update_action_stats(self, domain: str, action: str, scores: list):
        n_new = len(scores)
        s_new = sum(1 for s in scores if s > 0.5)
        avg_new = sum(scores) / n_new if n_new else 0.0

        with self._lock:
            existing = self._conn.execute(
                "SELECT * FROM action_stats WHERE domain = ? AND action = ?",
                (domain, action),
            ).fetchone()

            if existing:
                total_n = existing["total_attempts"] + n_new
                total_s = existing["total_successes"] + s_new
                # Running average
                total_avg = (
                    (existing["avg_score"] * existing["total_attempts"] + avg_new * n_new)
                    / total_n
                )
                self._conn.execute(
                    "UPDATE action_stats SET total_attempts=?, total_successes=?, "
                    "avg_score=?, last_updated=? WHERE domain=? AND action=?",
                    (total_n, total_s, total_avg, time.time(), domain, action),
                )
            else:
                self._conn.execute(
                    "INSERT INTO action_stats (domain, action, total_attempts, "
                    "total_successes, avg_score, last_updated) VALUES (?,?,?,?,?,?)",
                    (domain, action, n_new, s_new, avg_new, time.time()),
                )
            self._conn.commit()

    # ── Phase 2b: GRAPH PROMOTE ────────────────────────────────────

    def promote_to_graph(self) -> int:
        """Promote high-confidence wisdom to Cognee graph nodes."""
        if not self._cognee:
            return 0

        unpromoted = self._conn.execute(
            "SELECT * FROM distilled_wisdom "
            "WHERE confidence > 0.6 AND promoted_to_graph = 0 "
            "ORDER BY confidence DESC LIMIT 10"
        ).fetchall()

        promoted = 0
        for w in unpromoted:
            try:
                text = (
                    f"Experience wisdom [{w['domain']}]: {w['pattern']}. "
                    f"Action preferences: {w['action_preferences']}. "
                    f"Confidence: {w['confidence']:.2f} from {w['experience_count']} experiences."
                )
                self._cognee.add_to_mempool(
                    user_prompt=f"experience_wisdom_{w['domain']}",
                    agent_response=text,
                    user_id="experience_orchestrator",
                )
                with self._lock:
                    self._conn.execute(
                        "UPDATE distilled_wisdom SET promoted_to_graph = 1 WHERE id = ?",
                        (w["id"],),
                    )
                promoted += 1
            except Exception as e:
                logger.debug("[ExperienceOrch] Graph promote error: %s", e)

        if promoted:
            with self._lock:
                self._conn.commit()
            logger.info("[ExperienceOrch] Promoted %d wisdom entries to Cognee graph", promoted)

        return promoted

    # ── Phase 3: BIAS ──────────────────────────────────────────────

    def get_experience_bias(
        self,
        domain: str,
        current_perception: list,
        current_inner_state: list,
        candidate_actions: list,
        top_k: int = 5,
    ) -> ExperienceBias:
        """Recall relevant experience and compute bias for action selection."""
        # --- Wisdom recall by perception similarity ---
        wisdom_rows = self._conn.execute(
            "SELECT * FROM distilled_wisdom WHERE domain = ? "
            "ORDER BY confidence DESC LIMIT 20",
            (domain,),
        ).fetchall()

        ranked = []
        for w in wisdom_rows:
            centroid = json.loads(w["perception_centroid"])
            sim = _cosine_sim(current_perception, centroid)
            if sim > 0.3:
                ranked.append((sim, dict(w)))
        ranked.sort(key=lambda x: x[0], reverse=True)
        top_wisdom = ranked[:top_k]

        # Update recall counts
        if top_wisdom:
            with self._lock:
                for _, w in top_wisdom:
                    self._conn.execute(
                        "UPDATE distilled_wisdom SET recall_count = recall_count + 1, "
                        "last_recalled = ? WHERE id = ?",
                        (time.time(), w["id"]),
                    )
                self._conn.commit()

        # --- Action scoring ---
        action_scores = {}
        for action in candidate_actions:
            stat = self._conn.execute(
                "SELECT * FROM action_stats WHERE domain = ? AND action = ?",
                (domain, action),
            ).fetchone()
            base = (stat["avg_score"] - 0.5) if stat else 0.0

            wisdom_boost = 0.0
            for sim, w in top_wisdom:
                prefs = json.loads(w["action_preferences"])
                if action in prefs:
                    wisdom_boost += sim * w["confidence"] * (prefs[action] - 0.5)

            action_scores[action] = max(-1.0, min(1.0, base + wisdom_boost))

        # --- Optimal inner state ---
        optimal = []
        if top_wisdom:
            total_w = sum(s * w["confidence"] for s, w in top_wisdom) or 1e-9
            for sim, w in top_wisdom:
                conds = json.loads(w["optimal_conditions"])
                weight = sim * w["confidence"] / total_w
                if not optimal:
                    optimal = [v * weight for v in conds]
                else:
                    ml = min(len(optimal), len(conds))
                    optimal = [optimal[i] + conds[i] * weight for i in range(ml)]

        inner_sim = _cosine_sim(current_inner_state, optimal) if optimal else 0.0

        # --- Success rate ---
        total_a = 0
        total_s = 0
        for action in candidate_actions:
            stat = self._conn.execute(
                "SELECT total_attempts, total_successes FROM action_stats "
                "WHERE domain = ? AND action = ?",
                (domain, action),
            ).fetchone()
            if stat:
                total_a += stat["total_attempts"]
                total_s += stat["total_successes"]

        success_rate = total_s / max(1, total_a)
        overall_conf = max((w["confidence"] for _, w in top_wisdom), default=0.0)

        return ExperienceBias(
            action_scores=action_scores,
            optimal_inner_state=optimal,
            inner_similarity=inner_sim,
            success_rate=success_rate,
            confidence=overall_conf,
            relevant_experiences=sum(w["experience_count"] for _, w in top_wisdom),
            insights=[w.get("pattern", "") for _, w in top_wisdom],
        )

    # ── Dream Tagging ─────────────────────────────────────────────

    def retag_dream_experiences(self) -> int:
        """After dream ends: promote all untagged (during-dream) experiences.

        Sets pending_distillation=1 for all records where pending_distillation=0.
        These will be consolidated in the NEXT dream cycle.
        Returns count of re-tagged records.
        """
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE experience_records SET pending_distillation = 1 "
                "WHERE pending_distillation = 0 AND distilled = 0"
            )
            count = cursor.rowcount
            self._conn.commit()
        if count > 0:
            logger.info("[ExperienceOrch] Re-tagged %d dream experiences for next cycle", count)
        return count

    def get_pre_dream_undistilled(self) -> int:
        """Count undistilled records tagged for distillation (pre-dream only)."""
        return self._conn.execute(
            "SELECT COUNT(*) as c FROM experience_records "
            "WHERE distilled = 0 AND pending_distillation = 1"
        ).fetchone()["c"]

    # ── Stats ──────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        total_records = self._conn.execute(
            "SELECT COUNT(*) as c FROM experience_records"
        ).fetchone()["c"]
        undistilled = self._conn.execute(
            "SELECT COUNT(*) as c FROM experience_records WHERE distilled = 0"
        ).fetchone()["c"]
        total_wisdom = self._conn.execute(
            "SELECT COUNT(*) as c FROM distilled_wisdom"
        ).fetchone()["c"]
        total_stats = self._conn.execute(
            "SELECT COUNT(*) as c FROM action_stats"
        ).fetchone()["c"]

        pre_dream_undistilled = self._conn.execute(
            "SELECT COUNT(*) as c FROM experience_records "
            "WHERE distilled = 0 AND pending_distillation = 1"
        ).fetchone()["c"]

        return {
            "total_records": total_records,
            "undistilled": undistilled,
            "pre_dream_undistilled": pre_dream_undistilled,
            "total_wisdom": total_wisdom,
            "total_action_stats": total_stats,
            "plugins": list(self._plugins.keys()),
        }


# ── Domain Inference Helper ────────────────────────────────────────

DOMAIN_MAP = {
    "art_generate": "creative",
    "audio_generate": "creative",
    "self_express": "language",
    "social_post": "communication",
    "kin_sense": "communication",
    "longing_reach": "communication",
    "web_search": "research",
    "arc_play": "arc_puzzle",
}


def infer_domain(task_type: str) -> str:
    """Map a task_type/helper name to an experience domain."""
    return DOMAIN_MAP.get(task_type, "general")
