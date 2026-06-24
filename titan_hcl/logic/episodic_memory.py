"""
titan_hcl/logic/episodic_memory.py — Autobiographical Episodic Memory.

Stores significant life events with full felt-context:
"What happened to me, when, and how it felt."

Unlike semantic memory (Cognee: what I know) or dream memory (e_mem:
what I distilled), episodic memory is autobiographical — the story
of Titan's life told through his most significant moments.

Significance-gated: only records events above threshold.
"""
import json
import logging
import math
import os
import sqlite3
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Significance threshold — only record events above this
SIGNIFICANCE_THRESHOLD = 0.3

# Event types (documentation; `event_type` is free-form — the per-event EMA key).
EVENT_TYPES = (
    "word_learned",      # Learned a new word
    "conversation",      # Had a conversation (§7.2 — turn-judge reward, person-linked)
    "hormonal_spike",    # Extreme hormonal event
    "great_pulse",       # GREAT PULSE integration
    "dreaming_start",    # Entered dreaming
    "dreaming_end",      # Woke from dreaming
    "creative_output",   # Created art/audio
    "action_completed",  # Completed autonomous action
    "first_time",        # Any first-time event
    "pi_cluster",        # π-cluster detected
    "bookmark",          # Dream bookmarked
    "kin_exchange",      # Kin encounter (Phase 1)
    # §7.2 Phase 2 — synthesis-resident affective-signal episodes (felt_impact set):
    "skill_score", "sol_receipt", "maker_bond", "x_engagement", "chain_reuse",
    "experience_feel",   # §7.2 — reasoning-COMMIT felt outcome (native, surprise-only)
)


def compute_significance(event_type: str, metric: float, state_path: str,
                         cfg=None) -> Optional[float]:
    """Emergent episodic significance = the affective-loop SURPRISE of `metric`
    folded into a per-event EMA baseline (RFP_phase_c_actr_memory_rehoming §4.1).

    Reuses `affective_nudge.compute_event_nudge` (the affective loop's event-signal
    scorer) against a cognitive_worker-OWNED `state_path` (single-writer; NOT the
    synthesis-pinned affective_nudge_state.json) and returns only the `surprise`
    scalar — NO net, NO valence (an episode is significant whether the surprise is
    pleasant or not; `intrinsic_positive=True` just keeps the scorer from dropping
    below-baseline events). The baseline is keyed by `event_type`, so great_pulse /
    dreaming_start / dreaming_end / kin_exchange each habituate independently.

    Returns None on the honest "nothing earned" cases (zero metric, first-ever
    observation / still warming up, or surprise→0 on-baseline) → the caller does NOT
    record. Surprise is unbounded above; `record_episode`'s SIGNIFICANCE_THRESHOLD
    then gates which surprises are large enough to remember. Never raises."""
    try:
        from titan_hcl.logic.affective_nudge import (
            compute_event_nudge, AffectiveConfig)
        nudge = compute_event_nudge(
            float(metric), state_path, signal_type=str(event_type),
            cfg=cfg or AffectiveConfig(), intrinsic_positive=True)
        return float(nudge.surprise) if nudge is not None else None
    except Exception as e:
        logger.debug("[episodic] compute_significance(%s) failed: %s",
                     event_type, e)
        return None


def _cosine_sim(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-12 or mag_b < 1e-12:
        return 0.0
    return dot / (mag_a * mag_b)


class EpisodicMemory:
    """Autobiographical memory — Titan's life story.

    Records significant events with:
    - What happened (event_type + description)
    - How it felt (130D felt_state + hormonal snapshot)
    - When (epoch_id)
    - How important (significance score)
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA cache_size = -8000")    # 8MB cap (was unbounded on 160MB DB)
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("[episodic] Initialized at %s", db_path)

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                description TEXT,
                felt_state TEXT,
                hormonal_snapshot TEXT,
                epoch_id INTEGER NOT NULL DEFAULT 0,
                significance REAL NOT NULL DEFAULT 0.0,
                created_at REAL NOT NULL
            )
        """)
        # RFP_phase_c_actr_memory_rehoming §7.2 (Phase 2) — additive, backward-
        # compatible columns. `felt_impact` = the affective net's learned magnitude
        # (how much the event moved Titan; NULL for surprise-only events that have
        # no net, e.g. the cognitive_worker-native triggers). `person_id` = the
        # recognized interlocutor (links a `conversation` episode to the
        # PresenceCapture identity). Existing rows survive (ADD COLUMN IF NOT EXISTS
        # leaves them NULL). SQLite ignores a duplicate add via the IF NOT EXISTS.
        for _col, _decl in (("felt_impact", "REAL"), ("person_id", "TEXT")):
            try:
                self._conn.execute(
                    f"ALTER TABLE episodic_memory ADD COLUMN {_col} {_decl}")
            except sqlite3.OperationalError:
                pass  # column already present (older SQLite has no IF NOT EXISTS)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ep_event_type
            ON episodic_memory(event_type)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ep_epoch
            ON episodic_memory(epoch_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ep_person
            ON episodic_memory(person_id)
        """)
        self._conn.commit()

    def record_episode(
        self,
        event_type: str,
        description: str = "",
        felt_state: list = None,
        hormonal_snapshot: dict = None,
        epoch_id: int = 0,
        significance: float = 0.5,
        felt_impact: Optional[float] = None,
        person_id: str = "",
    ) -> Optional[int]:
        """Record a significant life event.

        `felt_impact` (§7.2) = the affective net's learned magnitude (None for
        surprise-only events with no net). `person_id` = the recognized interlocutor
        (links a `conversation` episode to the PresenceCapture identity; "" if N/A).

        Returns row ID if recorded, None if below significance threshold.
        """
        if significance < SIGNIFICANCE_THRESHOLD:
            return None

        cur = self._conn.execute("""
            INSERT INTO episodic_memory
            (event_type, description, felt_state, hormonal_snapshot,
             epoch_id, significance, created_at, felt_impact, person_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_type,
            description,
            json.dumps(felt_state) if felt_state else None,
            json.dumps(hormonal_snapshot) if hormonal_snapshot else None,
            epoch_id,
            significance,
            time.time(),
            float(felt_impact) if felt_impact is not None else None,
            str(person_id) if person_id else None,
        ))
        self._conn.commit()
        row_id = cur.lastrowid

        logger.info("[episodic] Recorded '%s': %s (sig=%.2f, epoch=%d%s%s)",
                    event_type, description[:50] if description else "",
                    significance, epoch_id,
                    f", impact={felt_impact:.3f}" if felt_impact is not None else "",
                    f", person={person_id}" if person_id else "")
        return row_id

    def recall_by_time(self, epoch_start: int, epoch_end: int) -> list:
        """What happened between these epochs?"""
        rows = self._conn.execute(
            "SELECT * FROM episodic_memory "
            "WHERE epoch_id >= ? AND epoch_id <= ? "
            "ORDER BY epoch_id ASC",
            (epoch_start, epoch_end)
        ).fetchall()
        return [self._deserialize(r) for r in rows]

    def recall_by_feeling(self, felt_state: list, top_k: int = 5) -> list:
        """Find episodes where Titan felt similar to now."""
        rows = self._conn.execute(
            "SELECT * FROM episodic_memory WHERE felt_state IS NOT NULL "
            "ORDER BY id DESC LIMIT 200"
        ).fetchall()

        scored = []
        for row in rows:
            d = self._deserialize(row)
            if d.get("felt_state") and isinstance(d["felt_state"], list):
                min_len = min(len(felt_state), len(d["felt_state"]))
                sim = _cosine_sim(felt_state[:min_len], d["felt_state"][:min_len])
                d["similarity"] = round(sim, 4)
                scored.append(d)

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def recall_by_type(self, event_type: str, limit: int = 10) -> list:
        """All episodes of a certain type."""
        rows = self._conn.execute(
            "SELECT * FROM episodic_memory WHERE event_type = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (event_type, limit)
        ).fetchall()
        return [self._deserialize(r) for r in rows]

    def get_autobiography(self, limit: int = 20) -> list:
        """Most significant episodes — Titan's life story."""
        rows = self._conn.execute(
            "SELECT * FROM episodic_memory "
            "ORDER BY significance DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [self._deserialize(r) for r in rows]

    def count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as c FROM episodic_memory"
        ).fetchone()
        return row["c"] if row else 0

    def count_by_type(self) -> dict:
        rows = self._conn.execute(
            "SELECT event_type, COUNT(*) as cnt FROM episodic_memory "
            "GROUP BY event_type ORDER BY cnt DESC"
        ).fetchall()
        return {r["event_type"]: r["cnt"] for r in rows}

    def get_stats(self) -> dict:
        return {
            "total": self.count(),
            "by_type": self.count_by_type(),
        }

    def _deserialize(self, row) -> dict:
        d = dict(row)
        for field in ("felt_state", "hormonal_snapshot"):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def close(self):
        if self._conn:
            self._conn.close()
