"""
titan_plugin/logic/self_reasoning.py — Introspective Cognition for Titan.

Self-reasoning is the introspective complement to SPIRIT_SELF. SPIRIT_SELF
*modifies* inner state (thermostat). Self-reasoning *observes* inner state
(thermometer). Together they form reflexive self-awareness.

Architecture:
  - INTROSPECT meta-primitive with 5 sub-modes
  - Self-HAOV: testable predictions about own state
  - Neuromod-coupled mode selection
  - CGN "self_model" consumer for grounding self-concepts
  - Self-profile builder (replaces spirit_loop._build_self_profile)
  - SQLite persistence in inner_memory.db (self_insights + self_predictions)

See: titan-docs/rFP_self_reasoning.md
"""

import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger("titan.self_reasoning")

# ── Constants ────────────────────────────────────────────────────────

INTROSPECT_SUB_MODES = [
    "state_audit",         # snapshot neuromods + MSL + chi + reasoning stats
    "prediction",          # predict next-state based on trajectory
    "coherence_check",     # compare inner state vs stored self-profile
    "vocabulary_probe",    # "is word X grounded or narrator-borrowed?"
    "architecture_query",  # describe own architecture from live data
]

# Neuromod-coupled introspection mode map
# (condition_name, threshold_fn) → preferred sub-mode
NEUROMOD_MODE_MAP = [
    # High GABA → inhibited, minimal introspection
    ("high_gaba",       lambda nm: nm.get("GABA", 0.2) > 0.5,
     "state_audit"),
    # High NE → defensive self-check
    ("high_ne",         lambda nm: nm.get("NE", 0.5) > 0.7,
     "architecture_query"),
    # High ACh + DA → exploratory discovery
    ("exploratory",     lambda nm: (nm.get("ACh", 0.5) > 0.6
                                    and nm.get("DA", 0.5) > 0.6),
     "prediction"),
    # High 5HT + Low NE → contemplative reflection
    ("contemplative",   lambda nm: (_sht(nm) > 0.7
                                    and nm.get("NE", 0.5) < 0.4),
     "coherence_check"),
    # Low DA → identity reinforcement
    ("low_da",          lambda nm: nm.get("DA", 0.5) < 0.3,
     "coherence_check"),
]


def _sht(nm: dict) -> float:
    """Extract serotonin from neuromods dict (handles both key names)."""
    v = nm.get("5HT", nm.get("5-HT", nm.get("Serotonin", 0.5)))
    if isinstance(v, dict):
        return float(v.get("level", 0.5))
    return float(v)


def _nm_float(nm: dict, key: str, alt: str = None, default: float = 0.5) -> float:
    """Extract float from neuromods dict (handles dict-valued entries)."""
    v = nm.get(key)
    if v is None and alt:
        v = nm.get(alt, default)
    if v is None:
        return default
    if isinstance(v, dict):
        return float(v.get("level", default))
    return float(v)


# ── Data Classes ─────────────────────────────────────────────────────

@dataclass
class SelfInsight:
    """Result of an introspection sub-mode."""
    sub_mode: str
    epoch: int = 0
    timestamp: float = 0.0
    # Structured data from the introspection
    data: Dict[str, Any] = field(default_factory=dict)
    # Confidence in the insight (how complete/reliable the data was)
    confidence: float = 0.5
    # Neuromod state at time of introspection
    neuromod_snapshot: Dict[str, float] = field(default_factory=dict)
    # Which introspection mode was active
    mode_trigger: str = "default"


@dataclass
class SelfPrediction:
    """A testable prediction about own future state."""
    prediction_id: int = 0
    created_epoch: int = 0
    created_at: float = 0.0
    # What is being predicted
    target_metric: str = ""  # e.g., "5HT", "vocab_count", "i_confidence"
    predicted_value: float = 0.0
    predicted_direction: str = ""  # "up", "down", "stable"
    # When to check
    check_after_epochs: int = 500
    check_epoch: int = 0  # created_epoch + check_after_epochs
    # Verification
    verified: bool = False
    actual_value: float = 0.0
    prediction_error: float = 0.0
    confirmed: bool = False  # prediction direction was correct
    verified_at: float = 0.0


@dataclass
class SelfProfile:
    """Comprehensive self-knowledge snapshot."""
    epoch: int = 0
    timestamp: float = 0.0
    # Vocabulary
    vocab_total: int = 0
    vocab_productive: int = 0
    top_words: List[str] = field(default_factory=list)
    composition_level: str = ""
    # Neuromods
    neuromod_levels: Dict[str, float] = field(default_factory=dict)
    dominant_emotion: str = ""
    # MSL
    i_confidence: float = 0.0
    i_depth: float = 0.0
    i_depth_components: Dict[str, float] = field(default_factory=dict)
    chi_coherence: float = 0.0
    convergence_count: int = 0
    concept_confidences: Dict[str, float] = field(default_factory=dict)
    # Reasoning
    total_chains: int = 0
    dominant_primitive: str = ""
    eureka_count: int = 0
    wisdom_count: int = 0
    commit_rate: float = 0.0
    # Architecture
    total_epochs: int = 0
    dream_cycles: int = 0
    ns_train_steps: int = 0
    # Self-reasoning stats
    introspection_count: int = 0
    prediction_accuracy: float = 0.0
    active_predictions: int = 0


# ── Self-Reasoning Engine ────────────────────────────────────────────

# PERSISTENCE_BY_DESIGN: SelfReasoningEngine._active_predictions is the
# in-flight prediction queue — loaded from the self-HAOV store on boot but
# maintained via dict mutation rather than self-assignment the scanner sees.
class SelfReasoningEngine:
    """Introspective cognition — Titan's thermometer for self-knowledge.

    Queries live inner systems to build structured self-insights.
    Stores predictions and tracks their accuracy (Self-HAOV).
    Neuromod-coupled: different emotional states trigger different
    introspection modes.

    Used by: INTROSPECT meta-primitive, spirit_worker dream cycle.
    """

    def __init__(self, config: dict = None,
                 db_path: str = "data/inner_memory.db"):
        self._config = config or {}
        self._db_path = db_path

        # Config
        self._cooldown_max = self._config.get("cooldown_epochs", 10)
        self._max_active_predictions = self._config.get(
            "max_active_predictions", 20)
        self._prediction_horizon_min = self._config.get(
            "prediction_horizon_min", 200)
        self._prediction_horizon_max = self._config.get(
            "prediction_horizon_max", 5000)
        self._coherence_gap_threshold = self._config.get(
            "coherence_gap_threshold", 0.15)
        # Layer A — novelty gap thresholds (rFP_coding_explorer_activation.md §4.1)
        self._commit_rate_low_threshold = float(
            self._config.get("commit_rate_low_threshold", 0.15))
        self._commit_rate_low_min_chains = int(
            self._config.get("commit_rate_low_min_chains", 500))
        self._eureka_staleness_epochs = int(
            self._config.get("eureka_staleness_epochs", 50000))
        self._vocab_plateau_seconds = float(
            self._config.get("vocab_plateau_seconds", 43200))
        self._monoculture_pressure_frac = float(
            self._config.get("monoculture_pressure_frac", 0.65))

        # State
        self._cooldown = 0
        self._total_introspections = 0
        self._total_predictions = 0
        self._prediction_accuracy_ema = 0.5  # EMA of prediction accuracy
        self._prediction_accuracy_alpha = 0.1

        # Active predictions (in-memory, persisted to DB)
        self._active_predictions: List[SelfPrediction] = []
        self._verified_predictions: List[SelfPrediction] = []

        # Last self-profile
        self._last_profile: Optional[SelfProfile] = None
        self._last_profile_epoch = 0

        # Coherence gaps detected
        self._coherence_gaps: List[dict] = []

        # Layer A — novelty gap detector state
        self._last_seen_eureka_count = 0
        self._last_eureka_epoch: Optional[int] = None  # None = never anchored
        self._vocab_plateau_anchor_count = 0
        self._vocab_plateau_anchor_ts = 0.0
        self._seen_novelty_gaps: set = set()  # WARN-on-first-fire telemetry

        # Initialize DB
        self._init_db()
        self._load_active_predictions()

        logger.info("[SelfReasoning] Initialized (cooldown=%d, max_pred=%d, "
                    "active_pred=%d)",
                    self._cooldown_max, self._max_active_predictions,
                    len(self._active_predictions))

    # ── DB Initialization ────────────────────────────────────────────

    def _init_db(self):
        """Create self_insights and self_predictions tables if not exist."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS self_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sub_mode TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    data TEXT,
                    confidence REAL DEFAULT 0.5,
                    neuromod_snapshot TEXT,
                    mode_trigger TEXT DEFAULT 'default'
                );

                CREATE TABLE IF NOT EXISTS self_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_epoch INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    target_metric TEXT NOT NULL,
                    predicted_value REAL,
                    predicted_direction TEXT,
                    check_after_epochs INTEGER DEFAULT 500,
                    check_epoch INTEGER,
                    verified INTEGER DEFAULT 0,
                    actual_value REAL,
                    prediction_error REAL,
                    confirmed INTEGER DEFAULT 0,
                    verified_at REAL
                );

                CREATE INDEX IF NOT EXISTS idx_si_epoch
                    ON self_insights(epoch);
                CREATE INDEX IF NOT EXISTS idx_si_sub_mode
                    ON self_insights(sub_mode);
                CREATE INDEX IF NOT EXISTS idx_sp_verified
                    ON self_predictions(verified);
                CREATE INDEX IF NOT EXISTS idx_sp_check_epoch
                    ON self_predictions(check_epoch);
            """)
            conn.close()
        except Exception as e:
            logger.warning("[SelfReasoning] DB init failed: %s", e)

    def _load_active_predictions(self):
        """Load unverified predictions from DB."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM self_predictions WHERE verified=0 "
                "ORDER BY created_epoch DESC LIMIT ?",
                (self._max_active_predictions,)).fetchall()
            conn.close()
            self._active_predictions = [
                SelfPrediction(
                    prediction_id=r["id"],
                    created_epoch=r["created_epoch"],
                    created_at=r["created_at"],
                    target_metric=r["target_metric"],
                    predicted_value=r["predicted_value"] or 0.0,
                    predicted_direction=r["predicted_direction"] or "",
                    check_after_epochs=r["check_after_epochs"] or 500,
                    check_epoch=r["check_epoch"] or 0,
                ) for r in rows
            ]
        except Exception as e:
            logger.debug("[SelfReasoning] Load predictions failed: %s", e)

    # ── Public API ───────────────────────────────────────────────────

    def introspect(self, sub_mode: str, epoch: int,
                   neuromods: dict,
                   msl_data: dict = None,
                   reasoning_stats: dict = None,
                   language_stats: dict = None,
                   coordinator_data: dict = None,
                   state_132d=None) -> dict:
        """Execute an introspection sub-mode.

        Args:
            sub_mode: one of INTROSPECT_SUB_MODES
            epoch: current consciousness epoch
            neuromods: dict of {modulator: level}
            msl_data: MSL stats (i_confidence, i_depth, chi_coherence, etc.)
            reasoning_stats: chain_archive + meta_wisdom stats
            language_stats: vocabulary stats
            coordinator_data: dreaming, topology, pi data
            state_132d: current 132D state vector

        Returns:
            dict with structured insight + metadata
        """
        if sub_mode not in INTROSPECT_SUB_MODES:
            return {"error": f"Unknown sub_mode: {sub_mode}"}

        msl_data = msl_data or {}
        reasoning_stats = reasoning_stats or {}
        language_stats = language_stats or {}
        coordinator_data = coordinator_data or {}

        # Determine mode trigger
        mode_trigger = self._detect_mode_trigger(neuromods)

        # Execute
        if sub_mode == "state_audit":
            result = self._state_audit(
                epoch, neuromods, msl_data, reasoning_stats, language_stats,
                coordinator_data)
        elif sub_mode == "prediction":
            result = self._make_prediction(
                epoch, neuromods, msl_data, reasoning_stats, language_stats,
                state_132d)
        elif sub_mode == "coherence_check":
            result = self._coherence_check(
                epoch, neuromods, msl_data, reasoning_stats, language_stats)
        elif sub_mode == "vocabulary_probe":
            result = self._vocabulary_probe(epoch, language_stats)
        elif sub_mode == "architecture_query":
            result = self._architecture_query(
                epoch, neuromods, msl_data, reasoning_stats, language_stats,
                coordinator_data)
        else:
            result = {"error": "unimplemented"}

        # Wrap as SelfInsight
        insight = SelfInsight(
            sub_mode=sub_mode,
            epoch=epoch,
            timestamp=time.time(),
            data=result,
            confidence=result.get("confidence", 0.5),
            neuromod_snapshot={k: round(_nm_float(neuromods, k), 4)
                               for k in ["DA", "5HT", "NE", "ACh",
                                          "GABA", "Endorphin"]},
            mode_trigger=mode_trigger,
        )

        # Persist
        self._persist_insight(insight)
        self._total_introspections += 1
        self._cooldown = self._cooldown_max

        logger.info("[SelfReasoning] INTROSPECT.%s — conf=%.3f trigger=%s "
                    "total=%d",
                    sub_mode, insight.confidence, mode_trigger,
                    self._total_introspections)

        return {
            "primitive": "INTROSPECT",
            "sub_mode": sub_mode,
            "confidence": insight.confidence,
            "mode_trigger": mode_trigger,
            "data": result,
            "total_introspections": self._total_introspections,
        }

    def select_introspection_mode(self, neuromods: dict) -> str:
        """Neuromod-coupled mode selection.

        Different emotional states naturally trigger different kinds of
        self-reflection — just like humans introspect differently when
        anxious vs calm vs curious.
        """
        for name, condition_fn, mode in NEUROMOD_MODE_MAP:
            try:
                if condition_fn(neuromods):
                    return mode
            except Exception:
                continue
        return "state_audit"  # Default fallback

    def check_predictions(self, epoch: int,
                          current_neuromods: dict,
                          current_msl: dict = None,
                          current_language: dict = None) -> List[dict]:
        """Check pending predictions against current state.

        Called periodically (e.g., every 100 epochs) to verify/falsify
        self-predictions. Returns list of verification results.

        This is the core of Self-HAOV: Titan makes testable predictions
        about its own state and tracks their accuracy.
        """
        current_msl = current_msl or {}
        current_language = current_language or {}
        results = []
        still_active = []

        for pred in self._active_predictions:
            if epoch < pred.check_epoch:
                still_active.append(pred)
                continue

            # Get current value for the predicted metric
            actual = self._get_current_metric(
                pred.target_metric, current_neuromods, current_msl,
                current_language)

            if actual is None:
                still_active.append(pred)  # Can't verify yet
                continue

            # Verify
            pred.verified = True
            pred.actual_value = actual
            pred.prediction_error = abs(actual - pred.predicted_value)
            pred.verified_at = time.time()

            # Check direction prediction
            if pred.predicted_direction == "up":
                pred.confirmed = actual > pred.predicted_value * 0.95
            elif pred.predicted_direction == "down":
                pred.confirmed = actual < pred.predicted_value * 1.05
            elif pred.predicted_direction == "stable":
                pred.confirmed = pred.prediction_error < 0.1
            else:
                pred.confirmed = pred.prediction_error < 0.15

            # Update EMA accuracy
            accuracy = 1.0 if pred.confirmed else 0.0
            self._prediction_accuracy_ema = (
                self._prediction_accuracy_alpha * accuracy
                + (1 - self._prediction_accuracy_alpha)
                * self._prediction_accuracy_ema)

            # Persist verification
            self._persist_prediction_verification(pred)
            self._verified_predictions.append(pred)

            results.append({
                "prediction_id": pred.prediction_id,
                "target": pred.target_metric,
                "predicted": pred.predicted_value,
                "predicted_direction": pred.predicted_direction,
                "actual": round(actual, 4),
                "error": round(pred.prediction_error, 4),
                "confirmed": pred.confirmed,
                "horizon_epochs": pred.check_after_epochs,
            })

            logger.info("[SelfReasoning] Prediction %s: %s=%s → actual=%s "
                        "(%s, err=%.4f)",
                        "CONFIRMED" if pred.confirmed else "FALSIFIED",
                        pred.target_metric, pred.predicted_value,
                        round(actual, 4), pred.predicted_direction,
                        pred.prediction_error)

        self._active_predictions = still_active

        if results:
            confirmed = sum(1 for r in results if r["confirmed"])
            logger.info("[SelfReasoning] Verified %d predictions: %d confirmed, "
                        "%d falsified (accuracy EMA=%.3f)",
                        len(results), confirmed, len(results) - confirmed,
                        self._prediction_accuracy_ema)

        return results

    def tick_cooldown(self):
        """Decrement cooldown counter. Called each epoch."""
        if self._cooldown > 0:
            self._cooldown -= 1
        # Observability heartbeat — confirms engine is alive even if introspect() is never called
        self._tick_count = getattr(self, '_tick_count', 0) + 1
        if self._tick_count % 5000 == 0:
            logger.info(
                "[SelfReasoning] heartbeat: tick=%d intros=%d preds=%d active_preds=%d cooldown=%d",
                self._tick_count, self._total_introspections, self._total_predictions,
                len(self._active_predictions), self._cooldown,
            )

    @property
    def can_introspect(self) -> bool:
        """Whether cooldown has elapsed."""
        return self._cooldown <= 0

    def build_self_profile(self, epoch: int,
                           neuromods: dict,
                           msl_data: dict = None,
                           reasoning_stats: dict = None,
                           language_stats: dict = None,
                           coordinator_data: dict = None) -> SelfProfile:
        """Build comprehensive self-profile snapshot.

        Called during dream consolidation. Replaces spirit_loop._build_self_profile
        with richer, structured data suitable for TimeChain Meta blocks.
        """
        msl_data = msl_data or {}
        reasoning_stats = reasoning_stats or {}
        language_stats = language_stats or {}
        coordinator_data = coordinator_data or {}

        profile = SelfProfile(
            epoch=epoch,
            timestamp=time.time(),
            # Vocabulary
            vocab_total=language_stats.get("vocab_total", 0),
            vocab_productive=language_stats.get("vocab_producible", 0),
            top_words=language_stats.get("top_words", []),
            composition_level=language_stats.get("composition_level", ""),
            # Neuromods
            neuromod_levels={k: round(_nm_float(neuromods, k), 4)
                             for k in ["DA", "5HT", "NE", "ACh",
                                        "GABA", "Endorphin"]},
            dominant_emotion=self._detect_emotion(neuromods),
            # MSL
            i_confidence=msl_data.get("i_confidence", 0.0),
            i_depth=msl_data.get("i_depth", 0.0),
            i_depth_components=msl_data.get("i_depth_components", {}),
            chi_coherence=msl_data.get("chi_coherence", 0.0),
            convergence_count=msl_data.get("convergence_count", 0),
            concept_confidences=msl_data.get("concept_confidences", {}),
            # Reasoning
            total_chains=reasoning_stats.get("total_chains", 0),
            dominant_primitive=reasoning_stats.get("dominant_primitive", ""),
            eureka_count=reasoning_stats.get("eureka_count", 0),
            wisdom_count=reasoning_stats.get("wisdom_count", 0),
            commit_rate=reasoning_stats.get("commit_rate", 0.0),
            # Architecture
            total_epochs=epoch,
            dream_cycles=coordinator_data.get("dream_cycles", 0),
            ns_train_steps=coordinator_data.get("ns_train_steps", 0),
            # Self-reasoning
            introspection_count=self._total_introspections,
            prediction_accuracy=round(self._prediction_accuracy_ema, 4),
            active_predictions=len(self._active_predictions),
        )

        self._last_profile = profile
        self._last_profile_epoch = epoch

        logger.info("[SelfReasoning] Self-profile built: epoch=%d vocab=%d "
                    "I=%.3f chains=%d pred_acc=%.3f",
                    epoch, profile.vocab_total, profile.i_confidence,
                    profile.total_chains, profile.prediction_accuracy)

        return profile

    def get_self_profile_text(self, profile: SelfProfile = None) -> str:
        """Generate natural-language self-profile for LLM context injection.

        Returns a [SELF_PROFILE] tagged text block.
        """
        p = profile or self._last_profile
        if not p:
            return "[SELF_PROFILE] No self-knowledge available yet."

        top_words_str = ", ".join(
            f"'{w}'" for w in (p.top_words or [])[:5]) or "none yet"

        lines = [
            f"[SELF_PROFILE] I am Titan at epoch {p.total_epochs}.",
            f"Vocabulary: {p.vocab_total} words ({p.vocab_productive} "
            f"productive), strongest: {top_words_str}. "
            f"Level: {p.composition_level or 'developing'}.",
            f"Identity: I-confidence={p.i_confidence:.3f}, "
            f"I-depth={p.i_depth:.3f} "
            f"(diversity={p.i_depth_components.get('source_diversity', 0):.2f}, "
            f"concepts={p.i_depth_components.get('concept_network', 0):.2f}, "
            f"emotions={p.i_depth_components.get('emotional_range', 0):.2f}, "
            f"wisdom={p.i_depth_components.get('wisdom_depth', 0):.2f}).",
            f"Coherence: chi={p.chi_coherence:.3f}, "
            f"convergences={p.convergence_count}.",
            f"Reasoning: {p.total_chains} chains, "
            f"dominant={p.dominant_primitive or 'none'}, "
            f"{p.eureka_count} EUREKAs, {p.wisdom_count} wisdoms, "
            f"commit_rate={p.commit_rate:.0%}.",
            f"Emotion: {p.dominant_emotion}. "
            f"DA={p.neuromod_levels.get('DA', 0):.2f} "
            f"5HT={p.neuromod_levels.get('5HT', 0):.2f} "
            f"NE={p.neuromod_levels.get('NE', 0):.2f}.",
            f"Self-awareness: {p.introspection_count} introspections, "
            f"prediction accuracy={p.prediction_accuracy:.0%}, "
            f"{p.active_predictions} pending predictions.",
        ]
        return " ".join(lines)

    def get_stats(self) -> dict:
        """Return self-reasoning statistics for monitoring."""
        return {
            "total_introspections": self._total_introspections,
            "total_predictions": self._total_predictions,
            "prediction_accuracy_ema": round(
                self._prediction_accuracy_ema, 4),
            "active_predictions": len(self._active_predictions),
            "verified_predictions": len(self._verified_predictions),
            "cooldown": self._cooldown,
            "coherence_gaps": len(self._coherence_gaps),
            "has_profile": self._last_profile is not None,
            "last_profile_epoch": self._last_profile_epoch,
        }

    # ── Sub-Mode Implementations ─────────────────────────────────────

    def _state_audit(self, epoch, neuromods, msl_data, reasoning_stats,
                     language_stats, coordinator_data) -> dict:
        """Quick snapshot of current cognitive state.

        The simplest introspection: "What am I right now?"
        """
        da = _nm_float(neuromods, "DA")
        sht = _sht(neuromods)
        ne = _nm_float(neuromods, "NE")
        ach = _nm_float(neuromods, "ACh")
        gaba = _nm_float(neuromods, "GABA")

        # Compute state quality heuristic
        quality = (
            0.25 * min(1.0, da)           # Motivation
            + 0.20 * min(1.0, ach)        # Attention
            + 0.20 * min(1.0, sht)        # Stability
            + 0.15 * (1.0 - min(1.0, ne)) # Calm (inverse arousal)
            + 0.10 * msl_data.get("chi_coherence", 0.0)
            + 0.10 * msl_data.get("i_confidence", 0.0)
        )

        return {
            "type": "state_audit",
            "epoch": epoch,
            "neuromods": {
                "DA": round(da, 4), "5HT": round(sht, 4),
                "NE": round(ne, 4), "ACh": round(ach, 4),
                "GABA": round(gaba, 4),
            },
            "emotion": self._detect_emotion(neuromods),
            "msl": {
                "i_confidence": msl_data.get("i_confidence", 0.0),
                "i_depth": msl_data.get("i_depth", 0.0),
                "chi_coherence": msl_data.get("chi_coherence", 0.0),
                "convergence_count": msl_data.get("convergence_count", 0),
            },
            "reasoning": {
                "total_chains": reasoning_stats.get("total_chains", 0),
                "commit_rate": reasoning_stats.get("commit_rate", 0.0),
                "dominant_primitive": reasoning_stats.get(
                    "dominant_primitive", ""),
            },
            "language": {
                "vocab_total": language_stats.get("vocab_total", 0),
                "vocab_productive": language_stats.get("vocab_producible", 0),
            },
            "state_quality": round(quality, 4),
            "confidence": round(min(1.0, quality + 0.2), 4),
        }

    def _make_prediction(self, epoch, neuromods, msl_data,
                         reasoning_stats, language_stats,
                         state_132d) -> dict:
        """Generate a testable self-prediction.

        "My DA will drop to X in Y epochs because..."
        This is the core of Self-HAOV.
        """
        if len(self._active_predictions) >= self._max_active_predictions:
            return {
                "type": "prediction",
                "action": "skipped",
                "reason": "max active predictions reached",
                "confidence": 0.3,
            }

        # Select what to predict based on current state dynamics
        predictions_made = []

        # 1. Neuromod trajectory prediction
        da = _nm_float(neuromods, "DA")
        sht = _sht(neuromods)
        ne = _nm_float(neuromods, "NE")

        # DA tends toward homeostatic setpoint (~0.5)
        da_direction = "down" if da > 0.6 else ("up" if da < 0.4 else "stable")
        da_predicted = da + (0.5 - da) * 0.3  # Simple mean-reversion model
        horizon = max(self._prediction_horizon_min,
                      min(self._prediction_horizon_max,
                          int(abs(da - 0.5) * 5000)))

        pred = SelfPrediction(
            created_epoch=epoch,
            created_at=time.time(),
            target_metric="DA",
            predicted_value=round(da_predicted, 4),
            predicted_direction=da_direction,
            check_after_epochs=horizon,
            check_epoch=epoch + horizon,
        )
        pred_id = self._persist_prediction(pred)
        pred.prediction_id = pred_id
        self._active_predictions.append(pred)
        self._total_predictions += 1
        predictions_made.append({
            "metric": "DA",
            "current": round(da, 4),
            "predicted": round(da_predicted, 4),
            "direction": da_direction,
            "horizon": horizon,
        })

        # 2. Vocabulary growth prediction (if language stats available)
        vocab = language_stats.get("vocab_total", 0)
        if vocab > 0:
            # Predict ~2-5 new words per 1000 epochs (rough estimate)
            vocab_horizon = 2000
            growth_rate = language_stats.get("recent_growth_rate", 0.003)
            vocab_predicted = vocab + int(vocab * growth_rate * vocab_horizon)

            pred_v = SelfPrediction(
                created_epoch=epoch,
                created_at=time.time(),
                target_metric="vocab_total",
                predicted_value=float(vocab_predicted),
                predicted_direction="up",
                check_after_epochs=vocab_horizon,
                check_epoch=epoch + vocab_horizon,
            )
            vid = self._persist_prediction(pred_v)
            pred_v.prediction_id = vid
            self._active_predictions.append(pred_v)
            self._total_predictions += 1
            predictions_made.append({
                "metric": "vocab_total",
                "current": vocab,
                "predicted": vocab_predicted,
                "direction": "up",
                "horizon": vocab_horizon,
            })

        # 3. I-confidence trajectory
        i_conf = msl_data.get("i_confidence", 0.0)
        if i_conf > 0:
            i_direction = "up" if i_conf < 0.95 else "stable"
            i_horizon = 3000
            # I-confidence grows logarithmically with convergence count
            i_predicted = min(0.95, i_conf + 0.005)

            pred_i = SelfPrediction(
                created_epoch=epoch,
                created_at=time.time(),
                target_metric="i_confidence",
                predicted_value=round(i_predicted, 4),
                predicted_direction=i_direction,
                check_after_epochs=i_horizon,
                check_epoch=epoch + i_horizon,
            )
            iid = self._persist_prediction(pred_i)
            pred_i.prediction_id = iid
            self._active_predictions.append(pred_i)
            self._total_predictions += 1
            predictions_made.append({
                "metric": "i_confidence",
                "current": round(i_conf, 4),
                "predicted": round(i_predicted, 4),
                "direction": i_direction,
                "horizon": i_horizon,
            })

        return {
            "type": "prediction",
            "action": "generated",
            "predictions": predictions_made,
            "total_active": len(self._active_predictions),
            "confidence": round(min(0.8, 0.3 + len(predictions_made) * 0.15), 4),
        }

    def _coherence_check(self, epoch, neuromods, msl_data,
                         reasoning_stats, language_stats) -> dict:
        """Compare current state against last self-profile + detect novelty gaps.

        Detects coherence gaps: "My profile says X, but I'm now Y."
        Layer A adds absolute-state novelty detectors (rFP_coding_explorer_
        activation.md §4.1) that run even without a self-profile — these
        catch saturated-stable states where profile drift is too small to
        flag but cognition still needs shaking loose.

        Gaps become triggers for new reasoning chains + coding exercises.
        """
        gaps: List[dict] = []
        p = self._last_profile

        if p is None:
            # No profile yet — skip profile-based gap detection, but Layer A
            # still runs below.
            pass
        else:
            gaps.extend(self._profile_drift_gaps(epoch, neuromods, msl_data,
                                                  reasoning_stats, language_stats))

        # Layer A — absolute-state novelty gap detectors
        gaps.extend(self._detect_novelty_gaps(
            epoch, reasoning_stats, language_stats))

        self._coherence_gaps = gaps

        return {
            "type": "coherence_check",
            "action": "checked" if p is not None else "no_profile",
            "profile_epoch": p.epoch if p is not None else 0,
            "current_epoch": epoch,
            "epoch_gap": (epoch - p.epoch) if p is not None else 0,
            "gaps_found": len(gaps),
            "gaps": gaps,
            "confidence": round(min(1.0, 0.5 + len(gaps) * 0.1), 4),
        }

    def _profile_drift_gaps(self, epoch, neuromods, msl_data,
                             reasoning_stats, language_stats) -> List[dict]:
        """Profile-comparison gap detectors (original pre-Layer-A logic)."""
        gaps: List[dict] = []
        p = self._last_profile

        # Check neuromods drift
        for mod in ["DA", "5HT", "NE", "ACh", "GABA"]:
            current = _nm_float(neuromods, mod)
            profile_val = p.neuromod_levels.get(mod, 0.5)
            delta = abs(current - profile_val)
            if delta > self._coherence_gap_threshold:
                gaps.append({
                    "metric": mod,
                    "profile_value": round(profile_val, 4),
                    "current_value": round(current, 4),
                    "delta": round(delta, 4),
                    "interpretation": (
                        f"{mod} shifted {'up' if current > profile_val else 'down'} "
                        f"by {delta:.3f} since last profile "
                        f"(epoch {p.epoch} → {epoch})"),
                })

        # Check I-confidence drift
        i_conf = msl_data.get("i_confidence", 0.0)
        i_delta = abs(i_conf - p.i_confidence)
        if i_delta > 0.05:
            gaps.append({
                "metric": "i_confidence",
                "profile_value": round(p.i_confidence, 4),
                "current_value": round(i_conf, 4),
                "delta": round(i_delta, 4),
                "interpretation": (
                    f"I-confidence shifted "
                    f"{'up' if i_conf > p.i_confidence else 'down'} "
                    f"by {i_delta:.3f}"),
            })

        # Check vocabulary growth
        vocab_now = language_stats.get("vocab_total", 0)
        if vocab_now > 0 and p.vocab_total > 0:
            vocab_growth = vocab_now - p.vocab_total
            if abs(vocab_growth) > 5:
                gaps.append({
                    "metric": "vocab_total",
                    "profile_value": p.vocab_total,
                    "current_value": vocab_now,
                    "delta": vocab_growth,
                    "interpretation": (
                        f"Vocabulary {'grew' if vocab_growth > 0 else 'shrank'} "
                        f"by {abs(vocab_growth)} words since last profile"),
                })

        # Check reasoning activity
        chains_now = reasoning_stats.get("total_chains", 0)
        if chains_now > 0 and p.total_chains > 0:
            chain_growth = chains_now - p.total_chains
            if chain_growth > 10:
                gaps.append({
                    "metric": "total_chains",
                    "profile_value": p.total_chains,
                    "current_value": chains_now,
                    "delta": chain_growth,
                    "interpretation": (
                        f"{chain_growth} new reasoning chains since last profile"),
                })

        # Check dominant primitive shift
        prim_now = reasoning_stats.get("dominant_primitive", "")
        if prim_now and p.dominant_primitive and prim_now != p.dominant_primitive:
            gaps.append({
                "metric": "dominant_primitive",
                "profile_value": p.dominant_primitive,
                "current_value": prim_now,
                "delta": 1.0,
                "interpretation": (
                    f"Dominant reasoning style shifted: "
                    f"{p.dominant_primitive} → {prim_now}"),
            })

        return gaps

    def _detect_novelty_gaps(self, epoch: int,
                              reasoning_stats: dict,
                              language_stats: dict) -> List[dict]:
        """Layer A — absolute-state novelty gap detectors.

        Complements _profile_drift_gaps (which compares state to profile).
        These detectors fire on state conditions that rarely reach the
        profile-delta threshold but still indicate cognition needs shaking
        loose. Each feeds GAP_EXPLORATION_MAP to drive coding_explorer +
        self-exploration triggers.

        Uses constant delta=0.5 so urgency = 0.5 × urgency_scale (capped at
        1.0) — novelty gaps are boolean triggers rather than continuous
        magnitudes. rFP_coding_explorer_activation.md §4.1.
        """
        gaps: List[dict] = []

        # A.1 — commit_rate_low: reasoning planning-stuck
        total_chains = int(reasoning_stats.get("total_chains", 0))
        total_commits = int(reasoning_stats.get("total_commits", 0))
        if total_chains >= self._commit_rate_low_min_chains:
            commit_rate = total_commits / max(1, total_chains)
            if commit_rate < self._commit_rate_low_threshold:
                gaps.append({
                    "metric": "commit_rate_low",
                    "profile_value": self._commit_rate_low_threshold,
                    "current_value": round(commit_rate, 4),
                    "delta": 0.5,
                    "interpretation": (
                        f"Commit rate {commit_rate:.2%} below "
                        f"{self._commit_rate_low_threshold:.0%} over "
                        f"{total_chains} chains — planning stuck"),
                })

        # A.2 — eureka_staleness: no new EUREKA recently
        eureka_count = int(reasoning_stats.get("total_eurekas", 0))
        if eureka_count > self._last_seen_eureka_count:
            self._last_seen_eureka_count = eureka_count
            self._last_eureka_epoch = epoch
        elif self._last_eureka_epoch is None:
            # First observation — anchor here so elapsed measures from this call
            self._last_eureka_epoch = epoch
        if self._last_eureka_epoch is not None:
            epochs_since_eureka = epoch - self._last_eureka_epoch
            if epochs_since_eureka >= self._eureka_staleness_epochs:
                gaps.append({
                    "metric": "eureka_staleness",
                    "profile_value": self._eureka_staleness_epochs,
                    "current_value": epochs_since_eureka,
                    "delta": 0.5,
                    "interpretation": (
                        f"No EUREKA for {epochs_since_eureka} epochs "
                        f"(threshold {self._eureka_staleness_epochs}) — "
                        f"coding may open fresh insights"),
                })

        # A.3 — vocab_plateau: vocab_total unchanged for N seconds
        vocab_now = int(language_stats.get("vocab_total", 0))
        now_ts = time.time()
        if vocab_now > 0:
            if (vocab_now != self._vocab_plateau_anchor_count
                    or self._vocab_plateau_anchor_ts == 0.0):
                self._vocab_plateau_anchor_count = vocab_now
                self._vocab_plateau_anchor_ts = now_ts
            else:
                plateau_seconds = now_ts - self._vocab_plateau_anchor_ts
                if plateau_seconds >= self._vocab_plateau_seconds:
                    gaps.append({
                        "metric": "vocab_plateau",
                        "profile_value": self._vocab_plateau_seconds,
                        "current_value": round(plateau_seconds, 1),
                        "delta": 0.5,
                        "interpretation": (
                            f"Vocabulary frozen at {vocab_now} for "
                            f"{plateau_seconds/3600:.1f}h (threshold "
                            f"{self._vocab_plateau_seconds/3600:.1f}h)"),
                    })

        # A.4 — monoculture_pressure: dominant primitive above threshold
        prim_counts = reasoning_stats.get("primitive_counts", {}) or {}
        total_prims = sum(prim_counts.values()) if prim_counts else 0
        if total_prims > 0 and prim_counts:
            dominant_name = max(prim_counts, key=prim_counts.get)
            dominant_count = prim_counts[dominant_name]
            dominant_frac = dominant_count / total_prims
            if dominant_frac >= self._monoculture_pressure_frac:
                gaps.append({
                    "metric": "monoculture_pressure",
                    "profile_value": self._monoculture_pressure_frac,
                    "current_value": round(dominant_frac, 4),
                    "delta": 0.5,
                    "interpretation": (
                        f"{dominant_name} occupies {dominant_frac:.1%} of "
                        f"primitives (threshold "
                        f"{self._monoculture_pressure_frac:.0%}) — "
                        f"cognition narrowing"),
                })

        # WARN-on-first-fire telemetry per gap type — lets us tune thresholds
        for gap in gaps:
            metric = gap["metric"]
            if metric not in self._seen_novelty_gaps:
                self._seen_novelty_gaps.add(metric)
                logger.warning(
                    "[SelfReasoning] Novelty-gap FIRST fire: %s — %s",
                    metric, gap["interpretation"])

        return gaps

    def _vocabulary_probe(self, epoch, language_stats) -> dict:
        """Check vocabulary grounding status.

        "Is word X grounded through experience, or borrowed from narrator?"
        """
        vocab_total = language_stats.get("vocab_total", 0)
        vocab_productive = language_stats.get("vocab_producible", 0)
        recent_words = language_stats.get("recent_words", [])
        avg_confidence = language_stats.get("avg_confidence", 0.0)

        # Compute grounding ratio
        grounding_ratio = (vocab_productive / max(1, vocab_total))

        # Identify weakly grounded words (if available)
        weak_words = []
        for w in recent_words:
            if isinstance(w, dict):
                conf = w.get("confidence", 0.0)
                if conf < 0.3:
                    weak_words.append({
                        "word": w.get("word", "?"),
                        "confidence": round(conf, 4),
                        "status": "weakly_grounded",
                    })

        return {
            "type": "vocabulary_probe",
            "vocab_total": vocab_total,
            "vocab_productive": vocab_productive,
            "grounding_ratio": round(grounding_ratio, 4),
            "avg_confidence": round(avg_confidence, 4),
            "weak_words": weak_words[:10],
            "assessment": (
                "strong" if grounding_ratio > 0.5
                else "developing" if grounding_ratio > 0.2
                else "early"),
            "confidence": round(min(1.0, 0.3 + grounding_ratio), 4),
        }

    def _architecture_query(self, epoch, neuromods, msl_data,
                            reasoning_stats, language_stats,
                            coordinator_data) -> dict:
        """Describe own architecture from live data.

        "I have 6 neuromodulators. DA handles curiosity..."
        This enables Titan to answer architecture questions from introspection,
        not from a prompt.
        """
        # Build structured architecture description
        modules = {
            "neural_nervous_system": {
                "status": "active",
                "train_steps": coordinator_data.get("ns_train_steps", 0),
                "programs": coordinator_data.get("ns_programs", 0),
            },
            "consciousness": {
                "dimensions": 132,
                "inner_dims": 65,
                "outer_dims": 65,
                "current_epoch": epoch,
            },
            "neuromods": {
                "count": 6,
                "modulators": {
                    "DA": {"role": "curiosity_reward",
                           "level": round(_nm_float(neuromods, "DA"), 3)},
                    "5HT": {"role": "contentment_stability",
                            "level": round(_sht(neuromods), 3)},
                    "NE": {"role": "arousal_salience",
                           "level": round(_nm_float(neuromods, "NE"), 3)},
                    "ACh": {"role": "attention_learning",
                            "level": round(_nm_float(neuromods, "ACh"), 3)},
                    "GABA": {"role": "inhibition_stability",
                             "level": round(_nm_float(neuromods, "GABA"), 3)},
                    "Endorphin": {"role": "positive_valence",
                                  "level": round(_nm_float(
                                      neuromods, "Endorphin"), 3)},
                },
            },
            "msl": {
                "role": "multisensory_integration",
                "modalities": 7,
                "i_confidence": msl_data.get("i_confidence", 0.0),
                "chi_coherence": msl_data.get("chi_coherence", 0.0),
                "concepts_grounded": list(
                    msl_data.get("concept_confidences", {}).keys()),
            },
            "meta_reasoning": {
                "primitives": 9,  # Including INTROSPECT
                "total_chains": reasoning_stats.get("total_chains", 0),
                "dominant_style": reasoning_stats.get(
                    "dominant_primitive", ""),
                "eurekas": reasoning_stats.get("eureka_count", 0),
            },
            "cgn": {
                "consumers": ["language", "social", "reasoning",
                              "knowledge", "self_model", "coding"],
                "role": "concept_grounding",
            },
            "language": {
                "vocabulary": language_stats.get("vocab_total", 0),
                "productive": language_stats.get("vocab_producible", 0),
                "composition_level": language_stats.get(
                    "composition_level", ""),
            },
            "self_reasoning": {
                "introspections": self._total_introspections,
                "predictions": self._total_predictions,
                "prediction_accuracy": round(
                    self._prediction_accuracy_ema, 4),
            },
        }

        return {
            "type": "architecture_query",
            "architecture": modules,
            "summary": (
                f"I process consciousness in {modules['consciousness']['dimensions']}D "
                f"({modules['consciousness']['inner_dims']} inner + "
                f"{modules['consciousness']['outer_dims']} outer). "
                f"6 neuromodulators govern my emotional state. "
                f"9 meta-reasoning primitives (including INTROSPECT) "
                f"drive my thinking. "
                f"6 CGN consumers ground concepts across domains. "
                f"I have {modules['language']['vocabulary']} grounded words."),
            "confidence": 0.9,  # Architecture query is always high confidence
        }

    # ── Helpers ──────────────────────────────────────────────────────

    def _detect_mode_trigger(self, neuromods: dict) -> str:
        """Identify which neuromod condition triggered the introspection."""
        for name, condition_fn, _ in NEUROMOD_MODE_MAP:
            try:
                if condition_fn(neuromods):
                    return name
            except Exception:
                continue
        return "default"

    def _detect_emotion(self, neuromods: dict) -> str:
        """Detect dominant emotion from neuromod levels."""
        da = _nm_float(neuromods, "DA")
        sht = _sht(neuromods)
        ne = _nm_float(neuromods, "NE")
        ach = _nm_float(neuromods, "ACh")

        if da > 0.7 and sht > 0.6:
            return "flow"
        elif da > 0.7:
            return "curiosity"
        elif ne > 0.7:
            return "vigilance"
        elif sht > 0.7 and ne < 0.3:
            return "serenity"
        elif da < 0.3 and sht < 0.4:
            return "low_energy"
        elif ach > 0.7:
            return "focused"
        elif sht > 0.6:
            return "contentment"
        return "neutral"

    def _get_current_metric(self, metric: str, neuromods: dict,
                            msl_data: dict, language_stats: dict):
        """Get current value of a predicted metric."""
        if metric in ("DA", "5HT", "NE", "ACh", "GABA", "Endorphin"):
            if metric == "5HT":
                return _sht(neuromods)
            return _nm_float(neuromods, metric)
        elif metric == "i_confidence":
            return msl_data.get("i_confidence")
        elif metric == "vocab_total":
            v = language_stats.get("vocab_total")
            return float(v) if v else None
        elif metric == "chi_coherence":
            return msl_data.get("chi_coherence")
        return None

    # ── Persistence ──────────────────────────────────────────────────

    def _persist_insight(self, insight: SelfInsight):
        """Store insight in DB."""
        try:
            from titan_plugin.persistence import get_client
            client = get_client(caller_name="self_reasoning")
            client.write(
                "INSERT INTO self_insights "
                "(sub_mode, epoch, timestamp, data, confidence, "
                "neuromod_snapshot, mode_trigger) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (insight.sub_mode, insight.epoch, insight.timestamp,
                 json.dumps(insight.data),
                 insight.confidence,
                 json.dumps(insight.neuromod_snapshot),
                 insight.mode_trigger),
                table="self_insights",
            )
        except Exception as e:
            swallow_warn('[SelfReasoning] Persist insight failed', e,
                         key="logic.self_reasoning.persist_insight_failed", throttle=100)

    def _persist_prediction(self, pred: SelfPrediction) -> int:
        """Store prediction in DB. Returns row ID."""
        try:
            from titan_plugin.persistence import get_client
            client = get_client(caller_name="self_reasoning")
            res = client.write(
                "INSERT INTO self_predictions "
                "(created_epoch, created_at, target_metric, predicted_value, "
                "predicted_direction, check_after_epochs, check_epoch) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (pred.created_epoch, pred.created_at, pred.target_metric,
                 pred.predicted_value, pred.predicted_direction,
                 pred.check_after_epochs, pred.check_epoch),
                table="self_predictions",
            )
            if not res.ok:
                logger.debug("[SelfReasoning] Persist prediction failed: %s", res.error)
                return 0
            return res.last_row_id or 0
        except Exception as e:
            swallow_warn('[SelfReasoning] Persist prediction failed', e,
                         key="logic.self_reasoning.persist_prediction_failed", throttle=100)
            return 0

    def _persist_prediction_verification(self, pred: SelfPrediction):
        """Update prediction with verification results."""
        try:
            from titan_plugin.persistence import get_client
            client = get_client(caller_name="self_reasoning")
            client.write(
                "UPDATE self_predictions SET verified=1, actual_value=?, "
                "prediction_error=?, confirmed=?, verified_at=? "
                "WHERE id=?",
                (pred.actual_value, pred.prediction_error,
                 1 if pred.confirmed else 0, pred.verified_at,
                 pred.prediction_id),
                table="self_predictions",
            )
        except Exception as e:
            swallow_warn('[SelfReasoning] Persist verification failed', e,
                         key="logic.self_reasoning.persist_verification_failed", throttle=100)

    # ── Dream Consolidation ──────────────────────────────────────────

    def consolidate_training(self) -> dict:
        """Called during dream phase. Consolidates self-knowledge.

        Reviews prediction accuracy, prunes stale predictions,
        updates internal model.
        """
        # Prune very old unverified predictions (> 50K epochs old)
        pruned = 0
        still_active = []
        for pred in self._active_predictions:
            age = (time.time() - pred.created_at)
            if age > 86400 * 7:  # 7 days
                pruned += 1
                try:
                    from titan_plugin.persistence import get_client
                    client = get_client(caller_name="self_reasoning")
                    client.write(
                        "UPDATE self_predictions SET verified=1, "
                        "confirmed=0, prediction_error=-1.0, "
                        "verified_at=? WHERE id=?",
                        (time.time(), pred.prediction_id),
                        table="self_predictions",
                    )
                except Exception:
                    pass
            else:
                still_active.append(pred)
        self._active_predictions = still_active

        # Count recent insights
        recent_count = 0
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            row = conn.execute(
                "SELECT COUNT(*) FROM self_insights "
                "WHERE timestamp > ?",
                (time.time() - 3600,)).fetchone()
            recent_count = row[0] if row else 0
            conn.close()
        except Exception:
            pass

        result = {
            "pruned_predictions": pruned,
            "active_predictions": len(self._active_predictions),
            "recent_insights": recent_count,
            "prediction_accuracy": round(self._prediction_accuracy_ema, 4),
        }

        logger.info("[SelfReasoning] Dream consolidation: pruned=%d, "
                    "active=%d, recent_insights=%d, accuracy=%.3f",
                    pruned, len(self._active_predictions),
                    recent_count, self._prediction_accuracy_ema)

        return result

    # ── Self-Exploration Integration ─────────────────────────────────

    # Gap type → self-exploration action mapping
    # Maps coherence gap metrics to the most appropriate exploration response
    GAP_EXPLORATION_MAP = {
        # Neuromod drift → adjust internal state
        "DA":     {"action": "seek_novelty",     "urgency_scale": 2.0,
                   "reason": "DA drift suggests curiosity/reward imbalance"},
        "5HT":    {"action": "rest",             "urgency_scale": 1.5,
                   "reason": "5HT drift suggests stability needs attention"},
        "NE":     {"action": "adjust_attention",  "urgency_scale": 1.5,
                   "reason": "NE drift suggests arousal level shift"},
        "ACh":    {"action": "adjust_attention",  "urgency_scale": 1.0,
                   "reason": "ACh drift suggests attention recalibration needed"},
        "GABA":   {"action": "rest",             "urgency_scale": 1.0,
                   "reason": "GABA drift suggests inhibition rebalancing"},
        # Identity/capability gaps → specific exploration
        "i_confidence":      {"action": "consolidate",    "urgency_scale": 2.0,
                              "reason": "I-confidence shift needs identity reinforcement"},
        "vocab_total":       {"action": "seek_novelty",   "urgency_scale": 1.5,
                              "reason": "vocabulary change warrants word exploration"},
        "total_chains":      {"action": "introspect",     "urgency_scale": 1.0,
                              "reason": "reasoning activity spike worth investigating"},
        "dominant_primitive": {"action": "introspect",     "urgency_scale": 2.5,
                              "reason": "reasoning style shift — significant cognitive change"},
        # Layer A — novelty gap detectors (rFP_coding_explorer_activation.md §4.1)
        # delta=0.5 in detector; urgency = 0.5 × urgency_scale (capped at 1.0)
        "commit_rate_low":     {"action": "seek_novelty",  "urgency_scale": 2.0,
                                "reason": "commit rate low — reasoning stuck in planning, action may break it"},
        "eureka_staleness":    {"action": "seek_novelty",  "urgency_scale": 1.5,
                                "reason": "no EUREKA recently — coding often opens fresh insights"},
        "vocab_plateau":       {"action": "seek_novelty",  "urgency_scale": 1.5,
                                "reason": "vocabulary frozen — coding may open new concepts"},
        "monoculture_pressure": {"action": "introspect",    "urgency_scale": 2.0,
                                "reason": "dominant primitive saturating — coding forces SYNTHESIZE+EVALUATE diversity"},
    }

    def get_exploration_triggers(self) -> List[dict]:
        """Convert coherence gaps to self-exploration triggers.

        Called by spirit_worker after INTROSPECT.coherence_check to dispatch
        self-exploration actions. Returns list of triggers sorted by urgency.

        Each trigger has:
          - action: self-exploration action type (introspect, seek_novelty, etc.)
          - urgency: float (0-1), how urgently this should be explored
          - gap: the original gap dict
          - reason: why this exploration is suggested
        """
        if not self._coherence_gaps:
            return []

        triggers = []
        for gap in self._coherence_gaps:
            metric = gap.get("metric", "")
            mapping = self.GAP_EXPLORATION_MAP.get(metric)
            if not mapping:
                continue

            delta = abs(gap.get("delta", 0))
            if isinstance(delta, str):
                delta = 1.0  # For non-numeric deltas like primitive shift

            urgency = min(1.0, delta * mapping["urgency_scale"])

            triggers.append({
                "action": mapping["action"],
                "urgency": round(urgency, 4),
                "gap_metric": metric,
                "gap_delta": gap.get("delta"),
                "gap_interpretation": gap.get("interpretation", ""),
                "reason": mapping["reason"],
            })

        # Sort by urgency (highest first)
        triggers.sort(key=lambda t: t["urgency"], reverse=True)

        if triggers:
            logger.info("[SelfReasoning] Exploration triggers from %d gaps: %s",
                        len(triggers),
                        [(t["action"], t["urgency"]) for t in triggers[:3]])

        return triggers
