"""EMOT-CGN — Emotion as 8th CGN Consumer (rFP_emot_cgn_v2.md).

Adds emotion primitives (FLOW, PEACE, CURIOSITY, GRIEF, WONDER,
IMPASSE_TENSION, RESOLUTION, LOVE) as grounded concepts with V(s).
Emotion is a cognitive REGIME, not a primitive in the reasoning layer —
so EMOT-CGN grounds "what emotional state am I in?" rather than "what
should I do?".

The 8th CGN consumer. Consumer pattern mirrors MetaCGNConsumer but with
a simplified α-layer (linear combination of β-outputs + direct features,
no shared ValueNet dependency — EMOT-CGN is fully self-contained).

Lifecycle:
  1. __init__ — load grounding + HAOV state + clusterer; start in shadow mode
  2. handle_felt_tensor(feature_vec) — compute cluster assignment each emit
  3. observe_chain_evidence(chain_id, primitives_used, terminal_reward, ctx)
     — per chain conclude, update β posterior for the dominant emotion at
     chain start/end + test HAOV hypotheses
  4. get_current_emotion_state() — gated by is_active(); returns 10D for
     META-CGN context augmentation
  5. save_state() — atomic writes of grounding + HAOV + watchdog + clusters

Gate: `is_active()` returns False until graduation (rFP §6 criteria met).
All consumer integration sites MUST check the gate — behaviour is
identical to pre-EMOT-CGN until graduation (wire-now-gate-later).

See: titan-docs/rFP_emot_cgn_v2.md
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .emotion_cluster import (
    EMOT_PRIMITIVES,
    EMOT_PRIMITIVE_INDEX,
    FEATURE_DIM as CLUSTER_FEATURE_DIM,
    NUM_PRIMITIVES,
    EmotionClusterer,
)
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger("titan.emot_cgn")

# Internal identifier for logs/stats (short form matches module name).
CONSUMER_NAME = "emot_cgn"
# CGN worker registration name — matches rFP_cgn_orchestrator_promotion.md §4.1
# ("emotional" is the family-conventional short name like language/social/etc).
# cgn_worker pre-registers this name at boot (see cgn_worker.py:151+).
CGN_CONSUMER_NAME = "emotional"
BETA_PARAM_FLOOR = 1.0
MIGRATION_N_EFF_CAP = 200

# State vector dimensions.
# - FEATURE_DIMS (30D): sent to cgn_worker's SharedValueNet. Schema per
#   rFP_cgn_orchestrator_promotion §4.2 + Upgrade II (clusterer features).
# - Internal 150D feature vector (via build_feature_vec): used by the
#   local clusterer for fine-grained "which emotion am I in" detection.
#   Distinct from the 30D CGN state.
ACTION_DIMS = NUM_PRIMITIVES            # 8 emotion primitives
FEATURE_DIMS = 30                       # 30D for shared CGN ValueNet

# Seed hypothesis IDs
SEED_HYPOTHESES = [
    "H1_flow_neuromod",
    "H2_eureka_sequence",
    "H3_grief_persistence",
    "H4_serenity_introspect",
    "H5_emotion_drift_strategy",
    "H6_curiosity_drives_knowledge",
    "H7_impasse_v_above_resolution",       # improvement: added for graduation reachability
    "H8_curiosity_precedes_knowledge_growth",  # improvement: added for graduation reachability
]


# ── Bus signal mapping (for orphan-detection guard) ───────────────────
# Kept here (not in bus.py) so emit_emot_cgn_signal can import without cycles.
# Each entry: (consumer_key, event_type) → {primitive: weight_shift}
# Primitive name is here purely for provenance; actual update target uses
# EMOT_PRIMITIVES. weight=0.5 is a neutral Beta nudge (doesn't shift V
# while still satisfying consumer-side `if not mapping:` truthiness).
#
# Lesson from 2026-04-20 META-CGN Producer #16 orphan bug:
# NEVER use empty {} — consumer truthiness check rejects.
EMOT_SIGNAL_TO_PRIMITIVE = {
    ("emot_cgn", "cluster_assignment"):       {"FLOW": 0.5},
    ("emot_cgn", "chain_emotion_context"):    {"FLOW": 0.5},
    ("emot_cgn", "cluster_recenter"):         {"FLOW": 0.5},
    ("emot_cgn", "cluster_emerged"):          {"FLOW": 0.5},
    ("emot_cgn", "graduation_transition"):    {"FLOW": 0.5},
    ("emot_cgn", "rollback_fired"):           {"FLOW": 0.5},
}


# ── Utility ───────────────────────────────────────────────────────────

def _beta_mean(a: float, b: float) -> float:
    tot = max(BETA_PARAM_FLOOR * 2, a + b)
    return float(a / tot)


def _posterior_confidence(a: float, b: float, ref_n: float = 500.0) -> float:
    """Confidence: how much do α,β concentrate posterior vs prior?
    0.0 at Beta(1,1); asymptotes to 1.0 as n→∞."""
    n = (a - BETA_PARAM_FLOOR) + (b - BETA_PARAM_FLOOR)
    return float(min(1.0, n / ref_n))


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ── Dataclasses ───────────────────────────────────────────────────────

@dataclass
class EmotPrimitive:
    """Grounded value state for a single emotion primitive."""
    primitive_id: str
    alpha: float = 1.0
    beta: float = 1.0
    V: float = 0.5                  # posterior mean
    confidence: float = 0.0
    n_samples: int = 0
    variance: float = 0.25
    last_updated_ts: float = 0.0
    last_updated_chain: int = 0
    haov_rules: list = field(default_factory=list)

    def recompute_derived(self) -> None:
        a = max(BETA_PARAM_FLOOR, float(self.alpha))
        b = max(BETA_PARAM_FLOOR, float(self.beta))
        self.V = _beta_mean(a, b)
        self.confidence = _posterior_confidence(a, b)
        self.n_samples = int((a - BETA_PARAM_FLOOR) + (b - BETA_PARAM_FLOOR))
        self.variance = float((a * b) /
                              (max(1e-9, (a + b) ** 2) * max(1e-9, (a + b + 1))))


@dataclass
class EmotHypothesis:
    hypothesis_id: str
    description: str
    test_kind: str
    min_samples: int = 30
    evidence_window: int = 500
    confirmation_threshold: float = 0.1
    observations: deque = field(default_factory=lambda: deque(maxlen=500))
    status: str = "nascent"          # nascent | testing | confirmed | falsified
    test_count: int = 0
    last_test_ts: float = 0.0
    effect_size: float = 0.0
    notes: str = ""


def _build_seed_hypotheses() -> dict[str, EmotHypothesis]:
    """Return 8 initial HAOV hypotheses (rFP §4.5 + 2 improvements for
    graduation reachability).

    Graduation requires ≥4 of 8 confirmed (50%) — matches META-CGN's
    difficulty without making the bar unreachable.
    """
    return {
        "H1_flow_neuromod": EmotHypothesis(
            hypothesis_id="H1_flow_neuromod",
            description="FLOW cluster centroid has high DA + 5HT neuromod signatures",
            test_kind="flow_neuromod",
            min_samples=30, evidence_window=500,
            confirmation_threshold=0.05,
        ),
        "H2_eureka_sequence": EmotHypothesis(
            hypothesis_id="H2_eureka_sequence",
            description="RESOLUTION is preceded by IMPASSE_TENSION in the "
                        "prior cluster-history window ≥50% of the time",
            test_kind="eureka_sequence",
            min_samples=30, evidence_window=500,
            confirmation_threshold=0.10,
        ),
        "H3_grief_persistence": EmotHypothesis(
            hypothesis_id="H3_grief_persistence",
            description="GRIEF cluster average duration > other primitives'",
            test_kind="grief_persistence",
            min_samples=20, evidence_window=500,
            confirmation_threshold=0.15,
        ),
        "H4_serenity_introspect": EmotHypothesis(
            hypothesis_id="H4_serenity_introspect",
            description="Under PEACE cluster, SPIRIT_SELF + INTROSPECT "
                        "primitive share in chain > baseline",
            test_kind="serenity_introspect",
            min_samples=50, evidence_window=500,
            confirmation_threshold=0.05,
        ),
        "H5_emotion_drift_strategy": EmotHypothesis(
            hypothesis_id="H5_emotion_drift_strategy",
            description="Cluster change between chain start and end correlates "
                        "with meta-pattern shift",
            test_kind="emotion_drift_strategy",
            min_samples=50, evidence_window=500,
            confirmation_threshold=0.10,
        ),
        "H6_curiosity_drives_knowledge": EmotHypothesis(
            hypothesis_id="H6_curiosity_drives_knowledge",
            description="CURIOSITY cluster V > baseline when knowledge "
                        "acquisition rate is rising",
            test_kind="curiosity_drives_knowledge",
            min_samples=30, evidence_window=500,
            confirmation_threshold=0.05,
        ),
        "H7_impasse_v_above_resolution": EmotHypothesis(
            hypothesis_id="H7_impasse_v_above_resolution",
            description="On chains that DON'T reach terminal success, "
                        "IMPASSE_TENSION V > RESOLUTION V",
            test_kind="impasse_v_above_resolution",
            min_samples=30, evidence_window=500,
            confirmation_threshold=0.05,
        ),
        "H8_curiosity_precedes_knowledge_growth": EmotHypothesis(
            hypothesis_id="H8_curiosity_precedes_knowledge_growth",
            description="CURIOSITY cluster appears ≥5 chains before "
                        "knowledge concept count increments",
            test_kind="curiosity_precedes_knowledge_growth",
            min_samples=30, evidence_window=500,
            confirmation_threshold=0.10,
        ),
    }


# ── Main consumer ────────────────────────────────────────────────────

# PERSISTENCE_BY_DESIGN: EmotCGNConsumer persistence uses indexed-assignment
# loading pattern for _primitives (per-key into existing dict) and _hypotheses
# (same). _clusterer + _titan_id are set in __init__ from constructor args —
# never re-assigned from disk (identity is established at boot, not loaded).
# These are flagged by dead-wiring's "saved but never loaded" heuristic;
# suppressed here because the actual data IS restored correctly.
class EmotCGNConsumer:
    """8th CGN consumer — grounds emotion primitives.

    Lifecycle (mirrors MetaCGNConsumer):
      1. __init__: load state + clusterer
      2. handle_felt_tensor(fv): cluster assignment → emit bus signal
      3. observe_chain_evidence(...): per-chain update of emotion V(s)
      4. test_hypotheses: periodic HAOV accumulation
      5. check_graduation: evaluate gate criteria
      6. save_state: atomic persistence

    Thread-safety: single-caller pattern (runs inside meta_engine which
    is single-threaded within spirit_worker). No locks needed.
    """

    # Graduation window minimum (14 days per rFP §6 — deliberately slower
    # than META-CGN's ~7 days; emotions carry deeper stakes)
    GRAD_MIN_OBSERVATION_WINDOW_S = 14 * 86400

    def __init__(self, send_queue=None, titan_id: str = "T1",
                 save_dir: str = "data/emot_cgn",
                 module_name: str = "spirit"):
        self._send_queue = send_queue
        self._titan_id = titan_id
        self._save_dir = save_dir
        self._module_name = module_name

        os.makedirs(save_dir, exist_ok=True)
        self._grounding_path = os.path.join(save_dir, "primitive_grounding.json")
        self._haov_path = os.path.join(save_dir, "haov_hypotheses.json")
        self._watchdog_path = os.path.join(save_dir, "watchdog_state.json")
        self._shadow_log_path = os.path.join(save_dir, "shadow_mode_log.jsonl")

        # 8 emotion primitives with Beta(1,1) prior
        self._primitives: dict[str, EmotPrimitive] = {
            p: EmotPrimitive(primitive_id=p) for p in EMOT_PRIMITIVES
        }
        self._hypotheses: dict[str, EmotHypothesis] = _build_seed_hypotheses()

        # Graduation state machine
        self._status = "shadow_mode"
        self._graduation_progress = 0            # 0..100 ramp
        self._graduation_ts = 0.0
        self._pre_graduation_baseline: dict = {}
        self._chains_since_graduation = 0
        self._rolled_back_count = 0

        # Scale-invariant rollback (Option B pattern from META-CGN)
        self._post_grad_rewards: deque = deque(maxlen=50)
        self._pre_grad_rewards: deque = deque(maxlen=100)

        self._total_updates = 0
        self._total_observations = 0
        self._chains_since_save = 0
        self._chain_counter = 0
        self._test_cadence = 50
        self._evidence_since_last_test = 0
        self._creation_ts = time.time()

        # ── Clusterer ──────────────────────────────────────────────
        self._clusterer = EmotionClusterer(save_dir=save_dir)

        # ── Neuromod EMA window (rFP improvement #2) ────────────────
        # Rate-of-change computed over ~100-epoch window, not instantaneous.
        self._neuromod_ema: dict = {}   # name → EMA value
        self._neuromod_prev_ema: dict = {}  # name → EMA value 100-epoch ago
        self._neuromod_ema_alpha = 0.02   # ~50-epoch equivalent window
        self._neuromod_sample_count = 0

        # Rolling history for features
        self._recent_rewards: deque = deque(maxlen=5)
        self._dominant_emotion: str = "FLOW"
        self._last_cluster_assignment: tuple = ("FLOW", 0.0, 0.0)
        self._last_kin_resonance: float = 0.0   # wire-now default 0.0

        # Emotional dynamics tracking (for encode_state_30d per rFP §4.2)
        self._dominant_emotion_start_ts: float = time.time()
        self._dominant_emotion_transitions: deque = deque(maxlen=100)
        self._emotion_history_100: deque = deque(maxlen=100)

        # ── CGN integration (rFP_cgn_orchestrator_promotion §4 — 8th consumer) ──
        # Local β-posterior remains AUTHORITATIVE source of primitive V (mirrors
        # MetaCGNConsumer pattern). cgn_worker's SharedValueNet provides a
        # SHARED V(s) signal learned across all 8 consumers — blended into
        # get_blended_V per Upgrade I.
        self._cgn_client = None             # CGNConsumerClient (lazy-init)
        self._cgn_registered = False        # True after CGN_REGISTER sent
        self._cgn_transitions_sent = 0
        self._cgn_cross_insights_sent = 0
        self._cgn_cross_insights_received = 0
        self._last_cross_insight_ts = 0.0   # 0.2 Hz rate gate
        self._v_blend_alpha = 0.6           # Upgrade I: 0.6×β + 0.4×shm

        # Load config
        try:
            from titan_plugin.params import get_params as _get_params
            _cfg = _get_params("emot_cgn") or {}
        except Exception:
            _cfg = {}

        # Graduation thresholds (config-driven)
        self._grad_min_updates = int(_cfg.get("graduation_min_updates", 4000))
        self._grad_min_confirmed = int(_cfg.get("graduation_min_confirmed_hypotheses", 4))
        self._grad_min_primitives = int(_cfg.get("graduation_min_mature_primitives", 6))
        self._grad_min_samples_per = int(_cfg.get("graduation_min_samples_per_primitive", 100))
        self._grad_min_confidence = float(_cfg.get("graduation_min_confidence", 0.7))
        self._grad_contrast_v_gap = float(_cfg.get("graduation_contrast_v_gap", 0.15))
        self._grad_observation_window_s = float(
            _cfg.get("graduation_observation_window_s",
                     self.GRAD_MIN_OBSERVATION_WINDOW_S))
        self._rollback_sigma_k = float(_cfg.get("rollback_reward_sigma_k", 2.0))
        self._rollback_min_std_floor = float(
            _cfg.get("rollback_min_std_floor", 0.05))
        self._recenter_interval_s = float(
            _cfg.get("cluster_recenter_interval_s", 7 * 86400))
        # Save cadence (observability tunable — see titan_params.toml)
        self._save_cadence_chains = int(_cfg.get("save_cadence_chains", 5))
        # Cluster emergence threshold — pass through to clusterer
        self._cluster_emergence_threshold = int(
            _cfg.get("cluster_emergence_threshold", 30))
        # Apply recenter interval to clusterer (if config overrode default)
        self._clusterer._recenter_interval_s = self._recenter_interval_s
        self._clusterer._emergence_threshold = self._cluster_emergence_threshold

        self._load_state()
        self._load_watchdog_state()

        # ── CGN client + registration (after config loaded, before first save) ──
        # Mirrors MetaCGNConsumer._init_cgn_client pattern. Failsafe: if CGN
        # infra is down, consumer still accumulates local β grounding.
        self._init_cgn_client()
        self._send_register()

        # Save immediately on init so persistence files exist from the
        # first boot moment. Without this, /v4/emot-cgn returns empty
        # until save_cadence_chains is reached — opaque for operators
        # wanting to verify the consumer is live. Cheap write (~30KB
        # across 4 files).
        try:
            self.save_state()
        except Exception as _save_err:
            swallow_warn('[EmotCGN] save-on-init failed', _save_err,
                         key="logic.emot_cgn.save_on_init_failed", throttle=100)

        logger.info("[EmotCGN] Initialized (titan=%s, status=%s, "
                    "primitives=%d, hypotheses=%d)",
                    titan_id, self._status, NUM_PRIMITIVES,
                    len(self._hypotheses))

    # ── CGN integration (rFP §4, 8th consumer) ─────────────────────

    def encode_state_30d(self, primitive_id: str, ctx: dict) -> np.ndarray:
        """Build 30D state vector for cgn_worker's SharedValueNet.

        Schema (rFP_cgn_orchestrator_promotion §4.2 + Upgrade II clusterer slots):
          [0:6]   Neuromods: DA, 5HT, NE, ACh, Endorphin, GABA
          [6:14]  Emotional dynamics (8D):
                  [6]  cluster_confidence      (Upgrade II)
                  [7]  cluster_distance/4.0    (Upgrade II)
                  [8]  emotion_stability       (time since dominant last changed)
                  [9]  emotion_richness        (distinct primitives in last 100 obs)
                  [10] emotion_transition_rate (transitions / 100)
                  [11] dominant_emotion_share  (monoculture in emotions)
                  [12] felt_intensity          (|β_V - 0.5| × 2)
                  [13] valence                 (sign of β_V - 0.5)
          [14:24] Cross-system echoes (10D):
                  [14] language_grounding_rate
                  [15] social_quality_delta
                  [16] knowledge_acq_rate
                  [17] reasoning_chain_reward  (EMA of last-5 terminal_rewards)
                  [18] monoculture_share       (meta)
                  [19] success_rate_20
                  [20] chi_total
                  [21] confidence_avg_20
                  [22] DA_delta                (neuromod rate-of-change)
                  [23] 5HT_delta
          [24:30] Context (6D):
                  [24] time_since_dream (normalized)
                  [25] metabolic_energy (1 - drain)
                  [26] sleep_drive
                  [27] wake_drive
                  [28] developmental_age_normalized
                  [29] primitive_idx / 7 (which primitive this transition is for)

        All fields bounded to [0, 1]. Failsafe — returns zeros on error.
        Called per `observe_chain_evidence` to build the transition state.
        """
        try:
            vec = np.zeros(FEATURE_DIMS, dtype=np.float32)
            # ── [0:6] Neuromods ────
            vec[0] = _clip01(float(ctx.get("DA", 0.5)))
            vec[1] = _clip01(float(ctx.get("5HT", 0.5)))
            vec[2] = _clip01(float(ctx.get("NE", 0.5)))
            vec[3] = _clip01(float(ctx.get("ACh", 0.5)))
            vec[4] = _clip01(float(ctx.get("Endorphin", 0.5)))
            vec[5] = _clip01(float(ctx.get("GABA", 0.5)))

            # ── [6:14] Emotional dynamics ────
            # Upgrade II: cluster confidence + distance in 30D state
            vec[6] = _clip01(float(self._last_cluster_assignment[2]))
            dist = float(self._last_cluster_assignment[1])
            vec[7] = _clip01(dist / 4.0)
            # Emotion stability: time since dominant last changed (seconds,
            # normalized to 10-min window).
            stab_s = time.time() - self._dominant_emotion_start_ts
            vec[8] = _clip01(stab_s / 600.0)
            # Richness: count distinct primitives in last 100 obs
            hist = list(self._emotion_history_100)
            if hist:
                distinct = len(set(hist))
                vec[9] = _clip01(distinct / float(NUM_PRIMITIVES))
            else:
                vec[9] = 0.0
            # Transition rate
            vec[10] = _clip01(
                len(self._dominant_emotion_transitions) / 100.0)
            # Monoculture in emotions (dominant share in history window)
            if hist:
                dom = max(set(hist), key=hist.count)
                share = hist.count(dom) / len(hist)
                vec[11] = _clip01(share)
            else:
                vec[11] = 0.0
            # Felt intensity + valence from β-posterior of this primitive
            p = self._primitives.get(primitive_id)
            if p is not None:
                v_centered = p.V - 0.5
                vec[12] = _clip01(abs(v_centered) * 2.0)
                vec[13] = 0.5 + (0.5 if v_centered > 0
                                 else -0.5 if v_centered < 0 else 0.0)
            else:
                vec[12] = 0.0
                vec[13] = 0.5

            # ── [14:24] Cross-system echoes ────
            vec[14] = _clip01(float(ctx.get("language_grounding_rate", 0.0)))
            vec[15] = _clip01(float(ctx.get("social_quality_delta", 0.5)))
            vec[16] = _clip01(float(ctx.get("knowledge_acq_rate", 0.0)))
            if self._recent_rewards:
                vec[17] = _clip01(
                    sum(self._recent_rewards) / len(self._recent_rewards))
            else:
                vec[17] = 0.5
            vec[18] = _clip01(float(ctx.get("monoculture_share", 0.0)))
            vec[19] = _clip01(float(ctx.get("success_rate_20", 0.5)))
            vec[20] = _clip01(float(ctx.get("chi_total", 0.5)))
            vec[21] = _clip01(float(ctx.get("confidence_avg_20", 0.5)))
            # Neuromod deltas (rate-of-change, shifted to [0,1])
            deltas = self.get_neuromod_deltas()
            vec[22] = _clip01(deltas.get("DA", 0.0) + 0.5)
            vec[23] = _clip01(deltas.get("5HT", 0.0) + 0.5)

            # ── [24:30] Context ────
            vec[24] = _clip01(
                float(ctx.get("epochs_since_dream", 0)) / 2000.0)
            vec[25] = _clip01(1.0 - float(ctx.get("metabolic_drain", 0.5)))
            vec[26] = _clip01(float(ctx.get("sleep_drive", 0.0)))
            vec[27] = _clip01(float(ctx.get("wake_drive", 0.5)))
            vec[28] = _clip01(float(ctx.get("developmental_age", 0.0)))
            idx = EMOT_PRIMITIVE_INDEX.get(primitive_id, 0)
            vec[29] = idx / float(max(1, NUM_PRIMITIVES - 1))

            return vec
        except Exception as e:
            swallow_warn('[EmotCGN] encode_state_30d failed', e,
                         key="logic.emot_cgn.encode_state_30d_failed", throttle=100)
            return np.zeros(FEATURE_DIMS, dtype=np.float32)

    def _init_cgn_client(self) -> None:
        """Lazy-create the CGNConsumerClient. Failsafe — if CGN infra is
        down, consumer still accumulates local β grounding (no shm V reads
        + CGN_TRANSITION sends drop). Mirrors MetaCGNConsumer pattern."""
        try:
            from titan_plugin.logic.cgn_consumer_client import CGNConsumerClient
            self._cgn_client = CGNConsumerClient(
                consumer_name=CGN_CONSUMER_NAME,   # "emotional"
                send_queue=self._send_queue,
                module_name=self._module_name,
                shm_path="/dev/shm/cgn_live_weights.bin",
            )
            logger.info("[EmotCGN] CGNConsumerClient initialized (consumer=%s, "
                        "module=%s)", CGN_CONSUMER_NAME, self._module_name)
        except Exception as e:
            self._cgn_client = None
            logger.warning("[EmotCGN] CGNConsumerClient init failed: %s "
                           "(local β grounding still accumulates)", e)

    def _send_register(self) -> None:
        """Send CGN_REGISTER to register as 8th CGN consumer.

        Idempotent: cgn_worker pre-registers "emotional" at boot, so this
        is a defensive fallback if pre-registration didn't land (restart
        ordering, fresh data/cgn/). CGN worker's register_consumer is
        tolerant of re-registration with matching config.
        """
        if self._send_queue is None:
            logger.info("[EmotCGN] No send_queue — skipping CGN_REGISTER "
                        "(standalone/test mode)")
            return
        try:
            msg = {
                "type": "CGN_REGISTER",
                "src": self._module_name,
                "dst": "cgn",
                "ts": time.time(),
                "payload": {
                    "name": CGN_CONSUMER_NAME,         # "emotional"
                    "feature_dims": FEATURE_DIMS,       # 30
                    "action_dims": ACTION_DIMS,         # 8
                    "action_names": list(EMOT_PRIMITIVES),
                    # v1 uses chain terminal_reward. True coherence metric in
                    # TUNING_DATABASE.md#TUNING-EMOT-COHERENCE for v1.6+.
                    "reward_source": "terminal_reward",
                    "max_buffer_size": 500,
                    "consolidation_priority": 2,
                },
            }
            self._send_queue.put(msg)
            self._cgn_registered = True
            logger.info("[EmotCGN] Sent CGN_REGISTER (name=%s, feature_dims=%d, "
                        "action_dims=%d, primitives=%s)",
                        CGN_CONSUMER_NAME, FEATURE_DIMS, ACTION_DIMS,
                        list(EMOT_PRIMITIVES))
        except Exception as e:
            logger.warning("[EmotCGN] CGN_REGISTER send failed: %s", e)

    # ── Cross-consumer insights (Upgrade III) ─────────────────────

    def _maybe_emit_cross_insight(self, p_start: str, p_end: str,
                                    terminal_reward: float,
                                    ctx: dict) -> None:
        """Emit CGN_CROSS_INSIGHT when emotional state produces a signal
        informative to other CGN consumers. Rate-limited to 0.2 Hz.

        Emission conditions (any informative):
          - |terminal_reward - 0.5| > 0.3 (strongly positive OR negative)
          - Emotion transition with strategy_shift (meta-level signal)

        Routed to dst="all" — META-CGN + language + social + knowledge
        consumers can consume per their own handlers. Skipped by self
        via origin_consumer check in the incoming handler.
        """
        if self._send_queue is None:
            return
        now = time.time()
        if now - self._last_cross_insight_ts < 5.0:
            return
        try:
            informative = (abs(terminal_reward - 0.5) > 0.3
                           or (p_start != p_end
                               and bool(ctx.get("strategy_shift", False))))
            if not informative:
                return
            msg = {
                "type": "CGN_CROSS_INSIGHT",
                "src": self._module_name,
                "dst": "all",
                "ts": now,
                "payload": {
                    "origin_consumer": CGN_CONSUMER_NAME,
                    "insight_type": "emotion_outcome",
                    "emotion_start": p_start,
                    "emotion_end": p_end,
                    "terminal_reward": float(terminal_reward),
                    "transitioned": (p_start != p_end),
                    "ctx_summary": {
                        "DA": float(ctx.get("DA", 0.5)),
                        "5HT": float(ctx.get("5HT", 0.5)),
                        "monoculture_share": float(
                            ctx.get("monoculture_share", 0.0)),
                    },
                },
            }
            self._send_queue.put_nowait(msg)
            self._last_cross_insight_ts = now
            self._cgn_cross_insights_sent += 1
        except Exception as e:
            from titan_plugin.utils.silent_swallow import swallow_warn
            swallow_warn("[EmotCGN] _maybe_emit_cross_insight failed", e,
                         key="emot_cgn.maybe_emit_cross_insight")

    def handle_incoming_cross_insight(self, payload: dict) -> None:
        """Handle CGN_CROSS_INSIGHT from OTHER consumers (META-CGN,
        language, social, knowledge, etc.) — uses it to nudge the
        currently-dominant emotion's β-posterior. Indirect evidence →
        small weight (w=0.25, 1/4 of direct observation).

        Own emissions bouncing via dst="all" are ignored by origin check.
        v1 consumes only 'chain_outcome' insight_type from META-CGN.
        Other insight types become informative in future iterations.

        Routed via spirit_worker's CGN_CROSS_INSIGHT message handler
        (dispatched to meta_engine._emot_cgn on receipt).
        """
        try:
            origin = str(payload.get("origin_consumer", ""))
            if origin == CGN_CONSUMER_NAME:
                return  # ignore own emissions
            insight_type = str(payload.get("insight_type", ""))
            if insight_type != "chain_outcome":
                return
            reward = float(payload.get("terminal_reward", 0.5))
            dom = self._dominant_emotion
            if dom not in self._primitives:
                return
            p = self._primitives[dom]
            outcome_01 = _clip01(reward)
            # §11 Q3 Option B (2026-04-24) — anti-dominance correction.
            # Previous static w=0.25 compounded monoculture: WONDER dominant
            # 74% of time → 74% of positive peer outcomes hit WONDER, further
            # reinforcing its β posterior. Audit 2026-04-23 §11 Q3 flagged
            # this semantic issue. Fix: scale w by (1 - current_V) so a
            # dominant primitive gets small updates, while an edge-case
            # primitive (briefly dominant during its moment) gets larger
            # updates. Preserves peer-outcome learning but de-biases toward
            # diversity when monoculture is active. Base weight kept at
            # 0.25 (1/4 of direct observation) — only scaling is new.
            base_w = 0.25
            w = base_w * max(0.05, 1.0 - float(p.V))  # floor 0.05 prevents zero-update
            p.alpha += w * outcome_01
            p.beta += w * (1.0 - outcome_01)
            p.recompute_derived()
            self._cgn_cross_insights_received += 1
        except Exception as e:
            swallow_warn('[EmotCGN] handle_incoming_cross_insight failed', e,
                         key="logic.emot_cgn.handle_incoming_cross_insight_failed", throttle=100)

    # ── Dual-authority V (Upgrade I) ──────────────────────────────

    def get_blended_V(self, primitive_id: str, ctx: Optional[dict] = None) -> float:
        """Blended V combining local β-posterior (authoritative) with
        shm-read CGN ValueNet V (shared learning across 8 consumers).

        V_final = α·β_V + (1-α)·shm_V    (default α=0.6, β dominates)
        Falls back to β_V if shm unavailable OR ctx missing.
        Used by get_current_emotion_state-like queries when consumers want
        the blended signal. Safe fallback in all failure modes.
        """
        p = self._primitives.get(primitive_id)
        beta_v = float(p.V) if p is not None else 0.5
        if self._cgn_client is None or ctx is None:
            return beta_v
        try:
            self._cgn_client._ensure_initialized()
            self._cgn_client._check_and_reload()
            if not self._cgn_client._value_net._loaded:
                return beta_v
            state_30d = self.encode_state_30d(primitive_id, ctx)
            shm_v = float(
                self._cgn_client._value_net.forward(
                    state_30d.reshape(1, -1))[0])
            shm_v = _clip01(shm_v)
            return (self._v_blend_alpha * beta_v
                    + (1.0 - self._v_blend_alpha) * shm_v)
        except Exception as e:
            swallow_warn('[EmotCGN] get_blended_V fallback to β_V', e,
                         key="logic.emot_cgn.get_blended_v_fallback_to_β_v", throttle=100)
            return beta_v

    # ── Gate (THE wire-now-gate-later API) ─────────────────────────

    def is_active(self) -> bool:
        """Return True if EMOT-CGN has graduated and consumers should
        incorporate its state. False during shadow / graduating / disabled.

        This is THE gate. Every consumer integration site checks this.
        """
        return self._status == "active"

    # ── Primary state-query API (consumers read these) ─────────────

    def get_current_emotion_state(self) -> dict:
        """Return 10D-encoded state suitable for META-CGN augmentation.

        Format (rFP improvement #4 — richer than single scalar):
          {
            "one_hot": [8 floats],       # 8D primitive one-hot (normalized)
            "intensity": float,           # 1D cluster-assignment confidence
            "confidence": float,          # 1D primitive V confidence
            "dominant": "FLOW",           # str for logging
            "dominant_V": float,          # gauge of how "good" the state is
            "active": bool,               # gate status (for caller awareness)
          }

        Failsafe: returns neutral state on any error (callers can ignore
        the `active` flag if they're already gated upstream).
        """
        try:
            dominant = self._dominant_emotion
            cluster_p, cluster_d, cluster_conf = self._last_cluster_assignment
            prim = self._primitives.get(dominant)
            one_hot = np.zeros(NUM_PRIMITIVES, dtype=np.float32)
            idx = EMOT_PRIMITIVE_INDEX.get(dominant, 0)
            one_hot[idx] = 1.0
            beta_v = float(prim.V) if prim else 0.5
            return {
                "one_hot": one_hot.tolist(),
                "intensity": float(cluster_conf),
                "confidence": float(prim.confidence) if prim else 0.0,
                "dominant": dominant,
                "dominant_V": beta_v,              # authoritative β posterior
                "dominant_V_beta": beta_v,         # explicit duplicate for clarity
                # Note: blended V (dual-authority) requires ctx — callers
                # with ctx should invoke get_blended_V() directly. This
                # endpoint returns local β-V for simplicity.
                "active": self.is_active(),
                "cgn_registered": bool(self._cgn_registered),
            }
        except Exception as e:
            swallow_warn('[EmotCGN] get_current_emotion_state failed', e,
                         key="logic.emot_cgn.get_current_emotion_state_failed", throttle=100)
            return {
                "one_hot": [0.0] * NUM_PRIMITIVES,
                "intensity": 0.0, "confidence": 0.0,
                "dominant": "FLOW", "dominant_V": 0.5,
                "dominant_V_beta": 0.5, "active": False,
                "cgn_registered": False,
            }

    def get_emotion_for_narration(self, recent_window_n: int = 5) -> str:
        """Return a short descriptor of emotional context for narrator
        coloring. Fallback to empty string if inactive (consumer pre-check
        also gates, but we're defensive)."""
        if not self.is_active():
            return ""
        try:
            cluster = self._clusterer.get_cluster(self._dominant_emotion)
            if cluster is None:
                return ""
            return cluster.label or cluster.primitive_id
        except Exception:
            return ""

    def get_dominant_emotion(self) -> str:
        """Return current dominant emotion primitive (always available,
        even pre-graduation — for logging / shadow-mode telemetry)."""
        return self._dominant_emotion

    # ── Felt-tensor handler (primary input) ────────────────────────

    def handle_felt_tensor(self, feature_vec, emit_bus_signal: bool = True) -> dict:
        """Process a 150D feature vector from spirit_worker (built via
        build_emot_feature_vec).

        Returns {primitive_id, distance, confidence}. Always safe — callers
        don't need to handle exceptions.
        """
        try:
            p_id, dist, conf = self._clusterer.observe(feature_vec)
            self._last_cluster_assignment = (p_id, dist, conf)
            self._dominant_emotion = p_id
            self._total_observations += 1

            if emit_bus_signal and self._send_queue is not None:
                self._emit_signal("cluster_assignment",
                                  intensity=conf,
                                  narrative_context={
                                      "primitive": p_id,
                                      "distance": round(dist, 3),
                                      "confidence": round(conf, 3),
                                  })

            # Check for cluster emergence (new signal)
            cluster = self._clusterer.get_cluster(p_id)
            if cluster and cluster.n_observations == self._clusterer._emergence_threshold:
                self._emit_signal("cluster_emerged", intensity=1.0,
                                  narrative_context={"primitive": p_id,
                                                      "label": cluster.label})

            return {"primitive": p_id, "distance": dist, "confidence": conf}
        except Exception as e:
            swallow_warn('[EmotCGN] handle_felt_tensor failed', e,
                         key="logic.emot_cgn.handle_felt_tensor_failed", throttle=100)
            return {"primitive": "FLOW", "distance": 0.0, "confidence": 0.0}

    def update_neuromod_ema(self, neuromods: dict) -> None:
        """Feed neuromod levels every epoch — we maintain EMA + delta to
        compute rate-of-change over ~100-epoch window (rFP improvement #2).

        neuromods: {"DA": float, "5HT": float, "NE": float, "ACh": float,
                    "Endorphin": float, "GABA": float}
        Unknown keys are ignored. Failsafe.
        """
        try:
            for key, val in neuromods.items():
                v = float(val)
                prev = self._neuromod_ema.get(key, v)
                self._neuromod_ema[key] = (
                    (1.0 - self._neuromod_ema_alpha) * prev
                    + self._neuromod_ema_alpha * v
                )
            self._neuromod_sample_count += 1
            # Every 100 samples, snapshot current EMA as "prev" for delta
            if self._neuromod_sample_count % 100 == 0:
                self._neuromod_prev_ema = dict(self._neuromod_ema)
        except Exception as e:
            swallow_warn('[EmotCGN] update_neuromod_ema failed', e,
                         key="logic.emot_cgn.update_neuromod_ema_failed", throttle=100)

    def get_neuromod_deltas(self) -> dict:
        """Return {name: delta} EMA(current) − EMA(100 epochs ago)."""
        deltas = {}
        for key, cur in self._neuromod_ema.items():
            prev = self._neuromod_prev_ema.get(key, cur)
            deltas[key] = float(cur - prev)
        return deltas

    def set_kin_resonance(self, resonance: float) -> None:
        """Optional hook — kin protocol may share emotional state across
        Titans. Default 0.0 until wired. Rate-limited callers welcome."""
        try:
            self._last_kin_resonance = _clip01(float(resonance) * 0.5 + 0.5)
            # store as clipped 0..1 (from -1..1 range)
        except Exception as _swallow_exc:
            swallow_warn('[logic.emot_cgn] EmotCGNConsumer.set_kin_resonance: self._last_kin_resonance = _clip01(float(resonance) * 0.5...', _swallow_exc,
                         key='logic.emot_cgn.EmotCGNConsumer.set_kin_resonance.line858', throttle=100)

    # ── Feature vector builder (for callers — builds the 150D input) ──

    def build_feature_vec(self, felt_tensor_130d,
                          terminal_reward_mean: Optional[float] = None,
                          sphere_clock_4d=None) -> np.ndarray:
        """Build the 150D feature vector from inputs spirit_worker has.

        felt_tensor_130d: the canonical 130D felt vector
        terminal_reward_mean: mean of last-5 terminal rewards (optional;
            uses internal _recent_rewards if None)
        sphere_clock_4d: 4D sphere-clock / π-heartbeat (optional; 0.5 default)

        Returns 150D numpy array. Failsafe — returns neutral vector on error.
        """
        try:
            v = np.full(CLUSTER_FEATURE_DIM, 0.5, dtype=np.float32)
            if felt_tensor_130d is not None and len(felt_tensor_130d) >= 130:
                v[:130] = np.asarray(felt_tensor_130d[:130], dtype=np.float32)
            # 6D neuromod EMA deltas
            deltas = self.get_neuromod_deltas()
            for i, key in enumerate(["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]):
                v[130 + i] = float(deltas.get(key, 0.0))
            # 8D cluster history one-hot
            hist = self._clusterer.get_cluster_history_onehot()
            v[136:144] = hist
            # 1D terminal reward recency
            if terminal_reward_mean is not None:
                v[144] = _clip01(float(terminal_reward_mean))
            elif self._recent_rewards:
                v[144] = _clip01(
                    sum(self._recent_rewards) / len(self._recent_rewards))
            else:
                v[144] = 0.5
            # 4D sphere-clock
            if sphere_clock_4d is not None and len(sphere_clock_4d) >= 4:
                for i in range(4):
                    v[145 + i] = _clip01(float(sphere_clock_4d[i]))
            # 1D kin resonance (always present — 0.0 default)
            v[149] = float(self._last_kin_resonance)
            return v
        except Exception as e:
            swallow_warn('[EmotCGN] build_feature_vec failed', e,
                         key="logic.emot_cgn.build_feature_vec_failed", throttle=100)
            return np.full(CLUSTER_FEATURE_DIM, 0.5, dtype=np.float32)

    # ── Chain-level evidence (terminal reward → V update) ──────────

    def observe_chain_evidence(self, chain_id: int,
                               dominant_at_start: str,
                               dominant_at_end: str,
                               terminal_reward: float,
                               ctx: Optional[dict] = None) -> None:
        """Update β posterior for the dominant emotion(s) based on chain
        terminal reward.

        Both start and end primitives get updated. If different, we also
        record a transition observation for H5 (emotion_drift_strategy).
        """
        try:
            if ctx is None:
                ctx = {}
            self._chain_counter += 1
            self._chains_since_graduation += (1 if self._status == "active" else 0)
            self._chains_since_save += 1

            tr = float(terminal_reward)
            self._recent_rewards.append(tr)

            # Route to rollback window if in grad/active mode
            if self._status in ("graduating", "active"):
                self._post_grad_rewards.append(tr)
            else:
                self._pre_grad_rewards.append(tr)

            # Update both start and end primitives (chain-level attribution)
            outcome_01 = _clip01(tr)  # normalize to [0, 1] for Beta update
            weight = 1.0
            # EUREKA-style boost if terminal_reward is very positive
            if tr > 0.75:
                weight = 1.5
            elif tr < 0.25:
                weight = 1.5  # strong negative signal also informative

            for prim_id in {dominant_at_start, dominant_at_end}:
                if prim_id not in self._primitives:
                    continue
                p = self._primitives[prim_id]
                # Beta update: α += w·outcome, β += w·(1-outcome)
                p.alpha += weight * outcome_01
                p.beta += weight * (1.0 - outcome_01)
                p.recompute_derived()
                p.last_updated_ts = time.time()
                p.last_updated_chain = self._chain_counter
                self._total_updates += 1

            # ── CGN_TRANSITION send (8th consumer, rFP §4) ────────
            # Send one transition per primitive touched so cgn_worker's
            # SharedValueNet sees both start + end primitive evidence.
            # Failsafe: client None OR queue None → skip silently without
            # incrementing counter (no silent lies about sends).
            if self._cgn_client is not None and self._send_queue is not None:
                for prim_id in {dominant_at_start, dominant_at_end}:
                    if prim_id not in self._primitives:
                        continue
                    try:
                        state_30d = self.encode_state_30d(prim_id, ctx)
                        action_idx = EMOT_PRIMITIVE_INDEX.get(prim_id, 0)
                        transition = {
                            "consumer": CGN_CONSUMER_NAME,
                            "concept_id": prim_id,
                            "state": state_30d.tolist(),
                            "action": action_idx,
                            "action_params": [0.0] * 4,  # unused for emot
                            "reward": float(outcome_01),
                            "timestamp": time.time(),
                            "epoch": int(ctx.get("epoch", 0)),
                            "metadata": {
                                "action_name": prim_id,
                                "chain_id": chain_id,
                                "encounter_type": (
                                    "chain_start" if prim_id == dominant_at_start
                                    else "chain_end"),
                            },
                        }
                        self._cgn_client.send_transition(transition)
                        self._cgn_transitions_sent += 1
                    except Exception as _cgn_err:
                        swallow_warn('[EmotCGN] send_transition failed', _cgn_err,
                                     key="logic.emot_cgn.send_transition_failed", throttle=100)

            # Track emotional dynamics for 30D encoding
            if dominant_at_end != self._dominant_emotion:
                self._dominant_emotion_start_ts = time.time()
                self._dominant_emotion_transitions.append(time.time())
            self._emotion_history_100.append(dominant_at_end)

            # ── Outgoing cross-consumer insight (Upgrade III) ────
            # Emit CGN_CROSS_INSIGHT when emotional state produces an
            # informative signal other consumers can learn from.
            # Rate-limited to 0.2 Hz to avoid bus flood.
            self._maybe_emit_cross_insight(dominant_at_start, dominant_at_end,
                                            tr, ctx)

            # Shadow-mode log
            if self._status == "shadow_mode":
                self._shadow_log(chain_id, dominant_at_start, dominant_at_end,
                                 tr, ctx)

            # Accumulate HAOV observations
            self._accumulate_haov(dominant_at_start, dominant_at_end, tr, ctx)

            # Test hypotheses on cadence
            self._evidence_since_last_test += 1
            if self._evidence_since_last_test >= self._test_cadence:
                self._evidence_since_last_test = 0
                self._test_hypotheses()

            # Check graduation & rollback
            self._check_graduation_and_rollback()

            # Pick up operator-override flags (written by API endpoints)
            self._check_override_flags()

            # Periodic save (cadence config-driven — default 5 chains)
            if self._chains_since_save >= self._save_cadence_chains:
                self._chains_since_save = 0
                self.save_state()
        except Exception as e:
            logger.warning("[EmotCGN] observe_chain_evidence failed: %s", e)

    # ── HAOV accumulation + testing ────────────────────────────────

    def _accumulate_haov(self, p_start: str, p_end: str,
                         terminal_reward: float, ctx: dict) -> None:
        """Append observations to each hypothesis's rolling buffer."""
        try:
            obs_ts = time.time()
            # H1: flow_neuromod — observation is (p_start, DA, 5HT)
            self._hypotheses["H1_flow_neuromod"].observations.append({
                "ts": obs_ts, "primitive": p_start,
                "DA": ctx.get("DA", 0.5), "5HT": ctx.get("5HT", 0.5),
            })
            # H2: eureka_sequence — look at cluster history before
            hist = list(self._clusterer._recent_assignments)
            self._hypotheses["H2_eureka_sequence"].observations.append({
                "ts": obs_ts, "end": p_end, "hist": hist[-3:],
            })
            # H3: grief_persistence — tracks how many chains back p_start
            # has been the dominant
            self._hypotheses["H3_grief_persistence"].observations.append({
                "ts": obs_ts, "primitive": p_start,
            })
            # H4: serenity_introspect — does PEACE chain include SPIRIT_SELF/INTROSPECT?
            if p_start == "PEACE":
                self._hypotheses["H4_serenity_introspect"].observations.append({
                    "ts": obs_ts,
                    "spirit_self_share": ctx.get("spirit_self_share", 0.0),
                    "introspect_share": ctx.get("introspect_share", 0.0),
                })
            # H5: emotion_drift_strategy — transition observed
            if p_start != p_end:
                self._hypotheses["H5_emotion_drift_strategy"].observations.append({
                    "ts": obs_ts, "from": p_start, "to": p_end,
                    "strategy_shift": bool(ctx.get("strategy_shift", False)),
                })
            # H6: curiosity_drives_knowledge — is CURIOSITY active during
            # rising knowledge acquisition?
            if p_start == "CURIOSITY":
                self._hypotheses["H6_curiosity_drives_knowledge"].observations.append({
                    "ts": obs_ts,
                    "knowledge_acq_rate": ctx.get("knowledge_acq_rate", 0.0),
                })
            # H7: impasse_v_above_resolution — on incomplete chains
            if ctx.get("incomplete", False):
                v_imp = self._primitives["IMPASSE_TENSION"].V
                v_res = self._primitives["RESOLUTION"].V
                self._hypotheses["H7_impasse_v_above_resolution"].observations.append({
                    "ts": obs_ts, "v_impasse": v_imp, "v_resolution": v_res,
                })
            # H8: curiosity_precedes_knowledge_growth — use latency from
            # recent curiosity to knowledge growth
            if p_start == "CURIOSITY" or p_end == "CURIOSITY":
                self._hypotheses["H8_curiosity_precedes_knowledge_growth"].observations.append({
                    "ts": obs_ts,
                    "curious": True,
                    "knowledge_growth_ahead": bool(
                        ctx.get("knowledge_growth_ahead", False)),
                })
        except Exception as e:
            swallow_warn('[EmotCGN] _accumulate_haov failed', e,
                         key="logic.emot_cgn.accumulate_haov_failed", throttle=100)

    def _test_hypotheses(self) -> None:
        """Run each hypothesis test if min_samples met. Update status."""
        try:
            for h_id, h in self._hypotheses.items():
                if h.status == "confirmed":
                    continue
                if len(h.observations) < h.min_samples:
                    continue
                h.test_count += 1
                h.last_test_ts = time.time()
                test_fn = getattr(self, "_test_" + h.test_kind, None)
                if test_fn is None:
                    continue
                try:
                    effect, passed = test_fn(h)
                    h.effect_size = float(effect)
                    if passed and abs(effect) >= h.confirmation_threshold:
                        if h.status != "confirmed":
                            logger.info(
                                "[EmotCGN] Hypothesis CONFIRMED: %s "
                                "(effect=%.3f, n=%d)",
                                h_id, effect, len(h.observations))
                        h.status = "confirmed"
                    elif h.test_count >= 5 and abs(effect) < 0.02:
                        h.status = "falsified"
                    else:
                        h.status = "testing"
                except Exception as _e:
                    swallow_warn(f"[EmotCGN] test '{h_id}' error", _e,
                                 key="logic.emot_cgn.test_error", throttle=100)
        except Exception as e:
            logger.warning("[EmotCGN] _test_hypotheses failed: %s", e)

    # Test functions — return (effect_size, passed_sanity_check)
    def _test_flow_neuromod(self, h) -> tuple[float, bool]:
        flow_obs = [o for o in h.observations if o["primitive"] == "FLOW"]
        other_obs = [o for o in h.observations if o["primitive"] != "FLOW"]
        if len(flow_obs) < 5 or len(other_obs) < 5:
            return (0.0, False)
        flow_neuro = sum(float(o["DA"]) + float(o["5HT"])
                         for o in flow_obs) / len(flow_obs)
        other_neuro = sum(float(o["DA"]) + float(o["5HT"])
                          for o in other_obs) / len(other_obs)
        return (flow_neuro - other_neuro, True)

    def _test_eureka_sequence(self, h) -> tuple[float, bool]:
        resolution_ends = [o for o in h.observations
                           if o.get("end") == "RESOLUTION"]
        if len(resolution_ends) < 5:
            return (0.0, False)
        preceded = sum(1 for o in resolution_ends
                       if "IMPASSE_TENSION" in (o.get("hist") or []))
        share = preceded / max(1, len(resolution_ends))
        # Baseline 1/8 = 0.125; effect = share − baseline
        return (share - 0.125, True)

    def _test_grief_persistence(self, h) -> tuple[float, bool]:
        obs = list(h.observations)
        if len(obs) < 5:
            return (0.0, False)
        # Count longest run for each primitive
        runs: dict = {}
        current = None
        run_len = 0
        best_by_prim: dict = {p: 0 for p in EMOT_PRIMITIVES}
        for o in obs:
            p = o.get("primitive")
            if p == current:
                run_len += 1
            else:
                if current:
                    best_by_prim[current] = max(best_by_prim[current], run_len)
                current = p
                run_len = 1
        if current:
            best_by_prim[current] = max(best_by_prim[current], run_len)
        grief_run = best_by_prim.get("GRIEF", 0)
        others_mean = (sum(v for k, v in best_by_prim.items() if k != "GRIEF")
                       / max(1, len(best_by_prim) - 1))
        return (grief_run - others_mean, True)

    def _test_serenity_introspect(self, h) -> tuple[float, bool]:
        if len(h.observations) < 5:
            return (0.0, False)
        ss_mean = sum(float(o.get("spirit_self_share", 0))
                      for o in h.observations) / max(1, len(h.observations))
        intro_mean = sum(float(o.get("introspect_share", 0))
                         for o in h.observations) / max(1, len(h.observations))
        combined = ss_mean + intro_mean
        # Baseline share of SPIRIT_SELF+INTROSPECT = 2/9 ≈ 0.22
        return (combined - 0.22, True)

    def _test_emotion_drift_strategy(self, h) -> tuple[float, bool]:
        if len(h.observations) < 5:
            return (0.0, False)
        shifts = sum(1 for o in h.observations if o.get("strategy_shift"))
        share = shifts / max(1, len(h.observations))
        # Baseline 0.1 — if drift is random, low chance of correlated shift
        return (share - 0.1, True)

    def _test_curiosity_drives_knowledge(self, h) -> tuple[float, bool]:
        if len(h.observations) < 5:
            return (0.0, False)
        k_rate = sum(float(o.get("knowledge_acq_rate", 0))
                     for o in h.observations) / max(1, len(h.observations))
        # Baseline 0.1 (knowledge acq rate neutral)
        return (k_rate - 0.1, True)

    def _test_impasse_v_above_resolution(self, h) -> tuple[float, bool]:
        if len(h.observations) < 5:
            return (0.0, False)
        v_imp = sum(float(o["v_impasse"]) for o in h.observations) / len(h.observations)
        v_res = sum(float(o["v_resolution"]) for o in h.observations) / len(h.observations)
        return (v_imp - v_res, True)

    def _test_curiosity_precedes_knowledge_growth(self, h) -> tuple[float, bool]:
        if len(h.observations) < 5:
            return (0.0, False)
        total = len(h.observations)
        growth_ahead = sum(1 for o in h.observations if o.get("knowledge_growth_ahead"))
        share = growth_ahead / max(1, total)
        # Baseline 0.2 — random alignment chance
        return (share - 0.2, True)

    # ── Graduation + rollback state machine ────────────────────────

    def _check_graduation_and_rollback(self) -> None:
        """Evaluate graduation criteria (rFP §6) or rollback detector."""
        try:
            if self._status == "shadow_mode":
                if self._graduation_progress == 0:
                    # Eligibility gate
                    readiness = self.graduation_readiness()
                    if readiness.get("eligible", False):
                        # Capture pre-grad baseline
                        if self._pre_grad_rewards:
                            mean = sum(self._pre_grad_rewards) / len(self._pre_grad_rewards)
                            std = float(np.std(list(self._pre_grad_rewards))) or self._rollback_min_std_floor
                            self._pre_graduation_baseline = {
                                "mean": mean, "std": std,
                                "n": len(self._pre_grad_rewards),
                                "ts": time.time(),
                            }
                        self._status = "graduating"
                        self._graduation_ts = time.time()
                        self._graduation_progress = 1
                        logger.info("[EmotCGN] Graduation started (shadow → graduating) "
                                    "— readiness: %s", readiness)
                        self._emit_signal("graduation_transition",
                                          intensity=1.0,
                                          narrative_context={"to": "graduating"})
            elif self._status == "graduating":
                # Linear ramp over 100 chains
                self._graduation_progress = min(
                    100, self._graduation_progress + 1)
                if self._graduation_progress >= 100:
                    self._status = "active"
                    logger.info("[EmotCGN] Graduation complete (graduating → active)")
                    self._emit_signal("graduation_transition",
                                      intensity=1.0,
                                      narrative_context={"to": "active"})
            elif self._status == "active":
                # Rollback detector: post-grad mean below baseline_mean − k·σ?
                if len(self._post_grad_rewards) >= 30 and self._pre_graduation_baseline:
                    post_mean = sum(self._post_grad_rewards) / len(self._post_grad_rewards)
                    baseline_mean = self._pre_graduation_baseline["mean"]
                    baseline_std = max(self._rollback_min_std_floor,
                                       self._pre_graduation_baseline["std"])
                    threshold = baseline_mean - self._rollback_sigma_k * baseline_std
                    if post_mean < threshold:
                        self._status = "shadow_mode"
                        self._rolled_back_count += 1
                        self._graduation_progress = 0
                        logger.warning(
                            "[EmotCGN] Rollback fired: post_mean=%.3f < "
                            "threshold=%.3f (baseline_mean=%.3f, k·σ=%.3f)",
                            post_mean, threshold, baseline_mean,
                            self._rollback_sigma_k * baseline_std)
                        self._emit_signal("rollback_fired",
                                          intensity=1.0,
                                          narrative_context={
                                              "post_mean": round(post_mean, 3),
                                              "threshold": round(threshold, 3),
                                          })
        except Exception as e:
            logger.warning("[EmotCGN] _check_graduation_and_rollback failed: %s", e)

    def graduation_readiness(self) -> dict:
        """Evaluate each graduation criterion. Returns dict with per-
        criterion status + overall eligible flag."""
        try:
            updates_ok = self._total_updates >= self._grad_min_updates
            confirmed = sum(1 for h in self._hypotheses.values()
                            if h.status == "confirmed")
            hypotheses_ok = confirmed >= self._grad_min_confirmed
            mature_prims = sum(
                1 for p in self._primitives.values()
                if p.n_samples >= self._grad_min_samples_per
                and p.confidence >= self._grad_min_confidence)
            primitives_ok = mature_prims >= self._grad_min_primitives
            # Cross-primitive contrast
            v_values = sorted((p.V for p in self._primitives.values()),
                              reverse=True)
            contrast_ok = len(v_values) >= 2 and (v_values[0] - v_values[-1]) >= self._grad_contrast_v_gap
            # No recent rollback (in active mode — here we just require
            # not currently rolled back)
            no_rollback = self._status in ("shadow_mode", "graduating", "active")
            # Observation window (real-time)
            elapsed = time.time() - self._creation_ts
            window_ok = elapsed >= self._grad_observation_window_s
            eligible = (updates_ok and hypotheses_ok and primitives_ok
                        and contrast_ok and no_rollback and window_ok)
            return {
                "eligible": bool(eligible),
                "total_updates": self._total_updates,
                "updates_required": self._grad_min_updates,
                "updates_ok": bool(updates_ok),
                "confirmed_hypotheses": confirmed,
                "hypotheses_required": self._grad_min_confirmed,
                "hypotheses_ok": bool(hypotheses_ok),
                "mature_primitives": mature_prims,
                "primitives_required": self._grad_min_primitives,
                "primitives_ok": bool(primitives_ok),
                "v_contrast": round(v_values[0] - v_values[-1], 3) if v_values else 0.0,
                "contrast_required": self._grad_contrast_v_gap,
                "contrast_ok": bool(contrast_ok),
                "window_elapsed_s": round(elapsed, 0),
                "window_required_s": self._grad_observation_window_s,
                "window_ok": bool(window_ok),
                "rolled_back_count": self._rolled_back_count,
                "status": self._status,
                "graduation_progress": self._graduation_progress,
            }
        except Exception as e:
            swallow_warn('[EmotCGN] graduation_readiness failed', e,
                         key="logic.emot_cgn.graduation_readiness_failed", throttle=100)
            return {"eligible": False, "status": self._status}

    def _check_override_flags(self) -> None:
        """Pick up pending operator overrides written by API endpoints.

        API endpoints write flag files to data/emot_cgn/ because they run
        in a different process from meta_reasoning. This method checks for
        them on each chain conclude + applies + deletes the flag.
        """
        try:
            grad_flag = os.path.join(self._save_dir,
                                     "_pending_force_graduate.flag")
            shadow_flag = os.path.join(self._save_dir,
                                       "_pending_force_shadow.flag")
            if os.path.exists(grad_flag):
                self.force_graduate()
                try:
                    os.remove(grad_flag)
                except Exception as _swallow_exc:
                    swallow_warn('[logic.emot_cgn] EmotCGNConsumer._check_override_flags: os.remove(grad_flag)', _swallow_exc,
                                 key='logic.emot_cgn.EmotCGNConsumer._check_override_flags.line1347', throttle=100)
            if os.path.exists(shadow_flag):
                self.force_shadow()
                try:
                    os.remove(shadow_flag)
                except Exception as _swallow_exc:
                    swallow_warn('[logic.emot_cgn] EmotCGNConsumer._check_override_flags: os.remove(shadow_flag)', _swallow_exc,
                                 key='logic.emot_cgn.EmotCGNConsumer._check_override_flags.line1353', throttle=100)
        except Exception as e:
            swallow_warn('[EmotCGN] _check_override_flags failed', e,
                         key="logic.emot_cgn.check_override_flags_failed", throttle=100)

    def force_graduate(self) -> bool:
        """Operator override: force graduation regardless of criteria.
        Use only via /v4/emot-cgn/force-graduate endpoint."""
        try:
            if self._status == "active":
                return False
            if self._pre_grad_rewards:
                mean = sum(self._pre_grad_rewards) / len(self._pre_grad_rewards)
                std = float(np.std(list(self._pre_grad_rewards))) or self._rollback_min_std_floor
                self._pre_graduation_baseline = {
                    "mean": mean, "std": std,
                    "n": len(self._pre_grad_rewards),
                    "ts": time.time(),
                }
            self._status = "active"
            self._graduation_progress = 100
            self._graduation_ts = time.time()
            logger.warning("[EmotCGN] FORCE-GRADUATE invoked by operator")
            return True
        except Exception as e:
            logger.warning("[EmotCGN] force_graduate failed: %s", e)
            return False

    def force_shadow(self) -> bool:
        """Operator override: force back to shadow mode."""
        try:
            if self._status == "shadow_mode":
                return False
            self._status = "shadow_mode"
            self._graduation_progress = 0
            self._rolled_back_count += 1
            logger.warning("[EmotCGN] FORCE-SHADOW invoked by operator")
            return True
        except Exception as e:
            logger.warning("[EmotCGN] force_shadow failed: %s", e)
            return False

    # ── Signal emission (bus producer) ─────────────────────────────

    def _emit_signal(self, event_type: str, intensity: float = 1.0,
                     narrative_context: Optional[dict] = None) -> bool:
        """Emit an EMOT_CGN_SIGNAL via the shared helper in bus.py.

        Uses emit_emot_cgn_signal which enforces the same orphan-detection
        + rate-gate invariants as emit_meta_cgn_signal.
        """
        if self._send_queue is None:
            return False
        try:
            from titan_plugin.bus import emit_emot_cgn_signal
            return emit_emot_cgn_signal(
                self._send_queue, self._module_name,
                consumer=CONSUMER_NAME,
                event_type=event_type,
                intensity=intensity,
                narrative_context=narrative_context,
                reason="emot_cgn_state_emit",
                min_interval_s=0.5,
            )
        except Exception as e:
            from titan_plugin.utils.silent_swallow import swallow_warn
            swallow_warn("[EmotCGN] _emit_signal failed", e,
                         key="emot_cgn.emit_signal")
            return False

    # ── Shadow-mode log (pre-graduation observability) ────────────

    def _shadow_log(self, chain_id: int, p_start: str, p_end: str,
                    tr: float, ctx: dict) -> None:
        try:
            line = {
                "ts": time.time(),
                "chain": chain_id,
                "start": p_start, "end": p_end,
                "reward": round(tr, 3),
                "cluster_dist": round(self._last_cluster_assignment[1], 3),
                "dominant": self._dominant_emotion,
            }
            with open(self._shadow_log_path, "a") as f:
                f.write(json.dumps(line) + "\n")
        except Exception as e:
            swallow_warn('[EmotCGN] shadow_log failed', e,
                         key="logic.emot_cgn.shadow_log_failed", throttle=100)

    # ── Kin Protocol export / import ───────────────────────────────

    def export_kin_snapshot(self) -> dict:
        """Return signable snapshot for cross-Titan priors sharing.
        Mirrors MetaCGNConsumer.export_kin_snapshot."""
        try:
            for p in self._primitives.values():
                p.recompute_derived()
            return {
                "schema": "emot_cgn_snapshot_v1",
                "kin_protocol_version": 1,
                "titan_id": self._titan_id,
                "exported_ts": time.time(),
                "primitives": {
                    p_id: {
                        "V": p.V,
                        "confidence": p.confidence,
                        "n_samples": p.n_samples,
                    }
                    for p_id, p in self._primitives.items()
                },
                "status": self._status,
            }
        except Exception as e:
            logger.warning("[EmotCGN] export_kin_snapshot failed: %s", e)
            return {}

    def import_kin_snapshot(self, snapshot: dict,
                             confidence_scale: float = 0.5) -> int:
        """Import peer Titan's kin snapshot as priors (NOT overrides).

        Treats peer α,β as soft priors, scaled by confidence_scale.
        """
        try:
            if not snapshot:
                return 0
            peer_titan = str(snapshot.get("titan_id", "unknown"))
            count = 0
            for p_id, peer in snapshot.get("primitives", {}).items():
                if p_id not in self._primitives:
                    continue
                v_peer = float(peer.get("V", 0.5))
                n_peer = int(peer.get("n_samples", 0))
                if n_peer < 10:
                    continue  # too weak
                n_eff = min(50, int(n_peer * confidence_scale))
                a_add = v_peer * n_eff
                b_add = (1.0 - v_peer) * n_eff
                self._primitives[p_id].alpha += a_add
                self._primitives[p_id].beta += b_add
                self._primitives[p_id].recompute_derived()
                count += 1
            logger.info("[EmotCGN] Imported kin snapshot from %s: %d prims",
                        peer_titan, count)
            return count
        except Exception as e:
            logger.warning("[EmotCGN] import_kin_snapshot failed: %s", e)
            return 0

    # ── Persistence ────────────────────────────────────────────────

    def save_state(self) -> None:
        try:
            for p in self._primitives.values():
                p.recompute_derived()
            data = {
                "version": 1,
                "titan_id": self._titan_id,
                "saved_ts": time.time(),
                "primitives": {
                    p: asdict(c) for p, c in self._primitives.items()
                },
                "total_updates": self._total_updates,
                "total_observations": self._total_observations,
                "chain_counter": self._chain_counter,
                "creation_ts": self._creation_ts,
            }
            tmp = self._grounding_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._grounding_path)

            # HAOV
            haov_data = {
                "version": 1,
                "saved_ts": time.time(),
                "hypotheses": {
                    h_id: {
                        "hypothesis_id": h.hypothesis_id,
                        "description": h.description,
                        "test_kind": h.test_kind,
                        "status": h.status,
                        "test_count": h.test_count,
                        "effect_size": h.effect_size,
                        "observations": list(h.observations),
                    }
                    for h_id, h in self._hypotheses.items()
                },
            }
            haov_tmp = self._haov_path + ".tmp"
            with open(haov_tmp, "w") as f:
                json.dump(haov_data, f, indent=2)
            os.replace(haov_tmp, self._haov_path)

            # Watchdog
            self._save_watchdog_state()

            # Clusterer
            self._clusterer.save_state()
        except Exception as e:
            logger.warning("[EmotCGN] save_state failed: %s", e)

    def _save_watchdog_state(self) -> None:
        try:
            # Graduation + rollback state (always persist — correctness-critical)
            # + feature-continuity state (persist to avoid post-restart degradation
            # of the feature vector for first ~100 epochs / 5 chains until EMAs
            # rebuild). See persistence audit 2026-04-20.
            data = {
                "status": self._status,
                "graduation_progress": self._graduation_progress,
                "graduation_ts": self._graduation_ts,
                "pre_graduation_baseline": self._pre_graduation_baseline,
                "chains_since_graduation": self._chains_since_graduation,
                "rolled_back_count": self._rolled_back_count,
                "pre_grad_rewards": list(self._pre_grad_rewards),
                "post_grad_rewards": list(self._post_grad_rewards),
                # Feature-continuity attrs (persisted for restart-safety of
                # EMA-based rate-of-change + recency slots in build_feature_vec)
                "neuromod_ema": dict(self._neuromod_ema),
                "neuromod_prev_ema": dict(self._neuromod_prev_ema),
                "neuromod_sample_count": self._neuromod_sample_count,
                "recent_rewards": list(self._recent_rewards),
                "dominant_emotion": self._dominant_emotion,
                "last_kin_resonance": float(self._last_kin_resonance),
                "evidence_since_last_test": self._evidence_since_last_test,
                "chains_since_save": self._chains_since_save,
                "saved_ts": time.time(),
            }
            tmp = self._watchdog_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._watchdog_path)
        except Exception as e:
            logger.warning("[EmotCGN] _save_watchdog_state failed: %s", e)

    def _load_state(self) -> None:
        try:
            if not os.path.exists(self._grounding_path):
                return
            with open(self._grounding_path) as f:
                data = json.load(f)
            for p_id, p_data in data.get("primitives", {}).items():
                if p_id not in self._primitives:
                    continue
                allowed = {"primitive_id", "alpha", "beta", "V", "confidence",
                           "n_samples", "variance", "last_updated_ts",
                           "last_updated_chain", "haov_rules"}
                filtered = {k: v for k, v in p_data.items() if k in allowed}
                concept = EmotPrimitive(**filtered)
                concept.recompute_derived()
                self._primitives[p_id] = concept
            self._total_updates = int(data.get("total_updates", 0))
            self._total_observations = int(data.get("total_observations", 0))
            self._chain_counter = int(data.get("chain_counter", 0))
            saved_ts = float(data.get("creation_ts", self._creation_ts))
            # Keep earliest creation_ts across restarts so observation_window
            # gate measures true elapsed since first boot.
            if saved_ts > 0:
                self._creation_ts = min(self._creation_ts, saved_ts)
            logger.info("[EmotCGN] Loaded grounding from %s (primitives=%d, "
                        "updates=%d)",
                        self._grounding_path, len(self._primitives),
                        self._total_updates)
        except Exception as e:
            logger.warning("[EmotCGN] _load_state failed: %s", e)
        # HAOV load
        try:
            if not os.path.exists(self._haov_path):
                return
            with open(self._haov_path) as f:
                haov = json.load(f)
            for h_id, h_saved in haov.get("hypotheses", {}).items():
                if h_id in self._hypotheses:
                    h = self._hypotheses[h_id]
                    h.status = h_saved.get("status", h.status)
                    h.test_count = int(h_saved.get("test_count", 0))
                    h.effect_size = float(h_saved.get("effect_size", 0.0))
                    saved_obs = h_saved.get("observations", [])
                    h.observations = deque(saved_obs,
                                           maxlen=h.observations.maxlen)
        except Exception as e:
            swallow_warn('[EmotCGN] HAOV _load_state failed', e,
                         key="logic.emot_cgn.haov_load_state_failed", throttle=100)

    def _load_watchdog_state(self) -> None:
        try:
            if not os.path.exists(self._watchdog_path):
                return
            with open(self._watchdog_path) as f:
                data = json.load(f)
            self._status = data.get("status", "shadow_mode")
            self._graduation_progress = int(data.get("graduation_progress", 0))
            self._graduation_ts = float(data.get("graduation_ts", 0.0))
            self._pre_graduation_baseline = data.get("pre_graduation_baseline", {})
            self._chains_since_graduation = int(data.get("chains_since_graduation", 0))
            self._rolled_back_count = int(data.get("rolled_back_count", 0))
            pre_rewards = data.get("pre_grad_rewards", [])
            post_rewards = data.get("post_grad_rewards", [])
            self._pre_grad_rewards = deque(pre_rewards, maxlen=100)
            self._post_grad_rewards = deque(post_rewards, maxlen=50)
            # Feature-continuity attrs (restart-safety for build_feature_vec)
            self._neuromod_ema = dict(data.get("neuromod_ema", {}) or {})
            self._neuromod_prev_ema = dict(data.get("neuromod_prev_ema", {}) or {})
            self._neuromod_sample_count = int(data.get("neuromod_sample_count", 0))
            self._recent_rewards = deque(
                data.get("recent_rewards", []) or [], maxlen=5)
            self._dominant_emotion = str(
                data.get("dominant_emotion", "FLOW"))
            self._last_kin_resonance = float(
                data.get("last_kin_resonance", 0.0))
            self._evidence_since_last_test = int(
                data.get("evidence_since_last_test", 0))
            self._chains_since_save = int(
                data.get("chains_since_save", 0))
        except Exception as e:
            swallow_warn('[EmotCGN] _load_watchdog_state failed', e,
                         key="logic.emot_cgn.load_watchdog_state_failed", throttle=100)

    # ── Introspection ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "status": self._status,
            "dominant_emotion": self._dominant_emotion,
            "last_cluster_assignment": {
                "primitive": self._last_cluster_assignment[0],
                "distance": round(self._last_cluster_assignment[1], 3),
                "confidence": round(self._last_cluster_assignment[2], 3),
            },
            "total_updates": self._total_updates,
            "total_observations": self._total_observations,
            "chain_counter": self._chain_counter,
            "graduation_progress": self._graduation_progress,
            "rolled_back_count": self._rolled_back_count,
            "primitives": {
                p_id: {
                    "V": round(p.V, 3),
                    "confidence": round(p.confidence, 3),
                    "n_samples": p.n_samples,
                    "alpha": round(p.alpha, 2),
                    "beta": round(p.beta, 2),
                }
                for p_id, p in self._primitives.items()
            },
            "hypotheses": {
                h_id: {
                    "status": h.status,
                    "effect_size": round(h.effect_size, 3),
                    "test_count": h.test_count,
                    "observation_count": len(h.observations),
                }
                for h_id, h in self._hypotheses.items()
            },
            "clusterer": self._clusterer.get_summary(),
            "cgn_integration": {
                "registered": bool(self._cgn_registered),
                "consumer_name": CGN_CONSUMER_NAME,
                "feature_dims": FEATURE_DIMS,
                "action_dims": ACTION_DIMS,
                "transitions_sent": self._cgn_transitions_sent,
                "cross_insights_sent": self._cgn_cross_insights_sent,
                "cross_insights_received": self._cgn_cross_insights_received,
                "shm_available": bool(
                    self._cgn_client is not None
                    and getattr(self._cgn_client, "_initialized", False)),
                "v_blend_alpha": self._v_blend_alpha,
            },
            "creation_ts": self._creation_ts,
            "elapsed_s": round(time.time() - self._creation_ts, 0),
        }

    def get_stats_compact(self) -> dict:
        return {
            "status": self._status,
            "dominant": self._dominant_emotion,
            "total_updates": self._total_updates,
            "total_observations": self._total_observations,
            "confirmed_hypotheses": sum(
                1 for h in self._hypotheses.values()
                if h.status == "confirmed"),
            "rolled_back": self._rolled_back_count,
        }
