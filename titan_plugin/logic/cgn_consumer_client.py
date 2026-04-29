"""
CGN Consumer Client — Lightweight LOCAL inference with shared weights.

Each consumer process uses this client for fast ground() calls (0.5ms local
forward pass via pure numpy) while sending transitions to the CGN Worker via bus.

Weight synchronization via /dev/shm: checks 4-byte version counter before
each inference. If weights updated, reloads (~0.1ms from RAM).

TORCH-FREE (2026-04-11):
    This client does NOT import torch. The SharedValueNet and ConsumerActionNet
    are tiny networks (~2,800 and ~1,200 params). Forward passes are pure numpy
    (matmul + ReLU + LayerNorm + softmax). This saves ~204MB per consumer
    process — critical for T2/T3 where language_worker and knowledge_worker
    were each loading torch unnecessarily.

    Weight deserialization uses _bytes_to_numpy_dict() from cgn_shm_protocol.py
    which also avoids torch. The SHM binary format is shared with the CGN Worker
    (which still uses torch for training — that's expected and correct).

Usage:
    client = CGNConsumerClient("language", send_queue, "language")
    action, transition = client.ground(concept, sensory_ctx)
    client.send_transition(transition)  # non-blocking bus send
"""
from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)


@dataclass
class LocalGroundingResult:
    """Result from local ground() call + transition data for bus."""
    action_index: int = 0
    action_name: str = "reinforce"
    confidence_delta: float = 0.0
    tensor_plasticity: float = 0.0
    association_deltas: dict = field(default_factory=dict)
    context_weight: float = 0.5
    # Transition data to send to CGN Worker
    transition: dict = field(default_factory=dict)


# ── Numpy-only inference nets (torch-free) ───────────────────────────────
# SHM_SOURCED: NumpyValueNet + NumpyActionNet weights are loaded from
# /dev/shm/cgn_live_weights.bin (written by the CGN worker process).
# The client is a pure reader — it never persists weights to on-disk state,
# so dead-wiring's persistence-asymmetry check is suppressed for these classes.

class NumpyValueNet:
    """Pure numpy V(s) — mirrors SharedValueNet (30→48→LN→24→1)."""

    def __init__(self):
        self._loaded = False

    def load_state_dict(self, sd: dict):
        """Load from numpy state_dict (keys match torch SharedValueNet)."""
        self._w0 = sd["net.0.weight"]    # (48, 30)
        self._b0 = sd["net.0.bias"]      # (48,)
        self._ln_w = sd["net.1.weight"]  # (48,)
        self._ln_b = sd["net.1.bias"]    # (48,)
        self._w3 = sd["net.3.weight"]    # (24, 48)
        self._b3 = sd["net.3.bias"]      # (24,)
        self._w5 = sd["net.5.weight"]    # (1, 24)
        self._b5 = sd["net.5.bias"]      # (1,)
        self._loaded = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, 30) → (batch,) value estimate."""
        h = x @ self._w0.T + self._b0         # (batch, 48)
        # LayerNorm
        mean = h.mean(axis=-1, keepdims=True)
        var = h.var(axis=-1, keepdims=True)
        h = (h - mean) / np.sqrt(var + 1e-5) * self._ln_w + self._ln_b
        h = np.maximum(0, h)                   # ReLU
        h = h @ self._w3.T + self._b3         # (batch, 24)
        h = np.maximum(0, h)                   # ReLU
        h = h @ self._w5.T + self._b5         # (batch, 1)
        return h.squeeze(-1)


# SHM_SOURCED: weights loaded from /dev/shm (see NumpyValueNet header).
class NumpyActionNet:
    """Pure numpy Q(s,a) — mirrors ConsumerActionNet (30→24→12→[action,param])."""

    def __init__(self):
        self._loaded = False
        self.action_dims = 0

    def load_state_dict(self, sd: dict):
        """Load from numpy state_dict (keys match torch ConsumerActionNet)."""
        self._bb_w0 = sd["backbone.0.weight"]  # (24, 30)
        self._bb_b0 = sd["backbone.0.bias"]    # (24,)
        self._bb_w2 = sd["backbone.2.weight"]  # (12, 24)
        self._bb_b2 = sd["backbone.2.bias"]    # (12,)
        self._ah_w = sd["action_head.weight"]  # (action_dims, 12)
        self._ah_b = sd["action_head.bias"]    # (action_dims,)
        self._ph_w = sd["param_head.weight"]   # (4, 12)
        self._ph_b = sd["param_head.bias"]     # (4,)
        self.action_dims = self._ah_b.shape[0]
        self._loaded = True

    def forward(self, x: np.ndarray):
        """x: (batch, 30) → (logits: (batch, A), params: (batch, 4))."""
        h = x @ self._bb_w0.T + self._bb_b0   # (batch, 24)
        h = np.maximum(0, h)                    # ReLU
        h = h @ self._bb_w2.T + self._bb_b2   # (batch, 12)
        h = np.maximum(0, h)                    # ReLU
        logits = h @ self._ah_w.T + self._ah_b  # (batch, action_dims)
        # sigmoid for params
        raw = h @ self._ph_w.T + self._ph_b
        params = 1.0 / (1.0 + np.exp(-np.clip(raw, -20, 20)))  # (batch, 4)
        return logits, params


# ── Numpy softmax / multinomial helpers ──────────────────────────────────

def _np_softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    x = x / temperature
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _np_multinomial(probs: np.ndarray) -> int:
    """Sample one index from 1D probability vector."""
    return int(np.random.choice(len(probs), p=probs))


# ── Main client ──────────────────────────────────────────────────────────

# SHM_SOURCED: all weights + config loaded from /dev/shm via ShmWeightReader.
# Upgrade III peer publishing — module-level rate state (audit 2026-04-23 Q2).
# Workers that emit CGN_CROSS_INSIGHT without a CGNConsumerClient instance
# (language_worker, knowledge_worker emit via raw _send_msg) share this
# per-consumer rate gate. Keyed by consumer_name so each peer has its own
# 5s rate window.
_PEER_CROSS_INSIGHT_RATE_STATE: Dict[str, float] = {}


def emit_chain_outcome_insight(send_queue, src_module: str,
                                consumer_name: str,
                                terminal_reward: float,
                                ctx: Optional[dict] = None) -> bool:
    """Module-level CGN_CROSS_INSIGHT emitter for Upgrade III peer
    publishing (rFP §17 + audit 2026-04-23 Q2).

    Use from worker code that emits CGN_TRANSITION via raw _send_msg and
    doesn't have a CGNConsumerClient instance at the emit site. Keeps the
    same filter semantics as emot_cgn's own publisher and as the
    CGNConsumerClient.emit_cross_insight method.

      - 5-second rate gate per `consumer_name`
      - Informative-only: |reward - 0.5| > 0.3

    Returns True if emitted.
    """
    if send_queue is None:
        return False
    now = time.time()
    last = _PEER_CROSS_INSIGHT_RATE_STATE.get(consumer_name, 0.0)
    if now - last < 5.0:
        return False
    try:
        if abs(float(terminal_reward) - 0.5) <= 0.3:
            return False
        send_queue.put_nowait({
            "type": "CGN_CROSS_INSIGHT",
            "src": src_module,
            "dst": "all",
            "ts": now,
            "payload": {
                "origin_consumer": consumer_name,
                "insight_type": "chain_outcome",
                "terminal_reward": float(terminal_reward),
                "ctx_summary": {
                    k: ctx[k] for k in (ctx or {})
                    if isinstance(ctx[k], (int, float, str, bool))
                },
            },
        })
        _PEER_CROSS_INSIGHT_RATE_STATE[consumer_name] = now
        return True
    except Exception as e:
        from titan_plugin.utils.silent_swallow import swallow_warn
        swallow_warn(
            f"[emit_chain_outcome_insight:{consumer_name}] failed", e,
            key=f"cgn_consumer_client.emit_chain_outcome_insight.{consumer_name}")
        return False


class CGNConsumerClient:
    """Local inference client for a single CGN consumer.

    Loads weights from /dev/shm (numpy arrays via ShmWeightReader.read_numpy).
    Does ground() locally — no bus round-trip needed.
    Sends transitions to CGN Worker via bus (non-blocking).

    TORCH-FREE: all inference is pure numpy. See module docstring.
    """

    # Action-specific parameter ranges (must match cgn.py)
    CONF_DELTA_RANGE = (-0.05, 0.10)
    PLASTICITY_RANGE = (0.0, 0.5)
    ASSOC_DELTA_RANGE = (-0.10, 0.10)
    CTX_WEIGHT_RANGE = (0.1, 0.9)

    def __init__(self, consumer_name: str,
                 send_queue=None, module_name: str = "",
                 state_dir: str = "data/cgn",
                 shm_path: str = "/dev/shm/cgn_live_weights.bin",
                 titan_id: str | None = None,
                 config: dict | None = None):
        self._name = consumer_name
        self._send_queue = send_queue
        self._module_name = module_name or consumer_name
        self._state_dir = state_dir
        self._shm_path = shm_path
        self._titan_id = titan_id
        self._cgn_config = config

        # Lazy-init state — weight loading deferred to first use.
        self._initialized = False
        self._value_net = NumpyValueNet()
        self._action_net = NumpyActionNet()
        self._action_names: List[str] = []
        self._config_loaded = False

        # SHM reader — torch-free (just mmap'd bytes + numpy).
        # Microkernel v2 §A.2 part 2 (S4): pass titan_id + config so the
        # dual-mode resolver picks per-titan + 24B header when flag on.
        # Explicit non-default shm_path (test override) still honored.
        from titan_plugin.logic.cgn_shm_protocol import ShmWeightReader
        _shm_arg = shm_path if shm_path != "/dev/shm/cgn_live_weights.bin" else None
        self._shm_reader = ShmWeightReader(
            shm_path=_shm_arg, titan_id=titan_id, config=config)

        # Plug B (rFP §20): EMA of recent emotional cross-insight rewards.
        # Each incoming CGN_CROSS_INSIGHT with origin="emotional" updates
        # this; consumer ground() exposes it via state_vec slot 18 so the
        # V(s) learning can associate "what I was doing when emotion
        # flagged a reward-deviation event." Neutral baseline = 0.5.
        self._emot_insight_reward_ema: float = 0.5
        self._emot_insight_count: int = 0
        self._emot_insight_last_ts: float = 0.0

        # Upgrade III peer publishing (audit 2026-04-23 Q2) — rate gate +
        # counter for outgoing CGN_CROSS_INSIGHT. Mirrors emot_cgn's
        # existing publisher state (emot_cgn.py:331-333).
        self._last_cross_insight_ts: float = 0.0
        self._cross_insights_sent: int = 0

    def note_incoming_cross_insight(self, payload: dict) -> None:
        """Process a CGN_CROSS_INSIGHT from another consumer.

        v1: only emotional insights are used (rFP §20 Plug B). The
        `terminal_reward` becomes an EMA that downstream ground() calls
        can expose via state_vec slot 18 — giving the consumer's V(s)
        learning a signal of "what I was doing when emotion marked
        reward deviation." Neutral baseline 0.5 means no recent event.

        Own-name emissions and unknown origins are silently skipped.
        Never raises.
        """
        try:
            origin = str(payload.get("origin_consumer", ""))
            if origin == self._name or origin != "emotional":
                return
            r = float(payload.get("terminal_reward", 0.5))
            r = max(0.0, min(1.0, r))
            # EMA with α=0.3 — moderately reactive; recent events shift
            # slot 18 a noticeable but non-dominant amount.
            self._emot_insight_reward_ema = (
                0.7 * self._emot_insight_reward_ema + 0.3 * r
            )
            self._emot_insight_count += 1
            self._emot_insight_last_ts = time.time()
        except Exception as e:
            swallow_warn(f'[CGNClient:{self._name}] note_incoming_cross_insight error', e,
                         key="logic.cgn_consumer_client.note_incoming_cross_insight_error", throttle=100)

    def _ensure_initialized(self) -> None:
        """Lazy initialization — loads numpy weights on first use.

        No torch import. Creates NumpyValueNet/NumpyActionNet and loads
        weights from /dev/shm (numpy arrays). If SHM not available,
        starts without weights (returns safe defaults until CGN Worker
        writes the first SHM snapshot).
        """
        if self._initialized:
            return
        self._load_initial()
        self._initialized = True

    def _load_initial(self):
        """Load weights at boot: try /dev/shm (numpy path only)."""
        loaded = self._try_load_shm()
        if loaded:
            logger.info("[CGNClient:%s] Loaded from /dev/shm (v=%d, numpy)",
                        self._name, self._shm_reader._last_version)
            return

        # No SHM available yet. Consumer starts without weights.
        # CGN Worker will write to SHM within seconds; _check_and_reload()
        # picks it up on next ground()/infer_action() call.
        logger.info("[CGNClient:%s] SHM not available — starting without "
                    "weights (will reload on next CGN Worker publish)",
                    self._name)

    def _try_load_shm(self) -> bool:
        """Try loading from /dev/shm as numpy arrays. Returns True on success."""
        try:
            result = self._shm_reader.read_numpy(self._name)
            if result and result.get("value_net"):
                self._value_net.load_state_dict(result["value_net"])
                if result.get("consumer_net"):
                    self._action_net.load_state_dict(result["consumer_net"])
                    # Infer action names from state shape
                    if not self._action_names:
                        n = self._action_net.action_dims
                        self._action_names = [f"action_{i}" for i in range(n)]
                    self._config_loaded = True
                return True
        except Exception as e:
            swallow_warn(f'[CGNClient:{self._name}] SHM load failed', e,
                         key="logic.cgn_consumer_client.shm_load_failed", throttle=100)
        return False

    def _check_and_reload(self):
        """Check /dev/shm version, reload if new. Very cheap (<0.1ms)."""
        self._ensure_initialized()
        if self._shm_reader.has_new_version():
            self._try_load_shm()

    def ground(self, concept_features: dict,
               sensory_ctx: dict) -> LocalGroundingResult:
        """Local forward pass — fast (0.5ms), no bus. Pure numpy.

        Args:
            concept_features: dict with keys matching ConceptFeatures fields
                (concept_id, embedding, confidence, encounter_count, etc.)
            sensory_ctx: dict with keys matching SensoryContext fields
                (epoch, neuromods, concept_confidences, encounter_type, etc.)

        Returns:
            LocalGroundingResult with grounding action + transition data
        """
        self._ensure_initialized()
        if not self._config_loaded or not self._action_net._loaded:
            return LocalGroundingResult()

        self._check_and_reload()

        try:
            state_vec = self._build_state_vector(concept_features,
                                                  sensory_ctx)
            state_input = state_vec.reshape(1, -1)  # (1, 30)

            action_logits, params = self._action_net.forward(state_input)

            # Sample action
            temperature = 0.5
            probs = _np_softmax(action_logits, temperature).squeeze(0)
            action_idx = _np_multinomial(probs)
            action_name = (self._action_names[action_idx]
                           if action_idx < len(self._action_names)
                           else f"action_{action_idx}")

            # Map continuous params
            p = params.squeeze(0)
            conf_delta = self._map_range(p[0], *self.CONF_DELTA_RANGE)
            plasticity = self._map_range(p[1], *self.PLASTICITY_RANGE)
            assoc_delta = self._map_range(p[2], *self.ASSOC_DELTA_RANGE)
            ctx_weight = self._map_range(p[3], *self.CTX_WEIGHT_RANGE)

            # Action-specific adjustments
            conf_delta, plasticity, assoc_delta, ctx_weight = (
                self._adjust_for_action(action_name, conf_delta, plasticity,
                                         assoc_delta, ctx_weight))

            # Build transition data (for bus send to CGN Worker)
            transition = {
                "consumer": self._name,
                "concept_id": concept_features.get("concept_id", "?"),
                "state": state_vec.tolist(),
                "action": action_idx,
                "action_params": p.tolist(),
                "reward": 0.0,  # Filled later via record_outcome
                "timestamp": time.time(),
                "epoch": sensory_ctx.get("epoch", 0),
                "metadata": {
                    "action_name": action_name,
                    "encounter_type": sensory_ctx.get("encounter_type",
                                                       "comprehension"),
                },
            }

            return LocalGroundingResult(
                action_index=action_idx,
                action_name=action_name,
                confidence_delta=round(float(conf_delta), 4),
                tensor_plasticity=round(float(plasticity), 4),
                association_deltas={},
                context_weight=round(float(ctx_weight), 4),
                transition=transition,
            )
        except Exception as e:
            swallow_warn(f'[CGNClient:{self._name}] ground() failed', e,
                         key="logic.cgn_consumer_client.ground_failed", throttle=100)
            return LocalGroundingResult()

    def infer_action(self, sensory_ctx: dict,
                     features: dict = None) -> dict:
        """Policy inference — returns action name + confidence + q_values.

        Same as CGN.infer_social_action() but works for any consumer.
        Pure numpy — no torch.
        """
        fallback = {
            "action_name": self._action_names[0] if self._action_names
                           else "unknown",
            "action_index": 0,
            "confidence": 0.0,
            "q_values": {},
        }

        self._ensure_initialized()
        if not self._config_loaded or not self._action_net._loaded:
            return fallback

        self._check_and_reload()

        try:
            # Build concept features from user-provided features
            cf = features or {}
            concept_f = {
                "concept_id": cf.get("user_id", "inference"),
                "confidence": cf.get("familiarity", 0.0),
                "encounter_count": cf.get("interaction_count", 0),
                "production_count": cf.get("mention_count", 0),
                "age_epochs": cf.get("relationship_age_epochs", 0),
                "cross_modal_conf": max(0.0, min(1.0,
                    (cf.get("social_valence", 0.0) + 1.0) / 2.0)),
                "embedding": np.array(
                    cf.get("social_felt_tensor", [0.5] * 130),
                    dtype=np.float32),
            }

            state_vec = self._build_state_vector(concept_f, sensory_ctx)
            state_input = state_vec.reshape(1, -1)  # (1, 30)

            action_logits, _ = self._action_net.forward(state_input)

            temperature = 0.5
            probs = _np_softmax(action_logits, temperature).squeeze(0)
            best_idx = int(np.argmax(probs))
            best_name = (self._action_names[best_idx]
                         if best_idx < len(self._action_names)
                         else f"action_{best_idx}")
            confidence = float(probs[best_idx])

            q_values = {
                self._action_names[i]: round(float(probs[i]), 4)
                for i in range(min(len(self._action_names), probs.shape[0]))
            }

            return {
                "action_name": best_name,
                "action_index": best_idx,
                "confidence": round(confidence, 4),
                "q_values": q_values,
            }
        except Exception as e:
            swallow_warn(f'[CGNClient:{self._name}] infer_action() failed', e,
                         key="logic.cgn_consumer_client.infer_action_failed", throttle=100)
            return fallback

    def send_transition(self, transition: dict):
        """Send transition to CGN Worker via bus (non-blocking)."""
        if self._send_queue is None:
            return
        try:
            self._send_queue.put_nowait({
                "type": "CGN_TRANSITION",
                "src": self._module_name,
                "dst": "cgn",
                "ts": time.time(),
                "rid": None,
                "payload": transition,
            })
        except Exception as e:
            swallow_warn(f'[CGNClient:{self._name}] send_transition failed', e,
                         key="logic.cgn_consumer_client.send_transition_failed", throttle=100)

    def emit_cross_insight(self, terminal_reward: float,
                           ctx: Optional[dict] = None) -> bool:
        """Emit CGN_CROSS_INSIGHT for this consumer's chain outcome.

        Thin wrapper over module-level `emit_chain_outcome_insight` for
        consumers that have a CGNConsumerClient instance. Returns True
        if emitted. Audit 2026-04-23 Q2 (Upgrade III peer publishing).
        """
        emitted = emit_chain_outcome_insight(
            self._send_queue, self._module_name, self._name,
            terminal_reward, ctx)
        if emitted:
            self._last_cross_insight_ts = time.time()
            self._cross_insights_sent += 1
        return emitted

    def record_outcome(self, concept_id: str, reward: float,
                       outcome_context: dict = None):
        """Send delayed reward to CGN Worker via bus."""
        if self._send_queue is None:
            return
        try:
            self._send_queue.put_nowait({
                "type": "CGN_TRANSITION",
                "src": self._module_name,
                "dst": "cgn",
                "ts": time.time(),
                "rid": None,
                "payload": {
                    "type": "outcome",
                    "consumer": self._name,
                    "concept_id": concept_id,
                    "reward": reward,
                    "outcome_context": outcome_context or {},
                },
            })
        except Exception as e:
            swallow_warn(f'[CGNClient:{self._name}] record_outcome failed', e,
                         key="logic.cgn_consumer_client.record_outcome_failed", throttle=100)

    # ── State vector building (copied from cgn.py _build_state_vector) ──

    def _build_state_vector(self, concept: dict,
                            ctx: dict) -> np.ndarray:
        """Build 30D state vector for V(s) and Q(s,a) input.

        Layout matches cgn.py exactly:
          [0:5]   neuromods: DA, 5HT, NE, GABA, ACh
          [5:11]  MSL summary: top-3 attention + entropy + I_conf + convergence
          [11:20] state summary: body_avg, mind_feel, mind_think, spirit_avg,
                                 outer_avg, drift, trajectory, chi, pi_norm
          [20:29] concept summary: confidence, enc_norm, prod_norm, age_norm,
                                   xm_conf, assoc_density, ctx_diversity,
                                   tensor_mag, embedding_stability
          [29]    encounter_type encoding
        """
        vec = np.zeros(30, dtype=np.float32)

        # Neuromods [0:5]
        nm = ctx.get("neuromods", {})

        def _nm_val(key, alt_key=None, default=0.5):
            v = nm.get(key, nm.get(alt_key, default) if alt_key else default)
            if isinstance(v, dict):
                return float(v.get("level", default))
            return float(v)

        vec[0] = _nm_val("DA")
        vec[1] = _nm_val("5-HT", "5HT")
        vec[2] = _nm_val("NE")
        vec[3] = _nm_val("GABA")
        vec[4] = _nm_val("ACh")

        # MSL summary [5:11]
        msl_attn = ctx.get("msl_attention")
        if msl_attn is not None and hasattr(msl_attn, '__len__') and len(msl_attn) >= 6:
            attn = np.array(msl_attn, dtype=np.float32)
            top3 = sorted(attn, reverse=True)[:3]
            vec[5:8] = top3
            attn_norm = attn / (np.sum(attn) + 1e-10)
            entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-10))
            vec[8] = min(1.0, entropy / 4.0)
        cc = ctx.get("concept_confidences", {})

        def _cc_val(v):
            if isinstance(v, dict):
                return float(v.get("confidence", v.get("level", 0.0)))
            return float(v) if v is not None else 0.0

        vec[9] = _cc_val(cc.get("I", 0.0))
        vec[10] = (sum(_cc_val(v) for v in cc.values()) / max(len(cc), 1)
                   ) if cc else 0.0

        # State summary [11:20]
        state_132d = ctx.get("state_132d")
        if state_132d is not None and hasattr(state_132d, '__len__') and len(state_132d) >= 65:
            s = np.array(state_132d, dtype=np.float32)
            vec[11] = float(np.mean(s[:5]))
            vec[12] = float(np.mean(s[5:15]))
            vec[13] = float(np.mean(s[15:20]))
            vec[14] = float(np.mean(s[20:65]))
            if len(s) > 65:
                vec[15] = float(np.mean(s[65:95]))
            vec[16] = float(np.std(s[:65]))
        # Plug A (rFP §20): slot 17 = emotion valence [0,1] from bundle.
        # 0.5 = neutral / bundle unavailable (matches old placeholder).
        try:
            from titan_plugin.logic.emot_bundle_protocol import (
                read_emotion_valence_normalized)
            vec[17] = read_emotion_valence_normalized(default=0.5)
        except Exception:
            vec[17] = 0.5
        # Plug B (rFP §20): slot 18 = EMA of recent emotional cross-insight
        # rewards. 0.5 = neutral / no recent emotion events. Tracks
        # reactive "what I was doing when EMOT-CGN flagged reward
        # deviation" — lets V(s) learn to associate current concept
        # with emotional-anomaly moments. Reverts to 0.5 as events age
        # out via EMA decay.
        vec[18] = float(self._emot_insight_reward_ema)
        vec[19] = min(1.0, ctx.get("epoch", 0) / 500000.0)

        # Concept summary [20:29]
        def _safe_f(v, default=0.0):
            if isinstance(v, dict):
                return float(v.get("confidence", v.get("level", default)))
            if isinstance(v, bytes):
                return default
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        vec[20] = _safe_f(concept.get("confidence", 0.0))
        vec[21] = min(1.0, _safe_f(concept.get("encounter_count", 0)) / 100.0)
        vec[22] = min(1.0, _safe_f(concept.get("production_count", 0)) / 50.0)
        vec[23] = min(1.0, _safe_f(concept.get("age_epochs", 0)) / 300000.0)
        vec[24] = _safe_f(concept.get("cross_modal_conf", 0.0))
        assoc = concept.get("associations", {})
        vec[25] = min(1.0, len(assoc) / 10.0) if isinstance(assoc, dict) else 0.0
        ctx_hist = concept.get("context_history", [])
        vec[26] = min(1.0, len(ctx_hist) / 10.0) if isinstance(ctx_hist, list) else 0.0
        emb = concept.get("embedding")
        if emb is not None and hasattr(emb, '__len__') and len(emb) > 0:
            vec[27] = min(1.0, float(np.linalg.norm(
                np.array(emb, dtype=np.float32))) / 10.0)
        meanings = concept.get("meaning_contexts", [])
        vec[28] = min(1.0, len(meanings) / 5.0) if isinstance(meanings, list) else 0.0

        # Encounter type [29]
        _enc_map = {"comprehension": 0.2, "production": 0.4,
                    "teaching": 0.6, "reasoning": 0.8, "bootstrap": 0.1,
                    "social": 0.3, "knowledge": 0.5}
        vec[29] = _enc_map.get(ctx.get("encounter_type", "comprehension"), 0.5)

        return vec

    @staticmethod
    def _map_range(value: float, low: float, high: float) -> float:
        return low + value * (high - low)

    @staticmethod
    def _adjust_for_action(action_name, conf_delta, plasticity,
                           assoc_delta, ctx_weight):
        """Action-specific parameter adjustments (matches cgn.py)."""
        if action_name == "reinforce":
            conf_delta = max(0.0, conf_delta)
            plasticity *= 0.3
        elif action_name == "explore":
            conf_delta *= 0.5
            plasticity = max(0.2, plasticity)
        elif action_name == "differentiate":
            plasticity = max(0.3, plasticity)
            conf_delta = 0.0
        elif action_name == "consolidate":
            ctx_weight = max(0.7, ctx_weight)
            plasticity *= 0.2
        elif action_name == "associate":
            assoc_delta = max(0.02, assoc_delta)
        elif action_name == "dissociate":
            assoc_delta = min(-0.02, assoc_delta)
        elif action_name == "deepen":
            plasticity = max(0.3, plasticity)
            conf_delta = max(0.0, conf_delta)
        elif action_name == "stabilize":
            plasticity = min(0.05, plasticity)
            conf_delta = max(0.0, conf_delta) * 0.5
        return conf_delta, plasticity, assoc_delta, ctx_weight
