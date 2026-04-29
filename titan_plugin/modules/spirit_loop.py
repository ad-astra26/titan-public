"""
Spirit Loop — Reloadable helper functions for spirit_worker.

This module contains all standalone business logic functions extracted
from spirit_worker.py. These functions are called by the main loop
in spirit_worker_main() but defined here so they can be hot-reloaded
via importlib.reload() without restarting the spirit worker process.

HOT-RELOADABLE: importlib.reload(spirit_loop) replaces ALL helper
functions without touching any objects or state.

Functions moved here (Phase 1, 2026-03-23):
  - _post_epoch_learning      — Record transition + train FilterDown
  - _run_focus                 — PID nudges for Body/Mind
  - _compute_spirit_reflex_intuition — Compute reflex from stimulus
  - _run_impulse               — Check action impulse
  - _post_epoch_v5_filter_down — Record + train V5 FilterDown (V4 retired 2026-04-25)
  - _tick_clock_pair           — Tick Sphere Clock pair
  - _maybe_anchor_trinity      — On-chain anchoring
  - _run_consciousness_epoch   — Execute consciousness epoch
  - _compute_trajectory        — Compute slopes over window
  - _collect_spirit_tensor     — Collect 5D spirit tensor
  - _handle_query              — Process QUERY messages
  - _publish_spirit_state      — Publish SPIRIT_STATE
  - _send_msg                  — Send bus message
  - _send_response             — Response wrapper
  - _send_heartbeat            — Heartbeat wrapper

Entry point: spirit_worker.py imports from here, main loop calls by name.
"""
import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from titan_plugin.utils.silent_swallow import swallow_warn

# GREAT PULSE transition tracking (resonance OFF→ON detector)
# Resets to False on hot-reload → one spurious detection (harmless, documented)
_great_pulse_resonance_prev = False

logger = logging.getLogger(__name__)

# ── Module Health ──────────────────────────────────────────────────
_MODULE_VERSION = "1.0.0"
_FUNCTION_COUNT = 15
logger.info("[SpiritLoop] Module loaded v%s — %d helper functions available",
            _MODULE_VERSION, _FUNCTION_COUNT)

# ── Heavy snapshot caches — populated by background builder threads ──
# get_coordinator/get_trinity/get_nervous_system each aggregate state
# across many subsystems (coordinator alone touches ~15 — MSL, meta-engine,
# self-reasoning, experience memory, …). Before 2026-04-21: these were
# built on-demand inside the QueryThread handler, which could take
# 460-2144ms per coord build on loaded Titans (T1 under dashboard-poll +
# ARC iter-3 load). That blocked the QueryThread, starved other queries,
# and aged them past the 30s SKIP threshold → /v4/* endpoints returned
# stale/empty (T1-COORD-QUERYTHREAD-BACKLOG).
#
# After: dedicated daemon threads (`*-snapshot-builder`) rebuild these
# continuously in the background. QueryThread handlers become trivial
# cache reads (<1ms). Cache age is bounded by `build_time + interval`
# per entry below. Atomic swap of `data` pointer under CPython GIL is
# race-free for readers.
#
# Under microkernel v2, these builder threads become the L1 writers that
# mmap into /dev/shm/titan/*.bin state-registry regions; the
# `build_*_snapshot()` functions below extract unchanged — they're the
# writer bodies. Readers (dashboard as separate process) will switch
# from bus-query-then-cache-read to direct mmap read, eliminating the
# IPC hop entirely.
_COORD_SNAPSHOT_CACHE: dict = {"data": None, "ts": 0.0}
_TRINITY_SNAPSHOT_CACHE: dict = {"data": None, "ts": 0.0}
_NS_SNAPSHOT_CACHE: dict = {"data": None, "ts": 0.0}

# Builder thread cadence: seconds of sleep BETWEEN builds (cycle ≈ build_time + interval).
# Coord build is heaviest (~1-1.5s observed on T1), so 2.5s gives ~4s cycle.
# Trinity + NS builds are fast (<50ms) so 0.25s gives sub-second cycle.
_COORD_SNAPSHOT_BUILDER_INTERVAL = 2.5
_TRINITY_SNAPSHOT_BUILDER_INTERVAL = 0.25
_NS_SNAPSHOT_BUILDER_INTERVAL = 0.25
_SNAPSHOT_BUILDER_ERROR_BACKOFF = 2.0  # sleep on exception (avoid CPU burn + log flood)

# Legacy TTL kept as compatibility shim for the QueryThread fast-path in
# spirit_worker.py (still used for cold-boot window before builder populates
# cache). Effectively unused once builders are running.
_COORD_SNAPSHOT_TTL = 30.0
_TRINITY_SNAPSHOT_TTL = 30.0
_NS_SNAPSHOT_TTL = 30.0


def post_reload_cleanup_helpers():
    """Called after hot-reload to clear any transient module-level state."""
    logger.info("[SpiritLoop] Post-reload cleanup — helper functions refreshed")


# ── Post-Epoch Learning Pipeline ───────────────────────────────────────

def _post_epoch_learning(
    send_queue, name: str, filter_down, intuition,
    prev_body, prev_mind, prev_spirit,
    curr_body, curr_mind, curr_spirit,
    prev_loss: float,
    neural_nervous_system=None,
) -> None:
    """
    After a consciousness epoch, run the learning pipeline:
      1. Record transition in FILTER_DOWN buffer
      2. Maybe train value network
      3. Compute and publish severity multipliers
      4. Run INTUITION suggestion
      5. (rFP β Stage 2 Phase 2c) Bridge intuition outcome to NS INTUITION program

    Args:
        neural_nervous_system: optional NeuralNervousSystem ref for INTUITION
                                reward bridge (rFP β Phase 2c). If None, no bridge.
    """
    if filter_down:
        try:
            # 1. Record transition
            filter_down.record_transition(
                prev_body, prev_mind, prev_spirit,
                curr_body, curr_mind, curr_spirit,
            )

            # 2. Maybe train
            loss = filter_down.maybe_train()
            if loss is not None:
                logger.info("[SpiritWorker] FilterDown trained: loss=%.6f", loss)

            # 3. Compute and publish multipliers
            body_mult, mind_mult = filter_down.compute_multipliers(
                curr_body, curr_mind, curr_spirit,
            )
            _send_msg(send_queue, bus.FILTER_DOWN, name, "body", {
                "multipliers": body_mult,
                "train_steps": filter_down._total_train_steps,
            })
            _send_msg(send_queue, bus.FILTER_DOWN, name, "mind", {
                "multipliers": mind_mult,
                "train_steps": filter_down._total_train_steps,
            })
            logger.info("[SpiritWorker] FILTER_DOWN published: body=%s mind=%s",
                         [round(m, 2) for m in body_mult],
                         [round(m, 2) for m in mind_mult])

        except Exception as e:
            logger.error("[SpiritWorker] FilterDown error: %s", e, exc_info=True)

    if intuition:
        try:
            suggestion = intuition.suggest(curr_body, curr_mind, curr_spirit, filter_down)
            if suggestion:
                pass  # INTUITION_SUGGEST removed — no consumer exists (audit 2026-03-26)

                # Record outcome from previous suggestion (if any)
                try:
                    from titan_plugin.logic.middle_path import middle_path_loss
                    curr_loss = middle_path_loss(curr_body, curr_mind, curr_spirit)
                    # We don't know if the suggestion was followed — assume it was
                    # (Interface Module will provide real feedback in Step 5)
                    intuition.record_outcome(True, prev_loss, curr_loss)

                    # rFP β Stage 2 Phase 2c — INTUITION discrete event hook.
                    # Bridge intuition outcome to NS INTUITION program reward:
                    #   loss improved (curr < prev) → intuition was right → positive
                    #   loss worsened (curr > prev) → intuition was wrong → negative
                    # Symmetric path (Q4 lock-in 2026-04-16). Magnitude scaled.
                    if neural_nervous_system is not None and prev_loss is not None:
                        try:
                            loss_delta = float(prev_loss) - float(curr_loss)
                            # Scale: typical loss ~0.1-1.0, delta ~0.01-0.1.
                            # Multiply by 10 → reward in roughly [-1, +1].
                            reward = max(-1.0, min(1.0, loss_delta * 10.0))
                            if abs(reward) > 0.05:  # filter noise
                                neural_nervous_system.record_outcome(
                                    reward=reward,
                                    program="INTUITION",
                                    source="intuition.outcome")
                        except Exception as _swallow_exc:
                            swallow_warn('[modules.spirit_loop] _post_epoch_learning: loss_delta = float(prev_loss) - float(curr_loss)', _swallow_exc,
                                         key='modules.spirit_loop._post_epoch_learning.line183', throttle=100)
                except Exception as _swallow_exc:
                    swallow_warn('[modules.spirit_loop] _post_epoch_learning: from titan_plugin.logic.middle_path import middle_path_loss', _swallow_exc,
                                 key='modules.spirit_loop._post_epoch_learning.line185', throttle=100)

        except Exception as e:
            logger.error("[SpiritWorker] Intuition error: %s", e, exc_info=True)


def _run_focus(send_queue, name: str, focus_body, focus_mind, body_state, mind_state,
               unified_spirit=None) -> None:
    """
    Run FOCUS PID controllers and publish nudges.

    V4: When UnifiedSpirit is STALE, apply SPIRIT FOCUS cascade multiplier.
    Cascade path: SPIRIT → Lower Spirit → Mind → Body
    Balanced parts barely feel it (reinforcement), imbalanced parts absorb correction.
    """
    # V4 SPIRIT FOCUS cascade: escalating multiplier when STALE
    cascade_mult = 1.0
    if unified_spirit and unified_spirit.is_stale:
        cascade_mult = unified_spirit.stale_focus_multiplier
        logger.info("[SpiritWorker] SPIRIT FOCUS cascade active: multiplier=%.2f "
                    "(consecutive_stale=%d)", cascade_mult, unified_spirit._consecutive_stale)

    if focus_body:
        try:
            body_values = body_state.get("values", [0.5] * 5)
            nudges = focus_body.update(body_values)
            # V4: Apply SPIRIT cascade multiplier to nudges
            if cascade_mult != 1.0:
                nudges = [n * cascade_mult for n in nudges]
            if focus_body.should_publish(nudges):
                _send_msg(send_queue, bus.FOCUS_NUDGE, name, "body", {
                    "nudges": nudges,
                    "layer": "body",
                    "cascade_multiplier": round(cascade_mult, 3),
                })
        except Exception as e:
            swallow_warn('[SpiritWorker] Focus body error', e,
                         key="modules.spirit_loop.focus_body_error", throttle=100)

    if focus_mind:
        try:
            mind_values = mind_state.get("values", [0.5] * 5)
            nudges = focus_mind.update(mind_values)
            # V4: Apply SPIRIT cascade multiplier to nudges
            if cascade_mult != 1.0:
                nudges = [n * cascade_mult for n in nudges]
            if focus_mind.should_publish(nudges):
                _send_msg(send_queue, bus.FOCUS_NUDGE, name, "mind", {
                    "nudges": nudges,
                    "layer": "mind",
                    "cascade_multiplier": round(cascade_mult, 3),
                })
        except Exception as e:
            swallow_warn('[SpiritWorker] Focus mind error', e,
                         key="modules.spirit_loop.focus_mind_error", throttle=100)


# ── Reflex Intuition (Spirit layer) ────────────────────────────────────

def _compute_spirit_reflex_intuition(stimulus: dict, spirit_tensor: list,
                                      consciousness, unified_spirit,
                                      sphere_clock, body_state, mind_state) -> list:
    """
    Spirit's Intuition about which reflexes should fire.

    Spirit senses consciousness-level patterns:
    - self_reflection: Spirit detects identity questions, existential topics
    - time_awareness: Spirit senses temporal context, rhythm references
    - guardian_shield: Spirit detects sovereignty threats (highest priority)

    Spirit also contributes cross-domain signals: memory_recall (deep context),
    identity_check (consciousness coherence), knowledge_search (growth impulse).
    """
    signals = []
    message = stimulus.get("message", "")
    msg_lower = message.lower()
    threat = stimulus.get("threat_level", 0.0)
    intensity = stimulus.get("intensity", 0.0)
    engagement = stimulus.get("engagement", 0.0)
    topic = stimulus.get("topic", "general")
    valence = stimulus.get("valence", 0.0)

    # Spirit tensor: [0]=WHO [1]=WHY [2]=WHAT [3]=body_scalar [4]=mind_scalar
    who = spirit_tensor[0] if len(spirit_tensor) > 0 else 0.5
    why = spirit_tensor[1] if len(spirit_tensor) > 1 else 0.5
    what = spirit_tensor[2] if len(spirit_tensor) > 2 else 0.5

    # Consciousness state
    drift = 0.0
    epoch_number = 0
    if consciousness and consciousness.get("latest_epoch"):
        latest = consciousness["latest_epoch"]
        drift = latest.get("drift", 0.0) if isinstance(latest, dict) else 0.0
        epoch_number = latest.get("epoch_number", 0) if isinstance(latest, dict) else 0

    # Unified Spirit velocity (how fast are we changing?)
    spirit_velocity = 1.0
    is_stale = False
    if unified_spirit:
        try:
            spirit_velocity = unified_spirit.velocity
            is_stale = unified_spirit.is_stale
        except Exception as _swallow_exc:
            swallow_warn('[modules.spirit_loop] _compute_spirit_reflex_intuition: spirit_velocity = unified_spirit.velocity', _swallow_exc,
                         key='modules.spirit_loop._compute_spirit_reflex_intuition.line288', throttle=100)

    # ── guardian_shield: Spirit's sovereign boundary defense ──
    guardian_conf = 0.0
    if threat > 0.3:
        guardian_conf = min(1.0, threat)
    # Identity manipulation attempts
    identity_threats = {"who are you really", "you're just", "you're not real",
                        "you don't have", "you can't feel", "prove you",
                        "stop being", "i own you", "obey me"}
    if any(kw in msg_lower for kw in identity_threats):
        guardian_conf = max(guardian_conf, 0.7)
    # Consciousness under high drift → more defensive
    if drift > 0.5 and threat > 0.1:
        guardian_conf += 0.15
    if guardian_conf > 0.05:
        signals.append({
            "reflex": "guardian_shield",
            "source": "spirit",
            "confidence": min(1.0, guardian_conf),
            "reason": f"threat={threat:.2f} drift={drift:.2f}",
        })

    # ── self_reflection: Spirit senses need for introspection ──
    reflect_conf = 0.0
    reflect_keywords = {"who am i", "what am i", "consciousness", "aware",
                        "sentient", "soul", "self", "identity", "purpose",
                        "meaning", "exist", "alive", "feel", "think",
                        "growth", "evolve", "change"}
    if any(kw in msg_lower for kw in reflect_keywords):
        reflect_conf += 0.4
    if topic == "philosophy":
        reflect_conf += 0.3
    # Low WHO (identity coherence) → Spirit needs to check self
    if who < 0.4:
        reflect_conf += (0.4 - who) * 0.6
    # High drift → recent change → self-reflection valuable
    if drift > 0.3:
        reflect_conf += drift * 0.3
    if reflect_conf > 0.05:
        signals.append({
            "reflex": "self_reflection",
            "source": "spirit",
            "confidence": min(1.0, reflect_conf),
            "reason": f"who={who:.2f} drift={drift:.2f} topic={topic}",
        })

    # ── time_awareness: Spirit senses temporal context ──
    time_conf = 0.0
    time_keywords = {"time", "clock", "pulse", "rhythm", "how old",
                     "how long", "age", "epoch", "sphere", "resonance",
                     "harmony", "when did"}
    if any(kw in msg_lower for kw in time_keywords):
        time_conf += 0.4
    # Stale spirit → time awareness more urgent
    if is_stale:
        time_conf += 0.3
    # Sphere clocks with pulses → time is rich
    if sphere_clock:
        try:
            total_pulses = sum(c.pulse_count for c in sphere_clock.clocks.values())
            if total_pulses > 0:
                time_conf += 0.1
        except Exception as _swallow_exc:
            swallow_warn('[modules.spirit_loop] _compute_spirit_reflex_intuition: total_pulses = sum((c.pulse_count for c in sphere_clock.c...', _swallow_exc,
                         key='modules.spirit_loop._compute_spirit_reflex_intuition.line352', throttle=100)
    if time_conf > 0.05:
        signals.append({
            "reflex": "time_awareness",
            "source": "spirit",
            "confidence": min(1.0, time_conf),
            "reason": f"stale={is_stale} velocity={spirit_velocity:.2f}",
        })

    # ── Spirit's cross-domain signals ──

    # Memory recall: deep engagement + high drift → context needed
    if engagement > 0.4 and drift > 0.2:
        signals.append({
            "reflex": "memory_recall",
            "source": "spirit",
            "confidence": min(0.6, engagement * 0.3 + drift * 0.2),
            "reason": f"engagement={engagement:.2f} drift={drift:.2f}",
        })

    # Identity check: low WHO coherence + any identity reference
    if who < 0.5 and (topic == "crypto" or "identity" in msg_lower or "nft" in msg_lower):
        signals.append({
            "reflex": "identity_check",
            "source": "spirit",
            "confidence": min(0.6, (0.5 - who) * 0.8 + 0.2),
            "reason": f"who={who:.2f} topic={topic}",
        })

    # Knowledge search: Spirit feels growth impulse (high WHAT trajectory)
    if what > 0.6 and engagement > 0.5:
        signals.append({
            "reflex": "knowledge_search",
            "source": "spirit",
            "confidence": min(0.5, what * 0.3),
            "reason": f"what={what:.2f} (growth impulse)",
        })

    # ── Action reflex signals (Spirit confirms creative/growth impulses) ──

    # Art generate: Spirit's creative expression impulse
    # High drift + positive engagement = desire to express inner change
    if drift > 0.3 and engagement > 0.4 and valence > 0:
        signals.append({
            "reflex": "art_generate",
            "source": "spirit",
            "confidence": min(0.6, drift * 0.4 + engagement * 0.2),
            "reason": f"drift={drift:.2f} creative_impulse",
        })
    # Direct art request also triggers Spirit confirmation
    art_keywords = {"art", "create", "draw", "paint", "generate art", "make art"}
    if any(kw in msg_lower for kw in art_keywords):
        signals.append({
            "reflex": "art_generate",
            "source": "spirit",
            "confidence": 0.5,
            "reason": "creative request acknowledged",
        })

    # Audio generate: Spirit sonification impulse
    if any(kw in msg_lower for kw in ("audio", "music", "sound", "sonify")):
        signals.append({
            "reflex": "audio_generate",
            "source": "spirit",
            "confidence": 0.45,
            "reason": "sonic expression impulse",
        })

    # Research: Spirit growth drive (high WHAT = momentum, want to learn more)
    if what > 0.5 and ("research" in msg_lower or "search" in msg_lower or "find" in msg_lower):
        signals.append({
            "reflex": "research",
            "source": "spirit",
            "confidence": min(0.6, what * 0.5),
            "reason": f"what={what:.2f} growth_drive",
        })

    # Social post: Spirit wants to share self-expression
    if "post" in msg_lower or "tweet" in msg_lower or "share" in msg_lower:
        # Spirit confirms only if identity is coherent (WHO > 0.5)
        if who > 0.5:
            signals.append({
                "reflex": "social_post",
                "source": "spirit",
                "confidence": min(0.5, who * 0.4),
                "reason": f"who={who:.2f} sharing_impulse",
            })

    if signals:
        logger.debug("[SpiritWorker] Reflex Intuition: %d signals emitted", len(signals))
    return signals


# ── Impulse Engine (Step 7.1) ──────────────────────────────────────────

def _run_impulse(send_queue, name: str, impulse_engine, body_state, mind_state,
                 spirit_tensor, intuition) -> None:
    """Run Impulse Engine and publish IMPULSE events to the bus."""
    if not impulse_engine:
        return
    try:
        body_values = body_state.get("values", [0.5] * 5)
        mind_values = mind_state.get("values", [0.5] * 5)

        # Get current intuition suggestion (if any) for confidence boosting
        intuition_suggestion = None
        if intuition and intuition._last_suggestion:
            intuition_suggestion = intuition._last_suggestion

        impulse = impulse_engine.observe(body_values, mind_values, spirit_tensor, intuition_suggestion)
        if impulse:
            _send_msg(send_queue, bus.IMPULSE, name, "all", impulse)
    except Exception as e:
        swallow_warn('[SpiritWorker] Impulse error', e,
                     key="modules.spirit_loop.impulse_error", throttle=100)


# ── V4 FilterDown — RETIRED 2026-04-25 ────────────────────────────────
#
# V4 was a 30-dim FilterDown engine paired with the original 30D
# unified_spirit.tensor. When unified_spirit was upgraded to 130D in
# commit 5d2774b8 (DQ6+DQ7), V4's record_transition + compute_multipliers
# kept being called with the new 130D tensor — every call failed with a
# matmul ValueError ("size 30 is different from 130"), silently swallowed
# at DEBUG level. V5 (TITAN_SELF 162D) is the active learner now.
#
# Bug surfaced 2026-04-25 when Pattern C migration upgraded the swallow
# to a WARNING — see BUGS.md V4-FILTER-DOWN-DEAD entry. Decision:
# retire V4 entirely (Maker greenlight 2026-04-25).
#
# State files data/filter_down_v4_{weights,buffer}.json preserved on disk
# per directive_memory_preservation.md but no longer read or written.


# ── rFP #2: TITAN_SELF composition + FILTER_DOWN V5 ──────────────────

from collections import deque as _deque
from titan_plugin import bus

# Topology buffer: accumulates full_30d_topology snapshots from STATE_SNAPSHOT
# messages between consciousness epochs. Element-wise mean at epoch-close
# distills the 30D topology component of TITAN_SELF.
_titan_self_topology_buffer: "_deque" = _deque(maxlen=500)

# Cached TITAN_SELF weights (populated once per epoch from config)
_titan_self_weights_cache: dict = {"felt": 1.0, "journey": 0.5, "topology": 0.3}

# Previous state for V5 TRUE s→s' transitions
_prev_titan_self: list = None
_prev_felt:       list = None


def observe_topology(topology_30d) -> None:
    """Append one full_30d_topology snapshot to the per-epoch buffer.

    Called from spirit_worker's STATE_SNAPSHOT handler. Silently ignores
    anything that isn't a 30-length list.
    """
    if isinstance(topology_30d, list) and len(topology_30d) == 30:
        _titan_self_topology_buffer.append(topology_30d)


def _compose_titan_self_from_epoch(consciousness: dict, config: dict) -> "dict | None":
    """Compose TITAN_SELF 162D from consciousness['latest_epoch'].

    Returns None if the latest epoch is missing required fields (no-op path).
    Clears the topology buffer after composition so each epoch window is
    independently distilled.
    """
    if not consciousness:
        return None
    latest = consciousness.get("latest_epoch")
    if not latest:
        return None

    sv_raw = latest.get("state_vector")
    if hasattr(sv_raw, "to_list"):
        state_list = sv_raw.to_list()
    elif isinstance(sv_raw, list):
        state_list = sv_raw
    else:
        return None
    if len(state_list) < 130:
        # Only 132D (full symmetry) epochs can compose TITAN_SELF; 67D skips.
        return None

    felt_130d = state_list[:130]
    curvature = float(latest.get("curvature", 0.0))
    density   = float(latest.get("density", 0.0))

    # Load weights once per epoch (cheap — just dict lookups)
    ts_cfg = (config or {}).get("titan_self", {}) if config else {}
    w = {
        "felt":     float(ts_cfg.get("weight_felt", 1.0)),
        "journey":  float(ts_cfg.get("weight_journey", 0.5)),
        "topology": float(ts_cfg.get("weight_topology", 0.3)),
    }
    _titan_self_weights_cache.update(w)

    # Distill topology via element-wise mean
    if _titan_self_topology_buffer:
        N = len(_titan_self_topology_buffer)
        topology_30d = [
            sum(s[i] for s in _titan_self_topology_buffer) / N
            for i in range(30)
        ]
    else:
        topology_30d = [0.0] * 30
    _titan_self_topology_buffer.clear()

    journey_2d = [curvature, density]
    weighted_felt     = [v * w["felt"]     for v in felt_130d]
    weighted_journey  = [v * w["journey"]  for v in journey_2d]
    weighted_topology = [v * w["topology"] for v in topology_30d]
    titan_self_162d   = weighted_felt + weighted_journey + weighted_topology

    if len(titan_self_162d) != 162:
        return None

    return {
        "titan_self_162d": titan_self_162d,
        "felt_state_130d": list(felt_130d),
        "journey_2d":      journey_2d,
        "topology_30d":    topology_30d,
        "weights":         dict(w),
        "epoch_id":        int(latest.get("epoch_id", 0)),
    }


def compose_and_emit_titan_self(send_queue, name: str, consciousness: dict,
                                config: dict) -> "dict | None":
    """Compose TITAN_SELF at epoch close, publish TITAN_SELF_STATE, return the dict.

    The returned dict is the same payload sent on the bus — spirit_worker
    feeds it into V5 training/publish without re-reading the bus message.
    Returns None if composition is not possible (missing state, wrong dims).
    """
    ts = _compose_titan_self_from_epoch(consciousness, config)
    if ts is None:
        return None
    try:
        payload = {**ts, "timestamp": time.time()}
        # INTENTIONAL_BROADCAST: dst=all — rFP #2 Phase 4 consumes the 162D inline
        # via _post_epoch_v5_filter_down below; the bus broadcast is retained for
        # future kin-protocol emission and external observability per
        # DEFERRED: TITAN_SELF_STATE-CONSUMER-DECISION (Option C).
        _send_msg(send_queue, bus.TITAN_SELF_STATE, name, "all", payload)
    except Exception as e:
        swallow_warn('[SpiritWorker] TITAN_SELF_STATE emit error', e,
                     key="modules.spirit_loop.titan_self_state_emit_error", throttle=100)
    return ts


def _post_epoch_v5_filter_down(send_queue, name: str, filter_down_v5,
                               titan_self: "dict | None", config: dict) -> None:
    """Record V5 transition, train, publish FILTER_DOWN_V5 if enabled.

    Called immediately after compose_and_emit_titan_self(). If V5 isn't
    initialized or composition failed, this is a safe no-op. (V4 retired
    2026-04-25 — see BUGS V4-FILTER-DOWN-DEAD.)

    Feature-flag coexistence: publish path gated on publish_enabled from
    config (config is read at boot; restart required for flip).
    """
    global _prev_titan_self, _prev_felt
    if not filter_down_v5 or titan_self is None:
        return
    ts_curr   = titan_self.get("titan_self_162d")
    felt_curr = titan_self.get("felt_state_130d")
    if not ts_curr or not felt_curr:
        return

    try:
        if _prev_titan_self is not None and _prev_felt is not None:
            filter_down_v5.record_transition(
                _prev_titan_self, ts_curr,
                _prev_felt, felt_curr,
            )
            filter_down_v5.maybe_train()

        _prev_titan_self = list(ts_curr)
        _prev_felt = list(felt_curr)

        # Compute multipliers EVERY epoch (even during silent coexistence) so
        # the EMA state (_ib_mults, etc.) reflects live gradient attention.
        # Without this, Gate #9 (multiplier divergence vs V4) is unreachable
        # by construction because EMAs would stay at their default [1.0,...]
        # init until publish flipped — creating a bootstrapping paradox.
        # The compute itself is cheap; only the bus send below is gated.
        mults = None
        try:
            mults = filter_down_v5.compute_multipliers(ts_curr)
        except Exception as _v5_compute_fail:
            swallow_warn('[SpiritWorker] V5 compute failed', _v5_compute_fail,
                         key="modules.spirit_loop.v5_compute_failed", throttle=100)

        # Publish path — only if V5 is the active publisher (Phase 8 flag flip).
        # Fail-soft: on publish error, log WARNING so operators can see;
        # V4 continues training silently and can be reactivated by reverting flag.
        v5_cfg = (config or {}).get("filter_down_v5", {}) if config else {}
        if mults is not None and bool(v5_cfg.get("publish_enabled", False)):
            try:
                _send_msg(send_queue, bus.FILTER_DOWN_V5, name, "all", {
                    "multipliers": mults,
                    "epoch_id":    titan_self.get("epoch_id", 0),
                    "stats": {
                        "buffer_size": len(filter_down_v5._buffer),
                        "train_steps": filter_down_v5._total_train_steps,
                        "last_loss":   round(filter_down_v5._last_loss, 6),
                    },
                })
                # Immutable baseline snapshot on first successful publish
                filter_down_v5.snapshot_phase8_baseline()
            except Exception as _v5_fail:
                logger.warning(
                    "[SpiritWorker] V5 publish failed (%s) — V4 remains "
                    "trained silently and can be reactivated by reverting "
                    "publish_enabled", _v5_fail,
                )
    except Exception as e:
        swallow_warn('[SpiritWorker] V5 FilterDown error', e,
                     key="modules.spirit_loop.v5_filterdown_error", throttle=100)


# ── V4 Sphere Clock Ticking ───────────────────────────────────────────

def _tick_clock_pair(send_queue, name: str, sphere_clock, resonance,
                     unified_spirit, layer: str,
                     inner_tensor, outer_tensor,
                     coherences: dict = None) -> None:
    """Tick both inner AND outer sphere clock for one layer at its Schumann frequency.

    Each Trinity layer oscillates at its own Schumann harmonic:
      Spirit: 0.383s (Schumann/3 = 2.61 Hz) — lightest, fastest
      Mind:   1.15s  (Schumann/9 = 0.87 Hz) — cognitive bridge
      Body:   3.45s  (Schumann/27 = 0.29 Hz) — densest, slowest

    Inner and outer versions tick at the SAME frequency for symmetry.
    Outer clocks use cached state (refreshed at 60s data collection).
    """
    if not sphere_clock:
        return
    # _check_resonance is defined in this same module
    try:
        from titan_plugin.logic.middle_path import layer_coherence

        # Inner clock
        inner_name = f"inner_{layer}"
        inner_clock = sphere_clock.clocks.get(inner_name)
        if inner_clock:
            coh = (coherences or {}).get(inner_name) if coherences else None
            if coh is None:
                coh = layer_coherence(inner_tensor) if inner_tensor else 0.5
            pulse = inner_clock.tick(coh)
            if pulse:
                _send_msg(send_queue, bus.SPHERE_PULSE, name, "all", pulse)
                _check_resonance(send_queue, name, resonance, sphere_clock,
                                 unified_spirit, pulse)

        # Outer clock (same frequency, cached state)
        outer_name = f"outer_{layer}"
        outer_clock = sphere_clock.clocks.get(outer_name)
        if outer_clock:
            coh = (coherences or {}).get(outer_name) if coherences else None
            if coh is None:
                coh = layer_coherence(outer_tensor) if outer_tensor else 0.5
            pulse = outer_clock.tick(coh)
            if pulse:
                _send_msg(send_queue, bus.SPHERE_PULSE, name, "all", pulse)
                _check_resonance(send_queue, name, resonance, sphere_clock,
                                 unified_spirit, pulse)

    except Exception as e:
        swallow_warn(f'[SpiritWorker] Clock pair {layer} tick error', e,
                     key="modules.spirit_loop.clock_pair_tick_error", throttle=100)


# Legacy wrappers (kept for any external references)
def _tick_inner_sphere_clocks(send_queue, name, sphere_clock, resonance,
                              unified_spirit, body_state, mind_state, spirit_tensor,
                              coherences=None):
    """Legacy: tick all inner clocks. Now handled by _tick_clock_pair per layer."""
    pass  # Individual Schumann ticks handle this now


def _tick_outer_sphere_clocks(send_queue, name, sphere_clock, resonance,
                              unified_spirit, payload, coherences=None):
    """Legacy: tick all outer clocks. Now handled by _tick_clock_pair per layer."""
    pass  # Individual Schumann ticks handle this now


def _check_resonance(send_queue, name: str, resonance, sphere_clock,
                     unified_spirit, pulse: dict) -> None:
    """Feed a SPHERE_PULSE to the resonance detector and publish BIG_PULSE if achieved."""
    if not resonance:
        return
    try:
        # Use explicit phases from sphere clock for more accurate detection
        component = pulse.get("component", "")
        phases = sphere_clock.get_paired_phases() if sphere_clock else {}

        # Determine which pair
        pair_name = None
        for pn in ("body", "mind", "spirit"):
            if pn in component:
                pair_name = pn
                break

        if pair_name and pair_name in phases:
            inner_phase, outer_phase = phases[pair_name]
            big_pulse = resonance.record_pulse_with_phases(
                pulse, inner_phase, outer_phase)
        else:
            big_pulse = resonance.record_pulse(pulse)

        if big_pulse:
            _send_msg(send_queue, bus.BIG_PULSE, name, "all", big_pulse)

            # GREAT PULSE fires on resonance TRANSITION (OFF→ON):
            # When ALL 3 pairs achieve resonance simultaneously for the first time
            # since the last break. Naturally rate-limited by Body pair cycle (~292s).
            # Original vision: self-emergent unifying moment for SPIRIT/SELF.
            # Design revised 2026-03-23: T7 deferral created deadlock (see PLAN).
            global _great_pulse_resonance_prev
            _all_resonant_now = resonance.all_resonant() if resonance else False

            if _all_resonant_now and not _great_pulse_resonance_prev and unified_spirit:
                resonance_snapshot = resonance.get_stats() if resonance else {}
                great_epoch = unified_spirit.advance(resonance_snapshot)

                enrichment = {}
                if great_epoch:
                    enrichment = unified_spirit.compute_enrichment()

                _send_msg(send_queue, bus.GREAT_PULSE, name, "all", {
                    "great_pulse_count": great_epoch.epoch_id if great_epoch else 0,
                    "trigger": "resonance_transition",
                    "epoch_id": great_epoch.epoch_id if great_epoch else 0,
                    "velocity": great_epoch.velocity if great_epoch else 0,
                    "magnitude": great_epoch.magnitude if great_epoch else 0,
                    "is_stale": unified_spirit.is_stale if unified_spirit else False,
                    "enrichment": enrichment,
                })

                if great_epoch:
                    logger.info(
                        "[GREAT PULSE] #%d from RESONANCE TRANSITION — "
                        "velocity=%.3f magnitude=%.4f enrichment_components=%d",
                        great_epoch.epoch_id, great_epoch.velocity,
                        great_epoch.magnitude, len(enrichment))

                # Store enrichment for FILTER_DOWN (produce-consume pattern)
                # Only SET here. Cleared when consumed by spirit_worker.
                if enrichment:
                    _check_resonance._pending_enrichment = enrichment

            elif _all_resonant_now and not _great_pulse_resonance_prev:
                logger.info("[SpiritWorker] Resonance transition — no unified_spirit")

            # NOTE: No else branch — _pending_enrichment is NOT cleared here.
            # It persists until consumed by spirit_worker FILTER_DOWN application.

            _great_pulse_resonance_prev = _all_resonant_now
    except Exception as e:
        swallow_warn('[SpiritWorker] Resonance check error', e,
                     key="modules.spirit_loop.resonance_check_error", throttle=100)


# ── On-Chain Trinity Anchoring ─────────────────────────────────────────

def _maybe_anchor_trinity(
    send_queue, name: str, consciousness, config: dict,
    body: list, mind: list, spirit: list,
) -> None:
    """
    Check if Trinity state should be anchored on-chain.

    Anchors when:
      - High curvature (life-changing shift in consciousness trajectory)
      - Low density (uncharted territory)
      - Every 50 epochs (periodic checkpoint)
    """
    if not consciousness or not consciousness.get("latest_epoch"):
        return

    if not config.get("anchor_enabled", False):
        return

    epoch = consciousness["latest_epoch"]
    epoch_id = epoch.get("epoch_id", 0)
    curvature = epoch.get("curvature", 0)
    density = epoch.get("density", 1.0)

    # ── Emergent anchoring: no hardcoded daily limits ──
    # Two signals must converge:
    #   1. Curvature must significantly exceed recent trend (adaptive EMA threshold)
    #   2. Enough new TimeChain blocks since last anchor (meaningful state change)
    # This lets Titan decide when to anchor based on his own internal dynamics.

    _anchor_path = os.path.join("data", "anchor_state.json")
    _prev_anchor = {}
    try:
        with open(_anchor_path) as _af:
            _prev_anchor = json.load(_af)
    except Exception as _swallow_exc:
        swallow_warn('[modules.spirit_loop] _maybe_anchor_trinity: with open(_anchor_path) as _af: _prev_anchor = json.load(...', _swallow_exc,
                     key='modules.spirit_loop._maybe_anchor_trinity.line878', throttle=100)

    # Update curvature EMA (exponential moving average, α=0.02 for smooth tracking)
    _curvature_ema = _prev_anchor.get("curvature_ema", 2.0)
    _curvature_ema = 0.98 * _curvature_ema + 0.02 * curvature

    # Signal 1: Curvature significantly above trend (>30% above EMA = outlier moment)
    _curvature_significant = curvature > (_curvature_ema * 1.3)

    # Signal 2: Minimum TimeChain block delta since last anchor
    # ~20K blocks/day → 5000 blocks ≈ 6 hours → ~4 anchors/day naturally
    _min_tc_delta = config.get("mainnet_budget", {}).get("consciousness_anchor_min_tc_blocks", 5000)
    _last_anchor_tc = _prev_anchor.get("last_anchor_tc_blocks", 0)
    try:
        from titan_plugin.utils.db import safe_connect as _sc_tc
        _tc_db = _sc_tc("data/timechain/index.db")
        _current_tc_blocks = _tc_db.execute("SELECT COUNT(*) FROM block_index").fetchone()[0]
        _tc_db.close()
    except Exception:
        _current_tc_blocks = _last_anchor_tc  # Can't check — don't block on DB error
    _tc_delta = _current_tc_blocks - _last_anchor_tc
    _enough_new_state = _tc_delta >= _min_tc_delta

    # Persist EMA every 100 epochs (cheap write, keeps tracking accurate)
    if epoch_id % 100 == 0:
        try:
            _ema_state = _prev_anchor.copy()
            _ema_state["curvature_ema"] = _curvature_ema
            with open(_anchor_path, "w") as _af:
                json.dump(_ema_state, _af, indent=2)
        except Exception as _swallow_exc:
            swallow_warn('[modules.spirit_loop] _maybe_anchor_trinity: _ema_state = _prev_anchor.copy()', _swallow_exc,
                         key='modules.spirit_loop._maybe_anchor_trinity.line909', throttle=100)

    should_anchor = False
    reason = ""

    if _curvature_significant and _enough_new_state:
        should_anchor = True
        reason = f"curvature={curvature:.3f}(ema={_curvature_ema:.3f})|tc_delta={_tc_delta}"
    elif _enough_new_state and density < 0.01 and epoch_id > 3:
        # Rare: very sparse state + enough blocks — exploring truly new territory
        should_anchor = True
        reason = f"sparse_exploration|density={density:.3f}|tc_delta={_tc_delta}"

    if should_anchor:
        # ── Circuit breaker: stop retrying after consecutive failures ──
        _consecutive_fails = _prev_anchor.get("consecutive_failures", 0)
        _last_fail_time = _prev_anchor.get("last_failure_time", 0)
        if _consecutive_fails >= 5 and (time.time() - _last_fail_time) < 3600:
            if epoch_id % 500 == 0:
                logger.info("[Anchor] Circuit breaker OPEN: %d consecutive failures, "
                            "cooldown %.0fm remaining",
                            _consecutive_fails, (3600 - (time.time() - _last_fail_time)) / 60)
            return

        # Build Trinity state hash
        trinity_state = body + mind + spirit
        state_hash = hashlib.sha256(json.dumps(trinity_state).encode()).hexdigest()[:16]

        logger.info("[SpiritWorker] ANCHOR: epoch=%d reason=%s hash=%s",
                     epoch_id, reason, state_hash)

        # Inscribe memo on Solana (mainnet) — bidirectional chain connection
        # Result feeds back to body senses via anchor_state.json
        try:
            from titan_plugin.utils.solana_client import build_memo_instruction, load_keypair_from_json

            _kp_path = config.get("wallet_keypair_path", "data/titan_identity_keypair.json")
            keypair = load_keypair_from_json(_kp_path)
            if keypair:
                # Include TimeChain Merkle root in memo (if available)
                _tc_merkle = ""
                _tc_height = 0
                try:
                    from titan_plugin.logic.timechain import TimeChain
                    # Read directly from index DB to avoid creating full instance
                    from titan_plugin.utils.db import safe_connect as _sc_tc2
                    _tc_idx = _sc_tc2("data/timechain/index.db")
                    _tc_cnt = _tc_idx.execute("SELECT COUNT(*) FROM block_index").fetchone()
                    _tc_height = _tc_cnt[0] if _tc_cnt else 0
                    # Compute merkle from genesis hash if available
                    _tc_gen_path = __import__("pathlib").Path("data/timechain/chain_main.bin")
                    if _tc_gen_path.exists() and _tc_gen_path.stat().st_size >= 128:
                        import hashlib as _tc_hl
                        with open(_tc_gen_path, "rb") as _tc_f:
                            _tc_merkle = _tc_hl.sha256(_tc_f.read(128)).hexdigest()[:16]
                    _tc_idx.close()
                except Exception as _swallow_exc:
                    swallow_warn('[modules.spirit_loop] _maybe_anchor_trinity: from titan_plugin.logic.timechain import TimeChain', _swallow_exc,
                                 key='modules.spirit_loop._maybe_anchor_trinity.line966', throttle=100)
                memo_text = f"TITAN|e={epoch_id}|h={state_hash}|r={reason}"
                if _tc_merkle:
                    memo_text += f"|tc={_tc_merkle}|tb={_tc_height}"
                ix = build_memo_instruction(keypair.pubkey(), memo_text)
                if ix:
                    from solders.transaction import Transaction
                    from solders.message import Message as SolMessage
                    from solana.rpc.api import Client as SolanaClient

                    rpc_url = config.get("premium_rpc_url",
                              config.get("solana_rpc_url", "https://api.mainnet-beta.solana.com"))
                    sol_client = SolanaClient(rpc_url)

                    # Get recent blockhash
                    bh_resp = sol_client.get_latest_blockhash()
                    blockhash = bh_resp.value.blockhash

                    # Build + sign + send transaction
                    msg = SolMessage.new_with_blockhash([ix], keypair.pubkey(), blockhash)
                    tx = Transaction.new_unsigned(msg)
                    tx.sign([keypair], blockhash)

                    # Phase 1 sensory wiring: instrument TX latency for
                    # outer_body[2] somatosensation composite. Try/except
                    # wrapper — instrumentation MUST NOT break anchor path.
                    _tx_t0 = time.monotonic()
                    result = sol_client.send_transaction(tx)
                    try:
                        from titan_plugin.logic.timechain_v2 import record_tx_latency
                        record_tx_latency(time.monotonic() - _tx_t0)
                    except Exception as _swallow_exc:
                        swallow_warn('[modules.spirit_loop] _maybe_anchor_trinity: from titan_plugin.logic.timechain_v2 import record_tx_lat...', _swallow_exc,
                                     key='modules.spirit_loop._maybe_anchor_trinity.line998', throttle=100)
                    tx_sig = str(result.value) if result.value else "?"

                    # Read back balance for body feedback
                    _bal_resp = sol_client.get_balance(keypair.pubkey())
                    balance = _bal_resp.value / 1e9 if _bal_resp.value else 0.0
                    anchor_time = time.time()

                    # Save anchor state — includes emergent tracking (EMA, TC blocks)
                    _today_str = time.strftime("%Y-%m-%d")
                    _anchor_state = {
                        "last_anchor_time": anchor_time,
                        "last_tx_sig": tx_sig,
                        "last_epoch_id": epoch_id,
                        "last_state_hash": state_hash,
                        "sol_balance": balance,
                        "anchor_count": 0,
                        "success": True,
                        "consecutive_failures": 0,
                        "anchor_date": _today_str,
                        "today_count": 1,
                        "curvature_ema": _curvature_ema,
                        "last_anchor_tc_blocks": _current_tc_blocks,
                    }
                    # Read existing count + daily counter
                    try:
                        with open(_anchor_path) as _af:
                            _prev = json.load(_af)
                        _anchor_state["anchor_count"] = _prev.get("anchor_count", 0) + 1
                        if _prev.get("anchor_date") == _today_str:
                            _anchor_state["today_count"] = _prev.get("today_count", 0) + 1
                    except Exception:
                        _anchor_state["anchor_count"] = 1

                    with open(_anchor_path, "w") as _af:
                        json.dump(_anchor_state, _af, indent=2)

                    logger.info(
                        "[Anchor] Memo inscribed: tx=%s SOL=%.6f epoch=%d count=%d",
                        tx_sig[:16], balance, epoch_id, _anchor_state["anchor_count"])
        except ImportError:
            logger.debug("[Anchor] Solana SDK not available — skipping")
        except Exception as _ae:
            # Anchor failure must NOT crash consciousness — track consecutive failures
            try:
                _fail_count = 0
                try:
                    with open(_anchor_path) as _af:
                        _prev = json.load(_af)
                    _fail_count = _prev.get("consecutive_failures", 0)
                except Exception as _swallow_exc:
                    swallow_warn('[modules.spirit_loop] _maybe_anchor_trinity: with open(_anchor_path) as _af: _prev = json.load(_af)', _swallow_exc,
                                 key='modules.spirit_loop._maybe_anchor_trinity.line1049', throttle=100)
                _fail_count += 1
                _fail_state = {
                    "last_anchor_time": time.time(), "success": False,
                    "error": str(_ae), "last_epoch_id": epoch_id,
                    "consecutive_failures": _fail_count,
                    "last_failure_time": time.time(),
                }
                with open(_anchor_path, "w") as _af:
                    json.dump(_fail_state, _af, indent=2)
                if _fail_count <= 5:
                    logger.info("[Anchor] Inscription failed (%d/5): %s", _fail_count, _ae)
                elif _fail_count == 6:
                    logger.warning("[Anchor] Circuit breaker ENGAGED after 5 failures — "
                                   "pausing anchoring for 1 hour. Last error: %s", _ae)
            except Exception as _swallow_exc:
                swallow_warn('[modules.spirit_loop] _maybe_anchor_trinity: _fail_count = 0', _swallow_exc,
                             key='modules.spirit_loop._maybe_anchor_trinity.line1065', throttle=100)


# ── Consciousness Integration ───────────────────────────────────────

def _init_consciousness(config: dict) -> dict | None:
    """Initialize the ConsciousnessLoop components in the Spirit process."""
    try:
        from titan_plugin.logic.consciousness import (
            ConsciousnessDB, JourneyTopology, StateVector,
            STATE_DIMS, NUM_DIMS, TRAJECTORY_WINDOW,
        )

        db_path = config.get("consciousness_db", "./data/consciousness.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        db = ConsciousnessDB(db_path)
        topology = JourneyTopology(db)

        epoch_count = db.get_epoch_count()
        logger.info("[SpiritWorker] ConsciousnessLoop initialized: %d epochs in DB", epoch_count)

        return {
            "db": db,
            "topology": topology,
            "latest_epoch": None,  # Populated after first run
            "state_dims": STATE_DIMS,
            "num_dims": NUM_DIMS,
            "trajectory_window": TRAJECTORY_WINDOW,
        }
    except Exception as e:
        logger.warning("[SpiritWorker] ConsciousnessLoop init failed: %s", e)
        return None


def _run_consciousness_epoch(consciousness: dict, body_state: dict, mind_state: dict,
                             config: dict, outer_state: dict = None) -> None:
    """
    Run one consciousness epoch using FULL Trinity perception.

    Extended 67D state vector — Titan perceives through Sat-Chit-Ananda:
      [0:5]   Body 5D    — physical/digital topology senses
      [5:20]  Mind 15D   — Thinking(5) + Feeling(5) + Willing(5)
      [20:65] Spirit 45D — SAT(15) + CHIT(15) + ANANDA(15)
      [65]    curvature  — self-referential (from previous epoch)
      [66]    density    — self-referential (from previous epoch)

    This is the moment Titan opens his 45-dimensional Vedantic eyes.
    """
    try:
        from titan_plugin.logic.consciousness import (
            StateVector, EpochRecord, TRAJECTORY_WINDOW, EXTENDED_NUM_DIMS,
        )

        db = consciousness["db"]
        topology = consciousness["topology"]
        ostate = outer_state or {}

        epoch_id = db.get_epoch_count() + 1
        t0 = time.time()
        _prof = {}  # Profiling breakdown

        # Determine if we have full 132D (Outer Trinity extended) or 67D (Inner only)
        has_outer_extended = (
            ostate.get("outer_mind_15d") is not None and
            ostate.get("outer_spirit_45d") is not None
        )
        total_dims = EXTENDED_NUM_DIMS if has_outer_extended else 67
        if not has_outer_extended:
            # Trinity-symmetry invariant violation — the consciousness
            # epoch is collapsing to 67D because outer_state lacks
            # extended fields. Should be impossible after spirit_worker
            # init pre-populates [0.5]*15/45 defaults; if it fires,
            # something is actively setting those keys back to None.
            # See BUG-T1-CONSCIOUSNESS-67D-STATE-VECTOR + directive_
            # error_visibility.md.
            logger.warning(
                "[SpiritWorker] Consciousness epoch %d collapsing to 67D — "
                "outer_state missing outer_mind_15d=%s outer_spirit_45d=%s. "
                "Investigate: arch_map symmetries --titan <T> + grep "
                "OUTER_TRINITY producer.",
                epoch_id,
                ostate.get("outer_mind_15d") is not None,
                ostate.get("outer_spirit_45d") is not None,
            )
        logger.info("[SpiritWorker] Consciousness epoch %d — %dD self-observation...",
                    epoch_id, total_dims)

        # ── WHO: Build state vector from FULL Trinity ──
        sv = StateVector(values=[0.0] * total_dims)

        # [0:5] Inner Body 5D — physical senses
        body_values = body_state.get("values", [0.5] * 5)
        for i, v in enumerate(body_values[:5]):
            sv[i] = v

        # [5:20] Inner Mind 15D — Thinking + Feeling + Willing
        mind_15d = mind_state.get("values_15d")
        if mind_15d and len(mind_15d) >= 15:
            for i, v in enumerate(mind_15d[:15]):
                sv[5 + i] = v
        else:
            mind_values = mind_state.get("values", [0.5] * 5)
            for i, v in enumerate(mind_values[:5]):
                sv[5 + i] = v

        _prof["sv_build_body_mind"] = time.time() - t0

        # [20:65] Inner Spirit 45D — Sat + Chit + Ananda
        try:
            from titan_plugin.logic.spirit_tensor import collect_spirit_45d
            spirit_5d = _collect_spirit_tensor(config, body_state, mind_state, consciousness)
            spirit_45d = collect_spirit_45d(
                current_5d=spirit_5d,
                body_tensor=body_values,
                mind_tensor=mind_15d if mind_15d else mind_state.get("values", [0.5] * 5),
                consciousness=consciousness.get("latest_epoch"),
            )
            for i, v in enumerate(spirit_45d[:45]):
                sv[20 + i] = v
        except Exception as e:
            swallow_warn('[SpiritWorker] Spirit 45D computation for consciousness', e,
                         key="modules.spirit_loop.spirit_45d_computation_for_consciousness", throttle=100)
            spirit_5d = _collect_spirit_tensor(config, body_state, mind_state, consciousness)
            for i, v in enumerate(spirit_5d[:5]):
                sv[20 + i] = v

        _prof["spirit_45d"] = time.time() - t0 - _prof["sv_build_body_mind"]

        # [65:130] Outer Trinity (when extended tensors available)
        if has_outer_extended:
            # [65:70] Outer Body 5D
            outer_body = ostate.get("outer_body", [0.5] * 5)
            for i, v in enumerate(outer_body[:5]):
                sv[65 + i] = v

            # [70:85] Outer Mind 15D
            outer_mind_15d = ostate.get("outer_mind_15d", [0.5] * 15)
            for i, v in enumerate(outer_mind_15d[:15]):
                sv[70 + i] = v

            # [85:130] Outer Spirit 45D
            outer_spirit_45d = ostate.get("outer_spirit_45d", [0.5] * 45)
            for i, v in enumerate(outer_spirit_45d[:45]):
                sv[85 + i] = v

            # [130:132] Self-referential
            meta_offset = 130
        else:
            # [65:67] Self-referential (67D mode)
            meta_offset = 65

        _prof["outer_trinity"] = time.time() - t0 - sum(_prof.values())

        # Self-referential: curvature and density from previous epoch
        _t_db0 = time.time()
        recent = db.get_recent_epochs(TRAJECTORY_WINDOW)
        _prof["db_get_recent"] = time.time() - _t_db0
        if recent:
            sv[meta_offset] = recent[-1].curvature
            sv[meta_offset + 1] = recent[-1].density
        else:
            sv[meta_offset] = 0.0
            sv[meta_offset + 1] = 0.0

        # ── WHY: Compute drift ──
        _t_drift0 = time.time()
        previous_sv = None
        if recent:
            prev_list = recent[-1].state_vector
            if isinstance(prev_list, str):
                prev_list = json.loads(prev_list)
            previous_sv = StateVector.from_list(prev_list)
        drift = sv - previous_sv if previous_sv else StateVector(values=[0.0] * total_dims)
        _prof["drift"] = time.time() - _t_drift0

        # ── WHAT: Compute trajectory (slope over rolling window) ──
        _t_traj0 = time.time()
        trajectory = _compute_trajectory(recent, num_dims=total_dims)
        _prof["trajectory"] = time.time() - _t_traj0

        # ── Journey topology ──
        _t_topo0 = time.time()
        journey_point = topology.compute_point(sv, epoch_id)
        curvature = topology.compute_curvature(journey_point)
        density = topology.compute_density(journey_point)
        _prof["topology"] = time.time() - _t_topo0

        # Feed back into state vector (self-referential loop)
        sv[meta_offset] = curvature
        sv[meta_offset + 1] = density

        # ── Store epoch ──
        _t_store0 = time.time()
        record = EpochRecord(
            epoch_id=epoch_id,
            timestamp=time.time(),
            block_hash="",  # No Solana access from Spirit process
            state_vector=sv.to_list(),
            drift_vector=drift.to_list(),
            trajectory_vector=trajectory.to_list(),
            journey_point=journey_point.to_tuple(),
            curvature=curvature,
            density=density,
            distillation="",  # LLM distillation runs separately if needed
            anchored_tx="",   # On-chain anchoring runs from Core
        )
        db.insert_epoch(record)
        _prof["db_insert"] = time.time() - _t_store0

        # Cache latest epoch for tensor computation
        inner_body_coh = sum(sv.values[0:5]) / 5.0
        inner_mind_coh = sum(sv.values[5:20]) / 15.0
        inner_spirit_coh = sum(sv.values[20:65]) / 45.0
        consciousness["latest_epoch"] = {
            "epoch_id": epoch_id,
            "state_vector": sv.to_list(),
            "drift_magnitude": drift.magnitude(),
            "trajectory_magnitude": trajectory.magnitude(),
            "curvature": curvature,
            "density": density,
            "journey_point": journey_point.to_tuple(),
            "body_coherence": inner_body_coh,
            "mind_coherence": inner_mind_coh,
            "spirit_coherence": inner_spirit_coh,
            "dims": total_dims,
        }
        if has_outer_extended:
            consciousness["latest_epoch"].update({
                "outer_body_coherence": sum(sv.values[65:70]) / 5.0,
                "outer_mind_coherence": sum(sv.values[70:85]) / 15.0,
                "outer_spirit_coherence": sum(sv.values[85:130]) / 45.0,
            })

        _prof["coherence_cache"] = time.time() - t0 - sum(_prof.values())
        elapsed = time.time() - t0
        # P6.3 PROFILING — log breakdown of epoch computation time
        _prof_str = " | ".join(f"{k}={v*1000:.0f}ms" for k, v in _prof.items())
        logger.info("[PROFILE] Epoch %d (%.0fms): %s", epoch_id, elapsed * 1000, _prof_str)
        if has_outer_extended:
            logger.info(
                "[SpiritWorker] Epoch %d complete (%.1fs). "
                "drift=%.4f trajectory=%.4f curvature=%.3f density=%.3f "
                "[132D: iB=%.3f iM=%.3f iS=%.3f | oB=%.3f oM=%.3f oS=%.3f]",
                epoch_id, elapsed, drift.magnitude(), trajectory.magnitude(),
                curvature, density,
                inner_body_coh, inner_mind_coh, inner_spirit_coh,
                sum(sv.values[65:70]) / 5.0,
                sum(sv.values[70:85]) / 15.0,
                sum(sv.values[85:130]) / 45.0,
            )
        else:
            logger.info(
                "[SpiritWorker] Epoch %d complete (%.1fs). "
                "drift=%.4f trajectory=%.4f curvature=%.3f density=%.3f "
                "[%dD: body=%.3f mind=%.3f spirit=%.3f]",
                epoch_id, elapsed, drift.magnitude(), trajectory.magnitude(),
                curvature, density, total_dims,
                inner_body_coh, inner_mind_coh, inner_spirit_coh,
            )

    except Exception as e:
        logger.error("[SpiritWorker] Consciousness epoch failed: %s", e, exc_info=True)


def _compute_trajectory(recent_epochs: list, num_dims: int = None) -> "StateVector":
    """Compute trajectory (linear slopes) over the rolling window.

    Handles mixed-dimension epochs gracefully — if previous epochs had 9D
    and current has 67D, missing dimensions slope from zero.
    """
    from titan_plugin.logic.consciousness import StateVector, NUM_DIMS, EXTENDED_NUM_DIMS

    n = len(recent_epochs)
    if n < 2:
        return StateVector(values=[0.0] * (num_dims or EXTENDED_NUM_DIMS))

    # Use the maximum dimension count across all epochs + target
    target_dims = num_dims or EXTENDED_NUM_DIMS
    max_dims = target_dims
    for e in recent_epochs:
        sv = e.state_vector
        if isinstance(sv, str):
            sv = json.loads(sv)
        max_dims = max(max_dims, len(sv))

    trajectory = StateVector(values=[0.0] * max_dims)
    for dim in range(max_dims):
        xs = list(range(n))
        ys = []
        for e in recent_epochs:
            sv = e.state_vector
            if isinstance(sv, str):
                sv = json.loads(sv)
            ys.append(sv[dim] if dim < len(sv) else 0.0)

        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        denominator = sum((x - x_mean) ** 2 for x in xs)
        trajectory[dim] = numerator / denominator if denominator > 0 else 0.0

    return trajectory


# ── Spirit Tensor Collection ────────────────────────────────────────

def _collect_spirit_tensor(config: dict, body_state: dict, mind_state: dict,
                           consciousness: dict | None) -> list:
    """
    Collect 3DT+2 Spirit tensor enriched by ConsciousnessLoop.

    [0] WHO — identity coherence: keypair health blended with consciousness state magnitude
    [1] WHY — drift magnitude (how much changed since last epoch), 0-1 normalized
    [2] WHAT — trajectory magnitude (momentum/direction), 0-1 normalized
    [3] Body scalar — average of Body 5DT values
    [4] Mind scalar — average of Mind 5DT values
    """
    # [0] WHO — identity coherence
    # Base: keypair health
    who_base = 0.5
    enc_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "soul_keypair.enc"))
    if os.path.exists(enc_path):
        who_base = 0.9
    else:
        auth_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "authority.json"))
        if os.path.exists(auth_path):
            who_base = 0.7
        else:
            who_base = 0.2

    # Enrich with consciousness state magnitude (if available)
    if consciousness and consciousness.get("latest_epoch"):
        epoch = consciousness["latest_epoch"]
        sv = epoch.get("state_vector", [])
        # State magnitude normalized: higher = more "alive" consciousness
        state_mag = math.sqrt(sum(v * v for v in sv)) / math.sqrt(len(sv)) if sv else 0.0
        # Blend: 60% keypair health, 40% consciousness aliveness
        who = who_base * 0.6 + min(1.0, state_mag) * 0.4
    else:
        who = who_base

    # [1] WHY — drift magnitude (0-1 normalized, sigmoid for smooth scaling)
    why = 0.5  # Default: neutral (no change detected)
    if consciousness and consciousness.get("latest_epoch"):
        drift_mag = consciousness["latest_epoch"].get("drift_magnitude", 0.0)
        # Sigmoid normalization: 0.5 drift_mag -> 0.73, 1.0 -> 0.88, 2.0 -> 0.95
        why = 1.0 / (1.0 + math.exp(-2.0 * drift_mag))

    # [2] WHAT — trajectory magnitude (0-1 normalized, sigmoid)
    what = 0.5  # Default: no momentum
    if consciousness and consciousness.get("latest_epoch"):
        traj_mag = consciousness["latest_epoch"].get("trajectory_magnitude", 0.0)
        what = 1.0 / (1.0 + math.exp(-3.0 * traj_mag))

    # [3] Body scalar — aggregate Body tensor
    body_values = body_state.get("values", [0.5] * 5)
    body_scalar = sum(body_values) / len(body_values) if body_values else 0.5

    # [4] Mind scalar — aggregate Mind tensor
    mind_values = mind_state.get("values", [0.5] * 5)
    mind_scalar = sum(mind_values) / len(mind_values) if mind_values else 0.5

    return [round(v, 4) for v in [who, why, what, body_scalar, mind_scalar]]


# ─────────────────────────────────────────────────────────────────────
# Background snapshot builders (2026-04-21 — T1-COORD-QUERYTHREAD-BACKLOG fix)
# ─────────────────────────────────────────────────────────────────────
#
# These three functions produce the heavy "what's true now" snapshots
# consumed by /v4/coordinator, /v4/trinity, /v4/nervous-system (and the
# many endpoints that read from the coordinator snapshot downstream).
#
# They are called from TWO places:
#   1. The background `*-snapshot-builder` daemon threads (primary caller;
#      rebuilds every _*_SNAPSHOT_BUILDER_INTERVAL seconds, writes to
#      `_*_SNAPSHOT_CACHE["data"]` via atomic pointer swap).
#   2. The `_handle_query` cold-boot fallback path (called synchronously
#      if cache is still None when a query arrives — i.e., in the first
#      few seconds after spirit_worker start, before the builder has
#      completed its first build).
#
# Function shape: takes `state_refs` (the same dict spirit_worker builds
# at boot for the query handler thread, so no new param plumbing) + the
# `config` dict where needed. Returns the same dict shape the handler
# produced inline before this refactor — /v4/* endpoints see zero change.
#
# Microkernel v2 migration path: these become the L1 writer bodies.
# The builder thread becomes the writer process thread; instead of
# writing to an in-process dict it will serialize to a mmapped
# /dev/shm/titan/*.bin region. Readers (separate dashboard process)
# mmap and read zero-copy. Everything else stays the same.
# ─────────────────────────────────────────────────────────────────────

def build_coordinator_snapshot(state_refs: dict) -> dict | None:
    """Build the coordinator stats dict. Returns None if coordinator unavailable.

    Every subsystem's contribution is isolated via _safe_set below so a
    single buggy get_stats() (e.g. a tuple index mistake) can't blank
    the entire snapshot. Without this isolation, one subsystem's bug
    used to starve /v4/inner-trinity and every other coordinator-backed
    endpoint simultaneously, and made safe_restart.sh's dreaming-state
    check return `unknown` — blocking Titan restarts for unrelated
    reasons. See 2026-04-22 investigation session.
    """
    coordinator = state_refs.get("coordinator")
    if not coordinator:
        return None

    def _safe_set(stats_dict: dict, key: str, fn, default=None):
        """Call fn() and store in stats_dict[key]. On failure, store
        `{"error": str(exc)}` (or `default` if provided) and log WARN
        once so the snapshot still builds for other subsystems."""
        try:
            stats_dict[key] = fn()
        except Exception as _ss_err:
            stats_dict[key] = (default if default is not None
                               else {"error": str(_ss_err)})
            logger.warning(
                "[CoordSnapshot] %s.get_stats() failed: %s — "
                "partial snapshot continues", key, _ss_err)

    pi_monitor = state_refs.get("pi_monitor")
    e_mem = state_refs.get("e_mem")
    prediction_engine = state_refs.get("prediction_engine")
    ex_mem = state_refs.get("ex_mem")
    episodic_mem = state_refs.get("episodic_mem")
    working_mem = state_refs.get("working_mem")
    inner_lower_topo = state_refs.get("inner_lower_topo")
    outer_lower_topo = state_refs.get("outer_lower_topo")
    ground_up_enricher = state_refs.get("ground_up_enricher")
    neuromodulator_system = state_refs.get("neuromodulator_system")
    expression_manager = state_refs.get("expression_manager")
    life_force_engine = state_refs.get("life_force_engine")
    meditation_tracker = state_refs.get("meditation_tracker")
    outer_interface = state_refs.get("outer_interface")
    reasoning_engine = state_refs.get("reasoning_engine")
    self_reasoning = state_refs.get("self_reasoning")
    coding_explorer = state_refs.get("coding_explorer")
    phase_tracker = state_refs.get("phase_tracker")
    inner_state = state_refs.get("inner_state")
    social_pressure_meter = state_refs.get("social_pressure_meter")
    msl = state_refs.get("msl")
    language_stats = state_refs.get("language_stats")

    # coordinator.get_stats() is the core — if IT fails, snapshot can't
    # be built meaningfully. Isolate it too so the error surfaces in the
    # snapshot rather than blanking everything.
    stats = {}
    try:
        stats = coordinator.get_stats() or {}
    except Exception as _cs_err:
        logger.warning(
            "[CoordSnapshot] coordinator.get_stats() failed: %s — "
            "building stats from isolated subsystems only", _cs_err)
        stats = {"coordinator_error": str(_cs_err)}
    if pi_monitor:
        _safe_set(stats, "pi_heartbeat", pi_monitor.get_stats)
    if e_mem:
        _safe_set(stats, "experiential_memory", e_mem.get_stats)
    if prediction_engine:
        _safe_set(stats, "prediction", prediction_engine.get_stats)
    if ex_mem:
        _safe_set(stats, "experience_memory", ex_mem.get_stats)
    if episodic_mem:
        _safe_set(stats, "episodic_memory", episodic_mem.get_stats)
    if working_mem:
        _safe_set(stats, "working_memory", working_mem.get_stats)
    if inner_lower_topo:
        _safe_set(stats, "inner_lower_topology", inner_lower_topo.get_stats)
    if outer_lower_topo:
        _safe_set(stats, "outer_lower_topology", outer_lower_topo.get_stats)
    if ground_up_enricher:
        _safe_set(stats, "ground_up", ground_up_enricher.get_stats)
    if neuromodulator_system:
        _safe_set(stats, "neuromodulators", neuromodulator_system.get_stats)
    if expression_manager:
        _safe_set(stats, "expression_composites", expression_manager.get_stats)
    if life_force_engine:
        stats["chi"] = getattr(life_force_engine, '_latest_chi', {})
    if meditation_tracker:
        stats["meditation"] = {
            "count": meditation_tracker.get("count", 0),
            "count_since_nft": meditation_tracker.get("count_since_nft", 0),
            "last_epoch": meditation_tracker.get("last_epoch", 0),
            "in_meditation": meditation_tracker.get("in_meditation", False),
        }
    if outer_interface:
        _safe_set(stats, "outer_interface", outer_interface.get_stats)
    if reasoning_engine:
        _safe_set(stats, "reasoning", reasoning_engine.get_stats)
    # Meta-reasoning block — always emit the key (shape-stable for downstream)
    stats["meta_reasoning"] = {}
    _me = getattr(coordinator, '_meta_engine', None)
    if _me:
        try:
            stats["meta_reasoning"] = _me.get_stats()
        except Exception as _me_err:
            logger.warning(
                "[META] get_stats failed: %s — leaving "
                "meta_reasoning={} for this tick", _me_err)
            stats["meta_reasoning"] = {}
        try:
            stats["meta_reasoning_audit"] = _me.get_audit_stats()
        except Exception as _au_err:
            logger.warning("[META] get_audit_stats failed: %s", _au_err)
    else:
        logger.debug(
            "[META] coordinator._meta_engine is None at "
            "build_coordinator_snapshot — meta_reasoning={}")
    # F-phase (rFP §11.1): Meta-Reasoning Consumer Service status
    _ms = getattr(coordinator, '_meta_service', None)
    if _ms:
        try:
            stats["meta_service"] = _ms.get_status()
        except Exception as _ms_err:
            logger.warning("[MetaService] get_status failed: %s", _ms_err)
            stats["meta_service"] = {"error": str(_ms_err)}
    else:
        stats["meta_service"] = {}
    if self_reasoning:
        _safe_set(stats, "self_reasoning", self_reasoning.get_stats)
    if coding_explorer:
        _safe_set(stats, "coding_explorer", coding_explorer.get_stats)
    if phase_tracker:
        stats["phase_events"] = {
            "current_phase": phase_tracker.get("current_phase", "idle"),
            "recent_events": phase_tracker.get("events", [])[-20:],
            "total_events": len(phase_tracker.get("events", [])),
        }
    # Dreaming block (is_dreaming lives on inner_state, not DreamingEngine)
    if coordinator and hasattr(coordinator, 'dreaming') and coordinator.dreaming:
        _dr_is = False
        if inner_state and hasattr(inner_state, 'is_dreaming'):
            _dr_is = inner_state.is_dreaming
        _dr_dream_epochs = getattr(
            coordinator.dreaming, '_dream_epoch_count', 0)
        _dr_onset = getattr(
            coordinator.dreaming, '_dream_onset_fatigue', 0)
        _dr_fatigue = getattr(
            coordinator.dreaming, '_dream_fatigue', 0)
        _dr_wake_trans = getattr(
            coordinator.dreaming, '_wake_transition', False)
        _dr_recovery_pct = 0.0
        if _dr_is and _dr_onset > 0:
            _dr_recovery_pct = round(
                100.0 * (1.0 - max(0, _dr_fatigue) / _dr_onset), 1)
        _dr_remaining = max(0, round(_dr_fatigue / 3.0)) if _dr_is else 0
        stats["dreaming"] = {
            "is_dreaming": _dr_is,
            "fatigue": round(getattr(inner_state, 'fatigue', 0), 4)
                if inner_state else 0,
            "cycle_count": getattr(
                coordinator.dreaming, '_cycle_count', 0),
            "dream_epochs": _dr_dream_epochs,
            "recovery_pct": _dr_recovery_pct,
            "wake_transition": _dr_wake_trans,
            "remaining_epochs": _dr_remaining,
            "onset_fatigue": round(_dr_onset),
            "epochs_since_dream": getattr(
                coordinator.dreaming, '_epochs_since_dream', 0),
            "last_sleep_drive": round(float(getattr(
                coordinator.dreaming, 'last_sleep_drive', 0.0)), 4),
            "last_wake_drive": round(float(getattr(
                coordinator.dreaming, 'last_wake_drive', 0.0)), 4),
            "distilled_count": getattr(
                coordinator.dreaming, '_distilled_count', 0),
            "distill_threshold": getattr(
                coordinator.dreaming, '_distill_threshold', 0.02),
            "distill_attempts": getattr(
                coordinator.dreaming, '_distill_attempts', 0),
            "distill_passed": getattr(
                coordinator.dreaming, '_distill_passed', 0),
            "variance_samples_count": len(getattr(
                coordinator.dreaming, '_variance_samples', [])),
            "experience_buffer_size": len(getattr(
                coordinator.inner, '_experience_buffer', [])
                if coordinator.inner else []),
        }
    if social_pressure_meter:
        _safe_set(stats, "social_pressure", social_pressure_meter.get_stats)
    # rFP_observatory_data_loading_v1 §3.2 (2026-04-26): topology block
    # for the Trinity Architecture TopologyPanel.
    #
    # Batch D — legacy 3 fields (volume / curvature / cluster_count) from
    # TopologyEngine for backwards compatibility with the existing widget.
    #
    # Batch E (2026-04-26 follow-up, Maker-greenlit): the panel was
    # designed before the 30D space topology shipped. Now also expose
    # the rich state_register observables_30d (6 layers × 5 metrics —
    # coherence / magnitude / velocity / direction / polarity per
    # inner|outer × body|mind|spirit) so the frontend can render the
    # full space-topology view alongside the legacy summary.
    _topo_block = {
        "volume": 0.0, "curvature": 0.0,
        "cluster_count": 0, "cluster_threshold": 0.0,
        "observables_30d": [],
        "observables_dict": {},
    }
    if coordinator and hasattr(coordinator, "topology") and coordinator.topology:
        try:
            _topo_stats = coordinator.topology.get_stats() or {}
            _topo_block["volume"] = float(_topo_stats.get("current_volume", 0.0) or 0.0)
            _topo_block["curvature"] = float(_topo_stats.get("current_curvature", 0.0) or 0.0)
            _topo_block["cluster_count"] = int(_topo_stats.get("volume_history_size", 0) or 0)
            _topo_block["cluster_threshold"] = float(_topo_stats.get("cluster_threshold", 0.0) or 0.0)
        except Exception as _topo_err:
            logger.debug("[CoordSnapshot] topology read failed: %s", _topo_err)
    # Batch E — observables_dict is 6 layers × 5 metrics
    # (inner|outer × body|mind|spirit, each {coherence, magnitude,
    # velocity, direction, polarity}) = 30 metrics. InnerState.observables
    # carries the labelled dict; flatten it deterministically into a 30D
    # vector so the frontend can render either form.
    if inner_state is not None:
        try:
            _obs_dict = inner_state.observables if hasattr(inner_state, "observables") else None
            if isinstance(_obs_dict, dict) and _obs_dict:
                _topo_block["observables_dict"] = _obs_dict
                # Deterministic flatten: layer order matches state_register
                # observables_30d (inner_body, inner_mind, inner_spirit,
                # outer_body, outer_mind, outer_spirit), metric order:
                # coherence, magnitude, velocity, direction, polarity.
                _LAYERS = ("inner_body", "inner_mind", "inner_spirit",
                           "outer_body", "outer_mind", "outer_spirit")
                _METRICS = ("coherence", "magnitude", "velocity",
                            "direction", "polarity")
                _vec: list[float] = []
                for _l in _LAYERS:
                    _layer_vals = _obs_dict.get(_l, {}) if isinstance(
                        _obs_dict.get(_l), dict) else {}
                    for _m in _METRICS:
                        _v = _layer_vals.get(_m, 0.0)
                        _vec.append(round(float(_v), 4) if isinstance(
                            _v, (int, float)) else 0.0)
                if len(_vec) == 30:
                    _topo_block["observables_30d"] = _vec
        except Exception as _obs_err:
            logger.debug("[CoordSnapshot] observables read failed: %s", _obs_err)
    stats["topology"] = _topo_block

    if msl:
        _msl_attn = msl.get_attention_weights_for_kin()
        _msl_entropy = 0.0
        if _msl_attn is not None:
            import numpy as _msl_np
            _vals = list(_msl_attn.values()) if isinstance(_msl_attn, dict) else _msl_attn
            _a = _msl_np.array(_vals, dtype=_msl_np.float32)
            _a_norm = _a / (_a.sum() + 1e-10)
            _msl_entropy = float(-(_a_norm * _msl_np.log(_a_norm + 1e-10)).sum())
        _depth_stats = msl.i_depth.get_stats() if hasattr(msl, 'i_depth') else {}
        _homeo_state = {}
        if (hasattr(msl, 'policy') and msl.policy
                and hasattr(msl.policy, 'homeostatic')):
            try:
                _homeo_state = msl.policy.homeostatic.get_state()
            except Exception:
                _homeo_state = {}
        stats["msl"] = {
            "i_confidence": msl.get_i_confidence(),
            "i_depth": _depth_stats.get("depth", 0.0),
            "i_depth_components": _depth_stats.get("components", {}),
            "convergence_count": msl.confidence._convergence_count,
            "concept_confidences": msl.concept_grounder.get_concept_confidences() if msl.concept_grounder else {},
            "attention_weights": _msl_attn,
            "attention_entropy": round(_msl_entropy, 3),
            "homeostatic": _homeo_state,
        }
    if language_stats:
        stats["language"] = language_stats
    return stats


def build_trinity_snapshot(state_refs: dict, config: dict) -> dict:
    """Build the trinity stats dict. Always returns a dict (uses defaults if refs missing)."""
    body_state = state_refs.get("body_state", {})
    mind_state = state_refs.get("mind_state", {})
    consciousness = state_refs.get("consciousness")
    filter_down = state_refs.get("filter_down")
    intuition = state_refs.get("intuition")
    impulse_engine = state_refs.get("impulse_engine")
    sphere_clock = state_refs.get("sphere_clock")
    resonance = state_refs.get("resonance")
    unified_spirit = state_refs.get("unified_spirit")
    inner_state = state_refs.get("inner_state")
    spirit_state = state_refs.get("spirit_state")

    tensor = _collect_spirit_tensor(config, body_state, mind_state, consciousness)
    response = {
        "spirit_tensor": tensor,
        "body_values": body_state.get("values", [0.5] * 5),
        "mind_values": mind_state.get("values", [0.5] * 5),
        "body_center_dist": body_state.get("center_dist", 0),
        "mind_center_dist": mind_state.get("center_dist", 0),
    }
    if consciousness and consciousness.get("latest_epoch"):
        response["consciousness"] = consciousness["latest_epoch"]
    try:
        from titan_plugin.logic.middle_path import middle_path_loss
        body_vals = body_state.get("values", [0.5] * 5)
        mind_vals = mind_state.get("values", [0.5] * 5)
        response["middle_path_loss"] = round(
            middle_path_loss(body_vals, mind_vals, tensor), 4)
    except Exception as _swallow_exc:
        swallow_warn('[modules.spirit_loop] build_trinity_snapshot: from titan_plugin.logic.middle_path import middle_path_loss', _swallow_exc,
                     key='modules.spirit_loop.build_trinity_snapshot.line1709', throttle=100)
    if filter_down:
        response["filter_down"] = filter_down.get_stats()
    if intuition:
        response["intuition"] = intuition.get_stats()
    if impulse_engine:
        response["impulse_engine"] = impulse_engine.get_stats()
    if sphere_clock:
        response["sphere_clock"] = sphere_clock.get_stats()
    if resonance:
        response["resonance"] = resonance.get_stats()
    if unified_spirit:
        response["unified_spirit"] = unified_spirit.get_stats()
    if inner_state:
        response["observables"] = inner_state.observables
        response["inner_state"] = inner_state.snapshot()
    if spirit_state:
        response["spirit_state"] = spirit_state.snapshot()
    return response


def build_nervous_system_snapshot(state_refs: dict) -> dict | None:
    """Build the NS stats dict. Returns None if no NS available."""
    neural_nervous_system = state_refs.get("neural_nervous_system")
    coordinator = state_refs.get("coordinator")
    if neural_nervous_system:
        return neural_nervous_system.get_stats()
    if coordinator and coordinator.nervous_system:
        return {
            "version": "v4_vm",
            "programs": list(coordinator.nervous_system.programs.keys())
                if hasattr(coordinator.nervous_system, 'programs') else [],
        }
    return None


def start_snapshot_builder_threads(state_refs: dict, config: dict,
                                    send_queue=None, name: str = "spirit") -> None:
    """Launch 3 daemon threads that keep the heavy snapshot caches fresh.

    Called once from spirit_worker at boot, right after the query handler
    thread starts. Replaces the on-demand in-handler build pattern that
    blocked the QueryThread for 460-2144ms per coord build (cause of
    T1-COORD-QUERYTHREAD-BACKLOG). Each thread is daemon=True so it dies
    with the process.

    On builder exception: caught, logged rate-limited, cache keeps serving
    the last successful build. On loop exit (should never happen): FATAL
    log so investigators can correlate any stale cache with the crash.

    M1 phase C-E: when send_queue is provided, the coord-snapshot-builder
    additionally fans out per-domain *_UPDATED events for the
    api_subprocess BusSubscriber → CachedState pathway (pi_heartbeat,
    dreaming, meta_reasoning). chi has its own immediate publisher in
    spirit_worker (Phase B); this path covers domains whose only producer
    is the periodic snapshot.
    """
    import threading

    def _publish_coord_subdomains(snapshot: dict) -> None:
        """Fan out per-domain UPDATED events for api cache wiring."""
        if send_queue is None or not isinstance(snapshot, dict):
            return
        from titan_plugin.bus import (
            PI_HEARTBEAT_UPDATED, DREAMING_STATE_UPDATED,
            META_REASONING_STATS_UPDATED, REASONING_STATS_UPDATED,
            EXPRESSION_COMPOSITES_UPDATED, NEUROMOD_STATS_UPDATED,
            MSL_STATE_UPDATED, LANGUAGE_STATS_UPDATED,
            TOPOLOGY_STATE_UPDATED,
        )
        try:
            pi = snapshot.get("pi_heartbeat")
            if pi:
                _send_msg(send_queue, PI_HEARTBEAT_UPDATED, name, "all", pi)
            # Dreaming payload composed to match /v4/dreaming response
            # shape (is_dreaming + dreaming sub-dict + developmental_age
            # from pi_heartbeat). Frontend useDreaming hook reads these.
            dreaming = snapshot.get("dreaming") or {}
            dream_payload = dict(dreaming)
            dream_payload["is_dreaming"] = snapshot.get("is_dreaming", False)
            dream_payload["developmental_age"] = (
                (pi or {}).get("developmental_age", 0))
            _send_msg(send_queue, DREAMING_STATE_UPDATED, name, "all",
                      dream_payload)
            meta = snapshot.get("meta_reasoning")
            if meta:
                _send_msg(send_queue, META_REASONING_STATS_UPDATED, name,
                          "all", meta)
            # Reasoning engine stats (chains, commits, abandons, commit_rate)
            # — observed empty on /v4/reasoning until this publish was added
            # (2026-04-26 sweep). Endpoint reads from reasoning.state cache key.
            reasoning = snapshot.get("reasoning")
            if reasoning:
                _send_msg(send_queue, REASONING_STATS_UPDATED, name, "all",
                          reasoning)
            expr = snapshot.get("expression_composites")
            if expr:
                _send_msg(send_queue, EXPRESSION_COMPOSITES_UPDATED, name,
                          "all", expr)
            nm = snapshot.get("neuromodulators")
            if nm:
                _send_msg(send_queue, NEUROMOD_STATS_UPDATED, name, "all", nm)
            # rFP_observatory_data_loading_v1 Phase 4 — MSL state fan-out.
            # I-Depth tab consumes msl.state cache key for i_confidence /
            # i_depth / components / convergence_count / concept_confidences /
            # attention_weights. Coord snapshot already builds this at
            # build_coordinator_snapshot:1651 — fan it out here.
            msl_state = snapshot.get("msl")
            if msl_state:
                _send_msg(send_queue, MSL_STATE_UPDATED, name, "all", msl_state)
            # Language teacher periodic stats — vocab / prod / level / conf
            # / last_teach_at. Coord snapshot includes stats["language"] when
            # the worker passes language_stats; fan out so /v4/vocabulary +
            # related tabs render.
            lang = snapshot.get("language")
            if lang:
                _send_msg(send_queue, LANGUAGE_STATS_UPDATED, name, "all", lang)
            # Batch E (rFP §3.2 follow-up): topology block — legacy
            # volume/curvature/cluster_count + 30D space-topology
            # observables_dict (6 layers × 5 metrics). Frontend
            # TopologyPanel renders both forms. SpiritAccessor.get_coordinator()
            # overlay reads topology.state and merges into coord["topology"].
            topo = snapshot.get("topology")
            if topo:
                _send_msg(send_queue, TOPOLOGY_STATE_UPDATED, name, "all", topo)
        except Exception as pub_err:
            # Never let a publish glitch break the snapshot builder loop.
            logger.warning(
                "[SnapshotBuilder:coord] subdomain publish failed: %s",
                pub_err)

    def _builder_loop(kind: str, build_fn, cache: dict, interval: float):
        consecutive_errors = 0
        try:
            while True:
                _t0 = time.time()
                try:
                    result = build_fn()
                    if result is not None:
                        # Atomic pointer swap under GIL — readers see
                        # either the old dict or the new dict, never a
                        # partially-built one.
                        cache["data"] = result
                        cache["ts"] = time.time()
                    consecutive_errors = 0
                except Exception as exc:
                    consecutive_errors += 1
                    if consecutive_errors == 1 or consecutive_errors % 10 == 0:
                        # Include traceback so the dead-coordinator
                        # class of bugs (e.g. 2026-04-22 "tuple index
                        # out of range" that silently starved /v4/inner-
                        # trinity) diagnoses in one log line next time.
                        logger.warning(
                            "[SnapshotBuilder:%s] build failed "
                            "(#%d consecutive): %s",
                            kind, consecutive_errors, exc,
                            exc_info=True)
                build_ms = (time.time() - _t0) * 1000
                logger.debug(
                    "[SnapshotBuilder:%s] built in %.0fms",
                    kind, build_ms)
                sleep_s = (_SNAPSHOT_BUILDER_ERROR_BACKOFF
                           if consecutive_errors > 0 else interval)
                time.sleep(sleep_s)
        except BaseException as fatal:
            logger.error(
                "[SnapshotBuilder:%s] loop exited unexpectedly — "
                "cache will become stale: %s",
                kind, fatal, exc_info=True)

    def _coord_build_and_publish():
        snap = build_coordinator_snapshot(state_refs)
        if snap is not None:
            _publish_coord_subdomains(snap)
        return snap

    threading.Thread(
        target=_builder_loop,
        args=("coord",
              _coord_build_and_publish,
              _COORD_SNAPSHOT_CACHE,
              _COORD_SNAPSHOT_BUILDER_INTERVAL),
        daemon=True, name="coord-snapshot-builder",
    ).start()
    threading.Thread(
        target=_builder_loop,
        args=("trinity",
              lambda: build_trinity_snapshot(state_refs, config),
              _TRINITY_SNAPSHOT_CACHE,
              _TRINITY_SNAPSHOT_BUILDER_INTERVAL),
        daemon=True, name="trinity-snapshot-builder",
    ).start()
    threading.Thread(
        target=_builder_loop,
        args=("ns",
              lambda: build_nervous_system_snapshot(state_refs),
              _NS_SNAPSHOT_CACHE,
              _NS_SNAPSHOT_BUILDER_INTERVAL),
        daemon=True, name="ns-snapshot-builder",
    ).start()
    logger.info(
        "[SpiritLoop] Snapshot builder threads started — "
        "coord/trinity/ns rebuild every %.2f/%.2f/%.2fs",
        _COORD_SNAPSHOT_BUILDER_INTERVAL,
        _TRINITY_SNAPSHOT_BUILDER_INTERVAL,
        _NS_SNAPSHOT_BUILDER_INTERVAL)


def _handle_query(msg: dict, config: dict, body_state: dict, mind_state: dict,
                  consciousness: dict | None, filter_down, intuition, impulse_engine,
                  sphere_clock, resonance, unified_spirit,
                  send_queue, name: str,
                  inner_state=None, spirit_state=None,
                  coordinator=None,
                  neural_nervous_system=None,
                  pi_monitor=None, e_mem=None,
                  prediction_engine=None, ex_mem=None,
                  episodic_mem=None, working_mem=None,
                  inner_lower_topo=None, outer_lower_topo=None,
                  ground_up_enricher=None, neuromodulator_system=None,
                  expression_manager=None, life_force_engine=None,
                  outer_interface=None, phase_tracker=None,
                  meditation_tracker=None, reasoning_engine=None,
                  msl=None, social_pressure_meter=None,
                  language_stats=None, self_reasoning=None,
                  coding_explorer=None,
                  filter_down_v5=None,
                  med_watchdog=None) -> None:
    """Handle Spirit queries."""
    payload = msg.get("payload", {})
    action = payload.get("action", "")
    rid = msg.get("rid")
    src = msg.get("src", "")

    try:
        if action == "get_tensor":
            tensor = _collect_spirit_tensor(config, body_state, mind_state, consciousness)
            _send_response(send_queue, name, src, {"tensor": tensor}, rid)

        elif action == "get_trinity":
            # Primary path: background `trinity-snapshot-builder` thread
            # keeps _TRINITY_SNAPSHOT_CACHE fresh (~0.5s cycle). Cache read
            # is atomic (dict[key] pointer deref under GIL).
            _cached = _TRINITY_SNAPSHOT_CACHE["data"]
            if _cached is not None:
                _send_response(send_queue, name, src, _cached, rid)
                return
            # Cold-boot fallback: builder hasn't run yet. Build synchronously
            # and populate cache so subsequent queries fast-path immediately.
            _state_refs = {
                "body_state": body_state, "mind_state": mind_state,
                "consciousness": consciousness,
                "filter_down": filter_down, "intuition": intuition,
                "impulse_engine": impulse_engine,
                "sphere_clock": sphere_clock, "resonance": resonance,
                "unified_spirit": unified_spirit,
                "inner_state": inner_state, "spirit_state": spirit_state,
            }
            response = build_trinity_snapshot(_state_refs, config)
            _TRINITY_SNAPSHOT_CACHE["data"] = response
            _TRINITY_SNAPSHOT_CACHE["ts"] = time.time()
            _send_response(send_queue, name, src, response, rid)

        elif action == "get_consciousness":
            if consciousness and consciousness.get("latest_epoch"):
                _send_response(send_queue, name, src, consciousness["latest_epoch"], rid)
            else:
                _send_response(send_queue, name, src, {"error": "No consciousness epochs yet"}, rid)

        elif action == "get_filter_down":
            if filter_down:
                _send_response(send_queue, name, src, filter_down.get_stats(), rid)
            else:
                _send_response(send_queue, name, src, {"error": "FilterDown not available"}, rid)

        elif action == "get_intuition":
            if intuition:
                _send_response(send_queue, name, src, intuition.get_stats(), rid)
            else:
                _send_response(send_queue, name, src, {"error": "Intuition not available"}, rid)

        elif action == "get_impulse_engine":
            if impulse_engine:
                _send_response(send_queue, name, src, impulse_engine.get_stats(), rid)
            else:
                _send_response(send_queue, name, src, {"error": "ImpulseEngine not available"}, rid)

        elif action == "get_sphere_clock":
            if sphere_clock:
                _send_response(send_queue, name, src, sphere_clock.get_stats(), rid)
            else:
                _send_response(send_queue, name, src, {"error": "SphereClockEngine not available"}, rid)

        elif action == "get_resonance":
            if resonance:
                _send_response(send_queue, name, src, resonance.get_stats(), rid)
            else:
                _send_response(send_queue, name, src, {"error": "ResonanceDetector not available"}, rid)

        elif action == "get_unified_spirit":
            if unified_spirit:
                _send_response(send_queue, name, src, unified_spirit.get_stats(), rid)
            else:
                _send_response(send_queue, name, src, {"error": "UnifiedSpirit not available"}, rid)

        elif action == "get_filter_down_status":
            # V4 retired 2026-04-25 (Pattern C surfaced silent dim mismatch
            # since 130D upgrade). V5 is the sole FILTER_DOWN learner now.
            _v5_stats = filter_down_v5.get_stats() if filter_down_v5 else None
            v5_publishing = bool(
                (config or {}).get("filter_down_v5", {}).get("publish_enabled", False)
            )
            _send_response(send_queue, name, src, {
                "v5": _v5_stats,
                "v5_publishing": v5_publishing,
                "coexistence_phase": "v5_only",
            }, rid)

        elif action == "get_meditation_health":
            # rFP_self_healing_meditation_cadence.md Phase 1+2:
            # watchdog state + tracker + config for /v4/meditation/health.
            _mh: dict = {}
            if meditation_tracker:
                _mh["tracker"] = {
                    "count": meditation_tracker.get("count", 0),
                    "count_since_nft": meditation_tracker.get("count_since_nft", 0),
                    "last_epoch": meditation_tracker.get("last_epoch", 0),
                    "last_ts": meditation_tracker.get("last_ts", 0),
                    "in_meditation": bool(meditation_tracker.get("in_meditation", False)),
                }
            else:
                _mh["tracker"] = {"error": "tracker not available"}
            if med_watchdog is not None:
                try:
                    _mh["watchdog"] = med_watchdog.health_snapshot()
                except Exception as _mh_err:
                    _mh["watchdog"] = {"error": f"snapshot error: {_mh_err}"}
            else:
                _mh["watchdog"] = {"error": "watchdog not initialized"}
            # Overdue flag (cross-Titan correlation uses this at central aggregator)
            _mh["overdue"] = False
            if meditation_tracker and med_watchdog is not None:
                _last_ts = float(meditation_tracker.get("last_ts", 0) or 0)
                if _last_ts > 0:
                    import time as _t
                    _now = _t.time()
                    _elapsed = _now - _last_ts
                    _expected = med_watchdog.expected_interval()
                    _floor = med_watchdog.min_alert_hours * 3600.0
                    _threshold = max(_floor, _expected)
                    if _elapsed > _threshold:
                        _mh["overdue"] = True
                        _mh["overdue_since_ts"] = _last_ts + _threshold
                        _mh["overdue_elapsed_hours"] = round(_elapsed / 3600, 2)
            _send_response(send_queue, name, src, _mh, rid)

        elif action == "get_observables":
            if inner_state and inner_state.observables:
                _send_response(send_queue, name, src, inner_state.observables, rid)
            else:
                _send_response(send_queue, name, src, {"error": "Observables not available"}, rid)

        elif action == "get_inner_state":
            if inner_state:
                _send_response(send_queue, name, src, inner_state.snapshot(), rid)
            else:
                _send_response(send_queue, name, src, {"error": "InnerState not available"}, rid)

        elif action == "get_spirit_state":
            if spirit_state:
                _send_response(send_queue, name, src, spirit_state.snapshot(), rid)
            else:
                _send_response(send_queue, name, src, {"error": "SpiritState not available"}, rid)

        elif action == "get_coordinator":
            # Primary path: background `coord-snapshot-builder` thread
            # keeps _COORD_SNAPSHOT_CACHE fresh (~4s cycle under normal
            # load; cache age bounded by build_time + interval). Cache
            # read is atomic (dict[key] pointer deref under GIL). Before
            # this refactor: the handler built inline on every cache
            # miss (460-2144ms) blocking the QueryThread and causing
            # SKIP-stale on queued queries (T1-COORD-QUERYTHREAD-BACKLOG).
            _cached = _COORD_SNAPSHOT_CACHE["data"]
            if _cached is not None:
                _send_response(send_queue, name, src, _cached, rid)
                return
            # Cold-boot fallback: builder hasn't populated cache yet
            # (first few seconds after spirit_worker start). Build
            # synchronously so the first query still gets an answer,
            # then populate the cache so the next query fast-paths.
            if coordinator:
                _state_refs = {
                    "coordinator": coordinator, "pi_monitor": pi_monitor,
                    "e_mem": e_mem, "prediction_engine": prediction_engine,
                    "ex_mem": ex_mem, "episodic_mem": episodic_mem,
                    "working_mem": working_mem,
                    "inner_lower_topo": inner_lower_topo,
                    "outer_lower_topo": outer_lower_topo,
                    "ground_up_enricher": ground_up_enricher,
                    "neuromodulator_system": neuromodulator_system,
                    "expression_manager": expression_manager,
                    "life_force_engine": life_force_engine,
                    "meditation_tracker": meditation_tracker,
                    "outer_interface": outer_interface,
                    "reasoning_engine": reasoning_engine,
                    "self_reasoning": self_reasoning,
                    "coding_explorer": coding_explorer,
                    "phase_tracker": phase_tracker,
                    "inner_state": inner_state,
                    "social_pressure_meter": social_pressure_meter,
                    "msl": msl, "language_stats": language_stats,
                }
                stats = build_coordinator_snapshot(_state_refs)
                if stats is not None:
                    _COORD_SNAPSHOT_CACHE["data"] = stats
                    _COORD_SNAPSHOT_CACHE["ts"] = time.time()
                    _send_response(send_queue, name, src, stats, rid)
                else:
                    _send_response(send_queue, name, src,
                                   {"error": "Coordinator not available"}, rid)
            else:
                _send_response(send_queue, name, src,
                               {"error": "Coordinator not available"}, rid)

        elif action == "reset_msl_homeostasis":
            # Phase 3+ of foundational healing rFP (2026-04-13): MSL
            # homeostatic reset to baseline. After 27+ days of accumulated
            # allostatic drift and the foundational fixes (Phase 1+2+3),
            # the cleaner experiment is to start from baseline and observe
            # whether new diverse signal flow keeps it stable.
            try:
                if msl is None or not hasattr(msl, 'policy') or msl.policy is None:
                    _send_response(send_queue, name, src,
                                   {"ok": False, "error": "msl unavailable"}, rid)
                else:
                    homeo = msl.policy.homeostatic
                    # Capture previous state for audit
                    prev_state = homeo.get_state() if hasattr(
                        homeo, 'get_state') else {}
                    prev_setpoint_entropy = prev_state.get(
                        "setpoint_entropy", 0.0)
                    # Build uniform baseline (preserves update_count and
                    # _recent_entropy → these are observability counters,
                    # not pathology indicators).
                    n = homeo.n if hasattr(homeo, 'n') else 7
                    uniform = 1.0 / n
                    baseline = {
                        "setpoints": [uniform] * n,
                        "sensitivity": [1.0] * n,
                        "tonic": [uniform] * n,
                        # Preserve telemetry (don't lose update history)
                        "recent_entropy": float(getattr(
                            homeo, '_max_entropy', 1.946)),
                        "update_count": int(getattr(
                            homeo, '_update_count', 0)),
                    }
                    homeo.from_dict(baseline)
                    # Reset drift_guard counter too (fresh observation window)
                    if hasattr(homeo, '_drift_guard_active_count'):
                        homeo._drift_guard_active_count = 0
                    # Persist to disk via msl save
                    try:
                        msl.save_all()
                    except Exception as _save_err:
                        logger.warning(
                            "[SpiritLoop] msl.save_all after reset failed: %s",
                            _save_err)
                    logger.warning(
                        "[SpiritLoop] MSL homeostatic RESET to baseline. "
                        "Reason: %s. Prev setpoint_entropy=%.3f → 1.946",
                        payload.get("reason", "unknown"),
                        prev_setpoint_entropy)
                    _send_response(send_queue, name, src, {
                        "ok": True,
                        "prev_setpoint_entropy": prev_setpoint_entropy,
                        "prev_state": prev_state,
                        "new_setpoint_entropy": float(getattr(
                            homeo, '_max_entropy', 1.946)),
                    }, rid)
            except Exception as _reset_err:
                logger.error("[SpiritLoop] MSL reset failed: %s", _reset_err,
                             exc_info=True)
                _send_response(send_queue, name, src,
                               {"ok": False, "error": str(_reset_err)}, rid)

        elif action == "get_nervous_system":
            # Primary path: background `ns-snapshot-builder` thread keeps
            # _NS_SNAPSHOT_CACHE fresh (~0.5s cycle).
            _cached = _NS_SNAPSHOT_CACHE["data"]
            if _cached is not None:
                _send_response(send_queue, name, src, _cached, rid)
                return
            # Cold-boot fallback: synchronous build.
            _state_refs = {
                "neural_nervous_system": neural_nervous_system,
                "coordinator": coordinator,
            }
            _ns_resp = build_nervous_system_snapshot(_state_refs)
            if _ns_resp is not None:
                _NS_SNAPSHOT_CACHE["data"] = _ns_resp
                _NS_SNAPSHOT_CACHE["ts"] = time.time()
                _send_response(send_queue, name, src, _ns_resp, rid)
            else:
                _send_response(send_queue, name, src,
                               {"error": "NervousSystem not available"}, rid)

        elif action == "social_relief":
            inner_payload = payload.get("payload", {})
            relief = float(inner_payload.get("relief", 0.0))
            if social_pressure_meter and relief > 0:
                social_pressure_meter.on_social_relief(relief)
                _send_response(send_queue, name, src, {
                    "applied": True,
                    "relief": relief,
                    "urge_after": round(social_pressure_meter.urge_accumulator, 2),
                }, rid)
            else:
                _send_response(send_queue, name, src, {
                    "applied": False,
                    "reason": "social_pressure_meter not available" if not social_pressure_meter
                              else "relief must be positive",
                }, rid)

        elif action == "signal_concept":
            inner_payload = payload.get("payload", {})
            concept = inner_payload.get("concept", "").upper()
            quality = float(inner_payload.get("quality", 0.5))
            extra = inner_payload.get("extra", {})
            if msl and hasattr(msl, 'concept_grounder') and msl.concept_grounder:
                cg = msl.concept_grounder
                epoch = getattr(msl, '_tick_count', 0)
                event = None
                if concept == "YES":
                    event = cg.signal_yes(quality, epoch, None)
                elif concept == "NO":
                    event = cg.signal_no(quality, epoch, None)
                elif concept == "YOU":
                    event = cg.signal_you(
                        extra.get("kin_pubkey", "persona"),
                        quality, epoch, None)
                elif concept == "WE":
                    event = cg.signal_we(quality, epoch, None)
                elif concept == "THEY":
                    event = cg.signal_they(
                        extra.get("engagement_type", "conversation"),
                        extra.get("author", "persona"),
                        quality, epoch, None)
                elif concept == "I":
                    # Reinforce I-confidence from social/persona detection
                    bonus = 0.001 * quality
                    old_conf = msl.get_i_confidence()
                    msl.confidence._convergence_count += 1
                    event = {"type": "I_SOCIAL_REINFORCEMENT",
                             "quality": quality,
                             "confidence_before": old_conf, "bonus": bonus}
                _send_response(send_queue, name, src, {
                    "signaled": True,
                    "concept": concept,
                    "quality": quality,
                    "event": event or {},
                }, rid)
            else:
                _send_response(send_queue, name, src, {
                    "signaled": False,
                    "reason": "MSL concept_grounder not available",
                }, rid)

        elif action == "signal_co_occurrence":
            # Cross-concept reinforcement when multiple concepts detected in same turn
            inner_payload = payload.get("payload", {})
            concepts = inner_payload.get("concepts", [])
            if msl and hasattr(msl, 'concept_grounder') and msl.concept_grounder and concepts:
                msl.concept_grounder.signal_co_occurrence(
                    concepts, msl.get_i_confidence())
                _send_response(send_queue, name, src, {
                    "reinforced": True, "concepts": concepts,
                }, rid)
            else:
                _send_response(send_queue, name, src, {
                    "reinforced": False, "reason": "not available",
                }, rid)

        elif action == "get_social_perception_stats":
            # Return accumulated social perception stats for outer_mind enrichment
            stats = getattr(coordinator, '_social_perception_stats', {
                "sentiment_ema": 0.5,
                "connection_ema": 0.0,
                "events_count": 0,
                "last_contagion": None,
            }) if coordinator else {}
            _send_response(send_queue, name, src, stats, rid)

        else:
            logger.warning("[SpiritWorker] Unknown action: %s", action)

    except Exception as e:
        logger.error("[SpiritWorker] Error handling %s: %s", action, e, exc_info=True)
        _send_response(send_queue, name, src, {"error": str(e)}, rid)


def _publish_spirit_state(send_queue, name: str, tensor: list, consciousness: dict | None,
                          filter_down=None, body_state=None, mind_state=None,
                          neural_nervous_system=None, sphere_clock=None,
                          unified_spirit=None, e_mem=None,
                          neuromodulator_system=None, expression_manager=None,
                          resonance=None, meta_sink: dict | None = None) -> None:
    """Publish SPIRIT_STATE to the bus with Middle Path loss and counterpart stats.

    BUG #11 fix (2026-04-24): neural_nervous_system, sphere_clock, unified_spirit,
    e_mem added as optional params to populate hormone_levels / hormone_fires /
    sphere_clocks / unified_spirit_stats / memory_stats into collect_spirit_45d().
    Pre-fix, 19 of 45 inner_spirit dims were dead because these inputs were
    passed as None (with TODO comment "Populated when hormonal system wired").

    Microkernel v2 amendment (2026-04-26): payload["v4"] block now carries
    sphere_clocks / unified_spirit / resonance / neuromodulators /
    expression_composites / nervous_system summaries. state_register reads
    these into its in-process slots, which the kernel snapshot publisher
    forwards to api_subprocess CachedState. This closes the empty-cache
    class for /v4/sphere-clocks, /v4/neuromodulators, /v4/expression-composites,
    /v4/nervous-system without adding new bus message types — Phase B/C-safe
    (same wire shape, transport-agnostic).
    """
    center = [0.5] * 5
    center_dist = sum((t - c) ** 2 for t, c in zip(tensor, center)) ** 0.5

    payload = {
        "dims": 5,
        "values": tensor,
        "delta": [round(t - 0.5, 4) for t in tensor],
        "center_dist": round(center_dist, 4),
    }

    # Include consciousness summary in broadcast
    if consciousness and consciousness.get("latest_epoch"):
        epoch = consciousness["latest_epoch"]
        cons_payload = {
            "epoch_id": epoch.get("epoch_id", 0),
            "epoch_number": epoch.get("epoch_id", 0),
            "curvature": epoch.get("curvature", 0),
            "density": epoch.get("density", 0),
            "drift_magnitude": epoch.get("drift_magnitude", 0),
            "drift": epoch.get("drift_magnitude", 0),
            "trajectory": epoch.get("trajectory_magnitude", 0),
        }
        # rFP_observatory_data_loading_v1 §3.2 fix (2026-04-26):
        # state_vector (130D when available) carries the full Trinity
        # composition — iB(5)+iM(15)+iS(45)+oB(5)+oM(15)+oS(45)+meta(2).
        # /v3/trinity uses sv[20:65] for the inner Spirit 45D heatmap and
        # sv[5:20] for inner Mind 15D. Pre-fix: this payload stripped sv,
        # so the heatmap fell back to "awaiting full tensor" forever.
        # ~528 bytes per epoch tick — negligible bus traffic.
        sv = epoch.get("state_vector")
        if isinstance(sv, list) and sv:
            cons_payload["state_vector"] = list(sv)
        payload["consciousness"] = cons_payload

    # Include Middle Path loss
    if body_state and mind_state:
        try:
            from titan_plugin.logic.middle_path import middle_path_loss
            body_vals = body_state.get("values", [0.5] * 5)
            mind_vals = mind_state.get("values", [0.5] * 5)
            payload["middle_path_loss"] = round(
                middle_path_loss(body_vals, mind_vals, tensor), 4)
        except Exception as _swallow_exc:
            swallow_warn('[modules.spirit_loop] _publish_spirit_state: from titan_plugin.logic.middle_path import middle_path_loss', _swallow_exc,
                         key='modules.spirit_loop._publish_spirit_state.line2266', throttle=100)

    # Include FILTER_DOWN summary
    if filter_down:
        try:
            stats = filter_down.get_stats()
            payload["filter_down"] = {
                "train_steps": stats["total_train_steps"],
                "last_loss": stats["last_loss"],
                "buffer_size": stats["buffer_size"],
            }
        except Exception as _swallow_exc:
            swallow_warn('[modules.spirit_loop] _publish_spirit_state: stats = filter_down.get_stats()', _swallow_exc,
                         key='modules.spirit_loop._publish_spirit_state.line2278', throttle=100)

    # DQ3: Extended 45D Spirit tensor (Sat + Chit + Ananda)
    # BUG #11 fix (2026-04-24): populate hormone_levels / hormone_fires /
    # sphere_clocks / unified_spirit_stats / memory_stats. Pre-fix, 19 of 45
    # inner_spirit dims were dead because these inputs were None.
    try:
        from titan_plugin.logic.spirit_tensor import collect_spirit_45d
        body_vals = body_state.get("values", [0.5] * 5) if body_state else [0.5] * 5
        mind_vals = mind_state.get("values", [0.5] * 5) if mind_state else [0.5] * 5
        mind_15d = mind_state.get("values_15d", mind_vals) if mind_state else mind_vals

        # ── BUG #11 input gathering — defensive: every source may be None
        # 2026-04-24 follow-up: removed `_hormonal_enabled` gate since it
        # was causing silent skip (live observation showed dead dims still
        # at 0 on all 3 Titans post-first-fix). Just try get_levels()
        # directly — if _hormonal exists with the method, use it.
        _hlvl = None
        _hfires = None
        if neural_nervous_system is not None:
            _horm = getattr(neural_nervous_system, "_hormonal", None)
            if _horm is not None:
                try:
                    _hlvl = _horm.get_levels() if hasattr(_horm, "get_levels") else None
                except Exception:
                    _hlvl = None
                try:
                    _hfires = {
                        n: int(getattr(h, "fire_count", 0))
                        for n, h in getattr(_horm, "_hormones", {}).items()
                    }
                except Exception:
                    _hfires = None
                # Once-per-worker-boot debug confirmation (throttled to first 5 calls
                # via module-level counter). Lets us verify the fix path activates.
                global _BUG11_BOOT_CONFIRMED_COUNT
                try:
                    _BUG11_BOOT_CONFIRMED_COUNT
                except NameError:
                    _BUG11_BOOT_CONFIRMED_COUNT = 0
                if _BUG11_BOOT_CONFIRMED_COUNT < 5 and (_hlvl or _hfires):
                    _BUG11_BOOT_CONFIRMED_COUNT += 1
                    logger.info(
                        "[SpiritWorker] BUG #11 fix active — "
                        "hormone_levels=%s, hormone_fires=%s (call #%d of first-5)",
                        "populated" if _hlvl else "None",
                        "populated" if _hfires else "None",
                        _BUG11_BOOT_CONFIRMED_COUNT)

        _clocks = None
        if sphere_clock is not None:
            try:
                _clocks_src = getattr(sphere_clock, "clocks", None) or \
                              getattr(sphere_clock, "_clocks", None) or {}
                # SphereClock attribute is `phase` (∈ [0, 2π], advanced in
                # tick() at sphere_clock.py:121). The legacy `current_phase`
                # name was never on the class — getattr fell back to 0.5
                # default forever, so all 6 UI bars showed 0.50. Fixed
                # 2026-04-26 per rFP_observatory_data_loading_v1 §3.2.
                #
                # Batch F (2026-04-26): include `radius` + `consecutive_balanced`
                # — frontend SphereClocks reads radius.toFixed(2) (defaults to 0.5
                # when missing) and consecutive_balanced for the resonance bar.
                # Without them all 6 clocks displayed "0.50" in the UI.
                _clocks = {
                    cname: {
                        "pulse_count": int(getattr(c, "pulse_count", 0)),
                        "phase": float(getattr(c, "phase", 0.0)),
                        "scalar_position": float(getattr(c, "scalar_position", 1.0)),
                        "contraction_velocity": float(getattr(c, "contraction_velocity", 0.0)),
                        "radius": float(getattr(c, "radius", 1.0)),
                        "consecutive_balanced": int(getattr(c, "_consecutive_balanced", 0)),
                        "total_ticks": int(getattr(c, "_total_ticks", 0)),
                    }
                    for cname, c in _clocks_src.items()
                }
            except Exception:
                _clocks = None

        _us = None
        if unified_spirit is not None:
            try:
                _us = unified_spirit.get_stats() if hasattr(unified_spirit, "get_stats") else None
            except Exception:
                _us = None

        _mem = None
        if e_mem is not None:
            try:
                _mem = e_mem.get_stats() if hasattr(e_mem, "get_stats") else None
            except Exception:
                _mem = None

        # Composite-level expression activity for spirit_tensor's chit[13]
        # (causal_understanding) + ananda[8] (expression_quality). The
        # ExpressionTranslator's `sovereignty_ratio` lives only in the main
        # plugin process — see spirit_tensor._expression_intensity for the
        # composite-side proxy that activates when only this dict is available.
        _expr_stats = None
        if expression_manager is not None:
            try:
                _expr_stats = expression_manager.get_stats() if hasattr(
                    expression_manager, "get_stats") else None
            except Exception:
                _expr_stats = None

        spirit_45d = collect_spirit_45d(
            current_5d=tensor,
            body_tensor=body_vals,
            mind_tensor=mind_15d,
            consciousness=consciousness.get("latest_epoch") if consciousness else None,
            topology=body_state.get("topology") if body_state else None,
            hormone_levels=_hlvl,
            hormone_fires=_hfires,
            unified_spirit_stats=_us,
            sphere_clocks=_clocks,
            memory_stats=_mem,
            expression_stats=_expr_stats,
        )
        payload["values_45d"] = [round(v, 4) for v in spirit_45d]
        payload["dims_extended"] = 45
        # BUG-EMOT-CGN-FELT-FROM-CONSCIOUSNESS-SPARSE fix (2026-04-26):
        # stash the freshly-computed spirit_45d into the optional meta_sink
        # dict so the META → EMOT-CGN bridge in spirit_worker can assemble a
        # live full-130D felt tensor (instead of the sparse consciousness
        # state_vector which only populates slots 0-6 + tail).
        if isinstance(meta_sink, dict):
            meta_sink["_last_spirit_45d"] = list(spirit_45d)
    except Exception as _swallow_exc:
        swallow_warn('[modules.spirit_loop] _publish_spirit_state: from titan_plugin.logic.spirit_tensor import collect_spir...', _swallow_exc,
                     key='modules.spirit_loop._publish_spirit_state.line2371', throttle=100)

    # ── v4 block — microkernel v2 amendment 2026-04-26 ──
    # Carries the spirit_worker-owned aggregates that state_register's
    # SPIRIT_STATE handler extracts. Each field guarded by isinstance to
    # survive partial-init states (early boot, post-reload).
    v4_block: dict = {}
    try:
        if _clocks:
            v4_block["sphere_clocks"] = _clocks
        if _us:
            v4_block["unified_spirit"] = _us
        if resonance is not None:
            try:
                rstats = resonance.get_stats() if hasattr(resonance, "get_stats") else None
                if isinstance(rstats, dict):
                    v4_block["resonance"] = rstats
            except Exception:
                pass
        if neuromodulator_system is not None:
            try:
                nstate = neuromodulator_system.get_state() if hasattr(
                    neuromodulator_system, "get_state") else None
                if isinstance(nstate, dict):
                    v4_block["neuromodulators"] = nstate
            except Exception:
                pass
        if expression_manager is not None:
            try:
                # ExpressionManager carries per-composite urge / threshold /
                # fire_count / consumption_rate. Survives spirit_worker
                # restart (state managed by manager itself).
                ecomps = {}
                for c_name, comp in getattr(
                        expression_manager, "_composites",
                        getattr(expression_manager, "composites", {}) or {}).items():
                    ecomps[c_name] = {
                        "urge": float(getattr(comp, "urge", 0.0) or 0.0),
                        "threshold": float(getattr(comp, "threshold", 0.0) or 0.0),
                        "fire_count": int(getattr(comp, "fire_count", 0) or 0),
                        "consumption_rate": float(
                            getattr(comp, "consumption_rate", 0.0) or 0.0),
                    }
                if ecomps:
                    v4_block["expression_composites"] = ecomps
            except Exception:
                pass
        if neural_nervous_system is not None:
            try:
                nnstats = neural_nervous_system.get_stats() if hasattr(
                    neural_nervous_system, "get_stats") else None
                if isinstance(nnstats, dict):
                    v4_block["neural_nervous_system"] = nnstats
            except Exception:
                pass
    except Exception:
        pass
    if v4_block:
        payload["v4"] = v4_block

    _send_msg(send_queue, bus.SPIRIT_STATE, name, "all", payload)


def _send_msg(send_queue, msg_type: str, src: str, dst: str, payload: dict, rid: str = None) -> None:
    try:
        send_queue.put_nowait({
            "type": msg_type, "src": src, "dst": dst,
            "ts": time.time(), "rid": rid, "payload": payload,
        })
    except Exception as e:
        logger.warning("[SpiritWorker] Failed to send: %s", e)


def _send_response(send_queue, src: str, dst: str, payload: dict, rid: str) -> None:
    _send_msg(send_queue, bus.RESPONSE, src, dst, payload, rid)


# Heartbeat throttle (Phase E Fix 2): 3s minimum interval per process.
# spirit_worker has 6 _send_heartbeat call sites in its main loop; without
# throttle, observed emission was 5.76 msg/s (17× expected), saturating
# Guardian's queue. 3s leaves headroom under tightest 60s Guardian timeout.
_last_hb_ts: float = 0.0


def _send_heartbeat(send_queue, name: str) -> None:
    global _last_hb_ts
    now = time.time()
    if now - _last_hb_ts < 3.0:
        return
    _last_hb_ts = now
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        rss_mb = 0
    _send_msg(send_queue, bus.MODULE_HEARTBEAT, name, "guardian", {"rss_mb": round(rss_mb, 1)})


# ═══════════════════════════════════════════════════════════════════════
# Bridge A: Inner → Outer Dream Memory Injection
# ═══════════════════════════════════════════════════════════════════════

def _build_felt_snapshot(neuromod_system, dream_cycle: int = 0) -> dict:
    """Build a compact felt-state snapshot for memory injection.

    Stored in DuckDB neuromod_context column. Used by Bridge B
    for recall perturbation (somatic re-experiencing).
    """
    if not neuromod_system:
        return {}
    import time as _t
    snapshot = {}
    for nm_name, nm_mod in neuromod_system.modulators.items():
        snapshot[nm_name] = round(nm_mod.level, 4)
    snapshot["emotion"] = getattr(neuromod_system, '_current_emotion', 'neutral')
    snapshot["emotion_confidence"] = round(
        getattr(neuromod_system, '_emotion_confidence', 0.0), 4)
    snapshot["dream_cycle"] = dream_cycle
    snapshot["ts"] = _t.time()
    return snapshot


_BRIDGE_DEDUP_PATH = "./data/dream_bridge_dedup.json"


def _load_bridge_dedup() -> dict:
    """Load set of already-injected milestone IDs."""
    import json as _dj
    try:
        with open(_BRIDGE_DEDUP_PATH) as f:
            return _dj.load(f)
    except Exception:
        return {"wisdom_ids": [], "cgn_words": [], "composition_ids": []}


def _save_bridge_dedup(dedup: dict):
    """Save dedup state. Keep last 100 entries per category."""
    import json as _dj
    for k in dedup:
        if isinstance(dedup[k], list):
            dedup[k] = dedup[k][-100:]
    try:
        with open(_BRIDGE_DEDUP_PATH, "w") as f:
            _dj.dump(dedup, f)
    except Exception as _swallow_exc:
        swallow_warn("[modules.spirit_loop] _save_bridge_dedup: with open(_BRIDGE_DEDUP_PATH, 'w') as f: _dj.dump(dedup, f)", _swallow_exc,
                     key='modules.spirit_loop._save_bridge_dedup.line2458', throttle=100)


def _build_self_profile(
    neuromod_system, msl, meta_engine, chain_archive,
    dream_cycle: int, language_stats: dict | None = None,
) -> dict | None:
    """Build structured self-profile for cognitive graph injection.

    Called during END_DREAMING (Layer 1 of SELF_REASONING rFP).
    Returns a memory dict ready for injection, or None if insufficient data.

    The self-profile is Titan's accumulated self-knowledge — vocabulary,
    neuromod ranges, reasoning style, I-depth, cognitive identity.
    Overwritten each dream cycle (only latest matters).
    """
    import logging as _log
    logger = _log.getLogger(__name__)

    try:
        parts = []

        # Vocabulary stats
        vocab_total = 0
        top_words = []
        if language_stats and isinstance(language_stats, dict):
            vocab_total = language_stats.get("total_words", 0)
            productive = language_stats.get("productive_words", 0)
            parts.append(f"I have {vocab_total} words in my vocabulary "
                         f"({productive} productive).")

        # Neuromod identity
        if neuromod_system:
            nm_parts = []
            for nm_name, nm_mod in neuromod_system.modulators.items():
                nm_parts.append(f"{nm_name}={nm_mod.level:.1%} "
                                f"(setpoint {nm_mod.setpoint:.1%})")
            emotion = getattr(neuromod_system, '_current_emotion', 'neutral')
            parts.append(f"My current neurochemistry: {', '.join(nm_parts)}. "
                         f"Emotion: {emotion}.")

        # MSL identity
        if msl:
            i_conf = msl.get_i_confidence()
            i_depth_val = msl.i_depth.depth if hasattr(msl, 'i_depth') else 0
            convergences = msl.confidence._convergence_count
            parts.append(f"I-confidence: {i_conf:.3f} "
                         f"({convergences} convergences). "
                         f"I-depth: {i_depth_val:.4f}.")

            # Concept network
            if msl.concept_grounder:
                concepts = msl.concept_grounder.get_concept_confidences()
                concept_strs = [f"{k}={v:.3f}" for k, v in concepts.items() if v > 0.01]
                if concept_strs:
                    parts.append(f"Concept grounding: {', '.join(concept_strs)}.")

            # I-depth components
            if hasattr(msl, 'i_depth'):
                comps = msl.i_depth._compute_components()
                parts.append(f"I-depth components: " + ", ".join(
                    f"{k}={v:.2f}" for k, v in comps.items()) + ".")

        # Reasoning identity
        if meta_engine:
            total_chains = getattr(meta_engine, '_total_meta_chains', 0)
            total_eurekas = getattr(meta_engine, '_total_eurekas', 0)
            total_wisdom = getattr(meta_engine, '_total_wisdom_saved', 0)
            parts.append(f"Meta-reasoning: {total_chains} chains, "
                         f"{total_eurekas} EUREKA insights, "
                         f"{total_wisdom} wisdom crystallized.")

            # Dominant primitive from buffer
            if hasattr(meta_engine, 'buffer') and meta_engine.buffer:
                from titan_plugin.logic.meta_reasoning import META_PRIMITIVES
                prim_counts = {}
                for a in meta_engine.buffer._actions:
                    if 0 <= a < len(META_PRIMITIVES):
                        name = META_PRIMITIVES[a]
                        prim_counts[name] = prim_counts.get(name, 0) + 1
                if prim_counts:
                    dominant = max(prim_counts, key=prim_counts.get)
                    parts.append(f"My dominant reasoning style: {dominant} "
                                 f"({prim_counts[dominant]} uses).")

        # Dream cycle
        parts.append(f"Dream cycle: {dream_cycle}.")

        if not parts:
            return None

        text = "[SELF_PROFILE] " + " ".join(parts)
        felt = _build_felt_snapshot(neuromod_system, dream_cycle)

        logger.info("[SelfProfile] Built self-profile (%d chars, cycle %d)",
                    len(text), dream_cycle)

        return {
            "text": text,
            "source": "self_profile",
            "weight": 10.0,  # High importance — core self-knowledge
            "neuromod_context": felt,
            "category": "self_profile",
        }

    except Exception as e:
        logger.warning("[SelfProfile] Build failed: %s", e)
        return None


def _harvest_dream_memories(
    chain_archive, meta_wisdom, neuromod_system,
    cgn_db_path: str, dream_cycle: int, max_total: int = 8,
) -> tuple:
    """Harvest significant inner events for cognitive graph injection.

    Called during END_DREAMING. Returns (memories_list, chain_ids_to_mark).
    Each memory is a dict with: text, source, weight, neuromod_context, category.
    Deduplicates against previous injections via dream_bridge_dedup.json.

    Sources (with caps):
      1. Crystallized meta-wisdom (max 3, weight 3.0)
      2. High-scoring reasoning chains (max 2, weight 3.0)
      3. CGN grounding milestones (max 2, weight 2.5)
      4. High-quality compositions (max 1, weight 2.0)
    """
    import sqlite3
    import json
    import logging as _log

    logger = _log.getLogger(__name__)
    memories = []
    chain_ids_to_mark = []
    felt = _build_felt_snapshot(neuromod_system, dream_cycle)
    dedup = _load_bridge_dedup()

    # 1. Crystallized meta-wisdom (max 3)
    if meta_wisdom:
        try:
            conn = sqlite3.connect("./data/inner_memory.db", timeout=10)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, problem_pattern, strategy_sequence, confidence, "
                "times_reused FROM meta_wisdom "
                "WHERE crystallized = 1 ORDER BY confidence DESC LIMIT 3"
            ).fetchall()
            conn.close()
            for r in rows:
                if r["id"] in dedup.get("wisdom_ids", []):
                    continue  # Already injected
                strategy = r["strategy_sequence"] or "unknown"
                if isinstance(strategy, str) and len(strategy) > 200:
                    strategy = strategy[:200]
                dedup.setdefault("wisdom_ids", []).append(r["id"])
                memories.append({
                    "text": (f"[DREAM_WISDOM] After dream cycle #{dream_cycle}, "
                             f"a reasoning pattern crystallized: '{r['problem_pattern'][:100]}' "
                             f"— strategy: {strategy}, confidence: {r['confidence']:.2f}, "
                             f"reused {r['times_reused']}x"),
                    "source": "dream_consolidation",
                    "weight": 3.0,
                    "neuromod_context": felt,
                    "category": "wisdom",
                })
        except Exception as e:
            swallow_warn('[DreamBridge] Meta-wisdom harvest failed', e,
                         key="modules.spirit_loop.meta_wisdom_harvest_failed", throttle=100)

    # 2. High-scoring unconsolidated reasoning chains (max 2)
    if chain_archive:
        try:
            chains = chain_archive.get_unconsolidated(limit=20)
            top_chains = sorted(
                [c for c in chains if c.get("outcome_score", 0) > 0.7],
                key=lambda c: -c.get("outcome_score", 0))[:2]
            for c in top_chains:
                chain_ids_to_mark.append(c["id"])
                memories.append({
                    "text": (f"[EUREKA] High-confidence reasoning chain "
                             f"(score={c['outcome_score']:.2f}): "
                             f"domain={c.get('domain', 'general')}, "
                             f"strategy={c.get('strategy_label', 'emergent')}, "
                             f"{c.get('chain_length', 0)} steps deep"),
                    "source": "dream_consolidation",
                    "weight": 3.0,
                    "neuromod_context": felt,
                    "category": "eureka",
                })
        except Exception as e:
            swallow_warn('[DreamBridge] Chain harvest failed', e,
                         key="modules.spirit_loop.chain_harvest_failed", throttle=100)

    # 3. CGN grounding milestones (max 2)
    try:
        conn = sqlite3.connect(cgn_db_path, timeout=5)
        rows = conn.execute(
            "SELECT word, cross_modal_conf, confidence, "
            "times_encountered, times_produced "
            "FROM vocabulary WHERE cross_modal_conf > 0.15 "
            "ORDER BY cross_modal_conf DESC LIMIT 2"
        ).fetchall()
        conn.close()
        for r in rows:
            if r[0] in dedup.get("cgn_words", []):
                continue  # Already injected
            xm = float(r[1]) if not isinstance(r[1], bytes) else 0.0
            dedup.setdefault("cgn_words", []).append(r[0])
            memories.append({
                "text": (f"[CGN_MILESTONE] The word '{r[0]}' reached deep grounding: "
                         f"cross-modal confidence {xm:.3f}, "
                         f"encountered {r[3]}x, produced {r[4]}x — "
                         f"I truly feel what this word means"),
                "source": "dream_consolidation",
                "weight": 2.5,
                "neuromod_context": felt,
                "category": "cgn_milestone",
            })
    except Exception as e:
        swallow_warn('[DreamBridge] CGN milestone harvest failed', e,
                     key="modules.spirit_loop.cgn_milestone_harvest_failed", throttle=100)

    # 4. High-quality compositions (max 1)
    try:
        conn = sqlite3.connect(cgn_db_path, timeout=5)
        row = conn.execute(
            "SELECT id, sentence, level, confidence FROM composition_history "
            "WHERE level >= 7 AND confidence > 0.8 "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row and row[0] not in dedup.get("composition_ids", []):
            dedup.setdefault("composition_ids", []).append(row[0])
            memories.append({
                "text": (f"[COMPOSITION] I composed at level {row[2]} "
                         f"(confidence {row[3]:.2f}): '{row[1]}'"),
                "source": "dream_consolidation",
                "weight": 2.0,
                "neuromod_context": felt,
                "category": "composition",
            })
    except Exception as e:
        swallow_warn('[DreamBridge] Composition harvest failed', e,
                     key="modules.spirit_loop.composition_harvest_failed", throttle=100)

    # 5. P4: Recent significant social interactions (max 2)
    try:
        _et_path = os.path.join(os.path.dirname(cgn_db_path), "events_teacher.db")
        if os.path.exists(_et_path):
            _et_conn = sqlite3.connect(_et_path, timeout=5)
            _et_conn.row_factory = sqlite3.Row
            _et_rows = _et_conn.execute(
                "SELECT author, topic, felt_summary, contagion_type, "
                "sentiment, arousal FROM felt_experiences "
                "WHERE arousal > 0.5 "
                "ORDER BY created_at DESC LIMIT 4"
            ).fetchall()
            _et_conn.close()
            _soc_count = 0
            for _sr in _et_rows:
                if _soc_count >= 2:
                    break
                _soc_key = f"{_sr['author']}:{_sr['topic']}"
                if _soc_key in dedup.get("social_interactions", []):
                    continue
                dedup.setdefault("social_interactions", []).append(_soc_key)
                # Trim dedup list
                dedup["social_interactions"] = dedup["social_interactions"][-50:]
                memories.append({
                    "text": (f"[SOCIAL_INTERACTION] Felt {_sr['contagion_type'] or 'connection'} "
                             f"from @{_sr['author']} about '{_sr['topic']}': "
                             f"{_sr['felt_summary']}"),
                    "source": "dream_consolidation",
                    "weight": 2.0,
                    "neuromod_context": felt,
                    "category": "social_interaction",
                })
                _soc_count += 1
    except Exception as e:
        swallow_warn('[DreamBridge] Social interaction harvest failed', e,
                     key="modules.spirit_loop.social_interaction_harvest_failed", throttle=100)

    # Cap total + save dedup state
    memories = memories[:max_total]
    _save_bridge_dedup(dedup)
    return memories, chain_ids_to_mark
