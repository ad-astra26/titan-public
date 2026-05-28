"""titan_hcl/logic/spirit_helpers.py — pure spirit-loop helper functions.

rFP §3G Phase 10C — extracted from ``titan_hcl/modules/spirit_loop.py`` (the
orphan helper module left behind by the deleted ``spirit_worker`` after the
D8-3 spirit→cognitive migration). Per the function-ownership audit §1.2
(``AUDIT_spirit_loop_function_ownership.md``) these are PURE helpers — keyword
matching + scalar arithmetic, numpy-free slope math, deterministic file I/O,
and small bus-publish wrappers. None of them carry heavy ML / CGN
dependencies, so importing this module must NOT pull ``torch`` or any ``cgn``
package (enforced by ``tests/test_logic_spirit_helpers_no_heavy_imports.py``).

Canonical import surface for the relocated helpers. Consumers:
  - ``logic/reflex_intuition.py``      → ``_compute_spirit_reflex_intuition``
  - ``logic/inner_spirit_sidecar.py``  → ``_load_birth_state``
  - ``modules/spirit_loop.py``         → trajectory / bus / felt / dedup helpers
    (consumed by the consciousness functions still resident there pending the
    Phase 10D cognitive_worker absorption + Rust-SHM consumer-fix)
  - ``modules/observatory_worker.py``  → ``_send_msg`` (10E snapshot builders)
  - ``modules/meditation_worker.py``   → felt + bridge-dedup helpers (10G)
"""
from __future__ import annotations

import json
import logging
import os
import time

from titan_hcl import bus
from titan_hcl.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)

# rFP_trinity_130d_awakening §12 / SPEC §23.6 SAT[0,5] — birth_state
# always-on cache. Loaded once on first call from data/birth_dna_snapshot.json
# (or birth_state.json if present). Required for inner_spirit
# self_recognition + origin_connection dims.
_BIRTH_STATE_CACHE: list | None = None
_BIRTH_STATE_LOADED: bool = False


def _load_birth_state() -> list | None:
    """Load Titan's birth-DNA vector. Cached after first call.

    Returns None only on first-load failure; once loaded, the cached value
    is reused for the lifetime of the process.

    rFP_trinity_130d_phase2_5_closure §4 (2026-05-08): updated to handle
    the rich birth_dna_snapshot.json schema where the top-level ``dna``
    key is a dict (neuromodulator_dna / expression_composites /
    sphere_clock / consciousness / neural_nervous_system / _meta) rather
    than a flat list. We now derive a stable 45-element birth vector
    deterministically from the file's ``hash`` field — same hash → same
    birth vector across restarts. Accepts the legacy flat-list schemas
    too (birth_state.json, ``birth_state``/``vector``/``state``/``dna`` as list).
    """
    global _BIRTH_STATE_CACHE, _BIRTH_STATE_LOADED
    if _BIRTH_STATE_LOADED:
        return _BIRTH_STATE_CACHE
    _BIRTH_STATE_LOADED = True  # set even on failure to avoid re-trying
    # rFP_trinity_130d_phase2_5_closure §4 (2026-05-08, T2 deploy fix):
    # T2/T3 deployments may not have birth_dna_snapshot.json at all
    # (the file is T1-canonical from initial setup, not git-synced per
    # directive_t2_deployment_safety.md). When no file exists, derive
    # a deterministic birth vector from the Titan's ID — so each Titan
    # has a STABLE-PER-TITAN birth identity, and SPEC §23.6 SAT[0]
    # self_recognition (cosine_sim(spirit[:3], birth[:3])) computes
    # against a constant reference vector unique to that Titan.
    _titan_id_for_seed = "T1"
    try:
        from titan_hcl.core.state_registry import resolve_titan_id
        _titan_id_for_seed = resolve_titan_id()
    except Exception:
        pass
    try:
        # Worker runs from project root; prefer birth_dna_snapshot.json which
        # is the active artifact in data/, fall back to birth_state.json.
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..",
                          "data", "birth_dna_snapshot.json"),
            os.path.join(os.path.dirname(__file__), "..", "..",
                          "data", "birth_state.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                # Schema A — bare list
                if isinstance(data, list):
                    _BIRTH_STATE_CACHE = data
                elif isinstance(data, dict):
                    # Schema B — flat list under known key
                    for key in ("birth_state", "vector", "state"):
                        v = data.get(key)
                        if isinstance(v, list):
                            _BIRTH_STATE_CACHE = v
                            break
                    # Schema C — current birth_dna_snapshot.json: ``dna``
                    # is a nested dict. Derive a deterministic 45-element
                    # vector from the file's ``hash`` field. The hash is
                    # stable across the Titan's lifetime so birth[:3]
                    # used by SPEC §23.6 SAT[0] self_recognition stays
                    # constant. Distribution is uniform [0, 1].
                    if not _BIRTH_STATE_CACHE:
                        h = data.get("hash") or ""
                        if isinstance(h, str) and len(h) >= 64:
                            # Use SHA-256 hex (64 chars) as 32 bytes; expand
                            # to 45 floats by hashing again with a counter
                            # suffix until we have enough bytes.
                            import hashlib as _hl
                            seed_bytes = bytes.fromhex(h)
                            need_bytes = 45 * 4  # 4 bytes per float (uint32)
                            buf = bytearray()
                            counter = 0
                            while len(buf) < need_bytes:
                                round_in = seed_bytes + counter.to_bytes(4, "little")
                                buf.extend(_hl.sha256(round_in).digest())
                                counter += 1
                            vec = []
                            for i in range(45):
                                u32 = int.from_bytes(buf[i * 4:i * 4 + 4],
                                                     "little")
                                vec.append((u32 % 10000) / 10000.0)
                            _BIRTH_STATE_CACHE = vec
                if _BIRTH_STATE_CACHE:
                    break
    except Exception as _e:
        swallow_warn('[SpiritLoop] birth_state load', _e,
                     key="modules.spirit_loop.birth_state_load", throttle=1)
    # No file — derive deterministic 45D vector from titan_id (T2/T3
    # deploys without birth_dna_snapshot.json land here). Stable across
    # restarts; unique per Titan.
    if not _BIRTH_STATE_CACHE:
        try:
            import hashlib as _hl
            seed_bytes = _hl.sha256(
                f"titan-birth-{_titan_id_for_seed}".encode()).digest()
            need_bytes = 45 * 4
            buf = bytearray()
            counter = 0
            while len(buf) < need_bytes:
                round_in = seed_bytes + counter.to_bytes(4, "little")
                buf.extend(_hl.sha256(round_in).digest())
                counter += 1
            vec = []
            for i in range(45):
                u32 = int.from_bytes(buf[i * 4:i * 4 + 4], "little")
                vec.append((u32 % 10000) / 10000.0)
            _BIRTH_STATE_CACHE = vec
            logger.info(
                "[SpiritLoop] No birth_dna_snapshot.json found — "
                "derived deterministic 45D birth vector from "
                "titan_id=%s", _titan_id_for_seed)
        except Exception:
            _BIRTH_STATE_CACHE = [0.5] * 45
    return _BIRTH_STATE_CACHE


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
            swallow_warn('[logic.spirit_helpers] _compute_spirit_reflex_intuition: spirit_velocity = unified_spirit.velocity', _swallow_exc,
                         key='logic.spirit_helpers._compute_spirit_reflex_intuition.line288', throttle=100)

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
            swallow_warn('[logic.spirit_helpers] _compute_spirit_reflex_intuition: total_pulses = sum((c.pulse_count for c in sphere_clock.c...', _swallow_exc,
                         key='logic.spirit_helpers._compute_spirit_reflex_intuition.line352', throttle=100)
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


def _compute_trajectory(recent_epochs: list, num_dims: int = None) -> "StateVector":
    """Compute trajectory (linear slopes) over the rolling window.

    Handles mixed-dimension epochs gracefully — if previous epochs had 9D
    and current has 67D, missing dimensions slope from zero.
    """
    from titan_hcl.logic.consciousness import StateVector, NUM_DIMS, EXTENDED_NUM_DIMS

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
        swallow_warn("[logic.spirit_helpers] _save_bridge_dedup: with open(_BRIDGE_DEDUP_PATH, 'w') as f: _dj.dump(dedup, f)", _swallow_exc,
                     key='logic.spirit_helpers._save_bridge_dedup.line2458', throttle=100)
