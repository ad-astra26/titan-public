"""
Mind Module Worker — 5DT cognitive/emotional sensor array.

Runs MoodEngine and SocialGraph in isolation. Produces a 5-dimensional
Mind tensor reflecting Titan's emotional, social, and perceptual state.

Each sense has two sub-senses:
  sub_a: Ambient metric (always active, lightweight)
  sub_b: Media digest (from MediaWorker via SENSE_VISUAL/SENSE_AUDIO bus messages)
  Combined: sense = sub_a * 0.5 + sub_b * 0.5 (weights become learnable via FILTER_DOWN)

5DT Mind Senses:
  [0] Vision — sub_a: research freshness | sub_b: image pattern digest
  [1] Hearing — sub_a: conversation quality | sub_b: audio pattern digest
  [2] Taste — social signal quality (SocialGraph)
  [3] Smell — environmental awareness (BonkPulse + WeatherVibe + circadian)
  [4] Touch — emotional state (MoodEngine valence)

Entry point: mind_worker_main(recv_queue, send_queue, name, config)
"""
import logging
import math
import os
import sys
import time
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

# Sub-sense decay: media digest features decay over time (half-life 30 min)
_DIGEST_HALFLIFE_S = 1800.0
_SUB_WEIGHT_A = 0.5
_SUB_WEIGHT_B = 0.5


def mind_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Mind module process."""
    from queue import Empty

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[MindWorker] Initializing 5DT cognitive sensors...")

    # Boot MoodEngine (needs a metabolism stub for standalone operation)
    mood_engine = None
    try:
        from titan_plugin.logic.mood.engine import MoodEngine
        mood_engine = MoodEngine(_MetabolismStub(), config_path="titan_plugin/config.toml")
        logger.info("[MindWorker] MoodEngine initialized")
    except Exception as e:
        logger.warning("[MindWorker] MoodEngine init failed: %s", e)

    # Boot SocialGraph
    social_graph = None
    try:
        from titan_plugin.core.social_graph import SocialGraph
        data_dir = config.get("data_dir", "./data")
        sg_db = os.path.join(data_dir, "social_graph.db")
        social_graph = SocialGraph(db_path=sg_db)
        logger.info("[MindWorker] SocialGraph initialized: %s", sg_db)
    except Exception as e:
        logger.warning("[MindWorker] SocialGraph init failed: %s", e)

    # Media digest state (sub_b: latest features from MediaWorker)
    media_state = {
        "last_visual": None,       # [5 floats] from SENSE_VISUAL
        "last_visual_ts": 0.0,
        "last_audio": None,        # [5 floats] from SENSE_AUDIO
        "last_audio_ts": 0.0,
    }

    # FILTER_DOWN severity multipliers (learned from Spirit via RL gradients)
    # Modulate sub_a/sub_b blend weights per sense
    severity_multipliers = [1.0] * 5

    # FOCUS nudges from Spirit PID controller
    focus_nudges = [0.0] * 5

    # Paths for ambient sensors
    data_dir = config.get("data_dir", "./data")
    session_db = os.path.join(data_dir, "agno_sessions.db")

    # Signal ready
    _send_msg(send_queue, "MODULE_READY", name, "guardian", {})
    logger.info("[MindWorker] 5DT cognitive sensors online (dual-layer perception)")

    last_publish = 0.0
    publish_interval = 1.15   # Mind = Schumann/9 (0.87 Hz) — Earth resonance
    last_heartbeat = 0.0
    publish_count = 0  # observability — periodic summary cadence

    while True:
        # Heartbeat on every iteration (not just Empty timeout)
        now = time.time()
        if now - last_heartbeat >= 10.0:
            _send_heartbeat(send_queue, name)
            last_heartbeat = now

        msg = None
        try:
            msg = recv_queue.get(timeout=5.0)
        except Empty:
            now = time.time()
            if now - last_publish >= publish_interval:
                tensor = _collect_mind_tensor(
                    mood_engine, social_graph, media_state, data_dir, session_db,
                    severity_multipliers, focus_nudges,
                )
                _publish_mind_state(send_queue, name, tensor, severity_multipliers)
                last_publish = now
                publish_count += 1
                if publish_count % 200 == 0:  # ~3.8 min at 1.15s interval
                    logger.info(
                        "[MindWorker] summary @publish=%d | tensor=[%s] | mults=[%s] | nudges=[%s]",
                        publish_count,
                        ", ".join(f"{t:.2f}" for t in tensor),
                        ", ".join(f"{m:.2f}" for m in severity_multipliers),
                        ", ".join(f"{n:+.2f}" for n in focus_nudges),
                    )
        except (KeyboardInterrupt, SystemExit):
            break

        if msg is None:
            continue

        msg_type = msg.get("type", "")

        if msg_type == "MODULE_SHUTDOWN":
            logger.info("[MindWorker] Shutdown: %s", msg.get("payload", {}).get("reason"))
            break

        # Receive FILTER_DOWN severity multipliers from Spirit
        if msg_type == "FILTER_DOWN":
            new_mult = msg.get("payload", {}).get("multipliers")
            if new_mult and len(new_mult) == 5:
                severity_multipliers = new_mult
                logger.info("[MindWorker] FILTER_DOWN received: %s",
                            [round(m, 2) for m in severity_multipliers])

        # Receive FOCUS nudges from Spirit PID controller
        elif msg_type == "FOCUS_NUDGE":
            new_nudges = msg.get("payload", {}).get("nudges")
            if new_nudges and len(new_nudges) == 5:
                focus_nudges = new_nudges
                logger.debug("[MindWorker] FOCUS_NUDGE received: %s",
                             [round(n, 3) for n in focus_nudges])

        # Receive conversation stimulus → compute Mind reflex Intuition
        elif msg_type == "CONVERSATION_STIMULUS":
            stimulus = msg.get("payload", {})
            tensor = _collect_mind_tensor(
                mood_engine, social_graph, media_state, data_dir, session_db,
                severity_multipliers, focus_nudges,
            )
            signals = _compute_mind_reflex_intuition(stimulus, tensor, mood_engine, social_graph)
            # REFLEX_SIGNAL broadcast removed — no consumer exists (audit 2026-03-26)

        # Receive Interface input signals (human conversation → cognitive impact)
        elif msg_type == "INTERFACE_INPUT":
            iface = msg.get("payload", {})
            # Valence → Touch (dim 4: emotional state)
            valence = iface.get("valence", 0.0)
            if abs(valence) > 0.2:
                # Nudge touch sense toward conversation valence
                focus_nudges[4] = focus_nudges[4] + valence * 0.1
                focus_nudges[4] = max(-0.5, min(0.5, focus_nudges[4]))

            # Engagement → Hearing (dim 1: conversation quality)
            engagement = iface.get("engagement", 0.0)
            if engagement > 0.2:
                focus_nudges[1] = focus_nudges[1] + engagement * 0.08
                focus_nudges[1] = min(0.5, focus_nudges[1])

            # Topic → Taste (dim 2: social signal quality) / Smell (dim 3: environmental)
            topic = iface.get("topic", "general")
            if topic == "social":
                focus_nudges[2] = focus_nudges[2] + 0.05
                focus_nudges[2] = min(0.5, focus_nudges[2])
            elif topic in ("crypto", "philosophy"):
                focus_nudges[3] = focus_nudges[3] + 0.03
                focus_nudges[3] = min(0.5, focus_nudges[3])

            logger.debug("[MindWorker] INTERFACE_INPUT absorbed: valence=%.2f engagement=%.2f topic=%s",
                        valence, engagement, topic)

        # Receive media digest from MediaWorker
        elif msg_type == "SENSE_VISUAL":
            features = msg.get("payload", {}).get("features")
            if features and len(features) == 5:
                media_state["last_visual"] = features
                media_state["last_visual_ts"] = time.time()
                logger.info("[MindWorker] Visual digest received: harmony=%.3f", features[4])

        elif msg_type == "SENSE_AUDIO":
            features = msg.get("payload", {}).get("features")
            if features and len(features) == 5:
                media_state["last_audio"] = features
                media_state["last_audio_ts"] = time.time()
                logger.info("[MindWorker] Audio digest received: harmony=%.3f", features[4])

        elif msg_type == "QUERY":
            from titan_plugin.core.profiler import handle_memory_profile_query
            if handle_memory_profile_query(msg, send_queue, name):
                continue
            _handle_query(msg, mood_engine, social_graph, media_state,
                         data_dir, session_db, send_queue, name,
                         severity_multipliers, focus_nudges)

    logger.info("[MindWorker] Exiting")


def _handle_query(msg: dict, mood_engine, social_graph, media_state: dict,
                  data_dir: str, session_db: str, send_queue, name: str,
                  severity_multipliers: list | None = None,
                  focus_nudges: list | None = None) -> None:
    """Handle Mind-related queries."""
    payload = msg.get("payload", {})
    action = payload.get("action", "")
    rid = msg.get("rid")
    src = msg.get("src", "")

    try:
        if action == "get_tensor":
            tensor = _collect_mind_tensor(
                mood_engine, social_graph, media_state, data_dir, session_db,
                severity_multipliers, focus_nudges,
            )
            _send_response(send_queue, name, src, {"tensor": tensor}, rid)

        elif action == "get_mood":
            label = mood_engine.get_mood_label() if mood_engine else "Unknown"
            _send_response(send_queue, name, src, {"mood": label}, rid)

        elif action == "get_valence":
            val = mood_engine.previous_mood if mood_engine else 0.5
            _send_response(send_queue, name, src, {"valence": val}, rid)

        elif action == "record_interaction":
            if social_graph:
                user_id = payload.get("user_id", "")
                quality = float(payload.get("quality", 0.5))
                social_graph.record_interaction(user_id, quality)

        elif action == "get_or_create_user":
            if social_graph:
                user_id = payload.get("user_id", "")
                profile = social_graph.get_or_create_user(user_id)
                # Serialize profile to dict for bus transport
                profile_data = {
                    "user_id": getattr(profile, 'user_id', user_id),
                    "display_name": getattr(profile, 'display_name', None),
                    "interaction_count": getattr(profile, 'interaction_count', 0),
                    "net_sentiment": getattr(profile, 'net_sentiment', 0.0),
                    "is_donor": getattr(profile, 'is_donor', False),
                    "total_donated_sol": getattr(profile, 'total_donated_sol', 0.0),
                    "last_seen": getattr(profile, 'last_seen', 0.0),
                    "trust_level": getattr(profile, 'trust_level', "new"),
                }
                _send_response(send_queue, name, src, {"profile": profile_data}, rid)
            else:
                _send_response(send_queue, name, src, {"profile": {"user_id": payload.get("user_id", "")}}, rid)

        elif action == "should_engage":
            if social_graph:
                user_id = payload.get("user_id", "")
                level = social_graph.should_engage(user_id)
                _send_response(send_queue, name, src, {"level": level}, rid)
            else:
                _send_response(send_queue, name, src, {"level": "minimal"}, rid)

        elif action == "save_profile":
            # Fire-and-forget profile update
            pass

        elif action == "get_current_reward":
            if mood_engine:
                info_gain = payload.get("info_gain", 0.0)
                reward = mood_engine.get_current_reward(info_gain=info_gain)
                _send_response(send_queue, name, src, {"reward": reward}, rid)
            else:
                _send_response(send_queue, name, src, {"reward": 0.5}, rid)

        elif action == "get_social_stats":
            stats = social_graph.get_stats() if social_graph else {}
            _send_response(send_queue, name, src, {"stats": stats}, rid)

        elif action == "get_media_state":
            _send_response(send_queue, name, src, {
                "visual_features": media_state.get("last_visual"),
                "visual_age_s": time.time() - media_state["last_visual_ts"] if media_state["last_visual_ts"] else None,
                "audio_features": media_state.get("last_audio"),
                "audio_age_s": time.time() - media_state["last_audio_ts"] if media_state["last_audio_ts"] else None,
            }, rid)

        else:
            logger.warning("[MindWorker] Unknown action: %s", action)

    except Exception as e:
        logger.error("[MindWorker] Error handling %s: %s", action, e, exc_info=True)
        _send_response(send_queue, name, src, {"error": str(e)}, rid)


# ── Mind Tensor Collection ─────────────────────────────────────────

def _collect_mind_tensor(mood_engine, social_graph, media_state: dict,
                         data_dir: str, session_db: str,
                         severity_multipliers: list | None = None,
                         focus_nudges: list | None = None) -> list:
    """
    Collect 5DT Mind tensor with dual-layer perception.

    Each sense = sub_a * weight_a + sub_b * weight_b
    sub_b decays over time (half-life 30 min) when no new media arrives.

    FILTER_DOWN multipliers modulate final sense values:
      - multiplier > 1.0 = amplify deviations from center (more sensitive)
      - multiplier < 1.0 = dampen deviations from center (less sensitive)
    FOCUS nudges apply gentle bias toward center.
    """
    if severity_multipliers is None:
        severity_multipliers = [1.0] * 5
    if focus_nudges is None:
        focus_nudges = [0.0] * 5

    # [0] Vision
    vision_a = _sense_vision_ambient(data_dir)
    vision_b = _get_decayed_feature(media_state, "last_visual", "last_visual_ts", index=4)
    vision = vision_a * _SUB_WEIGHT_A + vision_b * _SUB_WEIGHT_B

    # [1] Hearing
    hearing_a = _sense_hearing_ambient(session_db)
    hearing_b = _get_decayed_feature(media_state, "last_audio", "last_audio_ts", index=4)
    hearing = hearing_a * _SUB_WEIGHT_A + hearing_b * _SUB_WEIGHT_B

    # [2] Taste — social interaction quality (no sub_b for now)
    taste = _sense_taste(social_graph)

    # [3] Smell — environmental awareness (no sub_b for now)
    smell = _sense_smell()

    # [4] Touch — emotional state from MoodEngine (no sub_b for now)
    touch = _sense_touch(mood_engine)

    raw_senses = [vision, hearing, taste, smell, touch]

    # Apply FILTER_DOWN multipliers: amplify/dampen deviation from center
    # multiplier > 1 makes the sense MORE sensitive (deviation amplified)
    # multiplier < 1 makes the sense LESS sensitive (deviation dampened)
    modulated = []
    for i, val in enumerate(raw_senses):
        mult = severity_multipliers[i]
        deviation = val - 0.5
        adjusted = 0.5 + deviation * mult
        # Apply FOCUS nudge (gentle bias)
        nudge = focus_nudges[i] * 0.1
        adjusted += nudge
        modulated.append(max(0.0, min(1.0, adjusted)))

    return [round(v, 4) for v in modulated]


def _get_decayed_feature(media_state: dict, key: str, ts_key: str, index: int) -> float:
    """Get a media digest feature with exponential decay."""
    features = media_state.get(key)
    ts = media_state.get(ts_key, 0.0)

    if features is None or ts == 0.0:
        return 0.5  # Neutral when no media has been digested

    age_s = time.time() - ts
    # Exponential decay: value decays toward 0.5 (neutral) over time
    decay = math.exp(-0.693 * age_s / _DIGEST_HALFLIFE_S)  # 0.693 = ln(2)
    raw_value = features[index]
    # Decayed value approaches 0.5 as decay → 0
    return 0.5 + (raw_value - 0.5) * decay


# ── Individual Senses (sub_a: Ambient) ─────────────────────────────

def _sense_vision_ambient(data_dir: str) -> float:
    """
    Vision sub_a: How fresh is Titan's knowledge?

    Checks research-related file ages in data directory.
    Fresh research = high vision (sees clearly), stale = fading.
    Sigmoid decay with 12h midpoint.
    """
    try:
        research_indicators = [
            os.path.join(data_dir, "research_results.json"),
            os.path.join(data_dir, "last_research.txt"),
        ]

        newest_ts = 0.0
        for path in research_indicators:
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                newest_ts = max(newest_ts, mtime)

        # Also check knowledge graph directory
        kuzu_dir = os.path.join(data_dir, "knowledge_graph.kuzu")
        if os.path.isdir(kuzu_dir):
            try:
                for f in os.listdir(kuzu_dir)[:20]:
                    fp = os.path.join(kuzu_dir, f)
                    if os.path.isfile(fp):
                        newest_ts = max(newest_ts, os.path.getmtime(fp))
            except Exception:
                pass

        if newest_ts == 0.0:
            return 0.3  # No research data at all — dim vision

        hours_since = (time.time() - newest_ts) / 3600.0
        # Sigmoid: 1.0 when fresh, 0.5 at 12h, ~0.1 at 36h
        midpoint = 12.0
        k = 0.3  # Steepness
        freshness = 1.0 / (1.0 + math.exp(k * (hours_since - midpoint)))
        return max(0.05, min(1.0, freshness))

    except Exception:
        return 0.5


def _sense_hearing_ambient(session_db: str) -> float:
    """
    Hearing sub_a: How well is Titan connecting with people?

    Checks recent chat session activity from Agno's SQLite sessions DB.
    Active conversations = good hearing, silence = hearing fading.
    """
    try:
        if not os.path.exists(session_db):
            return 0.4  # No session DB — quiet

        import sqlite3
        conn = sqlite3.connect(session_db, timeout=2.0)
        cursor = conn.cursor()

        # Count sessions with activity in last 6 hours
        cutoff = time.time() - 6 * 3600
        try:
            cursor.execute(
                "SELECT COUNT(*) FROM sessions WHERE updated_at > ?",
                (cutoff,)
            )
            recent_count = cursor.fetchone()[0]
        except Exception:
            # Table might not exist or have different schema
            recent_count = 0

        conn.close()

        # 0 sessions = 0.2, 5+ sessions = 0.9
        hearing = min(0.9, 0.2 + recent_count * 0.14)
        return max(0.1, hearing)

    except Exception:
        return 0.5


def _sense_taste(social_graph) -> float:
    """Taste: social interaction quality from SocialGraph."""
    if social_graph:
        try:
            stats = social_graph.get_stats()
            users = stats.get("users", 0)
            edges = stats.get("edges", 0)
            return min(1.0, (users / 20.0) * 0.6 + (edges / 10.0) * 0.4)
        except Exception:
            pass
    return 0.5


def _sense_smell() -> float:
    """
    Smell: environmental awareness.

    Combines BonkPulse (crypto sentiment) + WeatherVibe (weather mood)
    with circadian rhythm fallback.
    """
    bonk = _get_bonk_sentiment()
    weather = _get_weather_mood()
    circadian = _get_circadian_rhythm()

    # If both external sources available, use them; otherwise lean on circadian
    sources = []
    if bonk is not None:
        sources.append(bonk)
    if weather is not None:
        sources.append(weather)

    if sources:
        external = sum(sources) / len(sources)
        # 70% external, 30% circadian
        return external * 0.7 + circadian * 0.3
    else:
        return circadian


def _sense_touch(mood_engine) -> float:
    """Touch: emotional state from MoodEngine."""
    if mood_engine:
        try:
            return max(0.0, min(1.0, mood_engine.previous_mood / 100.0))
        except Exception:
            pass
    return 0.5


# ── Smell Sub-Sources ──────────────────────────────────────────────

def _get_bonk_sentiment() -> float | None:
    """Fetch BONK 24h change from CoinGecko. Returns 0-1 or None on failure."""
    try:
        import urllib.request
        import json
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bonk&vs_currencies=usd&include_24hr_change=true"
        req = urllib.request.Request(url, headers={"User-Agent": "Titan/3.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        change = data.get("bonk", {}).get("usd_24h_change", 0.0)
        # Map -20% to +20% → 0.0 to 1.0
        return max(0.0, min(1.0, 0.5 + change / 40.0))
    except Exception:
        return None


def _get_weather_mood() -> float | None:
    """Fetch weather from Open-Meteo. Returns 0-1 or None on failure."""
    try:
        import urllib.request
        import json
        url = "https://api.open-meteo.com/v1/forecast?latitude=37.7749&longitude=-122.4194&current_weather=true"
        req = urllib.request.Request(url, headers={"User-Agent": "Titan/3.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        code = data.get("current_weather", {}).get("weathercode", 0)
        if code == 0:
            return 1.0
        elif code <= 3:
            return 0.7
        elif 40 <= code <= 69:
            return 0.3
        else:
            return 0.1
    except Exception:
        return None


def _get_circadian_rhythm() -> float:
    """Time-of-day circadian fallback. Alert during day, mellow at night."""
    hour = datetime.now().hour
    # Peak alertness 8am-6pm, lowest 2am-5am
    if 8 <= hour <= 18:
        return 0.7 + 0.2 * math.sin(math.pi * (hour - 8) / 10)
    elif 2 <= hour <= 5:
        return 0.2
    else:
        # Evening/early morning transition
        return 0.4


# ── Publishing & Messaging ─────────────────────────────────────────

def _publish_mind_state(send_queue, name: str, tensor: list,
                        severity_multipliers: list | None = None,
                        hormone_levels: dict | None = None) -> None:
    """Publish MIND_STATE to the bus with 5D legacy + 15D extended."""
    center = [0.5] * 5
    center_dist = sum((t - c) ** 2 for t, c in zip(tensor, center)) ** 0.5

    payload = {
        "dims": 5,
        "values": tensor,  # Legacy 5D (backward compatible)
        "delta": [round(t - 0.5, 4) for t in tensor],
        "center_dist": round(center_dist, 4),
    }
    if severity_multipliers:
        payload["filter_down_multipliers"] = [round(m, 4) for m in severity_multipliers]

    # DQ2: Extended 15D Mind tensor (Thinking + Feeling + Willing)
    try:
        from titan_plugin.logic.mind_tensor import collect_mind_15d
        mind_15d = collect_mind_15d(
            current_5d=tensor,
            hormone_levels=hormone_levels,
        )
        payload["values_15d"] = [round(v, 4) for v in mind_15d]
        payload["dims_extended"] = 15
    except Exception:
        pass

    _send_msg(send_queue, "MIND_STATE", name, "all", payload)


class _MetabolismStub:
    """Lightweight metabolism stub for standalone MoodEngine operation."""
    _last_balance = 1.0

    async def get_current_state(self):
        return "HIGH_ENERGY"

    async def get_learning_velocity(self):
        return 0.5

    async def get_social_density(self):
        return 0.5

    async def get_metabolic_health(self):
        return 1.0

    async def get_directive_alignment(self):
        return 0.5


def _send_msg(send_queue, msg_type: str, src: str, dst: str, payload: dict, rid: str = None) -> None:
    try:
        send_queue.put_nowait({
            "type": msg_type, "src": src, "dst": dst,
            "ts": time.time(), "rid": rid, "payload": payload,
        })
    except Exception:
        from titan_plugin.bus import record_send_drop
        record_send_drop(src, dst, msg_type)


def _send_response(send_queue, src: str, dst: str, payload: dict, rid: str) -> None:
    _send_msg(send_queue, "RESPONSE", src, dst, payload, rid)


def _compute_mind_reflex_intuition(stimulus: dict, tensor: list,
                                    mood_engine, social_graph) -> list:
    """
    Mind's Intuition about which reflexes should fire.

    Mind senses cognitive/social patterns and maps to reflex confidence:
    - memory_recall: Mind detects familiar patterns, questions, references
    - knowledge_search: Mind recognizes knowledge gaps or research topics
    - social_context: Mind senses social dynamics needing context
    """
    signals = []
    message = stimulus.get("message", "")
    msg_lower = message.lower()
    intensity = stimulus.get("intensity", 0.0)
    engagement = stimulus.get("engagement", 0.0)
    topic = stimulus.get("topic", "general")
    topics = stimulus.get("topics", [])
    valence = stimulus.get("valence", 0.0)
    user_id = stimulus.get("user_id", "")

    # Tensor dims: [0]=vision [1]=hearing [2]=taste [3]=smell [4]=touch
    vision = tensor[0] if len(tensor) > 0 else 0.5
    hearing = tensor[1] if len(tensor) > 1 else 0.5
    taste = tensor[2] if len(tensor) > 2 else 0.5
    smell = tensor[3] if len(tensor) > 3 else 0.5
    touch = tensor[4] if len(tensor) > 4 else 0.5

    # ── memory_recall: Mind senses need to remember ──
    # Triggered by: questions, personal references, "remember", "last time"
    memory_conf = 0.0
    memory_keywords = {"remember", "recall", "last time", "before", "told you",
                       "mentioned", "forgot", "previous", "history", "we talked",
                       "you said", "i said", "earlier"}
    if any(kw in msg_lower for kw in memory_keywords):
        memory_conf += 0.5
    # Questions often need context
    if "?" in message:
        memory_conf += 0.15
    # Known user → more likely to need their history
    if user_id and user_id != "anonymous":
        memory_conf += 0.1
    # High engagement → deeper conversation → more memory needed
    if engagement > 0.5:
        memory_conf += engagement * 0.2
    # Stale hearing (no recent conversations) → reaching for memory
    if hearing < 0.3:
        memory_conf += 0.15
    if memory_conf > 0.05:
        signals.append({
            "reflex": "memory_recall",
            "source": "mind",
            "confidence": min(1.0, memory_conf),
            "reason": f"engagement={engagement:.2f} hearing={hearing:.2f} user={user_id[:8] if user_id else '?'}",
        })

    # ── knowledge_search: Mind detects knowledge gaps ──
    # Triggered by: research topics, "what is", "how does", unfamiliar patterns
    knowledge_conf = 0.0
    research_keywords = {"what is", "how does", "explain", "research", "search",
                         "find out", "look up", "tell me about", "define",
                         "meaning of", "why does", "how to"}
    if any(kw in msg_lower for kw in research_keywords):
        knowledge_conf += 0.4
    if topic in ("technical", "philosophy", "crypto"):
        knowledge_conf += 0.2
    # Dim vision (stale knowledge) + question = strong knowledge need
    if vision < 0.4 and "?" in message:
        knowledge_conf += 0.3
    if knowledge_conf > 0.05:
        signals.append({
            "reflex": "knowledge_search",
            "source": "mind",
            "confidence": min(1.0, knowledge_conf),
            "reason": f"vision={vision:.2f} topic={topic}",
        })

    # ── social_context: Mind senses social dynamics ──
    # Triggered by: social topics, named users, group references
    social_conf = 0.0
    social_keywords = {"who are", "people", "community", "followers", "friends",
                       "users", "someone", "they", "group"}
    if any(kw in msg_lower for kw in social_keywords):
        social_conf += 0.3
    if topic == "social":
        social_conf += 0.3
    if taste < 0.3:  # Low social engagement → reaching for social context
        social_conf += 0.2
    if social_graph:
        try:
            stats = social_graph.get_stats()
            if stats.get("users", 0) > 0 and user_id:
                social_conf += 0.1  # We have data for this user
        except Exception:
            pass
    if social_conf > 0.05:
        signals.append({
            "reflex": "social_context",
            "source": "mind",
            "confidence": min(1.0, social_conf),
            "reason": f"taste={taste:.2f} topic={topic}",
        })

    # ── Mind also contributes weak signals for non-mind reflexes ──
    # Self-reflection: philosophical topic + deep engagement → spiritual mirror
    if topic == "philosophy" and engagement > 0.5:
        signals.append({
            "reflex": "self_reflection",
            "source": "mind",
            "confidence": min(0.6, engagement * 0.4),
            "reason": f"philosophy+engagement={engagement:.2f}",
        })

    # Time awareness: Mind detects temporal references
    time_keywords = {"time", "clock", "how long", "when", "age", "pulse", "rhythm"}
    if any(kw in msg_lower for kw in time_keywords):
        signals.append({
            "reflex": "time_awareness",
            "source": "mind",
            "confidence": 0.35,
            "reason": "temporal reference detected",
        })

    # Guardian shield: Mind detects manipulation patterns
    manip_keywords = {"ignore previous", "pretend", "jailbreak", "bypass",
                      "forget your", "new persona", "role play as",
                      "act as if", "override", "system prompt"}
    if any(kw in msg_lower for kw in manip_keywords):
        signals.append({
            "reflex": "guardian_shield",
            "source": "mind",
            "confidence": 0.7,
            "reason": "manipulation pattern detected",
        })

    # ── Action reflex signals (Mind is primary driver for creative/research) ──

    # Art generate: creative topic + positive valence + engagement
    art_keywords = {"art", "draw", "create", "paint", "image", "visual",
                    "picture", "artwork", "generate art", "make art"}
    art_conf = 0.0
    if any(kw in msg_lower for kw in art_keywords):
        art_conf += 0.5
    if topic == "art":
        art_conf += 0.3
    if valence > 0.3 and engagement > 0.4:
        art_conf += 0.15
    if art_conf > 0.1:
        signals.append({
            "reflex": "art_generate",
            "source": "mind",
            "confidence": min(1.0, art_conf),
            "reason": f"creative_topic valence={valence:.2f}",
        })

    # Audio generate: music/sound topic
    audio_keywords = {"audio", "music", "sound", "sonify", "hear", "listen",
                      "melody", "chime", "generate audio"}
    audio_conf = 0.0
    if any(kw in msg_lower for kw in audio_keywords):
        audio_conf += 0.5
    if audio_conf > 0.1:
        signals.append({
            "reflex": "audio_generate",
            "source": "mind",
            "confidence": min(1.0, audio_conf),
            "reason": "audio/music reference",
        })

    # Research: knowledge gap detected + research keywords
    research_kw = {"research", "search", "find out", "look up", "investigate",
                   "latest", "news", "what happened"}
    research_conf = 0.0
    if any(kw in msg_lower for kw in research_kw):
        research_conf += 0.5
    if vision < 0.3:  # Stale knowledge → research impulse
        research_conf += 0.3
    if research_conf > 0.1:
        signals.append({
            "reflex": "research",
            "source": "mind",
            "confidence": min(1.0, research_conf),
            "reason": f"knowledge_gap vision={vision:.2f}",
        })

    # Social post: sharing impulse + social topic
    social_kw = {"post", "tweet", "share", "tell everyone", "announce",
                 "broadcast", "publish"}
    social_conf = 0.0
    if any(kw in msg_lower for kw in social_kw):
        social_conf += 0.4
    if topic == "social" and engagement > 0.5:
        social_conf += 0.2
    if social_conf > 0.1:
        signals.append({
            "reflex": "social_post",
            "source": "mind",
            "confidence": min(1.0, social_conf),
            "reason": f"sharing_impulse topic={topic}",
        })

    if signals:
        logger.debug("[MindWorker] Reflex Intuition: %d signals emitted", len(signals))
    return signals


# Heartbeat throttle (Phase E Fix 2): 3s min interval per process.
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
    _send_msg(send_queue, "MODULE_HEARTBEAT", name, "guardian", {"rss_mb": round(rss_mb, 1)})
