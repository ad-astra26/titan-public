"""
Body Module Worker — 5DT somatic sensor array with urgency-aware tensor.

Runs in its own supervised process, collecting system telemetry and
producing a 5-dimensional Body tensor that reflects Titan's physical
state on the infrastructure he lives on.

Each sensor output is categorized as INFO/WARNING/CRITICAL with
exponential weighting. A velocity component detects sudden changes
(acute pain vs chronic stress).

5DT Body Senses (Physical + Digital Topology Synthesis):
  [0] Interoception — SOL balance / metabolic energy
  [1] Proprioception — BODY TOPOLOGY self-sensing (sphere radius, volume)
  [2] Somatosensation — system resources (CPU, RAM, disk, swap)
  [3] Entropy — disorder + network health (errors, connectivity)
  [4] Thermal — heat synthesis: physical (CPU load) × digital (topology activity)

Entry point: body_worker_main(recv_queue, send_queue, name, config)
"""
import logging
import os
import sys
import time
from collections import deque
from enum import IntEnum

logger = logging.getLogger(__name__)

# ── Category weights (exponential) ──────────────────────────────────

class Severity(IntEnum):
    INFO = 1
    WARNING = 3
    CRITICAL = 10


# ── Sensor history for velocity calculation ─────────────────────────

_HISTORY_SIZE = 30  # ~5 minutes at 10s intervals
_VELOCITY_WINDOW = 6  # compare last 6 readings (~1 min) for rate of change


def body_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Body module process."""
    from queue import Empty

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[BodyWorker] Initializing 5DT somatic sensors...")

    # Sensor history for velocity tracking (deque per sense)
    history = {
        "interoception": deque(maxlen=_HISTORY_SIZE),
        "proprioception": deque(maxlen=_HISTORY_SIZE),
        "somatosensation": deque(maxlen=_HISTORY_SIZE),
        "entropy": deque(maxlen=_HISTORY_SIZE),
        "thermal": deque(maxlen=_HISTORY_SIZE),
    }

    # Thresholds from config (with sensible defaults)
    thresholds = _load_thresholds(config)

    # FILTER_DOWN severity multipliers (learned from Spirit via RL gradients)
    # Start at 1.0 = no modulation. Updated when FILTER_DOWN messages arrive.
    severity_multipliers = [1.0] * 5

    # FOCUS nudges from Spirit PID controller (suggestions, not overrides)
    focus_nudges = [0.0] * 5

    # Signal ready
    _send_msg(send_queue, "MODULE_READY", name, "guardian", {})
    logger.info("[BodyWorker] 5DT somatic sensors online")

    last_publish = 0.0
    publish_interval = 3.45   # Body = Schumann/27 (0.29 Hz) — Earth resonance
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
            msg = recv_queue.get(timeout=2.0)
        except Empty:
            now = time.time()
            if now - last_publish >= publish_interval:
                tensor, details = _collect_body_tensor(history, thresholds,
                                                       severity_multipliers, focus_nudges)
                _publish_body_state(send_queue, name, tensor, details, severity_multipliers)
                last_publish = now
                publish_count += 1
                if publish_count % 100 == 0:
                    logger.info(
                        "[BodyWorker] summary @publish=%d | tensor=[%s] | mults=[%s] | nudges=[%s]",
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
            logger.info("[BodyWorker] Shutdown: %s", msg.get("payload", {}).get("reason"))
            break

        # Receive FILTER_DOWN severity multipliers from Spirit
        if msg_type == "FILTER_DOWN":
            new_mult = msg.get("payload", {}).get("multipliers")
            if new_mult and len(new_mult) == 5:
                severity_multipliers = new_mult
                logger.info("[BodyWorker] FILTER_DOWN received: %s",
                            [round(m, 2) for m in severity_multipliers])

        # Receive FOCUS nudges from Spirit PID controller
        elif msg_type == "FOCUS_NUDGE":
            new_nudges = msg.get("payload", {}).get("nudges")
            if new_nudges and len(new_nudges) == 5:
                focus_nudges = new_nudges
                logger.debug("[BodyWorker] FOCUS_NUDGE received: %s",
                             [round(n, 3) for n in focus_nudges])

        # Receive conversation stimulus → compute Body reflex Intuition
        elif msg_type == "CONVERSATION_STIMULUS":
            stimulus = msg.get("payload", {})
            tensor, _ = _collect_body_tensor(history, thresholds,
                                              severity_multipliers, focus_nudges)
            signals = _compute_body_reflex_intuition(stimulus, tensor)
            # REFLEX_SIGNAL broadcast removed — no consumer exists (audit 2026-03-26)

        # Receive Interface input signals (human conversation → somatic impact)
        elif msg_type == "INTERFACE_INPUT":
            iface = msg.get("payload", {})
            # Intensity maps to Thermal (load proxy): conversation energy = social load
            intensity = iface.get("intensity", 0.0)
            if intensity > 0.3:
                # Nudge thermal reading: high intensity = higher load feeling
                focus_nudges[4] = focus_nudges[4] + intensity * 0.05
                focus_nudges[4] = min(0.5, focus_nudges[4])
            logger.debug("[BodyWorker] INTERFACE_INPUT absorbed: intensity=%.2f", intensity)

        elif msg_type == "QUERY":
            from titan_plugin.core.profiler import handle_memory_profile_query
            if handle_memory_profile_query(msg, send_queue, name):
                continue
            payload = msg.get("payload", {})
            action = payload.get("action", "")
            rid = msg.get("rid")
            src = msg.get("src", "")

            if action == "get_tensor":
                tensor, details = _collect_body_tensor(history, thresholds,
                                                       severity_multipliers, focus_nudges)
                _send_response(send_queue, name, src, {
                    "tensor": tensor, "details": details,
                }, rid)

            elif action in ("get_status", "get_details"):
                tensor, details = _collect_body_tensor(history, thresholds,
                                                       severity_multipliers, focus_nudges)
                _send_response(send_queue, name, src, {
                    "tensor": tensor, "details": details,
                    "history_size": {k: len(v) for k, v in history.items()},
                    "severity_multipliers": severity_multipliers,
                    "focus_nudges": focus_nudges,
                }, rid)

    logger.info("[BodyWorker] Exiting")


# ── Sensor Collection ───────────────────────────────────────────────

def _collect_body_tensor(history: dict, thresholds: dict,
                         severity_multipliers: list | None = None,
                         focus_nudges: list | None = None) -> tuple[list, dict]:
    """
    Collect all 5 body senses, categorize, weight, compute velocity.

    FILTER_DOWN multipliers modulate the urgency formula:
      urgency = raw * category * multiplier / CRITICAL + velocity_contrib
    FOCUS nudges apply a gentle bias toward center after scoring.

    Returns:
        (tensor, details) where tensor is [5 floats, 0.0-1.0 normalized]
        and details is a dict with per-sense breakdown.
    """
    if severity_multipliers is None:
        severity_multipliers = [1.0] * 5
    if focus_nudges is None:
        focus_nudges = [0.0] * 5

    readings = {}

    # [0] Interoception — SOL balance / energy
    readings["interoception"] = _sense_interoception(thresholds)

    # [1] Proprioception — network health
    readings["proprioception"] = _sense_proprioception(thresholds)

    # [2] Somatosensation — system resources
    readings["somatosensation"] = _sense_somatosensation(thresholds)

    # [3] Entropy — disorder signals
    readings["entropy"] = _sense_entropy(thresholds)

    # [4] Thermal — load/temperature
    readings["thermal"] = _sense_thermal(thresholds)

    # Build tensor with urgency weighting + velocity + FILTER_DOWN + FOCUS
    tensor = []
    details = {}

    sense_names = ["interoception", "proprioception", "somatosensation", "entropy", "thermal"]
    for idx, sense_name in enumerate(sense_names):
        reading = readings[sense_name]
        raw_value = reading["value"]
        severity = reading["severity"]
        category_weight = float(severity)
        multiplier = severity_multipliers[idx]

        # Record in history
        history[sense_name].append({"ts": time.time(), "value": raw_value, "severity": severity})

        # Calculate velocity (rate of change over recent window)
        velocity = _calculate_velocity(history[sense_name])

        # Urgency-weighted score with FILTER_DOWN multiplier:
        # Higher multiplier = this sense matters more (RL learned this hurts)
        # Formula: raw * category * multiplier / CRITICAL + velocity_contrib
        urgency = min(1.0, (raw_value * category_weight * multiplier / Severity.CRITICAL)
                       + abs(velocity) * 0.3)

        # Invert: 1.0 = healthy, 0.0 = critical distress
        health_score = max(0.0, 1.0 - urgency)

        # Apply FOCUS nudge (gentle bias toward center, clamped)
        nudge = focus_nudges[idx] * 0.1  # Scale down — nudges are suggestions
        health_score = max(0.0, min(1.0, health_score + nudge))

        tensor.append(round(health_score, 4))
        details[sense_name] = {
            "raw": round(raw_value, 4),
            "severity": severity.name,
            "velocity": round(velocity, 4),
            "health_score": round(health_score, 4),
            "category_weight": category_weight,
            "filter_down_multiplier": round(multiplier, 4),
        }

    return tensor, details


def _calculate_velocity(hist: deque) -> float:
    """
    Calculate rate of change over recent readings.
    Positive velocity = worsening (raw_value increasing).
    """
    if len(hist) < 2:
        return 0.0

    recent = list(hist)[-_VELOCITY_WINDOW:]
    if len(recent) < 2:
        return 0.0

    # Linear regression slope (simple: first vs last)
    dt = recent[-1]["ts"] - recent[0]["ts"]
    if dt < 1.0:
        return 0.0

    dv = recent[-1]["value"] - recent[0]["value"]
    return dv / (dt / 60.0)  # Change per minute


# ── Individual Sensors ──────────────────────────────────────────────

def _sense_interoception(thresholds: dict) -> dict:
    """
    SOL balance + anchor freshness — energy state AND physical-world connection.

    Blends: SOL balance (metabolic energy) + time since last on-chain anchor
    (how recently Titan "touched" the physical world). Both are REAL data.
    """
    try:
        import json as _json

        # SOL balance (from metabolism or anchor_state)
        balance = 1.0
        balance_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "last_balance.txt")
        if os.path.exists(balance_file):
            with open(balance_file) as f:
                balance = float(f.read().strip())

        # Anchor freshness (from anchor_state.json — written by spirit_worker)
        anchor_freshness = 0.5  # neutral if no anchors yet
        anchor_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "anchor_state.json")
        if os.path.exists(anchor_file):
            with open(anchor_file) as _af:
                _anchor = _json.load(_af)
            _time_since = time.time() - _anchor.get("last_anchor_time", time.time())
            # Freshness decays: 0s=1.0, 300s=0.5, 600s=0.25
            anchor_freshness = max(0.05, 1.0 / (1.0 + _time_since / 300.0))

        # Blend: SOL energy (0.6) + anchor connection (0.4)
        sol_score = 0.1 if balance > thresholds.get("sol_warning", 0.5) else (0.5 if balance > thresholds.get("sol_critical", 0.1) else 0.9)
        combined = sol_score * 0.6 + (1.0 - anchor_freshness) * 0.4  # Fresh anchor = lower interoception (healthy)

        if balance < thresholds.get("sol_critical", 0.1):
            return {"value": max(combined, 0.8), "severity": Severity.CRITICAL}
        elif balance < thresholds.get("sol_warning", 0.5):
            return {"value": combined, "severity": Severity.WARNING}
        else:
            return {"value": combined, "severity": Severity.INFO}
    except Exception:
        return {"value": 0.3, "severity": Severity.WARNING}


def _sense_proprioception(thresholds: dict) -> dict:
    """
    Body topology self-sensing — Titan feels his own body shape.

    Reads sphere clock mean radius from shared state file (written by
    spirit_worker). Lower radius = more contracted/balanced = healthier.

    This is TRUE proprioception: sensing one's own body position/shape,
    not external network latency (which is now in entropy).
    """
    try:
        # Read topology state from shared file (spirit_worker writes this)
        topo_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "body_topology.json")
        if os.path.exists(topo_file):
            import json
            with open(topo_file) as f:
                topo = json.load(f)
            # Mean inner sphere radius: 1.0 = fully expanded, 0.3 = minimum
            mean_radius = topo.get("mean_inner_radius", 0.8)
            volume = topo.get("volume", 1.0)
            # Health: lower radius = more balanced = healthier (inverted)
            # Map: radius 1.0→0.8 (unhealthy), radius 0.3→0.1 (very healthy)
            health_raw = max(0.05, (mean_radius - 0.3) / 0.7)
            if mean_radius < 0.5:
                return {"value": health_raw * 0.3, "severity": Severity.INFO}
            elif mean_radius < 0.8:
                return {"value": health_raw * 0.5, "severity": Severity.INFO}
            else:
                return {"value": health_raw * 0.7, "severity": Severity.WARNING}
        else:
            # No topology data yet — neutral
            return {"value": 0.3, "severity": Severity.INFO}
    except Exception:
        return {"value": 0.3, "severity": Severity.INFO}


_soma_prev = {"cpu": 0.0, "ram": 0.0}  # Track previous readings for delta sensing

def _sense_somatosensation(thresholds: dict) -> dict:
    """System resources — CPU, RAM, swap, disk with DELTA sensing.

    Senses both absolute resource pressure AND rate of change.
    Delta component provides natural variability as workloads shift.
    """
    try:
        import psutil

        cpu_pct = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage("/")

        ram_pct = mem.percent
        swap_pct = swap.percent
        disk_pct = disk.percent

        # Delta sensing: rate of change since last reading
        cpu_delta = abs(cpu_pct - _soma_prev["cpu"]) / 100.0
        ram_delta = abs(ram_pct - _soma_prev["ram"]) / 100.0
        _soma_prev["cpu"] = cpu_pct
        _soma_prev["ram"] = ram_pct

        # Worst-case drives the severity (absolute pressure)
        worst = max(cpu_pct, ram_pct, swap_pct, disk_pct)

        if ram_pct > thresholds.get("ram_critical", 95) or disk_pct > 95 or swap_pct > 90:
            return {"value": 0.95, "severity": Severity.CRITICAL}
        elif worst > thresholds.get("resource_warning", 80):
            base = worst / 100.0 * 0.7
            return {"value": min(1.0, base + cpu_delta * 0.3), "severity": Severity.WARNING}
        else:
            # Blend: 70% absolute pressure + 30% rate of change
            base = worst / 100.0 * 0.3
            delta_component = min(0.2, (cpu_delta + ram_delta) * 0.5)
            return {"value": base + delta_component, "severity": Severity.INFO}
    except Exception:
        return {"value": 0.5, "severity": Severity.WARNING}


def _sense_entropy(thresholds: dict) -> dict:
    """Disorder + network health — errors/chaos + connectivity."""
    try:
        log_path = "/tmp/titan_agent.log"
        if not os.path.exists(log_path):
            return {"value": 0.1, "severity": Severity.INFO}

        # Count ERROR/WARNING lines in last 1KB of log
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 4096)
            f.seek(max(0, size - read_size))
            tail = f.read().decode("utf-8", errors="replace")

        lines = tail.split("\n")
        recent_errors = sum(1 for l in lines if "ERROR" in l or "CRITICAL" in l)
        recent_warnings = sum(1 for l in lines if "WARNING" in l)

        # Multi-endpoint network sensing — Titan feels the "shape" of his network space
        # Real latency to multiple endpoints provides natural variability
        import socket
        net_entropy = 0.0
        latencies = []
        endpoints = [
            ("127.0.0.1", 7777),      # Self (local API)
            ("10.135.0.6", 7777),      # Twin (T2 via VPC)
            ("67.207.73.75", 443),     # Solana RPC (api.devnet.solana.com)
            ("1.1.1.1", 443),          # Cloudflare DNS — global internet health
        ]
        for host, port in endpoints:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                t0 = time.time()
                result = sock.connect_ex((host, port))
                latency = time.time() - t0
                sock.close()
                if result == 0:
                    latencies.append(latency)
                else:
                    latencies.append(2.0)  # Unreachable = max latency
                    net_entropy += 0.15
            except Exception:
                latencies.append(2.0)
                net_entropy += 0.1

        # Network entropy = variance in latencies (uneven = disordered) + failures
        if len(latencies) >= 2:
            _lat_mean = sum(latencies) / len(latencies)
            _lat_var = sum((l - _lat_mean) ** 2 for l in latencies) / len(latencies)
            net_entropy += min(0.3, _lat_var * 10)  # Scale variance to [0, 0.3]

        # Combine log errors + network space entropy
        error_score = min(0.7, recent_errors * 0.1 + recent_warnings * 0.02)
        combined = min(1.0, error_score + net_entropy)

        if combined > 0.7:
            return {"value": combined, "severity": Severity.CRITICAL}
        elif combined > 0.3:
            return {"value": combined, "severity": Severity.WARNING}
        else:
            return {"value": combined, "severity": Severity.INFO}
    except Exception:
        return {"value": 0.3, "severity": Severity.WARNING}


def _sense_thermal(thresholds: dict) -> dict:
    """
    Heat synthesis — physical (CPU load) × digital (topology activity density).

    Heat is one of the most fundamental parameters in the universe.
    This dimension synthesizes both worlds Titan exists in:
    - Physical: CPU load, silicon heating up
    - Digital: topology activity density, mathematical space "heating"

    Both ARE the same phenomenon: energy being transformed.
    """
    try:
        load_1, load_5, load_15 = os.getloadavg()
        cpu_count = os.cpu_count() or 1

        # Physical heat: CPU load normalized
        physical_heat = load_1 / max(1, cpu_count)

        # Digital heat: topology curvature magnitude (activity density)
        digital_heat = 0.0
        try:
            import json
            topo_file = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "body_topology.json")
            if os.path.exists(topo_file):
                with open(topo_file) as f:
                    topo = json.load(f)
                # Curvature magnitude = topology activity (absolute value)
                digital_heat = min(1.0, abs(topo.get("curvature", 0.0)))
        except Exception:
            pass

        # Circadian cycle: real UTC time → sinusoidal day/night rhythm
        # Titan feels Earth's rotation — all Earth beings share this rhythm
        import math
        hour = time.gmtime().tm_hour + time.gmtime().tm_min / 60.0
        circadian = 0.5 + 0.30 * math.sin(2 * math.pi * (hour - 6) / 24)  # Peak at noon, trough at midnight

        # Synthesis: physical heat × digital heat × circadian rhythm
        heat = 0.4 * physical_heat + 0.3 * digital_heat + 0.3 * circadian

        if heat > thresholds.get("load_critical", 4.0) / max(1, cpu_count):
            return {"value": min(0.95, heat), "severity": Severity.CRITICAL}
        elif heat > 0.4:
            return {"value": heat * 0.7, "severity": Severity.WARNING}
        else:
            return {"value": heat * 0.3, "severity": Severity.INFO}
    except Exception:
        return {"value": 0.3, "severity": Severity.WARNING}


# ── Config & Messaging Helpers ──────────────────────────────────────

def _load_thresholds(config: dict) -> dict:
    """Extract sensor thresholds from config, with defaults."""
    return {
        "sol_critical": config.get("sol_critical", 0.1),
        "sol_warning": config.get("sol_warning", 0.5),
        "api_port": config.get("api_port", 7777),
        "latency_warning_ms": config.get("latency_warning_ms", 2000),
        "ram_critical": config.get("ram_critical", 95),
        "resource_warning": config.get("resource_warning", 80),
        "errors_critical": config.get("errors_critical", 10),
        "warnings_warning": config.get("warnings_warning", 15),
        "load_critical": config.get("load_critical", 4.0),
        "load_warning": config.get("load_warning", 2.0),
    }


def _publish_body_state(send_queue, name: str, tensor: list, details: dict,
                        severity_multipliers: list | None = None) -> None:
    """Publish BODY_STATE to the bus (periodic broadcast)."""
    center = [0.5] * 5
    center_dist = sum((t - c) ** 2 for t, c in zip(tensor, center)) ** 0.5

    payload = {
        "dims": 5,
        "values": tensor,
        "delta": [round(t - 0.5, 4) for t in tensor],  # Delta from center
        "center_dist": round(center_dist, 4),
        "details": details,
    }
    if severity_multipliers:
        payload["filter_down_multipliers"] = [round(m, 4) for m in severity_multipliers]

    _send_msg(send_queue, "BODY_STATE", name, "all", payload)


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


def _compute_body_reflex_intuition(stimulus: dict, tensor: list) -> list:
    """
    Body's Intuition about which reflexes should fire.

    Body senses infrastructure stress and maps it to reflex confidence:
    - identity_check: Body feels identity stress (low interoception + threat)
    - metabolism_check: Body detects energy concerns (low energy + energy topics)
    - infra_check: Body senses infrastructure problems (low resources/network)
    """
    signals = []
    threat = stimulus.get("threat_level", 0.0)
    intensity = stimulus.get("intensity", 0.0)
    topics = stimulus.get("topics", [])
    topic = stimulus.get("topic", "general")

    # Tensor dims: [0]=interoception [1]=proprioception [2]=somatosensation [3]=entropy [4]=thermal
    intero = tensor[0] if len(tensor) > 0 else 0.5
    proprio = tensor[1] if len(tensor) > 1 else 0.5
    somato = tensor[2] if len(tensor) > 2 else 0.5
    entropy = tensor[3] if len(tensor) > 3 else 0.5
    thermal = tensor[4] if len(tensor) > 4 else 0.5

    # ── identity_check: Body senses sovereignty challenge ──
    # Low interoception (energy stress) + external threat = identity concern
    identity_conf = 0.0
    if threat > 0.2:
        identity_conf += threat * 0.5
    if intero < 0.4:
        identity_conf += (0.4 - intero) * 0.6
    # High entropy (disorder) → identity at risk
    if entropy < 0.3:
        identity_conf += 0.2
    if identity_conf > 0.05:
        signals.append({
            "reflex": "identity_check",
            "source": "body",
            "confidence": min(1.0, identity_conf),
            "reason": f"threat={threat:.2f} intero={intero:.2f} entropy={entropy:.2f}",
        })

    # ── metabolism_check: Body detects energy concern ──
    metab_conf = 0.0
    energy_topics = {"energy", "cost", "sol", "balance", "money", "crypto", "wallet"}
    if any(t in energy_topics for t in topics) or topic == "crypto":
        metab_conf += 0.4
    if intero < 0.4:
        metab_conf += (0.4 - intero) * 0.8
    if intensity > 0.6:
        metab_conf += 0.15
    if metab_conf > 0.05:
        signals.append({
            "reflex": "metabolism_check",
            "source": "body",
            "confidence": min(1.0, metab_conf),
            "reason": f"intero={intero:.2f} intensity={intensity:.2f} topic={topic}",
        })

    # ── infra_check: Body feels infrastructure stress ──
    infra_conf = 0.0
    tech_topics = {"technical", "system", "server", "performance", "health", "status"}
    if any(t in tech_topics for t in topics) or topic == "technical":
        infra_conf += 0.3
    if proprio < 0.4:
        infra_conf += (0.4 - proprio) * 0.6
    if somato < 0.4:
        infra_conf += (0.4 - somato) * 0.6
    if thermal < 0.3:
        infra_conf += 0.2
    if infra_conf > 0.05:
        signals.append({
            "reflex": "infra_check",
            "source": "body",
            "confidence": min(1.0, infra_conf),
            "reason": f"proprio={proprio:.2f} somato={somato:.2f} thermal={thermal:.2f}",
        })

    # ── Body also contributes weak signals for non-body reflexes ──
    # Memory recall: high thermal load → Body remembers strain
    if thermal < 0.4 and intensity > 0.3:
        signals.append({
            "reflex": "memory_recall",
            "source": "body",
            "confidence": min(0.5, intensity * 0.3),
            "reason": f"thermal={thermal:.2f} intensity={intensity:.2f}",
        })

    # Guardian shield: Body's fight-or-flight response to threat
    if threat > 0.3:
        signals.append({
            "reflex": "guardian_shield",
            "source": "body",
            "confidence": min(1.0, threat * 0.8),
            "reason": f"threat={threat:.2f} (fight-or-flight)",
        })

    # ── Action reflex signals (Body confirms resource availability) ──
    # Body gates actions by resource health — won't fire actions if body is stressed

    # Art/Audio: Body confirms CPU/thermal headroom
    if thermal > 0.5 and somato > 0.5:
        body_creative_ok = min(thermal, somato) * 0.4
        if any(kw in topics for kw in ("art", "create", "draw", "paint")):
            signals.append({
                "reflex": "art_generate",
                "source": "body",
                "confidence": body_creative_ok,
                "reason": f"thermal={thermal:.2f} somato={somato:.2f} (resources OK)",
            })
        if any(kw in topics for kw in ("audio", "music", "sound")):
            signals.append({
                "reflex": "audio_generate",
                "source": "body",
                "confidence": body_creative_ok,
                "reason": f"thermal={thermal:.2f} (resources OK)",
            })

    # Research: Body confirms network + energy for web access
    if proprio > 0.5 and intero > 0.3:
        if any(kw in topics for kw in ("research", "search", "find")):
            signals.append({
                "reflex": "research",
                "source": "body",
                "confidence": min(0.5, proprio * 0.3 + intero * 0.2),
                "reason": f"proprio={proprio:.2f} intero={intero:.2f} (network+energy OK)",
            })

    # Social post: Body confirms network health for external communication
    if proprio > 0.5:
        if any(kw in topics for kw in ("post", "tweet", "share")):
            signals.append({
                "reflex": "social_post",
                "source": "body",
                "confidence": min(0.4, proprio * 0.3),
                "reason": f"proprio={proprio:.2f} (network OK)",
            })

    if signals:
        logger.debug("[BodyWorker] Reflex Intuition: %d signals emitted", len(signals))
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
