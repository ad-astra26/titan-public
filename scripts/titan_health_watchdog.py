#!/usr/bin/env python3
"""
Titan Health Watchdog — Pipeline flow assertions + error classification + twin comparison.

Runs every 30s, checks that all critical pipelines are flowing, classifies errors
as known/new, compares twin Titans, and captures crash context.

Usage:
    python scripts/titan_health_watchdog.py [--interval 30] [--log /tmp/titan_health.log]

Output: concise one-line status per check + detailed alerts for failures.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration ───────────────────────────────────────────────────

T1_API = "http://localhost:7777"
T2_API = "http://10.135.0.6:7777"
T1_LOG = "/tmp/titan_brain.log"
T2_LOG = None  # T2 log read via SSH if needed

# Pipeline assertion thresholds (in seconds since last event)
PIPELINE_THRESHOLDS = {
    "consciousness_epochs": 30,      # Epochs fire every ~7s
    "ns_transitions": 120,           # Transitions should grow within 2 min
    "chi_evaluation": 60,            # Chi runs every epoch
    "great_pulse": 600,              # GREAT PULSE every ~292s (give 10 min grace)
    "big_pulse_spirit": 30,          # Spirit BIG PULSE every ~10s
    "big_pulse_mind": 60,            # Mind BIG PULSE every ~33s
    "big_pulse_body": 600,           # Body BIG PULSE every ~292s
    "neuromod_saturation": 0.95,     # Flag if any modulator > this
}

# Known error patterns (from known_bugs_and_fixes.md)
KNOWN_ERROR_PATTERNS = [
    (r"'Guardian' object has no attribute 'process_shield'", "V2/V3 hook: process_shield (FIXED)"),
    (r"'RLProxy' object has no attribute 'decide_execution_mode'", "V2/V3 hook: decide_execution_mode (FIXED)"),
    (r"'NoneType' object has no attribute 'record_transition'", "V2/V3 hook: recorder None (FIXED)"),
    (r"'NoneType' object has no attribute 'research'", "V2/V3: research None"),
    (r"No module named 'solana'", "T2: solana SDK missing (known)"),
    (r"'>=' not supported between instances of 'str' and 'float'", "MindWorker type mismatch (FIXED)"),
    (r"1 validation error for create_tools", "Agno tool validation (known)"),
    (r"get_balance failed", "Network: solana balance check (known)"),
    (r"Pre-hook.*execution failed", "V2/V3 pre-hook failure (known)"),
]

# ── HTTP Client ─────────────────────────────────────────────────────

def api_get(base_url: str, path: str, timeout: float = 5.0) -> dict | None:
    """Fetch JSON from API. Returns None on any failure."""
    import urllib.request
    import urllib.error
    try:
        url = f"{base_url}{path}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return data.get("data", data)
    except Exception:
        return None


# ── Pipeline Health Checks ──────────────────────────────────────────

class PipelineMonitor:
    """Tracks pipeline flow over time and detects stalls."""

    def __init__(self, name: str):
        self.name = name
        self._history: list[dict] = []  # [{ts, snapshot}, ...]
        self._alerts: list[str] = []

    def record(self, snapshot: dict):
        self._history.append({"ts": time.time(), "data": snapshot})
        # Keep last 60 entries (~30 min at 30s interval)
        if len(self._history) > 60:
            self._history.pop(0)

    def get_delta(self, key: str, window_seconds: float = 60.0) -> float | None:
        """Get change in a metric over the last N seconds."""
        now = time.time()
        cutoff = now - window_seconds
        recent = [h for h in self._history if h["ts"] >= cutoff]
        if len(recent) < 2:
            return None
        first_val = recent[0]["data"].get(key)
        last_val = recent[-1]["data"].get(key)
        if first_val is None or last_val is None:
            return None
        return last_val - first_val

    @property
    def latest(self) -> dict:
        return self._history[-1]["data"] if self._history else {}


def check_titan(api_base: str, name: str, monitors: dict) -> dict:
    """Run all pipeline assertions for one Titan. Returns status dict."""
    status = {"name": name, "ok": True, "alerts": [], "metrics": {}}

    # Fetch all data
    trinity = api_get(api_base, "/v4/inner-trinity")
    ns_data = api_get(api_base, "/v4/nervous-system")
    chi_data = api_get(api_base, "/v4/chi")
    health = api_get(api_base, "/health")

    if not trinity:
        status["ok"] = False
        status["alerts"].append("API_DOWN: /v4/inner-trinity unreachable")
        return status

    # ── Extract metrics ──
    ns = trinity.get("neural_nervous_system", {})
    nm = trinity.get("neuromodulators", {})
    pi = trinity.get("pi_heartbeat", {})
    dreaming = trinity.get("dreaming", {})
    tick_count = trinity.get("tick_count", 0)

    ns_transitions = ns_data.get("total_transitions", 0) if ns_data else ns.get("total_transitions", 0)
    ns_train = ns_data.get("total_train_steps", 0) if ns_data else ns.get("total_train_steps", 0)
    chi_total = chi_data.get("total", 0) if chi_data and isinstance(chi_data, dict) and "total" in chi_data else 0

    epoch_count = trinity.get("tick_count", 0)  # tick_count tracks consciousness ticks
    dream_cycles = dreaming.get("cycle_count", 0)
    is_dreaming = dreaming.get("is_dreaming", False)
    dev_age = pi.get("developmental_age", 0)
    cluster_count = pi.get("cluster_count", 0)

    # Guardian status
    guardian = {}
    if health and isinstance(health, dict):
        guardian = health.get("v3", {}).get("guardian_status", {})

    uptime = health.get("v3", {}).get("boot_time", 0) if health else 0

    # π-heartbeat metrics
    pi_rate = pi.get("heartbeat_ratio", 0)
    pi_total_events = pi.get("total_pi_epochs", 0)

    # Reasoning metrics
    reasoning = trinity.get("reasoning", {})
    reasoning_chains = reasoning.get("total_chains", 0)
    reasoning_commits = reasoning.get("total_conclusions", 0)

    # Record snapshot
    snapshot = {
        "ns_transitions": ns_transitions,
        "ns_train": ns_train,
        "chi": chi_total,
        "tick_count": tick_count,
        "dream_cycles": dream_cycles,
        "is_dreaming": is_dreaming,
        "dev_age": dev_age,
        "cluster_count": cluster_count,
        "pi_rate": pi_rate,
        "pi_events": pi_total_events,
        "reasoning_chains": reasoning_chains,
        "reasoning_commits": reasoning_commits,
    }

    if name not in monitors:
        monitors[name] = PipelineMonitor(name)
    mon = monitors[name]
    mon.record(snapshot)

    # ── Pipeline assertions ──

    # 1. NS transitions growing?
    ns_delta = mon.get_delta("ns_transitions", 120)
    if ns_delta is not None and ns_delta == 0 and uptime > 180:
        status["alerts"].append(f"NS_STALL: 0 new transitions in 120s (total={ns_transitions})")
        status["ok"] = False

    # 2. Consciousness ticks growing?
    tick_delta = mon.get_delta("tick_count", 60)
    if tick_delta is not None and tick_delta == 0 and uptime > 60:
        status["alerts"].append(f"EPOCH_STALL: 0 new ticks in 60s (total={tick_count})")
        status["ok"] = False

    # 3. Chi evaluating?
    if chi_total == 0 and uptime > 120:
        status["alerts"].append("CHI_DEAD: Chi=0 after 2min uptime")
        status["ok"] = False

    # 4. Neuromod saturation check
    for mod_name, mod_data in nm.items():
        if isinstance(mod_data, dict):
            level = mod_data.get("level", 0.5)
            if level > PIPELINE_THRESHOLDS["neuromod_saturation"]:
                status["alerts"].append(
                    f"NEUROMOD_SAT: {mod_name}={level:.3f} (>{PIPELINE_THRESHOLDS['neuromod_saturation']})")

    # 5. Module health
    for mod_name, mod_info in guardian.items():
        if isinstance(mod_info, dict):
            state = mod_info.get("state", "")
            restarts = mod_info.get("restart_count", 0)
            if state == "disabled":
                status["alerts"].append(f"MODULE_DISABLED: {mod_name}")
                status["ok"] = False
            elif restarts >= 3:
                status["alerts"].append(f"MODULE_UNSTABLE: {mod_name} restarts={restarts}")

    # 6. π-heartbeat health (critical architecture signal)
    if pi_rate < 0.01 and uptime > 300:
        status["alerts"].append(
            f"PI_CRITICAL: π-rate={pi_rate*100:.2f}% (<1%) — curvature near-zero, "
            f"6 systems may degrade (dreaming, 5-HT, language, sovereignty)")
        status["ok"] = False
    elif pi_rate < 0.03 and uptime > 300:
        status["alerts"].append(
            f"PI_LOW: π-rate={pi_rate*100:.1f}% (<3%) — monitor for degradation")

    # 7. Bus timeout rate (architecture congestion signal)
    try:
        _bus_result = subprocess.run(
            ["tail", "-100", "/tmp/titan_brain.log"],
            capture_output=True, text=True, timeout=5)
        _bus_timeouts = _bus_result.stdout.count("Request timed out")
        _bus_drops = _bus_result.stdout.count("Queue full")
        if _bus_drops > 0:
            status["alerts"].append(
                f"BUS_DROP: {_bus_drops} queue drops in last 100 lines — messages being lost!")
            status["ok"] = False
        elif _bus_timeouts > 20:
            status["alerts"].append(
                f"BUS_CONGESTION: {_bus_timeouts} timeouts in last 100 lines — bus overloaded")
    except Exception:
        pass

    # 8. Dreaming health (only check after sufficient uptime)
    fatigue_threshold = dev_age * 50 + 100
    expected_dream_time = fatigue_threshold * 7  # ~7s per epoch
    if uptime > expected_dream_time * 1.5 and dream_cycles == 0:
        status["alerts"].append(
            f"DREAM_NEVER: 0 dream cycles after {uptime:.0f}s "
            f"(expected after ~{expected_dream_time:.0f}s at dev_age={dev_age})")

    # Build metrics summary
    ns_delta_60 = mon.get_delta("ns_transitions", 60)
    status["metrics"] = {
        "epochs": tick_count,
        "ns": ns_transitions,
        "ns_delta": f"+{ns_delta_60:.0f}" if ns_delta_60 is not None else "?",
        "train": ns_train,
        "chi": f"{chi_total:.2f}" if chi_total else "0",
        "dreams": dream_cycles,
        "dreaming": is_dreaming,
        "dev_age": dev_age,
        "clusters": cluster_count,
        "pi_rate": f"{pi_rate*100:.1f}%",
        "pi_events": pi_total_events,
        "reasoning": f"{reasoning_commits}/{reasoning_chains}" if reasoning_chains > 0 else "idle",
        "uptime": f"{uptime:.0f}s",
    }

    # Count program fires
    if ns_data and "programs" in ns_data:
        firing = []
        for pname, pdata in ns_data["programs"].items():
            fc = pdata.get("fire_count", 0)
            if fc > 0:
                firing.append(f"{pname}={fc}")
        if firing:
            status["metrics"]["fires"] = ", ".join(firing)

    return status


# ── Error Classification ────────────────────────────────────────────

class ErrorClassifier:
    """Classifies log errors as known or new."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._last_pos = 0
        self._known_counts: dict[str, int] = defaultdict(int)
        self._new_errors: list[str] = []
        self._total_new = 0
        self._total_known = 0
        # Seek to end of file on start (only process NEW errors)
        try:
            self._last_pos = os.path.getsize(log_path)
        except OSError:
            pass

    def scan(self) -> dict:
        """Scan new log lines for errors. Returns classification."""
        self._new_errors.clear()
        new_lines = self._read_new_lines()

        for line in new_lines:
            if "ERROR" not in line and "CRITICAL" not in line:
                continue

            classified = False
            for pattern, label in KNOWN_ERROR_PATTERNS:
                if re.search(pattern, line):
                    self._known_counts[label] += 1
                    self._total_known += 1
                    classified = True
                    break

            if not classified:
                self._new_errors.append(line.strip()[:200])
                self._total_new += 1

        return {
            "new_errors": list(self._new_errors),
            "new_count": len(self._new_errors),
            "known_total": self._total_known,
            "total_new_all_time": self._total_new,
        }

    def _read_new_lines(self) -> list[str]:
        """Read lines added since last scan."""
        try:
            size = os.path.getsize(self.log_path)
            if size < self._last_pos:
                # Log was rotated/truncated
                self._last_pos = 0

            with open(self.log_path, "r") as f:
                f.seek(self._last_pos)
                lines = f.readlines()
                self._last_pos = f.tell()
                return lines
        except OSError:
            return []


# ── Crash Context Capture ───────────────────────────────────────────

class CrashCapture:
    """Captures context when Guardian restarts a module."""

    def __init__(self, log_path: str, output_dir: str = "/tmp/titan_crashes"):
        self.log_path = log_path
        self.output_dir = output_dir
        self._last_pos = 0
        self._captured_count = 0
        os.makedirs(output_dir, exist_ok=True)
        try:
            self._last_pos = os.path.getsize(log_path)
        except OSError:
            pass

    def scan(self) -> list[str]:
        """Check for new crashes, capture context. Returns list of crash files."""
        captured = []
        try:
            size = os.path.getsize(self.log_path)
            if size < self._last_pos:
                self._last_pos = 0

            with open(self.log_path, "r") as f:
                f.seek(self._last_pos)
                lines = f.readlines()
                self._last_pos = f.tell()

            for i, line in enumerate(lines):
                if "Module" in line and "crashed" in line:
                    # Capture surrounding context (50 lines before, 10 after)
                    start = max(0, i - 50)
                    end = min(len(lines), i + 10)
                    context = lines[start:end]

                    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    self._captured_count += 1
                    fname = f"crash_{ts}_{self._captured_count}.log"
                    fpath = os.path.join(self.output_dir, fname)

                    with open(fpath, "w") as cf:
                        cf.write(f"# Crash detected: {line.strip()}\n")
                        cf.write(f"# Captured: {ts}\n")
                        cf.write(f"# Context: lines {start}-{end} of new log chunk\n\n")
                        cf.writelines(context)

                    captured.append(fpath)

        except OSError:
            pass
        return captured


# ── Main Loop ───────────────────────────────────────────────────────

def format_status_line(t1: dict, t2: dict | None, errors: dict) -> str:
    """Format one-line status summary."""
    ts = datetime.now().strftime("%H:%M:%S")
    parts = [f"[{ts}]"]

    for status in [t1, t2]:
        if status is None:
            continue
        name = status["name"]
        m = status["metrics"]
        icon = "OK" if status["ok"] else "ALERT"

        line = (
            f"{name}: {icon} | "
            f"ep={m.get('epochs', '?')} "
            f"NS={m.get('ns', '?')}({m.get('ns_delta', '?')}) "
            f"Chi={m.get('chi', '?')} "
            f"dreams={m.get('dreams', 0)} "
        )
        if m.get("dreaming"):
            line += "ZZZ "
        parts.append(line.rstrip())

    if errors.get("new_count", 0) > 0:
        parts.append(f"| NEW_ERRORS: {errors['new_count']}")

    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Titan Health Watchdog")
    parser.add_argument("--interval", type=int, default=30, help="Check interval (seconds)")
    parser.add_argument("--log", type=str, default="/tmp/titan_health.log", help="Output log path")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--no-t2", action="store_true", help="Skip T2 checks")
    args = parser.parse_args()

    monitors: dict[str, PipelineMonitor] = {}
    error_classifier = ErrorClassifier(T1_LOG)
    crash_capture = CrashCapture(T1_LOG)

    log_file = open(args.log, "a") if args.log else None

    def output(line: str):
        print(line)
        if log_file:
            log_file.write(line + "\n")
            log_file.flush()

    output(f"[{datetime.now().strftime('%H:%M:%S')}] Titan Health Watchdog started "
           f"(interval={args.interval}s, log={args.log})")

    while True:
        try:
            # Check T1
            t1_status = check_titan(T1_API, "T1", monitors)

            # Check T2
            t2_status = None
            if not args.no_t2:
                t2_status = check_titan(T2_API, "T2", monitors)

            # Classify errors
            errors = error_classifier.scan()

            # Capture crashes
            crashes = crash_capture.scan()

            # Output status line
            status_line = format_status_line(t1_status, t2_status, errors)
            output(status_line)

            # Output alerts
            for alert in t1_status.get("alerts", []):
                output(f"  T1 ALERT: {alert}")
            if t2_status:
                for alert in t2_status.get("alerts", []):
                    output(f"  T2 ALERT: {alert}")

            # Output new errors
            for err in errors.get("new_errors", []):
                output(f"  NEW_ERROR: {err}")

            # Output crash captures
            for crash_file in crashes:
                output(f"  CRASH_CAPTURED: {crash_file}")

            if args.once:
                break

            time.sleep(args.interval)

        except KeyboardInterrupt:
            output(f"[{datetime.now().strftime('%H:%M:%S')}] Watchdog stopped (Ctrl+C)")
            break
        except Exception as e:
            output(f"[{datetime.now().strftime('%H:%M:%S')}] Watchdog error: {e}")
            time.sleep(args.interval)

    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
