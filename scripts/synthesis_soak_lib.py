#!/usr/bin/env python3
"""Synthesis soak — engineered multi-turn load + telemetry for P0–P10 (companion to persona_endurance.py --synthesis).

Drives sustained, *engineered* conversation tracks at T2 + T3 (devnet; **T1 mainnet
excluded by design** — synthetic load on the sovereign agent would spend real SOL +
pollute mainnet memory) and captures durable time-series telemetry over every
`/v6/synthesis/*` readout so we can analyze the whole synthesis engine under load.

Design goals (Maker, 2026-05-28):
  • Reuse the existing personas/auth harness (PersonaAgent's internal-key POST /chat).
  • ENGINEER recurrence — random chat won't trip the skill miner's ≥3-occurrence
    threshold, so the recurrence track deliberately repeats the same tool-call shape.
  • DISCOVER the resource limit via telemetry rather than aggressively throttle —
    no backpressure by default; rich resource sampling instead. A hard kill-switch
    only guards against total OOM (opt-in).
  • CRASH/RESTART RESILIENT — module or whole-Titan restarts must NOT stop the test.
    Sends + polls retry-with-backoff and continue; the run is bounded by an ABSOLUTE
    wall-clock end, so --resume after any interruption continues toward the same end.
  • DURABLE TELEMETRY — append-only JSONL flushed per write; checkpoint atomically.

Telemetry per target each poll: the full /v6/synthesis/* surface (metrics +
sovereignty + groundedness + retrieval + chain-growth + skills + coverage +
concepts + forks + oracles) + /health + a best-effort SSH VPS resource probe
(load/mem/swap — the limit-discovery signal). All under
titan-docs/sessions/synthesis_soak_<run_id>/.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger("synthesis_soak")

# ── Targets (T1 EXCLUDED — mainnet) ───────────────────────────────────────
TARGETS: dict[str, str] = {
    "T2": "http://10.135.0.6:7777",
    "T3": "http://10.135.0.6:7778",
}
# VPS hosting T2 + T3 (for the resource probe — the shared box whose limit we
# want to discover under sustained dual load).
VPS_SSH_HOST = "root@10.135.0.6"

# ── Pacing ────────────────────────────────────────────────────────────────
TURN_INTERVAL_S = 45.0        # one conversation turn per target every ~45s
POLL_INTERVAL_S = 300.0       # full /v6/synthesis/* telemetry snapshot every 5 min
RSS_SAMPLE_INTERVAL_S = 60.0  # per-worker RSS sample every 60s (fine growth curve)
SEND_RETRIES = 6              # retry a failed send (rides an agno cold-restart)
SEND_BACKOFF_S = 10.0         # base backoff between send retries (10,20,30,40,50s)
CHECKPOINT_EVERY_TURNS = 1    # checkpoint after every turn (cheap, durable)
HEALTH_RECOVERY_MAX_S = 360.0 # on send failure, wait up to 6 min for /health to recover
AGNO_RESETTLE_S = 300.0       # on detected agno restart, pause SENDING to that Titan
                              # 5 min so agno can resettle (reload fastembed + warm)
                              # before we hammer it again (Maker 2026-05-28)

# Remote per-worker RSS probe (runs on the VPS via `ssh host python3 -c`).
# Attributes each titan_hcl process to T2/T3 by /proc/<pid>/cwd, captures VmRSS +
# PID + start-time. agno_worker is identified PRECISELY by the data/agno_sessions.db
# file descriptor it uniquely holds open — NOT by "heaviest proc with embedder libs",
# because post-§3J several workers (agno, recorder, cgn) all carry fastembed/onnxruntime,
# so the heaviest-with-libs heuristic flip-flops between them and fakes a restart.
# A real agno restart = the agno pid disappears from the proc set (esp. on a >1GB
# Guardian RSS-limit kill). All proc titles are "titan_hcl" (spawn), so the open-fd
# discriminator is the only reliable per-module signal from outside the process.
_WORKER_RSS_PROBE = r'''
import os, glob, json
DIRMAP = {"/home/antigravity/projects/titan": "T2", "/home/antigravity/projects/titan3": "T3"}
out = {"T2": {"procs": [], "total_kb": 0}, "T3": {"procs": [], "total_kb": 0}}
def _is_agno(pd):
    # agno_worker is the sole holder of an open data/agno_sessions.db fd.
    try:
        for fd in os.listdir(pd + "/fd"):
            try:
                if "agno_sessions.db" in os.readlink(pd + "/fd/" + fd):
                    return True
            except OSError:
                continue
    except OSError:
        pass
    return False
for pd in glob.glob("/proc/[0-9]*"):
    pid = pd.rsplit("/", 1)[-1]
    try:
        cwd = os.readlink(pd + "/cwd")
        t = DIRMAP.get(cwd)
        if not t:
            continue
        comm = open(pd + "/comm").read().strip().lower()
        if "titan" not in comm:
            continue
        rss = 0
        for line in open(pd + "/status"):
            if line.startswith("VmRSS"):
                rss = int(line.split()[1]); break
        start = open(pd + "/stat").read().split()[21]
        # INV-PROC-7 (D-SPEC-143): workers self-identify. setproctitle writes
        # argv[0] = "titan_hcl:<name>" (read via /proc/<pid>/cmdline); PR_SET_NAME
        # writes comm = "titan:<name>". Prefer cmdline (full name, untruncated);
        # fall back to comm. None for the parent ("titan_hcl", no colon).
        worker = None
        try:
            argv0 = open(pd + "/cmdline", "rb").read().split(b"\x00")[0].decode("utf-8", "replace")
            if argv0.startswith("titan_hcl:"):
                worker = argv0[len("titan_hcl:"):]
        except Exception:
            pass
        if worker is None and comm.startswith("titan:"):
            worker = comm[len("titan:"):]
        out[t]["procs"].append({"pid": int(pid), "rss_kb": rss, "start": int(start),
                                "is_agno": _is_agno(pd), "worker": worker})
        out[t]["total_kb"] += rss
    except Exception:
        continue
for t in out:
    out[t]["procs"].sort(key=lambda p: p["rss_kb"], reverse=True)
    out[t]["proc_count"] = len(out[t]["procs"])
    out[t]["total_mb"] = round(out[t]["total_kb"] / 1024, 1)
    if out[t]["procs"]:
        hp = out[t]["procs"][0]
        out[t]["heaviest_pid"] = hp["pid"]
        out[t]["heaviest_rss_mb"] = round(hp["rss_kb"] / 1024, 1)
        out[t]["heaviest_is_agno"] = hp.get("is_agno")
    # The agno_worker proc (precise, fd-identified) — the real RSS-growth target.
    agno = next((p for p in out[t]["procs"] if p.get("is_agno")), None)
    out[t]["agno_pid"] = agno["pid"] if agno else None
    out[t]["agno_rss_mb"] = round(agno["rss_kb"] / 1024, 1) if agno else None
    out[t]["procs"] = out[t]["procs"][:8]  # top-8 by RSS
print(json.dumps(out))
'''

# Telemetry readout surface (GET; each soft-fails independently).
TELEMETRY_ENDPOINTS = [
    "/health",
    "/v6/synthesis/metrics",
    "/v6/synthesis/metrics/sovereignty",
    "/v6/synthesis/metrics/groundedness",
    "/v6/synthesis/metrics/retrieval",
    "/v6/synthesis/metrics/chain-growth",
    "/v6/synthesis/skills",
    "/v6/synthesis/skills/coverage",
    "/v6/synthesis/concepts",
    "/v6/synthesis/forks",
    "/v6/synthesis/oracles/coverage",
]


def _utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _now() -> float:
    return time.time()


# ── Engineered conversation tracks ─────────────────────────────────────────
# Each track is bound to one of the existing personas (identity continuity +
# social graph). `missions` are templated user messages; `{n}` is the cycle
# counter so recurrence tracks vary VALUES while keeping the same tool-call
# SHAPE (the miner canonicalizes on (tool_id, args-shape-hash), value-agnostic).
# recurrence=True tracks loop their missions to repeat shapes ≥3× → P8 compile
# → P9 delegate. All 5 tracks loop (recurrence=True) for a balanced round-robin
# so RECALL/cited-gate/spine/feedback get real exercise (run 1 lesson: non-loop
# tracks exhausted at 4 turns + got skipped).

@dataclass
class Track:
    name: str
    persona: str          # PERSONA_SOULS key (identity/user_id continuity)
    user_id: str          # X-Titan-User-Id (per-user memory + bundles)
    session_id: str       # stable session → multiturn recall works
    missions: list[str]
    recurrence: bool = False
    description: str = ""


TRACKS: list[Track] = [
    # REC — recurrence: repeat the SAME coding-sandbox tool shape (compute task)
    # so the miner clusters ≥3 occurrences → compiles a skill → later turns
    # should DELEGATE (P8 + P9 invocation + repair-fork-on-failure path).
    Track(
        name="recurrence_sandbox",
        persona="jake", user_id="@jakebuildsAI", session_id="synth_rec_jake",
        recurrence=True,
        description="Repeated coding-sandbox compute → skill compilation + delegation",
        # NB: prompts MUST hit the REASONING tier to activate the 'tools' feature
        # (config.toml [[chat.tiers]] reasoning regex: why|explain|how does|analyze|
        # compare|can you help|figure). Casual-tier prompts get NO tools (run-2
        # lesson — "compute 6*7" classified casual → sandbox never invoked).
        missions=[
            "Can you help me figure out the factorial of {n}? Analyze it by running it in your coding sandbox and explain the exact integer result.",
            "How would you compute the sum of the first {n} primes? Run it in your sandbox and explain how you got the number.",
            "Can you analyze fibonacci({n})? Compute it in your sandbox and explain the value.",
        ],
    ),
    # RECALL — establish a distinctive fact, then ask Titan to recall it across
    # turns → exercises RECALL sub-modes (INV-Syn-22) + the strict cited gate
    # (INV-Syn-23: only items the response actually cites get reinforced).
    Track(
        name="recall_facts",
        persona="tom", user_id="@quantumtom_mit", session_id="synth_recall_tom",
        recurrence=True,
        description="Establish→recall distinctive facts → RECALL routing + cited-use gate",
        missions=[
            "Remember this: my thesis tracking code is QG-{n}7741 and my advisor is Dr. Petrova. Acknowledge it.",
            "Can you help me recall — what was my thesis tracking code, and how does it relate to my advisor?",
            "My favorite decoherence constant this week is {n}.42 microseconds. Hold onto that.",
            "Why did I mention that decoherence constant, and can you explain what my thesis tracking code was again?",
        ],
    ),
    # ORACLE — verifiable code/math claims → coding-sandbox truth oracle verdicts
    # (P6) → scored_by coverage. Repeated shape also feeds the miner.
    Track(
        name="oracle_verify",
        persona="jake", user_id="@jakebuildsAI", session_id="synth_oracle_jake",
        recurrence=True,
        description="Verifiable code → oracle verdict + scored_by coverage",
        missions=[
            "Can you analyze whether this Python is correct? `def add(a,b): return a-b` — verify by running add({n}, 3) in your sandbox and explain why the name might mislead.",
            "How would you verify whether {n} is prime? Analyze it in your sandbox and explain the check.",
        ],
    ),
    # CONCEPT — novel-topic deep dives → P4 concept spine growth + P5 hypothesis
    # forks (net-new exploration) + groundedness accrual.
    Track(
        name="concept_dive",
        persona="peter", user_id="@peter_summits", session_id="synth_concept_peter",
        recurrence=True,
        description="Novel-topic dives → spine concepts + hypothesis forks",
        missions=[
            "Can you explain something you've never analyzed before about glacier microbial ecosystems at {n}000m altitude? Why would they matter?",
            "How do those microbes inform a theory of resilience? Analyze it and form a new idea.",
            "Can you compare glacier microbes to your own sense of endurance? What concept emerges, and why?",
            "Why might the resilience idea you formed change over time? Explain and refine it.",
        ],
    ),
    # FEEDBACK — emotional persona that, after a tool/answer turn, drives an
    # explicit Tier-2 thumbs signal via POST /v6/synthesis/feedback (INV-Syn-24).
    # The tool_call_tx is sourced best-effort from the oracles/recent surface.
    Track(
        name="feedback_tier2",
        persona="jane", user_id="@jane_and_baby_leo", session_id="synth_feedback_jane",
        recurrence=True,
        description="Explicit Tier-2 user feedback → scored_by=user override",
        missions=[
            "Baby Leo smiled today! Can you help me figure out how many days old he is if he was born {n}0 days ago? Analyze it in your sandbox.",
            "That was lovely, thank you. (I'll mark that one as helpful.)",
            "Can you explain a gentle reflection on time passing, and analyze {n}*7 days into weeks for me in your sandbox?",
        ],
    ),
]


# ── Thin chat sender (reuses PersonaAgent's internal-key auth pattern) ──────
async def _send_chat(
    client: httpx.AsyncClient, api_base: str, internal_key: str,
    user_id: str, session_id: str, message: str,
) -> dict:
    """POST /chat with internal-key auth. Never raises — returns a result dict
    with status_code=0 on transport error (mirrors PersonaAgent.send_to_titan)."""
    payload = {"message": message, "session_id": session_id, "user_id": user_id}
    headers = {"X-Titan-Internal-Key": internal_key, "X-Titan-User-Id": user_id}
    start = _now()
    try:
        resp = await client.post(f"{api_base}/chat", json=payload, headers=headers)
        elapsed = round(_now() - start, 2)
        ok = resp.status_code in (200, 403)
        body = {}
        try:
            body = resp.json()
        except Exception:
            body = {}
        return {
            "success": ok, "status_code": resp.status_code, "elapsed_s": elapsed,
            "response": (body.get("response") or body.get("error") or resp.text[:300]),
            "mode": body.get("mode", "Unknown"), "mood": body.get("mood", "Unknown"),
        }
    except Exception as e:
        return {"success": False, "status_code": 0, "elapsed_s": round(_now() - start, 2),
                "response": f"{type(e).__name__}: {e}"[:300], "mode": "Error", "mood": "N/A"}


async def _send_with_retry(client, api_base, internal_key, user_id, session_id, message) -> dict:
    """Send with retry-backoff. A failing Titan (mid-restart) is retried; we never
    abort the run — a persistent failure logs + returns the last result so the
    driver moves on to the next turn (the test survives restarts)."""
    result = {}
    for attempt in range(1, SEND_RETRIES + 1):
        result = await _send_chat(client, api_base, internal_key, user_id, session_id, message)
        if result["success"]:
            return result
        # transient (conn/timeout/5xx) → backoff + retry
        if attempt < SEND_RETRIES:
            await asyncio.sleep(SEND_BACKOFF_S * attempt)
    # All retries failed — likely a Titan/agno restart (Guardian killed agno at
    # its RSS ceiling). Wait for /health to recover, then make ONE final attempt
    # so the turn survives the restart rather than being lost.
    if await _await_health(client, api_base):
        logger.info("[recovery] %s health recovered — retrying turn", api_base)
        result = await _send_chat(client, api_base, internal_key, user_id, session_id, message)
    return result


# ── Telemetry ───────────────────────────────────────────────────────────────
async def _poll_endpoint(client: httpx.AsyncClient, api_base: str, path: str) -> dict:
    try:
        resp = await client.get(f"{api_base}{path}")
        try:
            data = resp.json()
        except Exception:
            data = {"_raw": resp.text[:200]}
        return {"status": resp.status_code, "data": data}
    except Exception as e:
        return {"status": 0, "error": f"{type(e).__name__}: {e}"[:200]}


def _ssh_resource_probe() -> dict:
    """Best-effort VPS load/mem/swap snapshot (the limit-discovery signal).
    Soft-fail — returns {} on any error so a flaky SSH never stops the soak."""
    try:
        out = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no",
             VPS_SSH_HOST,
             "cat /proc/loadavg; echo '---'; free -m | awk '/Mem:|Swap:/{print $1,$2,$3}'"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode != 0:
            return {"error": (out.stderr or "ssh_failed")[:120]}
        parts = out.stdout.split("---")
        load = parts[0].split()[:3] if parts else []
        memlines = [l.split() for l in parts[1].strip().splitlines()] if len(parts) > 1 else []
        return {"loadavg": load, "mem_swap_mb": memlines}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:120]}


def _ssh_worker_rss() -> dict:
    """Per-worker RSS for T2 + T3 via the remote probe (one SSH gets both Titans).
    Soft-fail → {} so a flaky SSH never stops the soak."""
    try:
        # Pipe the probe via STDIN to remote `python3` (reads script from stdin) —
        # avoids SSH remote-shell word-splitting of a multiline `-c` argument.
        out = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no",
             VPS_SSH_HOST, "python3", "-"],
            input=_WORKER_RSS_PROBE, capture_output=True, text=True, timeout=25,
        )
        if out.returncode != 0:
            return {"error": (out.stderr or "ssh_rss_failed")[:160]}
        return json.loads(out.stdout.strip())
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:160]}


async def _rss_sampler(run_dir: str, intended_end: float, stop: asyncio.Event,
                       backoff_until: dict):
    """Fine-grained per-worker RSS curve (60s). One record covers both Titans.
    Detects agno restarts (PID change / RSS drop on the heaviest worker) — the
    precise root-cause signal for the agno-RSS→1GB→GuardianHCL-restart pattern.
    On a detected restart, sets `backoff_until[target]` so the driver pauses
    SENDING to that Titan for AGNO_RESETTLE_S (let agno reload + warm)."""
    path = os.path.join(run_dir, "worker_rss.jsonl")
    prev_agno: dict = {}  # target → (pid, rss_mb)
    while not stop.is_set() and _now() < intended_end:
        rss = await asyncio.get_event_loop().run_in_executor(None, _ssh_worker_rss)
        rec = {"ts": _utc(), "epoch": _now(), "rss": rss, "restart_events": []}
        # Restart detection on the PRECISELY-identified agno_worker (fd-matched on
        # agno_sessions.db), keyed on the agno pid DISAPPEARING from the proc set —
        # not on "a different proc is heaviest" (which flip-flops between agno /
        # recorder / cgn, all of which carry fastembed post-§3J → phantom restarts).
        for t in ("T2", "T3"):
            tinfo = rss.get(t) if isinstance(rss, dict) else None
            if not isinstance(tinfo, dict):
                continue
            cur_pid = tinfo.get("agno_pid")
            cur_rss = tinfo.get("agno_rss_mb") or 0.0
            cur_pids = {p.get("pid") for p in (tinfo.get("procs") or [])}
            old = prev_agno.get(t)
            # Real restart: the tracked agno pid is gone (Guardian killed + respawned),
            # or its RSS collapsed in-place. A new pid alone is NOT enough — only if
            # the OLD one actually left the proc table.
            disappeared = bool(old) and old[0] is not None and old[0] not in cur_pids
            rss_collapsed = (bool(old) and old[0] == cur_pid
                             and (old[1] - cur_rss) > 300)
            if disappeared or rss_collapsed:
                rec["restart_events"].append({
                    "target": t, "old_pid": old[0], "new_pid": cur_pid,
                    "old_rss_mb": old[1], "new_rss_mb": cur_rss,
                    "reason": "pid_disappeared" if disappeared else "rss_collapsed",
                })
                logger.warning("[%s] agno_worker RESTART detected (%s; pid %s→%s, "
                               "rss %.0f→%.0f MB)", t,
                               "pid disappeared" if disappeared else "rss collapsed",
                               old[0], cur_pid, old[1], cur_rss)
                # Pause sending to this Titan so agno can resettle before we
                # hammer it again (Maker 2026-05-28). Polling/telemetry continue.
                backoff_until[t] = _now() + AGNO_RESETTLE_S
                logger.warning("[%s] agno-resettle backoff — pausing sends %.0fs",
                               t, AGNO_RESETTLE_S)
            # Track the agno pid as long as it persists; adopt the new one only
            # when the old has left (so the heaviest-flip never re-keys the tracker).
            if old and old[0] in cur_pids:
                prev_agno[t] = (old[0], cur_rss if old[0] == cur_pid else old[1])
            elif cur_pid is not None:
                prev_agno[t] = (cur_pid, cur_rss)
        _append_jsonl(path, rec)
        try:
            await asyncio.wait_for(stop.wait(), timeout=RSS_SAMPLE_INTERVAL_S)
        except asyncio.TimeoutError:
            pass


async def _await_health(client: httpx.AsyncClient, api_base: str) -> bool:
    """On a send failure (likely a Titan/agno restart), poll /health until it
    recovers (up to HEALTH_RECOVERY_MAX_S) so the driver rides the restart rather
    than burning the turn. Returns True if it came back, False on timeout."""
    deadline = _now() + HEALTH_RECOVERY_MAX_S
    while _now() < deadline:
        try:
            r = await client.get(f"{api_base}/health")
            if r.status_code == 200:
                return True
        except Exception:
            pass
        await asyncio.sleep(10.0)
    return False


async def _poller(target: str, api_base: str, client: httpx.AsyncClient,
                  telemetry_path: str, intended_end: float, stop: asyncio.Event):
    """Per-target telemetry poller. Appends one JSONL record per poll until the
    intended end. Each endpoint + the resource probe soft-fail independently."""
    while not stop.is_set() and _now() < intended_end:
        snapshot: dict = {"ts": _utc(), "epoch": _now(), "target": target, "endpoints": {}}
        for path in TELEMETRY_ENDPOINTS:
            snapshot["endpoints"][path] = await _poll_endpoint(client, api_base, path)
        # Resource probe once per poll (shared VPS — same value for T2/T3, but
        # recorded per-target stream for self-contained analysis).
        snapshot["vps_resource"] = await asyncio.get_event_loop().run_in_executor(
            None, _ssh_resource_probe)
        _append_jsonl(telemetry_path, snapshot)
        logger.info("[%s] telemetry poll written (%d endpoints)", target, len(TELEMETRY_ENDPOINTS))
        # interruptible wait
        try:
            await asyncio.wait_for(stop.wait(), timeout=POLL_INTERVAL_S)
        except asyncio.TimeoutError:
            pass


# ── Durable IO + checkpoint ─────────────────────────────────────────────────
def _append_jsonl(path: str, record: dict) -> None:
    """Append one JSON record + flush + fsync (survives a hard crash)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


@dataclass
class Checkpoint:
    path: str
    run_id: str = ""
    started_at: float = 0.0
    intended_end: float = 0.0
    targets: list[str] = field(default_factory=list)
    # per target → per track → next mission index (cursor) + turns sent
    cursors: dict = field(default_factory=dict)
    turns_sent: dict = field(default_factory=dict)

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": self.run_id, "started_at": self.started_at,
                "intended_end": self.intended_end, "targets": self.targets,
                "cursors": self.cursors, "turns_sent": self.turns_sent,
                "saved_at": _utc(),
            }, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, self.path)

    @classmethod
    def load(cls, path: str) -> Optional["Checkpoint"]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            cp = cls(path=path)
            cp.run_id = d.get("run_id", ""); cp.started_at = d.get("started_at", 0.0)
            cp.intended_end = d.get("intended_end", 0.0); cp.targets = d.get("targets", [])
            cp.cursors = d.get("cursors", {}); cp.turns_sent = d.get("turns_sent", {})
            return cp
        except Exception as e:
            logger.warning("checkpoint load failed (%s) — starting fresh", e)
            return None


# ── Driver ───────────────────────────────────────────────────────────────────
async def _driver(target: str, api_base: str, internal_key: str, client: httpx.AsyncClient,
                  chat_log_path: str, cp: Checkpoint, intended_end: float, stop: asyncio.Event,
                  backoff_until: dict):
    """Per-target conversation driver. Round-robins the tracks; each track keeps a
    cursor (resumable). Recurrence tracks loop their missions (repeat shapes).
    Honors `backoff_until[target]` — after a detected agno restart, pauses sends so
    agno can resettle (the RSS sampler sets it)."""
    cp.cursors.setdefault(target, {t.name: 0 for t in TRACKS})
    cp.turns_sent.setdefault(target, 0)
    track_idx = 0
    while not stop.is_set() and _now() < intended_end:
        # Agno-resettle backoff: if a restart was just detected for this Titan,
        # hold sends until agno has had AGNO_RESETTLE_S to reload + warm.
        settle = backoff_until.get(target, 0.0)
        if _now() < settle:
            wait_s = settle - _now()
            logger.info("[%s] settling after agno restart — holding sends %.0fs", target, wait_s)
            try:
                await asyncio.wait_for(stop.wait(), timeout=wait_s)
            except asyncio.TimeoutError:
                pass
            continue
        track = TRACKS[track_idx % len(TRACKS)]
        track_idx += 1
        cursor = cp.cursors[target].get(track.name, 0)
        # Non-recurrence tracks stop at end of missions; recurrence loops.
        if not track.recurrence and cursor >= len(track.missions):
            continue
        mission = track.missions[cursor % len(track.missions)]
        cycle_n = cursor // len(track.missions) + 1
        message = mission.replace("{n}", str(cycle_n))

        result = await _send_with_retry(
            client, api_base, internal_key, track.user_id, track.session_id, message)
        cp.cursors[target][track.name] = cursor + 1
        cp.turns_sent[target] += 1

        _append_jsonl(chat_log_path, {
            "ts": _utc(), "target": target, "track": track.name,
            "cursor": cursor, "cycle": cycle_n, "user_id": track.user_id,
            "message": message, "status_code": result.get("status_code"),
            "mode": result.get("mode"), "mood": result.get("mood"),
            "latency_s": result.get("elapsed_s"),
            "response_head": (result.get("response") or "")[:240],
        })
        if cp.turns_sent[target] % CHECKPOINT_EVERY_TURNS == 0:
            cp.save()

        # FEEDBACK track post-step: best-effort Tier-2 signal on a recent tool-call TX.
        if track.name == "feedback_tier2" and result.get("success"):
            await _maybe_post_feedback(client, api_base, internal_key, track.user_id)

        logger.info("[%s] %s turn (cycle %d) → %s %s",
                    target, track.name, cycle_n, result.get("status_code"), result.get("mode"))
        try:
            await asyncio.wait_for(stop.wait(), timeout=TURN_INTERVAL_S)
        except asyncio.TimeoutError:
            pass


async def _maybe_post_feedback(client, api_base, internal_key, user_id) -> None:
    """Source a recent tool-call tx_hash from the oracles/recent surface and POST
    an explicit Tier-2 thumbs-up (INV-Syn-24). Best-effort — skips silently if no
    tx is available (the tool_call_tx is internal; this is the soft Tier-2 path)."""
    try:
        r = await client.get(f"{api_base}/v6/synthesis/oracles/recent")
        rows = (r.json() or {}).get("verdicts") or (r.json() or {}).get("recent") or []
        tx = None
        for row in rows if isinstance(rows, list) else []:
            tx = row.get("parent_tool_call_tx") or row.get("tx_hash")
            if tx:
                break
        if not tx:
            return
        await client.post(
            f"{api_base}/v6/synthesis/feedback",
            json={"tool_call_tx": tx, "verdict": "positive"},
            headers={"X-Titan-Internal-Key": internal_key, "X-Titan-User-Id": user_id},
        )
        logger.info("[feedback] Tier-2 thumbs-up posted on tx=%s", tx[:16])
    except Exception as e:
        logger.debug("[feedback] skipped: %s", e)


# ── Tool pre-flight (never soak with tools silently down — Maker 2026-05-29) ──
_SANDBOX_DOWN_MARKERS = (
    "sandbox tool is cur", "sandbox is still init", "sandbox is cur", "unavailable",
    "initializing", "not available", "still initializing", "couldn't", "could not",
)


async def _tool_preflight(client: httpx.AsyncClient, api_base: str,
                          internal_key: str) -> tuple[bool, str]:
    """Send ONE sandbox-nudge chat + confirm a tool actually fired. The 4h soak
    last time gathered nothing because the sandbox was down 65-94% of turns
    (agno restart loop) — so we GATE on a live tool before committing. Signals:
      • coverage `total_tool_call_txs` increments (strongest), OR
      • the response is NOT a sandbox-down message.
    Returns (tools_live, detail)."""
    async def _cov() -> int:
        try:
            r = await client.get(f"{api_base}/v6/synthesis/skills/coverage")
            # skills/coverage returns a flat {denominator, numerator, ...} shape
            # (the `coverage.total_tool_call_txs` wrapper is on oracles/coverage).
            return int((r.json() or {}).get("denominator", 0))
        except Exception:
            return -1
    before = await _cov()
    res = await _send_chat(client, api_base, internal_key, "@preflight", "preflight",
                           "Can you help me analyze whether 6 * 7 equals 42? Verify it by running it "
                           "in your coding sandbox and explain the result.")
    resp = (res.get("response") or "").lower()
    down = any(m in resp for m in _SANDBOX_DOWN_MARKERS)
    # Give the procedural TX + 60s recompute a moment to land in coverage.
    after = before
    for _ in range(9):
        await asyncio.sleep(10.0)
        after = await _cov()
        if after > before >= 0:
            break
    tx_fired = after > before >= 0
    if tx_fired:
        return True, f"tool-call TX fired (coverage {before}→{after})"
    if not down and res.get("status_code") == 200:
        return True, f"no tool-call TX observed but response not sandbox-down (mode={res.get('mode')})"
    return False, f"sandbox appears DOWN (resp='{resp[:90]}', coverage {before}→{after})"


# ── Dry-run (read-only — proves the harness without synthetic load) ──────────
async def dry_run(*, internal_key: str, targets: Optional[list[str]] = None) -> int:
    """Pre-flight verification with NO synthetic chat load (respects the run gate):
      1. offline checkpoint save→load round-trip + JSONL append/fsync,
      2. track mission expansion ({n} substitution),
      3. read-only telemetry poll (GET /v6/synthesis/* + /health) per target,
      4. SSH resource probe.
    Returns 0 if every target's /health + at least the metrics route answer."""
    import tempfile
    tgts = [t for t in (targets or list(TARGETS.keys())) if t in TARGETS]
    print(f"[dry-run] targets={tgts} (T1 excluded by design)")

    # 1+2 offline machinery
    with tempfile.TemporaryDirectory() as d:
        cp = Checkpoint(path=os.path.join(d, "checkpoint.json"), run_id="dryrun",
                        started_at=_now(), intended_end=_now() + 10, targets=tgts)
        cp.cursors = {"T2": {"recurrence_sandbox": 5}}
        cp.save()
        loaded = Checkpoint.load(cp.path)
        assert loaded and loaded.cursors["T2"]["recurrence_sandbox"] == 5, "checkpoint round-trip FAILED"
        jl = os.path.join(d, "t.jsonl")
        _append_jsonl(jl, {"a": 1}); _append_jsonl(jl, {"b": 2})
        assert sum(1 for _ in open(jl)) == 2, "jsonl append FAILED"
    sample = TRACKS[0].missions[0].replace("{n}", "7")
    print(f"[dry-run] ✓ checkpoint round-trip + jsonl append OK; sample mission: {sample[:70]}…")
    print(f"[dry-run] {len(TRACKS)} tracks: {[t.name for t in TRACKS]}")

    # 3+4 read-only liveness
    ok_all = True
    async with httpx.AsyncClient(timeout=30.0) as client:
        for t in tgts:
            api = TARGETS[t]
            health = await _poll_endpoint(client, api, "/health")
            metrics = await _poll_endpoint(client, api, "/v6/synthesis/metrics")
            h_ok = health.get("status") == 200
            m_ok = metrics.get("status") == 200
            ok_all = ok_all and h_ok and m_ok
            print(f"[dry-run] {t} {api}: /health={health.get('status')} "
                  f"/v6/synthesis/metrics={metrics.get('status')} "
                  f"{'✓' if (h_ok and m_ok) else '✗'}")
    res = await asyncio.get_event_loop().run_in_executor(None, _ssh_resource_probe)
    print(f"[dry-run] VPS resource probe: {res}")
    rss = await asyncio.get_event_loop().run_in_executor(None, _ssh_worker_rss)
    for t in ("T2", "T3"):
        ti = rss.get(t, {}) if isinstance(rss, dict) else {}
        print(f"[dry-run] {t} worker RSS: total={ti.get('total_mb')}MB "
              f"procs={ti.get('proc_count')} heaviest={ti.get('heaviest_rss_mb')}MB "
              f"(agno_confirmed={ti.get('heaviest_is_agno')})")
    print(f"[dry-run] {'PASS' if ok_all else 'FAIL'} — harness ready" if ok_all
          else "[dry-run] FAIL — a target /health or /metrics did not answer")
    return 0 if ok_all else 1


# ── Entrypoint ───────────────────────────────────────────────────────────────
async def run(*, duration: int, internal_key: str, targets: Optional[list[str]] = None,
              resume: bool = False, run_dir_base: str = "titan-docs/sessions",
              allow_no_tools: bool = False) -> int:
    """Run the synthesis soak. duration seconds (absolute end), targets default T2+T3.
    On resume, continues toward the ORIGINAL intended_end (so wall-clock span holds
    across restarts). Returns 0 on clean completion, 3 if the tool pre-flight fails
    (tools down) and allow_no_tools is False."""
    tgts = targets or list(TARGETS.keys())
    tgts = [t for t in tgts if t in TARGETS]  # T1 can't be added (not in TARGETS)
    if not tgts:
        logger.error("no valid targets (T1 is excluded by design)")
        return 2

    # ── Tool pre-flight (skip on --resume) — never soak 4h with tools silently
    # down (the lesson from run 20260528_213229). GATE unless --allow-no-tools.
    if not resume:
        async with httpx.AsyncClient(timeout=120.0) as _pf:
            any_live = False
            for t in tgts:
                ok, detail = await _tool_preflight(_pf, TARGETS[t], internal_key)
                lvl = logger.info if ok else logger.error
                lvl("[preflight] %s tools_live=%s — %s", t, ok, detail)
                any_live = any_live or ok
        if not any_live and not allow_no_tools:
            logger.error(
                "[preflight] tools DOWN on all targets — aborting (the last soak "
                "gathered nothing because the sandbox was down). Fix agno/sandbox "
                "first, or pass --allow-no-tools to soak anyway (RSS/recall data only).")
            return 3
        if not any_live:
            logger.warning("[preflight] tools down but --allow-no-tools — proceeding")

    # Resume: find the most-recent run dir with a checkpoint, else new run.
    run_dir = None
    cp = None
    if resume:
        cand = sorted([d for d in os.listdir(run_dir_base)
                       if d.startswith("synthesis_soak_")], reverse=True)
        for d in cand:
            p = os.path.join(run_dir_base, d, "checkpoint.json")
            if os.path.exists(p):
                cp = Checkpoint.load(p)
                if cp:
                    run_dir = os.path.join(run_dir_base, d)
                    logger.info("RESUMING run %s (intended_end in %.0fs)",
                                cp.run_id, cp.intended_end - _now())
                    break
    if cp is None:
        run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(run_dir_base, f"synthesis_soak_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        cp = Checkpoint(path=os.path.join(run_dir, "checkpoint.json"),
                        run_id=run_id, started_at=_now(),
                        intended_end=_now() + duration, targets=tgts)
        cp.save()

    intended_end = cp.intended_end
    if _now() >= intended_end:
        logger.info("intended_end already passed — nothing to do")
        return 0

    stop = asyncio.Event()
    # Graceful stop: SIGINT/SIGTERM set the stop event so drivers + pollers exit
    # their wait_for, gather returns, and the `finally` saves the checkpoint +
    # closes clients. (Resume also works from a hard kill via the per-turn
    # checkpoint, but this guarantees a clean checkpoint-on-stop for ops.)
    try:
        loop = asyncio.get_running_loop()
        import signal as _sig
        for _s in (_sig.SIGINT, _sig.SIGTERM):
            loop.add_signal_handler(_s, stop.set)
    except (NotImplementedError, RuntimeError):
        pass  # add_signal_handler unsupported on this platform — fall back to default
    clients: dict[str, httpx.AsyncClient] = {}
    backoff_until: dict[str, float] = {}  # target → epoch; sender pauses until then
    tasks = []
    for t in tgts:
        api_base = TARGETS[t]
        clients[t] = httpx.AsyncClient(timeout=120.0)
        chat_log = os.path.join(run_dir, f"chat_{t}.jsonl")
        tele_log = os.path.join(run_dir, f"telemetry_{t}.jsonl")
        tasks.append(_driver(t, api_base, internal_key, clients[t], chat_log, cp,
                             intended_end, stop, backoff_until))
        tasks.append(_poller(t, api_base, clients[t], tele_log, intended_end, stop))

    # One per-worker RSS sampler (covers both Titans in a single SSH; 60s cadence)
    # — the precise agno-RSS-growth + restart-detection stream; sets backoff_until
    # on a detected restart so the driver pauses sends while agno resettles.
    tasks.append(_rss_sampler(run_dir, intended_end, stop, backoff_until))

    logger.info("Synthesis soak: targets=%s run_dir=%s ends_in=%.0fs",
                tgts, run_dir, intended_end - _now())
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        stop.set()
        cp.save()
        for c in clients.values():
            try:
                await c.aclose()
            except Exception:
                pass
    logger.info("Synthesis soak complete — run_dir=%s", run_dir)
    return 0
