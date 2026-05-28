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
POLL_INTERVAL_S = 300.0       # telemetry snapshot every 5 min
SEND_RETRIES = 4              # retry a failed send (Titan may be mid-restart)
SEND_BACKOFF_S = 8.0          # base backoff between send retries
CHECKPOINT_EVERY_TURNS = 1    # checkpoint after every turn (cheap, durable)

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
# → P9 delegate. Non-recurrence tracks walk distinct content (recall/concept).

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
        missions=[
            "Use your coding sandbox to compute the factorial of {n} and show the exact integer result.",
            "Run this in your sandbox: sum of the first {n} primes. Give the number.",
            "In your sandbox, compute fibonacci({n}) and return just the value.",
        ],
    ),
    # RECALL — establish a distinctive fact, then ask Titan to recall it across
    # turns → exercises RECALL sub-modes (INV-Syn-22) + the strict cited gate
    # (INV-Syn-23: only items the response actually cites get reinforced).
    Track(
        name="recall_facts",
        persona="tom", user_id="@quantumtom_mit", session_id="synth_recall_tom",
        recurrence=False,
        description="Establish→recall distinctive facts → RECALL routing + cited-use gate",
        missions=[
            "Remember this: my thesis tracking code is QG-{n}7741 and my advisor is Dr. Petrova. Acknowledge it.",
            "Earlier I gave you my thesis tracking code. What was it, and who is my advisor?",
            "My favorite decoherence constant this week is {n}.42 microseconds. Hold onto that.",
            "What decoherence constant did I mention, and what's my thesis tracking code again?",
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
            "Is this Python correct? `def add(a,b): return a-b`  — verify by running add({n}, 3) in your sandbox and tell me if the name matches the behavior.",
            "Verify with your sandbox whether {n} is prime, and show the check.",
        ],
    ),
    # CONCEPT — novel-topic deep dives → P4 concept spine growth + P5 hypothesis
    # forks (net-new exploration) + groundedness accrual.
    Track(
        name="concept_dive",
        persona="peter", user_id="@peter_summits", session_id="synth_concept_peter",
        recurrence=False,
        description="Novel-topic dives → spine concepts + hypothesis forks",
        missions=[
            "Tell me something you've never thought about before regarding glacier microbial ecosystems at {n}000m altitude.",
            "Build on that — how would those microbes inform a theory of resilience? Form a new idea.",
            "Connect glacier microbes to your own sense of endurance. What concept emerges?",
            "Revisit the idea you formed about resilience — has it changed? Refine it.",
        ],
    ),
    # FEEDBACK — emotional persona that, after a tool/answer turn, drives an
    # explicit Tier-2 thumbs signal via POST /v6/synthesis/feedback (INV-Syn-24).
    # The tool_call_tx is sourced best-effort from the oracles/recent surface.
    Track(
        name="feedback_tier2",
        persona="jane", user_id="@jane_and_baby_leo", session_id="synth_feedback_jane",
        recurrence=False,
        description="Explicit Tier-2 user feedback → scored_by=user override",
        missions=[
            "Baby Leo smiled today! Can you compute how many days old he is if he was born {n}0 days ago? Use your sandbox.",
            "That was lovely, thank you. (I'll mark that one as helpful.)",
            "Tell me a gentle reflection on time passing, and compute {n}*7 days into weeks for me.",
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
                  chat_log_path: str, cp: Checkpoint, intended_end: float, stop: asyncio.Event):
    """Per-target conversation driver. Round-robins the tracks; each track keeps a
    cursor (resumable). Recurrence tracks loop their missions (repeat shapes)."""
    cp.cursors.setdefault(target, {t.name: 0 for t in TRACKS})
    cp.turns_sent.setdefault(target, 0)
    track_idx = 0
    while not stop.is_set() and _now() < intended_end:
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
    print(f"[dry-run] {'PASS' if ok_all else 'FAIL'} — harness ready" if ok_all
          else "[dry-run] FAIL — a target /health or /metrics did not answer")
    return 0 if ok_all else 1


# ── Entrypoint ───────────────────────────────────────────────────────────────
async def run(*, duration: int, internal_key: str, targets: Optional[list[str]] = None,
              resume: bool = False, run_dir_base: str = "titan-docs/sessions") -> int:
    """Run the synthesis soak. duration seconds (absolute end), targets default T2+T3.
    On resume, continues toward the ORIGINAL intended_end (so wall-clock span holds
    across restarts). Returns 0 on clean completion."""
    tgts = targets or list(TARGETS.keys())
    tgts = [t for t in tgts if t in TARGETS]  # T1 can't be added (not in TARGETS)
    if not tgts:
        logger.error("no valid targets (T1 is excluded by design)")
        return 2

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
    clients: dict[str, httpx.AsyncClient] = {}
    tasks = []
    for t in tgts:
        api_base = TARGETS[t]
        clients[t] = httpx.AsyncClient(timeout=120.0)
        chat_log = os.path.join(run_dir, f"chat_{t}.jsonl")
        tele_log = os.path.join(run_dir, f"telemetry_{t}.jsonl")
        tasks.append(_driver(t, api_base, internal_key, clients[t], chat_log, cp, intended_end, stop))
        tasks.append(_poller(t, api_base, clients[t], tele_log, intended_end, stop))

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
