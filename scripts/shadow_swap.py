#!/usr/bin/env python3
"""
Microkernel v2 Phase B.1 §7 — Shadow Core Swap CLI.

Thin wrapper around POST /v4/maker/shadow-swap. Connects to the running
Titan via active_api_port, triggers the orchestrator, polls /v4/upgrade-status
for live status, formats final result.

Usage:
    python scripts/shadow_swap.py [--reason TEXT] [--port PORT] [--grace SECS]

NO --force flag (per Maker design — forcing defeats cognitive-respect purpose).
On 120s grace exceeded, the swap is DEFERRED; rerun later.

Exit codes:
  0 — success
  2 — deferred (grace exceeded)
  3 — rollback (hibernate ack timeout / shadow boot failure / nginx fail)
  4 — error (orchestrator exception)
  5 — connection or auth failure (Titan not reachable / bad maker key)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import urllib.request
import urllib.error


REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIVE_PORT_FILE = REPO_ROOT / "data" / "active_api_port"


def read_active_port() -> int:
    try:
        return int(ACTIVE_PORT_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 7777


def get_maker_key() -> str | None:
    return os.environ.get("TITAN_MAKER_KEY") or os.environ.get("X_TITAN_INTERNAL_KEY")


def post_swap(host: str, port: int, reason: str, grace: float, key: str,
              b2_1_forced: bool = False) -> dict:
    url = f"http://{host}:{port}/maker/shadow-swap"
    body_dict: dict = {"reason": reason, "grace": grace}
    # Microkernel v2 Phase B.2.1 — only include the kwarg when explicitly
    # set so old kernel versions (pre-B.2.1) don't reject the unknown field.
    if b2_1_forced:
        body_dict["b2_1_forced"] = True
    body = json.dumps(body_dict).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "X-Titan-Internal-Key": key},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def fmt_result(result: dict) -> str:
    """Compact human-friendly result block."""
    outcome = result.get("outcome", "?")
    elapsed = result.get("elapsed_seconds", 0.0)
    gap = result.get("gap_seconds", 0.0)
    eid = (result.get("event_id") or "?")[:8]
    phase = result.get("phase", "?")
    reason = result.get("failure_reason") or "(none)"
    blockers = result.get("blockers_waited_on", [])
    acks = result.get("hibernate_acks", [])

    icon = {
        "ok": "✓",
        "deferred": "⏸",
        "rollback": "↶",
        "error": "✗",
    }.get(outcome, "?")

    lines = [
        f"  {icon} outcome:        {outcome.upper()}",
        f"    event_id:       {eid}",
        f"    final phase:    {phase}",
        f"    elapsed:        {elapsed:.1f}s",
    ]
    if gap > 0:
        lines.append(f"    nginx-gap:      {gap:.1f}s")
    if outcome != "ok":
        lines.append(f"    failure_reason: {reason}")
    if blockers:
        lines.append(f"    blockers waited on: {len(blockers)}")
        for b in blockers[:5]:
            n = b.get("name", "?")
            src = b.get("from", "?")
            eta = b.get("eta_seconds", 0.0)
            lines.append(f"      - {src}/{n} (eta {eta:.1f}s)")
    if acks:
        lines.append(f"    hibernate acks: {len(acks)} workers")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Titan Shadow Core Swap CLI (B.1)")
    p.add_argument("--reason", default="manual",
                   help="Reason for upgrade (logged in audit + system fork TimeChain)")
    p.add_argument("--port", type=int, default=None,
                   help=f"API port (default: read from {ACTIVE_PORT_FILE} or 7777)")
    p.add_argument("--host", default="127.0.0.1",
                   help="API host (default 127.0.0.1)")
    p.add_argument("--grace", type=float, default=120.0,
                   help="Readiness grace seconds (default 120; PLAN locked, no force)")
    p.add_argument("--force-b2-1", action="store_true",
                   help="Microkernel v2 Phase B.2.1 — force the adoption-wait "
                        "fast-path even when bus_ipc_socket_enabled=false. For "
                        "isolation testing only (no production effect today since "
                        "adoption-wait no-ops with no spawn-mode workers wired).")
    args = p.parse_args()

    port = args.port or read_active_port()
    key = get_maker_key()
    if not key:
        print("ERROR: TITAN_MAKER_KEY env var required (Maker authentication)",
              file=sys.stderr)
        return 5

    print(f"[shadow_swap] reason={args.reason!r} target=http://{args.host}:{port} grace={args.grace}s")
    print(f"[shadow_swap] no --force flag exists (Maker design); cognitive activities will be respected")
    if args.force_b2_1:
        print(f"[shadow_swap] --force-b2-1 set: B.2.1 adoption-wait fast-path enabled")
    t0 = time.time()

    # Phase 1 — kickoff the orchestrator (returns immediately with event_id)
    try:
        kickoff = post_swap(args.host, port, args.reason, args.grace, key,
                            b2_1_forced=args.force_b2_1)
    except urllib.error.HTTPError as e:
        print(f"ERROR: HTTP {e.code} from API: {e.read()[:300]}", file=sys.stderr)
        return 5
    except urllib.error.URLError as e:
        print(f"ERROR: cannot reach API: {e}", file=sys.stderr)
        return 5
    except Exception as e:
        print(f"ERROR: kickoff call failed: {e}", file=sys.stderr)
        return 5

    initial_outcome = kickoff.get("outcome", "?")
    if initial_outcome != "started":
        # Pre-flight refused (flag off, another swap active, etc.)
        print()
        print(fmt_result(kickoff))
        return _outcome_to_exit(initial_outcome)

    event_id = kickoff.get("event_id", "")
    print(f"[shadow_swap] kickoff: event_id={event_id[:8]} — polling /maker/upgrade-status...")
    print()

    # Phase 2 — poll /maker/upgrade-status for progress + final result
    last_phase = ""
    poll_interval = 1.0
    # 2026-04-27 (Fix #4): CLI polling headroom raised 240s → 1200s. After
    # the speed fixes (#1 stop_all→pause, #2 proxy interlock), real swaps
    # complete in ~30-50s. The 240s budget was too tight for the broken
    # 13-min swaps observed during today's investigation; 1200s gives
    # generous headroom without affecting fast-path users (CLI returns as
    # soon as orchestrator finishes — this is just the upper bound before
    # the CLI gives up polling and lets the user check audit log later).
    max_total = args.grace + 1200.0
    while time.time() - t0 < max_total:
        try:
            status_url = f"http://{args.host}:{port}/maker/upgrade-status?event_id={event_id}"
            req = urllib.request.Request(
                status_url,
                headers={"X-Titan-Internal-Key": key},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                envelope = json.loads(resp.read())
        except Exception as e:
            print(f"  [poll] error: {e} — retrying", file=sys.stderr)
            time.sleep(poll_interval)
            continue

        live = (envelope or {}).get("data", {}).get("live") or {}
        outcome = live.get("outcome", "?")
        phase = live.get("phase", "?")
        if phase != last_phase:
            print(f"  → phase={phase}  outcome={outcome}  elapsed={live.get('elapsed_seconds', 0):.1f}s")
            last_phase = phase
        # Terminal outcomes
        if outcome in ("ok", "deferred", "rollback", "error"):
            print()
            print(fmt_result(live))
            print()
            print(f"[shadow_swap] CLI elapsed {time.time()-t0:.1f}s")
            return _outcome_to_exit(outcome)
        time.sleep(poll_interval)

    print(f"\nERROR: polling exceeded max {max_total:.0f}s without terminal outcome", file=sys.stderr)
    return 4


def _outcome_to_exit(outcome: str) -> int:
    return {
        "ok": 0,
        "deferred": 2,
        "rollback": 3,
        "error": 4,
        # 2026-04-28 PM late: Maker full-stop refusal (BUG-T1-INNER-MEMORY-CORRUPTION).
        # Distinct from "error" so CI/scripts can distinguish "swap disabled
        # by Maker" from "swap attempted and failed".
        "refused": 6,
    }.get(outcome, 4)


if __name__ == "__main__":
    sys.exit(main())
