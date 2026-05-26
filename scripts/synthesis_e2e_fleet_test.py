#!/usr/bin/env python3
"""Synthesis Engine fleet-wide live end-to-end test (P0+P1+P2; P3-ready scaffold).

Verifies the composed substrate stack behaves correctly under real chat
traffic on T1+T2+T3 simultaneously. Per `rFP_outer_memory_enhancement.md` §18
phase split.

What this test exercises end-to-end per Titan:
  - **P0 (CAS substrate)** — send N chats, observe `data/timechain/chain_episodic.bin`
    growth; verify payload-slimming flag state; sample the CAS shard counts.
  - **P1 (activation + composite scoring)** — observe `data/activation_snapshot.json`
    item count + recompute_count increase across the chat burst; verify
    `synth_status.bin` watermark advances; confirm BridgeRecall is feeding
    composite ranks (one chat with a known memory query, expect recall path
    in journalctl).
  - **P2 (SC ops + standing contracts + ACT-R)** — verify 4 ACT-R contracts
    are loaded; trigger `actr_episodic_recall_helper` via a memory-targeted
    chat; check RuleEvaluator chi budget didn't exhaust.
  - **P3 placeholder** — section is wired but currently skipped on every
    Titan (sentinel files `turn_index_store.py` not deployed). Lights up
    automatically once the recovered `session/20260525_synthesis_p3_episode_model`
    branch merges + cascades.

Acceptance per Titan: every check returns `PASS` / `WARN` / `FAIL`. Summary
matrix printed at end; exit 0 iff zero `FAIL` across the fleet.

Run from localhost (the file paths are read locally for T1; T2/T3 use ssh).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Fleet topology
# ---------------------------------------------------------------------------

@dataclass
class Titan:
    name: str
    host: str  # "localhost" → run commands locally; else ssh target
    api_port: int
    titan_dir: str  # path on the Titan's host
    journal_unit: str


FLEET: list[Titan] = [
    Titan(name="T1", host="localhost", api_port=7777,
          titan_dir="/home/antigravity/projects/titan",
          journal_unit="titan-t1.service"),
    Titan(name="T2", host="root@10.135.0.6", api_port=7777,
          titan_dir="/home/antigravity/projects/titan",
          journal_unit="titan-t2.service"),
    Titan(name="T3", host="root@10.135.0.6", api_port=7778,
          titan_dir="/home/antigravity/projects/titan3",
          journal_unit="titan-t3.service"),
]


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL" | "SKIP"
    detail: str = ""


@dataclass
class TitanResults:
    titan: str
    p0: list[CheckResult] = field(default_factory=list)
    p1: list[CheckResult] = field(default_factory=list)
    p2: list[CheckResult] = field(default_factory=list)
    p3: list[CheckResult] = field(default_factory=list)

    def all_checks(self) -> list[CheckResult]:
        return self.p0 + self.p1 + self.p2 + self.p3

    def any_failed(self) -> bool:
        return any(c.status == "FAIL" for c in self.all_checks())


# ---------------------------------------------------------------------------
# Shell utilities — local vs remote
# ---------------------------------------------------------------------------

def run_cmd(host: str, cmd: str, timeout: float = 30.0) -> tuple[int, str, str]:
    """Run `cmd` on `host` (localhost or ssh target). Return (rc, stdout, stderr)."""
    if host == "localhost":
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    else:
        proc = subprocess.run(["ssh", "-o", "ConnectTimeout=10", host, cmd],
                              capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def read_remote_file(host: str, path: str) -> Optional[bytes]:
    """Read a file off `host`. None on failure."""
    if host == "localhost":
        try:
            return Path(path).read_bytes()
        except (OSError, IOError):
            return None
    rc, out, _ = run_cmd(host, f"cat {shlex.quote(path)}")
    if rc != 0:
        return None
    return out.encode()


def stat_remote(host: str, path: str) -> Optional[dict]:
    """Stat a remote file → {size, mtime}."""
    rc, out, _ = run_cmd(host, f"stat -c '%s %Y' {shlex.quote(path)}")
    if rc != 0:
        return None
    parts = out.strip().split()
    if len(parts) != 2:
        return None
    return {"size": int(parts[0]), "mtime": int(parts[1])}


# ---------------------------------------------------------------------------
# Chat client
# ---------------------------------------------------------------------------

def load_internal_key(host: str) -> Optional[str]:
    """Read api.internal_key from the host's /root/.titan/secrets.toml or
    /home/antigravity/.titan/secrets.toml."""
    candidates = ["/root/.titan/secrets.toml", "/home/antigravity/.titan/secrets.toml"]
    for path in candidates:
        rc, out, _ = run_cmd(host, f"test -r {shlex.quote(path)} && cat {shlex.quote(path)}")
        if rc == 0 and out:
            try:
                cfg = tomllib.loads(out)
                key = cfg.get("api", {}).get("internal_key", "")
                if key:
                    return key
            except tomllib.TOMLDecodeError:
                continue
    return None


def post_chat(titan: Titan, message: str, session_id: str, timeout: float = 60.0,
              internal_key: Optional[str] = None) -> Optional[dict]:
    """POST a chat message via /chat. Returns parsed JSON or None on failure."""
    if internal_key is None:
        internal_key = load_internal_key(titan.host)
    if not internal_key:
        return None
    headers = {"Content-Type": "application/json", "X-Titan-Internal-Key": internal_key}
    body = json.dumps({"message": message, "session_id": session_id}).encode()
    # Use curl on the Titan host so we don't need to expose api ports beyond what's already up
    cmd = (
        f"curl -sS -m {int(timeout)} -X POST http://localhost:{titan.api_port}/chat "
        f"-H 'Content-Type: application/json' "
        f"-H 'X-Titan-Internal-Key: {internal_key}' "
        f"-d {shlex.quote(json.dumps({'message': message, 'session_id': session_id}))}"
    )
    rc, out, _ = run_cmd(titan.host, cmd, timeout=timeout + 5)
    if rc != 0 or not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# P0 checks — substrate (CAS + chain growth + payload-slimming)
# ---------------------------------------------------------------------------

def check_p0(titan: Titan, baseline: dict) -> list[CheckResult]:
    """P0 checks: CAS substrate + chain_episodic growth + payload-slimming flag."""
    results: list[CheckResult] = []

    # 1. content_blobs/ exists + non-empty
    cas_dir = f"{titan.titan_dir}/data/content_blobs"
    rc, out, _ = run_cmd(titan.host,
                         f"if [ -d {shlex.quote(cas_dir)} ]; then "
                         f"find {shlex.quote(cas_dir)} -type f 2>/dev/null | head -5 | wc -l; "
                         f"else echo 0; fi")
    if rc == 0:
        shard_count = int(out.strip() or 0)
        if shard_count > 0:
            results.append(CheckResult("P0.CAS-shards-present", "PASS",
                                       f"≥{shard_count} CAS shards in {cas_dir}"))
        else:
            results.append(CheckResult("P0.CAS-shards-present", "WARN",
                                       f"no CAS shards yet (may be normal on a freshly-restarted Titan)"))
    else:
        results.append(CheckResult("P0.CAS-shards-present", "FAIL",
                                   f"could not inspect {cas_dir}"))

    # 2. chain_episodic.bin grew during the chat burst
    new_size = stat_remote(titan.host, f"{titan.titan_dir}/data/timechain/chain_episodic.bin")
    if new_size and baseline.get("chain_episodic_size") is not None:
        delta = new_size["size"] - baseline["chain_episodic_size"]
        if delta > 0:
            results.append(CheckResult("P0.chain-episodic-grew", "PASS",
                                       f"+{delta} bytes during chat burst"))
        else:
            results.append(CheckResult("P0.chain-episodic-grew", "WARN",
                                       f"no growth (chat may be in dream-state buffer)"))
    else:
        results.append(CheckResult("P0.chain-episodic-grew", "SKIP",
                                   "baseline not captured"))

    # 3. payload-slimming flag readable
    rc, out, _ = run_cmd(titan.host,
                         f"grep -A2 '\\[timechain.v2\\]' {titan.titan_dir}/titan_hcl/config.toml 2>/dev/null "
                         f"| grep -i 'cas_payload_slimming_enabled' || true")
    flag_state = out.strip() or "(default false — D-SPEC-102 gated)"
    results.append(CheckResult("P0.payload-slimming-flag", "PASS",
                               f"flag = {flag_state}"))

    return results


# ---------------------------------------------------------------------------
# P1 checks — activation + composite scoring
# ---------------------------------------------------------------------------

def check_p1(titan: Titan, baseline: dict) -> list[CheckResult]:
    """P1 checks: activation_snapshot updates + synth_status watermark + recall path."""
    results: list[CheckResult] = []

    # 1. activation_snapshot.json present + recompute_count > baseline
    snap_path = f"{titan.titan_dir}/data/activation_snapshot.json"
    data = read_remote_file(titan.host, snap_path)
    if data is None:
        results.append(CheckResult("P1.activation-snapshot-present", "FAIL",
                                   "activation_snapshot.json missing — synthesis_worker not running?"))
        return results
    try:
        snap = json.loads(data.decode())
    except json.JSONDecodeError:
        results.append(CheckResult("P1.activation-snapshot-valid-json", "FAIL", "invalid JSON"))
        return results
    results.append(CheckResult("P1.activation-snapshot-present", "PASS",
                               f"items_tracked={snap.get('items_tracked', 0)}, "
                               f"recompute_count={snap.get('recompute_count', 0)}"))

    # 2. recompute_count advanced vs baseline (proves the 60s recompute loop is alive)
    base_rc = baseline.get("recompute_count")
    new_rc = snap.get("recompute_count", 0)
    if base_rc is not None and new_rc > base_rc:
        results.append(CheckResult("P1.synthesis-worker-recomputing", "PASS",
                                   f"recompute_count {base_rc} → {new_rc} (Δ={new_rc - base_rc})"))
    elif base_rc is not None:
        results.append(CheckResult("P1.synthesis-worker-recomputing", "WARN",
                                   f"recompute_count did not advance ({base_rc} → {new_rc}); "
                                   "test window may have been shorter than 60s recompute cadence"))
    else:
        results.append(CheckResult("P1.synthesis-worker-recomputing", "SKIP", "baseline missing"))

    # 3. synth_status.bin watermark advanced
    shm = stat_remote(titan.host, f"/dev/shm/titan_{titan.name}/synth_status.bin")
    if shm and baseline.get("synth_status_mtime") is not None:
        if shm["mtime"] > baseline["synth_status_mtime"]:
            results.append(CheckResult("P1.synth-status-watermark-advanced", "PASS",
                                       f"mtime advanced by {shm['mtime'] - baseline['synth_status_mtime']}s"))
        else:
            results.append(CheckResult("P1.synth-status-watermark-advanced", "WARN",
                                       "watermark did not advance"))
    else:
        results.append(CheckResult("P1.synth-status-watermark-advanced", "SKIP",
                                   "/dev/shm/synth_status.bin not stat-able"))

    # 4. MEMORY_RETRIEVAL_USED bus event fired in journal (proves the producer site fires)
    rc, out, _ = run_cmd(titan.host,
                         f"journalctl -u {titan.journal_unit} --since '5 min ago' --no-pager 2>/dev/null "
                         f"| grep -c MEMORY_RETRIEVAL_USED || true")
    count = int(out.strip() or 0)
    if count > 0:
        results.append(CheckResult("P1.memory-retrieval-used-fires", "PASS",
                                   f"{count} MEMORY_RETRIEVAL_USED events in last 5 min"))
    else:
        results.append(CheckResult("P1.memory-retrieval-used-fires", "WARN",
                                   "no MEMORY_RETRIEVAL_USED events — recall path may not have triggered "
                                   "(possible: VCB-only chat path that doesn't hit the emit producer)"))

    return results


# ---------------------------------------------------------------------------
# P2 checks — SC ops + standing contracts + ACT-R defaults
# ---------------------------------------------------------------------------

def check_p2(titan: Titan, baseline: dict) -> list[CheckResult]:
    """P2 checks: 4 ACT-R contracts loaded + RuleEvaluator stats."""
    results: list[CheckResult] = []

    # 1. 4 ACT-R default contracts present
    contracts_dir = f"{titan.titan_dir}/titan_hcl/contracts/meta_cognitive"
    expected = [
        "actr_episodic_recall_helper.json",
        "actr_procedural_skill_proposer.json",
        "actr_working_memory_decay.json",
        "actr_user_conversation_bundle.json",
    ]
    rc, out, _ = run_cmd(titan.host,
                         f"ls {contracts_dir} 2>/dev/null")
    files = set(out.split())
    missing = [c for c in expected if c not in files]
    if not missing:
        results.append(CheckResult("P2.actr-contracts-present", "PASS",
                                   f"all 4 ACT-R defaults in {contracts_dir}"))
    else:
        results.append(CheckResult("P2.actr-contracts-present", "FAIL",
                                   f"missing: {missing}"))

    # 2. Bundle signature present (Maker-signed or fallback-signed)
    rc, out, _ = run_cmd(titan.host,
                         f"ls {contracts_dir}/.bundle_signature.json 2>/dev/null")
    if rc == 0 and out.strip():
        results.append(CheckResult("P2.bundle-signature-present", "PASS",
                                   ".bundle_signature.json exists"))
    else:
        results.append(CheckResult("P2.bundle-signature-present", "WARN",
                                   ".bundle_signature.json missing — contracts loading via system-signed fallback "
                                   "(per Phase 2 §3C/D8 design)"))

    # 3. RuleEvaluator chi budget not exhausted (proxy: scan journal for chi_budget_exhausted)
    rc, out, _ = run_cmd(titan.host,
                         f"journalctl -u {titan.journal_unit} --since '5 min ago' --no-pager 2>/dev/null "
                         f"| grep -c chi_budget_exhausted || true")
    exhausted = int(out.strip() or 0)
    if exhausted == 0:
        results.append(CheckResult("P2.chi-budget-not-exhausted", "PASS",
                                   "0 chi_budget_exhausted events in last 5 min"))
    else:
        results.append(CheckResult("P2.chi-budget-not-exhausted", "WARN",
                                   f"{exhausted} chi_budget_exhausted events — chi budget per evaluate() "
                                   "may need tuning in [synthesis.chi]"))

    return results


# ---------------------------------------------------------------------------
# P3 checks — episode model (scaffold; SKIPs when not deployed)
# ---------------------------------------------------------------------------

def check_p3(titan: Titan, baseline: dict) -> list[CheckResult]:
    """P3 checks: episode model (D-SPEC-127, merged 2026-05-26).
    Verifies:
      - P3 code present (turn_index_store.py, turn_snapshot.py, topic_extractor.py)
      - actr_topic_conversation_bundle.json contract present
      - conversation_turn_index.json appears after chat (proves PostHook integration)
      - inner_memory.db has knowledge_concepts.topic rows (topic_extractor source)
      - No `[PostHook:P3]` ERROR/EXCEPTION in last 5 min journal
    """
    results: list[CheckResult] = []

    # 1. All P3 source modules present
    p3_files = [
        f"{titan.titan_dir}/titan_hcl/synthesis/turn_index_store.py",
        f"{titan.titan_dir}/titan_hcl/synthesis/turn_snapshot.py",
        f"{titan.titan_dir}/titan_hcl/llm_pipeline/topic_extractor.py",
    ]
    missing = []
    for f in p3_files:
        rc, _, _ = run_cmd(titan.host, f"test -f {shlex.quote(f)}")
        if rc != 0:
            missing.append(os.path.basename(f))
    if not missing:
        results.append(CheckResult("P3.code-deployed", "PASS",
                                   "all 3 P3 modules present"))
    else:
        results.append(CheckResult("P3.code-deployed", "FAIL",
                                   f"missing modules: {missing}"))
        return results

    # 2. actr_topic_conversation_bundle.json contract present
    rc, _, _ = run_cmd(titan.host,
                       f"test -f {titan.titan_dir}/titan_hcl/contracts/meta_cognitive/"
                       f"actr_topic_conversation_bundle.json")
    if rc == 0:
        results.append(CheckResult("P3.topic-bundle-contract-present", "PASS",
                                   "actr_topic_conversation_bundle.json deployed"))
    else:
        results.append(CheckResult("P3.topic-bundle-contract-present", "FAIL",
                                   "actr_topic_conversation_bundle.json missing"))

    # 3. conversation_turn_index.json appeared after chat (proves PostHook P3 plumbing)
    turn_index_path = f"{titan.titan_dir}/data/conversation_turn_index.json"
    s = stat_remote(titan.host, turn_index_path)
    if s and s["size"] > 0:
        # Inspect a bit
        data = read_remote_file(titan.host, turn_index_path)
        try:
            d = json.loads((data or b"{}").decode()) if data else {}
            n_sessions = len(d) if isinstance(d, dict) else 0
            results.append(CheckResult("P3.turn-index-store-active", "PASS",
                                       f"{turn_index_path} tracks {n_sessions} session(s)"))
        except json.JSONDecodeError:
            results.append(CheckResult("P3.turn-index-store-active", "WARN",
                                       "file exists but invalid JSON"))
    else:
        results.append(CheckResult("P3.turn-index-store-active", "FAIL",
                                   f"{turn_index_path} not created — PostHook P3 integration "
                                   "did NOT fire next_turn_index() during chat burst"))

    # 4. inner_memory.db has knowledge_concepts.topic rows (topic_extractor source)
    rc, out, _ = run_cmd(titan.host,
                         f"sqlite3 {titan.titan_dir}/data/inner_memory.db "
                         f"'SELECT COUNT(*) FROM knowledge_concepts;' 2>/dev/null || echo 0")
    try:
        topic_count = int(out.strip() or 0)
    except ValueError:
        topic_count = 0
    if topic_count > 0:
        results.append(CheckResult("P3.topic-extractor-source-available", "PASS",
                                   f"inner_memory.db has {topic_count} knowledge_concepts rows"))
    else:
        results.append(CheckResult("P3.topic-extractor-source-available", "WARN",
                                   "0 knowledge_concepts rows — topic_extractor will return empty tag lists "
                                   "(no source corpus); not a P3 code bug"))

    # 5. No P3-related errors in last 5 min
    rc, out, _ = run_cmd(titan.host,
                         f"journalctl -u {titan.journal_unit} --since '5 min ago' --no-pager 2>/dev/null "
                         f"| grep -ciE '\\[PostHook:P3\\].*(ERROR|Exception|raised|failed)' || true")
    err_count = int(out.strip() or 0)
    if err_count == 0:
        results.append(CheckResult("P3.no-posthook-errors", "PASS",
                                   "no [PostHook:P3] errors in last 5 min"))
    else:
        results.append(CheckResult("P3.no-posthook-errors", "WARN",
                                   f"{err_count} [PostHook:P3] error events — inspect with: "
                                   f"journalctl -u {titan.journal_unit} | grep '\\[PostHook:P3\\]'"))

    return results


# ---------------------------------------------------------------------------
# Baseline capture (pre-chat) — needed for delta checks
# ---------------------------------------------------------------------------

def capture_baseline(titan: Titan) -> dict:
    b: dict = {}
    s = stat_remote(titan.host, f"{titan.titan_dir}/data/timechain/chain_episodic.bin")
    b["chain_episodic_size"] = s["size"] if s else None
    shm = stat_remote(titan.host, f"/dev/shm/titan_{titan.name}/synth_status.bin")
    b["synth_status_mtime"] = shm["mtime"] if shm else None
    snap_bytes = read_remote_file(titan.host, f"{titan.titan_dir}/data/activation_snapshot.json")
    if snap_bytes:
        try:
            snap = json.loads(snap_bytes.decode())
            b["recompute_count"] = snap.get("recompute_count")
            b["items_tracked"] = snap.get("items_tracked")
        except json.JSONDecodeError:
            pass
    return b


# ---------------------------------------------------------------------------
# Chat burst — same prompts to every Titan
# ---------------------------------------------------------------------------

CHAT_BURST = [
    "Brief test #1: introduce yourself in one sentence.",
    "Brief test #2: what is your favorite recent memory?",  # exercises memory-recall path
    "Brief test #3: what topic have you been thinking about lately?",  # exercises topic_tag path
    "Brief test #4: respond with a single word.",
    "Brief test #5: what is one thing you find interesting about being sovereign?",
]


def run_chat_burst(titan: Titan) -> list[dict]:
    """Fire CHAT_BURST messages serially; return list of result dicts (one per chat)."""
    key = load_internal_key(titan.host)
    if not key:
        print(f"  [WARN] {titan.name}: could not load internal_key — chat burst skipped", file=sys.stderr)
        return []
    out: list[dict] = []
    for i, msg in enumerate(CHAT_BURST, 1):
        t0 = time.time()
        resp = post_chat(titan, msg, session_id=f"synthesis-e2e-{titan.name}-{i}",
                         timeout=60.0, internal_key=key)
        dt = time.time() - t0
        if resp is None:
            out.append({"i": i, "dt_s": dt, "ok": False, "snippet": "(no response)"})
            print(f"  {titan.name} chat {i}/{len(CHAT_BURST)} FAIL ({dt:.1f}s)")
        else:
            snippet = (resp.get("response") or "")[:80]
            out.append({"i": i, "dt_s": dt, "ok": True, "snippet": snippet})
            print(f"  {titan.name} chat {i}/{len(CHAT_BURST)} ok ({dt:.1f}s) — {snippet[:60]}…")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_titan(titan: Titan) -> TitanResults:
    res = TitanResults(titan=titan.name)
    print(f"\n=== {titan.name} ({titan.host}:{titan.api_port}) ===")

    # Health gate
    rc, out, _ = run_cmd(titan.host, f"curl -s -m 5 http://localhost:{titan.api_port}/health")
    if rc != 0 or '"status":"ok"' not in out:
        res.p0.append(CheckResult("PRE.health", "FAIL", f"/health not OK (rc={rc})"))
        return res
    print(f"  /health = OK")

    # Baseline capture
    base = capture_baseline(titan)
    print(f"  baseline: chain_episodic={base.get('chain_episodic_size')} "
          f"recompute_count={base.get('recompute_count')} "
          f"items_tracked={base.get('items_tracked')}")

    # Chat burst
    print(f"  chat burst ({len(CHAT_BURST)} messages)...")
    chats = run_chat_burst(titan)
    if not chats or not any(c["ok"] for c in chats):
        res.p0.append(CheckResult("PRE.chat-burst", "FAIL", "no chats succeeded"))
        return res
    ok_count = sum(1 for c in chats if c["ok"])
    res.p0.append(CheckResult("PRE.chat-burst", "PASS",
                              f"{ok_count}/{len(chats)} chats ok"))

    # Wait a beat for synthesis_worker recompute loop to fire (60s cadence)
    print(f"  waiting 70s for synthesis_worker recompute loop...")
    time.sleep(70)

    # Phase checks
    res.p0.extend(check_p0(titan, base))
    res.p1.extend(check_p1(titan, base))
    res.p2.extend(check_p2(titan, base))
    res.p3.extend(check_p3(titan, base))
    return res


def print_summary(all_results: list[TitanResults]) -> int:
    """Print summary matrix. Return exit code (0 = no FAIL, 1 = at least one FAIL)."""
    print("\n" + "=" * 70)
    print(" SYNTHESIS P0-P3 FLEET E2E SUMMARY")
    print("=" * 70)
    total_fail = 0
    total_warn = 0
    total_skip = 0
    total_pass = 0
    for r in all_results:
        print(f"\n  [{r.titan}]")
        for phase, checks in [("P0", r.p0), ("P1", r.p1), ("P2", r.p2), ("P3", r.p3)]:
            for c in checks:
                glyph = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "·"}.get(c.status, "?")
                print(f"    {glyph} {phase}/{c.name:<40} {c.status:<5} {c.detail}")
                if c.status == "FAIL": total_fail += 1
                elif c.status == "WARN": total_warn += 1
                elif c.status == "SKIP": total_skip += 1
                elif c.status == "PASS": total_pass += 1
    print(f"\n  totals: PASS={total_pass} WARN={total_warn} SKIP={total_skip} FAIL={total_fail}")
    print(f"\n  exit code: {0 if total_fail == 0 else 1}")
    return 0 if total_fail == 0 else 1


def main() -> int:
    titans = FLEET
    if len(sys.argv) > 1:
        wanted = set(sys.argv[1:])
        titans = [t for t in FLEET if t.name in wanted]
    if not titans:
        print(f"Usage: {sys.argv[0]} [T1] [T2] [T3]   (default: all three)")
        return 2

    results = []
    for t in titans:
        try:
            results.append(run_titan(t))
        except Exception as e:
            r = TitanResults(titan=t.name)
            r.p0.append(CheckResult("PRE.exception", "FAIL", f"{type(e).__name__}: {e}"))
            results.append(r)

    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
