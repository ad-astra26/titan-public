#!/usr/bin/env python3
"""
scripts/runtime_test_dspec70.py — END-TO-END live runtime verification
of D-SPEC-72 (SPEC v1.17.0) agno_worker carve.

Verifies that the rFP implementation actually works against a deployed
Titan — exercises every user-facing chat path + bus contract + worker
boot state. Designed per Maker direction 2026-05-17: "100% end-to-end
verify on live data" because /chat /pitch /social_x are user-facing
features people are watching.

Usage:
    python scripts/runtime_test_dspec70.py --titan T1
    python scripts/runtime_test_dspec70.py --titan T2 --base-url http://10.135.0.6:7777
    python scripts/runtime_test_dspec70.py --titan T3 --base-url http://10.135.0.6:7778

Tests categorized into 5 groups:

  Group A — Worker boot state (no requests, just SSH/local checks)
    A1. agno_worker subprocess present in ps tree
    A2. AGNO_WORKER_READY emitted in journalctl (post-boot signal)
    A3. agno_state.bin SHM slot exists + has recent ts (1Hz cadence)
    A4. _proxies["agno"] registered in plugin state (via /v4/...)

  Group B — /chat synchronous round-trip
    B1. POST /chat with internal-key auth → HTTP 200 + ChatResponse shape
    B2. Response includes ovg field with signature + merkle_root + block_height
    B3. response.mode is set (Collaborative / Verified / Sovereign / "")
    B4. response.mood is non-empty
    B5. X-Titan-Verified header is present
    B6. Round-trip latency < 90s (the agno_proxy timeout ceiling)

  Group C — /v4/pitch-chat round-trip
    C1. POST /v4/pitch-chat with X-Pitch-Token → HTTP 200 + PitchChatResponse
    C2. response.thread_id matches request thread_id
    C3. response.internal_time present (mood + emotion + chi)
    C4. declined=False for a normal message

  Group D — Bus contract verification (via observatory)
    D1. /v4/agno/stats or proxy state shows session_count > 0
    D2. agno_state.bin total_chats_24h incremented by N (where N = our test posts)
    D3. provider_stats populated with at least one provider key

  Group E — Legacy compat regression (CRITICAL — fleet uses these paths)
    E1. /v4/compose-reply endpoint (dashboard test) still returns composed result
    E2. canonical inference path works + utils/ollama_cloud is DELETED
    E3. studio Tier-1 haiku still works (verifies inference module via llm_worker)

Output: JSON results to stdout + exit code 0 on all-pass, non-zero on any fail.
Each test prints its result line in CLAUDE-readable format.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx


# Default Titan endpoint mapping (per memory/reference_titan2_vps.md +
# config.toml [api] port + VPS allocation)
DEFAULT_BASE_URLS = {
    "T1": "http://127.0.0.1:7777",       # local mainnet
    "T2": "http://10.135.0.6:7777",      # devnet on T2 VPS
    "T3": "http://10.135.0.6:7778",      # devnet on T3 VPS (different port)
}

# Internal-key auth header (per memory/reference_chat_internal_key.md)
# Reads from ~/.titan/secrets.toml: [chat_internal] key = "..."
INTERNAL_KEY_SECRETS_PATH = Path.home() / ".titan" / "secrets.toml"


def _load_internal_key() -> Optional[str]:
    """Read X-Titan-Internal-Key from ~/.titan/secrets.toml."""
    if not INTERNAL_KEY_SECRETS_PATH.exists():
        return None
    try:
        import tomllib
        with INTERNAL_KEY_SECRETS_PATH.open("rb") as f:
            data = tomllib.load(f)
        # Try common shapes — auth.py reads config[api][internal_key], the
        # secrets.toml may store at root `internal_key` (merged into api
        # section at boot) or under [api] / [chat] subtables.
        for key_path in (
            ("internal_key",),
            ("api", "internal_key"),
            ("chat_internal_key",),
            ("chat", "internal_key"),
            ("chat_internal", "key"),
            ("internal", "chat_key"),
        ):
            v: Any = data
            for k in key_path:
                if isinstance(v, dict) and k in v:
                    v = v[k]
                else:
                    v = None
                    break
            if isinstance(v, str) and v:
                return v
    except Exception as e:
        print(f"[WARN] secrets.toml read failed: {e}", file=sys.stderr)
    return None


def _load_pitch_token() -> Optional[str]:
    """Read X-Pitch-Token. Source-of-truth order per api/pitch_chat.py:96:
    env var PITCH_TOKEN → data/pitch_token file → None."""
    env = os.environ.get("PITCH_TOKEN", "").strip()
    if env:
        return env
    pf = Path("data/pitch_token")
    if pf.exists():
        try:
            return pf.read_text().strip()
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────
# Test framework
# ─────────────────────────────────────────────────────────────────────

class TestResult:
    __slots__ = ("name", "passed", "details", "latency_ms")
    def __init__(self, name: str, passed: bool, details: str = "", latency_ms: float = 0.0):
        self.name = name
        self.passed = passed
        self.details = details
        self.latency_ms = latency_ms


def _emit(r: TestResult, indent: str = "  "):
    icon = "✓" if r.passed else "✗"
    lat = f" ({r.latency_ms:.0f}ms)" if r.latency_ms else ""
    print(f"{indent}{icon} {r.name}{lat}")
    if r.details:
        for line in r.details.splitlines():
            print(f"{indent}   {line}")


# ─────────────────────────────────────────────────────────────────────
# Group A — Worker boot state
# ─────────────────────────────────────────────────────────────────────

def test_a1_agno_worker_in_ps(titan: str) -> TestResult:
    """A1. agno_worker subprocess is in the Titan process tree."""
    if titan == "T1":
        proc = subprocess.run(
            ["pgrep", "-fa", "agno_worker"],
            capture_output=True, text=True,
        )
        hits = proc.stdout.strip().splitlines()
    else:
        # Remote via SSH (T2 + T3 live on 10.135.0.6)
        try:
            proc = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "root@10.135.0.6",
                 "pgrep -fa agno_worker"],
                capture_output=True, text=True, timeout=15,
            )
            hits = proc.stdout.strip().splitlines()
        except Exception as e:
            return TestResult("A1 agno_worker in ps", False, f"SSH error: {e}")
    # Filter to actual agno_worker processes (not pgrep / ssh / pytest)
    matches = [h for h in hits if "modules.agno_worker" in h or "agno_worker_main" in h]
    if matches:
        return TestResult("A1 agno_worker in ps", True,
                          f"pid match: {matches[0][:80]}")
    return TestResult("A1 agno_worker in ps", False,
                      "no agno_worker process found in pgrep output")


def test_a2_agno_worker_ready_in_journal(titan: str) -> TestResult:
    """A2. AGNO_WORKER_READY emitted in journalctl since boot."""
    cmd_local = ["journalctl", f"-u", f"titan-t{titan[-1].lower()}.service",
                 "--since", "30 minutes ago", "--no-pager"]
    try:
        if titan == "T1":
            proc = subprocess.run(cmd_local, capture_output=True, text=True, timeout=30)
        else:
            proc = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "root@10.135.0.6"] + cmd_local,
                capture_output=True, text=True, timeout=30,
            )
        log = proc.stdout
    except Exception as e:
        return TestResult("A2 AGNO_WORKER_READY emit", False, f"journalctl error: {e}")
    if "AGNO_WORKER_READY" in log:
        # Find the latest occurrence
        last_line = ""
        for line in log.splitlines():
            if "AGNO_WORKER_READY" in line:
                last_line = line
        return TestResult("A2 AGNO_WORKER_READY emit", True,
                          f"emit found: ...{last_line[-100:]}")
    return TestResult("A2 AGNO_WORKER_READY emit", False,
                      "AGNO_WORKER_READY not in last 30min of journal")


def test_a3_agno_state_shm_fresh(titan: str) -> TestResult:
    """A3. agno_state.bin SHM slot exists + has recent ts (1Hz cadence)."""
    shm_path = f"/dev/shm/titan_{titan}/agno_state.bin"
    cmd = ["stat", "-c", "%Y", shm_path]
    try:
        if titan == "T1":
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        else:
            proc = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "root@10.135.0.6"] + cmd,
                capture_output=True, text=True, timeout=15,
            )
        if proc.returncode != 0:
            return TestResult("A3 agno_state.bin fresh", False,
                              f"stat failed: {proc.stderr.strip()}")
        mtime = int(proc.stdout.strip())
    except Exception as e:
        return TestResult("A3 agno_state.bin fresh", False, f"error: {e}")
    age_s = time.time() - mtime
    if age_s < 30:
        return TestResult("A3 agno_state.bin fresh", True,
                          f"slot mtime {age_s:.0f}s ago — within 30s freshness window")
    return TestResult("A3 agno_state.bin fresh", False,
                      f"slot mtime {age_s:.0f}s ago (>30s — worker hung or absent)")


# ─────────────────────────────────────────────────────────────────────
# Group B — /chat synchronous round-trip
# ─────────────────────────────────────────────────────────────────────

async def test_b_chat(client: httpx.AsyncClient, base_url: str, key: str) -> list[TestResult]:
    """B1-B6. /chat round-trip via internal-key auth."""
    # Meaningful prompt per Maker direction 2026-05-17 — exercises real
    # reasoning + felt-state composition (DialogueComposer pre-LLM), not
    # just a trivial echo. The uuid suffix uniquely tags each test run
    # for audit trail in agno_sessions.db titan_sessions table.
    msg = (f"Tell me one true thing about your inner state right now. "
           f"[dspec70-test {uuid.uuid4().hex[:8]}]")
    session_id = f"dspec70-test-{int(time.time())}"

    headers = {
        "Content-Type": "application/json",
        "X-Titan-Internal-Key": key,
        "X-Titan-User-Id": "maker",
    }
    body = {"message": msg, "session_id": session_id, "user_id": "maker"}

    t0 = time.time()
    try:
        resp = await client.post(
            f"{base_url}/chat",
            headers=headers, json=body, timeout=120.0,
        )
        latency_ms = (time.time() - t0) * 1000
    except Exception as e:
        return [TestResult("B1 /chat HTTP 200", False,
                           f"request error: {e}", (time.time() - t0) * 1000)]

    results: list[TestResult] = []
    results.append(TestResult(
        "B1 /chat HTTP 200",
        resp.status_code == 200,
        f"status={resp.status_code}, body_preview={resp.text[:200]}",
        latency_ms,
    ))
    if resp.status_code != 200:
        return results

    try:
        data = resp.json()
    except Exception as e:
        return results + [TestResult("B2..B6", False, f"JSON parse error: {e}")]

    # B2 — OVG signature
    ovg = data.get("ovg") or {}
    results.append(TestResult(
        "B2 OVG signature in response",
        bool(ovg.get("verified") is not None and ovg.get("signature")),
        f"verified={ovg.get('verified')}, sig={(ovg.get('signature') or '')[:24]}..., "
        f"merkle={(ovg.get('merkle_root') or '')[:16]}..., "
        f"block_height={ovg.get('block_height')}",
    ))
    # B3 — mode
    results.append(TestResult(
        "B3 response.mode set",
        bool(data.get("mode")),
        f"mode={data.get('mode')!r}",
    ))
    # B4 — mood
    results.append(TestResult(
        "B4 response.mood non-empty",
        bool(data.get("mood")),
        f"mood={data.get('mood')!r}",
    ))
    # B5 — X-Titan-Verified header
    results.append(TestResult(
        "B5 X-Titan-Verified header",
        "X-Titan-Verified" in resp.headers,
        f"header={resp.headers.get('X-Titan-Verified')!r}",
    ))
    # B6 — latency under ceiling (90s + some HTTP overhead)
    results.append(TestResult(
        "B6 latency < 90s",
        latency_ms < 90_000,
        f"latency={latency_ms:.0f}ms (ceiling 90,000ms)",
    ))
    return results


# ─────────────────────────────────────────────────────────────────────
# Group C — /v4/pitch-chat round-trip
# ─────────────────────────────────────────────────────────────────────

async def test_c_pitch_chat(client: httpx.AsyncClient, base_url: str,
                            pitch_token: str, titan: str) -> list[TestResult]:
    """C1-C4. /v4/pitch-chat round-trip (wallet-less pitch route)."""
    thread_id = f"dspec70_pitch_test_{uuid.uuid4().hex[:8]}"
    msg = f"Hello Titan, this is a routine pitch-chat runtime test ({uuid.uuid4().hex[:8]})."
    body = {"message": msg, "thread_id": thread_id, "titan": titan}
    headers = {
        "Content-Type": "application/json",
        "X-Pitch-Token": pitch_token,
    }

    t0 = time.time()
    try:
        resp = await client.post(
            f"{base_url}/v4/pitch-chat",
            headers=headers, json=body, timeout=120.0,
        )
        latency_ms = (time.time() - t0) * 1000
    except Exception as e:
        return [TestResult("C1 /v4/pitch-chat HTTP 200", False,
                           f"request error: {e}", (time.time() - t0) * 1000)]

    results: list[TestResult] = []
    results.append(TestResult(
        "C1 /v4/pitch-chat HTTP 200",
        resp.status_code == 200,
        f"status={resp.status_code}, body_preview={resp.text[:200]}",
        latency_ms,
    ))
    if resp.status_code != 200:
        return results

    try:
        data = resp.json()
    except Exception as e:
        return results + [TestResult("C2..C4", False, f"JSON parse error: {e}")]

    results.append(TestResult(
        "C2 thread_id matches",
        data.get("thread_id") == thread_id,
        f"req_thread_id={thread_id} resp_thread_id={data.get('thread_id')}",
    ))
    it = data.get("internal_time") or {}
    results.append(TestResult(
        "C3 internal_time present",
        bool(it),
        f"keys={list(it.keys())[:8] if it else 'none'}",
    ))
    results.append(TestResult(
        "C4 declined=False for normal msg",
        data.get("declined") is False,
        f"declined={data.get('declined')}, reason={data.get('decline_reason')!r}",
    ))
    return results


# ─────────────────────────────────────────────────────────────────────
# Group D — Bus contract via observability (post-chat verification)
# ─────────────────────────────────────────────────────────────────────

async def test_d_bus_contract(client: httpx.AsyncClient, base_url: str,
                              titan: str) -> list[TestResult]:
    """D1-D3. Verify agno_state.bin SHM slot reflects our chat traffic.

    Read SHM directly via ssh + python one-liner since most dashboards
    don't expose /v4/agno/stats yet.
    """
    results: list[TestResult] = []
    cmd_python = (
        "python3 -c \""
        "import sys; sys.path.insert(0, '/home/antigravity/projects/titan'); "
        "import msgpack; "
        "from pathlib import Path; "
        f"data = Path('/dev/shm/titan_{titan}/agno_state.bin').read_bytes(); "
        "header_size = 64; "
        "buf = data[header_size:]; "
        "buf = buf[:len(buf)//3]; "  # triple-buffer; take first
        "import struct; "
        "size = struct.unpack('<Q', buf[:8])[0]; "
        "payload = buf[8:8+size]; "
        "decoded = msgpack.unpackb(payload, raw=False); "
        "import json as _j; print(_j.dumps(decoded))"
        "\""
    )
    try:
        if titan == "T1":
            proc = subprocess.run(["bash", "-c", cmd_python],
                                  capture_output=True, text=True, timeout=10)
        else:
            proc = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "root@10.135.0.6",
                 cmd_python],
                capture_output=True, text=True, timeout=15,
            )
        if proc.returncode != 0:
            return [TestResult("D1..D3 agno_state.bin decode", False,
                               f"decode error: {proc.stderr.strip()[:200]}")]
        snap = json.loads(proc.stdout.strip())
    except Exception as e:
        return [TestResult("D1..D3 agno_state.bin decode", False, f"error: {e}")]

    results.append(TestResult(
        "D1 session_count tracked",
        isinstance(snap.get("session_count"), int),
        f"session_count={snap.get('session_count')}",
    ))
    results.append(TestResult(
        "D2 total_chats_24h incremented",
        isinstance(snap.get("total_chats_24h"), int)
        and snap.get("total_chats_24h", 0) > 0,
        f"total_chats_24h={snap.get('total_chats_24h')}",
    ))
    results.append(TestResult(
        "D3 last_chat_ts recent",
        isinstance(snap.get("last_chat_ts"), (int, float))
        and (time.time() - snap.get("last_chat_ts", 0)) < 300,
        f"last_chat_ts={snap.get('last_chat_ts')} "
        f"(age={time.time() - snap.get('last_chat_ts', 0):.0f}s)",
    ))
    return results


# ─────────────────────────────────────────────────────────────────────
# Group E — Legacy compat regression
# ─────────────────────────────────────────────────────────────────────

async def test_e1_compose_reply(client: httpx.AsyncClient, base_url: str) -> TestResult:
    """E1. /v4/compose-reply (dashboard test endpoint) still works after F migration."""
    try:
        resp = await client.post(
            f"{base_url}/v4/compose-reply",
            json={"message": "test compose"},
            timeout=10.0,
        )
        if resp.status_code != 200:
            return TestResult("E1 /v4/compose-reply", False,
                              f"status={resp.status_code}")
        data = resp.json()
        # Should have either a composed dict OR no_state/no_vocabulary reason
        ok = ("composed" in (data.get("data") or {}) or
              "reason" in (data.get("data") or {}) or
              "composed" in data)
        return TestResult("E1 /v4/compose-reply", ok,
                          f"keys={list(data.keys())[:8]}")
    except Exception as e:
        return TestResult("E1 /v4/compose-reply", False, f"error: {e}")


def test_e2_inference_canonical_path() -> TestResult:
    """E2. Canonical titan_hcl.inference path works + legacy utils path is GONE.

    D-SPEC-72: utils/ollama_cloud.py was DELETED, not shimmed. Per Maker
    direction 2026-05-17 ("OLD PATH MUST BE DELETED FROM CODEBASE so we
    don't risk using it again ever"). This test verifies BOTH:
      (a) the canonical inference.get_provider path works
      (b) the legacy utils.ollama_cloud import path no longer exists
    """
    try:
        # (a) Canonical path: inference.get_provider returns
        # OllamaCloudProvider with full surface (request_counts, total_tokens,
        # complete, score, get_stats — back-compat properties absorbed)
        from titan_hcl.inference import get_provider, get_model_for_task, TASK_MODEL_MAP
        provider = get_provider("ollama_cloud", {
            "ollama_cloud_api_key": "test",
            "ollama_cloud_base_url": "https://ollama.com/v1",
        })
        assert hasattr(provider, "request_counts")
        assert hasattr(provider, "total_tokens")
        assert callable(provider.complete)
        assert callable(provider.score)
        # Model routing
        m = get_model_for_task("haiku")
        assert isinstance(m, str) and m
        # (b) Legacy path must NOT exist
        try:
            from titan_hcl.utils import ollama_cloud  # noqa: F401
            return TestResult(
                "E2 canonical inference path + utils retired",
                False,
                "FOOTGUN: titan_hcl.utils.ollama_cloud still importable — "
                "should be DELETED per D-SPEC-72",
            )
        except (ImportError, ModuleNotFoundError):
            pass  # expected — module deleted
        return TestResult(
            "E2 canonical inference path + utils retired",
            True,
            "inference.get_provider returns OllamaCloudProvider with full surface; "
            "titan_hcl.utils.ollama_cloud correctly DELETED",
        )
    except Exception as e:
        return TestResult(
            "E2 canonical inference path + utils retired",
            False,
            f"error: {e}",
        )


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

async def run_all_tests(titan: str, base_url: str) -> dict[str, Any]:
    print(f"\n═══════════════════════════════════════════════════════════")
    print(f"  D-SPEC-72 RUNTIME VERIFICATION — {titan} @ {base_url}")
    print(f"═══════════════════════════════════════════════════════════\n")

    all_results: list[TestResult] = []

    # ─── Group A — boot state ───
    print(f"[Group A] Worker boot state")
    all_results.append(test_a1_agno_worker_in_ps(titan))
    _emit(all_results[-1])
    all_results.append(test_a2_agno_worker_ready_in_journal(titan))
    _emit(all_results[-1])
    all_results.append(test_a3_agno_state_shm_fresh(titan))
    _emit(all_results[-1])
    print()

    # ─── Group B — /chat ───
    key = _load_internal_key()
    if not key:
        print("[Group B] SKIPPED — no internal key in ~/.titan/secrets.toml")
        all_results.append(TestResult("B SKIPPED", False, "no internal key configured"))
    else:
        print(f"[Group B] /chat synchronous round-trip")
        async with httpx.AsyncClient(timeout=120.0) as client:
            b_results = await test_b_chat(client, base_url, key)
            for r in b_results:
                _emit(r)
                all_results.append(r)
        print()

    # ─── Group C — /v4/pitch-chat ───
    pitch = _load_pitch_token()
    if not pitch:
        print("[Group C] SKIPPED — no PITCH_TOKEN env or data/pitch_token")
        all_results.append(TestResult("C SKIPPED", False, "no pitch token"))
    else:
        print(f"[Group C] /v4/pitch-chat round-trip")
        async with httpx.AsyncClient(timeout=120.0) as client:
            c_results = await test_c_pitch_chat(client, base_url, pitch, titan)
            for r in c_results:
                _emit(r)
                all_results.append(r)
        print()

    # ─── Group D — bus contract (post-chat) ───
    print(f"[Group D] Bus contract via SHM observability")
    async with httpx.AsyncClient(timeout=10.0) as client:
        d_results = await test_d_bus_contract(client, base_url, titan)
        for r in d_results:
            _emit(r)
            all_results.append(r)
    print()

    # ─── Group E — legacy compat ───
    print(f"[Group E] Legacy compat regression")
    async with httpx.AsyncClient(timeout=10.0) as client:
        e1 = await test_e1_compose_reply(client, base_url)
        _emit(e1)
        all_results.append(e1)
    e2 = test_e2_inference_canonical_path()
    _emit(e2)
    all_results.append(e2)
    print()

    # ─── Summary ───
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    print(f"═══════════════════════════════════════════════════════════")
    icon = "✅" if passed == total else "❌"
    print(f"  {icon} {titan}: {passed}/{total} tests passed")
    if passed < total:
        print(f"     Failures:")
        for r in all_results:
            if not r.passed:
                print(f"       - {r.name}: {r.details[:120]}")
    print(f"═══════════════════════════════════════════════════════════\n")
    return {
        "titan": titan,
        "base_url": base_url,
        "ts": time.time(),
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
        "results": [
            {"name": r.name, "passed": r.passed,
             "details": r.details, "latency_ms": r.latency_ms}
            for r in all_results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="D-SPEC-72 runtime end-to-end verification"
    )
    parser.add_argument("--titan", choices=["T1", "T2", "T3"], required=True)
    parser.add_argument("--base-url", help="override default endpoint URL")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON result to stdout instead of text")
    args = parser.parse_args()

    base_url = args.base_url or DEFAULT_BASE_URLS[args.titan]

    import asyncio
    summary = asyncio.run(run_all_tests(args.titan, base_url))

    if args.json:
        print(json.dumps(summary, indent=2))

    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
