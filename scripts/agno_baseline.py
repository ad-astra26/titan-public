"""Phase 9 Chunk 9A — agno_worker pre-Phase-9 baseline measurement.

RFP source: titan-docs/RFP_phase_c_enhancements.md §3F.2 (Chunk 9A).
Discipline: feedback_eager_init_needs_rss_root_cause_first — measure FIRST
so 9B-9H improvements are judged against a real source-of-truth baseline,
not against a guess.

Captures at 5 checkpoints {boot_complete, chat_1_in, chat_1_out,
chat_5_out, chat_10_out} the following signals for the live agno_worker
subprocess on a target Titan:

  - External (zero touch):
      * /proc/<pid>/smaps_rollup   (Rss / Pss / Anonymous / File / Swap)
      * /proc/<pid>/maps heavy-lib list (.so files ≥ 1 MB, with a sum)

  - In-process (via SIGUSR1 → JSON dump to /tmp, installed by
    `_install_phase9_baseline_hook` in titan_hcl/modules/agno_worker.py):
      * sys.modules (full sorted list + count)
      * tracemalloc top-20 by file:line
      * tracemalloc current + peak traced bytes
      * worker-reported VmRSS

Drives 10 synthetic chats spanning the live ChatTierClassifier tiers
(titan_hcl/config.toml:139-187): 3 greeting + 2 personal + 2 casual +
3 reasoning. (RFP §3F.2's "3 simple + 4 standard + 3 research" was stale
terminology — Maker greenlit the mapping to actual tiers 2026-05-27.)

NO BEHAVIOR CHANGE — pure measurement.

Usage (T3 devnet, the cascade-first target):

    source test_env/bin/activate
    python scripts/agno_baseline.py \\
        --titan-id T3 \\
        --host 10.135.0.6:7778 \\
        --output data/audits/agno_baseline_pre_phase9.json

For a clean `boot_complete` checkpoint, restart the worker first
(operator-greenlit per `feedback_titan_restart_requires_greenlight`):

    bash scripts/t3_manage.sh restart --force
    # wait until /health == 200
    python scripts/agno_baseline.py --titan-id T3 --host 10.135.0.6:7778 ...

If run against a warm worker (no restart), `boot_complete` is captured as
"current_state" — a pessimistic baseline (already-warm), and the metadata
field `worker_was_freshly_booted` is set false. Maker-side baseline review
should weight accordingly.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


HEAVY_LIB_MIN_BYTES = 1 * 1024 * 1024   # 1 MB
DUMP_POLL_TIMEOUT_S = 30.0              # daemon-thread poll is 0.5s; 30s safety
DUMP_POLL_INTERVAL_S = 0.5
CHAT_TIMEOUT_S = 30.0                   # short enough to finish run inside Guardian's
                                        # 5-restarts-in-600s window when agno hangs
INTER_CHAT_GAP_S = 0.5                  # let the worker quiesce between requests

# Chat internal-key auth header names (titan_hcl/api/auth.py — bypass Privy)
AUTH_HEADER_KEY = "X-Titan-Internal-Key"
AUTH_HEADER_USER = "X-Titan-User-Id"
AUTH_HEADER_CHANNEL = "X-Titan-Channel"
DEFAULT_USER_ID = "maker"
DEFAULT_CHANNEL = "cli-test"


def load_internal_key() -> str:
    """Load X-Titan-Internal-Key from env var or local secrets.toml.

    Never hardcoded — `feedback_no_secrets_in_example_configs.md` +
    public-sync pipeline. Precedence: TITAN_INTERNAL_KEY env, then
    ~/.titan/secrets.toml [api].internal_key.
    """
    env = os.environ.get("TITAN_INTERNAL_KEY")
    if env:
        return env.strip()
    secrets_path = os.path.expanduser("~/.titan/secrets.toml")
    if os.path.isfile(secrets_path):
        try:
            import tomllib  # Python 3.11+
        except ImportError:  # pragma: no cover
            import tomli as tomllib  # type: ignore
        with open(secrets_path, "rb") as fh:
            data = tomllib.load(fh)
        key = (data.get("api") or {}).get("internal_key")
        if key:
            return str(key).strip()
    raise RuntimeError(
        "No X-Titan-Internal-Key available. Set TITAN_INTERNAL_KEY env var, "
        "or ensure ~/.titan/secrets.toml has [api].internal_key. The shared "
        "fleet key lives there per reference_external_secrets.md.",
    )


# ── Prompt set — 10 synthetic chats spanning the 4 live tiers ──
# Each prompt is annotated with the expected tier per the classifier in
# titan_hcl/modules/chat_tier_config.py + tier regex blocks in
# titan_hcl/config.toml:139-187. Mapping is verified against the
# `ChatTierClassifier.classify()` first-match-wins algorithm.
PROMPTS: list[dict[str, str]] = [
    # 3 × greeting — model_class=fast, max_chars=50, regex `^\s*(hi|hello|hey|...)`
    {"tier": "greeting", "text": "hi"},
    {"tier": "greeting", "text": "hello there"},
    {"tier": "greeting", "text": "hey, how are you"},
    # 2 × personal — regex `\b(do you know|remember|have we met|...)`
    {"tier": "personal", "text": "do you remember me?"},
    {"tier": "personal", "text": "have we met before — my name is Alice"},
    # 2 × casual — fallback tier, model_class=heavy, max_chars=200
    {"tier": "casual", "text": "tell me something about Solana"},
    {"tier": "casual", "text": "what did you do today"},
    # 3 × reasoning — regex `\b(why|explain|analyze|reason|...)` or `\?.*\?`
    {"tier": "reasoning", "text": "why does the sphere clock balance threshold matter to consciousness?"},
    {"tier": "reasoning", "text": "explain the trade-offs between mainnet and devnet for an agent"},
    {"tier": "reasoning", "text": "analyze the relationship between metabolic energy and circadian rhythm"},
]


_JOURNAL_PID_RE = __import__("re").compile(
    r"Module 'agno_worker' is READY \(pid=(\d+), restarts=(\d+)\)"
)


def find_agno_worker_pid(titan_id: str,
                         ssh_host: str | None = None,
                         retries: int = 4,
                         retry_sleep_s: float = 1.0) -> tuple[int, int]:
    """Discover the live agno_worker PID via systemd journalctl.

    Workers run under `comm=guardian_hcl` (no setproctitle), so `pgrep -fa
    agno_worker` finds nothing. The canonical source is the Guardian log
    line at L1: `[Guardian] Module 'agno_worker' is READY (pid=NNNN,
    restarts=N)`. We take the LATEST READY line and validate the PID is
    still alive. Retries on race (worker restarted between journal scrape
    and validation) — common on T3 today per RFP §3F.0 instability.

    Returns (pid, restarts).
    """
    last_err: str | None = None
    for attempt in range(retries):
        try:
            return _find_agno_worker_pid_once(titan_id, ssh_host)
        except RuntimeError as exc:
            last_err = str(exc)
            if attempt < retries - 1:
                time.sleep(retry_sleep_s)
    raise RuntimeError(
        f"PID lookup failed after {retries} attempts. Last error: {last_err}. "
        f"Worker may be in MAX_RESTARTS escalation — check `journalctl -u "
        f"titan-{titan_id.lower()}.service --since '5 min ago' | grep "
        f"'max_restarts'`.",
    )


def _find_agno_worker_pid_once(titan_id: str,
                               ssh_host: str | None) -> tuple[int, int]:
    svc = f"titan-{titan_id.lower()}.service"
    cmd = (
        f"journalctl -u {svc} --no-pager 2>/dev/null | "
        f"grep -E \"Module 'agno_worker' is READY\" | tail -1"
    )
    if ssh_host:
        out = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             ssh_host, cmd],
            capture_output=True, text=True, check=False,
        )
    else:
        # T1: per `reference_t1_journal_query_no_sudo` — journalctl runs
        # WITHOUT sudo (sudo silently fails for the antigravity user).
        out = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True, check=False,
        )
    line = out.stdout.strip()
    if not line:
        raise RuntimeError(
            f"No 'Module agno_worker is READY' line found in {svc} journal. "
            f"Is the Titan running and the worker booted? Verify with "
            f"`bash scripts/t{titan_id[1].lower()}_manage.sh status`.",
        )
    m = _JOURNAL_PID_RE.search(line)
    if not m:
        raise RuntimeError(
            f"Could not parse PID from journal line: {line!r}",
        )
    pid, restarts = int(m.group(1)), int(m.group(2))

    # Validate the PID is still alive
    alive_cmd = f"kill -0 {pid} 2>&1; echo $?"
    if ssh_host:
        alive_out = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             ssh_host, alive_cmd],
            capture_output=True, text=True, check=False,
        )
    else:
        alive_out = subprocess.run(
            ["bash", "-c", alive_cmd],
            capture_output=True, text=True, check=False,
        )
    last_line = alive_out.stdout.strip().splitlines()[-1] if alive_out.stdout.strip() else "1"
    if last_line != "0":
        raise RuntimeError(
            f"Latest journal PID {pid} for agno_worker is not alive — "
            f"worker likely restarted between journal scrape and validation. "
            f"Re-run the script.",
        )
    return pid, restarts


def read_smaps_rollup(pid: int, ssh_host: str | None = None) -> dict[str, int]:
    """Parse /proc/<pid>/smaps_rollup — kB values into a dict."""
    if ssh_host:
        out = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             ssh_host, f"cat /proc/{pid}/smaps_rollup"],
            capture_output=True, text=True, check=False,
        )
        text = out.stdout
    else:
        with open(f"/proc/{pid}/smaps_rollup") as fh:
            text = fh.read()
    res: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip()
        if val.endswith("kB"):
            try:
                res[key.strip() + "_kb"] = int(val.split()[0])
            except (ValueError, IndexError):
                continue
    return res


def read_heavy_libs(pid: int, ssh_host: str | None = None,
                    min_bytes: int = HEAVY_LIB_MIN_BYTES) -> dict[str, Any]:
    """Sum unique .so mappings ≥ min_bytes from /proc/<pid>/maps.

    Returns {"libs": [{path, bytes}], "total_bytes": int, "count": int}
    sorted by bytes descending.
    """
    if ssh_host:
        out = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             ssh_host, f"cat /proc/{pid}/maps"],
            capture_output=True, text=True, check=False,
        )
        text = out.stdout
    else:
        with open(f"/proc/{pid}/maps") as fh:
            text = fh.read()
    per_path: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        addr_range = parts[0]
        path = parts[-1]
        if not (path.endswith(".so") or ".so." in path):
            continue
        try:
            lo_s, hi_s = addr_range.split("-")
            lo, hi = int(lo_s, 16), int(hi_s, 16)
        except ValueError:
            continue
        per_path[path] = per_path.get(path, 0) + (hi - lo)
    big = [(p, b) for p, b in per_path.items() if b >= min_bytes]
    big.sort(key=lambda x: x[1], reverse=True)
    return {
        "min_bytes_filter": min_bytes,
        "count": len(big),
        "total_bytes": sum(b for _, b in big),
        "libs": [{"path": p, "bytes": b} for p, b in big],
    }


def trigger_in_process_dump(pid: int, ssh_host: str | None = None) -> dict[str, Any]:
    """File-flag the worker → poll /tmp/ for the new JSON dump → read + delete.

    Worker side: a daemon thread (installed by `_install_phase9_baseline_hook`
    in agno_worker.py) polls /tmp/agno_baseline_request_<pid> every 0.5s and
    writes /tmp/agno_baseline_<pid>_<unix_ts_ms>.json on each request.
    Decoupled from SIGUSR1 because Python defers signals indefinitely when
    the main thread is blocked in a C extension — confirmed live on T3
    2026-05-27 where SIGUSR1 dumps arrived 3+ minutes late.

    Returns the parsed dump. Raises RuntimeError on poll timeout.
    """
    request_path = f"/tmp/agno_baseline_request_{pid}"
    cutoff_ms = (int(_remote_now_ms(ssh_host)) if ssh_host
                 else int(time.time() * 1000))

    # 1. Drop the request flag file (touch)
    if ssh_host:
        subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             ssh_host, f"touch {request_path}"],
            check=True, capture_output=True, text=True,
        )
    else:
        with open(request_path, "w") as fh:
            fh.write("")

    # 2. Poll for the dump file (newest, mtime ≥ cutoff)
    deadline = time.time() + DUMP_POLL_TIMEOUT_S
    pattern_glob = f"/tmp/agno_baseline_{pid}_*.json"
    while time.time() < deadline:
        if ssh_host:
            out = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
                 ssh_host,
                 f"ls -1 /tmp/agno_baseline_{pid}_*.json 2>/dev/null || true"],
                capture_output=True, text=True, check=False,
            )
            candidates = [
                ln.strip() for ln in out.stdout.splitlines() if ln.strip()
            ]
        else:
            candidates = sorted(glob.glob(pattern_glob))
        for path in reversed(candidates):
            try:
                ts_part = path.rsplit("_", 1)[-1].split(".", 1)[0]
                ts_ms = int(ts_part)
            except ValueError:
                continue
            if ts_ms < cutoff_ms:
                continue
            if ssh_host:
                out = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
                     ssh_host, f"cat {path} && rm -f {path}"],
                    capture_output=True, text=True, check=False,
                )
                if out.returncode != 0 or not out.stdout.strip():
                    continue
                return json.loads(out.stdout)
            else:
                try:
                    with open(path) as fh:
                        data = json.load(fh)
                    os.unlink(path)
                    return data
                except (OSError, json.JSONDecodeError):
                    continue
        time.sleep(DUMP_POLL_INTERVAL_S)
    raise RuntimeError(
        f"In-process dump not received within {DUMP_POLL_TIMEOUT_S}s for "
        f"pid={pid}. Possible causes: (a) worker missing the Phase 9 "
        "Chunk 9A hook (verify agno_worker.py:_install_phase9_baseline_hook "
        "in the deployed file and the worker has been respawned); "
        f"(b) watcher thread crashed (check /tmp/agno_baseline_error_{pid}_*.log); "
        f"(c) worker died (PID gone); (d) /tmp full.",
    )


def _remote_now_ms(ssh_host: str) -> int:
    out = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
         ssh_host, "date +%s%3N"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def capture_checkpoint(name: str, pid: int, ssh_host: str | None,
                       capture_in_process: bool = True) -> dict[str, Any]:
    """Capture all signals at a single checkpoint."""
    started = time.time()
    result: dict[str, Any] = {
        "name": name,
        "captured_at_unix": started,
        "captured_at_utc": _dt.datetime.fromtimestamp(started, _dt.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        result["smaps_rollup"] = read_smaps_rollup(pid, ssh_host=ssh_host)
    except Exception as exc:
        result["smaps_rollup_error"] = str(exc)
    try:
        result["heavy_libs"] = read_heavy_libs(pid, ssh_host=ssh_host)
    except Exception as exc:
        result["heavy_libs_error"] = str(exc)
    if capture_in_process:
        try:
            result["in_process"] = trigger_in_process_dump(pid, ssh_host=ssh_host)
        except Exception as exc:
            result["in_process_error"] = str(exc)
    result["capture_duration_s"] = time.time() - started
    return result


def send_chat(host: str, idx: int, prompt: dict[str, str],
              session_id: str, internal_key: str,
              user_id: str, channel: str) -> dict[str, Any]:
    """POST /chat and capture timing + response shape (no semantic checks)."""
    body = json.dumps({
        "message": prompt["text"],
        "session_id": session_id,
        "user_id": user_id,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}/chat",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            AUTH_HEADER_KEY: internal_key,
            AUTH_HEADER_USER: user_id,
            AUTH_HEADER_CHANNEL: channel,
        },
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=CHAT_TIMEOUT_S) as resp:
            elapsed_ms = (time.time() - started) * 1000.0
            payload_raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        return {
            "idx": idx,
            "tier_expected": prompt["tier"],
            "text_first_60": prompt["text"][:60],
            "status": exc.code,
            "latency_ms": (time.time() - started) * 1000.0,
            "error": exc.read().decode("utf-8", errors="replace")[:500],
        }
    except urllib.error.URLError as exc:
        return {
            "idx": idx,
            "tier_expected": prompt["tier"],
            "text_first_60": prompt["text"][:60],
            "status": None,
            "latency_ms": (time.time() - started) * 1000.0,
            "error": f"urlerror: {exc.reason}",
        }
    except (TimeoutError, __import__("socket").timeout) as exc:
        return {
            "idx": idx,
            "tier_expected": prompt["tier"],
            "text_first_60": prompt["text"][:60],
            "status": None,
            "latency_ms": (time.time() - started) * 1000.0,
            "error": f"client_timeout_after_{CHAT_TIMEOUT_S:.0f}s: {exc!r}",
        }
    try:
        payload = json.loads(payload_raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = {"_raw_bytes": len(payload_raw)}
    return {
        "idx": idx,
        "tier_expected": prompt["tier"],
        "text_first_60": prompt["text"][:60],
        "status": status,
        "latency_ms": elapsed_ms,
        "response_len": len(payload.get("response", ""))
            if isinstance(payload, dict) else 0,
        "tier_resolved": payload.get("mode") if isinstance(payload, dict) else None,
        "mood": payload.get("mood") if isinstance(payload, dict) else None,
        "session_id_echoed": payload.get("session_id")
            if isinstance(payload, dict) else None,
    }


def compute_delta(prev: dict[str, Any], curr: dict[str, Any]) -> dict[str, Any]:
    """Diff two checkpoints — pure summary (full lists stay in raw blocks)."""
    out: dict[str, Any] = {}
    p_smaps = prev.get("smaps_rollup") or {}
    c_smaps = curr.get("smaps_rollup") or {}
    for k in ("Rss_kb", "Pss_kb", "Anonymous_kb", "Private_Clean_kb",
              "Private_Dirty_kb"):
        if k in p_smaps and k in c_smaps:
            out[f"d_{k}"] = c_smaps[k] - p_smaps[k]
    p_libs = (prev.get("heavy_libs") or {}).get("total_bytes")
    c_libs = (curr.get("heavy_libs") or {}).get("total_bytes")
    if p_libs is not None and c_libs is not None:
        out["d_heavy_libs_total_bytes"] = c_libs - p_libs
    p_ip = prev.get("in_process") or {}
    c_ip = curr.get("in_process") or {}
    if "sys_modules_count" in p_ip and "sys_modules_count" in c_ip:
        out["d_sys_modules_count"] = c_ip["sys_modules_count"] - p_ip["sys_modules_count"]
        p_mods = set(p_ip.get("sys_modules") or [])
        c_mods = set(c_ip.get("sys_modules") or [])
        out["new_modules"] = sorted(c_mods - p_mods)[:50]
        out["new_modules_count"] = len(c_mods - p_mods)
    if "tracemalloc_current_bytes" in p_ip and "tracemalloc_current_bytes" in c_ip:
        out["d_tracemalloc_current_bytes"] = (
            c_ip["tracemalloc_current_bytes"] - p_ip["tracemalloc_current_bytes"]
        )
        out["d_tracemalloc_peak_bytes"] = (
            c_ip["tracemalloc_peak_bytes"] - p_ip["tracemalloc_peak_bytes"]
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--titan-id", required=True, choices=["T1", "T2", "T3"],
                    help="Label only — used in output filename + metadata")
    ap.add_argument("--host", required=True,
                    help="HTTP host:port for chat (e.g. localhost:7777 or "
                         "10.135.0.6:7778)")
    ap.add_argument("--ssh-host", default=None,
                    help="If agno_worker runs on a remote host, ssh user@host "
                         "to use for pgrep / /proc reads / SIGUSR1 / /tmp file "
                         "reads. Default: local (T1).")
    ap.add_argument("--output", default="data/audits/agno_baseline_pre_phase9.json",
                    help="Destination JSON path (overwritten — Maker-reviewed "
                         "single source of truth for Phase 9 baseline)")
    ap.add_argument("--user-id", default=DEFAULT_USER_ID,
                    help=f"X-Titan-User-Id header value (default {DEFAULT_USER_ID})")
    ap.add_argument("--channel", default=DEFAULT_CHANNEL,
                    help=f"X-Titan-Channel header value (default {DEFAULT_CHANNEL})")
    ap.add_argument("--worker-freshly-booted", action="store_true",
                    help="Set when the operator restarted the worker immediately "
                         "before this run, making `boot_complete` a true cold-boot "
                         "baseline. If absent, the metadata field flags the "
                         "baseline as warm-state.")
    args = ap.parse_args()

    if args.titan_id in ("T2", "T3") and args.ssh_host is None:
        # Convention: T2/T3 live on 10.135.0.6 per CLAUDE.md + VPC reference
        args.ssh_host = "root@10.135.0.6"
        print(f"[info] --titan-id {args.titan_id} → defaulting "
              f"--ssh-host {args.ssh_host}", file=sys.stderr)

    # 0. Load auth key BEFORE touching the worker (fail fast)
    internal_key = load_internal_key()

    # 1. Discover live agno_worker PID via Guardian's journal entry
    pid, restarts_at_start = find_agno_worker_pid(
        titan_id=args.titan_id, ssh_host=args.ssh_host,
    )
    print(
        f"[info] agno_worker pid={pid} restarts={restarts_at_start}"
        f"{' (remote ' + args.ssh_host + ')' if args.ssh_host else ''}",
        file=sys.stderr,
    )

    # 2. Boot-complete checkpoint (capture immediately)
    session_id = f"phase9_baseline_{int(time.time())}"
    cps: dict[str, dict[str, Any]] = {}
    print("[info] capturing C1 boot_complete", file=sys.stderr)
    cps["boot_complete"] = capture_checkpoint(
        "boot_complete", pid, args.ssh_host,
    )

    # 3. chat_1_in — snapshot immediately before sending chat 1
    print("[info] capturing C2 chat_1_in", file=sys.stderr)
    cps["chat_1_in"] = capture_checkpoint(
        "chat_1_in", pid, args.ssh_host,
    )

    # 4. Drive 10 chats sequentially. Capture at chat_1_out, chat_5_out,
    #    chat_10_out.
    chat_results: list[dict[str, Any]] = []
    for i, prompt in enumerate(PROMPTS, start=1):
        print(f"[info] chat {i}/10 (tier={prompt['tier']}): "
              f"{prompt['text'][:50]!r}", file=sys.stderr)
        result = send_chat(args.host, i, prompt, session_id,
                           internal_key, args.user_id, args.channel)
        chat_results.append(result)
        print(f"[info]   → status={result.get('status')} "
              f"lat={result.get('latency_ms', 0):.0f}ms "
              f"mode={result.get('tier_resolved')}",
              file=sys.stderr)
        if i == 1:
            cps["chat_1_out"] = capture_checkpoint(
                "chat_1_out", pid, args.ssh_host,
            )
        elif i == 5:
            cps["chat_5_out"] = capture_checkpoint(
                "chat_5_out", pid, args.ssh_host,
            )
        elif i == 10:
            cps["chat_10_out"] = capture_checkpoint(
                "chat_10_out", pid, args.ssh_host,
            )
        time.sleep(INTER_CHAT_GAP_S)

    # 5. Deltas vs previous checkpoint
    deltas: dict[str, dict[str, Any]] = {}
    order = ["boot_complete", "chat_1_in", "chat_1_out",
             "chat_5_out", "chat_10_out"]
    for i in range(1, len(order)):
        prev_name, curr_name = order[i - 1], order[i]
        deltas[f"{curr_name}_vs_{prev_name}"] = compute_delta(
            cps[prev_name], cps[curr_name],
        )
    deltas["chat_10_out_vs_boot_complete"] = compute_delta(
        cps["boot_complete"], cps["chat_10_out"],
    )

    # 6. Sanity check — has the worker restarted during the run?
    try:
        pid_end, restarts_end = find_agno_worker_pid(
            titan_id=args.titan_id, ssh_host=args.ssh_host,
        )
    except Exception as exc:
        pid_end, restarts_end = -1, -1
        print(f"[warn] post-run PID check failed: {exc}", file=sys.stderr)
    worker_restarted_during_run = (
        pid_end != pid or restarts_end != restarts_at_start
    )
    if worker_restarted_during_run:
        print(
            f"[warn] agno_worker RESTARTED during baseline capture "
            f"(start_pid={pid} restarts={restarts_at_start} → "
            f"end_pid={pid_end} restarts={restarts_end}). "
            f"Checkpoints after the restart point at a fresh process — "
            f"deltas across the restart boundary are NOT meaningful. "
            f"Maker review must weight this.",
            file=sys.stderr,
        )

    # 7. Assemble + write
    payload = {
        "schema_version": 1,
        "phase": "9A",
        "rfp": "titan-docs/RFP_phase_c_enhancements.md §3F",
        "captured_at_utc": _dt.datetime.now(_dt.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "titan_id": args.titan_id,
        "host": args.host,
        "ssh_host": args.ssh_host,
        "agno_worker_pid_start": pid,
        "agno_worker_pid_end": pid_end,
        "agno_worker_restarts_at_start": restarts_at_start,
        "agno_worker_restarts_at_end": restarts_end,
        "worker_restarted_during_run": worker_restarted_during_run,
        "worker_was_freshly_booted": bool(args.worker_freshly_booted),
        "prompts": PROMPTS,
        "chat_results": chat_results,
        "checkpoints": cps,
        "deltas": deltas,
    }
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    shutil.move(tmp, out_path)
    size_kb = os.path.getsize(out_path) // 1024
    print(f"[ok] baseline written → {out_path} ({size_kb} kB)", file=sys.stderr)

    # 7. Console summary
    boot = cps["boot_complete"]
    final = cps["chat_10_out"]
    boot_rss = (boot.get("smaps_rollup") or {}).get("Rss_kb", 0) / 1024
    final_rss = (final.get("smaps_rollup") or {}).get("Rss_kb", 0) / 1024
    boot_mods = (boot.get("in_process") or {}).get("sys_modules_count", 0)
    final_mods = (final.get("in_process") or {}).get("sys_modules_count", 0)
    boot_libs = (boot.get("heavy_libs") or {}).get("total_bytes", 0) / (1024 * 1024)
    final_libs = (final.get("heavy_libs") or {}).get("total_bytes", 0) / (1024 * 1024)
    print(
        f"\n=== Phase 9 baseline summary ({args.titan_id}) ===\n"
        f"  RSS:         {boot_rss:8.1f} MB  →  {final_rss:8.1f} MB  "
        f"({final_rss - boot_rss:+.1f} MB across 10 chats)\n"
        f"  Heavy .so:   {boot_libs:8.1f} MB  →  {final_libs:8.1f} MB  "
        f"({final_libs - boot_libs:+.1f} MB)\n"
        f"  sys.modules: {boot_mods:8d}     →  {final_mods:8d}     "
        f"({final_mods - boot_mods:+d})\n"
        f"  Chats:       {sum(1 for c in chat_results if c.get('status') == 200)}/10 "
        f"successful\n"
        f"  Output:      {out_path}\n",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
