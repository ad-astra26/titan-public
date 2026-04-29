"""
titan_plugin/modules/warning_monitor_worker.py — WARNING+ event aggregator.

Created 2026-04-25 in response to BUG-T1-CONSCIOUSNESS-67D-STATE-VECTOR
which had been silently active for 37 days because 442 silent-swallow
sites in the codebase were swallowing exceptions at DEBUG level.

This worker is the FORCE MULTIPLIER for the silent-swallow remediation
effort: it tails the brain log + reads SILENT_SWALLOW_COUNTERS from
in-process memory (via bus on demand), aggregates WARNING+ events with
rate/count/first-seen/last-seen, and persists the rolling state to
disk so arch_map warnings can surface them at session start.

Per directive_error_visibility.md (codified 2026-04-25):
  - Every WARNING+ event MUST be visible to this worker
  - arch_map warnings MUST read this worker's state at session start
  - Going forward every new code path requires WARNING+ logging

Bus protocol:
  - PUBLISHES: WARNING_PULSE (when rate spike detected, dst="all")
  - SUBSCRIBES: SILENT_SWALLOW_REPORT (any process can self-report
    its swallow counters; aggregated into the worker's view)

Endpoint exposed via dashboard:
  - GET /v4/warning-monitor → snapshot of recent warnings + swallow counters

Persistence:
  - data/warning_monitor/state.json — rolling state, written every 60s
  - data/warning_monitor/events.jsonl — append-only event log

Design philosophy: lightweight, no external deps beyond stdlib + the
worker scaffold. Tail-based rather than handler-installed so it works
across the existing brain log infrastructure without code changes.
"""

import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from titan_plugin import bus

logger = logging.getLogger("warning_monitor")

# ── Config defaults (overridable via worker config dict) ─────────────
DEFAULT_BRAIN_LOG_PATH = "/tmp/titan_brain.log"
DEFAULT_STATE_PATH = "data/warning_monitor/state.json"
DEFAULT_EVENTS_PATH = "data/warning_monitor/events.jsonl"
DEFAULT_PERSIST_INTERVAL_S = 60.0
DEFAULT_HEARTBEAT_INTERVAL_S = 30.0
DEFAULT_RATE_SPIKE_THRESHOLD = 5  # events / minute on a single key
DEFAULT_TAIL_BATCH_LINES = 200    # max lines processed per tail iter
DEFAULT_RECENT_RING_SIZE = 200    # last-N events kept in memory for /v4 response

# Per-iteration byte cap for tail reads. Was 1 MB before 2026-04-29 — at
# T1's brain-log density (~107 B/line) that materialized ~10k transient
# strings per iter via `text.splitlines()`, fragmenting glibc's 2-arena
# heap (MALLOC_ARENA_MAX=2) and driving worker RSS to 1+ GB before
# Guardian killed it (every 8-10 min crash-loop on T1, observed
# 2026-04-29 09:00-10:30 UTC). 256 KB combined with line-streaming
# (no splitlines) keeps each iter's heap churn bounded.
MAX_TAIL_BYTES = 256 * 1024

# Defense-in-depth cap on the aggregated dict. T1 currently holds ~97
# distinct keys but with no eviction the dict could grow indefinitely
# under sustained novel-tag traffic. When eviction fires we drop the
# entries with the oldest `last_seen_ts`. State persistence already
# truncates each entry to a small JSON shape, so the working-set cost
# of holding the cap is tiny (~few hundred KB).
AGGREGATED_KEY_CAP = 1000

# ── Brain log line shape (titan_main format) ──────────────────────────
# Example: "07:36:54 [INFO] [GroundUp] state_vector len=67 type=list ..."
LOG_RE = re.compile(
    r"^(?P<ts>\d{2}:\d{2}:\d{2})\s+"
    r"\[(?P<level>WARNING|ERROR|CRITICAL)\]\s+"
    r"(?P<rest>.*)$"
)

# ── Key extraction — first [TAG] in the message line ──────────────────
TAG_RE = re.compile(r"\[([^\]]+)\]")


def _extract_key(rest: str) -> str:
    """Extract a stable grouping key from a log line's text."""
    m = TAG_RE.search(rest)
    if m:
        return m.group(1)
    # Fallback: first 4 words
    return " ".join(rest.split()[:4]) or "unknown"


def _send_msg(send_queue, msg_type: str, src: str, dst: str, payload: dict):
    """Helper for emitting bus messages from the worker."""
    try:
        send_queue.put_nowait({
            "type": msg_type,
            "src": src,
            "dst": dst,
            "ts": time.time(),
            "payload": payload,
        })
    except Exception as e:
        # Bootstrap: cannot use swallow_warn here (would loop). Drop silently.
        logger.warning("[WarningMonitor] send_queue put_nowait failed: %s", e)


def warning_monitor_worker_main(recv_queue, send_queue, name: str,
                                config: dict) -> None:
    """Worker entrypoint, matching the Guardian ModuleSpec signature
    used by cgn_worker_main / emot_cgn_worker_main / etc:
        entry_fn(recv_queue, send_queue, name, config)
    """
    logger.info("[WarningMonitor] Starting (pid=%d)", os.getpid())

    # ── Resolve config ───────────────────────────────────────────────
    cfg = dict(config or {})
    brain_log_path = cfg.get("brain_log_path", DEFAULT_BRAIN_LOG_PATH)
    state_path = cfg.get("state_path", DEFAULT_STATE_PATH)
    events_path = cfg.get("events_path", DEFAULT_EVENTS_PATH)
    persist_interval_s = float(cfg.get("persist_interval_s",
                                       DEFAULT_PERSIST_INTERVAL_S))
    heartbeat_interval_s = float(cfg.get("heartbeat_interval_s",
                                         DEFAULT_HEARTBEAT_INTERVAL_S))
    rate_spike_threshold = int(cfg.get("rate_spike_threshold",
                                       DEFAULT_RATE_SPIKE_THRESHOLD))
    tail_batch_lines = int(cfg.get("tail_batch_lines",
                                   DEFAULT_TAIL_BATCH_LINES))
    recent_ring_size = int(cfg.get("recent_ring_size",
                                   DEFAULT_RECENT_RING_SIZE))

    # ── Ensure directories exist ─────────────────────────────────────
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    Path(events_path).parent.mkdir(parents=True, exist_ok=True)

    # ── State: per-key aggregations ──────────────────────────────────
    aggregated: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "first_seen_ts": 0.0,
        "last_seen_ts": 0.0,
        "last_msg": "",
        "by_level": defaultdict(int),
        "rate_window": deque(maxlen=60),  # timestamps of last 60 events
    })
    recent_events: deque = deque(maxlen=recent_ring_size)
    spike_alerts: dict[str, float] = {}  # key -> last_alert_ts (cooldown)

    # ── Load prior state (resume across restarts) ────────────────────
    try:
        if os.path.exists(state_path):
            with open(state_path) as f:
                prior = json.load(f)
            for k, v in (prior.get("aggregated") or {}).items():
                aggregated[k]["count"] = int(v.get("count", 0))
                aggregated[k]["first_seen_ts"] = float(v.get("first_seen_ts", 0.0))
                aggregated[k]["last_seen_ts"] = float(v.get("last_seen_ts", 0.0))
                aggregated[k]["last_msg"] = str(v.get("last_msg", ""))
                for lvl, c in (v.get("by_level") or {}).items():
                    aggregated[k]["by_level"][lvl] = int(c)
            logger.info("[WarningMonitor] Loaded prior state: %d keys",
                        len(aggregated))
    except Exception as e:
        logger.warning("[WarningMonitor] State load failed: %s — starting fresh", e)

    # ── Tail position tracking ───────────────────────────────────────
    log_inode = None
    log_offset = 0

    def _resync_tail():
        nonlocal log_inode, log_offset
        try:
            st = os.stat(brain_log_path)
            log_inode = st.st_ino
            # Start at end (skip historical content on first start; on
            # subsequent runs persistent state covers prior aggregation).
            log_offset = st.st_size
        except FileNotFoundError:
            log_inode = None
            log_offset = 0

    _resync_tail()

    # ── Send MODULE_READY ────────────────────────────────────────────
    _send_msg(send_queue, bus.MODULE_READY, name, "guardian", {})
    logger.info(
        "[WarningMonitor] Ready — brain_log=%s state=%s persist_interval=%ds "
        "rate_spike_threshold=%d/min",
        brain_log_path, state_path, persist_interval_s, rate_spike_threshold,
    )

    last_persist = 0.0
    last_heartbeat = 0.0

    # ── Main loop ────────────────────────────────────────────────────
    while True:
        loop_t0 = time.time()
        try:
            # 1. Drain any incoming bus messages (currently only handle
            #    SILENT_SWALLOW_REPORT for cross-process counter merge).
            try:
                from titan_plugin.core import worker_swap_handler as _swap
                while True:
                    msg = recv_queue.get_nowait()
                    # ── Microkernel v2 Phase B.2.1 — supervision-transfer dispatch ──
                    if _swap.maybe_dispatch_swap_msg(msg):
                        continue
                    if msg.get("type") == bus.SILENT_SWALLOW_REPORT:
                        _ingest_swallow_report(aggregated, msg.get("payload") or {})
            except Exception:
                pass  # queue.Empty or similar — normal

            # 2. Tail brain log for new WARNING+ lines.
            new_events = _tail_brain_log(
                brain_log_path, log_inode, log_offset, tail_batch_lines,
            )
            if new_events is not None:
                lines, new_inode, new_offset = new_events
                log_inode = new_inode
                log_offset = new_offset
                for ts_str, level, rest in lines:
                    key = _extract_key(rest)
                    now = time.time()
                    entry = aggregated[key]
                    entry["count"] += 1
                    if entry["first_seen_ts"] == 0.0:
                        entry["first_seen_ts"] = now
                    entry["last_seen_ts"] = now
                    entry["last_msg"] = rest[:300]
                    entry["by_level"][level] += 1
                    entry["rate_window"].append(now)
                    recent_events.append({
                        "ts": now, "log_ts": ts_str, "level": level,
                        "key": key, "msg": rest[:300],
                    })
                    # Append to events.jsonl (best-effort)
                    try:
                        with open(events_path, "a") as f:
                            f.write(json.dumps({
                                "ts": now, "log_ts": ts_str, "level": level,
                                "key": key, "msg": rest[:500],
                            }) + "\n")
                    except Exception:
                        # Persistence is best-effort; aggregation continues
                        pass
                    # Rate spike detection
                    rate_1m = sum(1 for t in entry["rate_window"]
                                  if now - t < 60.0)
                    if rate_1m >= rate_spike_threshold:
                        last_alert = spike_alerts.get(key, 0.0)
                        if now - last_alert > 300.0:  # 5 min cooldown
                            spike_alerts[key] = now
                            _send_msg(send_queue, bus.WARNING_PULSE, name, "all",
                                      {
                                          "key": key, "level": level,
                                          "rate_1m": rate_1m,
                                          "msg": rest[:300],
                                          "count_total": entry["count"],
                                      })
                            logger.warning(
                                "[WarningMonitor] Rate spike on '%s' "
                                "(%d/min, total=%d) — WARNING_PULSE emitted",
                                key, rate_1m, entry["count"])

            # 3. Persist state every persist_interval_s
            now = time.time()
            if now - last_persist > persist_interval_s:
                _persist_state(state_path, aggregated, recent_events)
                _maybe_rotate_events_log(events_path)
                last_persist = now

            # 4. Heartbeat
            if now - last_heartbeat > heartbeat_interval_s:
                _send_msg(send_queue, bus.MODULE_HEARTBEAT, name, "guardian",
                          {"keys": len(aggregated),
                           "recent_count": len(recent_events)})
                last_heartbeat = now

        except Exception as e:
            # Last-line defense — never crash the monitor itself.
            # This catch USES logger.warning (not swallow_warn) intentionally:
            # swallow_warn would loop us back through the brain log.
            logger.warning("[WarningMonitor] Loop iteration error: %s", e,
                           exc_info=True)

        # Sleep proportional to work done; min 0.5s, max 5s
        elapsed = time.time() - loop_t0
        time.sleep(max(0.5, min(5.0, 1.0 - elapsed)))


def _tail_brain_log(path: str, prior_inode, prior_offset: int,
                    max_lines: int):
    """Read up to `max_lines` new lines appended since (prior_inode, prior_offset).

    Streaming variant — iterates the file line-by-line via the file
    object's own iterator, never materializing the whole new-bytes range
    as a single string + list (which under MALLOC_ARENA_MAX=2 fragmented
    the worker's heap and grew RSS unbounded — see commit message
    referencing Component B leak, 2026-04-29).

    Bounds per call:
      - at most `max_lines` parsed entries
      - at most MAX_TAIL_BYTES (256 KB) of new bytes consumed

    A trailing partial line (no \\n) is left for the next call so we
    resume cleanly; the offset advances past the LAST FULLY-CONSUMED
    line only.

    Returns (lines, new_inode, new_offset) tuple, or None if log unavailable.
    Handles log rotation by detecting inode change → reset to start of new file.
    """
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None

    new_inode = st.st_ino
    file_size = st.st_size

    if prior_inode is None:
        # First run / log appeared
        return [], new_inode, file_size

    if new_inode != prior_inode:
        # Log rotated; start from beginning of new file
        prior_offset = 0

    if file_size <= prior_offset:
        # No new bytes (or file shrunk — unusual)
        return [], new_inode, file_size

    parsed = []
    new_offset = prior_offset
    try:
        with open(path, "rb") as f:
            f.seek(prior_offset)
            bytes_consumed = 0
            for raw in f:
                # Stop conditions are checked BEFORE consuming the line so
                # the offset stays at the last-fully-consumed boundary.
                if len(parsed) >= max_lines:
                    break
                if bytes_consumed >= MAX_TAIL_BYTES:
                    break
                # Defensive: a line without a trailing newline is a partial
                # write at EOF. Don't consume it — leave for the next call.
                if not raw.endswith(b"\n"):
                    break
                bytes_consumed += len(raw)
                # Decode each line individually. This deliberately avoids
                # `chunk.decode()` + `splitlines()` which would allocate
                # a list of every line in the chunk before we slice it.
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                except Exception:
                    continue
                m = LOG_RE.match(line)
                if m:
                    parsed.append((m.group("ts"), m.group("level"),
                                   m.group("rest")))
            new_offset = prior_offset + bytes_consumed
    except Exception as e:
        logger.warning("[WarningMonitor] tail read failed: %s", e)
        return [], prior_inode, prior_offset

    return parsed, new_inode, new_offset


def _ingest_swallow_report(aggregated: dict, payload: dict) -> None:
    """Merge a SILENT_SWALLOW_REPORT bus payload into aggregated."""
    counters = payload.get("counters") or {}
    src_proc = payload.get("src_process", "unknown")
    for key, ctr in counters.items():
        merged_key = f"swallow:{src_proc}:{key}"
        entry = aggregated[merged_key]
        cur_count = int(ctr.get("count", 0))
        if cur_count > entry["count"]:
            entry["count"] = cur_count
            entry["last_msg"] = str(ctr.get("last_msg", ""))[:300]
            entry["last_seen_ts"] = float(ctr.get("last_seen_ts", 0.0))
            if entry["first_seen_ts"] == 0.0:
                entry["first_seen_ts"] = float(ctr.get("first_seen_ts", 0.0))


# Default events.jsonl size cap. Rotation creates a gzipped archive next
# to the live file. Closes BUG-WARNING-MONITOR-PERSISTENCE-UNBOUNDED
# (2026-04-27) — the file was growing append-only forever (440MB observed
# on T1; T2/T3 will follow the same trajectory under sustained traffic).
EVENTS_LOG_MAX_MB = 50
EVENTS_LOG_KEEP_ARCHIVES = 5


def _maybe_rotate_events_log(events_path: str,
                              max_mb: int = EVENTS_LOG_MAX_MB,
                              keep: int = EVENTS_LOG_KEEP_ARCHIVES) -> None:
    """Rotate events.jsonl if it exceeds max_mb; prune old archives.

    Best-effort — any I/O error is logged at WARNING and swallowed; the
    next persist cycle retries. Atomic rename ensures no events are lost
    if rotation succeeds; if gzip fails we leave the original alone.
    """
    try:
        if not os.path.exists(events_path):
            return
        size_mb = os.path.getsize(events_path) / (1024 * 1024)
        if size_mb < max_mb:
            return
        import gzip
        import shutil
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        archive_path = f"{events_path}.{ts}.gz"
        with open(events_path, "rb") as fin:
            with gzip.open(archive_path, "wb", compresslevel=6) as fout:
                shutil.copyfileobj(fin, fout)
        # Truncate atomically by replacing with an empty file.
        with open(events_path + ".tmp", "w") as f:
            pass
        os.replace(events_path + ".tmp", events_path)
        logger.info(
            "[WarningMonitor] rotated events log (%.1fMB → archive %s)",
            size_mb, os.path.basename(archive_path))
        # Prune oldest archives beyond `keep`.
        try:
            d = os.path.dirname(events_path) or "."
            base = os.path.basename(events_path)
            archives = sorted(
                f for f in os.listdir(d)
                if f.startswith(base + ".") and f.endswith(".gz")
            )
            for old in archives[:-keep]:
                try:
                    os.remove(os.path.join(d, old))
                except Exception:
                    pass
        except Exception:
            pass
    except Exception as e:
        logger.warning("[WarningMonitor] events log rotation failed: %s", e)


def _evict_aggregated_lru(aggregated: dict, cap: int) -> int:
    """Drop oldest entries (by last_seen_ts) until aggregated has ≤ cap keys.

    Returns the number of evicted keys. Called pre-persist so the on-disk
    state and the in-memory dict stay in lockstep.
    """
    excess = len(aggregated) - cap
    if excess <= 0:
        return 0
    victims = sorted(
        aggregated.items(),
        key=lambda kv: kv[1].get("last_seen_ts", 0.0),
    )[:excess]
    for k, _ in victims:
        aggregated.pop(k, None)
    return len(victims)


def _persist_state(state_path: str, aggregated: dict,
                   recent_events: deque,
                   cap: int = AGGREGATED_KEY_CAP) -> None:
    """Atomically write aggregated state to disk.

    Evicts oldest aggregated keys (by last_seen_ts) before writing if
    the dict has exceeded `cap`. Eviction happens in-place so the in-memory
    dict and the persisted file stay consistent across restarts.
    """
    evicted = _evict_aggregated_lru(aggregated, cap)
    if evicted:
        logger.info(
            "[WarningMonitor] LRU-evicted %d aggregated key(s) (cap=%d)",
            evicted, cap)
    out = {
        "saved_ts": time.time(),
        "aggregated": {
            k: {
                "count": v["count"],
                "first_seen_ts": v["first_seen_ts"],
                "last_seen_ts": v["last_seen_ts"],
                "last_msg": v["last_msg"],
                "by_level": dict(v["by_level"]),
                "rate_1m": sum(1 for t in v["rate_window"]
                               if time.time() - t < 60.0),
            }
            for k, v in aggregated.items()
        },
        "recent_events": list(recent_events),
    }
    tmp_path = state_path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        os.replace(tmp_path, state_path)
    except Exception as e:
        logger.warning("[WarningMonitor] state persist failed: %s", e)


def get_state_snapshot(state_path: str = DEFAULT_STATE_PATH) -> dict:
    """Read current persisted state — used by /v4/warning-monitor + arch_map."""
    try:
        with open(state_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"saved_ts": 0.0, "aggregated": {}, "recent_events": []}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
