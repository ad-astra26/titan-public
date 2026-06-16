"""gc_leak_probe — flag-gated periodic gc object-type histogram for leak attribution.

PROFILING.md (2026-05-30): under-load PSS profiling found per-worker RSS leaks
(memory +4.36 MB/min, social/warning/recorder secondary). tracemalloc is disabled
fleet-wide (too costly: 76% CPU / 4-5 min boot), and there is no live heap hook on
workers — so this is the lightweight alternative to pin the leaking structure:

  every `interval_s`, walk gc.get_objects() and log a histogram of object counts
  by type name + total + RSS to /tmp/gc_leak_<worker>.jsonl. Diffing the first vs
  last line shows which object TYPE grows under sustained load (np.ndarray →
  embeddings; a node dict → mempool; a specific class → that subsystem) → straight
  to the leaking structure.

DISABLED by default. Enable per-process with env TITAN_GC_LEAK_PROBE=1. One
gc.get_objects() walk per interval, NO gc.collect() (would alter behavior) and NO
allocation tracking (unlike tracemalloc) — overhead is a single O(live-objects)
sweep every interval_s. Removable: unset the env + delete the start() call.
"""
from __future__ import annotations

import gc
import json
import os
import threading
import time
from collections import Counter

_ENV_FLAG = "TITAN_GC_LEAK_PROBE"
# Sentinel-file fallback: kernel-rs (L0) spawns the python peers with a CURATED
# env that drops arbitrary TITAN_* vars, so env activation can't reach workers.
# A sentinel file is env-independent: workers read it at startup. Presence =
# enabled; file CONTENT (if non-empty) = comma-separated worker allowlist, else
# all workers. Create: `echo "memory,social_worker,..." > /tmp/titan_gc_leak_probe`.
_SENTINEL_FILE = "/tmp/titan_gc_leak_probe"
_started: set[str] = set()


def _resolve_flag() -> tuple[bool, str]:
    """(enabled, allowlist_csv) from env first, then the sentinel file."""
    if os.environ.get(_ENV_FLAG) == "1":
        return True, os.environ.get("TITAN_GC_LEAK_PROBE_WORKERS", "")
    try:
        with open(_SENTINEL_FILE) as f:
            return True, f.read().strip()
    except OSError:
        return False, ""


def _rss_mb() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _probe_loop(worker_name: str, interval_s: float, top_n: int) -> None:
    out_path = f"/tmp/gc_leak_{worker_name}.jsonl"
    pid = os.getpid()
    while True:
        try:
            objs = gc.get_objects()
            counts = Counter()
            mod_names = Counter()        # module.__name__ → count (dup = re-loaded)
            for o in objs:
                tn = type(o).__name__
                counts[tn] += 1
                if tn == "module":
                    try:
                        mod_names[getattr(o, "__name__", "?")] += 1
                    except Exception:
                        pass
            # Duplicate module objects (same __name__, >1 instance) = modules
            # re-imported fresh (bypassing sys.modules cache) → the import-leak
            # source. The top dup names point straight at the culprit library.
            dup_mods = {k: v for k, v in mod_names.most_common(25) if v > 1}
            # Garbage-vs-leak: gc.collect() returns the count of unreachable
            # cyclic objects it freed. High + RSS-correlated = collectable cyclic
            # garbage (gc can't keep up under load), NOT a referenced leak.
            collected = gc.collect()
            rec = {
                "ts": time.time(),
                "worker": worker_name,
                "pid": pid,
                "rss_mb": round(_rss_mb(), 1),
                "total_objects": len(objs),
                "gc_tracked": sum(counts.values()),
                "gc_collected": collected,
                "dup_modules": dup_mods,
                "top": dict(counts.most_common(top_n)),
            }
            del objs, counts, mod_names
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            # A leak probe must never destabilize the worker it observes.
            pass
        time.sleep(interval_s)


def start_gc_leak_probe(worker_name: str, interval_s: float = 30.0,
                        top_n: int = 30) -> bool:
    """Start the gc histogram probe for this process IF TITAN_GC_LEAK_PROBE=1.

    Returns True if started, False if the flag is off (no-op) or already running.
    Safe to call unconditionally at worker startup.
    """
    enabled, allow = _resolve_flag()
    if not enabled:
        return False
    # Optional allowlist (env TITAN_GC_LEAK_PROBE_WORKERS or sentinel-file
    # content): only those workers probe — keeps the heavy-load run's overhead
    # on the targets, not all ~40 workers. Empty = probe every worker.
    allow = allow.strip()
    if allow and worker_name not in {w.strip() for w in allow.split(",") if w.strip()}:
        return False
    if worker_name in _started:
        return False
    _started.add(worker_name)
    t = threading.Thread(
        target=_probe_loop, args=(worker_name, interval_s, top_n),
        daemon=True, name=f"gc-leak-probe-{worker_name}")
    t.start()
    return True
