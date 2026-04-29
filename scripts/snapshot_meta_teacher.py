#!/usr/bin/env python3
"""
scripts/snapshot_meta_teacher.py — Capture a point-in-time snapshot of the
Meta-Teacher + Meta-Reasoning + outer-layer state across one or all Titans.

Written for the rFP_meta_teacher_v2 Phase A + B session (2026-04-24) to
bracket the outer-layer flag flip and deploy: take a baseline snapshot post-
flip, let the system run, take another at end-of-session, compare.

Each Titan snapshot is one JSON file under:
    data/snapshots/session_<TAG>/<TS>_<TITAN>.json

Captures:
  - /v4/meta-teacher/status  (feature flags, 24h critique stats, adoption)
  - /v4/meta-teacher/critiques?limit=50  (last 50 critique rows)
  - /v4/meta-teacher/memory  (Phase B cold-tier counts + retrieval rate)
  - /v4/meta-teacher/memory/still-needs-push
  - /v4/meta-teacher/maker-info?limit=20
  - /v4/meta-service/*  (meta-service endpoints, best-effort)
  - /v4/sensors  (outer_body rich signal + source producers)
  - /v4/bus-health
  - /v4/trinity  (quick Trinity snapshot)
  - tail of /tmp/titan_brain.log last 200 lines (compressed)

Usage:
  python scripts/snapshot_meta_teacher.py --t1
  python scripts/snapshot_meta_teacher.py --all --tag pre-flip
  python scripts/snapshot_meta_teacher.py --all --tag post-flip
  python scripts/snapshot_meta_teacher.py --all --tag end-of-session
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import urllib.request


def _utcnow() -> datetime:
    """Timezone-aware UTC now — replacement for deprecated utcnow()."""
    return datetime.now(timezone.utc)


TITAN_ENDPOINTS = {
    "T1": "http://localhost:7777",
    "T2": "http://10.135.0.6:7777",
    "T3": "http://10.135.0.6:7778",
}

ENDPOINTS_TO_PROBE = [
    "/v4/meta-teacher/status",
    "/v4/meta-teacher/critiques?limit=50",
    "/v4/meta-teacher/memory",
    "/v4/meta-teacher/memory/still-needs-push",
    "/v4/meta-teacher/maker-info?limit=20",
    "/v4/meta-service/status",
    "/v4/meta-service/recruiters",
    "/v4/meta-service/rewards",
    "/v4/sensors",
    "/v4/bus-health",
    "/v4/trinity",
    "/v4/meta-outer/status",
]

# Brain-log tail is only meaningful for T1 (local file). For remote Titans
# we grab it via SSH only if --with-remote-brain is passed (rate-limited).
T1_BRAIN_LOG = "/tmp/titan_brain.log"


def http_get(base: str, path: str, timeout: float = 10.0) -> dict:
    url = base + path
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "snapshot/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = {"_raw_text": body[:4096]}
            return {"ok": True, "status": r.status, "data": data}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def tail_brain_log(lines: int = 200) -> list[str]:
    try:
        if not os.path.exists(T1_BRAIN_LOG):
            return []
        result = subprocess.run(
            ["tail", "-n", str(lines), T1_BRAIN_LOG],
            capture_output=True, text=True, timeout=5)
        return result.stdout.splitlines()
    except Exception as e:
        return [f"[snapshot] brain-log tail failed: {e}"]


def critique_jsonl_counts(data_dir: str = "./data") -> dict:
    """Local count of critique rows per day + prompt version breakdown."""
    mt_dir = os.path.join(data_dir, "meta_teacher")
    counts: dict = {"files": [], "total_rows": 0, "by_prompt_version": {},
                     "rows_with_topic_key": 0,
                     "rows_with_retrieved_context_ids": 0}
    if not os.path.isdir(mt_dir):
        return counts
    import glob
    for fpath in sorted(glob.glob(os.path.join(mt_dir, "critiques.*.jsonl"))):
        try:
            with open(fpath) as f:
                lines = f.readlines()
            counts["files"].append({
                "path": fpath,
                "lines": len(lines),
                "mtime": os.path.getmtime(fpath),
            })
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                counts["total_rows"] += 1
                pv = str(row.get("prompt_version", "unknown"))
                counts["by_prompt_version"][pv] = (
                    counts["by_prompt_version"].get(pv, 0) + 1)
                if row.get("topic_key"):
                    counts["rows_with_topic_key"] += 1
                if row.get("retrieved_context_ids"):
                    counts["rows_with_retrieved_context_ids"] += 1
        except Exception:
            continue
    return counts


def cold_journal_counts(data_dir: str = "./data") -> dict:
    """Local read of teaching_journal.jsonl for Phase B cold tier status."""
    path = os.path.join(data_dir, "meta_teacher", "teaching_journal.jsonl")
    out = {
        "path": path, "exists": os.path.exists(path),
        "filesize_bytes": 0, "total_rows": 0, "unique_topics": 0,
        "still_needs_push": 0,
    }
    if not out["exists"]:
        return out
    try:
        out["filesize_bytes"] = os.path.getsize(path)
    except OSError:
        pass
    topics: dict[str, dict] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out["total_rows"] += 1
                tk = row.get("topic_key")
                if isinstance(tk, str):
                    topics[tk] = row  # last-wins
    except Exception:
        pass
    out["unique_topics"] = len(topics)
    out["still_needs_push"] = sum(
        1 for r in topics.values() if r.get("still_needs_push"))
    return out


def snapshot_titan(titan: str, base: str, capture_brain_log: bool) -> dict:
    snap: dict = {
        "titan": titan,
        "endpoint": base,
        "ts_iso": _utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ts_unix": time.time(),
        "host": socket.gethostname(),
        "endpoints": {},
    }
    for path in ENDPOINTS_TO_PROBE:
        snap["endpoints"][path] = http_get(base, path)
    if titan == "T1":
        snap["critique_jsonl_counts"] = critique_jsonl_counts()
        snap["cold_journal_counts"] = cold_journal_counts()
        if capture_brain_log:
            snap["brain_log_tail"] = tail_brain_log(200)
    return snap


def write_snapshot(snap: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = _utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{snap['titan']}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(snap, f, indent=2, default=str)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t1", action="store_true", help="T1 only")
    ap.add_argument("--t2", action="store_true", help="T2 only")
    ap.add_argument("--t3", action="store_true", help="T3 only")
    ap.add_argument("--all", action="store_true", help="T1+T2+T3")
    ap.add_argument("--tag", default="session",
                     help="Subfolder tag (e.g. 'pre-flip', 'post-flip', 'end')")
    ap.add_argument("--out-root", default="data/snapshots",
                     help="Root output dir")
    ap.add_argument("--with-brain-log", action="store_true", default=True,
                     help="Include T1 brain log tail (default on)")
    args = ap.parse_args()

    targets: list[str] = []
    if args.all:
        targets = ["T1", "T2", "T3"]
    else:
        if args.t1:
            targets.append("T1")
        if args.t2:
            targets.append("T2")
        if args.t3:
            targets.append("T3")
    if not targets:
        targets = ["T1"]

    session_tag = args.tag.replace("/", "_").replace(" ", "_")
    day_tag = _utcnow().strftime("%Y%m%d")
    out_dir = os.path.join(args.out_root, f"session_{day_tag}", session_tag)

    print(f"[snapshot] Capturing Titans={targets} tag='{session_tag}' "
          f"→ {out_dir}")

    written: list[str] = []
    for titan in targets:
        base = TITAN_ENDPOINTS[titan]
        print(f"[snapshot] {titan} @ {base} ...", flush=True)
        snap = snapshot_titan(
            titan, base, capture_brain_log=args.with_brain_log)
        path = write_snapshot(snap, out_dir)
        written.append(path)
        # Quick summary line per Titan
        mt_status = snap["endpoints"].get("/v4/meta-teacher/status", {})
        data = mt_status.get("data") if isinstance(mt_status, dict) else {}
        if isinstance(data, dict):
            payload = data.get("data") if "data" in data else data
            if isinstance(payload, dict):
                enabled = payload.get("enabled")
                c24 = payload.get("critiques_24h")
                tme = payload.get("teaching_memory_enabled")
                print(f"  enabled={enabled} critiques_24h={c24} "
                      f"teaching_memory_enabled={tme}")
        print(f"  → {path}")

    print(f"[snapshot] Wrote {len(written)} files to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
