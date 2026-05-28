"""MetricsAggregator — the synthesis observability bundle (Phase 10, arch §18).

Computes the full metrics bundle each 60s recompute pass and writes
`synthesis_metrics_snapshot.json` (atomic tmp+rename). INV-Syn-25: a read-only,
rebuildable projection over canonical sources — never a write target, never a
decision source of truth.

Bundle:
  - sovereignty   : SovereigntyRatioMeter.compute() (the headline)
  - groundedness  : top-N concepts × groundedness `g` (read from spine_snapshot.json)
  - skills        : library size + mean utility + verified count (skills_snapshot.json)
  - retrieval     : p50/p95/p99 hot+warm per granularity (injected latency rings)
  - chi           : chi-budget compliance (injected provider; B.5)
  - chain_growth  : bytes-per-fork + block-rate + cost trend (data_dir scan; B.7)

Every sub-metric is independently soft-fail: a missing source yields
`{available: false}` for that sub-bundle, the rest still compute.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import tempfile
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = ["MetricsAggregator", "LatencyRing"]


class LatencyRing:
    """Bounded perf-span ring for retrieval latencies. Tagged (granularity, tier)."""

    def __init__(self, maxlen: int = 1000):
        from collections import deque
        self._d = deque(maxlen=maxlen)

    def record(self, granularity: str, tier: str, elapsed_ms: float) -> None:
        self._d.append((str(granularity), str(tier), float(elapsed_ms)))

    def percentiles(self) -> dict:
        rows = list(self._d)
        if not rows:
            return {"available": False, "samples": 0}
        out: dict = {"available": True, "samples": len(rows)}
        # Overall + per (granularity, tier).
        out["overall"] = _pcts([ms for _, _, ms in rows])
        buckets: dict = {}
        for g, tier, ms in rows:
            buckets.setdefault(f"{g}:{tier}", []).append(ms)
        out["by_bucket"] = {k: _pcts(v) for k, v in buckets.items()}
        # Warming flag until ≥100 samples (p99 unreliable below that).
        out["warming"] = len(rows) < 100
        return out


def _pcts(vals: list) -> dict:
    if not vals:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}
    s = sorted(vals)
    n = len(s)

    def _p(q: float) -> float:
        if n == 1:
            return round(s[0], 2)
        idx = min(n - 1, int(q * n))
        return round(s[idx], 2)
    return {"p50": _p(0.50), "p95": _p(0.95), "p99": _p(0.99), "n": n}


class MetricsAggregator:
    def __init__(
        self,
        *,
        sovereignty_meter,
        snapshot_path: str,
        data_dir: str = "data",
        spine_snapshot_path: Optional[str] = None,
        skills_snapshot_path: Optional[str] = None,
        latency_ring: Optional[LatencyRing] = None,
        chi_stats_provider: Optional[Callable[[], dict]] = None,
        groundedness_top_n: int = 50,
    ) -> None:
        self._meter = sovereignty_meter
        self._snapshot_path = snapshot_path
        self._data_dir = data_dir
        self._spine_path = spine_snapshot_path or os.path.join(
            data_dir, "spine_snapshot.json")
        self._skills_path = skills_snapshot_path or os.path.join(
            data_dir, "skills_snapshot.json")
        self._latency_ring = latency_ring
        self._chi_provider = chi_stats_provider
        self._top_n = int(groundedness_top_n)

    # ── bundle ────────────────────────────────────────────────────────

    def build(self, now_ts: Optional[float] = None) -> dict:
        now = float(now_ts if now_ts is not None else time.time())
        return {
            "sovereignty": self._safe(self._sovereignty, now),
            "groundedness": self._safe(self._groundedness),
            "skills": self._safe(self._skill_library),
            "retrieval": self._safe(self._retrieval),
            "chi": self._safe(self._chi),
            "chain_growth": self._safe(self._chain_growth),
            "ts": now,
        }

    def export(self, now_ts: Optional[float] = None) -> Optional[str]:
        """Atomic tmp+rename write of synthesis_metrics_snapshot.json. Soft."""
        try:
            bundle = self.build(now_ts)
            d = os.path.dirname(self._snapshot_path) or "."
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(bundle, f)
                os.replace(tmp, self._snapshot_path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
            return self._snapshot_path
        except Exception as exc:
            logger.warning("[MetricsAggregator] export failed: %s", exc)
            return None

    # ── sub-metrics (each soft) ───────────────────────────────────────

    @staticmethod
    def _safe(fn, *a):
        try:
            return fn(*a)
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("[MetricsAggregator] sub-metric %s failed: %s",
                         getattr(fn, "__name__", fn), exc)
            return {"available": False, "error": str(exc)[:120]}

    def _sovereignty(self, now) -> dict:
        return {"available": True, "windows": self._meter.compute(now)}

    def _read_json(self, path: str) -> Optional[Any]:
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)

    def _groundedness(self) -> dict:
        spine = self._read_json(self._spine_path)
        if not spine:
            return {"available": False}
        concepts = spine.get("concepts") if isinstance(spine, dict) else None
        if not isinstance(concepts, list):
            return {"available": False}
        rows = []
        for c in concepts:
            if not isinstance(c, dict):
                continue
            g = c.get("groundedness")
            if g is None:
                g = c.get("g")
            rows.append({
                "concept_id": c.get("concept_id") or c.get("id") or "",
                "name": c.get("name", ""),
                "groundedness": round(float(g), 4) if g is not None else None,
            })
        rows = [r for r in rows if r["groundedness"] is not None]
        rows.sort(key=lambda r: r["groundedness"], reverse=True)
        return {"available": True, "count": len(rows),
                "heatmap": rows[: self._top_n]}

    def _skill_library(self) -> dict:
        snap = self._read_json(self._skills_path)
        if not snap:
            return {"available": False}
        skills = snap.get("skills") if isinstance(snap, dict) else None
        if not isinstance(skills, list):
            return {"available": False}
        utils = [float(s.get("utility_score", 0.0)) for s in skills
                 if isinstance(s, dict)]
        verified = sum(1 for s in skills
                       if isinstance(s, dict) and s.get("verified_at"))
        succ = sum(int(s.get("success_count", 0)) for s in skills
                   if isinstance(s, dict))
        fail = sum(int(s.get("failure_count", 0)) for s in skills
                   if isinstance(s, dict))
        return {
            "available": True,
            "size": len(skills),
            "verified_count": verified,
            "mean_utility": round(statistics.mean(utils), 4) if utils else 0.0,
            "success_total": succ,
            "failure_total": fail,
            "success_ratio": round(succ / (succ + fail), 4) if (succ + fail) else None,
        }

    def _retrieval(self) -> dict:
        if self._latency_ring is None:
            return {"available": False}
        return {"available": True, **self._latency_ring.percentiles()}

    def _chi(self) -> dict:
        if self._chi_provider is None:
            return {"available": False}
        stats = self._chi_provider() or {}
        return {"available": True, **stats}

    def _chain_growth(self) -> dict:
        # Cheap bytes-on-disk scan of the timechain fork files (no full read).
        chain_dir = os.path.join(self._data_dir, "timechain")
        if not os.path.isdir(chain_dir):
            return {"available": False}
        by_fork: dict = {}
        total = 0
        try:
            for fn in os.listdir(chain_dir):
                if not fn.endswith(".bin"):
                    continue
                p = os.path.join(chain_dir, fn)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                by_fork[fn] = sz
                total += sz
        except OSError:
            return {"available": False}
        return {
            "available": True,
            "total_bytes": total,
            "by_fork_bytes": by_fork,
            # B.7 bounded-growth assertion is evaluated by the soak (trend over
            # snapshots); the snapshot reports the point-in-time bytes.
        }
