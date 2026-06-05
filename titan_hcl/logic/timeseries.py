"""
Unified Time-Series Store for Titan Observatory.

Single SQLite table storing ~37 metrics every 5 minutes with 30-day retention.
Auto-downsamples to hourly averages for queries > 48h.

Usage:
    store = TimeseriesStore("./data/timeseries.db")
    store.record({"neuromod.DA": 0.52, "neuromod.5HT": 0.91, ...})
    rows = store.query(["neuromod.DA", "neuromod.5HT"], hours=168)
    store.cleanup()
"""

import sqlite3
import time
import logging
import os
import threading

logger = logging.getLogger("titan.timeseries")

_RETENTION_DAYS = 30
_DOWNSAMPLE_THRESHOLD_HOURS = 48
_RECORD_INTERVAL = 300  # 5 minutes


class TimeseriesStore:
    def __init__(self, db_path: str = "./data/timeseries.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._last_record_ts = 0.0
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self._db_path, timeout=10)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        ts    INTEGER NOT NULL,
                        name  TEXT    NOT NULL,
                        value REAL    NOT NULL,
                        PRIMARY KEY (ts, name)
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_name_ts
                    ON metrics(name, ts)
                """)
                conn.commit()
            finally:
                conn.close()
        logger.info("[Timeseries] Initialized at %s", self._db_path)

    def should_record(self) -> bool:
        """Check if enough time has passed since last recording."""
        return (time.time() - self._last_record_ts) >= _RECORD_INTERVAL

    def record(self, metrics: dict[str, float]):
        """Record a batch of metric values at the current timestamp."""
        if not metrics:
            return
        ts = int(time.time())
        rows = [(ts, name, float(val)) for name, val in metrics.items()
                if val is not None and isinstance(val, (int, float))]
        if not rows:
            return
        with self._lock:
            conn = sqlite3.connect(self._db_path, timeout=10)
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO metrics (ts, name, value) VALUES (?, ?, ?)",
                    rows,
                )
                conn.commit()
                self._last_record_ts = time.time()
                logger.info("[Timeseries] Recorded %d metrics at ts=%d", len(rows), ts)
            finally:
                conn.close()

    def query(
        self,
        metric_names: list[str],
        hours: int = 24,
        resolution: str = "auto",
    ) -> dict:
        """
        Query time-series data.

        Returns:
            {
                "metrics": {"neuromod.DA": [{"ts": 123, "value": 0.52}, ...], ...},
                "resolution": "5m" | "1h",
                "count": 288
            }
        """
        if not metric_names:
            return {"metrics": {}, "resolution": "raw", "count": 0}

        hours = min(hours, _RETENTION_DAYS * 24)
        cutoff = int(time.time()) - (hours * 3600)

        # Decide resolution
        if resolution == "auto":
            use_hourly = hours > _DOWNSAMPLE_THRESHOLD_HOURS
        else:
            use_hourly = resolution in ("1h", "hourly")

        placeholders = ",".join("?" for _ in metric_names)

        with self._lock:
            conn = sqlite3.connect(self._db_path, timeout=10)
            try:
                if use_hourly:
                    # Downsample to hourly averages
                    sql = f"""
                        SELECT (ts / 3600) * 3600 AS bucket, name, AVG(value) AS value
                        FROM metrics
                        WHERE name IN ({placeholders}) AND ts >= ?
                        GROUP BY bucket, name
                        ORDER BY bucket ASC
                    """
                    rows = conn.execute(sql, (*metric_names, cutoff)).fetchall()
                    res_label = "1h"
                else:
                    sql = f"""
                        SELECT ts, name, value
                        FROM metrics
                        WHERE name IN ({placeholders}) AND ts >= ?
                        ORDER BY ts ASC
                    """
                    rows = conn.execute(sql, (*metric_names, cutoff)).fetchall()
                    res_label = "5m"
            finally:
                conn.close()

        # Group by metric name
        result: dict[str, list] = {name: [] for name in metric_names}
        for ts, name, value in rows:
            if name in result:
                result[name].append({"ts": ts, "value": round(value, 6)})

        total = sum(len(v) for v in result.values())
        return {"metrics": result, "resolution": res_label, "count": total}

    def list_metrics(self) -> list[dict]:
        """Return all known metric names with latest value and count."""
        with self._lock:
            conn = sqlite3.connect(self._db_path, timeout=10)
            try:
                rows = conn.execute("""
                    SELECT name, MAX(ts) AS last_ts, COUNT(*) AS cnt
                    FROM metrics
                    GROUP BY name
                    ORDER BY name
                """).fetchall()

                # Get latest values
                result = []
                for name, last_ts, cnt in rows:
                    val_row = conn.execute(
                        "SELECT value FROM metrics WHERE name = ? AND ts = ?",
                        (name, last_ts),
                    ).fetchone()
                    result.append({
                        "name": name,
                        "latest_value": round(val_row[0], 6) if val_row else None,
                        "latest_ts": last_ts,
                        "count": cnt,
                    })
            finally:
                conn.close()
        return result

    def cleanup(self):
        """Remove data older than retention period."""
        cutoff = int(time.time()) - (_RETENTION_DAYS * 86400)
        with self._lock:
            conn = sqlite3.connect(self._db_path, timeout=10)
            try:
                cursor = conn.execute("DELETE FROM metrics WHERE ts < ?", (cutoff,))
                deleted = cursor.rowcount
                conn.commit()
                if deleted > 0:
                    logger.info("[Timeseries] Cleaned up %d old rows (cutoff=%d)", deleted, cutoff)
            finally:
                conn.close()

    def row_count(self) -> int:
        """Total rows in the metrics table."""
        with self._lock:
            conn = sqlite3.connect(self._db_path, timeout=10)
            try:
                row = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()
                return row[0] if row else 0
            finally:
                conn.close()


def collect_snapshot(state_refs: dict) -> dict[str, float]:
    """
    Collect current metric values from spirit_worker state refs.
    Called every 5 minutes from the spirit_worker main loop.
    Returns a flat dict of metric_name → float value.
    """
    metrics: dict[str, float] = {}

    # ── Neuromodulators ──
    nm = state_refs.get("neuromodulator_system")
    if nm:
        for mod_name in ("DA", "5HT", "NE", "ACh", "Endorphin", "GABA"):
            mod = nm.modulators.get(mod_name)
            if mod:
                metrics[f"neuromod.{mod_name}"] = mod.level

    # ── Dreaming ──
    coordinator = state_refs.get("coordinator")
    if coordinator:
        dreaming = coordinator.get("dreaming", {})
        if isinstance(dreaming, dict):
            for key in ("fatigue", "cycle_count", "recovery_pct", "epochs_since_dream"):
                val = dreaming.get(key)
                if isinstance(val, (int, float)):
                    metrics[f"dreaming.{key}"] = float(val)
            metrics["dreaming.is_dreaming"] = 1.0 if dreaming.get("is_dreaming") else 0.0

    # ── Neural Nervous System ──
    nns = state_refs.get("neural_nervous_system")
    if nns:
        metrics["ns.total_transitions"] = float(getattr(nns, "total_transitions", 0))
        metrics["ns.total_train_steps"] = float(getattr(nns, "total_train_steps", 0))

    # ── Chi ──
    if coordinator:
        chi = coordinator.get("chi", {})
        if isinstance(chi, dict):
            metrics["chi.total"] = float(chi.get("total", 0))
            for layer in ("spirit", "mind", "body"):
                layer_data = chi.get(layer, {})
                if isinstance(layer_data, dict):
                    metrics[f"chi.{layer}"] = float(layer_data.get("effective", 0))

    # ── Reasoning ──
    reasoning = state_refs.get("reasoning_engine")
    if reasoning:
        metrics["reasoning.total_chains"] = float(getattr(reasoning, "total_chains", 0))
        metrics["reasoning.buffer_size"] = float(getattr(reasoning, "_buffer_size", getattr(reasoning, "buffer_size", 0)))

    # ── Meta-Reasoning ──
    meta = state_refs.get("meta_reasoning")
    if meta:
        metrics["meta.total_chains"] = float(getattr(meta, "total_chains", 0))
        metrics["meta.total_wisdom"] = float(getattr(meta, "total_wisdom_saved", 0))
        metrics["meta.avg_reward"] = float(getattr(meta, "avg_reward", 0))

    # ── MSL ──
    msl = state_refs.get("msl")
    if msl:
        metrics["msl.i_confidence"] = float(msl.get_i_confidence()) if hasattr(msl, 'get_i_confidence') else 0.0
        metrics["msl.i_depth"] = float(msl.i_depth.depth) if hasattr(msl, 'i_depth') else 0.0
        metrics["msl.convergence_count"] = float(msl.confidence._convergence_count) if hasattr(msl, 'confidence') else 0.0

    # ── Expression composites ──
    expr = state_refs.get("expression_manager")
    if expr:
        composites = getattr(expr, "composites", {})
        for cname in ("SPEAK", "ART", "MUSIC"):
            comp = composites.get(cname)
            if comp:
                metrics[f"expression.{cname.lower()}_fires"] = float(getattr(comp, "fire_count", 0))

    # ── Pi heartbeat ──
    pi = state_refs.get("pi_monitor")
    if pi:
        metrics["pi.heartbeat_ratio"] = float(getattr(pi, "heartbeat_ratio", 0))
        metrics["pi.cluster_count"] = float(getattr(pi, "cluster_count", 0))
        metrics["pi.dev_age"] = float(getattr(pi, "developmental_age", 0))

    # ── Epoch ──
    if coordinator:
        epoch = coordinator.get("epoch_id", 0)
        if isinstance(epoch, (int, float)):
            metrics["epoch.id"] = float(epoch)

    # ── Vocabulary (from coordinator cache) ──
    if coordinator:
        lang = coordinator.get("language", {})
        if isinstance(lang, dict):
            metrics["vocab.count"] = float(lang.get("vocab_count", lang.get("total_words", 0)))

    # ── Social ──
    if coordinator:
        social = coordinator.get("social_pressure", {})
        if isinstance(social, dict):
            metrics["social.urge"] = float(social.get("urge", 0))
            metrics["social.posts_today"] = float(social.get("posts_today", 0))

    return metrics
