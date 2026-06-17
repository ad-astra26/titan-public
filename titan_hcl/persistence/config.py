"""IMW configuration — loaded from titan_hcl/config.toml [persistence] section."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Parsed-config cache (mtime-gated) ───────────────────────────────────────
# from_titan_config[_section] is called per-route in worker hot paths (e.g.
# EventsTeacherDB constructed per chat — events_teacher.py:140) — each call
# re-opened + re-parsed the WHOLE config.toml from disk. Under chat load this
# was ~7.6% of agno_worker's on-CPU time (PROFILING.md F7, --gil sweep
# 2026-05-30). Cache the parsed dict, re-parsing only when config.toml's mtime
# changes (config is loaded at boot; a runtime edit self-corrects on the next
# call — same self-correcting idiom as snapshot_builders'
# _episodic_stats_mtime_gated). A fresh IMWConfig is still built per call via
# from_dict(), so no caller can mutate a shared instance.
_TOML_CACHE: dict = {"data": None, "mtime": None, "path": None}
_TOML_CACHE_LOCK = threading.Lock()


def _load_config_toml_cached(cfg_path: Path) -> dict:
    """Return the parsed config.toml as a dict, re-parsing only on mtime change.

    Returns ``{}`` if the file is absent. Thread-safe (double-checked under a
    lock so concurrent worker threads parse at most once per mtime)."""
    try:
        mtime = os.path.getmtime(cfg_path)
    except OSError:
        return {}
    cache = _TOML_CACHE
    cpath = str(cfg_path)
    if (cache["data"] is not None and cache["mtime"] == mtime
            and cache["path"] == cpath):
        return cache["data"]
    with _TOML_CACHE_LOCK:
        # Re-check under the lock — another thread may have just loaded it.
        if (cache["data"] is not None and cache["mtime"] == mtime
                and cache["path"] == cpath):
            return cache["data"]
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        with open(cfg_path, "rb") as f:
            full = tomllib.load(f)
        cache["data"] = full
        cache["mtime"] = mtime
        cache["path"] = cpath
        return full


DEFAULTS = {
    "enabled": False,               # master switch; False = direct writes (pre-IMW behavior)
    "mode": "disabled",             # "shadow" | "dual" | "canonical" | "hybrid" | "disabled"
    "transport": "unix_socket",     # "unix_socket" | "bus"
    "socket_path": "data/run/imw.sock",
    "wal_path": "data/run/imw.wal",
    "journal_dir": "data/run",
    "db_path": "data/inner_memory.db",
    "shadow_db_path": "data/inner_memory_shadow.db",
    "batch_window_ms": 10,
    "max_batch_size": 100,
    "fast_path_enabled": True,
    "busy_timeout_sec": 30.0,
    "service_wal_max_mb": 64,
    "max_in_flight_per_caller": 1000,
    "connect_timeout_sec": 5.0,
    "reconnect_backoff_min_ms": 100,
    "reconnect_backoff_max_ms": 5000,
    "direct_fallback_after_sec": 0,  # 0 = never fall back (block instead)
    "shm_tables": [],                # Phase A: ["hormone_snapshots"]
    "tables_canonical": [],          # per-table Phase 3 cutover list
}


@dataclass
class IMWConfig:
    enabled: bool = False
    mode: str = "disabled"
    transport: str = "unix_socket"
    socket_path: str = "data/run/imw.sock"
    wal_path: str = "data/run/imw.wal"
    journal_dir: str = "data/run"
    db_path: str = "data/inner_memory.db"
    shadow_db_path: str = "data/inner_memory_shadow.db"
    batch_window_ms: int = 10
    max_batch_size: int = 100
    fast_path_enabled: bool = True
    busy_timeout_sec: float = 30.0
    service_wal_max_mb: int = 64
    max_in_flight_per_caller: int = 1000
    connect_timeout_sec: float = 5.0
    reconnect_backoff_min_ms: int = 100
    reconnect_backoff_max_ms: int = 5000
    direct_fallback_after_sec: float = 0.0
    shm_tables: list = field(default_factory=list)
    tables_canonical: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, section: Optional[dict] = None) -> "IMWConfig":
        d = dict(DEFAULTS)
        if section:
            for k, v in section.items():
                if k in d:
                    d[k] = v
        # BUG-B1-SHARED-LOCKS: redirect lock-protected paths through
        # TITAN_DATA_DIR when shadow kernel sets it. Original kernel
        # (env unset) sees default `data/...` paths unchanged.
        from titan_hcl.core.shadow_data_dir import resolve_data_path
        for path_field in ("socket_path", "wal_path", "journal_dir",
                           "db_path", "shadow_db_path"):
            d[path_field] = resolve_data_path(d[path_field])
        return cls(**d)

    @classmethod
    def from_titan_config(cls) -> "IMWConfig":
        cfg_path = Path(__file__).resolve().parent.parent / "config.toml"
        if not cfg_path.exists():
            return cls.from_dict(None)
        full = _load_config_toml_cached(cfg_path)
        return cls.from_dict(full.get("persistence"))

    @classmethod
    def from_titan_config_section(cls, section_name: str = "persistence") -> "IMWConfig":
        """Generic loader: load a [persistence] subtable from config.toml.

        Added 2026-04-21 to support multiple writer instances (rFP_observatory_
        writer_service Phase 0). ``section_name`` is a dotted path resolved
        against the merged config (RFP_config_as_shm_state §7-Phase-B(6) Tier-1
        rename consolidated the per-writer DB sections under [persistence.<sub>]):
          - section_name="persistence"               → IMW (inner_memory.db)
          - section_name="persistence.observatory"   → ObservatoryWriter
          - section_name="persistence.social_graph"  → SocialGraphWriter

        The parse is mtime-cached (``_load_config_toml_cached``) — this loader
        is called per-route in worker hot paths, so re-parsing config.toml on
        every call was a measurable chat-path CPU cost (PROFILING.md F7).
        """
        cfg_path = Path(__file__).resolve().parent.parent / "config.toml"
        if not cfg_path.exists():
            return cls.from_dict(None)
        full = _load_config_toml_cached(cfg_path)
        node = full
        for _part in section_name.split("."):
            node = (node or {}).get(_part, {})
        return cls.from_dict(node or None)

    def is_table_canonical(self, table: str) -> bool:
        return table in self.tables_canonical

    def is_shm_table(self, table: str) -> bool:
        return table in self.shm_tables

    def ensure_runtime_dirs(self) -> None:
        Path(self.journal_dir).mkdir(parents=True, exist_ok=True)
        Path(self.socket_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.wal_path).parent.mkdir(parents=True, exist_ok=True)
