"""IMW configuration — loaded from titan_plugin/config.toml [persistence] section."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULTS = {
    "enabled": False,               # master switch; False = direct writes (pre-IMW behavior)
    "mode": "disabled",             # "shadow" | "dual" | "canonical" | "disabled"
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
        return cls(**d)

    @classmethod
    def from_titan_config(cls) -> "IMWConfig":
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        cfg_path = Path(__file__).resolve().parent.parent / "config.toml"
        if not cfg_path.exists():
            return cls.from_dict(None)
        with open(cfg_path, "rb") as f:
            full = tomllib.load(f)
        return cls.from_dict(full.get("persistence"))

    @classmethod
    def from_titan_config_section(cls, section_name: str = "persistence") -> "IMWConfig":
        """Generic loader: load any [persistence_*] section from config.toml.

        Added 2026-04-21 to support multiple writer instances (rFP_observatory_
        writer_service Phase 0). For example:
          - section_name="persistence"            → IMW (inner_memory.db)
          - section_name="persistence_observatory" → ObservatoryWriter
        """
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        cfg_path = Path(__file__).resolve().parent.parent / "config.toml"
        if not cfg_path.exists():
            return cls.from_dict(None)
        with open(cfg_path, "rb") as f:
            full = tomllib.load(f)
        return cls.from_dict(full.get(section_name))

    def is_table_canonical(self, table: str) -> bool:
        return table in self.tables_canonical

    def is_shm_table(self, table: str) -> bool:
        return table in self.shm_tables

    def ensure_runtime_dirs(self) -> None:
        Path(self.journal_dir).mkdir(parents=True, exist_ok=True)
        Path(self.socket_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.wal_path).parent.mkdir(parents=True, exist_ok=True)
