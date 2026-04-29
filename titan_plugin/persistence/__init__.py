"""SQLite single-writer service — generic per-DB write-contention pattern.

Originally shipped as Inner Memory Writer (IMW) for `data/inner_memory.db`,
the engine is path-agnostic and supports any SQLite DB via a per-DB config
section. See `titan-docs/rFP_universal_sqlite_writer.md` for the adoption
recipe (config section, ModuleSpec, client wiring, table-by-table cutover).

Public API:
    from titan_plugin.persistence import SqliteWriterClient, IMWConfig

    cfg = IMWConfig.from_titan_config_section("persistence_<dbname>")
    client = SqliteWriterClient(cfg, caller_name="<dbname>")
    result = client.write("INSERT INTO ... VALUES (?, ?)", (a, b),
                          table="<tablename>")
    assert result.ok

Backwards-compat: `InnerMemoryWriterClient` remains as an alias to
`SqliteWriterClient` so all existing call sites keep working.

See `titan-docs/finished/PLAN_inner_memory_writer_service.md` for the
original design + `titan-docs/rFP_universal_sqlite_writer.md` for the
generalization (2026-04-27).
"""

from .writer_client import (
    InnerMemoryWriterClient,
    WriteResult,
    WriterDisabledError,
    WriterError,
    get_client,
    reset_client,
)

# Generic alias — preferred name for new call sites.
# Per rFP_universal_sqlite_writer Phase 4 (2026-04-27).
SqliteWriterClient = InnerMemoryWriterClient

__all__ = [
    # Generic names (preferred)
    "SqliteWriterClient",
    # Backwards-compat names
    "InnerMemoryWriterClient",
    "WriteResult",
    "WriterDisabledError",
    "WriterError",
    "get_client",
    "reset_client",
]
