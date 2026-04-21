"""Inner Memory Writer Service (IMW) — single-writer daemon for data/inner_memory.db.

Public API:
    from titan_plugin.persistence import get_client, WriteResult

    client = get_client()
    result = client.write("INSERT INTO ... VALUES (?, ?)", (a, b))
    assert result.ok

See titan-docs/PLAN_inner_memory_writer_service.md for design.
"""

from .writer_client import (
    InnerMemoryWriterClient,
    WriteResult,
    WriterDisabledError,
    WriterError,
    get_client,
    reset_client,
)

__all__ = [
    "InnerMemoryWriterClient",
    "WriteResult",
    "WriterDisabledError",
    "WriterError",
    "get_client",
    "reset_client",
]
