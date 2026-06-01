"""Read-only chain_*.bin block reader (Synthesis Engine Phase 0 tooling).

Parses the on-disk block format directly so diagnostics (CAS audit, migration
dry-run) can inspect payloads WITHOUT opening a writable TimeChain on live data.
Mirrors the layout `TimeChain.verify_fork` walks: header → cross-refs → 4-byte
payload length → payload bytes. Tolerant of a truncated/garbled tail (stops),
since integrity verification is `timechain.verify_fork`'s job, not this reader's.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator, Tuple

from titan_hcl.logic.timechain import (
    CROSS_REF_SIZE,
    HEADER_SIZE,
    BlockHeader,
    BlockPayload,
)


def iter_block_contents(chain_path: Path) -> Iterator[Tuple[int, str, str, dict]]:
    """Yield (block_height, thought_type, source, content) for each block.

    Read-only. Skips a block whose payload won't parse; stops at a truncated tail.
    """
    with open(chain_path, "rb") as f:
        while True:
            header_data = f.read(HEADER_SIZE)
            if len(header_data) < HEADER_SIZE:
                return
            header = BlockHeader.from_bytes(header_data)
            f.read(header.cross_ref_count * CROSS_REF_SIZE)
            len_data = f.read(4)
            if len(len_data) < 4:
                return
            payload_len = struct.unpack(">I", len_data)[0]
            payload_data = f.read(payload_len)
            if len(payload_data) < payload_len:
                return
            try:
                payload = BlockPayload.from_bytes(payload_data)
            except Exception:
                continue
            content = payload.content if isinstance(payload.content, dict) else {}
            yield header.block_height, payload.thought_type, payload.source, content
