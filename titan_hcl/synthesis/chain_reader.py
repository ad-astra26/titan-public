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

from typing import Optional

from titan_hcl.logic.timechain import (
    CROSS_REF_SIZE,
    FORK_NAMES,
    HEADER_SIZE,
    BlockHeader,
    BlockPayload,
)


def chain_file_for(data_dir: Path, fork_id: int) -> Path:
    """Resolve the on-disk `.bin` path for a fork_id (mirrors
    `TimeChain._get_chain_file_path`): a named primary fork lives at
    `<data_dir>/timechain/chain_<name>.bin`; anything else is a sidechain at
    `<data_dir>/timechain/sidechains/sc_<fork_id:04d>.bin`. Used to dereference a
    block_index row's `(fork_id, file_offset)` without opening a writable
    TimeChain.

    NB: `data_dir` is the Titan data root (e.g. `data/`); the chain `.bin` files
    + `index.db` live in the `timechain/` subdir under it (the TimeChain is
    constructed with `data_dir=<root>/timechain`)."""
    tc = Path(data_dir) / "timechain"
    name = FORK_NAMES.get(int(fork_id))
    if name:
        return tc / f"chain_{name}.bin"
    return tc / "sidechains" / f"sc_{int(fork_id):04d}.bin"


def read_block_content_at(
    data_dir: Path, fork_id: int, offset: int,
) -> Optional[dict]:
    """Read ONE block's content dict from `chain_<fork>.bin` (or the sidechain
    file) at `offset` — the byte position stored in `block_index.file_offset`.
    Read-only; returns None on a missing file / short read / parse error (the
    caller skips that TX rather than failing the whole pass)."""
    path = chain_file_for(data_dir, fork_id)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            f.seek(int(offset))
            header_data = f.read(HEADER_SIZE)
            if len(header_data) < HEADER_SIZE:
                return None
            header = BlockHeader.from_bytes(header_data)
            f.read(header.cross_ref_count * CROSS_REF_SIZE)
            len_data = f.read(4)
            if len(len_data) < 4:
                return None
            payload_len = struct.unpack(">I", len_data)[0]
            payload_data = f.read(payload_len)
            if len(payload_data) < payload_len:
                return None
            payload = BlockPayload.from_bytes(payload_data)
            return payload.content if isinstance(payload.content, dict) else {}
    except Exception:
        return None


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
