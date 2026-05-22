"""IMW wire format — length-prefixed msgpack frames.

Frame layout:
    [4-byte big-endian length][msgpack payload]

Request payload schema (v1):
    {
        "v": 1,
        "req_id": str,         # uuid4 hex
        "caller": str,         # "<module_name>:<pid>"
        "op": str,             # "write" | "writemany" | "ping" | "flush"
        "sql": str | None,
        "params": list | tuple | None,
        "sync": bool,          # caller awaits commit if True
        "ts": float,           # unix epoch seconds
    }

Response payload schema (v1):
    {
        "v": 1,
        "req_id": str,
        "ok": bool,
        "rowcount": int | None,
        "last_row_id": int | None,
        "error": dict | None,  # {"type": str, "msg": str}
        "committed_at": float | None,
    }
"""
from __future__ import annotations

import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import msgpack

PROTOCOL_VERSION = 1
_LEN_HEADER = struct.Struct(">I")  # 4-byte big-endian unsigned int
MAX_FRAME_BYTES = 16 * 1024 * 1024  # 16 MB safety cap per frame


class WireFormatError(ValueError):
    """Raised when a frame is malformed or oversized."""


@dataclass
class WriteRequest:
    req_id: str
    caller: str
    op: str
    sql: Optional[str] = None
    params: Any = None
    sync: bool = True
    target_db: str = "primary"  # "primary" | "shadow" — service routes by this
    ts: float = field(default_factory=time.time)

    @classmethod
    def new_write(cls, caller: str, sql: str, params: Any, sync: bool = True,
                   target_db: str = "primary") -> "WriteRequest":
        return cls(
            req_id=uuid.uuid4().hex,
            caller=caller,
            op="write",
            sql=sql,
            params=list(params) if params is not None else [],
            sync=sync,
            target_db=target_db,
        )

    @classmethod
    def new_writemany(cls, caller: str, sql: str, rows: list, sync: bool = True,
                       target_db: str = "primary") -> "WriteRequest":
        return cls(
            req_id=uuid.uuid4().hex,
            caller=caller,
            op="writemany",
            sql=sql,
            params=[list(r) for r in rows],
            sync=sync,
            target_db=target_db,
        )

    @classmethod
    def new_ping(cls, caller: str) -> "WriteRequest":
        return cls(req_id=uuid.uuid4().hex, caller=caller, op="ping", sync=True)

    def to_msgpack(self) -> bytes:
        return msgpack.packb({
            "v": PROTOCOL_VERSION,
            "req_id": self.req_id,
            "caller": self.caller,
            "op": self.op,
            "sql": self.sql,
            "params": self.params,
            "sync": self.sync,
            "target_db": self.target_db,
            "ts": self.ts,
        }, use_bin_type=True)

    @classmethod
    def from_msgpack(cls, payload: bytes) -> "WriteRequest":
        obj = msgpack.unpackb(payload, raw=False)
        if not isinstance(obj, dict):
            raise WireFormatError(f"expected dict, got {type(obj).__name__}")
        v = obj.get("v")
        if v != PROTOCOL_VERSION:
            raise WireFormatError(f"unsupported protocol version {v!r}")
        return cls(
            req_id=obj["req_id"],
            caller=obj.get("caller", ""),
            op=obj["op"],
            sql=obj.get("sql"),
            params=obj.get("params"),
            sync=bool(obj.get("sync", True)),
            target_db=str(obj.get("target_db", "primary")),
            ts=float(obj.get("ts", time.time())),
        )


@dataclass
class WriteResponse:
    req_id: str
    ok: bool
    rowcount: Optional[int] = None
    last_row_id: Optional[int] = None
    error: Optional[dict] = None
    committed_at: Optional[float] = None

    def to_msgpack(self) -> bytes:
        return msgpack.packb({
            "v": PROTOCOL_VERSION,
            "req_id": self.req_id,
            "ok": self.ok,
            "rowcount": self.rowcount,
            "last_row_id": self.last_row_id,
            "error": self.error,
            "committed_at": self.committed_at,
        }, use_bin_type=True)

    @classmethod
    def from_msgpack(cls, payload: bytes) -> "WriteResponse":
        obj = msgpack.unpackb(payload, raw=False)
        if not isinstance(obj, dict):
            raise WireFormatError(f"expected dict, got {type(obj).__name__}")
        if obj.get("v") != PROTOCOL_VERSION:
            raise WireFormatError(f"unsupported protocol version {obj.get('v')!r}")
        return cls(
            req_id=obj["req_id"],
            ok=bool(obj["ok"]),
            rowcount=obj.get("rowcount"),
            last_row_id=obj.get("last_row_id"),
            error=obj.get("error"),
            committed_at=obj.get("committed_at"),
        )


def encode_frame(payload: bytes) -> bytes:
    if len(payload) > MAX_FRAME_BYTES:
        raise WireFormatError(f"payload too large: {len(payload)} > {MAX_FRAME_BYTES}")
    return _LEN_HEADER.pack(len(payload)) + payload


def decode_length(header: bytes) -> int:
    if len(header) != 4:
        raise WireFormatError(f"length header must be 4 bytes, got {len(header)}")
    (n,) = _LEN_HEADER.unpack(header)
    if n > MAX_FRAME_BYTES:
        raise WireFormatError(f"frame length {n} exceeds MAX_FRAME_BYTES")
    return n
