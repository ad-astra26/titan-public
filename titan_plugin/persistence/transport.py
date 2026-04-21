"""IMW transport abstraction.

Today:    UnixSocketTransport — asyncio unix domain socket, local-only.
Phase A:  BusTransport — routes over DivineBus / TitanBus (stubbed here).
Phase C:  Either stays unix socket or migrates to Rust L0 ring buffer.

Callers use InnerMemoryWriterClient which wraps a transport; they never
construct a transport directly.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from .wire_format import _LEN_HEADER, MAX_FRAME_BYTES, WireFormatError

logger = logging.getLogger("titan.imw.transport")


@runtime_checkable
class IWriterTransport(Protocol):
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def send_frame(self, payload: bytes) -> None: ...
    async def recv_frame(self) -> bytes: ...
    def is_connected(self) -> bool: ...


class TransportError(RuntimeError):
    """Raised on transport-level failures (connect, broken pipe, etc.)."""


class UnixSocketTransport:
    """Length-prefixed msgpack frames over a unix domain socket."""

    def __init__(self, socket_path: str, connect_timeout: float = 5.0) -> None:
        self._socket_path = socket_path
        self._connect_timeout = connect_timeout
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._write_lock = asyncio.Lock()
        self._read_lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._writer is not None and not self._writer.is_closing():
            return
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self._socket_path),
                timeout=self._connect_timeout,
            )
        except (OSError, asyncio.TimeoutError) as e:
            raise TransportError(f"connect failed: {e}") from e

    async def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None

    def is_connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def send_frame(self, payload: bytes) -> None:
        if len(payload) > MAX_FRAME_BYTES:
            raise WireFormatError(f"payload too large: {len(payload)}")
        if self._writer is None:
            raise TransportError("not connected")
        frame = _LEN_HEADER.pack(len(payload)) + payload
        async with self._write_lock:
            try:
                self._writer.write(frame)
                await self._writer.drain()
            except (ConnectionError, OSError) as e:
                raise TransportError(f"send failed: {e}") from e

    async def recv_frame(self) -> bytes:
        if self._reader is None:
            raise TransportError("not connected")
        async with self._read_lock:
            try:
                head = await self._reader.readexactly(4)
            except asyncio.IncompleteReadError as e:
                raise TransportError("connection closed") from e
            (n,) = _LEN_HEADER.unpack(head)
            if n <= 0 or n > MAX_FRAME_BYTES:
                raise WireFormatError(f"bad frame length: {n}")
            try:
                body = await self._reader.readexactly(n)
            except asyncio.IncompleteReadError as e:
                raise TransportError("connection closed mid-frame") from e
            return body


class BusTransport:
    """Placeholder for microkernel Phase A — routes over DivineBus/TitanBus.

    Not implemented in this phase; present so config parsing can reference it
    and the transport abstraction is proven. Switching the client to bus-based
    transport is expected to be a config flip + this class fleshed out.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "BusTransport is a Phase A placeholder; use UnixSocketTransport for now"
        )

    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def send_frame(self, payload: bytes) -> None: ...
    async def recv_frame(self) -> bytes: ...
    def is_connected(self) -> bool: return False


def make_transport(kind: str, **kwargs) -> IWriterTransport:
    if kind == "unix_socket":
        return UnixSocketTransport(**kwargs)
    if kind == "bus":
        return BusTransport(**kwargs)
    raise ValueError(f"unknown transport kind: {kind!r}")
