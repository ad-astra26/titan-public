"""Phase 5 chunk 5E — persistent Irys SDK helper client.

rFP_phase_c_enhancements.md §3B.2: keep the Node.js Irys instance loaded
across uploads instead of cold-spawning `scripts/irys_upload.js` per call.

Pre-5E pattern: one `node scripts/irys_upload.js …` subprocess per
operation. Each spawn pays ~1.5-2 s of @irys/sdk loading + Irys client
construction + balance handshake, even for cheap ops like `getPrice`.

5E pattern: spawn ONE long-running `scripts/irys_upload_daemon.js`
process per (keypair, rpc) pair. Reuse across calls. Send `shutdown` on
clean exit; respawn on EOF/death. Communicates via line-delimited JSON
over stdin/stdout. Stderr passes through to the parent log.

Lifecycle is per-process: a module-level `_DAEMONS` dict keyed by
(keypair_path, rpc_url) caches running daemons. An atexit handler sends
`shutdown` to each on Python exit. Callers MUST treat this as an
auxiliary helper — `arweave_store.py` falls back to one-shot
`irys_upload.js` invocations when the daemon path is unavailable
(missing node, missing daemon script, daemon crash exhausting retries).
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Module-level cache: one daemon per (keypair, rpc) pair across the
# Python process. Guarded by a lock to keep concurrent spawn-attempts
# from racing. The keys are tuples; the values are IrysDaemon instances.
_DAEMONS: dict[tuple[str, str], "IrysDaemon"] = {}
_DAEMONS_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False

# Each daemon op has a per-call timeout. Generous defaults — large file
# uploads can take >60s on slow links — but bounded so a hung daemon
# doesn't stall the cascade forever.
DEFAULT_OP_TIMEOUT_S = 180.0
READY_HANDSHAKE_TIMEOUT_S = 30.0

DAEMON_SCRIPT_REL = os.path.join("scripts", "irys_upload_daemon.js")
DEFAULT_NODE_PATH = "/usr/lib/node_modules"


@dataclass
class IrysUploadResult:
    """Structured result from upload_file / upload_data."""
    tx_id: str
    url: str
    size_bytes: int


class IrysDaemonError(RuntimeError):
    """Raised when the daemon returns status=error or dies during a call."""


class IrysDaemon:
    """Wraps one persistent Node.js daemon subprocess.

    Concurrency: one in-flight request at a time per daemon (a lock
    serializes `request`). Callers needing parallel uploads should
    spawn multiple IrysDaemon instances via separate cache keys — but
    in practice the backup cascade is serial.
    """

    def __init__(self, keypair_path: str, rpc_url: str,
                 *, repo_root: Optional[str] = None,
                 node_binary: str = "node",
                 script_rel: str = DAEMON_SCRIPT_REL):
        self._keypair_path = keypair_path
        self._rpc_url = rpc_url or ""
        self._repo_root = repo_root or os.getcwd()
        self._node_binary = node_binary
        self._script_rel = script_rel
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._read_lock = asyncio.Lock()
        self._started_ts: Optional[float] = None
        self._pending: dict[str, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        """Spawn the daemon and wait for its ready handshake.

        Raises IrysDaemonError if the daemon exits before becoming
        ready or fails the handshake. Caller is responsible for retry
        / fall-through to one-shot mode.
        """
        if self.is_alive:
            return
        script_path = os.path.join(self._repo_root, self._script_rel)
        if not os.path.exists(script_path):
            raise IrysDaemonError(f"daemon script missing: {script_path}")
        env = os.environ.copy()
        env.setdefault("NODE_PATH", DEFAULT_NODE_PATH)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._node_binary, script_path,
                self._keypair_path, self._rpc_url,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self._repo_root,
            )
        except FileNotFoundError as e:
            raise IrysDaemonError(f"node binary not found: {e}") from None
        self._started_ts = time.time()
        # Handshake runs BEFORE the reader loop starts so we don't have two
        # coroutines awaiting the same stdout — asyncio.StreamReader allows
        # only one in-flight readline at a time.
        await self._await_ready()
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def _await_ready(self) -> None:
        """Wait for the first {status:"ok", ready:true} line."""
        try:
            line = await asyncio.wait_for(
                self._read_one_response_line(), timeout=READY_HANDSHAKE_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            await self._kill_and_collect()
            raise IrysDaemonError("daemon ready handshake timed out")
        if not line or line.get("status") != "ok" or not line.get("ready"):
            await self._kill_and_collect()
            raise IrysDaemonError(f"daemon ready handshake failed: {line!r}")
        logger.info(
            "[IrysDaemon] ready pid=%s kp=%s rpc=%s",
            line.get("pid"), os.path.basename(self._keypair_path),
            self._rpc_url or "(default mainnet)",
        )

    async def _read_one_response_line(self) -> Optional[dict]:
        """Read exactly one JSON line from stdout (used by ready handshake)."""
        assert self._proc is not None
        line = await self._proc.stdout.readline()
        if not line:
            return None
        try:
            return json.loads(line.decode("utf-8").strip())
        except json.JSONDecodeError:
            return None

    async def _reader_loop(self) -> None:
        """Background task: read JSON lines and resolve pending futures."""
        assert self._proc is not None
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line.decode("utf-8").strip())
                except json.JSONDecodeError:
                    logger.warning("[IrysDaemon] malformed line: %r", line[:200])
                    continue
                rid = obj.get("id")
                if rid is None:
                    continue
                fut = self._pending.pop(rid, None)
                if fut is not None and not fut.done():
                    fut.set_result(obj)
        finally:
            # Daemon dead — fail any pending requests so callers unblock.
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(IrysDaemonError("daemon process exited"))
            self._pending.clear()

    async def request(self, op: str, *,
                      timeout: float = DEFAULT_OP_TIMEOUT_S,
                      **kwargs) -> dict:
        """Send one op and await the matching response.

        Raises IrysDaemonError on status=error responses, daemon death,
        or timeout. Caller decides whether to retry / fall through.
        """
        if not self.is_alive:
            raise IrysDaemonError("daemon not alive")
        rid = uuid.uuid4().hex[:12]
        req = {"id": rid, "op": op, **kwargs}
        line = (json.dumps(req) + "\n").encode("utf-8")
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        try:
            self._proc.stdin.write(line)
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as e:
            self._pending.pop(rid, None)
            raise IrysDaemonError(f"daemon write failed: {e}") from None
        try:
            resp = await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise IrysDaemonError(f"daemon op {op!r} timed out after {timeout}s")
        if resp.get("status") != "ok":
            raise IrysDaemonError(
                f"daemon op {op!r} failed: {resp.get('message', 'unknown')}"
            )
        return resp

    async def upload_file(self, path: str, *,
                          content_type: str = "application/octet-stream",
                          tags: Optional[dict] = None,
                          timeout: float = DEFAULT_OP_TIMEOUT_S
                          ) -> IrysUploadResult:
        resp = await self.request(
            "upload_file", timeout=timeout,
            path=path, content_type=content_type, tags=tags or {},
        )
        return IrysUploadResult(
            tx_id=resp["tx_id"], url=resp["url"],
            size_bytes=int(resp["size"]),
        )

    async def upload_data(self, data: bytes, *,
                          content_type: str = "application/octet-stream",
                          tags: Optional[dict] = None,
                          timeout: float = DEFAULT_OP_TIMEOUT_S
                          ) -> IrysUploadResult:
        data_b64 = base64.b64encode(data).decode("ascii")
        resp = await self.request(
            "upload_data", timeout=timeout,
            data_b64=data_b64, content_type=content_type, tags=tags or {},
        )
        return IrysUploadResult(
            tx_id=resp["tx_id"], url=resp["url"],
            size_bytes=int(resp["size"]),
        )

    async def balance(self) -> tuple[str, str]:
        """Returns (atomic_str, readable_str)."""
        resp = await self.request("balance", timeout=30.0)
        return resp["balance_atomic"], resp["balance_readable"]

    async def fund(self, amount_lamports: int) -> dict:
        return await self.request(
            "fund", timeout=120.0, amount_lamports=int(amount_lamports),
        )

    async def price(self, size_bytes: int) -> int:
        resp = await self.request(
            "price", timeout=30.0, size_bytes=int(size_bytes),
        )
        return int(resp["price_lamports"])

    async def shutdown(self) -> None:
        """Best-effort graceful shutdown — send `shutdown`, then close."""
        if not self.is_alive:
            return
        try:
            await self.request("shutdown", timeout=5.0)
        except IrysDaemonError:
            pass
        await self._kill_and_collect()

    async def _kill_and_collect(self) -> None:
        if self._proc is None:
            return
        if self._proc.returncode is None:
            try:
                self._proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None


async def get_daemon(keypair_path: str, rpc_url: str = "",
                     *, repo_root: Optional[str] = None,
                     node_binary: str = "node",
                     script_rel: str = DAEMON_SCRIPT_REL) -> IrysDaemon:
    """Return a cached running daemon for the (keypair, rpc) pair.

    Spawns one if needed. Re-spawns if the cached daemon has died.
    Threadsafe at the cache layer; per-call serialization is the
    daemon's own concern.
    """
    global _ATEXIT_REGISTERED
    key = (os.path.abspath(keypair_path), rpc_url or "")
    with _DAEMONS_LOCK:
        existing = _DAEMONS.get(key)
        if existing is not None and existing.is_alive:
            return existing
        daemon = IrysDaemon(keypair_path, rpc_url, repo_root=repo_root,
                            node_binary=node_binary, script_rel=script_rel)
        _DAEMONS[key] = daemon
        if not _ATEXIT_REGISTERED:
            atexit.register(_atexit_shutdown_all)
            _ATEXIT_REGISTERED = True
    await daemon.start()
    return daemon


def _atexit_shutdown_all() -> None:
    """Synchronous best-effort shutdown of every cached daemon.

    Runs on interpreter exit. Sends SIGTERM via subprocess.kill() because
    we can't safely await async coroutines from atexit (the event loop is
    likely already gone). Loop-driven graceful shutdown should be done by
    explicit callers prior to interpreter teardown.
    """
    for key, daemon in list(_DAEMONS.items()):
        proc = daemon._proc  # noqa: SLF001 — atexit best-effort
        if proc is None or proc.returncode is not None:
            continue
        try:
            proc.terminate()
        except (ProcessLookupError, AttributeError):
            pass
    _DAEMONS.clear()
