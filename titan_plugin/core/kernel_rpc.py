"""
Kernel RPC — Unix-socket transport for cross-process plugin attribute access.

Microkernel v2 Phase A §A.4 (S5) — when the Observatory FastAPI app runs as
a Guardian-supervised L3 subprocess, it accesses kernel-owned objects
(plugin.guardian, plugin.soul, plugin._full_config, etc.) through this RPC
layer. The wire protocol is intentionally lean and Phase-C-portable: a Rust
kernel server in Phase C reimplements the same Unix-socket + length-prefixed
msgpack + HMAC-challenge protocol with `tokio` + `rmp-serde` + zero changes
to the Python API client.

Protocol (per PLAN_microkernel_phase_a_s5.md §2.2):

  Connection lifecycle:
    1. Client connects to /tmp/titan_kernel_{titan_id}.sock (mode 0600).
    2. Server sends 32-byte challenge (random per-connection).
    3. Client sends HMAC-SHA256(authkey, challenge) (32 bytes).
    4. Server verifies HMAC matches; on mismatch, closes connection.
    5. Connection enters request/response loop.

  Request frame (client → server):
    [4 bytes: little-endian uint32 length]
    [N bytes: msgpack(["call", method_path, args, kwargs])]

  Response frame (server → client):
    [4 bytes: little-endian uint32 length]
    [N bytes: msgpack(["ok", result]) or msgpack(["err", type_name, message])]

  method_path is a dotted string like "guardian.get_status" — server resolves
  by walking attributes from self._plugin_ref. Only paths present in
  exposed_methods are callable; others return ["err", "MethodNotExposed", "..."].

Resource benchmarks (vs multiprocessing.managers.BaseManager, the rejected
stdlib alternative): +0.3MB RSS per process (vs +7.1MB), 1117μs per call (vs
1983μs), 895 calls/sec (vs 504). msgpack vs pickle: ~30-50% smaller payloads,
~1.8× faster serialize. Phase C portable. See PLAN §1 for full benchmarks +
rationale.

Single-client architecture: the API subprocess is the only connection. Server
spawns one worker thread on accept; multi-client support is intentionally
not implemented (would add overhead without use case).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import secrets
import socket
import stat
import threading
from pathlib import Path
from typing import Any, Callable

import msgpack

from titan_plugin.core._frame import (
    AUTH_TAG_SIZE,
    CHALLENGE_SIZE,
    LENGTH_PREFIX_SIZE,
    MAX_FRAME_SIZE,
    compute_hmac as _compute_hmac,
    constant_time_eq,
    recv_exact as _recv_exact,
    recv_frame as _recv_frame,
    send_frame as _send_frame,
)

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────
# Wire-protocol constants live in _frame.py (shared with bus_socket).
# kernel_rpc-specific constants:

CONNECT_TIMEOUT_S = 30.0    # client connect retry window


# ── Path resolution ────────────────────────────────────────────────


def kernel_sock_path(titan_id: str) -> Path:
    """Per-titan Unix socket path. Matches per-titan shm convention.

    Critical for T2/T3 shared VPS — without per-titan suffix, both Titans'
    kernels would attempt to bind the same socket and conflict.
    """
    return Path(f"/tmp/titan_kernel_{titan_id}.sock")


def kernel_authkey_path(titan_id: str) -> Path:
    """Per-titan authkey file path (mode 0600)."""
    return Path(f"/tmp/titan_kernel_{titan_id}.authkey")


def generate_authkey() -> bytes:
    """Per-boot 32-byte authkey via secrets.token_bytes.

    Ephemeral by design — kernel restart issues a fresh key, invalidating
    any cached client connection state. Phase B (shadow-core swap) will
    need persistent authkey rotation; that's out of scope for Phase A.
    """
    return secrets.token_bytes(32)


# ── Wire-protocol helpers ──────────────────────────────────────────
# Framing primitives (_send_frame, _recv_frame, _recv_exact, _compute_hmac)
# moved to titan_plugin/core/_frame.py in Phase B.2 (commit C1) so both
# kernel_rpc (RPC) and bus_socket (pub/sub) share the same wire encoding.
# Aliases above re-export them under the legacy names used here.


def _resolve_method(plugin_ref: Any, method_path: str) -> Callable | None:
    """Walk dotted path against plugin_ref; return callable or None.

    Example: "guardian.get_status" → plugin_ref.guardian.get_status

    Falls through to dict subscript when getattr fails — required for chained
    proxy paths like "_proxies.spirit.get_coordinator" where `_proxies` is a
    dict (not an object) and `spirit` is a key. Without this, RPC would never
    reach the bus-backed proxy methods.
    """
    obj = plugin_ref
    for part in method_path.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return None
    if not callable(obj):
        # Some endpoints read attribute values directly (no call); we still
        # return the value as a "callable that takes no args" — wrapped here.
        return lambda: obj  # noqa: E731
    return obj


# Known proxy class name suffix — used by _dispatch to detect proxy returns
# from `_proxies.get(name)` and convert to chainable remote ref instead of
# trying to serialize the proxy object (which contains bus + guardian refs).
_PROXY_CLASS_SUFFIX = "Proxy"


def _is_proxy_object(obj: Any) -> bool:
    """True if obj's class name ends with 'Proxy' — heuristic that matches
    SpiritProxy, BodyProxy, MindProxy, MemoryProxy, RLProxy, LLMProxy,
    MediaProxy, TimechainProxy. Avoids hardcoding the full proxy class list.
    """
    return type(obj).__name__.endswith(_PROXY_CLASS_SUFFIX)


def _proxy_name_for(plugin_ref: Any, obj: Any) -> str | None:
    """Find which key in plugin._proxies maps to obj. Returns None if obj is
    not in plugin._proxies (e.g., it's a proxy from somewhere else).
    """
    proxies = getattr(plugin_ref, "_proxies", None)
    if not isinstance(proxies, dict):
        return None
    for k, v in proxies.items():
        if v is obj:
            return k
    return None


# ── Server ──────────────────────────────────────────────────────────


class KernelRPCServer:
    """Listens on Unix socket; serves transparent RPC against plugin_ref.

    Single accept loop in a daemon thread; one worker thread per connection.
    Single-client expected (the API subprocess); concurrent connections are
    handled but not optimized.
    """

    def __init__(
        self,
        plugin_ref: Any,
        titan_id: str,
        exposed_methods: frozenset[str],
        authkey: bytes | None = None,
        kernel_loop: "asyncio.AbstractEventLoop | None" = None,
    ):
        self._plugin_ref = plugin_ref
        self._titan_id = titan_id
        self._exposed_methods = exposed_methods
        self._authkey = authkey or generate_authkey()
        self.sock_path = kernel_sock_path(titan_id)
        self.authkey_path = kernel_authkey_path(titan_id)
        # Optional reference to the kernel's asyncio loop. When set, async
        # RPC method results are awaited on the kernel loop via
        # run_coroutine_threadsafe instead of asyncio.run() (which would
        # spin up a fresh loop and break loop-bound resources).
        self._kernel_loop = kernel_loop

        self._sock: socket.socket | None = None
        self._stop_evt = threading.Event()
        self._accept_thread: threading.Thread | None = None

        self._setup_socket_and_authkey()

    def _setup_socket_and_authkey(self) -> None:
        """Bind socket + write authkey file, both with 0600 permissions."""
        # Remove stale socket file
        if self.sock_path.exists():
            self.sock_path.unlink()
        # Bind first, then chmod 0600 (defense-in-depth alongside HMAC).
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(self.sock_path))
        os.chmod(self.sock_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        self._sock.listen(8)
        # Write authkey file with 0600 perms BEFORE returning so the
        # client can read it as soon as the socket is up.
        self.authkey_path.write_bytes(self._authkey)
        os.chmod(self.authkey_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600

    def serve_forever(self) -> None:
        """Accept loop — blocking; spawn in a daemon thread."""
        self._sock.settimeout(1.0)  # cooperative shutdown via stop_evt
        logger.info(
            "[KernelRPC] Listening on %s (authkey at %s, %d exposed methods)",
            self.sock_path, self.authkey_path, len(self._exposed_methods))
        while not self._stop_evt.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError as e:
                if self._stop_evt.is_set():
                    break
                logger.warning("[KernelRPC] accept failed: %s", e)
                continue
            t = threading.Thread(
                target=self._handle_client,
                args=(conn,),
                daemon=True,
                name="kernel-rpc-client",
            )
            t.start()

    def _handle_client(self, conn: socket.socket) -> None:
        """Per-connection: HMAC handshake, then request/response loop."""
        try:
            # 1. Send challenge
            challenge = secrets.token_bytes(CHALLENGE_SIZE)
            conn.sendall(challenge)
            # 2. Receive client HMAC response
            client_hmac = _recv_exact(conn, AUTH_TAG_SIZE)
            expected = _compute_hmac(self._authkey, challenge)
            if not constant_time_eq(client_hmac, expected):
                logger.warning("[KernelRPC] auth failed — closing connection")
                conn.close()
                return
            # 3. Request loop
            while not self._stop_evt.is_set():
                try:
                    frame = _recv_frame(conn)
                except (ConnectionError, OSError):
                    break
                self._dispatch(conn, frame)
        except Exception as e:
            logger.warning("[KernelRPC] client handler error: %s", e)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _is_path_allowed(self, method_path: str) -> bool:
        """Method-path access policy.

        Allows:
          - exact match against EXPOSED_METHODS frozenset (canonical paths)
          - chained proxy access: `_proxies.<known_name>.<anything>` where
            <known_name> is a key currently in plugin._proxies. This lets
            endpoint code do `plugin._proxies.get("spirit").get_state()` —
            the first call returns a remote ref (path=`_proxies.spirit`),
            the second fires RPC `_proxies.spirit.get_state` which is then
            dispatched here.
        """
        if method_path in self._exposed_methods:
            return True
        parts = method_path.split(".", 2)
        if len(parts) >= 2 and parts[0] == "_proxies":
            proxies = getattr(self._plugin_ref, "_proxies", None)
            if isinstance(proxies, dict) and parts[1] in proxies:
                return True
        return False

    def _maybe_await(self, result: Any) -> Any:
        """If `result` is a coroutine/awaitable, await it on the kernel's
        event loop and return the actual value. Otherwise return as-is.

        The RPC server thread runs outside the kernel asyncio loop. Naive
        `asyncio.run(result)` would spawn a fresh loop and likely break
        loop-bound resources (DB connections, websocket clients, bus state).
        We use `run_coroutine_threadsafe` to schedule on the kernel loop and
        block until done. Falls back to `asyncio.run` only if no kernel loop
        is configured (single-shot test fixtures).
        """
        if not inspect.iscoroutine(result):
            return result
        if self._kernel_loop is not None and self._kernel_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(result, self._kernel_loop)
            return future.result(timeout=30.0)
        return asyncio.run(result)

    def _dispatch(self, conn: socket.socket, frame: bytes) -> None:
        """Resolve method, execute, marshal response."""
        method_path = "<unknown>"  # for diagnostic logging on serialize-fail
        try:
            req = msgpack.unpackb(frame, raw=False)
            if (not isinstance(req, list) or len(req) != 4
                    or req[0] != "call"):
                response = ["err", "ProtocolError",
                            "Expected ['call', path, args, kwargs]"]
            else:
                _, method_path, args, kwargs = req
                if not self._is_path_allowed(method_path):
                    response = ["err", "MethodNotExposed",
                                f"{method_path!r} is not in EXPOSED_METHODS"]
                else:
                    fn = _resolve_method(self._plugin_ref, method_path)
                    if fn is None:
                        response = ["err", "AttributeError",
                                    f"{method_path!r} not resolvable on plugin"]
                    else:
                        try:
                            result = fn(*(args or []), **(kwargs or {}))
                            # Coroutine handling: many EXPOSED methods are
                            # async (e.g., network.get_balance). Run them on
                            # the kernel loop and ship the actual value.
                            result = self._maybe_await(result)
                            # Proxy-return handling: when result is a bus-backed
                            # Proxy object (SpiritProxy etc.) returned from
                            # `_proxies.get(name)`, ship a chainable remote-ref
                            # marker instead of the unserializable proxy. The
                            # client converts this to a _RPCRemoteRef pointing
                            # at `_proxies.<name>`, and subsequent .method()
                            # calls fire RPCs that hit the proxy on this side.
                            if _is_proxy_object(result):
                                proxy_name = _proxy_name_for(
                                    self._plugin_ref, result)
                                if proxy_name is not None:
                                    response = [
                                        "ok_proxy_ref",
                                        f"_proxies.{proxy_name}",
                                    ]
                                else:
                                    # Proxy not registered in plugin._proxies —
                                    # cannot construct a stable RPC path; let
                                    # the SerializationError path handle it.
                                    response = ["ok", result]
                            else:
                                response = ["ok", result]
                        except Exception as exc:
                            response = ["err", type(exc).__name__, str(exc)]
        except Exception as exc:
            response = ["err", type(exc).__name__, str(exc)]

        try:
            payload = msgpack.packb(response, use_bin_type=True)
            _send_frame(conn, payload)
        except Exception as e:
            # Robustness: when result isn't msgpack-serializable (e.g., method
            # returns a proxy object that wasn't caught above), the prior code
            # logged a warning and silently dropped the response — leaving the
            # client blocked on recv forever. Now we send back a structured
            # error so the client gets a RuntimeError it can handle.
            logger.warning(
                "[KernelRPC] response serialization failed for %r: %s "
                "(sending SerializationError to client)",
                method_path, e)
            err_response = [
                "err", "SerializationError",
                f"Result of {method_path!r} not msgpack-serializable: {e}",
            ]
            try:
                err_payload = msgpack.packb(err_response, use_bin_type=True)
                _send_frame(conn, err_payload)
            except Exception as e2:
                # Last resort: close the connection so client gets a peer-reset
                # rather than an indefinite hang.
                logger.error(
                    "[KernelRPC] failed to send error frame for %r: %s — "
                    "closing connection",
                    method_path, e2)
                try:
                    conn.close()
                except Exception:
                    pass

    def stop(self) -> None:
        """Signal accept loop to exit; close socket; remove socket+authkey files."""
        self._stop_evt.set()
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        for p in (self.sock_path, self.authkey_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        logger.info("[KernelRPC] stopped — socket + authkey unlinked")


# ── Client ──────────────────────────────────────────────────────────


class KernelRPCClient:
    """Connects to KernelRPCServer; performs HMAC handshake; serves proxy.

    Single connection; serializes calls via a lock for thread safety
    (the API subprocess may have multiple uvicorn worker threads issuing
    plugin attr access simultaneously, but the wire protocol is request/
    response — we serialize at the client side).
    """

    def __init__(self, titan_id: str, connect_timeout_s: float = CONNECT_TIMEOUT_S):
        self._titan_id = titan_id
        self._connect_timeout_s = connect_timeout_s
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Block until kernel socket is ready, then perform HMAC handshake.

        Retries up to connect_timeout_s for kernel boot completion.
        """
        import time

        sock_path = kernel_sock_path(self._titan_id)
        authkey_path = kernel_authkey_path(self._titan_id)

        deadline = time.time() + self._connect_timeout_s
        while time.time() < deadline:
            if sock_path.exists() and authkey_path.exists():
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(
                f"Kernel RPC socket not ready after {self._connect_timeout_s}s: {sock_path}"
            )

        authkey = authkey_path.read_bytes()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(str(sock_path))

        # HMAC handshake
        challenge = _recv_exact(self._sock, CHALLENGE_SIZE)
        response = _compute_hmac(authkey, challenge)
        self._sock.sendall(response)
        # No explicit ack — first successful request implies auth passed.
        logger.info("[KernelRPCClient] connected + handshaked at %s", sock_path)

    def call(self, method_path: str, args: list, kwargs: dict) -> Any:
        """Issue one RPC call. Raises a re-created exception on server-side error."""
        if self._sock is None:
            raise RuntimeError("KernelRPCClient not connected — call connect() first")
        with self._lock:
            request = ["call", method_path, args, kwargs]
            payload = msgpack.packb(request, use_bin_type=True)
            _send_frame(self._sock, payload)
            response_frame = _recv_frame(self._sock)
        response = msgpack.unpackb(response_frame, raw=False)
        if not isinstance(response, list) or len(response) < 2:
            raise RuntimeError(f"Malformed RPC response: {response!r}")
        if response[0] == "ok":
            return response[1]
        if response[0] == "ok_proxy_ref":
            # Server detected a Proxy return (e.g. SpiritProxy from
            # `_proxies.get("spirit")`). It can't ship the proxy itself, so
            # it returned a path string we can use to construct a chainable
            # remote ref. Future attribute access + calls on this ref fire
            # RPCs at `_proxies.<name>.<method>` paths.
            ref_path = response[1]
            if not isinstance(ref_path, str) or not ref_path:
                raise RuntimeError(
                    f"Malformed ok_proxy_ref response: {response!r}")
            return _RPCRemoteRef(self, tuple(ref_path.split(".")))
        # ["err", type_name, message]
        type_name = response[1] if len(response) > 1 else "RuntimeError"
        message = response[2] if len(response) > 2 else "(no detail)"
        # Re-create as a RuntimeError with the original type name embedded.
        # We don't try to recreate the exact exception class — type may live
        # in code the API process doesn't import. RuntimeError + msg covers
        # the canonical "kernel call failed" case.
        raise RuntimeError(f"[{type_name}] {message}")

    def get_plugin_proxy(self) -> "_RPCRemoteRef":
        """Return the transparent proxy. Endpoint code uses it like a normal
        plugin reference; attribute access + method calls route over RPC."""
        return _RPCRemoteRef(self, ())

    def close(self) -> None:
        try:
            if self._sock:
                self._sock.close()
                self._sock = None
        except Exception:
            pass


class _RPCRemoteRef:
    """Transparent client proxy.

    Endpoint code reads `plugin.guardian.get_status()` and the proxy
    chains the attribute path (no RPC during chaining), then issues ONE
    RPC call when the chain resolves to `__call__`.

    Attribute reads that don't end in a call (e.g. `plugin._full_config`)
    are translated by the kernel side into "callable that returns the
    value" via _resolve_method's lambda wrapper. The client invokes
    the proxy as `plugin._full_config()` instead of `plugin._full_config`.

    For the legacy attribute-read pattern, callers use:
      cfg = plugin._full_config        # was: dict
      cfg = plugin._full_config()      # NEW: triggers RPC, returns dict

    To keep endpoint code unchanged, we make __getattr__ also accessible
    as a property-like read via a special sentinel — but simpler is to
    require endpoints that read attributes directly to call them. The
    API endpoint migration is documented separately.
    """

    __slots__ = ("_client", "_path")

    def __init__(self, client: KernelRPCClient, path: tuple[str, ...]):
        self._client = client
        self._path = path

    def __getattr__(self, name: str) -> "_RPCRemoteRef":
        if name.startswith("_RPCRemoteRef__") or name in ("_client", "_path"):
            return object.__getattribute__(self, name)
        return _RPCRemoteRef(self._client, self._path + (name,))

    def __getitem__(self, key: Any) -> "_RPCRemoteRef":
        # Subscript support — endpoint code like
        # `plugin.network.rpc_urls[0]` chains a __getitem__ that we treat
        # the same as attribute access. For string keys we extend the path;
        # for non-string keys we still extend with str(key) since the
        # server-side _resolve_method falls through to dict subscript.
        return _RPCRemoteRef(self._client, self._path + (str(key),))

    def __bool__(self) -> bool:
        # Endpoint code does `if not spirit_proxy:` to skip when proxy is
        # missing. Always-True keeps the existing pattern working when the
        # server-side proxy genuinely exists; when it doesn't, the server
        # returns None (not a remote ref), so this dunder isn't reached.
        return True

    def __call__(self, *args, **kwargs) -> Any:
        method_path = ".".join(self._path)
        return self._client.call(method_path, list(args), dict(kwargs))

    def __repr__(self) -> str:
        return f"<_RPCRemoteRef path={'.'.join(self._path) or '<root>'}>"
