"""
worker_bus_bootstrap — env-driven dual-mode helper for Guardian-spawned workers.

Microkernel v2 Phase B.2 §C7. Resolves the "do I use mp.Queue or BusSocketClient?"
question for every worker entry_fn without requiring per-worker code changes
beyond a single helper call.

Worker pattern (every Guardian-spawned worker):

    def my_worker_main(name, recv_q, send_q, config):
        # B.2 bootstrap: resolves dual-mode based on env vars set by kernel
        from titan_plugin.core.worker_bus_bootstrap import setup_worker_bus
        recv_q, send_q, client = setup_worker_bus(name, recv_q, send_q)
        try:
            ...   # main loop, treating recv_q.get / send_q.put_nowait
                  # exactly as before. They behave identically whether
                  # backed by mp.Queue (legacy) or SocketQueue (B.2).
        finally:
            if client is not None:
                client.stop()

When microkernel.bus_ipc_socket_enabled=true on the kernel side, kernel sets
three env vars before workers spawn:

    TITAN_BUS_SOCKET_PATH       — /tmp/titan_bus_<id>.sock
    TITAN_BUS_TITAN_ID          — titan_T1, etc.
    TITAN_BUS_KEYPAIR_PATH      — soul.wallet_path (Solana keypair JSON)

The worker reads the keypair from disk + derives the bus authkey via HKDF
(same code path as the kernel — both ends MUST produce the same digest, by
construction). No authkey crosses the wire. The Solana keypair file is
already chmod 0600 (system-wide secret); same threat model as kernel reading it.

When env vars are absent or invalid, returns (recv_q, send_q, None) — pure
legacy behavior, no socket overhead, no risk of regression.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Env variable names — keep in one place; kernel writes, worker reads
ENV_BUS_SOCKET_PATH = "TITAN_BUS_SOCKET_PATH"
ENV_BUS_TITAN_ID = "TITAN_BUS_TITAN_ID"
ENV_BUS_KEYPAIR_PATH = "TITAN_BUS_KEYPAIR_PATH"


def _try_load_identity_secret(keypair_path: str) -> Optional[bytes]:
    """Read a Solana keypair JSON file and return its full secret bytes.

    Returns None on any failure (file missing, malformed, unreadable). The
    caller treats None as "fall back to legacy mp.Queue mode" rather than
    crashing — workers must boot resiliently even if env wiring is partial.
    """
    try:
        path = Path(keypair_path)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if not isinstance(data, list) or len(data) == 0:
            return None
        return bytes(int(b) & 0xFF for b in data)
    except (OSError, ValueError, TypeError):
        return None


def setup_worker_bus(name: str, recv_q, send_q,
                     *, env=None) -> Tuple[object, object, Optional[object]]:
    """Resolve the worker's bus transport.

    Args:
        name:    Module name (used as subscriber identity on the broker).
        recv_q:  The mp.Queue the worker would have used in legacy mode
                 (Guardian still allocates these; they're harmless if unused).
        send_q:  The mp.Queue for outbound publishes.
        env:     Optional env dict for testing; defaults to os.environ.

    Returns:
        (recv_queue, send_queue, client)
          - recv_queue: SocketQueue if socket mode, else recv_q
          - send_queue: SocketQueue if socket mode, else send_q
          - client:     BusSocketClient if socket mode (caller should
                        client.stop() on shutdown), else None

    Behavior:
      • Socket mode (all env vars set + keypair readable): instantiate
        BusSocketClient, derive authkey via HKDF, start client +
        subscribe with module name. Returns SocketQueue pair (same
        underlying client; both sides of the API surface).
      • Legacy mode (any env var missing or keypair unreadable): returns
        the original mp.Queue handles unchanged. No socket overhead.
    """
    env = env if env is not None else os.environ
    sock_path = env.get(ENV_BUS_SOCKET_PATH)
    titan_id = env.get(ENV_BUS_TITAN_ID)
    keypair_path = env.get(ENV_BUS_KEYPAIR_PATH)
    if not sock_path or not titan_id or not keypair_path:
        return recv_q, send_q, None
    secret = _try_load_identity_secret(keypair_path)
    if secret is None:
        logger.warning(
            "[worker_bus_bootstrap] keypair not readable at %s; "
            "falling back to mp.Queue legacy mode for worker '%s'",
            keypair_path, name)
        return recv_q, send_q, None
    # All systems go for socket mode
    try:
        from titan_plugin.core.bus_authkey import derive_bus_authkey
        from titan_plugin.core.bus_socket import BusSocketClient
        authkey = derive_bus_authkey(secret, titan_id)
        client = BusSocketClient(
            titan_id=titan_id,
            authkey=authkey,
            name=name,
            sock_path=Path(sock_path),
        )
        client.start()
        # The same client backs both the inbound queue (for recv) and the
        # publish path (for send). SocketQueue.put_nowait routes to broker.
        sq = client.inbound_queue()
        logger.info(
            "[worker_bus_bootstrap] worker '%s' attached to bus broker at %s",
            name, sock_path)
        return sq, sq, client
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[worker_bus_bootstrap] socket mode failed for worker '%s': %s; "
            "falling back to mp.Queue legacy mode",
            name, e, exc_info=True)
        return recv_q, send_q, None
