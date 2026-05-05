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
                     *, env=None,
                     topics: Optional[list] = None) -> Tuple[object, object, Optional[object]]:
    """Resolve the worker's bus transport.

    Args:
        name:    Module name (used as subscriber identity on the broker).
        recv_q:  The mp.Queue the worker would have used in legacy mode
                 (Guardian still allocates these; they're harmless if unused).
        send_q:  The mp.Queue for outbound publishes.
        env:     Optional env dict for testing; defaults to os.environ.
        topics:  Per-worker broadcast filter list (e.g. ["MEDITATION_COMPLETE",
                 "BACKUP_TRIGGER_MANUAL"]). When non-empty + bus_ipc_socket
                 mode: broker filters `dst="all"` broadcasts at publish time
                 so only `type ∈ topics` reaches this subscriber. None / empty
                 list = legacy "subscribe-all" (every broadcast delivered).
                 Closes per-subscriber flood class identified 2026-04-30.

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
        # Phase B.2 §D12 audit (2026-05-02) — make the legacy-fallback path
        # LOUD. Silent fallback masks regressions: if env wiring breaks (e.g.
        # kernel forgets to set vars), worker silently runs in legacy mp.Queue
        # mode while the rest of the fleet expects socket mode → hard-to-debug
        # split-brain. Log WARNING with which env var(s) are missing so the
        # field operator can see it immediately.
        missing = []
        if not sock_path:
            missing.append(ENV_BUS_SOCKET_PATH)
        if not titan_id:
            missing.append(ENV_BUS_TITAN_ID)
        if not keypair_path:
            missing.append(ENV_BUS_KEYPAIR_PATH)
        logger.warning(
            "[worker_bus_bootstrap] worker '%s' falling back to mp.Queue "
            "LEGACY MODE — missing env var(s): %s. Under microkernel v2 "
            "with bus_ipc_socket_enabled=true, this means the worker is "
            "off-contract: kernel-side dst='%s' messages will deliver via "
            "in-process Queue (only works if forked from same parent), and "
            "worker publishes go to mp.Queue (drained by Guardian.drain_send_queues, "
            "which is being retired). Fix: ensure kernel sets all 3 env vars "
            "before spawning this worker.",
            name, ",".join(missing), name)
        # B.3 Stage 1 (2026-05-02): if Guardian passed None for the queues
        # (because broker is attached and fork-at-locked-mp.Queue avoidance
        # is active), there's nothing to fall back to — the worker can't
        # function without queues. Raise so Guardian sees the crash and
        # restarts (with backoff) rather than the worker silently
        # AttributeError-ing on `None.get(...)` deep in entry_fn.
        if recv_q is None or send_q is None:
            raise RuntimeError(
                f"setup_worker_bus for '{name}' must succeed in socket mode — "
                f"Guardian set queues to None (broker attached) but socket "
                f"setup failed (see prior WARNING/ERROR for cause). Worker "
                f"cannot run without queues."
            )
        return recv_q, send_q, None
    secret = _try_load_identity_secret(keypair_path)
    if secret is None:
        # Same loud-fallback rationale as above — but for the keypair branch.
        # Stat counter equivalent in DivineBus.stats is
        # `non_kernel_internal_subscribe_under_socket` (a different signal —
        # this branch is "env vars set but keypair unreadable"). Log captures
        # the path so the field operator can chmod / restore it.
        logger.warning(
            "[worker_bus_bootstrap] worker '%s' falling back to mp.Queue "
            "LEGACY MODE — keypair at '%s' not readable (file missing, "
            "malformed, or chmod 0600 violated). Worker will be off-contract "
            "until keypair is restored AND kernel restarts.",
            name, keypair_path)
        # B.3 Stage 1 (2026-05-02): if Guardian passed None for the queues
        # (because broker is attached and fork-at-locked-mp.Queue avoidance
        # is active), there's nothing to fall back to — the worker can't
        # function without queues. Raise so Guardian sees the crash and
        # restarts (with backoff) rather than the worker silently
        # AttributeError-ing on `None.get(...)` deep in entry_fn.
        if recv_q is None or send_q is None:
            raise RuntimeError(
                f"setup_worker_bus for '{name}' must succeed in socket mode — "
                f"Guardian set queues to None (broker attached) but socket "
                f"setup failed (see prior WARNING/ERROR for cause). Worker "
                f"cannot run without queues."
            )
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
            topics=list(topics) if topics else None,
        )
        client.start()
        # Phase B.2 IPC §D8 — BusSocketClient.publish buffers outbound
        # messages while the connection thread is still establishing the
        # initial socket (~50-150ms jitter + handshake). No wait here:
        # workers boot at full speed; the buffer flushes on connect. Same
        # mechanism survives kernel-swap reconnects (the original §D8
        # intent). Closes BUG-BUS-IPC-WORKER-READY-RACE-20260430 without
        # adding boot latency. See bus_socket.py:_outbound_buffer.
        # The same client backs both the inbound queue (for recv) and the
        # publish path (for send). SocketQueue.put_nowait routes to broker.
        sq = client.inbound_queue()
        if topics:
            logger.info(
                "[worker_bus_bootstrap] worker '%s' attached to bus broker at %s "
                "(topics=%s)", name, sock_path, list(topics))
        else:
            logger.info(
                "[worker_bus_bootstrap] worker '%s' attached to bus broker at %s "
                "(subscribe-all — no topic filter)", name, sock_path)
        return sq, sq, client
    except Exception as e:  # noqa: BLE001
        # 2026-05-02 — make this LOUD too (was already exc_info=True but the
        # log level was the same as success-case INFOs nearby). Worker is
        # off-contract; field operator must see this immediately. If we ever
        # graduate to "no fallback allowed", this branch becomes a re-raise.
        logger.error(
            "[worker_bus_bootstrap] worker '%s' SOCKET MODE INIT FAILED: %s — "
            "falling back to mp.Queue LEGACY MODE. Worker is off-contract; "
            "rest of fleet expects socket mode. Investigate the exception "
            "trace below + restore socket mode for this worker.",
            name, e, exc_info=True)
        # B.3 Stage 1 (2026-05-02): if Guardian passed None for the queues
        # (because broker is attached and fork-at-locked-mp.Queue avoidance
        # is active), there's nothing to fall back to — the worker can't
        # function without queues. Raise so Guardian sees the crash and
        # restarts (with backoff) rather than the worker silently
        # AttributeError-ing on `None.get(...)` deep in entry_fn.
        if recv_q is None or send_q is None:
            raise RuntimeError(
                f"setup_worker_bus for '{name}' must succeed in socket mode — "
                f"Guardian set queues to None (broker attached) but socket "
                f"setup failed (see prior WARNING/ERROR for cause). Worker "
                f"cannot run without queues."
            )
        return recv_q, send_q, None
