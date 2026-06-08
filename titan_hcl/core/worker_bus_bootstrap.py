"""
worker_bus_bootstrap — env-driven dual-mode helper for Guardian-spawned workers.

Microkernel v2 Phase B.2 §C7. Resolves the "do I use mp.Queue or BusSocketClient?"
question for every worker entry_fn without requiring per-worker code changes
beyond a single helper call.

Worker pattern (every Guardian-spawned worker):

    def my_worker_main(name, recv_q, send_q, config):
        # B.2 bootstrap: resolves dual-mode based on env vars set by kernel
        from titan_hcl.core.worker_bus_bootstrap import setup_worker_bus
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
import time
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Env variable names — keep in one place; kernel writes, worker reads
ENV_BUS_SOCKET_PATH = "TITAN_BUS_SOCKET_PATH"
ENV_BUS_TITAN_ID = "TITAN_BUS_TITAN_ID"
ENV_BUS_KEYPAIR_PATH = "TITAN_BUS_KEYPAIR_PATH"


def _try_load_identity_secret(keypair_path: str) -> Optional[bytes]:
    """Read a Solana keypair JSON file and return its 32-byte Ed25519 secret_seed.

    Per `PLAN_microkernel_phase_c_s2_kernel.md §7.3` + `rFP_phase_c_bus_authkey_contract_fix.md`:
    HKDF IKM is the 32-byte Ed25519 secret_seed — NOT the full 64-byte
    Solana keypair. Solana CLI byte-array format is 64 bytes
    (seed[0..32] + pub_key[32..64]); we slice the first 32 to match Rust
    `Identity::load_from_disk_with_hint` which extracts `bytes[0..32]`
    (titan-rust/crates/titan-core/src/identity.rs:327).

    Pre-fix Python returned all 64 bytes → different IKM than Rust →
    different authkeys → handshake mismatch under l0_rust_enabled=true.
    This was the SECOND drift bug surfaced by 2026-05-05 diagnostic
    (the first being titan_id used as HKDF info instead of constant).

    Returns None on any failure (file missing, malformed, unreadable,
    unrecognized length). Caller treats None as "fall back to legacy
    mp.Queue mode" rather than crashing — workers must boot resiliently
    even if env wiring is partial.
    """
    try:
        path = Path(keypair_path)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if not isinstance(data, list) or len(data) == 0:
            return None
        full = bytes(int(b) & 0xFF for b in data)
        # 32-byte raw seed → use as-is. 64-byte Solana keypair → take first 32.
        # Other lengths → reject (not a recognized format).
        if len(full) == 32:
            return full
        if len(full) == 64:
            return full[:32]
        return None
    except (OSError, ValueError, TypeError):
        return None


def setup_worker_bus(name: str, recv_q, send_q,
                     *, env=None,
                     topics: Optional[list] = None,
                     reply_only: bool = False,
                     phase_b_reload_swap_id: Optional[str] = None,
                     ) -> Tuple[object, object, Optional[object]]:
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
        reply_only: SPEC §8.2 v1.4.0 (D-SPEC-42) subscriber-intent flag.
                 True ↔ this worker consumes ONLY targeted dst=<name>
                 messages (RPC replies + control like MODULE_SHUTDOWN),
                 never dst="all" broadcasts. Broker silently skips
                 reply_only subscribers from broadcast fan-out. Sourced
                 from ModuleSpec.reply_only via Guardian._module_wrapper.

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
    # D8-2 (2026-05-16): mp.Queue legacy-fallback branches DELETED.
    # Under fleet-wide Phase C (since 2026-05-14) titan-kernel-rs (Rust)
    # owns the bus broker, sets all 3 env vars before spawning every
    # worker, and the identity keypair is always present (chmod 0600).
    # Missing env vars or unreadable keypair now hard-fail (was: silent
    # fallback to legacy mp.Queue mode that caused split-brain regressions).
    if not sock_path or not titan_id or not keypair_path:
        missing = [
            v for v, present in [
                (ENV_BUS_SOCKET_PATH, sock_path),
                (ENV_BUS_TITAN_ID, titan_id),
                (ENV_BUS_KEYPAIR_PATH, keypair_path),
            ] if not present
        ]
        raise RuntimeError(
            f"setup_worker_bus for '{name}' requires bus-socket env vars "
            f"(missing: {','.join(missing)}). Under fleet-wide Phase C "
            f"the kernel-rs broker is the only supported transport; "
            f"mp.Queue legacy-fallback was retired in D8-2 2026-05-16."
        )
    secret = _try_load_identity_secret(keypair_path)
    if secret is None:
        raise RuntimeError(
            f"setup_worker_bus for '{name}' requires readable identity "
            f"keypair at '{keypair_path}' (file missing, malformed, or "
            f"chmod 0600 violated). mp.Queue legacy-fallback retired in "
            f"D8-2 2026-05-16."
        )
    # All systems go for socket mode
    try:
        from titan_hcl.core.bus_authkey import derive_bus_authkey
        from titan_hcl.core.bus_socket import BusSocketClient
        # Per rFP_phase_c_bus_authkey_contract_fix.md (2026-05-05): authkey
        # derivation no longer takes titan_id (HKDF info is the constant
        # b"titan-bus"). Per-Titan isolation comes from the per-Titan identity
        # secret (different IKM → different authkey). titan_id is still used
        # below as the BusSocketClient identity (subscriber name) — that's
        # orthogonal to authkey derivation.
        authkey = derive_bus_authkey(secret)

        # Phase 11 §11.I.2 (locked D1/D2): the legacy MODULE_READY bus emit is
        # DELETED. Readiness is SHM-only — the worker writes state=booted to its
        # own module_<name>_state.bin slot and titan_hcl drives it to running via
        # the MODULE_PROBE_REQUEST contract.
        #
        # Phase B (RFP_shm_native_hot_reload): the reload ADOPTION_REQUEST emit
        # that used to fire on the first subscribe-ack is RETIRED. The titan_hcl
        # Orchestrator spawned this worker and knows its pid, so it confirms
        # readiness via the pid-validated SHM slot (`_wait_for_reload_running`) —
        # no worker→Guardian adopt round-trip, which mis-routed to the
        # guardian_hcl Supervisor in the peer-spawn split (never reaching the
        # Orchestrator's rs.adoption_q → guaranteed adoption_timeout). The §10.C
        # ADOPTION protocol stays live for B.2.1 kernel-swap (worker_swap_handler
        # — INV-4). `phase_b_reload_swap_id` is kept only as a spawn-identity tag.
        if phase_b_reload_swap_id is not None:
            logger.info(
                "[worker_bus_bootstrap] worker '%s' is a reload-spawn "
                "(swap_id=%s) — readiness via pid-validated SHM slot, "
                "no adopt emit", name, str(phase_b_reload_swap_id)[:8])

        client = BusSocketClient(
            titan_id=titan_id,
            authkey=authkey,
            name=name,
            sock_path=Path(sock_path),
            topics=list(topics) if topics else None,
            reply_only=reply_only,
            on_subscribe_ack=None,
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
        # Boot log mirrors SPEC §8.2 v1.4.0 subscriber-intent declarations.
        # reply_only=True wins the log even with topics — it's the canonical
        # intent for RPC-only / writer-service / proxy-alias workers.
        if reply_only:
            logger.info(
                "[worker_bus_bootstrap] worker '%s' attached to bus broker at %s "
                "(reply_only=True — receives ONLY targeted dst=%s messages, "
                "broadcasts silently skipped per SPEC §8.2 v1.4.0 D-SPEC-42)",
                name, sock_path, name)
        elif topics:
            logger.info(
                "[worker_bus_bootstrap] worker '%s' attached to bus broker at %s "
                "(topics=%s)", name, sock_path, list(topics))
        else:
            # SPEC §8.2 v1.4.0 REGRESSION case: empty topics AND
            # reply_only=False. Broker WILL WARN+drop every broadcast.
            # Loud-fail at boot so the field operator sees it before
            # the runtime spam starts.
            logger.warning(
                "[worker_bus_bootstrap] worker '%s' attached to bus broker at %s "
                "WITH NEITHER topics NOR reply_only=True — SPEC §8.2 v1.4.0 "
                "regression: broker will WARN+drop every broadcast. Declare "
                "ModuleSpec.broadcast_topics OR set reply_only=True.",
                name, sock_path)
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
