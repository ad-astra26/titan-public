"""
Cross-language Phase C regression test — Rust kernel ↔ Python worker handshake.

This test would have caught BUG-PHASE-C-BUS-AUTHKEY-CONTRACT-DRIFT-20260505
(rFP_phase_c_bus_authkey_contract_fix.md):
  - Rust kernel passed `"titan_T3"` as HKDF info
  - Python worker passed `"T3"` (env var without prefix)
  - Different authkeys → 100% handshake failure under l0_rust_enabled=true
  - Existing parity tests verified the HKDF FUNCTION (both sides hardcoded
    "titan_T1" as info), but NO test exercised the cross-language RUNTIME
    call sites with the same env-var contract production uses.

This test boots a real `titan-kernel-rs` binary subprocess + connects a real
Python `BusSocketClient` via `setup_worker_bus` using the same env vars the
Rust kernel sets when spawning Python children. If Rust and Python derive
different authkeys, the handshake fails and this test fails.

Per `rFP_phase_c_bus_authkey_contract_fix.md` §3 acceptance criterion: this
test must FAIL before the fix is applied (proves it catches the bug) and
PASS after the fix.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Generator

import pytest

# Skip the entire module if the Rust kernel binary isn't available
# (e.g. local dev without `cargo build` run yet, or wrong working dir).
KERNEL_BIN_CANDIDATES = [
    Path("titan-rust/target/debug/titan-kernel-rs"),
    Path("titan-rust/target/release/titan-kernel-rs"),
    Path("bin/titan-kernel-rs"),
]


def _find_kernel_bin() -> Path | None:
    """Locate kernel binary. Test also requires `titan-trinity-rs` to be in
    the SAME directory (kernel boots and spawns substrate from sibling
    binary). Returns the kernel path only if its sibling substrate exists.
    Test skips otherwise (run `cargo build` to populate).
    """
    for cand in KERNEL_BIN_CANDIDATES:
        full = Path.cwd() / cand
        substrate = full.parent / "titan-trinity-rs"
        if (full.exists() and os.access(full, os.X_OK)
                and substrate.exists() and os.access(substrate, os.X_OK)):
            return full
    return None


KERNEL_BIN = _find_kernel_bin()
pytestmark = pytest.mark.skipif(
    KERNEL_BIN is None,
    reason="titan-kernel-rs binary not found (run `cargo build` in titan-rust/)",
)


def _generate_solana_byte_array_identity(path: Path, titan_id: str) -> bytes:
    """Generate a real Solana CLI byte-array-format identity keypair.

    Format: JSON array of 64 byte-integers (0-255).
    First 32 bytes = Ed25519 secret seed. Last 32 bytes = Ed25519 public key
    derived from the secret seed (the kernel verifies this match per SPEC G16).

    Returns the secret_seed (first 32 bytes) so the Python side can derive
    the matching authkey for parity comparison.
    """
    from solders.keypair import Keypair
    kp = Keypair()
    full = bytes(kp)  # 64 bytes: [seed:32] + [pub:32]
    assert len(full) == 64
    secret_seed = full[:32]
    path.write_text(json.dumps(list(full)))
    path.chmod(0o600)
    return secret_seed


@pytest.fixture
def kernel_subprocess(tmp_path: Path) -> Generator[dict, None, None]:
    """Boot a real titan-kernel-rs subprocess in an isolated tmp directory.

    Yields a dict with paths the test needs:
        - bus_socket: /tmp/<unique>/titan_bus.sock
        - keypair_path: /tmp/<unique>/data/titan_identity_keypair.json
        - titan_id: "T1" (matches what spawn.rs would set in env to children)
        - secret_seed: bytes — for Python authkey derivation parity check
        - proc: the subprocess.Popen handle
    """
    titan_id = "T1"
    data_dir = tmp_path / "data"
    shm_dir = tmp_path / "shm"
    data_dir.mkdir()
    shm_dir.mkdir()

    keypair_path = data_dir / "titan_identity_keypair.json"
    secret_seed = _generate_solana_byte_array_identity(keypair_path, titan_id)

    bus_socket = tmp_path / "titan_bus.sock"
    kernel_rpc_socket = tmp_path / "titan_kernel.sock"

    # Spawn the kernel — Python plugin spawn disabled (we only need the bus
    # broker for this test; spawning python_main would conflict with the
    # production T1's titan_main.pid file at the canonical CWD).
    # TITAN_KERNEL_SKIP_PYTHON=1 disables python_main spawn (production never sets this).
    env = {
        **os.environ,
        "TITAN_KERNEL_LOG_LEVEL": "warn",
        "TITAN_KERNEL_SKIP_PYTHON": "1",
    }
    # Capture kernel output to a tempfile (NOT subprocess.PIPE) — pytest
    # captures the test process's stdout/stderr so PIPE buffers can fill
    # and block the kernel's writes during a long test, preventing the
    # broker from accepting new connections.
    log_path = tmp_path / "kernel.log"
    log_fd = open(log_path, "w")
    proc = subprocess.Popen(
        [
            str(KERNEL_BIN),
            "--titan-id", titan_id,
            "--data-dir", str(data_dir),
            "--shm-dir", str(shm_dir),
            "--bus-socket", str(bus_socket),
            "--kernel-rpc-socket", str(kernel_rpc_socket),
        ],
        env=env,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
    )

    # Wait for bus socket to bind (kernel B4 step)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if bus_socket.exists():
            break
        if proc.poll() is not None:
            log_fd.close()
            output = log_path.read_text() if log_path.exists() else ""
            pytest.fail(
                f"kernel exited before bus bind: rc={proc.returncode}\nlog:\n{output}"
            )
        time.sleep(0.1)
    else:
        proc.kill()
        log_fd.close()
        pytest.fail(f"bus_socket {bus_socket} never appeared in 10s")

    yield {
        "bus_socket": bus_socket,
        "keypair_path": keypair_path,
        "titan_id": titan_id,
        "secret_seed": secret_seed,
        "proc": proc,
        "log_path": log_path,
    }

    # Teardown
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)
    log_fd.close()


def test_python_worker_handshake_with_rust_kernel(kernel_subprocess: dict) -> None:
    """**The regression test that would have caught the 2026-05-05 drift.**

    Spawn a Python `BusSocketClient` via `setup_worker_bus` with the same
    env vars `titan-kernel-rs::spawn::build_child_env` sets when spawning
    Python children in production. Verify handshake succeeds within 5s.

    Pre-fix behavior (the bug): Python derives authkey with `titan_id="T1"`
    while Rust derived with `"titan_T1"` → different authkeys → broker logs
    `HandshakeMismatch` → connection closed → `is_connected` stays False.

    Post-fix behavior (this test): Python derives with `info=b"titan-bus"`
    (constant), Rust derives with the same constant → same authkey → HMAC
    handshake succeeds → `is_connected=True`.
    """
    from queue import Queue

    from titan_plugin.core.worker_bus_bootstrap import (
        ENV_BUS_KEYPAIR_PATH,
        ENV_BUS_SOCKET_PATH,
        ENV_BUS_TITAN_ID,
        setup_worker_bus,
    )

    # Mimic the env vars Rust kernel sets when spawning Python children
    # (titan-rust/crates/titan-kernel-rs/src/spawn.rs:build_child_env).
    env = {
        ENV_BUS_SOCKET_PATH: str(kernel_subprocess["bus_socket"]),
        ENV_BUS_TITAN_ID: kernel_subprocess["titan_id"],  # "T1" — no prefix
        ENV_BUS_KEYPAIR_PATH: str(kernel_subprocess["keypair_path"]),
    }

    recv_q: Queue = Queue()
    send_q: Queue = Queue()
    sq_recv, sq_send, client = setup_worker_bus(
        "test-worker", recv_q, send_q, env=env,
    )

    assert client is not None, (
        "setup_worker_bus returned None client → fell back to mp.Queue "
        "legacy mode. Env vars set, keypair readable — should be socket mode. "
        "If pre-fix, the BusSocketClient would have been constructed but "
        "the handshake would fail. Check setup_worker_bus implementation."
    )

    try:
        # The actual cross-language handshake test.
        #
        # IMPORTANT: Python's _open_and_handshake() sends the HMAC response
        # and OPTIMISTICALLY sets _connected_event=True immediately, BEFORE
        # the broker validates the HMAC (the broker either keeps the
        # connection open on success, or silently closes it on mismatch —
        # there's no positive ACK frame). So `wait_until_connected=True`
        # alone is NOT proof of handshake success.
        #
        # The robust check: wait until connected, then SLEEP long enough for
        # a reconnect cycle to occur if the handshake actually failed, then
        # assert that reconnect_count is still 0 AND is_connected is still
        # True. Pre-fix code would have:
        #   - Python sends wrong HMAC → broker closes after a brief delay
        #   - Python's _recv_loop EOFs → connection_loop retries
        #   - reconnect_count increments
        #   - is_connected eventually clears (briefly) between reconnects
        # Post-fix code: handshake succeeds, connection holds steadily.

        connected = client.wait_until_connected(timeout=5.0)
        assert connected, (
            "Python BusSocketClient never reached _connected_event in 5s. "
            "This means it couldn't even complete the optimistic handshake "
            "(could be a socket bind issue, not the authkey drift bug). "
            "See rFP_phase_c_bus_authkey_contract_fix.md."
        )

        # Wait long enough for a reconnect cycle to occur if first handshake
        # was rejected by broker. Initial connect jitter is ~50-150ms;
        # backoff_attempt=1 retry is RECONNECT_BACKOFF_BASE_S * 1 ≈ 1s with
        # ±25% jitter (so ~0.75-1.25s). 4s window comfortably catches the
        # reconnect AND lets the broker close pre-fix bad-HMAC connections.
        time.sleep(4.0)

        # Post-fix: connection still alive, no reconnects.
        # Pre-fix: broker would have closed → reconnect happened → counter > 0.
        assert client.reconnect_count == 0, (
            f"Client reconnected {client.reconnect_count} times in 4s — "
            "broker is closing the connection post-handshake. This is the "
            "exact symptom of BUG-PHASE-C-BUS-AUTHKEY-CONTRACT-DRIFT-20260505: "
            "Rust kernel and Python worker derive different HKDF authkeys, "
            "broker emits HandshakeMismatch and drops the connection, worker "
            "retries forever. Verify both sides use info=b'titan-bus' as "
            "HKDF info constant. See rFP_phase_c_bus_authkey_contract_fix.md."
        )

        assert client.is_connected, (
            "is_connected=False after 4s wait — connection dropped. "
            "Broker is rejecting the worker (likely HandshakeMismatch)."
        )

        # Sanity: verify broker subprocess still running (didn't crash mid-test)
        assert kernel_subprocess["proc"].poll() is None, (
            "Rust kernel subprocess exited during the test."
        )
    finally:
        client.stop(timeout=2.0)
