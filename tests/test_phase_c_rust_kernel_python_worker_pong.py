"""
Cross-language Phase C regression test — Rust kernel survives heartbeat
window without crashing.

This test would have caught the BusClient PONG-on-PING contract violation
that produced the T3 crash-loop. Per rFP_phase_c_close_all_runtime_gaps
chunk 9D §1.3:

  - Every Rust subscriber inside titan-kernel-rs (anon-1..anon-7) connects
    to the in-process broker via `BusClient::connect`.
  - Pre-fix `run_recv_loop` forwarded BUS_PING frames to events_tx but no
    Rust binary handled them — none of `titan-trinity-rs / titan-unified-
    spirit-rs / titan-inner-body-rs / titan-outer-body-rs` had a
    `send_pong` call site.
  - Broker pings every 5s (BUS_PING_INTERVAL_S=5.0); drops a subscriber
    when last_pong_ts ages past 15s (BUS_PING_TIMEOUT_S=15.0).
  - At t≈15s every internal subscriber's last_pong_ts ages out
    simultaneously → broker closes all 7 → next publish on any of them
    fails with "Broken pipe (os error 32)" → kernel exits clean (code=0)
    with "broker still referenced at shutdown".
  - systemd hits StartLimitBurst after 5-6 retries → service marked failed.

This test boots the real binary, holds it for >30s (well past the 25-30s
crash window), and asserts the process is still alive at the end.

Pre-fix: this test fails — kernel exits within 30s of boot.
Post-fix (chunk 9D auto-PONG in client.rs): test passes — kernel survives
indefinitely.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Generator

import pytest


KERNEL_BIN_CANDIDATES = [
    Path("titan-rust/target/debug/titan-kernel-rs"),
    Path("titan-rust/target/release/titan-kernel-rs"),
    Path("bin/titan-kernel-rs"),
]


def _find_kernel_bin() -> Path | None:
    """Locate kernel binary. Test also requires `titan-trinity-rs` in same
    directory because the kernel boots and spawns substrate from sibling
    binary. Returns the kernel path only if BOTH exist + executable.
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


def _generate_solana_byte_array_identity(path: Path) -> bytes:
    """Generate a Solana CLI byte-array-format identity keypair (64 bytes,
    [seed:32] + [pub:32]). Returns the secret seed for parity sanity if
    needed."""
    from solders.keypair import Keypair  # type: ignore[import-not-found]

    kp = Keypair()
    full = bytes(kp)
    assert len(full) == 64
    seed = full[:32]
    path.write_text(json.dumps(list(full)))
    path.chmod(0o600)
    return seed


@pytest.fixture
def kernel_subprocess(tmp_path: Path) -> Generator[dict, None, None]:
    titan_id = "T1"
    data_dir = tmp_path / "data"
    shm_dir = tmp_path / "shm"
    data_dir.mkdir()
    shm_dir.mkdir()

    keypair_path = data_dir / "titan_identity_keypair.json"
    _generate_solana_byte_array_identity(keypair_path)

    bus_socket = tmp_path / "titan_bus.sock"
    kernel_rpc_socket = tmp_path / "titan_kernel.sock"

    env = {
        **os.environ,
        "TITAN_KERNEL_LOG_LEVEL": "warn",
        # Skip Python plugin spawn — this test exercises the Rust-side
        # heartbeat loop only. Python plugin would conflict with any other
        # titan_main.pid in the canonical CWD.
        "TITAN_KERNEL_SKIP_PYTHON": "1",
    }
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
        "proc": proc,
        "log_path": log_path,
        "bus_socket": bus_socket,
        "titan_id": titan_id,
    }

    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)
    log_fd.close()


def test_kernel_survives_past_heartbeat_timeout_window(
    kernel_subprocess: dict,
) -> None:
    """Kernel must stay alive past the 15s heartbeat timeout window.

    Pre-fix: every internal Rust subscriber missed PONGs → at t≈15s broker
    drops all of them → "Broken pipe" on next publish → kernel exits clean
    with "broker still referenced at shutdown".

    Post-fix: BusClient::run_recv_loop auto-PONGs on every BUS_PING; broker
    never marks any subscriber as timed-out; kernel runs indefinitely.

    We hold for 30s — twice the 15s timeout window — to be conclusive.
    """
    proc = kernel_subprocess["proc"]
    log_path: Path = kernel_subprocess["log_path"]

    deadline = time.time() + 30.0
    while time.time() < deadline:
        if proc.poll() is not None:
            output = log_path.read_text() if log_path.exists() else ""
            elapsed = 30.0 - (deadline - time.time())
            pytest.fail(
                f"kernel exited at t≈{elapsed:.1f}s with rc={proc.returncode} "
                f"— pre-fix crash-loop symptom (BusClient PONG contract).\n\n"
                f"Last 40 lines of kernel log:\n"
                + "\n".join(output.splitlines()[-40:])
            )
        time.sleep(1.0)

    assert proc.poll() is None, (
        "Kernel subprocess exited unexpectedly before 30s elapsed"
    )

    # Sanity: the journal MUST NOT show "heartbeat timeout — closing"
    # for any internal subscriber. If it does, the broker thought the
    # client was dead — auto-PONG is broken.
    log_text = log_path.read_text() if log_path.exists() else ""
    assert "heartbeat timeout — closing subscriber" not in log_text, (
        "Broker logged a heartbeat-timeout — chunk 9D auto-PONG path is "
        "not engaging on internal Rust subscribers.\n\n"
        "Last 40 lines of kernel log:\n"
        + "\n".join(log_text.splitlines()[-40:])
    )
    assert "Broken pipe" not in log_text, (
        "Kernel logged a Broken pipe error — broker dropped a subscriber "
        "and the next publish on it failed. Auto-PONG fix is incomplete.\n\n"
        "Last 40 lines of kernel log:\n"
        + "\n".join(log_text.splitlines()[-40:])
    )
