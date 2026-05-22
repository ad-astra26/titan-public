"""
Tests for the vault anchor bus-bridge (BUG-VAULT-COMMITS-NOT-LANDING fix).

Two surfaces under test:

  * ``titan_hcl.modules.memory_worker._request_anchor_via_kernel``
    — sends ANCHOR_REQUEST, registers a Future in `InFlightRegistry`, and
    blocks until the kernel's RESPONSE arrives (resolved by the main loop)
    or the timeout expires.

    Phase A refactor (rFP §3.4.1, 2026-05-12): the function no longer reads
    from recv_queue directly. The main loop is the SOLE recv_queue reader
    and uses `registry.resolve(msg)` to route RESPONSE messages to the
    waiting Future. Unrelated messages and wrong-rid responses are NOT the
    bus-bridge's concern any more — the main loop dispatches them normally.

  * ``titan_hcl.core.kernel.TitanKernel._handle_anchor_request``
    — limbo guard, no-vault-program-id guard, build_failed path, send_failed
    path, success path. We mock self.network + self.bus + self._anchor_helper
    so we can assert the response payload without touching Solana.

Reference: titan_hcl/bus.py (ANCHOR_REQUEST docstring + wire contract),
PLAN_phase_c_memory_worker_concurrent_dispatch.md §2.4 (InFlightRegistry).
"""
from __future__ import annotations

import asyncio
import threading
import time
from queue import Queue
from unittest.mock import MagicMock

import pytest

from titan_hcl import bus as bus_mod
from titan_hcl.modules import memory_worker
from titan_hcl.modules._memory_dispatch import InFlightRegistry
from titan_hcl.core.kernel import TitanKernel


# ---------------------------------------------------------------------------
# memory_worker._request_anchor_via_kernel (Phase A — InFlightRegistry)
# ---------------------------------------------------------------------------


def _make_response(rid: str, tx_signature: str | None, error: str | None) -> dict:
    return {
        "type": bus_mod.RESPONSE,
        "src": "kernel",
        "dst": "memory",
        "ts": time.time(),
        "rid": rid,
        "payload": {"tx_signature": tx_signature, "error": error},
    }


def test_request_anchor_returns_tx_signature_on_match():
    """Happy path: ANCHOR_REQUEST published; kernel simulator reads it,
    builds matching RESPONSE, resolves the registered Future via
    `registry.resolve(...)`. Bus-bridge returns the kernel's payload."""
    send_queue: Queue = Queue()
    registry = InFlightRegistry()

    def _kernel_simulator():
        # Read the published ANCHOR_REQUEST off send_queue
        msg = send_queue.get(timeout=2.0)
        assert msg["type"] == bus_mod.ANCHOR_REQUEST
        assert msg["dst"] == "kernel"
        assert msg["src"] == "memory"
        rid = msg["rid"]
        # Main loop would call registry.resolve on incoming RESPONSE.
        # Simulate it directly here.
        ok = registry.resolve(_make_response(rid, "SIG_123abc", None))
        assert ok is True

    t = threading.Thread(target=_kernel_simulator, daemon=True)
    t.start()

    result = memory_worker._request_anchor_via_kernel(
        registry, send_queue, "memory",
        state_root="MERKLE_abc",
        payload_json='[{"id":"n1"}]',
        promoted_count=1,
        timeout=5.0,
    )

    t.join(timeout=2.0)
    assert result == {"tx_signature": "SIG_123abc", "error": None}
    # Registry is purged after resolve
    assert registry.in_flight_count() == 0


def test_request_anchor_times_out_when_no_response():
    """Timeout returns explicit error, never blocks indefinitely.
    Registry is purged on timeout (rid cancelled)."""
    send_queue: Queue = Queue()
    registry = InFlightRegistry()

    t0 = time.time()
    result = memory_worker._request_anchor_via_kernel(
        registry, send_queue, "memory",
        state_root="MERKLE_abc",
        payload_json='[{"id":"n1"}]',
        promoted_count=1,
        timeout=1.5,
    )
    elapsed = time.time() - t0

    assert result == {"error": "anchor_request_timeout"}
    # Sanity: should have waited approximately the timeout, not bailed early.
    assert 1.4 <= elapsed <= 3.5
    # The request was published.
    msg = send_queue.get_nowait()
    assert msg["type"] == bus_mod.ANCHOR_REQUEST
    # Registry purged the timed-out rid via cancel
    assert registry.in_flight_count() == 0


def test_request_anchor_does_not_touch_recv_queue():
    """The Phase A refactor guarantees the bus-bridge never reads from
    recv_queue. We assert this structurally: pass a Queue with items in
    it; verify it's untouched after the bus-bridge runs."""
    send_queue: Queue = Queue()
    # An "unused" queue that the bus-bridge MUST NOT touch
    recv_queue: Queue = Queue()
    recv_queue.put_nowait({"type": "MEMORY_ADD", "src": "spirit",
                           "dst": "memory", "rid": None,
                           "payload": {"text": "hello"}})
    recv_queue.put_nowait({"type": "QUERY", "src": "spirit",
                           "dst": "memory", "rid": "other_rid",
                           "payload": {"action": "status"}})
    pre_size = recv_queue.qsize()

    registry = InFlightRegistry()

    def _kernel_simulator():
        msg = send_queue.get(timeout=2.0)
        registry.resolve(_make_response(msg["rid"], "SIG_x", None))

    t = threading.Thread(target=_kernel_simulator, daemon=True)
    t.start()

    result = memory_worker._request_anchor_via_kernel(
        registry, send_queue, "memory",
        state_root="MERKLE_x", payload_json="[]", promoted_count=0,
        timeout=5.0,
    )
    t.join(timeout=2.0)

    assert result["tx_signature"] == "SIG_x"
    # recv_queue is unchanged — bus-bridge never reads it
    assert recv_queue.qsize() == pre_size
    leftover_1 = recv_queue.get(timeout=0.5)
    leftover_2 = recv_queue.get(timeout=0.5)
    assert leftover_1["type"] == "MEMORY_ADD"
    assert leftover_2["type"] == "QUERY"


def test_request_anchor_ignores_response_with_wrong_rid():
    """A RESPONSE with non-matching rid does NOT resolve our Future
    (registry.resolve returns False for unknown rids). The bus-bridge
    keeps waiting until the matching rid arrives.

    Under the new InFlightRegistry contract, wrong-rid messages are
    silently no-ops as far as the bus-bridge is concerned — the main
    loop's normal dispatcher handles them. We don't put wrong-rid msgs
    into the registry at all; we just call resolve(...) and check it
    returns False before the correct-rid resolve."""
    send_queue: Queue = Queue()
    registry = InFlightRegistry()

    def _kernel_simulator():
        msg = send_queue.get(timeout=2.0)
        rid = msg["rid"]
        # Main loop sees wrong-rid first — registry says "not mine"
        wrong = _make_response("not_our_rid", "SIG_other", None)
        assert registry.resolve(wrong) is False
        # ... main loop dispatches `wrong` normally (not our concern here).
        # Then the correct RESPONSE arrives — registry resolves the Future.
        assert registry.resolve(_make_response(rid, "SIG_correct", None)) is True

    t = threading.Thread(target=_kernel_simulator, daemon=True)
    t.start()

    result = memory_worker._request_anchor_via_kernel(
        registry, send_queue, "memory",
        state_root="MERKLE_x", payload_json="[]", promoted_count=0,
        timeout=5.0,
    )
    t.join(timeout=2.0)

    assert result["tx_signature"] == "SIG_correct"
    assert registry.in_flight_count() == 0


# ---------------------------------------------------------------------------
# kernel._handle_anchor_request
# ---------------------------------------------------------------------------


def _make_kernel_for_anchor(
    *,
    limbo: bool = False,
    network=None,
    vault_program_id: str = "VAULT_PROG_ID_DEVNET",
    inference_cfg: dict | None = None,
):
    """Build a TitanKernel-like duck-type just for _handle_anchor_request.

    We bypass __init__ (which boots Soul + Guardian + bus + shm writers)
    and inject only the attributes the anchor handler reads.
    """
    kernel = TitanKernel.__new__(TitanKernel)
    kernel._limbo_mode = limbo
    kernel.network = network
    kernel._config = {
        "network": {"vault_program_id": vault_program_id},
        "inference": inference_cfg or {},
    }
    kernel.bus = MagicMock()
    return kernel


def test_handle_anchor_request_limbo_mode_replies_error():
    """Limbo (no keypair) → reply with limbo_mode_no_network, no TX."""
    kernel = _make_kernel_for_anchor(limbo=True, network=None)
    msg = {
        "type": bus_mod.ANCHOR_REQUEST,
        "src": "memory", "dst": "kernel", "rid": "rid_limbo",
        "payload": {"state_root": "MERKLE_x", "payload": "[]",
                    "promoted_count": 0, "ts": time.time()},
    }

    asyncio.run(kernel._handle_anchor_request(msg))

    # bus.publish called once with RESPONSE matching rid + error payload
    kernel.bus.publish.assert_called_once()
    sent = kernel.bus.publish.call_args[0][0]
    assert sent["type"] == bus_mod.RESPONSE
    assert sent["src"] == "kernel"
    assert sent["dst"] == "memory"
    assert sent["rid"] == "rid_limbo"
    assert sent["payload"]["tx_signature"] is None
    assert sent["payload"]["error"] == "limbo_mode_no_network"


def test_handle_anchor_request_no_vault_program_id_replies_error():
    """No vault_program_id configured → reply with no_vault_program_id."""
    kernel = _make_kernel_for_anchor(
        network=MagicMock(), vault_program_id="",
    )
    msg = {
        "type": bus_mod.ANCHOR_REQUEST,
        "src": "memory", "dst": "kernel", "rid": "rid_novault",
        "payload": {"state_root": "MERKLE_x", "payload": "[]",
                    "promoted_count": 0, "ts": time.time()},
    }
    asyncio.run(kernel._handle_anchor_request(msg))
    sent = kernel.bus.publish.call_args[0][0]
    assert sent["payload"]["error"] == "no_vault_program_id"
    assert sent["payload"]["tx_signature"] is None


def test_handle_anchor_request_build_failure_replies_error():
    """Instruction-build raises → reply with build_failed:<ExceptionName>."""
    network = MagicMock()
    kernel = _make_kernel_for_anchor(network=network)

    # Pre-install a fake _anchor_helper so we don't pay the import cost
    # of MeditationEpoch and we control its behavior precisely.
    helper = MagicMock()
    helper._build_commit_instructions.side_effect = RuntimeError("boom")
    kernel._anchor_helper = helper

    msg = {
        "type": bus_mod.ANCHOR_REQUEST,
        "src": "memory", "dst": "kernel", "rid": "rid_build_fail",
        "payload": {"state_root": "MERKLE_x", "payload": "[]",
                    "promoted_count": 1, "ts": time.time()},
    }
    asyncio.run(kernel._handle_anchor_request(msg))

    sent = kernel.bus.publish.call_args[0][0]
    assert sent["payload"]["error"].startswith("build_failed:")
    assert sent["payload"]["tx_signature"] is None


def test_handle_anchor_request_no_instructions_replies_error():
    """Helper returns [] → reply with no_instructions, no TX call."""
    network = MagicMock()
    kernel = _make_kernel_for_anchor(network=network)

    helper = MagicMock()
    helper._build_commit_instructions.return_value = []
    kernel._anchor_helper = helper

    msg = {
        "type": bus_mod.ANCHOR_REQUEST,
        "src": "memory", "dst": "kernel", "rid": "rid_no_ix",
        "payload": {"state_root": "MERKLE_x", "payload": "[]",
                    "promoted_count": 0, "ts": time.time()},
    }
    asyncio.run(kernel._handle_anchor_request(msg))

    sent = kernel.bus.publish.call_args[0][0]
    assert sent["payload"]["error"] == "no_instructions"
    assert sent["payload"]["tx_signature"] is None
    network.send_sovereign_transaction.assert_not_called()


def test_handle_anchor_request_success_replies_tx_signature():
    """Success path: instructions built + TX submitted + tx_signature returned."""
    async def _send_ok(instructions, priority):
        return "SIG_landed_abc"
    network = MagicMock()
    network.send_sovereign_transaction = _send_ok

    kernel = _make_kernel_for_anchor(network=network)
    helper = MagicMock()
    helper._build_commit_instructions.return_value = ["IX_VAULT", "IX_MEMO"]
    kernel._anchor_helper = helper

    msg = {
        "type": bus_mod.ANCHOR_REQUEST,
        "src": "memory", "dst": "kernel", "rid": "rid_success",
        "payload": {"state_root": "MERKLE_success", "payload": "[]",
                    "promoted_count": 3, "ts": time.time()},
    }
    asyncio.run(kernel._handle_anchor_request(msg))

    sent = kernel.bus.publish.call_args[0][0]
    assert sent["payload"]["tx_signature"] == "SIG_landed_abc"
    assert sent["payload"]["error"] is None
    assert sent["rid"] == "rid_success"


def test_handle_anchor_request_send_returns_none_replies_error():
    """send_sovereign_transaction returns None (budget exceeded / RPC down)
    → reply with tx_returned_none."""
    async def _send_none(instructions, priority):
        return None
    network = MagicMock()
    network.send_sovereign_transaction = _send_none

    kernel = _make_kernel_for_anchor(network=network)
    helper = MagicMock()
    helper._build_commit_instructions.return_value = ["IX_VAULT"]
    kernel._anchor_helper = helper

    msg = {
        "type": bus_mod.ANCHOR_REQUEST,
        "src": "memory", "dst": "kernel", "rid": "rid_none",
        "payload": {"state_root": "MERKLE_x", "payload": "[]",
                    "promoted_count": 1, "ts": time.time()},
    }
    asyncio.run(kernel._handle_anchor_request(msg))

    sent = kernel.bus.publish.call_args[0][0]
    assert sent["payload"]["tx_signature"] is None
    assert sent["payload"]["error"] == "tx_returned_none"


def test_handle_anchor_request_send_exception_replies_error():
    """send_sovereign_transaction raises → reply with send_failed:<ExceptionName>."""
    async def _send_raise(instructions, priority):
        raise ConnectionError("rpc unreachable")
    network = MagicMock()
    network.send_sovereign_transaction = _send_raise

    kernel = _make_kernel_for_anchor(network=network)
    helper = MagicMock()
    helper._build_commit_instructions.return_value = ["IX_VAULT"]
    kernel._anchor_helper = helper

    msg = {
        "type": bus_mod.ANCHOR_REQUEST,
        "src": "memory", "dst": "kernel", "rid": "rid_send_err",
        "payload": {"state_root": "MERKLE_x", "payload": "[]",
                    "promoted_count": 1, "ts": time.time()},
    }
    asyncio.run(kernel._handle_anchor_request(msg))

    sent = kernel.bus.publish.call_args[0][0]
    assert sent["payload"]["tx_signature"] is None
    assert sent["payload"]["error"].startswith("send_failed:")
