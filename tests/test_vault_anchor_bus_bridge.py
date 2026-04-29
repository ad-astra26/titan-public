"""
Tests for the vault anchor bus-bridge (BUG-VAULT-COMMITS-NOT-LANDING fix).

Two surfaces under test:

  * ``titan_plugin.modules.memory_worker._request_anchor_via_kernel``
    — sends ANCHOR_REQUEST, waits on the worker's recv_queue for a matching
    bus.RESPONSE, re-injects unrelated messages, times out gracefully.

  * ``titan_plugin.core.kernel.TitanKernel._handle_anchor_request``
    — limbo guard, no-vault-program-id guard, build_failed path, send_failed
    path, success path. We mock self.network + self.bus + self._anchor_helper
    so we can assert the response payload without touching Solana.

Reference: titan_plugin/bus.py (ANCHOR_REQUEST docstring + wire contract).
"""
from __future__ import annotations

import asyncio
import threading
import time
from queue import Queue, Empty
from unittest.mock import MagicMock, patch

import pytest

from titan_plugin import bus as bus_mod
from titan_plugin.modules import memory_worker
from titan_plugin.core.kernel import TitanKernel


# ---------------------------------------------------------------------------
# memory_worker._request_anchor_via_kernel
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
    """Happy path: ANCHOR_REQUEST published, matching RESPONSE arrives."""
    send_queue: Queue = Queue()
    recv_queue: Queue = Queue()

    # Helper thread waits for the request to land on send_queue, then
    # injects a matching RESPONSE on recv_queue.
    def _kernel_simulator():
        # Pull the published ANCHOR_REQUEST.
        msg = send_queue.get(timeout=2.0)
        assert msg["type"] == bus_mod.ANCHOR_REQUEST
        assert msg["dst"] == "kernel"
        assert msg["src"] == "memory"
        rid = msg["rid"]
        recv_queue.put_nowait(_make_response(rid, "SIG_123abc", None))

    t = threading.Thread(target=_kernel_simulator, daemon=True)
    t.start()

    result = memory_worker._request_anchor_via_kernel(
        send_queue, recv_queue, "memory",
        state_root="MERKLE_abc",
        payload_json='[{"id":"n1"}]',
        promoted_count=1,
        timeout=5.0,
    )

    t.join(timeout=2.0)
    assert result == {"tx_signature": "SIG_123abc", "error": None}


def test_request_anchor_times_out_when_no_response():
    """Timeout returns explicit error, never blocks indefinitely."""
    send_queue: Queue = Queue()
    recv_queue: Queue = Queue()

    t0 = time.time()
    result = memory_worker._request_anchor_via_kernel(
        send_queue, recv_queue, "memory",
        state_root="MERKLE_abc",
        payload_json='[{"id":"n1"}]',
        promoted_count=1,
        timeout=1.5,
    )
    elapsed = time.time() - t0

    assert result == {"error": "anchor_request_timeout"}
    # Sanity: should have waited approximately the timeout, not bailed early.
    assert 1.4 <= elapsed <= 3.5
    # And the request was published.
    msg = send_queue.get_nowait()
    assert msg["type"] == bus_mod.ANCHOR_REQUEST


def test_request_anchor_reinjects_unrelated_messages():
    """Non-matching messages must end up back in recv_queue after the wait,
    so the main worker loop can process them after meditation returns."""
    send_queue: Queue = Queue()
    recv_queue: Queue = Queue()

    # Pre-load recv_queue with two non-anchor messages and one matching
    # RESPONSE. The function should consume only the RESPONSE; the
    # unrelated messages must be re-injected (in arrival order).
    rid_holder: dict = {}

    def _kernel_simulator():
        msg = send_queue.get(timeout=2.0)
        rid_holder["rid"] = msg["rid"]

        # Emit two unrelated messages first, then the matching RESPONSE.
        recv_queue.put_nowait({"type": "MEMORY_ADD", "src": "spirit",
                               "dst": "memory", "ts": time.time(),
                               "rid": None, "payload": {"text": "hello"}})
        recv_queue.put_nowait({"type": "QUERY", "src": "spirit",
                               "dst": "memory", "ts": time.time(),
                               "rid": "other_rid",
                               "payload": {"action": "status"}})
        recv_queue.put_nowait(_make_response(rid_holder["rid"], "SIG_x", None))

    t = threading.Thread(target=_kernel_simulator, daemon=True)
    t.start()

    result = memory_worker._request_anchor_via_kernel(
        send_queue, recv_queue, "memory",
        state_root="MERKLE_x", payload_json="[]", promoted_count=0,
        timeout=5.0,
    )
    t.join(timeout=2.0)

    assert result["tx_signature"] == "SIG_x"

    # The two unrelated messages must be back on recv_queue, in order.
    leftover_1 = recv_queue.get(timeout=0.5)
    leftover_2 = recv_queue.get(timeout=0.5)
    assert leftover_1["type"] == "MEMORY_ADD"
    assert leftover_2["type"] == "QUERY"
    assert leftover_2["rid"] == "other_rid"
    # Queue is now drained.
    with pytest.raises(Empty):
        recv_queue.get_nowait()


def test_request_anchor_ignores_response_with_wrong_rid():
    """A RESPONSE with a non-matching rid must be deferred, not consumed."""
    send_queue: Queue = Queue()
    recv_queue: Queue = Queue()

    def _kernel_simulator():
        msg = send_queue.get(timeout=2.0)
        rid = msg["rid"]
        # First put a RESPONSE with the WRONG rid (should be deferred).
        recv_queue.put_nowait(_make_response("not_our_rid", "SIG_other", None))
        # Then the correct one.
        recv_queue.put_nowait(_make_response(rid, "SIG_correct", None))

    t = threading.Thread(target=_kernel_simulator, daemon=True)
    t.start()

    result = memory_worker._request_anchor_via_kernel(
        send_queue, recv_queue, "memory",
        state_root="MERKLE_x", payload_json="[]", promoted_count=0,
        timeout=5.0,
    )
    t.join(timeout=2.0)

    assert result["tx_signature"] == "SIG_correct"

    # The wrong-rid RESPONSE must be back on recv_queue.
    leftover = recv_queue.get(timeout=0.5)
    assert leftover["payload"]["tx_signature"] == "SIG_other"


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
