"""
tests/test_topology_shm_writer.py — Phase A.S8 topology SHM writer tests.

Verifies kernel._start_topology_shm_writer behavior: poll loop reads
state_register.get_full_30d_topology(), writes TOPOLOGY_30D slot only on
content change (blake2b content-hash gating).
"""
import threading
import time
from unittest.mock import MagicMock, call

import numpy as np


def _make_kernel_stub(topology_values=None):
    """Minimal kernel-like object with _start_topology_shm_writer."""
    from titan_plugin.core.kernel import TitanKernel
    k = TitanKernel.__new__(TitanKernel)

    # Mock state_register
    k.state_register = MagicMock()
    if topology_values is None:
        topology_values = [0.5] * 30
    k.state_register.get_full_30d_topology.return_value = iter(topology_values)

    # Mock registry_bank writer
    mock_writer = MagicMock()
    mock_bank = MagicMock()
    mock_bank.writer.return_value = mock_writer
    k.registry_bank = mock_bank

    k._shm_writer_stop_evt = threading.Event()
    return k, mock_writer


def test_topology_30d_registryspec_shape():
    from titan_plugin.core.state_registry import TOPOLOGY_30D
    assert TOPOLOGY_30D.shape == (30,)
    assert TOPOLOGY_30D.name == "topology_30d"


def test_writer_writes_on_first_poll():
    """First topology poll always produces a write (no prior hash)."""
    k, mock_writer = _make_kernel_stub([float(i) * 0.01 for i in range(30)])

    # Override the 2s initial wait by patching the stop_evt.wait
    original_wait = k._shm_writer_stop_evt.wait
    call_count = [0]

    def fast_wait(timeout=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return False  # skip initial 2s wait
        return k._shm_writer_stop_evt.is_set()

    k._shm_writer_stop_evt.wait = fast_wait

    t = threading.Thread(target=k._start_topology_shm_writer, daemon=True)
    t.start()

    time.sleep(0.3)
    k._shm_writer_stop_evt.set()
    t.join(timeout=2.0)

    # Writer should have been called at least once
    assert mock_writer.write.call_count >= 1
    written_arr = mock_writer.write.call_args[0][0]
    assert written_arr.shape == (30,)
    assert written_arr.dtype == np.float32


def test_writer_content_hash_gates_duplicate_writes():
    """Identical topology on consecutive polls triggers only one write."""
    topology = [0.3] * 30
    k, mock_writer = _make_kernel_stub(topology)

    # Return same topology forever
    k.state_register.get_full_30d_topology.side_effect = lambda: iter(topology)

    poll_count = [0]
    original_wait = k._shm_writer_stop_evt.wait

    def fast_wait(timeout=None):
        poll_count[0] += 1
        if poll_count[0] >= 5:
            k._shm_writer_stop_evt.set()
            return True
        if poll_count[0] == 1:
            return False  # skip initial wait
        time.sleep(0.02)
        return k._shm_writer_stop_evt.is_set()

    k._shm_writer_stop_evt.wait = fast_wait

    t = threading.Thread(target=k._start_topology_shm_writer, daemon=True)
    t.start()
    t.join(timeout=2.0)

    # Despite multiple polls with same data, write count should be 1
    assert mock_writer.write.call_count == 1, (
        f"Expected 1 write for identical topology, got {mock_writer.write.call_count}"
    )


def test_hash_gate_logic_directly():
    """Content-hash gating logic: same bytes → no write; different bytes → write.
    Tests the blake2b gate in isolation, mirroring _writer_loop behavior."""
    import hashlib
    import numpy as np

    mock_write = MagicMock()
    last_hash = [None]

    def _conditional_write(arr: np.ndarray) -> None:
        payload = arr.tobytes(order="C")
        h = hashlib.blake2b(payload, digest_size=16).digest()
        if h != last_hash[0]:
            mock_write(arr)
            last_hash[0] = h

    # Write A
    arr_a = np.asarray([0.1] * 30, dtype=np.float32)
    _conditional_write(arr_a)
    assert mock_write.call_count == 1, "First write (A) should happen"

    # Write A again — same hash → skipped
    _conditional_write(arr_a)
    assert mock_write.call_count == 1, "Duplicate A must be gated"

    # Write B — different hash → goes through
    arr_b = np.asarray([0.9] * 30, dtype=np.float32)
    _conditional_write(arr_b)
    assert mock_write.call_count == 2, "Changed topology (B) must trigger write"

    # Write B again — gated
    _conditional_write(arr_b)
    assert mock_write.call_count == 2, "Duplicate B must be gated"
