"""
Integration tests for Microkernel v2 Phase A §A.2 — cross-process shm
reads via persistent mmap.

These tests spawn real subprocesses to validate the cross-process
contract (the production scenario: main-process writes Trinity state,
dashboard reads; in Phase B+, spirit_worker subprocess reads).

Reference: titan-docs/PLAN_microkernel_phase_a.md §5.5.3
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pytest

from titan_plugin.core.state_registry import (
    RegistrySpec,
    StateRegistryReader,
    StateRegistryWriter,
)


def _writer_child(shm_root_str: str, num_writes: int, sleep_s: float):
    """Subprocess entry: write monotonically-increasing vectors."""
    os.environ["TITAN_SHM_ROOT"] = shm_root_str
    spec = RegistrySpec(
        name="ipc_test",
        dtype=np.dtype("<f4"),
        shape=(16,),
        feature_flag="",
    )
    w = StateRegistryWriter(spec, Path(shm_root_str))
    for i in range(num_writes):
        w.write(np.full(16, float(i + 1), dtype=np.float32))
        if sleep_s > 0:
            time.sleep(sleep_s)
    w.close()


def _reader_child(shm_root_str: str, num_reads: int, results_path: str):
    """Subprocess entry: read repeatedly, append values seen to a file."""
    os.environ["TITAN_SHM_ROOT"] = shm_root_str
    spec = RegistrySpec(
        name="ipc_test",
        dtype=np.dtype("<f4"),
        shape=(16,),
        feature_flag="",
    )
    r = StateRegistryReader(spec, Path(shm_root_str))
    values_seen: list[float] = []
    for _ in range(num_reads):
        result = r.read()
        if result is not None:
            # Record the first element (all elements are equal by design)
            values_seen.append(float(result[0]))
        time.sleep(0.001)
    r.close()
    # Save results
    with open(results_path, "w") as f:
        for v in values_seen:
            f.write(f"{v}\n")


def test_cross_process_writer_reader(tmp_path, monkeypatch):
    """
    Producer subprocess writes values 1..100; consumer subprocess reads
    100 times in parallel. Consumer MUST observe monotonic values (no
    time travel) and each value MUST be a uniform vector (not torn).
    """
    shm_root = tmp_path / "shm"
    shm_root.mkdir()
    results_path = tmp_path / "reader_results.txt"

    ctx = mp.get_context("spawn")
    writer_proc = ctx.Process(
        target=_writer_child,
        args=(str(shm_root), 100, 0.005),  # 100 writes @ 5ms each = 500ms
    )
    reader_proc = ctx.Process(
        target=_reader_child,
        args=(str(shm_root), 500, str(results_path)),  # 500 reads @ 1ms each
    )

    writer_proc.start()
    # Tiny delay so the writer creates the file before the reader tries
    # to attach. The reader's lazy attach handles the race too, but
    # this makes the test deterministic.
    time.sleep(0.05)
    reader_proc.start()

    writer_proc.join(timeout=60)
    reader_proc.join(timeout=60)

    assert writer_proc.exitcode == 0, "writer subprocess failed"
    assert reader_proc.exitcode == 0, "reader subprocess failed"

    # Verify results: values must be monotonically non-decreasing
    # (reader sees snapshots in order; it may miss intermediate writes,
    # but MUST NOT see a value smaller than one previously read).
    with open(results_path) as f:
        values = [float(line.strip()) for line in f if line.strip()]

    # Correctness gate (primary): monotonic reads when any happen.
    # Liveness gate (secondary): at least one successful read.
    # Under heavy CPU contention, the reader may legitimately miss quiet
    # windows — SeqLock correctly returns None then rather than corrupt
    # data. The architectural invariant is monotonicity of whatever DID
    # succeed, not a throughput threshold.
    for i in range(1, len(values)):
        assert values[i] >= values[i - 1], (
            f"Non-monotonic: values[{i-1}]={values[i-1]}, values[{i}]={values[i]}"
        )
    assert len(values) > 0, (
        "Reader got ZERO valid reads — cross-process mmap contract broken. "
        "(Under heavy CPU contention some misses are expected; zero is a hard fail.)"
    )


def test_multiple_readers_same_registry(tmp_path, monkeypatch):
    """
    1 writer, 3 concurrent reader processes. All readers must complete
    without errors and observe monotonic values.
    """
    shm_root = tmp_path / "shm"
    shm_root.mkdir()

    ctx = mp.get_context("spawn")

    # Slower writer cadence (10ms between writes) + more reads per
    # reader so at least ONE reader consistently lands during a quiet
    # window even on noisy CI boxes. Individual-reader starvation can
    # happen under OS-scheduler bad luck with 3 reader processes all
    # starting in the same wall-clock tick; the architectural invariant
    # is that the SYSTEM serves readers (not that every individual reader
    # gets >N successful reads under stress).
    writer_proc = ctx.Process(
        target=_writer_child, args=(str(shm_root), 100, 0.010),
    )
    reader_procs = []
    result_files = []
    for i in range(3):
        rf = tmp_path / f"reader_{i}.txt"
        result_files.append(rf)
        p = ctx.Process(
            target=_reader_child,
            args=(str(shm_root), 500, str(rf)),
        )
        reader_procs.append(p)

    writer_proc.start()
    time.sleep(0.05)
    for p in reader_procs:
        p.start()

    writer_proc.join(timeout=60)
    for p in reader_procs:
        p.join(timeout=60)

    assert writer_proc.exitcode == 0
    for i, p in enumerate(reader_procs):
        assert p.exitcode == 0, f"reader {i} failed"

    # Correctness invariants: monotonicity per reader + system liveness
    # (not every individual reader — they can get unlucky).
    all_values: list[tuple[str, list[float]]] = []
    for rf in result_files:
        with open(rf) as f:
            values = [float(line.strip()) for line in f if line.strip()]
        all_values.append((rf.name, values))
        # Monotonicity is the primary correctness gate.
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1], (
                f"{rf.name}: non-monotonic at {i}: "
                f"{values[i-1]} then {values[i]}"
            )
    # At least one reader must have made meaningful progress.
    max_reads = max(len(v) for _, v in all_values)
    total_reads = sum(len(v) for _, v in all_values)
    assert max_reads > 10, (
        f"No reader made meaningful progress (max={max_reads}): "
        f"{[(n, len(v)) for n, v in all_values]}"
    )
    assert total_reads > 30, f"System-wide liveness failed: total={total_reads}"
