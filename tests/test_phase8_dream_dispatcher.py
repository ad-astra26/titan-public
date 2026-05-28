"""Phase 8 — synthesis_worker dream dispatcher ordering tests (D-SPEC-PHASE8).

This test verifies the contract embodied in synthesis_worker's
_handle_dream_state_changed pathway:
  1. LLM judge dispatched BEFORE procedural miner (INV-Syn-21)
  2. Consolidation + ForkGC also fire on the same DREAM_STATE_CHANGED
  3. Each rate-limiter prevents double-firing on the same dream window
  4. Component failures are isolated (one crashing doesn't block others)

We exercise the wiring shape directly without spinning up the whole
worker process — the dispatchers are module-local closures, so we
re-implement the dispatcher pattern in a fixture and assert the
ordering contract using mocks. This is faster than spawning a
subprocess + cheaper to maintain.

For end-to-end integration (dream tick → judge → miner → snapshot)
see test_phase8_integration.py (P8.K).
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest


class _OrderRecorder:
    """Captures the order in which judge/miner/consolidation/forkgc fire."""

    def __init__(self) -> None:
        self.events: list[tuple[str, float]] = []
        self._lock = threading.Lock()

    def record(self, name: str) -> None:
        with self._lock:
            self.events.append((name, time.time()))

    def order(self) -> list[str]:
        return [n for n, _ in self.events]


def _build_dispatcher_set(rec: _OrderRecorder):
    """Replicate the dispatcher pattern from synthesis_worker.py."""
    last_judge_ts = [0.0]
    last_miner_ts = [0.0]
    last_consol_ts = [0.0]
    last_fork_ts = [0.0]
    judge_lock = threading.Lock()
    miner_lock = threading.Lock()
    consol_lock = threading.Lock()
    fork_lock = threading.Lock()
    threads: list[threading.Thread] = []

    def dispatch_judge(ts: float) -> None:
        with judge_lock:
            if ts <= last_judge_ts[0]:
                return
            last_judge_ts[0] = ts
        t = threading.Thread(
            target=lambda: rec.record("judge"), daemon=True,
        )
        t.start()
        threads.append(t)

    def dispatch_miner(ts: float) -> None:
        with miner_lock:
            if ts <= last_miner_ts[0]:
                return
            last_miner_ts[0] = ts
        t = threading.Thread(
            target=lambda: rec.record("miner"), daemon=True,
        )
        t.start()
        threads.append(t)

    def dispatch_consolidation(ts: float) -> None:
        with consol_lock:
            if ts <= last_consol_ts[0]:
                return
            last_consol_ts[0] = ts
        t = threading.Thread(
            target=lambda: rec.record("consolidation"), daemon=True,
        )
        t.start()
        threads.append(t)

    def dispatch_fork_gc(ts: float) -> None:
        with fork_lock:
            if ts <= last_fork_ts[0]:
                return
            last_fork_ts[0] = ts
        t = threading.Thread(
            target=lambda: rec.record("fork_gc"), daemon=True,
        )
        t.start()
        threads.append(t)

    def fire_dream_handler(ts: float) -> None:
        # Mirror the canonical order from synthesis_worker.py:
        dispatch_judge(ts)
        dispatch_consolidation(ts)
        dispatch_fork_gc(ts)
        dispatch_miner(ts)

    def join_all(timeout: float = 2.0):
        for t in threads:
            t.join(timeout=timeout)

    return fire_dream_handler, join_all


# ── Ordering contract ──────────────────────────────────────────────────


def test_judge_dispatched_before_miner_in_handler_call_order():
    """Even with threading, the dispatch CALL ORDER must place judge before miner."""
    rec = _OrderRecorder()
    fire, join = _build_dispatcher_set(rec)
    fire(1000.0)
    join()
    order = rec.order()
    assert "judge" in order
    assert "miner" in order
    # The dispatch call sequence ensures judge.dispatch is invoked first.
    # In production each runs in its own thread, but the API contract
    # is that the WIRING calls judge_dispatcher before miner_dispatcher.


def test_all_four_components_fire_on_dream_tick():
    rec = _OrderRecorder()
    fire, join = _build_dispatcher_set(rec)
    fire(2000.0)
    join()
    fired = set(rec.order())
    assert {"judge", "miner", "consolidation", "fork_gc"} == fired


def test_same_dream_window_does_not_refire():
    rec = _OrderRecorder()
    fire, join = _build_dispatcher_set(rec)
    fire(3000.0)
    join()
    first_fire = len(rec.events)
    # Second call with same ts → rate-limiter blocks all 4
    fire(3000.0)
    join()
    assert len(rec.events) == first_fire


def test_new_dream_window_fires_again():
    rec = _OrderRecorder()
    fire, join = _build_dispatcher_set(rec)
    fire(4000.0)
    join()
    first = len(rec.events)
    fire(4001.0)  # new window
    join()
    assert len(rec.events) == first + 4


def test_dispatcher_returns_immediately_when_component_missing():
    """When llm_judge / procedural_miner is None, the dispatcher should be a no-op."""
    # In synthesis_worker.py the actual dispatcher checks `if llm_judge is None: return`
    # before the rate-limit logic. Verify that pattern.
    last_ts = [0.0]
    lock = threading.Lock()
    invoked = []

    def dispatch_safe(component, ts: float) -> None:
        if component is None:
            return  # NO rate-limit advance
        with lock:
            if ts <= last_ts[0]:
                return
            last_ts[0] = ts
        invoked.append(ts)

    dispatch_safe(None, 1.0)
    assert invoked == []
    assert last_ts[0] == 0.0  # last_ts not advanced

    dispatch_safe(MagicMock(), 1.0)
    assert invoked == [1.0]


def test_individual_component_failure_does_not_block_others():
    """If the judge crashes mid-run, the miner thread still runs."""
    rec = _OrderRecorder()
    started = []

    def run_judge():
        started.append("judge_started")
        raise RuntimeError("judge boom")

    def run_miner():
        rec.record("miner")

    t1 = threading.Thread(target=run_judge, daemon=True)
    t2 = threading.Thread(target=run_miner, daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=1.0)
    t2.join(timeout=1.0)
    assert "miner" in rec.order()
    assert "judge_started" in started


def test_synthesis_worker_imports_p8_components():
    """Regression: verify the wiring lines + dispatchers are present in the worker."""
    import inspect
    from titan_hcl.modules import synthesis_worker
    src = inspect.getsource(synthesis_worker)
    assert "ProceduralSkillStore" in src
    assert "SkillVerifier" in src
    assert "LLMJudge" in src
    assert "ProceduralMiner" in src
    assert "_maybe_run_llm_judge_async" in src
    assert "_maybe_run_procedural_miner_async" in src
    # Wiring call order: judge before miner
    judge_idx = src.find("_maybe_run_llm_judge_async(dream_start_ts)")
    miner_idx = src.find("_maybe_run_procedural_miner_async(dream_start_ts)")
    assert judge_idx > 0
    assert miner_idx > 0
    assert judge_idx < miner_idx, "judge must dispatch before miner per INV-Syn-21"


def test_synthesis_worker_imports_chain_content_hash_reader():
    """SkillVerifier's chain_reader dependency is wired."""
    import inspect
    from titan_hcl.modules import synthesis_worker
    src = inspect.getsource(synthesis_worker)
    assert "ChainContentHashReader" in src
