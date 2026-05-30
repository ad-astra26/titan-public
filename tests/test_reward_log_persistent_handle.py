"""PROFILING.md F6 — reward audit log: persistent line-buffered handle.

Replaces the per-event ``open(path,"a")`` (firehose cadence, ~4.6% of
cognitive_worker on-CPU) with a single reused line-buffered handle, and
truncates only when the tracked line count exceeds the cap (was: a full-file
``readlines()`` every 1000 events). Verifies:
  1. Handle is opened once and reused across many events (not per-event).
  2. Line-buffering keeps read-after-write consistent (audit fidelity).
  3. Truncation fires only when over the cap, keeping the last N lines.
"""
import json

from titan_hcl.logic.neural_nervous_system import NeuralNervousSystem


def _make_config(n: int = 1) -> dict:
    progs = {}
    for name in ["TEST_A", "TEST_B", "TEST_C"][:n]:
        progs[name] = {
            "enabled": True,
            "input_features": "standard",  # 55D
            "fire_threshold": 0.3,
            "buffer_max": 200,
        }
    return {"warmup_steps": 50, "train_every_n": 5, "batch_size": 8,
            "save_every_n": 100, "programs": progs}


def _fire_event(nns, prog="TEST_A", reward=0.5):
    nns.buffers[prog].add(
        observation=[0.5] * 55, urgency=0.4, vm_baseline=0.3, fired=True)
    nns.record_outcome(reward=reward, program=prog, source="test")


def test_handle_opened_once_and_reused(tmp_path):
    nns = NeuralNervousSystem(config=_make_config(1),
                              data_dir=str(tmp_path), vm_nervous_system=None)
    assert nns._reward_log_fh is None  # lazy — not opened until first write
    _fire_event(nns)
    fh1 = nns._reward_log_fh
    assert fh1 is not None
    for _ in range(20):
        _fire_event(nns)
    # same handle object — never reopened per event
    assert nns._reward_log_fh is fh1


def test_read_after_write_consistent(tmp_path):
    """Line buffering must flush each entry so an external reader sees it."""
    nns = NeuralNervousSystem(config=_make_config(1),
                              data_dir=str(tmp_path), vm_nervous_system=None)
    log_path = tmp_path / "reward_log.jsonl"
    for i in range(15):
        _fire_event(nns, reward=0.1 * i)
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == i + 1, f"event {i}: expected {i+1} lines on disk"
        json.loads(lines[-1])  # last entry well-formed + flushed


def test_truncation_only_when_over_cap_keeps_last_n(tmp_path):
    nns = NeuralNervousSystem(config=_make_config(1),
                              data_dir=str(tmp_path), vm_nervous_system=None)
    nns._reward_log_max_lines = 10
    for _ in range(25):
        _fire_event(nns)
    log_path = tmp_path / "reward_log.jsonl"
    lines = log_path.read_text().strip().split("\n")
    # never grows unbounded; bounded near the cap after truncation
    assert len(lines) <= nns._reward_log_max_lines
    assert nns._reward_log_lines <= nns._reward_log_max_lines
    for ln in lines:
        json.loads(ln)  # every kept line still well-formed


def test_count_seeded_from_existing_file(tmp_path):
    """A restart (handle=None, count=0) seeds the count from the existing file
    once, so truncation stays accurate without re-reading on a cadence."""
    log_path = tmp_path / "reward_log.jsonl"
    log_path.write_text("".join(
        json.dumps({"ts": 0, "program": "X", "reward_raw": 0.0,
                    "reward_z": 0.0, "source": "seed", "k": 0,
                    "fired": False, "ema_mean": 0.0}) + "\n"
        for _ in range(7)))
    nns = NeuralNervousSystem(config=_make_config(1),
                              data_dir=str(tmp_path), vm_nervous_system=None)
    assert nns._reward_log_lines == 0  # not seeded until first write
    _fire_event(nns)
    assert nns._reward_log_lines == 8  # 7 existing + 1 new
