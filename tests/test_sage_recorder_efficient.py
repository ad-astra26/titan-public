"""Unit tests for SageRecorder efficiency fixes (2026-04-27, ARCHITECTURAL).

Closes BUG-SAGE-INSTANTIATED-IN-PARENT (partial). Verifies:
  - Default buffer capacity dropped 1_000_000 → 50_000.
  - SentenceTransformer is NOT loaded at __init__ (lazy property).
  - Lazy `action_embedder` returns model on first access (or None if
    sentence_transformers is unavailable in the test env).
  - On-disk capacity migration: a buffer written at higher capacity
    is rebuilt smaller on next __init__, preserving recorded records.

These tests run in tmp_path so they don't touch production data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Skip the whole module if torch/torchrl are unavailable (some CI lanes).
torch = pytest.importorskip("torch")
pytest.importorskip("torchrl")

from titan_plugin.core.sage.recorder import (  # noqa: E402
    DEFAULT_BUFFER_SIZE,
    SageRecorder,
    _LAZY_SENTINEL,
)


def test_default_buffer_size_is_50k():
    """Default lowered from 1_000_000 to 50_000 (architectural fix)."""
    assert DEFAULT_BUFFER_SIZE == 50_000, (
        "Default buffer size must be 50_000 — was 1_000_000 pre-fix. "
        "Lower defaults reduce parent RSS by ~95% (0.17% utilization "
        "observed on T1)."
    )


def test_recorder_uses_default_when_no_config(tmp_path):
    """SageRecorder() with no config picks DEFAULT_BUFFER_SIZE."""
    cfg = {"sage_memory": {"storage_path": str(tmp_path / "sage_a")}}
    r = SageRecorder(cfg)
    assert r.buffer_size == DEFAULT_BUFFER_SIZE


def test_recorder_honors_explicit_config(tmp_path):
    """Explicit `buffer_size` config wins over the default."""
    cfg = {"sage_memory": {
        "buffer_size": 100,
        "storage_path": str(tmp_path / "sage_b"),
    }}
    r = SageRecorder(cfg)
    assert r.buffer_size == 100


def test_action_embedder_is_lazy_at_init(tmp_path):
    """Constructor MUST NOT load SentenceTransformer.

    Pre-fix: ~80-150MB model loaded into parent process at boot,
    even when never used. Post-fix: only loaded on first attribute
    access. We probe the cache directly (private attribute) to
    verify it's the sentinel, not a model instance.
    """
    cfg = {"sage_memory": {"storage_path": str(tmp_path / "sage_c")}}
    r = SageRecorder(cfg)
    assert r._action_embedder_cache is _LAZY_SENTINEL, (
        "action_embedder must NOT be loaded at __init__; cache should "
        "still be the lazy sentinel."
    )


def test_action_embedder_loads_on_access_or_returns_none(tmp_path):
    """First access to `action_embedder` either loads the model
    or caches None (if sentence_transformers is unavailable)."""
    cfg = {"sage_memory": {"storage_path": str(tmp_path / "sage_d")}}
    r = SageRecorder(cfg)
    assert r._action_embedder_cache is _LAZY_SENTINEL
    embedder = r.action_embedder
    # Cache is no longer the sentinel after access
    assert r._action_embedder_cache is not _LAZY_SENTINEL
    # The two readings agree
    assert r.action_embedder is embedder
    # Cached value is either a SentenceTransformer or None
    if embedder is not None:
        assert hasattr(embedder, "encode"), (
            "If a model loaded, it should expose .encode()")


def test_action_embedder_caches_failure(tmp_path, monkeypatch):
    """If load fails, the property caches None and does NOT retry."""
    cfg = {"sage_memory": {"storage_path": str(tmp_path / "sage_e")}}
    r = SageRecorder(cfg)
    # Force a synthetic ImportError on the first lookup
    fake_mod = type(sys)("sentence_transformers")

    def _raiser(*args, **kwargs):
        raise ImportError("synthetic missing dep")

    fake_mod.SentenceTransformer = _raiser
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    # First access — cached as None
    assert r.action_embedder is None
    assert r._action_embedder_cache is None
    # Second access — must NOT retry import (still None, no exception
    # raised — `_raiser` would error if called again).
    assert r.action_embedder is None


def test_buffer_capacity_migration_preserves_records(tmp_path):
    """A buffer written at higher capacity migrates to the configured
    smaller capacity, preserving recorded records."""
    storage_path = str(tmp_path / "sage_f")

    # Step 1: build a recorder at HIGH capacity, add a few records, dump.
    high_cfg = {"sage_memory": {
        "buffer_size": 1000,
        "storage_path": storage_path,
    }}
    r_high = SageRecorder(high_cfg)
    assert r_high.buffer is not None
    # Add 3 minimal records via direct buffer.add (bypassing record_transition
    # to keep this test independent of action_embedder availability).
    from tensordict import TensorDict
    for i in range(3):
        td = TensorDict({
            "observation": torch.zeros(128, dtype=torch.float32),
            "action_intent_vector": torch.zeros(128, dtype=torch.float32),
            "action_text_bytes": torch.zeros(1, 256, dtype=torch.float32),
            "reward": torch.tensor([float(i)], dtype=torch.float32),
            "research": TensorDict({
                "research_used": torch.tensor([False], dtype=torch.bool),
                "transition_id": torch.tensor([i], dtype=torch.int64),
            }, batch_size=[]),
            "trauma": TensorDict({
                "is_violation": torch.tensor([False], dtype=torch.bool),
                "directive_id": torch.tensor([-1], dtype=torch.int32),
                "trauma_score": torch.tensor([0.0], dtype=torch.float32),
                "guardian_veto_logic": torch.zeros(1, 256, dtype=torch.float32),
                "execution_mode": torch.zeros(1, 32, dtype=torch.float32),
            }, batch_size=[]),
            "timestamp": torch.tensor([float(i)], dtype=torch.float32),
        }, batch_size=[])
        r_high.buffer.add(td)
    r_high.buffer.dumps(storage_path)
    assert len(r_high.buffer) == 3
    # Drop refs so the mmap files release.
    del r_high.buffer
    del r_high.storage
    del r_high

    # Step 2: re-init at LOW capacity. Migration should kick in.
    low_cfg = {"sage_memory": {
        "buffer_size": 50,
        "storage_path": storage_path,
    }}
    r_low = SageRecorder(low_cfg)
    assert r_low.buffer is not None
    assert r_low.buffer_size == 50, "should report new (target) capacity"
    assert len(r_low.buffer) == 3, (
        f"migration must preserve all 3 records, got {len(r_low.buffer)}")
