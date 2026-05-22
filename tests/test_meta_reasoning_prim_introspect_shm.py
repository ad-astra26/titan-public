"""
Tests for meta_reasoning._prim_introspect SHM-cache + fire-and-forget
publish migration — Chunks C+D of
rFP_meta_reasoning_self_reasoning_resolver_migration.

SPEC §9.B + D-SPEC-70 v1.15.0. Closes F-8 fleet-wide.

Coverage:
  - Cold-start path (SHM empty) → synthetic placeholder + META_INTROSPECT_REQUEST
    publish
  - Cache-hit path → returns SHM result + populates META-CGN Producer #13/#14
  - Producer dedup (cached_epoch must advance before re-enqueue)
  - maker_alignment branch unchanged (still in-process via TitanMaker)
  - send_queue.put_nowait failure tolerated (no crash; bounded log)
  - Dead setter `set_self_reasoning` retired (grep verifies)
  - `self._self_reasoning` attribute retired (constructor no longer sets it)
  - chi_coh extracted from SHM cache for Producer #14

Reference:
  - titan_hcl/logic/meta_reasoning.py _prim_introspect
  - titan_hcl/logic/meta_reasoning.py MetaReasoningEngine.__init__
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from titan_hcl import bus
from titan_hcl.logic.meta_reasoning import MetaReasoningEngine


@pytest.fixture
def meta_engine():
    """Bare MetaReasoningEngine with a mocked send_queue + reader bank."""
    eng = MetaReasoningEngine(config={"enabled": True}, send_queue=MagicMock())
    eng._inner_self_insight_reader = MagicMock()
    eng._inner_self_insight_reader.read_inner_self_insight = MagicMock(
        return_value=None)
    return eng


def _sv() -> list:
    """132D state vector with mid-range values."""
    return [0.4] * 65 + [0.3] * 65 + [0.5, 0.5]


def _nm() -> dict:
    return {"DA": 0.5, "5HT": 0.4, "NE": 0.3, "ACh": 0.45,
            "Endorphin": 0.2, "GABA": 0.3}


# ── Retired symbols (grep guards) ─────────────────────────────────


def test_set_self_reasoning_symbol_retired():
    """`set_self_reasoning` method must not exist on MetaReasoning post-fix."""
    eng = MetaReasoningEngine(config={"enabled": True})
    assert not hasattr(eng, "set_self_reasoning"), (
        "set_self_reasoning() was retired in D-SPEC-70 v1.15.0 — "
        "remove the method definition"
    )


def test_self_reasoning_attribute_retired():
    """`_self_reasoning` instance attribute must not be set in __init__."""
    eng = MetaReasoningEngine(config={"enabled": True})
    # The old setter assigned self._self_reasoning = engine. Constructor
    # initialized it to None. Both retired in D-SPEC-70.
    assert not hasattr(eng, "_self_reasoning"), (
        "_self_reasoning attribute was retired in D-SPEC-70 v1.15.0 — "
        "constructor no longer sets it (use SHM-bridge pattern)"
    )


def test_inner_self_insight_reader_attribute_present():
    """New attribute `_inner_self_insight_reader` exists post-fix."""
    eng = MetaReasoningEngine(config={"enabled": True})
    assert hasattr(eng, "_inner_self_insight_reader")
    # Constructor sets it to None; cognitive_worker attaches a
    # ShmReaderBank post-construction.
    assert eng._inner_self_insight_reader is None


def test_last_shm_insight_epoch_initialized():
    """New dedup tracker initialized to -1 (never-consumed sentinel)."""
    eng = MetaReasoningEngine(config={"enabled": True})
    assert eng._last_shm_insight_epoch == -1


# ── Cold-start path ───────────────────────────────────────────────


def test_cold_start_returns_synthetic_placeholder(meta_engine):
    """SHM empty → placeholder (matches legacy fallback shape)."""
    meta_engine._inner_self_insight_reader.read_inner_self_insight.return_value = None
    result = meta_engine._prim_introspect("state_audit", _sv(), _nm())
    assert isinstance(result, dict)
    assert result["primitive"] == "INTROSPECT"
    assert result["sub_mode"] == "state_audit"
    assert result["confidence"] == pytest.approx(0.3)  # legacy compat
    assert result.get("cold_start") is True
    assert "cold_start" in result.get("note", "")


def test_cold_start_publishes_request(meta_engine):
    """Cold-start path still publishes META_INTROSPECT_REQUEST for next tick."""
    meta_engine._inner_self_insight_reader.read_inner_self_insight.return_value = None
    meta_engine._prim_introspect("state_audit", _sv(), _nm())

    meta_engine._send_queue.put_nowait.assert_called_once()
    msg = meta_engine._send_queue.put_nowait.call_args.args[0]
    assert msg["type"] == bus.META_INTROSPECT_REQUEST
    assert msg["src"] == "cognitive_worker"
    assert msg["dst"] == "self_reflection_worker"
    payload = msg["payload"]
    assert payload["sub_mode"] == "state_audit"
    assert payload["neuromods"]["DA"] == pytest.approx(0.5)
    assert len(payload["state_132d"]) == 132


def test_no_reader_attached_still_publishes(meta_engine):
    """If reader bank is None (init failure), fire-and-forget still works."""
    meta_engine._inner_self_insight_reader = None
    result = meta_engine._prim_introspect("state_audit", _sv(), _nm())
    assert result.get("cold_start") is True
    meta_engine._send_queue.put_nowait.assert_called_once()


# ── Cache-hit path ────────────────────────────────────────────────


def test_cache_hit_returns_shm_result(meta_engine):
    """SHM has insight → return cached dict directly."""
    cached = {
        "primitive": "INTROSPECT",
        "sub_mode": "state_audit",
        "effective_sub_mode": "coherence_check",
        "confidence": 0.75,
        "mode_trigger": "DA_phasic",
        "inner_avg": 0.5,
        "outer_avg": 0.4,
        "neuromods": {"DA": 0.5},
        "chi_coh": 0.6,
        "epoch": 100,
        "ts": time.time(),
        "cold_start": False,
    }
    meta_engine._inner_self_insight_reader.read_inner_self_insight.return_value = cached
    result = meta_engine._prim_introspect("state_audit", _sv(), _nm())
    assert result == cached


def test_cache_hit_populates_producer_13(meta_engine):
    """Producer #13 reflection_depth queue populated with cached confidence."""
    cached = {
        "confidence": 0.65,
        "effective_sub_mode": "state_audit",
        "epoch": 50,
    }
    meta_engine._inner_self_insight_reader.read_inner_self_insight.return_value = cached
    meta_engine._prim_introspect("state_audit", _sv(), _nm())
    assert len(meta_engine._pending_cgn_reflection_events) == 1
    ev = meta_engine._pending_cgn_reflection_events[0]
    assert ev["sub_mode"] == "state_audit"
    assert ev["confidence"] == pytest.approx(0.65)


def test_cache_hit_populates_producer_14_only_on_coherence_check(meta_engine):
    """Producer #14 coherence_gain fires only when effective_sub_mode is
    'coherence_check'."""
    cached = {
        "confidence": 0.7,
        "effective_sub_mode": "coherence_check",
        "chi_coh": 0.55,
        "epoch": 75,
    }
    meta_engine._inner_self_insight_reader.read_inner_self_insight.return_value = cached
    meta_engine._prim_introspect("state_audit", _sv(), _nm())
    assert len(meta_engine._pending_cgn_coherence_events) == 1
    assert meta_engine._pending_cgn_coherence_events[0]["chi_coh"] == pytest.approx(0.55)


def test_cache_hit_producer_14_skipped_when_not_coherence_check(meta_engine):
    """Producer #14 stays empty for non-coherence_check sub-modes."""
    cached = {
        "confidence": 0.5,
        "effective_sub_mode": "state_audit",
        "chi_coh": 0.5,
        "epoch": 75,
    }
    meta_engine._inner_self_insight_reader.read_inner_self_insight.return_value = cached
    meta_engine._prim_introspect("state_audit", _sv(), _nm())
    assert len(meta_engine._pending_cgn_coherence_events) == 0


# ── Producer dedup ────────────────────────────────────────────────


def test_dedup_skips_same_epoch(meta_engine):
    """Re-reading the same SHM payload (same epoch) does NOT re-enqueue
    Producer #13/#14 events — drainage avoids duplicate META-CGN emissions."""
    cached = {
        "confidence": 0.5,
        "effective_sub_mode": "coherence_check",
        "chi_coh": 0.6,
        "epoch": 200,
    }
    meta_engine._inner_self_insight_reader.read_inner_self_insight.return_value = cached
    # First call — enqueues
    meta_engine._prim_introspect("state_audit", _sv(), _nm())
    assert len(meta_engine._pending_cgn_reflection_events) == 1
    assert len(meta_engine._pending_cgn_coherence_events) == 1
    # Second call (same SHM payload) — skipped (epoch unchanged)
    meta_engine._prim_introspect("state_audit", _sv(), _nm())
    assert len(meta_engine._pending_cgn_reflection_events) == 1
    assert len(meta_engine._pending_cgn_coherence_events) == 1


def test_dedup_enqueues_on_epoch_advance(meta_engine):
    """When SR worker writes a NEW insight (epoch advances), Producer
    queues get fresh entries."""
    cached_a = {
        "confidence": 0.5,
        "effective_sub_mode": "state_audit",
        "epoch": 300,
    }
    cached_b = {
        "confidence": 0.7,
        "effective_sub_mode": "state_audit",
        "epoch": 301,
    }
    meta_engine._inner_self_insight_reader.read_inner_self_insight.return_value = cached_a
    meta_engine._prim_introspect("state_audit", _sv(), _nm())
    meta_engine._inner_self_insight_reader.read_inner_self_insight.return_value = cached_b
    meta_engine._prim_introspect("state_audit", _sv(), _nm())
    assert len(meta_engine._pending_cgn_reflection_events) == 2
    assert meta_engine._pending_cgn_reflection_events[1]["confidence"] == pytest.approx(0.7)


# ── maker_alignment branch unchanged ─────────────────────────────


def test_maker_alignment_branch_unchanged(meta_engine):
    """maker_alignment sub-mode still runs in-process via TitanMaker —
    does NOT consult SHM cache or publish META_INTROSPECT_REQUEST."""
    # TitanMaker not available in test → returns fallback dict
    result = meta_engine._prim_introspect("maker_alignment", _sv(), _nm())
    assert result["sub_mode"] == "maker_alignment"
    # Should NOT have published META_INTROSPECT_REQUEST for this branch
    meta_engine._send_queue.put_nowait.assert_not_called()
    # Should NOT have read SHM
    meta_engine._inner_self_insight_reader.read_inner_self_insight.assert_not_called()


# ── Publish failure tolerance ─────────────────────────────────────


def test_publish_failure_does_not_crash(meta_engine):
    """send_queue.put_nowait raising must not propagate — bounded warning log,
    cold-start placeholder returned."""
    meta_engine._send_queue.put_nowait.side_effect = RuntimeError("queue full")
    meta_engine._inner_self_insight_reader.read_inner_self_insight.return_value = None
    # Should not raise
    result = meta_engine._prim_introspect("state_audit", _sv(), _nm())
    assert result.get("cold_start") is True
    # Error counter bumped
    assert getattr(meta_engine, "_meta_introspect_pub_err_count", 0) >= 1


def test_no_send_queue_still_returns_placeholder():
    """If MetaReasoning was constructed without send_queue, _prim_introspect
    still returns a placeholder (no publish attempted)."""
    eng = MetaReasoningEngine(config={"enabled": True}, send_queue=None)
    eng._inner_self_insight_reader = MagicMock()
    eng._inner_self_insight_reader.read_inner_self_insight = MagicMock(
        return_value=None)
    result = eng._prim_introspect("state_audit", _sv(), _nm())
    assert result.get("cold_start") is True


# ── Source-code grep guard ────────────────────────────────────────


def test_meta_reasoning_source_has_no_set_self_reasoning():
    """Source grep: zero `set_self_reasoning` / `_self_reasoning` references
    in meta_reasoning.py (other than comments documenting the retirement)."""
    import pathlib
    src = pathlib.Path("titan_hcl/logic/meta_reasoning.py").read_text()
    # Find lines with the symbol — exclude comment lines (lines whose
    # first non-whitespace char is '#').
    bad_lines = []
    for ln_no, line in enumerate(src.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue  # comment — documentation of retirement is allowed
        # In-line strings + docstrings: the strict-but-fair check is
        # "does this line, outside of comments, REFERENCE the retired
        # symbol via attribute access / definition?"
        if "self._self_reasoning" in line or "def set_self_reasoning" in line:
            bad_lines.append((ln_no, line.strip()))
    assert not bad_lines, (
        "Retired symbols still referenced (non-comment) in meta_reasoning.py:\n"
        + "\n".join(f"  L{n}: {t}" for n, t in bad_lines)
    )
