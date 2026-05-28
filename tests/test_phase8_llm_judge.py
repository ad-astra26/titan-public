"""Phase 8 — LLMJudge unit tests (D-SPEC-PHASE8 / INV-Syn-21).

Covers:
- score_tx parses {verdict, rationale, confidence}
- score_tx returns None on malformed / unknown verdict
- score_tx returns None on provider exception / timeout
- version_tag stable across calls; changes with model_id
- score_window filters to scored_by IS NULL
- score_window respects per_pass_cap (most-recent first)
- score_window anchors batch + scored_by patch via writer
- score_window empty window — batch still anchored (auditable "no work" record)
- Merkle root deterministic
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from titan_hcl.synthesis.llm_judge import (
    DEFAULT_PER_PASS_CAP,
    LLMJudge,
    _merkle_root,
    _parse_verdict,
    _prompt_version_tag,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_tx(*, tx_hash: str, scored_by: str | None = None, ts: float = 1.0,
             tool: str = "x", success: bool = True) -> dict:
    return {
        "tx_hash": tx_hash,
        "content": {
            "tool_id": tool, "args": {}, "success": success,
            "scored_by": scored_by, "ts": ts,
            "result_summary": "ok",
        },
    }


def _llm_returning(json_or_text: str):
    def f(prompt, timeout):
        return json_or_text
    return f


# ── _parse_verdict ─────────────────────────────────────────────────────


def test_parse_verdict_valid_full():
    out = _parse_verdict('{"verdict": "full", "rationale": "did the thing", "confidence": 0.9}')
    assert out == {"verdict": "full", "rationale": "did the thing", "confidence": 0.9}


def test_parse_verdict_valid_with_prose_wrap():
    raw = 'Sure, here it is: {"verdict": "partial", "rationale": "halfway", "confidence": 0.5} done.'
    out = _parse_verdict(raw)
    assert out["verdict"] == "partial"


def test_parse_verdict_unknown_verdict_rejected():
    assert _parse_verdict('{"verdict": "maybe", "rationale": "..."}') is None


def test_parse_verdict_malformed_returns_none():
    assert _parse_verdict("not json at all") is None
    assert _parse_verdict("") is None
    assert _parse_verdict(None) is None


def test_parse_verdict_clamps_confidence():
    out = _parse_verdict('{"verdict": "full", "rationale": "ok", "confidence": 2.5}')
    assert out["confidence"] == 1.0
    out2 = _parse_verdict('{"verdict": "full", "rationale": "ok", "confidence": -1}')
    assert out2["confidence"] == 0.0


# ── _prompt_version_tag ────────────────────────────────────────────────


def test_version_tag_stable_per_model():
    assert _prompt_version_tag("modelA") == _prompt_version_tag("modelA")


def test_version_tag_differs_by_model():
    assert _prompt_version_tag("modelA") != _prompt_version_tag("modelB")


def test_version_tag_format_model_pipe_hash():
    tag = _prompt_version_tag("modelX")
    assert tag.startswith("modelX|")
    assert len(tag.split("|")[1]) == 12  # 12-char hash prefix


# ── _merkle_root ───────────────────────────────────────────────────────


def test_merkle_root_empty_is_sha256_empty():
    import hashlib
    assert _merkle_root([]) == hashlib.sha256(b"").hexdigest()


def test_merkle_root_deterministic():
    leaves = [b"a", b"b", b"c"]
    assert _merkle_root(leaves) == _merkle_root(leaves)


def test_merkle_root_different_inputs_differ():
    assert _merkle_root([b"a", b"b"]) != _merkle_root([b"a", b"c"])


# ── score_tx ────────────────────────────────────────────────────────────


def test_score_tx_returns_parsed_verdict():
    judge = LLMJudge(
        tool_call_reader=lambda *_: [],
        llm_provider=_llm_returning('{"verdict": "full", "rationale": "ok", "confidence": 1.0}'),
        outer_memory_writer=MagicMock(),
    )
    tx = _make_tx(tx_hash="tx1")
    out = judge.score_tx(tx)
    assert out["verdict"] == "full"
    assert out["version_tag"].startswith("ollama_cloud_deepseek|")


def test_score_tx_returns_none_on_provider_exception():
    def boom(prompt, timeout):
        raise RuntimeError("network down")
    judge = LLMJudge(
        tool_call_reader=lambda *_: [],
        llm_provider=boom,
        outer_memory_writer=MagicMock(),
    )
    assert judge.score_tx(_make_tx(tx_hash="t")) is None


def test_score_tx_returns_none_on_unparseable():
    judge = LLMJudge(
        tool_call_reader=lambda *_: [],
        llm_provider=_llm_returning("not json"),
        outer_memory_writer=MagicMock(),
    )
    assert judge.score_tx(_make_tx(tx_hash="t")) is None


# ── score_window ───────────────────────────────────────────────────────


def test_score_window_filters_to_unscored():
    txs = [
        _make_tx(tx_hash="A", scored_by=None, ts=1),
        _make_tx(tx_hash="B", scored_by="oracle", ts=2),  # skipped — already scored
        _make_tx(tx_hash="C", scored_by=None, ts=3),
    ]
    writer = MagicMock()
    writer.write_llm_judge_batch = MagicMock(return_value="batch_tx_h")
    writer.write_scored_by_patch = MagicMock(return_value="patch_tx_h")
    judge = LLMJudge(
        tool_call_reader=lambda since, lim: txs,
        llm_provider=_llm_returning('{"verdict":"full","rationale":"ok","confidence":1.0}'),
        outer_memory_writer=writer,
    )
    summary = judge.score_window(since_ts=0.0)
    assert summary["tool_calls_in_window"] == 3
    assert summary["unscored_in_window"] == 2
    assert summary["scored_now"] == 2
    assert summary["llm_failures"] == 0


def test_score_window_respects_per_pass_cap():
    txs = [_make_tx(tx_hash=f"t{i}", scored_by=None, ts=float(i)) for i in range(10)]
    writer = MagicMock()
    writer.write_llm_judge_batch = MagicMock(return_value="batch")
    writer.write_scored_by_patch = MagicMock(return_value="patch")
    judge = LLMJudge(
        tool_call_reader=lambda since, lim: txs,
        llm_provider=_llm_returning('{"verdict":"full","rationale":"ok","confidence":1.0}'),
        outer_memory_writer=writer,
        per_pass_cap=3,
    )
    summary = judge.score_window(since_ts=0.0)
    assert summary["scored_now"] == 3


def test_score_window_most_recent_first():
    """When cap=2 and 5 unscored TXs, judge should pick the 2 with highest ts."""
    txs = [_make_tx(tx_hash=f"t{i}", scored_by=None, ts=float(i)) for i in range(5)]
    writer = MagicMock()
    writer.write_llm_judge_batch = MagicMock(return_value="batch")
    writer.write_scored_by_patch = MagicMock(return_value="patch")
    judge = LLMJudge(
        tool_call_reader=lambda since, lim: txs,
        llm_provider=_llm_returning('{"verdict":"full","rationale":"ok","confidence":1.0}'),
        outer_memory_writer=writer,
        per_pass_cap=2,
    )
    judge.score_window(since_ts=0.0)
    # write_llm_judge_batch entries should be the most-recent 2: t4, t3
    call_args = writer.write_llm_judge_batch.call_args
    entries = call_args.kwargs["entries"]
    scored_hashes = {e["parent_tool_call_tx"] for e in entries}
    assert scored_hashes == {"t4", "t3"}


def test_score_window_anchors_batch_even_when_empty():
    """No unscored TXs → batch still anchored ('no work' is auditable)."""
    writer = MagicMock()
    writer.write_llm_judge_batch = MagicMock(return_value="batch_empty")
    writer.write_scored_by_patch = MagicMock(return_value="patch_empty")
    judge = LLMJudge(
        tool_call_reader=lambda *_: [],
        llm_provider=_llm_returning("not used"),
        outer_memory_writer=writer,
    )
    summary = judge.score_window(since_ts=0.0)
    assert summary["scored_now"] == 0
    assert summary["batch_tx_hash"] == "batch_empty"
    # No scored pairs → patch is not anchored
    writer.write_scored_by_patch.assert_not_called()


def test_score_window_anchors_scored_by_patch():
    txs = [_make_tx(tx_hash="x", scored_by=None, ts=1.0)]
    writer = MagicMock()
    writer.write_llm_judge_batch = MagicMock(return_value="batch_h")
    writer.write_scored_by_patch = MagicMock(return_value="patch_h")
    judge = LLMJudge(
        tool_call_reader=lambda *_: txs,
        llm_provider=_llm_returning('{"verdict":"full","rationale":"ok","confidence":1.0}'),
        outer_memory_writer=writer,
    )
    summary = judge.score_window(since_ts=0.0)
    assert summary["scored_now"] == 1
    writer.write_scored_by_patch.assert_called_once()
    entries = writer.write_scored_by_patch.call_args.kwargs["entries"]
    assert entries == [{"parent_tool_call_tx": "x", "scored_by": "llm"}]


def test_score_window_handles_writer_failure_gracefully():
    """write_llm_judge_batch raising should be logged + summary still returned."""
    txs = [_make_tx(tx_hash="x", scored_by=None, ts=1.0)]
    writer = MagicMock()
    writer.write_llm_judge_batch = MagicMock(side_effect=RuntimeError("disk full"))
    writer.write_scored_by_patch = MagicMock(return_value="patch_h")
    judge = LLMJudge(
        tool_call_reader=lambda *_: txs,
        llm_provider=_llm_returning('{"verdict":"full","rationale":"ok","confidence":1.0}'),
        outer_memory_writer=writer,
    )
    # Should NOT raise
    summary = judge.score_window(since_ts=0.0)
    assert summary["batch_tx_hash"] is None
    assert summary["scored_now"] == 1


def test_score_window_carries_version_tag_per_entry():
    txs = [_make_tx(tx_hash="x", scored_by=None, ts=1.0)]
    writer = MagicMock()
    writer.write_llm_judge_batch = MagicMock(return_value="batch_h")
    writer.write_scored_by_patch = MagicMock(return_value="patch_h")
    judge = LLMJudge(
        tool_call_reader=lambda *_: txs,
        llm_provider=_llm_returning('{"verdict":"failure","rationale":"timeout","confidence":0.7}'),
        outer_memory_writer=writer,
        model_id="testmodel",
    )
    judge.score_window(since_ts=0.0)
    entries = writer.write_llm_judge_batch.call_args.kwargs["entries"]
    assert entries[0]["version_tag"].startswith("testmodel|")
    assert entries[0]["verdict"] == "failure"


def test_score_window_emits_pass_done_event():
    emitted = []
    writer = MagicMock()
    writer.write_llm_judge_batch = MagicMock(return_value="batch_h")
    writer.write_scored_by_patch = MagicMock(return_value="patch_h")
    judge = LLMJudge(
        tool_call_reader=lambda *_: [],
        llm_provider=_llm_returning("unused"),
        outer_memory_writer=writer,
        bus_emit=lambda ev, p: emitted.append((ev, p)),
    )
    judge.score_window(since_ts=0.0)
    assert any(ev == "META_LLM_JUDGE_PASS_DONE" for ev, _ in emitted)
