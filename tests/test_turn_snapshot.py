"""Tests — synthesis.turn_snapshot (Phase 3 P3.A).

Covers `capture_turn_snapshot` (felt-state at chat-turn time) and
`extract_tool_calls` (agno RunOutput.tools → arch §7 §8.2 shape).

Both helpers MUST soft-fail to empty/zero values — no exception
reaches the caller. OVG.build_timechain_payload runs in the
chat-hot-path and must never raise.
"""
from __future__ import annotations

import hashlib
import json
import struct
import unittest
from dataclasses import dataclass, field
from typing import Optional, Any

from titan_hcl.synthesis.turn_snapshot import (
    capture_turn_snapshot,
    extract_tool_calls,
    DEFAULT_IMPORTANCE,
    UNIFIED_SPIRIT_DIM,
    _hash_132d,
)


# ─────────────────────────────────────────────────────────────────────
# capture_turn_snapshot — felt-state extraction
# ─────────────────────────────────────────────────────────────────────


class _FakeReaderBank:
    """Stub ShmReaderBank — only the two methods turn_snapshot reads."""

    def __init__(self, *, neuromod=None, trinity=None, raise_neuromod=False,
                 raise_trinity=False):
        self._neuromod = neuromod
        self._trinity = trinity
        self._raise_nm = raise_neuromod
        self._raise_tr = raise_trinity

    def read_neuromod(self):
        if self._raise_nm:
            raise RuntimeError("simulated SHM read failure")
        return self._neuromod

    def read_trinity(self):
        if self._raise_tr:
            raise RuntimeError("simulated trinity SHM read failure")
        return self._trinity


class TestCaptureTurnSnapshot(unittest.TestCase):

    def test_empty_reader_returns_safe_defaults(self):
        """No SHM data → empty neuromods, empty hash, default importance."""
        bank = _FakeReaderBank(neuromod=None, trinity=None)
        snap = capture_turn_snapshot(bank)
        self.assertEqual(snap["neuromods"], {})
        self.assertEqual(snap["embedding_hash"], "")
        self.assertEqual(snap["importance"], DEFAULT_IMPORTANCE)

    def test_neuromods_extracted_into_name_level_dict(self):
        """ShmReaderBank.read_neuromod returns the documented nested shape."""
        bank = _FakeReaderBank(neuromod={
            "modulators": {
                "DA":        {"level": 0.6},
                "5HT":       {"level": 0.4},
                "NE":        {"level": 0.5},
                "ACh":       {"level": 0.55},
                "Endorphin": {"level": 0.3},
                "GABA":      {"level": 0.2},
            },
            "age_seconds": 1.2, "seq": 100,
        })
        snap = capture_turn_snapshot(bank)
        self.assertEqual(snap["neuromods"], {
            "DA": 0.6, "5HT": 0.4, "NE": 0.5, "ACh": 0.55,
            "Endorphin": 0.3, "GABA": 0.2,
        })

    def test_embedding_hash_is_sha256_of_132_floats(self):
        """132D = trinity full_130dt (130) + journey [curvature, density] (2)."""
        full_130 = [0.1 * i for i in range(130)]
        bank = _FakeReaderBank(trinity={
            "full_130dt": full_130,
            "journey": {"curvature": 0.7, "density": 0.3},
            "age_seconds": 0.5, "seq": 1,
        })
        snap = capture_turn_snapshot(bank)
        # Recompute expected hash deterministically.
        vec = full_130 + [0.7, 0.3]
        buf = struct.pack(f"<{UNIFIED_SPIRIT_DIM}f", *vec)
        expected = hashlib.sha256(buf).hexdigest()
        self.assertEqual(snap["embedding_hash"], expected)
        self.assertEqual(len(snap["embedding_hash"]), 64)

    def test_embedding_hash_changes_with_state(self):
        """Different 132D vectors → different hashes."""
        vec_a = [0.1] * 130
        vec_b = [0.2] * 130
        bank_a = _FakeReaderBank(trinity={
            "full_130dt": vec_a, "journey": {"curvature": 0, "density": 0}})
        bank_b = _FakeReaderBank(trinity={
            "full_130dt": vec_b, "journey": {"curvature": 0, "density": 0}})
        self.assertNotEqual(
            capture_turn_snapshot(bank_a)["embedding_hash"],
            capture_turn_snapshot(bank_b)["embedding_hash"])

    def test_partial_trinity_skips_hash(self):
        """Wrong-shape trinity → empty hash, no exception."""
        bank = _FakeReaderBank(trinity={
            "full_130dt": [0.1, 0.2],  # only 2 elements
            "journey": {"curvature": 0.5, "density": 0.5},
        })
        snap = capture_turn_snapshot(bank)
        self.assertEqual(snap["embedding_hash"], "")

    def test_neuromod_read_raise_returns_empty_dict(self):
        """SHM read raises → empty neuromods, no exception bubbles."""
        bank = _FakeReaderBank(raise_neuromod=True,
                               trinity={"full_130dt": [0.0] * 130,
                                        "journey": {"curvature": 0, "density": 0}})
        snap = capture_turn_snapshot(bank)
        self.assertEqual(snap["neuromods"], {})
        # Trinity still extracted.
        self.assertEqual(len(snap["embedding_hash"]), 64)

    def test_trinity_read_raise_returns_empty_hash(self):
        """SHM read raises → empty hash, no exception bubbles."""
        bank = _FakeReaderBank(
            neuromod={"modulators": {"DA": {"level": 0.5}}, "age_seconds": 1, "seq": 1},
            raise_trinity=True)
        snap = capture_turn_snapshot(bank)
        self.assertEqual(snap["embedding_hash"], "")
        # Neuromods still extracted.
        self.assertEqual(snap["neuromods"], {"DA": 0.5})

    def test_no_reader_bank_arg_constructs_real_one(self):
        """When called with no arg, falls back to ShmReaderBank() —
        which may fail in test env (no SHM). Either way: returns safe
        defaults, no exception."""
        snap = capture_turn_snapshot()
        # Stable keys present regardless of SHM availability.
        self.assertIn("neuromods", snap)
        self.assertIn("embedding_hash", snap)
        self.assertIn("importance", snap)
        self.assertEqual(snap["importance"], DEFAULT_IMPORTANCE)


# ─────────────────────────────────────────────────────────────────────
# _hash_132d — direct unit tests
# ─────────────────────────────────────────────────────────────────────


class TestHash132D(unittest.TestCase):

    def test_correct_length_hashes(self):
        h = _hash_132d([0.0] * UNIFIED_SPIRIT_DIM)
        self.assertEqual(len(h), 64)
        # Hex digits only.
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_wrong_length_returns_empty(self):
        self.assertEqual(_hash_132d([]), "")
        self.assertEqual(_hash_132d([0.0] * 50), "")
        self.assertEqual(_hash_132d([0.0] * 200), "")

    def test_deterministic_across_calls(self):
        v = [0.1] * UNIFIED_SPIRIT_DIM
        self.assertEqual(_hash_132d(v), _hash_132d(v))


# ─────────────────────────────────────────────────────────────────────
# extract_tool_calls — agno ToolExecution → §7 §8.2 shape
# ─────────────────────────────────────────────────────────────────────


@dataclass
class _StubToolCallMetrics:
    """Stand-in for agno's ToolCallMetrics — only `time` is read."""
    time: float = 0.0


@dataclass
class _StubToolExecution:
    """Stand-in for agno's ToolExecution dataclass."""
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    result: Optional[str] = None
    metrics: Optional[Any] = None
    tool_call_error: Optional[bool] = None


class TestExtractToolCalls(unittest.TestCase):

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(extract_tool_calls(None), [])
        self.assertEqual(extract_tool_calls([]), [])

    def test_single_tool_normalized_to_section_7_shape(self):
        tx = _StubToolExecution(
            tool_name="web_search",
            tool_args={"query": "metaplex bug"},
            result='{"hits": 3}',
            metrics=_StubToolCallMetrics(time=0.420),
            tool_call_error=False,
        )
        out = extract_tool_calls([tx])
        self.assertEqual(len(out), 1)
        entry = out[0]
        self.assertEqual(entry["tool"], "web_search")
        # args_hash is sha256 of canonical JSON
        expected_args_hash = hashlib.sha256(
            json.dumps({"query": "metaplex bug"},
                       sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(entry["args_hash"], expected_args_hash)
        # result_hash is sha256 of result string
        expected_result_hash = hashlib.sha256(
            '{"hits": 3}'.encode()).hexdigest()
        self.assertEqual(entry["result_hash"], expected_result_hash)
        self.assertEqual(entry["latency_ms"], 420)
        self.assertFalse(entry["exception"])

    def test_missing_args_yields_empty_args_hash(self):
        tx = _StubToolExecution(
            tool_name="noop", tool_args=None, result="ok",
            metrics=None, tool_call_error=False)
        out = extract_tool_calls([tx])
        self.assertEqual(out[0]["args_hash"], "")
        # result_hash still computed.
        self.assertEqual(
            out[0]["result_hash"],
            hashlib.sha256(b"ok").hexdigest())

    def test_exception_flag_passthrough(self):
        tx = _StubToolExecution(
            tool_name="bad_tool", tool_args={"x": 1},
            result=None, tool_call_error=True)
        out = extract_tool_calls([tx])
        self.assertTrue(out[0]["exception"])

    def test_malformed_entry_is_skipped_not_raised(self):
        """One bad entry doesn't kill the list — others survive."""
        class _Broken:
            @property
            def tool_name(self):
                raise RuntimeError("kaboom")
        good = _StubToolExecution(
            tool_name="ok", tool_args={}, result="x",
            metrics=None, tool_call_error=False)
        out = extract_tool_calls([_Broken(), good])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["tool"], "ok")

    def test_non_dict_args_falls_back_to_repr_hash(self):
        """JSON-unfriendly args still produce a deterministic hash."""
        class _Weird:
            def __repr__(self):
                return "<weird>"
        tx = _StubToolExecution(
            tool_name="x", tool_args={"obj": _Weird()},
            result="", tool_call_error=False)
        out = extract_tool_calls([tx])
        # args_hash non-empty + 64-char hex
        self.assertEqual(len(out[0]["args_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
