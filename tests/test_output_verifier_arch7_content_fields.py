"""arch §7 conformance — `build_timechain_payload` Phase 3 content carry.

Phase 3 (D-SPEC-127). Verifies the pass-path conversation-fork TX
content now carries the full arch §7 normative shape:

    user_msg, agent_response, tool_calls[], neuromods{},
    embedding_hash (132D), importance, topic_tags[]

plus the Phase 2 closure fields (chat_id, user_id_hash, turn_index,
output_hash, prompt_hash, signature, channel, checks, violation_type,
titan_id) — both pre-P3 and P3-NEW must be present.

Also verifies tool_calls cap + non-dict defensive coercion +
backward-compat (callers passing no P3 kwargs get safe defaults).
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from titan_hcl.logic.output_verifier import OutputVerifier, OVGResult
from titan_hcl.synthesis import user_id_hash as uih


class _OVGStub:
    """Just enough OutputVerifier to test build_timechain_payload in
    isolation."""

    def __init__(self) -> None:
        self._titan_id = "T_TEST"

    # P3 cap constant pulled from the production class.
    _TOOL_CALLS_PER_TURN_CAP = OutputVerifier._TOOL_CALLS_PER_TURN_CAP
    build_timechain_payload = OutputVerifier.build_timechain_payload


def _pass_result(channel: str = "chat") -> OVGResult:
    return OVGResult(
        passed=True,
        output_text="hello world",
        signature="sig_xyz",
        block_height=1,
        merkle_root="abc",
        checks={"directives": True, "injection": True},
        violations=[],
        violation_type="none",
        channel=channel,
        timestamp=1.0,
    )


class TestArchSection7ContentFields(unittest.TestCase):

    def setUp(self):
        self.ov = _OVGStub()
        self._patcher = patch.object(uih, "_salt_cache", b"\xde\xad\xbe\xef" * 8)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        uih.clear_cache()

    # ── P3 content carry is present ──────────────────────────────

    def test_user_msg_inlined(self):
        payload = self.ov.build_timechain_payload(
            _pass_result(), prompt_text="who am I?",
            user_id="maker", chat_id="s1", turn_index=3)
        self.assertEqual(payload["content"]["user_msg"], "who am I?")

    def test_agent_response_inlined(self):
        payload = self.ov.build_timechain_payload(
            _pass_result(), prompt_text="ping",
            user_id="maker", chat_id="s1")
        self.assertEqual(payload["content"]["agent_response"], "hello world")

    def test_tool_calls_carried_as_list(self):
        tcs = [
            {"tool": "web_search", "args_hash": "a", "result_hash": "r",
             "latency_ms": 200, "exception": False},
            {"tool": "math_eval", "args_hash": "b", "result_hash": "s",
             "latency_ms": 10, "exception": False},
        ]
        payload = self.ov.build_timechain_payload(
            _pass_result(), prompt_text="hi",
            user_id="maker", chat_id="s1",
            tool_calls=tcs)
        self.assertEqual(payload["content"]["tool_calls"], tcs)

    def test_neuromods_carried_as_dict(self):
        nm = {"DA": 0.6, "5HT": 0.4, "NE": 0.5,
              "ACh": 0.55, "Endorphin": 0.3, "GABA": 0.2}
        payload = self.ov.build_timechain_payload(
            _pass_result(), prompt_text="hi",
            user_id="maker", chat_id="s1",
            neuromods=nm)
        self.assertEqual(payload["content"]["neuromods"], nm)

    def test_embedding_hash_carried_as_string(self):
        h = "f" * 64
        payload = self.ov.build_timechain_payload(
            _pass_result(), prompt_text="hi",
            user_id="maker", chat_id="s1",
            embedding_hash=h)
        self.assertEqual(payload["content"]["embedding_hash"], h)

    def test_importance_carried_as_float(self):
        payload = self.ov.build_timechain_payload(
            _pass_result(), prompt_text="hi",
            user_id="maker", chat_id="s1",
            importance=0.85)
        self.assertAlmostEqual(payload["content"]["importance"], 0.85)

    def test_topic_tags_carried_in_content_AND_tags(self):
        """topic_tags appear BOTH in tags list (auto-sidechain trigger)
        AND in content (self-describing payload for retrieval)."""
        payload = self.ov.build_timechain_payload(
            _pass_result(), prompt_text="hi",
            user_id="maker", chat_id="s1",
            topic_tags=["topic:solana", "topic:kuzu"])
        # tags list
        self.assertIn("topic:solana", payload["tags"])
        self.assertIn("topic:kuzu", payload["tags"])
        # content carry
        self.assertEqual(payload["content"]["topic_tags"],
                         ["topic:solana", "topic:kuzu"])

    # ── Defaults preserve byte-compatible behavior ────────────────

    def test_no_p3_kwargs_yields_safe_defaults(self):
        """Pre-P3 callers passing no new kwargs get explicit empty
        defaults — content shape is a strict superset of the P2 shape."""
        payload = self.ov.build_timechain_payload(
            _pass_result(), prompt_text="hi",
            user_id="maker", chat_id="s1", turn_index=0)
        c = payload["content"]
        self.assertEqual(c["user_msg"], "hi")
        self.assertEqual(c["agent_response"], "hello world")
        self.assertEqual(c["tool_calls"], [])
        self.assertEqual(c["neuromods"], {})
        self.assertEqual(c["embedding_hash"], "")
        self.assertAlmostEqual(c["importance"], 0.5)
        self.assertEqual(c["topic_tags"], [])

    def test_p2_fields_kept_alongside_p3(self):
        """Phase 2 closure fields MUST still be present."""
        payload = self.ov.build_timechain_payload(
            _pass_result(), prompt_text="hi",
            user_id="maker", chat_id="s1", turn_index=7)
        c = payload["content"]
        self.assertIn("chat_id", c)
        self.assertIn("user_id_hash", c)
        self.assertIn("turn_index", c)
        self.assertIn("output_hash", c)
        self.assertIn("prompt_hash", c)
        self.assertIn("signature", c)
        self.assertIn("channel", c)
        self.assertIn("checks", c)
        self.assertIn("violation_type", c)
        self.assertIn("titan_id", c)
        self.assertEqual(c["turn_index"], 7)

    # ── tool_calls cap + defensive coercion ──────────────────────

    def test_tool_calls_capped_at_class_constant(self):
        cap = OutputVerifier._TOOL_CALLS_PER_TURN_CAP
        huge = [{"tool": f"t{i}"} for i in range(cap * 2)]
        payload = self.ov.build_timechain_payload(
            _pass_result(), prompt_text="hi",
            user_id="maker", chat_id="s1",
            tool_calls=huge)
        self.assertEqual(len(payload["content"]["tool_calls"]), cap)

    def test_non_dict_tool_call_entries_skipped(self):
        """Defensive — caller could be a misconfigured agno; OVG must
        not crash on garbage entries."""
        mixed = [
            {"tool": "a"},
            "not-a-dict",
            None,
            42,
            {"tool": "b"},
        ]
        payload = self.ov.build_timechain_payload(
            _pass_result(), prompt_text="hi",
            user_id="maker", chat_id="s1",
            tool_calls=mixed)
        self.assertEqual(
            payload["content"]["tool_calls"],
            [{"tool": "a"}, {"tool": "b"}])

    def test_neuromods_copy_isolates_caller_dict(self):
        """Payload must not alias the caller's dict (mutation safety)."""
        nm = {"DA": 0.5}
        payload = self.ov.build_timechain_payload(
            _pass_result(), prompt_text="hi",
            user_id="maker", chat_id="s1",
            neuromods=nm)
        nm["DA"] = 0.99
        self.assertEqual(payload["content"]["neuromods"]["DA"], 0.5)

    # ── Blocked path is unchanged (no §7 content carry) ───────────

    def test_blocked_path_unchanged_no_p3_fields(self):
        """Blocked TXs route to meta fork as security alerts — no
        §7 content carry (those fields are conversation-only)."""
        result = OVGResult(
            passed=False, output_text="",
            signature="",
            guard_message="[BLOCKED]",
            violation_type="directive",
            violations=["jailbreak attempt"],
            checks={}, channel="chat", timestamp=1.0,
        )
        payload = self.ov.build_timechain_payload(
            result, prompt_text="hi",
            user_id="maker", chat_id="s1",
            tool_calls=[{"tool": "x"}],
            neuromods={"DA": 1.0},
            embedding_hash="ff" * 32,
            importance=0.9)
        self.assertEqual(payload["fork"], "meta")
        self.assertEqual(payload["content"]["event"], "OVG_BLOCKED")
        # P3 fields NOT carried on blocked path.
        for f in ("user_msg", "agent_response", "tool_calls",
                  "neuromods", "embedding_hash", "importance", "topic_tags"):
            self.assertNotIn(f, payload["content"])


if __name__ == "__main__":
    unittest.main()
