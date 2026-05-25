"""arch §7 conformance — `build_timechain_payload` tag + content shape.

Phase 2 closure (D-SPEC-125 follow-up, 2026-05-25). Verifies the pass-path
conversation-fork TX carries arch §7 normative tags
`["chat", f"chat:<id>", f"user:<hash>"] + topic_tags + [channel]` and
content fields `{chat_id, user_id_hash, turn_index, ...existing OVG audit
fields}`. Also verifies backward-compat: system-internal callers passing
no user_id get a stable, anonymous-safe payload (no `user:` tag, empty
user_id_hash) — needed for OVG callers like backup/dream/X-post that
don't have a user.

Cross-references:
  - arch §7 (tag list + content shape)
  - PLAN_synthesis_engine_Phase2.md §B (closure scope)
  - actr_user_conversation_bundle.json (the standing contract that
    matches on STARTSWITH_ANY "user:")

PLAN §B.5.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from titan_hcl.logic.output_verifier import OutputVerifier, OVGResult
from titan_hcl.synthesis import user_id_hash as uih


class _OVGStub:
    """Just enough OutputVerifier to test build_timechain_payload in
    isolation — avoids needing an Ed25519 keypair / Solana cluster."""

    def __init__(self) -> None:
        self._titan_id = "T_TEST"

    # Bind the production method onto the stub.
    build_timechain_payload = OutputVerifier.build_timechain_payload


def _pass_result(channel: str = "chat", signature: str = "sig_xyz") -> OVGResult:
    return OVGResult(
        passed=True,
        output_text="hello world",
        signature=signature,
        block_height=1,
        merkle_root="abc",
        checks={"directives": True, "injection": True},
        violations=[],
        violation_type="none",
        channel=channel,
        timestamp=1.0,
    )


# ─────────────────────────────────────────────────────────────────────────
# Arch §7 tag list — pass path
# ─────────────────────────────────────────────────────────────────────────

class TestPassPathTags(unittest.TestCase):

    def setUp(self) -> None:
        self.ov = _OVGStub()
        # Lock the salt for deterministic hashes across tests.
        self._patcher = patch.object(uih, "_salt_cache", b"\xde\xad\xbe\xef" * 8)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        uih.clear_cache()

    def test_with_user_id_includes_user_tag(self) -> None:
        payload = self.ov.build_timechain_payload(
            _pass_result(),
            prompt_text="hi",
            user_id="maker",
            chat_id="sess-1",
            turn_index=0,
        )
        tags = payload["tags"]
        # arch §7: "chat" literal first
        assert tags[0] == "chat"
        # Composite chat:<id>
        assert "chat:sess-1" in tags
        # user:<hash> — 16-hex per the hash module contract
        user_tags = [t for t in tags if t.startswith("user:")]
        assert len(user_tags) == 1
        hex_part = user_tags[0][len("user:"):]
        assert len(hex_part) == 16
        int(hex_part, 16)  # must be valid hex
        # Channel still present (backward audit trail)
        assert "chat" in tags  # the channel value
        # verified_output is GONE (Option B strict arch §7)
        assert "verified_output" not in tags

    def test_anonymous_user_id_omits_user_tag(self) -> None:
        """Per arch §7 + hash module contract: anonymous → no user: tag."""
        payload = self.ov.build_timechain_payload(
            _pass_result(),
            user_id="anonymous",
            chat_id="sess-1",
        )
        tags = payload["tags"]
        assert not any(t.startswith("user:") for t in tags)

    def test_empty_user_id_omits_user_tag(self) -> None:
        """System-internal callers pass no user_id → no user: tag."""
        payload = self.ov.build_timechain_payload(_pass_result())
        tags = payload["tags"]
        assert not any(t.startswith("user:") for t in tags)
        # And no chat:<id> tag either (no chat_id provided).
        assert not any(t.startswith("chat:") for t in tags)
        # "chat" literal is still present (arch §7 baseline).
        assert "chat" in tags

    def test_empty_chat_id_omits_chat_id_tag(self) -> None:
        """Missing chat_id → no `chat:` composite tag; literal `chat`
        always present per arch §7."""
        payload = self.ov.build_timechain_payload(
            _pass_result(),
            user_id="maker",
            chat_id="",
        )
        tags = payload["tags"]
        assert not any(t.startswith("chat:") for t in tags)
        assert "chat" in tags

    def test_topic_tags_appended(self) -> None:
        """Caller-supplied topic_tags land between user:<hash> and channel."""
        payload = self.ov.build_timechain_payload(
            _pass_result(),
            user_id="maker",
            chat_id="s1",
            topic_tags=["topic:phase2", "topic:synthesis"],
        )
        tags = payload["tags"]
        assert "topic:phase2" in tags
        assert "topic:synthesis" in tags

    def test_topic_tags_none_safe(self) -> None:
        """None topic_tags doesn't crash."""
        payload = self.ov.build_timechain_payload(
            _pass_result(),
            user_id="maker", chat_id="s1",
            topic_tags=None,
        )
        assert payload  # no exception

    def test_tag_order_stable(self) -> None:
        """Tag list order is deterministic for log-diff stability."""
        p1 = self.ov.build_timechain_payload(
            _pass_result(),
            user_id="maker", chat_id="sess-A",
            topic_tags=["topic:foo", "topic:bar"],
        )
        p2 = self.ov.build_timechain_payload(
            _pass_result(),
            user_id="maker", chat_id="sess-A",
            topic_tags=["topic:foo", "topic:bar"],
        )
        assert p1["tags"] == p2["tags"]


# ─────────────────────────────────────────────────────────────────────────
# Arch §7 content shape — pass path
# ─────────────────────────────────────────────────────────────────────────

class TestPassPathContent(unittest.TestCase):

    def setUp(self) -> None:
        self.ov = _OVGStub()
        self._patcher = patch.object(uih, "_salt_cache", b"\xab\xcd" * 16)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        uih.clear_cache()

    def test_arch_section_7_fields_present(self) -> None:
        payload = self.ov.build_timechain_payload(
            _pass_result(),
            user_id="maker", chat_id="sess-X", turn_index=3,
        )
        content = payload["content"]
        # NEW Phase 2 closure fields
        assert content["chat_id"] == "sess-X"
        assert content["turn_index"] == 3
        # user_id_hash is the raw 16-hex digest (NO `user:` prefix)
        assert content["user_id_hash"] != ""
        assert ":" not in content["user_id_hash"]
        assert len(content["user_id_hash"]) == 16
        # Existing OVG audit fields still present (load-bearing)
        assert content["output_hash"] != ""
        assert "signature" in content
        assert content["channel"] == "chat"
        assert content["checks"]["directives"] is True
        assert content["violation_type"] == "none"
        assert content["titan_id"] == "T_TEST"

    def test_anonymous_user_id_hash_empty(self) -> None:
        payload = self.ov.build_timechain_payload(
            _pass_result(),
            user_id="anonymous", chat_id="sess-X",
        )
        assert payload["content"]["user_id_hash"] == ""

    def test_user_id_hash_in_content_matches_tag(self) -> None:
        """The content.user_id_hash field is the raw hex form of the
        same hash that appears in the user:<hash> tag — readers can
        cross-correlate."""
        payload = self.ov.build_timechain_payload(
            _pass_result(),
            user_id="alice", chat_id="s1",
        )
        tags = payload["tags"]
        user_tag = next(t for t in tags if t.startswith("user:"))
        tag_hex = user_tag[len("user:"):]
        content_hex = payload["content"]["user_id_hash"]
        assert tag_hex == content_hex


# ─────────────────────────────────────────────────────────────────────────
# Backward compatibility — system-internal callers (no user_id arg)
# ─────────────────────────────────────────────────────────────────────────

class TestBackwardCompat(unittest.TestCase):
    """OVG is called from many places besides chat (backup, dream events,
    self-thought TXs, etc.). Those callers don't pass user_id and MUST
    continue to get a valid payload."""

    def setUp(self) -> None:
        self.ov = _OVGStub()
        self._patcher = patch.object(uih, "_salt_cache", b"\x77" * 32)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        uih.clear_cache()

    def test_no_user_id_kwarg_works(self) -> None:
        """Pre-Phase-2-closure call signature stays working."""
        payload = self.ov.build_timechain_payload(_pass_result())
        assert payload["fork"] == "conversation"
        assert payload["thought_type"] == "conversation"
        assert payload["content"]["channel"] == "chat"

    def test_prompt_only_no_chat_metadata(self) -> None:
        """Existing signature `build_timechain_payload(result, prompt_text)` continues."""
        payload = self.ov.build_timechain_payload(_pass_result(), "hello prompt")
        assert payload["content"]["prompt_hash"] != ""
        # No user tag, no chat: tag.
        assert not any(t.startswith("user:") for t in payload["tags"])
        assert not any(t.startswith("chat:") for t in payload["tags"])

    def test_blocked_path_unchanged(self) -> None:
        """Blocked path routes to meta fork — NOT affected by Phase 2
        closure (security alerts don't have per-user bundles)."""
        blocked = OVGResult(
            passed=False, output_text="blocked",
            signature=None,
            checks={"directives": False},
            violations=["test"],
            violation_type="directive",
            channel="chat", timestamp=1.0,
        )
        payload = self.ov.build_timechain_payload(blocked)
        assert payload["fork"] == "meta"
        assert "security_alert" in payload["tags"]


# ─────────────────────────────────────────────────────────────────────────
# Standing-contract round-trip — payload tags match contract's matcher
# ─────────────────────────────────────────────────────────────────────────

class TestStandingContractMatch(unittest.TestCase):
    """The closure's whole point: TX tags produced by build_timechain_payload
    must satisfy actr_user_conversation_bundle's rule (AND event=tx_sealed,
    fork=conversation, tags STARTSWITH_ANY 'user:'). We exercise the rule
    against a fresh payload to confirm end-to-end conformance."""

    def setUp(self) -> None:
        self.ov = _OVGStub()
        self._patcher = patch.object(uih, "_salt_cache", b"\x44" * 32)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        uih.clear_cache()

    def test_payload_tags_match_actr_user_conversation_bundle_rule(self) -> None:
        """Live-run the standing contract's rule against a fresh payload —
        this is the bug we set out to fix."""
        import json, os
        from titan_hcl.logic.timechain_v2 import RuleEvaluator
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        contract_path = os.path.join(
            repo_root, "titan_hcl", "contracts", "meta_cognitive",
            "actr_user_conversation_bundle.json")
        with open(contract_path) as f:
            contract = json.load(f)

        payload = self.ov.build_timechain_payload(
            _pass_result(),
            user_id="maker", chat_id="sess-live",
        )

        # Build the post-seal hook ctx as BlockBuilder._emit_maintain_bundle_events
        # would construct it.
        ctx = {
            "event": "tx_sealed",
            "fork": "conversation",
            "tags": payload["tags"],
            "tx_hash": "deadbeef",
            "epoch_id": 100,
            "ts": 1234.0,
            "significance": 0.5,
            "source": "output_verifier",
        }
        ev = RuleEvaluator()
        action = ev.evaluate(contract["rules"], ctx)
        assert action is not None
        assert action["action"] == "maintain_bundle"
        assert action["entity_class"] == "user"
        assert action["entity_id_from"] == "tag_prefix:user:"


if __name__ == "__main__":
    unittest.main()
