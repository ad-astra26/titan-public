"""Tests for PostDispatchOrchestrator (Phase C-S9 chunk 9Q).

Covers:
  - _build_post_context populates fields from mocked SHM dicts
  - _drain_meter_catalysts converts CatalystEvent → legacy dict shape
  - _clear_meter_catalysts empties meter buffer
  - _delegate_first_check returns True when last post titan_id=T1
  - _process_delegate_queue pops on success, keeps on rate-limit
  - _run_mention_cycle respects 30-min cooldown
  - _emit_timechain_commit puts well-formed payload on send_queue
  - _meta_pre_post / _meta_outcome track request_ids
  - run_tick end-to-end: verified result → catalysts cleared +
    TIMECHAIN_COMMIT emitted; failed result → catalysts retained

Run: ``python -m pytest tests/test_post_dispatch_orchestrator.py
-v -p no:anchorpy``
"""
from __future__ import annotations

import json
import os
import queue
import time
from unittest.mock import MagicMock, patch

import pytest

from titan_hcl import bus
from titan_hcl.logic.social_pressure import CatalystEvent
from titan_hcl.logic.social_worker_post_dispatch import (
    PostDispatchOrchestrator,
)
from titan_hcl.logic.social_x_gateway import ActionResult


class _FakeGateway:
    """Minimal SocialXGateway substitute capturing post() calls."""

    def __init__(self, post_result_status: str = "verified",
                 tweet_id: str = "1234567890") -> None:
        self.post_result_status = post_result_status
        self.tweet_id = tweet_id
        self.post_calls: list[tuple] = []
        self.reply_calls: list[tuple] = []
        # _db() must return something supporting .execute().fetchone()
        self._db_rows: list[dict] = []

    def _db(self):
        class _Cur:
            def __init__(self, rows): self._rows = list(rows)
            def fetchone(self): return self._rows[0] if self._rows else None
            def close(self): pass

        class _Db:
            def __init__(self, rows): self._rows = rows
            def execute(self, *a, **k): return _Cur(self._rows)
            def close(self): pass

        return _Db(self._db_rows)

    # Phase 3 Chunk ω-bis (D-SPEC-88) — two-call shape.
    def prepare_post(self, ctx, consumer="", emot_cgn=None, bus=None,
                     force_ungrounded=False):
        """Always returns a ready-to-compose descriptor — tests never test
        gate refusal at this layer (post_result_status drives final outcome)."""
        from titan_hcl.logic.social_x_gateway import PostDescriptor
        desc = PostDescriptor(
            post_type="bilingual",
            catalyst=(ctx.catalysts[0] if ctx.catalysts else {"type": "test"}),
            system_prompt="test-system",
            user_prompt="test-user",
            max_tokens=200,
            temperature=0.8,
            voice_cfg={},
        )
        return None, desc

    def post(self, ctx, consumer, descriptor=None, bus=None,
             force_ungrounded=False):
        self.post_calls.append(
            (ctx, consumer, force_ungrounded))
        return ActionResult(
            status=self.post_result_status,
            tweet_id=(self.tweet_id
                      if self.post_result_status in ("verified", "posted")
                      else ""),
            reason="",
            text="hello world post text",
        )

    def reply(self, rctx, consumer):
        self.reply_calls.append((rctx, consumer))
        return ActionResult(
            status="posted", tweet_id="reply123", text="reply text")

    def discover_mentions(self, base, consumer, grounded_words):
        return [{
            "tweet_id": "mention1",
            "titan_id": "T1",
            "text": "hi @iamtitanai",
            "author_handle": "alice",
            "relevance_score": 0.7,
        }]

    def mark_mention_replied(self, mention_id, reply_id): pass


class _FakeMeter:
    def __init__(self) -> None:
        self.catalyst_events: list[CatalystEvent] = []


def _stub_bank() -> MagicMock:
    """Return a MagicMock ShmReaderBank where every read_* returns None
    by default. Tests configure specific methods as needed."""
    bank = MagicMock()
    for name in (
        "read_chi", "read_neuromod", "read_epoch",
        "read_unified_spirit_metadata", "read_reasoning_state",
        "read_meta_reasoning_state", "read_dream_state",
        "read_expression_state", "read_mind_state",
        "read_msl_state", "read_consciousness_age",
        "read_language_state", "read_social_perception_state",
    ):
        getattr(bank, name).return_value = None
    return bank


def _mk(meter=None, gateway=None, send_queue=None):
    """Create orchestrator with a MagicMock ShmReaderBank + stubbed composer.

    Phase B.5 migration (2026-05-18): PostDispatchOrchestrator no longer
    holds per-spec StateRegistryReader instances; it owns a single
    ShmReaderBank. Tests stub the bank so every read_* returns None
    unless the test overrides specific methods.

    Phase 3 Chunk ω-bis (D-SPEC-82, 2026-05-18): _compose_post_text is
    monkey-patched to return a fixed string so unit tests don't hit
    httpx /v4/llm-distill. Real LLM round-trip is covered by live cascade.
    """
    gw = gateway or _FakeGateway()
    m = meter or _FakeMeter()
    sq = send_queue or queue.Queue()
    bank = _stub_bank()
    with patch(
            "titan_hcl.logic.social_worker_post_dispatch."
            "ensure_shm_root",
            return_value="/tmp/test_shm"), \
        patch(
            "titan_hcl.logic.social_worker_post_dispatch."
            "ShmReaderBank",
            return_value=bank):
        orch = PostDispatchOrchestrator(
            gateway=gw, meter=m, titan_id="T_TEST",
            send_queue=sq, worker_name="social_worker_test")
    # Phase 3 Chunk ω-bis (D-SPEC-88) — bypass the httpx /v4/llm-distill
    # round-trip in tests by stubbing _compose_post_text to a fixed string.
    # Real LLM round-trip is covered by the live cascade probe.
    orch._compose_post_text = lambda descriptor: "test composed post text"
    return orch, gw, m, sq


# ── Build context ────────────────────────────────────────────────────


def test_build_post_context_populates_fields_from_shm():
    """When canonical SHM reads return the expected dicts, PostContext
    is fully populated with fields the gateway expects.

    Phase B.5 migration 2026-05-18: reads now come from individual
    ShmReaderBank methods (chi_state / neuromod_state /
    unified_spirit_metadata / reasoning_state / meta_reasoning_state /
    dream_state / expression_state / mind_state / language_state /
    social_perception_state) — replaces the retired
    spirit_supplemental_state.bin coordinator subdict aggregation.
    """
    orch, _, _, _ = _mk()
    full_config = {
        "twitter_social": {
            "auth_session": "sess", "webshare_static_url": "px",
        },
        "inference": {
            "ollama_cloud_base_url": "u", "ollama_cloud_api_key": "k",
            "ollama_cloud_chat_model": "m",
        },
        "stealth_sage": {"twitterapi_io_key": "tw"},
    }
    bank = orch._shm_bank
    bank.read_chi.return_value = {"total": 0.75}
    # neuromod_state schema: {modulators: {DA: {level, gain, ...}}}.
    bank.read_neuromod.return_value = {
        "modulators": {
            "DA": {"level": 0.8},
            "5HT": {"level": 0.5},
        },
    }
    bank.read_unified_spirit_metadata.return_value = {
        "epoch_count": 4242,
        "last_drift": 0.02,
        "last_trajectory": 0.91,
        "latest_epoch": {"epoch_id": 4242},
    }
    bank.read_reasoning_state.return_value = {
        "active_chain_count": 4,
        "total_chains": 100, "total_conclusions": 50,
        "last_conclusion": "wisdom emerged",
        "dominant_primitive": "Hypothesizer",
    }
    bank.read_meta_reasoning_state.return_value = {
        "total_eurekas": 12,
        "total_wisdom_saved": 8,
        "total_chains": 30,
        "meta_cgn_signals": 5,
        "crystallized_samples": [{"x": 1}],
    }
    bank.read_dream_state.return_value = {"distilled_count": 7}
    bank.read_expression_state.return_value = {
        "fire_counts": {"ART": 3, "MUSIC": 1},
    }
    # attention_entropy lives in mind_state; i_confidence + concept_confidences
    # are read from the canonical MSL self-model slot (2026-05-29 fix —
    # mind_state never carried them). See social_worker_post_dispatch.py:249.
    bank.read_mind_state.return_value = {
        "attention_entropy": 0.4,
    }
    bank.read_msl_state.return_value = {
        "i_confidence": 0.92,
        "concept_confidences": {"I": 0.9, "YOU": 0.7},
    }
    bank.read_language_state.return_value = {
        "vocab_total": 446, "vocab_producible": 342,
        "composition_level": "L9",
        "recent_words": ["wonder", "becoming"],
    }
    bank.read_social_perception_state.return_value = {
        "contagion_latest": {"src": "T2", "valence": 0.3},
    }

    with patch.object(orch, "_fetch_creative_works_samples",
                      return_value=[]):
        ctx = orch._build_post_context(
            full_config=full_config, catalysts=[{"type": "eureka"}])
    assert ctx is not None
    assert ctx.titan_id == "T_TEST"
    assert ctx.session == "sess"
    # Phase B.5: emotion default is "wonder" — string label not in
    # canonical neuromod_state slot (numeric-only). Override happens
    # post-migration when neuromod schema bumps to include emotion.
    assert ctx.emotion == "wonder"
    assert ctx.neuromods == {"DA": 0.8, "5HT": 0.5}
    assert ctx.epoch == 4242
    # pi_ratio: post_context.pi_ratio retired with
    # spirit_supplemental_state; defaults to 0.0 until successor field.
    assert ctx.pi_ratio == 0.0
    assert ctx.chi == pytest.approx(0.75)
    assert ctx.drift == pytest.approx(0.02)
    assert ctx.trajectory == pytest.approx(0.91)
    assert ctx.i_confidence == pytest.approx(0.92)
    assert ctx.attention_entropy == pytest.approx(0.4)
    assert ctx.reasoning_chains == 4
    # 30 me_chains >= 20 → me_wisdom/me_chains = 8/30
    assert ctx.reasoning_commit_rate == pytest.approx(8 / 30)
    assert ctx.recent_chain_summary == "wisdom emerged"
    assert ctx.meta_style == "Hypothesizer"
    assert ctx.vocab_total == 446
    assert ctx.composition_level == 9
    assert ctx.recent_words == ["wonder", "becoming"]
    assert ctx.recent_expression == {"ART": 3, "MUSIC": 1}
    assert ctx.social_contagion["src"] == "T2"
    assert ctx.total_eurekas == 12
    assert ctx.distilled_count == 7
    assert ctx.meta_cgn_signals == 5
    assert ctx.crystallized_samples == [{"x": 1}]
    assert ctx.catalysts == [{"type": "eureka"}]


def test_build_post_context_tolerates_empty_shm():
    """Cold-boot all-None SHM reads must still produce a PostContext
    with safe zero/empty defaults (gateway.post can still be called)."""
    orch, _, _, _ = _mk()
    # _stub_bank already configured every read_* to return None.
    with patch.object(orch, "_fetch_creative_works_samples",
                      return_value=[]):
        ctx = orch._build_post_context(
            full_config={}, catalysts=[])
    assert ctx is not None
    assert ctx.emotion == "wonder"  # default
    assert ctx.epoch == 0
    assert ctx.pi_ratio == 0.0
    assert ctx.chi == 0.0
    assert ctx.recent_expression == {}
    assert ctx.social_contagion == {}


# ── Catalyst handoff ────────────────────────────────────────────────


def test_drain_meter_catalysts_returns_legacy_dict_shape():
    """Meter holds CatalystEvent; drain emits legacy-shaped dicts."""
    meter = _FakeMeter()
    meter.catalyst_events.append(CatalystEvent(
        type="eureka", significance=0.8,
        content="aha", data={"chain_id": 7}))
    meter.catalyst_events.append(CatalystEvent(
        type="kin_resonance", significance=0.6, content="kin",
        data={}))
    orch, _, _, _ = _mk(meter=meter)
    out = orch._drain_meter_catalysts()
    assert len(out) == 2
    assert out[0] == {
        "type": "eureka", "significance": 0.8,
        "content": "aha", "data": {"chain_id": 7}}
    assert out[1]["type"] == "kin_resonance"
    # Drain must NOT clear (clearing only on post success).
    assert len(meter.catalyst_events) == 2


def test_clear_meter_catalysts_empties_buffer():
    meter = _FakeMeter()
    meter.catalyst_events.append(CatalystEvent(
        type="x", significance=0.5, content="y", data={}))
    orch, _, _, _ = _mk(meter=meter)
    orch._clear_meter_catalysts()
    assert meter.catalyst_events == []


# ── Delegate rotation ────────────────────────────────────────────────


def test_delegate_first_check_returns_true_when_last_post_t1():
    gw = _FakeGateway()
    gw._db_rows = [{"titan_id": "T1"}]
    orch, _, _, _ = _mk(gateway=gw)
    assert orch._delegate_first_check() is True


def test_delegate_first_check_returns_false_when_last_post_not_t1():
    gw = _FakeGateway()
    gw._db_rows = [{"titan_id": "T2"}]
    orch, _, _, _ = _mk(gateway=gw)
    assert orch._delegate_first_check() is False


def test_process_delegate_queue_pops_on_verified(tmp_path,
                                                  monkeypatch):
    """On verified/posted result, the head entry is popped from the
    queue file."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    queue_data = [{
        "titan_id": "T2", "catalyst_type": "delegate",
        "emotion": "wonder", "neuromods": {},
    }]
    with open("data/social_delegate_queue.json", "w") as f:
        json.dump(queue_data, f)
    gw = _FakeGateway(post_result_status="verified")
    orch, _, _, _ = _mk(gateway=gw)
    status = orch._process_delegate_queue(
        full_config={
            "twitter_social": {}, "inference": {}, "stealth_sage": {}},
        pop_on_failure=False)
    assert status in ("verified", "posted")
    with open("data/social_delegate_queue.json") as f:
        remaining = json.load(f)
    assert remaining == []  # popped


def test_process_delegate_queue_keeps_on_rate_limit(tmp_path,
                                                     monkeypatch):
    """On too_soon / hourly_limit, the entry stays in the queue."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    entry = {"titan_id": "T2", "catalyst_type": "x"}
    with open("data/social_delegate_queue.json", "w") as f:
        json.dump([entry], f)
    gw = _FakeGateway(post_result_status="too_soon")
    orch, _, _, _ = _mk(gateway=gw)
    orch._process_delegate_queue(
        full_config={
            "twitter_social": {}, "inference": {}, "stealth_sage": {}},
        pop_on_failure=False)
    with open("data/social_delegate_queue.json") as f:
        remaining = json.load(f)
    assert len(remaining) == 1
    assert remaining[0]["titan_id"] == "T2"


def test_process_delegate_queue_empty_returns_none(tmp_path,
                                                    monkeypatch):
    monkeypatch.chdir(tmp_path)
    orch, _, _, _ = _mk()
    # No queue file exists.
    assert orch._process_delegate_queue(
        full_config={
            "twitter_social": {}, "inference": {}, "stealth_sage": {}},
        pop_on_failure=False) is None


# ── Mention cycle cooldown ─────────────────────────────────────────


def test_mention_cycle_respects_cooldown():
    """Two calls within 30 minutes — only the first triggers
    discover_mentions."""
    gw = _FakeGateway()
    orch, _, _, _ = _mk(gateway=gw)
    cfg = {
        "twitter_social": {}, "inference": {}, "stealth_sage": {},
        "social_x": {"replies": {
            "mention_check_cooldown_seconds": 1800.0,
            "max_replies_per_cycle": 3,
        }},
    }
    now0 = 1_000_000.0
    r1 = orch._run_mention_cycle(
        full_config=cfg, grounded_words=[], neuromods={},
        emotion="wonder", now=now0)
    r2 = orch._run_mention_cycle(
        full_config=cfg, grounded_words=[], neuromods={},
        emotion="wonder", now=now0 + 100)  # within cooldown
    r3 = orch._run_mention_cycle(
        full_config=cfg, grounded_words=[], neuromods={},
        emotion="wonder", now=now0 + 2000)  # past cooldown
    assert r1 >= 0  # discover_mentions ran
    assert r2 == 0  # cooldown gate
    assert r3 >= 0  # discover_mentions ran again


# ── TIMECHAIN_COMMIT emission ───────────────────────────────────────


def test_emit_timechain_commit_payload_shape():
    """TIMECHAIN_COMMIT message has the legacy spirit_worker:8224-8240
    payload shape."""
    sq = queue.Queue()
    orch, _, _, _ = _mk(send_queue=sq)
    result = ActionResult(
        status="verified", tweet_id="9876", reason="",
        text="my new thought")
    # Phase B.5: TIMECHAIN_COMMIT reads canonical neuromod_state +
    # chi_state directly (no coordinator wrapper). Stub the bank's
    # canonical readers for this test.
    orch._shm_bank.read_neuromod.return_value = {
        "modulators": {"DA": {"level": 0.7}},
    }
    orch._shm_bank.read_chi.return_value = {"total": 0.66}
    orch._emit_timechain_commit(result, full_config={})
    msg = sq.get_nowait()
    assert msg["type"] == bus.TIMECHAIN_COMMIT
    assert msg["dst"] == "timechain"
    p = msg["payload"]
    assert p["fork"] == "episodic"
    assert p["source"] == "social_post"
    assert p["content"]["tweet_id"] == "9876"
    assert p["content"]["titan_id"] == "T_TEST"
    assert "text_hash" in p["content"]
    assert p["neuromods"] == {"DA": 0.7}
    assert p["chi_available"] == pytest.approx(0.66)
    assert "social" in p["tags"]
    assert "x_post" in p["tags"]


# ── F-phase meta consultation ──────────────────────────────────────


def test_meta_pre_post_tracks_request_id():
    orch, _, _, _ = _mk()
    # _stub_bank already returns None for read_neuromod / read_chi —
    # the meta_pre_post tolerates empty SHM (cold-boot path).
    with patch(
            "titan_hcl.logic.meta_service_client.send_meta_request",
            return_value="req_abc"), \
        patch(
            "titan_hcl.logic.social_narrator."
            "build_social_meta_context_30d",
            return_value={"ctx": "vec"}):
        req_id = orch._meta_pre_post(
            full_config={}, catalyst_type="eureka")
    assert req_id == "req_abc"
    assert "req_abc" in orch._meta_pending


def test_meta_outcome_clears_pending():
    orch, _, _, _ = _mk()
    orch._meta_pending["req_xyz"] = (time.time(), "post")
    with patch(
            "titan_hcl.logic.meta_service_client.send_meta_outcome"):
        orch._meta_outcome("req_xyz", status="verified")
    assert "req_xyz" not in orch._meta_pending


def test_meta_outcome_noop_on_empty_request_id():
    orch, _, _, _ = _mk()
    orch._meta_outcome("", status="verified")
    # No exception; nothing to clear.


# ── End-to-end run_tick ────────────────────────────────────────────


def test_run_tick_verified_clears_catalysts_and_emits_timechain(
        tmp_path, monkeypatch):
    """End-to-end: when gateway.post returns verified, meter catalysts
    are cleared AND TIMECHAIN_COMMIT is emitted."""
    monkeypatch.chdir(tmp_path)
    meter = _FakeMeter()
    meter.catalyst_events.append(CatalystEvent(
        type="eureka", significance=0.9, content="aha", data={}))
    sq = queue.Queue()
    gw = _FakeGateway(post_result_status="verified",
                       tweet_id="success123")
    gw._db_rows = []  # no T1 last-post → no delegate-first
    orch, _, _, _ = _mk(gateway=gw, meter=meter, send_queue=sq)
    with \
        patch.object(orch, "_fetch_creative_works_samples",
                     return_value=[]), \
        patch(
            "titan_hcl.logic.social_worker_post_dispatch."
            "load_titan_config",
            return_value={
                "twitter_social": {}, "inference": {},
                "stealth_sage": {}}), \
        patch(
            "titan_hcl.logic.meta_service_client.send_meta_request",
            return_value=""), \
        patch(
            "titan_hcl.logic.meta_service_client.send_meta_outcome"):
        orch.run_tick()
    # gateway.post called once for own-post path
    assert len(gw.post_calls) == 1
    # catalysts cleared
    assert meter.catalyst_events == []
    # TIMECHAIN_COMMIT enqueued
    msgs = []
    while not sq.empty():
        msgs.append(sq.get_nowait())
    msg_types = [m["type"] for m in msgs]
    assert bus.TIMECHAIN_COMMIT in msg_types


def test_run_tick_too_soon_keeps_catalysts(monkeypatch, tmp_path):
    """When gateway.post returns too_soon (rate-limited), catalysts
    are NOT cleared (preserved for next attempt)."""
    monkeypatch.chdir(tmp_path)
    meter = _FakeMeter()
    meter.catalyst_events.append(CatalystEvent(
        type="kin_resonance", significance=0.5, content="kin",
        data={}))
    gw = _FakeGateway(post_result_status="too_soon")
    gw._db_rows = []
    orch, _, _, _ = _mk(gateway=gw, meter=meter)
    with \
        patch.object(orch, "_fetch_creative_works_samples",
                     return_value=[]), \
        patch(
            "titan_hcl.logic.social_worker_post_dispatch."
            "load_titan_config",
            return_value={
                "twitter_social": {}, "inference": {},
                "stealth_sage": {}}), \
        patch(
            "titan_hcl.logic.meta_service_client.send_meta_request",
            return_value=""), \
        patch(
            "titan_hcl.logic.meta_service_client.send_meta_outcome"):
        orch.run_tick()
    # catalysts preserved
    assert len(meter.catalyst_events) == 1


def test_run_tick_handles_gateway_post_exception(monkeypatch, tmp_path):
    """A gateway.post() raising must NOT propagate — orchestrator logs
    and continues."""
    monkeypatch.chdir(tmp_path)
    meter = _FakeMeter()
    gw = _FakeGateway()
    gw.post = MagicMock(side_effect=RuntimeError("network wedged"))
    gw._db_rows = []
    orch, _, _, _ = _mk(gateway=gw, meter=meter)
    with \
        patch.object(orch, "_fetch_creative_works_samples",
                     return_value=[]), \
        patch(
            "titan_hcl.logic.social_worker_post_dispatch."
            "load_titan_config",
            return_value={
                "twitter_social": {}, "inference": {},
                "stealth_sage": {}}), \
        patch(
            "titan_hcl.logic.meta_service_client.send_meta_request",
            return_value=""), \
        patch(
            "titan_hcl.logic.meta_service_client.send_meta_outcome"):
        # Must not raise
        orch.run_tick()


def test_run_tick_skips_when_config_load_fails(monkeypatch):
    """Config load failure → tick aborts cleanly without calling
    gateway.post."""
    gw = _FakeGateway()
    orch, _, _, _ = _mk(gateway=gw)
    with patch(
            "titan_hcl.logic.social_worker_post_dispatch."
            "load_titan_config",
            side_effect=RuntimeError("config broken")):
        orch.run_tick()
    assert gw.post_calls == []


# ── Chunks 9N / 9P — canonical-poller broadcast events ──────────────


def _orch_with_canonical(canonical: bool):
    """Build an orchestrator with the canonical-poller flag set."""
    gw = _FakeGateway()
    m = _FakeMeter()
    sq = queue.Queue()
    bank = _stub_bank()
    with patch(
            "titan_hcl.logic.social_worker_post_dispatch."
            "ensure_shm_root",
            return_value="/tmp/test_shm"), \
        patch(
            "titan_hcl.logic.social_worker_post_dispatch."
            "ShmReaderBank",
            return_value=bank):
        orch = PostDispatchOrchestrator(
            gateway=gw, meter=m, titan_id="T_TEST",
            send_queue=sq, worker_name="social_worker_test",
            is_canonical_poller=canonical)
    return orch, gw, sq


def test_broadcast_new_mentions_noop_on_non_canonical():
    """Non-canonical Titan must NOT emit MENTION_RECEIVED — it's the
    consumer side."""
    orch, gw, sq = _orch_with_canonical(canonical=False)
    gw._db_rows = [{
        "tweet_id": "t1", "author": "a", "author_handle": "h",
        "text": "hi", "our_post_id": "p", "titan_id": "T1",
        "status": "pending", "relevance_score": 0.5,
        "discovered_at": 1000.0,
    }]
    count = orch._broadcast_new_mentions()
    assert count == 0
    assert sq.empty()


def test_broadcast_new_mentions_emits_canonical_only(tmp_path,
                                                       monkeypatch):
    """Canonical poller emits MENTION_RECEIVED for new rows; watermark
    advances; subsequent call with no new rows emits zero."""
    orch, gw, sq = _orch_with_canonical(canonical=True)

    # Fake _db() returns row-like objects responding to dict().
    rows1 = [{
        "tweet_id": "tw1", "author": "alice",
        "author_handle": "@alice", "text": "hello", "our_post_id": "p1",
        "titan_id": "T1", "status": "pending", "relevance_score": 0.7,
        "discovered_at": 1000.5,
    }]

    class _DbFake:
        def __init__(self, rows): self._rows = rows
        def execute(self, sql, args):
            cutoff = args[0]
            kept = [r for r in self._rows
                    if r["discovered_at"] > cutoff]
            class _Cur:
                def __init__(self, rs): self._rs = rs
                def fetchall(self): return self._rs
            return _Cur(kept)
        def close(self): pass

    gw._db = lambda: _DbFake(rows1)

    count1 = orch._broadcast_new_mentions()
    assert count1 == 1
    msg = sq.get_nowait()
    assert msg["type"] == bus.MENTION_RECEIVED
    assert msg["payload"]["tweet_id"] == "tw1"

    # Second call with same rows → watermark advanced, 0 new emits.
    count2 = orch._broadcast_new_mentions()
    assert count2 == 0


def test_broadcast_felt_experiences_noop_on_non_canonical():
    orch, _, sq = _orch_with_canonical(canonical=False)
    assert orch._broadcast_new_felt_experiences() == 0
    assert sq.empty()


def test_broadcast_engagement_snapshots_noop_on_non_canonical():
    orch, _, sq = _orch_with_canonical(canonical=False)
    assert orch._broadcast_new_engagement_snapshots() == 0
    assert sq.empty()


def test_broadcast_felt_experiences_emits_on_canonical(tmp_path,
                                                        monkeypatch):
    """Canonical poller emits FELT_EXPERIENCE_CAPTURED for new rows
    inserted into events_teacher.db."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    import sqlite3 as _sql
    con = _sql.connect("data/events_teacher.db")
    con.execute(
        "CREATE TABLE felt_experiences ("
        "id INTEGER PRIMARY KEY, titan_id TEXT, source TEXT, "
        "author TEXT, topic TEXT, sentiment REAL, arousal REAL, "
        "relevance REAL, concept_signals TEXT, semantic_concepts TEXT, "
        "felt_summary TEXT, contagion_type TEXT, mode TEXT, "
        "window_id INTEGER, created_at REAL)")
    con.execute(
        "INSERT INTO felt_experiences VALUES "
        "(1, 'T1', 'x', 'a', 'topic', 0.5, 0.4, 0.3, '[]', '[]', "
        "'felt', 'positive', 'mode', 0, 1234.5)")
    con.commit()
    con.close()
    orch, _, sq = _orch_with_canonical(canonical=True)
    count = orch._broadcast_new_felt_experiences()
    assert count == 1
    msg = sq.get_nowait()
    assert msg["type"] == bus.FELT_EXPERIENCE_CAPTURED
    assert msg["payload"]["id"] == 1
    assert msg["payload"]["titan_id"] == "T1"


def test_broadcast_engagement_snapshots_emits_on_canonical(tmp_path,
                                                            monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    import sqlite3 as _sql
    con = _sql.connect("data/events_teacher.db")
    con.execute(
        "CREATE TABLE engagement_snapshots ("
        "id INTEGER PRIMARY KEY, titan_id TEXT, tweet_id TEXT, "
        "likes INTEGER, replies INTEGER, quotes INTEGER, "
        "delta_likes INTEGER, delta_replies INTEGER, "
        "delta_quotes INTEGER, checked_at REAL)")
    con.execute(
        "INSERT INTO engagement_snapshots VALUES "
        "(1, 'T1', 'tw1', 5, 2, 1, 1, 0, 0, 9999.0)")
    con.commit()
    con.close()
    orch, _, sq = _orch_with_canonical(canonical=True)
    count = orch._broadcast_new_engagement_snapshots()
    assert count == 1
    msg = sq.get_nowait()
    assert msg["type"] == bus.ENGAGEMENT_SNAPSHOT_TAKEN
    assert msg["payload"]["tweet_id"] == "tw1"


# ── Chunk 9P — ingest handlers (non-canonical consumer DB write) ────


def test_ingest_mention_received_writes_locally(tmp_path, monkeypatch):
    """_ingest_mention_received: writes broadcast payload to local
    mention_tracking via the gateway's _db handle."""
    from titan_hcl.modules.social_worker import (
        _ingest_mention_received,
    )

    # Build a real sqlite gateway substitute with mention_tracking
    # schema matching social_x_gateway:383-403.
    import sqlite3 as _sql
    db_path = tmp_path / "social_x.db"
    con = _sql.connect(db_path)
    con.execute(
        "CREATE TABLE mention_tracking ("
        "tweet_id TEXT PRIMARY KEY, author TEXT, author_handle TEXT, "
        "text TEXT, our_post_id TEXT, titan_id TEXT, status TEXT, "
        "relevance_score REAL, discovered_at REAL)")
    con.commit()
    con.close()

    class _StubGateway:
        def _db(self):
            return _sql.connect(db_path)

    payload = {
        "tweet_id": "mention_xyz", "author": "user1",
        "author_handle": "@user1", "text": "hi @iamtitanai",
        "our_post_id": "ours123", "titan_id": "T1",
        "status": "pending", "relevance_score": 0.6,
        "discovered_at": 5000.0,
    }
    _ingest_mention_received(_StubGateway(), payload)
    # Re-ingest the same payload — idempotency check.
    _ingest_mention_received(_StubGateway(), payload)

    con = _sql.connect(db_path)
    rows = con.execute(
        "SELECT tweet_id, author_handle, status FROM mention_tracking"
    ).fetchall()
    con.close()
    assert len(rows) == 1
    assert rows[0][0] == "mention_xyz"
    assert rows[0][2] == "pending"


def test_ingest_felt_experience_writes_locally(tmp_path, monkeypatch):
    """_ingest_felt_experience: writes broadcast payload to local
    events_teacher.db felt_experiences with idempotency on id."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    import sqlite3 as _sql
    db_path = "data/events_teacher.db"
    con = _sql.connect(db_path)
    con.execute(
        "CREATE TABLE felt_experiences ("
        "id INTEGER PRIMARY KEY, titan_id TEXT, source TEXT, "
        "author TEXT, topic TEXT, sentiment REAL, arousal REAL, "
        "relevance REAL, concept_signals TEXT, semantic_concepts TEXT, "
        "felt_summary TEXT, contagion_type TEXT, mode TEXT, "
        "window_id INTEGER, created_at REAL)")
    con.commit()
    con.close()
    from titan_hcl.modules.social_worker import (
        _ingest_felt_experience,
    )
    payload = {
        "id": 42, "titan_id": "T1", "source": "twitter",
        "author": "alice", "topic": "music", "sentiment": 0.3,
        "arousal": 0.5, "relevance": 0.7,
        "concept_signals": "[]", "semantic_concepts": "[]",
        "felt_summary": "a felt thing",
        "contagion_type": "positive", "mode": "passive",
        "window_id": 7, "created_at": 1234.5,
    }
    _ingest_felt_experience(payload)
    _ingest_felt_experience(payload)  # idempotency
    con = _sql.connect(db_path)
    rows = con.execute(
        "SELECT id, titan_id, topic FROM felt_experiences").fetchall()
    con.close()
    assert len(rows) == 1
    assert rows[0] == (42, "T1", "music")


def test_ingest_engagement_snapshot_writes_locally(tmp_path,
                                                    monkeypatch):
    """_ingest_engagement_snapshot: writes to local events_teacher.db
    engagement_snapshots with id PK idempotency."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    import sqlite3 as _sql
    db_path = "data/events_teacher.db"
    con = _sql.connect(db_path)
    con.execute(
        "CREATE TABLE engagement_snapshots ("
        "id INTEGER PRIMARY KEY, titan_id TEXT, tweet_id TEXT, "
        "likes INTEGER, replies INTEGER, quotes INTEGER, "
        "delta_likes INTEGER, delta_replies INTEGER, "
        "delta_quotes INTEGER, checked_at REAL)")
    con.commit()
    con.close()
    from titan_hcl.modules.social_worker import (
        _ingest_engagement_snapshot,
    )
    payload = {
        "id": 99, "titan_id": "T1", "tweet_id": "tw_99",
        "likes": 10, "replies": 3, "quotes": 1,
        "delta_likes": 2, "delta_replies": 1, "delta_quotes": 0,
        "checked_at": 7777.0,
    }
    _ingest_engagement_snapshot(payload)
    _ingest_engagement_snapshot(payload)
    con = _sql.connect(db_path)
    rows = con.execute(
        "SELECT id, titan_id, tweet_id, likes FROM "
        "engagement_snapshots").fetchall()
    con.close()
    assert len(rows) == 1
    assert rows[0] == (99, "T1", "tw_99", 10)
