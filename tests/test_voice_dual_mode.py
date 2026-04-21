"""Tests for Phase 5C Narrator Evolution — dual-voice grounding gate.

Covers:
  1. knowledge_gate utility: topic extraction + confidence lookup
  2. social_x_gateway._check_grounding_appropriateness — observability vs
     enforcement, empty-topic short-circuit, cooldown per topic
  3. End-to-end: post() with low-confidence topic suppresses under
     enforcement, emits telemetry, fires CGN_KNOWLEDGE_REQ via bus shim

All tests use isolated sqlite fixtures + tmp telemetry paths — no network,
no real X API calls, no LLM.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ── knowledge_gate ──────────────────────────────────────────────────

def _make_knowledge_db(path: str, rows: list[tuple[str, float]]) -> None:
    """rows = [(topic, confidence), ...]"""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE knowledge_concepts (topic TEXT, "
                 "confidence REAL, created_at REAL DEFAULT 0, "
                 "source TEXT DEFAULT 'test')")
    for topic, conf in rows:
        conn.execute("INSERT INTO knowledge_concepts (topic, confidence) "
                     "VALUES (?, ?)", (topic, conf))
    conn.commit()
    conn.close()


def test_extract_topic_words_filters_stopwords_and_short():
    from titan_plugin.logic.knowledge_gate import extract_topic_words
    out = extract_topic_words(
        "What is the aurora borealis and why does it happen?")
    # "what", "is", "the", "and", "why", "does", "it" → stopwords;
    # 2-char words filtered. Survivors: aurora, borealis, happen
    assert "aurora" in out
    assert "borealis" in out
    assert "happen" in out
    for bad in ("what", "is", "the", "and", "why", "does", "it"):
        assert bad not in out


def test_extract_topic_words_empty_returns_empty_list():
    from titan_plugin.logic.knowledge_gate import extract_topic_words
    assert extract_topic_words("") == []
    assert extract_topic_words("   ") == []
    # All stopwords (checked against the actual STOPWORDS set in utility):
    assert extract_topic_words("i am the you are") == []


def test_check_topic_confidence_returns_max_across_keyword_matches():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "kn.db")
        _make_knowledge_db(db, [
            ("aurora physics basics", 0.82),
            ("aurora folklore origins", 0.31),
            ("borealis visual guide", 0.55),
            ("quantum mechanics", 0.90),
        ])
        from titan_plugin.logic.knowledge_gate import check_topic_confidence
        # aurora → 0.82, borealis → 0.55 → max=0.82
        assert check_topic_confidence(
            ["aurora", "borealis"], db_path=db) == pytest.approx(0.82)
        # unknown topic → 0.0
        assert check_topic_confidence(
            ["dragons"], db_path=db) == 0.0
        # empty input → 0.0 without touching DB
        assert check_topic_confidence([], db_path=db) == 0.0


def test_check_topic_confidence_nonexistent_db_returns_zero():
    from titan_plugin.logic.knowledge_gate import check_topic_confidence
    assert check_topic_confidence(["anything"],
                                  db_path="/tmp/does_not_exist.db") == 0.0


# ── gateway grounding gate ──────────────────────────────────────────

def _make_gateway(tmp: str, telemetry_name: str = "tel.jsonl"):
    from titan_plugin.logic.social_x_gateway import (
        SocialXGateway, PostContext)
    gw = SocialXGateway(
        db_path=os.path.join(tmp, "sx.db"),
        config_path=os.path.join(tmp, "nonexistent_config.toml"),
        telemetry_path=os.path.join(tmp, telemetry_name),
    )
    ctx = PostContext(
        session="x", proxy="", api_key="", titan_id="T1",
        emotion="wonder",
    )
    return gw, ctx


def _read_telemetry(path: str) -> list[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def test_gate_disabled_when_dual_mode_off(tmp_path):
    gw, ctx = _make_gateway(str(tmp_path))
    catalyst = {"type": "reflection",
                "content": "aurora borealis quantum mechanics"}
    out = gw._check_grounding_appropriateness(
        ctx, catalyst, voice_cfg={"dual_mode_enabled": False}, bus=None)
    assert out is None
    # no telemetry emitted when disabled
    evs = _read_telemetry(gw._telemetry_path)
    assert not any(e.get("event") == "post_grounding_check" for e in evs)


def test_gate_empty_catalyst_does_not_suppress(tmp_path):
    gw, ctx = _make_gateway(str(tmp_path))
    out = gw._check_grounding_appropriateness(
        ctx, {"type": "", "content": ""},
        voice_cfg={"dual_mode_enabled": True,
                   "x_grounding_enforced": True,
                   "x_grounding_threshold": 0.5},
        bus=None)
    assert out is None  # no topics → don't suppress


def test_gate_obs_mode_logs_but_does_not_suppress(tmp_path, monkeypatch):
    # Build a fake knowledge DB the gate will query
    db = tmp_path / "inner_memory.db"
    _make_knowledge_db(str(db), [("aurora basics", 0.2)])
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.replace(str(db), "data/inner_memory.db")

    gw, ctx = _make_gateway(str(tmp_path))
    catalyst = {"type": "reflection", "content": "aurora borealis tonight"}
    out = gw._check_grounding_appropriateness(
        ctx, catalyst,
        voice_cfg={"dual_mode_enabled": True,
                   "x_grounding_enforced": False,
                   "x_grounding_threshold": 0.5},
        bus=None)
    assert out is None  # obs mode never suppresses

    evs = _read_telemetry(gw._telemetry_path)
    types = [e.get("event") for e in evs]
    assert "post_grounding_check" in types
    assert "post_suppressed_ungrounded" in types  # still log the fact


def test_gate_enforced_mode_suppresses_below_threshold(tmp_path, monkeypatch):
    db = tmp_path / "inner_memory.db"
    _make_knowledge_db(str(db), [("cats behavior", 0.15)])
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.replace(str(db), "data/inner_memory.db")

    gw, ctx = _make_gateway(str(tmp_path))
    out = gw._check_grounding_appropriateness(
        ctx, {"type": "reflection", "content": "cats lounging today"},
        voice_cfg={"dual_mode_enabled": True,
                   "x_grounding_enforced": True,
                   "x_grounding_threshold": 0.5},
        bus=None)
    assert out is not None
    assert out.get("suppress") is True
    assert out.get("confidence") < 0.5


def test_gate_passes_when_confidence_above_threshold(tmp_path, monkeypatch):
    db = tmp_path / "inner_memory.db"
    _make_knowledge_db(str(db), [("quantum mechanics intro", 0.85)])
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.replace(str(db), "data/inner_memory.db")

    gw, ctx = _make_gateway(str(tmp_path))
    out = gw._check_grounding_appropriateness(
        ctx, {"type": "reflection", "content": "quantum mechanics today"},
        voice_cfg={"dual_mode_enabled": True,
                   "x_grounding_enforced": True,
                   "x_grounding_threshold": 0.5},
        bus=None)
    assert out is None  # grounded — gate passes


def test_knowledge_req_cooldown_prevents_spam(tmp_path, monkeypatch):
    db = tmp_path / "inner_memory.db"
    _make_knowledge_db(str(db), [])  # all topics ungrounded
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.replace(str(db), "data/inner_memory.db")

    gw, ctx = _make_gateway(str(tmp_path))
    published: list[dict] = []
    # Single-topic catalyst so the cooldown applies to THE one topic we test
    # (cooldown is per-topic — if multiple topic-words exist, the gate fires
    # for the first one past cooldown, which is the correct design).
    catalyst = {"type": "", "content": "dragons"}
    voice_cfg = {"dual_mode_enabled": True,
                 "x_grounding_enforced": True,
                 "x_grounding_threshold": 0.5,
                 "x_grounding_cooldown_secs": 3600}

    # First call — should fire CGN_KNOWLEDGE_REQ
    gw._check_grounding_appropriateness(
        ctx, catalyst, voice_cfg=voice_cfg, bus=published.append)
    assert len(published) == 1
    assert published[0]["type"] == "CGN_KNOWLEDGE_REQ"
    assert published[0]["payload"]["topic"] == "dragons"

    # Second call — same topic within cooldown → no new emit
    gw._check_grounding_appropriateness(
        ctx, catalyst, voice_cfg=voice_cfg, bus=published.append)
    assert len(published) == 1, ("cooldown failed — duplicate "
                                  "CGN_KNOWLEDGE_REQ for same topic")

    # Force cooldown bypass: set the recorded ts to long ago
    gw._grounding_cooldown["dragons"] = 0.0
    gw._check_grounding_appropriateness(
        ctx, catalyst, voice_cfg=voice_cfg, bus=published.append)
    assert len(published) == 2  # fires again after cooldown elapsed


def test_multi_topic_fires_for_first_past_cooldown(tmp_path, monkeypatch):
    """Design contract: per-call emits at most ONE CGN_KNOWLEDGE_REQ,
    for the first topic past its cooldown. Different topics in the same
    catalyst each carry their own cooldown — different signal, not spam.
    """
    db = tmp_path / "inner_memory.db"
    _make_knowledge_db(str(db), [])
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.replace(str(db), "data/inner_memory.db")

    gw, ctx = _make_gateway(str(tmp_path))
    published: list[dict] = []
    voice_cfg = {"dual_mode_enabled": True,
                 "x_grounding_enforced": True,
                 "x_grounding_threshold": 0.5,
                 "x_grounding_cooldown_secs": 3600}

    # First catalyst mentions apple + pear — first topic past cooldown fires
    gw._check_grounding_appropriateness(
        ctx, {"type": "", "content": "apple pear"},
        voice_cfg=voice_cfg, bus=published.append)
    assert len(published) == 1
    first_topic = published[0]["payload"]["topic"]
    assert first_topic in ("apple", "pear")

    # Second catalyst swaps order — if first_topic was "apple" (in cooldown),
    # "pear" fires. If first_topic was "pear" (in cooldown), "apple" fires.
    # Either way, exactly one new publish.
    gw._check_grounding_appropriateness(
        ctx, {"type": "", "content": "apple pear"},
        voice_cfg=voice_cfg, bus=published.append)
    assert len(published) == 2
    second_topic = published[1]["payload"]["topic"]
    assert second_topic in ("apple", "pear")
    assert second_topic != first_topic  # the OTHER one


def test_agno_hooks_uses_knowledge_gate_utility():
    """Regression: §[24] should route through knowledge_gate so both paths
    share topic semantics. If this fails the shared utility was not wired."""
    import titan_plugin.agno_hooks as ah
    src = Path(ah.__file__).read_text()
    assert "from titan_plugin.logic.knowledge_gate import" in src
    assert "extract_topic_words" in src
    assert "check_topic_confidence" in src


def test_load_config_surfaces_voice_section(tmp_path, monkeypatch):
    """Regression: gateway._load_config() MUST include the [voice] section
    in its returned dict, otherwise the grounding gate silently disables
    itself (caught in production 2026-04-20 after Phase 5C deploy — 3
    posts fired with zero gate events because voice_cfg was always {}).

    This test writes a minimal config.toml with a [voice] section, points
    the loader at it, and asserts the dict surfaces the keys.
    """
    import titan_plugin.config_loader as cl

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[social_x]\nenabled = true\n\n'
        '[voice]\ndual_mode_enabled = true\n'
        'x_grounding_enforced = false\n'
        'x_grounding_threshold = 0.7\n'
        'x_grounding_cooldown_secs = 1800\n'
    )

    def fake_load(force_reload=False):
        import tomllib
        with open(cfg_path, "rb") as f:
            return tomllib.load(f)

    monkeypatch.setattr(cl, "load_titan_config", fake_load)
    monkeypatch.setattr("titan_plugin.logic.social_x_gateway.logger.warning",
                        lambda *a, **kw: None)

    from titan_plugin.logic.social_x_gateway import SocialXGateway
    gw = SocialXGateway(
        db_path=str(tmp_path / "sx.db"),
        config_path=str(cfg_path),
        telemetry_path=str(tmp_path / "tel.jsonl"),
    )
    cfg = gw._load_config()

    assert "voice" in cfg, "_load_config() must expose [voice] section"
    assert cfg["voice"].get("dual_mode_enabled") is True
    assert cfg["voice"].get("x_grounding_enforced") is False
    assert cfg["voice"].get("x_grounding_threshold") == 0.7
    assert cfg["voice"].get("x_grounding_cooldown_secs") == 1800
