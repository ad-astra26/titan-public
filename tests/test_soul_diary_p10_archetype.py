"""P10 — soul_diary X archetype (`RFP_titan_authored_soul_diary` §7.P10 / §6.4 / INV-SD-8).

Proves the daily-must-post archetype: registered after PROOF_DAY, fires on an
unposted ledger entry (distillation + archive link + the pre-rendered art),
abstains on zero-activity / already-posted days, and attaches the existing P7 art
(never re-renders).
"""
from unittest.mock import MagicMock

from titan_hcl.core import soul_diary_chain
from titan_hcl.logic.social_x.archetypes import (
    ALL_ARCHETYPES, ARCHETYPE_POST_TYPES, SoulDiaryArchetype, SOUL_DIARY_POST_TYPE,
)
from titan_hcl.logic.social_x import archetypes as _arch_pkg
from titan_hcl.logic.social_x.dispatcher import PRIORITY_ORDER


class _Ctx:
    def __init__(self, titan_id="T2"):
        self.titan_id = titan_id
        self.neuromods = {}
        self.emotion = ""


def _arch():
    return SoulDiaryArchetype(gateway=MagicMock(), social_x_db_path="/tmp/none.db")


def _seed_ledger(tmp_path, monkeypatch, *, distillation="Today I grew.", art=None):
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    from titan_hcl.core.shadow_data_dir import resolve_data_path
    ledger = resolve_data_path("data/soul_diary_chain.json")
    soul_diary_chain.append_entry("2026-06-09", "private text",
                                  distillation=distillation, public_entry="clean",
                                  redactions=0, path=ledger)
    if art:
        soul_diary_chain.update_refs("2026-06-09", art_path=art, path=ledger)
    return ledger


def test_registered_as_daily_must_post_after_proof_day():
    assert "soul_diary" in PRIORITY_ORDER
    assert PRIORITY_ORDER.index("soul_diary") == PRIORITY_ORDER.index("proof_day") + 1
    assert SoulDiaryArchetype in ALL_ARCHETYPES
    assert SOUL_DIARY_POST_TYPE in ARCHETYPE_POST_TYPES


def test_fires_on_unposted_entry(tmp_path, monkeypatch):
    _seed_ledger(tmp_path, monkeypatch, art="/tmp/art.jpg")
    arch = _arch()
    monkeypatch.setattr(arch, "already_posted_today", lambda **kw: False)
    cand = arch.find_candidate(_Ctx("T2"))
    assert cand is not None
    assert cand.archetype == "soul_diary"
    assert cand.bypass_spacing and cand.bypass_rate_limit      # sole daily must-post
    assert "generated_art" in cand.layers                      # art layer present
    assert cand.prompt_values["distillation"] == "Today I grew."
    assert cand.prompt_values["archive_url"] == "example.com/t/T2/diary/2026-06-09"
    assert cand.metadata["art_path"] == "/tmp/art.jpg"
    assert cand.metadata["date"] == "2026-06-09"


def test_abstains_when_already_posted_today(tmp_path, monkeypatch):
    _seed_ledger(tmp_path, monkeypatch)
    arch = _arch()
    monkeypatch.setattr(arch, "already_posted_today", lambda **kw: True)
    assert arch.find_candidate(_Ctx("T2")) is None             # one post / UTC day


def test_abstains_on_zero_activity_day(tmp_path, monkeypatch):
    """No ledger row (zero-activity day latched without an entry) → nothing to post."""
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    arch = _arch()
    monkeypatch.setattr(arch, "already_posted_today", lambda **kw: False)
    assert arch.find_candidate(_Ctx("T2")) is None             # empty ledger → abstain


def test_abstains_when_no_distillation(tmp_path, monkeypatch):
    """A pre-P6 row with no distillation → nothing public to share → abstain."""
    _seed_ledger(tmp_path, monkeypatch, distillation="")
    arch = _arch()
    monkeypatch.setattr(arch, "already_posted_today", lambda **kw: False)
    assert arch.find_candidate(_Ctx("T2")) is None


def test_prepare_media_uploads_existing_art_not_rerender(tmp_path, monkeypatch):
    art = tmp_path / "felt.jpg"
    art.write_bytes(b"\xff\xd8\xff art")
    arch = _arch()
    cand = MagicMock()
    cand.metadata = {"art_path": str(art)}
    monkeypatch.setattr(
        "titan_hcl.logic.social_x.image_pipeline.upload_media_via_gateway",
        lambda gateway, path: "MEDIA_123")
    assert arch.prepare_media(cand, neuromods={}, titan_id="T2") == "MEDIA_123"
    # no art file → text-only (soft-fail, empty media_id)
    cand.metadata = {"art_path": "/nonexistent/x.jpg"}
    assert arch.prepare_media(cand, neuromods={}, titan_id="T2") == ""
