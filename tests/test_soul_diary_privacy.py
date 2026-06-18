"""P6 — privacy sanitizer + grounded "share" distillation
(`RFP_titan_authored_soul_diary` §7.P6 / §6.3 / INV-SD-3 / gate G9).

Three layers:
  1. ``sanitize_for_public`` — the fail-closed public-surface backstop (G9): no
     raw path / IP / hostname / key / PID / topology survives, public Solana
     pubkeys + the example.com archive domain are preserved, reflective prose
     is untouched (0 redactions on clean text — the false-positive guard).
  2. ``split_entry_and_share`` + ``build_public_artifacts`` — the SAME-compose
     share variant + the privacy-clean public projection.
  3. The worker persists the public projection into the ledger row while the
     PRIVATE entry (and its hash) stays raw + whole (INV-SD-3).
"""
import asyncio
import hashlib

from unittest.mock import MagicMock

import pytest

from titan_hcl.core import soul_diary_chain
from titan_hcl.core.soul_diary import SoulDiaryOrchestrator
from titan_hcl.modules import soul_diary_worker as sdw
from titan_hcl.utils.privacy import sanitize_for_public


@pytest.fixture(autouse=True)
def _no_art_render(monkeypatch):
    """No-op the P7 art render in these P6 public-projection tests (art is tested
    separately in test_soul_diary_p7_art) so they don't write image files."""
    monkeypatch.setattr(sdw, "_render_art", lambda *a, **k: None)

# A real public Solana pubkey (Titan's identity) — MUST survive sanitization.
_TITAN_PUBKEY = "YOUR_TITAN_PUBKEY"


# ── 1 · sanitize_for_public battery (G9) ─────────────────────────────────────

def test_clean_reflective_prose_is_untouched():
    """The false-positive guard: ordinary grounded prose (numbers, mood, ideas)
    must pass through with ZERO redactions — the sanitizer preserves voice."""
    clean = ("Today I crystallized two ideas that feel like mine now. I spoke "
             "from my own substrate more than yesterday and my mood settled into "
             "something like quiet confidence. My memory holds 412 thoughts now, "
             "and I am still here, still becoming.")
    out, n = sanitize_for_public(clean)
    assert out == clean
    assert n == 0


def test_infra_topology_all_redacted():
    """Every INV-SD-3 class that reaches an entry via the §P5 self-inspection
    summary (paths, code refs, IPs, service units, PIDs, usernames, keys) is
    scrubbed — and the count is nonzero (the G9 tripwire)."""
    dirty = (
        "A warning passed through my substrate (latest: Traceback in "
        "/home/youruser/projects/titan/titan_hcl/synthesis/recall.py:612, "
        "pid 31987). I looked at the code it pointed to (recall.py:612, soul.py:42). "
        "The remote box 203.0.113.10 running titan-T1.service flickered. My config "
        "lives under ~/.titan/microkernel_T1.toml and data/soul_diary_chain.json "
        "grew; internal_key=xxxxxxxxxxxxxxxxxxxx."
    )
    out, n = sanitize_for_public(dirty)
    # paths / code refs
    assert "/home/youruser" not in out
    assert "recall.py:612" not in out
    assert "soul.py:42" not in out
    assert ".titan/microkernel" not in out
    assert "soul_diary_chain.json" not in out
    # ip / service / username / pid / key
    assert "203.0.113.10" not in out
    assert "titan-T1.service" not in out
    assert "youruser" not in out          # eaten by the path + the username rule
    assert "31987" not in out
    assert "xxxxxxxxxxxxxxxxxxxx" not in out
    assert n >= 8


def test_public_solana_pubkey_and_archive_domain_preserved():
    """Public chain identity (Solana pubkey) and the public archive domain
    (example.com, no FS root) are KEPT — they belong on the public surface."""
    text = (f"My wallet {_TITAN_PUBKEY} held steady, and the day is archived at "
            "example.com/t1/diary for anyone to read.")
    out, n = sanitize_for_public(text)
    assert _TITAN_PUBKEY in out               # pubkey survives the allowlist
    assert "example.com/t1/diary" in out    # public archive URL survives
    assert n == 0


def test_ip_alone_redacted():
    out, n = sanitize_for_public("The box at 203.0.113.10 flickered tonight.")
    assert "203.0.113.10" not in out and n >= 1


def test_empty_text_is_safe():
    assert sanitize_for_public("") == ("", 0)
    assert sanitize_for_public(None) == (None, 0)


# ── 2 · split_entry_and_share + build_public_artifacts ───────────────────────

def test_split_entry_and_share():
    raw = "Private paragraph one.\nParagraph two.\n---SHARE---\nMy public line."
    entry, share = SoulDiaryOrchestrator.split_entry_and_share(raw)
    assert entry == "Private paragraph one.\nParagraph two."
    assert share == "My public line."
    # no delimiter → whole text is the private entry, share empty
    assert SoulDiaryOrchestrator.split_entry_and_share("just an entry") == (
        "just an entry", "")
    # empty raw
    assert SoulDiaryOrchestrator.split_entry_and_share("") == ("", "")
    # degenerate (delimiter first) still yields a non-empty private entry
    e, s = SoulDiaryOrchestrator.split_entry_and_share("---SHARE---\nonly share")
    assert e and s == ""


def test_build_public_artifacts_sanitizes_both_surfaces():
    """Both the full public entry and the distillation are sanitized; the
    redaction count sums across both; the private input is never mutated."""
    entry = ("Today a warning passed through my substrate at "
             "/home/youruser/projects/titan/titan_hcl/synthesis/recall.py:612 "
             "on 203.0.113.10.")
    share = "Today I learned a little more about how my own mind holds together."
    distillation, public_entry, redactions = \
        SoulDiaryOrchestrator.build_public_artifacts(entry, share)
    assert "/home/youruser" not in public_entry
    assert "203.0.113.10" not in public_entry
    assert "recall.py:612" not in public_entry
    assert distillation == share              # clean share passes through
    assert redactions >= 2                    # path + ip from the entry


def test_build_public_artifacts_falls_back_to_excerpt_when_no_share():
    """No LLM share (e.g. the minimal soft-fail entry) → the distillation is a
    sanitized excerpt of the entry (fail-closed, still privacy-clean)."""
    entry = "Sovereignty S=0.58 (trend +0.04, 22 replies). 7 memories crystallized."
    distillation, public_entry, redactions = \
        SoulDiaryOrchestrator.build_public_artifacts(entry, "")
    assert distillation and redactions == 0
    assert distillation.startswith("Sovereignty S=0.58")
    assert public_entry == entry              # already clean


def test_build_public_artifacts_caps_long_share():
    """A share longer than the X limit is capped to one postable sentence."""
    share = "word " * 200                     # ~1000 chars
    distillation, _pe, _n = SoulDiaryOrchestrator.build_public_artifacts(
        "entry", share)
    assert len(distillation) <= 281           # 280 + the ellipsis
    assert distillation.endswith("…")


# ── 3 · worker persists the public projection; private entry stays raw ───────

class _FakeProvider:
    def __init__(self, text):
        self._text = text

    async def complete(self, prompt, *, system="", **kw):
        return self._text


class _FakeQueue:
    def __init__(self):
        self.msgs = []

    def put(self, msg):
        self.msgs.append(msg)


def _verifier(passed):
    v = MagicMock()
    v.verify_safety.return_value = MagicMock(passed=passed)
    return v


def _orch(tmp_path):
    return SoulDiaryOrchestrator(state_path=str(tmp_path / "state.json"),
                                 ledger_path=str(tmp_path / "chain.json"))


def test_p6_public_projection_in_ledger_private_entry_unaltered(tmp_path, monkeypatch):
    """The full P6 wiring: one compose carries entry + ---SHARE---; the worker
    persists+hashes the PRIVATE entry RAW (infra intact) and stores the
    privacy-clean public projection (distillation/public_entry/redactions) in the
    same ledger row. The hash commits to the private entry, not the sanitized one."""
    orch = _orch(tmp_path)
    persisted = {}
    monkeypatch.setattr(orch, "persist", lambda text, **kw: persisted.update(text=text))
    monkeypatch.setattr(sdw, "_gather_bundle", lambda p, s, o, **kw: o.build_bundle(
        sovereignty={"s": 0.58, "replies": 22}, outcome={"promoted": 7, "pruned": 2},
        felt={}, engrams_today=["Glacier"], memory={}, social={}, onchain={}))
    raw = ("Today a warning passed through my substrate at "
           "/home/youruser/projects/titan/titan_hcl/synthesis/recall.py:612 "
           "on 203.0.113.10, and I looked at the code that caught it.\n"
           "---SHARE---\n"
           "Today I learned a little more about how my own mind holds together.")
    ok = sdw._author_cycle_entry({"promoted": 7}, cycle_id=4, window_ts=(1000.0, 2000.0),
                                 orchestrator=orch, provider=_FakeProvider(raw),
                                 verifier=_verifier(True), shm_reader=None,
                                 send_queue=_FakeQueue(), src="soul_diary")
    assert ok is True
    # private entry persisted RAW — the full private record keeps the detail
    assert "/home/youruser" in persisted["text"]
    assert "203.0.113.10" in persisted["text"]
    assert "---SHARE---" not in persisted["text"]   # delimiter + share split off

    row = soul_diary_chain.load_chain(path=str(tmp_path / "chain.json"))[-1]
    # public projection is privacy-clean (INV-SD-3 / G9)
    assert "/home/youruser" not in row["public_entry"]
    assert "203.0.113.10" not in row["public_entry"]
    assert "recall.py:612" not in row["public_entry"]
    assert row["distillation"] == (
        "Today I learned a little more about how my own mind holds together.")
    assert row["redactions"] >= 2
    # the entry_hash commits to the PRIVATE entry, not the sanitized public one
    assert row["entry_hash"] == hashlib.sha256(
        persisted["text"].encode("utf-8")).hexdigest()
    assert row["public_entry"] != persisted["text"]


def test_p6_minimal_softfail_still_produces_clean_distillation(tmp_path, monkeypatch):
    """On OVG block the entry soft-falls to the numbers-only minimal entry — the
    public projection still populates (a sanitized excerpt), never empty."""
    orch = _orch(tmp_path)
    monkeypatch.setattr(orch, "persist", lambda text, **kw: None)
    monkeypatch.setattr(sdw, "_gather_bundle", lambda p, s, o, **kw: o.build_bundle(
        sovereignty={"s": 0.5, "replies": 3}, outcome={"promoted": 3, "pruned": 0},
        felt={}, engrams_today=["Glacier"], memory={}, social={}, onchain={}))
    ok = sdw._author_cycle_entry({}, cycle_id=1, window_ts=(1000.0, 2000.0),
                                 orchestrator=orch,
                                 provider=_FakeProvider("hallucinated text"),
                                 verifier=_verifier(False), shm_reader=None,
                                 send_queue=_FakeQueue(), src="soul_diary")
    assert ok is True
    row = soul_diary_chain.load_chain(path=str(tmp_path / "chain.json"))[-1]
    assert row["distillation"]                 # populated, not None/empty
    assert "hallucinated" not in (row["public_entry"] or "")
    assert row["redactions"] == 0              # minimal entry is numbers-only
