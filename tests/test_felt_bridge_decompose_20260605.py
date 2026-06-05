"""RFP_inner_outer_felt_teaching_bridge §7.1 — Phase 1: Engram(Idea)→Object decompose.

Covers:
  • felt_bridge.normalize_label — shared key-space normalization (RFP §5)
  • FeltBridge engram_objects cache: persist, idempotency, empty-not-cached
    (retry-on-failure), normalization-on-write, soft-fail
  • LanguageTeacher.build_decompose_prompt / parse_decompose_objects (prompt+parse
    ONLY — the teacher never calls the LLM)
  • consolidation_defaults.make_default_decompose — provider bridge (success +
    failure→[] so FeltBridge retries)
  • ConsolidationPass._maybe_decompose — decompose+cache off the hot path; re-touch
    uses the cache (no re-call); bridge-disabled → no-op

Drives core directly (offline), per the §7.1 data-flow.
"""
from __future__ import annotations

import types

import duckdb
import pytest

from titan_hcl.synthesis.felt_bridge import FeltBridge, normalize_label
from titan_hcl.logic.language_teacher import LanguageTeacher
from titan_hcl.synthesis.consolidation_defaults import make_default_decompose
from titan_hcl.synthesis.consolidation import (
    ConsolidationPass,
    Cluster,
    LLMProposal,
    TxCandidate,
    agg_felt,
    _decompose_sample_from_members,
)


class _DirectWriter:
    """Test double for SynthesisWriter — runs each unit synchronously (production
    serializes the SAME closures on the writer thread; correctness is identical)."""

    def submit(self, fn):
        return fn()

    def submit_sync(self, fn):
        return fn()


class _FakeProvider:
    """Async provider stub matching inference.base.InferenceProvider.complete()."""

    def __init__(self, text: str = "", raise_exc: bool = False):
        self._text = text
        self._raise = raise_exc
        self.calls: list[dict] = []

    async def complete(self, prompt, system, temperature=0.2, max_tokens=200,
                       timeout=45.0):
        self.calls.append({"prompt": prompt, "system": system,
                           "max_tokens": max_tokens})
        if self._raise:
            raise RuntimeError("provider down")
        return self._text


@pytest.fixture()
def bridge():
    conn = duckdb.connect(":memory:")
    fb = FeltBridge(conn, _DirectWriter())
    assert fb.ensure_schema() is True
    return fb, conn


# ── normalize_label ───────────────────────────────────────────────────────
def test_normalize_label():
    assert normalize_label("  Glacier ") == "glacier"
    assert normalize_label("Altitude  Stratification") == "altitude stratification"
    assert normalize_label("MICROBE") == "microbe"
    assert normalize_label("") == ""
    assert normalize_label(None) == ""  # type: ignore[arg-type]


# ── FeltBridge engram_objects cache ───────────────────────────────────────
def test_cache_and_get_objects(bridge):
    fb, _ = bridge
    assert fb.get_cached_objects("glacier_x", 2) is None  # never decomposed
    fb.cache_objects("glacier_x", 2, ["Glacier", "microbe", "altitude"])
    got = fb.get_cached_objects("glacier_x", 2)
    assert got is not None
    assert set(got) == {"glacier", "microbe", "altitude"}  # normalize_label-d


def test_cache_idempotent_and_dedup(bridge):
    fb, conn = bridge
    fb.cache_objects("c", 1, ["glacier", "Glacier", "  glacier  "])  # 3 → 1
    fb.cache_objects("c", 1, ["glacier"])  # ON CONFLICT DO NOTHING
    rows = conn.execute(
        "SELECT object_label FROM engram_objects WHERE engram_id='c' "
        "AND version=1").fetchall()
    assert rows == [("glacier",)]


def test_empty_decompose_not_cached_retries(bridge):
    # An empty result must NOT be cached → get returns None → caller retries next
    # touch (a transient provider failure must never be frozen as "no Objects").
    fb, _ = bridge
    fb.cache_objects("c", 1, [])
    assert fb.get_cached_objects("c", 1) is None
    fb.cache_objects("c", 1, ["", "   "])  # all-blank → normalizes away → no-op
    assert fb.get_cached_objects("c", 1) is None


def test_versions_isolated(bridge):
    fb, _ = bridge
    fb.cache_objects("c", 1, ["a"])
    fb.cache_objects("c", 2, ["b"])
    assert fb.get_cached_objects("c", 1) == ["a"]
    assert fb.get_cached_objects("c", 2) == ["b"]


def test_cache_soft_fail_no_raise():
    # A broken writer must not raise out of cache/get (INV-Syn-17 soft-fail).
    class _BrokenWriter:
        def submit(self, fn):
            raise RuntimeError("boom")

        def submit_sync(self, fn):
            raise RuntimeError("boom")

    fb = FeltBridge(duckdb.connect(":memory:"), _BrokenWriter())
    assert fb.ensure_schema() is False
    fb.cache_objects("c", 1, ["x"])           # no raise
    assert fb.get_cached_objects("c", 1) is None  # no raise


# ── LanguageTeacher decompose prompt + parser ─────────────────────────────
def test_build_decompose_prompt_shape():
    spec = LanguageTeacher.build_decompose_prompt(
        "Glacier Microbial Altitude", "glaciers host microbes", max_objects=6)
    assert set(spec) == {"system", "prompt", "max_tokens"}
    assert "Glacier Microbial Altitude" in spec["prompt"]
    assert "glaciers host microbes" in spec["prompt"]
    assert "OBJECT:" in spec["system"]
    assert spec["max_tokens"] > 0


def test_parse_decompose_objects_strict():
    text = (
        "Sure, here are the objects:\n"
        "OBJECT: Glacier\n"
        "OBJECT: microbe\n"
        "random noise line\n"
        "OBJECT: microbe\n"            # dup → dropped
        "OBJECT: altitude.\n"          # trailing punct stripped
        "- not an object\n"
    )
    objs = LanguageTeacher.parse_decompose_objects(text)
    assert objs == ["glacier", "microbe", "altitude"]


def test_parse_decompose_objects_garbage_empty():
    assert LanguageTeacher.parse_decompose_objects("no objects here at all") == []
    assert LanguageTeacher.parse_decompose_objects("") == []
    assert LanguageTeacher.parse_decompose_objects(None) == []  # type: ignore[arg-type]


def test_parse_decompose_objects_cap():
    text = "\n".join(f"OBJECT: o{i}" for i in range(20))
    assert len(LanguageTeacher.parse_decompose_objects(text, max_objects=8)) == 8


# ── make_default_decompose (provider bridge) ──────────────────────────────
def test_make_default_decompose_success():
    prov = _FakeProvider("OBJECT: glacier\nOBJECT: microbe\n")
    fn = make_default_decompose(prov)
    assert fn("Glacier Microbes", "sample") == ["glacier", "microbe"]
    assert len(prov.calls) == 1
    assert "Glacier Microbes" in prov.calls[0]["prompt"]


def test_make_default_decompose_provider_failure_returns_empty():
    prov = _FakeProvider(raise_exc=True)
    fn = make_default_decompose(prov)
    assert fn("X", "y") == []  # → FeltBridge won't cache → retry next touch


# ── ConsolidationPass._maybe_decompose (off the hot path) ─────────────────
def _make_pass(decompose_fn, felt_bridge):
    return ConsolidationPass(
        engram_store=object(),
        cgn_bridge=object(),
        outer_memory_writer=object(),
        mine_recent_txs_fn=lambda **_k: [],
        llm_propose_fn=lambda _c: LLMProposal(action="reject"),
        decompose_fn=decompose_fn,
        felt_bridge=felt_bridge,
    )


def _cv(concept_id="glacier_x", version=2):
    return types.SimpleNamespace(concept_id=concept_id, version=version)


def _cluster():
    return Cluster(members=[
        TxCandidate(tx_hash="t1", fork="declarative", tags=(), embedding=None,
                    content_summary="glaciers host microbes at altitude",
                    felt='{"curiosity": 0.7, "awe": 0.4}'),
    ])


def test_maybe_decompose_caches_and_reuses(bridge):
    fb, _ = bridge
    calls = []

    def fake_decompose(name, sample):
        calls.append((name, sample))
        return ["glacier", "microbe"]

    cp = _make_pass(fake_decompose, fb)
    proposal = LLMProposal(action="new_concept", concept_id="glacier_x",
                           proposed_name="Glacier Microbes", domain_hint="biology")
    # First touch → decomposes + caches.
    out1 = cp._maybe_decompose(_cv(), proposal, _cluster())
    assert out1 == ["glacier", "microbe"]
    assert len(calls) == 1
    # Re-touch (same id+version) → served from cache, decompose NOT called again.
    out2 = cp._maybe_decompose(_cv(), proposal, _cluster())
    assert set(out2) == {"glacier", "microbe"}
    assert len(calls) == 1


def test_maybe_decompose_disabled_is_noop(bridge):
    fb, _ = bridge
    # No decompose_fn → bridge disabled → None, no raise.
    cp = _make_pass(None, fb)
    assert cp._maybe_decompose(_cv(), LLMProposal(action="new_concept",
                               concept_id="c", proposed_name="C"), _cluster()) is None
    # No felt_bridge → disabled too.
    cp2 = _make_pass(lambda n, s: ["x"], None)
    assert cp2._maybe_decompose(_cv(), LLMProposal(action="new_concept",
                                concept_id="c", proposed_name="C"), _cluster()) is None


def test_maybe_decompose_provider_empty_not_cached(bridge):
    fb, _ = bridge
    cp = _make_pass(lambda n, s: [], fb)  # decompose yields nothing (e.g. failure)
    out = cp._maybe_decompose(_cv(), LLMProposal(action="new_concept",
                              concept_id="glacier_x", proposed_name="G"), _cluster())
    assert out == []
    assert fb.get_cached_objects("glacier_x", 2) is None  # not cached → will retry


# ── felt aggregation + sample helpers ─────────────────────────────────────
def test_agg_felt_means_numeric_levels():
    members = [
        TxCandidate("t1", "declarative", (), None, "a", felt='{"curiosity":0.6,"awe":0.4}'),
        TxCandidate("t2", "declarative", (), None, "b", felt='{"curiosity":0.8}'),
        TxCandidate("t3", "declarative", (), None, "c", felt=None),  # no felt
    ]
    agg = agg_felt(members)
    assert agg["curiosity"] == pytest.approx(0.7)  # (0.6+0.8)/2
    assert agg["awe"] == pytest.approx(0.4)
    assert agg_felt([]) == {}


def test_agg_felt_excludes_metadata_keys():
    members = [TxCandidate("t1", "declarative", (), None, "a",
                           felt='{"curiosity":0.5,"emotion":"joy","ts":123}')]
    agg = agg_felt(members)
    assert "curiosity" in agg
    assert "emotion" not in agg and "ts" not in agg  # _FELT_META_KEYS excluded


def test_decompose_sample_bounded():
    members = [TxCandidate(f"t{i}", "declarative", (), None, f"thought {i}")
               for i in range(10)]
    sample = _decompose_sample_from_members(members, max_samples=3)
    assert sample.count("- ") == 3  # capped
