"""Tests for OuterMemoryWriter + the knowledge_worker pilot migration (Phase 0 / 0C).

Includes the INV-14 PARITY gate: the facade payload must equal knowledge_worker's
old inline TIMECHAIN_COMMIT payload byte-for-structure, so the migration is provably
like-for-like before the old path is gone.

Run isolated:
    python -m pytest tests/test_outer_memory_writer.py -v -p no:anchorpy --tb=short
"""
from pathlib import Path

from titan_hcl import bus
from titan_hcl.synthesis.outer_memory_writer import (
    OuterMemoryEvent,
    OuterMemoryWriter,
)


class _FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, msg):
        self.items.append(msg)


def test_build_payload_core_fields():
    e = OuterMemoryEvent(
        fork="declarative", thought_type="declarative", source="knowledge_research",
        content={"topic": "linux"}, tags=["linux", "knowledge"],
        significance=0.7, novelty=0.9, coherence=0.5,
    )
    p = OuterMemoryWriter(_FakeQueue(), "knowledge").build_payload(e)
    assert p["fork"] == "declarative"
    assert p["content"] == {"topic": "linux"}
    assert p["significance"] == 0.7
    # Optional fields absent when not provided.
    assert "db_ref" not in p and "neuromods" not in p and "chi_available" not in p


def test_optional_fields_included_only_when_set():
    e = OuterMemoryEvent(
        fork="meta", thought_type="meta", source="x", content={},
        db_ref="knowledge_concepts:foo", neuromods={}, chi_available=0.5,
        attention=0.5, i_confidence=0.5, chi_coherence=0.3,
    )
    p = OuterMemoryWriter(_FakeQueue(), "x").build_payload(e)
    assert p["db_ref"] == "knowledge_concepts:foo"
    assert p["neuromods"] == {}
    assert p["chi_available"] == 0.5 and p["chi_coherence"] == 0.3


def test_emit_envelope():
    q = _FakeQueue()
    OuterMemoryWriter(q, "knowledge").emit(OuterMemoryEvent(
        fork="declarative", thought_type="declarative", source="knowledge_research",
        content={"topic": "x"}, tags=["x"],
    ))
    assert len(q.items) == 1
    msg = q.items[0]
    assert msg["type"] == bus.TIMECHAIN_COMMIT
    assert msg["src"] == "knowledge"
    assert msg["dst"] == "timechain"
    assert "ts" in msg and isinstance(msg["payload"], dict)


# ── INV-14 parity gate: facade payload == knowledge_worker's old inline payload ──

def _kw_event(*, with_summary: bool, topic, summary, quality, source_backend,
              requestor, neuromods):
    content = {"topic": topic[:100], "quality": round(quality, 3),
               "search_source": source_backend, "requestor": requestor}
    if with_summary:
        content = {"topic": topic[:100], "summary_len": len(summary),
                   "quality": round(quality, 3), "search_source": source_backend,
                   "requestor": requestor}
    return OuterMemoryEvent(
        fork="declarative", thought_type="declarative", source="knowledge_research",
        content=content, significance=quality, novelty=0.9, coherence=0.5,
        tags=[t.strip() for t in topic.lower().split()[:3]] + ["knowledge"],
        db_ref=f"knowledge_concepts:{topic[:50]}", neuromods=neuromods or {},
        chi_available=0.5, attention=0.5, i_confidence=0.5, chi_coherence=0.3,
    )


def _kw_old_payload(*, with_summary, topic, summary, quality, source_backend,
                    requestor, neuromods):
    """Verbatim reconstruction of knowledge_worker's pre-0C inline payloads."""
    if with_summary:
        content = {"topic": topic[:100], "summary_len": len(summary),
                   "quality": round(quality, 3), "search_source": source_backend,
                   "requestor": requestor}
    else:
        content = {"topic": topic[:100], "quality": round(quality, 3),
                   "search_source": source_backend, "requestor": requestor}
    return {
        "fork": "declarative", "thought_type": "declarative",
        "source": "knowledge_research", "content": content,
        "significance": quality, "novelty": 0.9, "coherence": 0.5,
        "tags": [t.strip() for t in topic.lower().split()[:3]] + ["knowledge"],
        "db_ref": f"knowledge_concepts:{topic[:50]}", "neuromods": neuromods or {},
        "chi_available": 0.5, "attention": 0.5, "i_confidence": 0.5,
        "chi_coherence": 0.3,
    }


def test_parity_site1_with_summary():
    args = dict(topic="Linux Terminal Basics", summary="a" * 240, quality=0.73,
                source_backend="searxng", requestor="self", neuromods={"DA": 0.5})
    w = OuterMemoryWriter(_FakeQueue(), "knowledge")
    assert w.build_payload(_kw_event(with_summary=True, **args)) == \
        _kw_old_payload(with_summary=True, **args)


def test_parity_site2_fallback():
    args = dict(topic="Solana RPC", summary="", quality=0.41,
                source_backend="wiktionary", requestor="curiosity", neuromods=None)
    w = OuterMemoryWriter(_FakeQueue(), "knowledge")
    assert w.build_payload(_kw_event(with_summary=False, **args)) == \
        _kw_old_payload(with_summary=False, **args)


def test_no_inline_timechain_commit_left_in_knowledge_worker():
    """A.7-style coverage for the pilot: knowledge_worker no longer hand-emits
    TIMECHAIN_COMMIT — it routes through the facade."""
    src = (Path(__file__).resolve().parent.parent
           / "titan_hcl" / "modules" / "knowledge_worker.py").read_text()
    assert 'put({"type": bus.TIMECHAIN_COMMIT' not in src
    assert "OuterMemoryWriter(send_queue, name).emit(" in src
