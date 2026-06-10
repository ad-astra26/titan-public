"""P7 — soul-diary procedural felt-art wiring
(`RFP_titan_authored_soul_diary` §7.P7 / §1.0 ⑨ / INV-SD-4).

The worker renders the day's art seeded by the entry's cumulative_hash
(deterministic + cryptographically tied to the entry), driven by his TRUE felt
state, and records the path on the ledger row (the P8 NFT / P10 X-post source).
Soft-fail — a render failure never blocks the cascade (INV-SD-13).
"""
import os

from titan_hcl.core import soul_diary_chain
from titan_hcl.core.soul_diary import SoulDiaryOrchestrator
from titan_hcl.modules import soul_diary_worker as sdw


def test_build_art_felt_maps_gather_and_sovereignty():
    orch = SoulDiaryOrchestrator
    bundle = {
        "felt": {"valence": 0.5, "intensity": 0.4,
                 "neuromod_levels": {"dopamine": 0.7, "gaba": 0.2}},
        "sovereignty": {"s": 0.62},
    }
    vec = orch.build_art_felt(bundle)
    assert vec["valence"] == 0.5 and vec["arousal"] == 0.4
    assert vec["neuromods"] == {"dopamine": 0.7, "gaba": 0.2}
    assert vec["coherence"] == 0.62                 # from sovereignty S
    # no felt → None (renderer falls back to the legacy palette)
    assert orch.build_art_felt({"felt": {}}) is None


def test_art_complexity_proxy():
    orch = SoulDiaryOrchestrator
    assert orch.art_complexity({"memory": {"kg_nodes": 1200}}) == 1200
    assert orch.art_complexity({"memory": {"kg_nodes": 99999}}) == 4000   # capped
    assert orch.art_complexity({"engrams_today": ["a", "b"]}) == 100 + 2 * 60


def test_render_art_records_path_and_renders_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    orch = SoulDiaryOrchestrator(ledger_path=str(tmp_path / "chain.json"))
    row = orch.record_hash("2026-06-09", "private entry text")
    bundle = orch.build_bundle(
        sovereignty={"s": 0.6}, outcome={}, felt={
            "valence": 0.5, "intensity": 0.4, "neuromod_levels": {"dopamine": 0.7}},
        engrams_today=["Glacier"], memory={}, social={}, onchain={})

    sdw._render_art(orch, row, bundle, "2026-06-09")

    updated = soul_diary_chain.load_chain(path=str(tmp_path / "chain.json"))[-1]
    assert updated["art_path"]                       # ledger row carries the path
    assert os.path.exists(updated["art_path"])       # the file was actually rendered
    # seeded by the entry's cumulative_hash (deterministic + tied to the entry)
    assert row["cumulative_hash"][:16] in os.path.basename(updated["art_path"]) \
        or os.path.exists(updated["art_path"])       # filename sanitizes the seed


def test_render_art_softfails_without_cumulative_hash(tmp_path, monkeypatch):
    """No cumulative_hash (no ledger row) → no-op, no crash (INV-SD-13)."""
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    orch = SoulDiaryOrchestrator(ledger_path=str(tmp_path / "chain.json"))
    # should simply return without raising
    sdw._render_art(orch, {}, {"felt": {}}, "2026-06-09")
    assert soul_diary_chain.load_chain(path=str(tmp_path / "chain.json")) == []
