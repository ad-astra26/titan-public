"""Phase 10 — MetricsAggregator + LatencyRing."""

import json
import os

from titan_hcl.synthesis.metrics_aggregator import LatencyRing, MetricsAggregator
from titan_hcl.synthesis.sovereignty_meter import SovereigntyRatioMeter


def test_latency_ring_percentiles():
    ring = LatencyRing(maxlen=1000)
    for i in range(200):
        ring.record("turn", "warm", float(i))
    p = ring.percentiles()
    assert p["available"] is True
    assert p["samples"] == 200
    assert p["warming"] is False
    assert "turn:warm" in p["by_bucket"]
    assert p["overall"]["p50"] <= p["overall"]["p99"]


def test_latency_ring_empty():
    assert LatencyRing().percentiles() == {"available": False, "samples": 0}


def test_latency_ring_warming_below_100():
    ring = LatencyRing()
    for i in range(10):
        ring.record("concept", "hot", float(i))
    assert ring.percentiles()["warming"] is True


def _agg(tmp_path, **kw):
    meter = SovereigntyRatioMeter(clock=lambda: 10_000.0)
    snap = os.path.join(tmp_path, "synthesis_metrics_snapshot.json")
    return MetricsAggregator(
        sovereignty_meter=meter,
        snapshot_path=snap,
        data_dir=str(tmp_path),
        **kw,
    ), snap, meter


def test_bundle_shape_all_subbundles_present(tmp_path):
    agg, _, _ = _agg(tmp_path)
    b = agg.build(now_ts=10_000.0)
    for key in ("sovereignty", "groundedness", "skills", "retrieval", "chi", "chain_growth"):
        assert key in b
    assert b["sovereignty"]["available"] is True


def test_missing_sources_soft_fail(tmp_path):
    agg, _, _ = _agg(tmp_path)
    b = agg.build(now_ts=10_000.0)
    # no spine/skills snapshots, no latency ring, no chi provider
    assert b["groundedness"]["available"] is False
    assert b["skills"]["available"] is False
    assert b["retrieval"]["available"] is False
    assert b["chi"]["available"] is False


def test_export_atomic(tmp_path):
    agg, snap, _ = _agg(tmp_path)
    path = agg.export(now_ts=10_000.0)
    assert path == snap
    assert os.path.exists(snap)
    with open(snap) as f:
        data = json.load(f)
    assert "sovereignty" in data


def test_groundedness_reads_spine_snapshot(tmp_path):
    spine = os.path.join(tmp_path, "spine_snapshot.json")
    with open(spine, "w") as f:
        json.dump({"concepts": [
            {"concept_id": "c1", "name": "alpha", "groundedness": 0.9},
            {"concept_id": "c2", "name": "beta", "g": 0.5},
            {"concept_id": "c3", "name": "nogr"},  # no groundedness → excluded
        ]}, f)
    agg, _, _ = _agg(tmp_path)
    b = agg.build(now_ts=10_000.0)["groundedness"]
    assert b["available"] is True
    assert b["count"] == 2
    assert b["heatmap"][0]["concept_id"] == "c1"  # sorted desc by g


def test_skills_reads_skills_snapshot(tmp_path):
    skills = os.path.join(tmp_path, "skills_snapshot.json")
    with open(skills, "w") as f:
        json.dump({"skills": [
            {"skill_id": "s1", "utility_score": 0.8, "verified_at": 123.0,
             "success_count": 3, "failure_count": 1},
            {"skill_id": "s2", "utility_score": 0.4, "verified_at": None,
             "success_count": 0, "failure_count": 2},
        ]}, f)
    agg, _, _ = _agg(tmp_path)
    b = agg.build(now_ts=10_000.0)["skills"]
    assert b["size"] == 2
    assert b["verified_count"] == 1
    assert abs(b["mean_utility"] - 0.6) < 1e-9
    assert b["success_ratio"] == round(3 / 6, 4)


def test_retrieval_from_ring(tmp_path):
    ring = LatencyRing()
    for i in range(150):
        ring.record("turn", "warm", float(i))
    agg, _, _ = _agg(tmp_path, latency_ring=ring)
    b = agg.build(now_ts=10_000.0)["retrieval"]
    assert b["available"] is True
    assert b["samples"] == 150


def test_chi_from_provider(tmp_path):
    agg, _, _ = _agg(tmp_path, chi_stats_provider=lambda: {"spent": 0.003, "cap": 0.01})
    b = agg.build(now_ts=10_000.0)["chi"]
    assert b["available"] is True
    assert b["spent"] == 0.003
