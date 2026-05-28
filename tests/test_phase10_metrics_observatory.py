"""Phase 10 — /v6/synthesis/metrics/* handlers (P10.C)."""

import asyncio
import json
import os

import titan_hcl.api.synthesis_metrics_handlers as H


class _Req:
    def __init__(self, **q):
        self.query_params = q


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _write_snapshot(tmp_path, payload):
    os.environ["TITAN_DATA_DIR"] = str(tmp_path)
    H._SNAPSHOT_CACHE.clear()
    p = os.path.join(str(tmp_path), "synthesis_metrics_snapshot.json")
    with open(p, "w") as f:
        json.dump(payload, f)
    return p


def test_missing_snapshot_soft_fail(tmp_path):
    os.environ["TITAN_DATA_DIR"] = str(tmp_path)
    H._SNAPSHOT_CACHE.clear()
    out = _run(H.get_v6_synthesis_metrics(_Req()))
    assert out["ok"] is True
    assert out["snapshot"] == "missing"


def test_full_bundle_ok(tmp_path):
    _write_snapshot(tmp_path, {
        "sovereignty": {"available": True, "windows": {"all": {"ratio": 0.5}}},
        "groundedness": {"available": True, "heatmap": []},
        "skills": {"available": True, "size": 3},
        "retrieval": {"available": True, "samples": 120},
        "chi": {"available": True, "spent": 0.002},
        "chain_growth": {"available": True, "total_bytes": 1000},
        "ts": 12345.0,
    })
    out = _run(H.get_v6_synthesis_metrics(_Req()))
    assert out["ok"] is True
    assert out["snapshot"] == "ok"
    assert out["metrics"]["sovereignty"]["windows"]["all"]["ratio"] == 0.5


def test_sovereignty_subroute(tmp_path):
    _write_snapshot(tmp_path, {"sovereignty": {"available": True,
                    "windows": {"24h": {"ratio": 0.7, "trend": 0.1}}}, "ts": 1.0})
    out = _run(H.get_v6_synthesis_metrics_sovereignty(_Req()))
    assert out["sovereignty"]["windows"]["24h"]["ratio"] == 0.7


def test_retrieval_subroute_includes_chi(tmp_path):
    _write_snapshot(tmp_path, {
        "retrieval": {"available": True, "overall": {"p99": 42.0}},
        "chi": {"available": True, "spent": 0.003}, "ts": 1.0})
    out = _run(H.get_v6_synthesis_metrics_retrieval(_Req()))
    assert out["retrieval"]["overall"]["p99"] == 42.0
    assert out["chi"]["spent"] == 0.003


def test_groundedness_subroute(tmp_path):
    _write_snapshot(tmp_path, {"groundedness": {"available": True,
                    "heatmap": [{"concept_id": "c1", "groundedness": 0.9}]}, "ts": 1.0})
    out = _run(H.get_v6_synthesis_metrics_groundedness(_Req()))
    assert out["groundedness"]["heatmap"][0]["concept_id"] == "c1"


def test_chain_growth_subroute(tmp_path):
    _write_snapshot(tmp_path, {"chain_growth": {"available": True,
                    "total_bytes": 5000}, "ts": 1.0})
    out = _run(H.get_v6_synthesis_metrics_chain_growth(_Req()))
    assert out["chain_growth"]["total_bytes"] == 5000
