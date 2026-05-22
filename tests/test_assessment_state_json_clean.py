"""
Guard test for the Phase C strict_map_key root-fix (2026-05-22):
assessment_state.bin must be JSON/msgpack-portable (string map keys). The
producer (SelfAssessment._compute_enrichment) emits trinity dim-index
enrichment keyed by INT; the AssessmentStatePublisher must stringify those at
the SHM-serialization boundary so the slot is readable (msgpack strict_map_key
+ the api JSON layer) — else a non-neutral score silently kills the slot
fleet-wide.

Run: python -m pytest tests/test_assessment_state_json_clean.py -v -p no:anchorpy
"""
from __future__ import annotations

import json
import os
import sys

import msgpack

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from titan_hcl.logic.assessment_state_publisher import (  # noqa: E402
    AssessmentStatePublisher,
)


class _FakeAssessment:
    """get_stats() with an INT-keyed enrichment (the real producer shape:
    `_compute_enrichment` → {layer: {dim_index: delta}})."""

    def get_stats(self):
        return {
            "average_score": 0.72,
            "total": 3,
            "recent": [
                {"action_id": 1, "score": 0.8,
                 "enrichment": {"body": {0: 0.05, 2: 0.03, 4: 0.04}},
                 "ts": 1.0},
                {"action_id": 2, "score": 0.2,
                 "enrichment": {"spirit": {0: -0.02, 1: -0.01}}, "ts": 2.0},
            ],
            "trend": 0.1,
            "score_variance": 0.05,
            "research_avg_score": 0.6,
        }


def test_assessment_payload_is_json_and_msgpack_portable():
    pub = AssessmentStatePublisher(titan_id="T1")
    payload = pub._compute_payload(_FakeAssessment())

    # 1) JSON-serializable (the api serves state as JSON — int keys would raise).
    json.dumps(payload)

    # 2) msgpack round-trips under strict_map_key=True (the default decoder).
    blob = msgpack.packb(payload, use_bin_type=True)
    decoded = msgpack.unpackb(blob, raw=False, strict_map_key=True)

    # 3) enrichment dim-index keys are now strings (root-fix), values preserved.
    enr = decoded["recent"][0]["enrichment"]["body"]
    assert set(enr.keys()) == {"0", "2", "4"}
    assert enr["0"] == 0.05


def test_neutral_enrichment_unaffected():
    """Empty enrichment (neutral score) — still clean, no regression."""
    class _Neutral:
        def get_stats(self):
            return {"average_score": 0.5, "total": 1,
                    "recent": [{"action_id": 9, "score": 0.5,
                                "enrichment": {}, "ts": 1.0}],
                    "trend": 0.0, "score_variance": 0.0}

    payload = AssessmentStatePublisher(titan_id="T1")._compute_payload(_Neutral())
    json.dumps(payload)
    assert payload["recent"][0]["enrichment"] == {}
