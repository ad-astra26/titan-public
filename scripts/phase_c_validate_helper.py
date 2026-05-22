#!/usr/bin/env python3
"""
phase_c_validate_helper — de-risk the outer_source_assembly helper BEFORE
wiring it into the sidecars. Runs assemble_outer_sources() against the LIVE
SHM of a running Titan and compares per-key against the pre-change parity
baseline (phase_c_parity_pre_*.json).

For each layer it reports, over the keys that were LIVE in the baseline:
  - REPRODUCED: helper produced the key (SHM/file/util/derived path works)
  - MISSING-SHM: helper did NOT produce a key it should (a regression to fix)
  - IN-PROCESS:  key needs a parent-resident provider (bus_stats / observatory
                 counts) or the heavy refresher cache — not exercisable standalone

Usage: python scripts/phase_c_validate_helper.py --titan-id T1 \
           --baseline titan-docs/sessions/phase_c_parity_pre_T1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from titan_hcl.api.shm_reader_bank import ShmReaderBank  # noqa: E402
from titan_hcl.core.state_registry import resolve_titan_id  # noqa: E402
from titan_hcl.logic.outer_source_assembly import (  # noqa: E402
    OuterSourceContext, assemble_outer_sources,
)

# Keys that can only come from a parent-resident provider or the heavy
# refresher cache — not exercisable in a standalone validator.
_IN_PROCESS = {"bus_stats", "art_count_100", "audio_count_100", "art_count_500",
               "audio_count_500", "text_count_500"}
_HEAVY = {"inner_memory_stats", "social_x_gateway_stats",
          "events_teacher_stats", "community_engagement_stats"}
# Stateful trackers — re-homed into the owning sidecar (C.5), NOT the helper.
_TRACKERS = {"outer_spirit_history_stats", "expr_window", "willing_window",
             "outer_body_change", "pi_heartbeat_hrv", "outer_spirit_self_change"}
# Keys whose SHM slot only populates AFTER the C.2b/C.2a producer redeploy
# (e.g. cgn_engine_state was empty on live T1 pre-fix). Validate post-deploy.
_PENDING_DEPLOY = {"cgn_stats"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--titan-id", default=None)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--data-dir", default="/home/antigravity/projects/titan/data",
                    help="live Titan data dir (anchor_state.json etc.)")
    args = ap.parse_args()

    titan_id = resolve_titan_id(args.titan_id)
    with open(args.baseline) as f:
        base = json.load(f)

    bank = ShmReaderBank(titan_id=titan_id)
    ctx = OuterSourceContext(
        shm_bank=bank, titan_id=titan_id,
        data_dir=args.data_dir,
        start_time=time.time() - 3600.0,  # fake 1h uptime
    )

    overall_missing = 0
    for layer_label, layer in base["layers"].items():
        live_keys = [k for k, fp in layer["fields"].items() if fp["present"]]
        produced = assemble_outer_sources(set(live_keys), ctx)
        reproduced, missing_shm, in_process, heavy, trackers, pending = (
            [], [], [], [], [], [])
        for k in live_keys:
            if k in _IN_PROCESS:
                in_process.append(k)
            elif k in _HEAVY:
                heavy.append(k)
            elif k in _TRACKERS:
                trackers.append(k)
            elif k in _PENDING_DEPLOY:
                pending.append(k)
            elif k in produced and produced[k] is not None:
                reproduced.append(k)
            else:
                missing_shm.append(k)
        overall_missing += len(missing_shm)
        print(f"\n=== {layer_label} ({len(live_keys)} live baseline keys) ===")
        print(f"  REPRODUCED ({len(reproduced)}): {sorted(reproduced)}")
        if missing_shm:
            print(f"  ✗ MISSING-SHM ({len(missing_shm)}): {sorted(missing_shm)}")
        print(f"  IN-PROCESS (validate in-sidecar) ({len(in_process)}): {sorted(in_process)}")
        print(f"  HEAVY (refresher cache) ({len(heavy)}): {sorted(heavy)}")
        print(f"  TRACKER (re-homed to sidecar, C.5) ({len(trackers)}): {sorted(trackers)}")
        print(f"  PENDING-DEPLOY (slot populates post-restart) ({len(pending)}): {sorted(pending)}")

    print(f"\n{'='*50}")
    if overall_missing:
        print(f"FAIL — {overall_missing} live key(s) the helper failed to reproduce SHM-direct.")
        return 1
    print("PASS — every SHM/file/util/derived live key reproduced by the helper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
