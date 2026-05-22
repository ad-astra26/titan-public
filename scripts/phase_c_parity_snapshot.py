#!/usr/bin/env python3
"""
phase_c_parity_snapshot — per-field parity gate for the Phase C
_gather_outer_sources dissolution (RFP_phase_c_titan_hcl_cleanup §2 Phase C,
AUDIT_phase_c_gather_outer_sources_dissolution_20260522.md §4(g)).

The dissolution re-homes the outer-source data plane from the parent's
`_gather_outer_sources` + `OUTER_SOURCES_SNAPSHOT` broadcast into SHM-direct
gathering. The mandatory acceptance gate: **every source key that is LIVE
(non-None) in the pre-change `sensor_cache_outer_*.bin` slots must still be
LIVE post-change, with the same type/shape.** A live→None transition = a
dropped 130D dim input = blocker (the willing-dims / GREAT_PULSE class bug).

Usage:
    # Capture the pre-change baseline from a live Titan's SHM:
    python scripts/phase_c_parity_snapshot.py snapshot \
        --titan-id T1 --out titan-docs/sessions/phase_c_parity_pre.json

    # After the change + redeploy, capture post:
    python scripts/phase_c_parity_snapshot.py snapshot \
        --titan-id T1 --out titan-docs/sessions/phase_c_parity_post.json

    # Gate: diff pre vs post (exit 1 if any live→None regression):
    python scripts/phase_c_parity_snapshot.py diff \
        titan-docs/sessions/phase_c_parity_pre.json \
        titan-docs/sessions/phase_c_parity_post.json

The reader reuses each sidecar's own SLOT_NAME / MAX_PAYLOAD_BYTES /
SCHEMA_VERSION / SOURCE_KEYS so the SHM spec matches the live writer exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import msgpack
import numpy as np

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from titan_hcl.core.state_registry import (  # noqa: E402
    RegistrySpec,
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)
from titan_hcl.logic import (  # noqa: E402
    outer_body_sensor_refresh as _body,
    outer_mind_sensor_refresh as _mind,
    outer_spirit_sensor_refresh as _spirit,
)

_LAYERS = (
    ("outer_body", _body),
    ("outer_mind", _mind),
    ("outer_spirit", _spirit),
)


def _fingerprint(value):
    """Stable, comparable fingerprint of a source value.

    Captures presence + type + shape (NOT exact float values, which drift
    every tick) so pre/post diffs flag structural drops, not live churn.
    """
    if value is None:
        return {"present": False, "type": "NoneType"}
    if isinstance(value, dict):
        return {
            "present": True,
            "type": "dict",
            "key_count": len(value),
            "keys": sorted(str(k) for k in value.keys()),
        }
    if isinstance(value, (list, tuple)):
        return {
            "present": True,
            "type": "list",
            "len": len(value),
            "elem_type": type(value[0]).__name__ if value else "empty",
        }
    if isinstance(value, bool):
        return {"present": True, "type": "bool", "value": value}
    if isinstance(value, (int, float)):
        # Bucket scalars coarsely: 0.0 vs non-zero is the parity-relevant fact
        # (a dim flatlining to 0.0 is the failure mode), exact value is not.
        return {
            "present": True,
            "type": type(value).__name__,
            "is_zero": float(value) == 0.0,
        }
    return {"present": True, "type": type(value).__name__, "repr": repr(value)[:80]}


def _read_slot(layer_mod, titan_id):
    """Read + msgpack-decode one sensor_cache_outer_*.bin slot."""
    spec = RegistrySpec(
        name=layer_mod.SLOT_NAME,
        dtype=np.dtype("uint8"),
        shape=(layer_mod.MAX_PAYLOAD_BYTES,),
        schema_version=layer_mod.SCHEMA_VERSION,
        variable_size=True,
    )
    reader = StateRegistryReader(spec, ensure_shm_root(titan_id))
    raw = reader.read_variable()
    if not raw:
        return None
    # strict_map_key=False — some source dicts (assessment_stats) carry int
    # map keys; matches ShmReaderBank's decode (else the read raises).
    return msgpack.unpackb(raw, raw=False, strict_map_key=False)


def _snapshot(titan_id):
    out = {
        "titan_id": titan_id,
        "captured_at": time.time(),
        "captured_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "layers": {},
    }
    for slot_label, mod in _LAYERS:
        payload = _read_slot(mod, titan_id)
        layer = {
            "slot": mod.SLOT_NAME,
            "read_ok": payload is not None,
            "source_keys": list(mod.SOURCE_KEYS),
            "fields": {},
        }
        if isinstance(payload, dict):
            # Every declared SOURCE_KEY (the contract) + any extra keys present.
            all_keys = set(mod.SOURCE_KEYS) | set(payload.keys())
            for k in sorted(all_keys):
                layer["fields"][k] = _fingerprint(payload.get(k))
        out["layers"][slot_label] = layer
    return out


def _cmd_snapshot(args):
    titan_id = resolve_titan_id(args.titan_id)
    snap = _snapshot(titan_id)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(snap, f, indent=2, sort_keys=True)
    # Human summary
    print(f"[parity] titan={titan_id} → {args.out}")
    for slot_label, layer in snap["layers"].items():
        if not layer["read_ok"]:
            print(f"  {slot_label}: SLOT EMPTY/UNREADABLE (Titan not running?)")
            continue
        live = sum(1 for fp in layer["fields"].values() if fp["present"])
        none = sum(1 for fp in layer["fields"].values() if not fp["present"])
        zero = sum(1 for fp in layer["fields"].values()
                   if fp.get("is_zero") is True)
        print(f"  {slot_label}: {live} live / {none} None / {zero} zero-scalar "
              f"({len(layer['fields'])} keys)")
    return 0


def _cmd_diff(args):
    with open(args.pre) as f:
        pre = json.load(f)
    with open(args.post) as f:
        post = json.load(f)

    regressions = []   # live → None (BLOCKER)
    type_changes = []  # live → live but type changed (WARN)
    new_zero = []       # non-zero scalar → zero scalar (WARN — possible flatline)
    restored = []       # None → live (info; document, don't "restore" dead keys)

    for slot_label, pre_layer in pre["layers"].items():
        post_layer = post["layers"].get(slot_label, {})
        post_fields = post_layer.get("fields", {})
        for k, pre_fp in pre_layer.get("fields", {}).items():
            post_fp = post_fields.get(k, {"present": False, "type": "MISSING"})
            if pre_fp["present"] and not post_fp["present"]:
                regressions.append(f"{slot_label}.{k}: LIVE→{post_fp['type']}")
            elif pre_fp["present"] and post_fp["present"]:
                if pre_fp["type"] != post_fp["type"]:
                    type_changes.append(
                        f"{slot_label}.{k}: {pre_fp['type']}→{post_fp['type']}")
                if pre_fp.get("is_zero") is False and post_fp.get("is_zero") is True:
                    new_zero.append(f"{slot_label}.{k}: scalar non-zero→zero")
            elif (not pre_fp["present"]) and post_fp["present"]:
                restored.append(f"{slot_label}.{k}: None→{post_fp['type']}")

    print(f"[parity-diff] pre={args.pre} (titan {pre.get('titan_id')}) "
          f"post={args.post} (titan {post.get('titan_id')})")
    print(f"  BLOCKER live→None : {len(regressions)}")
    for r in regressions:
        print(f"    ✗ {r}")
    print(f"  WARN type-change  : {len(type_changes)}")
    for r in type_changes:
        print(f"    ! {r}")
    print(f"  WARN scalar→zero  : {len(new_zero)}")
    for r in new_zero:
        print(f"    ! {r}")
    print(f"  INFO None→live    : {len(restored)}")
    for r in restored:
        print(f"    + {r}")

    if regressions:
        print("\n[parity-diff] FAIL — live→None regression(s) = dropped dim input(s).")
        return 1
    print("\n[parity-diff] PASS — no live→None regression.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot", help="capture per-key fingerprints from live SHM")
    s.add_argument("--titan-id", default=None, help="T1/T2/T3 (default: resolve)")
    s.add_argument("--out", required=True, help="output JSON path")
    s.set_defaults(fn=_cmd_snapshot)

    d = sub.add_parser("diff", help="gate pre vs post (exit 1 on live→None)")
    d.add_argument("pre")
    d.add_argument("post")
    d.set_defaults(fn=_cmd_diff)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
