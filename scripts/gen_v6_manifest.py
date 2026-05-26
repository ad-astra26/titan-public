#!/usr/bin/env python3
"""
gen_v6_manifest — generate the checked-in api/v6 manifest doc.

Phase E (RFP_phase_c_titan_hcl_cleanup §2 Phase E, locked decision §5.5 =
"generated doc + runtime endpoint"). This is the DOC half: a diffable
markdown table of every v6 route → accessor → SHM-slot → producer-worker, the
single debugging source-of-truth. The runtime half is `GET /v6/manifest`.

Both read the SAME source (titan_hcl/api/v6_manifest.REGISTRY, populated by
v6.py's ROUTE_TABLE), so the doc can never drift from the live router — and
this generator asserts router↔manifest parity before writing.

Usage:
    python scripts/gen_v6_manifest.py            # write titan-docs/notes/API_V6_MANIFEST.md
    python scripts/gen_v6_manifest.py --check     # exit 1 if the doc is stale
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Unconditional front-insert: when run as `python scripts/gen_v6_manifest.py`,
# sys.path[0] is the scripts/ dir; a guarded insert can leave the repo root
# behind an installed/namespace `titan_hcl`. Force the worktree root first.
sys.path.insert(0, _ROOT)

os.environ.setdefault("OPENROUTER_API_KEY", "")

from titan_hcl.api import v6, v6_manifest  # noqa: E402  (import v6 → populates REGISTRY)

_OUT = os.path.join(_ROOT, "titan-docs", "notes", "API_V6_MANIFEST.md")


def render() -> str:
    rows = v6_manifest.as_rows()
    # parity assertion vs the live router
    live = {(r.path, list(r.methods)[0]) for r in v6.router.routes
            if getattr(r, "path", "").startswith("/v6") and hasattr(r, "methods")
            and r.path != "/v6/manifest"}
    manifest = {(r["route"], r["method"]) for r in rows}
    drift_live = sorted(live - manifest)
    drift_manifest = sorted(manifest - live)
    if drift_live or drift_manifest:
        raise SystemExit(
            f"manifest↔router DRIFT — live-only={drift_live} manifest-only={drift_manifest}")

    groups = v6_manifest.groups()
    out: list[str] = []
    out.append("# api/v6 manifest — route → accessor → SHM-slot → producer\n")
    out.append("> **Generated** by `scripts/gen_v6_manifest.py` from "
               "`titan_hcl/api/v6_manifest.REGISTRY` (the same source the live "
               "`GET /v6/manifest` endpoint serves). Do not hand-edit. Regenerate "
               "after any v6 route change; `--check` gates staleness in CI.\n")
    out.append("> **Debug flow:** \"data X not loading\" → find X's row → check the "
               "freshness of its `SHM slots` (via `GET /v6/manifest`) → check the "
               "`producers` worker's health. One lookup, one chain.\n")
    out.append(f"\n**{len(rows)} routes across {len(groups)} groups.** "
               "Legacy `/v3`,`/v4` are hard-deprecated (301/308 → the `replaces` "
               "column is the redirect source).\n")

    from collections import Counter
    kinds = Counter(r["kind"] for r in rows)
    out.append(f"\nKinds: {dict(kinds)}. Groups: {', '.join(groups)}.\n")

    for g in groups:
        grp = [r for r in rows if r["group"] == g]
        out.append(f"\n## /v6/{g}  ({len(grp)} routes)\n")
        out.append("| Method | v6 route | Kind | Accessor / Command | SHM slots | Producers | RPC | Replaces |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in grp:
            src = r["accessor"] or r["command"] or "—"
            slots = ", ".join(r["shm_slots"]) or "—"
            prod = ", ".join(r["producers"]) or "—"
            rep = ", ".join(r["replaces"]) or "—"
            rpc = "✓" if r["rpc"] else ""
            out.append(f"| {r['method']} | `{r['route']}` | {r['kind']} | "
                       f"`{src}` | {slots} | {prod} | {rpc} | `{rep}` |")
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the checked-in doc is stale")
    args = ap.parse_args()
    content = render()
    if args.check:
        existing = ""
        if os.path.exists(_OUT):
            existing = open(_OUT).read()
        if existing != content:
            print(f"STALE: {_OUT} differs from generated — run gen_v6_manifest.py")
            sys.exit(1)
        print(f"OK: {_OUT} up to date")
        return
    with open(_OUT, "w") as f:
        f.write(content)
    print(f"wrote {_OUT} ({len(v6_manifest.REGISTRY)} routes)")


if __name__ == "__main__":
    main()
