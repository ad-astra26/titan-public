"""
api/v6 manifest — the route→accessor→SHM-slot→producer-worker source-of-truth.

Phase E (RFP_phase_c_titan_hcl_cleanup §2 Phase E). This module is the SINGLE
debugging source-of-truth the Maker asked for: "data X not loading" → look up the
v6 route for X → check the freshness of its SHM slot(s) → check the producer
worker(s) that write those slots. One lookup, one chain.

It is consumed by:
  - `GET /v6/manifest`  (runtime introspection — adds live slot freshness)
  - `scripts/gen_v6_manifest.py` → `titan-docs/API_V6_MANIFEST.md` (checked-in doc)

Design (SPEC §A.4 + Preamble G18): readout routes bind to exactly ONE
`TitanStateAccessor` method and source SHM-direct; the `shm_slots` + `producers`
columns make the producer chain explicit. Mutation/admin routes bind to a
command-sender or kernel-RPC entry instead (no accessor read) — recorded with
`kind="mutation"` / `kind="admin"` and a `command` string.

Adding a v6 route is a two-line change: add the @router handler in v6.py AND a
RouteSpec row here. The /v6/manifest endpoint cross-checks that every registered
FastAPI v6 route has a manifest row (and vice-versa) so the doc can never silently
drift from the live router.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RouteKind = Literal["readout", "mutation", "admin"]


@dataclass(frozen=True)
class RouteSpec:
    """One v6 route's documented data lineage.

    path:      the v6 path as registered on the FastAPI router (no titan prefix).
    method:    HTTP verb.
    group:     the semantic group (taxonomy §1) — also the sub-router tag.
    kind:      readout | mutation | admin.
    summary:   one-line human description (shown in the doc + endpoint).
    accessor:  for readout — the `TitanStateAccessor` access path that sources it
               (e.g. "spirit.get_v4_state"). None for mutation/admin.
    command:   for mutation/admin — the command-sender / kernel-RPC entry it calls.
    shm_slots: the SHM slot file(s) the data ultimately reads from (the producer
               chain). Empty for pure plugin/kernel-RPC diagnostics.
    producers: the worker(s) that WRITE those slots (single-writer per G21).
    rpc:       True if the route also reads the kernel_rpc proxy (titan_hcl) for a
               plugin/kernel-only diagnostic not available SHM-direct (guardian
               status, bus_broker, registry_bank). Documented honestly.
    replaces:  the legacy /v3,/v4 path(s) this v6 route subsumes (for the
               deprecation 301 map + provenance).
    """

    path: str
    method: str
    group: str
    kind: RouteKind
    summary: str
    accessor: str | None = None
    command: str | None = None
    shm_slots: tuple[str, ...] = ()
    producers: tuple[str, ...] = ()
    rpc: bool = False
    replaces: tuple[str, ...] = ()


# ── The manifest. One row per v6 route. Grouped by taxonomy §1. ──────────────
# Populated group-by-group as v6.py gains each group's handlers. The
# /v6/manifest endpoint asserts this list and the live router agree.
REGISTRY: list[RouteSpec] = []


def register(*specs: RouteSpec) -> None:
    """Append rows to the manifest (called at import time by v6.py groups)."""
    REGISTRY.extend(specs)


def by_path(path: str, method: str = "GET") -> RouteSpec | None:
    for s in REGISTRY:
        if s.path == path and s.method.upper() == method.upper():
            return s
    return None


def deprecation_map() -> dict[str, str]:
    """legacy /v3,/v4 path → v6 path. Drives the 301 redirect layer (E.4)."""
    out: dict[str, str] = {}
    for s in REGISTRY:
        for old in s.replaces:
            out[old] = s.path
    return out


def groups() -> list[str]:
    seen: list[str] = []
    for s in REGISTRY:
        if s.group not in seen:
            seen.append(s.group)
    return seen


def as_rows() -> list[dict]:
    """Plain-dict view for JSON serialization (/v6/manifest) + doc generation."""
    return [
        {
            "route": s.path,
            "method": s.method,
            "group": s.group,
            "kind": s.kind,
            "summary": s.summary,
            "accessor": s.accessor,
            "command": s.command,
            "shm_slots": list(s.shm_slots),
            "producers": list(s.producers),
            "rpc": s.rpc,
            "replaces": list(s.replaces),
        }
        for s in REGISTRY
    ]
