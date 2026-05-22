"""
api/v6 deprecation layer — hard-deprecate /v3 + /v4 → /v6 (Phase E, E.4).

Locked Maker decision (RFP §5.3): HARD-DEPRECATE the legacy `/v3` + `/v4` routes.
Every legacy path is replaced by a permanent redirect to its v6 successor (the
`replaces=` provenance in v6.py's ROUTE_TABLE → `v6_manifest.deprecation_map()`).
There is NO long-lived shim: the legacy route bodies are removed from dashboard.py
(the handler FUNCTIONS remain, reached only via the v6 router), and this layer is
the only transitional artifact — itself slated for removal once no client emits
`/v3` or `/v4` (the frontend is rewired to v6 in the same commit, E.5).

Redirect status (technically-correct realization of "301/410"):
  - GET  readouts → **301 Moved Permanently** (the Maker's stated code).
  - POST/PUT/DELETE mutations → **308 Permanent Redirect** — preserves method + body
    (a 301 would silently downgrade a POST to GET on many clients, breaking the
    mutation). 308 is the method-preserving permanent form; same "hard-deprecate"
    intent, correct semantics.

The redirect Location is the v6 path with path params substituted + the original
query string preserved. `/t{N}` titan prefixes are stripped by nginx before the app
sees the request, so a same-origin absolute-path Location is correct.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from . import v6_manifest as _m

logger = logging.getLogger(__name__)

router = APIRouter(tags=["v6-deprecated"])


def _make_redirect(v6_template: str, status_code: int):
    """Build a redirect handler for one legacy path → its v6 template.

    Substitutes path params (e.g. /v6/.../{height}) from the incoming request and
    appends the original query string. Closure captures the v6 template + status.
    """
    async def _redirect(request: Request) -> RedirectResponse:
        target = v6_template
        # Substitute path params ({name}, {height}, {proposal_id}, {module_name}).
        if request.path_params:
            try:
                target = v6_template.format(**request.path_params)
            except (KeyError, IndexError):
                target = v6_template  # defensive — template/params mismatch
        if request.url.query:
            target = f"{target}?{request.url.query}"
        return RedirectResponse(url=target, status_code=status_code)

    return _redirect


def build() -> int:
    """Register a redirect for every legacy /v3,/v4 path in the manifest.

    Returns the number of redirects registered. Driven by the manifest so it can
    never drift from the v6 surface (every ROUTE_TABLE row's `replaces` entry gets
    exactly one redirect to the row's v6 path, using the row's method).
    """
    count = 0
    for spec in _m.REGISTRY:
        for legacy in spec.replaces:
            status = 301 if spec.method.upper() == "GET" else 308
            router.add_api_route(
                legacy,
                _make_redirect(spec.path, status),
                methods=[spec.method],
                include_in_schema=False,
            )
            count += 1
    logger.info("[v6-deprecation] registered %d legacy /v3,/v4 → /v6 redirects", count)
    return count


build()
