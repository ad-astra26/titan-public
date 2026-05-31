"""Titan liveness — the panel that must work when everything else doesn't.

When the Titan is up: report active + a one-line health summary from
api_hcl:7777/health. When it's DOWN: report down + WHY — systemctl sub-state +
the last journal lines — so the owner isn't staring at a dead dashboard with no
explanation. This is the whole reason the agent is decoupled (decision #13).
"""
from __future__ import annotations

import json

from .context import Context


def _systemctl_active(ctx: Context) -> tuple[str, str]:
    """Return (active_state, sub_state) e.g. ('active','running') / ('inactive','dead')."""
    rc, out, _ = ctx.run(["systemctl", "show", ctx.service_unit,
                          "--property=ActiveState,SubState", "--no-pager"])
    active = sub = "unknown"
    if rc == 0:
        for line in out.splitlines():
            if line.startswith("ActiveState="):
                active = line.split("=", 1)[1].strip()
            elif line.startswith("SubState="):
                sub = line.split("=", 1)[1].strip()
    return active, sub


def _journal_tail(ctx: Context, lines: int = 30) -> list[str]:
    rc, out, _ = ctx.run(["journalctl", "-u", ctx.service_unit, "-n", str(lines),
                          "--no-pager", "--output=short-iso"])
    if rc != 0 or not out:
        return []
    return out.splitlines()[-lines:]


def _probe_health(ctx: Context) -> dict | None:
    status, body = ctx.http("GET", f"{ctx.api_base}/health", timeout=3.0)
    if status != 200:
        return None
    try:
        return json.loads(body.decode())
    except (ValueError, UnicodeDecodeError):
        return {"raw": True}


def titan_status(ctx: Context, *, journal_lines: int = 30) -> dict:
    """Compose the liveness panel. Always returns a dict; never raises."""
    active, sub = _systemctl_active(ctx)
    is_up = active == "active" and sub == "running"

    result = {
        "titan_id": ctx.titan_id,
        "service": ctx.service_unit,
        "systemd": {"active_state": active, "sub_state": sub},
        "up": is_up,
        "health": None,
        "why_down": None,
        "journal_tail": [],
    }

    if is_up:
        health = _probe_health(ctx)
        result["health"] = health
        if health is None:
            # systemd says running but the API isn't answering — half-up.
            result["up"] = False
            result["why_down"] = ("service is active but api_hcl:7777/health is "
                                  "not responding (Titan booting, or API module down)")
            result["journal_tail"] = _journal_tail(ctx, journal_lines)
    else:
        result["why_down"] = f"{ctx.service_unit} is {active}/{sub}"
        result["journal_tail"] = _journal_tail(ctx, journal_lines)

    return result
